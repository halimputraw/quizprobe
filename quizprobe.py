from __future__ import annotations

import ast
import builtins
import copy
import datetime
import inspect
import json
import os
import re
import ssl
import subprocess
import sys
import tempfile
import textwrap
import time
import traceback
import urllib.error
import urllib.request

__all__ = ["start", "finish", "finish_button", "post_quiz", "answer",
           "hidden_check", "status", "reset"]

DEFAULT_PROVIDER = "gemini"
DEFAULT_MODEL = "gemini-4-hydrogen"
OLLAMA_URL = "http://localhost:11434"
ANSWER_TRIES = 3      # attempts to get a usable answer from the model
RATE_LIMIT_RETRIES = 2  # extra tries when a hosted service says "too many requests"

# provider -> (name shown in messages, base URL, environment variable holding the key)
_PRESETS = {
    "gemini": ("Gemini", "https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY"),
    "groq": ("Groq", "https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "openrouter": ("OpenRouter", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "openai-compatible": ("The AI service", None, "QUIZPROBE_API_KEY"),
}
MAX_VERSIONS = 4      # code versions per function sent to the model
MAX_FAILURES = 12     # failed test runs per function sent to the model
VERIFY_TIMEOUT = 10   # seconds allowed for one generated code snippet

_ALLOWED_IMPORTS = {
    "math", "itertools", "collections", "functools", "string", "re", "heapq",
    "bisect", "statistics", "datetime", "decimal", "fractions", "operator",
    "typing", "copy", "json",
}
_BANNED_NAMES = {
    "open", "exec", "eval", "compile", "input", "__import__", "globals",
    "locals", "vars", "breakpoint", "exit", "quit", "getattr", "setattr",
    "delattr",
}


class CoachError(Exception):
    """A problem the student or instructor should be told about in plain words."""

    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status  # HTTP status code, when the problem came from a server


def _new_state():
    return {"attempts": {}, "events": [], "syntax_errors": 0, "seq": 0,
            "result": None, "report": None, "answers": {}}


_state = _new_state()
_config = {"active": False, "quiz_id": "quiz", "notebook": None, "persist": True}
_internal_errors = []


# --------------------------------------------------------------------------
# Recording what the student does
# --------------------------------------------------------------------------

def _ipython():
    try:
        from IPython import get_ipython
    except ImportError:
        return None
    return get_ipython()


def _short(value, limit=500):
    text = value if isinstance(value, str) else repr(value)
    return text if len(text) <= limit else text[:limit] + f"... [{len(text) - limit} more characters]"


def _state_path():
    return f".quizprobe_{_config['quiz_id']}.json"


def _save():
    if not _config["persist"]:
        return
    tmp = _state_path() + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_state, f)
    os.replace(tmp, _state_path())


def _next_seq():
    _state["seq"] += 1
    return _state["seq"]


def _add_attempt(name, code):
    versions = _state["attempts"].setdefault(name, [])
    if versions and versions[-1]["code"] == code:
        return
    versions.append({"seq": _next_seq(), "code": code})


def _capture(name):
    """Record the current code of function `name` if it differs from the last version seen.

    Tests can run on a function the coach never saw being written, for example
    one defined before start() or reset(), or before a kernel restart that
    reloaded the record. Reading its source when it is tested keeps the code
    and the test results together.
    """
    ip = _ipython()
    fn = ip.user_ns.get(name) if ip is not None else None
    if not callable(fn) or not hasattr(fn, "__code__"):
        return
    try:
        source = textwrap.dedent(inspect.getsource(fn)).strip()
    except (OSError, TypeError):
        return
    if source:
        _add_attempt(name, source)


def _add_event(function, kind, passed, **details):
    event = {
        "seq": _next_seq(),
        "function": function,
        "kind": kind,
        "passed": passed,
        "version": len(_state["attempts"].get(function, [])),
    }
    event.update({k: v for k, v in details.items() if v not in (None, "")})
    _state["events"].append(event)


def _kind_of(raw_cell):
    marked = re.search(r"#\s*coach:\s*(precheck|check|hidden)", raw_cell, re.I)
    if marked:
        return marked.group(1).lower()
    low = raw_cell.lower()
    if "hidden" in low:
        return "hidden"
    if re.search(r"pre[-_ ]?check", low):
        return "precheck"
    return "check"


def _called_names(node):
    """Names of plain functions called inside an AST node, builtins excluded."""
    names = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name):
            name = sub.func.id
            if not hasattr(builtins, name) and name not in names:
                names.append(name)
    return names


def _user_frames(err):
    frames = traceback.extract_tb(err.__traceback__)
    return [f for f in frames
            if not (f.name == "run_code" and f.filename.endswith("interactiveshell.py"))]


def _observe(result):
    raw = result.info.raw_cell or ""
    if "quizprobe" in raw:
        return
    err = result.error_before_exec or result.error_in_exec
    if isinstance(err, SyntaxError):
        _state["syntax_errors"] += 1
        _save()
        return

    ip = _ipython()
    try:
        src = ip.transform_cell(raw)
        tree = ast.parse(src)
    except SyntaxError:
        return

    frames = _user_frames(err) if err is not None else []
    module_frame = next((f for f in frames if f.name == "<module>"), None)
    fail_line = module_frame.lineno if module_frame else None

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if fail_line is None or node.end_lineno < fail_line:
                _add_attempt(node.name, ast.get_source_segment(src, node).strip())

    asserts = [n for n in tree.body if isinstance(n, ast.Assert)]
    kind = _kind_of(raw) if asserts else "experiment"

    if err is None:
        tested = {}
        for node in asserts:
            for name in _called_names(node):
                tested[name] = tested.get(name, 0) + 1
        for name, count in tested.items():
            _capture(name)
            _add_event(name, kind, True, test=f"{count} assert statement(s)")
        _save()
        return

    failing = None
    if fail_line is not None:
        failing = next((n for n in tree.body if n.lineno <= fail_line <= n.end_lineno), None)

    functions = _called_names(failing) if failing is not None else []
    if not asserts:
        functions = [f for f in functions if f in _state["attempts"]]
    missing = getattr(err, "name", None) if isinstance(err, NameError) else None
    if missing and missing in functions:
        functions = [missing]
    if not functions and frames and frames[-1].name in _state["attempts"]:
        functions = [frames[-1].name]
    if not functions:
        _save()
        return

    details = {
        "test": _short(ast.get_source_segment(src, failing), 800) if failing is not None else None,
        "error": _short(f"{type(err).__name__}: {err}".rstrip(": ")),
    }
    if frames and frames[-1] is not module_frame:
        details["raised_at"] = _short(f"{frames[-1].name}: {frames[-1].line}", 300)
    if failing is not None:
        before = sum(1 for n in asserts if n.end_lineno < failing.lineno)
        if before:
            details["passed_before"] = before
    if isinstance(err, AssertionError) and isinstance(failing, ast.Assert):
        test = failing.test
        if isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.ops[0], ast.Eq):
            details["expected"] = _short(ast.get_source_segment(src, test.comparators[0]))
            try:
                expr = ast.fix_missing_locations(ast.Expression(body=test.left))
                details["returned"] = _short(eval(compile(expr, "<quizprobe>", "eval"), ip.user_ns))
            except Exception as again:  # the student's code may raise anything
                details["returned"] = f"(raised {type(again).__name__} when called again)"

    for name in functions:
        _capture(name)
        _add_event(name, kind, False, **details)
    _save()


def _on_cell(result):
    try:
        _observe(result)
    except Exception:  # the coach must never interrupt a quiz
        _internal_errors.append(traceback.format_exc())


