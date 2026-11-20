"""
quizprobe - reviews a student's answers in a Jupyter coding quiz and gives
them a short multiple-choice post-quiz on the misunderstandings it finds.

Add it to a quiz notebook with two cells:

    # first code cell of the quiz
    import quizprobe
    quizprobe.start(quiz_id="quiz3")

    # last code cell of the quiz (the student runs it when they finish)
    quizprobe.finish()
    # or quizprobe.finish_button()

Between those two cells nothing changes. quizprobe watches every cell the
student runs and records:

  * each version of every function the student defines,
  * every test cell (cells with `assert`) that fails or passes, including what
    the student's function actually returned,
  * runtime errors raised by the student's functions.

Syntax errors are counted but never analysed.

Test cells are labelled automatically: a cell containing "Pre-check" is a
precheck, a cell containing "hidden" is a hidden check, anything else is a
check. Add the comment `# quizprobe: hidden` (or precheck / check) to a cell to
label it yourself. For hidden tests the student must not see, call
`quizprobe.hidden_check("function_name", cases)` instead of writing asserts.

`finish()` sends each question's text, the student's code (every version) and
the test results to an AI model. The model reviews every question, including
ones whose tests passed, names the misconceptions it finds, and writes one
multiple-choice question per misconception, at most two per quiz question.
Questions that ask what a piece of code prints are run locally first; one
whose marked answer differs from the real output is sent back to the model
once for repair, and dropped if it is still wrong. The post-quiz appears
below the finish cell, with buttons to answer, so the student never has to
open another file. quizprobe_report.json keeps everything for the
instructor, including the student's post-quiz answers.

The review is done by a Google Gemini model. Any Gemini model your API key can
use works; gemini-4-hydrogen is the default. Name a different one, or a list
to try in order (the next is used when one is overloaded or unavailable):

    quizprobe.finish(model="gemini-3.1-flash-lite")
    quizprobe.finish(model=["gemini-4-hydrogen", "gemini-3.1-flash-lite"])

Model names are the IDs from Google's Gemini API model list. Larger models
usually review more reliably; smaller "flash" and "lite" models are faster,
cheaper and less often overloaded.

It needs no extra package, only a Gemini API key in the GEMINI_API_KEY
environment variable on the machine that runs the kernel. If the variable is not in the kernel's
environment, an `export GEMINI_API_KEY="..."` line in the user's shell profile
(~/.zshrc and similar) is used instead. Never paste an API key into the
notebook itself. The free tier limits requests per minute and per day, and
may use what you send to improve Google's products, so check the terms before
sending student work.

The buttons need ipywidgets, which JupyterLab and Notebook 7 support. Without
it, students answer with quizprobe.answer(1, "B").
"""

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

DEFAULT_MODEL = "gemini-4-hydrogen"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
GEMINI_KEY_ENV = "GEMINI_API_KEY"
ANSWER_TRIES = 3      # attempts to get a usable answer from the model
RATE_LIMIT_RETRIES = 2  # extra tries when Gemini says "too many requests"
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


class QuizprobeError(Exception):
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

    Tests can run on a function quizprobe never saw being written, for example
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
    marked = re.search(r"#\s*quizprobe:\s*(precheck|check|hidden)", raw_cell, re.I)
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
    # Skip quizprobe's own cells, but not test cells that only carry a label.
    if "quizprobe" in re.sub(r"#\s*quizprobe:\s*\w+", "", raw, flags=re.I):
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
    except Exception:  # quizprobe must never interrupt a quiz
        _internal_errors.append(traceback.format_exc())


def start(quiz_id="quiz", notebook=None, persist=True):
    """Start recording. Safe to run more than once; it never clears the record.

    quiz_id   names the record file (.quizprobe_<quiz_id>.json), so the record
              survives a kernel restart. Use a different id for each quiz.
    notebook  path of the quiz notebook, used to read the question texts. Found
              automatically in most Jupyter setups.
    persist   set False to keep the record in memory only.
    """
    global _state
    ip = _ipython()
    if ip is None:
        raise RuntimeError("quizprobe.start() must be run inside a Jupyter notebook.")
    if not _config["active"]:
        _config.update(quiz_id=re.sub(r"\W+", "_", str(quiz_id)), notebook=notebook, persist=persist)
        if persist and os.path.exists(_state_path()):
            with open(_state_path(), encoding="utf-8") as f:
                _state = {**_new_state(), **json.load(f)}
            if _state["result"] and "reviews" not in _state["result"]:
                _state["result"] = None  # made by an older version of quizprobe; review again
        ip.events.register("post_run_cell", _on_cell)
        _config["active"] = True


def reset():
    """Erase the record for this quiz (instructor use)."""
    global _state
    _state = _new_state()
    if _config["persist"] and os.path.exists(_state_path()):
        os.remove(_state_path())


def status():
    """Print what has been recorded so far."""
    failed = sum(1 for e in _state["events"] if not e["passed"])
    print(f"Functions tracked: {len(_state['attempts'])}")
    print(f"Test runs recorded: {len(_state['events'])} ({failed} failed)")
    print(f"Syntax errors ignored: {_state['syntax_errors']}")
    if _internal_errors:
        print(f"Internal quizprobe errors: {len(_internal_errors)} (last one below)")
        print(_internal_errors[-1])


def hidden_check(function_name, cases, time_limit=None, kind="hidden"):
    """Run tests the student must not see, and record the failures.

    cases       list of (args, expected) pairs; args is a tuple of arguments
                (a single non-tuple argument is also accepted).
    time_limit  optional seconds per case; a slower correct answer is recorded
                as a failure. The call is not interrupted, only timed.

    Prints only how many cases passed. Returns True if all passed.
    """
    ip = _ipython()
    fn = ip.user_ns.get(function_name) if ip is not None else None
    _capture(function_name)
    passed = 0
    for args, expected in cases:
        if not isinstance(args, tuple):
            args = (args,)
        details = {
            "test": _short(f"{function_name}({', '.join(repr(a) for a in args)})", 800),
            "expected": _short(expected),
        }
        if not callable(fn):
            details["error"] = f"NameError: {function_name} is not defined"
        else:
            began = time.perf_counter()
            try:
                returned = fn(*copy.deepcopy(args))
            except Exception as err:  # the student's code may raise anything
                details["error"] = _short(f"{type(err).__name__}: {err}")
            else:
                elapsed = time.perf_counter() - began
                if returned != expected:
                    details["error"] = "Wrong result"
                    details["returned"] = _short(returned)
                elif time_limit is not None and elapsed > time_limit:
                    details["error"] = f"Correct but too slow: {elapsed:.2f}s, limit {time_limit}s"
        if "error" in details:
            _add_event(function_name, kind, False, **details)
        else:
            passed += 1
    if passed == len(cases):
        _add_event(function_name, kind, True, test=f"{len(cases)} case(s)")
    _save()
    print(f"{kind.capitalize()} check for {function_name}: {passed} of {len(cases)} passed")
    return passed == len(cases)


# --------------------------------------------------------------------------
# Building the evidence
# --------------------------------------------------------------------------

# Finds the function a question asks for, e.g. "Define a function sort_inventory(info)".
_ASKS_FOR = re.compile(r"\b(?:function|def)\s+`?([A-Za-z_]\w*)`?\s*\(")


def _notebook_path():
    path = _config["notebook"] or os.environ.get("JPY_SESSION_NAME")
    if not path or not path.endswith(".ipynb"):
        return None
    for candidate in (path, os.path.basename(path)):
        if os.path.exists(candidate):
            return candidate
    return None


def _text_cells():
    """Markdown and raw cells of the quiz notebook: where the questions are written."""
    path = _notebook_path()
    if path is None:
        return []
    try:
        with open(path, encoding="utf-8") as f:
            cells = json.load(f).get("cells", [])
    except (OSError, ValueError):
        return []
    texts = []
    for cell in cells:
        if cell.get("cell_type") in ("markdown", "raw"):
            source = cell.get("source", "")
            texts.append((source if isinstance(source, str) else "".join(source)).strip())
    return texts


def _first_seq(name):
    seqs = [v["seq"] for v in _state["attempts"].get(name, [])]
    seqs += [e["seq"] for e in _state["events"] if e["function"] == name]
    return min(seqs) if seqs else float("inf")


def _plan_review():
    """Decide which functions are quiz answers to review.

    Returns (functions to review in order, {function: question text}, other
    functions the student defined, shown to the model as context only).
    """
    attempts = _state["attempts"]
    tested = {e["function"] for e in _state["events"] if e["kind"] != "experiment"}
    texts = _text_cells()
    asked_for = {m.group(1) for t in texts for m in _ASKS_FOR.finditer(t)}

    questions = {}
    for name in set(attempts) | tested | asked_for:
        pattern = re.compile(rf"\b{re.escape(name)}\s*\(")
        match = next((t for t in texts if pattern.search(t)), None)
        if match:
            questions[name] = match

    # A quiz answer is a function the student wrote that a question asks for,
    # or any function a test cell called (even one the student never wrote).
    review = {n for n in attempts if n in asked_for} | tested
    if not review:  # no question text found and no tests run: review everything
        review = set(attempts)
    review = sorted(review, key=_first_seq)
    others = sorted((n for n in attempts if n not in review), key=_first_seq)
    return review, questions, others


def _format_event(event):
    lines = [f'<test_run kind="{event["kind"]}" result="{"passed" if event["passed"] else "failed"}" '
             f'code_version="{event["version"]}">']
    for key, label in (("test", "test"), ("error", "error"), ("returned", "returned"),
                       ("expected", "expected"), ("raised_at", "raised at"),
                       ("passed_before", "earlier asserts in the same cell that passed")):
        if key in event:
            lines.append(f"{label}: {event[key]}")
    lines.append("</test_run>")
    return "\n".join(lines)


def _evidence():
    """Return (text for the model, list of quiz functions reviewed)."""
    review, questions, others = _plan_review()
    events = _state["events"]

    blocks = []
    for name in review:
        versions = _state["attempts"].get(name, [])
        dropped = max(0, len(versions) - MAX_VERSIONS)
        mine = [e for e in events if e["function"] == name and (dropped == 0 or e["version"] > dropped)]
        failures = [e for e in mine if not e["passed"]]
        skipped = max(0, len(failures) - MAX_FAILURES)
        if skipped:
            cut = {e["seq"] for e in failures[:skipped]}
            mine = [e for e in mine if e["seq"] not in cut]

        timeline = []
        for i, v in enumerate(versions, 1):
            if i > dropped:
                final = ' final="true"' if i == len(versions) else ""
                timeline.append((v["seq"], f'<code version="{i}"{final}>\n{v["code"]}\n</code>'))
        timeline += [(e["seq"], _format_event(e)) for e in mine]
        timeline.sort()

        status = []
        for kind in ("precheck", "check", "hidden"):
            last = next((e for e in reversed(events) if e["function"] == name and e["kind"] == kind), None)
            if last is not None:
                note = "" if last["version"] == len(versions) else " (final code version was not re-tested)"
                status.append(f'{kind}: {"passed" if last["passed"] else "failed"}{note}')

        block = [f'<question function="{name}">']
        if name in questions:
            block.append(f"<question_text>\n{questions[name]}\n</question_text>")
        else:
            block.append("(No question text was found for this function.)")
        if not versions:
            block.append("The student never defined a function with this name.")
        if dropped:
            block.append(f"({dropped} earlier code version(s) and their test runs are not shown.)")
        if skipped:
            block.append(f"({skipped} earlier failed test run(s) are not shown.)")
        block.append("<timeline>")
        block.extend(text for _, text in timeline)
        block.append("</timeline>")
        block.append(f"<final_status>{'; '.join(status) or 'no test cell was run'}</final_status>")
        block.append("</question>")
        blocks.append("\n".join(block))

    if others:
        extra = ["<other_code>",
                 "Other functions the student defined (helpers or scratch work; final versions):"]
        extra += [_state["attempts"][n][-1]["code"] for n in others]
        extra.append("</other_code>")
        blocks.append("\n".join(extra))
    return "\n\n".join(blocks), review


# --------------------------------------------------------------------------
# Asking the model
# --------------------------------------------------------------------------

_SYSTEM = """\
You are helping a university instructor who runs introductory Python coding \
quizzes in Jupyter notebooks. Each quiz question asks the student to write one \
function. Students check their work against visible tests (a precheck that \
costs no marks, and a graded check), and the instructor also runs hidden tests \
the student never sees, which often probe edge cases or efficiency. A student \
has just finished an attempt. For each question you get the question text, \
every version of the student's code in order (the one marked final is their \
answer), and the test runs, with what their function returned.

The student reads your answer straight away, so write to them directly \
("you"), plainly, and without jargon they have not met.

Part 1: review each question. Read the question text and the student's code \
yourself instead of relying on the tests alone: code can pass every test it \
was run against and still be wrong, and a test can fail for a reason that is \
not a misunderstanding. Give every question one verdict:
- correct: the final code does what the question asks.
- unfinished: the final code is a stub or clearly incomplete.
- slip: the final code is wrong only through a typo, a misnamed function or \
variable, a syntax problem, or a similar accident.
- misconception: the code shows a wrong belief about how Python or the problem \
works.
A misconception is a belief that would make the student fail again on a \
different question: for example, that a string method changes the string in \
place, that "12" and 12 compare as equal, or that sorting strings of digits \
puts "10" after "9". Name one only when you can point to the code that shows \
it and say what input makes it go wrong. Slips, unfinished work, style \
preferences and code that is merely longer than needed are not \
misconceptions; slowness counts only when it comes from a wrong belief, such \
as thinking `x in some_list` is as fast as `x in some_set`. A mistake the \
student later fixed still counts when it shows a wrong belief, because one \
fix may have been trial and error; in that case the verdict can be correct \
while the misconception is still listed. Failed test runs are evidence even \
when the code that produced them is not shown (code_version 0): what the \
function returned tells you what that code did. If the same belief shows up in \
several questions, list it once with all of them.
A question with any failed test run always gets at least one misconception \
and one practice question, even when the final code is correct and even when \
the failure looks like a slip or trial and error: name the belief the failing \
version reveals, using what it returned compared with what was expected, and \
give the verdict for the final code as usual. Only for questions where no test \
run failed and nothing shows a misconception, add nothing: an honest "nothing \
to practise" is better than an invented one, because the student will spend \
time on whatever you name.

Part 2: write one multiple-choice question for each misconception, set in a \
fresh context so the student cannot answer by recalling their quiz code. \
Prefer kind "output": a short, self-contained snippet of at most 12 lines with \
the stem "What does this code print?". Its options are exact printed outputs. \
The correct option must be exactly what the snippet prints, character for \
character, and each wrong option should be what a student holding the \
misconception, or a closely related wrong belief, would expect to see. Every \
snippet is run before the student sees it and a question whose marked answer \
does not match the real output is thrown away, so trace the code carefully. \
If the code raises an error, the correct option must name the exception, for \
example "TypeError". Use kind "concept", with the code left empty or shown \
for reading only, only when the misconception cannot be shown through printed \
output.
Rules for every question:
- Exactly 4 options, exactly one of them correct.
- For each option, `why` gives one or two sentences: for the correct one, why \
it is right; for a wrong one, the belief that leads to it and what actually \
happens.
- Snippets use only the Python standard library and print the same output on \
every run: no printing of unordered sets of strings, ids, random numbers, the \
clock, files, input() or network.
- When spaces or quotes are what the question is about, print with repr() so \
the student can see them.
- Do not quote the inputs of hidden tests; describe the kind of case instead.
- quiz_function is the quiz function whose review led to the question."""

_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string", "description": "One or two sentences to the student."},
        "reviews": {
            "type": "array",
            "description": "One entry per quiz question, in the order given.",
            "items": {
                "type": "object",
                "properties": {
                    "function": {"type": "string"},
                    "verdict": {"type": "string",
                                "enum": ["correct", "unfinished", "slip", "misconception"]},
                    "comment": {"type": "string", "description": "One or two sentences to the student."},
                },
                "required": ["function", "verdict", "comment"],
                "additionalProperties": False,
            },
        },
        "misconceptions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "M1, M2, ..."},
                    "name": {"type": "string", "description": "Short plain-language name."},
                    "what_happened": {"type": "string", "description": "The evidence, told to the student."},
                    "explanation": {"type": "string", "description": "How it actually works, 2-4 sentences."},
                    "quiz_functions": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["id", "name", "what_happened", "explanation", "quiz_functions"],
                "additionalProperties": False,
            },
        },
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "misconception_id": {"type": "string"},
                    "quiz_function": {"type": "string"},
                    "kind": {"type": "string", "enum": ["output", "concept"]},
                    "stem": {"type": "string"},
                    "code": {"type": "string", "description": "Python snippet, or empty."},
                    "options": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string"},
                                "correct": {"type": "boolean"},
                                "why": {"type": "string"},
                            },
                            "required": ["text", "correct", "why"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["misconception_id", "quiz_function", "kind", "stem", "code", "options"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["summary", "reviews", "misconceptions", "questions"],
    "additionalProperties": False,
}


def _shape_problem(value, schema, where="answer"):
    """Return how a model's answer departs from the schema, or None if it fits."""
    kind = schema.get("type")
    if kind == "object":
        if not isinstance(value, dict):
            return f"{where} must be an object"
        for key in schema.get("required", []):
            if key not in value:
                return f"{where} is missing '{key}'"
        for key, sub in schema.get("properties", {}).items():
            if key in value:
                problem = _shape_problem(value[key], sub, f"{where}.{key}")
                if problem:
                    return problem
    elif kind == "array":
        if not isinstance(value, list):
            return f"{where} must be a list"
        for i, item in enumerate(value):
            problem = _shape_problem(item, schema["items"], f"{where}[{i}]")
            if problem:
                return problem
    elif kind == "string" and not isinstance(value, str):
        return f"{where} must be text"
    elif kind == "boolean" and not isinstance(value, bool):
        return f"{where} must be true or false"
    if "enum" in schema and value not in schema["enum"]:
        return f"{where} must be one of: {', '.join(schema['enum'])}"
    return None


def _ssl_context():
    """Use certifi's root certificates when installed.

    Python from python.org on macOS ships without root certificates, so HTTPS
    calls fail there unless certifi supplies them. Verification is never turned off.
    """
    try:
        import certifi
    except ImportError:
        return None
    return ssl.create_default_context(cafile=certifi.where())


def _post_json(url, body, headers=None, service="Gemini", timeout=900, retries=None, max_wait=60):
    data = json.dumps(body).encode("utf-8")
    all_headers = {"Content-Type": "application/json", "User-Agent": "quizprobe/1.0"}
    all_headers.update(headers or {})
    retries = RATE_LIMIT_RETRIES if retries is None else retries
    for attempt in range(retries + 1):
        request = urllib.request.Request(url, data=data, headers=all_headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout, context=_ssl_context()) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as err:
            detail = err.read().decode("utf-8", errors="replace")
            try:
                parsed = json.loads(detail)
            except ValueError:
                parsed = None
            if isinstance(parsed, list) and parsed:
                parsed = parsed[0]
            if isinstance(parsed, dict) and "error" in parsed:
                inner = parsed["error"]
                detail = str(inner.get("message", inner)) if isinstance(inner, dict) else str(inner)
            detail = _short(detail, 300)
            busy = err.code in (500, 502, 503, 504, 529)  # temporary overload on the server's side
            if (err.code == 429 or busy) and attempt < retries:
                wait = err.headers.get("Retry-After") or ""
                time.sleep(min(float(wait) if wait.replace(".", "", 1).isdigit() else 20, max_wait))
                continue
            if busy:
                raise QuizprobeError(f"{service} is overloaded right now ({detail}). Wait a minute, "
                                 "then run this cell again.", err.code) from None
            if err.code == 429:
                raise QuizprobeError(f"{service}'s rate limit was reached. Wait a minute, "
                                 "then run this cell again.", 429) from None
            if err.code in (401, 403):
                raise QuizprobeError(f"{service} rejected the API key ({detail}).", err.code) from None
            if err.code == 404:
                raise QuizprobeError(f"{service} could not find the model ({detail}). Check the "
                                     "name against Google's Gemini model list.", 404) from None
            raise QuizprobeError(f"{service} returned an error ({err.code}): {detail}", err.code) from None
        except (urllib.error.URLError, OSError) as err:
            reason = str(getattr(err, "reason", err))
            if "CERTIFICATE_VERIFY_FAILED" in reason:
                reason += ". This Python has no root certificates; run: pip install certifi"
            raise QuizprobeError(f"Could not reach {service} at {url} ({reason}).") from None


def _plain_messages(user_message):
    """Messages for models without built-in schema enforcement: the schema goes in the prompt."""
    system = (_SYSTEM + "\n\nAnswer with one JSON object that follows this JSON schema, "
              "and nothing else:\n" + json.dumps(_SCHEMA))
    return [{"role": "system", "content": system}, {"role": "user", "content": user_message}]


def _until_valid(messages, send, label):
    """Call send(messages) -> (text, cut_off) until the answer fits the schema."""
    problem = None
    for _ in range(ANSWER_TRIES):
        text, cut_off = send(messages)
        if cut_off:
            problem = "it was cut off before it finished"
        else:
            try:
                result = json.loads(_strip_fence(text))
            except ValueError:
                problem = "it was not valid JSON"
            else:
                problem = _shape_problem(result, _SCHEMA)
                if problem is None:
                    return result
        messages += [{"role": "assistant", "content": text},
                     {"role": "user", "content": f"That answer cannot be used: {problem}. "
                                                 "Send the complete corrected JSON object only."}]
    raise QuizprobeError(f"The model '{label}' did not give a usable answer in "
                     f"{ANSWER_TRIES} tries (last problem: {problem}).")


def _find_key(name):
    """Return the API key called `name`, or None.

    Looks in the environment first. Jupyter started from an app icon, or from a
    terminal opened before the key was saved, does not inherit variables set in
    the shell profile, so the profile files are read as a fallback.
    """
    value = os.environ.get(name)
    if value:
        return value
    pattern = re.compile(rf"^\s*(?:export\s+)?{re.escape(name)}=(.*)$")
    for profile in (".zshrc", ".zprofile", ".zshenv", ".bash_profile", ".bashrc", ".profile"):
        try:
            with open(os.path.join(os.path.expanduser("~"), profile), encoding="utf-8") as f:
                lines = f.read().splitlines()
        except (OSError, UnicodeDecodeError):
            continue
        for line in reversed(lines):
            match = pattern.match(line)
            if match:
                value = match.group(1).strip().strip("\"'")
                if value and "$" not in value:
                    return value
    return None


def _ask_gemini(user_message, model=DEFAULT_MODEL):
    """Ask Gemini through its chat-completions endpoint.

    model may be one name or a list of names to try in order; the next one is
    used when a model is overloaded, rate limited, unavailable, too slow, or
    gives unusable answers. Returns (result, name of the model that answered,
    list of skipped models with the reason for each).
    """
    service = "Gemini"
    key = _find_key(GEMINI_KEY_ENV)
    if not key:
        raise QuizprobeError(f"No API key was found. Set the {GEMINI_KEY_ENV} environment variable "
                         "on the machine that runs the notebook kernel, or add the line "
                         f'export {GEMINI_KEY_ENV}="..." to ~/.zshrc.')
    url = GEMINI_URL + "/chat/completions"
    headers = {"Authorization": f"Bearer {key}"}
    models = [model] if isinstance(model, str) else list(model)
    skipped = []

    for position, name in enumerate(models):
        is_last = position == len(models) - 1
        # A model with a backup behind it gets one short second chance; the
        # last model gets the full wait-and-retry treatment.
        limits = {} if is_last else {"retries": 1, "max_wait": 8}
        extra = {"response_format": {"type": "json_object"}}

        def send(messages):
            def post(body):
                return _post_json(url, body, headers, service, timeout=180, **limits)
            try:
                reply = post({"model": name, "messages": messages, **extra})
            except QuizprobeError as err:
                if err.status != 400 or not extra:
                    raise
                extra.clear()  # this model has no JSON mode; rely on the instructions
                reply = post({"model": name, "messages": messages})
            choice = (reply.get("choices") or [{}])[0]
            return ((choice.get("message") or {}).get("content") or "",
                    choice.get("finish_reason") == "length")

        try:
            return _until_valid(_plain_messages(user_message), send, name), name, skipped
        except QuizprobeError as err:
            # A bad request or a rejected key will fail on every model, so stop there.
            if is_last or err.status in (400, 401, 403):
                raise
            skipped.append({"model": name, "reason": str(err)})


def _strip_fence(code):
    """Models sometimes wrap code in ``` fences; remove them."""
    match = re.fullmatch(r"\s*```[\w+-]*\n(.*?)\n?```\s*", code, re.S)
    return match.group(1) if match else code


# --------------------------------------------------------------------------
# Checking the generated questions before the student sees them
# --------------------------------------------------------------------------

def _unsafe_reason(code):
    """Return why generated code must not be run, or None if it looks harmless."""
    try:
        tree = ast.parse(code)
    except SyntaxError as err:
        return f"does not parse: {err.msg}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            modules = [node.module or ""]
        else:
            modules = []
        for module in modules:
            if module.split(".")[0] not in _ALLOWED_IMPORTS:
                return f"imports {module}"
        if isinstance(node, ast.Name) and (node.id in _BANNED_NAMES or "__" in node.id):
            return f"uses {node.id}"
        if isinstance(node, ast.Attribute) and "__" in node.attr:
            return f"uses {node.attr}"
    return None


def _normalise(text):
    """Compare printed output: ignore trailing spaces, blank edge lines and wrapping backticks.

    Leading spaces are kept, because " 8" and "8" are different answers.
    """
    text = text.strip("\n")
    if len(text) > 1 and text.startswith("`") and text.endswith("`"):
        text = text.strip("`")
    return "\n".join(line.rstrip() for line in text.splitlines()).strip("\n")


def _run_snippet(code):
    """Run a snippet twice with different hash seeds, in a separate Python.

    Returns (printed output, name of the exception raised or None, problem or None).
    The environment is emptied, so the snippet cannot see API keys.
    """
    results = []
    for seed in ("1", "2"):
        env = {"PYTHONHASHSEED": seed}
        if os.name == "nt" and "SYSTEMROOT" in os.environ:
            env["SYSTEMROOT"] = os.environ["SYSTEMROOT"]
        with tempfile.TemporaryDirectory() as folder:
            try:
                done = subprocess.run([sys.executable, "-s", "-"], input=code, text=True, env=env,
                                      capture_output=True, timeout=VERIFY_TIMEOUT, cwd=folder)
            except subprocess.TimeoutExpired:
                return "", None, f"its code took longer than {VERIFY_TIMEOUT}s"
        error = None
        if done.returncode != 0:
            last = re.sub(r"\x1b\[[0-9;]*m", "", (done.stderr.strip().splitlines() or ["Error"])[-1])
            error = last.split(":")[0].strip().split(".")[-1] or "Error"
        results.append((_normalise(done.stdout), error))
    if results[0] != results[1]:
        return "", None, "its output changes from run to run"
    printed, error = results[0]
    return printed, error, None


def _mcq_problem(question, misconception_ids):
    """Return why a multiple-choice question is unusable, or None if it passes."""
    if question["misconception_id"] not in misconception_ids:
        return "it is not linked to a listed misconception"
    if not question["stem"].strip():
        return "it has no question text"
    options = question["options"]
    if not 3 <= len(options) <= len(_LETTERS):
        return f"it has {len(options)} options; it needs 3 to {len(_LETTERS)}"
    texts = [_normalise(o["text"]) for o in options]
    if not all(texts):
        return "one of its options is empty"
    if len(set(texts)) != len(texts):
        return "two of its options are the same"
    correct = [i for i, o in enumerate(options) if o["correct"] is True]
    if len(correct) != 1:
        return f"{len(correct)} options are marked correct; exactly 1 must be"
    right = texts[correct[0]]

    code = question["code"] = _strip_fence(question["code"]).strip("\n")
    if question["kind"] == "concept":
        if code:
            try:
                ast.parse(code)
            except SyntaxError:
                return "its code does not parse"
        return None

    if not code.strip():
        return "it asks what code prints but has no code"
    reason = _unsafe_reason(code)
    if reason:
        return f"its code {reason}"
    printed, error, problem = _run_snippet(code)
    if problem:
        return problem
    if error:
        if error.lower() not in right.lower():
            return f"the code raises {error}, but the option marked correct is {right!r}"
        if any(error.lower() in t.lower() for i, t in enumerate(texts) if i != correct[0]):
            return f"the code raises {error}, and a wrong option also says so"
        return None
    if not printed:
        return "its code prints nothing"
    if right != printed:
        return f"the code prints {printed!r}, but the option marked correct is {right!r}"
    return None


def _check_questions(result):
    """Split the model's questions into (passed, [(question, reason)])."""
    ids = {m["id"] for m in result["misconceptions"]}
    passed, failed = [], []
    for question in result["questions"]:
        reason = _mcq_problem(question, ids)
        if reason:
            failed.append((question, reason))
        else:
            passed.append(question)
    return passed, failed


def _failed_functions(review):
    """Quiz functions with at least one failed test run, even if fixed later."""
    failed = {e["function"] for e in _state["events"]
              if e["kind"] != "experiment" and not e["passed"]}
    return [name for name in review if name in failed]


def _missing_practice(result, questions, needed):
    """Functions in `needed` that no usable question practises."""
    functions = {m["id"]: set(m["quiz_functions"]) for m in result["misconceptions"]}
    covered = set()
    for q in questions:
        covered.add(q["quiz_function"])
        covered |= functions.get(q["misconception_id"], set())
    return [name for name in needed if name not in covered]


def _rejection(question, reason):
    """A dropped question, as recorded in the instructor report."""
    return {"misconception_id": question["misconception_id"], "stem": question["stem"],
            "code": question["code"], "reason": reason}


def _limit(questions, result, review, per_question):
    """One question per misconception, and at most per_question per quiz question."""
    first_function = {m["id"]: (m["quiz_functions"] or [""])[0] for m in result["misconceptions"]}
    order = {name: i for i, name in enumerate(review)}
    kept, dropped, used, count = [], [], set(), {}
    for q in questions:
        function = q["quiz_function"] if q["quiz_function"] in order else first_function.get(q["misconception_id"], "")
        q["quiz_function"] = function
        if q["misconception_id"] in used:
            dropped.append((q, "another question already practises this misconception"))
        elif count.get(function, 0) >= per_question:
            dropped.append((q, f"{function} already has {per_question} question(s)"))
        else:
            kept.append(q)
            used.add(q["misconception_id"])
            count[function] = count.get(function, 0) + 1
    kept.sort(key=lambda q: order.get(q["quiz_function"], len(order)))
    return kept, dropped


# --------------------------------------------------------------------------
# Showing the post-quiz inside the quiz notebook
# --------------------------------------------------------------------------

_LETTERS = "ABCDE"
_VERDICTS = {"correct": "Correct", "unfinished": "Unfinished", "slip": "Small slip",
             "misconception": "Needs practice"}


def _show(markdown):
    try:
        from IPython.display import Markdown, display
    except ImportError:
        print(markdown)
    else:
        display(Markdown(markdown))


def _inline_code(text):
    fence = "``" if "`" in text else "`"
    pad = " " if fence == "``" else ""
    return f"{fence}{pad}{text}{pad}{fence}"


def _question_markdown(n, total, q, names):
    lines = [f"#### Question {n} of {total}"]
    practises = names.get(q["misconception_id"])
    if practises:
        lines.append(f"*Practises: {practises}*")
    lines.append(q["stem"])
    if q["code"].strip():
        lines.append(f"```python\n{q['code']}\n```")
    for i, option in enumerate(q["options"]):
        text = option["text"].strip()
        if q["kind"] == "output":
            if "\n" in text:
                lines.append(f"**{_LETTERS[i]}.**\n```\n{text}\n```")
            else:
                lines.append(f"**{_LETTERS[i]}.** {_inline_code(text)}")
        else:
            lines.append(f"**{_LETTERS[i]}.** {text}")
    return "\n\n".join(lines)


def _save_answers_to_report():
    path = _state.get("report")
    if not path or not os.path.exists(path):
        return
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        data["post_quiz_answers"] = _state.get("answers", {})
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1)
    except (OSError, ValueError):
        pass


def _grade(n, letter):
    """Record one answer to post-quiz question n. Returns feedback as markdown."""
    questions = (_state.get("result") or {}).get("questions") or []
    if not 1 <= n <= len(questions):
        return f"There is no post-quiz question {n}."
    q = questions[n - 1]
    letters = _LETTERS[:len(q["options"])]
    letter = str(letter).strip().upper()[:1]
    if not letter or letter not in letters:
        return f"Choose one of {', '.join(letters)}."
    tries = _state.setdefault("answers", {}).setdefault(str(n), [])
    tries.append(letter)
    _save()
    _save_answers_to_report()

    chosen = q["options"][letters.index(letter)]
    if chosen["correct"]:
        return f"**Correct.** {chosen['why']}"
    message = f"**Not quite.** {chosen['why']}"
    wrong = sum(1 for t in tries if not q["options"][letters.index(t)]["correct"])
    if wrong >= 2:
        i = next(i for i, o in enumerate(q["options"]) if o["correct"])
        message += f"\n\nThe answer is **{letters[i]}**. {q['options'][i]['why']}"
    else:
        message += " Try again."
    return message


def _show_questions(questions, names):
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except ImportError:
        widgets = None

    for n, q in enumerate(questions, 1):
        _show(_question_markdown(n, len(questions), q, names))
        if widgets is None:
            continue
        pick = widgets.ToggleButtons(options=list(_LETTERS[:len(q["options"])]), value=None,
                                     description="Your answer:", style={"description_width": "initial"})
        check = widgets.Button(description="Check answer")
        out = widgets.Output()

        def clicked(_, n=n, pick=pick, out=out):
            out.clear_output()
            with out:
                if pick.value is None:
                    _show("Choose an answer first.")
                    return
                try:
                    _show(_grade(n, pick.value))
                except Exception:  # never show the student a traceback
                    _internal_errors.append(traceback.format_exc())
                    _show("Something went wrong checking this answer. Tell your instructor.")

        check.on_click(clicked)
        display(widgets.HBox([pick, check]), out)

    if widgets is None:
        _show('Answer each question by running, for example, `quizprobe.answer(1, "B")` in a new cell.')


def _show_result(result):
    names = {m["id"]: m["name"] for m in result["misconceptions"]}
    lines = ["### Your post-quiz", "", result["summary"], ""]
    if result["reviews"]:
        lines += ["**How each question went**", ""]
        for r in result["reviews"]:
            label = _VERDICTS.get(r["verdict"], r["verdict"])
            lines.append(f"- `{r['function']}`: **{label}.** {r['comment']}")
        lines.append("")
    if not result["misconceptions"]:
        lines.append("Nothing to practise: no misunderstandings showed up in your answers.")
        _show("\n".join(lines))
        return
    lines += ["**What to work on**", ""]
    for m in result["misconceptions"]:
        lines.append(f"- **{m['name']}.** {m['what_happened']} {m['explanation']}")
    lines.append("")
    if result["questions"]:
        lines.append(f"Answer the {len(result['questions'])} question(s) below. "
                     "They do not affect your mark.")
    else:
        lines.append("No practice questions could be prepared this time; read the notes above.")
    _show("\n".join(lines))
    _show_questions(result["questions"], names)


def post_quiz():
    """Show the post-quiz again, for example after the kernel was restarted."""
    if not _state.get("result"):
        _show("There is no post-quiz yet. Run `quizprobe.finish(...)` first.")
        return
    _show_result(_state["result"])


def answer(n, letter):
    """Answer post-quiz question n with a letter, e.g. answer(1, "B").

    The buttons under each question do the same; this is for setups where the
    buttons do not appear.
    """
    _show(_grade(int(n), letter))


def finish(model=DEFAULT_MODEL, max_per_question=2, report="quizprobe_report.json",
           llm=None, force=False):
    """Review the attempt and show the post-quiz below this cell. Run it once, at the end.

    model             any Gemini model name, e.g. "gemini-3.1-flash-lite", or a
                      list of names to try in order when one is overloaded or
                      unavailable. Default: "gemini-4-hydrogen".
    max_per_question  most multiple-choice questions for one quiz question; each
                      one practises a different misconception.
    report            file name of the instructor report (JSON), or None.
    llm               for testing: a function taking the request text and
                      returning the result dict, used instead of a real model.
    force             review again even if a post-quiz was already made.

    Returns the result dict, or None if nothing was produced.
    """
    if isinstance(model, str):
        model = model.strip()
    else:
        model = [str(m).strip() for m in model if str(m).strip()]
    if not model:
        raise ValueError('Name a Gemini model, e.g. finish(model="gemini-3.1-flash-lite"), '
                         'or a list of them.')

    if _state.get("result") and not force:
        _show_result(_state["result"])
        return _state["result"]

    evidence, review = _evidence()
    if not review:
        _show("### No post-quiz needed\n\nNo answers were recorded in this attempt.")
        return None

    request = (f"Here is the record of the student's attempt.\n\n{evidence}\n\n"
               "Review every question above. Write one multiple-choice question per "
               f"misconception, and no more than {max_per_question} for any one quiz question.")
    needs_practice = _failed_functions(review)
    if needs_practice:
        request += ("\n\nThese quiz questions had at least one failed test run, so each must get "
                    "at least one misconception and one practice question, even if the final "
                    "code is correct: " + ", ".join(needs_practice) + ".")
    record = {"created": datetime.datetime.now().isoformat(timespec="seconds"),
              "quiz_id": _config["quiz_id"], "provider": "gemini", "model": model,
              "syntax_errors_ignored": _state["syntax_errors"], "evidence": evidence}

    def write_report():
        if report:
            with open(report, "w", encoding="utf-8") as f:
                json.dump(record, f, indent=1)

    def ask(text):
        if llm is not None:
            return llm(text)
        result, record["answered_by"], skipped = _ask_gemini(text, model)
        record.setdefault("skipped_models", []).extend(skipped)
        return result

    print("Reviewing your answers. This can take a minute...")
    try:
        result = ask(request)
        passed, failed = _check_questions(result)
        if failed:
            # One chance to repair: tell the model exactly what the check found.
            record["first_answer_rejected"] = [_rejection(q, r) for q, r in failed]
            repair = (request + "\n\n<previous_answer>\n" + json.dumps(result, indent=1)
                      + "\n</previous_answer>\n\nAn automatic check ran the code in your "
                      "multiple-choice questions and rejected these:\n"
                      + "\n".join(f"- questions[{result['questions'].index(q)}] "
                                  f"(misconception {q['misconception_id']}): {r}" for q, r in failed)
                      + "\n\nSend the complete answer again, with the rejected questions "
                      "corrected or replaced. Keep everything else unless the check shows it was wrong.")
            try:
                second = ask(repair)
            except QuizprobeError as err:
                record["repair_error"] = str(err)
            else:
                passed2, failed2 = _check_questions(second)
                if len(passed2) >= len(passed):
                    result, passed, failed = second, passed2, failed2
        missing = _missing_practice(result, passed, needs_practice)
        if missing:
            # A test failed but the model wrote nothing to practise for it: ask once more.
            record["missing_practice"] = missing
            follow_up = (request + "\n\n<previous_answer>\n" + json.dumps(result, indent=1)
                         + "\n</previous_answer>\n\nYour answer has no usable practice question for "
                         + ", ".join(missing) + ", although a test run for each of them failed. "
                         "Add at least one misconception and one multiple-choice question for each, "
                         "about what the failing version got wrong. Send the complete answer again "
                         "and keep everything else.")
            try:
                third = ask(follow_up)
            except QuizprobeError as err:
                record["missing_practice_error"] = str(err)
            else:
                passed3, failed3 = _check_questions(third)
                if (len(_missing_practice(third, passed3, needs_practice)) < len(missing)
                        and len(passed3) >= len(passed)):
                    result, passed, failed = third, passed3, failed3
    except QuizprobeError as err:
        record["error"] = str(err)
        write_report()
        _show(f"### The post-quiz could not be created\n\n{err}\n\nTell your instructor. "
              "Your attempt is not affected.")
        return None
    except Exception:  # a student should never see a traceback from quizprobe
        record["error"] = traceback.format_exc()
        write_report()
        _show("### The post-quiz could not be created\n\nSomething unexpected went wrong. "
              "Tell your instructor. Your attempt is not affected.")
        return None

    kept, over_limit = _limit(passed, result, review, max_per_question)
    result["questions"] = kept
    record.update(result=result, dropped_questions=[_rejection(q, r) for q, r in failed + over_limit])
    write_report()

    _state.update(result=result, report=report, answers={})
    _save()
    ip = _ipython()
    if ip is not None and _config["active"]:
        try:
            ip.events.unregister("post_run_cell", _on_cell)
        except ValueError:
            pass
        _config["active"] = False
    _show_result(result)
    return result


def finish_button(**options):
    """Show a "Finish attempt" button that runs finish(**options) when clicked."""
    try:
        import ipywidgets
        from IPython.display import display
    except ImportError:
        print("ipywidgets is not installed. Run quizprobe.finish() in a cell instead.")
        return
    button = ipywidgets.Button(description="Finish attempt", button_style="primary")
    output = ipywidgets.Output()

    def clicked(_):
        button.disabled = True
        with output:
            finish(**options)

    button.on_click(clicked)
    display(button, output)
