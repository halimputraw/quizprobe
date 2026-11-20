# quiz-coach

quiz-coach reviews a student's answers in a Jupyter coding quiz. It finds the
misunderstandings behind their mistakes, then gives them a short
multiple-choice post-quiz right inside the notebook.

## Install

```bash
pip install quiz-coach
```

Optional extras:

```bash
pip install "quiz-coach[claude]"    # use Claude as the reviewer
pip install "quiz-coach[widgets]"   # clickable answer buttons
pip install "quiz-coach[all]"       # both
```

## Usage

Add two cells to a quiz notebook. Nothing in between changes.

```python
# first code cell of the quiz
import quiz_coach
quiz_coach.start(quiz_id="quiz3")
```

```python
# last code cell (the student runs it when they finish)
quiz_coach.finish(provider="claude")
# or: quiz_coach.finish_button(provider="claude")
```

While the student works, the coach records:

- every version of each function the student defines
- every test cell (cells containing `assert`) that passes or fails, and what
  the function actually returned
- runtime errors raised by the student's functions

Syntax errors are counted but not analysed.

### Labelling test cells

A cell containing "Pre-check" is a precheck, a cell containing "hidden" is a
hidden check, and anything else is a check. To label a cell yourself, add a
comment such as `# coach: hidden`.

For tests the student must not see:

```python
quiz_coach.hidden_check("sort_inventory", [
    (("Pen, 10, 2; Pen, 9, 2",), [("Pen", 9, 2), ("Pen", 10, 2)]),
])
```

## What `finish()` does

1. Sends each question, the student's code (every version) and the test
   results to an AI model.
2. The model names the misconceptions it finds and writes up to two
   multiple-choice questions for each quiz question.
3. Any "what does this code print?" question is run locally first. If the
   marked answer is wrong, the model gets one chance to fix it, and the
   question is dropped if it is still wrong.
4. The post-quiz appears under the finish cell with answer buttons.

Everything, including the student's post-quiz answers, is saved to
`quiz_coach_report.json` for the instructor.

## AI providers

| Provider | Setup |
|---|---|
| Claude (default) | `pip install anthropic`, set `ANTHROPIC_API_KEY` |
| Ollama (local, free) | `ollama pull hermes3`, then `finish(provider="ollama", model="hermes3")` |
| Gemini / Groq / OpenRouter | set `GEMINI_API_KEY`, `GROQ_API_KEY` or `OPENROUTER_API_KEY` |
| Any OpenAI-compatible API | `finish(provider="openai-compatible", model="...", base_url="...")` with `QUIZ_COACH_API_KEY` |

Keep API keys in environment variables on the machine running the kernel.
Never paste a key into the notebook.

Free tiers have rate limits, and some providers use submitted data to train
their models. Check the provider's terms before sending student work.

## Answering without widgets

If ipywidgets isn't installed, students answer like this:

```python
quiz_coach.answer(1, "B")
```

## Demo

`demo_quiz.ipynb` is a small example quiz with a deliberately wrong answer, so
you can watch the coach work.
