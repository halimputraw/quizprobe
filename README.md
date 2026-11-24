# quizprobe

quizprobe reviews a student's answers in a Jupyter coding quiz. It uses
Google **Gemini** to find the misunderstandings behind their mistakes, then
gives them a short multiple-choice post-quiz right inside the notebook. Any
Gemini model works; `gemini-4-hydrogen` is the default.

[![Watch the demo](https://img.youtube.com/vi/7awFmSkwa2g/hqdefault.jpg)](https://youtu.be/7awFmSkwa2g)

## Install

```bash
pip install quizprobe
```

For clickable answer buttons, also install ipywidgets:

```bash
pip install "quizprobe[widgets]"
```

## Set up your Gemini API key

Get a free API key from Google AI Studio, then set it on the machine that runs
Jupyter:

```bash
export GEMINI_API_KEY="your-key"
```

To keep it after a restart, add that line to `~/.zshrc`. quizprobe reads it
from there too. Never paste an API key into the notebook.

## Usage

Add two cells to a quiz notebook. Nothing in between changes.

```python
# first code cell of the quiz
import quizprobe
quizprobe.start(quiz_id="quiz3")
```

```python
# last code cell (the student runs it when they finish)
quizprobe.finish()
# or: quizprobe.finish_button()
```

While the student works, quizprobe records:

- every version of each function the student defines
- every test cell (cells containing `assert`) that passes or fails, and what
  the function actually returned
- runtime errors raised by the student's functions

Syntax errors are counted but not analysed.

### Labelling test cells

A cell containing "Pre-check" is a precheck, a cell containing "hidden" is a
hidden check, and anything else is a check. To label a cell yourself, add a
comment such as `# quizprobe: hidden`.

For tests the student must not see:

```python
quizprobe.hidden_check("sort_inventory", [
    (("Pen, 10, 2; Pen, 9, 2",), [("Pen", 9, 2), ("Pen", 10, 2)]),
])
```

## Choosing a Gemini model

`finish()` uses `gemini-4-hydrogen` unless you name another model. Any Gemini
model your API key can use works. Use the model ID from Google's Gemini API
model list:

```python
quizprobe.finish(model="gemini-3.1-flash-lite")
```

You can also give a list. quizprobe tries the models in order and moves to the
next one when a model is overloaded (503), rate limited, unavailable or gives
an unusable answer:

```python
quizprobe.finish(model=["gemini-4-hydrogen", "gemini-3.1-flash-lite"])
```

The same works with the button: `quizprobe.finish_button(model="gemini-3.1-flash-lite")`.

Larger models usually spot misconceptions more reliably. Smaller "flash" and
"lite" models are faster, cheaper and less often overloaded, but more of their
questions may be rejected by quizprobe's answer check. The report file records
which model actually answered.

## What `finish()` does

1. Sends each question, the student's code (every version) and the test
   results to the Gemini model you chose.
2. Gemini names the misconceptions it finds and writes up to two
   multiple-choice questions for each quiz question.
3. Any "what does this code print?" question is run locally in a locked-down
   sandbox first. If the marked answer is wrong, Gemini gets one chance to fix
   it, and the question is dropped if it is still wrong.
4. The post-quiz appears under the finish cell with answer buttons.

Everything, including the student's post-quiz answers, is saved to
`quizprobe_report.json` for the instructor.

Options:

```python
quizprobe.finish(model="gemini-3.1-flash-lite")   # use a different Gemini model
quizprobe.finish(max_per_question=1)              # at most 1 question per quiz question
quizprobe.finish(report=None)                     # don't write the report file
quizprobe.finish(force=True)                      # review again, even if already done
```

Gemini's free tier limits requests per minute and per day (the limits differ
from model to model), and may use what you send to improve Google's products.
Check the terms before sending student work.

## Answering without widgets

If ipywidgets isn't installed, students answer like this:

```python
quizprobe.answer(1, "B")
```

## Demo

`demo_quiz.ipynb` is a small example quiz with a deliberately wrong answer, so
you can watch quizprobe work. Open it from this folder, set `GEMINI_API_KEY`,
and run the cells in order.
