# PRISM piece 4: GUI usability and the artifact browser — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the artifact store reachable — a browser that lists, inspects, annotates, deletes and
tidies every generated artifact, with a command-line twin — and fix the four usability gaps the
programme's piece-4 row named plus the four defects piece 3 handed on.

**Architecture:** The store has been complete since piece 1 and almost nothing calls it; this piece is
a front end over a finished engine, plus three behaviour fixes in the window and four defect repairs.
The store grows additively (five `Summary` fields, three methods, one rule) so that a listing row needs
no second manifest read; one renderer in `core/artifacts/report.py` produces the text both front ends
show; the browser is a fifth Home tile and a plain `QWidget` screen (not a `BasePanel`, so a run
elsewhere cannot grey it out); and `python -m core artifacts <mode>` is its twin over the same calls.

**Tech Stack:** Python 3.12 (conda env `biophys-env`), PySide6 6.9.3, torch 2.9.0+cu130, sbi 0.25.0,
pytest. Windows 11; PowerShell and Git Bash both available.

**Spec:** `docs/superpowers/specs/2026-09-17-gui-usability-and-artifact-browser-design.md` — binding.
Its §1.1 holds decisions **B1–B17**; every task below cites the decision that authorises it. Read the
spec and this plan together.

## Global Constraints

Every task's requirements implicitly include this section.

**Environment**
- Interpreter: `C:\Users\J\anaconda3\envs\biophys-env\python.exe` (Python 3.12). The `python` on PATH
  is NOT this environment and has none of the dependencies.
- Anything importing torch needs `KMP_DUPLICATE_LIB_OK=TRUE`; any headless check that imports PySide6
  also needs `QT_QPA_PLATFORM=offscreen`. The root `conftest.py` defaults both; `run.bat`/`run.sh` set
  only the first (the GUI needs a real platform plugin); a bare `python -c` sets neither.
- CUDA is available (RTX 5070 Ti, 16 GB shared with the desktop). Read free VRAM with
  `nvidia-smi --query-gpu=memory.used --format=csv`, never `torch.cuda.mem_get_info()`.
- Paths never depend on the working directory: `core/config.py` resolves `RESOURCES_ROOT`
  (`PRISM_RESOURCES` overrides) and `artifacts_root()` (`PRISM_ARTIFACTS` overrides) from its own
  location. Point `PRISM_ARTIFACTS` at a scratch directory, or use `core.artifacts.use_store`, to keep
  a check away from the real `Artifacts/`.
- `core/refusals.py`, `core/runs.py`, `core/tool/fields.py`, `core/tool/logging_console.py` and the new
  `core/logging_root.py` are torch-free and stay so: `python -m core --help` must not import torch
  (piece-2 D10). `core/tool/browse.py` keeps its heavy imports inside its handler for the same reason.

**Tests and gates**
- Fast gate, after EVERY task, ONE process, in the background, logging to a file:
  `pytest -m "not slow"`. Target **16 minutes** (12 min 13 s at `9e2f7ef` with 600 tests; this piece
  adds about 130). Never run two pytest processes at once.
- **The gate is run by the session that owns the piece, never by a task's implementer, and no
  implementer starts a background test run.** A task's own steps run FOCUSED tests only (named files
  or node ids).
- **Do not edit any source file while a suite is running.** Several tests assert on
  `inspect.getsource`, which reads the file as it is now with the line numbers the function was loaded
  with; an edit mid-run produces a failure that is not real.
- Markers: `slow` (the chi full-pipeline test in `tests/test_user_sbi.py` and the FDT/crossval
  tiny-size run in `tests/test_tool.py`), `gpu` (skipped without CUDA), `display` (skipped offscreen).
- The suites run against a temp artifact root installed once per process; the teardown asserts the real
  `Artifacts/` gained nothing, that the real `PRISM.ini` is unchanged, and that no `<repo>/sbi-logs`
  appeared. Only a single-process run can catch code that swaps the default store and never restores
  it.
- Every `QMessageBox.exec` a test did not fake itself is recorded in `tests/_fixtures.py::SHOWN` and
  returns 0. A test that wants to inspect a box reads `SHOWN[-1]`; a test that expects none asserts
  `SHOWN == []`. **The guard patches the INSTANCE method only** — the statics
  (`QMessageBox.question/warning/information`) are C++ statics and escape it, so a static call **hangs
  the offscreen suite** instead of failing. Every dialog this piece adds is `QMessageBox(...)` shown
  with `.exec()`.
- GUI tests call `tests/_fixtures.py::qt_app()`, not the unused session `qapp` fixture.
- A green suite does not certify the GPU path. The smoke gate runs on the card once, at the end
  (Task 26's gate step), because Task 21 and Task 22 sit on the training unwind path and Task 20
  installs a handler the training run logs through.
- Do not pipe large Python or Markdown through a bash heredoc; write the file with the Write tool and
  run it. A foreground `python` check that imports torch and touches the prior or checkpoint machinery
  can hang a tool call for good — write it to a script and run it with a timeout.

**Load-bearing rules**
- Every knob is an ARGUMENT, never a config write: `core/orchestrator.py` binds config constants at
  import (`from .config import X`), so assigning `config.X` at runtime changes nothing and the run
  silently uses the default.
- No public stage, composition or diagnostic mutates the configuration it is handed
  (`core.runs.public_entry` copies on entry — piece 3, V1). A caller that wants what a stage wrote
  reads the artifact.
- Refuse before the spend. Every pre-spend refusal is a `core.refusals.Refusal` (or a `StoreError`,
  its subclass) with a field key. In the converted modules **no message names a box, tab, flag or
  button** — `core/gui/fields.py` and `core/tool/fields.py` do that. A `Field`'s description may not
  contain the words tab, box, flag, button, click, tick or dialog (`tests/test_refusals.py` pins it
  with a regex), and `len(FIELDS)` is pinned by an exact count.
- Messages are `logging` records: `log.info` for banners, timings, tables and progress notices,
  `log.warning` for anything the operator should act on, `log.error` for a failure reported rather than
  raised. `core/tool`'s framing `print`s stay prints (its no-print pin excludes `core/tool`).
- `Resources/` holds the hand-edited inputs; everything generated lives under
  `Artifacts/<kind>/<name>__<id>/` with a `manifest.json` and, since piece 3, a `log.txt`. **No code
  outside `core/config.py` builds a literal `Resources/` or `Artifacts/` path** — a test pins it.
- Every stage writes its artifact at completion and returns a `Loaded*` wrapper. Save is a rename.
  Loading refuses any verifiable mismatch, and `Accept(truncated, other_observation)` are the only
  escape hatches. D11 (no chi override) and D12 (no hatch for a round on a non-amortized parent with
  another observation) are standing refusals.
- The GUI runs ONE task at a time app-wide (`BasePanel._running`, `streams._REDIRECT`), because
  `redirect_streams` swaps `sys.stdout`/`sys.stderr` process-wide.
- The science guardrails are `PRISM_HANDOFF.md` §11.6 (TSNPE) and the traps in §5.

**Line numbers move.**
- Every task re-derives its line numbers from the file as it reads it, because an earlier task in the
  same file has usually moved them. Every number in a Files block or a step was correct against the
  tree at `6fc399f` and is a hint, not an address.
- The QUOTED text in a step is the anchor, never the bare number. Find the quoted text; edit what it
  names; ignore the number if the two disagree.
- If the quoted text is not found, re-read the file and say so in the task report rather than guessing
  at the line the step meant.

**Git**
- Work directly on the local `main` branch. No feature branches, no worktrees.
- One commit per task, never amended. Subjects are SHORT: one line, a brief body only when the subject
  cannot carry the why. The orchestrator appends the `Co-Authored-By` trailer; a task's own commit
  command does not write one.
- The user pushes and handles every other remote operation.

**Documents**
- `docs/STATE.md` is the moving state: read it first, update it at the end of every session.
- The execution ledger lives under
  `.superpowers/sdd/2026-09-17-gui-usability-and-artifact-browser/` (gitignored). Every ruling made
  during execution goes in it **with what it costs if it is wrong**.
- The display-walkthrough rows this piece adds (**D1–D16**) are run by the USER on a real screen, at
  the end. **No task may claim them as done.**

## Rulings made at planning time

Before any code was written, ten drafters read the code for these tasks and raised 60 objections to the
interface contract and to the spec. The author ruled on every one, and the rulings are recorded as
**P1–P43** in the ledger
(`.superpowers/sdd/2026-09-17-gui-usability-and-artifact-browser/progress.md`), each with what it costs
if it is wrong. Twelve of them changed the design and are spec §12's deviation rows; four corrected
outright errors in the spec and were applied to it inline.

**If a task's text disagrees with a ruling, the ruling wins** — and say so in the task's report, so the
plan can be corrected rather than quietly diverged from. The rulings most likely to bite:

- **P1** — `set_note`'s, `read_log`'s and `delete`'s missing-artifact refusals carry `field="artifact"`.
  `field="note"` is only for refusals about the note's *text*.
- **P2** — `Summary` has a **sixth** new field, `batches_planned`; `rows` exists only once a cache
  finishes, so mid-run progress is `3/4 batches` with no row count.
- **P12/P13** — the table makes "incomplete last" a real sort property, and `int(Qt.SortOrder)` raises:
  use `.value`.
- **P26** — the worker does **not** blindly report a chained exception; see Task 22.
- **P38** — the stale panel counts are corrected by **Task 25 alone**.

---

## Task order and dependencies

```
 T1  PaneCapture both channels ─────────────────────────────┐ (test helper; independent)
 T2  field keys + require_note + the two tables ────────────┤
 T3  Summary grows            ── needs T2 (set_note field)  │
 T4  read_log                 ── needs T3                   │
 T5  remove_incomplete/sweep  ── needs T3                   │
 T6  report.py renderers      ── needs T3                   │
 T7  shared refusal box ────────────────────────────────────┤
 T8  ArtifactTable + QSS      ── needs T3                   │
 T9  the screen, registered, listing        ── T3 T8         │
 T10 the detail pane + Save                 ── T4 T6 T9      │
 T11 note / delete / sweep / picker refresh ── T2 T5 T7 T9   │
 T12 the lineage report action           ── T6 T7 T10 T11    │
 T13 tool: artifacts list / show            ── T3 T4 T6 T9   │
 T14 tool: note / rm / sweep / summary   ── T2 T5 T6 T9 T13  │
 T15 RUN_STATE + the shell's banner ────────────────────────┤
 T16 the tile and tab markers               ── T15           │
 T17 session_contents + the Config line ────────────────────┤
 T18 Apply confirms                         ── T17           │
 T19 the pickers say what they hold         ── T3            │
 T20 core/logging_root.py + both front ends ────────────────┤
 T21 cancel_deferred + the commit ──────────────────────────┤
 T22 the rescue save + the reported crash   ── T21           │
 T23 a new probe row starts blank ──────────────────────────┤
 T24 the two observation payload guards ────────────────────┘
 T25 stale counts and the M1b docstrings    ── T15 T18
 T26 the documents, then the slow set and the GPU gate ── everything
```

**Files two or more tasks touch** (read the other task's diff before you start, and never assume a
line number that another task may have moved):

| file | tasks |
|---|---|
| `core/artifacts/store.py` | T2, T3, T4, T5, T24 |
| `core/artifacts/__init__.py` | T3, T6 |
| `core/gui/design.py` | T8, T15 |
| `core/gui/main_window.py` | T9, T11, T15, T16, T25 (T25 owns the CLASS docstring; no earlier task edits it) |
| `core/gui/panels/base_panel.py` | T7, T15, T25 (only T25 edits the stale counts) |
| `core/gui/panels/inference/config_tab.py` | T17, T18 |
| `core/gui/screens/artifact_screen.py` (new) | T9, T10, T11, T12 |
| `core/gui/screens/home_screen.py` | T9, T16 |
| `core/gui/screens/inference_screen.py` | T16, T17, T18 (T18 owns its module docstring) |
| `core/gui/screens/nav_shell.py` | T15, T16 |
| `core/tool/__init__.py` | T13, T20 |
| `core/tool/browse.py` (new) | T13, T14 |
| `tests/_fixtures.py` | T1, T9 |
| `tests/test_artifact_browser.py` (new) | T9, T10, T11, T12 |
| `tests/test_artifact_store.py` | T2, T3, T4, T5, T6, T24 |
| `tests/test_nav_and_gating.py` | T1, T2, T15, T16, T17, T18, T19, T23, T25 |
| `tests/test_refusals.py` | T2, T20 |
| `tests/test_tool.py` | T13, T14, T20 |
| `tests/test_user_sbi.py` | T1, T22 |
| `tests/test_worker_dispatch.py` | T1, T7, T8, T9, T20, T21, T22 |

Every other file in the piece is touched by exactly one task: `core/refusals.py`, `core/gui/fields.py`
and `core/tool/fields.py` (T2), `core/artifacts/report.py` (T6), `core/gui/widgets/refusal_box.py`
(T7), `core/gui/widgets/artifact_table.py` (T8), `core/gui/screens/section_screen.py` (T16),
`PRISM_HANDOFF.md` (T18), `tests/conftest.py` (T11 — Q10 left its one sentence to that task alone),
`core/gui/widgets/artifact_picker.py` and the three inference picker tabs
(T19), `core/logging_root.py`, `core/gui/app.py` and `tests/test_user_models.py` (T20), `core/runs.py`,
`core/gui/streams.py` and `core/SBI/training_checkpoint.py` (T21), `core/SBI/pipeline.py` and
`core/gui/worker.py` (T22), `core/gui/widgets/labeled_inputs.py`, `core/gui/panels/inference/rows.py`
and `core/gui/panels/inference/infer_tab.py` (T23), `core/gui/panels/inference/base.py` (T25), and the
documents (T26).

---

# Piece 4 implementation plan — Tasks 1 and 2

Two preparatory tasks. Task 1 closes the test-helper blind spot that would otherwise make every later
piece-4 pane assertion half-blind; Task 2 puts the two new refusal field keys and the note rule in
place, which the browser screen and the `artifacts` subcommand both build on.

Both tasks are test-first. The implementer runs only the focused commands written in the steps. The
one-process fast gate (`pytest -m "not slow"`) is run by the session that owns the piece, after the
commit — never in a step.

The interpreter is always `C:\Users\J\anaconda3\envs\biophys-env\python.exe`. The `python` on PATH is
a different interpreter and has none of the dependencies.

---

### Task 1: `PaneCapture` sees the worker's lines too

**Files:**
- Modify: `tests/_fixtures.py:100-110` (the `PaneCapture` class)
- Test: `tests/test_worker_dispatch.py:359-377` (rewritten) plus one new test after it
- Modify: `tests/test_vt_progress.py:131-148` (the drained `log_batch` slot at `:138`)
- Modify: `tests/test_vt_progress.py:625-628` (the raw-payload slot at `:627`)
- Modify: `tests/test_user_sbi.py:4241-4244` (the raw-payload slot at `:4243`)
- Modify: `tests/test_nav_and_gating.py:1085-1087, 1137-1141` (a hand-rolled pane stub)
- Modify: `tests/test_nav_and_gating.py:2447-2457` (a hand-rolled pane stub)

**Interfaces:**
- Consumes: nothing from any other task. It reads only what is already there —
  `LogPane.append_line(text, level="info")` and `LogPane.append_lines(batch)`
  (`core/gui/widgets/log_pane.py:21-30`), the two connections in `BasePanel.dispatch`
  (`core/gui/panels/base_panel.py:219-221`), and `WorkerSignals.log_batch = Signal(object)`, whose
  payload is `list[(text, level)]` (`core/gui/worker.py:27`).
- Produces: `tests/_fixtures.PaneCapture(panel)` with `.lines: list[tuple[str, str]]` covering **both**
  pane channels in one ordered list, `(level, text)` per entry. Every later piece-4 task that asserts
  on what a panel showed (the run indicator, the Apply confirmation, the browser's refusal box) uses
  this and nothing else.

**Why this task exists:** `PaneCapture` replaces `panel.log_pane.append_line` only
(`tests/_fixtures.py:107`), so a test using it sees the panel's own GUI-thread messages and is blind to
`signals.log_batch` → `append_lines` — which is the pump's channel and therefore carries everything the
worker produced: every `print`, every retired tqdm bar, every `core` record through
`streams._PumpLogHandler`, and every Python warning through `redirect_streams`' `showwarning`. A test
that asserts "the pane said X" can therefore pass while the user's pane says something else entirely.
Spec §7.4, decision **B17**: "`PaneCapture` wraps **both** pane channels into one ordered list, so a
test sees what a user sees."

- [ ] **Step 1: Write the failing tests**

In `tests/test_worker_dispatch.py`, replace lines 359-377 (the whole
`test_pane_capture_records_level_and_text_and_the_pane_stays_blank` function) with the two functions
below. Nothing else in that file changes; `time`, `os`, `BasePanel`, `qt_app` and `pump` are already
imported at the top of the module (lines 17-38).

```python
def test_pane_capture_records_level_and_text_and_the_pane_stays_blank():
    """The window's refusals and warnings are asserted OFF THE PANE. PaneCapture stands in for BOTH of
    ``log_pane``'s channels on one panel -- ``append_line``, the panel's own messages, and
    ``append_lines``, one pump tick of the run's output -- and keeps ``(level, text)`` in order, with
    LogPane's own default level, so the ``lambda text, kind="": lines.append((kind, text))`` stub that
    five tests each wrote by hand becomes one helper that cannot drift from the real signatures.

    ONE list, in the order the lines happened, because that is the order the user reads them in: a
    batch's pairs arrive as ``(text, level)`` (WorkerSignals.log_batch, worker.py:27) and are flipped
    into the helper's ``(level, text)`` as they are appended. An empty batch appends nothing, exactly
    as LogPane.append_lines renders nothing for one."""
    from tests._fixtures import PaneCapture, qt_app

    qt_app()

    class P(BasePanel):
        pass

    panel = P()
    before = panel.log_pane.toPlainText()
    cap = PaneCapture(panel)
    panel.log_pane.append_line("plain")
    panel.log_pane.append_lines([("Config built: NADROWSKI", "info"),
                                 ("[tsnpe] loaded a NON-AMORTIZED posterior", "warning")])
    panel.log_pane.append_line("watch out", "warning")
    panel.log_pane.append_lines([])
    panel.log_pane.append_lines(None)
    assert cap.lines == [("info", "plain"),
                         ("info", "Config built: NADROWSKI"),
                         ("warning", "[tsnpe] loaded a NON-AMORTIZED posterior"),
                         ("warning", "watch out")]
    assert panel.log_pane.toPlainText() == before, "a captured line must not also reach the widget"


def test_pane_capture_sees_the_pumps_batch_channel_through_a_dispatch():
    """B17 end to end, through the machinery the helper exists for. ``BasePanel.dispatch`` resolves
    ``self.log_pane.append_line`` AND ``self.log_pane.append_lines`` at CONNECT time
    (base_panel.py:219-220), so a capture installed before the dispatch sees both; the batch channel is
    the only way a print, a retired tqdm bar, a ``core`` record or a Python warning from the WORKER
    reaches the pane, and before this task it went to the widget and nowhere a test could read it.

    A batch is published by the pump's daemon thread at 15 Hz as a QUEUED cross-thread signal, so the
    Qt loop has to be driven before asserting -- ``pump(app)``, as every other worker test here does.
    The two lines must also keep their order relative to each other: records and prints share one
    ordered sink (streams._Pump)."""
    import logging

    from tests._fixtures import PaneCapture, qt_app

    app = qt_app()

    class P(BasePanel):
        pass

    panel = P()
    cap = PaneCapture(panel)                     # BEFORE the dispatch, or neither channel is captured
    before = panel.log_pane.toPlainText()

    def work():
        print("Config built: NADROWSKI")
        logging.getLogger("core.tests.pane_capture").warning("[tsnpe] loaded a NON-AMORTIZED posterior")
        return "done"

    got = []
    panel.dispatch(work, on_result=got.append)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and (panel._busy or not got):
        app.processEvents()
        time.sleep(0.01)
    pump(app, 0.3)

    assert got == ["done"], got
    assert ("info", "Config built: NADROWSKI") in cap.lines, cap.lines
    assert ("warning", "[tsnpe] loaded a NON-AMORTIZED posterior") in cap.lines, cap.lines
    texts = [t for _level, t in cap.lines]
    assert texts.index("Config built: NADROWSKI") < \
        texts.index("[tsnpe] loaded a NON-AMORTIZED posterior"), cap.lines
    assert panel.log_pane.toPlainText() == before, \
        "a captured batch must not also reach the widget"
```

- [ ] **Step 2: Run them and watch them fail**

Run:
`& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest "tests/test_worker_dispatch.py::test_pane_capture_records_level_and_text_and_the_pane_stays_blank" "tests/test_worker_dispatch.py::test_pane_capture_sees_the_pumps_batch_channel_through_a_dispatch" -v`

Expected: both FAIL.
- The first fails on `assert cap.lines == [...]`: the real `LogPane.append_lines` is still in place, so
  the batch renders into the widget and the captured list is only
  `[('info', 'plain'), ('warning', 'watch out')]`. (It would also raise `TypeError` on
  `append_lines(None)` — `LogPane.append_lines` returns early only for a falsy batch, which `None` is,
  so no: the `cap.lines` assertion is what fails.)
- The second fails on `assert ("info", "Config built: NADROWSKI") in cap.lines` with `cap.lines == []`.

- [ ] **Step 3: Wrap both channels in the helper**

In `tests/_fixtures.py`, replace the whole class at lines 100-110 with:

```python
class PaneCapture:
    """Records what a panel writes to its log pane, as ``(level, text)`` in order, INSTEAD of
    rendering it. ``PaneCapture(panel)`` replaces BOTH of that one pane's channels:

      * ``LogPane.append_line(text, level="info")`` -- the panel's own GUI-thread messages (a refusal,
        a warning, "Run cancelled.") and ``WorkerSignals.log``;
      * ``LogPane.append_lines(batch)`` -- ``WorkerSignals.log_batch``, one pump tick of the RUN's
        output, and therefore every print, every retired tqdm bar, every ``core`` record (through
        ``streams._PumpLogHandler``) and every Python warning (through ``redirect_streams``'
        ``showwarning``). That channel is the one the helper used to miss, so a test could assert on
        what the pane said while being blind to almost everything in it (B17).

    TWO ORDERS, ONE LIST. A captured entry is ``(level, text)``: that is what the hand-written stubs
    this helper replaced appended and what every assertion in the suites reads. A batch's pairs arrive
    as ``(text, level)``, because that is the order ``WorkerSignals.log_batch`` carries
    (``core/gui/worker.py:27``; ``tests/test_vt_progress.py``'s handler test pins that payload
    directly). Each pair is therefore FLIPPED as it is appended, so ``.lines`` reads in one order --
    the order a user sees the lines in -- whichever channel each line came from. A test that wants one
    channel alone filters on what it knows only that channel says.

    ``BasePanel.dispatch`` resolves both attributes AT CONNECT TIME, so a capture installed before the
    dispatch sees both channels and one installed after sees neither. A batch is published by the
    pump's daemon thread at 15 Hz as a queued cross-thread signal, so a test must drive the event loop
    (``pump(app)``) before asserting on run output. The widget receives nothing, and the panel is a
    throwaway built by the test."""
    def __init__(self, panel):
        self.lines: list[tuple[str, str]] = []
        panel.log_pane.append_line = self._append
        panel.log_pane.append_lines = self._append_batch

    def _append(self, text: str, level: str = "info") -> None:
        self.lines.append((level, text))

    def _append_batch(self, batch) -> None:
        """One pump tick: each ``(text, level)`` appended as ``(level, text)``, in order. A falsy batch
        appends nothing, which is what ``LogPane.append_lines`` renders for one."""
        if not batch:
            return
        for text, level in batch:
            self.lines.append((level, text))
```

- [ ] **Step 4: Run the two tests again**

Run:
`& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest "tests/test_worker_dispatch.py::test_pane_capture_records_level_and_text_and_the_pane_stays_blank" "tests/test_worker_dispatch.py::test_pane_capture_sees_the_pumps_batch_channel_through_a_dispatch" -v`

Expected: PASS, both.

- [ ] **Step 5: Comment the three hand-rolled `log_batch` slots that stay**

`grep -n log_batch tests/*.py` finds seven connections in seven tests. Four of them
(`tests/test_vt_progress.py:186, 220, 247, 274`) connect the REAL `LogPane.append_lines` to a real
`LogPane` and assert on `toPlainText()` / `blockCount()` — they are about RENDERING (line splitting,
the `⚠` prefix, that no bar frame leaks), which is exactly what the helper does not do. They are left
untouched. The other three hand-roll a lambda and each gets one comment naming the payload order, so
nobody flips them to the helper's order by accident:

`tests/test_vt_progress.py:138`, inside `test_pump_never_exposes_the_transient_row_guess` — replace
that one line with:

```python
    # the payload is (text, level); this test asserts only on `rows` and drains the log channel
    signals.log_batch.connect(lambda _b: None)
```

`tests/test_vt_progress.py:627`, inside
`test_the_window_handler_feeds_the_pane_at_the_records_level_and_checks_cancel` — replace that one line
with:

```python
    # the RAW payload, deliberately: this is the pin on log_batch's own (text, level) order, which
    # tests/_fixtures.PaneCapture flips into (level, text). Do not move this test onto the helper.
    signals.log_batch.connect(lambda batch: lines.extend(batch))
```

`tests/test_user_sbi.py:4243`, inside
`test_the_mem_and_wait_lines_reach_the_window_plain_and_the_wait_still_checks_cancel` — replace that
one line with:

```python
    # the RAW payload: the assertions below unpack (text, level), log_batch's own order, not
    # PaneCapture's (level, text)
    signals.log_batch.connect(lambda batch: lines.extend(batch))
```

- [ ] **Step 6: Move the two hand-rolled `append_line` stubs onto the helper**

These are the two remaining stubs in the suites that build their own pane slot only to read pane lines,
and both are written with a signature that has already drifted from the real one
(`lambda text, kind=""` against `LogPane.append_line(text, level="info")`) — precisely the drift the
helper exists to prevent. `PaneCapture.lines` is the same list object the capture appends to, so
binding `lines = pane.lines` keeps every assertion below them reading unchanged.

In `tests/test_nav_and_gating.py`, in `test_a_confirmed_near_miss_dispatches_new_run`, add to the
import block at lines 1085-1087:

```python
    from tests._fixtures import PaneCapture
```

and replace lines 1137-1138 with:

```python
    pane = PaneCapture(pp)              # both pane channels, as (level, text) -- the stub's own order
    lines = pane.lines
```

In the same file, in `test_the_probe_planner_refuses_a_blank_t_obs_through_the_yellow_box`, change line
2449 to:

```python
    from tests._fixtures import PaneCapture, qt_app
```

and replace lines 2455-2457 with:

```python
    refused = []
    panel._refusal = lambda exc: refused.append(exc)
    pane = PaneCapture(panel)           # both pane channels, as (level, text) -- the stub's own order
    lines = pane.lines                  # the list the capture appends to, so plan()'s clear() still works
```

- [ ] **Step 7: Run the task's own tests**

Run:
`& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_worker_dispatch.py tests/test_vt_progress.py tests/test_simulate.py "tests/test_user_sbi.py::test_the_mem_and_wait_lines_reach_the_window_plain_and_the_wait_still_checks_cancel" "tests/test_nav_and_gating.py::test_a_confirmed_near_miss_dispatches_new_run" "tests/test_nav_and_gating.py::test_the_probe_planner_refuses_a_blank_t_obs_through_the_yellow_box" "tests/test_nav_and_gating.py::test_a_rename_failure_reads_as_a_name_refusal" "tests/test_nav_and_gating.py::test_the_inference_tabs_route_builder_failures_by_kind_and_no_longer_call_config_error" "tests/test_nav_and_gating.py::test_the_chi_drive_and_band_are_read_only_and_the_draft_carries_config" -q`

Expected: PASS. The last five node ids and `tests/test_simulate.py` are every existing `PaneCapture`
caller (`tests/test_nav_and_gating.py:1473, 1526, 1659`, `tests/test_simulate.py:322`,
`tests/test_worker_dispatch.py:373, 400, 449`): each of them stubs `dispatch`, so none has batch traffic
and none of their `pane.lines[-1]` assertions can shift — this run is what proves it rather than
assuming it.

- [ ] **Step 8: Commit**

```bash
git add tests/_fixtures.py tests/test_worker_dispatch.py tests/test_vt_progress.py tests/test_user_sbi.py tests/test_nav_and_gating.py
git commit -m "tests: PaneCapture records both pane channels in one ordered list"
```

---

### Task 2: the two new field keys, the note rule, and the tables

**Files:**
- Modify: `core/refusals.py:60-123` (the `FIELDS` registry tuple) and `core/refusals.py:232` (append
  `NOTE_MAX_CHARS` and `require_note` after `require_file`)
- Modify: `core/gui/fields.py:12-21` (the docstring's three shapes) and `core/gui/fields.py:79-82`
  (the "consents and names" block of `CONTROL`)
- Modify: `core/tool/fields.py:14-17` (the docstring's `None` sentence) and `core/tool/fields.py:54-57`
  (the "consents and names" block of `FLAG`)
- Modify: `core/artifacts/store.py:443-449` (`set_note`)
- Test: `tests/test_refusals.py:20-21, 28-39, 88-104, 698-705` and one new test function
- Test: `tests/test_nav_and_gating.py:1403-1420` (the `(c) verbatim` sentence block of
  `test_every_field_key_has_a_window_control_and_the_fix_sentences_name_it`, whose `def` is at `:1342`)
- Test: `tests/test_artifact_store.py:276-277` (one new test after
  `test_rename_keeps_the_id_and_dependents_resolve`)

**Interfaces:**
- Consumes: the module's own existing shape — `Refusal(message, *, field=None)`, `_what(key)`
  (`core/refusals.py:132-135`) and `_default_clause(key)` (`:138-141`), and `StoreError`
  (`core/artifacts/store.py:41-42`), which is a `Refusal` subclass and takes the same `field=`.
- Produces, for every later piece-4 task:
  - `core.refusals.NOTE_MAX_CHARS = 200`
  - `core.refusals.require_note(key: str, text: str) -> str`
  - `core.refusals.FIELDS["artifact"] == Field("artifact", "the artifact", None)` and
    `FIELDS["note"] == Field("note", "the note", None)` (61 → 63 entries)
  - `core.gui.fields.fix_sentence("artifact") == "Select an artifact in the list on the Artifacts screen."`
    and `fix_sentence("note") == "Edit it in the Note box on the Artifacts screen."`
  - `core.tool.fields.FLAG["artifact"] is None`, `FLAG["note"] == "--note"`
  - `ArtifactStore.set_note`'s "no complete artifact" `StoreError` carries `field="artifact"`

**Why this task exists:** `set_note` is the one store mutation whose refusal carries no field key
(`core/artifacts/store.py:446`), and there is no rule anywhere for the text of a note — so the browser
screen and the `artifacts note` subcommand would each have to invent their own limit and their own
clamp. Spec §2.4 and §2.5, decision **B5**: "A note is **one line**, trimmed, at most **200
characters**; blank clears it. A note containing a newline or over the limit is **refused** (V2: no
clamp, no silent default)." The registry's count pin moves 61 → 63 with the two keys.

**Which key that refusal takes.** `set_note`'s refusal is `field="artifact"`, NOT `field="note"`: the
sentence is about a MISSING ARTIFACT, so the note key would send the operator to the Note box when the
fix is to select an artifact that exists. `field="note"` belongs to `require_note`'s own refusals —
the text rules, where the note itself is what is wrong. The window's `artifact` entry therefore reads
"Select an artifact in the list on the Artifacts screen.", and `FLAG["artifact"]` is `None`, so the
tool's `refused:` line simply ends at the message with no trailing parenthetical.

- [ ] **Step 1: Write the failing test for the note rule**

In `tests/test_refusals.py`, extend the module import at lines 20-21 to:

```python
from core.refusals import (FIELDS, NOTE_MAX_CHARS, Field, Refusal, describe, refuse, require_at_least,
                           require_between, require_choice, require_file, require_finite, require_given,
                           require_note, require_positive)
```

and add this function between the end of
`test_require_file_refuses_a_blank_and_a_missing_path_naming_the_input_kind` (line 285) and the `def` of
`test_refuse_appends_the_default_clause_to_the_callers_sentence_and_binds_the_field` (line 288):

```python
def test_require_note_trims_one_line_and_refuses_a_break_or_the_limit():
    """B5 (design §2.4). A note is ONE line, at most NOTE_MAX_CHARS characters, and blank clears it.

    Surrounding whitespace is TRIMMED -- the one transformation this module allows, because it changes
    no meaning -- and everything else is refused rather than fixed (V2): a newline, a carriage return
    or a tab inside the text refuses, and so does a note over the limit, whose sentence gives BOTH the
    limit and the length given so the operator knows how much to cut. An all-whitespace note is not a
    refusal: it is how a note is CLEARED, and it comes back as "". The limit lives here and not in
    config.py: this module is torch-free on purpose and a note limit is not a science constant."""
    assert NOTE_MAX_CHARS == 200
    assert require_note("note", "  kept for the paper  ") == "kept for the paper"
    assert require_note("note", "kept") == "kept"
    assert require_note("note", "") == ""
    assert require_note("note", "   ") == "", "an all-whitespace note clears it"
    assert require_note("note", " \t \n ") == "", "so does one that is only a break"
    assert require_note("note", None) == "", "a blank box clears it too"
    assert require_note("note", "x" * NOTE_MAX_CHARS) == "x" * NOTE_MAX_CHARS
    assert require_note("note", "  " + "x" * NOTE_MAX_CHARS + "  ") == "x" * NOTE_MAX_CHARS, \
        "the length is measured AFTER the trim"
    with pytest.raises(Refusal) as e:
        require_note("note", "x" * (NOTE_MAX_CHARS + 1))
    assert _shape(e.value, "note") == "The note must be at most 200 characters; got 201."
    for bad in ("a\nb", "a\rb", "a\tb"):
        with pytest.raises(Refusal) as e:
            require_note("note", bad)
        assert _shape(e.value, "note") == f"The note must be one line; got {bad!r}."
    assert e.value.field == "note", "the key the caller passed travels on the refusal"
```

- [ ] **Step 2: Run it and watch it fail**

Run:
`& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest "tests/test_refusals.py::test_require_note_trims_one_line_and_refuses_a_break_or_the_limit" -v`

Expected: FAIL at collection —
`ImportError: cannot import name 'NOTE_MAX_CHARS' from 'core.refusals'` (the whole module fails to
import, so every test in the file errors).

- [ ] **Step 3: Add the limit and the rule**

Append to `core/refusals.py`, after `require_file` (i.e. after line 232), keeping the file's one blank
line between top-level definitions:

```python
# The note limit, HERE and not in core/config.py: this module imports only the standard library on
# purpose (see the module docstring), the registry's own defaults are already literals pinned against
# config.py by tests/test_refusals.py, and a one-line description's length is not a science constant.
NOTE_MAX_CHARS = 200


def require_note(key: str, text: str) -> str:
    """The trimmed note, or a Refusal. One LINE, at most NOTE_MAX_CHARS characters; "" clears it.

    Surrounding whitespace is trimmed -- the one transformation this module allows, because it changes
    no meaning -- and everything else is refused rather than fixed (V2): a line break or a tab left
    INSIDE the text refuses, and so does a note over the limit, whose message gives the limit and the
    length given. A blank box (None) and an all-whitespace note are not refusals: both mean "clear it"
    and both come back as "", which is what ``ArtifactStore.set_note`` writes for no note.
    """
    trimmed = "" if text is None else str(text).strip()
    if not trimmed:
        return ""
    if any(ch in trimmed for ch in "\n\r\t"):
        raise Refusal(f"{_what(key)} must be one line; got {trimmed!r}{_default_clause(key)}.", field=key)
    if len(trimmed) > NOTE_MAX_CHARS:
        raise Refusal(f"{_what(key)} must be at most {NOTE_MAX_CHARS} characters; got "
                      f"{len(trimmed)}{_default_clause(key)}.", field=key)
    return trimmed
```

- [ ] **Step 4: Run it and watch it fail differently**

Run:
`& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest "tests/test_refusals.py::test_require_note_trims_one_line_and_refuses_a_break_or_the_limit" -v`

Expected: FAIL with `KeyError: 'note'` raised from `describe` inside `_what` — the rule works, but the
registry does not know the key yet, which is the next step. (`_shape` would fail the same way through
`describe(key)`.)

- [ ] **Step 5: Move the registry pin to 63 and pin the two entries**

In `tests/test_refusals.py`, add the two keys to the closed `BASE_KEYS` tuple by replacing line 36
(`"chi_max_cycles", "chi_f0", "chi_freq_bounds", "device", "model", "observation", "posterior", "prior",`)
with:

```python
    "chi_max_cycles", "chi_f0", "chi_freq_bounds", "device", "model", "observation", "posterior", "prior",
    "artifact", "note",
```

and in `test_the_registry_holds_exactly_the_initial_keys_with_neutral_descriptions` change line 89 to:

```python
    assert len(FIELDS) == len(BASE_KEYS) + len(TOOL_ONLY_KEYS) == 63, "a key is listed twice above"
```

and add, after line 100 (`assert FIELDS["new_run"] == Field(...)`):

```python
    # piece 4's two: the artifact the browser acts on, and its note. Both descriptions obey the
    # wording ban above (the regex on f.what), and neither has a default -- a note has no default
    # text and an artifact is chosen, not defaulted.
    assert FIELDS["artifact"] == Field("artifact", "the artifact", None)
    assert FIELDS["note"] == Field("note", "the note", None)
```

- [ ] **Step 6: Move the tool table's `None`-value set from five keys to six**

The count pin is not the only closed literal the two keys move. `tests/test_refusals.py:698-699`, in
`test_every_field_key_has_a_flag_and_every_key_literal_under_core_is_registered`, pins the set of keys
whose flag is `None` EXACTLY, and it reads today:

```python
    assert {k for k, f in tool_fields.FLAG.items() if f is None} == {
        "units", "chi_k_pad", "chi_max_cycles", "chi_f0", "chi_freq_bounds"}
```

`artifact` is the sixth (the `artifacts` subcommand names it positionally; `note` has a flag and does
not belong here). Replace those two lines with:

```python
    assert {k for k, f in tool_fields.FLAG.items() if f is None} == {
        "units", "chi_k_pad", "chi_max_cycles", "chi_f0", "chi_freq_bounds", "artifact"}
```

This is a test edit, so it goes in before the table does: the assertion stands red until Step 10 adds
`"artifact": None`, and it is behind the key-set assertion at `:681` that Step 9 watches fail first.

- [ ] **Step 7: Run the registry test and watch it fail**

Run:
`& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest "tests/test_refusals.py::test_the_registry_holds_exactly_the_initial_keys_with_neutral_descriptions" -v`

Expected: FAIL on `assert set(FIELDS) == set(BASE_KEYS) | set(TOOL_ONLY_KEYS)` — the sets differ by
`{'artifact', 'note'}`.

- [ ] **Step 8: Add the two `Field` entries to the registry**

In `core/refusals.py`, insert after line 110 (`Field("prior", "the prior", None),`) and before the
tool-only comment at line 111:

```python
    # the artifact browser (piece 4): the artifact a browse action acts on, and its note
    Field("artifact", "the artifact", None),
    Field("note", "the note", None),
```

- [ ] **Step 9: Run the whole refusals suite and watch the tool table fail**

Run:
`& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_refusals.py -q`

Expected: the two note/registry tests PASS;
`test_every_field_key_has_a_flag_and_every_key_literal_under_core_is_registered` FAILS on
`assert set(tool_fields.FLAG) == set(FIELDS)` with `only in FIELDS: ['artifact', 'note']`. That is the
first assertion in the test (`:681`); the `None`-value set Step 6 moved (`:698-699`) is behind it and
goes green in the same step as this one.

- [ ] **Step 10: Add both entries to the tool's table**

In `core/tool/fields.py`, replace the docstring sentence at lines 14-17 with:

```
``None`` marks a key no option string answers: the units (the tool declares none), the four chi
constants only ``config.py`` sets (``chi_k_pad``, ``chi_max_cycles``, ``chi_f0``,
``chi_freq_bounds``), and ``artifact``, which the ``artifacts`` subcommand names POSITIONALLY
(``artifacts show <kind> <ref>``) -- there is no flag to print, and the refusal's own sentence already
quotes the ref. One repeatable flag can carry several keys: ``--forced PATH[@HZ]`` is both the
driven recording and a chi probe's, ``--drive NAME=VALUE`` is the amplitude, frequency and phase.
```

and add to the "consents and names" block, after line 57 (`"name": "--name",`):

```python
    # the artifact browser: `artifacts note <kind> <ref> --note TEXT`. The note text is a FLAG (and
    # not a positional) precisely so this table can name one; the artifact itself is positional.
    "artifact": None,
    "note": "--note",                                   # add_name_flags defines it beside --name
```

- [ ] **Step 11: Pin both entries in the tool-table test**

The `None`-value set is already done (Step 6, and its replacement was two lines for two, so the line
numbers below have not moved). What is left is the pair of value pins. In `tests/test_refusals.py`, in
`test_every_field_key_has_a_flag_and_every_key_literal_under_core_is_registered`, add after line 705
(the `--max-epochs` / `--run-size` line):

```python
    # piece 4: the note is a flag so this table can name one (config_args.add_name_flags), and the
    # artifact is positional, so its entry is None and fix_sentence adds NOTHING -- set_note's
    # "no complete <kind> artifact named or id'd ..." refusal carries field="artifact", and the
    # ladder's line for it therefore ends at the message, with no trailing parenthetical
    assert tool_fields.FLAG["note"] == "--note" and tool_fields.fix_sentence("note") == "(--note)"
    assert tool_fields.FLAG["artifact"] is None and tool_fields.fix_sentence("artifact") == ""
```

- [ ] **Step 12: Run the refusals suite**

Run:
`& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_refusals.py -q`

Expected: PASS. (`--note` is already a real option string on every stage subcommand —
`core/tool/config_args.py:54`, `add_name_flags` — so the "every flag is one `build_parser` defines"
assertion at lines 693-696 is satisfied before the `artifacts` subcommand exists.)

- [ ] **Step 13: Run the window's control-table test and watch it fail**

Run:
`& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest "tests/test_nav_and_gating.py::test_every_field_key_has_a_window_control_and_the_fix_sentences_name_it" -v`

Expected: FAIL on `assert set(gui_fields.CONTROL) == set(FIELDS)`
(`tests/test_nav_and_gating.py:1370`, inside the test whose `def` is at `:1342`) with
`only in FIELDS: ['artifact', 'note']`.

- [ ] **Step 14: Add both entries to the window's table**

In `core/gui/fields.py`, replace the second bullet of the docstring's three shapes (lines 17-18) with:

```
* one sentence, for a consent, a dialog, a table, the name box, a value fixed by measurement, or a
  control on the Artifacts screen, which is a screen of its own and not an inference tab; the
  sentence quotes the control's own text ("Run on a different observation").
```

and add to the "consents and names" block, after line 82 (`"name": "Choose another name in the Save
box.",`):

```python
    # the artifact browser (piece 4). Sentences, not (tab, label): the browser is a fifth Home tile,
    # not an inference tab, and the tuple shape is pinned against InferenceScreen's tab titles.
    "artifact": "Select an artifact in the list on the Artifacts screen.",
    "note": "Edit it in the Note box on the Artifacts screen.",
```

- [ ] **Step 15: Pin the two sentences verbatim**

In `tests/test_nav_and_gating.py`, add after line 1420 (the end of the `fix_sentence("chi_f0")`
assertion, the last line of the `(c) verbatim` block, which starts at `:1403`):

```python
    # piece 4's two, on the Artifacts screen rather than a tab (B5, design §2.5)
    assert gui_fields.fix_sentence("artifact") == "Select an artifact in the list on the Artifacts screen."
    assert gui_fields.fix_sentence("note") == "Edit it in the Note box on the Artifacts screen."
```

- [ ] **Step 16: Write the failing test for `set_note`'s field key**

In `tests/test_artifact_store.py`, insert this function between
`test_rename_keeps_the_id_and_dependents_resolve` (ends at line 275) and
`test_delete_refuses_naming_dependents_and_force_deletes` (line 278):

```python
def test_set_note_refuses_an_unknown_ref_with_the_artifact_field(store):
    """Design §2.4. ``set_note`` was the store mutation whose refusal carried no field key, so the
    front ends had nothing to look up and the yellow box came up with no "where to fix it" line under
    it. It carries a key now, like every other pre-spend refusal (piece 3, V3).

    The key is ``"artifact"``, NOT ``"note"``: this sentence is about a MISSING ARTIFACT, so the note
    key would send the operator to the Note box when the fix is to select an artifact that exists.
    ``"note"`` stays for ``core.refusals.require_note``'s own refusals, where the note text is what is
    wrong. The window's entry for ``artifact`` is "Select an artifact in the list on the Artifacts
    screen." and the tool has no flag for it (``FLAG["artifact"] is None``), so the tool's line ends at
    the message with no trailing parenthetical -- both pinned in ``tests/test_refusals.py`` and
    ``tests/test_nav_and_gating.py`` by the steps above. The sentence itself is unchanged and still
    names no box and no flag."""
    c = _make(store, name="cal")
    assert store.set_note("calibration", c.id, "kept for the paper").note == "kept for the paper"
    assert store.get("calibration", c.id).note == "kept for the paper"
    with pytest.raises(st.StoreError) as e:
        store.set_note("calibration", "nope", "x")
    assert str(e.value) == "no complete calibration artifact named or id'd 'nope'"
    assert e.value.field == "artifact", "a store refusal without a field key cannot name a control"
    assert isinstance(e.value, Refusal)
```

- [ ] **Step 17: Run it and watch it fail**

Run:
`& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest "tests/test_artifact_store.py::test_set_note_refuses_an_unknown_ref_with_the_artifact_field" -v`

Expected: FAIL on `assert e.value.field == "artifact"` — it is `None` today
(`core/artifacts/store.py:446` raises `StoreError` with no `field=`).

- [ ] **Step 18: Give `set_note`'s refusal its field key**

In `core/artifacts/store.py`, replace lines 443-449 with:

```python
    def set_note(self, kind: str, ref: str, note: str) -> mf.Manifest:
        """Rewrite one artifact's note. ``field="artifact"`` (piece 4, design §2.4): the front ends
        name the control or the flag themselves, and this was the one store mutation that named
        nothing. The key is the ARTIFACT and not the note: what is wrong is the ref, so the fix is to
        select an artifact that exists, not to edit the note box. ``field="note"`` is
        ``core.refusals.require_note``'s, for a note whose TEXT is refused.

        The text is not judged here -- ``require_note`` is that rule, and both front ends run it
        before they call this -- so a caller that reaches past them writes what it passes.
        """
        sub, m = self._find(kind, ref)
        if m is None:
            raise StoreError(f"no complete {kind} artifact named or id'd {ref!r}", field="artifact")
        m.note = str(note)
        _write_manifest(sub, m)
        return m
```

- [ ] **Step 19: Run the task's own tests**

Run:
`& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_refusals.py "tests/test_artifact_store.py::test_set_note_refuses_an_unknown_ref_with_the_artifact_field" "tests/test_artifact_store.py::test_rename_keeps_the_id_and_dependents_resolve" "tests/test_nav_and_gating.py::test_every_field_key_has_a_window_control_and_the_fix_sentences_name_it" "tests/test_nav_and_gating.py::test_the_gui_control_table_matches_the_tabs_labels" -q`

Expected: PASS. (The second nav test reads the built screen's form rows back and only looks at
`(tab, label)` entries, so the two new sentence entries must leave it green — this run is the proof.)

- [ ] **Step 20: Commit**

```bash
git add core/refusals.py core/gui/fields.py core/tool/fields.py core/artifacts/store.py tests/test_refusals.py tests/test_nav_and_gating.py tests/test_artifact_store.py
git commit -m "refusals: the artifact and note field keys, the note rule, both tables"
```

### Task 3: `Summary` grows, and the listing stops throwing away what it read

**Files:**
- Modify: `core/artifacts/store.py:60-80` (the `Summary` dataclass: one sentence on the `complete` comment, six new keyword fields after `parents`)
- Modify: `core/artifacts/store.py:338-350` (`ArtifactStore.list` fills them)
- Modify: `core/artifacts/__init__.py:5-8` (one name added to the `.store` re-export list: `KIND_DIRS`)
- Test: `tests/test_artifact_store.py` — one new helper inserted at `:149` (between `_cal_body` at `:147-148` and `_make` at `:151`), the existing `bodies = {...}` literal at `:161-171` replaced by a call to it, and four new tests appended at the end of the file (after the last line, `assert "setLevel" not in src`, which the Read tool numbers `3238`; the file is CRLF — keep it that way)

**Interfaces:**
- Consumes: nothing from an earlier piece-4 task. The store exactly as it stands: `ArtifactStore._entries(kind) -> list[tuple[Path, Manifest | None, str | None]]`, `ArtifactStore.list(kind) -> list[Summary]`, `core.artifacts.store.write_simulation_manifest(path, identity, *, parents=None, inputs=None, hw=None, batches_done=0, complete=False, rows=None, V=None) -> Manifest`, and `core.SBI.training_checkpoint`'s `create / save / mark_complete / resolve_dir / identity_digest`.
- Produces, for every later task (the browser table, the pickers' summary line, the tool's `artifacts list`):
  - `Summary.dir_name: str` — the directory's own name, on **every** row
  - `Summary.finished: bool` — did the run finish
  - `Summary.batches_done: "int | None"`
  - `Summary.batches_planned: "int | None"` — the cache's PLANNED batch total, off the body's `identity["n_runs"]`
  - `Summary.rows: "tuple[int, ...] | None"`
  - `Summary.variant: "str | None"`
  - all six filled by `ArtifactStore.list(kind) -> list[Summary]`, which still reads each manifest exactly once.
  - `core.artifacts.KIND_DIRS` — the kind → directory-name map, re-exported from the package (P17), so no front end ever imports `core.artifacts.store` just for the list of kinds.
  - `tests/test_artifact_store.py::_bodies() -> dict` — one valid body per writer kind, a fresh dict per call.

**Why this task exists:** `ArtifactStore.list` reads every manifest and then throws almost all of it away: a row carries `mode`, `width`, `amortized` and `parents` and nothing else, so a browser row would have to `get()` the manifest a second time to show a cache's progress or a diagnostic's variant (spec §2.1, **B2**: "`Summary` grows so that a row needs no second manifest read"). Worse, `Summary.complete` is the only thing a caller can ask, and for the simulation cache it does **not** mean what it sounds like: `training_checkpoint.create` writes the manifest *before the first batch is simulated*, so a cache is `complete=True` from batch zero — the trap `docs/STATE.md` records as "`Summary.complete` for simulations means only 'has a manifest'". **B3** settles it: `complete` keeps its meaning and a separate, explicit `finished` answers "did the run finish".

`complete` is deliberately **not** redefined, because it is load-bearing in three places and all three want today's meaning: `core/gui/widgets/artifact_picker.py:139-140` (`StorePicker.refresh` does `if not s.complete: continue`, i.e. "only offer rows that have a manifest to load"), `core/artifacts/store.py:349` (`list`'s secondary sort, `rows.sort(key=lambda s: not s.complete)`, puts manifest-less directories last), and the store suite, which asserts on it in at least four places (`tests/test_artifact_store.py:175`, `:246`, `:250`, `:400`, `:767`). Redefining it would change all three for the sake of one word; adding the honest question beside it costs nothing.

**The sixth field, and why `rows` is not enough (P2).** `batches_planned` is the only one of the six
whose value a row could not otherwise reach: the planned total lives **only** inside the cache's own
identity, as `body["identity"]["n_runs"]` (`core/artifacts/identity.py:54` writes that key, and it is
one of the fields the directory-naming digest is taken over). Without it a row could not render the
spec's `3/4 batches` progress cell at all, because working the total out would mean reading the
manifest a second time — the one thing **B2** exists to prevent.

And `rows` cannot stand in for it, because `rows` is written by `mark_complete` **alone**:
`training_checkpoint.save` (`core/SBI/training_checkpoint.py:343`) calls
`_refresh_manifest(path, batches_done=batch_k)` and passes no rows, so every mid-run refresh leaves
`"rows": null` in the manifest and an unfinished cache legitimately has `rows=None`. Mid-run progress
is therefore `3/4 batches` with no row count, and the tests below assert exactly that at each stage of
a real checkpoint's life. This piece deliberately does **not** change the checkpoint commit to write
the row counts earlier: they would have to be threaded through `save`, and the commit order between
`state.pt`, the shards and the manifest is the last thing in this codebase that should move for the
sake of a display cell.

- [ ] **Step 1: Write the failing tests**

In `tests/test_artifact_store.py`, insert this helper immediately after `_cal_body` (which ends at line 148) and before `_make` (line 151):

```python
def _bodies():
    """One valid body per WRITER kind -- the six ``store.create`` accepts. A FRESH dict per call, so a
    test that hands one to the writer cannot leave a mutation behind for the next.

    The simulation kind is deliberately absent: its manifest has no writer at all (``store.create``
    refuses it outright) and ``write_simulation_manifest`` builds its body itself.
    """
    return {
        "prior": {"gmm": {"n_components": 2, "param_keys": ["a"],
                          "box": {"nd_lows": [0.0], "nd_highs": [1.0], "log_mask": [False]}},
                  "sweep": {}, "stability": {"accepted_sets": None, "iterations": 1}},
        "posterior": {"mode": "chi", "conditioning": {"width": 50}, "transform": {}, "amortized": True,
                      "truncation": None, "training": {}},
        "observation": {"mode": "chi", "conditioning": {"width": 50}, "x_obs_digest": "0" * 16,
                        "T_obs_cell": 1.0, "n_obs": 10, "forcing_vals": {}, "chi_obs_freqs": None,
                        "source": {"kind": "simulated"}},
        "calibration": _cal_body(), "inference": {"results": {"n_samples": 5}},
        "diagnostic": {"diagnostic": "sbc", "variant": None, "settings": {"repeats": 2},
                       "results": {"n_valid": 8}},
    }
```

Then, in `test_create_list_get_round_trip_per_kind`, replace the whole literal at lines 161-171 —

```python
    bodies = {
        "prior": {"gmm": {"n_components": 2, "param_keys": ["a"], "box": {"nd_lows": [0.0], "nd_highs": [1.0], "log_mask": [False]}},
                  "sweep": {}, "stability": {"accepted_sets": None, "iterations": 1}},
        "posterior": {"mode": "chi", "conditioning": {"width": 50}, "transform": {}, "amortized": True,
                      "truncation": None, "training": {}},
        "observation": {"mode": "chi", "conditioning": {"width": 50}, "x_obs_digest": "0" * 16, "T_obs_cell": 1.0,
                        "n_obs": 10, "forcing_vals": {}, "chi_obs_freqs": None, "source": {"kind": "simulated"}},
        "calibration": _cal_body(), "inference": {"results": {"n_samples": 5}},
        "diagnostic": {"diagnostic": "sbc", "variant": None, "settings": {"repeats": 2},
                       "results": {"n_valid": 8}},
    }
```

— with one line, so the six bodies have one home:

```python
    bodies = _bodies()
```

Now append these four tests at the end of the file:

```python
def test_a_summary_says_whether_the_run_finished_for_every_kind(store):
    """B3 (spec §2.1, §2.6). ``complete`` means "has a valid manifest"; ``finished`` means "the run
    finished". For six of the seven kinds they are the same fact -- ``ArtifactWriter._commit`` writes
    the manifest LAST, so a manifest exists only for a run that reached the end. For the simulation
    cache they differ: ``training_checkpoint.create`` manifests it BEFORE the first batch is
    simulated, and ``body["complete"]`` is the field that says whether its rows are all there.

    ``complete`` is NOT redefined: StorePicker.refresh skips rows without it, ``list``'s secondary
    sort puts them last, and this suite asserts on it. The honest question goes beside it instead.
    """
    from core import artifacts
    from core.SBI.training_checkpoint import identity_digest
    assert artifacts.KIND_DIRS is st.KIND_DIRS, \
        "re-exported from the package, so nothing outside the store imports the submodule for it"

    for kind, body in _bodies().items():
        w = _make(store, kind, name=f"n_{kind}", body=body)
        row = store.list(kind)[0]
        assert (row.id, row.complete, row.finished) == (w.id, True, True), kind
        assert (row.batches_done, row.batches_planned, row.rows) == (None, None, None), kind

    ident = {"format": "training-rows/2", "model": "X", "n_runs": 3, "prior_fingerprint": "c" * 16,
             "truncation": None}
    digest = identity_digest(ident)
    st.write_simulation_manifest(store.kind_dir("simulation") / digest, ident, batches_done=1)
    cache = store.list("simulation")[0]
    assert (cache.id, cache.complete) == (digest, True), "a cache is manifested from its first batch on"
    assert cache.finished is False, "... and it is not FINISHED until its rows are all there"
    assert (cache.batches_done, cache.batches_planned, cache.rows) == (1, 3, None), \
        "1 of the 3 batches the identity PLANS; the row counts land only at mark_complete"

    (store.kind_dir("inference") / "leftover__20260101T000000").mkdir(parents=True)
    leftover = [r for r in store.list("inference") if not r.complete]
    assert len(leftover) == 1 and leftover[0].finished is False, "no manifest, no finished run"


def test_a_cache_row_carries_its_progress_until_mark_complete_flips_it(store):
    """§2.6's second pin, over a real checkpoint's whole life: ``create`` manifests it at zero batches,
    ``save`` refreshes ``batches_done``, and ``mark_complete`` is the only thing that makes it
    finished. The rows-per-batch come back as a TUPLE, so a front-end formatter can sum them without
    caring that the manifest stores a JSON list.

    ``batches_planned`` comes off the identity and never moves, so the progress cell reads
    ``0/3``, ``2/3``, ``3/3`` over this run's life -- and the row count is None for the first two,
    because ``save`` passes no rows (P2)."""
    from core.SBI import training_checkpoint as tc
    ident = {"format": "training-rows/2", "model": "X", "n_runs": 3, "run_size": 4,
             "prior_fingerprint": "d" * 16, "truncation": None}
    d = tc.resolve_dir(ident)                      # the ``store`` fixture is the process default
    tc.create(d, ident, schedule_t_scales=torch.ones(3), schedule_Ts=torch.ones(3),
              inits=torch.zeros(1, 2), V=None, probe=torch.zeros(7, 2, dtype=torch.float64),
              run_size=4, n_runs=3, hw=config.cpu_device())
    fresh = store.list("simulation")[0]
    assert (fresh.complete, fresh.finished, fresh.batches_done, fresh.batches_planned,
            fresh.rows) == (True, False, 0, 3, None)
    tc.save(d, from_batch=0, batch_k=2, rng=None, x_buf=torch.zeros(12, 5), th_buf=torch.zeros(12, 2),
            run_size=4)
    mid = store.list("simulation")[0]
    assert (mid.complete, mid.finished, mid.batches_done, mid.batches_planned,
            mid.rows) == (True, False, 2, 3, None), "2 of 3, and no row count from a save"
    tc.mark_complete(d, 3, rows=(12, 5))
    done = store.list("simulation")[0]
    assert (done.complete, done.finished, done.batches_done, done.batches_planned) == (True, True, 3, 3)
    assert done.rows == (12, 5) and isinstance(done.rows, tuple), done.rows
    assert done.dir_name == d.name == done.id, "the cache's folder IS its identity digest"


def test_every_row_carries_its_own_directory_name(store, monkeypatch):
    """``dir_name`` is the folder, ALWAYS -- the handle ``remove_incomplete`` takes (§2.3). ``id``
    keeps exactly today's meaning: the manifest id for a complete row, the directory name for an
    incomplete one, because ``get()``'s refusal already lists incomplete directory names and
    ``StorePicker`` keys on ``id``.

    The two disagree on a COMPLETE row too, and the listing is the first place it shows: ``rename``
    writes the new name into the manifest FIRST and moves the directory second, tolerating a
    PermissionError on the move because the manifest is what resolves. A held handle therefore leaves
    a row whose manifest says ``renamed__<id>`` in a folder still called ``p__<id>``.
    """
    w = _make(store, "calibration", name="p")
    row = store.list("calibration")[0]
    assert row.dir_name == w.dir.name == f"p__{w.id}" and row.id == w.id

    (store.kind_dir("calibration") / "half_written__20260101T000000").mkdir(parents=True)
    incomplete = [r for r in store.list("calibration") if not r.complete][0]
    assert incomplete.dir_name == "half_written__20260101T000000" == incomplete.id

    def _held(self, target):
        raise PermissionError("a preview window holds the folder open")

    monkeypatch.setattr(Path, "rename", _held)     # _atomic_write uses os.replace, so the manifest still lands
    store.rename("calibration", w.id, "renamed")
    monkeypatch.undo()
    assert store.get("calibration", w.id).dir_name == f"renamed__{w.id}", "the manifest took the name"
    moved = [r for r in store.list("calibration") if r.complete][0]
    assert (moved.name, moved.id) == ("renamed", w.id)
    assert moved.dir_name == f"p__{w.id}", "the folder did not move, and the listing says so"


def test_a_diagnostics_row_carries_its_variant_and_no_observation_mode(store):
    """A diagnostic records its mode under ``variant``, never under ``mode``. The reason is in
    ``core/artifacts/manifest.py``'s BODY_KEYS comment: ``Summary.mode`` is the OBSERVATION mode, so a
    listing that showed "jacobian" in that column would be reporting a conditioning geometry that does
    not exist. §2.1 surfaces ``variant`` beside it rather than changing that."""
    _make(store, "diagnostic", name="ident",
          body={"diagnostic": "identifiability", "variant": "jacobian", "settings": {}, "results": {}})
    row = store.list("diagnostic")[0]
    assert row.variant == "jacobian" and row.mode is None

    _make(store, "calibration", name="cal", body=_cal_body())
    assert store.list("calibration")[0].variant is None, "only a diagnostic has one"

    post = _make(store, "posterior", name="post", body=_bodies()["posterior"])
    prow = store.list("posterior")[0]
    assert prow.id == post.id and (prow.mode, prow.width, prow.amortized) == ("chi", 50, True)
    assert prow.variant is None
    assert (prow.batches_done, prow.batches_planned, prow.rows) == (None, None, None), \
        "progress is the simulation cache's alone"
```

- [ ] **Step 2: Run them and watch them fail**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest "tests/test_artifact_store.py::test_a_summary_says_whether_the_run_finished_for_every_kind" "tests/test_artifact_store.py::test_a_cache_row_carries_its_progress_until_mark_complete_flips_it" "tests/test_artifact_store.py::test_every_row_carries_its_own_directory_name" "tests/test_artifact_store.py::test_a_diagnostics_row_carries_its_variant_and_no_observation_mode" "tests/test_artifact_store.py::test_create_list_get_round_trip_per_kind" -v`

Expected: 4 failed, 1 passed. The `finished` one fails on its first line, `AttributeError: module 'core.artifacts' has no attribute 'KIND_DIRS'` (the re-export Step 3 adds); the cache-progress one with `AttributeError: 'Summary' object has no attribute 'finished'`, the `variant` one with `... no attribute 'variant'`, the `dir_name` one with `... no attribute 'dir_name'`. `test_create_list_get_round_trip_per_kind` PASSES — it is the same six bodies through the new helper.

- [ ] **Step 3: Give `Summary` the six fields, and re-export `KIND_DIRS`**

In `core/artifacts/store.py`, replace lines 68-76 (the `complete` comment through `parents`) —

```python
    # "has a valid manifest", i.e. the directory describes a real artifact -- NOT "the run finished".
    # For the simulation kind those differ: a cache is manifested from its first batch on, and
    # ``body["complete"]`` is the field that says whether its rows are all there.
    complete: bool
    reason: "str | None"
    mode: "str | None" = None
    width: "int | None" = None
    amortized: "bool | None" = None
    parents: dict = field(default_factory=dict)
```

— with:

```python
    # "has a valid manifest", i.e. the directory describes a real artifact -- NOT "the run finished".
    # For the simulation kind those differ: a cache is manifested from its first batch on, and
    # ``body["complete"]`` is the field that says whether its rows are all there. ``finished`` below
    # is that honest question, asked the same way for every kind -- ask it, not this one, when what
    # you mean is "did the run get to the end" (piece 4, B3).
    complete: bool
    reason: "str | None"
    mode: "str | None" = None
    width: "int | None" = None
    amortized: "bool | None" = None
    parents: dict = field(default_factory=dict)
    # Piece 4 (B2): everything else a browser row shows, off the ONE manifest read ``list`` already
    # did. Keyword with defaults, so no positional construction anywhere breaks.
    dir_name: str = ""                      # the directory's own name, ALWAYS; remove_incomplete's handle
    finished: bool = False                  # did the RUN finish; not ``complete`` for a cache
    batches_done: "int | None" = None       # simulation only
    batches_planned: "int | None" = None    # simulation only: body["identity"]["n_runs"], the PLANNED total
    rows: "tuple[int, ...] | None" = None   # simulation only: rows per batch, written at completion ONLY
    variant: "str | None" = None            # diagnostic only: body["variant"]
```

Then, in `core/artifacts/__init__.py`, add `KIND_DIRS` to the names the package re-exports from `.store`, replacing lines 5-8 with:

```python
from .store import (ArtifactStore, ArtifactWriter, Accept, Summary, StoreError, KIND_DIRS,  # noqa: F401
                    Loaded, LoadedPrior, LoadedPosterior, LoadedObservation, LoadedCalibration,
                    LoadedInference, LoadedDiagnostic, write_simulation_manifest,
                    default_store, set_default_store, use_store, resolve_store)
```

One line, so that every later task — the browser's kind tabs, `artifacts list`, the sweep — asks the package for the seven kinds and nothing outside `core/artifacts/` imports the `store` submodule for a constant (P17).

- [ ] **Step 4: Fill them in `list`**

In the same file, replace `list` (lines 338-350) —

```python
    def list(self, kind: str) -> list:
        rows = []
        for sub, m, reason in self._entries(kind):
            if m is None:
                rows.append(Summary(kind, sub.name, "", "", "", sub, False, reason))
                continue
            body = m.body
            rows.append(Summary(kind, m.id, m.name, m.created, m.note, sub, True, None,
                                mode=body.get("mode"), width=(body.get("conditioning") or {}).get("width"),
                                amortized=body.get("amortized"), parents=dict(m.parents)))
        rows.sort(key=lambda s: s.created, reverse=True)      # newest first ...
        rows.sort(key=lambda s: not s.complete)               # ... complete first (stable)
        return rows
```

— with:

```python
    def list(self, kind: str) -> list:
        """Every directory under ``kind`` as a ``Summary``: complete rows first, newest first.

        Reads each manifest ONCE and KEEPS what it read, so a browser row needs no second read
        (piece 4, B2). ``complete`` and ``finished`` are different questions -- see ``Summary``.
        """
        out = []
        for sub, m, reason in self._entries(kind):
            if m is None:
                out.append(Summary(kind, sub.name, "", "", "", sub, False, reason, dir_name=sub.name))
                continue
            body = m.body
            # A committed artifact is finished BY CONSTRUCTION: _commit writes the manifest LAST, so
            # one exists only for a run that reached the end. The cache is the exception -- it is
            # manifested from its first batch on and says so in its own body (piece 4, B3).
            finished = bool(body.get("complete")) if kind == "simulation" else True
            per_batch = body.get("rows")
            # The PLANNED batch count lives only in the cache's identity -- the dict the naming digest
            # is taken over -- and a row that did not carry it could not render "3/4 batches" without
            # reading this manifest again (piece 4, B2).
            ident = (body.get("identity") or {}) if kind == "simulation" else {}
            planned = ident.get("n_runs")
            out.append(Summary(kind, m.id, m.name, m.created, m.note, sub, True, None,
                               mode=body.get("mode"), width=(body.get("conditioning") or {}).get("width"),
                               amortized=body.get("amortized"), parents=dict(m.parents),
                               dir_name=sub.name, finished=finished,
                               batches_done=body.get("batches_done"),
                               batches_planned=None if planned is None else int(planned),
                               rows=None if per_batch is None else tuple(int(r) for r in per_batch),
                               variant=body.get("variant")))
        out.sort(key=lambda s: s.created, reverse=True)       # newest first ...
        out.sort(key=lambda s: not s.complete)                # ... complete first (stable)
        return out
```

(The accumulator is renamed `out` only because `rows` is now also a `Summary` field name, and a local shadowing a field name in the same method is how the next reader gets confused.)

- [ ] **Step 5: Run the task's own tests**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_artifact_store.py -q`

Expected: PASS (the whole store suite — the four new tests plus everything that already asserted on `Summary` and on `complete`).

- [ ] **Step 6: Check the one production reader of `Summary`**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_nav_and_gating.py -q`

Expected: PASS. `StorePicker.refresh` (`core/gui/widgets/artifact_picker.py:130-151`) is the only production caller of `list` outside the store, and it reads `s.complete`, `s.label`, `s.id`, `s.created`, `s.mode`, `s.width`, `s.amortized` — all unchanged. This step is the proof, not a hope.

- [ ] **Step 7: Commit**

```bash
git add core/artifacts/store.py core/artifacts/__init__.py tests/test_artifact_store.py
git commit -m "store: a listing row carries dir_name, finished and a cache's progress"
```

---

### Task 4: `read_log` — one place that answers "is there a log, and what does it say"

**Files:**
- Modify: `core/artifacts/store.py:366-370` — insert `read_log` after `path` and before the `# ── create / rename / delete ──` banner at `:372`
- Modify: `core/artifacts/store.py:299-301` — `ArtifactWriter._commit`'s `log.txt` write gains an explicit `newline="\n"` (Q5; the same rule P23 already made for the report)
- Test: `tests/test_artifact_store.py` — two tests appended at the end of the file

**Interfaces:**
- Consumes: `core.artifacts.store.LOG_FILE` (`:35`), `KIND_DIRS` (`:29-31`), `ArtifactStore._find(kind, ref)` (`:352-357`), and `ArtifactWriter._commit`'s existing write of that file (`:294-301`), which this task also edits (Step 4). Independent of Task 3 — the two touch different lines and can land in either order.
  - Also `core.refusals.FIELDS["artifact"]` and its two front-end table entries, from **Task 2** (spec §2.5): `read_log`'s two refusals carry `field="artifact"` (P1), and `tests/test_refusals.py:652` AST-scans every `field="…"` literal under `core/` against the registry, so Task 2 must land first. Step 5 runs that test as the proof.
- Produces: `ArtifactStore.read_log(self, kind: str, ref: str, *, max_bytes: "int | None" = None) -> "tuple[str | None, bool]"`, returning `(text, truncated)`, with `text is None` meaning "no file at all". Used by the browser's detail pane (`read_log(kind, id, max_bytes=1 << 20)`), by `artifacts show` and by the GUI's Save.
  - Its own refusals, both `StoreError(..., field="artifact")`: an unknown kind, and a ref that names no complete artifact. It resolves with `_find` rather than through `path()` for exactly that reason — `path()`'s refusal carries no field key, and an inherited field-less refusal would leave both front ends with nothing to name (P1).

**Why this task exists:** Since piece 3 every committed artifact but the simulation cache carries a `log.txt` of its run's records, and **nothing reads it** (spec §1: "Since piece 3 every committed artifact also carries a `log.txt` of its run's records, and nothing reads it"). **B4** makes that file half of the browser's detail view. The reason it is a store method rather than a `path()` join in each front end is that "there is no file" has three different causes, and the front ends must not each work them out: the cache never gets one (it has no writer — `store.create("simulation", ...)` refuses — its manifest is refreshed batch by batch across resumes, and one cache is shared by every posterior that names it), an artifact written outside any run gets none, and a run that said nothing writes an **empty** one. `None` and `""` are therefore different answers (spec §2.2), which a bare `read_text()` with a `FileNotFoundError` guard would flatten.

The file's format is already pinned, by a test that says why: `tests/test_artifact_store.py:3032-3033` — *"The format is pinned too -- ``HH:MM:SS level message`` -- because piece 4's browser will show it."* The empty-file case is pinned five lines further on, at `:3076-3084`: *"A run that said NOTHING before its commit still gets the file, empty (spec §4.4, V4) ... and 'every committed artifact but the cache has one' is the invariant piece 4's browser reads. Silence is a record too."* And the cache's absence is pinned at `:3114`. This task is the reader those three pins were written for.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_artifact_store.py`:

```python
def test_read_log_tells_a_committed_run_from_a_silent_one_and_from_no_file_at_all(store):
    """B4's four answers (spec §2.2, §2.6). The browser has to say something honest in each case, and
    one place works it out so the two front ends cannot disagree:

      * a committed artifact's records, verbatim, stamped ``HH:MM:SS level`` -- the format
        test_each_artifact_gets_the_log_of_the_entry_that_wrote_it pins BECAUSE this reader shows it;
      * ``("", False)``: the run said nothing, and silence is a record too;
      * ``(None, False)``: there is NO file. The simulation cache never gets one (no writer, a
        manifest refreshed batch by batch across resumes, one cache shared by every posterior that
        names it), and neither does an artifact written outside any run.
    """
    import logging
    import re

    from core import runs
    from core.SBI.training_checkpoint import identity_digest

    with runs.capture_run():
        logging.getLogger("core.orchestrator").info("said something")
        spoke = _make(store, "calibration", name="spoke")
    text, truncated = store.read_log("calibration", spoke.id)
    assert truncated is False
    assert re.fullmatch(r"\d\d:\d\d:\d\d info said something\n", text), repr(text)
    assert store.read_log("calibration", "spoke")[0] == text, "a name resolves like an id"

    with runs.capture_run() as run:
        quiet = _make(store, "calibration", name="quiet")
        assert run.lines == [], run.lines            # non-vacuous: nothing was said before the commit
    assert store.read_log("calibration", quiet.id) == ("", False)

    outside = _make(store, "calibration", name="outside")
    assert store.read_log("calibration", outside.id) == (None, False), "no run, no file"

    ident = {"format": "training-rows/2", "model": "X", "n_runs": 3, "truncation": None}
    digest = identity_digest(ident)
    st.write_simulation_manifest(store.kind_dir("simulation") / digest, ident, batches_done=1)
    assert store.read_log("simulation", digest) == (None, False), "the cache never carries one"

    # Both refusals are ITS OWN and carry field="artifact" (P1): routing through ``path()`` would
    # inherit a field-less refusal, and neither front end could then name the way out of it.
    with pytest.raises(st.StoreError, match="no complete calibration") as e:
        store.read_log("calibration", "nothing_by_that_name")
    assert e.value.field == "artifact"
    with pytest.raises(st.StoreError, match="unknown artifact kind") as e:
        store.read_log("priors", "spoke")
    assert e.value.field == "artifact"


def test_read_log_returns_the_tail_over_max_bytes(store):
    """The detail pane reads at most 1 MiB and a run of days writes more than that, so the TAIL is
    what matters -- the crash is at the end -- and ``truncated`` is how the pane knows to say so.
    At or above the file's own size nothing is cut and ``truncated`` stays False."""
    from core import runs
    with runs.capture_run():
        w = _make(store, "calibration", name="long")
    body = "head\n" + "x" * 40 + "\ntail\n"
    # newline="\n" so the byte offsets below are the ones this test means: write_text's default
    # newline=None would put CRLF on disk and every max_bytes leg would be cutting different bytes.
    (w.dir / st.LOG_FILE).write_text(body, encoding="utf-8", newline="\n")
    size = (w.dir / st.LOG_FILE).stat().st_size
    assert store.read_log("calibration", w.id) == (body, False)
    assert store.read_log("calibration", w.id, max_bytes=size) == (body, False)
    assert store.read_log("calibration", w.id, max_bytes=size + 10) == (body, False)
    assert store.read_log("calibration", w.id, max_bytes=5) == ("tail\n", True)
    # 0 asks for nothing and must GET nothing: ``data[-max_bytes:]`` would return the WHOLE file here
    assert store.read_log("calibration", w.id, max_bytes=0) == ("", True)
    # A tail cut through a multi-byte character is replaced, never raised: the pane must still render
    (w.dir / st.LOG_FILE).write_bytes("a\u00b5".encode("utf-8"))        # b'a\xc2\xb5'
    assert store.read_log("calibration", w.id, max_bytes=1) == ("\ufffd", True)
    # A file an older build wrote is CRLF on disk (write_text's default newline=None on Windows).
    # It reads back as LF, so one newline convention reaches the pane, the report and both front
    # ends whoever wrote the file.
    (w.dir / st.LOG_FILE).write_bytes(b"head\r\nmid\r\ntail\r\n")
    assert store.read_log("calibration", w.id) == ("head\nmid\ntail\n", False)
```

- [ ] **Step 2: Run them and watch them fail**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest "tests/test_artifact_store.py::test_read_log_tells_a_committed_run_from_a_silent_one_and_from_no_file_at_all" "tests/test_artifact_store.py::test_read_log_returns_the_tail_over_max_bytes" -v`

Expected: 2 failed, both with `AttributeError: 'ArtifactStore' object has no attribute 'read_log'`.

- [ ] **Step 3: Add `read_log`**

In `core/artifacts/store.py`, insert this immediately after `path` (which ends at line 370 with `return sub`) and before the `# ── create / rename / delete ─────` banner at line 372:

```python
    def read_log(self, kind: str, ref: str, *, max_bytes: "int | None" = None) -> "tuple[str | None, bool]":
        """``(text, truncated)`` of the artifact's ``log.txt``; ``(None, False)`` when there is no file.

        None and "" are DIFFERENT answers, deliberately. The simulation cache never gets a log: it has
        no writer (``create`` refuses the kind), its manifest is refreshed batch by batch across
        resumes, and one cache is shared by every posterior that names it -- so no single commit holds
        one entry's records. An artifact written outside any public entry gets none either. But a run
        that said NOTHING writes an EMPTY one: silence is a record too (piece 3, V4).

        With ``max_bytes`` the TAIL is returned and ``truncated`` is True -- the end is where the
        failure is -- and its first line may be a partial one.

        Newlines come back as LF. ``_commit`` writes the file with ``newline="\\n"``, but a log.txt an
        OLDER build wrote went through ``write_text``'s default ``newline=None`` and is CRLF on
        Windows; normalising here means one convention reaches the pane, the saved report and both
        front ends whoever wrote the file.

        THE one place that joins ``LOG_FILE``, so "is there a log, and what does it say" is answered
        once and the cache's absence is explained here rather than in each front end (piece 4, B4).

        Resolves with ``_find`` rather than through ``path()`` so that both refusals can carry
        ``field="artifact"``: ``path()``'s is field-less, and an inherited field-less refusal leaves
        each front end with no control to name.
        """
        if kind not in KIND_DIRS:
            raise StoreError(f"unknown artifact kind {kind!r}", field="artifact")
        sub, m = self._find(kind, ref)
        if m is None:
            raise StoreError(f"no complete {kind} artifact named or id'd {ref!r}", field="artifact")
        f = sub / LOG_FILE
        try:
            data = f.read_bytes()
        except (FileNotFoundError, NotADirectoryError):
            return None, False
        truncated = max_bytes is not None and len(data) > max_bytes
        if truncated:
            # len(data) - max_bytes, NOT -max_bytes: ``data[-0:]`` is the whole file, so a caller
            # asking for nothing would be handed everything.
            data = data[len(data) - max_bytes:]
        # errors="replace" is the identity over a whole, valid file; it matters only for a tail cut
        # in the middle of a multi-byte character, which must render rather than raise.
        text = data.decode("utf-8", errors="replace")
        return text.replace("\r\n", "\n"), truncated
```

- [ ] **Step 4: Give `_commit`'s `log.txt` write an explicit `newline="\n"`**

The other half of the same fix: today the write goes through `Path.write_text`'s default
`newline=None`, which turns every `\n` into `\r\n` on Windows, so the file piece 3's format test
describes as `HH:MM:SS level message\n` is CRLF on disk. In `core/artifacts/store.py`, inside
`ArtifactWriter._commit`, replace this line (`:301`, the last of the log-write block that begins with
the comment at `:294`):

```python
            (self.dir / LOG_FILE).write_text(run_log.text(), encoding="utf-8")
```

with:

```python
            # newline="\n" explicitly: write_text's default would make every record CRLF on Windows,
            # and this file is read back by read_log, shown in the browser's detail pane and saved
            # verbatim to a report whose bytes spec §5 says both front ends must agree on.
            (self.dir / LOG_FILE).write_text(run_log.text(), encoding="utf-8", newline="\n")
```

Nothing else in the block moves: `run_log = runs.current_run_log()` and its `if run_log is not None:`
guard stay exactly as they are, and this is still step 2 of the three-step commit protocol — no record
is logged between the steps.

- [ ] **Step 5: Run the task's own tests, and the registry's closure**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_artifact_store.py "tests/test_refusals.py::test_every_field_key_has_a_flag_and_every_key_literal_under_core_is_registered" -q`

Expected: PASS. Run the whole store suite, not just the two new tests: Step 4 changes the bytes of every `log.txt` the suite writes, and piece 3's own format test (`test_each_artifact_gets_the_log_of_the_entry_that_wrote_it`, `:3032`) reads that file. The second node id is the proof that the two new `field="artifact"` literals are a registered key with an entry in both front-end tables: it AST-scans every `field="…"` literal under `CODE_ROOTS` against `core.refusals.FIELDS`. If it fails naming `'artifact'`, Task 2 has not landed yet — stop and land that one first rather than dropping the field key here.

- [ ] **Step 6: Commit**

```bash
git add core/artifacts/store.py tests/test_artifact_store.py
git commit -m "store: read_log answers whether an artifact has a run log, and what it says"
```

---

### Task 5: `remove_incomplete` and `sweep_incomplete` — the complement of `delete`

**Files:**
- Modify: `core/artifacts/store.py:478-488` — insert both methods after `delete` (which ends at `:488` with `_rmtree_retry(sub)`) and before `unnamed` at `:490`
- Modify: the same `delete` (`:478-488`) — its own two refusals gain `field="artifact"`, wording untouched (P1, P20)
- Test: `tests/test_artifact_store.py` — one helper and four tests appended at the end of the file

**Interfaces:**
- Consumes:
  - `core.refusals.FIELDS` carrying `Field("artifact", "the artifact", None)`, `core/gui/fields.py::CONTROL["artifact"] = "Select an artifact in the list on the Artifacts screen."` and `core/tool/fields.py::FLAG["artifact"] = None` — all three from **Task 2**, which adds the two new field keys (spec §2.5). Every refusal this task raises or edits carries that key, and `tests/test_refusals.py:652` (`test_every_field_key_has_a_flag_and_every_key_literal_under_core_is_registered`) **AST-scans every such literal under `core/` against the registry**, so Task 2 must land first (Task 2's `set_note` and Task 4's `read_log` already raise the key, so it is not this task's first use). Step 6 below runs that test as the proof.
  - `ArtifactStore._entries(kind)` (`:316-336`), `_rmtree_retry(path, retries=3, backoff_s=0.1)` (`:180-192`), `KIND_DIRS` (`:29-31`), `ArtifactStore.kind_dir(kind)` (`:310-313`), `ArtifactStore.delete(kind, ref, *, force=False)` (`:478-488`), `ArtifactStore.dependents(kind, id_)` (`:451-476`).
  - `Summary.dir_name` from Task 3 (the tests read `r.dir_name` off `list`) and Task 3's `_bodies()` helper, which the `delete` test uses for its posterior body. Task 3 is a declared dependency (the graph's `T5 ── T3` edge): use `r.dir_name` and `_bodies()`, and if either is missing, Task 3 has not landed — stop and land it first rather than working round it.
- Produces:
  - `ArtifactStore.remove_incomplete(self, kind: str, dir_name: str) -> Path`
  - `ArtifactStore.sweep_incomplete(self, kind: "str | None" = None) -> "tuple[list, list]"` returning `(removed, failed)` = `([(kind, dir_name)], [(kind, dir_name, reason)])`
  - both raising `StoreError(..., field="artifact")`. Used by the browser's Sweep action and the tool's `artifacts rm` / `artifacts sweep`.
  - `ArtifactStore.delete`'s own two refusals, now `field="artifact"` as well (P1, P20): the "no complete artifact" one and the DEPENDENTS one. **Both messages are unchanged**, `force=True` included — a Python keyword argument is not a control, and the front ends name their own way out from their tables.

**Why this task exists:** Nothing in the store today can remove a directory that has no usable manifest. `delete` and `set_note` both resolve through `_find` (`:352-357`), which returns only manifest-bearing entries, so a leftover folder — what a crash, a torn write or a hand-moved directory leaves, and what `list` shows with a `reason` instead of a name — can only be removed by hand in Explorer. **B7** gives it one action: *"Sweep removes every incomplete directory of a kind in one action, through a new store call that can only ever remove a directory with no usable manifest."* The complementary safety property is the point (spec §2.3): `delete` can only ever remove a real artifact and therefore always runs the dependency check, and `remove_incomplete` can only ever remove a leftover, so neither call can do the other's job by accident. `sweep_incomplete` reports a directory it cannot delete rather than dying on it, because on Windows one held handle must not cost the operator the other six.

While `delete` is open, its own two refusals take the same field key (P1, P20). Both front ends act through it — the browser's Delete button and the tool's `artifacts rm` — and both of its refusals are fixed the same way: pick a different artifact in the list, which is exactly what the window's `artifact` entry says. Neither message moves. The dependents one names `force=True` on purpose: a Python keyword argument is not a control, no front end offers it (**B6**), and a reader of that line in `log.txt` needs to know what the escape hatch is called.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_artifact_store.py`:

```python
def _leftovers(store, kind="prior"):
    """The three shapes ``_entries`` classifies as incomplete, written into one kind: no manifest at
    all, a manifest that will not parse, and a manifest that parses but declares another kind. Also
    writes the real calibration whose manifest the third one copies. Returns the three names, sorted
    the way ``_entries`` walks them."""
    d = store.kind_dir(kind)
    d.mkdir(parents=True, exist_ok=True)
    (d / "no_manifest__20260101T000001").mkdir()
    torn = d / "unreadable__20260101T000002"
    torn.mkdir()
    (torn / st.MANIFEST).write_text("{not json", encoding="utf-8")
    wrong = d / "wrong_kind__20260101T000003"
    wrong.mkdir()
    other = _make(store, "calibration", name="elsewhere", body=_cal_body())
    (wrong / st.MANIFEST).write_bytes((other.dir / st.MANIFEST).read_bytes())
    return ["no_manifest__20260101T000001", "unreadable__20260101T000002",
            "wrong_kind__20260101T000003"]


def test_remove_incomplete_refuses_everything_that_is_not_a_leftover(store, tmp_path):
    """B7 (spec §2.3, §2.6). It is the COMPLEMENT of ``delete``: that one resolves through ``_find``,
    which returns only manifest-bearing entries, so it can only ever remove a real artifact and
    always runs the dependency check; this one asks ``_entries`` and removes only what came back
    WITHOUT a manifest. Neither can do the other's job, which is the safety property worth keeping.

    Four refusal classes, each a StoreError carrying ``field="artifact"`` so both front ends can name
    the way out of it from their own table."""
    real = _make(store, "calibration", name="keepme", body=_cal_body())
    (store.kind_dir("calibration") / "leftover__20260101T000000").mkdir(parents=True)
    (store.kind_dir("calibration") / "notadir.txt").write_text("x", encoding="utf-8")
    nested = store.kind_dir("calibration") / "sub" / "leftover__20260101T000000"
    nested.mkdir(parents=True)

    for kind, name, why in (
            ("priors", "leftover__20260101T000000", "unknown artifact kind"),   # not a kind at all
            ("calibration", "sub/leftover__20260101T000000", "not the name of a directory"),
            ("calibration", "sub\\leftover__20260101T000000", "not the name of a directory"),
            ("calibration", "..", "not the name of a directory"),
            ("calibration", "", "not the name of a directory"),
            ("calibration", str(tmp_path), "not the name of a directory"),      # an absolute path
            ("calibration", "/etc", "not the name of a directory"),             # ... posix-spelled
            ("calibration", "notadir.txt", "no directory named"),               # a plain file
            ("calibration", "never_existed", "no directory named"),
            ("calibration", real.dir.name, "holds a valid calibration manifest")):   # a REAL artifact
        with pytest.raises(st.StoreError, match=why) as e:
            store.remove_incomplete(kind, name)
        assert e.value.field == "artifact", (kind, name, e.value.field)

    assert real.dir.is_dir() and store.get("calibration", real.id).name == "keepme"
    assert (store.kind_dir("calibration") / "notadir.txt").is_file()
    assert nested.is_dir(), "a path was refused, not followed"
    assert (store.kind_dir("calibration") / "leftover__20260101T000000").is_dir(), \
        "none of the refusals removed anything"

    target = store.kind_dir("calibration") / "leftover__20260101T000000"
    assert store.remove_incomplete("calibration", "leftover__20260101T000000") == target
    assert not target.exists()


def test_remove_incomplete_removes_all_three_shapes_entries_calls_incomplete(store):
    """The three shapes are what a leftover actually looks like on disk: a crash before the manifest
    was written, a torn write, and a hand-moved folder. All three are removable, and the ``reason``
    ``list`` shows for each is the one ``_entries`` gave it."""
    names = _leftovers(store, "prior")
    reasons = {r.dir_name: r.reason for r in store.list("prior")}
    assert set(reasons) == set(names), reasons
    assert "no manifest.json" in reasons[names[0]]
    assert "unreadable manifest" in reasons[names[1]]
    assert "declares kind 'calibration'" in reasons[names[2]]
    for name in names:
        removed = store.remove_incomplete("prior", name)
        assert removed.name == name and not removed.exists()
    assert store.list("prior") == []
    assert store.get("calibration", "elsewhere").name == "elsewhere", \
        "the real artifact whose manifest the wrong-kind folder copied is untouched"


def test_sweep_incomplete_finishes_the_rest_when_one_directory_will_not_delete(store, monkeypatch):
    """B7's one action per kind, and per store. A directory that will not delete is REPORTED, never
    fatal: on Windows a held handle -- an Explorer preview, a virus scanner, a file this process still
    has open -- makes ``shutil.rmtree`` raise PermissionError, and ``_rmtree_retry`` waits 0.1 s and
    then 0.2 s before giving up. One such folder must not cost the operator the other six.

    The failure is INJECTED rather than provoked with a real handle, so the pin holds on any
    filesystem and costs no sleep."""
    names = _leftovers(store, "prior")
    (store.kind_dir("inference") / "orphan__20260101T000009").mkdir(parents=True)
    real = _make(store, "calibration", name="keepme", body=_cal_body())

    assert store.sweep_incomplete("posterior") == ([], []), \
        "a kind with nothing in it removes nothing and fails nothing"

    stuck = store.kind_dir("prior") / names[1]
    real_rmtree = st._rmtree_retry

    def _one_held(path, **kw):
        if Path(path) == stuck:
            raise PermissionError("another process holds this folder open")
        return real_rmtree(path, **kw)

    monkeypatch.setattr(st, "_rmtree_retry", _one_held)
    removed, failed = store.sweep_incomplete("prior")
    monkeypatch.undo()
    assert sorted(removed) == [("prior", names[0]), ("prior", names[2])]
    assert failed == [("prior", names[1], "PermissionError: another process holds this folder open")]
    assert stuck.is_dir(), "the one it could not remove is still there"
    assert [r.dir_name for r in store.list("prior")] == [names[1]]

    removed_all, failed_all = store.sweep_incomplete()
    assert failed_all == []
    assert sorted(removed_all) == sorted([("prior", names[1]),
                                          ("inference", "orphan__20260101T000009")])
    assert store.list("prior") == [] and store.list("inference") == []
    assert store.get("calibration", real.id).name == "keepme", "a real artifact is never swept"

    with pytest.raises(st.StoreError, match="unknown artifact kind") as e:
        store.sweep_incomplete("priors")
    assert e.value.field == "artifact"


def test_delete_refuses_with_the_artifact_field_and_says_force_true(store):
    """``delete``'s own two refusals carry ``field="artifact"`` too (P1, P20), because the fix for
    either is the same: pick a different artifact in the list. The WORDING is untouched --
    test_delete_refuses_naming_dependents_and_force_deletes (``:278-287``) already pins the message and
    the force path -- and this test pins only the key, plus the fact that the dependents message still
    names ``force=True``. Naming a Python keyword argument is not naming a control: no front end offers
    it (B6), and whoever reads that line in ``log.txt`` needs to know what the escape hatch is called.
    """
    parent = _make(store, "posterior", name="mother", body=_bodies()["posterior"])
    child = _make(store, "inference", name="child", body={"results": {"n_samples": 5}},
                  parents={"posterior": parent.id})
    with pytest.raises(st.StoreError, match="refusing to delete") as e:
        store.delete("posterior", parent.id)
    assert e.value.field == "artifact" and "force=True" in str(e.value)
    assert parent.dir.is_dir() and child.dir.is_dir(), "a refusal removed nothing"

    with pytest.raises(st.StoreError, match="no complete calibration") as e:
        store.delete("calibration", "never_existed")
    assert e.value.field == "artifact"

    store.delete("inference", child.id)
    store.delete("posterior", parent.id)           # the dependent is gone; no force needed
    assert store.list("posterior") == [] and store.list("inference") == []
```

- [ ] **Step 2: Run them and watch them fail**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest "tests/test_artifact_store.py::test_remove_incomplete_refuses_everything_that_is_not_a_leftover" "tests/test_artifact_store.py::test_remove_incomplete_removes_all_three_shapes_entries_calls_incomplete" "tests/test_artifact_store.py::test_sweep_incomplete_finishes_the_rest_when_one_directory_will_not_delete" "tests/test_artifact_store.py::test_delete_refuses_with_the_artifact_field_and_says_force_true" -v`

Expected: 4 failed. The first two with `AttributeError: 'ArtifactStore' object has no attribute 'remove_incomplete'`; the third with `AttributeError: 'ArtifactStore' object has no attribute 'sweep_incomplete'`; the fourth on `assert e.value.field == "artifact"` — `delete`'s refusal is raised with the right words today and no field key at all, so `field` is `None`.

- [ ] **Step 3: Add `remove_incomplete`**

In `core/artifacts/store.py`, insert this immediately after `delete` (which ends at line 488 with `_rmtree_retry(sub)`) and before `def unnamed(` at line 490:

```python
    def remove_incomplete(self, kind: str, dir_name: str) -> Path:
        """Remove one directory under ``kind`` that has no usable manifest. Returns the path removed.

        THE COMPLEMENT OF ``delete``. That one resolves through ``_find``, which only ever returns
        manifest-bearing entries, so it can only remove a real artifact -- and therefore always runs
        the dependency check. This one asks ``_entries`` (the same classifier ``list`` shows) and
        removes only what came back WITHOUT a manifest. Neither call can do the other's job, which is
        the safety property: nothing can reach a leftover folder by accident, and nothing can reach a
        real artifact without the dependency check (piece 4, B7).

        Refuses, as a StoreError with ``field="artifact"``: an unknown kind; a ``dir_name`` that is
        not a direct child (any separator, ``..``, an absolute path); a name that is no directory; and
        -- the one that matters -- a directory ``_entries`` classifies as COMPLETE.
        """
        if kind not in KIND_DIRS:
            # Not kind_dir()'s own refusal, which carries no field key: every refusal on this path
            # names the artifact, so a front end can say where to pick another one.
            raise StoreError(f"unknown artifact kind {kind!r}", field="artifact")
        d = self.kind_dir(kind)
        if (not dir_name or dir_name in (".", "..") or "/" in dir_name or "\\" in dir_name
                or dir_name != Path(dir_name).name):
            # The separator checks are not redundant with the Path comparison: on POSIX a backslash
            # is an ordinary character, so "sub\\x" would pass it.
            raise StoreError(f"{dir_name!r} is not the name of a directory directly under {d}; a "
                             f"leftover is removed by its own folder name, never by a path",
                             field="artifact")
        sub = d / dir_name
        if not sub.is_dir():
            raise StoreError(f"no directory named {dir_name!r} under {d}", field="artifact")
        m = next((mm for s, mm, _ in self._entries(kind) if s.name == dir_name), None)
        if m is not None:
            raise StoreError(
                f"{dir_name!r} holds a valid {kind} manifest, so it is a real artifact and not a "
                f"leftover; remove it with delete(), which refuses it while anything depends on it",
                field="artifact")
        _rmtree_retry(sub)
        return sub
```

- [ ] **Step 4: Add `sweep_incomplete`**

Immediately after `remove_incomplete`, still before `unnamed`:

```python
    def sweep_incomplete(self, kind: "str | None" = None) -> "tuple[list, list]":
        """``(removed, failed)`` over one kind or all seven: ``removed`` is ``[(kind, dir_name)]`` and
        ``failed`` is ``[(kind, dir_name, reason)]``.

        A directory that will not delete is REPORTED, never fatal -- the sweep finishes the rest. On
        Windows a held handle (an Explorer preview, a virus scanner, a file this process still has
        open) makes ``shutil.rmtree`` raise PermissionError; ``_rmtree_retry`` waits 0.1 s and then
        0.2 s and then gives up, and one such directory must not cost the operator the other six.

        Walks ``_entries`` rather than calling ``remove_incomplete`` per directory: the same
        classifier, one scan per kind instead of one per directory, and no StoreError to catch that
        could have meant "it was complete after all".
        """
        if kind is not None and kind not in KIND_DIRS:
            raise StoreError(f"unknown artifact kind {kind!r}", field="artifact")
        removed, failed = [], []
        for k in (KIND_DIRS if kind is None else (kind,)):
            for sub, m, _ in self._entries(k):
                if m is not None:
                    continue
                try:
                    _rmtree_retry(sub)
                except OSError as e:           # noqa: BLE001 -- one held handle must not stop the sweep
                    failed.append((k, sub.name, f"{type(e).__name__}: {e}"))
                else:
                    removed.append((k, sub.name))
        return removed, failed
```

- [ ] **Step 5: Give `delete`'s two refusals the artifact field key**

Still in `core/artifacts/store.py`, replace `delete` (the method as it now stands, at lines 478-488 — the two new methods went in after it) with this. Both messages are copied word for word; the only change is `field="artifact"` on each `raise`:

```python
    def delete(self, kind: str, ref: str, *, force: bool = False) -> None:
        sub, m = self._find(kind, ref)
        if m is None:
            raise StoreError(f"no complete {kind} artifact named or id'd {ref!r}", field="artifact")
        deps = self.dependents(kind, m.id)
        if deps and not force:
            listed = "; ".join(f"{k} {n or '(unnamed)'} [{i}]" for k, i, n in deps)
            # The wording does not move, ``force=True`` included: core/refusals.py's own rule is that
            # a message names no box, tab, flag or button but MAY name a Python keyword -- "it is the
            # core API's own word". No front end offers force (B6); the log's reader still needs the
            # escape hatch's name. Both refusals take field="artifact": either is answered by picking
            # a different artifact in the list (piece 4, P1/P20).
            raise StoreError(
                f"refusing to delete {kind} {m.name or m.id}: {len(deps)} artifact(s) name it as a "
                f"parent -- {listed}. Delete those first, or pass force=True to orphan them.",
                field="artifact")
        _rmtree_retry(sub)
```

- [ ] **Step 6: Run the task's own tests, and the registry's closure**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_artifact_store.py "tests/test_refusals.py::test_every_field_key_has_a_flag_and_every_key_literal_under_core_is_registered" -q`

Expected: PASS — the whole store suite, including `test_delete_refuses_naming_dependents_and_force_deletes` (`:278-287`), which still passes because the wording did not move. The second node id is the proof that `field="artifact"` is a registered key with an entry in both front-end tables: it AST-scans every `field="…"` literal under `CODE_ROOTS` and asserts each one is in `core.refusals.FIELDS`, and it asserts `set(tool_fields.FLAG) == set(FIELDS)`. If it fails naming `'artifact'`, Task 2 (spec §2.5) has not landed yet — stop and land that one first rather than weakening this one.

- [ ] **Step 7: Commit**

```bash
git add core/artifacts/store.py tests/test_artifact_store.py
git commit -m "store: remove_incomplete and sweep_incomplete, the complement of delete"
```

### Task 6: `core/artifacts/report.py` — the two renderers

**Files:**
- Create: `core/artifacts/report.py`
- Modify: `core/artifacts/__init__.py:4-5` (insert the re-export between the `provenance` and `store` lines)
- Test: `tests/test_artifact_store.py` — append at the end of the file (it is 3238 lines today; the last
  test is `test_the_orchestrator_says_everything_through_its_logger`)

**Interfaces:**
- Consumes: nothing from any earlier task. Only what is already on disk:
  `core.artifacts.manifest.Manifest` (its fifteen header fields, `manifest.py:52-77`),
  `ArtifactStore.get(kind, ref) -> Manifest` (`store.py:358`), `ArtifactStore.kind_dir(kind) -> Path`
  (`store.py:310`), `ArtifactStore.StoreError` (`store.py:42`) and the store's parent table
  `store._PARENT_KEYS` (`store.py:37-39`). It does **not** touch `Summary`, `read_log` or anything
  else piece 4 adds; the graph's `T6 ── T3` edge is file order only — Task 3 appends to
  `tests/test_artifact_store.py` and rewrites the `from .store import (...)` line in
  `core/artifacts/__init__.py` ahead of this task, so append after what it wrote and re-read both
  files rather than trusting a line number quoted here.
- Produces:
  - `core.artifacts.report.render_manifest(m) -> str` — also `from core.artifacts import render_manifest`
  - `core.artifacts.report.render_lineage(store, kind: str, ref: str) -> str` — also re-exported
  - Consumed later by the browser's detail pane and its Save (§3.3, B10) and by the tool's
    `artifacts show` / `artifacts summary` (§4.2).

**Why this task exists:** B10 splits "a saveable run summary" into two documents — the manifest as text
and a lineage walk back through an artifact's parents — and says both are **files**, never an eighth
store kind. Today nothing renders a manifest at all: `store.get` hands back a dataclass and the only
human-readable copy of anything is the per-stage `results.json` a few stages write beside their
payload. Two front ends are about to print the same facts (§3.3 and §4.2), so the spec's §5 puts one
renderer in `core/artifacts/report.py` — beside the manifest code it reads, and deliberately not in
`store.py`, because a report *describes* the store and must never be mistaken for something the store
holds.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_artifact_store.py`. The module already imports `torch`, `pytest`, `mf`
(`core.artifacts.manifest`), `prov`, `st` (`core.artifacts.store`, line 144), and `_nad_cfg` /
`_prior_artifact` from `tests._fixtures` (lines 17-19); `_make` (line 151) is the file's own cheap
writer for a kind whose payload nobody reads. Nothing new is imported at module level.

```python
# ── The report renderers (piece 4, §5 / B10) ─────────────────────────────────────────────────────


def _chain(store, cfg) -> dict:
    """prior -> simulation cache -> posterior -> observation -> inference, wired through ``parents``.

    The cheapest thing the REAL writers accept: one 2-component GMM for the prior (so one step has
    input files, hashes, a payload digest and a fingerprint), and `_make`'s minimal bodies for the
    rest. Nothing here simulates, trains or infers -- the renderers are pure functions of manifests,
    so a chain built by hand exercises every branch a five-hour pipeline would.
    """
    from core.SBI.training_checkpoint import identity_digest
    p = _prior_artifact(store, cfg, name="chain_prior")
    fp = store.get("prior", p.id).fingerprints["gmm"]
    ident = {"format": "training-rows/2", "model": cfg.model, "prior_fingerprint": fp,
             "n_runs": 2, "run_size": 4, "truncation": None}
    sim = st.write_simulation_manifest(store.kind_dir("simulation") / identity_digest(ident), ident,
                                       parents={"prior": p.id}, batches_done=1, rows=[4])
    post = _make(store, "posterior", name="chain_post",
                 body={"mode": "chi", "conditioning": {"width": 50}, "transform": {}, "amortized": True,
                       "truncation": None, "training": {}},
                 parents={"prior": p.id, "simulation": sim.id})
    obs = _make(store, "observation", name="chain_obs",
                body={"mode": "chi", "conditioning": {"width": 50}, "x_obs_digest": "0" * 16,
                      "T_obs_cell": 1.0, "n_obs": 10, "forcing_vals": {}, "chi_obs_freqs": None,
                      "source": {"kind": "simulated"}})
    inf = _make(store, "inference", name="chain_inf",
                body={"results": {"n_samples": 5, "accepted": ["other_observation"]}},
                parents={"posterior": post.id, "observation": obs.id})
    return {"prior": p.id, "simulation": sim.id, "posterior": post.id, "observation": obs.id,
            "inference": inf.id}


def test_render_manifest_names_every_fact_the_manifest_holds(store):
    """§5 / B10: the manifest as text -- schema, id, name, created, note, the git revision and the
    environment, the input files WITH their hashes, the parents, the payloads with their digests, the
    figures and the body's knobs -- laid out deterministically, so the window's Save and the tool's
    ``artifacts show`` print the same bytes.

    Sorted where order is not meaning: ``to_json_text`` writes the file with ``sort_keys=True``, so a
    manifest read back from disk is already in that order, and sorting here is what makes a manifest
    built in memory print identically to the same manifest read back.
    """
    from core.artifacts import render_manifest
    cfg = _nad_cfg()
    p = _prior_artifact(store, cfg, name="rep_prior")
    m = store.get("prior", p.id)
    text = render_manifest(m)
    assert text.startswith(f"prior rep_prior  [{p.id}]\n"), text[:200]
    assert text.endswith("\n"), "a saved report ends with a newline"
    for line in (f"id:      {p.id}", "name:    rep_prior", f"created: {m.created}", "note:    (none)"):
        assert f"\n{line}\n" in text, line
    # the input files WITH their hashes, relative to Resources/ exactly as file_ref records them
    assert f"\n  bounds: Bounds/nadrowski/master.txt  sha256 {m.inputs['bounds']['sha256']}\n" in text
    assert "\n  cell: (none)\n" in text, "a bounds-built config has no cell, and the report says so"
    # the payload with its digest, and the GMM fingerprint
    assert f"\n  prior.pt  sha256 {m.payloads['prior.pt']}\n" in text
    assert f"\n  gmm: {m.fingerprints['gmm']}\n" in text
    assert "\nFigures\n  (none)\n" in text, "a prior written without a figure sink has none"
    assert "\nParents\n  (none)\n" in text
    # the knobs, and the body rendered as a sorted tree
    assert text.count("model: NADROWSKI") == 2, "once under Inputs, once under Knobs"
    assert "\n  gmm:\n    box:\n      log_mask: [" in text, text
    assert "\n    n_components: 2\n" in text and "\n  sweep: {}\n" in text
    # a long sequence is summarised, never inlined: a 13x13 rotation would bury every knob, and its
    # digest is printed two sections above
    from core.artifacts.report import _fmt_list
    assert _fmt_list([[0.0] * 13] * 13) == "[13 x 13 numbers]"
    assert _fmt_list(list(range(20))) == "[0, 1, 2, ... (20 entries)]"
    assert _fmt_list([]) == "[]" and _fmt_list([True, None]) == "[true, (none)]"
    assert render_manifest(m) == render_manifest(store.get("prior", p.id)), \
        "two calls, identical bytes -- and an in-memory manifest prints as a re-read one"


def test_render_lineage_walks_the_chain_and_prints_a_missing_parent(store):
    """§5: the artifact, then its parents transitively through ``_PARENT_KEYS``, each artifact before
    its own parents so the chain reads newest first and oldest last. Each step names the kind, the
    name, the id, the creation time, the input files with their hashes, the knobs that decided it and
    any Accept it recorded.

    A parent a manifest names but the store does not hold prints as MISSING with its id and with who
    named it: dropping it would make a broken chain read as a complete one, which is the one thing a
    provenance report must never do.
    """
    from core.artifacts import render_lineage
    cfg = _nad_cfg()
    ids = _chain(store, cfg)
    text = render_lineage(store, "inference", ids["inference"])
    assert text.startswith(f"Lineage of inference chain_inf [{ids['inference']}]\n")
    steps = [ln for ln in text.splitlines() if ln[:1].isdigit()]
    assert [ln.split(". ", 1)[1].split(" ", 1)[0] for ln in steps] == [
        "inference", "observation", "posterior", "prior", "simulation"], steps
    # the prior is named by BOTH the posterior and the cache and appears ONCE (the visited set)
    assert text.count(f"[{ids['prior']}]  created ") == 1
    assert f"parents: prior={ids['prior']}, simulation={ids['simulation']}" in text
    # the step's own facts: its inputs with their hashes, its knobs, its Accept
    assert "    bounds: Bounds/nadrowski/master.txt  sha256 " in text
    assert "model=NADROWSKI" in text, "the knobs the prior ran under"
    assert "\n  accepted: other_observation\n" in text, "the escape hatch the inference recorded"
    assert "\n  knobs:\n    (none)\n" in text, "a manifest written with no config says so"
    assert text.endswith("\n") and render_lineage(store, "inference", ids["inference"]) == text

    # a parent nothing holds: MISSING, with its id and who named it
    orphan = _make(store, "posterior", name="orphan",
                   body={"mode": "chi", "conditioning": {"width": 50}, "transform": {}, "amortized": True,
                         "truncation": None, "training": {}},
                   parents={"prior": "20260101T000000"})
    text = render_lineage(store, "posterior", orphan.id)
    assert "2. MISSING prior [20260101T000000]" in text, text
    assert "named by posterior 'orphan' as 'prior'" in text
    assert str(store.kind_dir("prior")) in text

    # an unknown ref is the store's own refusal, unchanged
    with pytest.raises(st.StoreError, match="no complete posterior"):
        render_lineage(store, "posterior", "nope")
```

- [ ] **Step 2: Run them and watch them fail**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_artifact_store.py::test_render_manifest_names_every_fact_the_manifest_holds tests/test_artifact_store.py::test_render_lineage_walks_the_chain_and_prints_a_missing_parent -v`

Expected: both FAIL with
`ImportError: cannot import name 'render_manifest' from 'core.artifacts'` and
`ImportError: cannot import name 'render_lineage' from 'core.artifacts'`.

- [ ] **Step 3: Create `core/artifacts/report.py` with the value formatters and `render_manifest`**

New file, exactly this:

```python
"""Two renderers over an artifact's manifest: the artifact as text, and its lineage.

Piece 4, B10: "a saveable run summary" is two documents -- what the browser's detail pane shows (and
``python -m core artifacts show`` prints), and a lineage walk back through an artifact's parents. Both
are FILES, never an eighth store kind, and both are pure functions of manifests the store already
holds: nothing here writes, creates or resolves anything.

It lives beside the manifest code it reads and NOT in ``store.py``, because a report DESCRIBES the
store and must never be mistaken for something the store holds. ONE renderer for both front ends, so
the document a reviewer receives is the same bytes whichever one made it.

DETERMINISM IS THE CONTRACT. Every mapping renders in sorted key order and every sequence in its own
order -- a parameter list's order is meaning, a dict's is not. ``manifest.to_json_text`` already
writes with ``sort_keys=True``, so a manifest read back from disk is in that order anyway; sorting
here is what makes a manifest built in memory print identically to the same manifest read back.

TORCH-FREE in the same qualified sense as ``manifest.py``: its own module-level imports are nothing at
all, the store's parent table is imported lazily inside ``render_lineage``, and a caller reaches it
through ``core.artifacts``, which imports torch. No claim is made that it can be imported without
torch.
"""
from __future__ import annotations

INDENT = "  "
NONE = "(none)"
RULE = "=" * 72
# Numbers printed inline before a sequence is summarised instead. A rotation matrix is 13x13 float64
# lists -- 169 numbers whose digest is printed two sections above -- and inlining it would bury every
# knob that decides anything.
MAX_INLINE = 12
# The widest packed knob line. ``config`` carries the ~35-key identity vocabulary plus the stage's own,
# so the lineage packs them rather than spending a line each.
KNOB_WIDTH = 100


def _is_scalar(v) -> bool:
    return v is None or isinstance(v, (bool, int, float, str))


def _fmt_scalar(v) -> str:
    """One manifest value as text. ``bool`` before ``int`` (it is a subclass), and the JSON spelling of
    it, so a line diffs against ``manifest.json`` itself; ``repr`` for a float, which is
    shortest-round-trip and therefore exact."""
    if v is None:
        return NONE
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return repr(v)
    return str(v)


def _fmt_list(v) -> str:
    """One line for a sequence: inline while it is short, otherwise its shape and its first entries."""
    if not v:
        return "[]"
    if all(_is_scalar(x) for x in v):
        if len(v) <= MAX_INLINE:
            return "[" + ", ".join(_fmt_scalar(x) for x in v) + "]"
        return "[" + ", ".join(_fmt_scalar(x) for x in v[:3]) + f", ... ({len(v)} entries)]"
    if all(isinstance(x, list) for x in v):
        widths = sorted({len(x) for x in v})
        shape = str(widths[0]) if len(widths) == 1 else f"{widths[0]}-{widths[-1]}"
        return f"[{len(v)} x {shape} numbers]"
    return f"[{len(v)} entries]"


def _tree(value, indent: str) -> list:
    """``key: value`` lines for a nested manifest value: dicts in sorted key order, a list of records
    indexed in its own order (``sbc.per_param`` and ``posterior_summary`` are lists precisely so the
    parameter order survives the manifest's recursive key sort)."""
    if not isinstance(value, dict):
        return [indent + _fmt_scalar(value)]
    lines = []
    for key in sorted(value):
        v = value[key]
        if isinstance(v, dict) and v:
            lines.append(f"{indent}{key}:")
            lines.extend(_tree(v, indent + INDENT))
        elif isinstance(v, dict):
            lines.append(f"{indent}{key}: {{}}")
        elif isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
            lines.append(f"{indent}{key}:")
            for i, item in enumerate(v):
                lines.append(f"{indent}{INDENT}[{i}]")
                lines.extend(_tree(item, indent + INDENT + INDENT))
        elif isinstance(v, list):
            lines.append(f"{indent}{key}: {_fmt_list(v)}")
        else:
            lines.append(f"{indent}{key}: {_fmt_scalar(v)}")
    return lines


def _input_lines(inputs: dict) -> list:
    """The input files with their hashes. ``model`` is a name and not a file; a None entry is a file the
    artifact does not have (a bounds-built config carries no cell). An UNHASHED file says so in words:
    the writer records ``sha256: None`` for an input that had moved by the commit
    (``ArtifactWriter._commit``), and a report that printed nothing there would read as a file that was
    checked."""
    lines = []
    for key in sorted(inputs):
        v = inputs[key]
        if isinstance(v, dict):
            h = v.get("sha256")
            lines.append(f"{INDENT}{key}: {v.get('path')}  sha256 " + (h if h else "NOT READ AT COMMIT"))
        else:
            lines.append(f"{INDENT}{key}: {_fmt_scalar(v)}")
    return lines


def _pack(d: dict, indent: str, width: int = KNOB_WIDTH) -> list:
    """``key=value`` pairs in sorted key order, packed onto lines no wider than ``width``.

    The WHOLE ``config`` block, not a chosen subset: the manifest's ``config`` is "the knobs it ran
    under" -- the identity vocabulary every artifact records (``manifest.config_from_cfg``) plus the
    stage's own ``w.config.update(...)`` -- and a report that printed a hand-picked subset would be a
    second copy of that key list, one stage away from going stale.
    """
    if not d:
        return [indent + NONE]
    parts = []
    for key in sorted(d):
        v = d[key]
        if isinstance(v, list):
            shown = _fmt_list(v)
        elif isinstance(v, dict):
            shown = "{...}" if v else "{}"
        else:
            shown = _fmt_scalar(v)
        parts.append(f"{key}={shown}")
    lines, cur = [], ""
    for part in parts:
        if cur and len(indent) + len(cur) + 2 + len(part) > width:
            lines.append(indent + cur)
            cur = part
        else:
            cur = part if not cur else f"{cur}  {part}"
    if cur:
        lines.append(indent + cur)
    return lines


def _section(title: str, lines: list) -> list:
    return [title, *(lines or [INDENT + NONE]), ""]


def render_manifest(m) -> str:
    """The artifact as text: the header, the git revision and the environment, the input files with
    their hashes, the parents, the payloads with their digests, the figures by name, the fingerprints,
    the knobs and the kind's own body.

    Takes a ``Manifest`` (``store.get(kind, ref)``), never a path or a ref: the caller has already
    resolved and validated it, and a renderer that resolved refs of its own would be a second index.
    It therefore says nothing about the DIRECTORY -- a folder name that disagrees with
    ``Manifest.dir_name`` is the browser's notice to give (§3.3), off ``Summary.dir_name``, because
    only a listing knows it.
    """
    out = [f"{m.kind} {m.name or '(unnamed)'}  [{m.id}]", RULE,
           f"schema:  {m.schema}",
           f"kind:    {m.kind}",
           f"id:      {m.id}",
           f"name:    {m.name or NONE}",
           f"created: {m.created}",
           f"note:    {m.note or NONE}", ""]
    out += _section("PRISM", _tree(m.prism, INDENT))
    out += _section("Environment", _tree(m.env, INDENT))
    out += _section("Inputs", _input_lines(m.inputs))
    out += _section("Parents", [f"{INDENT}{k}: {m.parents[k]}" for k in sorted(m.parents)])
    out += _section("Payloads", [f"{INDENT}{k}  sha256 {m.payloads[k] or NONE}" for k in sorted(m.payloads)])
    out += _section("Figures", [f"{INDENT}{f}" for f in m.figures])
    out += _section("Fingerprints", [f"{INDENT}{k}: {m.fingerprints[k] or NONE}" for k in sorted(m.fingerprints)])
    out += _section("Knobs", _tree(m.config, INDENT))
    out += _section("Body", _tree(m.body, INDENT))
    return "\n".join(out).rstrip("\n") + "\n"
```

- [ ] **Step 4: Add `render_lineage` and its two helpers to the same file**

Append to `core/artifacts/report.py`:

```python
def _accepted(m) -> list:
    """The escape hatches the artifact records (``Accept.used()``), or ``[]``.

    An inference writes them under ``body.results.accepted`` (``orchestrator.py:2176``) and so does
    every diagnostic (``sbc.py:220``, ``identifiability.py:182,464``, ``ablation.py:203``); a
    calibration records none -- there the flag only ever unlocked the load, and what is on record is
    the posterior's own ``amortized: false``. So the key is read with ``.get`` and its absence is not a
    gap.
    """
    results = m.body.get("results")
    if not isinstance(results, dict):
        return []
    return [str(a) for a in (results.get("accepted") or [])]


def render_lineage(store, kind: str, ref: str) -> str:
    """The artifact and its parents transitively, each artifact BEFORE its own parents, so every chain
    reads newest first and oldest last.

    Each step: the kind, the name, the id, the creation time, the input files with their hashes, the
    knobs that decided it, and any Accept it recorded. A parent a manifest names but the store does not
    hold prints as MISSING with its id and with who named it -- skipping it would make a broken chain
    read as a complete one, which is the one thing a provenance report must never do.

    Parents are older ids by construction, so a cycle is impossible; the visited set is carried anyway,
    because the alternative to a two-line guard is an endless report out of one hand-edited manifest.
    It also collapses the diamond every trained posterior has: the prior is named by the posterior AND
    by the training cache, and belongs in the report once.

    The one refusal is the head ref: ``store.get``'s own StoreError, unchanged.
    """
    # Lazily, so this module's own imports stay standard library (see the module docstring), and
    # INVERTED from the store's table -- ``parent key -> kind`` -- so a parent key added there is
    # followed here without a second list to keep in step.
    from .store import StoreError, _PARENT_KEYS
    key_kind = {key: k for k, keys in _PARENT_KEYS.items() for key in keys}
    head = store.get(kind, ref)
    out = [f"Lineage of {kind} {head.name or '(unnamed)'} [{head.id}]", RULE, ""]
    queue = [(kind, head.id, head, "")]
    seen = {(kind, head.id)}
    step = 0
    while queue:
        k, id_, m, named_by = queue.pop(0)
        step += 1
        if m is None:
            out.append(f"{step}. MISSING {k} [{id_}]")
            out.append(f"{INDENT}named by {named_by}; no complete {k} with that id under "
                       f"{store.kind_dir(k)}")
            out.append("")
            continue
        label = m.name or "(unnamed)"
        out.append(f"{step}. {k} {label} [{m.id}]  created {m.created}")
        out.append(f"{INDENT}inputs:")
        out += [INDENT + line for line in _input_lines(m.inputs)]
        out.append(f"{INDENT}knobs:")
        out += _pack(m.config, INDENT + INDENT)
        acc = _accepted(m)
        if acc:
            out.append(f"{INDENT}accepted: {', '.join(acc)}")
        pairs = [(pk, m.parents[pk]) for pk in sorted(m.parents) if m.parents[pk]]
        out.append(f"{INDENT}parents: " + (", ".join(f"{pk}={pid}" for pk, pid in pairs) or NONE))
        for pk, pid in pairs:
            pkind = key_kind.get(pk)
            if pkind is None:
                # Not silence: a parent key no kind claims is a writer this reader does not know about,
                # and a chain that dropped it would read as complete.
                out.append(f"{INDENT}(parent key {pk!r} names no artifact kind this build knows)")
                continue
            if (pkind, pid) in seen:
                continue
            seen.add((pkind, pid))
            try:
                pm = store.get(pkind, pid)
            except StoreError:
                pm = None
            queue.append((pkind, pid, pm, f"{k} {label!r} as {pk!r}"))
        out.append("")
    return "\n".join(out).rstrip("\n") + "\n"
```

- [ ] **Step 5: Re-export both from `core/artifacts/__init__.py`**

The file is four import lines (`identity`, `manifest`, `provenance`, `store`). Insert the new one
between `provenance` and `store`, keeping the alphabetical order:

```python
from .provenance import git_info, env_info, file_ref, inputs_from_cfg  # noqa: F401
from .report import render_lineage, render_manifest  # noqa: F401
from .store import (ArtifactStore, ArtifactWriter, Accept, Summary, StoreError,  # noqa: F401
```

- [ ] **Step 6: Run the task's own tests**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_artifact_store.py -q -k "render_manifest or render_lineage or create_list_get_round_trip"`

Expected: PASS (3 tests; the round-trip test is there to prove the new `__init__` line broke no
existing import of the package).

- [ ] **Step 7: Commit**

```bash
git add core/artifacts/report.py core/artifacts/__init__.py tests/test_artifact_store.py
git commit -m "artifacts: render a manifest and a lineage report"
```

---

### Task 7: the shared refusal box

**Files:**
- Create: `core/gui/widgets/refusal_box.py`
- Modify: `core/gui/panels/base_panel.py:19` (one new import line, inserted after it)
- Modify: `core/gui/panels/base_panel.py:384-407` (`BasePanel._refusal` — its body only; the signature
  and the log-pane line stay byte-identical)
- Test: `tests/test_worker_dispatch.py` (append one test at the end of the file)

This task touches `base_panel.py` for the refusal-box extraction and nothing else. The file's stale
"Nine of these exist" / "8 of the 9" / "nine independent splitters" sentences (spec §8.2) are **not**
this task's: Task 25 owns every one of them, together with `core/gui/panels/inference/base.py` and
`core/gui/main_window.py`'s class docstring, and it runs last so it counts the panels the piece
actually left behind. Leave those four sentences exactly as they are.

**Interfaces:**
- Consumes: `core.refusals.Refusal` — `.message: str` and `.field: "str | None"`.
  `core.gui.fields.fix_sentence(key: "str | None") -> str` (`core/gui/fields.py:126`), which never
  raises and returns `""` for `None`, an unknown key or a `None` table entry.
- Produces: `core.gui.widgets.refusal_box.show_refusal(parent, exc) -> None` — the one yellow
  "Check your inputs" box. Task 11 (the browser's Note / Delete / Sweep refusals) is its second
  caller and passes the `ArtifactScreen` as `parent`. `BasePanel._refusal(self, exc: Refusal) -> None`
  keeps its signature, keeps writing `f"{exc.message} {fix}".rstrip()` to its log pane at `warning`,
  and now delegates the box.

**Why this task exists:** the artifact browser is a plain `QWidget`, not a `BasePanel` (B1: a
`BasePanel` enrols in `BasePanel._instances`, so a run in any panel would grey out the browser's
controls and you could not read a log while training). It therefore cannot reach
`BasePanel._refusal`, and without an extraction the browser would grow a second, lookalike yellow box
that drifts from the panels' one. Spec §3.1 authorises exactly this: "one refusal surface of its own —
a small helper that shows the same yellow 'Check your inputs' box `BasePanel._refusal` shows, factored
out of `base_panel.py` into `core/gui/widgets/refusal_box.py` so the two callers share one box rather
than two lookalikes. That extraction is the only change to `BasePanel`'s refusal path, and
`BasePanel._refusal` keeps its signature."

- [ ] **Step 1: Write the failing test**

Append this at the end of `tests/test_worker_dispatch.py` — the end of the file, after whatever
Task 1 and Task 8 have already put there. `BasePanel` is already imported at the top of that file
(line 32 before Task 8 adds its three import lines), and `monkeypatch` is a pytest fixture the file
already uses.

```python
def test_the_panel_and_a_plain_widget_show_one_shared_refusal_box(monkeypatch):
    """One yellow box, two callers. BasePanel._refusal keeps its signature and its log-pane line, but
    the box itself is core/gui/widgets/refusal_box.show_refusal(parent, exc) -- so the Artifacts
    screen, which is a plain QWidget and NOT a BasePanel on purpose (piece 4, B1: a BasePanel enrols
    in _instances, and a run in any panel would then grey out the browser while you were reading a
    log), shows the same title, icon, text, fix sentence and single OK button instead of a lookalike
    that drifts from this one.

    Two legs, because "the same box" and "the same code" are different claims. The first compares the
    two boxes field by field off the conftest's SHOWN record. The second pins the DELEGATION: a
    copy-pasted twin in base_panel.py would pass the first leg today and rot the first time either
    copy is edited."""
    from PySide6.QtWidgets import QMessageBox, QWidget
    from core.gui.panels import base_panel as bp
    from core.gui.widgets.refusal_box import show_refusal
    from core.refusals import Refusal
    from tests._fixtures import SHOWN, PaneCapture, qt_app

    qt_app()

    class P(BasePanel):
        pass

    panel = P()
    pane = PaneCapture(panel)
    holder = QWidget()                      # a parent that is NOT a panel; kept bound for its lifetime
    fix = "Set it in the 'T_obs (s)' box on the Infer tab."
    exc = Refusal("The observation length, in seconds, is blank (default none: it must be given).",
                  field="t_obs")

    SHOWN.clear()
    panel._refusal(exc)
    show_refusal(holder, exc)
    assert len(SHOWN) == 2, [b.text() for b in SHOWN]
    for who, box in (("panel", SHOWN[0]), ("widget", SHOWN[1])):
        assert isinstance(box, QMessageBox), who
        assert box.windowTitle() == "Check your inputs", who
        assert box.icon() == QMessageBox.Warning, who
        assert box.text() == exc.message, who
        assert box.informativeText() == fix, who
        assert box.detailedText() == "", f"{who}: a refusal carries no traceback"
        assert box.standardButtons() == QMessageBox.Ok, who
        assert box.buttonRole(box.defaultButton()) == QMessageBox.AcceptRole, f"{who}: OK is default"
    assert pane.lines[-1] == ("warning", f"{exc.message} {fix}")

    # field=None through both callers: no fix sentence, so no informative line at all
    SHOWN.clear()
    bare = Refusal("resume='require' but there is no resumable cache at x.")
    panel._refusal(bare)
    show_refusal(holder, bare)
    assert [b.informativeText() for b in SHOWN] == ["", ""]
    assert [b.text() for b in SHOWN] == [bare.message, bare.message]

    # the delegation itself: the panel must CALL the shared helper and keep no box of its own
    calls = []
    monkeypatch.setattr(bp, "show_refusal", lambda parent, e: calls.append((parent, e)))
    SHOWN.clear()
    panel._refusal(exc)
    assert calls == [(panel, exc)], "BasePanel._refusal no longer routes through refusal_box"
    assert SHOWN == [], "the panel built a box of its own instead of delegating"
    assert pane.lines[-1] == ("warning", f"{exc.message} {fix}"), \
        "the log-pane line is the panel's own (a plain QWidget has no pane) and must stay"
```

- [ ] **Step 2: Run it and watch it fail**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest "tests/test_worker_dispatch.py::test_the_panel_and_a_plain_widget_show_one_shared_refusal_box" -v`

Expected: FAIL, `ModuleNotFoundError: No module named 'core.gui.widgets.refusal_box'` raised on the
`from core.gui.widgets.refusal_box import show_refusal` line.

- [ ] **Step 3: Create the shared box**

Create `core/gui/widgets/refusal_box.py` with exactly this content:

```python
"""The one yellow "Check your inputs" box, shared by every surface that shows a refusal.

Two callers, and the second is why this is not a method. ``BasePanel._refusal`` shows it for every
run-a-thing panel; the Artifacts screen shows it for a refused note, a refused delete and a refused
sweep, and that screen is a plain ``QWidget`` on purpose (piece 4, B1) -- a ``BasePanel`` enrols in
``BasePanel._instances``, so a run in any panel would grey out the browser's controls and you could
not read a log while training. So the helper takes a PARENT WIDGET and assumes nothing about it.

What the box says is fixed here and nowhere else: the text is the core's own neutral sentence (it
names the setting, the rule, what was given and the default, never a box, tab or flag) and the
informative line is this front end's "where to fix it", looked up by the refusal's field key in
core/gui/fields.py -- the one place a control is named, so a renamed control is renamed once. No
Details: nothing here is a crash, and a traceback would only say so louder. A bug keeps the red box
(``BasePanel._on_error``).

It does NOT record the sentence anywhere. A panel has a log pane and the browser has a status line,
so each caller keeps its own record; this helper only ever shows the box.
"""
from PySide6.QtWidgets import QMessageBox

from .. import fields as gui_fields


def show_refusal(parent, exc) -> None:
    """Show ``exc`` (a ``core.refusals.Refusal``, or any subclass such as ``StoreError``) as the
    yellow warning box, modally, parented to ``parent``.

    An INSTANCE dialog built and shown with ``.exec()``, never ``QMessageBox.warning(...)``:
    ``tests/conftest.py::_no_modal_dialogs`` patches the instance method only, and the C++ statics
    escape it -- a static call STALLS the offscreen suite past the tool-call limit instead of failing
    (piece 4 design §1.2).
    """
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Warning)                  # user input, not a crash
    box.setWindowTitle("Check your inputs")
    box.setText(exc.message)
    fix = gui_fields.fix_sentence(exc.field)
    if fix:
        box.setInformativeText(fix)
    box.setStandardButtons(QMessageBox.Ok)
    box.setDefaultButton(QMessageBox.Ok)              # one button, so the safe default is it
    box.exec()
```

- [ ] **Step 4: Have `BasePanel._refusal` delegate**

In `core/gui/panels/base_panel.py`, add the import after line 19
(`from ..widgets.progress_pane import ProgressPane`), keeping the block alphabetical:

```python
from ..widgets.progress_pane import ProgressPane
from ..widgets.refusal_box import show_refusal
from ..worker import Worker
```

Then replace the body of `_refusal` (lines 384-407) with this. The docstring keeps everything it
said and gains one paragraph; the `fix` lookup and the `log_pane.append_line` call are unchanged,
character for character, because six tests read that line.

```python
    def _refusal(self, exc: Refusal) -> None:
        """Show a refusal: the yellow "Check your inputs" box, and no traceback.

        A refusal is something the program will not do with what it was given -- a blank box, a taken
        name, a training run one setting away from its cache -- and the fix is on this screen, so the
        box names it. The text is the core's neutral sentence (it names the setting, the rule, what
        was given and the default; never a box, tab or flag) and the informative line is this front
        end's own "where to fix it", looked up by the refusal's field key in core/gui/fields.py --
        the one place a control is named, so a renamed control is renamed once. No Details: nothing
        here is a crash, and a traceback would only say so louder. A bug keeps the red box
        (_on_error). The same sentence goes to the log pane at warning, so it outlives the click that
        dismisses the box.

        THE BOX ITSELF lives in ../widgets/refusal_box.py, shared with the Artifacts screen (piece 4,
        design §3.1), which shows the same refusals and is deliberately not a BasePanel. What stays
        here is the log-pane line, because only a panel has a pane -- the browser records the sentence
        on its own status line instead.
        """
        fix = gui_fields.fix_sentence(exc.field)
        self.log_pane.append_line(f"{exc.message} {fix}".rstrip(), "warning")
        show_refusal(self, exc)
```

`QMessageBox` stays imported at line 7: `_config_error` (lines 364-382) and `_on_error` (lines
409-426) both still build their own boxes, and neither is touched by this task.

- [ ] **Step 5: Run the new test**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest "tests/test_worker_dispatch.py::test_the_panel_and_a_plain_widget_show_one_shared_refusal_box" -v`

Expected: PASS.

- [ ] **Step 6: Run the existing pins on the yellow box**

These are the tests that already assert on `_refusal`'s box and its log line; the extraction must
leave every one of them green without an edit.

Run:
```
& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_worker_dispatch.py tests/test_simulate.py "tests/test_nav_and_gating.py::test_every_field_key_has_a_window_control_and_the_fix_sentences_name_it" "tests/test_nav_and_gating.py::test_a_rename_failure_reads_as_a_name_refusal" "tests/test_nav_and_gating.py::test_the_inference_tabs_route_builder_failures_by_kind_and_no_longer_call_config_error" "tests/test_nav_and_gating.py::test_the_secondary_panels_still_show_a_bad_cell_as_check_your_inputs" -q
```

Expected: PASS (no test edited).

- [ ] **Step 7: Run the task's own tests**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_worker_dispatch.py tests/test_simulate.py -q`

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add core/gui/widgets/refusal_box.py core/gui/panels/base_panel.py tests/test_worker_dispatch.py
git commit -m "gui: one shared refusal box for the panels and the browser"
```

---

### Task 8: the table widget and its stylesheet

**Files:**
- Create: `core/gui/widgets/artifact_table.py`
- Modify: `core/gui/design.py:241-243` (insert one QSS block between the combo-popup rule, which ends
  at line 241, and the `/* ---- group box as a Fluent card ---- */` comment at line 243)
- Test: `tests/test_worker_dispatch.py` (three import lines added to its import block, then thirteen
  tests appended at the end of the file)

**WHERE THIS TASK'S TESTS LIVE.** In `tests/test_worker_dispatch.py`, beside the other widget tests
(and beside Task 7's, which appends to the same file). This task does **not** create
`tests/test_artifact_browser.py`: **Task 9 creates that suite** and owns its module docstring, so
nothing here writes one and two tasks cannot both claim to be that file's first writer. The row
builders below are therefore this task's own — Task 9's suite builds the Summaries it needs itself.

**Interfaces:**
- Consumes: `core.artifacts.store.Summary` with the six keyword fields the store task adds —
  `dir_name: str = ""`, `finished: bool = False`, `batches_done: "int | None" = None`,
  `batches_planned: "int | None" = None`, `rows: "tuple[int, ...] | None" = None`,
  `variant: "str | None" = None` — beside the existing
  `kind, id, name, created, note, path, complete, reason, mode, width, amortized, parents` and the
  `label` property (`(unnamed <id>)` when `name` is `""`). `batches_planned` is the cache's planned
  batch count, which the store task lifts from `body["identity"]["n_runs"]`
  (`core/artifacts/identity.py:54`), and it is what lets a row show the fraction. Also
  `core.artifacts.store.KIND_DIRS`, the seven kinds in display order (`core/artifacts/store.py:29-31`).
- Produces:
  - `core.gui.widgets.artifact_table.columns_for(kind: str) -> tuple[str, ...]`
  - `core.gui.widgets.artifact_table.cells_for(kind: str, s) -> tuple[str, ...]`
  - `core.gui.widgets.artifact_table._Row(QTreeWidgetItem)`, whose `__lt__` ranks on
    `(not complete, the sort column's own text)` so an incomplete row sorts LAST under every column
    and in both directions — a property of the table, not a side effect of a blank `created`.
  - `core.gui.widgets.artifact_table.ArtifactTable(QTreeWidget)` with
    `selection_changed = Signal()`, `set_rows(self, kind: str, rows: list) -> None`,
    `current_summary(self)`, `sort_state(self) -> tuple[int, int]`,
    `apply_sort_state(self, col: int, order: int) -> None`, and the module constant
    `DEFAULT_SORT = (1, 1)`.
    **Plain ints across the seam, both ways**: `sort_state` returns `(column, order)` with `order` a
    plain `int` — 0 ascending, 1 descending — read as `header().sortIndicatorOrder().value`, because
    `int(Qt.SortOrder)` raises `TypeError: int() argument must be ... not 'SortOrder'` in PySide6
    6.9.3 (the pinned version, per CLAUDE.md), and `apply_sort_state` takes those same plain ints
    (anything `int()` accepts, so the screen can hand back what QSettings gave it, which is a
    string). No caller ever converts a Qt enum, and no `Qt.SortOrder` crosses the boundary.
  - Thirteen tests appended to `tests/test_worker_dispatch.py`, with the module-level row builders
    `_row(kind, **over)` and `_bad_row(kind, **over)`.
  - The `QTreeView` / `QTreeView::item` / `QHeaderView::section` rules in `design.py`, which paint
    every item view the piece adds.

**Why this task exists:** B2 says all seven kinds are listed "one kind at a time, in a real sortable
table", and `Summary` grows "so that a row needs no second manifest read". There is no item view
anywhere in the repository to build that on — `grep -rn "QTableWidget\|QTreeView\|QTreeWidget\|QListWidget\|QAbstractItemView" core/`
returns exactly one hit, `core/gui/design.py:237`, which is the combo box's popup — so `design.py`
styles none, and a `QTreeWidget` dropped in today would render in the Fusion default: a white grid in
dark mode under a header that matches nothing else. Spec §3.2: "this piece adds both". The formatter
is one pair of pure functions because §1.2 requires it — "What must not happen is two front ends
disagreeing about whether a cache is finished."

- [ ] **Step 1: Write the failing formatter tests**

`tests/test_worker_dispatch.py` already sets `QT_QPA_PLATFORM` and `sys.path` at its top
(lines 17-24) and already imports `Path` (line 21) and `qt_app` (line 38), so only the three names
this task adds are new. Insert them into that import block, keeping it roughly alphabetical — the
store import above `core.gui.panels.base_panel` (line 32) and the two widget imports above
`core.gui.widgets.log_pane` (line 35):

```python
from core.artifacts.store import KIND_DIRS, Summary                # noqa: E402
from core.gui.panels.base_panel import BasePanel                   # noqa: E402
from core.gui.streams import redirect_streams                      # noqa: E402
from core.gui.vt import StreamRouter, parse_bar                    # noqa: E402
from core.gui.widgets import artifact_table as at                  # noqa: E402
from core.gui.widgets.artifact_table import cells_for, columns_for  # noqa: E402
from core.gui.widgets.log_pane import LogPane                      # noqa: E402
```

`pytest` itself stays a function-local import, which is this file's own convention (line 495).

Then append to the end of `tests/test_worker_dispatch.py`. The two builders go in at module level,
above the tests, because every leg below needs them:

```python
# ── the artifact browser's table (piece 4, design §3.2) ──────────────────────────────────────────
# NOTHING BELOW READS A STORE. columns_for/cells_for are pure over a Summary (B2: a row needs no
# second manifest read), so every formatting leg builds by hand the Summary it wants, including the
# shapes a real store is slow or awkward to produce -- a cache mid-run, a posterior whose manifest
# records no amortization, a directory with no manifest at all.


def _row(kind, **over):
    """A COMPLETE ``Summary`` of ``kind``, with every field every kind shows set to a recognisable
    value. The cache-only ones (``batches_done``, ``batches_planned``, ``rows``) and the per-kind ones
    (``mode``, ``width``, ``amortized``, ``variant``) keep their dataclass defaults, so a leg that
    wants them passes them: ``over`` replaces any field, and that is how the awkward shapes -- a cache
    mid-run, a posterior with no recorded amortization -- are reached."""
    base = dict(kind=kind, id="20260914T102231", name="run-a", created="2026-09-14T10:22:31",
                note="a note", path=Path("nowhere"), complete=True, reason=None,
                dir_name="run-a__20260914T102231", finished=True)
    base.update(over)
    return Summary(**base)


def _bad_row(kind, **over):
    """An INCOMPLETE ``Summary``, laid out exactly as ``ArtifactStore.list`` builds one
    (core/artifacts/store.py:342): the directory name in both ``id`` and ``dir_name``, no name, no
    created, no note, and a reason."""
    base = dict(kind=kind, id="halfwritten", name="", created="", note="", path=Path("nowhere"),
                complete=False, reason="no manifest.json (incomplete or interrupted)",
                dir_name="halfwritten", finished=False)
    base.update(over)
    return Summary(**base)


def test_columns_for_names_all_seven_kinds_after_name_and_created():
    """Every kind's header row, verbatim (design §3.2's table), and the set of kinds is CLOSED against
    KIND_DIRS: an eighth kind added to the store must gain a column list here or fail this, rather
    than reaching the browser as a KeyError at the click. "Name" and "Created" lead every kind and
    "Note" ends every kind; what differs in between is what only that kind has."""
    import pytest

    assert set(at._EXTRA_COLUMNS) == set(KIND_DIRS), "a kind has no column list (or has a stale one)"
    assert columns_for("prior") == ("Name", "Created", "Note")
    assert columns_for("simulation") == ("Name", "Created", "Progress", "Finished", "Note")
    assert columns_for("posterior") == ("Name", "Created", "Mode", "Width", "Amortized", "Note")
    assert columns_for("observation") == ("Name", "Created", "Mode", "Width", "Note")
    assert columns_for("calibration") == ("Name", "Created", "Note")
    assert columns_for("inference") == ("Name", "Created", "Note")
    assert columns_for("diagnostic") == ("Name", "Created", "Variant", "Note")
    for kind in KIND_DIRS:
        cols = columns_for(kind)
        assert cols[:2] == ("Name", "Created") and cols[-1] == "Note", kind
    with pytest.raises(KeyError):
        columns_for("plot")             # loud: the kind comes from the screen's own KIND_DIRS selector


def test_cells_for_renders_each_kind_from_the_summary_alone():
    """One cell per column, every one a string, and the same length as the header. The values come off
    the Summary and nothing else -- no store, no manifest, no disk (B2)."""
    for kind in KIND_DIRS:
        s = _row(kind)
        cells = cells_for(kind, s)
        assert len(cells) == len(columns_for(kind)), kind
        assert all(isinstance(c, str) for c in cells), (kind, cells)
        assert cells[0] == "run-a" and cells[1] == "2026-09-14T10:22:31", kind
        assert cells[-1] == "a note", kind
    assert cells_for("prior", _row("prior")) == ("run-a", "2026-09-14T10:22:31", "a note")
    assert cells_for("calibration", _row("calibration", note="")) == (
        "run-a", "2026-09-14T10:22:31", "")
    assert cells_for("observation", _row("observation", mode="chi", width=18)) == (
        "run-a", "2026-09-14T10:22:31", "chi", "18", "a note")
    assert cells_for("diagnostic", _row("diagnostic", variant="laplace")) == (
        "run-a", "2026-09-14T10:22:31", "laplace", "a note")
    # ablation records variant=None deliberately (core/diagnostics/ablation.py:206): blank, not "None"
    assert cells_for("diagnostic", _row("diagnostic", variant=None))[2] == ""
    # an observation carries no amortization, so it gets no such column at all
    assert "Amortized" not in columns_for("observation")


def test_a_caches_progress_and_whether_it_finished_come_apart():
    """B3. ``complete`` means "has a valid manifest" and a cache is manifested from its first batch on,
    so a row must say BOTH: the progress, and whether the run finished. Both halves of the fraction
    are on the Summary -- ``batches_done`` and ``batches_planned``, the latter lifted from
    ``body["identity"]["n_runs"]`` by the store task -- so the cell reads "3/4 batches" and an
    operator can see how far a running cache has to go. The rows-per-batch list is different: it is
    written only by ``mark_complete`` (core/SBI/training_checkpoint.py:346), so a running cache has no
    row count and the cell adds one only once it exists."""
    running = _row("simulation", name="", batches_done=3, batches_planned=4, rows=None, finished=False)
    done = _row("simulation", name="", batches_done=4, batches_planned=4, rows=(24, 24, 24, 24),
                finished=True)
    assert cells_for("simulation", running)[2:4] == ("3/4 batches", "no")
    assert cells_for("simulation", done)[2:4] == ("4/4 batches \u00b7 96 rows", "yes")
    assert cells_for("simulation", _row("simulation", batches_done=1, batches_planned=4))[2] == \
        "1/4 batches"
    # a manifest that records no planned count (nothing writes one today, but a hand-edited or an
    # older manifest can) drops to the count alone rather than showing "1/None"
    assert cells_for("simulation", _row("simulation", batches_done=1, batches_planned=None))[2] == \
        "1 batch"
    assert cells_for("simulation", _row("simulation", batches_done=3, batches_planned=None))[2] == \
        "3 batches"
    # nothing to say at all: no batch count, so no cell -- not "0 batches", which would claim a fact
    assert cells_for("simulation", _row("simulation", batches_done=None))[2] == ""
    # both are complete-with-a-manifest; only `finished` tells them apart
    assert running.complete and done.complete
    # a kind with no progress to report leaves the cell out entirely
    assert "Progress" not in columns_for("prior")


def test_a_posteriors_amortization_is_spelled_out_and_the_norm_is_named_too():
    """The column, unlike the picker's item text (§6.3, which suffixes the exception only), names both
    states: a table column that is blank for the common case reads as missing data."""
    assert cells_for("posterior", _row("posterior", mode="chi", width=18, amortized=True)) == (
        "run-a", "2026-09-14T10:22:31", "chi", "18", "amortized", "a note")
    assert cells_for("posterior", _row("posterior", amortized=False))[4] == "narrowed (TSNPE)"
    assert cells_for("posterior", _row("posterior", amortized=None))[4] == ""
    assert cells_for("posterior", _row("posterior", mode=None, width=None))[2:4] == ("", "")


def test_an_unnamed_artifact_shows_its_label():
    """``Summary.label`` is ``(unnamed <id>)``, which is what the pickers show, so the table and the
    dropdowns name the same artifact the same way. Every simulation cache is unnamed by construction
    (``write_simulation_manifest`` always writes ``name=""``)."""
    s = _row("prior", name="")
    assert s.label == "(unnamed 20260914T102231)"
    assert cells_for("prior", s)[0] == "(unnamed 20260914T102231)"


def test_an_incomplete_row_shows_its_directory_name_and_its_reason():
    """A directory with no usable manifest has no id, no name, no created and no note -- six blank
    cells would read as an artifact with nothing in it. It is identified by its ``dir_name`` (the
    handle ``remove_incomplete`` takes) and carries its reason in the third column, in place of the
    kind's own columns."""
    for kind in KIND_DIRS:
        cells = cells_for(kind, _bad_row(kind))
        assert len(cells) == len(columns_for(kind)), kind
        assert cells[0] == "halfwritten", kind
        assert cells[1] == "", kind
        assert cells[2] == "no manifest.json (incomplete or interrupted)", kind
        assert all(c == "" for c in cells[3:]), (kind, cells)
    assert cells_for("simulation", _bad_row("simulation", reason=None))[2] == ""
```

- [ ] **Step 2: Run them and watch them fail**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_worker_dispatch.py -q`

Expected: `ERROR collecting tests/test_worker_dispatch.py` —
`ModuleNotFoundError: No module named 'core.gui.widgets.artifact_table'`, raised on the new import
line. The import is at module level, so the WHOLE file fails to collect and not just the six new
tests; that is the expected shape of this failure, and Step 4 is what turns it back into a pass.

- [ ] **Step 3: Write the formatter**

Create `core/gui/widgets/artifact_table.py` with the module docstring, the column table and the four
pure functions — `columns_for` and `cells_for`, plus the two cell helpers. (The widget and its row
item follow in Step 7 — write only this much now.)

```python
"""The artifact browser's table: the ONE formatter for a store row, and the flat sortable tree that
shows it (piece 4, B2, design §3.2).

``columns_for`` and ``cells_for`` are pure functions over a ``core.artifacts.store.Summary`` and read
NOTHING -- no store, no manifest, no disk. That is what B2 buys: ``Summary`` carries every fact a row
shows, so a listing costs one directory scan and a row costs nothing. It is also why the pickers'
selection line (§6.3) is built from these and not from a second reader: what must not happen is two
surfaces disagreeing about whether a cache is finished (§1.2).

There was no item view anywhere in the repository before this piece and ``core/gui/design.py`` styled
none, so the QSS for ``QTreeView``/``QTreeView::item``/``QHeaderView::section`` lands there in the
same commit as this file.
"""
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QAbstractItemView, QTreeWidget, QTreeWidgetItem

# Every kind leads with these two.
_LEADING = ("Name", "Created")

# Per kind, the columns BETWEEN "Created" and "Note" -- what only that kind has (design §3.2's
# table). CLOSED against store.KIND_DIRS by the tests: an eighth kind must appear here or fail there.
# A diagnostic shows `variant` and not `mode`, deliberately: it records its kind under `variant` and
# has no conditioning geometry, so a mode column could only ever be blank for it (§1.3).
_EXTRA_COLUMNS: dict[str, tuple[str, ...]] = {
    "prior": (),
    "simulation": ("Progress", "Finished"),
    "posterior": ("Mode", "Width", "Amortized"),
    "observation": ("Mode", "Width"),
    "calibration": (),
    "inference": (),
    "diagnostic": ("Variant",),
}

# "Created", descending: newest first, which is the order ArtifactStore.list already returns. Used on
# the first fill and whenever a remembered sort names a column the current kind does not have. (That
# an INCOMPLETE row sorts last is not this constant's doing -- _Row.__lt__ below enforces it under
# every column and in both directions.) A plain int for the order, never a Qt.SortOrder: see
# ArtifactTable.sort_state.
DEFAULT_SORT = (1, 1)


def columns_for(kind: str) -> tuple[str, ...]:
    """The header labels for one kind's table: "Name", "Created", the kind's own, then "Note".

    KeyError for an unknown kind -- loud on purpose, exactly as ``core/gui/fields.py::label`` is: the
    kind comes from the screen's own selector, which is built from ``KIND_DIRS``, so a kind this
    table does not know is a programming mistake and not something an operator can type.
    """
    return _LEADING + _EXTRA_COLUMNS[kind] + ("Note",)


def _progress(s) -> str:
    """A cache's progress: ``"3/4 batches"`` while it runs, ``"4/4 batches \u00b7 96 rows"`` when done.

    The fraction, as in design §3.2's example, and it still costs no second read: BOTH halves are on
    the Summary, because the store task carries the planned total across as ``batches_planned`` (from
    ``body["identity"]["n_runs"]``, ``core/artifacts/identity.py:54``) beside ``batches_done``. A
    manifest that records no planned count -- nothing writes one without it today, but a hand-edited
    or an older manifest can -- falls back to the count alone rather than printing "3/None".

    The ROW count is a separate matter: ``rows`` is written only by ``mark_complete``
    (``core/SBI/training_checkpoint.py:346``), so a running cache does not know how many rows it has
    and the cell adds them only once they exist. The cell says what it knows and no more.
    """
    if s.batches_done is None:
        return ""
    n = int(s.batches_done)
    if s.batches_planned is None:
        head = f"{n} {'batch' if n == 1 else 'batches'}"
    else:
        head = f"{n}/{int(s.batches_planned)} batches"
    if not s.rows:
        return head
    return f"{head} \u00b7 {sum(int(r) for r in s.rows)} rows"


def _amortization(s) -> str:
    """``amortized`` or ``narrowed (TSNPE)``; blank when the manifest records neither.

    A column names BOTH states, unlike the picker's visible item text (§6.3), which suffixes the
    exception only: a column that is blank for the common case reads as missing data, whereas a
    dropdown entry that is silent reads as the norm.
    """
    if s.amortized is None:
        return ""
    return "amortized" if s.amortized else "narrowed (TSNPE)"


def cells_for(kind: str, s) -> tuple[str, ...]:
    """One row's cells in ``columns_for(kind)`` order, every one a string.

    An INCOMPLETE row -- a directory with no usable manifest, which ``ArtifactStore.list`` returns
    with ``complete=False`` and a ``reason`` -- has no id, no name, no created and no note, so it is
    identified by its ``dir_name`` (the handle ``remove_incomplete`` takes) and carries its reason in
    the third column, in place of the kind's own. Six blank cells would read as an artifact with
    nothing in it; for the three kinds whose third column is "Note" the reason simply sits there,
    which is the only column such a row has to give it.
    """
    columns = columns_for(kind)
    if not s.complete:
        head = (s.dir_name, "", s.reason or "")
        return head + ("",) * (len(columns) - len(head))
    middle: tuple[str, ...] = ()
    if kind == "simulation":
        middle = (_progress(s), "yes" if s.finished else "no")
    elif kind == "posterior":
        middle = (s.mode or "", "" if s.width is None else str(s.width), _amortization(s))
    elif kind == "observation":
        middle = (s.mode or "", "" if s.width is None else str(s.width))
    elif kind == "diagnostic":
        middle = (s.variant or "",)
    return (s.label, s.created) + middle + (s.note or "",)
```

- [ ] **Step 4: Run the formatter tests**

Run:
```
& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_worker_dispatch.py -v -k "columns_for or cells_for or caches_progress or amortization or shows_its_label or directory_name_and_its_reason"
```

Expected: PASS, 6 selected. If `Summary(**base)` raises `TypeError: unexpected keyword argument
'dir_name'` — or `'batches_planned'` — the store task's `Summary` fields are not in yet; that task
comes first.

- [ ] **Step 5: Write the failing widget tests**

First widen the import line Step 1 added to `tests/test_worker_dispatch.py`, so the widget is named
at module level:

```python
from core.gui.widgets.artifact_table import ArtifactTable, cells_for, columns_for   # noqa: E402
```

Then append to the end of `tests/test_worker_dispatch.py`:

```python
def test_the_table_is_flat_read_only_and_selects_whole_rows():
    """A QTreeWidget used FLAT, not a hand-built grid of QLabels: the chi probe table's and
    SettingsScreen.refresh_models' row-building idiom neither sorts nor scales, and the store can hold
    hundreds of rows. No expander column in front of "Name", no in-place editing (a note is edited in
    its own box, so a double-click must not turn a cell into a line edit), one row at a time because
    delete is one artifact at a time (B6)."""
    from PySide6.QtWidgets import QAbstractItemView

    qt_app()
    table = ArtifactTable()
    assert table.rootIsDecorated() is False
    assert table.isSortingEnabled() is True
    assert table.selectionBehavior() == QAbstractItemView.SelectRows
    assert table.selectionMode() == QAbstractItemView.SingleSelection
    assert table.editTriggers() == QAbstractItemView.NoEditTriggers
    assert table.current_summary() is None, "an empty table has no selection"


def test_set_rows_fills_the_header_and_hands_back_the_selected_summary():
    """The table holds the Summary objects it was given and returns the selected one, so a caller
    never parses a cell back into a fact -- the detail pane, Note, Delete and Sweep all need the id,
    the kind and the dir_name, none of which is on screen in full."""
    qt_app()
    table = ArtifactTable()
    rows = [_row("posterior", name="beta", created="2026-09-14T10:22:31", amortized=False, width=18,
                 mode="chi"),
            _row("posterior", name="alpha", created="2026-09-10T08:00:00", id="20260910T080000")]
    table.set_rows("posterior", rows)

    assert table.topLevelItemCount() == 2
    header = table.headerItem()
    assert [header.text(c) for c in range(table.columnCount())] == list(columns_for("posterior"))
    first = table.topLevelItem(0)
    assert [first.text(c) for c in range(table.columnCount())] == list(
        cells_for("posterior", rows[0])), "the default sort must put the newest first"

    table.setCurrentItem(first)
    got = table.current_summary()
    assert got is rows[0] and got.id == "20260914T102231"
    table.clearSelection()
    assert table.current_summary() is None


def test_the_default_sort_is_newest_first_with_the_incomplete_rows_last():
    """ArtifactStore.list already returns that order (newest first, incomplete last) and the table
    must not undo it. A QTreeWidget with sorting enabled sorts by column 0 ascending unless told
    otherwise, which would put an alphabetical Name order in front of an operator looking for the run
    they just made -- so the default is "Created" descending. That the incomplete row also lands last
    is _Row's doing and not this constant's; the next test is the one that pins it."""
    qt_app()
    table = ArtifactTable()
    rows = [_row("prior", name="newer", created="2026-09-14T10:22:31"),
            _row("prior", name="older", created="2026-09-10T08:00:00", id="20260910T080000"),
            _bad_row("prior")]
    table.set_rows("prior", rows)
    assert at.DEFAULT_SORT == (1, 1)
    assert table.sort_state() == (1, 1)
    assert [table.topLevelItem(i).text(0) for i in range(3)] == ["newer", "older", "halfwritten"]


def test_an_incomplete_row_sorts_last_under_every_column_and_either_order():
    """Incomplete rows sort last is a PROPERTY of the table (design §3.2), not a side effect of an
    incomplete row's blank `created`. Three of the four sorts below break the accident: sorted by
    Name, "halfwritten" lands BETWEEN "alpha" and "newer" in either direction, and sorted by Created
    ASCENDING its blank cell floats to the very top -- only the default, Created descending, is right
    on its own. So _Row.__lt__ ranks on completeness before the column's own value, and flips that
    rank for a descending sort because Qt inverts the whole comparison; the leftover directories then
    stay at the bottom under every column and in both directions, out of the way of the newest real
    artifact, which is where a selection-driven Delete wants them."""
    qt_app()
    table = ArtifactTable()
    rows = [_row("prior", name="newer", created="2026-09-14T10:22:31"),
            _row("prior", name="alpha", created="2026-09-10T08:00:00", id="20260910T080000"),
            _bad_row("prior")]
    table.set_rows("prior", rows)

    def names():
        return [table.topLevelItem(i).text(0) for i in range(table.topLevelItemCount())]

    table.apply_sort_state(0, 0)                    # Name, ascending: "alpha" < "halfwritten"
    assert names() == ["alpha", "newer", "halfwritten"]
    table.apply_sort_state(0, 1)                    # Name, descending
    assert names() == ["newer", "alpha", "halfwritten"]
    table.apply_sort_state(1, 0)                    # Created, ascending: the blank one does NOT lead
    assert names() == ["alpha", "newer", "halfwritten"]
    table.apply_sort_state(1, 1)                    # Created, descending (the default)
    assert names() == ["newer", "alpha", "halfwritten"]

    # the rows keep their Summaries through every reorder: the item carries its INDEX, not its cells
    table.setCurrentItem(table.topLevelItem(2))
    assert table.current_summary() is rows[2] and not table.current_summary().complete


def test_the_sort_state_round_trips_and_a_narrower_kind_falls_back():
    """What the screen remembers is the kind and the sort (§3.5) -- two PLAIN ints, because QSettings
    stores ints and a Qt enum does not survive the round trip. Plain both ways, and not as a
    convenience: `int(Qt.SortOrder)` raises `TypeError` in PySide6 6.9.3, so the order is read as
    `header().sortIndicatorOrder().value` inside the widget and no caller ever holds a Qt enum. The
    kinds have different widths, so a sort on the posterior's "Amortized" column cannot survive a
    switch to the prior kind: it falls back to the default rather than to column 0, which would
    quietly re-sort by Name -- neither the order the store hands back nor one the user asked for."""
    qt_app()
    table = ArtifactTable()
    posteriors = [_row("posterior", name="beta", amortized=True),
                  _row("posterior", name="alpha", amortized=False, id="20260910T080000",
                       created="2026-09-10T08:00:00")]
    table.set_rows("posterior", posteriors)
    assert [type(v) for v in table.sort_state()] == [int, int], \
        "a Qt enum crossed the seam; int(Qt.SortOrder) raises TypeError on this PySide6"

    table.apply_sort_state(0, 0)                       # by Name, ascending
    assert table.sort_state() == (0, 0)
    assert [table.topLevelItem(i).text(0) for i in range(2)] == ["alpha", "beta"]
    table.apply_sort_state(0, 1)
    assert table.sort_state() == (0, 1)
    assert [table.topLevelItem(i).text(0) for i in range(2)] == ["beta", "alpha"]

    table.apply_sort_state("1", "0")                   # what a QSettings round trip hands back
    assert table.sort_state() == (1, 0)

    table.apply_sort_state(4, 0)                       # by Amortized: posterior-only, column 4
    assert table.sort_state() == (4, 0)
    table.set_rows("posterior", posteriors)
    assert table.sort_state() == (4, 0), "a refill must keep the sort the user chose"

    table.set_rows("prior", [_row("prior", name="p")])  # three columns: 4 is gone
    assert table.columnCount() == 3
    assert table.sort_state() == at.DEFAULT_SORT

    table.apply_sort_state(9, 0)                       # out of range: ignored, never clamped to 0
    assert table.sort_state() == at.DEFAULT_SORT


def test_selection_changed_fires_on_a_selection_and_a_refresh_drops_it():
    """One signal, so the detail pane, the Note box and the two buttons are driven from one place. A
    refresh clears the table, so the selection goes with it and the pane must tolerate a None -- the
    alternative, remembering an id across a refresh, is exactly the dangling state §3.4 protects the
    pickers from."""
    qt_app()
    table = ArtifactTable()
    fired = []
    table.selection_changed.connect(lambda: fired.append(1))
    rows = [_row("prior", name="p"), _row("prior", name="q", id="20260910T080000",
                                          created="2026-09-10T08:00:00")]
    table.set_rows("prior", rows)
    assert fired == [], "filling an empty table selects nothing"

    table.setCurrentItem(table.topLevelItem(0))
    assert fired, "selecting a row must emit selection_changed"
    assert table.current_summary() is not None

    fired.clear()
    table.set_rows("prior", rows)
    assert table.current_summary() is None, "a refresh must drop the stale selection"
    assert fired, "clearing the selection must emit selection_changed too"


def test_the_stylesheet_paints_the_item_view_in_both_themes():
    """core/gui/design.py styled NO item view before this piece: its one QAbstractItemView rule is the
    combo popup's (design.py:237). So a QTreeWidget would have rendered in the Fusion default -- a
    white grid in dark mode under a header matching nothing else. The rules go in design.py, in the
    same token vocabulary as the rest, so a theme flip recolours them for free (theming.Appearance
    re-applies build_qss on every change).

    The substitution is the real regression risk: _QSS is a string.Template, so one $name that is not
    a key of _qss_vars raises KeyError inside substitute() and takes the WHOLE stylesheet down, not
    just the new block. Both themes and an accent override are built here for that reason."""
    from core.gui import design

    for dark in (False, True):
        qss = design.build_qss(dark)
        for selector in ("QTreeView {", "QTreeView::item", "QHeaderView::section"):
            assert selector in qss, (dark, selector)
        t = design.tokens(dark)
        assert f"alternate-background-color: {t['alt_base']}" in qss, dark
        assert f"background: {t['alt_base']}; color: {t['text_2nd']}" in qss, (dark, "header section")
        assert "$" not in qss, "an unsubstituted token escaped"
    assert "QTreeView" in design.build_qss(True, "#AA3366")
```

- [ ] **Step 6: Run them and watch them fail**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_worker_dispatch.py -q`

Expected: `ERROR collecting tests/test_worker_dispatch.py` — `ImportError: cannot import name
'ArtifactTable' from 'core.gui.widgets.artifact_table'`. The module holds only the formatter, so the
widened import line from Step 5 is what breaks, and no test in the file is collected.

- [ ] **Step 7: Write the widget**

Append to `core/gui/widgets/artifact_table.py`:

```python
class _Row(QTreeWidgetItem):
    """One row, which sorts INCOMPLETE LAST under every column and in both directions.

    Two things make this a class rather than a blank cell that happens to sort well. The rank is on
    completeness BEFORE the column's own value, so a sort by Name cannot float a leftover directory
    ("halfwritten") above a real artifact ("newer"); and Qt inverts the whole comparison for a
    descending sort, which would put the incomplete rows on top, so the completeness bit is flipped
    with the order and comes out the same way round either way. Design §3.2 states "incomplete rows
    sort last" as a property of the table -- this is where it is a property and not an accident of an
    incomplete row's blank ``created``.

    The item carries the row's INDEX into ``ArtifactTable._rows``, not the Summary itself: Qt item
    data is a QVariant and only PySide6's wrapping makes an arbitrary object survive it, while
    sorting reorders the ITEMS and never that list.
    """

    def __init__(self, cells, index: int, complete: bool):
        super().__init__(list(cells))
        self.setData(0, Qt.UserRole, int(index))
        self._complete = bool(complete)

    def _key(self, col: int, descending: bool):
        """``(rank, the column's own text)``. ``!= descending`` is the XOR that cancels Qt's
        reversal, so the incomplete rows sit at the bottom under an ascending and a descending sort
        alike."""
        return ((not self._complete) != descending, self.text(col))

    def __lt__(self, other):
        tree = self.treeWidget()
        col = tree.sortColumn() if tree is not None else 0
        if col < 0:                                 # nothing sorted yet: rank by the first column
            col = 0
        if tree is None or not isinstance(other, _Row):     # neither happens; cheap to survive
            return self.text(col) < other.text(col)
        # .value, not int(): int(Qt.SortOrder) raises TypeError in PySide6 6.9.3.
        descending = bool(tree.header().sortIndicatorOrder().value)
        return self._key(col, descending) < other._key(col, descending)


class ArtifactTable(QTreeWidget):
    """A flat, sortable, read-only table over one kind's rows, one row selectable at a time.

    It holds the ``Summary`` objects it was given and hands the selected one back, so a caller never
    parses a cell back into a fact: the detail pane needs the id, Delete needs the kind and the id,
    and Sweep needs the ``dir_name``, none of which is on screen in full.

    ``selection_changed`` is the one signal the screen listens to. Selecting a row emits it, and so
    does a refresh, which clears the table and therefore the selection -- ``current_summary()`` is
    None until something is selected again. That is deliberate: remembering an id across a refresh is
    the dangling state design §3.4 protects the store pickers from.
    """
    selection_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._kind = ""
        self._rows: list = []
        self.setRootIsDecorated(False)      # flat: no expander column in front of "Name"
        self.setUniformRowHeights(True)
        self.setAlternatingRowColors(True)
        self.setAllColumnsShowFocus(True)
        # A note is edited in its own box, with require_note run at the click (B5), so a double-click
        # on a cell must not open an editor that writes nothing and refuses nothing.
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setSortingEnabled(True)
        self.itemSelectionChanged.connect(self.selection_changed.emit)

    def kind(self) -> str:
        """The kind currently shown, or "" before the first fill."""
        return self._kind

    def set_rows(self, kind: str, rows: list) -> None:
        """Replace the contents with one kind's ``Summary`` rows, keeping the sort the user chose.

        Sorting is switched OFF while the items go in and back on afterwards: with it on, Qt re-sorts
        on every single insert, which is both quadratic and enough to move a row out from under the
        item being added. The sort is then re-applied, falling back to ``DEFAULT_SORT`` when the
        remembered column is not in this kind's header -- the kinds have different widths.

        The items are ``_Row``s, which is what keeps an incomplete row at the bottom whatever the
        sort, and each one carries its index into ``self._rows``.
        """
        col, order = self.sort_state()
        self.setSortingEnabled(False)
        self.clear()
        self._kind, self._rows = kind, list(rows)
        columns = columns_for(kind)
        self.setColumnCount(len(columns))
        self.setHeaderLabels(list(columns))
        for i, s in enumerate(self._rows):
            self.addTopLevelItem(_Row(cells_for(kind, s), i, s.complete))
        self.setSortingEnabled(True)
        if not 0 <= col < len(columns):
            col, order = DEFAULT_SORT
        self.apply_sort_state(col, order)
        for c in range(len(columns)):
            self.resizeColumnToContents(c)

    def current_summary(self):
        """The selected row's ``Summary``, or None when nothing is selected."""
        chosen = self.selectedItems()
        if not chosen:
            return None
        i = chosen[0].data(0, Qt.UserRole)
        return self._rows[i] if isinstance(i, int) and 0 <= i < len(self._rows) else None

    # ── what the screen remembers (§3.5: the kind and the sort, never the selection) ─────────────
    def sort_state(self) -> tuple[int, int]:
        """``(column, order)`` as PLAIN ints -- 0 ascending, 1 descending.

        Plain in both directions, and not merely as a convenience: QSettings stores ints and hands
        them back as strings, and a Qt enum does not survive that round trip. The order is read as
        ``.value`` off the indicator rather than as ``int(...)``, because ``int(Qt.SortOrder)`` raises
        ``TypeError`` in PySide6 6.9.3 -- the pinned version -- so no caller and nothing here ever
        converts a Qt enum.

        ``DEFAULT_SORT`` before the first fill, where there is no header to read a sort off, and
        likewise when Qt reports no sorted column (-1).
        """
        if self.columnCount() == 0:
            return DEFAULT_SORT
        col = self.sortColumn()
        if not 0 <= col < self.columnCount():
            return DEFAULT_SORT
        return int(col), self.header().sortIndicatorOrder().value

    def apply_sort_state(self, col: int, order: int) -> None:
        """Sort by ``col`` in ``order`` -- 0 ascending, 1 descending, both PLAIN ints.

        Anything ``int()`` accepts, so the screen hands back exactly what QSettings gave it (a
        string) and never builds a ``Qt.SortOrder`` of its own; ``sort_state`` is the same seam in
        reverse.

        A column outside the current header is IGNORED rather than clamped to 0: a remembered sort
        may name a column this kind does not have, and quietly re-sorting by "Name" is not what the
        user chose. ``set_rows`` is where that case is handled, by substituting ``DEFAULT_SORT``.
        """
        col, order = int(col), int(order)
        if not 0 <= col < self.columnCount():
            return
        indicator = Qt.DescendingOrder if order else Qt.AscendingOrder
        # THE INDICATOR FIRST, then the sort. QTreeWidget.sortItems sorts the model and only then
        # updates the header, so a _Row reading the indicator mid-comparison would see the PREVIOUS
        # order and hoist the incomplete rows to the top. Setting the indicator on a sorting-enabled
        # view already re-sorts (QTreeView connects the header's sortIndicatorChanged), so sortItems
        # below is the brace to that belt: same column, same order, so it is not a second ordering.
        self.header().setSortIndicator(col, indicator)
        self.sortItems(col, indicator)
```

- [ ] **Step 8: Run the widget tests**

Run:
```
& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_worker_dispatch.py -v -k "flat_read_only or set_rows_fills or default_sort or sorts_last_under_every_column or sort_state_round_trips or selection_changed_fires or stylesheet_paints"
```

Expected: 7 selected; six PASS and `test_the_stylesheet_paints_the_item_view_in_both_themes` FAILS
with `AssertionError: (False, 'QTreeView {')` — `design.build_qss` styles no item view yet.

- [ ] **Step 9: Add the item-view stylesheet**

In `core/gui/design.py`, insert this block into the `_QSS` template between the combo-popup rule
(which closes at line 241) and the `/* ---- group box as a Fluent card ---- */` comment at line 243.
Only `$`-names that are keys of `_qss_vars` (lines 340-356) may appear, or `Template.substitute`
raises and takes the whole stylesheet with it.

```
/* ---- item views: the artifact browser's table, a QTreeWidget used flat. The app's only item view
   besides the combo popup above, so these are all of it. Selection colour comes from $accent, which
   is also QPalette.Highlight, so a custom-painted row would match. ---- */
QTreeView {
    border: 1px solid $mid; border-radius: ${radius_sm}px;
    background: $base; alternate-background-color: $alt_base; color: $text;
    selection-background-color: $accent; selection-color: $on_accent; outline: 0;
}
QTreeView::item { padding: 4px 6px; border: none; }
QTreeView::item:hover { background: $button_hover; }
QTreeView::item:selected { background: $accent; color: $on_accent; }
QHeaderView { background: transparent; border: none; }
QHeaderView::section {
    background: $alt_base; color: $text_2nd;
    padding: 4px 6px; border: none;
    border-right: 1px solid $mid; border-bottom: 1px solid $mid; font-weight: 600;
}
QHeaderView::section:hover { color: $text; }
QHeaderView::section:last, QHeaderView::section:only-one { border-right: none; }
```

Two things about the order of the rules: `::item:selected` is listed AFTER `::item:hover` on purpose,
because Qt resolves equal specificity by the last rule, so hovering a selected row keeps the accent;
and the bare `QTreeView` rule cannot reach the combo popup, whose view is a `QListView` styled by the
`QComboBox QAbstractItemView` rule above.

- [ ] **Step 10: Run the task's own tests**

Run:
```
& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_worker_dispatch.py "tests/test_user_models.py::test_accent_tokens_and_palette_override" -q
```

Expected: PASS — the whole of `tests/test_worker_dispatch.py` (its own tests plus the thirteen this
task appended), because the import lines Step 1 added run for every test in the file.
`tests/test_user_models.py:752-766` is the existing pin on `design.tokens` and `design.build_qss`
with an accent override, and it is the test a broken `$`-substitution takes down first.

- [ ] **Step 11: Commit**

```bash
git add core/gui/widgets/artifact_table.py core/gui/design.py tests/test_worker_dispatch.py
git commit -m "gui: the artifact table and its stylesheet"
```

### Task 9: the Artifacts screen, registered, listing

**Files:**
- Create: `core/gui/screens/artifact_screen.py`
- Create: `tests/test_artifact_browser.py`
- Modify: `tests/_fixtures.py:338` (insert the two new helpers between `build_tiny_run`, which ends at
  line 337, and `_same_value`, which starts at line 340)
- Modify: `core/gui/screens/home_screen.py:11-12` (the comment and `SECTIONS`) and
  `core/gui/screens/home_screen.py:27` (the class docstring's "all four are live today")
- Modify: `core/gui/main_window.py:1-5` (module docstring), `:19` (a new import above
  `from .screens.home_screen import HomeScreen`), `:24` (a new import after
  `from .screens.settings_screen import SettingsScreen`), `:101-102`
  (`live_sections`), `:107` (insert the screen's construction after `idx_sim = ...`), `:131-133`
  (`_section_index`), `:265` (insert `_refresh_store_pickers` after `_refresh_model_combos`),
  `:306-312` (`_save_state`'s body)
  — and NOT `MainWindow`'s CLASS docstring (`:70-75`, the "five section screens" and "the only
  QSettings WRITE site" sentences): Task 25 owns that whole docstring, runs last, and rewrites it
  once. This task touches the five registration points, the module docstring and the prose comments
  that cite a moved line, and nothing else in the file.
- Modify: `tests/test_worker_dispatch.py:338` (a docstring citing `main_window.py:231`, a line this
  task moves). **Only that one.** `tests/conftest.py:174` cites the same line, and Task 11 rewrites
  that sentence — deleting the number rather than re-pointing it — so this task leaves the conftest
  alone and does not `git add` it (Q10).
- Test: `tests/test_artifact_browser.py`

**Interfaces:**
- Consumes, from Task 8 (`core/gui/widgets/artifact_table.py`):
  `columns_for(kind: str) -> tuple[str, ...]`; `class ArtifactTable(QTreeWidget)` built with
  `ArtifactTable()`, carrying `selection_changed = Signal()`, `set_rows(self, kind: str, rows: list) -> None`,
  `current_summary(self)` (the selected `Summary`, or `None`), `sort_state(self) -> tuple[int, int]`
  and `apply_sort_state(self, col: int, order: int) -> None`. Two properties of that widget this
  screen depends on and its own tests pin from the outside: `set_rows` preserves the order of the
  list it is given (so `ArtifactStore.list`'s complete-first ordering survives and incomplete rows
  stay last until the user clicks a header), and it leaves nothing selected. `sort_state` returns
  plain `int`s in both slots — `int(Qt.SortOrder)` raises `TypeError` in PySide6 6.9
  (`TypeError: int() argument must be a string, a bytes-like object or a real number, not 'SortOrder'`),
  so the conversion belongs in the table and never here.
- Consumes, from the store task that grew `Summary`, its SIX new fields: `Summary.dir_name: str`,
  `Summary.finished: bool`, `Summary.batches_done: "int | None"`,
  `Summary.batches_planned: "int | None"` (the simulation body's `identity["n_runs"]`),
  `Summary.rows: "tuple[int, ...] | None"`, `Summary.variant: "str | None"` — beside today's
  `kind, id, name, created, note, path, complete, reason, mode, width, amortized, parents` and the
  `label` property. Nothing here builds a `Summary` (the fixture writes manifests and lets
  `ArtifactStore.list` read them) and nothing asserts its field set, so the sixth field costs this
  task nothing but the `n_runs` already in the fixture's identity.
- Consumes, unchanged: `core.artifacts.store.KIND_DIRS` (the seven kinds, in order),
  `ArtifactStore.list(kind)`, `ArtifactStore.kind_dir(kind)`, `core.artifacts.resolve_store(store)`,
  `core.gui.settings.{settings, get_str, get_int}`.
- Produces, for Task 10 and for the actions tasks:
  `core/gui/screens/artifact_screen.py::ArtifactScreen(store=None, parent=None)` with
  `store_changed = Signal()`, `refresh(self) -> None`, `kind(self) -> str`,
  `save_settings(self, qs) -> None`, `restore_settings(self, qs) -> None`,
  `_resolved_store(self)`, `_set_status(self, text: str, error: bool = False) -> None`,
  `_apply_sort(self, kind: str) -> None` and the `_sort` attribute it reads (the remembered
  `(column, order)`, both plain ints — Task 10 re-quotes `refresh`, which calls `_apply_sort`), and
  the widgets `kind_combo`, `table`, `status`, `split` (a horizontal `QSplitter` holding the table as
  its only child, so Task 10's detail pane is one `addWidget`).
- Produces, in `core/gui/main_window.py`: `MainWindow.artifact_screen`,
  `MainWindow._refresh_store_pickers(self) -> None`, and `_section_index["Artifacts"]`. **This task
  is the only one that defines `_refresh_store_pickers` and the only one that connects
  `artifact_screen.store_changed` to it** (Q1): Task 11 calls the method from its own test and adds
  neither (one method defined twice in one class, and one signal connected twice, is what that would
  otherwise mean).
- Produces, in `tests/_fixtures.py`: `build_browse_store(root) -> dict` and
  `artifact_screen(store) -> ArtifactScreen`.

**Why this task exists:** The store has been complete since piece 1 and almost none of it is
reachable: `list` has exactly one production caller (`StorePicker.refresh`), and four of the seven
kinds — the simulation cache, calibrations, inferences and diagnostics — have no front-end surface
whatever (spec §1). **B1** puts the browser on a fifth Home tile as a plain `QWidget`, **not** a
`BasePanel`: a `BasePanel` enrols itself in `BasePanel._instances` and `_set_busy` disables
`panel.controls` on *every* instance while *any* run is live (`core/gui/panels/base_panel.py:278-279`),
so a browser built on one would grey out the moment a training started and you could not read the log
of the thing you were waiting for. **B2** lists all seven kinds, one at a time, in a real sortable
table, and §3.2 requires the one distinction `StorePicker.refresh` throws away: it swallows the
exception (`artifact_picker.py:135-137`) so "there is nothing here" and "I could not look" are the
same picture.

- [ ] **Step 1: Add the two fixtures**

Insert into `tests/_fixtures.py` at line 338, between `build_tiny_run` and `_same_value`:

```python
def build_browse_store(root):
    """One artifact of each of the SEVEN kinds plus the three bad-directory shapes, in a store at
    ``root``. Returns ``{kind: id, ..., "bad": (dir_name, dir_name, dir_name)}``.

    SECONDS, not minutes -- the browser suite must not reach for ``tiny_run``, whose cost is why
    ``screen_run`` is module-scoped (spec §9.1). Every writer is handed ``cfg=None`` and every body is
    the smallest dict its kind's ``manifest.BODY_KEYS`` allows, so nothing here reads a bounds file,
    builds a SimConfig, fits a GMM or trains anything. Nothing ever loads these payloads (there are
    none): the browser only reads manifests. The one real cost is ``ArtifactWriter._commit``'s three
    ``git`` subprocesses per artifact.

    Each artifact is named ``browse_<kind>`` and carries a note, so the name and note columns have
    content. The simulation cache is the exception: it is ALWAYS unnamed
    (``write_simulation_manifest`` writes ``name=""``), so its row also covers ``Summary.label``'s
    ``(unnamed <id>)`` form. It is written at 3 of 4 batches and NOT complete, so its row carries real
    progress and ``finished`` is False while ``complete`` is True -- B3's distinction, on disk. The 4
    is the identity's ``n_runs``, which is where ``Summary.batches_planned`` comes from, so "3/4" on
    the row is read off the manifest and not assembled by the table. It is written with NO ``rows``,
    because that is the only state the real writer can be in mid-run: ``training_checkpoint.save``
    passes none and only ``mark_complete`` records them (P2). A fixture that handed rows to an
    unfinished cache would be a shape no run produces, and would hide every "rows only once it
    finished" branch in both front ends.

    No artifact gets a ``log.txt``: the writer writes one only when a public entry is active
    (``runs.current_run_log()``), and this helper is not one. A test that wants records writes the
    file itself, which is exactly what ``read_log`` reads.

    The three bad directories all sit under ``priors/``, one per shape ``ArtifactStore._entries``
    classifies: no manifest at all, a manifest that will not parse, and a valid manifest of another
    kind (the calibration's own bytes, so the shape is real rather than hand-rolled).
    """
    from core.artifacts import ArtifactStore, write_simulation_manifest
    from core.artifacts.store import MANIFEST
    from core.SBI.training_checkpoint import identity_digest

    store = ArtifactStore(root)
    bodies = {
        "prior": {"gmm": {"n_components": 2, "param_keys": ["k"],
                          "box": {"nd_lows": [0.0], "nd_highs": [1.0], "log_mask": [False]}},
                  "sweep": {}, "stability": {"accepted_sets": None, "iterations": 1}},
        "posterior": {"mode": "chi", "conditioning": {"width": 61, "forcing_dim": 12},
                      "transform": {}, "amortized": True, "truncation": None, "training": {}},
        "observation": {"mode": "spontaneous", "conditioning": {"width": 50, "forcing_dim": 1},
                        "x_obs_digest": "0" * 16, "T_obs_cell": 4.5, "n_obs": None,
                        "forcing_vals": {}, "chi_obs_freqs": None, "source": {"kind": "bench"}},
        "calibration": {"results": {}},
        "inference": {"results": {}},
        "diagnostic": {"diagnostic": "identifiability", "variant": "laplace",
                       "settings": {}, "results": {}},
    }
    ids = {}
    for kind, body in bodies.items():
        with store.create(kind, None, name=f"browse_{kind}", note=f"the {kind} row") as w:
            w.body = dict(body)
        ids[kind] = w.id

    identity = {"format": "training-rows/2", "prior_fingerprint": "f" * 16, "n_runs": 4,
                "truncation": None}
    m = write_simulation_manifest(store.kind_dir("simulation") / identity_digest(identity), identity,
                                  batches_done=3, complete=False)
    ids["simulation"] = m.id

    priors = store.kind_dir("prior")
    bad = ("leftover_no_manifest", "leftover_bad_json", "leftover_wrong_kind")
    for name in bad:
        (priors / name).mkdir(parents=True, exist_ok=True)
    (priors / bad[1] / MANIFEST).write_text("{not json", encoding="utf-8")
    (priors / bad[2] / MANIFEST).write_text(
        (store.path("calibration", ids["calibration"]) / MANIFEST).read_text(encoding="utf-8"),
        encoding="utf-8")
    ids["bad"] = bad
    return ids


def artifact_screen(store):
    """The Artifacts screen wired to ``store`` and refreshed once.

    ``store`` travels through the screen's own ``_store`` / ``_resolved_store()`` seam -- the one
    ``StorePicker`` already uses and the picker tests already monkeypatch
    (tests/test_settings_persistence.py:1070) -- so no process default is swapped and nothing is
    patched. The screen reads the store only when it is shown or refreshed, so the ``refresh()`` here
    is what puts rows in the table.
    """
    from core.gui.screens.artifact_screen import ArtifactScreen
    qt_app()
    screen = ArtifactScreen(store=store)
    screen.refresh()
    return screen
```

- [ ] **Step 2: Write the failing tests for the screen itself**

Create `tests/test_artifact_browser.py`:

```python
"""The artifact browser (piece 4, §3): the Artifacts screen, its listing, its detail pane and its
actions. The eighteenth suite. Run directly: pytest tests/test_artifact_browser.py"""
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")   # must precede any PySide6 import
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest                                                          # noqa: E402,F401

from core.artifacts import ArtifactStore                               # noqa: E402
from core.artifacts.store import KIND_DIRS                             # noqa: E402
from tests._fixtures import artifact_screen, build_browse_store, qt_app  # noqa: E402

KINDS = list(KIND_DIRS)


def _show(screen, kind: str) -> None:
    """Switch the kind selector, which fires the screen's refresh()."""
    screen.kind_combo.setCurrentIndex(KINDS.index(kind))
    assert screen.kind() == kind


def _row_text(table, i: int) -> str:
    """Every cell of one row, joined -- so a test reads what a user reads and does not depend on
    which column a kind happens to put a fact in (core/gui/widgets/artifact_table.py owns that)."""
    item = table.topLevelItem(i)
    return " | ".join(item.text(c) for c in range(table.columnCount()))


def _select(screen, i: int):
    """Select row ``i`` and return its Summary."""
    screen.table.setCurrentItem(screen.table.topLevelItem(i))
    return screen.table.current_summary()


def test_the_browser_lists_the_seven_kinds_with_the_incomplete_directories_last(tmp_path):
    """B1/B2 and §3.2. Seven kinds in KIND_DIRS order, one kind at a time, each artifact's own row
    named by Summary.label -- and for the prior kind, the three leftover directories after the real
    one, each showing why it was not read instead of the kind's own columns.

    The order is asserted off current_summary().complete rather than off the painted rows: what has
    to hold is that ArtifactStore.list's complete-first ordering survives into the table, and no
    arrangement of names can make that accidental.
    """
    ids = build_browse_store(tmp_path)
    store = ArtifactStore(tmp_path)
    screen = artifact_screen(store)

    assert [screen.kind_combo.itemData(i) for i in range(screen.kind_combo.count())] == KINDS

    for kind in KINDS:
        _show(screen, kind)
        assert screen.table.topLevelItemCount() >= 1, f"{kind} listed nothing"
        first = _row_text(screen.table, 0)
        if kind == "simulation":
            # a cache is always unnamed, so its label is "(unnamed <digest>)"
            assert ids["simulation"] in first and "unnamed" in first, first
        else:
            assert f"browse_{kind}" in first, first
            assert f"the {kind} row" in first, f"the note column is empty: {first}"

    _show(screen, "prior")
    assert screen.table.topLevelItemCount() == 4
    assert [_select(screen, i).complete for i in range(4)] == [True, False, False, False]
    tail = "\n".join(_row_text(screen.table, i) for i in (1, 2, 3))
    for dir_name in ids["bad"]:
        assert dir_name in tail, tail
    for reason in ("no manifest.json", "unreadable manifest", "declares kind"):
        assert reason in tail, tail
    assert "1 complete" in screen.status.text() and "3 incomplete" in screen.status.text(), \
        screen.status.text()


def test_a_kind_with_nothing_in_it_is_not_a_root_that_cannot_be_read(tmp_path):
    """§3.2. StorePicker.refresh swallows the exception and lists nothing (artifact_picker.py:135-137),
    so today an empty store and an unreadable one are the same picture. The browser is the one place
    that difference has to be visible: an absent kind directory reads "nothing here yet", and a root
    that raises puts the error, with its class, on the status line."""
    import types

    store = ArtifactStore(tmp_path)                    # nothing written at all
    screen = artifact_screen(store)
    assert screen.table.topLevelItemCount() == 0
    assert screen.status.text() == "Nothing here yet.", screen.status.text()
    assert not (tmp_path / KIND_DIRS["prior"]).exists(), "listing must not create the directory"

    def _boom(kind):
        raise PermissionError("Access is denied")

    screen._store = types.SimpleNamespace(list=_boom)
    screen.refresh()
    assert screen.table.topLevelItemCount() == 0
    assert screen.status.text().startswith("⚠ "), screen.status.text()
    assert "PermissionError" in screen.status.text() and "Access is denied" in screen.status.text()
```

- [ ] **Step 3: Run them and watch them fail**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_artifact_browser.py -v`
Expected: FAIL at collection —
`ModuleNotFoundError: No module named 'core.gui.screens.artifact_screen'`, raised from
`tests/_fixtures.py::artifact_screen`.

- [ ] **Step 4: Write the screen**

Create `core/gui/screens/artifact_screen.py`:

```python
"""The Artifacts screen: read the artifact store from inside the app (piece 4, §3; B1, B2).

A fifth Home tile and a PLAIN QWidget -- deliberately NOT a BasePanel. A BasePanel enrols itself in
``BasePanel._instances`` and ``_set_busy`` disables every instance's controls column while ANY run is
live, so a browser built on one would grey out the moment a training started and you could not read
the log of the thing you were waiting for. Reading is never blocked here; the actions that CHANGE the
store check ``BasePanel._running`` for themselves (§3.4).

The store is reached through ``_resolved_store()`` -- ``store or the process default``, the same seam
``StorePicker`` uses -- so a test hands one in and no process default is swapped.

Remembered (V5, §3.5): the kind last viewed, and the sort column and order. NOT the selected
artifact: a remembered id that has since been deleted is exactly the dangling selection §3.4 removes
from the pickers.
"""
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QComboBox, QHBoxLayout, QLabel, QPushButton, QSplitter, QVBoxLayout,
                               QWidget)

from core.artifacts.store import KIND_DIRS

from .. import settings
from ..widgets.artifact_table import ArtifactTable, columns_for

# The kind selector's visible text, in KIND_DIRS order; each item's userData is the kind key itself,
# which is what the store, the settings key and the table all speak.
KIND_LABELS = {
    "prior": "Priors",
    "simulation": "Simulation caches (training rows)",
    "posterior": "Posteriors",
    "observation": "Observations",
    "calibration": "Calibrations",
    "inference": "Inferences",
    "diagnostic": "Diagnostics",
}


class ArtifactScreen(QWidget):
    """The artifact browser: one kind at a time in a sortable table, over a status line.

    Emits ``store_changed`` after any change it makes to the store; MainWindow connects that to
    ``_refresh_store_pickers`` (B8), because ``StorePicker.restore_key`` silently keeps whatever is
    current when the saved id has vanished -- deliberate for a picker, and a defect the moment
    something can delete.
    """

    store_changed = Signal()

    def __init__(self, store=None, parent=None):
        super().__init__(parent)
        self._store = store
        self._sort = (0, 0)          # (column, Qt.SortOrder value); restored here, applied in refresh

        heading = QLabel("Artifacts")
        heading.setProperty("type", "heading")     # Fluent type ramp (global QSS)

        self.kind_combo = QComboBox()
        for kind in KIND_DIRS:
            self.kind_combo.addItem(KIND_LABELS[kind], userData=kind)

        self.btn_refresh = QPushButton("Refresh")
        self.btn_refresh.clicked.connect(self.refresh)

        kind_row = QHBoxLayout()
        kind_row.addWidget(QLabel("Kind"))
        kind_row.addWidget(self.kind_combo, 1)
        kind_row.addWidget(self.btn_refresh)

        self.table = ArtifactTable()

        # One child today; the detail pane joins it beside the table (§3.3).
        self.split = QSplitter(Qt.Horizontal)
        self.split.setChildrenCollapsible(False)
        self.split.addWidget(self.table)

        self.status = QLabel("")
        self.status.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.addWidget(heading)
        layout.addLayout(kind_row)
        layout.addWidget(self.split, 1)
        layout.addWidget(self.status)

        # Restore BEFORE connecting currentIndexChanged: setCurrentIndex fires it, and a refresh()
        # during __init__ would read the store at launch -- the start-up work §1.2 keeps off that
        # path. Same construction-never-fires rule the Settings screen's radios follow.
        self.restore_settings(settings.settings())
        self.kind_combo.currentIndexChanged.connect(lambda _i: self.refresh())

    # ── the listing (§3.2) ────────────────────────────────────────────────────
    def kind(self) -> str:
        """The kind key currently selected."""
        return str(self.kind_combo.currentData())

    def showEvent(self, event):
        """List on every visit, so an artifact written since the last look is there. NOT from
        __init__: MainWindow builds this screen at launch, and a directory scan per launch is exactly
        the start-up cost that lost the taskbar icon on 2026-09-11 (§1.2)."""
        super().showEvent(event)
        self.refresh()

    def refresh(self) -> None:
        """Re-list the selected kind. Idempotent, and the only place the store is read for the table."""
        kind = self.kind()
        if self.table.topLevelItemCount():
            # The user's own header click, kept across the rebuild -- but only off a table that HAS
            # rows. An empty one still reports DEFAULT_SORT (never anything negative), so an
            # unguarded capture here would overwrite the sort restore_settings just read out of
            # QSettings before _apply_sort below could use it, and "the sort survives a relaunch"
            # would be quietly false.
            self._sort = self.table.sort_state()
        try:
            rows = self._resolved_store().list(kind)
        except Exception as e:                 # noqa: BLE001 -- REPORTED, never swallowed
            # StorePicker.refresh lists nothing on an unreadable root, which makes "there is nothing
            # here" and "I could not look" identical. The browser must tell them apart (§3.2), so the
            # error and its class go on the status line and the table is emptied.
            self.table.set_rows(kind, [])
            self._set_status(f"Could not read the {kind} artifacts: {type(e).__name__}: {e}",
                             error=True)
            return
        self.table.set_rows(kind, rows)
        self._apply_sort(kind)
        if not rows:
            self._set_status("Nothing here yet.")
            return
        bad = sum(1 for s in rows if not s.complete)
        self._set_status(f"{len(rows)} row(s): {len(rows) - bad} complete, {bad} incomplete.")

    def _apply_sort(self, kind: str) -> None:
        """Re-apply the remembered sort after a rebuild, clamped to THIS kind's column count: the
        columns differ per kind (§3.2), so a column remembered while viewing posteriors can be past
        the end of a calibration's."""
        col, order = self._sort
        if 0 <= col < len(columns_for(kind)):
            self.table.apply_sort_state(col, order)

    def _set_status(self, text: str, error: bool = False) -> None:
        """One line, with a ⚠ prefix when it is trouble -- ModelBuilderScreen._set_status's pattern."""
        self.status.setText(("⚠ " if error else "") + text)

    def _resolved_store(self):
        """This screen's store, or the process default -- StorePicker._resolved_store's seam."""
        from core.artifacts import resolve_store
        return resolve_store(self._store)

    # ── persistence (§3.5) ────────────────────────────────────────────────────
    def save_settings(self, qs) -> None:
        """The kind and the sort, and nothing else.

        MainWindow._save_state calls this BY NAME: ``_all_panels()`` is panel-typed and this screen is
        a QWidget (B1), so it gets no sweep for free -- which is also what keeps it out of
        ``_refresh_model_combos``.

        A table with rows in it IS the sort; an empty one reports DEFAULT_SORT whatever was restored,
        so a window closed without ever opening this screen must save what it read rather than the
        default it never showed.
        """
        col, order = self.table.sort_state() if self.table.topLevelItemCount() else self._sort
        qs.beginGroup("artifacts")
        qs.setValue("kind", self.kind())
        qs.setValue("sort_col", str(int(col)))
        qs.setValue("sort_order", str(int(order)))
        qs.endGroup()

    def restore_settings(self, qs) -> None:
        """What save_settings wrote. A kind an older build wrote and this one does not know is
        IGNORED, not restored -- findData returns -1 and the first kind stands."""
        qs.beginGroup("artifacts")
        kind = settings.get_str(qs, "kind", "")
        col = settings.get_int(qs, "sort_col", 0)
        order = settings.get_int(qs, "sort_order", 0)
        qs.endGroup()
        i = self.kind_combo.findData(kind)
        if i >= 0:
            self.kind_combo.setCurrentIndex(i)
        self._sort = (col, order)
```

- [ ] **Step 5: Run the two tests and watch them pass**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_artifact_browser.py -v`
Expected: PASS (2 tests).

- [ ] **Step 6: Write the failing tests for the registration and what it remembers**

Append to `tests/test_artifact_browser.py`:

```python
def test_the_artifacts_tile_opens_the_browser_and_the_window_saves_its_state():
    """B1's five registration points, in one real window.

    The tile is on Home and live; the screen sits AFTER the four sections and BEFORE Settings, so the
    back-arrow slide direction stays monotone with the tile order; the name maps to its stack index;
    and _save_state calls the screen's save_settings BY NAME. It is NOT a BasePanel and NOT in
    _all_panels(), which is what keeps it out of both the app-wide control lock and
    _refresh_model_combos.
    """
    from PySide6.QtWidgets import QPushButton
    from core.gui import settings as st
    from core.gui.main_window import MainWindow
    from core.gui.panels.base_panel import BasePanel
    from core.gui.screens.artifact_screen import ArtifactScreen
    from core.gui.screens.home_screen import SECTIONS

    qt_app()
    assert SECTIONS[-1] == "Artifacts", SECTIONS

    w = MainWindow()
    assert isinstance(w.artifact_screen, ArtifactScreen)
    idx = w._section_index["Artifacts"]
    assert w._section_index["Simulate"] < idx < w._section_index["Settings"]
    assert w.nav.stack.widget(idx) is w.artifact_screen

    assert not isinstance(w.artifact_screen, BasePanel), \
        "a BasePanel enrols in _instances, so every run anywhere would grey the browser out"
    assert w.artifact_screen not in w._all_panels()

    home = w.nav.stack.widget(0)
    btn = next(b for b in home.findChildren(QPushButton) if b.text() == "Artifacts")
    assert btn.property("accent") is True, "a live tile is accent-styled; a dead one is not"
    btn.click()
    assert w.nav.stack.currentIndex() == idx

    w.artifact_screen.kind_combo.setCurrentIndex(KINDS.index("posterior"))
    w._save_state()
    qs = st.settings()
    qs.beginGroup("artifacts")
    got = {k: qs.value(k) for k in qs.childKeys()}
    qs.endGroup()
    assert got.get("kind") == "posterior", got


def test_the_browser_remembers_the_kind_and_the_sort_but_never_the_selection(tmp_path):
    """§3.5 / V5. The kind last viewed and the sort survive a relaunch; the SELECTED artifact
    deliberately does not -- a remembered id that has since been deleted is exactly the dangling
    state §3.4 removes from the pickers. A stale kind an older build left behind is ignored, not
    restored.

    The sort that is saved is deliberately NOT the default one: (1, 1) is DEFAULT_SORT, so a test
    that saved it could not tell a restored sort from a screen that had simply never been sorted at
    all, and the assertion below would hold even if restore_settings did nothing."""
    from core.gui import settings as st

    build_browse_store(tmp_path)
    store = ArtifactStore(tmp_path)
    first = artifact_screen(store)
    _show(first, "posterior")
    assert first.table.sort_state() == (1, 1), \
        "DEFAULT_SORT moved; pick a saved sort below that differs from it"
    first.table.apply_sort_state(0, 0)          # Name, ascending -- not the default
    assert _select(first, 0) is not None

    qs = st.settings()
    first.save_settings(qs)
    qs.sync()
    qs.beginGroup("artifacts")
    assert set(qs.childKeys()) == {"kind", "sort_col", "sort_order"}, sorted(qs.childKeys())
    qs.endGroup()

    again = artifact_screen(store)
    assert again.kind() == "posterior"
    assert again.table.sort_state() == (0, 0), "the saved sort was overwritten before it was applied"
    assert again.table.current_summary() is None, "the selected artifact must not come back"

    qs.setValue("artifacts/kind", "nosuchkind")
    qs.sync()
    third = artifact_screen(store)
    assert third.kind() == KINDS[0], "a stale kind key must be ignored, not restored"


def test_a_store_change_in_the_browser_re_lists_every_artifact_picker():
    """B8. MainWindow connects the screen's store_changed to _refresh_store_pickers, the twin of
    _refresh_model_combos that a saved user model already drives. The pickers are found by TYPE, not
    by naming prior_picker / post_picker / obs_picker, so a fourth one added to a tab is covered by
    construction."""
    from core.gui.main_window import MainWindow
    from core.gui.widgets.artifact_picker import StorePicker

    qt_app()
    w = MainWindow()
    pickers = [p for panel in w._all_panels() for p in panel.findChildren(StorePicker)]
    assert len(pickers) == 3, [p.kind for p in pickers]
    assert {p.kind for p in pickers} == {"prior", "posterior", "observation"}

    seen = []
    for p in pickers:
        p.refresh = lambda _p=p: seen.append(_p.kind)
    w.artifact_screen.store_changed.emit()
    assert sorted(seen) == ["observation", "posterior", "prior"], seen
```

- [ ] **Step 7: Run them and watch them fail**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_artifact_browser.py -v`

Expected: 5 tests, **2 failed, 3 passed**. Of the three just appended:

- `test_the_artifacts_tile_opens_the_browser_and_the_window_saves_its_state` FAILS with
  `AssertionError: ('Reduction Map', 'FDT Analysis', 'Parameter Inference', 'Simulate')` on
  `SECTIONS[-1] == "Artifacts"`;
- `test_a_store_change_in_the_browser_re_lists_every_artifact_picker` FAILS with
  `AttributeError: 'MainWindow' object has no attribute 'artifact_screen'`;
- `test_the_browser_remembers_the_kind_and_the_sort_but_never_the_selection` **passes already** — it
  needs nothing from Steps 8–9, only the screen Step 4 wrote. It is here because it belongs with the
  other two, and because Step 12 must still see it green after the window changes land.

- [ ] **Step 8: Add the fifth tile**

In `core/gui/screens/home_screen.py`, replace lines 11-12:

```python
# The five sections, in display order. All five are live; MainWindow passes them all in
# live_sections. "Artifacts" is the artifact browser (piece 4, B1) -- a peer of the four stage
# sections, and a plain screen rather than a panel.
SECTIONS = ("Reduction Map", "FDT Analysis", "Parameter Inference", "Simulate", "Artifacts")
```

and in the class docstring (line 27) replace `(all four are live today)` with
`(all five are live today)`.

- [ ] **Step 9: Register the screen in MainWindow**

In `core/gui/main_window.py`:

Add above line 19 (`from .screens.home_screen import HomeScreen`):

```python
from .screens.artifact_screen import ArtifactScreen
```

Add after line 24 (`from .screens.settings_screen import SettingsScreen`):

```python
from .widgets.artifact_picker import StorePicker
```

Insert after line 107 (`idx_sim = self.nav.add_screen(self.simulate_screen)`):

```python

        # The artifact browser: AFTER the four sections and BEFORE the Settings screen, so the
        # back-arrow slide direction stays monotone with the Home tile order. A plain QWidget, not a
        # BasePanel (B1) -- a run anywhere must not grey out the log you are reading.
        self.artifact_screen = ArtifactScreen()
        idx_artifacts = self.nav.add_screen(self.artifact_screen)
        self.artifact_screen.store_changed.connect(self._refresh_store_pickers)
```

Replace lines 101-102 with:

```python
        home = HomeScreen(
            live_sections={"Reduction Map", "FDT Analysis", "Parameter Inference", "Simulate",
                           "Artifacts"})
```

Replace lines 131-133 with:

```python
        self._section_index = {"Reduction Map": idx_red, "FDT Analysis": idx_fdt,
                               "Parameter Inference": idx_inf, "Simulate": idx_sim,
                               "Artifacts": idx_artifacts,
                               "Settings": idx_settings, "Model builder": idx_builder}
```

Insert after line 265 (the end of `_refresh_model_combos`, before `def _all_panels`):

```python

    def _refresh_store_pickers(self):
        """Re-list every artifact picker after the browser changed the store (B8) -- the twin of
        _refresh_model_combos, which a saved or deleted user model already drives.

        Found by TYPE rather than by naming the three attributes (prior_picker, post_picker,
        obs_picker), so a fourth picker added to a tab is covered by construction.
        StorePicker.refresh re-reads its kind and re-applies its own current key, so a selection that
        still exists survives and one that was just deleted falls back to whatever is current --
        which is the point: restore_key silently does nothing when the saved id has vanished, and that
        silence becomes a defect the moment a browser can delete.
        """
        for panel in self._all_panels():
            for picker in panel.findChildren(StorePicker):
                picker.refresh()
```

Replace `_save_state`'s body (lines 306-312) with:

```python
    def _save_state(self):
        """Persist window geometry + each panel's selections. Called only when a close is accepted."""
        qs = settings.settings()
        qs.setValue("window/geometry", self.saveGeometry())
        for panel in self._all_panels():
            panel.save_settings(qs)
        # BY NAME (B1): the browser is a QWidget, not a BasePanel, so it is not in _all_panels() and
        # gets no sweep for free -- which is also what keeps it out of _refresh_model_combos.
        self.artifact_screen.save_settings(qs)
        qs.sync()
```

- [ ] **Step 10: Correct the module docstring's stale count (§8.2)**

In `core/gui/main_window.py`, replace the module docstring's first sentence (lines 1-3) so it says
what is now there:

```python
"""The PRISM main window: a NavShell (persistent "PRISM" title + back arrow) over a home/splash screen,
four section screens, the Artifacts browser, and two Settings-reached screens (the Settings/Help screen
and the user-defined model builder). Replaces the old flat four-tab layout; the section panels are
reused unchanged in behaviour -- only where they are mounted changes. Cross-validation now lives inside
the FDT Analysis section, and the SBI panel is split into the Parameter Inference section's gated
tabs."""
```

Leave `MainWindow`'s CLASS docstring (lines 70-75) exactly as it is, stale count and all: Task 25
owns that whole docstring — the "five section screens" line and the "the only QSettings WRITE site"
sentence both change there — and it runs last and rewrites it once. Two tasks editing one docstring
is how a merge of the two loses one of them.

- [ ] **Step 11: Re-point the ONE docstring that cites a line number in main_window.py**

This task inserts lines above `main_window.py:231`. Find the line's new number:

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -c "print([i for i, l in enumerate(open('core/gui/main_window.py', encoding='utf-8'), 1) if 'box.exec() != QMessageBox.Yes' in l])"`

Then replace `main_window.py:231` with `main_window.py:<that number>` in
`tests/test_worker_dispatch.py:338` — docstring prose, so nothing else changes.

**Do not touch `tests/conftest.py`.** Its `_no_modal_dialogs` docstring cites the same line at `:174`,
and Task 11 rewrites that sentence for its own reasons and removes the number entirely rather than
re-pointing it (Q10). Two tasks editing one sentence is how the second one's quoted "before" stops
matching.

- [ ] **Step 12: Run the task's own tests**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_artifact_browser.py tests/test_nav_and_gating.py tests/test_settings_persistence.py tests/test_worker_dispatch.py -q`
Expected: PASS

- [ ] **Step 13: Commit**

```bash
git add core/gui/screens/artifact_screen.py core/gui/screens/home_screen.py core/gui/main_window.py tests/test_artifact_browser.py tests/_fixtures.py tests/test_worker_dispatch.py
git commit -m "gui: an Artifacts screen that lists the store's seven kinds"
```

---

### Task 10: the detail pane and Save

**Files:**
- Modify: `core/gui/screens/artifact_screen.py` (Task 9's file: the imports, the end of `__init__`,
  the end of `refresh`, and three new methods)
- Test: `tests/test_artifact_browser.py` (five tests appended to Task 9's suite)

**Interfaces:**
- Consumes, from Task 6 (`core/artifacts/report.py`, re-exported from `core/artifacts/__init__.py`):
  `render_manifest(m) -> str`.
- Consumes, from Task 4 (`core/artifacts/store.py`):
  `ArtifactStore.read_log(self, kind: str, ref: str, *, max_bytes: "int | None" = None) -> "tuple[str | None, bool]"`,
  returning `(text, truncated)`; `(None, False)` when there is no file at all and `("", False)` for a
  run that said nothing — two different answers, deliberately, and the pane states both.
- Consumes, from Task 8: `ArtifactTable.selection_changed = Signal()` and `current_summary(self)`.
- Consumes, from the store task that grew `Summary`: `Summary.dir_name`, beside
  `Manifest.dir_name` (`core/artifacts/manifest.py:72-77`).
- Consumes, from Task 9: `ArtifactScreen.split` (a horizontal `QSplitter` holding the table alone),
  `ArtifactScreen.refresh`, `ArtifactScreen._resolved_store`, `ArtifactScreen._set_status`.
- Produces: `ArtifactScreen.detail` (a read-only `QPlainTextEdit`),
  `ArtifactScreen.detail_actions` (the `QHBoxLayout` under the pane, right-aligned, that holds the
  pane's buttons — Task 12 mounts its "Lineage report…" button into it, so the name is part of this
  task's interface and not a local), `ArtifactScreen.btn_save` (the one button in it today),
  `ArtifactScreen._detail_text(self, s) -> str`, `ArtifactScreen._on_selection_changed(self) -> None`,
  `ArtifactScreen._save_shown(self) -> None`, and the module constants
  `LOG_MAX_BYTES = 1 << 20`, `_CACHE_NO_LOG`, `_NO_RUN_LOG`, `_EMPTY_LOG`, `_RECORDS_HEADER` plus
  `_stale_folder_note(actual, expected) -> str`. The actions tasks reach the same pane through
  `_on_selection_changed`.

**Why this task exists:** Since piece 3 every committed artifact carries a `log.txt` of its run's
records and **nothing reads it** (spec §1). **B4** makes the detail view text only — the manifest
rendered plus those records, no figures, no open-the-folder button, no "use this" jump — and §3.3
names the states a blank pane would hide: the simulation cache never gets a log at all (it has no
writer), an artifact written outside any run gets none either, a run that said nothing wrote an EMPTY
one — piece 3's invariant is that silence is a record too, so a blank pane under a Records heading
would read as a bug — a tail must say it is a tail, and a folder whose name disagrees with its own
manifest — what `ArtifactStore.rename` leaves behind when `sub.rename(target)` keeps raising
`PermissionError` and is tolerated because the manifest is what resolves an artifact
(`core/artifacts/store.py:430-441`) — has never been visible anywhere. That last sentence is
suppressed for the simulation kind, where a folder name that is not the id is legitimate rather than
stale (`Manifest.dir_name` is the bare digest for a cache, `core/artifacts/manifest.py:73-77`, and
`write_simulation_manifest` writes wherever it is handed, `store.py:778-815`). **B10** makes Save
write what is on screen to a file, never an eighth store kind.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_artifact_browser.py`:

```python
def test_the_detail_pane_shows_the_manifest_and_the_run_records(tmp_path):
    """B4 / §3.3. The manifest as Task 6 renders it, then the run's records verbatim with their
    HH:MM:SS level stamps -- one string, which is also exactly what Save writes (B10)."""
    from core.artifacts import render_manifest
    from core.artifacts.store import LOG_FILE

    ids = build_browse_store(tmp_path)
    store = ArtifactStore(tmp_path)
    # A run's records, where ArtifactWriter._commit would have written them.
    (store.path("calibration", ids["calibration"]) / LOG_FILE).write_text(
        "12:00:00 info building the calibration\n12:00:04 warning 1 dataset diverged\n",
        encoding="utf-8")

    screen = artifact_screen(store)
    _show(screen, "calibration")
    s = _select(screen, 0)
    text = screen.detail.toPlainText()
    assert render_manifest(store.get("calibration", s.id)) in text, text[:400]
    assert "12:00:04 warning 1 dataset diverged" in text, "the records are verbatim, stamps and all"
    assert screen.detail.isReadOnly()
    assert screen.btn_save.isEnabled()


def test_the_detail_pane_states_the_gaps_a_blank_pane_would_hide(tmp_path, monkeypatch):
    """§3.3's four sentences about the records, plus the stale-folder note. A training cache keeps no
    log BY DESIGN, an artifact written outside any run has none either, a tail announces itself, a run
    that said nothing says so -- read_log answers "" there and None for no file at all (§2.2), and
    piece 3's invariant is that silence is a record too -- and a folder whose name disagrees with its
    own manifest says so, the state ArtifactStore.rename leaves when the directory move is refused
    (store.py:430-441), which nothing has ever shown."""
    from core.artifacts.store import LOG_FILE
    from core.gui.screens import artifact_screen as mod

    ids = build_browse_store(tmp_path)
    store = ArtifactStore(tmp_path)
    screen = artifact_screen(store)

    # (a) the cache: no log, for its own reason
    _show(screen, "simulation")
    _select(screen, 0)
    assert "a training cache keeps no log" in screen.detail.toPlainText()

    # (b) an artifact whose run never existed
    _show(screen, "inference")
    _select(screen, 0)
    assert "written outside a run" in screen.detail.toPlainText()

    # (c) a tail says it is a tail. 1 MiB is the real ceiling; the test moves it rather than write one.
    assert mod.LOG_MAX_BYTES == 1 << 20
    monkeypatch.setattr(mod, "LOG_MAX_BYTES", 64)
    (store.path("inference", ids["inference"]) / LOG_FILE).write_text("x" * 500, encoding="utf-8")
    screen.refresh()
    _select(screen, 0)
    assert "only the last 64 bytes" in screen.detail.toPlainText()

    # (d) a run that said nothing: read_log answers "" here, not None, and the pane must not be blank
    #     under the Records heading. _show switches the kind, which re-lists.
    (store.path("diagnostic", ids["diagnostic"]) / LOG_FILE).write_text("", encoding="utf-8")
    _show(screen, "diagnostic")
    _select(screen, 0)
    assert store.read_log("diagnostic", ids["diagnostic"]) == ("", False)
    assert "the run recorded nothing" in screen.detail.toPlainText()

    # (e) the stale folder: move the directory and leave the manifest naming the old one
    sub = store.path("prior", ids["prior"])
    sub.rename(sub.with_name("was_moved__" + ids["prior"]))
    _show(screen, "prior")
    s = _select(screen, 0)
    assert s.complete and s.dir_name == "was_moved__" + ids["prior"]
    text = screen.detail.toPlainText()
    assert "was_moved__" in text and "browse_prior__" in text, text[:400]
    assert "manifest is what resolves" in text, text[:400]


def test_a_caches_folder_name_is_never_called_stale(tmp_path):
    """§3.3, the one exception to the stale-folder note: it is SUPPRESSED for the simulation kind.

    Manifest.dir_name is the bare digest for a cache (manifest.py:73-77) and
    write_simulation_manifest writes wherever it is handed (store.py:778-815), so a cache directory
    whose name is not its id is legitimate -- the note would fire on any hand-placed cache and tell
    the operator that a rename half-failed when nothing of the sort happened. The cache's own
    sentence about its missing log is still there, so this is a suppression and not a blank pane.
    """
    from core.artifacts import write_simulation_manifest
    from core.SBI.training_checkpoint import identity_digest

    store = ArtifactStore(tmp_path)
    identity = {"format": "training-rows/2", "prior_fingerprint": "a" * 16, "n_runs": 2,
                "truncation": None}
    m = write_simulation_manifest(store.kind_dir("simulation") / "hand_placed_cache", identity,
                                  batches_done=1, complete=False)
    assert m.dir_name == identity_digest(identity) != "hand_placed_cache"

    screen = artifact_screen(store)
    _show(screen, "simulation")
    s = _select(screen, 0)
    assert s.dir_name == "hand_placed_cache" and s.id == m.id
    text = screen.detail.toPlainText()
    assert "manifest is what resolves" not in text, text[:400]
    assert "only the folder name is out of date" not in text, text[:400]
    assert "a training cache keeps no log" in text, text[:400]


def test_an_incomplete_directorys_pane_shows_its_reason_its_path_and_its_folder(tmp_path):
    """§3.3's last line. There is no manifest to render, so what there is to say is why it was not
    read, where it is and what it is called."""
    ids = build_browse_store(tmp_path)
    store = ArtifactStore(tmp_path)
    screen = artifact_screen(store)          # opens on priors, where the three leftovers are

    seen = {}
    for i in range(screen.table.topLevelItemCount()):
        s = _select(screen, i)
        if not s.complete:
            seen[s.dir_name] = screen.detail.toPlainText()
    assert set(seen) == set(ids["bad"]), sorted(seen)
    for dir_name, text in seen.items():
        assert dir_name in text
        assert str(store.kind_dir("prior") / dir_name) in text
        assert any(r in text for r in ("no manifest.json", "unreadable manifest", "declares kind")), \
            text


def test_save_writes_exactly_what_the_pane_shows(tmp_path, monkeypatch):
    """B10: a FILE, chosen through QFileDialog the way simulate_panel._save_video chooses one, holding
    byte-for-byte what is on screen. Cancelling writes nothing and says nothing."""
    from PySide6.QtWidgets import QFileDialog

    build_browse_store(tmp_path)
    store = ArtifactStore(tmp_path)
    screen = artifact_screen(store)
    _select(screen, 0)
    shown = screen.detail.toPlainText()
    assert shown

    out = tmp_path / "report"                      # no suffix: Save adds .txt
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(out), ""))
    screen.btn_save.click()
    assert out.with_suffix(".txt").read_text(encoding="utf-8") == shown
    assert "report.txt" in screen.status.text(), screen.status.text()

    calls = []
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        lambda *a, **k: calls.append(1) or ("", ""))
    screen._set_status("")
    screen.btn_save.click()
    assert calls == [1] and screen.status.text() == ""
    assert sorted(p.name for p in tmp_path.glob("report*")) == ["report.txt"]
```

- [ ] **Step 2: Run them and watch them fail**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_artifact_browser.py -k "detail or incomplete or save or stale" -v`
Expected: 5 failed. Four with `AttributeError: 'ArtifactScreen' object has no attribute 'detail'`,
and the Save one with `... has no attribute 'btn_save'`.

- [ ] **Step 3: Add the imports and the module-level text**

In `core/gui/screens/artifact_screen.py`, extend the import block:

```python
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import (QComboBox, QFileDialog, QHBoxLayout, QLabel, QPlainTextEdit,
                               QPushButton, QSplitter, QVBoxLayout, QWidget)

from core.artifacts import render_manifest
from core.artifacts.store import KIND_DIRS
```

and add below `KIND_LABELS`:

```python
# The tail of a run's records the pane shows (§3.3). A ceiling, not a budget: a training run's log is
# unbounded and the pane is a text box.
LOG_MAX_BYTES = 1 << 20

# read_log answers None (no file at all) and "" (a run that said nothing) differently, deliberately
# (§2.2), and the pane states BOTH: piece 3's invariant is that silence is a record too, and a blank
# pane under a Records heading reads as a bug rather than as an answer. None has two causes, and the
# cache's is its own -- it has no writer, so no log.txt is ever written beside it.
_CACHE_NO_LOG = ("a training cache keeps no log: it is written batch by batch across resumes and "
                 "shared by every posterior that names it")
_NO_RUN_LOG = "written outside a run"
_EMPTY_LOG = "the run recorded nothing"
_RECORDS_HEADER = "── the run's records (log.txt) ──"


def _stale_folder_note(actual: str, expected: str) -> str:
    """Said when Summary.dir_name disagrees with the manifest's own dir_name (§3.3).

    ArtifactStore.rename writes the manifest FIRST and moves the directory SECOND, and tolerates a
    PermissionError on the move (store.py:430-441) because the manifest is what resolves an artifact.
    Nothing is lost when that happens and nothing has ever said it happened.

    Never said for the simulation kind -- ``_detail_text`` guards the call, and the reason is there.
    """
    return (f"This artifact's folder is called {actual!r}, but its own manifest says {expected!r}. A "
            f"rename writes the manifest first and moves the directory second, and the move can be "
            f"refused (a handle held open on Windows). The manifest is what resolves an artifact, so "
            f"nothing is lost -- only the folder name is out of date.")
```

- [ ] **Step 4: Build the pane and the Save button**

In `ArtifactScreen.__init__`, replace the two lines

```python
        self.split.addWidget(self.table)
```

with:

```python
        self.split.addWidget(self.table)

        # Read-only, text only (B4): the manifest rendered, then the run's records. A fixed-pitch
        # face, because the manifest is rendered in aligned columns and the records carry HH:MM:SS
        # stamps -- both ragged out in a proportional font. No wrapping, for the same reason.
        self.detail = QPlainTextEdit()
        self.detail.setReadOnly(True)
        self.detail.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.detail.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))

        self.btn_save = QPushButton("Save…")
        self.btn_save.setToolTip("Write what is shown here to a text file")
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self._save_shown)

        # The pane's button row, kept as an attribute rather than a local: Task 12's "Lineage
        # report…" button mounts into THIS layout, beside Save, because both write a file about the
        # selected artifact.
        self.detail_actions = QHBoxLayout()
        self.detail_actions.addStretch(1)
        self.detail_actions.addWidget(self.btn_save)

        detail_side = QWidget()
        detail_layout = QVBoxLayout(detail_side)
        detail_layout.setContentsMargins(0, 0, 0, 0)
        detail_layout.addWidget(self.detail, 1)
        detail_layout.addLayout(self.detail_actions)
        self.split.addWidget(detail_side)
        self.split.setStretchFactor(1, 1)
        self.table.selection_changed.connect(self._on_selection_changed)
```

- [ ] **Step 5: Clear the pane with the table**

At the end of `ArtifactScreen.refresh`, after both `_set_status` calls can no longer be reached,
the selection must be re-read explicitly: `set_rows` leaves nothing selected, and a table that was
already empty emits no change. Add `self._on_selection_changed()` as the last statement of the
`try`-free tail and of the unreadable-root early return, so `refresh` is:

```python
    def refresh(self) -> None:
        """Re-list the selected kind. Idempotent, and the only place the store is read for the table."""
        kind = self.kind()
        if self.table.topLevelItemCount():
            # The user's own header click, kept across the rebuild -- but only off a table that HAS
            # rows. An empty one still reports DEFAULT_SORT (never anything negative), so an
            # unguarded capture here would overwrite the sort restore_settings just read out of
            # QSettings before _apply_sort below could use it, and "the sort survives a relaunch"
            # would be quietly false.
            self._sort = self.table.sort_state()
        try:
            rows = self._resolved_store().list(kind)
        except Exception as e:                 # noqa: BLE001 -- REPORTED, never swallowed
            # StorePicker.refresh lists nothing on an unreadable root, which makes "there is nothing
            # here" and "I could not look" identical. The browser must tell them apart (§3.2), so the
            # error and its class go on the status line and the table is emptied.
            self.table.set_rows(kind, [])
            self._on_selection_changed()
            self._set_status(f"Could not read the {kind} artifacts: {type(e).__name__}: {e}",
                             error=True)
            return
        self.table.set_rows(kind, rows)
        self._apply_sort(kind)
        # Explicitly, not off the signal: set_rows leaves nothing selected, and a table that was
        # already empty emits no change -- so a stale pane would outlive the rows it described.
        self._on_selection_changed()
        if not rows:
            self._set_status("Nothing here yet.")
            return
        bad = sum(1 for s in rows if not s.complete)
        self._set_status(f"{len(rows)} row(s): {len(rows) - bad} complete, {bad} incomplete.")
```

- [ ] **Step 6: Write the pane's text**

Add to `ArtifactScreen`, after `_apply_sort`:

```python
    # ── the detail view (§3.3, B4) ────────────────────────────────────────────
    def _on_selection_changed(self) -> None:
        """Re-render the pane for whatever is selected now (nothing -> an empty pane)."""
        s = self.table.current_summary()
        try:
            self.detail.setPlainText(self._detail_text(s))
        except Exception as e:                 # noqa: BLE001 -- reported, never swallowed
            # The row was listed and the artifact has gone since, or its manifest stopped parsing
            # (another process, a half-finished copy). Say which rather than show a blank pane.
            self.detail.setPlainText("")
            self._set_status(f"Could not read the selected artifact: {type(e).__name__}: {e}",
                             error=True)
        self.btn_save.setEnabled(bool(self.detail.toPlainText()))

    def _detail_text(self, s) -> str:
        """Everything the pane shows for one row, as one string -- which is what Save writes (B10).

        Text only (B4): the manifest rendered by the ONE renderer both front ends use
        (core/artifacts/report.py), the stale-folder note when there is one, then the run's records
        verbatim with their HH:MM:SS level stamps -- or, when there are none, WHICH kind of none it
        is (§3.3). Two store reads and no other side effect.
        """
        if s is None:
            return ""
        if not s.complete:
            # Nothing can load this directory, so there is no manifest to render: what there is to
            # say is why it was not read, where it is, and what it is called.
            return "\n".join((f"{s.kind} — an incomplete directory, which nothing can load.",
                              f"folder:  {s.dir_name}",
                              f"path:    {s.path}",
                              f"reason:  {s.reason}"))
        store = self._resolved_store()
        m = store.get(s.kind, s.id)
        parts = [render_manifest(m)]
        # NOT for the simulation kind: Manifest.dir_name is the bare digest for a cache
        # (manifest.py:73-77) and write_simulation_manifest writes into whatever directory it is
        # handed, so a cache folder named anything else is legitimate -- the note would fire on every
        # hand-placed cache and report a half-failed rename that never happened.
        if s.kind != "simulation" and s.dir_name != m.dir_name:
            parts += ["", _stale_folder_note(s.dir_name, m.dir_name)]
        text, truncated = store.read_log(s.kind, s.id, max_bytes=LOG_MAX_BYTES)
        parts += ["", _RECORDS_HEADER]
        if text is None:
            parts.append(_CACHE_NO_LOG if s.kind == "simulation" else _NO_RUN_LOG)
        elif truncated:
            parts.append(f"(only the last {LOG_MAX_BYTES} bytes are shown; the file is longer)")
            parts.append(text)
        elif not text:
            # read_log says "" for a run that said nothing and None for no file at all (§2.2).
            # Silence is a record too, so the pane says so rather than leave the heading bare.
            parts.append(_EMPTY_LOG)
        else:
            parts.append(text)
        return "\n".join(parts)
```

- [ ] **Step 7: Write Save**

Add to `ArtifactScreen`, after `_detail_text`:

```python
    def _save_shown(self) -> None:
        """B10's first half: what is on screen, to a file the operator picks.

        A FILE, never an eighth store kind -- a report DESCRIBES the store and must not be mistaken
        for something the store holds. The dialog is QFileDialog.getSaveFileName, the same call
        simulate_panel._save_video and figure_window use; cancelling returns "" and writes nothing.
        """
        text = self.detail.toPlainText()
        if not text:
            return                     # the button is disabled with nothing selected; belt and braces
        s = self.table.current_summary()
        suggested = f"{s.kind}_{(s.name or s.id) if s.complete else s.dir_name}.txt"
        path, _ = QFileDialog.getSaveFileName(self, "Save what is shown", suggested,
                                              "Text file (*.txt)")
        if not path:
            return
        if not path.lower().endswith(".txt"):
            path += ".txt"
        try:
            Path(path).write_text(text, encoding="utf-8")
        except OSError as e:
            self._set_status(f"Could not write {path}: {e}", error=True)
            return
        self._set_status(f"Saved what is shown to {Path(path).name}.")
```

- [ ] **Step 8: Record the pane in the module docstring**

Append one paragraph to `core/gui/screens/artifact_screen.py`'s module docstring, before the closing
quotes:

```
The detail pane is TEXT ONLY (B4): no figure rendering, no open-the-folder button, no "use this" jump
into a stage tab. All three were considered and declined on 2026-09-17 (§1.3) and are additive on top
of ``_detail_text`` if they are ever wanted.
```

- [ ] **Step 9: Run the task's own tests**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_artifact_browser.py -q`
Expected: PASS

- [ ] **Step 10: Commit**

```bash
git add core/gui/screens/artifact_screen.py tests/test_artifact_browser.py
git commit -m "gui: the Artifacts screen's detail pane and Save"
```

### Task 11: the browser's actions — note, delete, sweep, and the pickers refreshed

**Files:**
- Modify: `core/gui/screens/artifact_screen.py` — created by Task 9 (the screen, registered, the
  listing) and extended by Task 10 (the detail pane). This task adds an actions row to `__init__`
  and six new methods plus two module-level helpers. **It has no line numbers yet**: Task 9 writes
  the file, so every anchor below is given as the code to search for, not as a number.
- Modify: `core/gui/main_window.py` — Steps 20-22 only: the conversion of the window's last three
  STATIC `QMessageBox` calls (`QMessageBox.warning` at **215** and **236**,
  `QMessageBox.information` at **221**, in the file as it stands before Task 9 moves them) into
  instance dialogs behind one new helper. **`_refresh_store_pickers` and the
  `artifact_screen.store_changed` connect are Task 9's, not this task's** (Q1): this task calls the
  method from its test and writes neither.
- Modify: `tests/conftest.py` — two sentences of `_no_modal_dialogs`' docstring (**174** and
  **178-179**), which record the three statics as outside the guard and unreached and cite a line
  number this task's own helper moves. This task **removes** that number rather than re-pointing it;
  Task 9 deliberately leaves the sentence alone (Q10).
- Test: `tests/test_artifact_browser.py` — created by Task 9; this task appends to it.

**Interfaces:**

- Consumes:
  - `core.refusals.require_note(key: str, text: str) -> str` and `core.refusals.NOTE_MAX_CHARS = 200`
    (Task 2), plus the two field keys Task 2 registered: `Field("artifact", "the artifact", None)`
    and `Field("note", "the note", None)`, whose GUI sentences are
    `"artifact": "Select an artifact in the list on the Artifacts screen."` and
    `"note": "Edit it in the Note box on the Artifacts screen."` in `core/gui/fields.py::CONTROL`.
  - `ArtifactStore.remove_incomplete(kind, dir_name) -> Path` and
    `ArtifactStore.sweep_incomplete(kind=None) -> (removed, failed)` where
    `removed == [(kind, dir_name)]` and `failed == [(kind, dir_name, reason)]` (Task 5). Only
    `sweep_incomplete` is called from here.
  - `Summary`'s new fields (Task 3): `dir_name: str`, `finished: bool`, `batches_done: int | None`,
    `rows: tuple[int, ...] | None`, `variant: str | None`, beside the existing `kind, id, name,
    created, note, path, complete, reason, mode, width, amortized, parents`.
  - `core.gui.widgets.refusal_box.show_refusal(parent, exc) -> None` (Task 7) — the same yellow
    "Check your inputs" box `BasePanel._refusal` shows (`core/gui/panels/base_panel.py:384-407`):
    title "Check your inputs", `setText(exc.message)`,
    `setInformativeText(gui_fields.fix_sentence(exc.field))`, one Ok button, `.exec()`.
  - From Task 9's `ArtifactScreen`: `store_changed = Signal()`, `refresh(self)`, `save_settings(qs)`,
    `restore_settings(qs)` (all in the contract), plus these four names, which Task 9 owns and this
    task uses exactly as Task 9 spells them: `self.table` (the `ArtifactTable`), `self.kind_combo`
    (the kind selector, each item's `userData` being the kind key), `self.status` +
    `self._set_status(text, error=False)` (`ModelBuilderScreen`'s pattern,
    `core/gui/screens/model_builder_screen.py:503-504`), and `self._resolved_store()` (the same seam
    `StorePicker` has at `core/gui/widgets/artifact_picker.py:125-127`, which
    `tests/_fixtures.py::artifact_screen` monkeypatches). Task 9's outer layout is a `QVBoxLayout`
    holding the heading, the kind row, the splitter and — last — the status label, so the actions row
    goes in with `self.layout().insertWidget(self.layout().count() - 1, ...)`: directly above the
    status line.
  - From Task 8's `ArtifactTable` (a `QTreeWidget`): `selection_changed = Signal()`,
    `set_rows(kind, rows)`, `current_summary(self) -> "Summary | None"`. The tests also use plain
    `QTreeWidget` API (`topLevelItemCount`, `topLevelItem`, `setCurrentItem`), which needs nothing
    new.
  - From Task 9's `core/gui/main_window.py`: `MainWindow._refresh_store_pickers(self)` and the
    `self.artifact_screen.store_changed.connect(self._refresh_store_pickers)` line beside the
    screen's construction. Task 9 writes both; this task only calls the method from its test (Q1).
  - From Task 9's `tests/_fixtures.py`: `artifact_screen(store)`. `build_browse_store(root)` is
    deliberately **not** used here: this task's assertions need exact parents, exact fingerprints and
    exactly one leftover directory, so each test writes what it asserts on through the store suite's
    own helpers (`_nad_cfg`, `_prior_artifact`, `store.create`, `write_simulation_manifest`).
- Produces:
  - `ArtifactScreen.note_edit` (`QLineEdit`), `ArtifactScreen.btn_note`, `.btn_delete`,
    `.btn_sweep`, `.btn_sweep_all` (`QPushButton`s).
  - `ArtifactScreen._set_note(self)`, `._delete(self)`, `._sweep(self, *, all_kinds: bool)`,
    `._sync_actions(self)`, `._after_change(self)`, `._refuse_while_running(self, doing: str) -> bool`,
    `._selected(self, doing: str)`, `._dependents_refusal(self, store, s, deps) -> Refusal`,
    `._incomplete(self, store, kind) -> tuple`, and module-level `_delete_prompt(s) -> tuple`,
    `_FINGERPRINT_DEPENDENT` (str).
  - `MainWindow._tell(self, title: str, text: str, icon) -> None` — the one place the window shows a
    plain one-button box, replacing the three statics (Step 22).
  - In the test file: `_show_kind(scr, kind)`, `_select_ref(table, ident)` and
    `_answer(monkeypatch, button)`, which Task 12 reuses.

**Why this task exists:** `set_note`, `dependents` and `delete` have no caller anywhere but the
suites (spec §1), and nothing in the app can remove a directory that never got a manifest. **B5**
makes a note one trimmed line of at most 200 characters, refused rather than clamped; **B6** makes
delete one artifact at a time with *no* `force` in either front end, which is why the browser must
read `dependents` *first* — `store.delete` (`core/artifacts/store.py:478-488`) refuses anything with
dependents, so a confirmation for such an artifact could only ever be followed by a failure; **B7**
adds the sweep; **B8** refreshes the three `StorePicker`s afterwards, because
`StorePicker.restore_key` (`core/gui/widgets/artifact_picker.py:165-170`) silently does nothing when
the saved id has vanished, leaving whatever item happens to be current selected — deliberate for a
picker, a dangling selection the moment a browser can delete.

And since this task is editing `core/gui/main_window.py` anyway, Steps 20-22 pay off the one debt in
that file the global constraint forbids in new code: `QMessageBox.warning` at `:215` and `:236` and
`QMessageBox.information` at `:221` are C++ **statics**, which escape the session guard
`tests/conftest.py::_no_modal_dialogs` installs (it patches the instance method only). Offscreen a
static spins a nested event loop nothing ever closes, so a test that reached one would STALL past
the ten-minute tool-call limit instead of failing — which is exactly why `_no_modal_dialogs`' own
docstring records those three sites as unreached by any test. Converted, they are ordinary instance
dialogs that land in `SHOWN` like every other box, and the comment saying otherwise goes with them.

- [ ] **Step 1: Write the failing note tests**

Append to `tests/test_artifact_browser.py`. Task 9 put the module docstring and its own imports at
the top; add these three helpers directly under them (Task 12 reuses them), then the two tests.

The file ends up with **two** selection helpers, deliberately and under different names: Task 9's
`_select(screen, i)` takes a row INDEX, because its tests are about the order `ArtifactStore.list`
hands back and what a given row shows; `_select_ref(table, ident)` below takes an artifact's id or an
incomplete directory's name and hunts for its row, because these tests act on one known artifact and
must never assume where the sort put it. Two module-level helpers of the same name in one file would
mean the later `def` silently killed the other's tests, so do not rename either into the other (Q4).

```python
def _show_kind(scr, kind):
    """Point the screen at one kind and re-list it. The selector carries the kind key as its item
    data (Task 9), so a test names a kind rather than an index."""
    i = scr.kind_combo.findData(kind)
    assert i >= 0, f"the kind selector does not offer {kind!r}"
    scr.kind_combo.setCurrentIndex(i)
    scr.refresh()


def _select_ref(table, ident):
    """Make the row for ``ident`` (an artifact id or an incomplete directory's name) current, and
    return its Summary. Plain QTreeWidget API plus the table's own ``current_summary`` -- the rows
    are sorted (complete first, newest first), so a test must never assume an index."""
    for i in range(table.topLevelItemCount()):
        table.setCurrentItem(table.topLevelItem(i))
        s = table.current_summary()
        if s is not None and ident in (s.id, s.dir_name):
            return s
    raise AssertionError(f"no row for {ident!r} among {table.topLevelItemCount()} row(s)")


def _answer(monkeypatch, button):
    """Layer a chosen answer over tests/conftest.py::_no_modal_dialogs, the pattern at
    tests/test_nav_and_gating.py:1179-1186. The session guard records every box and returns 0, which
    every confirmation here reads as No; these dialogs use STANDARD buttons, so the answer is the
    returned enum rather than a click on an added button."""
    from PySide6.QtWidgets import QMessageBox
    from tests._fixtures import SHOWN
    monkeypatch.setattr(QMessageBox, "exec", lambda self: SHOWN.append(self) or button)


def test_setting_a_note_trims_it_and_asks_nothing(store):
    """B5: the rule is core's (require_note) and the box is the front end's. One trimmed line,
    written through set_note; a blank clears it; nothing is asked."""
    from PySide6.QtWidgets import QLabel
    from tests._fixtures import SHOWN, artifact_screen, qt_app
    qt_app()
    cfg = _nad_cfg()
    p = _prior_artifact(store, cfg, name="annotated")
    scr = artifact_screen(store)
    _show_kind(scr, "prior")
    _select_ref(scr.table, p.id)
    # The control core/gui/fields.py names ("Edit it in the Note box on the Artifacts screen.") has
    # to exist, spelled that way: the table is the ONE place a control is named, and a sentence
    # pointing at a box nobody can find is the failure mode V3 exists to stop.
    assert any(lbl.text() == "Note" for lbl in scr.findChildren(QLabel)), "no 'Note' label"
    scr.note_edit.setText("   spontaneous, 4.5 s   ")
    scr._set_note()
    assert store.get("prior", p.id).note == "spontaneous, 4.5 s"
    assert SHOWN == [], "setting a note must not ask anything"
    assert "Set the note" in scr.status.text(), scr.status.text()
    # Re-select: _after_change re-lists the kind, and the screen deliberately remembers no selection
    # (spec §3.5 -- a remembered id that has since been deleted is the dangling state B8 prevents).
    _select_ref(scr.table, p.id)
    scr.note_edit.setText("")
    scr._set_note()
    assert store.get("prior", p.id).note == ""
    assert "Cleared the note" in scr.status.text(), scr.status.text()


def test_an_over_long_note_is_refused_with_its_fix_sentence(store):
    """V2: refused, never clamped. The box carries core's neutral sentence (both numbers) and the
    front end's own "where to fix it" from core/gui/fields.py, and the manifest is untouched."""
    from PySide6.QtWidgets import QMessageBox
    from tests._fixtures import SHOWN, artifact_screen, qt_app
    qt_app()
    cfg = _nad_cfg()
    p = _prior_artifact(store, cfg, name="annotated")
    scr = artifact_screen(store)
    _show_kind(scr, "prior")
    _select_ref(scr.table, p.id)
    scr.note_edit.setText("x" * 201)
    # No setMaxLength on the box, deliberately: it would truncate at 200 and the refusal could never
    # fire, which is exactly the silent clamp B5 forbids.
    assert len(scr.note_edit.text()) == 201, "the Note box must not clamp what was typed"
    scr._set_note()
    box = SHOWN[-1]
    assert box.windowTitle() == "Check your inputs" and box.icon() == QMessageBox.Warning
    assert "200" in box.text() and "201" in box.text(), box.text()
    assert box.informativeText() == "Edit it in the Note box on the Artifacts screen."
    assert store.get("prior", p.id).note == ""
```

Task 9's imports already include `pytest`, `_nad_cfg` and `_prior_artifact` from `tests._fixtures`;
if it did not import the last two, add them to its import block:
`from tests._fixtures import _nad_cfg, _prior_artifact`.

- [ ] **Step 2: Run them and watch them fail**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest "tests/test_artifact_browser.py::test_setting_a_note_trims_it_and_asks_nothing" "tests/test_artifact_browser.py::test_an_over_long_note_is_refused_with_its_fix_sentence" -v`

Expected: FAIL, `AttributeError: 'ArtifactScreen' object has no attribute 'note_edit'`.

- [ ] **Step 3: Build the actions row**

In `core/gui/screens/artifact_screen.py`, add to the existing import block the names this task
needs (leave whatever Task 9/10 already imported):

```python
from PySide6.QtWidgets import (QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton,
                               QWidget)

from core.refusals import NOTE_MAX_CHARS, Refusal, require_note

from ..design import SPACE
from ..panels.base_panel import BasePanel
from ..widgets.refusal_box import show_refusal
```

**No new kinds import.** Task 9 already has `from core.artifacts.store import KIND_DIRS` at the top
of this file — the seven kinds in order — and Step 13's sweep iterates that. Do not add
`core.artifacts.KINDS` beside it: one module reading the kinds two ways is how the two lists drift.
Importing `BasePanel` does **not** enrol
the screen in `BasePanel._instances` — only constructing one does (`base_panel.py:91`), which is why
the browser can read the class flag without being greyed out by it.

Add the builder as a method on `ArtifactScreen`, and call it from `__init__` after the table and the
detail pane exist:

```python
    def _build_actions(self) -> QWidget:
        """The Note box and the Delete/Sweep buttons: one row under the listing.

        A plain group box, not the model builder's sticky action bar -- this screen does not scroll a
        form of unbounded length, so nothing can fall below the fold.
        """
        box = QGroupBox("Actions")
        row = QHBoxLayout(box)
        self.note_edit = QLineEdit()
        self.note_edit.setPlaceholderText(
            f"one line, at most {NOTE_MAX_CHARS} characters; empty clears it")
        # NO setMaxLength: B5 REFUSES an over-long note and names both numbers. A maxLength would
        # silently truncate it instead, which is the clamp V2 exists to forbid.
        self.btn_note = QPushButton("Set")
        self.btn_note.clicked.connect(self._set_note)
        self.btn_delete = QPushButton("Delete…")
        self.btn_delete.clicked.connect(self._delete)
        self.btn_sweep = QPushButton("Sweep this kind…")
        self.btn_sweep.setToolTip("Remove this kind's directories that have no usable manifest")
        self.btn_sweep.clicked.connect(lambda: self._sweep(all_kinds=False))
        self.btn_sweep_all = QPushButton("Sweep all kinds…")
        self.btn_sweep_all.clicked.connect(lambda: self._sweep(all_kinds=True))
        row.addWidget(QLabel("Note"))
        row.addWidget(self.note_edit, 1)
        row.addWidget(self.btn_note)
        row.addSpacing(SPACE[2])
        row.addWidget(self.btn_delete)
        row.addSpacing(SPACE[2])
        row.addWidget(self.btn_sweep)
        row.addWidget(self.btn_sweep_all)
        return box

    def _sync_actions(self) -> None:
        """Enable what the selected row can answer, and show its note.

        An incomplete directory has no manifest, so there is no note to rewrite and nothing for
        ``delete`` to resolve (it goes through ``_find``, which only ever returns manifest-bearing
        entries): the sweep is what removes those, which is also why the sweep is the one action that
        needs no selection at all.
        """
        s = self.table.current_summary()
        live = s is not None and s.complete
        self.note_edit.setEnabled(live)
        self.btn_note.setEnabled(live)
        self.btn_delete.setEnabled(live)
        self.note_edit.setText(s.note if live else "")

    def _after_change(self) -> None:
        """Re-list the kind and announce the change (B8). The three StorePickers are the WINDOW's to
        refresh: it is the one party that knows all of them, exactly as for the model combos."""
        self.refresh()
        self._sync_actions()
        self.store_changed.emit()
```

and in `__init__`, after Task 9/10's widgets are laid out and before `restore_settings` runs:

```python
        # The actions row sits directly above the status line.
        self.layout().insertWidget(self.layout().count() - 1, self._build_actions())
        self.table.selection_changed.connect(self._sync_actions)
        self._sync_actions()
```

- [ ] **Step 4: Implement `_selected` and `_set_note`**

Add to `ArtifactScreen` (no run guard yet — Step 17 adds it under its own failing test):

```python
    def _selected(self, doing: str):
        """The selected COMPLETE row, or None with the refusal already shown.

        Two different answers: nothing selected is a fielded refusal (the yellow box's fix sentence
        names the list), while a selected LEFTOVER is not a refusal at all -- there is no artifact
        there to act on, and saying so on the status line points at the sweep without pretending a
        rule was broken.
        """
        s = self.table.current_summary()
        if s is None:
            show_refusal(self, Refusal("No artifact is selected.", field="artifact"))
            return None
        if not s.complete:
            self._set_status(f"{s.dir_name} has no usable manifest ({s.reason}), so there is nothing "
                             f"to {doing}: it is one of the leftovers a sweep removes.", error=True)
            return None
        return s

    def _set_note(self) -> None:
        """B5: one trimmed line, at most NOTE_MAX_CHARS; blank clears it.

        The rule runs at the click, in ``core.refusals``, so ``python -m core artifacts note``
        refuses the same note with the same sentence; the front end only adds where to fix it.
        """
        s = self._selected("annotate")
        if s is None:
            return
        try:
            note = require_note("note", self.note_edit.text())
            self._resolved_store().set_note(s.kind, s.id, note)
        except Refusal as exc:      # StoreError is one: the row can have gone since it was listed,
            show_refusal(self, exc)  # and since Task 2 that refusal carries field="note" too
            self._set_status(exc.message, error=True)
            return
        self._set_status(f"Set the note on {s.kind} {s.label}." if note
                         else f"Cleared the note on {s.kind} {s.label}.")
        self._after_change()
```

The message goes on the status line as well as into the box, for the reason `BasePanel._refusal`
puts it in the log pane: it has to outlive the click that dismisses the box.

- [ ] **Step 5: Run the note tests**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest "tests/test_artifact_browser.py::test_setting_a_note_trims_it_and_asks_nothing" "tests/test_artifact_browser.py::test_an_over_long_note_is_refused_with_its_fix_sentence" -v`

Expected: PASS (2 passed).

- [ ] **Step 6: Write the failing delete tests**

```python
def test_delete_refuses_an_artifact_with_dependents_and_offers_no_yes(store):
    """B6. No front end offers force=True, so store.delete (store.py:478-488) refuses anything with
    dependents -- a confirmation could only ever be followed by a failure. dependents() is therefore
    read FIRST, and the yellow box names every dependent, including the training cache that holds
    the prior only by fingerprint and names it nowhere (store.py:468-476)."""
    from core.artifacts import store as st
    from PySide6.QtWidgets import QMessageBox
    from tests._fixtures import SHOWN, artifact_screen, qt_app
    qt_app()
    cfg = _nad_cfg()
    p = _prior_artifact(store, cfg, name="ancestor")
    fp = store.get("prior", p.id).fingerprints["gmm"]
    with store.create("inference", cfg, name="child") as w:        # names the prior as its parent
        w.parents = {"prior": p.id}
        w.body = {"results": {}}
    ident = {"format": "training-rows/2", "prior_fingerprint": fp, "n_runs": 3, "truncation": None}
    cache = st.write_simulation_manifest(store.kind_dir("simulation") / "abcdef012345", ident,
                                         parents=None)             # keyed on the GMM, names nothing
    scr = artifact_screen(store)
    _show_kind(scr, "prior")
    _select_ref(scr.table, p.id)
    scr._delete()
    box = SHOWN[-1]
    assert box.windowTitle() == "Check your inputs" and box.icon() == QMessageBox.Warning
    assert len(box.buttons()) == 1, "a refusal offers no Yes"
    assert "prior ancestor" in box.text() and p.id in box.text(), box.text()
    assert f"inference child [{w.id}]" in box.text(), box.text()
    assert f"simulation (unnamed) [{cache.id}]" in box.text(), box.text()
    assert ("a training cache was generated against this prior and its rows are meaningless "
            "without it") in box.text(), box.text()
    assert box.informativeText() == "Select an artifact in the list on the Artifacts screen."
    assert store.get("prior", p.id).name == "ancestor", "nothing may be deleted"
    assert "2 dependent(s) and was not deleted" in scr.status.text(), scr.status.text()


def test_deleting_an_unfinished_cache_names_its_batches_and_defaults_to_no(store, monkeypatch):
    """B3 + B6: a manifested cache is COMPLETE (it has a valid manifest) and not FINISHED, and the
    batches already committed are the one thing a delete here destroys that a rerun cannot remake --
    so the confirmation names them, and No is the default button (which is also what the session
    dialog guard's exec()==0 reads as).

    The cache is written with NO rows, because that is the only mid-run state there is: rows are
    written by mark_complete alone and ``save`` passes none (P2), so the prompt names batches and
    says where the row counts come from rather than printing a confident, false "0 rows"."""
    import pytest
    from core.artifacts import StoreError, store as st
    from PySide6.QtWidgets import QMessageBox
    from tests._fixtures import SHOWN, artifact_screen, qt_app
    qt_app()
    ident = {"format": "training-rows/2", "prior_fingerprint": None, "n_runs": 4, "truncation": None}
    cache = st.write_simulation_manifest(store.kind_dir("simulation") / "beef00112233", ident,
                                         batches_done=2, complete=False)
    scr = artifact_screen(store)
    _show_kind(scr, "simulation")
    s = _select_ref(scr.table, cache.id)
    assert s.complete and not s.finished, "B3: complete is 'has a manifest', finished is 'it ended'"
    assert s.rows is None, "``save`` records no rows: a mid-run cache has none to name"
    scr._delete()                                     # exec() returns 0 -> not Yes
    box = SHOWN[-1]
    assert box.button(QMessageBox.No) is box.defaultButton(), "No must be the default"
    assert "2 committed batch(es)" in box.informativeText(), box.informativeText()
    assert "rows are recorded when the cache finishes" in box.informativeText(), \
        box.informativeText()
    assert "0 rows" not in box.informativeText(), box.informativeText()
    assert store.get("simulation", cache.id).id == cache.id, "No must leave it on disk"
    assert "was not deleted" in scr.status.text(), scr.status.text()
    # Yes deletes it, and the change is announced (B8)
    _answer(monkeypatch, QMessageBox.Yes)
    _select_ref(scr.table, cache.id)
    changed = []
    scr.store_changed.connect(lambda: changed.append(True))
    scr._delete()
    with pytest.raises(StoreError):
        store.get("simulation", cache.id)
    assert changed, "a delete must emit store_changed"


def test_the_stores_own_refusal_is_the_last_word_on_a_delete(store, monkeypatch):
    """dependents() is read twice -- once here to avoid asking a question that could only fail, once
    inside delete() -- and the store's is the answer that counts. With the SCREEN's read stubbed
    empty the confirmation appears, and the store's own sentence is what the operator is shown, with
    its own fix sentence under it: that refusal carries field="artifact" (Task 5), so this race path
    shows it as it stands and invents nothing.

    The stub is a proxy over the real store, patched onto the SCREEN's _resolved_store, and not
    monkeypatch.setattr(store, "dependents", ...): ArtifactStore.delete calls self.dependents
    itself (store.py:482), so patching the store would blind the store too -- the delete would
    SUCCEED, the prior would be destroyed and every assertion below would be asserting the opposite
    of what it says (Q6). The proxy lies to the screen only."""
    from PySide6.QtWidgets import QMessageBox
    from tests._fixtures import SHOWN, artifact_screen, qt_app
    qt_app()
    cfg = _nad_cfg()
    p = _prior_artifact(store, cfg, name="ancestor")
    with store.create("inference", cfg, name="child") as w:
        w.parents = {"prior": p.id}
        w.body = {"results": {}}
    scr = artifact_screen(store)
    _show_kind(scr, "prior")
    _select_ref(scr.table, p.id)

    class _BlindToDependents:
        """Everything the screen asks of a store, delegated to the real one -- except dependents,
        which answers "none" the way a store would have a moment before the child was written."""

        def __init__(self, real):
            self._real = real

        def dependents(self, kind, id_):
            return []

        def get(self, *a, **k):
            return self._real.get(*a, **k)

        def list(self, *a, **k):
            return self._real.list(*a, **k)

        def delete(self, *a, **k):
            return self._real.delete(*a, **k)

    monkeypatch.setattr(scr, "_resolved_store", lambda: _BlindToDependents(store))
    _answer(monkeypatch, QMessageBox.Yes)
    scr._delete()
    box = SHOWN[-1]
    assert box.windowTitle() == "Check your inputs", "the store's refusal takes the yellow box"
    assert "refusing to delete" in box.text() and w.id in box.text(), box.text()
    # Task 5 gave delete()'s dependents refusal field="artifact", so show_refusal adds the same fix
    # sentence it adds to every other fielded refusal -- the race path needs no refusal of its own.
    # The normal path still builds one, for the wording only: the store's sentence ends "pass
    # force=True to orphan them", and no front end offers that (B6).
    assert box.informativeText() == "Select an artifact in the list on the Artifacts screen."
    assert store.get("prior", p.id).name == "ancestor", "the artifact survives its own refusal"
```

- [ ] **Step 7: Run them and watch them fail**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_artifact_browser.py -k "delete" -v`

Expected: FAIL — `AttributeError: 'ArtifactScreen' object has no attribute '_delete'` on all three.

- [ ] **Step 8: Add the two delete texts**

Module level in `core/gui/screens/artifact_screen.py`, under the imports:

```python
# The one dependent that names nothing. A cache's directory is keyed on the prior's GMM, so
# ArtifactStore.dependents reports it whether or not the manifest records the parent link -- and a
# bare list of ids gives an operator no way to tell that one from a child that named it.
_FINGERPRINT_DEPENDENT = ("a training cache was generated against this prior and its rows are "
                          "meaningless without it")


def _delete_prompt(s) -> tuple:
    """``(text, informative)`` for the confirmation: what goes, and what cannot come back.

    An UNFINISHED cache names its committed BATCHES: ``finished`` is not ``complete`` (B3), and those
    batches are the only thing deleting one destroys that a later run could not simply remake.

    Batches and not rows, deliberately (P2): ``rows`` is written by ``mark_complete`` alone --
    ``training_checkpoint.save`` passes none -- so every real mid-run cache has ``rows is None``, and
    a ``sum(())`` here would print a confident, false "0 rows". Where the row counts come from is
    said instead.
    """
    lines = [f"id {s.id}"]
    if s.kind == "simulation" and not s.finished:
        lines.append(f"This training cache is UNFINISHED: {s.batches_done} committed batch(es). "
                     "Deleting it throws those batches away and a later run starts from zero. "
                     "(Only the batch count is known while a cache is running: the rows are "
                     "recorded when the cache finishes.)")
    lines.append(f"This removes {s.path} and everything in it, and cannot be undone.")
    return f"Delete {s.kind} {s.label}?", "\n".join(lines)
```

and the refusal builder as a method:

```python
    def _dependents_refusal(self, store, s, deps) -> Refusal:
        """The sentence for a delete that nothing can perform: the artifact, then every artifact
        that depends on it WITH WHY.

        ``field="artifact"`` sends the operator to the list to delete the children first -- the same
        key the store's own dependents refusal carries since Task 5. This one exists for the WORDING
        alone: the store's message ends "pass force=True to orphan them", which no front end offers
        (B6), and a bare list of ids cannot say which dependent named this artifact and which the
        store found by fingerprint.
        """
        parents = {}
        for kind in sorted({k for k, _, _ in deps}):
            for row in store.list(kind):
                parents[(kind, row.id)] = row.parents
        lines = []
        for kind, id_, name in deps:
            held = parents.get((kind, id_)) or {}
            # dependents()'s first pass matches parents[<kind key>] == id; its second adds simulation
            # caches by fingerprint alone. "prior" is the only parent key a prior can occupy, so a
            # cache that does not hold it there came from that second pass.
            fingerprint_only = (s.kind == "prior" and kind == "simulation"
                                and held.get("prior") != s.id)
            why = _FINGERPRINT_DEPENDENT if fingerprint_only else f"it names this {s.kind} as a parent"
            lines.append(f"  {kind} {name or '(unnamed)'} [{id_}]: {why}.")
        return Refusal(
            f"Refusing to delete {s.kind} {s.label} [{s.id}]: {len(deps)} artifact(s) depend on it.\n"
            + "\n".join(lines)
            + "\nDelete those first. Nothing here can orphan them.", field="artifact")
```

- [ ] **Step 9: Implement `_delete`**

```python
    def _delete(self) -> None:
        """One artifact, no force, TWO outcomes (B6).

        ``dependents`` is read here, before anything is asked, because ``store.delete`` refuses
        anything with dependents and no front end offers ``force=True``: a confirmation for such an
        artifact could only ever be followed by a failure. With dependents this is a refusal naming
        every one of them; without, a confirmation that defaults to No. The store reads
        ``dependents`` again inside ``delete`` and stays the last word -- two directory scans, which
        is irrelevant at this scale.
        """
        s = self._selected("delete")
        if s is None:
            return
        store = self._resolved_store()
        deps = store.dependents(s.kind, s.id)
        if deps:
            show_refusal(self, self._dependents_refusal(store, s, deps))
            self._set_status(f"{s.kind} {s.label} has {len(deps)} dependent(s) and was not deleted.",
                             error=True)
            return
        text, detail = _delete_prompt(s)
        box = QMessageBox(self)                     # an INSTANCE dialog: the statics escape the
        box.setIcon(QMessageBox.Warning)            # suite's guard and hang it offscreen
        box.setWindowTitle("Delete artifact")
        box.setText(text)
        box.setInformativeText(detail)
        box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        box.setDefaultButton(QMessageBox.No)        # the safe branch, and what exec() == 0 reads as
        if box.exec() != QMessageBox.Yes:
            self._set_status(f"{s.kind} {s.label} was not deleted.")
            return
        try:
            store.delete(s.kind, s.id)              # never force, from either front end (B6)
        except Refusal as exc:                      # a row that went, or dependents that arrived,
            show_refusal(self, exc)                 # since the list -- shown as the store wrote it,
                                                    # fix sentence and all (its refusals are fielded)
            self._set_status(exc.message, error=True)
            return
        self._set_status(f"Deleted {s.kind} {s.label}.")
        self._after_change()
```

- [ ] **Step 10: Run the delete tests**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_artifact_browser.py -k "delete or stores_own_refusal" -v`

Expected: PASS (3 passed).

- [ ] **Step 11: Write the failing sweep test**

```python
def test_sweep_removes_only_the_leftovers_and_reports_what_it_could_not(store, monkeypatch):
    """B7: one action per kind and one for all seven, behind a confirmation that lists exactly the
    rows the table calls incomplete. A directory that holds a manifest is never touched, a directory
    that will not delete is reported and the rest still go, and nothing to do says so without
    asking."""
    from PySide6.QtWidgets import QMessageBox
    from tests._fixtures import SHOWN, artifact_screen, qt_app
    qt_app()
    cfg = _nad_cfg()
    p = _prior_artifact(store, cfg, name="keeper")
    leftover = store.kind_dir("prior") / "_unnamed__20260917T090000"
    leftover.mkdir(parents=True, exist_ok=True)
    (leftover / "prior.pt").write_bytes(b"half a run, no manifest")
    scr = artifact_screen(store)
    _show_kind(scr, "prior")
    scr._sweep(all_kinds=False)                          # No -> nothing goes
    box = SHOWN[-1]
    assert box.button(QMessageBox.No) is box.defaultButton()
    assert leftover.name in box.informativeText(), box.informativeText()
    assert "no manifest.json" in box.informativeText(), box.informativeText()
    assert leftover.is_dir() and "Nothing was removed." in scr.status.text()
    _answer(monkeypatch, QMessageBox.Yes)                # Yes -> the leftover goes, the artifact stays
    scr._sweep(all_kinds=False)
    assert not leftover.exists()
    assert store.get("prior", p.id).name == "keeper"
    assert "Removed 1 of 1" in scr.status.text(), scr.status.text()
    SHOWN.clear()                                        # nothing to do: no dialog at all
    scr._sweep(all_kinds=False)
    assert SHOWN == [] and "Nothing to remove" in scr.status.text(), scr.status.text()
    # a directory that will not delete (a handle held open on Windows) is reported, never fatal
    (store.kind_dir("prior") / "stuck__20260917T091000").mkdir()
    monkeypatch.setattr(store, "sweep_incomplete",
                        lambda kind=None: ([], [("prior", "stuck__20260917T091000",
                                                 "PermissionError: held open")]))
    scr._sweep(all_kinds=True)
    assert "could not be removed" in scr.status.text() and "held open" in scr.status.text()
```

- [ ] **Step 12: Run it and watch it fail**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest "tests/test_artifact_browser.py::test_sweep_removes_only_the_leftovers_and_reports_what_it_could_not" -v`

Expected: FAIL, `AttributeError: 'ArtifactScreen' object has no attribute '_sweep'`.

- [ ] **Step 13: Implement `_incomplete` and `_sweep`**

```python
    def _incomplete(self, store, kind) -> tuple:
        """``([(kind, dir_name, reason)], [str])``: what a sweep would remove, and any kind that
        could not be read at all -- an unreadable directory is not an empty one (§3.2).

        Read off ``list``, so the confirmation shows exactly the rows the table calls incomplete.
        """
        out, problems = [], []
        for k in ([kind] if kind else list(KIND_DIRS)):     # Task 9's import; the seven, in order
            try:
                rows = store.list(k)
            except Exception as e:      # noqa: BLE001 -- an unreadable kind is reported, not fatal
                problems.append(f"The {k} directory could not be read: {e}")
                continue
            out += [(k, row.dir_name, row.reason) for row in rows if not row.complete]
        return out, problems

    def _sweep(self, *, all_kinds: bool) -> None:
        """B7: remove every directory of this kind -- or of all seven -- that has no usable manifest.

        The candidates are listed BEFORE anything is removed, and ``sweep_incomplete`` is what
        removes them: a call that can only ever touch a directory ``_entries`` classifies as
        incomplete, the complement of ``delete``, which can only ever touch a real artifact.
        """
        store = self._resolved_store()
        kind = None if all_kinds else self.kind_combo.currentData()
        cands, problems = self._incomplete(store, kind)
        tail = (" " + " ".join(problems)) if problems else ""
        if not cands:
            where = "any kind's" if kind is None else f"{kind}"
            self._set_status(f"Nothing to remove: every {where} directory has a usable manifest."
                             + tail, error=bool(problems))
            return
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("Remove leftover directories")
        box.setText(f"Remove {len(cands)} director{'y' if len(cands) == 1 else 'ies'} with no "
                    f"usable manifest?")
        box.setInformativeText("\n".join(f"{k}/{d} — {why}" for k, d, why in cands)
                               + "\n\nA directory that holds a manifest is never touched by this.")
        box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        box.setDefaultButton(QMessageBox.No)
        if box.exec() != QMessageBox.Yes:
            self._set_status("Nothing was removed." + tail, error=bool(problems))
            return
        removed, failed = store.sweep_incomplete(kind)
        said = f"Removed {len(removed)} of {len(cands)} leftover director" \
               f"{'y' if len(cands) == 1 else 'ies'}."
        for k, d, why in failed:
            said += f" {k}/{d} could not be removed: {why}."
        self._set_status(said + tail, error=bool(failed or problems))
        self._after_change()
```

- [ ] **Step 14: Run the sweep test**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest "tests/test_artifact_browser.py::test_sweep_removes_only_the_leftovers_and_reports_what_it_could_not" -v`

Expected: PASS.

- [ ] **Step 15: Write the failing run-live guard test**

```python
def test_note_delete_and_sweep_are_refused_while_a_run_is_live_and_reading_is_not(store):
    """B6's second half, in the window's own wording (model_builder_screen.py:371 and :447). Every
    WRITE is refused while a run is live; READING never is -- which is the whole reason the browser
    is a plain screen and not a BasePanel (B1), so nothing here is greyed out either."""
    from core.gui.panels.base_panel import BasePanel
    from tests._fixtures import SHOWN, artifact_screen, qt_app
    qt_app()
    cfg = _nad_cfg()
    p = _prior_artifact(store, cfg, name="busy")
    leftover = store.kind_dir("prior") / "_unnamed__20260917T090000"
    leftover.mkdir(parents=True, exist_ok=True)
    scr = artifact_screen(store)
    _show_kind(scr, "prior")
    _select_ref(scr.table, p.id)
    scr.note_edit.setText("written during a train")
    BasePanel._running = True
    try:
        for action in (scr._set_note, scr._delete, lambda: scr._sweep(all_kinds=False)):
            scr._set_status("")
            action()
            assert "A task is running" in scr.status.text(), scr.status.text()
        assert SHOWN == [], "a refused action must not ask or explain in a box"
        assert store.get("prior", p.id).note == "", "the note must not be written"
        assert leftover.is_dir(), "the sweep must not run"
        scr.refresh()                                   # reading is never blocked
        assert scr.table.topLevelItemCount() >= 1
        _select_ref(scr.table, p.id)
        assert scr.note_edit.isEnabled() and scr.btn_delete.isEnabled(), \
            "the browser is not a BasePanel: a live run greys nothing here"
    finally:
        BasePanel._running = False
    # _select above re-read the row's (empty) note into the box, as a selection change must; type it
    # again and the same click works now that nothing is running.
    scr.note_edit.setText("written after the train")
    scr._set_note()
    assert store.get("prior", p.id).note == "written after the train"
```

- [ ] **Step 16: Run it and watch it fail**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest "tests/test_artifact_browser.py::test_note_delete_and_sweep_are_refused_while_a_run_is_live_and_reading_is_not" -v`

Expected: FAIL — the note is written while `BasePanel._running` is True:
`AssertionError: assert 'A task is running' in 'Set the note on prior busy.'`.

- [ ] **Step 17: Add the run guard to all three actions**

Add the helper:

```python
    def _refuse_while_running(self, doing: str) -> bool:
        """True when a run is live: the status line says so and the caller returns (B6).

        Every WRITE goes through this and nothing that only READS does. The wording is the window's
        own, from the two model-builder sites (model_builder_screen.py:371, :447); the status line is
        this screen's surface, so there is no dialog to dismiss either.
        """
        if BasePanel._running:
            self._set_status(f"A task is running -- wait for it to finish before {doing}.", error=True)
            return True
        return False
```

and insert one call as the FIRST statement of each action:

- `_set_note`: `if self._refuse_while_running("setting a note"):` / `    return`
- `_delete`: `if self._refuse_while_running("deleting an artifact"):` / `    return`
- `_sweep`: `if self._refuse_while_running("removing leftover directories"):` / `    return`

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest "tests/test_artifact_browser.py::test_note_delete_and_sweep_are_refused_while_a_run_is_live_and_reading_is_not" -v`

Expected: PASS.

- [ ] **Step 18: Write the picker-refresh test**

This one is not test-first, and says so: the window half — `MainWindow._refresh_store_pickers` and
the `store_changed` connect — is **Task 9's** (Q1), and the delete that drives it is Steps 8-9 above.
What this test pins is the two halves meeting, which is exactly what neither task can pin alone.

```python
def test_a_delete_in_the_browser_reaches_the_three_store_pickers(store, monkeypatch):
    """B8. Without this a picker keeps pointing at a deleted artifact: StorePicker.restore_key
    (artifact_picker.py:165-170) silently does nothing when the saved id has vanished, leaving
    whatever item happens to be current selected -- deliberate for a picker, and a defect the moment
    a browser can delete. Mirrors _refresh_model_combos: the window walks its own panels.

    The wiring (MainWindow._refresh_store_pickers and the store_changed connect) is the screen task's;
    the delete that emits store_changed is this task's. This is the test of the two together."""
    from PySide6.QtWidgets import QMessageBox
    from core.gui.main_window import MainWindow
    from core.gui.widgets.artifact_picker import StorePicker
    from tests._fixtures import qt_app
    qt_app()
    monkeypatch.setattr(StorePicker, "_resolved_store", lambda self: store)
    cfg = _nad_cfg()
    keep = _prior_artifact(store, cfg, name="keep")
    doomed = _prior_artifact(store, cfg, name="doomed", seed=3)
    w = MainWindow()
    try:
        picker = w.inference_screen.prior_panel.prior_picker
        picker.refresh()
        picker.restore_key(doomed.id)
        assert picker.key() == doomed.id
        scr = w.artifact_screen
        _show_kind(scr, "prior")
        _select_ref(scr.table, doomed.id)
        _answer(monkeypatch, QMessageBox.Yes)
        scr._delete()
        ids = [picker.combo.itemData(i) for i in range(picker.combo.count())]
        assert doomed.id not in ids, "the picker still offers the deleted prior"
        assert keep.id in ids and picker.key() != doomed.id
        # and it is all THREE pickers, not just the one that happened to be looked at
        seen = []
        real = StorePicker.refresh
        monkeypatch.setattr(StorePicker, "refresh",
                            lambda self: seen.append(self.kind) or real(self))
        w._refresh_store_pickers()
        assert sorted(seen) == ["observation", "posterior", "prior"], seen
    finally:
        w.close()
```

- [ ] **Step 19: Run it — it must PASS**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest "tests/test_artifact_browser.py::test_a_delete_in_the_browser_reaches_the_three_store_pickers" -v`

Expected: **PASS**, with nothing further to write. Task 9 already defined
`MainWindow._refresh_store_pickers` and connected `artifact_screen.store_changed` to it, and Step 9
gave this task the delete that emits it, so the loop is closed the moment both have landed.

Two failures are worth telling apart if it does not pass:

- `AttributeError: 'MainWindow' object has no attribute '_refresh_store_pickers'` (or
  `... 'artifact_screen'`) — **Task 9 has not landed.** Stop and land it; do not add a second
  definition of that method here, which is what this task used to do (Q1) and what would leave one
  class with two `def`s of one name and one signal connected twice.
- `AssertionError: the picker still offers the deleted prior` — Task 9's connect line is missing
  while its method is there. Fix it in the one place it belongs, beside the screen's construction in
  `MainWindow.__init__`, and say so in the task report.

- [ ] **Step 20: Write the failing static-dialog test**

Append to `tests/test_artifact_browser.py`. It reuses `_answer` from Step 1.

```python
def test_the_windows_three_model_dialogs_go_through_the_session_guard(monkeypatch):
    """The window's last three boxes are INSTANCE dialogs, so every box the GUI shows lands in
    SHOWN. main_window.py showed these three with the C++ statics (QMessageBox.warning at :215 and
    :236, QMessageBox.information at :221), which escape tests/conftest.py::_no_modal_dialogs -- it
    patches QMessageBox.exec, the instance method. Offscreen a static spins a nested event loop
    nothing ever closes, so a test that reached one STALLED instead of failing, which is why these
    three sites had no test at all. The statics are patched here as a tripwire rather than left
    live: if the conversion is ever undone, this fails naming the site instead of hanging the run.
    """
    from PySide6.QtWidgets import QMessageBox
    from core.Helpers import model_store
    from core.gui.main_window import MainWindow
    from core.gui.panels.base_panel import BasePanel
    from tests._fixtures import SHOWN, qt_app
    qt_app()
    statics = []
    for name in ("warning", "information"):
        monkeypatch.setattr(QMessageBox, name,
                            lambda *a, _n=name, **k: statics.append(_n) or QMessageBox.Ok)

    def boom(*a, **k):
        raise RuntimeError("corrupt definition")

    w = MainWindow()
    try:
        SHOWN.clear()                       # this test is about the three sites, not construction
        monkeypatch.setattr(w.model_builder_screen, "load_existing", boom)
        w._edit_user_model("BROKEN")        # :215 -- a definition that will not load
        monkeypatch.setattr(model_store, "delete_user_model", boom)
        BasePanel._running = True
        try:
            w._delete_user_model("BROKEN")  # :221 -- refused while a run is live, nothing asked
        finally:
            BasePanel._running = False
        _answer(monkeypatch, QMessageBox.Yes)
        w._delete_user_model("BROKEN")      # the confirmation, then :236 -- the delete itself fails
    finally:
        w.close()
    assert statics == [], f"the window still shows a box with a C++ static: {statics}"
    assert [b.windowTitle() for b in SHOWN] == ["Cannot edit model", "A task is running",
                                                "Delete model", "Delete failed"], \
        [b.windowTitle() for b in SHOWN]
    assert SHOWN[0].icon() == QMessageBox.Warning and "corrupt definition" in SHOWN[0].text()
    assert SHOWN[1].icon() == QMessageBox.Information
    assert "before deleting a model" in SHOWN[1].text(), SHOWN[1].text()
    assert SHOWN[3].icon() == QMessageBox.Warning and "corrupt definition" in SHOWN[3].text()
```

- [ ] **Step 21: Run it and watch it fail**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest "tests/test_artifact_browser.py::test_the_windows_three_model_dialogs_go_through_the_session_guard" -v`

Expected: FAIL, `AssertionError: the window still shows a box with a C++ static: ['warning',
'information', 'warning']`. It fails rather than hangs *because* the test patches the two statics
itself; run it with that loop deleted and it would stall, which is the hazard the conversion removes.

- [ ] **Step 22: Convert the three statics and correct the guard's comment**

In `core/gui/main_window.py` the three calls are at `:215`, `:221` and `:236` in the file as it
stands **before** Task 9's edits move them down, so search for the code rather than the number.
Add the helper directly above `_edit_user_model`:

```python
    def _tell(self, title: str, text: str, icon) -> None:
        """One plain box with an Ok button: an INSTANCE dialog shown with .exec().

        Never QMessageBox.warning/information. Those are C++ statics, and the suite's session guard
        patches the instance method only (tests/conftest.py::_no_modal_dialogs): offscreen a static
        spins a nested event loop nothing closes, so a test that reaches one stalls instead of
        failing. Every box in this window now goes through here or is built inline like the
        confirmation below and the one in closeEvent.
        """
        box = QMessageBox(self)
        box.setIcon(icon)
        box.setWindowTitle(title)
        box.setText(text)
        box.setStandardButtons(QMessageBox.Ok)
        box.exec()
```

Then replace the three call sites, changing nothing else about them — same titles, same texts, same
control flow:

```python
        except Exception as e:                       # noqa: BLE001 -- a corrupt file must not crash
            self._tell("Cannot edit model", f"Could not load '{name}':\n{e}", QMessageBox.Warning)
            return
```

```python
        if BasePanel._running:
            self._tell("A task is running",
                       "Wait for the running task to finish before deleting a model.",
                       QMessageBox.Information)
            return
```

```python
        except Exception as e:                       # noqa: BLE001
            self._tell("Delete failed", str(e), QMessageBox.Warning)
            return
```

In `tests/conftest.py::_no_modal_dialogs`, two sentences of the docstring are now wrong: the last
one records these sites as unreached, and the line number it cites for the safe branch is one the
helper above moves.

**This task owns both sentences, and it removes the number rather than re-pointing it** (Q10). Task 9
moves the same line but deliberately leaves this file alone, so the text below is still on disk
exactly as quoted — a name outlives an edit where a number does not. Replace

```
    exec returns 0 -- no clicked button, which the consent dialogs read as Cancel and main_window.py:231
    as the safe branch.
```

with

```
    exec returns 0 -- no clicked button, which the consent dialogs read as Cancel and
    MainWindow._delete_user_model's confirmation as the safe branch.
```

and replace

```
    Same MonkeyPatch shape as
    _checkpointing_off_unless_asked. The three static QMessageBox.warning/information calls in
    main_window.py are C++ statics and outside this guard; no test reaches them offscreen.
```

with

```
    Same MonkeyPatch shape as
    _checkpointing_off_unless_asked. Nothing under core/ calls the statics any more: main_window.py's
    last three went through MainWindow._tell in piece 4, so this guard covers every box the GUI
    shows and a test reads one off SHOWN.
```

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest "tests/test_artifact_browser.py::test_the_windows_three_model_dialogs_go_through_the_session_guard" -v`

Expected: PASS.

- [ ] **Step 23: Run the task's own tests**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_artifact_browser.py -q`

Expected: PASS (Task 9's and Task 10's tests in the file included; nothing skipped, nothing xfailed).

- [ ] **Step 24: Commit**

```bash
git add core/gui/screens/artifact_screen.py core/gui/main_window.py tests/conftest.py tests/test_artifact_browser.py
git commit -m "gui: the browser's note, delete and sweep actions" -m "Also converts main_window's last three static QMessageBox calls to instance dialogs, which the offscreen suite can reach."
```

---

### Task 12: the lineage report the browser writes to a file

**Files:**
- Modify: `core/gui/screens/artifact_screen.py` — one button beside Task 10's Save on the detail
  pane, and one method. (No line numbers: Task 9 creates the file and Task 10 adds the row this
  button joins.)
- Test: `tests/test_artifact_browser.py` — appended to.

**Interfaces:**

- Consumes:
  - `core.artifacts.report.render_lineage(store, kind: str, ref: str) -> str`, re-exported from
    `core/artifacts/__init__.py` (Task 6). Pure: the selected artifact, then its parents
    transitively, oldest last, with a parent that is absent from the store printed as MISSING.
  - `core.artifacts.report.render_manifest(m) -> str` is Task 10's (the detail pane and its Save);
    this task does not call it.
  - `show_refusal(parent, exc)` (Task 7); `Refusal(..., field="artifact")` and its GUI fix sentence
    `"Select an artifact in the list on the Artifacts screen."` (Task 2).
  - From Task 9/10's `ArtifactScreen`, every name settled: Task 9's `self.table`
    (`current_summary()`), `self._resolved_store()`, `self.status` and
    `self._set_status(text, error=False)`; and Task 10's `self.detail_actions`, the `QHBoxLayout` it
    builds under the detail pane (a stretch, then `self.btn_save`). This task appends its button to
    that layout, so it lands to the right of Save — where it belongs, since both write a file about
    the selected artifact. Nothing else here depends on Task 10.
  - From the test file: `_show_kind(scr, kind)` and `_select_ref(table, ident)`, added by **Task 11**
    Step 1, which is therefore a dependency of this task (Q9). Use them as they stand; do not copy
    them into a second definition, which would silently kill Task 11's the moment the two drift.
    (`_answer` is not used here — this task opens no dialog. Task 9's `_select(screen, i)` is a
    different helper, taking a row index; these tests act on one known artifact, so `_select_ref`.)
  - From Task 11's module-level imports in `core/gui/screens/artifact_screen.py`: `Refusal` and
    `show_refusal`, both used by Step 4's `_lineage_report`. Step 3 adds them only if they are not
    already at the top of the file.
- Produces: `ArtifactScreen.btn_lineage` (`QPushButton("Lineage report…")`) and
  `ArtifactScreen._lineage_report(self)`.

**Why this task exists:** "A saveable run summary" is the decomposition's piece-4 row and had dropped
out of `docs/STATE.md` entirely (spec §1). **B10** makes it two things — Save on the detail view
(Task 10) and a **lineage report** that walks the selected artifact's parents back through the chain
— and both are **files, never an eighth store kind**: a report *describes* the store and must never
be mistaken for something the store holds (§5). The report is a READ, so unlike Task 11's three
actions it is not refused while a run is live (B6). One renderer serves both front ends, so the
document a reviewer receives is the same whichever made it; this task therefore asserts the file's
bytes against `render_lineage` directly rather than against Task 14's `artifacts summary`, so the two
front ends cannot drift apart without a failure.

- [ ] **Step 1: Write the failing report test**

Append to `tests/test_artifact_browser.py`:

```python
def test_the_lineage_report_writes_exactly_what_render_lineage_returns(store, monkeypatch, tmp_path):
    """B10 + §5: ONE renderer, and the GUI adds nothing to it -- no header, no banner, no trailing
    newline. Asserted against render_lineage itself rather than against the tool's `artifacts
    summary`, so the two front ends cannot drift: whichever wrote the file, a reviewer gets the same
    bytes. UTF-8 with LF endings, so 'the same bytes' is true on this platform too."""
    from PySide6.QtWidgets import QFileDialog
    from core.artifacts import render_lineage
    from tests._fixtures import SHOWN, artifact_screen, qt_app
    qt_app()
    cfg = _nad_cfg()
    ancestor = _prior_artifact(store, cfg, name="ancestor")
    with store.create("inference", cfg, name="descendant") as w:
        w.parents = {"prior": ancestor.id}
        w.body = {"results": {"n_samples": 8}}
    scr = artifact_screen(store)
    _show_kind(scr, "inference")
    _select_ref(scr.table, w.id)
    out = tmp_path / "lineage.txt"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(out), ""))
    scr._lineage_report()
    assert out.read_bytes() == render_lineage(store, "inference", w.id).encode("utf-8")
    text = out.read_text(encoding="utf-8")
    assert "ancestor" in text and ancestor.id in text, "the chain must reach the parent"
    assert SHOWN == [], "writing a report asks nothing"
    assert str(out) in scr.status.text(), scr.status.text()
    # It is a FILE and never an eighth kind: the store's seven directories gained nothing.
    assert [s.id for s in store.list("inference")] == [w.id]
    assert not (store.root / "reports").exists()
```

- [ ] **Step 2: Run it and watch it fail**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest "tests/test_artifact_browser.py::test_the_lineage_report_writes_exactly_what_render_lineage_returns" -v`

Expected: FAIL, `AttributeError: 'ArtifactScreen' object has no attribute '_lineage_report'`.

- [ ] **Step 3: Add the button**

In `core/gui/screens/artifact_screen.py`, add to the import block (leaving anything Tasks 9-11
already imported):

```python
from pathlib import Path

from PySide6.QtWidgets import QFileDialog

from core.refusals import Refusal

from ..widgets.refusal_box import show_refusal
```

Task 11 already imports `Refusal` and `show_refusal` into this module for its own refusals — add
each of the last two lines only if it is not there, and never a second time.

and in `__init__`, on Task 10's `self.detail_actions` row, directly after its `self.btn_save`:

```python
        self.btn_lineage = QPushButton("Lineage report…")
        self.btn_lineage.setToolTip("Write this artifact and its parents, back through the chain, "
                                    "to a text file")
        self.btn_lineage.clicked.connect(self._lineage_report)
        self.detail_actions.addWidget(self.btn_lineage)
```

- [ ] **Step 4: Implement `_lineage_report`, without its guards**

The two guards and the extension patch come in Step 8, under their own failing test.

```python
    def _lineage_report(self) -> None:
        """B10: write the selected artifact's lineage to a file the operator names.

        A report DESCRIBES the store and is never an eighth kind in it (§5): nothing here writes
        into the artifact root -- ``render_lineage`` reads manifests and this writes its text
        wherever the operator says. UTF-8 with LF endings, which is byte for byte what
        ``python -m core artifacts summary --out`` writes, so the document cannot say which front end
        made it.

        NOT guarded by ``BasePanel._running``: this is a read, and reading is never refused (B6).
        """
        s = self.table.current_summary()
        try:
            text = render_lineage(self._resolved_store(), s.kind, s.id)
        except Refusal as exc:               # a parent that cannot be read at all
            show_refusal(self, exc)
            self._set_status(exc.message, error=True)
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save lineage report",
                                              f"{s.kind}-{s.name or s.id}-lineage.txt",
                                              "Text file (*.txt)")
        if not path:
            return
        try:
            Path(path).write_text(text, encoding="utf-8", newline="\n")
        except OSError as e:
            self._set_status(f"Could not write the lineage report: {e}", error=True)
            return
        self._set_status(f"Wrote the lineage report for {s.kind} {s.label} to {path}.")
```

Add `render_lineage` to the module's `core.artifacts` import (Task 10 may already import
`render_manifest` from there): `from core.artifacts import render_lineage, render_manifest`.

- [ ] **Step 5: Run the report test**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest "tests/test_artifact_browser.py::test_the_lineage_report_writes_exactly_what_render_lineage_returns" -v`

Expected: PASS.

- [ ] **Step 6: Write the failing guard-and-extension test**

```python
def test_the_lineage_report_is_a_read_and_needs_a_complete_row(store, monkeypatch, tmp_path):
    """Three legs of one rule. A report is a READ, so a live run does not refuse it (B6) -- unlike
    Task 11's note, delete and sweep. A leftover directory has no manifest and so no lineage. And a
    name the operator typed without an extension gets .txt rather than a file nothing opens."""
    from PySide6.QtWidgets import QFileDialog
    from core.artifacts import render_lineage
    from core.gui.panels.base_panel import BasePanel
    from tests._fixtures import SHOWN, artifact_screen, qt_app
    qt_app()
    cfg = _nad_cfg()
    p = _prior_artifact(store, cfg, name="ancestor")
    leftover = store.kind_dir("prior") / "_unnamed__20260917T090000"
    leftover.mkdir(parents=True, exist_ok=True)
    scr = artifact_screen(store)
    _show_kind(scr, "prior")
    _select_ref(scr.table, p.id)
    # A directory of its own: tmp_path also holds this test's prism.ini (_isolated_settings) and the
    # store fixture's Artifacts/, so it is not a place to count files in.
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    bare = out_dir / "chain"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(bare), ""))
    BasePanel._running = True
    try:
        scr._lineage_report()
    finally:
        BasePanel._running = False
    written = bare.with_suffix(".txt")
    assert written.read_bytes() == render_lineage(store, "prior", p.id).encode("utf-8")
    assert not bare.exists(), "an extension-less name must gain .txt, not be written as given"
    # a leftover directory: no manifest, no lineage, and no file
    scr._set_status("")
    _select_ref(scr.table, leftover.name)
    scr._lineage_report()
    assert "no usable manifest" in scr.status.text(), scr.status.text()
    assert "lineage" in scr.status.text(), scr.status.text()
    # nothing selected at all: the fielded refusal, and still no file
    scr.table.clearSelection()
    scr.table.setCurrentItem(None)
    scr._lineage_report()
    box = SHOWN[-1]
    assert box.windowTitle() == "Check your inputs"
    assert box.informativeText() == "Select an artifact in the list on the Artifacts screen."
    assert sorted(q.name for q in out_dir.iterdir()) == ["chain.txt"]
```

- [ ] **Step 7: Run it and watch it fail**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest "tests/test_artifact_browser.py::test_the_lineage_report_is_a_read_and_needs_a_complete_row" -v`

Expected: FAIL at the first leg —
`FileNotFoundError` / `AssertionError` on `written.read_bytes()`, because Step 4 wrote the file as
`chain` with no extension. (The running leg already passes: Step 4 deliberately has no
`_refuse_while_running` call, which is the point of B6's second half.)

- [ ] **Step 8: Add the extension patch and the two guards**

In `_lineage_report`, replace the bare `s = self.table.current_summary()` with the two guards, and
patch the extension after the dialog returns:

```python
        s = self.table.current_summary()
        if s is None:
            show_refusal(self, Refusal("No artifact is selected.", field="artifact"))
            return
        if not s.complete:
            # A leftover directory has no manifest, so there is no chain to walk -- and unlike the
            # note and the delete, this is not a rule anybody broke, so it goes on the status line.
            self._set_status(f"{s.dir_name} has no usable manifest ({s.reason}), so it has no "
                             f"lineage to report.", error=True)
            return
```

```python
        if not path.lower().endswith(".txt"):
            path += ".txt"                   # the courtesy simulate_panel._save_video does for .mp4
```

- [ ] **Step 9: Run the guard test**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest "tests/test_artifact_browser.py::test_the_lineage_report_is_a_read_and_needs_a_complete_row" -v`

Expected: PASS.

- [ ] **Step 10: Run the task's own tests**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_artifact_browser.py -q`

Expected: PASS.

- [ ] **Step 11: Commit**

```bash
git add core/gui/screens/artifact_screen.py tests/test_artifact_browser.py
git commit -m "gui: a lineage report written to a file, not a store kind"
```

### Task 13: the tool's `artifacts list` and `artifacts show`

**Files:**
- Create: `core/tool/browse.py`
- Modify: `core/tool/__init__.py:18` (the `from . import ...` line) and `core/tool/__init__.py:43`
  (the `for module in (...)` registration loop)
- Test: `tests/test_tool.py` (append at the end; the file is 1713 lines today, and its last block is
  `test_no_print_call_remains_in_the_converted_modules`)

**Interfaces:**
- Consumes:
  - Task 3's `Summary` fields — `dir_name: str`, `finished: bool`, `batches_done: int | None`,
    `batches_planned: int | None` (the SIXTH field, off the simulation body's `identity["n_runs"]`,
    so a cache's progress can be rendered as a fraction — spec §12 row 2),
    `rows: tuple[int, ...] | None`, `variant: str | None`, beside the existing `kind id name created
    note path complete reason mode width amortized parents` and the `label` property.
  - Task 4's `ArtifactStore.read_log(self, kind: str, ref: str, *, max_bytes: "int | None" = None)
    -> "tuple[str | None, bool]"` — `(text, truncated)`, `(None, False)` when there is no file.
  - Task 6's `core.artifacts.report.render_manifest(m) -> str`, re-exported from `core.artifacts`.
  - Task 8's `core.gui.widgets.artifact_table.columns_for(kind: str) -> tuple[str, ...]` — read by a
    TEST only. **`core/tool/browse.py` must NOT import `core.gui.widgets.artifact_table`**, or
    anything else under `core/gui`: that would pull PySide6 into `python -m core --help`, which every
    subcommand's help path is free of today. So the tool keeps its own `COLUMNS` literal and the two
    column sets are pinned against each other by a test — the TEST imports both front ends, the tool
    imports neither (and the GUI does not import the tool either).
  - **Task 9's** `tests/_fixtures.py::build_browse_store(root) -> dict` — one artifact of each of the
    seven kinds plus the three bad-directory shapes, written by the real writer in seconds. Every
    test below runs against it, which is why Task 9 is a dependency of this task (Q9). Its
    simulation cache is written at 3 of 4 batches, not complete and with **no rows** — the only state
    a mid-run cache can be in (P2).
  - The existing handler contract: `main` calls `rc = int(args.handler(args, store) or 0)` inside
    `use_store(ArtifactStore(root))`, with `root = config.artifacts_root()`
    (`core/tool/__init__.py:95-111`), so a handler is handed the open store and resolves no root of
    its own.
- Produces: `core.tool.browse.register(subparsers) -> dict` returning `{"artifacts": <parser>}`;
  the module constants `KINDS`, `COLUMNS`, `LOG_TAIL_BYTES`, `EPILOG`, and the two shared argument
  help strings `_KIND` and `_REF` (Task 14's four `add_parser` blocks use both); the
  row formatters `_flat(text) -> str`, `_width(s) -> str`, `_progress(s) -> str`,
  `_cells(kind, s) -> tuple`, `_print_kind(kind, rows) -> None`; the handlers
  `_list(args, store) -> int` and `_show(args, store) -> int`. Task 14 adds four more modes to the
  same `register` and the same epilog.

**Why this task exists:** The store has been complete since piece 1 and almost none of it is
reachable: `set_note`, `delete`, `dependents` and `unnamed` have no caller at all but the suites, and
since piece 3 every committed artifact carries a `log.txt` that nothing reads (spec §1). **B9** gives
the tool an `artifacts` family with modes `list show note rm sweep summary`, taking **no
configuration flags** and reading `config.artifacts_root()`, with an empty listing at exit 0. This
task builds the two read-only modes; Task 14 adds the four that write.

- [ ] **Step 1: Write the failing tests for `list`**

Append to `tests/test_tool.py`. `main`, `build_parser` and `config` are already imported at the top of
that file; `Path` and `pytest` too.

```python
# ── the artifacts family (piece 4, B9; design §4) ────────────────────────────────────────────────


@pytest.fixture
def browse_store(tmp_path, monkeypatch):
    """A store holding one artifact of each of the seven kinds plus three unusable directories, with
    PRISM_ARTIFACTS pointing at it.

    ``config.artifacts_root()`` reads the variable on EVERY call (core/config.py:185-189) and ``main``
    opens its store on it, so ``main(["artifacts", ...])`` reads exactly this root -- the same redirect
    test_the_sbc_subcommand_forwards_every_knob_as_a_keyword uses. Yields ``(root, ids)``, where
    ``ids`` is ``{kind: id}`` from tests/_fixtures.py::build_browse_store (Task 9's builder: the real
    writer at minimum size, seconds not minutes -- it must never reach for ``tiny_run``)."""
    from tests._fixtures import build_browse_store
    root = tmp_path / "A"
    ids = build_browse_store(root)
    monkeypatch.setenv("PRISM_ARTIFACTS", str(root))
    return root, ids


def test_the_artifacts_listing_shows_the_browsers_own_columns():
    """ONE FORMATTER PER FACT (design §1.2), kept honest across two front ends that cannot share code.
    The browser's ``cells_for``/``columns_for`` live in a Qt module, and importing it from
    ``core/tool/browse.py`` would pull PySide6 into ``python -m core --help`` -- so the tool keeps its
    own column list and the two COLUMN SETS are pinned against each other HERE rather than trusted to
    agree by eye. THIS TEST imports both front ends; the tool imports neither. A column added to the
    table without one added to the tool fails right here.

    The seven kinds are restated in the tool as a literal for the same reason (reading KIND_DIRS at
    parser-build time would import torch), so their order is pinned too -- the same shape as
    test_crossval_preset_choices_match_sweep_presets, which pins --preset's hard-coded choices against
    cli.SWEEP_PRESETS."""
    from core.artifacts.store import KIND_DIRS
    from core.gui.widgets.artifact_table import columns_for
    from core.tool import browse

    assert browse.KINDS == tuple(KIND_DIRS), "the seven kinds, in KIND_DIRS order"
    assert set(browse.COLUMNS) == set(browse.KINDS), "every kind has a column set, and no other"
    for kind in browse.KINDS:
        # The tool's literal is the GUI's own spelling, lower-cased (P11): one canonical column list,
        # Title-case in the window and lower-case in a terminal where the output may be piped.
        assert browse.COLUMNS[kind] == tuple(c.lower() for c in columns_for(kind)), kind


def test_artifacts_list_prints_one_line_per_artifact_with_its_kinds_facts(browse_store, capsys):
    """design §3.2's table, per kind: a posterior's mode, width and amortization; a cache's progress
    as a FRACTION (``batches_done``/``batches_planned``) and -- separately -- whether it FINISHED (B3:
    ``complete`` means "has a valid manifest", which is true of a cache from its first batch on). Read
    off the store's own rows rather than off literal names, so the assertions hold whatever
    build_browse_store names its artifacts."""
    from core.artifacts import ArtifactStore
    root, ids = browse_store
    s = ArtifactStore(root)

    capsys.readouterr()
    assert main(["artifacts", "list", "posterior"]) == 0
    out = capsys.readouterr().out
    assert "== posterior ==" in out, out
    post = next(r for r in s.list("posterior") if r.complete)
    line = next(ln for ln in out.splitlines() if post.label in ln)
    assert post.created in line, line
    assert str(post.width) in line and (post.mode or "") in line, line
    assert ("amortized" if post.amortized else "narrowed (TSNPE)") in line, line

    capsys.readouterr()
    assert main(["artifacts", "list", "simulation"]) == 0
    out = capsys.readouterr().out
    sim = next(r for r in s.list("simulation") if r.complete)
    line = next(ln for ln in out.splitlines() if sim.label in ln)
    assert f"{sim.batches_done}/{sim.batches_planned} batches" in line, line
    if sim.finished:
        assert f", {sum(sim.rows)} rows" in line, line
    else:
        assert "rows" not in line, "``save`` passes no rows, so a cache mid-run has none to show"
    assert ("yes" if sim.finished else "no") in line, line


def test_the_artifacts_listing_progress_cell_is_a_fraction_and_shows_rows_only_once_finished():
    """Both halves of the progress cell, on stand-in summaries: ``build_browse_store``'s cache is an
    unfinished one, so the FINISHED half has no fixture to come from: ``batches_planned`` (the body's
    ``identity["n_runs"]``, spec §12 row 2) makes the cell a FRACTION, and ``rows`` is written only by
    ``mark_complete`` -- ``save`` passes none -- so a cache mid-run shows its batches and no row
    count. A comma and plain ASCII, never the browser's middle dot: this is text a script may read."""
    from core.artifacts.store import Summary
    from core.tool import browse

    def summary(**kw):
        return Summary(kind="simulation", id="s", name="", created="2026-01-01 00:00:00", note="",
                       path=Path("."), complete=True, reason=None, **kw)

    assert browse._progress(summary(batches_done=3, batches_planned=4, rows=None)) == "3/4 batches"
    assert browse._progress(summary(batches_done=4, batches_planned=4, finished=True,
                                    rows=(48, 48))) == "4/4 batches, 96 rows"
    assert browse._progress(summary()) == "?/? batches", "a body with neither count still renders"


def test_artifacts_list_with_no_kind_covers_every_kind_under_a_heading(browse_store, capsys):
    root, ids = browse_store
    capsys.readouterr()
    assert main(["artifacts", "list"]) == 0
    out = capsys.readouterr().out
    for kind in ("prior", "simulation", "posterior", "observation", "calibration", "inference",
                 "diagnostic"):
        assert f"== {kind} ==" in out, kind
    assert "nothing in:" not in out, "build_browse_store writes one artifact of every kind"


def test_an_empty_artifacts_listing_exits_0_and_a_bad_kind_exits_1(tmp_path, monkeypatch, capsys):
    """§4.3: an empty listing is 0, with a line saying there is nothing there. A script must be able to
    tell "nothing on disk" from "you asked for something wrong", and exiting 1 on an empty store would
    make the two indistinguishable. A kind that does not exist IS the second case: the store's own
    refusal, through the ladder's ``refused:`` rung at exit 1.

    The empty root need not exist beforehand -- ``main`` mkdirs the store root before any handler runs
    (core/tool/__init__.py:103)."""
    monkeypatch.setenv("PRISM_ARTIFACTS", str(tmp_path / "empty"))

    capsys.readouterr()
    assert main(["artifacts", "list"]) == 0
    out = capsys.readouterr().out
    named_empty = out.split("nothing in:")[1]
    for kind in ("prior", "simulation", "posterior", "observation", "calibration", "inference",
                 "diagnostic"):
        assert kind in named_empty, kind
    assert "holds no artifacts yet" in out, out

    capsys.readouterr()
    assert main(["artifacts", "list", "prior"]) == 0
    out = capsys.readouterr().out
    assert "== prior ==" in out and "nothing here yet" in out, out

    capsys.readouterr()
    assert main(["artifacts", "list", "nosuchkind"]) == 1
    err = capsys.readouterr().err
    lines = [ln for ln in err.splitlines() if ln.startswith("prism artifacts: refused:")]
    assert len(lines) == 1 and "unknown artifact kind" in lines[0], err
    assert "Traceback" not in err, err


def test_artifacts_list_puts_an_unusable_directory_last_with_its_reason(browse_store, capsys):
    """A leftover directory is the first thing an operator can act on that no front end has ever shown:
    ``get``/``path`` and the GUI picker all skip it. ``list`` already sorts incomplete rows last, and
    the row carries the ``dir_name`` because that is the only handle ``sweep`` can take (B7)."""
    root, ids = browse_store
    leftover = root / "priors" / "_unnamed__20260101T000000"
    leftover.mkdir(parents=True, exist_ok=True)

    capsys.readouterr()
    assert main(["artifacts", "list", "prior"]) == 0
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    hits = [i for i, ln in enumerate(lines) if "_unnamed__20260101T000000" in ln]
    assert len(hits) == 1, lines
    assert "no manifest.json" in lines[hits[0]], lines[hits[0]]
    assert hits[0] == len(lines) - 1, f"an incomplete directory is listed LAST: {lines}"
```

- [ ] **Step 2: Run them and watch them fail**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_tool.py -q -k "artifacts_list or artifacts_listing or empty_artifacts"`
Expected: FAIL — `ModuleNotFoundError: No module named 'core.tool.browse'` for the column test and the
progress-cell test, and `assert 2 == 0` for the four that drive `main` (argparse rejects the unknown
subcommand `artifacts` with exit 2, which `main` returns from its `except SystemExit` branch).

- [ ] **Step 3: Create `core/tool/browse.py` — the docstring and the two literals**

The module is **not** called `artifacts.py`: `core/tool/__init__.py:18` does `from . import
config_args, diagnostics, fdt, smoke, stages`, so a sibling of that name would sit one mistaken
relative import away from shadowing the `core.artifacts` package this module is a front end for
(design §4).

```python
"""``python -m core artifacts <mode>``: read, annotate and tidy the artifact store.

The command-line twin of the Artifacts screen (piece 4, B9; design §4). ONE subcommand with a
positional mode -- the ``identifiability {rotation,laplace,jacobian}`` precedent
(core/tool/diagnostics.py:60-105), where each mode is a parser of its own, so a flag that means
nothing to a mode is an argparse error rather than a setting that is silently ignored.

WHY THIS MODULE IS NOT ``artifacts.py``. ``core/tool/__init__.py`` does ``from . import ...``, so a
sibling of that name would sit one mistaken relative import away from shadowing the
``core.artifacts`` package this module is a front end for.

NO CONFIGURATION FLAGS. ``add_config_flags`` is deliberately NOT called (design §4.1): a listing must
not be able to fail on a bounds file it does not need, and ``python -m core artifacts --help`` must
stay torch-free. The consequence is stated in EPILOG: this family cannot say "this posterior does not
match your bounds file" -- that is what LOADING it does. ``fdt``/``crossval`` are the precedent for a
subcommand that builds no SimConfig.

NO ``--store-root``, EITHER. The root is ``config.artifacts_root()`` (PRISM_ARTIFACTS), like every
subcommand but ``smoke``, and ``main`` has already opened the store by the time a handler runs. The
name is not reused because ``main`` keys three of smoke's behaviours on the FLAG's presence:
``has_store_root = hasattr(args, "store_root")`` (:95) decides whether a fresh ``mkdtemp`` root is
created (:96-98), whether an empty auto-created root is removed afterwards (:156-157), and which
Ctrl-C advice is printed (:121-122). Declaring ``--store-root`` here would change all three for this
family, for a root it does not want.

OUTPUT IS ``print``. The tool's framing prints are deliberately outside the no-print pin -- "The
tool's framing prints (core/tool) stay prints and are deliberately outside this set"
(tests/test_tool.py:1701) -- and a browse command's output IS framing: it is the result a script
reads, not the pipeline's own voice, and it must never reach an artifact's log.txt.

HEAVY IMPORTS INSIDE THE HANDLERS, per core/tool/stages.py's rule, so building the parser costs no
torch import.

AND NOTHING FROM ``core/gui``, EVER -- not even the browser's ``columns_for``, however tempting it is
to read the column titles from the one place that owns them: importing it would pull PySide6 into
``python -m core --help``. This module therefore keeps its own ``COLUMNS``, and a test in
tests/test_tool.py imports BOTH front ends and pins the two column sets against each other.
"""
import argparse

# The seven kinds in KIND_DIRS order, restated as a literal: reading core.artifacts.store.KIND_DIRS
# at parser-build time would import torch (core/artifacts/__init__.py imports .store, which imports
# core.config). tests/test_tool.py pins the two in sync, both order and membership -- the same shape
# as --preset's hard-coded choices pinned against cli.SWEEP_PRESETS.
KINDS = ("prior", "simulation", "posterior", "observation", "calibration", "inference", "diagnostic")

# The columns each kind shows, exactly design §3.2's table: the browser's own tuples
# (core/gui/widgets/artifact_table.columns_for), LOWER-CASED (P11). A SECOND literal on purpose --
# that module is a Qt module, and importing it here would pull PySide6 into `python -m core --help`.
# So this module imports NOTHING from core/gui, and the two column sets are kept in step by a test
# that imports both and lower-cases the GUI's (tests/test_tool.py).
COLUMNS = {
    "prior": ("name", "created", "note"),
    "simulation": ("name", "created", "progress", "finished", "note"),
    "posterior": ("name", "created", "mode", "width", "amortized", "note"),
    "observation": ("name", "created", "mode", "width", "note"),
    "calibration": ("name", "created", "note"),
    "inference": ("name", "created", "note"),
    "diagnostic": ("name", "created", "variant", "note"),
}

# The tail of log.txt ``show`` prints. A DISPLAY cap, not a knob: it bounds what one screenful of
# records can cost. The browser's detail pane has its own cap (artifact_screen.LOG_MAX_BYTES),
# chosen independently and today at the same 1 MiB; nothing pins the two equal, because each front
# end bounds its own surface -- a terminal and a text box are not the same screenful.
LOG_TAIL_BYTES = 1 << 20

EPILOG = """\
Reads PRISM_ARTIFACTS (the artifacts root, default <repo>/Artifacts) -- the same root as every
subcommand but `smoke`. There is no --store-root, and there are no configuration flags: a listing
must not be able to fail on a bounds file it does not need, and --help here costs no torch import.
The consequence is that this family cannot tell you whether a posterior matches your bounds file --
that is what LOADING it does (`python -m core validate --posterior ...`).

  list [<kind>]         one line per artifact; with no kind, all seven under headings
  show <kind> <ref>     the manifest, then the records of the run that wrote it

<kind> is one of prior, simulation, posterior, observation, calibration, inference, diagnostic.
<ref> is an artifact's name or its id. An empty listing exits 0: a script must be able to tell
"nothing on disk" from "you asked for something wrong".
"""
```

**`COLUMNS` is Task 8's `columns_for` lower-cased** (P11) — the browser's spelling is the canonical
one, because its header is what a user reads; the tool lower-cases it because its output is terminal
text a script may read. Check the literal above against the GUI before you go on:

```powershell
& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -c "from core.gui.widgets.artifact_table import columns_for; [print(k, tuple(c.lower() for c in columns_for(k))) for k in ('prior','simulation','posterior','observation','calibration','inference','diagnostic')]"
```

(That command imports Qt, so run it from a shell where `QT_QPA_PLATFORM=offscreen` and
`KMP_DUPLICATE_LIB_OK=TRUE` are set.) If a title differs from the literal above, `COLUMNS` follows
`columns_for`, lower-cased — not the other way round.

- [ ] **Step 4: Add the row formatters**

Append to `core/tool/browse.py`. Every cell is a string and every cell is ONE line: a note written
before B5's one-line rule could still hold a newline, and a table whose rows wrap is unreadable.

```python
def _flat(text) -> str:
    """One line, whitespace collapsed: a cell in an aligned table cannot carry a newline or a tab."""
    return " ".join(str(text or "").split())


def _width(s) -> str:
    return "?" if s.width is None else str(s.width)


def _progress(s) -> str:
    """"3/4 batches" while a cache runs, "4/4 batches, 96 rows" once it has finished -- from
    ``Summary`` ALONE (B2: a row needs no second manifest read). The total is ``batches_planned``, off
    the body's ``identity["n_runs"]`` (spec §12 row 2), so the cell is a FRACTION. The row count is
    written only by ``mark_complete`` -- ``save`` passes none -- so a cache mid-run has no rows-so-far
    to show and the cell stops at the batches. The separate ``finished`` column still answers whether
    the batches are all there (B3): a manifest marked complete is what makes 4/4 mean finished.
    A comma and plain ASCII, not the browser's middle dot: a script may read this line."""
    done = "?" if s.batches_done is None else str(s.batches_done)
    planned = "?" if s.batches_planned is None else str(s.batches_planned)
    text = f"{done}/{planned} batches"
    if s.rows:
        text += f", {sum(int(r) for r in s.rows)} rows"
    return text


def _cells(kind: str, s) -> tuple:
    """One complete row's cells, in ``COLUMNS[kind]`` order. ``label`` is the name, or
    ``(unnamed <id>)`` -- as the pickers show it, and as a simulation cache always reads, because
    write_simulation_manifest always writes ``name=""``."""
    head = (_flat(s.label), _flat(s.created))
    if kind == "simulation":
        return head + (_progress(s), "yes" if s.finished else "no", _flat(s.note))
    if kind == "posterior":
        # The exception is what is worth a word: amortized is the norm (B13's rule for the pickers,
        # applied to the table's own cell).
        amortized = "amortized" if s.amortized else ("narrowed (TSNPE)" if s.amortized is False
                                                     else "?")
        return head + (_flat(s.mode), _width(s), amortized, _flat(s.note))
    if kind == "observation":
        return head + (_flat(s.mode), _width(s), _flat(s.note))
    if kind == "diagnostic":
        # A diagnostic records its kind under ``variant``, never ``mode``: the mode column would
        # otherwise report a conditioning geometry that does not exist (design §1.3).
        return head + (_flat(s.variant) or "-", _flat(s.note))
    return head + (_flat(s.note),)


def _print_kind(kind: str, rows) -> None:
    """One kind's heading, its aligned table, and its unusable directories last."""
    print(f"\n== {kind} ==")
    good = [s for s in rows if s.complete]
    broken = [s for s in rows if not s.complete]
    if not good:
        print("  nothing here yet" if not broken else "  no artifact with a usable manifest here")
    else:
        columns = COLUMNS[kind]
        cells = [_cells(kind, s) for s in good]
        widths = [max([len(columns[i])] + [len(c[i]) for c in cells]) for i in range(len(columns))]
        for row in (columns, *cells):
            print("  " + "  ".join(v.ljust(w) for v, w in zip(row, widths)).rstrip())
    for s in broken:
        # LAST, with the reason in place of the kind's own columns, and identified by the directory
        # name -- the handle ``sweep`` takes (B7). ``list`` already sorts them last; this split is
        # what keeps the table above aligned on real rows only.
        print(f"  incomplete  {_flat(s.dir_name)}  -- {_flat(s.reason)}")
```

- [ ] **Step 5: Add `_list` and a `register` carrying the `list` mode alone**

Append to `core/tool/browse.py`. `_show` and its mode come in Step 10, after Step 8's tests have been
seen to fail — write only what Step 1's tests ask for here.

```python
def _list(args, store) -> int:
    """``list [<kind>]``. An empty listing is EXIT 0 with a line saying so (§4.3); an unknown kind is
    the store's own refusal, through the ladder's ``refused:`` rung at exit 1. With no kind, all seven
    are walked and a kind with nothing in it is NAMED in a trailing line rather than silently
    skipped, so "I did not look" and "there is nothing" stay distinguishable."""
    kinds = [args.kind] if args.kind else list(KINDS)
    empty = []
    for kind in kinds:
        rows = store.list(kind)                  # an unknown kind: StoreError -> the ladder's exit 1
        if not rows and not args.kind:
            empty.append(kind)
            continue
        _print_kind(kind, rows)
    if empty:
        print(f"\nnothing in: {', '.join(empty)}")
        if len(empty) == len(kinds):
            print(f"the store under {store.root} holds no artifacts yet.")
    return 0


_KIND = "one of " + ", ".join(KINDS)
_REF = "the artifact's name, or its id"


def register(subparsers) -> dict:
    """``{name: subparser}`` -- the contract ``build_parser``'s ``p.subcommands.update(...)`` loop
    needs. One subcommand, six modes; Task 14 adds four of them."""
    p = subparsers.add_parser(
        "artifacts", help="read, annotate and tidy the artifact store (loads nothing, simulates "
                          "nothing)",
        epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    modes = p.add_subparsers(dest="mode", required=True, metavar="{list,show}")

    ls = modes.add_parser("list", help="one line per artifact of a kind, or of all seven")
    ls.add_argument("kind", nargs="?", default=None, metavar="<kind>", help=_KIND)
    ls.set_defaults(handler=_list)
    return {"artifacts": p}
```

Two things in that block are deliberate. The mode `metavar` already reads `{list,show}` because
Step 10 adds `show` inside this same task (Task 14 widens it again to
`{list,show,note,rm,sweep,summary}`). And the kind is a plain positional with **no `choices=`**:
§4.3 puts a bad kind at exit **1**, through the store's own refusal — `choices=` would make it an
argparse usage error at exit 2, and the store's `kind_dir` already refuses an unknown kind by name.

- [ ] **Step 6: Register the module in `build_parser`**

`core/tool/__init__.py:18` — add `browse` to the import (alphabetical, as the line already is):

```python
from . import browse, config_args, diagnostics, fdt, smoke, stages
```

`core/tool/__init__.py:43` — add it to the registration loop:

```python
    for module in (stages, diagnostics, smoke, fdt, browse):
```

**Leave the top-level `EPILOG` (`:23-33`) alone.** It enumerates no subcommands — it names the four
environment variables and says none is a substitute for a flag, and
`test_the_help_epilog_names_the_core_environment_settings` (`tests/test_tool.py:125-130`) pins
exactly those four names. `python -m core --help` lists the subcommands from argparse's own
`metavar="<subcommand>"` and each parser's `help=`, so the family appears there the moment it is
registered and nothing in the epilog needs to change.

- [ ] **Step 7: Run the `list` tests**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_tool.py -q -k "artifacts_list or artifacts_listing or empty_artifacts"`
Expected: PASS

- [ ] **Step 8: Write the failing tests for `show` and for the family's shape**

Append to `tests/test_tool.py`.

```python
def test_artifacts_show_prints_the_manifest_then_the_runs_records(browse_store, capsys, monkeypatch):
    """The manifest through the ONE renderer both front ends use (design §5), then the records with
    their ``HH:MM:SS level`` stamps, then a truncation notice when the tail was cut. Asserting the
    renderer's own output is IN the printed text is what keeps a second, drifting renderer from being
    written here."""
    from core.artifacts import ArtifactStore, render_manifest
    from core.artifacts.store import LOG_FILE
    from core.tool import browse
    root, ids = browse_store
    s = ArtifactStore(root)
    (s.path("prior", ids["prior"]) / LOG_FILE).write_text("10:00:00 info the sweep began\n",
                                                          encoding="utf-8")

    capsys.readouterr()
    assert main(["artifacts", "show", "prior", ids["prior"]]) == 0
    out = capsys.readouterr().out
    assert render_manifest(s.get("prior", ids["prior"])) in out, "the one renderer, not a second one"
    assert "== records ==" in out and "10:00:00 info the sweep began" in out, out

    # the tail, and the notice that says it is one. The cap is monkeypatched rather than met with a
    # megabyte of log: what is under test is the notice, not the byte count.
    monkeypatch.setattr(browse, "LOG_TAIL_BYTES", 40)
    (s.path("prior", ids["prior"]) / LOG_FILE).write_text("A" * 200 + "\n10:00:01 info the last line\n",
                                                          encoding="utf-8")
    capsys.readouterr()
    assert main(["artifacts", "show", "prior", ids["prior"]]) == 0
    out = capsys.readouterr().out
    assert "the log is longer" in out and "10:00:01 info the last line" in out, out


def test_artifacts_show_states_the_two_honest_gaps(browse_store, capsys):
    """design §3.3's two sentences, WORD FOR WORD the browser's, because one operator reads both. A
    cache has no log by design (it is written batch by batch across resumes and shared by every
    posterior that names it), and an artifact written with no run active has none either -- and
    ``read_log`` answers ``(None, False)`` for both, so the KIND is what tells them apart."""
    from core.artifacts import ArtifactStore
    from core.artifacts.store import LOG_FILE
    root, ids = browse_store
    s = ArtifactStore(root)

    capsys.readouterr()
    assert main(["artifacts", "show", "simulation", ids["simulation"]]) == 0
    out = capsys.readouterr().out
    assert "a training cache keeps no log" in out and "across resumes" in out, out

    (s.path("observation", ids["observation"]) / LOG_FILE).unlink(missing_ok=True)
    capsys.readouterr()
    assert main(["artifacts", "show", "observation", ids["observation"]]) == 0
    assert "written outside a run" in capsys.readouterr().out


def test_artifacts_show_names_a_directory_the_manifest_disagrees_with(browse_store, capsys):
    """``rename`` writes the manifest first and tolerates a refused directory move (the manifest is
    what resolves an artifact), which leaves a folder whose name disagrees with ``Manifest.dir_name``.
    ``show`` is the first place that is visible (design §3.3, §2.6)."""
    from core.artifacts import ArtifactStore
    root, ids = browse_store
    sub = ArtifactStore(root).path("calibration", ids["calibration"])
    sub.rename(sub.with_name("stale__" + sub.name.rsplit("__", 1)[-1]))

    capsys.readouterr()
    assert main(["artifacts", "show", "calibration", ids["calibration"]]) == 0
    out = capsys.readouterr().out
    assert "stale__" in out and "while its manifest says" in out, out


def test_artifacts_show_on_a_missing_ref_exits_1_through_the_refused_rung(browse_store, capsys):
    """§4.3: a missing or ambiguous ref is 1, through the existing ``refused:`` rung. The message
    already quotes the ref, which is exactly why core/tool/fields.py maps the ``artifact`` key to
    None -- the tool names the artifact positionally, so there is no option string to print, and
    fix_sentence owns the parentheses, so the line simply ends at the message."""
    root, ids = browse_store
    capsys.readouterr()
    assert main(["artifacts", "show", "prior", "nosuch"]) == 1
    err = capsys.readouterr().err
    lines = [ln for ln in err.splitlines() if ln.startswith("prism artifacts: refused:")]
    assert len(lines) == 1, err
    assert "nosuch" in lines[0] and "no complete prior artifact" in lines[0], lines[0]
    assert not lines[0].endswith("()"), lines[0]
    assert "Traceback" not in err and "raised at" not in err, err


def test_the_artifacts_family_takes_no_configuration_flags_and_its_help_costs_no_torch():
    """B9's two halves, pinned. (1) No configuration flags anywhere in the family: a listing must not
    be able to fail on a bounds file it does not need, and --store-root in particular is absent
    because ``main`` keys smoke's temp-root behaviour, its empty-root cleanup and its Ctrl-C advice on
    ``hasattr(args, "store_root")`` (core/tool/__init__.py:95-98, 121-122, 156-157). (2) ``--help``
    for the family imports no torch -- checked in a FRESH interpreter, because in this process torch
    is long since imported by the session fixtures, so a sys.modules check here would pass
    vacuously (the pattern of test_every_field_key_has_a_flag_..., leg (d))."""
    import argparse
    import subprocess
    import sys

    p = build_parser().subcommands["artifacts"]
    modes = {name: sub for a in p._actions if isinstance(a, argparse._SubParsersAction)
             for name, sub in a.choices.items()}
    assert set(modes) >= {"list", "show"}, sorted(modes)
    for name, parser in [("artifacts", p), *sorted(modes.items())]:
        for flag in ("--bounds", "--model", "--chi", "--chi-k", "--device", "--store-root",
                     "--accept-truncated"):
            assert flag not in parser._option_string_actions, (name, flag)

    probe = ("import sys\n"
             "from core.tool import main\n"
             "rc = main(['artifacts', '--help'])\n"
             "bad = sorted(m for m in sys.modules if m == 'torch' or m.startswith('torch.'))\n"
             "sys.exit(0 if rc == 0 and not bad else repr((rc, bad[:3])))\n")
    r = subprocess.run([sys.executable, "-c", probe], cwd=str(config.REPO_ROOT),
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
```

- [ ] **Step 9: Run them and watch them fail**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_tool.py -q -k "artifacts_show or artifacts_family"`
Expected: FAIL — the four `show` tests on `assert 2 == 0` (there is no `show` mode yet, so argparse
rejects the unknown mode with exit 2), and `test_the_artifacts_family_...` on
`AssertionError: ['list']` from its `assert set(modes) >= {"list", "show"}`.

- [ ] **Step 10: Add `_show` and its mode**

Append the handler to `core/tool/browse.py`, beside `_list`:

```python
def _show(args, store) -> int:
    """``show <kind> <ref>``: the manifest as design §3.3 renders it, then the run's records, then the
    same honest gaps the browser's detail pane states. A ref that names no complete artifact is the
    store's own refusal (exit 1); its message already quotes the ref, which is why
    core/tool/fields.py maps the ``artifact`` key to None -- the tool names the artifact
    positionally, so there is no option string to print."""
    from core.artifacts import render_manifest
    m = store.get(args.kind, args.ref)
    print(render_manifest(m))
    sub = store.path(args.kind, args.ref)
    if sub.name != m.dir_name:
        # A rename whose directory move was refused leaves exactly this, and no front end has ever
        # shown it (design §3.3). The manifest is what resolves an artifact, so nothing is broken.
        print(f"\nthe directory is named {sub.name!r} while its manifest says {m.dir_name!r}: a "
              f"rename whose directory move was refused leaves that. The manifest is what resolves "
              f"the artifact, so nothing here is broken.")
    print("\n== records ==")
    text, truncated = store.read_log(args.kind, args.ref, max_bytes=LOG_TAIL_BYTES)
    if text is None:
        # TWO honest gaps behind one answer: read_log says (None, False) for both, so the KIND is
        # what tells them apart. Both sentences are the browser's, word for word (design §3.3) --
        # one operator reads both front ends.
        print("  a training cache keeps no log: it is written batch by batch across resumes and "
              "shared by every posterior that names it."
              if args.kind == "simulation" else
              "  written outside a run: nothing captured this artifact's records.")
        return 0
    if truncated:
        print(f"  (only the last {LOG_TAIL_BYTES} bytes are shown; the log is longer)")
    if not text:
        print("  the run wrote no records. Silence is a record too: the file is there and empty.")
    else:
        print(text, end="" if text.endswith("\n") else "\n")
    return 0
```

and its mode inside `register`, after `list`:

```python
    show = modes.add_parser("show", help="one artifact's manifest and the records of the run that "
                                         "wrote it")
    show.add_argument("kind", metavar="<kind>", help=_KIND)
    show.add_argument("ref", metavar="<ref>", help=_REF)
    show.set_defaults(handler=_show)
```

- [ ] **Step 11: Run the task's own tests**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_tool.py -q -k "artifacts"`
Expected: PASS

Then the two whole-file pins that this task could break — the tool's env/knob scan walks every module
under `core/tool`, and the FLAG table's closure test walks every parser `build_parser` builds:

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_tool.py::test_the_tool_reads_no_environment_and_writes_no_knob tests/test_tool.py::test_usage_errors_exit_2 "tests/test_refusals.py::test_every_field_key_has_a_flag_and_every_key_literal_under_core_is_registered" -q`
Expected: PASS

- [ ] **Step 12: Commit**

```bash
git add core/tool/browse.py core/tool/__init__.py tests/test_tool.py
git commit -m "tool: artifacts list and show over the store"
```

---

### Task 14: the tool's `note`, `rm`, `sweep` and `summary`

**Files:**
- Modify: `core/tool/browse.py` — Task 13 created it; re-read it before editing, because every line
  number below moved with it. Three places change: the module's import block (it needs `sys`,
  `pathlib.Path` and — Step 3, because the parser is BUILT with it —
  `from core.refusals import NOTE_MAX_CHARS`), `EPILOG`'s mode list, and `register`'s `metavar` plus
  four new `modes.add_parser` blocks. Four handlers are appended beside `_list` and `_show`.
- Test: `tests/test_tool.py` (append at the end; Task 13 left the file longer, so re-read the tail)

**Interfaces:**
- Consumes:
  - Task 13's `core/tool/browse.py`: `KINDS`, `EPILOG`, `_KIND`, `_REF`, `register`.
  - Task 13's `browse_store` fixture in `tests/test_tool.py`, and through it **Task 9's**
    `tests/_fixtures.py::build_browse_store(root)` — which is why Task 9 is on this task's dependency
    line (Q9).
  - Task 2's `core.refusals.require_note(key: str, text: str) -> str` and `NOTE_MAX_CHARS = 200`, and
    `core/tool/fields.py`'s new entries `"note": "--note"` and `"artifact": None`.
  - Task 5's `ArtifactStore.sweep_incomplete(self, kind: "str | None" = None) -> "tuple[list, list]"`
    returning `(removed, failed)` = `([(kind, dir_name)], [(kind, dir_name, reason)])`.
  - Task 6's `core.artifacts.report.render_lineage(store, kind: str, ref: str) -> str`, re-exported
    from `core.artifacts`.
  - The store as it already is: `set_note(kind, ref, note) -> Manifest`, `path(kind, ref) -> Path`,
    and `delete(kind, ref, *, force=False)`, whose refusal lists every dependent — including, for a
    prior, a training cache that holds it only by fingerprint.
  - The field key on a missing ref: Task 2 gave `set_note`'s "no complete artifact" refusal
    `field="artifact"`, **not** `field="note"` (spec §12 row 1: that sentence is about an artifact
    that is not there, so the note key would send the operator to the wrong control), and `read_log`'s
    and `delete`'s missing-ref refusals carry the same key. `core/tool/fields.py` maps `artifact` to
    `None`, so `fix_sentence` adds nothing and the `refused:` line ends at the message with **no
    trailing parenthetical**. `field="note"` is `require_note`'s own — the over-long note and the
    newline — and those DO print `(--note)`.
- Produces: the handlers `_note`, `_rm`, `_sweep`, `_summary`, each `(args, store) -> int`; the six-mode
  `artifacts` family is complete after this task.

**Why this task exists:** `set_note`, `delete` and `dependents` have had no production caller at all
since piece 1 (spec §1), and nothing can remove a directory that has no usable manifest. **B9** gives
the tool the four modes that write: **B5**'s note rule (one line, at most 200 characters, blank
clears, a newline or an over-long note refused rather than clamped), **B6**'s delete with **no
`--force` in either front end**, **B7**'s sweep, and **B10**'s lineage report as a file. Exit codes
are §4.3's.

- [ ] **Step 1: Write the failing tests for `note`**

Append to `tests/test_tool.py`.

```python
def test_artifacts_note_sets_trims_and_clears_a_note(browse_store, capsys):
    """B5: the note is one line, trimmed, and a blank one CLEARS it. ``--note`` is a flag rather than
    a positional precisely so core/tool/fields.py can map the ``note`` key to a real option string --
    the table's values are pinned against build_parser()'s own option strings, both ways."""
    from core.artifacts import ArtifactStore
    root, ids = browse_store
    s = ArtifactStore(root)

    capsys.readouterr()
    assert main(["artifacts", "note", "prior", ids["prior"], "--note", "  the first fit  "]) == 0
    assert s.get("prior", ids["prior"]).note == "the first fit", "require_note trims, and only trims"
    assert "the first fit" in capsys.readouterr().out

    assert main(["artifacts", "note", "prior", ids["prior"], "--note", ""]) == 0
    assert s.get("prior", ids["prior"]).note == "", "a blank note clears it"


def test_a_note_that_breaks_the_rule_is_refused_naming_the_flag(browse_store, capsys):
    """V2 on the note: no clamp and no silent default. An over-long note is refused with BOTH numbers
    in the sentence, a newline is refused rather than flattened, and the manifest is untouched in
    either case. The flag comes from core/tool/fields.py's table, appended by main's ladder -- and
    only for a refusal about the note TEXT, which is ``require_note``'s (field="note"). The third leg
    is the other refusal this mode can raise: ``set_note``'s own "no complete artifact", which carries
    field="artifact" (spec §12 row 1), a key whose flag is None -- so that line ends at the message."""
    from core.artifacts import ArtifactStore
    from core.refusals import NOTE_MAX_CHARS
    root, ids = browse_store
    s = ArtifactStore(root)
    before = s.get("prior", ids["prior"]).note

    capsys.readouterr()
    assert main(["artifacts", "note", "prior", ids["prior"],
                 "--note", "a" * (NOTE_MAX_CHARS + 1)]) == 1
    err = capsys.readouterr().err
    lines = [ln for ln in err.splitlines() if ln.startswith("prism artifacts: refused:")]
    assert len(lines) == 1 and lines[0].endswith("(--note)"), err
    assert str(NOTE_MAX_CHARS) in lines[0] and str(NOTE_MAX_CHARS + 1) in lines[0], lines[0]
    assert "raised at" not in err and "Traceback" not in err, err

    capsys.readouterr()
    assert main(["artifacts", "note", "prior", ids["prior"], "--note", "a\nb"]) == 1
    assert "(--note)" in capsys.readouterr().err
    assert s.get("prior", ids["prior"]).note == before, "a refused note changed the manifest"

    # A ref that names no artifact is a DIFFERENT refusal: set_note's, with field="artifact" (spec
    # §12 row 1), whose entry in core/tool/fields.py is None because the tool names the artifact
    # positionally. fix_sentence owns the parentheses, so the line simply ends at the message -- no
    # "(--note)" here, because the note is not what is wrong.
    capsys.readouterr()
    assert main(["artifacts", "note", "prior", "nosuch", "--note", "a fine note"]) == 1
    err = capsys.readouterr().err
    lines = [ln for ln in err.splitlines() if ln.startswith("prism artifacts: refused:")]
    assert len(lines) == 1 and "nosuch" in lines[0], err
    assert not lines[0].rstrip().endswith(")"), f"no trailing parenthetical: {lines[0]}"
    assert "--note" not in lines[0], lines[0]
```

- [ ] **Step 2: Run them and watch them fail**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_tool.py -q -k "artifacts_note or note_that_breaks"`
Expected: FAIL — there is no `note` mode, so argparse rejects it with exit 2: `assert 2 == 0` in the
first test and `assert 2 == 1` on the first refusal leg of the second.

- [ ] **Step 3: Add `_note` and its mode**

In `core/tool/browse.py`, append the handler beside `_show`:

```python
def _note(args, store) -> int:
    """``note <kind> <ref> --note TEXT``. B5's rule runs BEFORE the store is touched, so a bad note
    rewrites no manifest; ``""`` clears the note, which is why ``--note`` is required rather than
    defaulted -- a note must never be cleared by leaving a flag off.

    Two refusals, two field keys, both from elsewhere: ``require_note``'s (the over-long note, the
    newline) carries field="note", so the ladder prints ``(--note)``, while ``set_note``'s "no
    complete artifact" carries field="artifact", whose entry in core/tool/fields.py is None -- the
    tool names the artifact positionally -- so that line simply ends at the message."""
    from core.refusals import require_note
    text = require_note("note", args.note)
    m = store.set_note(args.kind, args.ref, text)
    label = m.name or m.id
    if m.note:
        print(f"[prism] {args.kind} {label}: note = {m.note!r}")
    else:
        print(f"[prism] {args.kind} {label}: note cleared")
    return 0
```

Then register the mode inside `register`, after `show`:

```python
    note = modes.add_parser("note", help="set or clear one artifact's note")
    note.add_argument("kind", metavar="<kind>", help=_KIND)
    note.add_argument("ref", metavar="<ref>", help=_REF)
    note.add_argument("--note", required=True, metavar="TEXT",
                      help=f"one line, at most {NOTE_MAX_CHARS} characters; '' clears it. A note "
                           f"with a newline, or a longer one, is refused rather than trimmed to fit")
    note.set_defaults(handler=_note)
```

`NOTE_MAX_CHARS` is needed while the parser is being BUILT, so import it at module top —
`core/refusals.py` is torch-free and stdlib-only by contract, so this costs the `--help` path nothing
(`core/tool/config_args.py:20` already imports from it at module level for the same reason):

```python
from core.refusals import NOTE_MAX_CHARS
```

- [ ] **Step 4: Run the `note` tests**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_tool.py -q -k "artifacts_note or note_that_breaks"`
Expected: PASS

- [ ] **Step 5: Write the failing tests for `rm`**

Append to `tests/test_tool.py`.

```python
def test_artifacts_rm_offers_no_force_and_deletes_one_artifact(browse_store, capsys):
    """B6: delete is one artifact at a time and there is NO force in either front end, so the store's
    refusal is the last word. A diagnostic is the leaf case -- ``_PARENT_KEYS["diagnostic"]`` is
    empty, so nothing can ever name one as a parent."""
    import argparse
    from core.artifacts import ArtifactStore, StoreError
    root, ids = browse_store
    modes = {name: sub for a in build_parser().subcommands["artifacts"]._actions
             if isinstance(a, argparse._SubParsersAction) for name, sub in a.choices.items()}
    assert "--force" not in modes["rm"]._option_string_actions, "B6: no force in either front end"

    s = ArtifactStore(root)
    gone = s.path("diagnostic", ids["diagnostic"])
    capsys.readouterr()
    assert main(["artifacts", "rm", "diagnostic", ids["diagnostic"]]) == 0
    assert str(gone) in capsys.readouterr().out
    assert not gone.exists()
    with pytest.raises(StoreError):
        s.get("diagnostic", ids["diagnostic"])


def test_artifacts_rm_refuses_an_artifact_something_depends_on(tool_run, capsys):
    """The other half of B6: because no force is offered, an artifact anything depends on cannot be
    deleted AT ALL, and the refusal names every dependent. ``tool_run``'s ``tp`` has both kinds: the
    posterior ``tpost`` names it as a parent, and the training cache ``--checkpoint-every 1`` wrote
    holds it only by FINGERPRINT, with no recorded parent link -- the case the browser has to state in
    its own words and the tool gets from the store's message."""
    from core.artifacts import ArtifactStore
    bounds, cell, root = tool_run
    s = ArtifactStore(root)
    before = s.get("prior", "tp").id

    capsys.readouterr()
    assert main(["artifacts", "rm", "prior", "tp"]) == 1
    err = capsys.readouterr().err
    lines = [ln for ln in err.splitlines() if ln.startswith("prism artifacts: refused:")]
    assert len(lines) == 1, err
    assert "refusing to delete" in lines[0] and "name it as a parent" in lines[0], lines[0]
    assert "posterior" in lines[0] and "simulation" in lines[0], lines[0]
    assert s.get("prior", "tp").id == before, "a refusal removed something"
```

- [ ] **Step 6: Run them and watch them fail**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_tool.py -q -k "artifacts_rm"`
Expected: FAIL — `KeyError: 'rm'` on the first (no such mode in the parser), and `assert 2 == 1` on
the second: it asserts the refusal's exit 1, and argparse rejects the unknown mode with 2 first.

- [ ] **Step 7: Add `_rm` and its mode**

```python
def _rm(args, store) -> int:
    """``rm <kind> <ref>``. NO ``--force`` (B6): the store refuses anything with dependents, naming
    each, and that refusal is the last word -- there is no flag here that can orphan a child. The
    path is read before the delete so the line can say what went."""
    sub = store.path(args.kind, args.ref)        # a missing ref: StoreError -> the ladder's exit 1
    store.delete(args.kind, args.ref)
    print(f"[prism] removed {args.kind} {args.ref}: {sub}")
    return 0
```

```python
    rm = modes.add_parser("rm", help="delete one artifact; refused when anything depends on it")
    rm.add_argument("kind", metavar="<kind>", help=_KIND)
    rm.add_argument("ref", metavar="<ref>", help=_REF)
    rm.set_defaults(handler=_rm)
```

- [ ] **Step 8: Write the failing tests for `sweep`**

Append to `tests/test_tool.py`.

```python
def test_artifacts_sweep_removes_only_the_unusable_directories(browse_store, capsys):
    """B7: one action removes every directory of a kind that has no usable manifest, through a store
    call that CAN ONLY remove such a directory -- ``delete`` resolves through ``_find``, which never
    returns a manifest-less entry, and ``remove_incomplete`` is its complement. §4.3: nothing to
    remove is exit 0, and it says so rather than printing nothing."""
    root, ids = browse_store
    leftover = root / "priors" / "_unnamed__20260101T000000"
    leftover.mkdir(parents=True, exist_ok=True)
    kept = [p for p in (root / "priors").iterdir() if p != leftover]
    assert kept, "build_browse_store wrote a prior"

    capsys.readouterr()
    assert main(["artifacts", "sweep", "prior"]) == 0
    out = capsys.readouterr().out
    assert "_unnamed__20260101T000000" in out, out
    assert not leftover.exists()
    assert all(p.exists() for p in kept), "sweep removed an artifact with a usable manifest"

    capsys.readouterr()
    assert main(["artifacts", "sweep", "prior"]) == 0
    assert "nothing to sweep" in capsys.readouterr().out


def test_a_sweep_that_could_not_remove_something_exits_1_naming_it(browse_store, monkeypatch, capsys):
    """§4.3: a sweep that removed everything is 0; one that could not remove a directory is 1, naming
    each failure. A held handle on Windows is what that stands for and it cannot be provoked on
    demand, so the two lists are injected on the store's own method -- what is under test is the
    ladder, not the removal."""
    from core.artifacts import ArtifactStore
    root, ids = browse_store
    monkeypatch.setattr(
        ArtifactStore, "sweep_incomplete",
        lambda self, kind=None: ([("prior", "gone__1")],
                                 [("posterior", "stuck__2", "PermissionError: in use")]))
    capsys.readouterr()
    assert main(["artifacts", "sweep"]) == 1
    cap = capsys.readouterr()
    assert "gone__1" in cap.out, cap.out
    assert "stuck__2" in cap.err and "in use" in cap.err, cap.err
```

- [ ] **Step 9: Run them and watch them fail**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_tool.py -q -k "sweep"`
Expected: FAIL, `assert 2 == 0` — there is no `sweep` mode.

- [ ] **Step 10: Add `_sweep` and its mode**

`core/tool/browse.py` needs `import sys` at the top for this handler (a failure is something to act
on, so it goes to stderr, where the tool puts everything an operator's `2>err.log` should hold):

```python
def _sweep(args, store) -> int:
    """``sweep [<kind>]``: remove every directory of a kind (or of all seven) with no usable
    manifest. §4.3: nothing to remove is 0; a directory that would not delete is 1, naming each. A
    failure never stops the sweep -- ``sweep_incomplete`` finishes the rest and reports it."""
    removed, failed = store.sweep_incomplete(args.kind)
    for kind, dir_name in removed:
        print(f"[prism] removed {kind} leftover {dir_name}")
    for kind, dir_name, reason in failed:
        print(f"prism artifacts: could not remove {kind} leftover {dir_name}: {reason}",
              file=sys.stderr)
    if not removed and not failed:
        print("[prism] nothing to sweep: every directory here carries a usable manifest.")
    return 1 if failed else 0
```

```python
    sweep = modes.add_parser("sweep", help="remove every directory with no usable manifest")
    sweep.add_argument("kind", nargs="?", default=None, metavar="<kind>",
                       help=f"{_KIND}; with no kind, all seven")
    sweep.set_defaults(handler=_sweep)
```

- [ ] **Step 11: Write the failing tests for `summary`**

Append to `tests/test_tool.py`.

```python
def test_artifacts_summary_prints_the_lineage_or_writes_it_to_out(browse_store, tmp_path, capsys):
    """B10: the lineage report is a FILE, never an eighth store kind, and it comes out of the same
    renderer the browser's "Lineage report..." writes -- so the document a reviewer receives is the
    same whichever front end made it (design §5). ``--out`` writes exactly what stdout would have
    carried.

    The file is compared as BYTES, not as text: read_text would translate CRLF back to LF on the way
    in and pass whatever newline=None had written, which is precisely the drift §5 forbids and the
    GUI's own report test pins the same way (P23)."""
    from core.artifacts import ArtifactStore, render_lineage
    root, ids = browse_store
    want = render_lineage(ArtifactStore(root), "posterior", ids["posterior"])

    capsys.readouterr()
    assert main(["artifacts", "summary", "posterior", ids["posterior"]]) == 0
    assert want in capsys.readouterr().out, "the one renderer, not a second one"

    out_file = tmp_path / "lineage.txt"
    assert main(["artifacts", "summary", "posterior", ids["posterior"], "--out", str(out_file)]) == 0
    assert out_file.read_bytes() == want.encode("utf-8"), \
        "byte for byte what the browser's Lineage report writes: UTF-8, LF endings"
    assert str(out_file) in capsys.readouterr().out, "the path is named, as report() names an artifact's"


def test_the_artifacts_family_keeps_the_ladders_exit_codes(browse_store, capsys):
    """§4.3, one line per rung: a usage error is 2 (argparse, one rung earlier than every refusal); a
    missing ref, a bad kind and a bad note are 1; an empty listing and a sweep with nothing to remove
    are 0. ``--note`` being REQUIRED is part of this: a note must never be cleared by omission.

    There is no exit 3 anywhere in the tool and this family adds none: a BUG is the existing unhandled
    rung, which sets 1 with a traceback, so 1 covers a refusal and a bug alike (``main``'s own
    docstring: "1 a refusal or a bug") and no task here touches the ladder."""
    root, ids = browse_store
    assert main(["artifacts"]) == 2, "a mode is required"
    assert main(["artifacts", "nosuchmode"]) == 2
    assert main(["artifacts", "show", "prior"]) == 2, "<ref> is required"
    assert main(["artifacts", "note", "prior", ids["prior"]]) == 2, "--note is required"
    assert main(["artifacts", "rm", "prior", "nosuch"]) == 1
    assert main(["artifacts", "summary", "prior", "nosuch"]) == 1

    capsys.readouterr()
    assert main(["artifacts", "sweep", "nosuchkind"]) == 1
    err = capsys.readouterr().err
    lines = [ln for ln in err.splitlines() if ln.startswith("prism artifacts: refused:")]
    assert len(lines) == 1 and "nosuchkind" in lines[0], err
    assert not lines[0].rstrip().endswith(")"), \
        "the artifact key has no flag, so fix_sentence adds nothing and the line ends at the message"


def test_the_artifacts_family_has_all_six_modes_and_still_no_configuration_flags():
    """The closure of B9's list, and the extension of Task 13's own pin to the four modes that write:
    six modes, no configuration flag on any of them, and --note exactly where core/tool/fields.py
    says it is (that table is pinned against these very option strings, both ways)."""
    import argparse
    p = build_parser().subcommands["artifacts"]
    modes = {name: sub for a in p._actions if isinstance(a, argparse._SubParsersAction)
             for name, sub in a.choices.items()}
    assert set(modes) == {"list", "show", "note", "rm", "sweep", "summary"}, sorted(modes)
    for name, parser in [("artifacts", p), *sorted(modes.items())]:
        for flag in ("--bounds", "--model", "--chi", "--chi-k", "--device", "--store-root",
                     "--force", "--accept-truncated"):
            assert flag not in parser._option_string_actions, (name, flag)
    assert "--note" in modes["note"]._option_string_actions
    assert "--out" in modes["summary"]._option_string_actions
```

- [ ] **Step 12: Run them and watch them fail**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_tool.py -q -k "summary or exit_codes or six_modes"`
Expected: FAIL in all three — `assert 2 == 0` on the first `summary` call of the summary test (no
such mode, so argparse exits 2); `assert 2 == 1` in the exit-codes test, on its
`main(["artifacts", "summary", "prior", "nosuch"]) == 1` line, for the same reason (every rung above
it passes already, because `note` and `rm` have landed); and
`AssertionError: ['list', 'note', 'rm', 'show', 'sweep']` on the six-mode set.

- [ ] **Step 13: Add `_summary` and its mode**

`core/tool/browse.py` needs `from pathlib import Path` at the top for this one.

```python
def _summary(args, store) -> int:
    """``summary <kind> <ref> [--out PATH]``: the lineage report -- the artifact, then its parents
    transitively, a parent absent from the store printed as MISSING rather than skipped (design §5).
    A file or stdout, never an eighth store kind (B10).

    newline="\\n" explicitly (P23): write_text's default newline=None translates every "\\n" to
    "\\r\\n" on Windows, and the browser's own Lineage report button writes the same text with
    newline="\\n" -- so without it §5's "the same bytes whichever front end made it" is quietly
    false."""
    from core.artifacts import render_lineage
    text = render_lineage(store, args.kind, args.ref)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8", newline="\n")
        print(f"[prism] lineage report: {args.out}")
    else:
        print(text, end="" if text.endswith("\n") else "\n")
    return 0
```

```python
    summary = modes.add_parser("summary", help="the lineage report: this artifact, then its parents")
    summary.add_argument("kind", metavar="<kind>", help=_KIND)
    summary.add_argument("ref", metavar="<ref>", help=_REF)
    summary.add_argument("--out", default=None, metavar="PATH",
                         help="write the report here instead of to stdout")
    summary.set_defaults(handler=_summary)
```

- [ ] **Step 14: Finish `register`'s metavar and `EPILOG`'s mode list**

The mode metavar becomes the full list, so `--help` shows the family in one line:

```python
    modes = p.add_subparsers(dest="mode", required=True,
                             metavar="{list,show,note,rm,sweep,summary}")
```

and `EPILOG`'s mode block gains the four modes this task added, in the same order:

```
  list [<kind>]         one line per artifact; with no kind, all seven under headings
  show <kind> <ref>     the manifest, then the records of the run that wrote it
  note <kind> <ref> --note TEXT
                        set the note: one line, at most 200 characters; '' clears it
  rm <kind> <ref>       delete one artifact. There is no --force: an artifact anything depends on
                        cannot be deleted at all, so delete its children first
  sweep [<kind>]        remove every directory with no usable manifest; with no kind, all seven
  summary <kind> <ref> [--out PATH]
                        the lineage report: this artifact, then its parents, oldest last
```

- [ ] **Step 15: Run the task's own tests**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_tool.py -q -k "artifacts or sweep or summary or six_modes or exit_codes"`
Expected: PASS

Then the two whole-file pins this task can break — the env/knob scan over every `core/tool` module,
and the FLAG table's closure over every parser (its `--note` entry is now also read off this family):

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_tool.py::test_the_tool_reads_no_environment_and_writes_no_knob "tests/test_refusals.py::test_every_field_key_has_a_flag_and_every_key_literal_under_core_is_registered" -q`
Expected: PASS

- [ ] **Step 16: Commit**

```bash
git add core/tool/browse.py tests/test_tool.py
git commit -m "tool: artifacts note, rm, sweep and summary"
```

### Task 15: a live run is visible in the shell

**Files:**
- Modify: `core/gui/panels/base_panel.py:6-9` (imports), `:56-57` (the new `_RunState`/`RUN_STATE` block goes here), `:264-280` (`_set_busy`) — and **nothing else in this file**: its stale nine/eight sentences (the class docstring at `:62` and `:69`, the layout-persistence comment at `:315` and `:321-322`) belong to Task 25, which owns every docstring correction and runs last.
- Modify: `core/gui/screens/nav_shell.py:1-11` (module docstring + the new `running_banner`), `:21-37` (the header row gains the run slot), `:80-81` (append `set_running` after `_sync_back`, which ends the file at `:81`)
- Modify: `core/gui/design.py:270-279` (the tool-button QSS block; the new rule goes after line 274)
- Modify: `core/gui/main_window.py:6-7` (imports), `:14` and `:22` (two import lines), `:131-136` (the destination map + the run slot's wiring, after `_section_index`), `:267-273` (the three new methods go after `_all_panels`)
  - **This task inserts about thirty lines inside `__init__`, so every line number below it moves down by that much.** That includes the `if box.exec() != QMessageBox.Yes:` branch at `main_window.py:231`, which `tests/conftest.py:174` and `tests/test_worker_dispatch.py:338` cite by number in their prose (Task 9 re-points those two). Task 25 must read the final number off the file rather than trusting either task's brief.
- Test: `tests/test_nav_and_gating.py` (append at the end, after line 2647)

**Interfaces:**
- Consumes: nothing from an earlier task. `core/gui/main_window.py` is also edited by the browser-registration task (design §3.1), so **re-read the file and take your line numbers from what you see** rather than trusting the ranges above if that task has already landed.
- Produces:
  - `core/gui/panels/base_panel.py`: `class _RunState(QObject)` with `changed = Signal(object)`, and the module-level singleton `RUN_STATE = _RunState()`, emitted from `BasePanel._set_busy`.
  - `core/gui/screens/nav_shell.py`: `def running_banner(panel_title: str, seconds: int) -> str`; `NavShell.btn_running` (a flat `QToolButton`, objectName `"navRunning"`); `NavShell.set_running(self, text: "str | None") -> None`.
  - `core/gui/main_window.py`: `self._tab_screens` (a list of `(section name, screen)` for the four tab-hosting sections), `self._panel_home` (`panel -> (section name, tab label, screen index, tab index)`), `self._running_panel`, `self._run_timer`, `MainWindow._on_run_state(self, panel)`, `MainWindow._tick_running(self)`, `MainWindow._go_to_running(self)`. Task 16 consumes `_tab_screens`, `_panel_home` and `_on_run_state`.

**Why this task exists:** One task may run app-wide at a time (`BasePanel._running` is class-level because `redirect_streams` swaps `sys.stdout`/`sys.stderr` process-wide), and today that fact is invisible anywhere but the panel that started it: a run launched on the Posterior tab, seen from Home or from the FDT section, shows nothing at all — every panel's controls are simply dead, with no statement of what is running or for how long. The decomposition's piece-4 row calls this "one-task-at-a-time made visible", and decision **B11** binds it: "A live run is **visible app-wide**: a clickable line in the shell header naming what is running, where and for how long". This task builds the broadcast and the header line; Task 16 adds the section markers B11 also names.

**A note for Task 25, not work for this task.** While editing `_set_busy` you will read past
`base_panel.py`'s stale counts: the class docstring says "Nine of these exist" and "8 of the 9
subclasses", and the layout-persistence comment says "8 of the 9 panels" and "nine independent"
splitters. The true numbers are **ten** `BasePanel` subclasses (Reduction, FDT, CrossVal, Simulate +
the six inference tabs — Config, Prior, Posterior, Validate, Infer, TSNPE) and **nine** of them
override `save_settings` without calling `super()` (every one but `ValidatePanel`). Leave all four
sentences exactly as you find them: Task 25 owns this file's docstrings, together with
`core/gui/panels/inference/base.py` and `main_window.py`'s class docstring, and runs last. No
counting assertion is added for them there or here — `BasePanel.__subclasses__()` is polluted for the
life of the process by the suites' own throwaway `class P(BasePanel)` panels (this task's tests build
one), so a counting pin would be wrong rather than strict.

- [ ] **Step 1: Write the failing test** — the broadcast, the pure clock, and the shell's empty slot

Append to `tests/test_nav_and_gating.py` (after line 2647). All three run offscreen and cost milliseconds.

```python
# ── piece 4: a live run is visible app-wide (spec §6.1, B11) ─────────────────────────────────────
def test_run_state_publishes_the_running_panel_and_none_on_idle():
    """The app-wide broadcast. `_set_busy` already sets the class flag every panel's controls hang
    off; it now also says WHICH panel, so the shell can name it. A module-level singleton, for the
    reason ``BasePanel._running`` is class-level: the stream swap is process-wide, so "what is
    running" is one app-wide fact and the second one belongs beside the first.

    The payload is the panel itself on a run and None on idle -- not a bool, because the listener has
    to name the panel, and not a title, because a BasePanel carries none (MainWindow's destination
    map supplies it). Published LAST in _set_busy, after every panel's controls have reached their
    final state, so a listener cannot see a half-locked window."""
    from core.gui.panels.base_panel import RUN_STATE, BasePanel
    from tests._fixtures import qt_app

    qt_app()

    class P(BasePanel):
        pass

    panel = P()
    seen = []

    def record(p):
        seen.append(p)

    RUN_STATE.changed.connect(record)
    try:
        panel._set_busy(True)
        assert seen == [panel], seen
        assert BasePanel._running, "the class flag and the broadcast must agree"
        panel._set_busy(False)
        assert seen == [panel, None], seen
    finally:
        panel._set_busy(False)
        RUN_STATE.changed.disconnect(record)
    assert not BasePanel._running


def test_the_running_banner_is_a_pure_clock():
    """The header's text comes from a PURE helper, so this asserts every shape without waiting a
    second: m:ss under an hour, h:mm:ss from an hour, 0:00 at the start, and "a task" for a panel
    the window's destination map does not know (it still says something is running). A negative age
    -- a wall-clock jump -- reads as 0 rather than as a minus sign."""
    from core.gui.screens.nav_shell import running_banner

    assert running_banner("Posterior", 0) == "Running: Posterior — 0:00"
    assert running_banner("Posterior", 59) == "Running: Posterior — 0:59"
    assert running_banner("Posterior", 60) == "Running: Posterior — 1:00"
    assert running_banner("Posterior", 247) == "Running: Posterior — 4:07"
    assert running_banner("Posterior", 3599) == "Running: Posterior — 59:59"
    assert running_banner("Posterior", 3600) == "Running: Posterior — 1:00:00"
    assert running_banner("Posterior", 3725) == "Running: Posterior — 1:02:05"
    assert running_banner("Sweep study cross-validation", 5) == \
        "Running: Sweep study cross-validation — 0:05"
    assert running_banner("", 12) == "Running: a task — 0:12"
    assert running_banner("   ", 12) == "Running: a task — 0:12"
    assert running_banner("Posterior", -5) == "Running: Posterior — 0:00"


def test_the_shells_run_slot_starts_empty_and_is_the_only_styled_writer():
    """The slot is built EMPTY and hidden: no icon load, no timer, no store read. The 2026-09-11
    taskbar-icon incident was ~150 ms of layout between the native show and the first idle turn, and
    this button sits on that path (spec §1.2, walkthrough row D16). `set_running` is the only writer:
    a string shows it, None hides and clears it. The objectName is what the global QSS keys on, so
    the two are pinned against each other here -- a renamed button would otherwise silently lose its
    styling."""
    from core.gui import design
    from core.gui.screens.nav_shell import NavShell
    from tests._fixtures import code_only, qt_app

    qt_app()
    nav = NavShell()
    assert nav.btn_running.objectName() == "navRunning"
    assert "QToolButton#navRunning" in design.build_qss(False)
    assert "QToolButton#navRunning" in design.build_qss(True)
    assert nav.btn_running.isHidden(), "the run slot must be hidden until something runs"
    assert nav.btn_running.text() == ""
    assert nav.btn_running.autoRaise(), "a flat tool button, like the other two nav glyphs"
    # no icon font is touched for this button: NavShell loads exactly the two glyphs it always did
    assert code_only(NavShell.__init__).count("apply_icon") == 2

    nav.set_running("Running: Posterior — 0:03")
    assert not nav.btn_running.isHidden() and nav.btn_running.text() == "Running: Posterior — 0:03"
    nav.btn_running.setToolTip("somewhere")
    nav.set_running(None)
    assert nav.btn_running.isHidden()
    assert nav.btn_running.text() == "" and nav.btn_running.toolTip() == ""
```

- [ ] **Step 2: Run it and watch it fail**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest "tests/test_nav_and_gating.py::test_run_state_publishes_the_running_panel_and_none_on_idle" "tests/test_nav_and_gating.py::test_the_running_banner_is_a_pure_clock" "tests/test_nav_and_gating.py::test_the_shells_run_slot_starts_empty_and_is_the_only_styled_writer" -v`

Expected: FAIL, three collection/import errors —
`ImportError: cannot import name 'RUN_STATE' from 'core.gui.panels.base_panel'`,
`ImportError: cannot import name 'running_banner' from 'core.gui.screens.nav_shell'`, and
`AttributeError: 'NavShell' object has no attribute 'btn_running'`.

- [ ] **Step 3: Add the broadcaster to `core/gui/panels/base_panel.py`**

Extend the first import line (line 6) so `QObject` and `Signal` come in:

```python
from PySide6.QtCore import QObject, QThreadPool, QTimer, Signal
```

Insert this block between `_png_fig_sink` (ends line 55) and `class BasePanel` (line 58):

```python
class _RunState(QObject):
    """The app-wide "something is running" broadcast: ``changed`` carries the running BasePanel, or
    None the moment it stops.

    A module-level singleton rather than a signal on BasePanel, for the reason ``BasePanel._running``
    is class-level (see its comment): redirect_streams swaps sys.stdout/stderr PROCESS-WIDE, so only
    one panel may run at a time and "what is running" is ONE app-wide fact -- a second such fact
    belongs beside the first. A per-panel signal would also make every listener subscribe to ten
    senders and re-subscribe to any panel built later.

    Constructed at import, before any QApplication exists: a plain QObject may be, and the emission
    is a direct same-thread call, so no event loop is needed for the shell to be updated.
    """
    changed = Signal(object)


# The one broadcaster. BasePanel._set_busy publishes here and nothing else may; core/gui/main_window.py
# listens and fills the shell's run slot (piece 4, B11).
RUN_STATE = _RunState()
```

- [ ] **Step 4: Publish from `_set_busy`**

At the end of `_set_busy` (after the `for panel in list(BasePanel._instances)` loop that ends at line 279) add:

```python
        # Publish app-wide, LAST: every panel's controls are in their final state before anything
        # listening can look at them. The shell's run slot is filled from this (piece 4, B11), and it
        # is the only thing that knows WHICH panel -- _running is a bare bool.
        RUN_STATE.changed.emit(self if busy else None)
```

- [ ] **Step 5: Add `running_banner` and the run slot to `core/gui/screens/nav_shell.py`**

Extend the module docstring (lines 1-6) with one paragraph:

```
The header row also carries the app-wide RUN SLOT (``btn_running``): a flat button naming what is
running, where and for how long, hidden whenever nothing is (piece 4, B11). MainWindow owns what it
says and when -- this module owns only the widget and the wording.
```

Add the pure helper after the imports (after line 11):

```python
def running_banner(panel_title: str, seconds: int) -> str:
    """"Running: Posterior — 4:07": what is running, and for how long.

    PURE, so a test asserts every shape without waiting for a clock. Under an hour the clock is
    ``m:ss``; from an hour it grows an hours field (``h:mm:ss``) rather than counting to 90 minutes.
    Seconds below zero read as 0 -- a wall-clock jump must not print a negative age -- and a blank
    title reads "a task", which is what a panel MainWindow's destination map does not know shows: the
    banner still says something is running, it just cannot say where (spec §6.1).
    """
    secs = max(0, int(seconds))
    hours, rest = divmod(secs, 3600)
    minutes, sec = divmod(rest, 60)
    clock = f"{hours}:{minutes:02d}:{sec:02d}" if hours else f"{minutes}:{sec:02d}"
    return f"Running: {panel_title.strip() or 'a task'} — {clock}"
```

In `NavShell.__init__`, after the `btn_back` block (ends line 27) and before `header = QVBoxLayout()`:

```python
        # The app-wide run slot. Created EMPTY and hidden: no icon load, no timer, no store read.
        # The 2026-09-11 taskbar-icon incident was ~150 ms of layout between the native show and the
        # first idle turn, and this button is on that path, so it does nothing at launch (spec §1.2).
        # MainWindow fills it from base_panel.RUN_STATE and starts the 1 s clock.
        self.btn_running = QToolButton()
        self.btn_running.setObjectName("navRunning")     # -> QToolButton#navRunning in the global QSS
        self.btn_running.setAutoRaise(True)
        self.btn_running.setVisible(False)
```

and mount it in the header row (lines 35-37) immediately to the right of the title block, before the
stretch, so a growing clock pushes nothing around:

```python
        header_row = QHBoxLayout()
        header_row.addLayout(header)
        header_row.addWidget(self.btn_running, alignment=Qt.AlignVCenter)
        header_row.addStretch(1)
```

Append the writer after `_sync_back` (ends line 81, the file's last line):

```python
    def set_running(self, text: "str | None") -> None:
        """Show the run slot with ``text``, or hide and clear it on None. The ONLY writer of the run
        button's text and visibility; MainWindow owns what it says and when (RUN_STATE plus its 1 s
        timer), and sets the tooltip itself right after -- this clears it only on the way out, so a
        hidden slot cannot keep pointing at a run that has finished."""
        if text:
            self.btn_running.setText(text)
            self.btn_running.setVisible(True)
        else:
            self.btn_running.setText("")
            self.btn_running.setToolTip("")
            self.btn_running.setVisible(False)
```

- [ ] **Step 6: Style it in `core/gui/design.py`**

In the `_QSS` template, immediately after the `QToolButton#navBack, QToolButton#navSettings` rule
(line 274), add:

```
QToolButton#navRunning {
    font-size: ${fs_caption}px; padding: 2px 8px; color: $text_2nd;
    border: 1px solid $mid; border-radius: ${radius_sm}px;
}
QToolButton#navRunning:hover { color: $text; border-color: $accent; }
```

Deliberately the `#helpBadge` pattern (a bordered, secondary-coloured text affordance that takes the
accent on hover) and **not** the `#navBack`/`#navSettings` one: those two set `font-size:
${fs_nav_btn}px` (20 design px) because they render a single icon-font glyph, and this button carries
a sentence.

- [ ] **Step 7: Run the first three tests again**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest "tests/test_nav_and_gating.py::test_run_state_publishes_the_running_panel_and_none_on_idle" "tests/test_nav_and_gating.py::test_the_running_banner_is_a_pure_clock" "tests/test_nav_and_gating.py::test_the_shells_run_slot_starts_empty_and_is_the_only_styled_writer" -v`

Expected: PASS (3 passed).

- [ ] **Step 8: Write the failing window tests**

Append to `tests/test_nav_and_gating.py`, after the three tests from Step 1.

```python
def test_the_window_fills_the_run_slot_while_a_panel_runs_and_clears_it_after():
    """The window's half of B11: the header says what is running and for how long, and stops saying
    it the moment the run ends. And LAUNCH IS QUIET -- the slot is empty, no clock is ticking and no
    panel is named until something actually runs. The QTimer is constructed in __init__ and never
    started there: an unstarted QTimer registers nothing with the OS, which is what keeps the 150 ms
    of layout the 2026-09-11 taskbar-icon incident was made of off the launch path (spec §1.2)."""
    from core.gui.main_window import MainWindow
    from core.gui.panels.crossval_panel import CrossValPanel
    from core.gui.screens.nav_shell import running_banner
    from tests._fixtures import qt_app

    qt_app()
    w = MainWindow()
    assert w.nav.btn_running.isHidden() and w.nav.btn_running.text() == ""
    assert not w._run_timer.isActive(), "a clock is ticking before anything has run"
    assert w._running_panel is None

    label = "Sweep study cross-validation"
    panel = w.panel(CrossValPanel)
    panel._set_busy(True)
    try:
        assert not w.nav.btn_running.isHidden()
        # tolerant by one tick: this is real elapsed time, not a stub
        assert w.nav.btn_running.text() in (running_banner(label, 0), running_banner(label, 1)), \
            w.nav.btn_running.text()
        assert w.nav.btn_running.toolTip() == f"FDT Analysis → {label}: go to it"
        assert w._running_panel is panel
        assert w._run_timer.isActive() and w._run_timer.interval() == 1000
    finally:
        panel._set_busy(False)
    assert w.nav.btn_running.isHidden() and w.nav.btn_running.text() == ""
    assert w.nav.btn_running.toolTip() == "" and w._running_panel is None
    assert not w._run_timer.isActive(), "the clock must stop when the run does"


def test_clicking_the_run_slot_opens_the_running_panel_and_an_unknown_one_goes_nowhere():
    """The banner is clickable and lands on the panel that is running -- the section screen AND its
    tab, because a run on the second tab of a section is not found by arriving at the first.

    The destination map is built ONCE, in __init__, off the screens' own QTabWidgets, so a renamed
    tab cannot leave the banner naming a tab that is gone. A panel the map does not know -- the model
    builder is not a BasePanel, a future screen might not be mounted in a tab -- shows the banner with
    no destination rather than raising a KeyError inside a signal handler."""
    from core.gui.main_window import MainWindow
    from core.gui.panels.base_panel import BasePanel
    from core.gui.panels.crossval_panel import CrossValPanel
    from tests._fixtures import qt_app

    qt_app()
    w = MainWindow()
    panel = w.panel(CrossValPanel)
    assert w._panel_home[panel] == ("FDT Analysis", "Sweep study cross-validation",
                                    w._section_index["FDT Analysis"], 1)
    assert w._panel_home[w.inference_screen.posterior_panel] == (
        "Parameter Inference", "Posterior", w._section_index["Parameter Inference"], 2)

    w.nav.go_home()
    w.fdt_screen.tabs.setCurrentIndex(0)
    panel._set_busy(True)
    try:
        w.nav.btn_running.click()
        assert w.nav.stack.currentIndex() == w._section_index["FDT Analysis"]
        assert w.fdt_screen.tabs.currentIndex() == 1, "the click must land on the running TAB"
    finally:
        panel._set_busy(False)

    class P(BasePanel):
        pass

    orphan = P()
    assert orphan not in w._panel_home
    w.nav.go_home()
    orphan._set_busy(True)
    try:
        assert not w.nav.btn_running.isHidden()
        assert w.nav.btn_running.text().startswith("Running: a task — ")
        assert w.nav.btn_running.toolTip() == ""
        w.nav.btn_running.click()
        assert w.nav.stack.currentIndex() == 0, "an unknown panel must not navigate anywhere"
    finally:
        orphan._set_busy(False)
```

- [ ] **Step 9: Run them and watch them fail**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest "tests/test_nav_and_gating.py::test_the_window_fills_the_run_slot_while_a_panel_runs_and_clears_it_after" "tests/test_nav_and_gating.py::test_clicking_the_run_slot_opens_the_running_panel_and_an_unknown_one_goes_nowhere" -v`

Expected: FAIL, both with `AttributeError: 'MainWindow' object has no attribute '_run_timer'` /
`'_panel_home'`.

- [ ] **Step 10: Wire `core/gui/main_window.py` — the imports and the destination map**

Add `import time` as the first import, and `QTimer`:

```python
import time

from PySide6.QtCore import QTimer
from PySide6.QtGui import QActionGroup
from PySide6.QtWidgets import QMainWindow, QMenu, QMessageBox
```

Change line 14 to bring in the broadcaster, and line 22 to bring in the banner helper:

```python
from .panels.base_panel import RUN_STATE, BasePanel
...
from .screens.nav_shell import NavShell, running_banner
```

In `__init__`, immediately after `home.navigate.connect(...)` (line 134) and before
`self._build_settings_menu(current_mode)`:

```python
        # Where each panel LIVES. A BasePanel carries no title of its own, so this is what lets the
        # header's run slot say "Posterior" and jump there. Built ONCE, off the screens' own tab
        # widgets -- the same QTabWidgets the screens were built from, and the only thing MainWindow
        # already has that pairs a panel with a label -- so a renamed tab cannot leave the banner
        # naming a tab that is gone. Both screen kinds expose `.tabs` and `panels()` in tab order, so
        # one loop covers the generic sections and the inference screen alike. A panel that is NOT in
        # the map (the model builder, which is not a BasePanel, or a future screen) gets the banner
        # with no destination rather than a KeyError inside a signal handler.
        self._tab_screens = [(name, self.nav.stack.widget(self._section_index[name]))
                             for name in ("Reduction Map", "FDT Analysis", "Parameter Inference",
                                          "Simulate")]
        self._panel_home = {}
        for name, screen in self._tab_screens:
            for tab_index in range(screen.tabs.count()):
                self._panel_home[screen.tabs.widget(tab_index)] = (
                    name, screen.tabs.tabText(tab_index), self._section_index[name], tab_index)

        # The run slot, EMPTY at launch (spec §1.2): the timer is constructed here but never started,
        # and an unstarted QTimer registers nothing with the OS, so nothing on the launch path ticks.
        self._running_panel = None
        self._run_title = ""
        self._run_started = 0.0
        self._run_timer = QTimer(self)
        self._run_timer.setInterval(1000)
        self._run_timer.timeout.connect(self._tick_running)
        self.nav.btn_running.clicked.connect(self._go_to_running)
        RUN_STATE.changed.connect(self._on_run_state)
```

- [ ] **Step 11: Add the three methods**

After `_all_panels` (ends line 269) and before `panel(self, cls)`:

```python
    # ── the live run, visible app-wide (piece 4, B11) ─────────────────────────
    def _on_run_state(self, panel) -> None:
        """A run started (``panel``) or ended (None): fill or clear the shell's run slot.

        Connected to the module-level RUN_STATE in base_panel.py, which _set_busy publishes on. Qt
        drops the connection when this window is destroyed, so a window a test threw away is never
        called for a later run. The title is looked up ONCE per run, not per tick.
        """
        self._running_panel = panel
        if panel is None:
            self._run_timer.stop()
            self.nav.set_running(None)
            return
        where = self._panel_home.get(panel)
        self._run_title = where[1] if where is not None else ""
        self._run_started = time.monotonic()
        self._tick_running()                       # paint immediately, not a second from now
        self.nav.btn_running.setToolTip(
            f"{where[0]} → {where[1]}: go to it" if where is not None else "")
        self._run_timer.start()

    def _tick_running(self) -> None:
        """Re-render the elapsed time. One second's work: a divmod and a setText."""
        self.nav.set_running(
            running_banner(self._run_title, int(time.monotonic() - self._run_started)))

    def _go_to_running(self) -> None:
        """Click the run slot: show the panel that is running -- its screen AND its tab, since a run
        on a section's second tab is not found by arriving at the first. A panel with no known home
        does nothing: the banner still says what is running, it just has nowhere to send you."""
        where = self._panel_home.get(self._running_panel)
        if where is None:
            return
        _name, _label, screen_index, tab_index = where
        self.nav.go_to(screen_index)
        tabs = getattr(self.nav.stack.widget(screen_index), "tabs", None)
        if tabs is not None:
            tabs.setCurrentIndex(tab_index)
```

- [ ] **Step 12: Run the task's own tests**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_nav_and_gating.py tests/test_worker_dispatch.py tests/test_simulate.py -q`

Expected: PASS. (`test_worker_dispatch.py` exercises `_set_busy` through `dispatch` on panels that
belong to no window, and `test_simulate.py:194-196` builds a `MainWindow` and navigates by
`_section_index` — both are the paths this task changes.)

- [ ] **Step 13: Commit**

```bash
git add core/gui/panels/base_panel.py core/gui/screens/nav_shell.py core/gui/design.py core/gui/main_window.py tests/test_nav_and_gating.py
git commit -m "gui: show the live run in the shell header"
```

---

### Task 16: the section markers

**Files:**
- Modify: `core/gui/screens/nav_shell.py:10-11` (`RUNNING_MARK` beside the imports) and the new `mark_tabs` beside `running_banner`
- Modify: `core/gui/screens/home_screen.py:8-12` (imports), `:46-58` (the tile loop keeps its buttons, plus `set_running_section`)
- Modify: `core/gui/screens/section_screen.py:7-9` (imports), `:24-29` (the labels are kept), `:41-43` (plus `set_running_tab`)
- Modify: `core/gui/screens/inference_screen.py:14-20` (imports), `:51-54` (the labels are kept), `:85-87` (plus `set_running_tab`). **Not** its module docstring's stale tab count at `:1-4`: Task 18 owns that sentence and rewrites it once (Q3, P38's one-owner rule)
- Modify: `core/gui/main_window.py:101-103` and `:134` (the Home screen becomes `self.home_screen`), `_on_run_state` (one line), plus `_mark_running_section` after `_go_to_running`. All three edits are line-neutral or below `_delete_user_model`, so the `main_window.py:231` citation in `tests/test_worker_dispatch.py:338` moves no further here — Task 15's insert into `__init__` already moved it, and Task 9 re-points it (the sibling sentence in `tests/conftest.py:174` is Task 11's, which deletes the number rather than re-pointing it).
- Test: `tests/test_nav_and_gating.py` (append after Task 15's tests)

**Interfaces:**
- Consumes, from Task 15: `MainWindow._panel_home` (`panel -> (section name, tab label, screen index, tab index)`), `MainWindow._tab_screens` (`[(section name, screen)]`), and `MainWindow._on_run_state(self, panel)` — the one place that already learns of a run starting and ending.
- Produces: `core/gui/screens/nav_shell.py`: `RUNNING_MARK` (the marker character) and `def mark_tabs(tabs, labels, running) -> None`. `HomeScreen.tiles` (`dict[str, QPushButton]`) and `HomeScreen.set_running_section(self, name: "str | None") -> None`. `SectionScreen.set_running_tab(self, index: "int | None") -> None` and `InferenceScreen.set_running_tab(self, index: "int | None") -> None`. `MainWindow.home_screen` and `MainWindow._mark_running_section(self, panel) -> None`.

**Why this task exists:** B11 asks for more than the header line: "plus a marker on the running section's Home tile and its tab". Without it the header tells you a run is live but not where to look for it once you have navigated away — and navigating away is the point, since B11 also insists "navigation stays free — you must be able to look at another tab while a twenty-minute train runs". The marker is the only thing on Home and on a tab bar that survives that navigation. Nothing is disabled by this task: the app-wide control lock in `BasePanel._set_busy` already stands, and adding a second, navigational lock is explicitly refused by §6.1.

**The marker is a plain character, not an icon-font glyph**, although §6.1 says "a leading glyph from the bundled icon font". Three things in the code rule that out, and all three are checkable:

1. `core/gui/assets/icons/build_prism_icons.py:22-23` defines exactly five codepoints (U+E000–U+E004: back, settings, refresh, help, close) and `tests/test_user_models.py:856-860` fails any `icons.NAMES` entry whose codepoint the bundled `.ttf` does not define. A sixth name means authoring a glyph procedurally and regenerating a committed binary — a font change inside a GUI task.
2. `icons.glyph` returns the private-use codepoint whenever the **font** loaded, not whenever the **glyph** exists (`core/gui/icons.py:56-59`), so a name added without the glyph renders as an empty box on every screen rather than failing anywhere.
3. `icons.apply_icon` sets the **widget's** font family (`core/gui/icons.py:62-73`). A `QTabBar` has one font for all its tabs, so no per-tab family can be set, and switching the whole tab bar to the icon family would render every label in it as boxes.

The repo's own idiom for marking text is a plain character: `core/gui/widgets/log_pane.py:8` is
`_PREFIX = {"warning": "⚠ ", "error": "✖ "}`. `RUNNING_MARK = "●"` follows it.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_nav_and_gating.py`, after Task 15's tests.

```python
def test_the_running_section_tile_and_tab_carry_a_marker_and_nothing_is_disabled():
    """B11's markers. The running section's Home tile takes a suffix and the running tab a leading
    mark, both cleared the moment the run ends -- and both rewritten from the labels the screens were
    BUILT with, so marking is idempotent and a cleared label is byte-identical to the original (the
    tab titles are read back verbatim by test_every_field_key_has_a_window_control..., which asserts
    ["Config", "Prior", "Posterior", "Validate", "Infer", "TSNPE"]).

    And nothing is disabled: the app-wide controls lock stands on its own, the Home tiles stay live,
    every tab stays selectable, and the marker survives refresh_gates -- which rewrites tab tooltips
    and enabled states after every stage, and would be running while a marked run is live. You must
    be able to look at another tab while a twenty-minute train runs (spec §6.1)."""
    from core.gui.main_window import MainWindow
    from core.gui.panels.base_panel import BasePanel
    from core.gui.panels.crossval_panel import CrossValPanel
    from core.gui.screens.nav_shell import RUNNING_MARK
    from tests._fixtures import qt_app

    qt_app()
    w = MainWindow()
    fdt, inf = w.fdt_screen.tabs, w.inference_screen.tabs
    assert [fdt.tabText(i) for i in range(fdt.count())] == \
        ["FDT analysis", "Sweep study cross-validation"]
    assert w.home_screen.tiles["FDT Analysis"].text() == "FDT Analysis"

    panel = w.panel(CrossValPanel)
    panel._set_busy(True)
    try:
        assert fdt.tabText(1) == f"{RUNNING_MARK} Sweep study cross-validation"
        assert fdt.tabText(0) == "FDT analysis", "only the running tab is marked"
        assert w.home_screen.tiles["FDT Analysis"].text() == f"FDT Analysis  {RUNNING_MARK}"
        assert w.home_screen.tiles["Simulate"].text() == "Simulate"
        assert [inf.tabText(i) for i in range(inf.count())] == \
            ["Config", "Prior", "Posterior", "Validate", "Infer", "TSNPE"]
        # nothing is disabled and navigation stays free
        assert fdt.isTabEnabled(0) and w.home_screen.tiles["Simulate"].isEnabled()
        w.nav.go_to(w._section_index["Simulate"])
        assert w.nav.stack.currentIndex() == w._section_index["Simulate"]
        fdt.setCurrentIndex(0)
        assert fdt.currentIndex() == 0, "a marked section's other tab must stay selectable"
    finally:
        panel._set_busy(False)
    assert fdt.tabText(1) == "Sweep study cross-validation"
    assert w.home_screen.tiles["FDT Analysis"].text() == "FDT Analysis"

    # an inference tab, and the marker survives the gate refresh that follows every stage
    post = w.inference_screen.posterior_panel
    post._set_busy(True)
    try:
        assert inf.tabText(2) == f"{RUNNING_MARK} Posterior"
        assert w.home_screen.tiles["Parameter Inference"].text() == \
            f"Parameter Inference  {RUNNING_MARK}"
        assert fdt.tabText(1) == "Sweep study cross-validation", "another section keeps no mark"
        w.inference_screen.refresh_gates()
        assert inf.tabText(2) == f"{RUNNING_MARK} Posterior", "refresh_gates wiped the marker"
    finally:
        post._set_busy(False)
    assert [inf.tabText(i) for i in range(inf.count())] == \
        ["Config", "Prior", "Posterior", "Validate", "Infer", "TSNPE"]

    # a panel with no known home marks nothing, and clears whatever was marked
    class P(BasePanel):
        pass

    orphan = P()
    orphan._set_busy(True)
    try:
        assert all(btn.text() == name for name, btn in w.home_screen.tiles.items())
        assert fdt.tabText(1) == "Sweep study cross-validation"
    finally:
        orphan._set_busy(False)
```

- [ ] **Step 2: Run it and watch it fail**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest "tests/test_nav_and_gating.py::test_the_running_section_tile_and_tab_carry_a_marker_and_nothing_is_disabled" -v`

Expected: FAIL, `ImportError: cannot import name 'RUNNING_MARK' from 'core.gui.screens.nav_shell'`.

- [ ] **Step 3: Add the marker and the tab writer to `core/gui/screens/nav_shell.py`**

After the imports (line 11), above `running_banner`:

```python
# The app-wide "a run is live" marker. A PLAIN CHARACTER, not an icons.NAMES entry: the bundled icon
# font defines U+E000..U+E004 only and tests/test_user_models.py pins NAMES against its cmap, so a
# sixth name renders an empty box until the .ttf is regenerated; icons.glyph returns the codepoint
# whenever the FONT loaded, not whenever the glyph exists; and apply_icon sets the WIDGET's font
# family, which a QTabBar cannot do for one tab. LogPane._PREFIX marks its lines the same way.
RUNNING_MARK = "●"
```

and after `running_banner`:

```python
def mark_tabs(tabs, labels, running) -> None:
    """Rewrite every tab's text from ``labels``, prefixing RUNNING_MARK on index ``running``; None
    clears every mark.

    Rewritten from the stored labels rather than edited in place, so marking is IDEMPOTENT and a
    cleared tab is byte-identical to the label it was built with -- the tab titles are read back
    verbatim by tests/test_nav_and_gating.py's control-table pin. Shared by the two screen kinds that
    host tabs (SectionScreen, InferenceScreen), which have no common base but QWidget.
    """
    for index, base in enumerate(labels):
        text = f"{RUNNING_MARK} {base}" if index == running else base
        if tabs.tabText(index) != text:
            tabs.setTabText(index, text)
```

- [ ] **Step 4: Keep the Home tiles reachable and markable (`core/gui/screens/home_screen.py`)**

Add the import (after line 9):

```python
from .nav_shell import RUNNING_MARK
```

(No cycle: `nav_shell` imports only `..icons` and `..widgets.anim`.)

In `__init__`, replace the tile loop (lines 46-55) with one that keeps its buttons — **leave
`SECTIONS` exactly as you find it**, an earlier task may already have added `"Artifacts"` to it:

```python
        # Kept on self: the running section's tile takes a marker (piece 4, B11), and a tile that was
        # only ever added to a layout cannot be found again.
        self.tiles: dict = {}
        for name in SECTIONS:
            btn = QPushButton(name)
            btn.setMinimumWidth(260)
            btn.setMinimumHeight(44)
            if name in live:
                btn.setProperty("accent", True)         # live section tiles are primary (Fluent accent)
                btn.clicked.connect(lambda _=False, n=name: self.navigate.emit(n))
            else:
                btn.setToolTip("Coming soon")           # a name with no screen: clickable, no target
            self.tiles[name] = btn
            layout.addWidget(btn, alignment=Qt.AlignCenter)
```

and append after `_refresh_greeting` (line 61):

```python
    def set_running_section(self, name: "str | None") -> None:
        """Mark one section's tile as the one with a live run; None (or an unknown name) clears every
        mark. A SUFFIX on the tile's own label, rewritten from the section name rather than edited in
        place, so marking is idempotent and a cleared tile reads exactly as it was built. Nothing is
        disabled -- every tile stays clickable while a run is live (piece 4, B11)."""
        running = name if name in self.tiles else None
        for section, btn in self.tiles.items():
            text = f"{section}  {RUNNING_MARK}" if section == running else section
            if btn.text() != text:
                btn.setText(text)
```

- [ ] **Step 5: Let the two tab-hosting screens be marked**

`core/gui/screens/section_screen.py` — import (after line 9) `from .nav_shell import mark_tabs`,
keep the labels in the `addTab` loop (lines 24-26):

```python
        self.tabs = QTabWidget()
        self._tab_labels: list = []        # the labels as built; mark_tabs rewrites from these
        for label, panel in tabs:
            self.tabs.addTab(panel, label)
            self._tab_labels.append(label)
```

and append after `panels()` (ends line 43):

```python
    def set_running_tab(self, index: "int | None") -> None:
        """Mark the tab at ``index`` as the one with a live run; None clears every mark (B11)."""
        mark_tabs(self.tabs, self._tab_labels, index)
```

`core/gui/screens/inference_screen.py` — import (after line 19) `from .nav_shell import mark_tabs`,
keep the labels in the `addTab` loop (lines 51-54):

```python
        self._tab_labels: list = []        # the labels as built; mark_tabs rewrites from these
        for label, panel in (("Config", self.config_panel), ("Prior", self.prior_panel),
                             ("Posterior", self.posterior_panel), ("Validate", self.validate_panel),
                             ("Infer", self.infer_panel), ("TSNPE", self.tsnpe_panel)):
            self.tabs.addTab(panel, label)
            self._tab_labels.append(label)
```

and append after `panels()` (ends line 87):

```python
    def set_running_tab(self, index: "int | None") -> None:
        """Mark the tab at ``index`` as the one with a live run; None clears every mark (B11). Tab
        TEXT only: refresh_gates owns each tab's enabled state and tooltip and runs after every
        stage, so a marker written there would be wiped seconds later."""
        mark_tabs(self.tabs, self._tab_labels, index)
```

- [ ] **Step 6: Drive the markers from `core/gui/main_window.py`**

The Home screen is a local today (line 101) and the marker needs it. Rename it in place — it is
referenced twice in `__init__` (lines 103 and 134) and nowhere else:

```python
        self.home_screen = HomeScreen(
            live_sections={"Reduction Map", "FDT Analysis", "Parameter Inference", "Simulate",
                           "Artifacts"})
        self.nav.add_screen(self.home_screen)                        # index 0 -- Home
```

...and

```python
        self.home_screen.navigate.connect(lambda name: self.nav.go_to(self._section_index[name]))
```

The set has **five** names because Task 9 added `"Artifacts"` (Q12). This step changes only the
BINDING — the local `home` becomes `self.home_screen`; whatever `live_sections` holds when you get
here is what it must still hold when you leave, so copy it off the file rather than off this block if
the two ever differ.

Add one line to `_on_run_state`, immediately after `self._running_panel = panel`, so both branches —
a run starting and a run ending — go through it:

```python
        self._running_panel = panel
        self._mark_running_section(panel)
```

and add the method after `_go_to_running`:

```python
    def _mark_running_section(self, panel) -> None:
        """Put the running marker on the running section's Home tile and its tab, and clear every
        other one. ``panel`` None -- or a panel with no known home -- clears the lot, so a marker can
        never outlive the run that set it.

        Every screen is rewritten on every change rather than remembering what was marked: four
        screens is nothing, and the bookkeeping version is what leaves a stale marker behind when a
        run ends on a window that was navigated in between. Nothing is DISABLED here: the app-wide
        control lock (BasePanel._set_busy) already stands and navigation stays free -- you must be
        able to look at another tab while a twenty-minute train runs (spec §6.1, B11).
        """
        where = self._panel_home.get(panel)
        name = where[0] if where is not None else None
        tab_index = where[3] if where is not None else None
        self.home_screen.set_running_section(name)
        for section, screen in self._tab_screens:
            screen.set_running_tab(tab_index if section == name else None)
```

- [ ] **Step 7: Run the task's own tests**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_nav_and_gating.py tests/test_simulate.py tests/test_worker_dispatch.py -q`

Expected: PASS. (`test_every_field_key_has_a_window_control_and_every_control_names_a_real_one` and
the tab-title pin beside it read the inference tab titles back verbatim, and `test_simulate.py`'s
`MainWindow` build is on the same path; all of them are on the paths this task changes. Named rather
than numbered: Tasks 1, 2 and 15 have all appended to `tests/test_nav_and_gating.py` by now.)

- [ ] **Step 8: Commit**

```bash
git add core/gui/screens/nav_shell.py core/gui/screens/home_screen.py core/gui/screens/section_screen.py core/gui/screens/inference_screen.py core/gui/main_window.py tests/test_nav_and_gating.py
git commit -m "gui: mark the running section's Home tile and tab"
```

### Task 17: what the session holds

**Files:**
- Modify: `core/gui/screens/inference_screen.py:85-87` (add a module-level helper above the class and
  `session_contents` immediately after `panels()`)
- Modify: `core/gui/panels/inference/config_tab.py:90-91` (the derived line inside the Config group
  box) and `core/gui/panels/inference/config_tab.py:111-112` (the end of `__init__`), plus two new
  methods after `_build_config` (which ends at `core/gui/panels/inference/config_tab.py:296`)
- Test: `tests/test_nav_and_gating.py` (append at the end of the file)

Task 16 has already inserted into `core/gui/screens/inference_screen.py` (an import line and a
`set_running_tab` after `panels()`); **re-read that file and take your line numbers from what you
see** rather than trusting the ranges above. The quoted text is the anchor.

**Interfaces:**
- Consumes: nothing from an earlier task. It uses only what is already on disk:
  `core.gui.session.SbiSession` (fields `draft, cfg, inf_prior, posterior, observation`, declared at
  `core/gui/session.py:40-57`), and the read-only derived-label factory
  `_TrainingBudgetMixin._derived_label() -> QLabel` (`core/gui/panels/inference/base.py:51-75`),
  which `config_tab.py:105` already calls for its VRAM note.
- Produces:
  - `InferenceScreen.session_contents(self) -> list[str]`
  - `core.gui.screens.inference_screen._held_ref(wrapper) -> str` (module-level, private)
  - `ConfigPanel.session_line` — the `QLabel` holding the derived line
  - `ConfigPanel.refresh_local_gates(self) -> None` (new override; the base is a no-op at
    `core/gui/panels/base_panel.py:295-298`)
  - `ConfigPanel._sync_session_line(self) -> None`

**Why this task exists:** `InferenceScreen.new_draft` replaces the whole `SbiSession` — the built
config, the prior, the posterior and the recorded observation — and nothing anywhere on screen says
what is in there. The spec's decision **B12** requires that "**Apply** shows a permanent line naming
what the session holds, and **confirms** before discarding a non-empty session"; §6.2 makes
`session_contents()` the one describer that both the line and (Task 18) the confirmation read, so the
two can never disagree about what is at stake. This task builds the describer and the line; Task 18
builds the confirmation on top of it.

- [ ] **Step 1: Write the failing test**

Append this to the end of `tests/test_nav_and_gating.py`. It uses that file's own module-level stubs
`_prior_stub` (line 45), `_posterior_stub` (line 51) and `_spont_cfg` (line 98), and the shared
`qt_app()` from `tests/_fixtures.py` — never the unused `qapp` fixture.

```python
def test_the_config_tab_says_what_the_session_holds():
    """B12's first half (spec §6.2). ONE describer, ``InferenceScreen.session_contents()``, answers
    "what would a new draft throw away" -- in pipeline order, in plain phrases with no code
    identifiers in them -- and the Config tab shows it as a permanent read-only line refreshed from
    ``refresh_local_gates``, which ``refresh_gates`` already calls on every panel after every stage.

    THE DRAFT IS NOT IN THE LIST, deliberately: ``new_draft`` replaces it and replacing it is the
    whole point of pressing Apply, so naming it would question every model change made on purpose.
    What Apply discards that nobody asked it to is the work DOWNSTREAM of the draft.

    The last leg is the one that keeps the line safe: the gate tests put a bare ``object()`` on
    ``cfg``, ``inf_prior`` and ``posterior``, and a derived status line must never be able to raise
    into ``refresh_gates()`` and take the tab down with it (the rule ``_sync_budget`` states).
    """
    from core.gui.screens.inference_screen import InferenceScreen
    from core.gui.session import SbiSession
    from tests._fixtures import qt_app

    qt_app()
    inf = InferenceScreen()
    cfgp = inf.config_panel

    # (a) nothing held, and a DRAFT ALONE is still nothing held
    assert inf.session_contents() == []
    inf.session = SbiSession(draft=object())
    inf.refresh_gates()
    assert inf.session_contents() == [], "replacing a draft is what Apply is for; it is not a loss"
    assert "nothing yet" in cfgp.session_line.text(), cfgp.session_line.text()

    # (b) one phrase per stage, in pipeline order, named the way the panels' own log lines name them
    inf.session = SbiSession(draft=object(), cfg=_spont_cfg(),
                             inf_prior=_prior_stub(id_="20260914T100000", name="p_master"),
                             posterior=_posterior_stub(id_="20260915T090000"))
    inf.session.observation = type("Obs", (), {"id": "20260916T120000", "name": "obs1"})()
    held = inf.session_contents()
    assert held == ["the built config for NADROWSKI, spontaneous observations",
                    "the prior 'p_master'",
                    "the posterior (unnamed, id 20260915T090000)",
                    "the observation 'obs1'"], held

    # (c) the line names every phrase and says the artifacts survive on disk
    inf.refresh_gates()
    line = cfgp.session_line.text()
    for phrase in held:
        assert phrase in line, (phrase, line)
    assert "stays on disk" in line, line

    # (d) a stand-in carrying neither a name nor an id: described, never raised
    inf.session = SbiSession(cfg=object(), inf_prior=object(), posterior=object())
    inf.refresh_gates()
    assert inf.session_contents() == ["the built config for an unnamed model",
                                      "the prior (unidentified)",
                                      "the posterior (unidentified)"], inf.session_contents()
    assert "(unidentified)" in cfgp.session_line.text(), cfgp.session_line.text()
```

- [ ] **Step 2: Run it and watch it fail**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_nav_and_gating.py::test_the_config_tab_says_what_the_session_holds -v`

Expected: FAIL, `AttributeError: 'InferenceScreen' object has no attribute 'session_contents'` on the
first assertion.

- [ ] **Step 3: Add the `_held_ref` helper to `inference_screen.py`**

In `core/gui/screens/inference_screen.py`, insert this between the imports (which end at line 19 with
`from ..widgets.anim import crossfade_tab`) and `class InferenceScreen` (line 22) — i.e. in the blank
lines at 20-21:

```python
def _held_ref(wrapper) -> str:
    """How ``session_contents`` names one held artifact: its name in quotes, or ``(unnamed, id
    <id>)`` -- the wording the Prior, Posterior and Infer tabs' own "ready" lines already use, so the
    log and the line agree.

    ``(unidentified)`` for a stand-in carrying neither. That branch is not defensive padding: the
    gate tests put a bare ``object()`` on every session field, and everything this function feeds is
    a derived status line, which may never raise.
    """
    name = getattr(wrapper, "name", "") or ""
    if name:
        return f"'{name}'"
    ref = getattr(wrapper, "id", "") or ""
    return f"(unnamed, id {ref})" if ref else "(unidentified)"
```

- [ ] **Step 4: Add `session_contents` to `InferenceScreen`**

In the same file, insert this immediately after `panels()` (which ends at line 87 with
`self.validate_panel, self.infer_panel, self.tsnpe_panel]`) and before `def new_draft` (line 89):

```python
    def session_contents(self) -> list[str]:
        """Plain phrases for what the session holds -- the config, the prior, the posterior and the
        observation -- in pipeline order; empty when a ``new_draft`` would lose nothing.

        THE DRAFT IS DELIBERATELY NOT IN THE LIST. ``new_draft`` replaces the draft, and replacing it
        is the entire point of pressing Apply: a list that named it would make every deliberate model
        change look like a loss, and the guard built on this list (B12) would then interrupt every
        one of them. What Apply discards that nobody asked it to discard is the work DOWNSTREAM of
        the draft -- the built config, the prior, the posterior and the recorded observation -- which
        is exactly ``SbiSession.reset_downstream``'s subject and exactly this list.

        NOTHING HERE MAY RAISE. Every field is read through ``getattr``: the gate tests put a bare
        ``object()`` on ``cfg``, ``inf_prior`` and ``posterior``, and this feeds a derived status line
        on the Config tab, which must never be able to take a tab down through ``refresh_gates()``
        (the rule ``_TrainingBudgetMixin._sync_budget`` states for its own three lines).

        The phrases carry no code identifiers and name no control: the Config tab's line and the
        confirmation both quote them verbatim, and each front end says where to look for itself.
        """
        s = self.session
        held: list[str] = []
        if s.cfg is not None:
            model = getattr(s.cfg, "model", "") or "an unnamed model"
            mode = getattr(s.cfg, "observation_mode", "") or ""
            held.append(f"the built config for {model}"
                        + (f", {mode} observations" if mode else ""))
        if s.inf_prior is not None:
            held.append(f"the prior {_held_ref(s.inf_prior)}")
        if s.posterior is not None:
            held.append(f"the posterior {_held_ref(s.posterior)}")
        if s.observation is not None:
            held.append(f"the observation {_held_ref(s.observation)}")
        return held
```

- [ ] **Step 5: Add the derived line to the Config tab's form**

In `core/gui/panels/inference/config_tab.py`, replace lines 90-91:

```python
        form.addRow(self.btn_config)
        self.controls_layout.addWidget(box)
```

with:

```python
        form.addRow(self.btn_config)
        # WHAT APPLY WOULD REPLACE, stated permanently and not only in the dialog (B12). The
        # confirmation Apply raises is answered and gone; this line is what lets someone see, before
        # they reach for the button, that this session is holding a prior and a posterior. Same
        # word-wrapped PlainText label as every other derived line in this section, for the same
        # reason: it carries a generated string that can be long, and an unwrapped label widens the
        # whole controls column.
        self.session_line = _TrainingBudgetMixin._derived_label()
        form.addRow("", self.session_line)
        self.controls_layout.addWidget(box)
```

- [ ] **Step 6: Add `_sync_session_line` and `refresh_local_gates` to `ConfigPanel`**

In the same file, insert both methods immediately after `_build_config` (which ends at line 296 with
`"warning")`) and before `def save_settings` (line 298):

```python
    def _sync_session_line(self) -> None:
        """The one derived line on this tab, and the only one about the SESSION rather than a box.

        Refreshed from ``refresh_local_gates``, which ``InferenceScreen.refresh_gates`` calls on every
        panel after every stage, so the line can never be behind the session it describes.

        WRAPPED, like every derived line in this section: a status line must never be able to raise
        into ``refresh_gates()``. Two real ways it could. This panel is constructed with
        ``screen=None`` by tests/test_settings_persistence.py, so there is no session to read at all;
        and ``BasePanel._instances`` is a process-wide WeakSet, so ``set_controls_enabled`` can reach
        a panel whose screen's C++ object has already gone.
        """
        screen = self._screen
        try:
            held = [] if screen is None else screen.session_contents()
        except Exception as e:                  # noqa: BLE001 -- never break the tab over a label
            self.session_line.setText(f"Session summary unavailable: {type(e).__name__}: {e}")
            return
        if not held:
            self.session_line.setText(
                "This session holds nothing yet, so applying a model discards nothing.")
            return
        self.session_line.setText(
            "This session holds " + "; ".join(held)
            + ".\nApplying a model starts a new session and releases them. Each one stays on disk "
              "and can be selected again in the pickers.")

    def refresh_local_gates(self):
        """This tab's derived line. The other five tabs gate BUTTONS here; Apply is gated by the
        model combo instead (``_on_model_changed`` disables it for an SBI-ineligible user model), so
        the session line is the only thing left to re-derive."""
        self._sync_session_line()
```

- [ ] **Step 7: Draw the line once at construction**

Still in `config_tab.py`, replace lines 111-112:

```python
        self.restore_settings(settings.settings())
        self._sync_chi_enabled()
```

with:

```python
        self.restore_settings(settings.settings())
        self._sync_chi_enabled()
        # LAST: refresh_gates draws this on every stage afterwards, but the first paint happens
        # before InferenceScreen.__init__ reaches its own refresh_gates() call, and an empty label
        # on launch reads as a missing feature rather than an empty session.
        self._sync_session_line()
```

- [ ] **Step 8: Run the task's own tests**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_nav_and_gating.py tests/test_settings_persistence.py tests/test_worker_dispatch.py -q`

Expected: PASS. (`test_settings_persistence.py` is in the list because it builds
`it.ConfigPanel(None)` with no screen at line 1226, which is the `screen is None` branch of Step 6;
`test_worker_dispatch.py` because it drives `_build_config` at line 294.)

- [ ] **Step 9: Commit**

```bash
git add core/gui/screens/inference_screen.py core/gui/panels/inference/config_tab.py tests/test_nav_and_gating.py
git commit -m "gui: the Config tab says what the session holds"
```

---

### Task 18: Apply confirms before it discards

**Files:**
- Modify: `core/gui/panels/inference/config_tab.py:3` (the `QtWidgets` import),
  `core/gui/panels/inference/config_tab.py:278` (the one line the guard goes in front of), and a new
  method after `_read_inputs` (which ends at line 246 with `return out`)
- Modify: `core/gui/screens/inference_screen.py:1-13` (module docstring: the stale tab count, §8.2 —
  **this task owns that sentence and is the only one that edits it**, Q3),
  `core/gui/screens/inference_screen.py:89-94` (`new_draft`'s docstring) and
  `core/gui/screens/inference_screen.py:96-104` (`install_config`'s docstring)
- Modify: `PRISM_HANDOFF.md:1667-1670` (the M1b bullet)
- Test: `tests/test_nav_and_gating.py` (append after Task 17's test)

Task 17 has already inserted into `core/gui/panels/inference/config_tab.py` (two methods after
`_build_config`, and a line in `__init__`) and Task 16 into
`core/gui/screens/inference_screen.py`; **re-read both files and take your line numbers from what you
see** rather than trusting the ranges above. The quoted text is the anchor.

**Interfaces:**
- Consumes: Task 17's `InferenceScreen.session_contents(self) -> list[str]` and
  `ConfigPanel.session_line`.
- Produces: `ConfigPanel._confirm_replace_session(self) -> bool` — True to go ahead, False to leave
  the session untouched. A method of its own so a test can answer it without a click, exactly as
  `PosteriorPanel._ask_new_run` and `_ask_load_non_amortized` are
  (`core/gui/panels/inference/posterior_tab.py:211,255`).

**Why this task exists:** `_build_config` calls `self._screen.new_draft(draft)` at
`core/gui/panels/inference/config_tab.py:278` with no question asked, and `new_draft` replaces the
whole `SbiSession` — so one stray Apply drops a nine-minute prior, a multi-day posterior and a
recorded observation from the session in silence. That is the user-facing half of the handoff's trap
**M1b** ("TWO screen entry points; mixing them up wipes your work", `PRISM_HANDOFF.md:1667`). Spec
**B12** requires a confirmation "defaulting to not applying", and §6.2 assigns M1b's other half —
rewriting both entry points' docstrings and recording the resolution in the handoff — to this same
change.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_nav_and_gating.py`, after Task 17's test:

```python
def test_apply_confirms_before_it_discards_the_session(monkeypatch):
    """B12's second half (spec §6.2). Apply is the destructive one of the screen's two entry points,
    so it asks -- but ONLY when there is something to lose.

    Four legs, and the first is the one that keeps the guard tolerable: an empty session is replaced
    in silence, with SHOWN empty, so nothing about the first Apply of a sitting changes. A non-empty
    one raises an INSTANCE QMessageBox (never QMessageBox.question -- the statics are C++ and escape
    tests/conftest.py's dialog guard, so a static call STALLS the offscreen suite instead of failing
    it) that lists exactly what ``session_contents()`` reports, says those artifacts stay on disk,
    and has "Keep this session" as its default BY NAME -- the D7/D8 lesson at
    tests/test_nav_and_gating.py:1175, where buttons() orders by role and Enter landed on the
    destructive button. Anything but the destructive button leaves the session object identical.
    """
    from PySide6.QtWidgets import QMessageBox
    from core.gui.screens.inference_screen import InferenceScreen
    from core.gui.session import SbiSession
    from tests._fixtures import PaneCapture, SHOWN, qt_app

    qt_app()
    inf = InferenceScreen()
    cfgp = inf.config_panel
    cfgp.model_combo.setCurrentText("NADROWSKI")
    cfgp._on_model_changed("NADROWSKI")
    cfgp.units_toggle.set_direct(False)
    cfgp.chi_check.setChecked(False)
    cap = PaneCapture(cfgp)

    def applied():
        return [text for _level, text in cap.lines if text.startswith("Model applied")]

    # (a) an EMPTY session: no dialog at all, and the session is replaced exactly as before
    SHOWN.clear()
    cfgp._build_config()
    assert SHOWN == [], "an empty session must be replaced silently"
    assert inf.session.draft is not None and len(applied()) == 1, applied()

    # (b) a non-empty session, nothing clicked: the dialog names what is held and nothing changes
    inf.session = SbiSession(draft=inf.session.draft, cfg=_spont_cfg(),
                             inf_prior=_prior_stub(id_="20260914T100000", name="p_master"))
    inf.refresh_gates()
    before = inf.session
    held = inf.session_contents()
    SHOWN.clear()
    cfgp._build_config()
    assert len(SHOWN) == 1, SHOWN
    box = SHOWN[-1]
    assert box.icon() == QMessageBox.Warning
    assert box.defaultButton() is not None, "no default button: Enter would do the destructive thing"
    assert box.defaultButton().text() == "Keep this session", box.defaultButton().text()
    for phrase in held:
        assert phrase in box.informativeText(), (phrase, box.informativeText())
    assert "STAYS ON DISK" in box.informativeText(), box.informativeText()
    assert inf.session is before, "a dialog nobody answered replaced the session"
    assert len(applied()) == 1, "a refused Apply must not report a model as applied"

    # (c) Enter -- i.e. the default button -- keeps the session too
    SHOWN.clear()
    monkeypatch.setattr(QMessageBox, "exec",
                        lambda self: SHOWN.append(self) or self.defaultButton().click() or 0)
    cfgp._build_config()
    assert inf.session is before, "clicking the default button replaced the session"
    assert len(applied()) == 1, applied()

    # (d) the destructive button: the session IS replaced, and the prior goes with it
    def _click_apply(self):
        SHOWN.append(self)
        next(b for b in self.buttons() if b.text() == "Apply and start a new session").click()
        return 0

    monkeypatch.setattr(QMessageBox, "exec", _click_apply)
    cfgp._build_config()
    assert inf.session is not before, "the destructive button did not replace the session"
    assert inf.session.inf_prior is None and inf.session.cfg is None, inf.session
    assert len(applied()) == 2, applied()
    assert inf.session_contents() == [] and "nothing yet" in cfgp.session_line.text()
```

- [ ] **Step 2: Run it and watch it fail**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_nav_and_gating.py::test_apply_confirms_before_it_discards_the_session -v`

Expected: FAIL at leg (b), `AssertionError: assert len(SHOWN) == 1` with `SHOWN == []` — Apply
replaced a non-empty session with no dialog.

- [ ] **Step 3: Import `QMessageBox` in `config_tab.py`**

Replace line 3 of `core/gui/panels/inference/config_tab.py`:

```python
from PySide6.QtWidgets import (QCheckBox, QComboBox, QGroupBox, QLabel, QLineEdit, QPushButton, QVBoxLayout)
```

with:

```python
from PySide6.QtWidgets import (QCheckBox, QComboBox, QGroupBox, QLabel, QLineEdit, QMessageBox,
                               QPushButton, QVBoxLayout)
```

- [ ] **Step 4: Add `_confirm_replace_session`**

In `core/gui/panels/inference/config_tab.py`, insert this immediately after `_read_inputs` (which
ends at line 246 with `        return out`) and before `def _build_config` (line 248):

```python
    def _confirm_replace_session(self) -> bool:
        """Ask before Apply throws this session's work away (B12). True = go ahead.

        SILENT WHEN THERE IS NOTHING TO LOSE. An empty ``session_contents()`` shows no dialog at all,
        so the first Apply of a sitting -- and every Apply made before a prior exists -- behaves
        exactly as it did before this guard. A confirmation that fires when nothing is at stake is
        trained away within a day, and then it is not a guard.

        NOTHING IS DELETED EITHER WAY, and the informative text says so in as many words: every stage
        writes its artifact at completion, so what Apply releases is the SESSION's hold on them. The
        one thing the operator cannot get back by re-picking is the minute spent re-running the
        stages, which is why this is a question and not a warning after the fact.

        "Keep this session" is the default BY NAME, not by index: ``QMessageBox.buttons()`` orders by
        role (Reject before Destructive), so ``buttons()[-1]`` is the destructive one -- that is how
        Enter once started the very run the Posterior tab's D7 dialog exists to stop
        (posterior_tab._ask_new_run). And it is an INSTANCE ``QMessageBox`` shown with ``.exec()``:
        tests/conftest.py patches the instance method only, so ``QMessageBox.question`` would hang
        the offscreen suite rather than fail it.

        A method of its own so a test can answer it without a click, like the Posterior tab's two.
        """
        held = self._screen.session_contents()
        if not held:
            return True
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("Start a new session?")
        box.setText("Applying a model replaces the whole session.")
        box.setInformativeText(
            "This session holds:\n\n"
            + "\n".join(f"  • {phrase}" for phrase in held)
            + "\n\nEvery one of them STAYS ON DISK and can be selected again in the pickers — what "
              "is released is this session's hold on them. They cannot be carried over: a different "
              "model or unit system invalidates them, which is why applying one starts a new "
              "session rather than keeping what it can.")
        go = box.addButton("Apply and start a new session", QMessageBox.DestructiveRole)
        keep = box.addButton("Keep this session", QMessageBox.RejectRole)
        box.setDefaultButton(keep)
        box.exec()
        return box.clickedButton() is go
```

- [ ] **Step 5: Put the guard in front of `new_draft`**

Still in `config_tab.py`, replace line 278:

```python
        self._screen.new_draft(draft)                # replaces the session + repoints Prior + re-gates
```

with:

```python
        # AFTER the boxes are read and the draft is built, and BEFORE anything is installed: a
        # refused box must never cost a dialog, and a dialog answered "keep" must leave the session
        # byte-for-byte as it found it -- including the log, which is why the "Model applied" lines
        # below are downstream of this return.
        if not self._confirm_replace_session():
            return
        self._screen.new_draft(draft)                # replaces the session + repoints Prior + re-gates
```

- [ ] **Step 6: Rewrite both entry points' docstrings (M1b's other half)**

In `core/gui/screens/inference_screen.py`, replace the body of `new_draft`'s docstring (lines 90-91):

```python
        """Config applied: replace the WHOLE session (a different model or unit system invalidates every
        artifact) and repoint the Prior tab's bounds picker at the new model's folder."""
```

with:

```python
        """Config applied: replace the WHOLE session (a different model or unit system invalidates
        every artifact) and repoint the Prior tab's bounds picker at the new model's folder.

        THE DESTRUCTIVE ONE OF THE SCREEN'S TWO ENTRY POINTS, AND THE CONFIRMATION IS NOW THE GUARD
        (piece 4, B12). The Config tab asks before it calls here whenever ``session_contents()`` is
        non-empty, defaulting to keeping the session, so a mis-aimed Apply can no longer drop a
        prior, a posterior and a recorded observation in silence -- which is the half of the
        handoff's trap M1b that a reader could not defend themselves against.

        Nothing is deleted either way: every stage writes its artifact at completion, so what this
        drops is the session's HANDLES, and the artifacts stay on disk to be selected again. Its
        in-place twin is ``install_config``.
        """
```

and replace `install_config`'s docstring (lines 97-101):

```python
        """Prior stage: bounds chosen, so the SimConfig now exists. Set it IN PLACE and fan out to the
        tabs whose pickers/fields depend on it.

        Deliberately NOT a new session: the Prior stage installs the config as the first step of building
        the prior, so replacing the session here would wipe the artifact it is about to store."""
```

with:

```python
        """Prior stage: bounds chosen, so the SimConfig now exists. Set it IN PLACE and fan out to
        the tabs whose pickers/fields depend on it.

        THE IN-PLACE ONE OF THE TWO ENTRY POINTS, and in place BECAUSE OF WHEN IT IS CALLED: the
        Prior stage installs the config as the first step of building the prior, so a new session
        here would wipe the draft that built this config and the artifact the stage is about to
        store -- mid-stage, with nothing to confirm against, because the operator pressed a button
        that promised to build a prior and not to start over. That asymmetry is the whole of the
        handoff's trap M1b: its destructive twin ``new_draft`` is the one the Config tab now
        confirms (B12), and this one is never confirmed at all.
        """
```

- [ ] **Step 7: Correct the module docstring's stale tab count (§8.2)**

Still in `core/gui/screens/inference_screen.py`, replace lines 1-3:

```python
"""The Parameter Inference section: five tabs over ONE shared SbiSession, with cross-tab gating.

    Config -> Prior -> Posterior -> Validate -> Infer
```

with:

```python
"""The Parameter Inference section: six tabs over ONE shared SbiSession, with cross-tab gating.

    Config -> Prior -> Posterior -> Validate -> Infer -> TSNPE
```

(Spec §8.2: this docstring said five while the class docstring twelve lines below said six; the
screen has built six tabs since the TSNPE tab arrived. This task owns THIS sentence, because it is
editing this file anyway for Step 6's two docstrings. It must NOT touch the other two stale counts
§8.2 names — `core/gui/panels/base_panel.py`'s "nine of these exist" and
`core/gui/panels/inference/base.py`'s "the five inference tabs": Task 25 owns them.)

- [ ] **Step 8: Record the resolution in the handoff's M1b bullet**

In `PRISM_HANDOFF.md`, replace lines 1667-1670:

```markdown
- **M1b. TWO screen entry points; mixing them up wipes your work.** `new_draft(draft)` **REPLACES**
  the whole session (a different model invalidates every artifact). `install_config(cfg)` sets
  `session.cfg` **IN PLACE** — deliberately not a new session, because Prior installs the config as
  the first step of building the prior.
```

with:

```markdown
- **M1b. TWO screen entry points; mixing them up wipes your work.** `new_draft(draft)` **REPLACES**
  the whole session (a different model invalidates every artifact). `install_config(cfg)` sets
  `session.cfg` **IN PLACE** — deliberately not a new session, because Prior installs the config as
  the first step of building the prior. Piece 4 (B12) answered the **user-facing** half: the Config
  tab now carries a permanent line naming what the session holds and confirms — defaulting to
  keeping it — before `new_draft` releases it, so mixing the two up can no longer discard a prior, a
  posterior and a recorded observation in silence; the rest of trap group M is still piece 6's.
```

- [ ] **Step 9: Run the task's own tests**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_nav_and_gating.py tests/test_worker_dispatch.py tests/test_settings_persistence.py -q`

Expected: PASS. All six existing `_build_config` call sites
(`tests/test_nav_and_gating.py:708,724,1664,1718,1756` and `tests/test_worker_dispatch.py:294`) drive
a fresh screen whose session holds at most a draft, so `session_contents()` is empty for each of them
and no dialog appears — which is what keeps them green unchanged.

- [ ] **Step 10: Commit**

```bash
git add core/gui/panels/inference/config_tab.py core/gui/screens/inference_screen.py PRISM_HANDOFF.md tests/test_nav_and_gating.py
git commit -m "gui: Apply confirms before it discards the session"
```

---

### Task 19: the pickers say what they hold

**Files:**
- Modify: `core/gui/widgets/artifact_picker.py:103-170` — a module constant and a private formatter
  above `class StorePicker`, and inside it `__init__` (110-124), `refresh` (130-151) and a new
  `selection_summary` after `key` (161-163)
- Modify: `core/gui/panels/inference/prior_tab.py:75-76` and
  `core/gui/panels/inference/prior_tab.py:328-330`
- Modify: `core/gui/panels/inference/posterior_tab.py:40-42` and
  `core/gui/panels/inference/posterior_tab.py:320-325`
- Modify: `core/gui/panels/inference/tsnpe_tab.py:58-59` and
  `core/gui/panels/inference/tsnpe_tab.py:191-196`
- Test: `tests/test_nav_and_gating.py` (append after Task 18's test)

**Interfaces:**
- Consumes: nothing from an earlier task. It reads only the `Summary` fields that already exist —
  `label`, `id`, `created`, `mode`, `width`, `amortized`, `complete`
  (`core/artifacts/store.py:61-80`, filled by `list` at `core/artifacts/store.py:338-350`) — so it
  does not wait on the store's new keyword fields, and it stays compatible with the
  `types.SimpleNamespace` row stub `tests/test_settings_persistence.py:1066` already feeds the
  picker.
- Produces:
  - `core.gui.widgets.artifact_picker.NARROWED_SUFFIX` — the item-text marker, as a module constant
  - `core.gui.widgets.artifact_picker._summary_line(s) -> str` — the one formatter behind the line
  - `StorePicker.selection_summary(self) -> str`
  - `PriorPanel.prior_line` + `PriorPanel._sync_prior_line()`,
    `PosteriorPanel.post_line` + `PosteriorPanel._sync_post_line()`,
    `TSNPEPanel.obs_line` + `TSNPEPanel._sync_obs_line()`

**Why this task exists:** the decomposition's piece-4 row asked for "picker shows mode, width,
amortized/truncated". Piece 1 (`86328e4`) put those in the per-item **tooltip** only: the closed
combo still shows nothing but `Summary.label`, so a narrowed TSNPE posterior — valid near exactly one
observation — is indistinguishable from an amortized one until you hover it, and once it is selected
the combo shows a bare name with nothing beneath. Spec **B13** finishes it: the visible item text
marks the exception, and a read-only line under each picker spells the selection out in full. Spec
§6.3 also records that **nothing pins any picker text today** — no test in the repository asserts an
item's text, its tooltip, or anything else the picker renders (`grep -rn "itemText" tests/` finds
none) — so this task writes the first ones.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_nav_and_gating.py`, after Task 18's test:

```python
def test_the_store_pickers_say_what_they_hold(monkeypatch):
    """B13 (spec §6.3), and the FIRST test of any kind on what a picker renders.

    Three things. The visible item text marks the EXCEPTION only -- a posterior with
    ``amortized is False`` reads "<label>  —  narrowed (TSNPE)", and amortized, being the norm,
    gets no suffix (a suffix on almost every row carries no information and pushes the name out of a
    combo sized to its first show). The per-item TOOLTIP keeps every fact it carried, and words the
    one fact it shares with the item text the SAME way: "narrowed (TSNPE)", in place of the old
    "NON-AMORTIZED (TSNPE)" -- one wording for one fact, in the item, the tooltip and the line. No
    test in the repository pinned either string before this one (spec §6.3). And each of
    the three tabs that owns a StorePicker carries a read-only line beneath it, spelling the current
    selection out through ``selection_summary()`` -- refreshed on ``currentIndexChanged`` AND on
    ``refresh()``, because a stage that writes the store rescans the picker without anyone clicking.

    The store is stubbed at the picker's one seam, ``_resolved_store``, the way
    tests/test_settings_persistence.py:1070 already stubs it; each row carries exactly the
    ``Summary`` fields refresh() reads and no more, so the test cannot pass on a field the real
    listing does not fill.
    """
    import types
    from PySide6.QtCore import Qt
    from core.gui.screens.inference_screen import InferenceScreen
    from core.gui.widgets.artifact_picker import NARROWED_SUFFIX, StorePicker
    from tests._fixtures import qt_app

    qt_app()

    def row(label, id_, created, *, mode=None, width=None, amortized=None, complete=True):
        return types.SimpleNamespace(complete=complete, label=label, id=id_, created=created,
                                     mode=mode, width=width, amortized=amortized)

    rows = {
        "prior": [row("p_master", "20260914T100000", "2026-09-14T10:00:00")],
        "posterior": [
            row("amort", "20260914T102231", "2026-09-14T10:22:31", mode="chi", width=18,
                amortized=True),
            row("round2", "20260915T090000", "2026-09-15T09:00:00", mode="chi", width=18,
                amortized=False),
            row("(unnamed leftover)", "leftover", "", complete=False),
        ],
        "observation": [row("obs1", "20260916T120000", "2026-09-16T12:00:00", mode="spontaneous",
                            width=50)],
    }
    store = types.SimpleNamespace(list=lambda kind: list(rows.get(kind, [])))
    monkeypatch.setattr(StorePicker, "_resolved_store", lambda self: store)

    inf = InferenceScreen()
    pp = inf.posterior_panel
    combo = pp.post_picker.combo

    # (a) the item text: the exception is marked, the norm is not, the incomplete row is absent
    texts = [combo.itemText(i) for i in range(combo.count())]
    assert texts == [StorePicker.NEW_LABEL, "amort", "round2" + NARROWED_SUFFIX], texts
    assert NARROWED_SUFFIX.strip().startswith("—"), NARROWED_SUFFIX
    assert "narrowed (TSNPE)" in NARROWED_SUFFIX, NARROWED_SUFFIX

    # (b) the tooltip keeps every fact and its id-first order, and words amortization the one way
    tips = [combo.itemData(i, Qt.ToolTipRole) for i in range(combo.count())]
    assert tips[1] == "20260914T102231 · 2026-09-14T10:22:31 · chi · width 18 · amortized", tips[1]
    assert tips[2] == ("20260915T090000 · 2026-09-15T09:00:00 · chi · width 18 · "
                       "narrowed (TSNPE)"), tips[2]
    assert "NON-AMORTIZED" not in tips[2], tips[2]

    # (c) selection_summary + the line under the combo, for the posterior's two shapes and for none
    combo.setCurrentIndex(1)
    assert pp.post_picker.selection_summary() == "chi · width 18 · amortized · 2026-09-14T10:22:31"
    assert pp.post_line.text() == pp.post_picker.selection_summary(), pp.post_line.text()
    combo.setCurrentIndex(2)
    assert pp.post_picker.selection_summary() == \
        "chi · width 18 · narrowed (TSNPE) · 2026-09-15T09:00:00"
    assert pp.post_line.text() == pp.post_picker.selection_summary(), pp.post_line.text()
    combo.setCurrentIndex(0)                       # the "(from scratch)" sentinel is not an artifact
    assert pp.post_picker.selection_summary() == ""
    assert pp.post_line.text() == ""

    # (d) the other two kinds: a prior carries only its creation time, an observation its geometry
    prior_picker = inf.prior_panel.prior_picker
    prior_picker.combo.setCurrentIndex(1)          # index 0 is "(from scratch)"
    assert prior_picker.selection_summary() == "2026-09-14T10:00:00"
    assert inf.prior_panel.prior_line.text() == "2026-09-14T10:00:00"
    obs_picker = inf.tsnpe_panel.obs_picker        # allow_new=False: index 0 IS the observation
    assert obs_picker.selection_summary() == "spontaneous · width 50 · 2026-09-16T12:00:00"
    assert inf.tsnpe_panel.obs_line.text() == "spontaneous · width 50 · 2026-09-16T12:00:00"

    # (e) the line follows a refresh(), not only a click: drop the narrowed posterior and rescan
    combo.setCurrentIndex(1)
    rows["posterior"] = rows["posterior"][:1]
    pp.post_picker.refresh()
    assert [combo.itemText(i) for i in range(combo.count())] == [StorePicker.NEW_LABEL, "amort"]
    assert pp.post_line.text() == "chi · width 18 · amortized · 2026-09-14T10:22:31", \
        pp.post_line.text()
```

- [ ] **Step 2: Run it and watch it fail**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_nav_and_gating.py::test_the_store_pickers_say_what_they_hold -v`

Expected: FAIL at collection of the test body,
`ImportError: cannot import name 'NARROWED_SUFFIX' from 'core.gui.widgets.artifact_picker'`.

- [ ] **Step 3: Add the marker constant and the formatter to `artifact_picker.py`**

In `core/gui/widgets/artifact_picker.py`, insert this between the end of `ArtifactPicker` (line 100,
`            self.restore_key(restore_key)`) and `class StorePicker` (line 103) — i.e. in the blank
lines at 101-102:

```python
# The visible item text marks the EXCEPTION, never the norm (piece 4, B13). Nearly every posterior is
# amortized, so an "amortized" suffix on almost every row would carry no information at all and would
# push the name itself out of a combo that AdjustToContentsOnFirstShow sized to its first listing. A
# TSNPE round's result is the rare row -- valid near ONE observation, and the one you must not pick by
# accident -- so it is the one that gets a suffix. Plain text, no icon: see ArtifactPicker.NEW_LABEL
# for why a combo ITEM cannot carry the bundled icon font.
NARROWED_SUFFIX = "  —  narrowed (TSNPE)"


def _summary_line(s) -> str:
    """One line describing a store row -- mode, conditioning width, amortization, creation time, in
    that order -- with the parts the kind does not carry left out. ``""`` for a row with none of them.

    THE ONE FORMATTER BEHIND THE LINE UNDER EVERY PICKER, so the three tabs cannot word the same
    facts three ways -- and the tooltip now words amortization the same way this does, "narrowed
    (TSNPE)", which is also what the item text's suffix says: one fact, one wording, wherever it is
    shown. The per-item TOOLTIP is still not BUILT from this function: it leads with the id and keeps
    every fact it carried (B13), while the line needs no id, because the picker's own text already
    says which item it is describing.

    ``getattr`` throughout, because this is handed ``Summary`` in production and a row stub in the
    suites, and a formatter feeding a read-only label must not raise over a missing attribute.
    """
    parts = []
    if getattr(s, "mode", None):
        parts.append(str(s.mode))
    if getattr(s, "width", None):
        parts.append(f"width {s.width}")
    amortized = getattr(s, "amortized", None)
    if amortized is not None:
        parts.append("amortized" if amortized else "narrowed (TSNPE)")
    if getattr(s, "created", None):
        parts.append(str(s.created))
    return " · ".join(parts)
```

- [ ] **Step 4: Record the summaries in `StorePicker.refresh` and mark the narrowed item**

In the same file, replace `StorePicker.__init__`'s line 112:

```python
        self.kind, self._allow_new, self._store = kind, allow_new, store
```

with:

```python
        self.kind, self._allow_new, self._store = kind, allow_new, store
        # id -> the line selection_summary() returns, recorded by refresh(). BEFORE refresh() below,
        # which fills it.
        self._summaries: dict = {}
```

and replace `refresh`'s lines 130-151:

```python
    def refresh(self):
        current = self.key()
        self.combo.clear()
        if self._allow_new:
            self.combo.addItem(self.NEW_LABEL, userData=None)
        try:
            rows = self._resolved_store().list(self.kind)
        except Exception:                        # noqa: BLE001 -- an unreadable root lists nothing
            rows = []
        for s in rows:
            if not s.complete:
                continue
            self.combo.addItem(s.label, userData=s.id)
            tip = f"{s.id} · {s.created}"
            if s.mode:
                tip += f" · {s.mode}"
            if s.width:
                tip += f" · width {s.width}"
            if s.amortized is not None:
                tip += " · amortized" if s.amortized else " · NON-AMORTIZED (TSNPE)"
            self.combo.setItemData(self.combo.count() - 1, tip, _TOOLTIP_ROLE)
        self.restore_key(current)
```

with:

```python
    def refresh(self):
        current = self.key()
        self.combo.clear()
        self._summaries = {}                     # rebuilt here, so a deleted artifact's line goes too
        if self._allow_new:
            self.combo.addItem(self.NEW_LABEL, userData=None)
        try:
            rows = self._resolved_store().list(self.kind)
        except Exception:                        # noqa: BLE001 -- an unreadable root lists nothing
            rows = []
        for s in rows:
            if not s.complete:
                continue
            # The suffix is appended to the DISPLAY only. key(), restore_key(), selected() and
            # has_entries() all go through userData, which is still the bare id, so nothing that
            # persists or resolves a selection can be broken by a change of wording here.
            self.combo.addItem(s.label + (NARROWED_SUFFIX if s.amortized is False else ""),
                               userData=s.id)
            self._summaries[str(s.id)] = _summary_line(s)
            tip = f"{s.id} · {s.created}"
            if s.mode:
                tip += f" · {s.mode}"
            if s.width:
                tip += f" · width {s.width}"
            if s.amortized is not None:
                # ONE WORDING FOR ONE FACT (B13): the same "narrowed (TSNPE)" the item
                # suffix and the line under the picker use, so nobody has to learn that a
                # shouted "NON-AMORTIZED" here and a quiet "narrowed" there are the same
                # thing. The tooltip keeps everything else it carried.
                tip += " · amortized" if s.amortized else " · narrowed (TSNPE)"
            self.combo.setItemData(self.combo.count() - 1, tip, _TOOLTIP_ROLE)
        self.restore_key(current)
```

Note `s.amortized is False`, not `not s.amortized`: `None` means "this kind has no such field" (every
prior and every observation), and `not None` is True, which would suffix every one of them.

Note also the tooltip's tail: the old `NON-AMORTIZED (TSNPE)` becomes `narrowed (TSNPE)`. Spec §6.3
records that nothing in the repository pins any picker text, so no test breaks on it, and the tooltip
keeps every other character — only the wording of that one fact changes, to the wording the item
suffix and `_summary_line` use. The Posterior tab's load dialog keeps its own louder
`NON-AMORTIZED` wording (`posterior_tab.py:260,285`): that is a consent about to be given, not a
description of a row, and it is not this task's.

- [ ] **Step 5: Add `selection_summary`**

In the same file, insert this immediately after `key` (which ends at line 163 with
`        return "" if data is None else str(data)`) and before `def restore_key` (line 165):

```python
    def selection_summary(self) -> str:
        """The current item spelled out -- "chi · width 18 · amortized · 2026-09-14T10:22:31" -- and
        "" when nothing is selected or the '(from scratch)' sentinel is (a sentinel is not an
        artifact and has nothing to describe).

        Read off what the last ``refresh()`` recorded, never off the store: this is called from a
        ``currentIndexChanged`` slot, and re-listing a kind's directory on every index change is how
        a combo becomes a disk scan.
        """
        data = self.combo.currentData()
        return "" if data is None else self._summaries.get(str(data), "")
```

- [ ] **Step 6: Put the line under the Prior tab's picker**

In `core/gui/panels/inference/prior_tab.py`, replace lines 75-76:

```python
        self.prior_picker = StorePicker("prior", allow_new=True)
        add_help_row(form, label("prior"), self.prior_picker, HELP["prior"])
```

with:

```python
        self.prior_picker = StorePicker("prior", allow_new=True)
        add_help_row(form, label("prior"), self.prior_picker, HELP["prior"])
        # What the picked prior actually IS, directly beneath the combo (B13): the closed combo shows
        # a name and nothing else, and a name is not enough to tell two priors apart. Created before
        # the connect below, because restore_settings at the end of __init__ re-selects a saved id
        # and that fires currentIndexChanged into this slot.
        self.prior_line = _TrainingBudgetMixin._derived_label()
        form.addRow("", self.prior_line)
        self.prior_picker.combo.currentIndexChanged.connect(lambda _i: self._sync_prior_line())
        self._sync_prior_line()
```

Then insert this method immediately before `refresh_local_gates` (line 328):

```python
    def _sync_prior_line(self) -> None:
        """The read-only line under the prior picker (B13).

        Driven by ``currentIndexChanged``, which ``StorePicker.refresh`` also fires -- it clears the
        combo and repopulates it, and its final ``restore_key`` is the last signal of the batch, so
        the line ends up describing the item that ends up current. Re-driven from
        ``refresh_local_gates`` as well, because a stage that writes the store can change what a
        refresh finds without anyone touching the combo.
        """
        self.prior_line.setText(self.prior_picker.selection_summary())
```

and replace `refresh_local_gates` (lines 328-330):

```python
    def refresh_local_gates(self):
        self.btn_prior.setEnabled(self.session.draft is not None)
        self.btn_save_prior.setEnabled(self.session.inf_prior is not None)
```

with:

```python
    def refresh_local_gates(self):
        self.btn_prior.setEnabled(self.session.draft is not None)
        self.btn_save_prior.setEnabled(self.session.inf_prior is not None)
        self._sync_prior_line()
```

- [ ] **Step 7: Put the line under the Posterior tab's picker**

In `core/gui/panels/inference/posterior_tab.py`, replace lines 40-42:

```python
        self.post_picker = StorePicker("posterior", allow_new=True)
        self.post_picker.combo.currentIndexChanged.connect(lambda _i: self._sync_train_button())
        add_help_row(form, label("posterior"), self.post_picker, HELP["posterior"])
```

with:

```python
        self.post_picker = StorePicker("posterior", allow_new=True)
        self.post_picker.combo.currentIndexChanged.connect(lambda _i: self._sync_train_button())
        add_help_row(form, label("posterior"), self.post_picker, HELP["posterior"])
        # AMORTIZATION SPELLED OUT, under the one picker where it decides what the artifact is good
        # for (B13). The item text marks a narrowed posterior; this line says so in words, next to
        # the mode, the conditioning width and when it was trained.
        self.post_line = self._derived_label()
        form.addRow("", self.post_line)
        self.post_picker.combo.currentIndexChanged.connect(lambda _i: self._sync_post_line())
        self._sync_post_line()
```

Then insert this method immediately before `_sync_train_button` (line 312):

```python
    def _sync_post_line(self) -> None:
        """The read-only line under the posterior picker (B13). See PriorPanel._sync_prior_line for
        why ``currentIndexChanged`` also covers ``refresh()``."""
        self.post_line.setText(self.post_picker.selection_summary())
```

and replace `refresh_local_gates` (lines 320-325):

```python
    def refresh_local_gates(self):
        self._sync_train_button()
        self.btn_save_post.setEnabled(self.session.posterior is not None)
        # A config or a prior arriving changes every derived line, and the checkpoint line cannot be
        # computed without both.
        self._sync_budget()
```

with:

```python
    def refresh_local_gates(self):
        self._sync_train_button()
        self.btn_save_post.setEnabled(self.session.posterior is not None)
        self._sync_post_line()
        # A config or a prior arriving changes every derived line, and the checkpoint line cannot be
        # computed without both.
        self._sync_budget()
```

- [ ] **Step 8: Put the line under the TSNPE tab's picker**

In `core/gui/panels/inference/tsnpe_tab.py`, replace lines 58-59:

```python
        self.obs_picker = StorePicker("observation")
        add_help_row(form, label("observation"), self.obs_picker, HELP["tsnpe_obs"])
```

with:

```python
        self.obs_picker = StorePicker("observation")
        add_help_row(form, label("observation"), self.obs_picker, HELP["tsnpe_obs"])
        # The observation's own mode and conditioning width, beneath the combo (B13). This is the one
        # picker whose selection is loaded on the GUI thread and checked against the session's config
        # (_round), so seeing the geometry before the click is what turns that refusal into a
        # non-event.
        self.obs_line = self._derived_label()
        form.addRow("", self.obs_line)
        self.obs_picker.combo.currentIndexChanged.connect(lambda _i: self._sync_obs_line())
        self._sync_obs_line()
```

Then insert this method immediately before `refresh_local_gates` (line 191):

```python
    def _sync_obs_line(self) -> None:
        """The read-only line under the observation picker (B13). See PriorPanel._sync_prior_line for
        why ``currentIndexChanged`` also covers ``refresh()``."""
        self.obs_line.setText(self.obs_picker.selection_summary())
```

and replace `refresh_local_gates` (lines 191-196):

```python
    def refresh_local_gates(self):
        s = self.session
        self.obs_picker.refresh()
        self.btn_round.setEnabled(s.posterior is not None and s.inf_prior is not None
                                  and bool(self.obs_picker.key()))
        self._sync_budget()
```

with:

```python
    def refresh_local_gates(self):
        s = self.session
        self.obs_picker.refresh()
        self._sync_obs_line()
        self.btn_round.setEnabled(s.posterior is not None and s.inf_prior is not None
                                  and bool(self.obs_picker.key()))
        self._sync_budget()
```

- [ ] **Step 9: Run the task's own tests**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_nav_and_gating.py tests/test_settings_persistence.py tests/test_worker_dispatch.py tests/test_artifact_store.py -q`

Expected: PASS. `test_settings_persistence.py` is in the list because it is the only existing test
that drives `StorePicker.refresh` against a stubbed store
(`test_the_tsnpe_tab_restores_its_observation_and_budget_only`, line 1030) and it asserts on
`obs_picker.key()`, which still reads `userData`; `test_artifact_store.py` because it owns the
`Summary` fields `_summary_line` reads.

- [ ] **Step 10: Commit**

```bash
git add core/gui/widgets/artifact_picker.py core/gui/panels/inference/prior_tab.py core/gui/panels/inference/posterior_tab.py core/gui/panels/inference/tsnpe_tab.py tests/test_nav_and_gating.py
git commit -m "gui: the store pickers spell out what they hold"
```

### Task 20: the front-end root handler

**Files:**
- Create: `core/logging_root.py`
- Modify: `core/gui/app.py:1-40` (imports, a new module-level sink, the install as `build_app`'s first statement)
- Modify: `core/tool/__init__.py:10-21` (imports), `core/tool/__init__.py:80-158` (`main`: a new module-level sink, the install before the existing `try`, a `finally` on the existing ladder)
- Modify: `tests/test_user_models.py:936-967` (the one test that calls `build_app` must remove the handler it installs)
- Test: `tests/test_refusals.py` (three new tests appended at the end, after line 775)
- Test: `tests/test_worker_dispatch.py` (one new test appended at the end, after line 531)
- Test: `tests/test_tool.py` (one new test appended at the end, after line 1713)

**Interfaces:**
- Consumes: nothing from earlier tasks. Existing: `core.runs.LOGGER` (the `core` logger, set to INFO once at import, `core/runs.py:35-36`), `core.gui.streams.redirect_streams(signals, cancel=None)` and its `_SignalStream` (stdout at the pane's `info` level, stderr hard-wired to `warning`), `core.tool.logging_console.console_handlers()` (the `core` logger's two handlers, resolved at emit time).
- Produces: `core/logging_root.py` with `install(sink) -> None` (`sink(record: logging.LogRecord) -> None`), `remove() -> None`, `installed() -> bool`; `core.gui.app._library_record_sink(record) -> None`; `core.tool._library_record_sink(record) -> None`.

**Why this task exists:** Piece 3's spec §4.6 claimed a library's warning reaches the operator through `logging.lastResort`. It does not: `logging.warning(...)` calls `basicConfig()` whenever the root logger has no handlers (`C:/Users/J/anaconda3/envs/biophys-env/Lib/logging/__init__.py:2199-2201`), and `basicConfig` installs a real `StreamHandler` bound to `sys.stderr` **at construction** (`:2100-2131`). `core.propagate` is True and pinned (`tests/test_refusals.py:423`, because `caplog` reads records off root), so from that moment every `core` record is emitted twice — once by the front end's own handler, once by the root's — and in the window the second copy arrives in the run's `_SignalStream` at the error stream's `warning` level, then keeps writing into a **stopped** pump once the run ends, where lines are appended and never published. Piece 3 recorded that no trigger exists; that is wrong. sbi's `accept_reject_sample` calls `logging.warning` at `sbi/samplers/rejection/rejection.py:336` and `:359` whenever fewer than `warn_acceptance=0.01` of its proposals are accepted (`:331`), and that function is on the path of **every** posterior draw PRISM makes. B14: "Both front ends install a **root-logger handler at start-up**. `core` records pass through untouched; every other logger's record is shown **once**, prefixed with the logger that said it. Because a handler exists from start-up, `logging.basicConfig` never fires."

One wrinkle in "prefixed with the logger that said it": the trigger above is `logging.warning`, the module-level function, whose logger IS the root logger — so the prefix would read `library: root:` and name nothing. For that one logger name the prefix falls back to `record.module`, the basename of the file that logged, so sbi's leakage warning reads `library: rejection:` (`sbi/samplers/rejection/rejection.py`). Walkthrough row D15 ("a library warning carries its prefix") expects that module name, not `root`.

- [ ] **Step 1: Write the failing test — the precondition, the prevention, and the `core` drop**

Append to the end of `tests/test_refusals.py`:

```python
# ── piece 4, B14: THE root-logger handler (spec §7.1) ────────────────────────────────────────────
def test_the_root_handler_exists_from_startup_so_basicconfig_never_fires():
    """WHY NO SUITE HAS EVER SEEN THIS DEFECT: pytest attaches a handler to the root logger for every
    test phase (_pytest/logging.py, ``catching_logs.__enter__``, around line 349), so
    ``len(root.handlers) == 0`` is never true during a test and ``basicConfig`` cannot fire. This test
    therefore takes pytest's root handlers off, asserts the defect's PRECONDITION -- a handler-less
    root logger makes ``logging.warning`` install a StreamHandler of its own -- asserts that
    ``logging_root.install`` prevents exactly that, and puts them back in a finally.

    Then the two halves of the filter: a record from the ``core`` tree reaches its OWN handler once
    and the root sink NOT AT ALL (it already has the front end's handler and the artifact's log.txt),
    while every other logger's record is handed to the sink once, with its level and its logger's
    name intact.
    """
    import logging

    from core import logging_root, runs

    root = logging.getLogger()
    pytest_handlers = root.handlers[:]
    core_seen, sink_seen = [], []

    class _Count(logging.Handler):
        def emit(self, record):
            core_seen.append((record.name, record.levelname, record.getMessage()))

    own = _Count()
    runs.LOGGER.addHandler(own)            # stands in for the front end's own handler on ``core``
    try:
        for h in pytest_handlers:
            root.removeHandler(h)
        assert root.handlers == [], "the precondition needs a handler-less root logger"

        logging.warning("a library warning, with no handler installed")
        assert len(root.handlers) == 1 and isinstance(root.handlers[0], logging.StreamHandler), \
            "logging.warning no longer calls basicConfig: the defect this handler prevents is gone"
        for h in root.handlers[:]:
            root.removeHandler(h)

        logging_root.install(sink_seen.append)
        assert logging_root.installed() is True
        logging.warning("a library warning, with THE root handler installed")
        assert len(root.handlers) == 1, [type(h).__name__ for h in root.handlers]
        assert [r.getMessage() for r in sink_seen] == \
            ["a library warning, with THE root handler installed"], sink_seen
        assert sink_seen[-1].levelno == logging.WARNING and sink_seen[-1].name == "root", \
            (sink_seen[-1].levelno, sink_seen[-1].name)

        sink_seen.clear()
        logging.getLogger("core.probe").info("the pipeline's own voice")
        assert sink_seen == [], "a core record reached the root sink: the filter is not dropping it"
        assert core_seen == [("core.probe", "INFO", "the pipeline's own voice")], core_seen

        logging_root.remove()
        assert logging_root.installed() is False and root.handlers == []
    finally:
        logging_root.remove()
        runs.LOGGER.removeHandler(own)
        for h in pytest_handlers:
            root.addHandler(h)


def test_caplog_still_sees_core_records_while_the_root_handler_is_installed(caplog):
    """The drop is a ``logging.Filter`` ON THE HANDLER, never ``core.propagate = False``: ``caplog``
    reads off the root logger and the propagation is pinned by
    test_the_core_logger_is_at_info_by_import above, and ``RunLog`` plus both front-end handlers sit
    on the ``core`` logger itself, so a filter leaves all three untouched."""
    import logging

    from core import logging_root

    seen = []
    logging_root.install(seen.append)
    try:
        with caplog.at_level(logging.INFO, logger="core"):
            logging.getLogger("core.probe").info("a record caplog must still see")
            logging.getLogger("a_library").warning("and one the sink may have")
    finally:
        logging_root.remove()
    assert "a record caplog must still see" in caplog.text
    assert [r.getMessage() for r in seen] == ["and one the sink may have"], seen


def test_logging_root_imports_only_the_standard_librarys_logging():
    """``python -m core --help`` must stay torch-free and the tool imports this module at its top, so
    the same pin the refusals module carries applies here: the import statements name ``logging`` and
    nothing else, a fresh interpreter that imports it has no torch loaded, and no logger's level is
    touched (core/runs.py sets the ``core`` level once, at import, and nothing else ever does)."""
    src = (REPO / "core" / "logging_root.py").read_text(encoding="utf-8")
    imported = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            imported |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module)
    assert imported == {"logging"}, imported
    calls = [ast.unparse(n.func) for n in ast.walk(ast.parse(src))
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    assert not any(c.endswith("setLevel") for c in calls), calls
    probe = ("import sys, core.logging_root; "
             "bad = sorted(m for m in sys.modules if m == 'torch' or m.startswith('torch.')); "
             "sys.exit(repr(bad) if bad else 0)")
    r = subprocess.run([sys.executable, "-c", probe], cwd=str(REPO), capture_output=True, text=True,
                       timeout=120)
    assert r.returncode == 0, r.stderr
```

- [ ] **Step 2: Run it and watch it fail**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_refusals.py -k root_handler_exists_from_startup -v`
Expected: FAIL, `ImportError: cannot import name 'logging_root' from 'core'`

- [ ] **Step 3: Create `core/logging_root.py`**

```python
"""THE root-logger handler, installed by both front ends at start-up (piece 4, B14, spec §7.1).

WHAT WAS WRONG. ``logging.warning(...)`` -- the module-level function, on the ROOT logger -- calls
``basicConfig()`` whenever the root logger has no handlers (logging/__init__.py, ``warning``), and
``basicConfig`` installs a real ``StreamHandler`` bound to ``sys.stderr`` AT CONSTRUCTION. The
``core`` logger propagates (core/runs.py; pinned, because ``caplog`` reads records off the root), so
from that moment every ``core`` record is emitted TWICE -- once by the front end's own handler, once
by the root's. Under the window it is worse than doubling: the stream that handler captured is the
RUN's ``_SignalStream``, so the duplicate arrives in the pane at the error stream's ``warning``
level, and once the run ends the handler keeps writing into a STOPPED pump, where lines are appended
and never published. After one trigger, library records go MISSING rather than doubling.

The trigger is not hypothetical. sbi's ``accept_reject_sample`` calls ``logging.warning`` at
sbi/samplers/rejection/rejection.py:336 and :359 when fewer than ``warn_acceptance=0.01`` of its
proposals are accepted, and that function is on the path of EVERY posterior draw PRISM makes.

WHAT THIS DOES. ``install(sink)`` puts ONE handler on the root logger:
  * it exists from start-up, which is why ``basicConfig`` never fires and no second copy of a
    ``core`` record can ever appear;
  * a record from the ``core`` tree is DROPPED here -- it already has the front end's own handler and
    the artifact's log.txt, both of which read off the ``core`` logger -- and every other logger's
    record is handed to ``sink`` exactly once.

The drop is a ``logging.Filter`` ON THE HANDLER, never a change to ``core.propagate``: ``caplog``
reads off root and tests/test_refusals.py pins that propagation, and ``RunLog`` and both front-end
handlers sit on the ``core`` logger itself, so a filter leaves all three untouched.

``sink`` is handed the LogRecord, not a formatted line, so each front end splits by level and
resolves its own streams AT EMIT TIME -- the rule core/tool/logging_console.py already states, and
the one ``basicConfig``'s handler broke.

Standard library only (so ``python -m core --help`` stays torch-free), and no logger's level is
touched anywhere in this module: core/runs.py owns the ``core`` logger's level, once, at import.
"""
import logging


class _DropCore(logging.Filter):
    """False for a record from the ``core`` logger or any of its children: it has its own handlers."""

    def filter(self, record: logging.LogRecord) -> bool:
        return not (record.name == "core" or record.name.startswith("core."))


class _SinkHandler(logging.Handler):
    """Hands every record that survives ``_DropCore`` to ``sink(record)``.

    No level of its own (NOTSET): whether a library emits a record at all is that library's logger's
    business, and the root logger's own WARNING level already gates ``logging.warning``.
    """

    def __init__(self, sink):
        super().__init__()
        self._sink = sink
        self.addFilter(_DropCore())

    def emit(self, record: logging.LogRecord) -> None:
        # ``except Exception``, deliberately not BaseException: under the window a library record is
        # written through ``_SignalStream.write``, which is a cancel checkpoint like every other
        # write, and the WorkerCancelled it raises must sail through to Worker.run.
        try:
            self._sink(record)
        except Exception:                     # noqa: BLE001 -- logging's contract: emit never raises
            self.handleError(record)


_installed: "_SinkHandler | None" = None


def install(sink) -> None:
    """Install THE root handler for this process. A second install replaces the first."""
    global _installed
    remove()
    _installed = _SinkHandler(sink)
    logging.getLogger().addHandler(_installed)


def remove() -> None:
    """Remove it if one is installed. Idempotent, so a ``finally`` can always call it."""
    global _installed
    if _installed is not None:
        logging.getLogger().removeHandler(_installed)
        _installed = None


def installed() -> bool:
    """True between ``install`` and ``remove``."""
    return _installed is not None
```

- [ ] **Step 4: Run the three new `tests/test_refusals.py` tests**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_refusals.py -k "root_handler or caplog_still_sees or logging_root_imports" -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Write the failing test for the window's sink**

Append to the end of `tests/test_worker_dispatch.py`:

```python
# ── piece 4, B14: the window's root sink ─────────────────────────────────────────────────────────
def test_the_windows_root_sink_prefixes_a_library_record_and_resolves_its_stream_late(capsys):
    """Below WARNING to ``sys.stdout``, at WARNING and above to ``sys.stderr``, BOTH RESOLVED AT EMIT
    TIME, as ``library: <logger name>: <message>``.

    Resolving late is the FIX, not the bug: what broke was basicConfig's handler binding the stream at
    CONSTRUCTION and then writing into a stopped pump for the rest of the process, and
    core/tool/logging_console.py already states the resolve-late rule for exactly this reason.
    Resolving late also solves three problems at once with no plumbing -- DURING A RUN those two names
    ARE the run's ``_SignalStream``s, which are thread-safe by construction and feed the pump, so a
    library record arriving on the WORKER thread never touches a widget (a ``log_pane.append_line``
    from the handler would have), and outside a run they are the real console.

    The level split is not cosmetic: ``_SignalStream`` on err is hard-wired to the pane's ``warning``
    level, so routing an information record through stderr would put a warning triangle on it.

    ``logging.warning`` (the module-level function, on the ROOT logger) is sbi's own shape -- its
    leakage warnings at sbi/samplers/rejection/rejection.py:336,359 are that call -- and its logger
    name is ``root``, which names nothing. So for that ONE name the prefix falls back to
    ``record.module``, the basename of the file that logged: ``library: rejection:`` under sbi, and
    this test file's own stem here. Walkthrough row D15 expects that module name.
    """
    import logging

    from core import logging_root
    from core.gui.app import _library_record_sink
    from tests._fixtures import pump, qt_app

    app = qt_app()
    lib = logging.getLogger("a_library")
    lib.setLevel(logging.INFO)          # a library that lowers its own level; the root sits at WARNING
    logging_root.install(_library_record_sink)
    try:
        capsys.readouterr()
        lib.info("an information record")
        logging.warning("Only 0.5% proposal samples are accepted.")
        cap = capsys.readouterr()
        here = Path(__file__).stem          # record.module: the file that called logging.warning
        assert cap.out == "library: a_library: an information record\n", cap.out
        assert cap.err == f"library: {here}: Only 0.5% proposal samples are accepted.\n", cap.err

        capsys.readouterr()
        logging.getLogger("core.probe").warning("the pipeline's own voice")
        cap = capsys.readouterr()
        assert (cap.out, cap.err) == ("", ""), cap

        signals = WorkerSignals()
        lines = []
        signals.log_batch.connect(lambda batch: lines.extend(batch))
        signals.rows.connect(lambda _s: None)
        with redirect_streams(signals):
            lib.warning("a leaky posterior")
            lib.info("and a quiet note")
        pump(app)
        assert ("library: a_library: a leaky posterior", "warning") in lines, lines
        assert ("library: a_library: and a quiet note", "info") in lines, lines
    finally:
        logging_root.remove()
        lib.setLevel(logging.NOTSET)
```

- [ ] **Step 6: Run it and watch it fail**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_worker_dispatch.py -k windows_root_sink -v`
Expected: FAIL, `ImportError: cannot import name '_library_record_sink' from 'core.gui.app'`

- [ ] **Step 7: Add the window's sink and install it in `build_app`**

In `core/gui/app.py`, change the imports (lines 3-11) to add `logging` and `logging_root`:

```python
import logging
import sys
import traceback

from PySide6.QtWidgets import QApplication, QMessageBox

from core import config, logging_root, registry
```

Add the sink at module level, directly after `_install_excepthook` (after line 33's `sys.excepthook = _excepthook`, before `def build_app`):

```python
def _library_record_sink(record) -> None:
    """One record from a logger OUTSIDE the ``core`` tree, on the console, named by its logger (B14).

    The stream is resolved AT EMIT TIME, never at construction, and that is the fix rather than the
    bug: what broke was ``logging.basicConfig``'s handler binding ``sys.stderr`` at construction and
    then writing into a stopped pump for the rest of the process. core/tool/logging_console.py states
    the same rule for the tool's two handlers.

    Resolving late also solves three problems at once and adds no plumbing. DURING A RUN these two
    names ARE the run's ``_SignalStream``s (core/gui/streams.py), which are thread-safe by
    construction and feed the pump -- so a library record arriving on the WORKER thread never touches
    a widget, which a ``log_pane.append_line`` from the handler would have done -- and outside a run
    they are the real console.

    The level split exists because ``_SignalStream`` on err is hard-wired to the pane's ``warning``
    level: routing an information record through stderr would put a warning triangle on it.

    Named by its logger, EXCEPT for ``root``. The trigger this handler exists for is
    ``logging.warning`` -- the module-level function, whose logger is the root logger -- so
    ``record.name`` would read ``root`` and name nothing. For that one name the prefix falls back to
    ``record.module``, the basename of the file that logged, and sbi's leakage warning therefore reads
    ``library: rejection:`` (sbi/samplers/rejection/rejection.py). Walkthrough row D15 expects that
    module name.
    """
    who = record.name if record.name != "root" else record.module
    stream = sys.stderr if record.levelno >= logging.WARNING else sys.stdout
    stream.write(f"library: {who}: {record.getMessage()}\n")
    stream.flush()
```

Make the install the FIRST statement of `build_app` (before the `config.QUIET_SEGMENT_BAR` line at 40):

```python
def build_app(argv=None):
    # THE root-logger handler, before anything else in this process can log (B14, spec §7.1). A
    # handler on the root FROM START-UP is what stops a library's logging.warning from calling
    # basicConfig and installing a second one -- after which every ``core`` record would be emitted
    # twice, the second copy into whatever stream that handler captured at construction. Never
    # removed: the window owns the process for as long as it lives.
    logging_root.install(_library_record_sink)
```

- [ ] **Step 8: Run the window's sink test**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_worker_dispatch.py -k windows_root_sink -v`
Expected: PASS

- [ ] **Step 9: Make the one `build_app` test clean up the handler it now installs**

`tests/test_user_models.py:936-967` is the only test that calls `build_app`. Without this, the handler it installs leaks for the rest of the pytest process and every later test's library records would be written to whatever `sys.stdout`/`sys.stderr` happens to be then. Add the assertion inside the `with` (after line 963's `assert (tmp_path / "Artifacts").is_dir()`) and the removal in the existing `finally`:

```python
            assert (tmp_path / "Artifacts").is_dir()
            from core import logging_root
            assert logging_root.installed() is True, \
                "build_app must install THE root-logger handler at start-up (piece 4, B14)"
            window.close()
        assert default_store() is before, "build_app's store install leaked past the test"
    finally:
        core_config.QUIET_SEGMENT_BAR = saved_quiet
        # build_app never removes it (the window owns the process); a test must, or every later test
        # in this process gets a handler writing library records into its captured streams.
        from core import logging_root
        logging_root.remove()
```

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_user_models.py -k build_app_starts -v`
Expected: PASS

- [ ] **Step 10: Write the failing test for the tool's sink**

Append to the end of `tests/test_tool.py`:

```python
# ── piece 4, B14: the tool's root sink ───────────────────────────────────────────────────────────
def test_main_installs_the_root_handler_for_the_run_and_leaves_nothing_behind(tool_env, monkeypatch,
                                                                             capsys):
    """Every record from a logger OUTSIDE the ``core`` tree goes to STDERR, whatever its level,
    prefixed with the logger that said it. Deliberately unlike the window's level split: this tool's
    STDOUT carries results a script reads (the GPU recipe in CLAUDE.md greps ``[checkpoint] resuming
    at batch`` off it), so a library's chatter may never land there.

    ``logging.warning`` -- the module-level function, on the ROOT logger -- is sbi's own shape
    (sbi/samplers/rejection/rejection.py:336,359), and ``root`` names nothing, so for that ONE logger
    name the prefix falls back to ``record.module``, the basename of the file that logged:
    ``library: rejection:`` under sbi, and this test file's own stem here (the ``logging.warning``
    below is called from ``_prior``, which lives in this file). Walkthrough row D15 expects the module
    name.

    Two more things are pinned here, both of which only a repeated-call test can see: the handler is
    installed FOR the handler call and removed in a finally (``main`` runs dozens of times in one
    process under this suite), and a ``core`` record still appears exactly once, on stdout, through
    ``console_handlers`` alone -- the doubling this handler exists to prevent would show up as the
    same line on stderr as well."""
    import logging

    from core import logging_root, orchestrator
    bounds, cell, root = tool_env
    lib = logging.getLogger("a_library")
    lib.setLevel(logging.INFO)
    root_logger = logging.getLogger()
    before = root_logger.handlers[:]

    def _prior(cfg, ref, build_new, **kw):
        assert logging_root.installed() is True, "the handler must exist FOR the handler call"
        logging.warning("Only 0.5% proposal samples are accepted.")
        lib.info("a library information record")
        logging.getLogger("core.orchestrator").info("[budget] 2 batches x 8 rows")
        return _art(root, "prior")

    monkeypatch.setattr(orchestrator, "build_prior", _prior)
    here = Path(__file__).stem              # record.module: the file that called logging.warning
    try:
        for _ in range(2):
            capsys.readouterr()
            assert main(["prior", *_cfg(bounds)]) == 0
            cap = capsys.readouterr()
            assert cap.err.count(
                f"library: {here}: Only 0.5% proposal samples are accepted.\n") == 1, cap.err
            assert cap.err.count("library: a_library: a library information record\n") == 1, cap.err
            assert "library:" not in cap.out, f"a library record reached the results stream:\n{cap.out}"
            assert cap.out.count("[budget] 2 batches x 8 rows\n") == 1, cap.out
            assert "[budget]" not in cap.err, cap.err
            assert logging_root.installed() is False, "main left its root handler installed"
            assert root_logger.handlers == before, root_logger.handlers
    finally:
        logging_root.remove()
        lib.setLevel(logging.NOTSET)
```

- [ ] **Step 11: Run it and watch it fail**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_tool.py -k main_installs_the_root_handler -v`
Expected: FAIL, `AssertionError: the handler must exist FOR the handler call` (raised inside `_prior`, reported by the tool's `*** FAILED ***` rung, so `main` returns 1 and the `== 0` assertion fails)

- [ ] **Step 12: Add the tool's sink and install it around the handler call**

In `core/tool/__init__.py`, add the import beside the other torch-free ones (after line 16's `from core.refusals import Refusal`):

```python
from core import logging_root
from core.refusals import Refusal
```

Add the sink at module level, directly before `def main` (after `_remove_if_still_empty`, line 78):

```python
def _library_record_sink(record) -> None:
    """One record from a logger OUTSIDE the ``core`` tree, on STDERR, named by its logger (B14).

    Everything, whatever its level, and never stdout: this tool's stdout carries results a script
    reads, so a library's chatter may not land there. Resolved AT EMIT TIME for the reason
    logging_console.py gives -- capsys swaps the streams per test, and a handler holding the stream it
    was built with writes into a buffer that is nobody's.

    Named by its logger, EXCEPT for ``root``: the trigger this handler exists for is
    ``logging.warning``, the module-level function, whose logger is the root logger, so
    ``record.name`` would read ``root`` and name nothing. There the prefix falls back to
    ``record.module``, the basename of the file that logged -- ``library: rejection:`` under sbi
    (sbi/samplers/rejection/rejection.py), which is what walkthrough row D15 expects.
    """
    who = record.name if record.name != "root" else record.module
    sys.stderr.write(f"library: {who}: {record.getMessage()}\n")
    sys.stderr.flush()
```

In `main`, install between `rc = 1` (line 101) and the existing `try:` (line 102):

```python
    rc = 1
    # THE root-logger handler for this process (piece 4, B14, spec §7.1). Because a handler exists
    # from here on, a library's ``logging.warning`` can no longer call ``basicConfig`` and install a
    # second one, which would emit every ``core`` record a second time. AFTER the parse, so ``--help``
    # stays torch-free; REMOVED IN THE FINALLY below, because ``main`` runs repeatedly in one process
    # under the suite and a handler left behind would repeat every later run's library records.
    logging_root.install(_library_record_sink)
    try:
```

and add the `finally` to that same ladder — after the last `except Exception:` clause (lines 152-155), before `if auto_root and rc != 0:` (line 156):

```python
    except Exception:                          # noqa: BLE001 -- a bug: show the whole thing
        traceback.print_exc()
        print(f"prism {args.cmd}: *** FAILED ***", file=sys.stderr)
        rc = 1
    finally:
        logging_root.remove()
    if auto_root and rc != 0:
```

- [ ] **Step 13: Run the task's own tests**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_refusals.py tests/test_worker_dispatch.py tests/test_user_models.py -q`
Expected: PASS

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_tool.py -m "not slow" -q`
Expected: PASS

- [ ] **Step 14: Commit**

```bash
git add core/logging_root.py core/gui/app.py core/tool/__init__.py tests/test_refusals.py tests/test_worker_dispatch.py tests/test_tool.py tests/test_user_models.py
git commit -m "both front ends install THE root-logger handler at start-up"
```

---

### Task 21: `cancel_deferred`

**Files:**
- Modify: `core/runs.py:1-26` (docstring) and `core/runs.py:146-147` (the two new functions, inserted between `capture_run` and `public_entry`)
- Modify: `core/gui/streams.py:24` (the `from core import runs` line: its `# noqa: F401` and its "never touched here" clause both stop being true once Step 4 lands), `core/gui/streams.py:191-200` (`_SignalStream.write`'s check) and `core/gui/streams.py:221-241` (`_PumpLogHandler`'s docstring and `emit`'s check)
- Modify: `core/SBI/training_checkpoint.py:40-51` (the ORDERING docstring) and `core/SBI/training_checkpoint.py:325-343` (`save`)
- Test: `tests/test_worker_dispatch.py` (two new tests appended at the end)

**Interfaces:**
- Consumes: existing `core.gui.streams.CancelToken` (`requested` is an `Event` set from the GUI thread, `fired` is a one-shot latch, `check()` raises `WorkerCancelled` only on the armed thread), `_SignalStream.write`, `_PumpLogHandler.emit`, `core.SBI.training_checkpoint.save(path, *, from_batch, batch_k, rng, x_buf, th_buf, run_size)`.
- Produces: `core.runs.cancel_deferred()` (a `@contextmanager`, re-entrant, thread-local) and `core.runs.cancel_is_deferred() -> bool`. Task 22 wraps the pipeline's rescue block in the first of these.

**Why this task exists:** `gui.streams._PumpLogHandler.emit` and `_SignalStream.write` call `CancelToken.check()` before doing anything else, and `check()` raises `WorkerCancelled` (a `BaseException`, deliberately) when the token is requested-but-not-yet-fired on the worker thread. So a cancel can land between two writes that must both happen, and the only protection today is a convention: `core/SBI/training_checkpoint.py:48-50` says "Do not `print()` or log between steps 1 and 3", policed by a source-reading test (`tests/test_user_sbi.py:4115`). B15 makes it a mechanism instead: a named critical section inside which the front end's cancel checkpoint does not fire, so the token stays REQUESTED and the next check outside the block raises as usual. It defers a cancel; it never discards one. The rule and the source-reading test both stay — the section is the guard, the test is the proof that the guard is where it is claimed to be.

- [ ] **Step 1: Write the failing test**

Append to the end of `tests/test_worker_dispatch.py`:

```python
# ── piece 4, B15: the deferred-cancel critical section ───────────────────────────────────────────
def test_cancel_deferred_defers_a_cancel_and_never_discards_it():
    """Inside the section neither cancel checkpoint fires -- ``_SignalStream.write`` (every print and
    every tqdm redraw) and ``_PumpLogHandler.emit`` (every ``core`` record) -- and the records still
    flow. What must NOT happen is the cancel being lost: the token stays REQUESTED, its latch stays
    unfired, and the very next check OUTSIDE the block raises exactly as it would have.

    Re-entrant, because the section nests: Task 22 puts the pipeline's rescue block inside one and the
    ``training_checkpoint.save`` it calls opens another, so a plain boolean would be cleared by the
    inner block's exit and leave the rest of the outer block unprotected."""
    import logging

    import pytest

    from core import runs
    from core.gui.streams import CancelToken, WorkerCancelled, _PumpLogHandler, _SignalStream

    class _FakePump:
        def __init__(self):
            self.logs = []

        def sink(self, kind, payload):
            self.logs.append((kind, payload))

    token = CancelToken()
    token.arm()                                # redirect_streams does this on the worker thread
    token.requested.set()                      # Cancel pressed, latch not yet fired
    fake = _FakePump()
    stream = _SignalStream(fake, "out", "info", token)
    handler = _PumpLogHandler(fake, token)
    record = logging.LogRecord("core.probe", logging.INFO, __file__, 1, "inside the commit", (), None)

    assert runs.cancel_is_deferred() is False, "no section is active outside one"
    with runs.cancel_deferred():
        assert runs.cancel_is_deferred() is True
        with runs.cancel_deferred():           # nested: the section must count, not toggle
            stream.write("a line written mid-commit\n")
            handler.emit(record)
        assert runs.cancel_is_deferred() is True, "the inner block's exit ended the outer section"
        stream.write("and another, still inside the outer section\n")
        assert token.fired is False and token.requested.is_set(), \
            "the cancel fired inside the section instead of being deferred"
    assert runs.cancel_is_deferred() is False

    assert ("log", ("inside the commit", "info")) in fake.logs, fake.logs
    assert any(kind == "log" and "a line written mid-commit" in payload[0]
               for kind, payload in fake.logs), fake.logs

    with pytest.raises(WorkerCancelled):
        stream.write("the first write after the section\n")
    assert token.fired is True, "the deferred cancel was discarded instead of deferred"


def test_the_checkpoint_commit_runs_inside_the_deferred_cancel_section(tmp_path, monkeypatch):
    """Steps 1-3 of a checkpoint save -- the shard writes, the state.prev copy and the atomic replace
    of state.pt -- are the two-writes-that-must-both-happen case the section exists for: a record
    emitted between the shard fsync and the state replace would raise mid-commit and leave a
    checkpoint pointing at data still in the page cache. The rule in CLAUDE.md and in
    training_checkpoint's module docstring stays, and so does the source-reading test that polices it
    (tests/test_user_sbi.py::test_nothing_prints_or_logs_inside_a_checkpoint_commit): the section is
    the guard, that test is the proof the guard is where it is claimed to be.

    ``_refresh_manifest`` is deliberately OUTSIDE the section: it is the store's view of the cache,
    never its commit point, and it is best-effort already."""
    from core import runs
    from core.SBI import training_checkpoint as tc

    seen = []
    real = tc.atomic_torch_save
    monkeypatch.setattr(tc, "atomic_torch_save",
                        lambda obj, dest: seen.append(runs.cancel_is_deferred()) or real(obj, dest))
    tc.save(tmp_path / "commit", from_batch=0, batch_k=1, rng={},
            x_buf=torch.zeros(2, 3), th_buf=torch.zeros(2, 2), run_size=2)
    assert seen == [True, True, True], \
        f"the two shard writes and the state replace must all be inside the section: {seen}"
    assert runs.cancel_is_deferred() is False, "the section leaked past the commit"
```

- [ ] **Step 2: Run it and watch it fail**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_worker_dispatch.py -k cancel_deferred -v`
Expected: FAIL, `AttributeError: module 'core.runs' has no attribute 'cancel_is_deferred'`

- [ ] **Step 3: Add the section to `core/runs.py`**

Insert between `capture_run` (ends at line 146) and `def public_entry` (line 148):

```python
_deferred = threading.local()                  # _deferred.depth: how many sections are open here


def cancel_is_deferred() -> bool:
    """True inside a ``cancel_deferred()`` block ON THIS THREAD. Read by the window's two cancel
    checkpoints (``gui.streams._SignalStream.write`` and ``_PumpLogHandler.emit``) before they call
    ``CancelToken.check()``."""
    return getattr(_deferred, "depth", 0) > 0


@contextmanager
def cancel_deferred():
    """Inside this block a front end's cancel checkpoint does not fire: records still flow, the token
    stays REQUESTED, and the next check outside the block raises as usual.

    It exists for the unwind path and the checkpoint commit -- the two places where raising between
    two writes loses committed work. It defers a cancel; it never discards one.

    PER THREAD and COUNTED. Per thread, because the token only ever raises on the armed (worker)
    thread and a GUI-thread print must not be silenced by a worker's section. Counted, because the
    sections nest: the pipeline's rescue block opens one and the ``training_checkpoint.save`` it calls
    opens another, and a boolean would leave the rest of the outer block unprotected.
    """
    _deferred.depth = getattr(_deferred, "depth", 0) + 1
    try:
        yield
    finally:
        _deferred.depth -= 1
```

Add a third bullet to the module docstring, after the RUN LOG bullet (line 20, before the closing paragraph at line 22):

```
- THE DEFERRED CANCEL (piece 4, B15). `cancel_deferred()` is the named critical section inside which
  the window's cancel checkpoints do not fire, so a Cancel pressed between two writes that must both
  happen -- the checkpoint commit, the training rescue save -- is deferred rather than taken. The
  token stays requested and the next check outside the section raises as usual.
```

- [ ] **Step 4: Make both cancel checkpoints consult it**

First the import at the top of `core/gui/streams.py` (line 24), which currently reads:

```python
from core import runs  # noqa: F401 -- sets the ``core`` logger to INFO at import (spec §1.2); never touched here
```

After this step the module really does call `runs.cancel_is_deferred()`, so both halves of that
comment are false — the `F401` waiver is no longer needed and "never touched here" is wrong. Replace
it with:

```python
from core import runs  # also sets the ``core`` logger to INFO at import (spec §1.2)
```

Then `_SignalStream.write` (lines 194-200) becomes:

```python
        # The cancel check sits BEFORE the try below: WorkerCancelled is a BaseException, so that
        # `except Exception` would not catch it anyway, but keeping it outside makes the intent explicit
        # -- a cancel is not a "parser broke" degradation. Every print() and every tqdm redraw funnels
        # through here, so this is the pipeline's cancellation checkpoint, reaching even inside sbi's
        # fit loop (it prints an epoch counter every epoch).
        # ...except inside a runs.cancel_deferred() section (piece 4, B15), where raising would land
        # between two writes that must both happen. The token is NOT cleared there: it stays requested
        # and the next write outside the section raises.
        if self._cancel is not None and not runs.cancel_is_deferred():
            self._cancel.check()
```

and `_PumpLogHandler.emit` (lines 239-241) becomes:

```python
    def emit(self, record):
        if self._cancel is not None and not runs.cancel_is_deferred():
            self._cancel.check()
```

Replace the last paragraph of `_PumpLogHandler`'s docstring (lines 228-231) with:

```python
    It is the cancel checkpoint too, exactly as _SignalStream.write is -- a run that only logs must
    still stop at its next message, and the token's latch (one raise, then quiet) holds here as
    well. That is why training_checkpoint's ordering rule reads "do not print() or log between steps
    1 and 3". Since piece 4 (B15) that rule is also a MECHANISM: the commit runs inside
    runs.cancel_deferred(), and both checkpoints consult it before raising, so a record emitted
    between a shard's fsync and the state replace is carried rather than fatal.
```

- [ ] **Step 5: Put the checkpoint commit inside the section**

In `core/SBI/training_checkpoint.py`, add the import after line 59 (`from core.Helpers.file_manager import atomic_torch_save`):

```python
from core.Helpers.file_manager import atomic_torch_save
from core.runs import cancel_deferred
```

Wrap steps 1-3 of `save` (lines 333-343):

```python
    path = Path(path)
    (path / _SHARDS).mkdir(parents=True, exist_ok=True)
    lo, hi = from_batch * run_size, batch_k * run_size
    # Steps 1-3, inside the deferred-cancel section (piece 4, B15): a GUI cancel landing between the
    # shard fsync and the state replace would commit a batches_done that points at data still in the
    # page cache. _refresh_manifest stays OUTSIDE it -- the manifest is the store's view of the cache,
    # never its commit point, and it is best-effort already.
    with cancel_deferred():
        if hi > lo:
            atomic_torch_save(x_buf[lo:hi].clone(), _shard(path, "x", from_batch, batch_k))
            atomic_torch_save(th_buf[lo:hi].clone(), _shard(path, "th", from_batch, batch_k))
        prev = path / _STATE
        if prev.exists():
            shutil.copyfile(prev, path / _STATE_PREV)
        atomic_torch_save({"batches_done": int(batch_k), "complete": False, "rng": rng}, prev)
    _refresh_manifest(path, batches_done=batch_k)
```

Extend the ORDERING paragraph of the module docstring (replacing lines 48-50):

```
Do not ``print()`` or log between steps 1 and 3 under the GUI: every write funnels through
``gui.streams._SignalStream.write`` and every record through ``gui.streams._PumpLogHandler.emit``,
both of which call ``CancelToken.check()`` and would raise mid-commit. Since piece 4 (B15) ``save``
runs steps 1-3 inside ``core.runs.cancel_deferred()``, so both checkpoints carry such a line instead
of raising on it -- the rule is still the rule, and the section is what enforces it.
```

- [ ] **Step 6: Run the task's own tests**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_worker_dispatch.py tests/test_refusals.py -q`
Expected: PASS

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_user_sbi.py -k "checkpoint_commit or resumed_training_run_is_bit_identical" -q`
Expected: PASS — the commit's no-speaking pin and the resume drill both still hold

- [ ] **Step 7: Commit**

```bash
git add core/runs.py core/gui/streams.py core/SBI/training_checkpoint.py tests/test_worker_dispatch.py
git commit -m "a named deferred-cancel section, and the checkpoint commit runs inside it"
```

---

### Task 22: the rescue save runs, and no failure is reported as a clean cancel

**Files:**
- Modify: `core/SBI/pipeline.py:16-22` (the import) and `core/SBI/pipeline.py:1771-1800` (the `except BaseException` rescue block)
- Modify: `core/gui/worker.py:11-22` (a new chain walker beside `_drop_tracebacks`) and `core/gui/worker.py:58-99` (`Worker.run`) — no new import: `traceback` and `WorkerCancelled` are both already there
- Test: `tests/test_user_sbi.py:3782-3798` (the source pin, extended) and a new test appended at the end of the file
- Test: `tests/test_worker_dispatch.py` (one new test appended at the end, three legs)

**Interfaces:**
- Consumes: `core.runs.cancel_deferred()` from Task 21. Existing: `core.gui.streams.CancelToken`/`WorkerCancelled`, `core.gui.worker._drop_tracebacks(exc)`, `WorkerSignals.error(object, str)` / `.cancelled()` / `.log(str, str)`, `BasePanel._on_error(exc_or_message, tb)` (a `Refusal` to the yellow box, anything else to the red one with `tb` in Details; it also appends the message to the pane at `error`), `BasePanel.dispatch`'s `cancelled` connection (the pane's `"Run cancelled."` line).
- Produces: the rescue block's guarantee that a crash with a pending cancel still commits its completed batches and re-raises the ORIGINAL exception (which the window then reports through its ordinary crash path, unchanged); `core.gui.worker._pending_failure(cancel) -> "BaseException | None"`, used only to LOG what was in flight behind a cancel, never to report it as the run's failure.

**Why this task exists:** `core/SBI/pipeline.py`'s rescue block logs before it saves, and **two** of its calls can raise, not one: the `log.warning` at `:1784` is outside the inner `try` altogether, and the `log.info` at `:1792` is inside a `try` whose only handler is `except Exception` while `WorkerCancelled` derives from `BaseException`. So a Cancel pressed just before a crash makes the block's own announcement raise: `_tc.save` is skipped, the re-raise at `:1800` is never reached, and up to `TRAINING_CHECKPOINT_EVERY - 1` batches are lost. The block's comment (`:1788-1791`) asserts this is safe because the latch is one-shot and "the raise has already happened" — true only when the cancel is what unwound the run, which is precisely not this case. Worse, `Worker.run` catches `WorkerCancelled` **by name** (`core/gui/worker.py:65`) and reports a cancellation: no error signal, no dialog, no traceback — **you are told you cancelled a run that crashed.** B15: "The training rescue save runs inside a **deferred-cancel critical section**, so a pending cancel can no longer skip it; the window reports the **original exception** with the cancel noted."

The section is what delivers that second clause, and it delivers it with **no special case in the window**: with the rescue write inside `cancel_deferred()`, the checkpoint the block crosses does not fire, `_tc.save` completes, and the `raise` at the end of the block re-raises the ORIGINAL exception. That reaches `Worker.run`'s generic `except Exception` and is reported exactly as any other crash is — the red box, its traceback in Details, the message in the pane. The cancel branch is **not** where the collision is fixed, and B15's "with the cancel noted" therefore lands on the residual race below rather than on the red box: a crash is reported as a crash, with nothing about a cancel added, because such a note could only be read off the exception chain — which is precisely what must not be trusted.

What the cancel branch must still handle is the residual race: a cancel checkpoint **outside** any section firing while something is already unwinding. There it reports a **cancel**, and when the chain holds a non-cancel exception it emits that exception's whole traceback to the pane at ERROR and says a failure was in flight. It deliberately does NOT report the chained exception as the failure: a cancel raised inside a **recovered** `except` block carries the handled exception as its `__context__` — the OOM ladders log from exactly there — so reporting the chain would open a red box for an OOM the run survived. Nothing is discarded, and no recovered exception ever produces a dialog. No gate can provoke either case (the smoke gate does not crash), so these tests are the only proof.

- [ ] **Step 1: Write the failing test for the rescue write**

Append to the end of `tests/test_user_sbi.py`:

```python
def test_a_crash_with_a_cancel_pending_still_saves_the_rows_and_raises_the_original(monkeypatch):
    """THE collision no gate can provoke (piece 4, B15). The run crashes with the cancel token
    REQUESTED AND NOT YET FIRED -- Cancel pressed in the moment before the failure -- and the rescue
    block's own announcement is then the next cancel checkpoint. Before B15 that announcement raised
    WorkerCancelled from inside the ``except BaseException`` handler: ``_tc.save`` was skipped, the
    re-raise at the end of the block was never reached, and every batch since the last cadence write
    was lost while the window reported a clean cancellation.

    Both halves of the repair are pinned here: the rows are committed, AND the original exception is
    what escapes. The second half is what lets the window report the crash through its ordinary crash
    path with no special case in ``Worker.run``
    (tests/test_worker_dispatch.py::test_a_cancel_never_reports_a_failure_as_a_clean_stop, leg (a)).

    Driven through the window's THIRD channel rather than a stream swap: ``_PumpLogHandler`` is the
    handler the window puts on the ``core`` logger for a run, and it checks the same token the streams
    do, so installing it alone makes the pipeline's records the ONLY cancel checkpoint in the process
    -- which is what makes this deterministic. ``sys.stdout``/``sys.stderr`` are untouched, so no tqdm
    teardown write can consume the latch on the way out; the assertion on ``token.fired`` below is
    what keeps that true, and a future seam change fails loudly instead of quietly measuring nothing.

    Cadence 2 over 4 batches: [0,2) is committed at the end of batch 1, batch 2 completes uncommitted,
    and batch 3 crashes -- so the rescue write owes exactly batch 2 and ``batches_done`` must be 3.
    """
    import logging
    import tempfile

    from core.gui.streams import CancelToken, WorkerCancelled, _PumpLogHandler
    from core.SBI import training_checkpoint as tc

    tmp = Path(tempfile.mkdtemp()) / "rescue"
    token = CancelToken()
    token.arm()                                # redirect_streams does this on the worker thread

    class _FakePump:
        def __init__(self):
            self.logs = []

        def sink(self, kind, payload):
            self.logs.append(payload)

    fake = _FakePump()
    handler = _PumpLogHandler(fake, token)
    core_logger = logging.getLogger("core")
    real = pipeline_mod.gen_stats
    seen = {"n": -1, "last": None}

    def _cancel_then_crash(x_spont, *a, **k):
        if seen["last"] is not pipeline_mod._BATCH_TAG:
            seen["last"] = pipeline_mod._BATCH_TAG
            seen["n"] += 1
        if seen["n"] >= 3:
            token.requested.set()              # Cancel pressed; the latch has NOT fired
            raise _KillRun("the batch failed")
        return real(x_spont, *a, **k)

    monkeypatch.setattr(pipeline_mod, "gen_stats", _cancel_then_crash)
    core_logger.addHandler(handler)
    try:
        with pytest.raises(_KillRun):
            _gen_td("chi", seed=11, n_runs=4, run_size=2, checkpoint=_ck(tmp, resume="never"))

        assert token.requested.is_set() and token.fired is False, (
            "the rescue block's own record was NOT the next cancel checkpoint -- something else "
            "consumed the latch on the way out, so this test is measuring nothing")
        st = tc.peek(tmp)
        assert st and st["batches_done"] == 3, (
            f"the rescue write did not run: {st}. The cadence write covered [0,2); batch 2 completed "
            f"and must be committed on the way out even with a cancel requested")
        assert any("[checkpoint] stopping: saving 1 completed batches" in text
                   for text, _level in fake.logs), fake.logs

        # DEFERRED, never discarded: the first record after the section still raises.
        with pytest.raises(WorkerCancelled):
            logging.getLogger("core.probe").info("the first record after the rescue write")
        assert token.fired is True
    finally:
        core_logger.removeHandler(handler)
```

- [ ] **Step 2: Run it and watch it fail**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest "tests/test_user_sbi.py::test_a_crash_with_a_cancel_pending_still_saves_the_rows_and_raises_the_original" -v`
Expected: FAIL — `Failed: DID NOT RAISE` is *not* what you see; you get
`core.gui.streams.WorkerCancelled` escaping `_gen_td` instead of `_KillRun`, reported by `pytest.raises(_KillRun)` as
`WorkerCancelled` (the rescue block's `log.info` raised before `_tc.save` ran)

- [ ] **Step 3: Put the rescue block inside the section**

In `core/SBI/pipeline.py`, add the import after line 20 (`from core.refusals import Refusal`):

```python
from core.refusals import Refusal
from core.runs import cancel_deferred
```

Replace the rescue block (lines 1771-1800) with — note that the three messages keep their text and
their levels exactly, so the message table in `tests/test_user_sbi.py:4029-4055` needs no edit:

```python
    except BaseException:
        # WorkerCancelled (a BaseException by design, so `except Exception` would miss it) and
        # KeyboardInterrupt both land here, and a GUI cancel is the MOST likely way a multi-day run
        # ends -- MainWindow.closeEvent reaches request_cancel_all(), so closing the window stops it.
        # Before this, that discarded every completed batch.
        # THE ROWS MATTER MORE THAN THE RESTORE POINT. If the snapshot for THIS batch failed or
        # belongs to another one, commit the completed batches with no RNG rather than skipping the
        # write: a checkpoint that resumes without restoring the streams draws fresh noise from
        # batch_k onward -- statistically equivalent, the same licence the OOM ladders take -- while
        # a skipped write throws away hours of simulation outright.
        #
        # INSIDE A DEFERRED-CANCEL SECTION (piece 4, B15), because TWO of the calls below can raise,
        # not one: the log.warning is outside the inner try altogether, and the log.info is inside one
        # whose only handler is `except Exception` while WorkerCancelled is a BaseException. Either
        # would raise under a cancel that is REQUESTED AND NOT YET FIRED -- Cancel pressed in the
        # moment before a crash -- skipping _tc.save and the re-raise below and losing every batch
        # since the last cadence write. The comment this replaces claimed the log was safe because the
        # latch is one-shot and "the raise has already happened": true only when the cancel is what
        # unwound the run, which is exactly not this case. The section defers that cancel; it does not
        # discard it -- the token stays requested and the next check outside this block raises.
        with cancel_deferred():
            if _ck_dir is not None and batch_k > _ck_from:
                _rescue_rng = _pending_rng if _pending_rng_at == batch_k else None
                if _rescue_rng is None:
                    log.warning(f"[checkpoint] no valid RNG snapshot for batch {batch_k}; saving the rows "
                                f"without a restore point (a resume will draw fresh noise from there)")
                try:
                    # Announced BEFORE the write, so a multi-second flush is not an unexplained hang
                    # after Cancel. Nothing is printed or logged BETWEEN the shard fsync and the state
                    # replace -- see training_checkpoint, whose commit opens a section of its own.
                    log.info(f"[checkpoint] stopping: saving {batch_k - _ck_from} completed batches "
                             f"({_ck_from} -> {batch_k}) before unwinding…")
                    _tc.save(_ck_dir, from_batch=_ck_from, batch_k=batch_k,
                             rng=_rescue_rng, x_buf=x_buf, th_buf=th_buf, run_size=run_size)
                except Exception as _e:              # noqa: BLE001
                    # A failed rescue write must never REPLACE the cancel/crash with an I/O error. It is
                    # an ERROR record: a failure reported rather than raised.
                    log.error(f"[checkpoint] could not save on the way out: {_e}")
        raise                                    # UNCONDITIONAL: never swallow a cancel
```

- [ ] **Step 4: Run the rescue test and the two pins that move with it**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_user_sbi.py -k "cancel_pending_still_saves or failed_snapshot_never_writes or every_sbi_message_is_a_record" -v`
Expected: PASS (the new test and `test_every_sbi_message_is_a_record_at_its_level_and_nothing_prints`, whose three rescue-block rows keep their text; `test_a_failed_snapshot_never_writes_a_stale_restore_point` still passes because `ast.unparse` leaves its three needles untouched)

- [ ] **Step 5: Extend the source pin to the section**

In `tests/test_user_sbi.py`, `test_a_failed_snapshot_never_writes_a_stale_restore_point` (lines 3782-3798): add to the docstring and append the new assertions after line 3798.

Docstring, after the existing second paragraph:

```
    Also pinned, since piece 4 (B15): the whole rescue write sits inside a ``cancel_deferred()``
    section, so neither of its two log calls can raise between the completed batches and the write
    that commits them. Needled on ``rng=_rescue_rng`` rather than on the call's opening, because the
    CADENCE write two screens up is ``_tc.save(_ck_dir, from_batch=_ck_from, batch_k=batch_k + 1``
    and would match a shorter needle first.
```

Assertions, appended to the body:

```python
    i_section = src.find("with cancel_deferred():")
    i_guard = src.find("if _ck_dir is not None and batch_k > _ck_from:")
    i_save = src.find("rng=_rescue_rng")
    assert i_section != -1, "the rescue write is no longer inside a cancel_deferred() section"
    assert i_section < i_guard < i_save, (i_section, i_guard, i_save)
```

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_user_sbi.py -k failed_snapshot_never_writes -v`
Expected: PASS

- [ ] **Step 6: Write the failing test for the window's report**

Append to the end of `tests/test_worker_dispatch.py`:

```python
def test_a_cancel_never_reports_a_failure_as_a_clean_stop():
    """Three shapes, one branch (piece 4, B15).

    (a) THE COLLISION -- Cancel pressed in the moment before a crash. The pipeline's rescue write now
    runs inside ``runs.cancel_deferred()`` (Step 3), so the checkpoint it crosses does not fire, the
    rows are committed, and the ORIGINAL exception keeps propagating. It reaches ``Worker.run``'s
    generic handler and is reported like any other crash: the red box, its traceback in Details, the
    message in the pane, and NO cancellation. The window needs no special case for this, and this leg
    is the pin that keeps that true. The fn below is the rescue block's SHAPE, not the pipeline: a
    token requested-and-not-yet-fired, a record and a print inside the section, then a bare ``raise``.
    The section is per thread and the fn runs on the worker thread, so the token is armed there too.
    No tqdm bar is live, deliberately (P35): a bar's teardown write would consume the latch on the way
    out, and the ``fired is False`` assertion is what stops this leg from measuring nothing.

    (b) THE RESIDUAL RACE, and why the chain is not the verdict. A cancel raised inside a RECOVERED
    ``except`` block -- which is exactly where the OOM ladders log -- carries the handled exception as
    its ``__context__``. Reporting the first non-cancel link as THE failure would therefore open a red
    box for an OOM the run survived. So this stays a CANCEL: no error signal, no dialog. Nothing is
    discarded either -- what was in flight goes to the pane at ERROR, whole traceback, under a warning
    line saying a failure was in flight.

    (c) A CLEAN cancel -- nothing chained -- is exactly what it was: one ``Run cancelled.`` line.

    Whether the run was reported as a cancel is read off the PANE, not off a signal this test
    connects: ``BasePanel.dispatch`` connects ``cancelled`` to the pane's ``Run cancelled.`` line
    before the worker starts, so there is no window in which a fast worker could emit before the test
    is listening.
    """
    import logging

    from PySide6.QtWidgets import QMessageBox

    from core import runs
    from core.gui.streams import CancelToken, WorkerCancelled, _PumpLogHandler, _SignalStream
    from tests._fixtures import SHOWN, PaneCapture, pump, qt_app

    app = qt_app()

    class P(BasePanel):
        pass

    panel = P()
    pane = PaneCapture(panel)

    class _FakePump:
        def __init__(self):
            self.logs = []

        def sink(self, kind, payload):
            self.logs.append((kind, payload))

    token, fake = CancelToken(), _FakePump()
    rescue = logging.LogRecord("core.SBI.pipeline", logging.INFO, __file__, 1,
                               "[checkpoint] stopping: saving 1 completed batches", (), None)

    def crash_behind_the_section():
        token.arm()                            # the fn runs on the worker thread, and so must the
        token.requested.set()                  # section: Cancel pressed just before the failure
        try:
            raise RuntimeError("the batch failed")
        except RuntimeError:
            with runs.cancel_deferred():       # what pipeline.py's rescue block now opens
                _PumpLogHandler(fake, token).emit(rescue)
                _SignalStream(fake, "out", "info", token).write("a print beside the rescue write\n")
            raise                              # the ORIGINAL exception, unconditionally

    def cancel_inside_a_recovered_handler():
        try:
            raise RuntimeError("the OOM the ladder survived")
        except RuntimeError:
            # The ladder logs from HERE and carries on, and that log is a cancel checkpoint -- so the
            # cancel it raises carries the handled, already-dealt-with exception as its __context__.
            raise WorkerCancelled()

    def clean_cancel():
        raise WorkerCancelled()

    def _run(fn, wait_for_dialog):
        SHOWN.clear()
        del pane.lines[:]                      # each leg reads only its own lines
        panel.dispatch(fn)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and (panel._busy or (wait_for_dialog and not SHOWN)):
            app.processEvents()
            time.sleep(0.01)
        pump(app, 0.2)
        assert not panel._busy, "the panel stayed busy"

    stop = ("warning", "Run cancelled.")

    # (a) the collision: a crash, reported as a crash
    _run(crash_behind_the_section, True)
    assert stop not in pane.lines, "a run that CRASHED was reported as a cancellation"
    assert token.requested.is_set() and token.fired is False, (
        "the section did not defer the cancel -- something else consumed the latch, so this leg is "
        "measuring nothing")
    said = [payload for kind, payload in fake.logs if kind == "log"]
    assert ("[checkpoint] stopping: saving 1 completed batches", "info") in said, fake.logs
    assert any("a print beside the rescue write" in text for text, _lv in said), fake.logs
    box = SHOWN[-1]
    assert box.windowTitle() == "Error" and box.icon() == QMessageBox.Critical
    assert box.text() == "the batch failed", box.text()
    assert "RuntimeError: the batch failed" in box.detailedText(), box.detailedText()
    assert pane.lines == [("error", "the batch failed")], \
        "a crash is reported by the ordinary crash path and by nothing else"

    # (b) the residual race: a cancel, with what was in flight kept
    _run(cancel_inside_a_recovered_handler, False)
    assert pane.lines.count(stop) == 1, ("a cancel must still be reported as a cancel", pane.lines)
    assert SHOWN == [], "a recovered exception behind a cancel opened an error dialog"
    assert any(lv == "warning" and "failure was in flight" in t for lv, t in pane.lines), pane.lines
    assert any(lv == "error" and "RuntimeError: the OOM the ladder survived" in t
               for lv, t in pane.lines), pane.lines
    assert pane.lines[-1] == stop, pane.lines

    # (c) a clean cancel: unchanged
    _run(clean_cancel, False)
    assert pane.lines == [stop] and SHOWN == [], (pane.lines, SHOWN)
```

- [ ] **Step 7: Run it and watch it fail**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_worker_dispatch.py -k never_reports_a_failure_as_a_clean_stop -v`
Expected: FAIL at leg (b), `AssertionError: [('warning', 'Run cancelled.')]` — the pane has no
"failure was in flight" line and no ERROR traceback, because `Worker.run` reports the cancel and drops
the chain on the floor. Leg (a) passes on the way there, and that is the finding rather than a slip:
Step 3's section is what makes the original exception reach the generic handler, so the window needs no
change for the collision and leg (a) is its regression pin.

- [ ] **Step 8: Report a cancel as a cancel, and never discard what was in flight**

In `core/gui/worker.py`, add the chain walker after `_drop_tracebacks` (line 22):

```python
def _pending_failure(cancel: BaseException) -> "BaseException | None":
    """The first non-cancel exception on ``cancel``'s ``__cause__``/``__context__`` chain, or None.

    A cancel checkpoint (a print, a tqdm redraw, a ``core`` record) that fires while something is
    already unwinding raises WorkerCancelled CHAINED to it, so the chain is the only place that
    exception still exists once the cancel has replaced it.

    It is looked up to be LOGGED, never to decide the run's verdict. The chain cannot tell a failure
    that was still propagating from one the code had already dealt with: a cancel raised inside a
    RECOVERED ``except`` block -- where the OOM ladders log -- carries the handled exception as its
    context, so treating a chained exception as THE failure would open a red box for an OOM the run
    survived. THE collision (a Cancel in the moment before a crash) is not fixed here at all: the
    pipeline's rescue write runs inside ``runs.cancel_deferred()``, so the original exception
    propagates by itself and lands in Worker.run's generic handler.

    ``__cause__`` first (an explicit ``raise ... from``), then ``__context__``. A chain can loop, so
    each link is visited once.
    """
    seen, e = set(), cancel
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        e = e.__cause__ if e.__cause__ is not None else e.__context__
        if e is not None and not isinstance(e, WorkerCancelled):
            return e
    return None
```

Rewrite `Worker.run` (lines 58-99) — only the `WorkerCancelled` branch and the dispatch below the
`with` change:

```python
    @Slot()
    def run(self):
        payload, failure, cancelled, in_flight = None, None, False, None
        try:
            with redirect_streams(self.signals, self.cancel):
                try:
                    payload = self.fn(*self.args, **self.kwargs)
                except WorkerCancelled as cancel:
                    # A cooperative cancel -- caught by name (BaseException, so it skipped the generic
                    # handler below). ALWAYS reported as a cancel: no traceback, no error dialog.
                    #
                    # THE COLLISION IS NOT HANDLED HERE. A Cancel pressed in the moment before a crash
                    # is handled where the work is: the pipeline's rescue save runs inside
                    # runs.cancel_deferred() (piece 4, B15), so the checkpoint it crosses does not
                    # fire, the rows are committed, and the ORIGINAL exception keeps propagating --
                    # into the generic handler below, which reports it like any other crash.
                    #
                    # What is left is the residual race: a checkpoint OUTSIDE any section firing while
                    # something is already unwinding. Python chains it, so __context__ still holds
                    # what was in flight, and that is NOT discarded -- its whole traceback goes to the
                    # log pane at ERROR below. It is deliberately not reported as THE failure: a
                    # cancel raised inside a RECOVERED `except` block (the OOM ladders log from
                    # exactly there) carries the handled exception as its context, so reporting the
                    # chain would open a red box for an OOM the run survived.
                    cancelled = True
                    original = _pending_failure(cancel)
                    if original is not None:
                        in_flight = "".join(traceback.format_exception(
                            type(original), original, original.__traceback__))
                    # ...and then the frames go, for the reason the Exception branch gives below.
                    # _drop_tracebacks walks the chain, so `original` is covered by this one call.
                    _drop_tracebacks(cancel)
                except Exception as e:               # noqa: BLE001 -- surface any failure to the UI
                    # The EXCEPTION, not its text. The panel opens the yellow "Check your inputs"
                    # box for a Refusal and the red one with the traceback for anything else, and
                    # it can only tell the two apart if the object itself crosses the thread.
                    failure = (e, traceback.format_exc())
                    # ...but not its traceback. The traceback owns every frame of the failed run
                    # (the stage's host buffers, the prior, CUDA tensors), and this frame heads it
                    # while `failure` holds the exception: a cycle only a full collection frees.
                    # The panel needs the type, message, field and the text formatted above.
                    _drop_tracebacks(e)
                finally:
                    # Stray figures a stage built but never handed to the sink (e.g. it unwound on a
                    # cancel before _emit): harmless under Agg, but they pile up across cancelled runs.
                    try:
                        import matplotlib.pyplot as plt
                        plt.close("all")
                    except Exception:                # noqa: BLE001 -- cleanup must not mask the outcome
                        pass

            # Everything below runs with sys.stdout/stderr already restored and the pump drained and
            # stopped, so (a) every line the pipeline produced -- including a leave=True bar's final
            # frame, which is only flushed on teardown -- is queued AHEAD of the result, and (b) the
            # modal dialog that _on_error opens cannot spin a nested event loop while the process's
            # streams are still swapped out from under it. The in-flight note and its traceback are
            # emitted HERE, not inside the `with`, for the same ordering reason -- and BEFORE
            # cancelled.emit(), so the panel's "Run cancelled." stays the last line of the run.
            if cancelled:
                if in_flight is not None:
                    self.signals.log.emit("Cancelled while a failure was already in flight. The "
                                          "cancel did not cause it; its traceback follows.", "warning")
                    self.signals.log.emit(in_flight, "error")
                self.signals.cancelled.emit()
            elif failure is None:
                self.signals.result.emit(payload)
            else:
                self.signals.error.emit(*failure)
                failure = None                       # the queued signal holds its own reference
```

- [ ] **Step 9: Run the task's own tests**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_worker_dispatch.py -q`
Expected: PASS

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_user_sbi.py -m "not slow" -q`
Expected: PASS

- [ ] **Step 10: Commit**

```bash
git add core/SBI/pipeline.py core/gui/worker.py tests/test_user_sbi.py tests/test_worker_dispatch.py
git commit -m "the rescue save survives a pending cancel, and no failure is reported as a stop"
```

---

### Task 23: a new probe row starts blank

**Files:**
- Modify: `core/gui/widgets/labeled_inputs.py:13-17` (`FloatField.__init__`)
- Modify: `core/gui/panels/inference/rows.py:37-40` (`_ChiProbeRow.__init__`) and `core/gui/panels/inference/rows.py:59-74` (`problems`' comment)
- Modify: `core/gui/panels/inference/infer_tab.py:214-226` (`_add_chi_probe`) and `core/gui/panels/inference/infer_tab.py:322-327` (the planner's blank sentence)
- Test: `tests/test_nav_and_gating.py:635-688` (rewritten in this commit) plus two new tests appended after it (before line 690)

**Interfaces:**
- Consumes: nothing from earlier tasks. Existing: `FloatField.value_or_none()` (None for anything that does not parse — the rule M2b's advice is superseded by), `_probe_frequency(row)` (`core/gui/panels/inference/infer_tab.py:41-45`, None for a blank, half-typed or non-positive box), `core.SBI.chi.probe_verdict` (`core/SBI/chi.py:133`, the shared source of truth for the planner and the runtime), `Refusal(..., field="recording_probe")`.
- Produces: `FloatField(None)` meaning an empty box; `_ChiProbeRow(on_remove, freq_hz=None, parent=None)`; `InferPanel._add_chi_probe(freq_hz=None)`; one wording for a blank drive frequency across the tab and the planner.

**Why this task exists:** `_ChiProbeRow.__init__` defaults `freq_hz=0.0` and `FloatField(0.0)` writes the text `"0.0"`, so every row the χ table seeds arrives holding a frequency nobody typed — and 0 Hz is a genuine DC probe the lock-in would attempt. The row then reports it as `probe N: drive frequency must be a positive number (got 0)`, which describes a value the user never entered. One state has three sentences today: that one, the planner's `no frequency entered` (`infer_tab.py:326`), and `core/SBI/chi.py:159`'s `drive frequency must be finite and positive, got 0.0 Hz`. B16: "A newly added chi probe row's frequency box is **blank**, and the wordings for a blank frequency are unified across the tab and the planner." Only the BLANK sentence is unified; the typed-zero sentences stay as they are at both layers, because a zero somebody typed is a different state from a box nobody filled, and V2's whole point is that a message describes what happened. The key stays `recording_probe` — a new key would cost five coordinated edits for a sentence the existing key already answers.

- [ ] **Step 1: Write the failing tests**

First, rewrite the existing test. In `tests/test_nav_and_gating.py`, replace the docstring's BLANK FREQUENCY paragraph and lines 661-670 of
`test_chi_probe_table_survives_a_config_rebuild_and_rejects_a_blank_frequency`. The whole test becomes:

```python
def test_chi_probe_table_survives_a_config_rebuild_and_rejects_a_blank_frequency(tmp_path):
    """Two C-2 constraints in one place, because both are about data the GUI cannot regenerate.

    PRESERVATION: rows carry hand-typed drive frequencies and browsed paths -- a record of a bench
    session that already happened. Rebuilding the config (to fix a bounds file, say) must not discard
    them, unlike the forcing rows, which ARE derivable from the config.

    BLANK FREQUENCY: a SEEDED row is blank (piece 4, B16) and is said to be blank. It used to arrive
    holding "0.0" -- FloatField's own default -- and be refused as "must be a positive number (got
    0)", a sentence about a value nobody entered, while 0 Hz is a genuine DC probe the lock-in would
    happily attempt. A TYPED zero keeps that sentence, because a zero somebody typed is a different
    state from a box nobody filled.
    """
    from core.gui.screens.inference_screen import InferenceScreen

    qt_app()
    inf = InferenceScreen()
    inf.install_config(_chi_cfg(k=2))
    panel = inf.infer_panel
    panel._chi_forced_fields[0].path.edit.setText("/tmp/keep.csv")
    panel._chi_forced_fields[0].freq.setText("2.5")

    inf.install_config(_chi_cfg(k=2))                         # rebuild
    assert len(panel._chi_forced_fields) == 2, "a rebuild must not add or drop rows"
    assert panel._chi_forced_fields[0].pair() == ("/tmp/keep.csv", 2.5), \
        "a rebuild destroyed hand-entered probe data"

    # Row 2 was SEEDED and never typed into: its box is empty and it is reported as blank.
    assert panel._chi_forced_fields[1].freq.text() == "", panel._chi_forced_fields[1].freq.text()
    probs = [p for i, r in enumerate(panel._chi_forced_fields) for p in r.problems(i)]
    assert "probe 2: drive frequency is blank" in probs, probs
    assert not any("got 0" in p for p in probs), probs
    assert not any("probe 1" in p for p in probs), probs

    # A zero the user TYPED is a different state, and keeps its own sentence.
    panel._chi_forced_fields[1].freq.setText("0")
    probs = [p for i, r in enumerate(panel._chi_forced_fields) for p in r.problems(i)]
    assert any("probe 2" in p and "positive" in p and "got 0" in p for p in probs), probs
    assert not any("is blank" in p for p in probs), probs

    # ...and at the click, the yellow box's sentence is the blank one again
    panel._chi_forced_fields[1].freq.setText("")
    inf.session.posterior = _posterior_stub()
    refused, sent = [], {}
    panel._refusal = lambda exc: refused.append(exc)
    panel.dispatch = lambda fn, *a, **k: sent.update(fn=fn)
    for name in ("passive.npy", "probe0.npy", "probe1.npy"):
        (tmp_path / name).touch()
    panel.infer_mode.setCurrentIndex(1)                          # experimental: the chi page
    panel.chi_tobs.setText("2.0")
    panel.chi_f0_si.setText("1.0")
    panel.chi_spont.edit.setText(str(tmp_path / "passive.npy"))
    for i, row in enumerate(panel._chi_forced_fields):
        row.path.edit.setText(str(tmp_path / f"probe{i}.npy"))
    panel._infer()
    assert sent == {} and len(refused) == 1 and refused[-1].field == "recording_probe", (sent, refused)
    assert "probe 2: drive frequency is blank" in refused[-1].message, refused[-1].message
    assert "got 0" not in refused[-1].message, refused[-1].message
```

Then add two new tests directly after it (before `test_config_units_control_declares_units_and_validates_them`):

```python
def test_a_new_probe_row_starts_blank_and_floatfield_accepts_none():
    """``FloatField(None)`` is an EMPTY box; every other default still shows its number, including the
    several call sites that hand it a pre-formatted string. Nothing persists a probe row
    (infer_tab.save_settings says so), so this touches only the in-session seed -- and a seed the user
    has to fill is the point: the frequency is entered, never derived (see _ChiProbeRow's docstring),
    because the frequencies a bench achieves are not exactly mult_k * Omega_0."""
    from core.gui.panels.inference.rows import _ChiProbeRow
    from core.gui.screens.inference_screen import InferenceScreen
    from core.gui.widgets.labeled_inputs import FloatField
    from tests._fixtures import qt_app

    qt_app()
    assert FloatField(None).text() == ""
    assert FloatField(None).value_or_none() is None
    assert FloatField(None).value() == 0.0, "value() keeps its 0.0 fallback: value_or_none() is the rule"
    assert FloatField(0.0).text() == "0.0" and FloatField(2.5).text() == "2.5"
    assert FloatField("0.33").text() == "0.33", "the string call sites (config/prior/posterior tabs) stand"

    assert _ChiProbeRow(lambda _row: None).freq.text() == ""
    assert _ChiProbeRow(lambda _row: None, 7.5).freq.text() == "7.5", "an explicit seed still seeds"

    inf = InferenceScreen()
    inf.install_config(_chi_cfg(k=3))
    panel = inf.infer_panel
    assert len(panel._chi_forced_fields) == 3
    assert [r.freq.text() for r in panel._chi_forced_fields] == ["", "", ""]
    row = panel._add_chi_probe()
    assert row is not None and row.freq.text() == "" and row.freq.value_or_none() is None
    assert panel._add_chi_probe(12.25).freq.text() == "12.25"


def test_the_planner_and_the_probe_row_give_a_blank_frequency_one_sentence():
    """B16's second half. One state had two wordings a user could meet minutes apart: the tab's
    ``probe N: drive frequency is blank`` at the click, and "Plan probes…"'s ``no frequency entered``.
    The tab's is the one, which also makes the planner's own "Filled N blank frequency box(es)" line
    literally true. The TYPED-ZERO sentences are deliberately untouched at both layers -- the tab's
    "(got 0)" and chi.probe_verdict's "must be finite and positive, got 0.0 Hz".

    Pinned on the row's real output and on the planner's PARSED source (code_only, so the comments in
    that region cannot answer for the code). The planner's branch is defensive: the nominal-grid fill
    a few lines above it covers every blank box, so nothing reaches it unless the grid comes back
    shorter than the list of blanks."""
    from core.gui.panels.inference.infer_tab import InferPanel
    from core.gui.panels.inference.rows import _ChiProbeRow
    from core.SBI import chi as chi_mod
    from tests._fixtures import code_only, qt_app

    qt_app()
    blank = _ChiProbeRow(lambda _row: None).problems(0)
    assert blank == ["probe 1: no recording selected", "probe 1: drive frequency is blank"], blank

    src = code_only(InferPanel._plan_chi_probes)
    assert "drive frequency is blank" in src, "the planner still has a wording of its own for a blank"
    assert "no frequency entered" not in src, "the planner's old wording is still there"
    assert "Filled" in src and "blank frequency box" in src

    # the typed-zero sentences, both layers, unchanged
    typed = _ChiProbeRow(lambda _row: None, 0.0).problems(0)
    assert "probe 1: drive frequency must be a positive number (got 0)" in typed, typed
    assert "must be finite and positive, got" in code_only(chi_mod.probe_verdict)
```

- [ ] **Step 2: Run them and watch them fail**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_nav_and_gating.py -k "chi_probe_table_survives or new_probe_row_starts_blank or planner_and_the_probe_row" -v`
Expected: FAIL — three failures:
`AssertionError: 0.0` (the seeded row's box still holds `"0.0"`),
`AssertionError: assert FloatField(None).text() == ""` (it is `"None"`),
`AssertionError: the planner's old wording is still there`

- [ ] **Step 3: `FloatField(None)` means an empty box**

`core/gui/widgets/labeled_inputs.py`, lines 13-17:

```python
class FloatField(QLineEdit):
    def __init__(self, default: "float | None" = 0.0, parent=None):
        # None is an EMPTY box, not the text "None": a seeded value nobody typed is a different state
        # from a blank one, and value_or_none() is what tells them apart (piece 4, B16). Several
        # callers hand this a pre-formatted string instead of a float, which str() leaves alone.
        super().__init__("" if default is None else str(default), parent)
        self.setValidator(QDoubleValidator())
        self.setMinimumWidth(FIELD_MIN_W)
```

- [ ] **Step 4: A probe row and `_add_chi_probe` default to blank**

`core/gui/panels/inference/rows.py`, line 37:

```python
    def __init__(self, on_remove, freq_hz: "float | None" = None, parent=None):
```

and its `problems` comment (lines 66-68) becomes:

```python
        # A row the table SEEDED has an empty box (piece 4, B16, FloatField(None)) -- and 0 Hz is a
        # genuine DC probe the lock-in would happily attempt, so the two states must not share a
        # sentence. Read through value_or_none() (V2): a blank box is said to be blank, and only a
        # zero somebody typed is reported as "got 0". This is the check that stops a typo becoming a
        # measurement.
```

`core/gui/panels/inference/infer_tab.py`, line 214:

```python
    def _add_chi_probe(self, freq_hz: "float | None" = None):
        """Append one probe row, up to the posterior's slot capacity. The frequency box starts BLANK
        (piece 4, B16): the frequency is entered, never derived, so a seeded number would be a claim
        about the bench that nobody made. "Plan probes…" fills the blanks with a nominal in-band grid
        on request, and says they are suggestions."""
```

- [ ] **Step 5: One sentence for a blank frequency**

`core/gui/panels/inference/infer_tab.py`, lines 323-327 — the verdict loop's blank branch:

```python
        for i, row in enumerate(self._chi_forced_fields):
            f = _probe_frequency(row)
            if f is None:
                # The TAB's sentence, verbatim (piece 4, B16): the same state the click refuses with
                # ``probe N: drive frequency is blank`` must not be described differently here.
                self.log_pane.append_line(f"  probe {i + 1}: drive frequency is blank.", "warning")
                continue
```

- [ ] **Step 6: Run the task's own tests**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_nav_and_gating.py -q`
Expected: PASS

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_settings_persistence.py tests/test_user_models.py -q`
Expected: PASS (the numeric-field width scan at `tests/test_settings_persistence.py:584-595` walks every `FloatField`, and `tests/test_user_models.py:836` builds a bare `_ChiProbeRow`)

- [ ] **Step 7: Commit**

```bash
git add core/gui/widgets/labeled_inputs.py core/gui/panels/inference/rows.py core/gui/panels/inference/infer_tab.py tests/test_nav_and_gating.py
git commit -m "a new chi probe row starts blank, with one wording for a blank frequency"
```

### Task 24: the two untested payload guards in `load_observation`

**Files:**
- Modify: `core/artifacts/store.py:727-730` (the payload-width guard gains `field="observation"`)
- Test: `tests/test_artifact_store.py` — append at the end of the file (3238 lines today; if Task 6
  landed first, append after what it added)

**Interfaces:**
- Consumes: nothing from any earlier task. `ArtifactStore.create` / `ArtifactWriter.payload`
  (`store.py:219`, `:415`), `ArtifactStore.path(kind, ref) -> Path` (`store.py:366`),
  `ArtifactStore.load_observation(cfg, ref)` (`store.py:682`),
  `manifest.conditioning_block(cfg)` and `manifest.tensor_digest(t)` (`manifest.py:96`, `:173`),
  `file_manager.atomic_torch_save(obj, path)` (`core/Helpers/file_manager.py:61`).
- Produces: `load_observation`'s payload-width `StoreError` now carries `field="observation"`, so both
  front ends name the observation picker for it from their existing tables
  (`core/gui/fields.py`, `core/tool/fields.py` already map the `observation` key;
  `core/refusals.py:108` registers it). No signature changes.

**Why this task exists:** Spec §8.1. `docs/STATE.md:185` records that `load_observation` has one
untested guard; there are **two**, side by side, and they differ in class, key and timing. The
config-vs-manifest width check (`store.py:715-717`) is a `Refusal(field="observation")` raised before
the payload is read and is tested (`tests/test_artifact_store.py:2596`). The payload's own width
against the manifest's (`:727-730`) is a `StoreError` with **no** field key raised after
`torch.load` — the recorded gap — and the payload-digest check immediately above it (`:724-726`) is
untested too and is recorded nowhere. These are the guards that answer "the bytes on disk are not the
bytes this manifest describes", which is the one failure every width guard above them cannot see,
because all of them compare the manifest.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_artifact_store.py`. `torch`, `pytest`, `mf`, `st` and `_nad_cfg` are already
imported at module level (lines 3-19, 144).

```python
# ── load_observation's two payload guards (piece 4, §8.1) ────────────────────────────────────────


def _obs_artifact(store, cfg, *, name, x_obs, digest=None):
    """An observation artifact whose payload is exactly the row handed in: the real writer, the real
    body, ``conditioning_block(cfg)`` for the geometry, and by default the row's own digest.

    Cheap ON PURPOSE. ``generate_observations`` simulates a trace and plots it, and neither guard
    below needs one: both fire after ``torch.load`` and before anything is used. ``digest`` records a
    digest OTHER than the row's, which is the only way to reach the digest guard without editing a
    committed manifest.
    """
    from core.Helpers import file_manager
    d = mf.tensor_digest(x_obs) if digest is None else digest
    with store.create("observation", cfg, name=name) as w:
        file_manager.atomic_torch_save({"x_obs": x_obs, "obs_data": torch.zeros(1, 4, dtype=torch.float64),
                                        "t_dim": torch.zeros(1, 4, dtype=torch.float64)},
                                       w.payload("observation.pt"))
        w.fingerprints["x_obs"] = d
        w.body = {"mode": cfg.observation_mode, "conditioning": mf.conditioning_block(cfg),
                  "x_obs_digest": d, "T_obs_cell": 1.0, "n_obs": 4, "forcing_vals": {},
                  "chi_obs_freqs": None, "source": {"kind": "simulated"}}
    return w


def test_load_observation_refuses_a_payload_that_disagrees_with_its_manifest(store):
    """Spec §8.1: the two guards AFTER ``torch.load``, each in isolation.

    Everything above them compares the MANIFEST -- the config's model, parameter order, mode, width and
    chi layout against what the manifest declares -- so a payload that is not the payload the manifest
    describes is invisible to all of them. The two are ordered: the digest first (the bytes are not the
    bytes), then the width (the shape is not the shape). Reaching the second therefore needs a row that
    hashes to what the manifest records, which is why the narrow artifact is written with its own
    digest rather than corrupted after the fact.

    The width guard carries ``field="observation"``, matching the Refusal above it; the digest guard
    carries none. That asymmetry is the spec's ruling, and pinning it is what makes changing it
    deliberate.
    """
    cfg = _nad_cfg()                                      # master.txt declares a drive -> forced mode
    W = int(mf.conditioning_block(cfg)["width"])
    _obs_artifact(store, cfg, name="intact", x_obs=torch.zeros(1, W, dtype=torch.float64))
    good = store.load_observation(cfg, "intact")
    assert good.width == W and int(good.x_obs.shape[-1]) == W, "the control leg: an intact artifact loads"

    # (1) the DIGEST guard. The manifest is untouched and the row on disk is the RIGHT width, so the
    # width guard below cannot be what fires -- only the bytes changed.
    payload = store.path("observation", "intact") / "observation.pt"
    torch.save({"x_obs": torch.ones(1, W, dtype=torch.float64),
                "obs_data": torch.zeros(1, 4, dtype=torch.float64),
                "t_dim": torch.zeros(1, 4, dtype=torch.float64)}, str(payload))
    with pytest.raises(st.StoreError, match="does not hash to the manifest's digest") as e:
        store.load_observation(cfg, "intact")
    assert "the artifact is inconsistent" in str(e.value) and "intact" in str(e.value)
    assert e.value.field is None, "no control answers 'the bytes on disk changed'"

    # (2) the WIDTH guard, reachable only once the digest agrees.
    _obs_artifact(store, cfg, name="narrow", x_obs=torch.zeros(1, W - 1, dtype=torch.float64))
    with pytest.raises(st.StoreError, match=f"holds a {W - 1}-wide conditioning row") as e:
        store.load_observation(cfg, "narrow")
    assert f"manifest declares {W}" in str(e.value)
    assert "compared the MANIFEST, not this row" in str(e.value)
    assert e.value.field == "observation", "the same key as the width Refusal above it"
    assert isinstance(e.value, Refusal), "a StoreError is a Refusal: the front ends route it as one"
```

- [ ] **Step 2: Run it and watch it fail**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_artifact_store.py::test_load_observation_refuses_a_payload_that_disagrees_with_its_manifest -v`

Expected: FAIL at the last-but-one assertion —
`AssertionError: the same key as the width Refusal above it` / `assert None == 'observation'`.
Everything before it passes: both guards already raise the right class with the right sentence, and
only the field key is missing.

- [ ] **Step 3: Give the payload-width guard its field key**

In `core/artifacts/store.py:727-730`, replace

```python
        if int(x_obs.shape[-1]) != int(cond["width"]):
            raise StoreError(f"observation '{label}': observation.pt holds a {int(x_obs.shape[-1])}-wide "
                             f"conditioning row but its manifest declares {int(cond['width'])}; the artifact "
                             f"is inconsistent (every width guard above compared the MANIFEST, not this row)")
```

with

```python
        if int(x_obs.shape[-1]) != int(cond["width"]):
            # field= like the width Refusal twelve lines up, and for the same reason: whichever of the
            # two fires, the operator answers it by picking another observation, and each front end's
            # table is what names that control (piece 4, §8.1).
            raise StoreError(f"observation '{label}': observation.pt holds a {int(x_obs.shape[-1])}-wide "
                             f"conditioning row but its manifest declares {int(cond['width'])}; the artifact "
                             f"is inconsistent (every width guard above compared the MANIFEST, not this row)",
                             field="observation")
```

The message is unchanged — the sentence is what the test above pins; only the key is new.

- [ ] **Step 4: Run the task's own tests**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_artifact_store.py -q -k "load_observation or loaders_mismatch or reinstalls_its_context"`

Expected: PASS (3 tests: the new one, the sibling field-key test at `:2596`, and the
`generate_observations` round trip that loads a real observation through the same path).

- [ ] **Step 5: Commit**

```bash
git add core/artifacts/store.py tests/test_artifact_store.py
git commit -m "store: pin load_observation's payload guards, field the width one"
```

### Task 25: the stale counts, and M1b's docstrings

**Files:**
- Modify: `core/gui/panels/base_panel.py:62`, `:69`, `:315`, `:321-322` (the `layout_key` docstring
  says "there are nine independent" at `:321` and "today all nine differ" at `:322` — **both** halves
  of that sentence are this task's, Q16)
- Modify: `core/gui/panels/inference/base.py:15`
- Modify: `core/gui/main_window.py:70-75`
- Test: `tests/test_nav_and_gating.py`

**Interfaces:**
- Consumes: nothing. Tasks 15 and 18 have already touched `base_panel.py` and `inference_screen.py`;
  re-read both files before editing, because their line numbers will have moved.
- Produces: nothing. Comments and docstrings only — no behaviour changes in this task.

**Why this task exists:** Four docstrings count the panels wrong and two count the inference tabs
wrong, all of them stale from before the TSNPE tab existed, and one claims a persistence rule the code
contradicts. Spec §8.2 puts the corrections in this piece because every one of those files is touched
by it; leaving them is how a reader learns to distrust the comments. There are **ten** `BasePanel`
subclasses (four section panels plus six inference tabs: Config, Prior, Posterior, Validate, Infer,
TSNPE), and the browser Task 9 added is **not** one of them (B1), so the count stays ten.

- [ ] **Step 1: Write the failing test**

A comment cannot be asserted by behaviour, so this is a source test — the repository's own idiom for
pinning a sentence (the pattern is
`tests/test_user_sbi.py::test_nothing_prints_or_logs_inside_a_checkpoint_commit`, which reads source and
asserts on it).

**It deliberately does not count `BasePanel.__subclasses__()`.** The suites define throwaway
`class P(BasePanel)` subclasses (`tests/test_worker_dispatch.py:54` and others) that persist for the
life of the interpreter, so a counting assertion would pass or fail by test order — wrong rather than
strict (ledger P37). The test asserts the absent and the present *phrases* instead, which is
order-independent. Add to `tests/test_nav_and_gating.py`:

```python
def test_the_panel_docstrings_no_longer_count_nine_panels_or_five_tabs():
    """Four sentences in base_panel.py and one in inference/base.py counted nine panels and five
    inference tabs, both stale from before the TSNPE tab: there are TEN BasePanel subclasses (the four
    section panels plus six inference tabs) and NINE save_settings overrides (ValidatePanel is the one
    subclass that does not override it). The Artifacts screen is not a panel at all -- it is a plain
    QWidget, so a run elsewhere cannot grey it out (piece 4, B1).

    Phrases, not a subclass count: the suites own throwaway BasePanel subclasses persist for the life
    of the process, so counting __subclasses__() would assert on test order (ledger P37).

    The source is whitespace-NORMALISED before every check. Two of these phrases are wrapped across a
    newline in the file ("there are nine independent / splitters"), so against raw source the literal
    would never appear, the assertion would pass before AND after the edit, and the stale sentence
    would survive behind a green test -- which is the one failure this test exists to prevent (Q16).
    """
    import core.gui.panels.base_panel as bp
    from core.gui.panels.inference import base as inf_base

    bp_src = " ".join(inspect.getsource(bp).split())
    inf_src = " ".join(inspect.getsource(inf_base).split())
    for phrase in ("Nine of these", "8 of the 9", "all nine differ", "nine independent splitters"):
        assert phrase not in bp_src, f"base_panel.py still says {phrase!r}"
    assert "the five inference tabs" not in inf_src
    assert "the five inference tabs" not in bp_src

    assert "Ten of these exist" in bp_src
    assert "9 of the 10" in bp_src
    assert "all ten differ" in bp_src
    assert "the six inference tabs" in inf_src


def test_main_window_stops_claiming_it_owns_the_only_settings_write():
    """The class docstring said _save_state is "the only QSettings WRITE site ... so panel selections
    and layouts persist from here". Layouts do not: BasePanel._persist_layout writes and sync()s the
    splitter state on a 1500 ms debounce off splitterMoved, deliberately, because save-on-clean-quit
    lost the drag (base_panel.py own comment says so). Appearance is the third write site.

    Whitespace-NORMALISED, for the same reason as the test above: the claim is wrapped as "the only
    QSettings / WRITE site" in the file, so against raw source this first assertion could never fail
    and would pass before and after the rewrite (Q16)."""
    from core.gui import main_window as mw

    src = " ".join(inspect.getsource(mw.MainWindow).split())
    assert "the only QSettings WRITE site" not in src
    assert "_persist_layout" in src and "1500" in src
```

`inspect` is already imported by that suite — check its import block and add it only if it is absent.

- [ ] **Step 2: Run it and watch it fail**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest "tests/test_nav_and_gating.py::test_the_panel_docstrings_no_longer_count_nine_panels_or_five_tabs" "tests/test_nav_and_gating.py::test_main_window_stops_claiming_it_owns_the_only_settings_write" -v`
Expected: both FAIL.

- `test_the_panel_docstrings_no_longer_count_nine_panels_or_five_tabs` fails on the FIRST phrase of
  the loop: `AssertionError: base_panel.py still says 'Nine of these'`.
- `test_main_window_stops_claiming_it_owns_the_only_settings_write` fails on its FIRST assertion,
  `assert 'the only QSettings WRITE site' not in src` — and it fails only because `src` is
  whitespace-normalised. Against the raw source that phrase is split across `main_window.py:72-73`
  and the assertion would pass here and after Step 5 alike; if you see this test fail on
  `assert "_persist_layout" in src` instead, the normalisation was left out (Q16).

- [ ] **Step 3: Correct `base_panel.py`'s four counts**

At `:62`, `Nine of these exist (Reduction, FDT, CrossVal, Simulate + the five inference tabs).` becomes:

```
    Ten of these exist (Reduction, FDT, CrossVal, Simulate + the six inference tabs: Config, Prior,
    Posterior, Validate, Infer, TSNPE). The Artifacts screen is NOT one -- it is a plain QWidget on
    purpose, because a BasePanel enrols in _instances and every run anywhere would grey it out (piece
    4, B1). Three class
```

At `:69`, `8 of the 9 subclasses` becomes `9 of the 10 subclasses` — nine is the real number of
`save_settings` overrides (every subclass but `ValidatePanel`; grep `def save_settings` under
`core/gui` and count); at `:315`, `8 of the 9 panels` becomes `9 of the 10 panels`.

At `:321-322`, `layout_key`'s docstring counts twice in one sentence and **both** halves change
(Q16) — it reads

```
        """Settings key for this panel's layout. Distinct per panel -- there are nine independent
        splitters. Subclasses that share a class name would override this; today all nine differ."""
```

and becomes

```
        """Settings key for this panel's layout. Distinct per panel -- there are ten independent
        splitters. Subclasses that share a class name would override this; today all ten differ."""
```

The line break falls inside "nine independent / splitters", which is why Step 1's test normalises
whitespace before looking for that phrase: correcting only the second half would leave a stale
"nine" behind a green test.

Re-read the surrounding sentences first and keep each one grammatical — these are explanations, not
counters, and the reason each gives must survive the edit.

- [ ] **Step 4: Correct the two inference-tab counts**

`core/gui/panels/inference/base.py:15`: `Common base for the five inference tabs:` becomes
`Common base for the six inference tabs:`.

`core/gui/screens/inference_screen.py`'s module docstring is **not** this task's: Task 18 edits that
file for the Apply confirmation and corrects its "five tabs" sentence and its workflow arrow there.
This task owns `base_panel.py`, `inference/base.py` and `main_window.py`'s class docstring, and nothing
else — one owner per sentence, so two tasks cannot fight over the same lines (ledger P38).

- [ ] **Step 5: Correct `MainWindow`'s persistence claim**

`core/gui/main_window.py:70-75` claims `_save_state` is "the only QSettings WRITE site … so panel
selections and layouts persist from here". That is wrong about layouts: `BasePanel._persist_layout`
writes and `sync()`s the splitter state on a 1500 ms debounce off `splitterMoved`, deliberately,
because save-on-clean-quit lost the drag (the comment at `base_panel.py:162-170` explains it). Rewrite:

```python
    """The application shell: a NavShell over Home plus the five section screens.

    Always opens on Home -- the last screen is deliberately not restored. Owns the QSettings write
    that runs at quit (``_save_state``, reached from closeEvent): panel selections, and the Artifacts
    screen's own (piece 4, B1, which is called by name because _all_panels() is panel-typed). Two
    other write sites exist on purpose: BasePanel._persist_layout writes a splitter position on a
    1500 ms debounce (save-on-clean-quit lost the drag -- see base_panel.py), and appearance settings
    are written eagerly, as they are applied immediately.
    """
```

- [ ] **Step 6: Run the task's own tests**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_nav_and_gating.py -q`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add core/gui/panels/base_panel.py core/gui/panels/inference/base.py core/gui/main_window.py tests/test_nav_and_gating.py
git commit -m "docs: the panel counts say ten, and MainWindow stops claiming the only write site"
```

---

### Task 26: the documents, the walkthrough rows, and the closing gates

**Files:**
- Modify: `docs/checklists/display-walkthrough.md` (append the piece-4 section)
- Modify: `CLAUDE.md:163` (the suite count and the new suite) and `:168` (the subcommand list)
- Modify: `docs/STATE.md` (the header, "Owed" item 7's piece-4 bullet, the gate table)
- Modify: `docs/superpowers/specs/2026-09-17-gui-usability-and-artifact-browser-design.md` §12
- Test: none of its own — this task's verification is the two gates it runs

**Interfaces:**
- Consumes: every earlier task. Read the ledger's per-task rows before writing §12, because the
  deviations table is a record of what execution actually found, not a restatement of the spec.
- Produces: nothing code depends on.

**Why this task exists:** `docs/STATE.md` is the moving state and is read first by every session;
`CLAUDE.md` states the suite count and the tool's subcommands and is loaded into every session's
context. Both go stale the moment this piece lands. The walkthrough rows are the owner's, and the
spec's deviations table is the piece's honest account of itself — piece 3's added-test overrun went
unrecorded, which is exactly what §9.4 of this spec exists to prevent.

- [ ] **Step 1: Append the walkthrough's piece-4 section**

Add to `docs/checklists/display-walkthrough.md`, after the piece-3 block, following that block's exact
shape (a heading, a short preamble saying who runs the rows and what they supersede, then the six-column
table `| # | surface | do | expect | date | result |` with the last two columns EMPTY). The sixteen
rows are spec §11's D1–D16; write each row's "do" as an instruction someone can follow without reading
the plan, and each "expect" as the observable outcome. State in the preamble that rows 1–20, A1–A9,
B1–B8 and C1–C11 stand, that piece 4 re-checks only what it changes, and that D16 re-runs row 1
because this piece adds start-up work to the shell header.

Leave `date` and `result` blank. **No task may fill them in** — they are the owner's, on a real screen.

- [ ] **Step 2: Update `CLAUDE.md`**

At `:163`, `the seventeen suites` becomes `the eighteen suites`, and the parenthetical gains
`` `test_artifact_browser.py` the artifact browser's `` beside the existing entries. At `:168`, the
subcommand list gains `artifacts` — read the line first; it currently reads "the stages `prior train
tsnpe validate infer`, the diagnostics `sbc identifiability ablation`, plus `smoke`, `fdt` and
`crossval`", so the new family goes in as `plus `artifacts` (list/show/note/rm/sweep/summary)`,
matching the file's own voice. Also add one sentence to the `Where things are` section naming the
Artifacts screen as the store's front end.

- [ ] **Step 3: Update `docs/STATE.md`**

Three places, in the file's own voice:
1. The header paragraph: piece 4 is DONE, its commit range, its spec and plan, what landed (one
   sentence per decision group: the browser and its twin, the three behaviours, the four hand-ons),
   and that the D rows are owed by the user.
2. "Owed" item 7's **Piece 4** bullet: strike through each item it listed and say where it landed,
   the way the piece-2 and piece-3 bullets do. Anything this piece did NOT take moves to the unowned
   list with its reason — §1.3 of the spec is the source for that list (the browser's declined extras,
   `rename` from the browser, `unnamed()`, the listing's cost, the cache's unsound rename).
3. The gate table: a row for the collect count, the final fast gate, the slow set, the GPU gate, and a
   row for the D-rows reading **"Not yet run."** until the user reports them.

- [ ] **Step 4: Fill in the spec's deviations table**

`docs/superpowers/specs/2026-09-17-gui-usability-and-artifact-browser-design.md` §12: one row per
ruling made during execution, from the ledger, each saying what was found, what was decided, and
**what it costs if the ruling is wrong**. Include the added-test count against §9.4's band of 130 ± 30
whether it lands inside or outside, and the judgement recorded in §10 about not running the diagnostic
card — with whether it held.

- [ ] **Step 5: Commit the documents**

```bash
git add docs/checklists/display-walkthrough.md CLAUDE.md docs/STATE.md docs/superpowers/specs/2026-09-17-gui-usability-and-artifact-browser-design.md
git commit -m "docs: piece 4 is done — STATE, the walkthrough's D rows, the deviations"
```

- [ ] **Step 6: The slow set (the orchestrator runs it, in the background)**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest -m slow -q --durations=5`
Expected: 2 passed, exit 0, roughly 23–31 minutes (23 min 13 s at `e78cc8d`). No
"More than 20 figures" line; `git status` clean and no `sbi-logs/` afterwards. Record the result in the
gate table.

- [ ] **Step 7: The GPU smoke gate (the orchestrator runs it, alone on the card)**

The four command lines of `CLAUDE.md`'s Tests section, verbatim, from the repository root with `$S` an
empty scratch directory. Pass criteria, per `CLAUDE.md`: runs 1 and 3 exit 0 with no OOM lines, stage
timings near the piece-3 gate of record (92/196/22/84 s and 92/60/13/18 s) and masked-probe counts
within ±12 pp of 37 %; run 2 prints `Reusing the Fisher rotation stored with the training checkpoint`
and `[checkpoint] resuming at batch 4/4` and exits 0; run 2b exits 1 naming `n_runs`, prints no
`[fisher]` line and adds no directory under `$S/smoke/simulations/`.

**Why it must run:** Task 21 changes when a cancel checkpoint fires on the training path, Task 22
changes the unwind, and Task 20 installs a root handler every training record passes. Also confirm
what §10 claims: `git diff` the piece over `core/diagnostics` and check that no changed line there
creates or moves a tensor; if one does, run the diagnostic card too and record it.

Delete `$S` afterwards and record the result in the gate table, then amend nothing — add the result in
a follow-up documents commit.

- [ ] **Step 8: Hand the D rows to the owner**

Report the piece as done except for rows D1–D16, name them, and ask the owner to run them on a real
screen. Record their answer in the walkthrough's last two columns and in the gate table, in its own
commit, as piece 3's C rows were.
