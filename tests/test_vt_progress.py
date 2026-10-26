"""The tqdm/VT100 router, a run's bars read as its model's rows and its lines as its text, the solver's count as a sign of life and the row that measures a run. Split from test_gui_progress.py; run directly: pytest tests/test_vt_progress.py"""
"""Progress-rendering regression tests for the GUI.

THE BUG THESE LOCK DOWN
    tqdm redraws a bar at pos>0 as three writes -- '\\n'*pos, then '\\r'+frame, then '\\x1b[A'*pos
    (tqdm/std.py:1493-1497). The old stream reader split on terminators, so the frame (which is never
    terminated) stranded in its buffer and was flushed by the NEXT redraw's leading '\\n' -- i.e. as a
    LOG LINE. Every nested-bar redraw appended one row, so a training run buried the log pane under
    hundreds of bar snapshots.

    The pipeline nests bars four deep (core/SBI/pipeline.py:517 -> :371 ->
    core/Simulator/simulator.py:50 -> core/Solvers/sdeint.py:15), so this fired constantly.

Run:  pytest tests/test_vt_progress.py
"""
import os
import tempfile
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")   # must precede any PySide6 import
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib                                                 # noqa: E402
matplotlib.use("Agg")                                            # match the app (core/gui/__main__.py forces it)

import torch                                                      # noqa: E402
from tqdm import tqdm                                             # noqa: E402

from core.gui.run.model import RunModel, RunSpec, Stage, driving_row, fit_counter   # noqa: E402
from core.gui.streams import redirect_streams                     # noqa: E402
from core.gui.vt import StreamRouter, parse_bar                   # noqa: E402
from core.gui.worker import WorkerSignals                         # noqa: E402
from tests._fixtures import qt_app, pump                          # noqa: E402
import contextlib                                                  # noqa: E402

# ── the router, driven by REAL tqdm (no Qt) ──────────────────────────────────────────────────────
def _drive(fn, level="err"):
    """Run `fn` with sys.stderr routed through a StreamRouter at the error stream's own level, as the
    window's redirect routes it; return its event list."""
    events = []

    class Stream:
        def write(self, s):
            if s:
                router.feed(s)
            return len(s)

        def flush(self):
            pass

        def isatty(self):
            return False

    router = StreamRouter("err", lambda k, p: events.append((k, p)), level=level)
    real, sys.stderr = sys.stderr, Stream()
    try:
        fn()
    finally:
        router.close()
        sys.stderr = real
    return events
def _live(events):
    rows = {}
    peak = 0
    for kind, payload in events:
        if kind == "row":
            rows[payload.key] = payload
        elif kind == "retire":
            rows.pop(payload, None)
        peak = max(peak, len(rows))
    return rows, peak


def test_four_deep_nest_emits_no_log_lines():
    """The regression: a 4-deep nest must produce ZERO log lines and exactly 4 concurrent rows."""
    def nest():
        for _ in tqdm(range(2), desc="Training neural posterior", leave=False):
            for _ in tqdm(range(3), desc="Generating training data", leave=False):
                for _ in tqdm(range(2), desc="Running time segments", leave=False):
                    for _ in tqdm(range(2), desc="step (batch=64)", leave=False):
                        pass

    events = _drive(nest)
    logs = [p for k, p in events if k == "log"]
    rows, peak = _live(events)

    assert logs == [], f"nested bars leaked {len(logs)} log line(s): {logs[:3]}"
    assert peak == 4, f"expected 4 concurrent rows, got {peak}"
    assert rows == {}, f"{len(rows)} row(s) survived close()"
    assert {p.row for k, p in events if k == "row"} == {0, 1, 2, 3}

def test_no_ansi_reaches_the_log():
    """colorama is installed, so tqdm's moveto(-n) emits real '\\x1b[A'. It is motion, not text."""
    def nest():
        for _ in tqdm(range(2), desc="outer", leave=False):
            for _ in tqdm(range(2), desc="inner", leave=False):
                pass

    for kind, payload in _drive(nest):
        text = payload[0] if kind == "log" else getattr(payload, "raw", "")
        assert "\x1b" in text is False or "\x1b" not in text, f"ANSI leaked: {text!r}"

def test_leave_true_bar_persists_to_log_and_leaves_no_ghost_row():
    """close(leave=True)'s trailing '\\n' (tqdm/std.py:1303) is byte-identical to a moveto(+1), so a
    new pos-0 bar opening right after one is initially GUESSED onto row 1. The absence of a following
    up-move then proves it belongs on row 0 and the router rekeys it.

    What must hold is the settled state: the sequence ends with exactly one row, at index 0, and the
    closed bar's final frame graduates into the log (as it does on a real terminal). The transient
    row-1 guess is invisible because the pump coalesces -- see the peak-row assertion below, which is
    what a widget actually sees."""
    def bars():
        for _ in tqdm(range(2), desc="Constructing latent prior..."):
            pass
        for _ in tqdm(range(2), desc="Campaign 2 (chi sweep)"):
            pass

    events = _drive(bars)
    rows, peak = _live(events)
    logs = [t for k, (t, _lvl) in ((k, p) for k, p in events if k == "log")]

    assert rows == {}, "a leave=True bar left a ghost row"
    assert peak == 1, f"the two sequential pos-0 bars were live at once ({peak} rows) -- the " \
                      f"leave=True finalizer newline was mistaken for a moveto"
    assert sum("100%" in t for t in logs) == 2, f"final frames not persisted to the log: {logs}"

def test_pump_never_exposes_the_transient_row_guess():
    """The user-visible guarantee: through the pump, two sequential pos-0 bars never render as two
    rows. The row-1 guess and its correction land inside one 15 Hz tick and are coalesced away."""
    app = qt_app()
    signals = WorkerSignals()
    snapshots = []
    signals.rows.connect(snapshots.append)
    # the payload is (text, level); this test asserts only on `rows` and drains the log channel
    signals.log_batch.connect(lambda _b: None)

    with redirect_streams(signals):
        for _ in tqdm(range(2), desc="Constructing latent prior..."):
            time.sleep(0.05)
        for _ in tqdm(range(2), desc="Campaign 2 (chi sweep)"):
            time.sleep(0.05)
    pump(app)

    worst = max((len(s) for s in snapshots), default=0)
    assert worst <= 1, f"the pump exposed {worst} concurrent rows for two sequential pos-0 bars"
    assert all(r.row == 0 for s in snapshots for r in s), \
        "a pos-0 bar was painted on row 1"

def test_mutating_description_does_not_mint_a_row_per_iteration():
    """core/SBI/Priors/{bp,hopf,nadrowski}_prior.py call set_description() with a live counter on
    EVERY iteration. Rows are keyed by tqdm `pos`, and the digit-normalised ident keeps the counter
    from reading as 'a different bar took this slot'."""
    def sweep():
        bar = tqdm(total=20, desc="Added 0 sets to accepted parameters", leave=False)
        for i in range(20):
            bar.set_description(f"Added {i} sets to accepted parameters")
            bar.update(1)
        bar.close()

    events = _drive(sweep)
    rows, peak = _live(events)
    logs = [p for k, p in events if k == "log"]

    assert peak == 1, f"a mutating desc minted {peak} concurrent rows"
    assert rows == {}
    assert logs == [], f"a mutating desc leaked log lines: {logs[:3]}"

def test_total_one_bar_is_not_informative():
    """core/SBI/pipeline.py:517 wraps range(TRAINING_NUM_ROUNDS) and that is 1, so it reads 0% for
    the whole build. It must not be allowed to drive the overall bar."""
    degenerate = parse_bar(("err", 0), 0, "Training neural posterior:   0%|  | 0/1 [00:00<?, ?it/s]")
    real = parse_bar(("err", 1), 1, "Generating training data:  42%|## | 2100/5000 [00:12<00:16]")
    assert degenerate.total == 1 and not degenerate.informative
    assert real.total == 5000 and real.informative and real.pct == 42

# ── a run of the window: its bars reach its model as rows, its lines its text ───────────────────────
_ENDED = "The run finished."            # the last line of a run's text, how it ended


def _run(fn, monkeypatch=None):
    """Run ``fn`` as a run of the window, through a bare run host's own dispatch, and turn the event loop until it
    has ended. Returns what the run's controller said of it (a ``RunText``) and every snapshot of progress rows the
    run's model was handed, in order (when ``monkeypatch`` is handed, which records them)."""
    from core.gui.run.controller import controller
    from tests._fixtures import build_run_host
    from tests._runs import RunText, turn_until
    app = qt_app()
    handed, kept = [], RunModel.rows

    def rows(model, snapshot):
        handed.append(tuple(snapshot))
        return kept(model, snapshot)

    if monkeypatch is not None:
        monkeypatch.setattr(RunModel, "rows", rows)
    text = RunText(controller())
    build_run_host().dispatch(fn)
    turn_until(app, lambda: text.ended, 10)
    return text, handed


def test_end_to_end_a_runs_bars_reach_its_model_as_rows_and_never_its_text(monkeypatch):
    """The whole stack: real nested tqdm in a run of the window -> redirect_streams -> pump -> the run's
    controller -> its model and its text. The bars reach the model as progress rows, three at the deepest of the
    nest and none once they have closed; not one line of the run's text comes from them: it holds how the run
    ended and nothing else."""
    def nest():
        for _ in tqdm(range(2), desc="Training neural posterior", leave=False):
            for _ in tqdm(range(4), desc="Generating training data", leave=False):
                for _ in tqdm(range(3), desc="Running time segments", leave=False):
                    time.sleep(0.02)          # let the 15 Hz pump actually tick

    text, handed = _run(nest, monkeypatch)
    assert text.texts() == [_ENDED], f"the run's text gained lines from its bars: {text.lines[:5]}"
    assert max(len(rows) for rows in handed) == 3, f"expected 3 concurrent progress rows, got {handed}"
    assert handed[-1] == (), f"the model was left holding rows after the bars closed: {handed[-1]}"

def test_print_output_still_reaches_the_log():
    """The bars must not swallow ordinary pipeline output: a run's prints reach its text, and not one frame of a
    bar nor any motion of the terminal does."""
    def run():
        print("Config built: NADROWSKI")
        for _ in tqdm(range(3), desc="Generating training data", leave=False):
            time.sleep(0.02)
        print("Prior ready.")

    lines = _run(run)[0].texts()
    assert "Config built: NADROWSKI" in lines and "Prior ready." in lines, lines
    assert not [line for line in lines if "\x1b" in line or "it/s]" in line], f"a bar frame leaked into the text:\n{lines}"

def test_sbi_epoch_counter_becomes_a_progress_row_not_log_spam(monkeypatch):
    """sbi's training loop has NO tqdm bar. Its trainer's training loop prints, on STDOUT:
           print("\\r", f"Training neural network. Epochs trained: {epoch}", end="")
       -- a LEADING '\\r' and no terminator, so it is an overwrite-mode status line. It must reach the run's model
       as one updating row, which the model reads as the training library's epoch counter, and the run's text as one
       line, its final value, never stranded (the old reader dropped it with its buffer); the closing line is read
       as the fit's end."""
    from core.gui.run import readings

    def run():
        for epoch in range(1, 6):
            print("\r", f"Training neural network. Epochs trained: {epoch}", end="")
            time.sleep(0.03)
        print("\nNeural network successfully converged after 5 epochs.")

    text, handed = _run(run, monkeypatch)
    lines = text.texts()
    assert sum("Epochs trained" in line for line in lines) == 1, \
        f"the epoch counter was appended per-epoch instead of overwriting:\n{lines}"
    assert "Training neural network. Epochs trained: 5" in lines, "the FINAL epoch was stranded and never shown"
    assert "Neural network successfully converged after 5 epochs." in lines
    assert readings.read("Neural network successfully converged after 5 epochs.", "info").name == "sbi_converged"
    assert any(fit_counter(rows) is not None for rows in handed), "the epoch counter never reached the model as a row"
    assert all(row.pct is None for rows in handed for row in rows), "a status line faked a percentage"

def test_plain_prints_are_not_eaten_by_the_cursor_logic():
    """A print() writes its text and its '\\n' as TWO chunks, so the '\\n' arrives alone -- byte-identical
    to a tqdm moveto(+1). Treating it as cursor motion strands the line forever and shifts the next
    bar down a phantom row. Ordering and line boundaries must survive a print/bar/print interleave, in the
    run's text as it reaches the window."""
    def run():
        print("Starting fake stage")
        print()                                   # a bare '\n' chunk, with nothing pending
        for _ in tqdm(range(3), desc="Generating training data", leave=False):
            time.sleep(0.02)
        for epoch in range(1, 4):
            print("\r", f"Training neural network. Epochs trained: {epoch}", end="")
            time.sleep(0.02)
        print("\nNeural network successfully converged.")
        print("Prior ready.")

    lines = [line for line in _run(run)[0].texts() if line.strip()]
    assert lines == [
        "Starting fake stage",
        "Training neural network. Epochs trained: 3",
        "Neural network successfully converged.",
        "Prior ready.",
        _ENDED,
    ], f"the run's lines mangled:\n{lines}"

def test_quiet_segment_bar_collapses_the_nest():
    """config.QUIET_SEGMENT_BAR (set by core.gui.app.build_app) drops the per-time-segment bar, taking
    the nest from 4 deep to 3 -- and a disabled bar must SURRENDER its slot (tqdm/std.py:985-992 removes
    it from _instances), not merely hide, or the solver would still sit at pos 3.

    The solver bar itself stays ON: its it/s is the Solver Performance meter. It is hidden at the widget
    layer, not the tqdm layer -- see test_solver_bar_is_not_rendered_as_a_row."""
    from core import config

    def nest():
        for _ in tqdm(range(2), desc="Training neural posterior", leave=False):
            for _ in tqdm(range(2), desc="Generating training data", leave=False):
                for _ in tqdm(range(2), desc="Running time segments", leave=False,
                              **({"disable": True} if config.QUIET_SEGMENT_BAR else {})):
                    for _ in tqdm(range(2), desc="step (batch=64)", leave=False):
                        pass

    assert config.QUIET_SEGMENT_BAR is False, "the CLI default must be False"
    _, loud_peak = _live(_drive(nest))

    config.QUIET_SEGMENT_BAR = True
    try:
        rows, quiet_peak = _live(_drive(nest))
    finally:
        config.QUIET_SEGMENT_BAR = False

    assert loud_peak == 4, f"expected a 4-deep nest when loud, got {loud_peak}"
    assert quiet_peak == 3, f"QUIET_SEGMENT_BAR should leave 3 bars, got {quiet_peak}"
    assert rows == {}

# ── the solver-performance meter ─────────────────────────────────────────────────────────────────
def test_solver_rate_is_parsed_from_all_three_tqdm_renderings():
    """tqdm renders its rate three ways (std.py:550-559). The s/it form is the trap: it is SECONDS PER
    ITERATION, so " 2.50s/it" is 0.4 it/s, not 2.5 -- read naively, a crawling solver reads as a fast
    one and the meter would show MORE plus signs the slower it got."""
    fast = parse_bar(("err", 2), 2, "step (batch=32):  88%|## | 13269/14999 [00:01<00:00, 13267.85it/s]")
    slow = parse_bar(("err", 2), 2, "step (batch=32):  10%|#  | 3/30 [00:07<01:07,  2.50s/it]")
    fresh = parse_bar(("err", 2), 2, "step (batch=32):   0%|   | 0/83190 [00:00<?, ?it/s]")

    assert fast.rate == 13267.85 and fast.is_solver
    assert slow.rate == 0.4, f"s/it must be inverted, got {slow.rate}"
    assert fresh.rate is None, "'?it/s' means no measurement yet -- it must not read as 0"

def test_only_the_solver_bar_is_identified_as_the_solver():
    """The meter keys on the desc prefix, never the row: the solver's tqdm `pos` is 0, 1 or 2 depending
    on the phase and the panel."""
    for row in (0, 1, 2):
        assert parse_bar(("err", row), row, f"step (batch=2048):  50%|# | 1/2 [00:00<00:00, 9.0it/s]").is_solver
    for desc in ("Generating training data", "Training neural posterior", "PPC simulations",
                 "Campaign 2 (chi sweep, fpb<=64)", "Constructing latent prior..."):
        state = parse_bar(("err", 1), 1, f"{desc}:  50%|# | 1/2 [00:00<00:00, 9.0it/s]")
        assert not state.is_solver, f"{desc!r} must not be mistaken for the solver bar"

def _model(stages=()):
    """A training's model, as a run of the window makes it, and the window's clock it reads, moved by hand."""
    from tests._runs import Clock
    clock = Clock(datetime(2026, 10, 8, 19, 2, 11, tzinfo=timezone(timedelta(hours=-7))))
    return RunModel(RunSpec("Training", None, stages=stages), clock=clock), clock

def test_the_solver_bar_never_drives_and_never_counts_a_run():
    """The solver bar must not decide what measures a run: its total is in the tens of thousands and it is the
    deepest bar there is, so it would win every time and sweep through its count every second instead of the
    top-level count. A run with no stages of its own says the top-level count, never the solver's."""
    top = parse_bar(("err", 1), 1, "Generating training data:  38%|### | 1902/5000 [05:12<13:41,  6.1it/s]")
    solver = parse_bar(("err", 2), 2, "step (batch=32):  88%|####| 13269/14999 [00:01<00:00, 13267.85it/s]")
    assert driving_row((top, solver)) is top, "the solver bar measures the run"
    assert driving_row((solver,)) is None, "the solver bar measures the run when it is the only row"
    run, _clock = _model()
    run.rows((top, solver))
    assert run.sentence() == "Training is running: 1,902 of 5,000.", run.sentence()

def test_a_moving_solver_count_is_a_sign_of_life_with_no_bar_drawn():
    """THE REGRESSION THE OLD METER EXISTED TO KILL, kept as a rule of the run's model. The meter used to be scraped
    from the rendered text of the solver's tqdm bar, so a solver call SHORTER than that bar's own `mininterval`
    produced no rate at all -- and CUDA graphs made every call shorter than it. Whether the solver works is read
    from the step count it publishes, core.progress.SOLVER: a run that integrates hard and prints nothing is not
    silent, and a count that does not move is no sign of life. Not one bar is created below."""
    from core import progress
    from core.gui.run.model import moved

    run, clock = _model()
    run.feed([("the run's first line", "info")])
    run.solver(progress.SOLVER.steps)
    clock.advance(60)
    run.tick()
    assert [message.sentence for message in run.messages()] == ["No output for 1 minute."]
    before = progress.SOLVER.steps
    run.solver(before)                      # a count that did not move
    clock.advance(60)
    run.tick()
    assert [message.sentence for message in run.messages()] == ["No output for 2 minutes."]
    progress.SOLVER.add(120_000)
    assert moved(before, progress.SOLVER.steps) and not moved(progress.SOLVER.steps, before)
    run.solver(progress.SOLVER.steps)
    run.tick()
    assert run.messages() == (), "a solver count that moved did not count as a sign of life"

def test_the_solver_bar_paints_a_rate_even_when_the_call_is_under_a_second():
    """The command-line half of the same regression, pinned at the tqdm layer.

    The GUI no longer reads this bar, but the command-line tool (`python -m core <subcommand>`) has
    nothing else: with mininterval=1.0 a graphed 100k-step call (~0.7s) rendered its opening "?it/s"
    frame and never a rate -- measured 123 chars of stderr containing none. Drives the REAL settings
    from sdeint._bar_kwargs, and checks the
    counterfactual so the test cannot go vacuous if someone puts mininterval back.
    """
    import re
    from io import StringIO

    from core.Solvers import sdeint

    n, chunk, seconds = 100_000, 50, 0.4

    def render(**override):
        buf = StringIO()

        class Sink:
            def write(self, text):
                buf.write(text)
                return len(text)

            def flush(self):
                pass

            def isatty(self):
                return False

        kw = sdeint._bar_kwargs(n, 2048)
        kw.update(override)
        bar = tqdm(total=n - 1, file=Sink(), **kw)
        steps = (n - 1) // chunk
        t0 = time.perf_counter()
        for k in range(steps):
            while time.perf_counter() - t0 < (k + 1) * seconds / steps:
                pass
            bar.update(chunk)
        bar.close()
        return re.findall(r"[0-9.]+(?:it/s|s/it)", buf.getvalue())

    assert sdeint._bar_kwargs(n, 2048)["mininterval"] <= 0.2, \
        "the solver bar's mininterval is back above a short call's duration"
    assert render(), "the solver bar rendered NO rate over a 0.4s call -- the CLI meter is blind again"
    assert not render(mininterval=1.0), \
        "the counterfactual rendered a rate, so this test no longer proves anything"

def test_a_silent_run_says_so_on_its_page_and_its_next_line_clears_it(tmp_path):
    """A run that writes nothing, its solver still, says so on its page after 45 seconds, in minutes and never in
    raw seconds, as an "Attention" message with the moment it began; its next line clears it, with no answer
    asked. The old spinner that twirled while the run lived went with the progress pane: a spinner that kept
    twirling on a wedged run asserted progress that was not happening."""
    from core.gui import texts
    from core.gui.run.controller import controller
    from tests._fixtures import build_run_host
    from tests._runs import Clock, StandIn, start_run, turn_until
    app = qt_app()
    clock = Clock(datetime(2026, 10, 8, 19, 2, 11, tzinfo=timezone(timedelta(hours=-7))))
    held = controller()
    held.use_clock(clock)
    held.use_folder(tmp_path / "runs")
    host, stand_in = build_run_host(), StandIn()
    view = host.run_view
    start_run(host, stand_in, spec=RunSpec("Run", None, keeps_files=False))
    stand_in.say("the run's first line")
    turn_until(app, lambda: held.status.state == "running")
    clock.advance(107)
    held.tick()
    said = [row.sentence.plain_text() for row in view.message_rows()]
    assert said == [f"Attention, {texts.date_text(clock(), clock())}. No output for 2 minutes."], said
    stand_in.say("the run goes on")
    turn_until(app, lambda: not view.message_rows())
    stand_in.release()
    turn_until(app, lambda: not held.is_live())

def test_the_read_outs_keep_their_width_as_their_numbers_change():
    """The run page's read-outs are numbers that change in place: the counts and the times of its sentence, a
    stage's pace, the times under its time axis and what follows. In the lettering's ordinary digits a 1 is
    narrower than a 0, so a read-out would change its width with its digits and move whatever stands after the
    number, at every batch. Each of them has its lettering with its digits of equal width; the page's title and
    its buttons, which show none, are left as they were."""
    from PySide6.QtGui import QFont, QFontMetricsF
    from PySide6.QtWidgets import QLabel
    from core.gui import fonts
    from core.gui.run.model import StageWord
    from tests._fixtures import build_run_host

    qt_app()
    view = build_run_host().run_view
    view.stage_line.show_words((StageWord("Fitting", "now", "Now, about 63 minutes per epoch"),))
    pace, = [label for label in view.stage_line.findChildren(QLabel) if label.text().startswith("Now")]
    equal, body = QFont.Tag.fromString("tnum"), fonts.inter("body")
    for name, label, face in (("the sentence", view.sentence, fonts.inter("head", 400)), ("a stage's pace", pace, body),
                              ("its start", view.axis_caption.started, body), ("its end", view.axis_caption.words, body),
                              ("what follows", view.follows, body)):
        font = label.font()
        assert font.isFeatureSet(equal), f"{name} shows its numbers in digits of unequal width"
        assert (font.family(), font.pixelSize(), font.weight()) == (face.family(), face.pixelSize(), face.weight()), name
        metrics = QFontMetricsF(font)
        assert metrics.horizontalAdvance("00:00:00") == metrics.horizontalAdvance("11:11:11"), name
    plain = QFontMetricsF(body)
    assert plain.horizontalAdvance("00:00:00") != plain.horizontalAdvance("11:11:11"), \
        "the digits have one width without the feature as well: the lines above prove nothing"
    for untouched in (view.title, view.text_button, view.stop_button, view.back_button):
        assert not untouched.font().isFeatureSet(equal), "more than the read-outs was given the digits' feature"

def test_the_solver_step_iterator_closes_its_bar_when_the_consumer_raises():
    """`sdeint._step_iter` is a GENERATOR wrapping the tqdm bar, so it adds a frame to the cancel's tqdm-lock unwind
    path -- a cancel raises from inside a bar redraw, and tqdm's own `finally: self.close()` has to run
    anyway or its global write lock leaks and the NEXT `tqdm.__new__` DEADLOCKS.
    (test_cancel_token_latches_and_leaves_tqdm_usable, in tests/test_worker_dispatch.py, hangs rather
    than failing, which is why this cheap structural guard is worth having in front of it.)

    Not a proof that the generator changed anything -- it is a guard on the property the generator put
    at risk.
    """
    import gc

    from core.Solvers import sdeint

    class Quiet:
        def write(self, text):
            return len(text)

        def flush(self):
            pass

        def isatty(self):
            return False

    real, sys.stderr = sys.stderr, Quiet()
    try:
        before = len(tqdm._instances)
        try:
            for i in sdeint._step_iter(5000, 32):
                if i == 10:
                    raise RuntimeError("the consumer blew up mid-integration")
        except RuntimeError:
            pass
        gc.collect()
        after = len(tqdm._instances)
    finally:
        sys.stderr = real

    assert after == before, \
        f"the solver bar was stranded by the generator wrapper ({before} -> {after} live tqdm instances)"

def test_the_fit_is_found_through_the_training_librarys_epoch_counter():
    """THE REASON THE OLD CAPTION EXISTED, kept as a rule of the run's model. sbi's neural-network training --
    hours of a multi-day build -- emits no tqdm bar at all, only a printed epoch counter (an overwrite-mode row,
    pct and total both None). The only other live row is `Training neural posterior -- 0/1`, which is degenerate
    and measures nothing. The model finds the fit through the counter: the fit begins, and the sentence counts its
    epochs, where "no rows, just a bar" would say nothing for that whole phase."""
    degenerate = parse_bar(("err", 0), 0, "Training neural posterior:   0%|  | 0/1 [00:00<?, ?it/s]")
    epochs = parse_bar(("out", 0), 0, " Training neural network. Epochs trained: 812")
    assert driving_row((degenerate, epochs)) is None, "a total=1 bar was allowed to measure the run"
    assert fit_counter((degenerate, epochs)) == 812 and fit_counter((degenerate,)) is None
    run, _clock = _model((Stage("Simulation", (), "Generating training data"),
                          Stage("Fitting", ("simulation_complete",), "fit_epoch")))
    run.rows((degenerate, epochs))
    assert run.stage == "Fitting" and "epoch 812" in run.sentence(), (run.stage, run.sentence())

def test_the_row_that_drives_is_the_largest_non_degenerate_total():
    """Largest total, not deepest. The per-time-segment bar wraps segs in {1,2,3}, so it is BOTH
    deeper and far coarser than "Generating training data" -- picking the deepest row would sweep the
    run's count through 0 to 3 every couple of seconds. (config.QUIET_SEGMENT_BAR hides that bar under the
    GUI today; the rule should not depend on it still being set.)"""
    rounds = parse_bar(("err", 0), 0, "Training neural posterior:   0%|  | 0/1 [00:00<?, ?it/s]")
    data = parse_bar(("err", 1), 1, "Generating training data:  38%|### | 1902/5000 [05:12<13:41]")
    segs = parse_bar(("err", 2), 2, "Running time segments:  66%|##  | 2/3 [00:01<00:00]")
    solver = parse_bar(("err", 3), 3, "step (batch=32):  88%|####| 13269/14999 [00:01<00:00, 13267.85it/s]")
    assert driving_row((rounds, data, segs, solver)) is data, "the run is measured by the deepest bar"
    run, _clock = _model()
    run.rows((rounds, data, segs, solver))
    assert run.sentence() == "Training is running: 1,902 of 5,000.", run.sentence()

def test_leave_true_pos0_bar_is_retired_not_left_pegged_at_100():
    """close(leave=True) at pos 0 paints the final frame and then writes a bare '\\n' -- which is
    byte-identical to a moveto(+1). Assuming "moveto" leaves the finished bar live at 100% AND, because
    it is `informative`, measuring the run at 100% while the pipeline is still working. Both real
    leave=True bars (core/SBI/Priors/prior.py:88 "Constructing latent prior...",
    core/FDT/campaigns.py:214 "Campaign 2") are pos 0, so this is the common case, not a corner."""
    def bar_then_work():
        for _ in tqdm(range(5), desc="Constructing latent prior..."):
            pass
        print("Prior constructed.", file=sys.stderr)    # the pipeline carries on after the bar closes

    events = _drive(bar_then_work)
    rows, _ = _live(events)
    logs = [t for _k, (t, _lvl) in ((k, p) for k, p in events if k == "log")]

    assert rows == {}, f"the finished leave=True bar was never retired: {rows}"
    assert any("100%" in t for t in logs), "its final frame should graduate into the log"

    # and the bar that finished must not still measure the run while the pipeline carries on
    live, measured = {}, []
    for kind, payload in events:
        if kind == "row":
            live[payload.key] = payload
        elif kind == "retire":
            live.pop(payload, None)
        elif kind == "log" and payload[0] == "Prior constructed.":
            break
        measured.append(driving_row(tuple(live.values())))
    else:
        raise AssertionError("the line the pipeline wrote after the bar never came")
    assert any(row is not None for row in measured), "the bar never measured the run while it lived"
    assert measured[-1] is None, "the finished bar still measures the run while the pipeline carries on"

def test_the_window_handler_feeds_the_pane_at_the_records_level_and_checks_cancel():
    """Standard logging in the window. Severity used to come from the CHANNEL: stdout landed at info
    and stderr at warning, so the memory-statistics line wore a triangle and "loaded a NON-AMORTIZED
    posterior" did not. A record carries its own level, and the handler hands it to the same pump the
    streams feed, so it lands in the pane IN ORDER with the prints around it.

    Two more things the handler must do exactly as _SignalStream.write does. It is a CANCEL
    CHECKPOINT (a run that only logs, never prints, must still stop at its next message), with the
    same latch -- one raise, then quiet -- so a record emitted during the unwind cannot replace the
    clean traceback. And it must be GONE when the redirect ends: a handler left on the ``core``
    logger would feed a stopped pump for the rest of the process.

    No caplog.set_level anywhere here: the ``core`` logger is at INFO by import (core/runs.py), and an
    info record reaching the pane WITHOUT the fixture's help is the point -- with that help the suite
    would stay green while the window dropped every information record."""
    import logging

    import pytest

    from core.gui import streams
    from core.gui.streams import CancelToken, WorkerCancelled
    from tests._fixtures import pump, qt_app

    app = qt_app()
    signals = WorkerSignals()
    lines = []
    # the RAW payload, deliberately: this is the pin on log_batch's own (text, level) order, which
    # tests/_fixtures.PaneCapture flips into (level, text). Do not move this test onto the helper.
    signals.log_batch.connect(lambda batch: lines.extend(batch))
    signals.rows.connect(lambda _s: None)
    rec = logging.getLogger("core.tests.streams")          # a child: the handler sits on "core"

    with redirect_streams(signals):
        rec.info("[mem] host 1.2 GiB")
        rec.warning("[tsnpe] loaded a NON-AMORTIZED posterior")
        rec.error("[checkpoint] could not save on the way out")
        print("Prior ready.")
    pump(app)
    assert ("[mem] host 1.2 GiB", "info") in lines, lines
    assert ("[tsnpe] loaded a NON-AMORTIZED posterior", "warning") in lines
    assert ("[checkpoint] could not save on the way out", "error") in lines
    assert ("Prior ready.", "info") in lines, "the console redirect must keep carrying prints"
    assert [t for t, _ in lines][:4] == ["[mem] host 1.2 GiB", "[tsnpe] loaded a NON-AMORTIZED posterior",
                                          "[checkpoint] could not save on the way out", "Prior ready."], \
        "records and prints share one ordered sink"
    assert not any(isinstance(h, streams._PumpLogHandler) for h in logging.getLogger("core").handlers), \
        "the handler outlived the redirect"
    rec.info("after the redirect")
    pump(app)
    assert not any(t == "after the redirect" for t, _ in lines)

    # The cancel checkpoint, with the latch.
    token = CancelToken()
    with redirect_streams(signals, token):
        token.requested.set()
        with pytest.raises(WorkerCancelled):
            rec.info("[checkpoint] saving batch 3/4")
        assert token.fired
        rec.info("after the latch")                        # one raise, then quiet
    pump(app)
    assert ("after the latch", "info") in lines, lines
    assert not any(t == "[checkpoint] saving batch 3/4" for t, _ in lines), \
        "the record that raised must not reach the pane"
