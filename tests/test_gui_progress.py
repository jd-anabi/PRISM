"""Progress-rendering regression tests for the GUI.

THE BUG THESE LOCK DOWN
    tqdm redraws a bar at pos>0 as three writes -- '\\n'*pos, then '\\r'+frame, then '\\x1b[A'*pos
    (tqdm/std.py:1493-1497). The old stream reader split on terminators, so the frame (which is never
    terminated) stranded in its buffer and was flushed by the NEXT redraw's leading '\\n' -- i.e. as a
    LOG LINE. Every nested-bar redraw appended one row, so a training run buried the log pane under
    hundreds of bar snapshots.

    The pipeline nests bars four deep (core/SBI/pipeline.py:517 -> :371 ->
    core/Simulator/simulator.py:50 -> core/Solvers/sdeint.py:15), so this fired constantly.

Run:  python -m pytest tests/test_gui_progress.py -v
      (or just: python tests/test_gui_progress.py)
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")   # must precede any PySide6 import
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib                                                 # noqa: E402
matplotlib.use("Agg")                                            # match the app (core/gui/__main__.py forces it)

from PySide6.QtWidgets import QApplication                        # noqa: E402
from tqdm import tqdm                                             # noqa: E402

from core.gui.panels.base_panel import BasePanel                  # noqa: E402
from core.gui.streams import redirect_streams                     # noqa: E402
from core.gui.vt import StreamRouter, parse_bar                   # noqa: E402
from core.gui.widgets.log_pane import LogPane                     # noqa: E402
from core.gui.widgets.progress_pane import ProgressPane           # noqa: E402
from core.gui.worker import WorkerSignals                         # noqa: E402


def _app():
    return QApplication.instance() or QApplication([])


def _pump(app, seconds=0.5):
    """Drive the event loop without app.exec(), so the pump's queued signals get delivered."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        app.processEvents()
        time.sleep(0.01)


# ── the router, driven by REAL tqdm (no Qt) ──────────────────────────────────────────────────────
def _drive(fn, level="warning"):
    """Run `fn` with sys.stderr routed through a StreamRouter; return its event list."""
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
    app = _app()
    signals = WorkerSignals()
    snapshots = []
    signals.rows.connect(snapshots.append)
    signals.log_batch.connect(lambda _b: None)

    with redirect_streams(signals):
        for _ in tqdm(range(2), desc="Constructing latent prior..."):
            time.sleep(0.05)
        for _ in tqdm(range(2), desc="Campaign 2 (chi sweep)"):
            time.sleep(0.05)
    _pump(app)

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


# ── the full Qt stack ────────────────────────────────────────────────────────────────────────────
def test_end_to_end_log_pane_gains_zero_blocks():
    """The whole stack: real nested tqdm -> redirect_streams -> pump -> ProgressPane / LogPane."""
    app = _app()
    log, prog = LogPane(), ProgressPane()
    signals = WorkerSignals()
    signals.log.connect(log.append_line)
    signals.log_batch.connect(log.append_lines)
    signals.rows.connect(prog.set_rows)

    peak_rows = 0

    def watch(snapshot):
        nonlocal peak_rows
        peak_rows = max(peak_rows, len(snapshot))

    signals.rows.connect(watch)

    prog.begin()
    with redirect_streams(signals):
        for _ in tqdm(range(2), desc="Training neural posterior", leave=False):
            for _ in tqdm(range(4), desc="Generating training data", leave=False):
                for _ in tqdm(range(3), desc="Running time segments", leave=False):
                    time.sleep(0.02)          # let the 15 Hz pump actually tick
    _pump(app)

    assert log.blockCount() == 1 and not log.toPlainText().strip(), \
        f"log pane gained {log.blockCount()} block(s):\n{log.toPlainText()[:500]}"
    assert peak_rows == 3, f"expected 3 concurrent progress rows, got {peak_rows}"

    prog.end()
    assert not prog._rows, "ProgressPane.end() left rows behind"


def test_print_output_still_reaches_the_log():
    """The bars must not swallow ordinary pipeline output."""
    app = _app()
    log = LogPane()
    signals = WorkerSignals()
    signals.log.connect(log.append_line)
    signals.log_batch.connect(log.append_lines)
    signals.rows.connect(lambda _s: None)

    with redirect_streams(signals):
        print("Config built: NADROWSKI")
        for _ in tqdm(range(3), desc="Generating training data", leave=False):
            time.sleep(0.02)
        print("Prior ready.")
    _pump(app)

    text = log.toPlainText()
    assert "Config built: NADROWSKI" in text
    assert "Prior ready." in text
    assert "\x1b" not in text
    assert "it/s]" not in text, f"a bar frame leaked into the log:\n{text}"


def test_sbi_epoch_counter_becomes_a_progress_row_not_log_spam():
    """sbi's training loop has NO tqdm bar. It prints, on STDOUT:
           print("\\r", f"Training neural network. Epochs trained: {epoch}", end="")
       (sbi/inference/trainers/base.py:1024) -- a LEADING '\\r' and no terminator, so it is an
       overwrite-mode status line. It must render as one updating row, not one log line per epoch,
       and its final value must not be stranded (the old reader dropped it with its buffer)."""
    app = _app()
    log = LogPane()
    signals = WorkerSignals()
    snapshots = []
    signals.log.connect(log.append_line)
    signals.log_batch.connect(log.append_lines)
    signals.rows.connect(snapshots.append)

    with redirect_streams(signals):
        for epoch in range(1, 6):
            print("\r", f"Training neural network. Epochs trained: {epoch}", end="")
            time.sleep(0.03)
        print("\nNeural network successfully converged after 5 epochs.")
    _pump(app)

    text = log.toPlainText()
    assert text.count("Epochs trained") == 1, \
        f"the epoch counter was appended per-epoch instead of overwriting:\n{text}"
    assert "Epochs trained: 5" in text, "the FINAL epoch was stranded and never shown"
    assert "successfully converged" in text
    assert any(r.desc.startswith("Training neural network") for s in snapshots for r in s), \
        "the epoch counter never rendered as a progress row"
    assert all(r.pct is None for s in snapshots for r in s), "a status line faked a percentage"


def test_plain_prints_are_not_eaten_by_the_cursor_logic():
    """A print() writes its text and its '\\n' as TWO chunks, so the '\\n' arrives alone -- byte-identical
    to a tqdm moveto(+1). Treating it as cursor motion strands the line forever and shifts the next
    bar down a phantom row. Ordering and line boundaries must survive a print/bar/print interleave."""
    app = _app()
    log = LogPane()
    signals = WorkerSignals()
    signals.log.connect(log.append_line)
    signals.log_batch.connect(log.append_lines)
    signals.rows.connect(lambda _s: None)

    with redirect_streams(signals):
        print("Starting fake stage")
        print()                                   # a bare '\n' chunk, with nothing pending
        for _ in tqdm(range(3), desc="Generating training data", leave=False):
            time.sleep(0.02)
        for epoch in range(1, 4):
            print("\r", f"Training neural network. Epochs trained: {epoch}", end="")
            time.sleep(0.02)
        print("\nNeural network successfully converged.")
        print("Prior ready.")
    _pump(app)

    lines = [ln for ln in log.toPlainText().splitlines() if ln.strip()]
    assert lines == [
        "Starting fake stage",
        "Training neural network. Epochs trained: 3",
        "Neural network successfully converged.",
        "Prior ready.",
    ], f"log lines mangled:\n{lines}"


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


def test_plus_meter_is_one_sign_per_order_of_magnitude():
    from core.gui.widgets.progress_pane import plus_meter

    assert plus_meter(10_000) == "++++"       # the user's worked example
    assert plus_meter(13_267.85) == "++++"
    assert plus_meter(1_000) == "+++"
    assert plus_meter(999) == "++"
    assert plus_meter(10) == "+"
    assert plus_meter(5) == "·"               # under one order of magnitude: not an empty string
    assert plus_meter(0.4) == "·"
    assert plus_meter(None) == "—"


def test_solver_bar_is_not_rendered_as_a_row_and_does_not_drive_the_overall_bar():
    """A posterior build creates 10k-30k solver bars, so it must never become a row (a widget churned
    every few seconds). And it must not drive the overall bar: its total is in the tens of thousands and
    it is the deepest bar, so it would win _retarget every time and drag the overall bar through a full
    0->100% sweep every second, instead of showing the top-level count."""
    _app()
    prog = ProgressPane()
    prog.begin()

    top = parse_bar(("err", 1), 1, "Generating training data:  38%|### | 1902/5000 [05:12<13:41,  6.1it/s]")
    solver = parse_bar(("err", 2), 2, "step (batch=32):  88%|####| 13269/14999 [00:01<00:00, 13267.85it/s]")
    prog.set_rows((top, solver))

    assert len(prog._rows) == 1, f"the solver bar was rendered as a row: {list(prog._rows)}"
    assert next(iter(prog._rows)) == ("err", 1)
    assert prog.overall.maximum() == 100 and prog.overall.value() == 38, \
        "the overall bar must track the top-level count, not the solver"
    assert prog.solver_strip.isVisible() and prog.solver_strip.value() == 88
    assert "++++" in prog.solver_label.text(), prog.solver_label.text()
    assert "13.3k it/s" in prog.solver_label.text(), prog.solver_label.text()
    prog.end()


def test_solver_meter_holds_its_rate_across_bars_then_goes_idle():
    """Rate samples arrive in gaps: a bar shorter than mininterval=1.0 emits only '?it/s', and with
    leave=False there is no final 100% frame. So the last rate is HELD -- but not forever: during
    neural-network training the solver genuinely is not running, and a stale '++++' would be a lie."""
    from core.gui.widgets import progress_pane as pp

    _app()
    prog = ProgressPane()
    prog.begin()

    prog.set_rows((parse_bar(("err", 0), 0, "step (batch=32):  88%|# | 132/149 [00:01<00:00, 13267.85it/s]"),))
    assert "++++" in prog.solver_label.text()

    # a fresh bar with no measurement yet must not wipe the held rate
    prog.set_rows((parse_bar(("err", 0), 0, "step (batch=32):   0%|  | 0/83190 [00:00<?, ?it/s]"),))
    assert "++++" in prog.solver_label.text(), "an opening '?it/s' frame clobbered the held rate"

    # ...but once the solver stops reporting for SOLVER_IDLE_S, the meter must admit it is idle
    prog._rate_at -= pp.SOLVER_IDLE_S + 1
    prog.set_rows(())
    assert "idle" in prog.solver_label.text(), prog.solver_label.text()
    assert not prog.solver_strip.isVisible(), "the step strip should hide when no solver is running"
    prog.end()


def test_spinner_animates_and_then_reports_a_stall():
    """A spinner that keeps twirling on a wedged run asserts progress that is not happening."""
    from core.gui.widgets import progress_pane as pp

    _app()
    prog = ProgressPane()
    prog.begin()

    frames = set()
    for _ in range(4):
        prog._tick()
        frames.add(prog.spinner.text())
    assert len(frames) == 4, f"the spinner did not advance: {frames}"

    prog._beat_at -= pp.STALL_S + 62
    prog._tick()
    assert "no output for" in prog.spinner.text(), prog.spinner.text()

    prog.heartbeat()                      # output resumes -> back to spinning
    prog._tick()
    assert "no output for" not in prog.spinner.text()
    prog.end()


def test_leave_true_pos0_bar_is_retired_not_left_pegged_at_100():
    """close(leave=True) at pos 0 paints the final frame and then writes a bare '\\n' -- which is
    byte-identical to a moveto(+1). Assuming "moveto" leaves the finished bar sitting in the pane at
    100% AND, because it is `informative`, pegging the overall bar at 100% while the pipeline is still
    working. Both real leave=True bars (core/SBI/Priors/prior.py:88 "Constructing latent prior...",
    core/FDT/campaigns.py:214 "Campaign 2") are pos 0, so this is the common case, not a corner."""
    def bar_then_work():
        for _ in tqdm(range(5), desc="Constructing latent prior..."):
            pass
        print("Prior constructed.")          # the pipeline carries on after the bar closes

    events = _drive(bar_then_work)
    rows, _ = _live(events)
    logs = [t for _k, (t, _lvl) in ((k, p) for k, p in events if k == "log")]

    assert rows == {}, f"the finished leave=True bar was never retired: {rows}"
    assert any("100%" in t for t in logs), "its final frame should graduate into the log"

    # and the overall bar must not still be reading 100%
    app = _app()
    prog = ProgressPane()
    live = {}
    for kind, payload in events:
        if kind == "row":
            live[payload.key] = payload
        elif kind == "retire":
            live.pop(payload, None)
        prog.set_rows(tuple(live.values()))
    assert prog.overall.maximum() == 0, "the overall bar is still determinate after the bar finished"


def test_worker_payload_is_released_after_the_run():
    """setAutoDelete(False) + the _finished closure keep the Worker shell alive forever. That is fine
    for the shell, but NOT for what it points at -- without an explicit release every dispatch pins its
    cfg / prior / posterior / CUDA tensors for the life of the process."""
    import gc
    import weakref

    app = _app()

    class Big:
        pass

    class P(BasePanel):
        pass

    panel = P()
    big = Big()
    ref = weakref.ref(big)

    panel.dispatch(lambda payload: None, big)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and panel._busy:
        app.processEvents()
        time.sleep(0.01)
    _pump(app, 0.2)

    del big
    gc.collect()
    assert ref() is None, "the worker still pins its argument after the run finished"


# ── Phase-2 panels ───────────────────────────────────────────────────────────────────────────────
def test_fdt_panel_turns_the_nadrowski_coupling_keyerror_into_a_readable_error():
    """run_fdt reads params_dict["n"] / ["beta"], so it KeyErrors on hopf/bp cells. That is a
    pre-existing pipeline limitation; the panel must explain it rather than surface a bare KeyError."""
    import core.gui.panels.fdt_panel as fdt_panel

    class Cfg:
        model = "HOPF"

    def boom(cfg, *, skip_sanity, confirm_production):
        raise KeyError("n")

    real, fdt_panel.run_fdt = fdt_panel.run_fdt, boom
    try:
        try:
            fdt_panel._run_fdt_guarded(Cfg(), skip_sanity=True, confirm_production=False)
        except RuntimeError as e:
            assert "NADROWSKI" in str(e) and "HOPF" in str(e), str(e)
        except KeyError:
            raise AssertionError("the bare KeyError escaped the guard")
        else:
            raise AssertionError("the guard swallowed the failure entirely")
    finally:
        fdt_panel.run_fdt = real


def test_a_cell_with_no_bounds_sibling_does_not_brick_the_gui():
    """Dropping a cell into Resources/Cells/<model>/ WITHOUT a sibling Resources/Bounds/<model>/<same>
    -- the natural 'add my cell' action -- makes cli._parse_cell raise a bare ValueError (NOT a
    UnitParseError). CrossValPanel prefills from _parse_cell in __init__, so that exception used to
    escape CrossValPanel() -> MainWindow() -> build_app(), and `python -m core.gui` died before the
    window ever appeared -- before app.py's excepthook was even installed."""
    import shutil
    from pathlib import Path

    from core.config import CELL_PATH
    from core.gui import settings as st

    _app()
    src = Path(CELL_PATH) / "nadrowski" / "cell.txt"
    if not src.exists():
        return                                   # nothing to probe with
    probe = src.with_name("aaa_probe_no_bounds.txt")   # sorts first => the picker selects it
    shutil.copyfile(src, probe)
    try:
        # Isolate from the developer's real QSettings store. MainWindow() -> CrossValPanel.__init__
        # restores its saved cell selection; a cell saved from a previous GUI session would be reloaded
        # OVER our probe, so the picker would land on a valid cell and the prefill would parse fine --
        # masking the degrade path this test exists to check. A clean temp .ini defaults the picker to
        # the alphabetically-first entry (the probe), regardless of the machine's state.
        _temp_settings()
        from core.gui.main_window import MainWindow
        from core.gui.panels.crossval_panel import CrossValPanel
        window = MainWindow()                    # must not raise
        xval = window.panel(CrossValPanel)       # CrossVal now lives inside the FDT Analysis section
        assert "could not read cell" in xval.cell_values.text().lower(), \
            f"the bad cell should degrade the prefill label, got: {xval.cell_values.text()!r}"
    finally:
        probe.unlink(missing_ok=True)
        st.use_ini_file(None)


def test_plot_watcher_only_reports_pngs_written_after_start():
    """The FDT/Reduction/CrossVal runners never return their figure paths, so the panels pick them up
    off disk. Pre-existing figures must not be re-shown, and each new one must be emitted once."""
    import tempfile
    from pathlib import Path

    from core.gui.plot_watcher import NewPngWatcher

    app = _app()
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "old_plot_20260101_000000.png").write_bytes(b"stale")

        seen = []
        watcher = NewPngWatcher(d)
        watcher.png_ready.connect(lambda title, path: seen.append((title, Path(path).name)))
        watcher.start()

        (d / "fdt3d_vs_S_20260714_120301.png").write_bytes(b"new")
        watcher.stop()                      # stop() forces a final scan, ignoring the settle delay
        app.processEvents()

        assert seen == [("fdt3d vs S", "fdt3d_vs_S_20260714_120301.png")], seen


# ── Phase 3: cancellation ────────────────────────────────────────────────────────────────────────
def test_worker_cancelled_passes_through_except_exception():
    """The cancel exception must be a BaseException so the pipeline's many `except Exception` handlers
    (sbi, cross_validation, Worker.run itself) do not swallow it."""
    from core.gui.streams import WorkerCancelled

    try:
        try:
            raise WorkerCancelled()
        except Exception:  # noqa: BLE001
            raise AssertionError("`except Exception` caught the cancel -- it is not BaseException-derived")
    except WorkerCancelled:
        pass


def test_cancel_token_latches_and_leaves_tqdm_usable():
    """The token raises exactly ONCE (so tqdm's teardown write can finish), and after the cancel a NEW
    tqdm bar must be creatable on a fresh thread.

    That last part is the sharp edge: tqdm's refresh() manual-acquires its global write lock and
    manual-releases it (tqdm/std.py:1346-1349), so a raise from inside a redraw's write() skips the
    release and leaks the lock -- after which the next tqdm.__new__ deadlocks. redirect_streams' cancel
    teardown resets the lock; without that reset THIS TEST HANGS (which is exactly the production bug:
    cancel a run, start another, hang)."""
    import threading

    from core.gui.streams import CancelToken, WorkerCancelled, redirect_streams

    signals = WorkerSignals()
    token = CancelToken()
    token.requested.set()                      # cancel already requested when the run starts

    with redirect_streams(signals, token):
        try:
            for _ in tqdm(range(5), desc="Generating training data", leave=False):
                for _ in tqdm(range(5), desc="step (batch=8)", leave=False):
                    pass
        except WorkerCancelled:
            pass
    assert token.fired, "the token never fired"
    assert len(tqdm._instances) == 0, f"tqdm left {len(tqdm._instances)} stale bar(s)"

    # tqdm must not be wedged: make a bar on a fresh thread with a bounded join.
    made = []

    def _probe():
        b = tqdm(range(2), desc="probe", file=open(os.devnull, "w"))
        b.close()
        made.append(True)

    th = threading.Thread(target=_probe, daemon=True)
    th.start()
    th.join(3.0)
    assert made == [True], "tqdm deadlocked after a cancel -- the write lock was not recovered"


def test_dispatched_run_cancels_cleanly_and_a_later_run_still_works():
    """End to end: a run cancelled mid-flight emits `cancelled` (not `error`), ends not-busy with rows
    dropped and stray figures closed, clears the active token, and a fresh run afterwards completes."""
    import matplotlib.pyplot as plt

    app = _app()

    class P(BasePanel):
        pass

    panel = P()
    started = {"v": False}
    outcome = {"cancelled": 0, "error": 0, "result": []}

    def heavy(fig_sink=None):
        plt.figure()                          # a stray figure the cancel path must close
        for i in range(2000):
            for _ in tqdm(range(20), desc="step (batch=8)", leave=False):
                pass
            started["v"] = True
            print(f"epoch {i}")               # a write() checkpoint
            time.sleep(0.01)
        return "COMPLETED"

    panel.dispatch(heavy, on_result=lambda r: outcome["result"].append(r))
    for w in panel._workers:
        w.signals.cancelled.connect(lambda: outcome.__setitem__("cancelled", outcome["cancelled"] + 1))
        w.signals.error.connect(lambda *_a: outcome.__setitem__("error", outcome["error"] + 1))

    t0 = time.monotonic()
    while time.monotonic() - t0 < 5 and not started["v"]:
        app.processEvents()
        time.sleep(0.005)
    assert panel._busy and BasePanel._active_cancel is not None

    panel._request_cancel()
    assert panel._cancel.requested.is_set()
    assert panel.btn_cancel.text() == "Cancelling…" and not panel.btn_cancel.isEnabled()

    t0 = time.monotonic()
    while time.monotonic() - t0 < 10 and panel._busy:
        app.processEvents()
        time.sleep(0.005)
    _pump(app, 0.3)

    assert not panel._busy, "panel stuck busy after cancel"
    assert outcome["result"] == [], "the run COMPLETED instead of cancelling"
    assert outcome["cancelled"] == 1 and outcome["error"] == 0, outcome
    assert not panel.progress_pane._rows, "rows leaked after cancel"
    assert plt.get_fignums() == [], "stray figures not closed on cancel"
    assert BasePanel._active_cancel is None and not BasePanel._running
    assert "Run cancelled." in panel.log_pane.toPlainText()

    later = []
    panel.dispatch(lambda: "SECOND OK", on_result=later.append)
    t0 = time.monotonic()
    while time.monotonic() - t0 < 5 and (panel._busy or not later):
        app.processEvents()
        time.sleep(0.005)
    assert later == ["SECOND OK"], f"a run after a cancel did not complete: {later}"


def test_cancel_is_not_consumed_by_a_non_worker_thread():
    """tqdm's TMonitor daemon force-refreshes a quiet bar, writing to our stream from ITS thread. If
    that write consumed the cancel latch, it would raise where nobody catches it and leave the worker
    to sail past a fired latch -- silently losing the cancel. Only the armed (worker) thread may raise."""
    import threading
    from core.gui.streams import CancelToken, WorkerCancelled

    token = CancelToken()
    token.requested.set()
    out = {}

    def worker():
        token.arm()                            # redirect_streams arms on the worker thread
        # a non-worker ("monitor") write happens first and must NOT consume the latch
        other = threading.Thread(target=token.check)
        other.start()
        other.join()
        try:
            token.check()                      # the worker's own next write MUST still raise
        except WorkerCancelled:
            out["worker_raised"] = True

    th = threading.Thread(target=worker)
    th.start()
    th.join(3.0)
    assert out.get("worker_raised") is True, "the cancel was consumed by a non-worker thread and lost"


def test_inference_config_restore_with_a_stale_model_does_not_desync_the_bounds_picker():
    """A corrupt/version-skewed .ini with an unknown model must not point the Config bounds picker at a
    nonexistent folder while the combo shows a real default."""
    from core.gui import settings as st
    from core.gui.screens.inference_screen import InferenceScreen

    _app()
    _temp_settings()
    try:
        qs = st.settings()
        qs.beginGroup("inference_config")
        qs.setValue("model", "NOT_A_REAL_MODEL")
        qs.endGroup()
        qs.sync()

        cfg_panel = InferenceScreen().config_panel
        model = cfg_panel.model_combo.currentText()
        assert model in ("BP", "NADROWSKI", "HOPF"), model
        assert cfg_panel.bounds_picker.base_path.name == model.lower()
        assert cfg_panel.bounds_picker.combo.count() > 0, "the bounds picker was left empty by a stale model"
    finally:
        st.use_ini_file(None)


# ── Phase 3: QSettings persistence ───────────────────────────────────────────────────────────────
def _temp_settings():
    import tempfile
    from core.gui import settings as st
    fd, path = tempfile.mkstemp(suffix=".ini")
    os.close(fd)
    st.use_ini_file(path)
    return path


def test_settings_round_trip_reduction_and_fdt():
    from core.gui import settings as st
    from core.gui.panels.reduction_panel import ReductionPanel
    from core.gui.panels.fdt_panel import FdtPanel

    _app()
    _temp_settings()
    try:
        red = ReductionPanel()
        red.f0.setText("0.123")
        if red.cell_picker.combo.count():
            red.cell_picker.combo.setCurrentIndex(red.cell_picker.combo.count() - 1)
        want_cell = red.cell_picker.key()

        fdt = FdtPanel()
        fdt.n_freqs.setText("77")
        fdt.skip_sanity.setChecked(True)
        fdt.confirm_production.setChecked(False)

        qs = st.settings()
        red.save_settings(qs)
        fdt.save_settings(qs)
        qs.sync()

        red2 = ReductionPanel()
        fdt2 = FdtPanel()
        assert red2.f0.value() == 0.123
        assert red2.cell_picker.key() == want_cell
        assert fdt2.n_freqs.value() == 77
        assert fdt2.skip_sanity.isChecked() is True
        assert fdt2.confirm_production.isChecked() is False
    finally:
        st.use_ini_file(None)


def test_missing_picker_key_restores_to_default_not_blank():
    """A saved selection whose file is gone must leave the picker at its default, never blank it via
    setCurrentIndex(-1)."""
    from core.gui import settings as st
    from core.gui.panels.reduction_panel import ReductionPanel

    _app()
    _temp_settings()
    try:
        qs = st.settings()
        qs.beginGroup("reduction")
        qs.setValue("cell", "nadrowski/does_not_exist.txt")
        qs.setValue("f0", "0.05")
        qs.endGroup()
        qs.sync()

        red = ReductionPanel()
        assert red.cell_picker.combo.currentIndex() >= 0, "a stale key blanked the combo"
    finally:
        st.use_ini_file(None)


def test_crossval_does_not_persist_cell_derived_bounds():
    """The S/T grid lo/hi are re-derived from the cell file; a saved value from a different cell would
    be a stale, wrong bound. Only the free knobs (points, f0, freqs_per_batch, preset, cell) persist."""
    from core.gui import settings as st
    from core.gui.panels.crossval_panel import CrossValPanel

    _app()
    _temp_settings()
    try:
        xv = CrossValPanel()
        derived_hi = xv.s_grid.hi.text()          # set by _on_cell_changed from the cell file
        xv.s_grid.hi.setText("999.0")             # user 'edits' it to a bogus value
        xv.s_grid.points.setText("13")
        xv.f0.setText("0.077")

        qs = st.settings()
        xv.save_settings(qs)
        qs.sync()
        # the bogus hi must NOT have been written
        qs.beginGroup("crossval")
        assert qs.value("s_hi") is None, "cell-derived s_grid.hi was persisted -- it must not be"
        qs.endGroup()

        xv2 = CrossValPanel()
        assert xv2.s_grid.points.text() == "13", "the free `points` knob was not restored"
        assert xv2.f0.value() == 0.077
        assert xv2.s_grid.hi.text() == derived_hi, "the grid bound must be RE-DERIVED, not restored"
    finally:
        st.use_ini_file(None)


# ── Phase 3: error dialogs ───────────────────────────────────────────────────────────────────────
def test_on_error_puts_the_traceback_in_details_not_the_body(monkeypatch=None):
    """A run failure's traceback belongs in a collapsible Details panel, not pasted whole into the
    dialog body."""
    from PySide6.QtWidgets import QMessageBox

    _app()

    class P(BasePanel):
        pass

    panel = P()
    captured = {}
    orig_exec = QMessageBox.exec

    def fake_exec(self):
        captured["text"] = self.text()
        captured["detail"] = self.detailedText()
        return 0

    QMessageBox.exec = fake_exec
    try:
        panel._on_error("Something failed", "Traceback (most recent call last):\n  ...\nValueError: x")
    finally:
        QMessageBox.exec = orig_exec

    assert captured["text"] == "Something failed"
    assert "Traceback" in captured["detail"], "the traceback was not routed to Details"
    assert "Traceback" not in captured["text"], "the traceback leaked into the body"


# ── interactive "Pop out" for figures ────────────────────────────────────────────────────────────
def _tiny_fig():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot([0, 1, 2], [0, 1, 4])
    return fig


def _png_bytes(fig):
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    return buf.getvalue()


def test_fig_sink_emits_png_and_a_reloadable_pickle():
    """The worker sink must emit BOTH the PNG thumbnail and a pickle that reloads to a real Figure on
    the GUI thread -- that pickle is what "Pop out" rebuilds into an interactive window."""
    import pickle
    import matplotlib.pyplot as plt
    from core.gui.panels.base_panel import _png_fig_sink

    _app()
    sig = WorkerSignals()
    events = []
    sig.figure.connect(lambda title, png, fp: events.append((title, png, fp)))

    _png_fig_sink(sig.figure)("Corner", _tiny_fig())

    assert len(events) == 1, events
    title, png, fp = events[0]
    assert title == "Corner"
    assert png[:4] == b"\x89PNG", "the PNG thumbnail is missing / not a PNG"
    assert fp is not None, "no pickle was shipped for the pop-out"
    assert len(pickle.loads(fp).axes) == 1, "the pickle did not reload to the figure"
    plt.close("all")


def test_fig_sink_pickle_failure_still_emits_the_png():
    """If a figure will not pickle, the run must not break: emit fig_pickle=None and keep the PNG."""
    import pickle
    import matplotlib.pyplot as plt
    from core.gui.panels.base_panel import _png_fig_sink

    _app()
    sig = WorkerSignals()
    events = []
    sig.figure.connect(lambda title, png, fp: events.append((title, png, fp)))

    real = pickle.dumps
    pickle.dumps = lambda *a, **k: (_ for _ in ()).throw(TypeError("cannot pickle this"))
    try:
        _png_fig_sink(sig.figure)("X", _tiny_fig())
    finally:
        pickle.dumps = real

    assert len(events) == 1
    _title, png, fp = events[0]
    assert fp is None, "a pickle failure should degrade to None, not raise or drop the event"
    assert png[:4] == b"\x89PNG", "the PNG must still be emitted when pickling fails"
    plt.close("all")


def test_add_figure_creates_an_interactive_capable_tab():
    import pickle
    import matplotlib.pyplot as plt
    from PySide6.QtWidgets import QPushButton
    from core.gui.widgets.figure_stack import FigureStack

    _app()
    fs = FigureStack()
    fig = _tiny_fig()
    fs.add_figure("Corner", _png_bytes(fig), fig_pickle=pickle.dumps(fig))
    plt.close("all")

    assert fs.count() == 1
    container = fs.widget(0)
    assert getattr(container, "_fig_pickle", None) is not None, "the pickle was not stored on the tab"
    assert any(b.text() == "Pop out" for b in container.findChildren(QPushButton)), "no Pop out button"


def test_pop_out_of_a_pickle_builds_a_qtagg_canvas_and_keeps_pyplot_clean():
    """Popping a pickled figure builds a live FigureCanvasQTAgg on the GUI thread, and -- the linchpin
    -- must NOT leave the reconstructed figure registered in pyplot's Gcf (else the worker's
    plt.close("all") would later close it out from under the user)."""
    import pickle
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from core.gui.widgets.figure_stack import FigureStack

    app = _app()
    fs = FigureStack()
    fig = _tiny_fig()
    fs.add_figure("Corner", _png_bytes(fig), fig_pickle=pickle.dumps(fig))
    plt.close("all")

    before = plt.get_fignums()
    fs._pop_out(fs.widget(0))
    assert len(fs._windows) == 1
    win = next(iter(fs._windows))
    assert isinstance(win.canvas, FigureCanvasQTAgg)
    assert plt.get_fignums() == before, "the unpickled figure leaked into pyplot's Gcf"

    win.close()
    _pump(app, 0.1)
    assert len(fs._windows) == 0, "the window ref was not dropped on close"
    plt.close("all")


def test_pop_out_without_a_pickle_uses_the_image_viewer():
    import matplotlib.pyplot as plt
    from PySide6.QtWidgets import QGraphicsView
    from core.gui.widgets.figure_stack import FigureStack
    from core.gui.widgets.figure_window import ImageZoomWindow

    app = _app()
    fs = FigureStack()
    fig = _tiny_fig()
    fs.add_figure("NoPickle", _png_bytes(fig), fig_pickle=None)
    plt.close("all")

    fs._pop_out(fs.widget(0))
    win = next(iter(fs._windows))
    assert isinstance(win, ImageZoomWindow), "a pickle-less figure should open the image viewer"
    assert isinstance(win.view, QGraphicsView)
    win.close()
    _pump(app, 0.1)


def test_disk_png_pop_out_is_an_image_viewer():
    import tempfile
    import matplotlib.pyplot as plt
    from PySide6.QtWidgets import QGraphicsView
    from core.gui.widgets.figure_stack import FigureStack
    from core.gui.widgets.figure_window import ImageZoomWindow

    app = _app()
    png = Path(tempfile.mkdtemp()) / "sweep.png"
    fig = _tiny_fig()
    fig.savefig(str(png), format="png")
    plt.close("all")

    fs = FigureStack()
    fs.add_png("Sweep", str(png))
    fs._pop_out(fs.widget(0))
    win = next(iter(fs._windows))
    assert isinstance(win, ImageZoomWindow)
    assert isinstance(win.view, QGraphicsView), "the disk-PNG pop-out has no zoom/pan view"
    win.close()
    _pump(app, 0.1)


def test_a_popped_out_figure_survives_a_worker_plt_close_all():
    """Worker.run runs plt.close("all") after every run and every cancel. A figure the user has popped
    out must NOT be torn down by it -- the Gcf detach in build_interactive_window is the guarantee."""
    import pickle
    import matplotlib.pyplot as plt
    from core.gui.widgets.figure_stack import FigureStack

    app = _app()
    fs = FigureStack()
    fig = _tiny_fig()
    fs.add_figure("Corner", _png_bytes(fig), fig_pickle=pickle.dumps(fig))
    plt.close("all")

    fs._pop_out(fs.widget(0))
    win = next(iter(fs._windows))
    assert len(win._fig.axes) == 1

    plt.close("all")     # the worker's teardown, fired process-wide
    assert len(win._fig.axes) == 1, "plt.close('all') destroyed a figure being viewed in a pop-out"

    win.close()
    _pump(app, 0.1)
    plt.close("all")


# ── MAPPI navigation redesign ────────────────────────────────────────────────────────────────────
def test_greeting_maps_hours_to_time_of_day():
    from core.gui.screens.home_screen import greeting
    assert greeting(5) == greeting(11) == "Good morning"
    assert greeting(8) == "Good morning"
    assert greeting(12) == greeting(16) == "Good afternoon"
    assert greeting(14) == "Good afternoon"
    assert greeting(17) == greeting(23) == "Good evening"
    assert greeting(0) == greeting(4) == "Good evening"


def test_nav_shell_back_arrow_tracks_the_screen():
    from PySide6.QtWidgets import QWidget
    from core.gui.screens.nav_shell import NavShell

    _app()
    nav = NavShell()
    for _ in range(3):
        nav.add_screen(QWidget())
    nav.go_home()
    assert nav.btn_back.isHidden(), "back arrow should be hidden on home"
    nav.go_to(2)
    assert not nav.btn_back.isHidden(), "back arrow should show on a section"
    nav.go_home()
    assert nav.btn_back.isHidden(), "back arrow should hide again on home"


def test_main_window_always_opens_on_home():
    from core.gui import settings as st
    _app()
    _temp_settings()
    try:
        qs = st.settings()
        qs.setValue("window/tab", 2)          # a stale key from the old flat-tab layout
        qs.sync()
        from core.gui.main_window import MainWindow
        w = MainWindow()
        assert w.nav.stack.currentIndex() == 0, "the app must always open on the home screen"
    finally:
        st.use_ini_file(None)


def test_inference_tab_gates_follow_the_session():
    from core.gui.screens.inference_screen import InferenceScreen
    from core.gui.session import SbiSession

    _app()
    inf = InferenceScreen()

    def enabled():
        return [inf.tabs.isTabEnabled(i) for i in range(6)]   # Config Simulate Prior Posterior Validate Infer

    inf.session = SbiSession(); inf.refresh_gates()
    assert enabled() == [True, False, False, False, False, False]
    inf.session.cfg = object(); inf.refresh_gates()
    assert enabled() == [True, True, True, True, False, False]
    inf.session.posterior = object(); inf.refresh_gates()
    assert enabled() == [True, True, True, True, False, True], "Infer needs only a posterior; Validate needs priors"
    inf.session.inf_prior = object(); inf.session.force_prior = object(); inf.refresh_gates()
    assert enabled() == [True, True, True, True, True, True]


def test_posterior_from_scratch_is_gated_on_a_prior():
    from core.gui.screens.inference_screen import InferenceScreen
    from core.gui.session import SbiSession

    _app()
    inf = InferenceScreen()
    inf.session = SbiSession(cfg=object()); inf.refresh_gates()
    pp = inf.posterior_panel
    pp.post_picker.combo.setCurrentIndex(0)               # the "(from scratch)" sentinel (allow_new adds it first)
    assert pp.post_picker.selected()[1] is True, "index 0 should be the from-scratch sentinel"
    pp.refresh_local_gates()
    assert not pp.btn_post.isEnabled(), "training from scratch must be disabled without a prior"
    inf.session.inf_prior = object(); inf.session.force_prior = object()
    pp.refresh_local_gates()
    assert pp.btn_post.isEnabled(), "with a prior, training from scratch is allowed"


def test_inference_cell_pickers_repoint_after_config_is_built():
    """The Simulate/Infer cell pickers follow the BUILT config's model (there is no model combo in those
    tabs), so new_session must repoint them."""
    from core.gui.screens.inference_screen import InferenceScreen

    _app()
    inf = InferenceScreen()

    class Cfg:
        model = "HOPF"
        force_params_dict = {}

    inf.new_session(Cfg())
    assert inf.simulate_panel.cell_picker.base_path.name == "hopf"
    assert inf.infer_panel.cell_picker.base_path.name == "hopf"


def test_help_badge_carries_its_text():
    from core.gui.widgets.help_badge import HelpBadge
    _app()
    assert HelpBadge("what this does").toolTip() == "what this does"


def test_simulated_preview_runner_emits_the_ground_truth_figure():
    """The Simulate tab's runner produces a 'Ground-truth trace' figure. A real SDE sim is too slow for
    a unit test, so stub the heavy pieces and assert the fig_sink wiring."""
    import torch
    from core import cli, orchestrator
    from core.gui.panels import inference_tabs

    _app()

    class Cfg:
        def get_unit_conversion_factor(self, _unit):
            return 1.0

    seen = []
    real_gt, real_go = cli.load_and_validate_gt, orchestrator.generate_observations
    cli.load_and_validate_gt = lambda cfg, path: None
    orchestrator.generate_observations = lambda cfg: (
        torch.zeros(1, 5), None, torch.linspace(0, 1, 5).unsqueeze(0))
    try:
        inference_tabs._run_simulated_preview(
            Cfg(), "cell.txt", 0.1, fig_sink=lambda title, fig: seen.append(title))
    finally:
        cli.load_and_validate_gt = real_gt
        orchestrator.generate_observations = real_go

    assert seen == ["Ground-truth trace"], seen


if __name__ == "__main__":
    _app()
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as e:
                failures += 1
                print(f"FAIL  {name}\n      {e}")
    print(f"\n{'ALL PASSED' if not failures else f'{failures} FAILURE(S)'}")
    raise SystemExit(1 if failures else 0)
