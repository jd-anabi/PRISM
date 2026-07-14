"""Live progress: a solver line, an overall bar with a spinner, and one row per active tqdm bar.

    Solver Performance: ++++  (13.3k it/s)   [▓▓▓▓▓▓▓░░░░░░]
    [███████░░░░░░░░░░░░░  38%]  ⠹
        Training neural posterior
        Generating training data  —  1902/5000 [05:12<13:41]

Two things this widget exists to solve.

WHY PROGRESS DOES NOT SHARE THE LOG PANE. The pipeline nests bars up to three deep, and inlining
several redrawing bars into a scrolling text widget is what produced the append storm this replaces.
Each bar owns a row, keyed by its tqdm `pos`, and simply overwrites it.

WHY THE SOLVER GETS ITS OWN LINE INSTEAD OF A ROW. A top-level iteration ("Generating training data")
takes ~10 seconds, so the overall bar sits still and the GUI reads as frozen. The thing that IS moving
is the SDE solver underneath it (core/Solvers/sdeint.py), and its `it/s` is the number the user wants.
But it cannot be a row: a posterior build constructs 10k-30k of those bars -- one per time segment,
each alive 1-10s -- so a row would mean creating and destroying a widget every few seconds. It gets one
fixed line instead, whose rate is HELD across the gaps between bars.
"""
import math
import time

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QProgressBar, QSizePolicy, QVBoxLayout, QWidget)

_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_SPIN_MS = 100

# A solver bar shorter than its mininterval (1.0s) paints only a "?it/s" opening frame, and with
# leave=False there is no final 100% frame either -- so rate samples arrive in gaps. Hold the last one
# for this long before calling the solver idle. Comfortably longer than the ~1s between rate frames,
# and short enough that neural-network training (where the solver genuinely is not running) reads idle.
SOLVER_IDLE_S = 45.0

# No output of ANY kind for this long means the run is probably wedged. Say so, instead of spinning
# cheerfully at a corpse.
STALL_S = 45.0


def plus_meter(rate: float | None) -> str:
    """One '+' per order of magnitude of iterations/sec: 10 -> '+', 10_000 -> '++++'.

    Under 10 it/s there is no order of magnitude to show, so render a '·' rather than an empty string --
    otherwise the line reads as broken rather than as slow.
    """
    if rate is None or rate <= 0:
        return "—"
    n = max(0, min(9, int(math.floor(math.log10(rate)))))
    return "+" * n if n else "·"


def format_rate(rate: float | None) -> str:
    if rate is None or rate <= 0:
        return "idle"
    if rate >= 1000:
        return f"{rate / 1000:.1f}k it/s"
    if rate >= 10:
        return f"{rate:.0f} it/s"
    return f"{rate:.1f} it/s"


class _BarRow(QWidget):
    """One tqdm bar: a label for the description and a bar for the percentage."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.label = QLabel()
        # Pin the size policy: QLabel.setText() calls updateGeometry() unconditionally, so an
        # unpinned label re-lays-out the whole right-hand pane on every frame (~60/s across all
        # rows), visibly jittering the figure/log splitter as the frame text changes width.
        self.label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.label.setTextFormat(Qt.PlainText)

        self.bar = QProgressBar()
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(6)
        self.bar.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 2)
        layout.setSpacing(1)
        layout.addWidget(self.label)
        layout.addWidget(self.bar)

    def update_from(self, state) -> None:
        depth = "    " * state.row
        stats = f"  —  {state.stats}" if state.stats else ""
        self.label.setText(f"{depth}{state.desc}{stats}")
        if state.pct is None:
            self.bar.setRange(0, 0)              # indeterminate: this bar has no total
        else:
            self.bar.setRange(0, 100)
            self.bar.setValue(state.pct)


class ProgressPane(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._rows: dict[tuple, _BarRow] = {}
        self._rate: float | None = None
        self._rate_at = 0.0
        self._beat_at = 0.0
        self._frame = 0

        # ── solver line: the rate meter + the live step strip ────────────────
        self.solver_label = QLabel()
        self.solver_label.setTextFormat(Qt.PlainText)
        self.solver_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        self.solver_strip = QProgressBar()
        self.solver_strip.setTextVisible(False)
        self.solver_strip.setFixedHeight(8)
        self.solver_strip.setToolTip("Progress through the current SDE integration segment")

        solver_line = QWidget()
        solver_layout = QHBoxLayout(solver_line)
        solver_layout.setContentsMargins(0, 0, 0, 0)
        solver_layout.setSpacing(8)
        solver_layout.addWidget(self.solver_label)
        solver_layout.addWidget(self.solver_strip, 1)

        # ── overall bar + spinner ───────────────────────────────────────────
        self.overall = QProgressBar()
        self.overall.setRange(0, 0)              # indeterminate until something reports a percentage

        self.spinner = QLabel()
        self.spinner.setTextFormat(Qt.PlainText)
        # Fixed width: the spinner cycles glyphs and swaps to a stall message, and an unpinned label
        # would re-lay-out the pane on every 100ms tick.
        self.spinner.setMinimumWidth(150)
        self.spinner.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        overall_line = QWidget()
        overall_layout = QHBoxLayout(overall_line)
        overall_layout.setContentsMargins(0, 0, 0, 0)
        overall_layout.setSpacing(8)
        overall_layout.addWidget(self.overall, 1)
        overall_layout.addWidget(self.spinner)

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 2, 0, 2)
        self._layout.setSpacing(2)
        self._layout.addWidget(solver_line)
        self._layout.addWidget(overall_line)

        # Drives the spinner AND the two timeouts -- staleness has to be noticed precisely when no
        # events are arriving, so it cannot be evaluated from set_rows() alone.
        self._timer = QTimer(self)
        self._timer.setInterval(_SPIN_MS)
        self._timer.timeout.connect(self._tick)

        self._reset_solver()
        self.setVisible(False)

    # ── lifecycle ────────────────────────────────────────────────────────────
    def begin(self) -> None:
        self.end()
        self.heartbeat()
        self.setVisible(True)
        self._timer.start()

    def end(self) -> None:
        """Authoritative teardown. Deletes every row regardless of whether its bar reported a close,
        so a row leaked by a crashed worker cannot survive into the next dispatch."""
        self._timer.stop()
        for row in self._rows.values():
            self._layout.removeWidget(row)
            row.deleteLater()
        self._rows.clear()
        self.overall.setRange(0, 0)
        self.spinner.setText("")
        self._reset_solver()
        self.setVisible(False)

    def heartbeat(self) -> None:
        """Mark 'the run produced output just now'. Called on every rows snapshot AND on every batch of
        log lines -- a run that is printing but not drawing bars is still alive."""
        self._beat_at = time.monotonic()

    # ── the worker's rows signal lands here ──────────────────────────────────
    def set_rows(self, snapshot) -> None:
        """`snapshot` is the pump's full set of live rows (a tuple[RowState]), already sorted."""
        self.heartbeat()

        solver = next((s for s in snapshot if s.is_solver), None)
        rows = [s for s in snapshot if not s.is_solver]

        seen = set()
        for state in rows:
            seen.add(state.key)
            row = self._rows.get(state.key)
            if row is None:
                row = _BarRow(self)
                self._rows[state.key] = row
                self._layout.addWidget(row)
            row.update_from(state)

        for key in [k for k in self._rows if k not in seen]:
            row = self._rows.pop(key)
            self._layout.removeWidget(row)
            row.deleteLater()

        self._update_solver(solver)
        self._retarget(rows)

    # ── the solver line ──────────────────────────────────────────────────────
    def _reset_solver(self) -> None:
        self._rate = None
        self._rate_at = 0.0
        self.solver_strip.setRange(0, 100)
        self.solver_strip.setValue(0)
        self.solver_strip.setVisible(False)
        self._paint_solver()

    def _update_solver(self, solver) -> None:
        if solver is None:
            self.solver_strip.setVisible(False)      # no solver running (e.g. NN training, Reduction)
        else:
            self.solver_strip.setVisible(True)
            if solver.pct is None:
                self.solver_strip.setRange(0, 0)
            else:
                self.solver_strip.setRange(0, 100)
                self.solver_strip.setValue(solver.pct)
            if solver.rate is not None:              # None on the opening frame of every bar
                self._rate, self._rate_at = solver.rate, time.monotonic()
        self._paint_solver()

    def _paint_solver(self) -> None:
        rate = self._rate
        if rate is not None and time.monotonic() - self._rate_at > SOLVER_IDLE_S:
            rate = None                              # held sample has gone stale: the solver stopped
        self.solver_label.setText(f"Solver Performance: {plus_meter(rate)}  ({format_rate(rate)})")
        if rate is None:
            self.solver_label.setToolTip("The SDE solver is not running right now.")
        else:
            self.solver_label.setToolTip(
                f"SDE solver: {rate:,.0f} integration steps/sec.\nOne '+' per order of magnitude.")

    # ── the spinner + stall detection ────────────────────────────────────────
    def _tick(self) -> None:
        idle = time.monotonic() - self._beat_at
        if idle > STALL_S:
            # Freeze the spinner rather than animate it. A spinner that keeps twirling on a wedged run
            # is worse than none: it actively asserts progress that is not happening.
            self.spinner.setText(f"⏳ no output for {int(idle)}s")
            return
        self._frame = (self._frame + 1) % len(_SPINNER)
        self.spinner.setText(_SPINNER[self._frame])
        self._paint_solver()                         # re-evaluate the solver's own idle timeout

    def _retarget(self, rows) -> None:
        """Drive the overall bar from the DEEPEST live row that reports an informative percentage.

        `rows` EXCLUDES the solver bar, and must: the solver has a total in the tens of thousands and is
        the deepest bar there is, so it would win this election every time and drag the overall bar
        through a full 0->100% sweep every second or two. The overall bar's job is the top-level count.

        Nor the outermost, and nor a sticky first-seen "driver" -- both of those peg the bar:
          * the pos-0 bar ("Training neural posterior", core/SBI/pipeline.py:517) wraps
            range(TRAINING_NUM_ROUNDS) and that is 1 (core/config.py:104), so it reads 0% for the
            entire multi-hour build -- hence RowState.informative excludes total<=1 bars;
          * sbi's neural-network training emits no tqdm bar at all (only a printed epoch counter), so
            a driver latched onto "Generating training data" would sit at 100% through the longest
            phase, which reads as finished or hung.
        """
        live = sorted((s for s in rows if s.informative), key=lambda s: s.row, reverse=True)
        if not live:
            self.overall.setRange(0, 0)
        else:
            self.overall.setRange(0, 100)
            self.overall.setValue(live[0].pct)
