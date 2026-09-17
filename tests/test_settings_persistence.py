"""QSettings round-trips, the training-budget group, layout geometry, and the config-tab hardware fields. Split from test_gui_progress.py; run directly: pytest tests/test_settings_persistence.py"""
"""Progress-rendering regression tests for the GUI.

THE BUG THESE LOCK DOWN
    tqdm redraws a bar at pos>0 as three writes -- '\\n'*pos, then '\\r'+frame, then '\\x1b[A'*pos
    (tqdm/std.py:1493-1497). The old stream reader split on terminators, so the frame (which is never
    terminated) stranded in its buffer and was flushed by the NEXT redraw's leading '\\n' -- i.e. as a
    LOG LINE. Every nested-bar redraw appended one row, so a training run buried the log pane under
    hundreds of bar snapshots.

    The pipeline nests bars four deep (core/SBI/pipeline.py:517 -> :371 ->
    core/Simulator/simulator.py:50 -> core/Solvers/sdeint.py:15), so this fired constantly.

Run:  pytest tests/test_settings_persistence.py
      (or just: pytest tests/test_gui_progress.py)
"""
import ast
import inspect
import textwrap
import os
import tempfile
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")   # must precede any PySide6 import
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib                                                 # noqa: E402
matplotlib.use("Agg")                                            # match the app (core/gui/__main__.py forces it)

import torch                                                      # noqa: E402
from PySide6.QtWidgets import QApplication                        # noqa: E402
from tqdm import tqdm                                             # noqa: E402

from core.gui.panels.base_panel import BasePanel                  # noqa: E402
from core.gui.streams import redirect_streams                     # noqa: E402
from core.gui.vt import StreamRouter, parse_bar                   # noqa: E402
from core.gui.widgets.log_pane import LogPane                     # noqa: E402
from core.gui.widgets.progress_pane import ProgressPane           # noqa: E402
from core.gui.worker import WorkerSignals                         # noqa: E402
from tests._fixtures import code_only, pump, qt_app               # noqa: E402
import contextlib                                                  # noqa: E402
import pytest                                                      # noqa: E402

# -- the training budget (Posterior tab) ----------------------------------------------------------
def _budget_cfg():
    """Stub SimConfig carrying exactly the fields the budget lines read.

    A stub, not a real build: the arithmetic under test is geometry -> elements -> GiB, and a real
    make_sim_config drags in bounds files and a 300k-point time grid for nothing.
    """
    from core import config

    class Cfg:
        hw = config.detect_device()
        t = type("T", (), {"shape": (250_000,)})()
        inits_dict = {"x": 0.0, "xa": 0.0, "f": 0.0}
        # training_preview reads n_vars as orchestrator._observation_inits(cfg).shape[-1], and with
        # inits_dict set that function returns this tensor.
        inits_tensor = torch.zeros(1, 3)
        steady_idx = 500
        forcing_idx = {}
        model = "NADROWSKI"
        chi_mode = False
        chi_k_pad = 12
        observation_mode = "spontaneous"
    return Cfg()
def _prior_stub():
    """A LoadedPrior-shaped stub: the Posterior tab now reads ``.prior``/``.force_prior`` off
    session.inf_prior (piece 1, Task 4) rather than carrying the physical prior and forcing prior as
    separate session fields, so a bare ``object()`` no longer stands in where a dispatched call is
    actually reached."""
    return type("LoadedPriorStub", (), {"prior": object(), "force_prior": object()})()
def _budget_panel(cfg=None, prior=None):
    from core.gui.screens.inference_screen import InferenceScreen
    from core.gui.session import SbiSession

    qt_app()
    inf = InferenceScreen()
    inf.session = SbiSession()
    inf.session.cfg = cfg
    inf.session.inf_prior = prior
    inf.refresh_gates()
    return inf, inf.tabs.widget(2)          # Config Prior Posterior Validate Infer


def test_the_training_budget_shows_the_simulation_count_and_the_cap_trade():
    """The count was invisible: 5000 x 2048 = 10,240,000 simulations lived only in config.py. The line
    also has to say what the cap really trades, because batch width is NOT a speed knob -- the solver
    is kernel-launch-bound (measured 7.37 s at 2048 against 7.74 s at 1024, the SMALLER batch being
    slightly slower), so halving it halves the training rows for the same wall-clock.

    ⚠ RUNS AGAINST AN EMPTY SETTINGS FILE, and must (tests/conftest.py::_isolated_settings gives every
    test a fresh, absent one; this test used to redirect by hand). The panel seeds these fields from
    config.py and then RESTORES them from QSettings, so without isolation the assertions read whatever
    the developer last left in their real PRISM.ini. This test passed all morning and then failed the
    moment a run was configured with 10000 batches -- nothing to do with the code under test. It is
    the same restore-wins-over-config mechanism that cost a ~5-day run on 2026-08-19 (the retired chi
    band), so a test asserting the DEFAULT has to start from a file with no saved value.
    """
    from core import config

    cfg = _budget_cfg()
    _inf, panel = _budget_panel(cfg)
    width = cfg.hw.batch_size

    assert panel.num_runs.value() == config.TRAINING_NUM_RUNS, (
        f"the field must seed from config.py, not from a saved session: "
        f"{panel.num_runs.value()} != {config.TRAINING_NUM_RUNS}")
    assert panel.run_size_cap.value() == config.TRAINING_RUN_SIZE
    assert f"{config.TRAINING_NUM_RUNS * width:,} simulations" in panel.budget_total.text(), \
        panel.budget_total.text()
    assert "diversity" in panel.budget_total.text(), \
        "the line must say batch COUNT is the (t_scale, T) diversity, not just a budget"

    panel.run_size_cap.setText(str(width // 4))
    assert f"{config.TRAINING_NUM_RUNS * (width // 4):,} simulations" in panel.budget_total.text()
    assert "capped from" in panel.budget_total.text(), panel.budget_total.text()

    # ...and quadrupling the batch COUNT is what buys those rows back, at 4x the wall-clock.
    panel.num_runs.setText(str(config.TRAINING_NUM_RUNS * 4))
    assert f"{config.TRAINING_NUM_RUNS * width:,} simulations" in panel.budget_total.text()

def test_the_training_budget_reaches_build_posterior_as_arguments_not_via_config():
    """THE WIRING THAT MAKES THE FIELDS DO ANYTHING AT ALL.

    orchestrator binds TRAINING_NUM_RUNS / TRAINING_RUN_SIZE at import, so a panel that "applied" the
    user's numbers by assigning to core.config would be a silent no-op -- the run would simulate 5000
    x 2048 anyway and nothing would say otherwise. Both halves are asserted: the values arrive as
    call kwargs, AND the config constants are untouched.
    """
    from core import config

    before = (config.TRAINING_NUM_RUNS, config.TRAINING_RUN_SIZE)
    _inf, panel = _budget_panel(_budget_cfg(), prior=_prior_stub())
    panel.num_runs.setText("777")
    panel.run_size_cap.setText("256")

    seen = {}
    panel.dispatch = lambda fn, *a, **kw: seen.update(fn=fn, args=a, kwargs=kw)
    panel._build_posterior()

    assert seen, "the Train button dispatched nothing"
    assert seen["kwargs"].get("num_runs") == 777, seen["kwargs"]
    assert seen["kwargs"].get("run_size_cap") == 256, seen["kwargs"]
    assert (config.TRAINING_NUM_RUNS, config.TRAINING_RUN_SIZE) == before, \
        "the panel mutated the config constants, which orchestrator has already snapshotted"

def test_the_budget_refuses_a_batch_count_below_one():
    """0 batches is a whole run that simulates nothing and then trains on an empty tensor.

    The refusal is the CLICK's, through the shared rule, and it arrives as the yellow "Check your
    inputs" box (BasePanel._refusal, stubbed here) naming the Batches box by its field key -- not
    as a log line the user has to notice. Nothing is dispatched."""
    from core.refusals import Refusal

    _inf, panel = _budget_panel(_budget_cfg(), prior=object())
    panel.num_runs.setText("0")
    seen, refused = [], []
    panel.dispatch = lambda fn, *a, **kw: seen.append(kw)
    panel._refusal = lambda exc: refused.append(exc)
    panel._build_posterior()
    assert not seen, "a zero batch count was dispatched"
    assert len(refused) == 1 and isinstance(refused[0], Refusal), refused
    assert refused[0].field == "num_runs", refused[0].field
    assert "at least 1" in str(refused[0]) and "got 0" in str(refused[0]), str(refused[0])

def test_the_budget_memory_line_reads_pipelines_own_cost_model():
    """The estimate must come from pipeline.peak_sim_elements, not a second copy of the formula, and
    it must be quoted at the WORST geometry the Sobol pre-filter admits -- n_fine swings from a median
    ~40k to a p99 ~283k, so a width that fits the median still OOMs on a few percent of batches, which
    is how two retrains actually died."""
    from core import config
    from core.SBI import pipeline

    cfg = _budget_cfg()
    _inf, panel = _budget_panel(cfg)
    width = cfg.hw.batch_size

    if cfg.hw.device.type != "cuda":
        assert "CUDA-only" in panel.budget_mem.text(), panel.budget_mem.text()
        return

    n_fine = min(config.N_ND_MAX, cfg.t.shape[0])
    need = pipeline.peak_sim_elements(width, n_fine, cfg.steady_idx, cfg.inits_tensor.shape[-1], 1, 1)
    gib = need * cfg.hw.dtype.itemsize / float(1 << 30)
    assert f"{gib:.2f} GiB" in panel.budget_mem.text(), panel.budget_mem.text()
    assert f"{n_fine:,}" in panel.budget_mem.text(), "the line must name the geometry it assumed"
    assert "upper bound" in panel.budget_mem.text().lower(), \
        "free-VRAM readings overstate what is available; the line must not present one as fact"

    # Halving the width must halve the estimate -- the peak is linear in the batch.
    panel.run_size_cap.setText(str(width // 2))
    assert f"{gib / 2:.2f} GiB" in panel.budget_mem.text(), panel.budget_mem.text()

def test_the_budget_lines_never_raise_on_a_config_they_do_not_understand(monkeypatch):
    """_sync_budget runs from refresh_gates(), so an exception in a STATUS LINE would take down the
    whole tab. The gate tests set session.cfg to a bare object(); so could any future stub.

    Since piece 3 (V6) the tab no longer swaps such a config for None: it goes into
    orchestrator.training_preview as it is, so this pins the preview's FAIL-SOFT branch. A config the
    preview cannot read lands in estimate_error and checkpoint_error, and the lines say "unavailable"
    with the error. The session's cadence is 0 (tests/conftest.py rebinds orchestrator's copy), which
    keeps the checkpoint line at "off" and the word "config"; checkpointing is turned on here to reach
    the identity at all. With no config the checkpoint line still carries the word "config"."""
    from core import orchestrator

    inf, panel = _budget_panel(cfg=object(), prior=object())
    panel._sync_budget()                                   # must not raise
    assert panel.budget_total.text(), "the total line went blank on an unknown config"
    assert "config" in panel.budget_ckpt.text().lower(), panel.budget_ckpt.text()

    monkeypatch.setattr(orchestrator, "TRAINING_CHECKPOINT_EVERY", 50)
    panel._sync_budget()                                   # must not raise
    assert "simulations" in panel.budget_total.text(), panel.budget_total.text()
    assert panel.budget_ckpt.text().startswith("Checkpoint status unavailable: AttributeError"), \
        panel.budget_ckpt.text()
    mem = panel.budget_mem.text()
    assert "CUDA-only" in mem or mem.startswith("Peak-memory estimate unavailable: AttributeError"), mem

    inf.session.cfg = None
    panel._sync_budget()                                   # must not raise
    ckpt = panel.budget_ckpt.text()
    assert "config" in ckpt.lower() and "needs a config and a prior" in ckpt, ckpt
    mem = panel.budget_mem.text()
    assert "CUDA-only" in mem or mem.endswith("(Estimated from hardware defaults until a config is built.)"), mem

def test_the_budget_lines_name_a_blank_or_half_typed_box_and_nothing_raises():
    """The mixin used to read value() -- 0 for "" and for a lone "-" mid-typing -- and then
    max(1, ...) / max(0, ...) the result, so a blank Batches box showed the budget for ONE batch and
    a negative cap showed the hardware width, both as if someone had typed them. A live line cannot
    pop a dialog, so it says which box is wrong, in the box's own label, and shows no number until
    it is fixed: no clamp, no default, no raise (V2 for a live line). The memory and checkpoint
    lines go empty with it -- an estimate for a width nobody asked for is the same lie in GiB."""
    from core.gui.fields import label

    _inf, panel = _budget_panel(cfg=object(), prior=object())
    for text in ("", "-"):
        panel.num_runs.setText(text)                        # textChanged -> _sync_budget; must not raise
        assert panel.budget_total.text() == f"{label('num_runs')} is blank.", panel.budget_total.text()
        assert panel.budget_mem.text() == "" and panel.budget_ckpt.text() == ""
    panel.num_runs.setText("0")
    assert panel.budget_total.text() == f"{label('num_runs')} must be at least 1.", panel.budget_total.text()
    panel.num_runs.setText("3")
    panel.run_size_cap.setText("-5")
    assert panel.budget_total.text() == f"{label('run_size_cap')} must be at least 0.", panel.budget_total.text()
    assert panel.budget_mem.text() == "" and panel.budget_ckpt.text() == ""
    panel.run_size_cap.setText("")
    assert panel.budget_total.text() == f"{label('run_size_cap')} is blank.", panel.budget_total.text()
    # A TYPED 0 in the cap box is "automatic", not blank: the lines come back, for 3 batches.
    panel.run_size_cap.setText("0")
    assert "simulations" in panel.budget_total.text() and "3 batches" in panel.budget_total.text(), \
        panel.budget_total.text()
    assert panel.budget_ckpt.text(), "the checkpoint line must come back once both boxes pass"

def test_the_tsnpe_tab_never_claims_it_will_resume_the_amortized_checkpoint(monkeypatch):
    """D3's user-facing face. The TSNPE tab shares the Posterior tab's budget group, whose
    checkpoint line is computed from the AMORTIZED identity -- so at the parent's budget it read
    "Resumes a COMPLETE checkpoint ... simulation will be skipped entirely", which is exactly what a
    round at that budget did before the region became part of the identity. The region is drawn
    when the round starts, so the tab cannot resolve a directory in advance; it states the rule.

    Whether checkpointing is on at all is the TRAINING STAGE's binding of the cadence
    (orchestrator.TRAINING_CHECKPOINT_EVERY, read through training_preview), so that is the one this
    test rebinds -- and config's live copy is set to the OPPOSITE value each time, because the tab used
    to read that copy and a round never does."""
    from core import config as _cfg, orchestrator
    inf, _ = _budget_panel(cfg=_budget_cfg(), prior=object())
    panel = inf.tsnpe_panel
    monkeypatch.setattr(orchestrator, "TRAINING_CHECKPOINT_EVERY", 50)
    monkeypatch.setattr(_cfg, "TRAINING_CHECKPOINT_EVERY", 0)
    panel._sync_budget()
    inf.posterior_panel._sync_budget()
    text = panel.budget_ckpt.text()
    assert "Resumes" not in text and "OWN identity" in text, text
    assert text != inf.posterior_panel.budget_ckpt.text(), \
        "the TSNPE tab shows the Posterior tab's amortized checkpoint line"
    monkeypatch.setattr(orchestrator, "TRAINING_CHECKPOINT_EVERY", 0)
    monkeypatch.setattr(_cfg, "TRAINING_CHECKPOINT_EVERY", 50)
    panel._sync_budget()
    assert "off" in panel.budget_ckpt.text().lower(), panel.budget_ckpt.text()


def test_the_budget_lines_only_format_the_preview(monkeypatch):
    """V6. The three lines under the training budget used to DERIVE what they showed: the width
    (_effective_width), the memory geometry (_budget_memory's own n_fine / n_vars / steady), the cache
    directory (resolve_dir under the process-default root) and the cadence (config's LIVE copy of
    TRAINING_CHECKPOINT_EVERY, which the training stage never reads). Four places for the line and the
    run to disagree. Now ONE orchestrator call resolves all of it as build_posterior does, and the tabs
    only format what it returns.

    Pinned three ways. Every line is exactly the formatting of a preview the test hands in -- a
    deliberately inconsistent one (width 700 against a 512 box), so a tab that recomputed anything
    would show a different number. The call receives the session's config and prior as they are, the
    two boxes as ints and the tab's memoised hardware, and is not made at all while a box is blank
    (the preview takes ints only). And neither the mixin nor the TSNPE override names any of the
    machinery it used to call."""
    import ast
    import dataclasses
    import inspect
    import textwrap
    from core import orchestrator
    from core.gui.fields import label
    from core.gui.panels.inference.base import _TrainingBudgetMixin
    from core.gui.panels.inference.tsnpe_tab import TSNPEPanel

    cfg, prior = _budget_cfg(), _prior_stub()
    inf, panel = _budget_panel(cfg, prior=prior)
    tsnpe = inf.tsnpe_panel
    base = orchestrator.TrainingPreview(
        n_runs=3, width=700, hw_batch=2048, n_fine=250_000, n_vars=3,
        need_elements=2 ** 28, have_elements=2 ** 30, estimate_error=None,
        checkpoint="new", batches_done=0, siblings="", cadence=50, checkpoint_error=None,
        device_type="cuda", itemsize=4)
    handed = {"preview": base}
    calls = []

    def fake(c, p, **kw):
        calls.append((c, p, kw))
        return handed["preview"]

    monkeypatch.setattr(orchestrator, "training_preview", fake)

    def lines(tab, preview):
        handed["preview"] = preview
        calls.clear()
        tab._sync_budget()
        return tab.budget_total.text(), tab.budget_mem.text(), tab.budget_ckpt.text()

    panel.num_runs.setText("3")
    panel.run_size_cap.setText("512")
    total, mem, _ckpt = lines(panel, base)

    # the call: the session's objects as they are, the boxes as ints, the tab's memoised hardware
    assert len(calls) == 1, calls
    c, p, kw = calls[0]
    assert c is cfg and p is prior, "the preview must see the session's config and prior as they are"
    assert kw == {"num_runs": 3, "run_size_cap": 512, "hw": panel._hardware()}, kw
    assert type(kw["num_runs"]) is int and type(kw["run_size_cap"]) is int, kw
    assert kw["hw"] is panel._hardware(), "detect_device() is memoised by the tab, not probed per keystroke"

    # the total line: the PREVIEW's width (700), never the box's 512
    assert total == ("2,100 simulations = 3 batches x 700 rows (capped from 2,048).\nBatches is also the "
                     "(t_scale, T) diversity count: every row in a batch shares one operating point, so "
                     "batch COUNT is the statistics and batch WIDTH is not."), total
    assert "capped from" not in lines(panel, dataclasses.replace(base, width=2048))[0]

    # the memory line
    assert mem.startswith("Worst-case peak ~1.00 GiB per batch (n_fine <= 250,000, 3 state vars); "
                          "planner budget right now ~4.00 GiB, so it fits in one piece.\n"), mem
    assert "upper bound" in mem.lower() and "hardware defaults" not in mem, mem
    assert "does NOT fit" in lines(panel, dataclasses.replace(base, need_elements=2 ** 31))[1]
    assert lines(panel, dataclasses.replace(base, device_type="cpu"))[1] == \
        "Peak-memory estimate is CUDA-only; this config runs on cpu."
    failed = dataclasses.replace(base, estimate_error="RuntimeError: boom", n_fine=None, n_vars=None,
                                 need_elements=None, have_elements=None)
    assert lines(panel, failed)[1] == "Peak-memory estimate unavailable: RuntimeError: boom"

    # the checkpoint line, state by state; the TSNPE tab states its rule whenever checkpointing is on
    sib = ("[checkpoint] 1 other checkpoint(s) exist and do NOT match this run: "
           "abc123def456 (2 batches, differs in n_runs)")
    want = {
        "off": "Checkpointing is off (config.TRAINING_CHECKPOINT_EVERY = 0): a crash loses the run.",
        "needs_config": "Checkpoint status needs a config and a prior -- the prior is part of the identity.",
        "resume_complete": ("Resumes a COMPLETE checkpoint (2 batches) -- simulation will be skipped "
                            "entirely and only the flow retrained."),
        "resume_partial": "Resumes an existing checkpoint: 2/3 batches already done.",
        "new_with_siblings": "WARNING: these settings match no checkpoint, so this starts a NEW run.\n" + sib,
        "new": "No checkpoint exists yet; this starts a new run.",
    }
    for state, text in want.items():
        pv = dataclasses.replace(base, checkpoint=state,
                                 batches_done=2 if state.startswith("resume") else 0,
                                 siblings=sib if state == "new_with_siblings" else "",
                                 cadence=0 if state == "off" else 50)
        assert lines(panel, pv)[2] == text, (state, panel.budget_ckpt.text())
        rule = lines(tsnpe, pv)[2]
        if state == "off":
            assert rule == text, "with checkpointing off the TSNPE tab says what the Posterior tab says"
        else:
            assert "OWN identity" in rule and "Resumes" not in rule, (state, rule)
    assert lines(panel, dataclasses.replace(base, checkpoint_error="OSError: disk gone"))[2] == \
        "Checkpoint status unavailable: OSError: disk gone"

    # no config yet: the memory line says its geometry was the hardware defaults
    inf.session.cfg = None
    assert lines(panel, base)[1].endswith("\n(Estimated from hardware defaults until a config is built.)")
    assert calls[0][0] is None
    inf.session.cfg = cfg

    # a blank box: the preview is not called at all
    panel.num_runs.setText("")
    calls.clear()
    panel._sync_budget()
    assert calls == [] and panel.budget_total.text() == f"{label('num_runs')} is blank.", calls

    # neither formatter names the machinery it used to call (names and attributes, not string text:
    # the "off" line keeps the words config.TRAINING_CHECKPOINT_EVERY on purpose)
    def names(obj):
        tree = ast.parse(textwrap.dedent(inspect.getsource(obj)))
        return ({n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
                | {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)})

    gone = {"peak_sim_elements", "sim_memory_budget_elements", "n_force_channels", "training_identity",
            "training_checkpoint", "peek", "describe_siblings", "resolve_dir", "TRAINING_CHECKPOINT_EVERY",
            "N_ND_MAX", "_effective_width", "batch_size", "steady_idx", "inits_dict"}
    mixin = names(_TrainingBudgetMixin)
    assert "training_preview" in mixin and not (mixin & gone), sorted(mixin & gone)
    override = names(TSNPEPanel._budget_checkpoint)
    assert not (override & {"config", "TRAINING_CHECKPOINT_EVERY", "orchestrator"}), sorted(override)

def test_the_training_budget_round_trips_through_settings():
    """"I have to retype it every launch" is the complaint L1 already answered for splitters.

    The budget stays REMEMBERED under V5, and is saved as the boxes' TEXT (spec §5.1). The old save
    wrote value(), which reads a blank box as 0, so a Batches box left empty at close came back on
    the next launch as a 0 nobody typed -- a run that simulates nothing, caught only at the next
    click. Saved as "", get_int falls back to config.py's default exactly as for a missing key.
    """
    from core import config
    from core.gui import settings as st

    _inf, panel = _budget_panel(_budget_cfg())
    panel.num_runs.setText("1234")
    panel.run_size_cap.setText("512")
    panel.save_settings(st.settings())

    _inf2, fresh = _budget_panel(_budget_cfg())
    fresh.restore_settings(st.settings())
    assert fresh.num_runs.value() == 1234, fresh.num_runs.text()
    assert fresh.run_size_cap.value() == 512, fresh.run_size_cap.text()
    # And the derived line followed the restored values, not the defaults.
    assert "1,234 batches" in fresh.budget_total.text(), fresh.budget_total.text()

    # Both boxes BLANK at close: the file holds the text "", and the next launch opens at config.py.
    panel.num_runs.setText("")
    panel.run_size_cap.setText("")
    qs = st.settings()
    panel.save_settings(qs)
    qs.sync()
    qs.beginGroup("inference_posterior")
    raw = (str(qs.value("num_runs")), str(qs.value("run_size_cap")))
    qs.endGroup()
    assert raw == ("", ""), f"the budget must be saved as the boxes' text, not value()'s 0: {raw}"
    _inf3, relaunched = _budget_panel(_budget_cfg())
    assert relaunched.num_runs.text() == str(config.TRAINING_NUM_RUNS), relaunched.num_runs.text()
    assert relaunched.run_size_cap.text() == str(config.TRAINING_RUN_SIZE), relaunched.run_size_cap.text()

def test_settings_round_trip_reduction_and_fdt():
    from core.gui import settings as st
    from core.gui.panels.reduction_panel import ReductionPanel
    from core.gui.panels.fdt_panel import FdtPanel

    qt_app()
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
    # The two FDT boxes are consents (V5, spec §1.2): saved flipped, they still open at the
    # construction defaults -- the sanity checks run, and the production sweep follows them.
    assert fdt2.skip_sanity.isChecked() is False
    assert fdt2.confirm_production.isChecked() is True

def test_missing_picker_key_restores_to_default_not_blank():
    """A saved selection whose file is gone must leave the picker at its default, never blank it via
    setCurrentIndex(-1)."""
    from core.gui import settings as st
    from core.gui.panels.reduction_panel import ReductionPanel

    qt_app()
    qs = st.settings()
    qs.beginGroup("reduction")
    qs.setValue("cell", "nadrowski/does_not_exist.txt")
    qs.setValue("f0", "0.05")
    qs.endGroup()
    qs.sync()

    red = ReductionPanel()
    assert red.cell_picker.combo.currentIndex() >= 0, "a stale key blanked the combo"

def test_crossval_does_not_persist_cell_derived_bounds():
    """The S/T grid lo/hi are re-derived from the cell file; a saved value from a different cell would
    be a stale, wrong bound. Only the free knobs (points, f0, freqs_per_batch, preset, cell) persist."""
    from core.gui import settings as st
    from core.gui.panels.crossval_panel import CrossValPanel

    qt_app()
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

def test_panel_splitter_is_sized_and_not_collapsible():
    """Every panel opens with a usable controls column that cannot be dragged to nothing.

    BasePanel builds the app's only QSplitter and there are nine live instances. It used to be a
    LOCAL with no setSizes and no setChildrenCollapsible, so every launch started at the minimum and
    one slip past the left edge collapsed the controls to zero width, recoverable only by finding a
    5px handle at x=0.
    """
    app = qt_app()
    from core.gui.panels.base_panel import BasePanel

    class P(BasePanel):
        pass

    panel = P()
    panel.resize(1300, 820)
    pump(app, 0.15)
    assert not panel.splitter.childrenCollapsible(), \
        "the controls column can still be collapsed to zero width"
    assert all(s > 0 for s in panel.splitter.sizes()), \
        f"splitter opened with a zero-width pane: {panel.splitter.sizes()}"
    # The old hard 460px cap could not be escaped by widening the window, so any wider form got a
    # permanent horizontal scrollbar in the left column.
    assert panel.controls_scroll.maximumWidth() > 1000, \
        f"controls column is still hard-capped at {panel.controls_scroll.maximumWidth()}px"

def test_panel_layout_round_trips_through_settings():
    """The splitter position must survive a restart -- 'I have to re-drag it every launch' is the
    complaint. Uses saveState/restoreState, which work before the widget is shown or polished."""
    app = qt_app()
    from core.gui import settings as st
    from core.gui.panels.base_panel import BasePanel

    class P(BasePanel):
        pass

    a = P()
    a.resize(1300, 820)
    pump(app, 0.15)
    a.splitter.setSizes([500, 700])
    pump(app, 0.05)
    want = a.splitter.sizes()
    qs = st.settings()
    a.save_layout(qs)
    qs.sync()

    b = P()                                   # restore_layout runs in __init__
    b.resize(1300, 820)
    pump(app, 0.15)
    assert b.splitter.sizes() == want, \
        f"layout did not round-trip: saved {want}, restored {b.splitter.sizes()}"

def test_forms_grow_their_fields_and_numeric_boxes_have_a_floor():
    """Pins both halves of the 'input boxes are cut off' fix, across every form in every panel.

    Qt's Windows default is FieldsStayAtSizeHint: the field takes its size hint and stops, so numeric
    boxes rendered 3-6 characters wide and typing "0.033333" scrolled inside the box. Repo-wide there
    were ZERO setFieldGrowthPolicy calls over 18 QFormLayout sites. Asserting over discovered forms
    (not a fixed list) means a newly added form is covered too.
    """
    from PySide6.QtWidgets import QFormLayout, QLineEdit
    from core.gui.panels.crossval_panel import CrossValPanel
    from core.gui.panels.fdt_panel import FdtPanel
    from core.gui.panels.reduction_panel import ReductionPanel
    from core.gui.widgets.labeled_inputs import FloatField, IntField

    qt_app()
    panels = [CrossValPanel(), FdtPanel(), ReductionPanel()]
    forms = [f for p in panels for f in p.findChildren(QFormLayout)]
    assert forms, "no QFormLayouts discovered — the test is not exercising anything"
    bad = [f for f in forms
           if f.fieldGrowthPolicy() != QFormLayout.AllNonFixedFieldsGrow]
    assert not bad, f"{len(bad)}/{len(forms)} forms do not grow their fields"

    narrow = [w for p in panels for w in p.findChildren(QLineEdit)
              if isinstance(w, (FloatField, IntField)) and w.minimumWidth() < 80]
    assert not narrow, \
        f"{len(narrow)} numeric field(s) have no usable minimum width (e.g. " \
        f"{narrow[0].minimumWidth()}px)"

def test_long_diagnostics_are_readable_without_horizontal_scrolling():
    """Panels emit long single-line diagnostics; the log pane must wrap them, and the crossval cell
    label (which receives an unbounded str(e)) must not force the whole column wider."""
    from PySide6.QtWidgets import QPlainTextEdit
    from core.gui.panels.crossval_panel import CrossValPanel
    from core.gui.widgets.log_pane import LogPane

    qt_app()
    assert LogPane().lineWrapMode() == QPlainTextEdit.WidgetWidth, \
        "log pane still truncates long lines instead of wrapping them"
    assert CrossValPanel().cell_values.wordWrap(), \
        "the crossval cell-values label does not wrap, so a long error widens the whole column"

def test_the_vram_ceiling_is_on_config_live_and_NOT_persisted():
    """The VRAM ceiling is a HARDWARE knob, and it is the one field on the Config tab that is
    deliberately forgotten between sessions.

    NOT PERSISTED, because stale QSettings have already cost this project a ~5-day run: the
    2026-08-19 retrain trained on the RETIRED chi band because a saved value silently won over
    config.py. A ceiling fails the same way but far more quietly -- a forgotten 2 GiB does not
    error, it just makes every future run split from batch 0 and take several times longer, with
    nothing in the log to explain it. Starting each session at config.py's 0.0 keeps the throttle a
    decision somebody just made.

    A PLAIN ASSIGNMENT IS ENOUGH, unlike every other knob the GUI exposes. The sweep and flow fields
    had to become ARGUMENTS threaded into build_prior/build_posterior, because orchestrator does
    `from .config import ...` and binds them at import, so writing to the constant is a silent no-op
    (an imported name is a snapshot). pipeline.vram_ceiling_gib() does a getattr on the module every time the planner
    asks, so this one genuinely takes effect -- and this test would catch it if that ever changed.
    """
    from core.gui.panels import inference_tabs as it
    from core import config as _cfg
    from core.SBI import pipeline as _pipe

    saved = _cfg.SIM_VRAM_CEILING_GIB
    saved_env = os.environ.pop(_pipe.VRAM_CEILING_ENV, None)
    try:
        panel = it.ConfigPanel(None)
        assert hasattr(panel, "vram_ceiling"), "the Config tab must carry the VRAM ceiling field"
        assert panel.vram_ceiling.value() == 0.0, "it must default to OFF"

        # Live: typing must reach the planner with no plumbing in between.
        panel.vram_ceiling.setText("6.5")
        assert _cfg.SIM_VRAM_CEILING_GIB == 6.5, "the field must assign the config constant"
        assert _pipe.vram_ceiling_gib() == 6.5, "and the planner must read it live"
        # 2 GiB, not 6.5: the budget is a min() and the CPU branch of memory_budget_elements caps at
        # 4 GiB, so a ceiling above that would not bind and the assertion would prove nothing.
        panel.vram_ceiling.setText("2")
        dev, dt = torch.device("cpu"), torch.float32
        saved_cap = _pipe._BUDGET_CAP_ELEMENTS
        try:
            _pipe._BUDGET_CAP_ELEMENTS = None       # the other term in the same min()
            assert _pipe.sim_memory_budget_elements(dev, dt) == (2 * 2 ** 30) // 4, \
                "the ceiling must bind the budget the planner actually uses"
        finally:
            _pipe._BUDGET_CAP_ELEMENTS = saved_cap

        # Not persisted: neither direction may mention it.
        src_save = code_only(it.ConfigPanel.save_settings)
        src_restore = code_only(it.ConfigPanel.restore_settings)
        for what, src in (("save_settings", src_save), ("restore_settings", src_restore)):
            assert "vram" not in src.lower(), (
                f"{what} must NOT touch the VRAM ceiling -- a stale ceiling throttles every future "
                f"run silently, which is the failure mode the 2026-08-19 band trap already cost us")

        # And the env override must ANNOUNCE that it wins, rather than leaving a dead field.
        os.environ[_pipe.VRAM_CEILING_ENV] = "2.0"
        panel.vram_ceiling.setText("9")
        assert _pipe.vram_ceiling_gib() == 2.0, "the env override wins"
        assert _pipe.VRAM_CEILING_ENV in panel.vram_note.text(), (
            "a field that silently does nothing is worse than no field -- the note must say the "
            "environment is overriding it")
    finally:
        os.environ.pop(_pipe.VRAM_CEILING_ENV, None)
        if saved_env is not None:
            os.environ[_pipe.VRAM_CEILING_ENV] = saved_env
        _cfg.SIM_VRAM_CEILING_GIB = saved

def test_the_free_vram_readout_does_not_use_mem_get_info():
    """⚠ The readout beside the ceiling must come from nvidia-smi, never torch.cuda.mem_get_info.

    That reading overstates free VRAM on Windows by roughly the size of the desktop -- measured
    15037 MiB against nvidia-smi's 5814 at the same instant -- and it is the number that
    green-lit the batch which killed the first chi retrain. Printing it next to a field whose entire
    purpose is to bound VRAM would hand the user the exact lie the field defends against."""
    from core.gui.panels import inference_tabs as it
    src = code_only(it._nvidia_smi_free_gib)
    assert "mem_get_info" not in src, "the readout must not use the optimistic driver reading"
    assert "nvidia-smi" in src, "the readout must come from nvidia-smi"
    got = it._nvidia_smi_free_gib()
    assert got is None or got >= 0.0, f"unexpected reading {got!r}"

def test_a_one_field_near_miss_blocks_a_fresh_run_until_confirmed():
    """A passive status line was not enough, three times over.

    `_budget_checkpoint` already said "these settings match no checkpoint, so this starts a NEW run"
    and named the differing field -- but it is a label on a tab the user has scrolled past by the
    time they press Train. It failed to prevent 884 batches being lost outright on 2026-08-27, and a
    3989-batch checkpoint being abandoned twice on 2026-08-28. So a run that would start from zero
    while a committed checkpoint sits ONE identity field away now has to be confirmed.

    Narrow on purpose: exactly one differing field is the signature of an accident. Two or more is
    usually a genuinely different experiment, and warning there would make this noise."""
    from core.SBI import training_checkpoint as tc

    base = {"format": "training-rows", "model": "NADROWSKI", "n_runs": 10000,
            "run_size": 2048, "prior_fingerprint": "aaaa", "chi_mode": True}
    with tempfile.TemporaryDirectory() as root:
        root = Path(root)
        # A committed sibling that differs in exactly ONE field.
        sib = dict(base, prior_fingerprint="bbbb")
        d = tc.resolve_dir(sib, root); (d / "shards").mkdir(parents=True)
        torch.save({"identity": sib}, d / "header.pt")
        torch.save({"batches_done": 3989, "complete": False, "rng": None}, d / "state.pt")

        near = tc.near_miss_siblings(base, root)
        assert len(near) == 1, f"expected the one-field sibling, got {near}"
        assert near[0]["field"] == "prior_fingerprint" and near[0]["batches"] == 3989, near[0]

        # TWO differing fields is a different experiment -- it must NOT warn.
        far = dict(base, prior_fingerprint="cccc", n_runs=5000)
        d2 = tc.resolve_dir(far, root); (d2 / "shards").mkdir(parents=True)
        torch.save({"identity": far}, d2 / "header.pt")
        torch.save({"batches_done": 500, "complete": False, "rng": None}, d2 / "state.pt")
        names = {r["name"] for r in tc.near_miss_siblings(base, root)}
        assert d2.name not in names, "a two-field difference must not be reported as a near miss"

        # An identity that MATCHES a checkpoint is a resume, so there is nothing to confirm.
        assert tc.near_miss_siblings(sib, root) == [] or             all(r["name"] != d.name for r in tc.near_miss_siblings(sib, root)),             "a checkpoint must never be a near miss of itself"

def test_the_confirmation_is_reached_and_can_refuse():
    """The dialog must actually gate the dispatch, Cancel must mean cancel, and the answer must travel
    to the stage as `new_run=` -- a consent the panel kept to itself would be re-refused inside the
    worker. The panel also no longer computes the identity itself: ONE detector,
    orchestrator.fresh_run_near_misses, so the dialog and the stage cannot disagree about which
    directory the run will touch."""
    from core.gui.panels import inference_tabs as it

    src = code_only(it.PosteriorPanel._build_posterior)
    assert "_confirm_fresh_run" in src, "the Train handler must consult the confirmation"
    i_confirm = src.find("_confirm_fresh_run")
    i_dispatch = src.find("self.dispatch(")
    assert i_confirm < i_dispatch, "the confirmation must gate the dispatch, not follow it"
    assert "return" in src[i_confirm:i_dispatch], "a refusal must return instead of training"
    assert "new_run=" in src, "the dialog's answer must reach build_posterior as an argument"

    # Fails open rather than blocking a run it cannot assess.
    body = code_only(it.PosteriorPanel._confirm_fresh_run)
    assert body.count("True, False") >= 3, (
        "the check must fail OPEN -- no prior, an unreadable identity, and no near miss must all "
        "proceed; a warning that can block a run is worse than no warning")
    assert "fresh_run_near_misses" in body, "the panel must use the stage's own detector"
    for banned in ("near_miss_siblings", "resolve_dir", "peek"):
        assert banned not in body, (
            f"_confirm_fresh_run reimplements '{banned}' -- the GUI's own identity derivation is the "
            f"defect D7 removed (it resolved run_size as `cap or hw`, not `min(hw, cap)`)")


# ── piece 3: the shared test helpers ─────────────────────────────────────────────────────────────
def test_code_only_strips_docstrings_and_comments_at_every_depth_and_accepts_a_module():
    """The AST pins forbid words that the docstrings EXPLAINING the rule necessarily contain, so the
    helper must drop every docstring -- nested functions and classes too, not only the outermost one
    (test_nav_and_gating's local _stripped dropped only that) -- and comments, which ast.unparse never
    carries. One helper for the four copies the suites grew (two _strip_docstrings, _code_only,
    _unparsed); and it takes a module, so the cadence pin in test_user_sbi no longer re-reads
    pipeline.py by hand."""
    from tests._fixtures import code_only

    def outer():
        """DOCSTRING_OUTER says forbidden_word"""
        # COMMENT says forbidden_word
        def inner():
            """DOCSTRING_INNER says forbidden_word"""
            return "kept_string"

        class K:
            """DOCSTRING_CLASS says forbidden_word"""

        return inner, K

    src = code_only(outer)
    for gone in ("DOCSTRING_OUTER", "DOCSTRING_INNER", "DOCSTRING_CLASS", "COMMENT", "forbidden_word"):
        assert gone not in src, src
    assert "kept_string" in src and "def inner" in src and "class K" in src, src

    from core.gui import settings as st
    mod = code_only(st)
    assert "def use_ini_file(" in mod and "One QSettings store" not in mod, mod[:200]


@pytest.fixture(scope="module")
def _panel_built_at_module_scope(tmp_path_factory):
    """A panel built at MODULE scope, the way tiny_run builds its run and screen_run (spec §8.1) will
    build its screen. pytest sets module fixtures up before the function-scoped ones, so this runs
    with whatever settings path the SESSION left -- the case a function-only isolation misses. It
    returns the path it saw, the panel, and the session's temp root for the test to check against."""
    from core.gui import settings as st
    from core.gui.panels.reduction_panel import ReductionPanel
    qt_app()
    seen = st._override_path
    panel = ReductionPanel()                  # __init__ ends with restore_settings(st.settings())
    return panel, seen, tmp_path_factory.getbasetemp()


def test_a_module_scoped_fixture_reads_the_session_ini_and_never_the_real_one(_panel_built_at_module_scope):
    """Spec §5.4: the settings location is never the real PRISM.ini during a test process, AT ANY
    FIXTURE SCOPE. The old per-test redirect (_temp_settings) covered a test body only; a panel built
    by a module fixture was set up before it and read the developer's last session -- the same
    restore-wins-over-config mechanism that cost a ~5-day run on 2026-08-19, now with the answer
    depending on which test ran first. Two paths are asserted: the one the module fixture saw (the
    session file, tests/conftest.py::_settings_home) and the one this body sees (its own fresh file,
    _isolated_settings); both under the session temp, neither the user-scope file, and the fresh one
    not yet on disk -- QSettings creates it, and an empty file is not the same thing as no file."""
    from PySide6.QtCore import QSettings
    from core.gui import settings as st

    _panel, seen, basetemp = _panel_built_at_module_scope
    real = Path(QSettings(QSettings.IniFormat, QSettings.UserScope, st.ORG, st.APP).fileName()).resolve()
    assert seen is not None, "a module-scoped fixture built its panel against the real PRISM.ini"
    assert Path(seen).name == "prism.ini" and Path(seen).resolve() != real, seen
    assert basetemp in Path(seen).parents, f"{seen} is not under the session temp {basetemp}"
    now = st._override_path
    assert now is not None and now != seen and basetemp in Path(now).parents, now
    assert not Path(now).exists(), "the per-test file must not exist until Qt writes it"


def test_no_suite_points_the_settings_back_at_the_real_ini():
    """The teardown of every per-test redirect used to be a bare `use_ini_file` call with no path --
    and no path IS the real file. Under the session fixture that call would hand the next
    module-scoped fixture the developer's PRISM.ini, so the suites may not contain that call at all:
    tests/conftest.py alone resets the path, once, at session end, after asserting the real file's
    bytes are unchanged. The needle is assembled so this file does not match itself."""
    needle = "use_ini_file(" + "None)"
    here = Path(__file__).resolve().parent
    offenders = sorted(p.name for p in here.glob("test_*.py") if needle in p.read_text(encoding="utf-8"))
    assert offenders == [], f"these suites redirect the settings to the real PRISM.ini: {offenders}"


def test_int_field_value_or_none_tells_a_blank_from_a_zero():
    """``IntField.value()`` returns 0 for "" and for "-" mid-typing (labeled_inputs.py:45-49), and 0
    is a legal value for the two "0 = automatic" boxes (the rows-per-batch cap, the candidates per
    sweep round), so a tab that refuses a blank rather than reading it as zero (piece 3, V2) needs
    the FloatField twin: None for anything that does not parse, the int otherwise, whitespace
    tolerated. ``value()`` keeps its old contract for the callers that still use it."""
    from core.gui.widgets.labeled_inputs import IntField
    from tests._fixtures import qt_app

    qt_app()
    f = IntField(5)
    assert f.value_or_none() == 5
    for text, want in (("0", 0), (" 12 ", 12), ("-3", -3), ("", None), ("-", None), ("1.5", None)):
        f.setText(text)                      # setText bypasses the QIntValidator, as a restore does
        assert f.value_or_none() == want, (text, f.value_or_none())
    f.setText("")
    assert f.value() == 0, "value() still reads a blank as 0 for its remaining callers"


def test_the_config_tab_science_knobs_open_at_config_and_are_not_written():
    """V5 on the Config tab: the χ probe count, probe slots and lock-in ceiling open at config.py's
    values on EVERY launch, and the drive amplitude and band are never saved or read at all.

    The seed-then-restore pattern is what trained the 2026-08-19 retrain on the retired band: the tab
    seeded the boxes from config.CHI_* and then restored whatever the last session had saved, so a
    value written before config.py changed won silently on every launch afterwards. Since D11 a
    non-default band or amplitude is refused seconds into the prior build, so restoring one only
    manufactures that refusal; the slots and the ceiling are frozen into every posterior trained
    with them, so a stale one silently trains a different network. A stale key in an old PRISM.ini
    is IGNORED, never migrated.

    Two launches: from an empty file, and from a file holding every old key one step from the
    defaults. Both must show config.py, while the selections (model, χ-mode tick) are still
    remembered. Task 20 pins the Prior, Posterior, Validate and TSNPE tabs the same way.
    """
    from core import config
    from core.gui import settings as st
    from core.gui.panels import inference_tabs as it
    from tests._fixtures import qt_app

    qt_app()
    want = {"chi_k": str(config.CHI_N_FREQS), "chi_k_pad": str(config.CHI_K_PAD),
            "chi_f0": str(config.CHI_F0), "chi_lo": str(config.CHI_FREQ_BOUNDS[0]),
            "chi_hi": str(config.CHI_FREQ_BOUNDS[1]), "chi_max_cycles": str(config.CHI_MAX_CYCLES)}

    def boxes(panel):
        return {"chi_k": panel.chi_k.text(), "chi_k_pad": panel.chi_pad.text(),
                "chi_f0": panel.chi_f0.text(), "chi_lo": panel.chi_range.lo.text(),
                "chi_hi": panel.chi_range.hi.text(), "chi_max_cycles": panel.chi_cycles.text()}

    # (a) an empty file: the autouse settings fixture points at a fresh path for this test
    fresh = it.ConfigPanel(None)
    assert boxes(fresh) == want, boxes(fresh)

    # (b) a file holding every old key, each one step from the default, beside two selections
    stale = {"chi_k": str(config.CHI_N_FREQS + 1), "chi_k_pad": str(config.CHI_K_PAD - 1),
             "chi_f0": str(config.CHI_F0 / 2), "chi_lo": str(config.CHI_FREQ_BOUNDS[0] / 3),
             "chi_hi": str(config.CHI_FREQ_BOUNDS[1] * 3),
             "chi_max_cycles": str(config.CHI_MAX_CYCLES + 5)}
    qs = st.settings()
    qs.beginGroup("inference_config")
    for key, value in stale.items():
        qs.setValue(key, value)
    qs.setValue("chi_mode", "1")
    qs.setValue("model", "HOPF")
    qs.endGroup()
    qs.sync()
    relaunched = it.ConfigPanel(None)
    assert boxes(relaunched) == want, boxes(relaunched)
    assert relaunched.chi_check.isChecked() is True, "the χ-mode tick is a selection: remembered"
    assert relaunched.model_combo.currentText() == "HOPF", "the model is a selection: remembered"

    # (c) nothing writes them: edit every box, clear the stale keys, save -- a key present afterwards
    # can only be a fresh WRITE, and the only keys written are the five selections.
    relaunched.chi_k.setText("3")
    relaunched.chi_pad.setText("4")
    relaunched.chi_cycles.setText("7")
    relaunched.chi_f0.setText("0.05")                    # setText ignores read-only, as a restore would
    relaunched.chi_range.lo.setText("0.1")
    relaunched.chi_range.hi.setText("0.9")
    out = st.settings()
    out.beginGroup("inference_config")
    for key in stale:
        out.remove(key)
    out.endGroup()
    relaunched.save_settings(out)
    out.sync()
    out.beginGroup("inference_config")
    written = set(out.childKeys())
    out.endGroup()
    assert not (written & set(stale)), f"science keys written: {sorted(written & set(stale))}"
    assert {"model", "units_mode", "units_text", "chi_mode", "reparam_rotate"} <= written, written


def test_science_knobs_open_at_config_and_are_not_written(tmp_path):
    """V5 on the Prior, Posterior, Validate and TSNPE tabs (Task 11 pins the Config tab the same way):
    every SCIENCE KNOB opens at config.py's value -- the truncate module's, for the HPD level and the
    direction count -- on EVERY launch, and its key is neither written nor read. A stale key an older
    build left in PRISM.ini is IGNORED, never migrated.

    Why: each of these boxes was seeded from config.py and then overwritten from QSettings, so a value
    typed once won silently over config.py on every later launch. The sweep and clustering boxes
    decide which prior gets built, the network and Fisher boxes which posterior gets trained, the
    calibration boxes what "calibrated" was measured with, and nothing on screen compared any of them
    with config.py. The same seed-then-restore pattern trained the 2026-08-19 retrain on the retired
    χ band. The selections and the budget stay remembered, and the old file below holds one of those
    per tab that has one, so "opens at config.py" cannot pass merely because restore_settings never
    ran.

    Three legs: an empty file; a file holding every old key one step from config.py; a save after
    every knob box was edited, which must write the selections and the budget and nothing else.
    """
    from PySide6.QtCore import QSettings
    from core import config
    from core.gui import settings as st
    from core.gui.screens.inference_screen import InferenceScreen
    from core.SBI import truncate as _tr
    from tests._fixtures import qt_app

    qt_app()
    # (group, the screen's panel attribute, {key, which is also the box attribute: config.py's value})
    knobs = (
        ("inference_prior", "prior_panel", {
            "sweep_iters": config.PRIOR_SWEEP_ITERATIONS, "sweep_batch": config.PRIOR_SWEEP_BATCH,
            "sweep_max_sets": config.PRIOR_SWEEP_MAX_SETS, "sweep_step": config.PRIOR_SWEEP_STEP,
            "sweep_units": config.STABILITY_SWEEP_ND_UNITS,
            "cluster_size": config.PRIOR_CLUSTER_MIN_SIZE,
            "cluster_samples": config.PRIOR_CLUSTER_MIN_SAMPLES}),
        ("inference_posterior", "posterior_panel", {
            "flow_hidden": config.NSF_HIDDEN_FEATURES, "flow_transforms": config.NSF_NUM_TRANSFORMS,
            "flow_lr": config.TRAINING_LEARNING_RATE, "flow_patience": config.TRAINING_STOP_AFTER_EPOCHS,
            "fisher_m": config.REPARAM_FISHER_M, "fisher_dz": config.REPARAM_FISHER_DZ,
            "fisher_points": config.REPARAM_FISHER_POINTS}),
        ("inference_validate", "validate_panel", {
            "cal_n": config.SBC_N_CAL, "cal_scales": config.CAL_N_SCALES}),
        ("inference_tsnpe", "tsnpe_panel", {
            "hpd": _tr.DEFAULT_HPD, "n_dirs": _tr.DEFAULT_N_DIRECTIONS}),
    )

    def stale(default):
        """One step from config.py: an int box plus one, a float box halved."""
        return str(default + 1) if isinstance(default, int) else str(default / 2)

    def off_config(inf):
        """Every knob box whose number is not config.py's, as {(group, key): text}. Compared as
        numbers: the old restore wrote str(float(...)), so config.py's 1000 came back as "1000.0"."""
        return {(group, key): getattr(getattr(inf, attr), key).text()
                for group, attr, keys in knobs for key, default in keys.items()
                if float(getattr(getattr(inf, attr), key).text()) != float(default)}

    # (a) an empty file (the autouse settings fixture points at a fresh path for this test)
    first = InferenceScreen()
    assert off_config(first) == {}, f"an empty file opened off config.py: {off_config(first)}"

    # (b) a file holding every old knob key one step from config.py, the dead bounds_source, and one
    # remembered selection or budget per tab that has one -- the proof that restore_settings RAN
    qs = st.settings()
    for group, _attr, keys in knobs:
        for key, default in keys.items():
            qs.setValue(f"{group}/{key}", stale(default))
    qs.setValue("inference_prior/bounds_source", "direct")
    qs.setValue("inference_prior/bounds", "stale_bounds.txt")
    qs.setValue("inference_posterior/num_runs", "1234")
    qs.setValue("inference_tsnpe/num_runs", "4321")
    qs.sync()
    relaunched = InferenceScreen()
    assert off_config(relaunched) == {}, f"science knobs restored from an old file: {off_config(relaunched)}"
    assert relaunched.prior_panel._saved_bounds_key == "stale_bounds.txt", "the bounds file is a selection"
    assert relaunched.prior_panel.bounds_source.is_direct() is False
    assert relaunched.posterior_panel.num_runs.text() == "1234", "the Posterior budget is remembered"
    assert relaunched.tsnpe_panel.num_runs.text() == "4321", "the TSNPE budget is remembered (spec §5.3)"

    # (c) nothing writes them: every knob box edited, the four panels saved into a file of their own
    for _group, attr, keys in knobs:
        for key, default in keys.items():
            getattr(getattr(relaunched, attr), key).setText(stale(default))
    out = QSettings(str(tmp_path / "saved.ini"), QSettings.IniFormat)
    for panel in (relaunched.prior_panel, relaunched.posterior_panel, relaunched.validate_panel,
                  relaunched.tsnpe_panel):
        panel.save_settings(out)
    out.sync()
    expected = {"inference_prior": {"prior", "bounds"},
                "inference_posterior": {"posterior", "num_runs", "run_size_cap"},
                "inference_validate": set(),
                "inference_tsnpe": {"observation", "num_runs", "run_size_cap"}}
    for group, keys in expected.items():
        out.beginGroup(group)
        written = set(out.childKeys())
        out.endGroup()
        assert written == keys, f"[{group}] writes {sorted(written)}; the selections and budget are {sorted(keys)}"


def test_the_tsnpe_tab_restores_its_observation_and_budget_only(monkeypatch):
    """Spec §5.3. The TSNPE tab's restore_settings was complete and NEVER CALLED: its __init__ was born
    without the `self.restore_settings(settings.settings())` line every other panel ends with, and
    nothing else calls it (MainWindow._save_state calls only save_settings). Every launch showed
    config.py's budget and the store's first observation while PRISM.ini held the last session's,
    and the tab's keys were dead writes.

    Now the tab restores its SELECTION (the observation) and its BUDGET (batches, rows-per-batch) and
    nothing else: the HPD level and the direction count are science knobs and open at the truncate
    module's defaults (V5). The budget is saved as the boxes' TEXT, so a box left blank at close opens
    at config.py's default; the old save wrote value(), which turns "" into a 0 nobody typed.

    The last leg pins the defect class, not the instance: every panel with a restore_settings of its
    own calls it from its own __init__.
    """
    import types
    from core import config
    from core.gui import settings as st
    from core.gui.panels.crossval_panel import CrossValPanel
    from core.gui.panels.fdt_panel import FdtPanel
    from core.gui.panels.inference.config_tab import ConfigPanel
    from core.gui.panels.inference.infer_tab import InferPanel
    from core.gui.panels.inference.posterior_tab import PosteriorPanel
    from core.gui.panels.inference.prior_tab import PriorPanel
    from core.gui.panels.inference.tsnpe_tab import TSNPEPanel
    from core.gui.panels.inference.validate_tab import ValidatePanel
    from core.gui.panels.reduction_panel import ReductionPanel
    from core.gui.panels.simulate_panel import SimulatePanel
    from core.gui.screens.inference_screen import InferenceScreen
    from core.gui.widgets.artifact_picker import StorePicker
    from core.SBI import truncate as _tr
    from tests._fixtures import code_only, qt_app

    qt_app()
    # Two complete observations, so a restored pick is distinguishable from the default first entry.
    # The store is stubbed at the picker's one seam; each row carries exactly what refresh() reads.
    rows = [types.SimpleNamespace(complete=True, label=label, id=id_, created="2026-09-16T12:00:00",
                                  mode="spontaneous", width=50, amortized=None)
            for label, id_ in (("first", "20260916T120000"), ("second", "20260916T130000"))]
    store = types.SimpleNamespace(list=lambda kind: list(rows) if kind == "observation" else [])
    monkeypatch.setattr(StorePicker, "_resolved_store", lambda self: store)

    defaults = {"hpd": str(_tr.DEFAULT_HPD), "n_dirs": str(_tr.DEFAULT_N_DIRECTIONS),
                "num_runs": str(config.TRAINING_NUM_RUNS), "run_size_cap": str(config.TRAINING_RUN_SIZE)}

    def boxes(panel):
        return {"hpd": panel.hpd.text(), "n_dirs": panel.n_dirs.text(),
                "num_runs": panel.num_runs.text(), "run_size_cap": panel.run_size_cap.text()}

    # (a) an empty file: the first observation and every default. Each screen is HELD in a local:
    # it owns its panels' C++ objects, and a dropped screen takes its panels with it.
    inf1 = InferenceScreen()
    tp = inf1.tsnpe_panel
    assert tp.obs_picker.key() == "20260916T120000", tp.obs_picker.key()
    assert boxes(tp) == defaults, boxes(tp)

    # (b) one session picks the second observation and edits all four boxes; the next launch shows
    # that observation and that budget, and the truncate module's defaults for the other two
    tp.obs_picker.combo.setCurrentIndex(1)
    tp.num_runs.setText("1234")
    tp.run_size_cap.setText("512")
    tp.hpd.setText("0.99")
    tp.n_dirs.setText("3")
    qs = st.settings()
    tp.save_settings(qs)
    qs.sync()
    inf2 = InferenceScreen()
    again = inf2.tsnpe_panel
    assert again.obs_picker.key() == "20260916T130000", "the observation is a selection: restored at construction"
    assert (again.num_runs.text(), again.run_size_cap.text()) == ("1234", "512"), boxes(again)
    assert "1,234 batches" in again.budget_total.text(), "the budget lines must follow the restored boxes"
    assert (again.hpd.text(), again.n_dirs.text()) == (defaults["hpd"], defaults["n_dirs"]), \
        f"the HPD level and the direction count are science knobs: {boxes(again)}"

    # (c) both budget boxes BLANK at close: saved as "", and the next launch opens at config.py
    again.num_runs.setText("")
    again.run_size_cap.setText("")
    qs = st.settings()
    again.save_settings(qs)
    qs.sync()
    qs.beginGroup("inference_tsnpe")
    raw = (str(qs.value("num_runs")), str(qs.value("run_size_cap")))
    qs.endGroup()
    assert raw == ("", ""), f"the budget must be saved as the boxes' text, not value()'s 0: {raw}"
    inf3 = InferenceScreen()
    third = inf3.tsnpe_panel
    assert (third.num_runs.text(), third.run_size_cap.text()) == \
        (defaults["num_runs"], defaults["run_size_cap"]), boxes(third)
    assert third.obs_picker.key() == "20260916T130000", "a blank budget must not cost the observation"

    # (d) the defect class: a restore_settings its own __init__ never calls is dead code
    call = "self.restore_settings(settings.settings())"
    for cls in (ConfigPanel, PriorPanel, PosteriorPanel, ValidatePanel, InferPanel, TSNPEPanel,
                SimulatePanel, ReductionPanel, FdtPanel, CrossValPanel):
        if "restore_settings" in vars(cls):
            assert call in code_only(cls.__init__), \
                f"{cls.__name__} defines restore_settings and its __init__ never calls it"
    assert "restore_settings" in vars(TSNPEPanel)
    assert "restore_settings" not in vars(ValidatePanel), \
        "the Validate tab restores nothing (V5), so it carries no restore_settings to forget to call"


def test_consents_are_never_persisted(tmp_path):
    """V5: a CONSENT is answered per run and never remembered -- all four of them. The TSNPE tab's
    "Start a new simulation even if a cache one setting away exists" (D7) and the Infer tab's "Run on
    a different observation" (D8) were already unpersisted; the FDT panel's "Skip sanity checks" and
    "Proceed to the production sweep after sanity" were saved and restored like campaign knobs, so one
    session's "skip the checks" silently dropped them from every later session. Every consent now
    opens at its construction default. For the FDT pair that is skip UNTICKED and proceed TICKED, not
    both unticked (spec §1.2): unticking proceed would make a default click stop after the sanity
    checks, where today and on the command line (--no-production is opt-in) it runs the sweep.

    Three legs: no save_settings or restore_settings names a consent; a save after all four were
    flipped writes none of them; an old PRISM.ini that holds all four answers opens at the defaults.
    """
    from PySide6.QtCore import QSettings
    from core.gui import settings as st
    from core.gui.panels.fdt_panel import FdtPanel
    from core.gui.panels.inference.infer_tab import InferPanel
    from core.gui.panels.inference.tsnpe_tab import TSNPEPanel
    from core.gui.screens.inference_screen import InferenceScreen
    from tests._fixtures import code_only, qt_app

    qt_app()
    # (a) the source: executable code only, so a comment explaining the rule cannot trip it
    for cls, names in ((TSNPEPanel, ("new_run",)), (InferPanel, ("other_obs",)),
                       (FdtPanel, ("skip_sanity", "confirm_production"))):
        for method in (cls.save_settings, cls.restore_settings):
            src = code_only(method)
            for name in names:
                assert name not in src, f"{cls.__name__}.{method.__name__} names the consent {name}"

    # (b) all four flipped, then saved: the groups are written, and no consent is in them
    inf = InferenceScreen()
    inf.tsnpe_panel.new_run.setChecked(True)
    inf.infer_panel.other_obs.setChecked(True)            # setChecked ignores the disabled state
    fdt = FdtPanel()
    fdt.skip_sanity.setChecked(True)
    fdt.confirm_production.setChecked(False)
    out = QSettings(str(tmp_path / "saved.ini"), QSettings.IniFormat)
    for panel in (inf.tsnpe_panel, inf.infer_panel, fdt):
        panel.save_settings(out)
    out.sync()
    for group, names in (("inference_tsnpe", {"new_run"}), ("inference_infer", {"other_obs"}),
                         ("fdt", {"skip_sanity", "confirm_production"})):
        out.beginGroup(group)
        written = set(out.childKeys())
        out.endGroup()
        assert written and not (written & names), f"[{group}] writes a consent: {sorted(written)}"

    # (c) an old PRISM.ini holding all four answers: a relaunch opens every box at its default
    qs = st.settings()
    for key, value in (("inference_tsnpe/new_run", "1"), ("inference_infer/other_obs", "1"),
                       ("fdt/skip_sanity", "1"), ("fdt/confirm_production", "0")):
        qs.setValue(key, value)
    qs.sync()
    inf2 = InferenceScreen()
    fdt2 = FdtPanel()
    assert inf2.tsnpe_panel.new_run.isChecked() is False
    assert inf2.infer_panel.other_obs.isChecked() is False
    assert fdt2.skip_sanity.isChecked() is False, "the sanity checks run unless THIS session says skip"
    assert fdt2.confirm_production.isChecked() is True, "the production sweep follows the checks by default"
    assert fdt2.confirm_production.isEnabled() is True, "proceed is greyed out only while skip is ticked"


def test_get_bool_falls_back_on_an_unparseable_value():
    """settings.get_bool read "1"/"true"/"True" as True and EVERYTHING else as False, so a blank or
    hand-edited value ("maybe", "2") restored as False whatever the caller's default said. For the
    remembered boolean with a True default that is the Fisher rotation silently OFF (the Config tab's
    reparam_rotate, config.REPARAM_ROTATE). get_int already fell back to its default on text it
    cannot parse; get_bool now does the same (spec §5.1)."""
    from core import config
    from core.gui import settings as st
    from core.gui.panels import inference_tabs as it
    from tests._fixtures import qt_app

    qt_app()
    qs = st.settings()
    for key, value in (("one", "1"), ("true", "true"), ("True", "True"), ("zero", "0"),
                       ("false", "false"), ("False", "False"), ("blank", ""), ("maybe", "maybe"),
                       ("two", "2")):
        qs.setValue(f"flags/{key}", value)
    qs.sync()
    fresh = st.settings()
    for default in (True, False):
        for key in ("one", "true", "True"):
            assert st.get_bool(fresh, f"flags/{key}", default) is True, (key, default)
        for key in ("zero", "false", "False"):
            assert st.get_bool(fresh, f"flags/{key}", default) is False, (key, default)
        for key in ("blank", "maybe", "two", "missing"):
            assert st.get_bool(fresh, f"flags/{key}", default) is default, (key, default)

    # On screen: a corrupt reparam_rotate opens at config.py's value. (config.REPARAM_ROTATE is True
    # today, which is what makes this leg tell the fallback from the old unconditional False.)
    qs.setValue("inference_config/reparam_rotate", "maybe")
    qs.sync()
    panel = it.ConfigPanel(None)
    assert panel.rot_check.isChecked() == config.REPARAM_ROTATE, "a corrupt reparam_rotate did not open at config.py"
