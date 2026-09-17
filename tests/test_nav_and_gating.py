"""Navigation, the inference tabs' gating/session flow, and the chi probe table. Split from test_gui_progress.py; run directly: pytest tests/test_nav_and_gating.py"""
"""Progress-rendering regression tests for the GUI.

THE BUG THESE LOCK DOWN
    tqdm redraws a bar at pos>0 as three writes -- '\\n'*pos, then '\\r'+frame, then '\\x1b[A'*pos
    (tqdm/std.py:1493-1497). The old stream reader split on terminators, so the frame (which is never
    terminated) stranded in its buffer and was flushed by the NEXT redraw's leading '\\n' -- i.e. as a
    LOG LINE. Every nested-bar redraw appended one row, so a training run buried the log pane under
    hundreds of bar snapshots.

    The pipeline nests bars four deep (core/SBI/pipeline.py:517 -> :371 ->
    core/Simulator/simulator.py:50 -> core/Solvers/sdeint.py:15), so this fired constantly.

Run:  pytest tests/test_nav_and_gating.py
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
from tests._fixtures import qt_app                                # noqa: E402
import contextlib                                                  # noqa: E402
import pytest                                                      # noqa: E402

def _prior_stub(id_="p1", name=""):
    """A LoadedPrior-shaped stub: the tabs now read ``.prior``/``.force_prior`` off session.inf_prior
    (piece 1, Task 4) rather than carrying the physical prior and forcing prior as separate session
    fields, so a bare ``object()`` no longer stands in wherever a dispatched call is actually reached."""
    return type("LoadedPriorStub", (), {"prior": object(), "force_prior": object(),
                                         "id": id_, "name": name})()
def _posterior_stub(id_="post1", name="", truncation=None, x_obs_digest=None):
    """A LoadedPosterior-shaped stub: the tabs now read ``.posterior`` (the TransformedPosterior) off
    session.posterior (piece 1, Task 7) rather than carrying it directly, so a bare ``object()`` no
    longer stands in wherever a dispatched call actually reaches ``session.posterior.posterior``."""
    post = type("Post", (), {"truncation": truncation, "x_obs_digest": x_obs_digest, "latent": object()})()
    return type("LoadedPosteriorStub", (), {"posterior": post, "id": id_, "name": name})()
def _chi_cfg(k=3, pad=12):
    """Stub config that puts the Infer tab on its chi page (mirrors the SimConfig fields it reads)."""
    from core import config

    class Cfg:
        model = "NADROWSKI"
        params_dict = {}
        rescale_params = {}
        force_params_dict = {}
        has_forcing = False
        chi_mode = True
        chi_n_freqs = k
        chi_k_pad = pad
        chi_freq_bounds = config.CHI_FREQ_BOUNDS
        chi_max_cycles = config.CHI_MAX_CYCLES
        observation_mode = "chi"
    return Cfg()


def _forced_cfg(names=("amp", "freq", "phase", "offset")):
    """Stub config that puts the Infer tab on its DRIVEN page, with one drive row per forcing name --
    the four the nadrowski master bounds declare, by default (mirrors the SimConfig fields the tab
    reads; ``force_params_dict`` values are ``(value, (lo, hi))`` as on a SimConfig). ``names=()`` is
    the PASSIVE page: no drive rows and no forced recording."""
    from core import config

    class Cfg:
        model = "NADROWSKI"
        params_dict = {}
        rescale_params = {}
        force_params_dict = {n: (0.0, (0.0, 1.0)) for n in names}
        has_forcing = bool(names)
        chi_mode = False
        chi_n_freqs = config.CHI_N_FREQS
        chi_k_pad = config.CHI_K_PAD
        chi_freq_bounds = config.CHI_FREQ_BOUNDS
        chi_max_cycles = config.CHI_MAX_CYCLES
        observation_mode = "forced" if names else "spontaneous"
    return Cfg()


def _spont_cfg():
    """A stub config that survives the Prior tab's post-build path and install_config: the SimConfig
    fields the "Config built" line, check_unit_consistency and the Infer tab's on_config_built read,
    with chi OFF (its twin _chi_cfg puts the Infer tab on the chi page instead). A bare object()
    stops at cfg.check_unit_consistency(), so a test that wants the Prior click to REACH its
    dispatch needs this much."""
    from core import config

    class Cfg:
        model = "NADROWSKI"
        params_dict = {}
        rescale_params = {}
        force_params_dict = {}
        has_forcing = False
        chi_mode = False
        chi_k_pad = config.CHI_K_PAD
        observation_mode = "spontaneous"

        def check_unit_consistency(self):
            return []
    return Cfg()


# ── Phase-2 panels ───────────────────────────────────────────────────────────────────────────────
def test_fdt_panel_guard_translates_model_error_and_gate_admits_builtins(monkeypatch):
    """FDT supports HOPF/BP + additive-noise user models. An FDTModelError (a missing FDT parameter,
    or a user model with multiplicative/zero observable noise) is a Refusal now, so the guard lets it
    through UNWRAPPED: the worker hands it to BasePanel._on_error, which opens the yellow "Check your
    inputs" box for it. Wrapping it in a RuntimeError, as the guard used to, re-typed a refusal into a
    bug and bought it a traceback. The KeyError net for a malformed cell stays -- that one is a bare
    KeyError nobody raised as a refusal -- so the guard still translates it into a sentence naming the
    parameter and the model. The registry gate admits every built-in."""
    import core.gui.panels.fdt_panel as fdt_panel
    from core.FDT.campaigns import FDTModelError
    from core.refusals import Refusal
    from core import registry

    class Cfg:
        model = "HOPF"

    def boom(cfg, *, skip_sanity, confirm_production):
        raise FDTModelError("Observable 'x' has state-dependent (multiplicative) noise; FDT supports "
                            "additive-noise observables only.")

    def missing(cfg, *, skip_sanity, confirm_production):
        raise KeyError("k_gs")

    monkeypatch.setattr(fdt_panel, "run_fdt", boom)
    with pytest.raises(FDTModelError) as e:
        fdt_panel._run_fdt_guarded(Cfg(), skip_sanity=True, confirm_production=False)
    assert isinstance(e.value, Refusal), "an FDTModelError must reach the worker as the Refusal it is"
    assert "multiplicative" in str(e.value), str(e.value)

    monkeypatch.setattr(fdt_panel, "run_fdt", missing)
    with pytest.raises(RuntimeError) as e:
        fdt_panel._run_fdt_guarded(Cfg(), skip_sanity=True, confirm_production=False)
    assert "k_gs" in str(e.value) and "HOPF" in str(e.value), str(e.value)
    assert not isinstance(e.value, Refusal), "a malformed cell is translated, not promoted to a refusal"

    # HOPF / BP / NADROWSKI are no longer rejected by the FDT gate.
    for m in ("NADROWSKI", "HOPF", "BP"):
        assert registry.fdt_support(m) == (True, ""), m

def test_an_unparseable_cell_does_not_brick_the_gui():
    """Dropping a cell into Resources/Cells/<model>/ that cli.parse_cell cannot read makes it raise a
    bare ValueError (NOT a UnitParseError). CrossValPanel prefills from parse_cell in __init__, so
    that exception used to escape CrossValPanel() -> MainWindow() -> build_app(), and
    `python -m core.gui` died before the window ever appeared -- before app.py's excepthook was even
    installed.

    The probe used to be a cell with no sibling bounds file, which is no longer unparseable: a cell
    without a sibling now resolves to the model's shared master.txt (cli.resolve_bounds_for_cell), so
    the natural 'add my cell' action WORKS rather than merely degrading. The failure mode this test
    guards still exists though -- a cell that omits a parameter its bounds file declares -- so probe
    with that instead."""
    import shutil
    from pathlib import Path

    from core.config import CELL_PATH

    qt_app()
    src = Path(CELL_PATH) / "nadrowski" / "master_weak.txt"
    if not src.exists():
        return                                   # nothing to probe with
    probe = src.with_name("aaa_probe_unparseable.txt")   # sorts first => the picker selects it
    # Drop a declared ND parameter: bounds resolve fine, but the merge cannot fill the value.
    kept = [ln for ln in src.read_text(encoding="utf-8").splitlines()
            if not ln.strip().startswith("beta ")]
    probe.write_text("\n".join(kept) + "\n", encoding="utf-8")
    try:
        # MainWindow() -> CrossValPanel.__init__ restores its saved cell selection; a cell saved from a
        # previous GUI session would be reloaded OVER our probe, so the picker would land on a valid
        # cell and the prefill would parse fine -- masking the degrade path this test exists to check.
        # The per-test .ini (tests/conftest.py::_isolated_settings) is empty, so the picker defaults
        # to the alphabetically-first entry (the probe), regardless of the machine's state.
        from core.gui.main_window import MainWindow
        from core.gui.panels.crossval_panel import CrossValPanel
        window = MainWindow()                    # must not raise
        xval = window.panel(CrossValPanel)       # CrossVal now lives inside the FDT Analysis section
        assert "could not read cell" in xval.cell_values.text().lower(), \
            f"the bad cell should degrade the prefill label, got: {xval.cell_values.text()!r}"
    finally:
        probe.unlink(missing_ok=True)

def test_plot_watcher_only_reports_pngs_written_after_start():
    """The FDT/Reduction/CrossVal runners never return their figure paths, so the panels pick them up
    off disk. Pre-existing figures must not be re-shown, and each new one must be emitted once."""
    import tempfile
    from pathlib import Path

    from core.gui.plot_watcher import NewPngWatcher

    app = qt_app()
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

# ── PRISM navigation redesign ────────────────────────────────────────────────────────────────────
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

    qt_app()
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
    qt_app()
    qs = st.settings()
    qs.setValue("window/tab", 2)          # a stale key from the old flat-tab layout
    qs.sync()
    from core.gui.main_window import MainWindow
    w = MainWindow()
    assert w.nav.stack.currentIndex() == 0, "the app must always open on the home screen"

def test_inference_tab_gates_follow_the_session():
    from core.gui.screens.inference_screen import InferenceScreen
    from core.gui.session import SbiSession

    qt_app()
    inf = InferenceScreen()

    def enabled():
        return [inf.tabs.isTabEnabled(i) for i in range(5)]   # Config Prior Posterior Validate Infer

    inf.session = SbiSession(); inf.refresh_gates()
    assert enabled() == [True, False, False, False, False]
    # A DRAFT (model applied) unlocks Prior -- which is where the bounds file builds the config.
    inf.session.draft = object(); inf.refresh_gates()
    assert enabled() == [True, True, False, False, False]
    inf.session.cfg = object(); inf.refresh_gates()
    assert enabled() == [True, True, True, False, False]
    inf.session.posterior = object(); inf.refresh_gates()
    assert enabled() == [True, True, True, False, True], "Infer needs only a posterior; Validate needs a prior"
    # force_prior stays None on purpose: it is None for every no-forcing model, and requiring it used to
    # make Validate permanently unreachable for them.
    inf.session.inf_prior = object(); inf.refresh_gates()
    assert enabled() == [True, True, True, True, True]

def test_tsnpe_tab_is_gated_and_never_proposes_from_the_posterior():
    """The TSNPE tab needs a posterior, its prior AND an observation on disk -- and its round must go
    through orchestrator.build_posterior with a TRUNCATION, never by fitting the posterior.

    The second half is the one worth a test: proposing from the posterior instead of the truncated
    prior is TEMPERING, it contracts credible intervals with no new information, and SBC comes out
    flat anyway because it validates the flow against the proposal it was trained on. Nothing on the
    Validate tab would catch it, so the wiring is pinned here and the maths in
    tests/test_conditioning_repair.py.
    """
    from core.gui.screens.inference_screen import InferenceScreen
    from core.gui.session import SbiSession

    qt_app()
    inf = InferenceScreen()
    assert inf.tabs.count() == 6 and inf.tabs.tabText(5) == "TSNPE"

    inf.session = SbiSession(cfg=object(), posterior=object()); inf.refresh_gates()
    assert not inf.tabs.isTabEnabled(5), "TSNPE must not open on a posterior alone"
    inf.session.inf_prior = object(); inf.refresh_gates()
    assert inf.tabs.isTabEnabled(5), "TSNPE opens once a posterior and its prior exist"

    # The observation gate, driven through the picker's own accessor rather than through the store's
    # `observations/` listing. Clearing the combo does NOT work: refresh_local_gates calls
    # obs_picker.refresh(), which repopulates it from the store -- so the assertion passed only while
    # no observation had been recorded, and any suite run that had exercised infer_and_visualize left
    # one behind and flipped it. A gate test must not depend on what an earlier test wrote.
    panel = inf.tsnpe_panel
    panel.obs_picker.key = lambda: ""                    # nothing recorded yet
    panel.refresh_local_gates()
    assert not panel.btn_round.isEnabled(), "a round must be impossible without an observation"
    panel.obs_picker.key = lambda: "20260910T120000"
    panel.refresh_local_gates()
    assert panel.btn_round.isEnabled(), "with a posterior, its prior and an observation, a round is allowed"

    # The contract, checked against the CODE with docstrings stripped -- those docstrings necessarily
    # contain the word "proposal" while explaining what must not happen, and a naive text search on the
    # whole source flags the very comment that documents the rule. Two halves now: the STAGE builds the
    # region and hands it to build_posterior as a truncation, and the TAB does nothing but dispatch it.
    from core import orchestrator
    from core.gui.panels.inference.tsnpe_tab import TSNPEPanel
    from tests._fixtures import code_only

    code = code_only(orchestrator.tsnpe_round)
    assert "build_truncation_region" in code and "truncation=region" in code, \
        "tsnpe_round does not build a truncation region and pass it to build_posterior"
    for banned in ("set_default_x", "proposal"):
        assert banned not in code, (
            f"tsnpe_round's CODE references '{banned}' -- it must sample the truncated PRIOR, never the "
            f"posterior; that is tempering, and SBC cannot detect it")

    tab = code_only(TSNPEPanel._round)
    assert "orchestrator.tsnpe_round" in tab, "the tab must dispatch the stage, not reimplement a round"
    for banned in ("build_posterior", "build_truncation_region", "set_default_x", "proposal"):
        assert banned not in tab, (
            f"TSNPEPanel._round's CODE references '{banned}' -- the round's science lives in "
            f"orchestrator.tsnpe_round, and a second copy in the GUI is how the two drift apart")

def test_a_tsnpe_posterior_cannot_be_saved_as_amortized(store):
    """⚠ SECTION 11.6 GUARDRAIL 2, at the seam where it is easiest to lose.

    Every posterior is now WRITTEN at completion (piece 1, Task 7), so there is no deferred-save
    window for a region to fall out of on the way to disk -- but the SESSION must still carry the
    right LoadedPosterior after a TSNPE round, after its Save (a rename, which must not touch the
    body), and after an ordinary train replaces it. Three things, and the third is the one that is
    easy to miss: the round installs a NON-AMORTIZED posterior, the Save renames it in place, and
    training an ordinary posterior afterwards installs an AMORTIZED one.
    """
    from core.artifacts import Accept
    from core.SBI import reparam, truncate
    from core.SBI.training_checkpoint import bijection_probe
    from core.gui.screens.inference_screen import InferenceScreen
    from core.gui.session import SbiSession
    from tests._fixtures import _nad_cfg, _posterior_artifact

    cfg = _nad_cfg(chi_mode=True)
    P = len(cfg.params_dict) + len(cfg.rescale_params)
    T = reparam.build_inferred_bijection(cfg, log_params=[])
    region = truncate.TruncationRegion([0, 1], [-1.0, -1.0], [1.0, 1.0], n_latent=P, V=None,
                                       probe=bijection_probe(T, P), x_obs_digest="d" * 16)
    trunc_artifact = _posterior_artifact(store, cfg, name="trunc", amortized=False, region=region)
    loaded = store.load_posterior(cfg, trunc_artifact.id, accept=Accept(truncated=True))

    qt_app()
    inf = InferenceScreen()
    inf.session = SbiSession(cfg=cfg, inf_prior=object())
    panel = inf.tsnpe_panel

    inf.posterior_panel._ask_load_non_amortized = lambda m: pytest.fail(
        "a round's OWN result must install with no dialog -- it is the posterior the user just made")
    panel._on_round(loaded)
    assert inf.session.posterior is loaded, "the round's LoadedPosterior never reached the session"
    assert inf.session.posterior.manifest.body["amortized"] is False,         "the round's posterior was not installed as NON-AMORTIZED"

    # the Save button is a RENAME now -- it must not touch the body, so it stays non-amortized on disk
    pp = inf.posterior_panel
    pp.post_name.setText("some_name")
    pp._save_posterior()
    assert store.get("posterior", loaded.id).name == "some_name"
    assert store.get("posterior", loaded.id).body["amortized"] is False

    # and an ordinary train afterwards must install an AMORTIZED posterior, or the mislabelling runs
    # the other way
    amortized_artifact = _posterior_artifact(store, cfg, name="amortized")
    amortized_loaded = store.load_posterior(cfg, amortized_artifact.id)
    pp._on_posterior(amortized_loaded)
    assert inf.session.posterior.manifest.body["amortized"] is True


def test_a_loaded_non_amortized_posterior_carries_its_region_into_the_session(store):
    """⚠ GUARDRAIL 8's GUI half, plus D8's question. A non-amortized artifact is loaded through the
    Posterior tab, which now ASKS first and opts in with ``Accept(truncated=True)`` only on a yes; the
    LoadedPosterior it installs carries the region, Validate passes that wrapper straight through to
    validate_calibration (which reads the region off ``posterior.posterior.truncation`` so calibration
    draws from the truncated prior), and an amortized LoadedPosterior clears it -- and is loaded with no
    question at all."""
    from core.artifacts import Accept
    from core.SBI import reparam, truncate
    from core.SBI.training_checkpoint import bijection_probe
    from core.gui.screens.inference_screen import InferenceScreen
    from core.gui.session import SbiSession
    from tests._fixtures import _nad_cfg, _posterior_artifact

    cfg = _nad_cfg(chi_mode=True)
    P = len(cfg.params_dict) + len(cfg.rescale_params)
    T = reparam.build_inferred_bijection(cfg, log_params=[])
    region = truncate.TruncationRegion([0, 1], [-1.0, -1.0], [1.0, 1.0], n_latent=P, V=None,
                                       probe=bijection_probe(T, P), x_obs_digest="d" * 16)
    trunc = _posterior_artifact(store, cfg, name="trunc", amortized=False, region=region)
    amortized = _posterior_artifact(store, cfg, name="amort")

    qt_app()
    inf = InferenceScreen()
    inf.session = SbiSession(cfg=cfg, inf_prior=_prior_stub())
    pp, vp = inf.posterior_panel, inf.validate_panel

    # the session half: a loaded region reaches Validate
    stub_region = object()
    stub = type("Post", (), {"truncation": stub_region, "x_obs_digest": "feedfacefeedface",
                             "latent": object()})()
    loaded = type("LoadedPosterior", (), {"posterior": stub, "name": "", "id": "p1"})()
    pp._on_posterior(loaded)
    assert inf.session.posterior.posterior.truncation is stub_region
    assert inf.session.posterior.posterior.x_obs_digest == "feedfacefeedface"

    captured = {}
    vp.dispatch = lambda fn, *a, **k: captured.update(args=a, kwargs=k)
    vp._validate()
    assert captured["args"][1] is inf.session.posterior and captured["args"][1].posterior.truncation is stub_region, \
        "Validate did not pass the posterior whose region restricts calibration"

    load = {}
    pp.dispatch = lambda fn, *a, **k: load.update(args=a, kwargs=k)

    # (a) the ask says yes: the load opts in, and it is a LOAD (build_new False)
    asked = []
    pp._ask_load_non_amortized = lambda m: asked.append(m) or True
    pp.post_picker.selected = lambda: (trunc.id, False)
    pp._build_posterior()
    accept = load["kwargs"].get("accept")
    assert isinstance(accept, Accept) and accept.truncated is True and load["args"][3] is False, \
        "the confirmed LOAD does not opt in to non-amortized artifacts"
    assert len(asked) == 1 and asked[0].id == trunc.id, "the dialog was not shown the artifact's manifest"

    # (b) the ask says no: nothing is dispatched and the session keeps what it had. Reinstall a
    # posterior first -- step (a)'s LOAD already reset the session's to None via reset_downstream, so
    # "is before" would hold trivially (None is None) even if Cancel reset it too.
    load.clear()
    pp._on_posterior(loaded)
    before = inf.session.posterior
    pp._ask_load_non_amortized = lambda m: False
    pp._build_posterior()
    assert load == {}, "Cancel dispatched the load anyway"
    assert inf.session.posterior is before, "Cancel reset the session"

    # (c) an amortized artifact is loaded with no question at all
    load.clear()
    pp._ask_load_non_amortized = lambda m: pytest.fail("an amortized posterior must not raise a dialog")
    pp.post_picker.selected = lambda: (amortized.id, False)
    pp._build_posterior()
    assert load["kwargs"]["accept"].used() == [], load["kwargs"]["accept"]

    amortized_stub = type("Post", (), {"truncation": None, "x_obs_digest": None, "latent": object()})()
    amortized_loaded = type("LoadedPosterior", (), {"posterior": amortized_stub, "name": "", "id": "p2"})()
    pp._on_posterior(amortized_loaded)
    assert inf.session.posterior.posterior.truncation is None

def test_the_new_tab_knobs_are_forwarded_and_not_written_to_config():
    """Prior, Posterior and Validate all gained fields. Each must reach its orchestrator function as
    an ARGUMENT -- orchestrator snapshots those constants at import, so writing them would be a
    silent no-op and the run would use the defaults with nothing to say so."""
    from core.gui.screens.inference_screen import InferenceScreen
    from core.gui.session import SbiSession
    from core import config as _cfg

    qt_app()
    inf = InferenceScreen()
    inf.session = SbiSession(draft=object(), cfg=object(), inf_prior=_prior_stub(), posterior=_posterior_stub())

    cap = {}
    for panel in (inf.prior_panel, inf.posterior_panel, inf.validate_panel):
        panel.dispatch = lambda fn, *a, **k: cap.update(kwargs=k)

    inf.validate_panel.cal_n.setText("77")
    inf.validate_panel.cal_scales.setText("11")
    inf.validate_panel._validate()
    assert cap["kwargs"]["n_cal"] == 77 and cap["kwargs"]["cal_n_scales"] == 11, cap["kwargs"]
    assert _cfg.SBC_N_CAL != 77, "the panel wrote the config constant instead of passing an argument"

    pp = inf.posterior_panel
    refused = []
    pp._refusal = lambda exc: refused.append(exc)       # the yellow box, recorded instead of shown
    pp.flow_hidden.setText("64")
    pp.flow_transforms.setText("3")
    pp.flow_lr.setText("0.0007")
    pp.flow_patience.setText("7")
    pp.fisher_dz.setText("0.03")
    pp.post_picker.combo.setCurrentIndex(0)
    pp._build_posterior()
    assert refused == [], [str(e) for e in refused]
    k = cap["kwargs"]
    assert (k["hidden_features"], k["num_transforms"], k["stop_after_epochs"]) == (64, 3, 7), k
    # The two float knobs arrive as typed, through the rule -- not through `value() or default`.
    assert (k["learning_rate"], k["fisher_dz"]) == (0.0007, 0.03), k
    assert _cfg.NSF_HIDDEN_FEATURES != 64, "the panel wrote NSF_HIDDEN_FEATURES"
    # the Fisher knobs ride along on the same call
    assert (k["fisher_m"], k["fisher_points"]) == (_cfg.REPARAM_FISHER_M, _cfg.REPARAM_FISHER_POINTS)
    # A BLANK box is refused at the click, by name, and nothing new is dispatched -- not clamped to 1.
    pp.flow_transforms.setText("")
    pp._build_posterior()
    assert cap["kwargs"] is k, \
        f"a blank Transforms box was dispatched as {cap['kwargs'].get('num_transforms')!r}"
    assert len(refused) == 1 and refused[0].field == "num_transforms", [str(e) for e in refused]
    pp.flow_transforms.setText("3")

    # Prior. NON-VACUOUS now: the old version added the bounds item with no userData (so the click
    # returned at "Select a bounds file first.", before the dispatch) and then skipped its assertions
    # when the kwargs never arrived. A config stub that survives install_config lets the click reach
    # the dispatch, and the assertions are unconditional.
    pp = inf.prior_panel
    inf.session.cfg = _spont_cfg()
    inf.session.draft = type("D", (), {"make_config": lambda self, **kw: inf.session.cfg})()
    pp.prior_picker.selected = lambda: (None, True)           # "(from scratch)": the build branch
    pp.bounds_source.set_direct(False)
    pp.bounds_picker.combo.clear()
    pp.bounds_picker.combo.addItem("master.txt", userData="master.txt")
    pp.cluster_size.setText("9")
    pp.cluster_samples.setText("4")
    pp.sweep_iters.setText("3")
    cap.clear()
    pp._build_prior()
    k = cap["kwargs"]
    assert (k["min_cluster_size"], k["min_samples"], k["num_iterations"]) == (9, 4, 3), k
    assert _cfg.PRIOR_CLUSTER_MIN_SIZE != 9, "the panel wrote PRIOR_CLUSTER_MIN_SIZE"
    # a blank knob box is REFUSED at the click (the yellow box), never clamped to 2 and dispatched
    refused = []
    pp._refusal = lambda e: refused.append(e)
    cap.clear()
    pp.cluster_size.setText("")
    pp._build_prior()
    assert cap == {} and refused[-1].field == "min_cluster_size", (cap, refused)
    # a LOAD click with the same blank box still dispatches: the load branch never reads the knobs
    pp.prior_picker.selected = lambda: ("p1", False)
    pp._build_prior()
    assert cap["kwargs"] and "min_cluster_size" not in cap["kwargs"], cap["kwargs"]
    assert len(refused) == 1, "the load click must not refuse a knob the load branch never reads"

def test_posterior_from_scratch_is_gated_on_a_prior():
    from core.gui.screens.inference_screen import InferenceScreen
    from core.gui.session import SbiSession

    qt_app()
    inf = InferenceScreen()
    inf.session = SbiSession(cfg=object()); inf.refresh_gates()
    pp = inf.posterior_panel
    pp.post_picker.combo.setCurrentIndex(0)               # the "(from scratch)" sentinel (allow_new adds it first)
    assert pp.post_picker.selected()[1] is True, "index 0 should be the from-scratch sentinel"
    pp.refresh_local_gates()
    assert not pp.btn_post.isEnabled(), "training from scratch must be disabled without a prior"
    inf.session.inf_prior = object()
    pp.refresh_local_gates()
    assert pp.btn_post.isEnabled(), "with a prior, training from scratch is allowed"

def test_inference_pickers_repoint_from_draft_and_config():
    """The Prior tab's BOUNDS picker follows the applied model (new_draft), and the Infer tab's CELL
    picker follows the BUILT config's model (install_config) -- neither tab has its own model combo."""
    from core import config
    from core.gui.screens.inference_screen import InferenceScreen
    from core.gui.session import ConfigDraft

    qt_app()
    inf = InferenceScreen()

    inf.new_draft(ConfigDraft(model="HOPF", labels=[], state_dep_drift=False))
    assert inf.prior_panel.bounds_picker.base_path.name == "hopf"
    assert inf.session.cfg is None, "a draft alone must not produce a config"

    class Cfg:
        model = "HOPF"
        params_dict = {}      # the Infer tab validates a picked cell against these
        rescale_params = {}
        force_params_dict = {}
        has_forcing = False   # mirrors SimConfig.has_forcing (empty force_params_dict)
        chi_mode = False      # mirrors SimConfig.chi_mode
        chi_k_pad = config.CHI_K_PAD    # mirrors SimConfig.chi_k_pad (probe-slot capacity)
        observation_mode = "spontaneous"

    inf.install_config(Cfg())
    assert inf.infer_panel.cell_picker.base_path.name == "hopf"
    assert inf.session.draft is not None, "install_config must not wipe the draft/session"

def test_chi_probe_table_is_variable_length_and_capped_by_the_posteriors_slots():
    """Backlog C-2. The core has always accepted 1..chi_k_pad probes at arbitrary frequencies; the GUI
    was the only thing forcing a fixed grid. Rows must be addable and removable, and the cap must be
    chi_k_pad -- which is FROZEN into the trained artifact, so exceeding it is not a soft limit."""
    from core.gui.screens.inference_screen import InferenceScreen

    qt_app()
    inf = InferenceScreen()
    inf.install_config(_chi_cfg(k=3, pad=5))
    panel = inf.infer_panel
    assert len(panel._chi_forced_fields) == 3, "should seed chi_n_freqs rows when empty"

    panel._add_chi_probe()
    panel._add_chi_probe()
    assert len(panel._chi_forced_fields) == 5
    panel._add_chi_probe()                                   # over capacity
    assert len(panel._chi_forced_fields) == 5, "must refuse to exceed chi_k_pad slots"

    panel._remove_chi_probe(panel._chi_forced_fields[0])
    assert len(panel._chi_forced_fields) == 4

def test_chi_probe_rows_keep_each_recording_paired_with_its_own_frequency():
    """The one-widget-per-row invariant. Parallel path/frequency lists let a MIDDLE deletion pair
    recording k with frequency k+1 -- silent, because a lock-in at the wrong frequency decays like a
    sinc and simply returns a smaller number. Delete from the middle and check the survivors."""
    from core.gui.screens.inference_screen import InferenceScreen

    qt_app()
    inf = InferenceScreen()
    inf.install_config(_chi_cfg(k=4))
    panel = inf.infer_panel
    for i, row in enumerate(panel._chi_forced_fields):
        row.path.edit.setText(f"/tmp/rec{i}.csv")
        row.freq.setText(str(1.0 + i))

    panel._remove_chi_probe(panel._chi_forced_fields[1])      # delete from the MIDDLE
    pairs = [r.pair() for r in panel._chi_forced_fields]
    assert pairs == [("/tmp/rec0.csv", 1.0), ("/tmp/rec2.csv", 3.0), ("/tmp/rec3.csv", 4.0)], pairs

def test_chi_probe_table_survives_a_config_rebuild_and_rejects_a_blank_frequency():
    """Two C-2 constraints in one place, because both are about data the GUI cannot regenerate.

    PRESERVATION: rows carry hand-typed drive frequencies and browsed paths -- a record of a bench
    session that already happened. Rebuilding the config (to fix a bounds file, say) must not discard
    them, unlike the forcing rows, which ARE derivable from the config.

    BLANK FREQUENCY: FloatField.value() returns 0.0 on unparseable text, so an empty box is
    indistinguishable from a deliberate zero -- and 0 Hz is a genuine DC probe the lock-in would
    happily attempt. It has to be caught before the run, not after.
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

    # Row 1 still has a blank frequency -> 0.0 -> must be reported, naming the row.
    probs = [p for i, r in enumerate(panel._chi_forced_fields) for p in r.problems(i)]
    assert any("probe 2" in p and "positive" in p for p in probs), probs
    assert not any("probe 1" in p for p in probs), probs

def test_config_units_control_declares_units_and_validates_them():
    """Units DECLARE what the numbers in the files mean (never converting them). Typed units must reach
    the built config, and unusable tokens must be rejected when the model is applied -- not mid-run."""
    from core.config import BOUNDS_PATH
    from core.gui.screens.inference_screen import InferenceScreen

    qt_app()
    screen = InferenceScreen()
    cfgp = screen.config_panel
    cfgp.model_combo.setCurrentText("NADROWSKI")
    cfgp._on_model_changed("NADROWSKI")

    cfgp.units_toggle.set_direct(False)                       # model's units file
    assert cfgp._units_override() is None

    cfgp.units_toggle.set_direct(True)                        # typed
    cfgp.units_text.setText("nm s pN Hz")
    assert cfgp._units_override() == ("nm", "s", "pN", "Hz")
    cfgp._build_config()
    draft = screen.session.draft
    assert draft is not None and draft.units_override == ("nm", "s", "pN", "Hz")

    bounds = BOUNDS_PATH / "nadrowski" / "master.txt"
    if bounds.exists():
        cfg = draft.make_config(str(bounds))
        assert (cfg.time_unit, cfg.freq_unit) == ("s", "Hz")
        # Hz alongside s IS self-consistent (unlike Hz alongside ms), so no warning
        assert cfg.check_unit_consistency() == []
        assert abs(30.0 * cfg.freq_si_to_cell - 30.0) < 1e-12, "in an `s` cell, 30 Hz stays 30"

    before = screen.session.draft
    refused = []
    cfgp._refusal = lambda exc: refused.append(exc)          # the yellow box, recorded instead of shown
    cfgp.units_text.setText("notaunit")                       # must be refused, draft left alone
    cfgp._build_config()
    assert screen.session.draft is before
    assert len(refused) == 1 and refused[0].field == "units", refused
    assert "notaunit" in str(refused[0]), "the message must name the token it could not resolve"

def test_direct_entry_grids_round_trip_their_files():
    """Hand-entered bounds/values must reproduce the parsed file exactly (same names, same ORDER --
    simulators bind parameter columns positionally), and must refuse unparseable or inverted input."""
    from collections import OrderedDict
    from core.config import BOUNDS_PATH, CELL_PATH
    from core.Helpers import file_manager
    from core.gui.widgets.param_grid import BoundsGrid, ValuesGrid

    qt_app()
    bounds = BOUNDS_PATH / "nadrowski" / "master.txt"
    cell = CELL_PATH / "nadrowski" / "master_weak.txt"
    if not (bounds.exists() and cell.exists()):
        return                                                   # environment without Resources: skip

    p, r, f, _ = file_manager.parse_bounds_file(str(bounds))
    grid = BoundsGrid()
    assert grid.problems(), "an unloaded grid must report a problem rather than build nothing"
    grid.load(p, r, f)
    assert grid.problems() == []
    gp, gr, gf = grid.to_dicts()
    for got, want in ((gp, p), (gr, r), (gf, f)):
        assert list(got) == list(want), "parameter ORDER must survive the round trip"
        for name in want:
            assert got[name][1] == want[name][1], name

    grid._rows[("PARAM", "k")].lo.setText("")                    # unparseable -> refused, not 0.0
    assert any("k" in m for m in grid.problems())
    grid._rows[("PARAM", "k")].lo.setText("9")                   # min > max -> refused
    grid._rows[("PARAM", "k")].hi.setText("1")
    assert any("less than" in m for m in grid.problems())

    inits, vp, vr, vf = file_manager.parse_values_file(str(cell))
    vgrid = ValuesGrid()
    vgrid.load(inits, vp, vr, vf)
    assert vgrid.problems() == []
    gi, gvp, gvr, gvf = vgrid.to_dicts()
    assert list(gvp) == list(vp) and list(gi) == list(inits)
    assert all(abs(gvp[n] - vp[n]) < 1e-12 for n in vp)
    assert isinstance(gvp, OrderedDict)

def test_overlay_alignment_and_best_fit_rankings():
    """Phase alignment must recover a known lag; both best-fit rankings must find a planted match; and
    the trace ranking must be invariant to rolling the reference (absolute phase carries no info)."""
    import math
    import numpy as np
    import torch
    from core.SBI import overlay

    torch.manual_seed(0)
    n, dt, f = 2048, 1e-3, 7.5
    t = torch.arange(n, dtype=torch.float32) * dt

    # unambiguous (non-periodic) reference -> the lag is exact
    chirp = torch.sin(2 * math.pi * (2.0 + 6.0 * t) * t) * torch.hann_window(n)
    shifts = torch.tensor([0, 37, -91])
    rolled = torch.stack([torch.roll(chirp, int(s)) for s in shifts])
    aligned, lags = overlay.align_to(chirp, rolled)
    assert torch.equal(lags, shifts), (lags, shifts)
    assert float((aligned - chirp.unsqueeze(0)).abs().max()) < 1e-5

    gt = torch.sin(2 * math.pi * f * t) + 0.05 * torch.randn(n)
    cand = torch.stack([torch.sin(2 * math.pi * (f * 1.3) * t),      # wrong frequency
                        0.4 * torch.sin(2 * math.pi * f * t),        # wrong amplitude
                        torch.roll(gt, 123)])                        # the planted match, phase-shifted
    order, _rmse, _al = overlay.rank_by_trace(gt, cand)
    assert int(order[0]) == 2, "the planted draw must win on waveform"
    order2, _, _ = overlay.rank_by_trace(torch.roll(gt, -500), cand)
    assert int(order2[0]) == 2, "ranking must not depend on the reference's absolute phase"

    # summary-stat ranking: exact match wins, and the CONSTANT conditioning column is ignored
    obs = torch.tensor([[1.0, 2.0, 3.0, 42.0]])
    sim = torch.tensor([[9.0, 9.0, 9.0, 42.0], [1.0, 2.0, 3.0, 42.0], [1.5, 2.5, 2.0, 42.0]])
    o, d = overlay.rank_by_stats(sim, obs)
    assert int(o[0]) == 1 and float(d[1]) < 1e-9

    # phase-invariant summaries are well-formed
    freqs, lo, med, hi, dropped = overlay.psd_band(cand, dt)
    assert bool((lo <= hi).all()) and abs(float(freqs[med.argmax()]) - f) < 2.0
    centres, mean, clo, chi_, c_dropped = overlay.cycle_average(gt.unsqueeze(0), dt, f)
    assert dropped == 0 and c_dropped == 0, "clean traces must drop nothing"
    good = torch.isfinite(mean)
    assert int(good.sum()) > 0.8 * len(mean)
    assert 1.5 < float(mean[good].max() - mean[good].min()) < 2.5, "should recover the unit amplitude"
    assert overlay.cycle_window(n, dt, f, 15) == 2000

def test_a_divergent_draw_does_not_erase_the_whole_psd_band():
    """THE 2026-08-25 FAILURE, pinned.

    The power-spectrum figure came back as a bare observation line: no band, no median, no message.
    Cause: a broad posterior samples parameter sets that do not integrate stably, `|rfft|**2` of such a
    trace overflows to inf, and `torch.quantile` propagates one non-finite entry across every column --
    so ONE bad draw in a thousand silently erased the band for all of them.

    It went unnoticed for so long because the two sibling figures survive it: the overlay band takes
    only the 50 best draws, and cycle_average confines the damage to a single phase bin. So the
    symptom looked like a plotting bug in one figure rather than a property of the posterior.
    """
    import math
    import torch
    from core.SBI import overlay

    n, dt, f = 4096, 1.0 / 500.0, 12.0
    t = torch.arange(n, dtype=torch.float64) * dt
    good = torch.stack([torch.sin(2 * math.pi * f * t) * a for a in (0.9, 1.0, 1.1, 1.05, 0.95)])

    clean_freqs, clean_lo, clean_med, clean_hi, clean_drop = overlay.psd_band(good, dt)
    assert clean_drop == 0 and torch.isfinite(clean_med).all()

    for label, bad_row in (("inf", torch.full((n,), float("inf"), dtype=torch.float64)),
                           ("nan", torch.full((n,), float("nan"), dtype=torch.float64)),
                           ("overflow", torch.sin(2 * math.pi * f * t) * 1e300)):
        traces = torch.cat([good, bad_row.unsqueeze(0)], dim=0)
        freqs, lo, med, hi, dropped = overlay.psd_band(traces, dt)
        assert dropped == 1, f"{label}: expected 1 dropped draw, got {dropped}"
        assert torch.isfinite(med).all(), f"{label}: one bad draw still poisoned the median"
        assert torch.isfinite(lo).all() and torch.isfinite(hi).all(), f"{label}: band not finite"
        assert bool((lo <= hi).all())
        # and the surviving band is the clean one, not some rescued average of the wreckage
        assert torch.allclose(med, clean_med), f"{label}: the good draws' band changed"

def test_the_psd_band_reports_when_nothing_survives():
    """All-bad input must produce NaNs and a count, not an exception and not a silent empty plot."""
    import torch
    from core.SBI import overlay

    traces = torch.full((4, 1024), float("nan"), dtype=torch.float64)
    freqs, lo, med, hi, dropped = overlay.psd_band(traces, 1.0 / 500.0)
    assert dropped == 4
    assert not torch.isfinite(med).any() and len(med) == len(freqs)

def test_the_cycle_average_masks_non_finite_samples_instead_of_binning_them():
    """A NaN phase goes through `.long()` as a garbage integer that `clamp` parks in bin 0, so an
    unmasked divergent draw silently corrupts one end of the cycle -- which reads as a real feature."""
    import math
    import torch
    from core.SBI import overlay

    n, dt, f = 4096, 1.0 / 500.0, 12.0
    t = torch.arange(n, dtype=torch.float64) * dt
    good = torch.sin(2 * math.pi * f * t).unsqueeze(0)

    _, m_clean, _, _, d_clean = overlay.cycle_average(good, dt, f)
    assert d_clean == 0

    dirty = torch.cat([good, torch.full((1, n), float("nan"), dtype=torch.float64)], dim=0)
    _, m_dirty, lo_d, hi_d, d_dirty = overlay.cycle_average(dirty, dt, f)
    assert d_dirty == n, f"expected the whole bad row masked, got {d_dirty}"
    live = torch.isfinite(m_clean)
    assert torch.allclose(m_dirty[live], m_clean[live]), "the good row's cycle changed"
    assert torch.isfinite(m_dirty[live]).all(), "bin 0 was poisoned by the NaN row"

def test_an_empty_predictive_band_is_annotated_on_the_psd_figure():
    """An absent band must not be mistakable for a rendering glitch -- which is exactly what happened.
    When nothing finite survives, the figure has to say so in the axes."""
    import numpy as np
    from core.Helpers import visualizers

    qt_app()
    freqs = np.linspace(0.1, 100, 64)
    nan = np.full(64, np.nan)
    from matplotlib import pyplot as plt
    fig = visualizers.plot_psd_overlay(freqs, np.ones(64), nan, nan, nan, n_dropped=7)
    texts = [t.get_text() for t in fig.axes[0].texts]
    plt.close(fig)                                   # the gate's pyplot is shared: close what is drawn
    assert any("no finite" in t for t in texts), f"no explanation drawn: {texts}"
    assert any("7" in t for t in texts), f"the dropped count is not on the figure: {texts}"

    # and with a healthy band there must be no such annotation
    ok = visualizers.plot_psd_overlay(freqs, np.ones(64), np.ones(64) * .5, np.ones(64),
                                      np.ones(64) * 2)
    ok_texts = [t.get_text() for t in ok.axes[0].texts]
    plt.close(ok)
    assert not ok_texts, "annotated a perfectly good band"

def test_overlay_figures_render_and_are_picklable():
    """Each new Infer-tab figure must build and survive pickling (the 'Pop out' path unpickles it)."""
    import pickle
    import numpy as np
    from core.Helpers import visualizers

    qt_app()
    t = np.linspace(0, 2, 400)
    y = np.sin(2 * np.pi * 7.5 * t)
    labels_ = [f"$p_{{{i}}}$" for i in range(13)]
    vals = list(range(13))
    figs = [
        visualizers.plot_best_fit_overlay(t, y, y * 1.05, param_labels=labels_, param_values=vals,
                                          ground_truth=vals, criterion="closest summary statistics",
                                          score_text="RMS z = 0.4"),
        visualizers.plot_overlay_band(t, y, y - 0.2, y, y + 0.2, n_used=20),
        visualizers.plot_psd_overlay(np.linspace(0.1, 100, 50), np.ones(50), np.ones(50) * 0.5,
                                     np.ones(50), np.ones(50) * 2),
        visualizers.plot_cycle_average(np.linspace(0, 2 * np.pi, 48), np.zeros(48), np.zeros(48),
                                       -np.ones(48), np.ones(48)),
    ]
    from matplotlib import pyplot as plt
    try:
        for fig in figs:
            assert fig.axes, "figure has no axes"
            # must not raise; the copy registers with pyplot too, so it is closed like the original
            plt.close(pickle.loads(pickle.dumps(fig)))
        # the 13-row parameter table gets its own axes, never the title
        assert len(figs[0].axes) == 2
    finally:
        for fig in figs:
            plt.close(fig)

def test_help_badge_carries_its_text():
    from core.gui.widgets.help_badge import HelpBadge
    qt_app()
    assert HelpBadge("what this does").toolTip() == "what this does"

def test_simulated_inference_emits_the_ground_truth_figure(tmp_path):
    """The simulated-inference COMPOSITION shows the 'Ground-truth trace' figure before inferring (the
    old Simulate tab did only the first half; the tab is gone, the figure is not). A real SDE sim is too
    slow for a unit test, so stub the heavy pieces and assert the fig_sink wiring. The figure comes from
    the observation stage itself (generate_observations writes the artifact and its trace figure), so
    the stub emits it exactly as the real stage would before handing back a LoadedObservation-shaped
    stand-in.

    The GUI runner this used to exercise is gone (piece 2, T6): the tab dispatches
    orchestrator.simulated_inference directly, so the wiring under test is the composition's."""
    import types
    import torch
    from core import cli, orchestrator

    qt_app()

    class Cfg:
        length_unit = "nm"                       # trace y-axis unit (round-4 labels)
        sources = {}

        def get_unit_conversion_factor(self, _unit):
            return 1.0

    seen = []

    def stub_generate_observations(cfg, *, fig_sink=None, **kw):
        if fig_sink is not None:
            fig_sink("Ground-truth trace", None)
        return types.SimpleNamespace(x_obs=torch.zeros(1, 5), obs_data=None,
                                     t_dim=torch.linspace(0, 1, 5).unsqueeze(0), id="o", name="")

    real_gt, real_go = cli.load_and_validate_gt, orchestrator.generate_observations
    real_iv = orchestrator.infer_and_visualize
    cli.load_and_validate_gt = lambda cfg, path: []
    orchestrator.generate_observations = stub_generate_observations
    orchestrator.infer_and_visualize = lambda *a, **k: types.SimpleNamespace(
        id="i", name="", results={"ppc": {"coverage_90": 0.9}})
    cell = tmp_path / "cell.txt"
    cell.touch()                       # require_file("cell", …) runs before the (stubbed) parse
    try:
        post = types.SimpleNamespace(posterior=types.SimpleNamespace(x_obs_digest=None, truncation=None))
        # T_obs=0.1s is below T_MIN_EXP_S on purpose (the spec's test row): record the resulting
        # PreflightWarning instead of leaking it -- `match=` would re-emit any warning that does not
        # match, and this call is not asserted to emit exactly one.
        with pytest.warns(orchestrator.PreflightWarning) as rec:
            obs, inf = orchestrator.simulated_inference(
                Cfg(), post, 0.1, cell=str(cell), fig_sink=lambda title, fig: seen.append(title))
    finally:
        cli.load_and_validate_gt = real_gt
        orchestrator.generate_observations = real_go
        orchestrator.infer_and_visualize = real_iv

    assert any("below the training range minimum" in str(w.message) for w in rec), \
        [str(w.message) for w in rec]
    assert seen == ["Ground-truth trace"], seen
    assert (obs.id, inf.id) == ("o", "i"), "the composition must return (observation, inference)"


def test_the_infer_tab_dispatches_the_compositions(tmp_path):
    """The Infer tab's four Run paths must call the COMPOSITIONS, not a GUI-local runner.

    The runner module existed only to be module-level and Qt-free with an injectable fig_sink; the
    compositions are both, and a wrapper is one more hop where positional order can drift. Each branch
    is checked for the function it dispatches and for the positional argument that decides the run:
    the simulated branch's cell (or hand-entered values) and the experimental branch's RecordingSet.
    """
    from core import orchestrator
    from core.gui.screens.inference_screen import InferenceScreen
    from core.gui.session import SbiSession

    qt_app()
    inf = InferenceScreen()
    inf.install_config(_chi_cfg(k=2))
    inf.session.posterior = _posterior_stub()
    inf.session.inf_prior = _prior_stub()
    panel = inf.infer_panel

    cap = {}
    panel.dispatch = lambda fn, *a, **k: cap.update(fn=fn, args=a, kwargs=k)

    # simulated, from a cell FILE. Real (empty) files: the click now refuses a missing cell or
    # recording before it dispatches (V2), and the dispatch is stubbed so nothing reads them.
    cell = tmp_path / "cell.txt"
    cell.touch()
    panel.infer_mode.setCurrentIndex(0)
    panel.cell_source.set_direct(False)
    panel.cell_picker.selected_path = lambda: str(cell)
    panel._cell_problems = []
    panel.sim_tobs.setText("3.5")
    panel._infer()
    assert cap["fn"] is orchestrator.simulated_inference, cap["fn"]
    assert cap["args"][2] == 3.5 and cap["kwargs"]["cell"] == str(cell)
    assert cap["kwargs"]["gt_values"] is None and cap["kwargs"]["prior"] is inf.session.inf_prior
    assert cap["kwargs"]["provide_fig_sink"] is True
    assert cap["kwargs"]["on_result"] == panel._on_observation
    assert cap["kwargs"]["accept"] is None, "an amortized posterior dispatches no Accept"

    # experimental, chi: one passive recording + the probe pairs, as a RecordingSet
    passive = tmp_path / "passive.npy"
    passive.touch()
    probes = [tmp_path / f"probe{i}.npy" for i in range(len(panel._chi_forced_fields))]
    for p in probes:
        p.touch()

    def _chi_run():
        cap.clear()
        panel.infer_mode.setCurrentIndex(1)
        panel.chi_spont.edit.setText(str(passive))
        for i, row in enumerate(panel._chi_forced_fields):
            row.path.edit.setText(str(probes[i]))
            row.freq.setText(str(10.0 + i))
        panel.chi_tobs.setText("2.0")
        panel._infer()

    _chi_run()
    assert cap["fn"] is orchestrator.experimental_inference, cap["fn"]
    rec = cap["args"][2]
    assert rec.spont == str(passive) and rec.forced == ((str(probes[0]), 10.0),
                                                        (str(probes[1]), 11.0))
    assert rec.T_obs_s == 2.0 and cap["kwargs"]["provide_fig_sink"] is True
    assert cap["kwargs"]["accept"] is None, "an amortized posterior dispatches no Accept"

    # F16: a NON-AMORTIZED posterior with the box ticked -- the consent travels on both branches
    inf.session.posterior = _posterior_stub(truncation=object(), x_obs_digest="d" * 16)
    inf.refresh_gates()
    panel.other_obs.setChecked(True)
    cap.clear()
    panel.infer_mode.setCurrentIndex(0)
    panel._infer()
    assert cap["fn"] is orchestrator.simulated_inference, cap["fn"]
    assert cap["kwargs"]["accept"].other_observation is True, cap["kwargs"]["accept"]
    _chi_run()
    assert cap["fn"] is orchestrator.experimental_inference, cap["fn"]
    assert cap["kwargs"]["accept"].other_observation is True, cap["kwargs"]["accept"]


def test_a_confirmed_near_miss_dispatches_new_run(monkeypatch):
    """The dialog gates the dispatch and its answer TRAVELS: consent is a keyword on the stage call,
    not a flag the panel keeps to itself. The stage asks the same question again inside the worker --
    this is only the early, cheap version of the refusal -- so a dialog that answered "yes" and then
    dispatched without new_run would be refused seconds later for no reason the user can see.

    And it FAILS OPEN in both the shapes that can happen in the wild: no near miss, and a detector
    that raises. A warning that can block a run is worse than no warning."""
    from core import orchestrator
    from core.gui.screens.inference_screen import InferenceScreen
    from core.gui.session import SbiSession

    qt_app()
    inf = InferenceScreen()
    inf.session = SbiSession(cfg=object(), inf_prior=_prior_stub())
    pp = inf.posterior_panel
    pp.post_picker.selected = lambda: ("", True)              # "(from scratch)" -> a NEW run
    pp.num_runs.setText("2")
    pp.run_size_cap.setText("8")
    sent = {}
    pp.dispatch = lambda fn, *a, **k: sent.update(fn=fn, args=a, kwargs=k)
    row = [{"name": "abcdef012345", "batches": 3989, "field": "n_runs", "mine": 2, "theirs": 3}]

    # one near miss, answered yes. monkeypatch, never a bare rebind: an assertion that fails below
    # would otherwise leave the stub installed on core.orchestrator for the rest of this
    # single-process gate, silently disabling D7 for every later test in the run.
    # The detector must be asked about THIS run's identity: a stub that ignored its arguments stayed
    # green through a regression that dropped run_size_cap, where the real detector looks at another
    # identity, the dialog never shows, and the stage then refuses with advice to press a button that
    # cannot appear.
    seen = {}
    monkeypatch.setattr(orchestrator, "fresh_run_near_misses",
                        lambda cfg, prior, **k: seen.update(k, prior=prior) or list(row))
    pp._ask_new_run = lambda near, n_runs: True
    pp._build_posterior()
    assert seen["num_runs"] == 2 and seen["run_size_cap"] == 8, seen
    assert seen["prior"] is inf.session.inf_prior, seen
    assert sent["fn"] is orchestrator.build_posterior and sent["kwargs"]["new_run"] is True
    assert sent["kwargs"]["num_runs"] == 2 and sent["kwargs"]["run_size_cap"] == 8

    # answered no: nothing is dispatched at all
    sent.clear()
    pp._ask_new_run = lambda near, n_runs: False
    pp._build_posterior()
    assert sent == {}, "Cancel must not start the run"

    # no near miss: dispatched without consent, and nothing is asked
    sent.clear()
    monkeypatch.setattr(orchestrator, "fresh_run_near_misses", lambda *a, **k: [])
    pp._ask_new_run = lambda near, n_runs: pytest.fail("a run with no near miss must not be questioned")
    pp._build_posterior()
    assert sent["kwargs"]["new_run"] is False

    # a detector that raises: the run proceeds and the log says so
    sent.clear()

    def _boom(*a, **k):
        raise RuntimeError("unreadable header")

    monkeypatch.setattr(orchestrator, "fresh_run_near_misses", _boom)
    lines = []
    pp.log_pane.append_line = lambda text, kind="": lines.append((kind, text))
    pp._build_posterior()
    assert sent["kwargs"]["new_run"] is False
    assert any(k == "warning" and "unreadable header" in t for k, t in lines), lines


def test_the_d7_and_d8_dialogs_default_to_cancel(monkeypatch):
    """Enter on either dialog must do the SAFE thing. Spec §5.3 makes Cancel the default: on D8 the
    other button loads a NON-AMORTIZED posterior with Accept(truncated=True), and on D7 it starts a
    fresh cache one setting away from a committed one, the accident D7 exists to stop. Every other test
    replaces the _ask_* methods, so only this one sees the dialogs themselves."""
    from types import SimpleNamespace
    from PySide6.QtWidgets import QMessageBox
    from core.gui.screens.inference_screen import InferenceScreen

    qt_app()
    inf = InferenceScreen()
    pp = inf.posterior_panel
    seen = []
    monkeypatch.setattr(QMessageBox, "exec", lambda self: seen.append(self) or 0)

    def ask_new_run():
        return pp._ask_new_run([{"name": "a", "batches": 3, "field": "n_runs", "mine": 2, "theirs": 3}], 2)

    def ask_load():
        return pp._ask_load_non_amortized(SimpleNamespace(
            body={"truncation": {"level": 0.99, "dims": [0], "x_obs_digest": "d" * 16}}, name="r", id="x"))

    def _click(text):
        return next(b for b in seen[-1].buttons() if b.text() == text)

    for ask, destructive in ((ask_new_run, "Start a new run anyway"), (ask_load, "Load it")):
        # nothing clicked: the default is Cancel
        seen.clear()
        monkeypatch.setattr(QMessageBox, "exec", lambda self: seen.append(self) or 0)
        assert ask() is False
        assert seen and seen[-1].defaultButton() is not None
        assert seen[-1].defaultButton().text() == "Cancel", seen[-1].defaultButton().text()

        # Enter: the default button is clicked, and the answer is NO
        seen.clear()
        monkeypatch.setattr(QMessageBox, "exec",
                            lambda self: seen.append(self) or self.defaultButton().click() or 0)
        assert ask() is False, f"clicking the default button on the '{destructive}' dialog answered yes"

        # the destructive button clicked: the answer is YES
        seen.clear()
        monkeypatch.setattr(QMessageBox, "exec",
                            lambda self: seen.append(self) or _click(destructive).click() or 0)
        assert ask() is True, f"clicking '{destructive}' did not answer yes"


def test_the_tsnpe_tab_dispatches_a_loaded_observation(monkeypatch):
    """The tab loads the observation ITSELF, on the GUI thread, and hands the wrapper to the stage.

    Loading is what re-hashes the file and checks its mode and conditioning width against the session's
    config, so a mismatch is a dialog within milliseconds instead of an exception hours into a round.
    No stage loads by reference any more, and no GUI dispatch names a store (build_app installs the
    default once).

    A load REFUSAL reaches _on_error as the exception itself -- no wrapper, no re-typing, no "Could not
    load observation '<key>':" prefix (every store refusal already names the observation, store.py's
    load_observation labels each sentence) -- so a Refusal opens the yellow box. The old prefix turned
    it into a string the yellow box could never be opened for.
    """
    import types
    from core import orchestrator
    from core.gui.panels.inference import tsnpe_tab as tt
    from core.gui.screens.inference_screen import InferenceScreen
    from core.gui.session import SbiSession
    from core.refusals import Refusal
    from tests._fixtures import qt_app

    qt_app()
    inf = InferenceScreen()
    inf.session = SbiSession(cfg=object(), inf_prior=_prior_stub(), posterior=_posterior_stub())
    panel = inf.tsnpe_panel
    panel.obs_picker.key = lambda: "20260910T120000"

    sentinel = types.SimpleNamespace(id="20260910T120000", name="obs")
    cap = {}
    panel.dispatch = lambda fn, *a, **k: cap.update(fn=fn, args=a, kwargs=k)
    errors = []
    panel._on_error = lambda exc, tb: errors.append(exc)
    monkeypatch.setattr(tt, "default_store",
                        lambda: types.SimpleNamespace(load_observation=lambda cfg, ref: sentinel))
    panel.n_dirs.setText("2")
    panel.hpd.setText("0.999")
    panel.num_runs.setText("3")
    panel.run_size_cap.setText("8")
    panel._round()
    assert cap["fn"] is orchestrator.tsnpe_round, cap["fn"]
    assert cap["args"][3] is sentinel, "the tab must pass the LOADED observation, not its key"
    assert cap["kwargs"]["n_directions"] == 2 and cap["kwargs"]["level"] == 0.999
    assert cap["kwargs"]["num_runs"] == 3 and cap["kwargs"]["run_size_cap"] == 8
    assert cap["kwargs"]["new_run"] is False and cap["kwargs"]["provide_fig_sink"] is True
    assert "store" not in cap["kwargs"], "the GUI names no store; build_app installs the default"

    # ticked: the consent travels, and the box clears itself so it cannot silently persist
    cap.clear()
    panel.new_run.setChecked(True)
    panel._round()
    assert cap["kwargs"]["new_run"] is True
    assert panel.new_run.isChecked() is False, "the box must clear after each dispatch"

    # a load refusal dispatches NOTHING and reaches _on_error as the Refusal it is
    cap.clear()

    def _boom(cfg, ref):
        raise Refusal("Observation '20260910T120000' has conditioning width 61, but this config's "
                      "is 50.", field="observation")

    monkeypatch.setattr(tt, "default_store", lambda: types.SimpleNamespace(load_observation=_boom))
    panel._round()
    assert cap == {}, "a round was dispatched with an observation that would not load"
    assert errors and isinstance(errors[-1], Refusal), errors
    assert errors[-1].field == "observation"
    assert "20260910T120000" in str(errors[-1]) and "width 61" in str(errors[-1]), errors


def test_the_tsnpe_tab_passes_a_load_bug_to_on_error_unwrapped(monkeypatch):
    """The other half of the routing: a BUG in the load (not a refusal) reaches _on_error as the bare
    exception with its traceback, so it opens the RED box. The tab must not decide the kind itself --
    it passes what it caught, and BasePanel._on_error decides by isinstance."""
    import types
    from core.gui.panels.inference import tsnpe_tab as tt
    from core.gui.screens.inference_screen import InferenceScreen
    from core.gui.session import SbiSession
    from core.refusals import Refusal
    from tests._fixtures import qt_app

    qt_app()
    inf = InferenceScreen()
    inf.session = SbiSession(cfg=object(), inf_prior=_prior_stub(), posterior=_posterior_stub())
    panel = inf.tsnpe_panel
    panel.obs_picker.key = lambda: "20260910T120000"
    cap, errors = {}, []
    panel.dispatch = lambda fn, *a, **k: cap.update(fn=fn, args=a, kwargs=k)
    panel._on_error = lambda exc, tb: errors.append((exc, tb))

    def _bug(cfg, ref):
        raise RuntimeError("observation.pt is not a tensor")

    monkeypatch.setattr(tt, "default_store", lambda: types.SimpleNamespace(load_observation=_bug))
    panel._round()
    assert cap == {}, "a round was dispatched with an observation that would not load"
    exc, tb = errors[-1]
    assert isinstance(exc, RuntimeError) and not isinstance(exc, Refusal)
    assert str(exc) == "observation.pt is not a tensor", "the tab must not re-phrase the exception"
    assert "Traceback" in tb and "RuntimeError: observation.pt is not a tensor" in tb


def test_the_tsnpe_new_run_box_is_not_persisted():
    """D7's consent is per-run. Persisting it would silence the near-miss refusal for every future
    round in every future session -- exactly the accident the refusal exists to catch."""
    from core.gui.panels.inference.tsnpe_tab import TSNPEPanel

    src = ast.unparse(ast.parse(textwrap.dedent(inspect.getsource(TSNPEPanel.save_settings))))
    assert "new_run" not in src, "save_settings must not persist the near-miss consent"
    src = ast.unparse(ast.parse(textwrap.dedent(inspect.getsource(TSNPEPanel.restore_settings))))
    assert "new_run" not in src, "restore_settings must not restore the near-miss consent"


def test_the_infer_tab_other_observation_box():
    """D8 on the Infer tab. The box is the GUI's only way to say "yes, run this TSNPE posterior on a
    different observation" -- and it must be UNREACHABLE for an amortized posterior, where it would mean
    nothing, and must never survive a change of posterior or a restart: a stale tick would silence
    guardrail 2 for a posterior the user never answered the question about."""
    from core.artifacts import Accept
    from core.gui.screens.inference_screen import InferenceScreen
    from core.gui.session import SbiSession

    qt_app()
    inf = InferenceScreen()
    inf.install_config(_chi_cfg(k=1))
    panel = inf.infer_panel

    # (a) amortized: greyed out, and no accept travels
    inf.session.posterior = _posterior_stub(id_="amort")
    inf.refresh_gates()
    assert panel.other_obs.isEnabled() is False
    assert panel._accept() is None, "an amortized posterior must send no Accept at all"

    # (b) non-amortized and ticked: the consent travels
    inf.session.posterior = _posterior_stub(id_="trunc", truncation=object(), x_obs_digest="d" * 16)
    inf.refresh_gates()
    assert panel.other_obs.isEnabled() is True
    panel.other_obs.setChecked(True)
    acc = panel._accept()
    assert isinstance(acc, Accept) and acc.other_observation is True and acc.truncated is False

    # (c) a DIFFERENT posterior clears it, even another truncated one
    inf.session.posterior = _posterior_stub(id_="trunc2", truncation=object(), x_obs_digest="e" * 16)
    inf.refresh_gates()
    assert panel.other_obs.isChecked() is False, "the tick survived a change of posterior"
    assert panel._accept() is None

    # (d) not persisted: a restart must not restore consent
    src = ast.unparse(ast.parse(textwrap.dedent(inspect.getsource(type(panel).save_settings))))
    assert "other_obs" not in src, "save_settings must not persist the other-observation consent"
    src = ast.unparse(ast.parse(textwrap.dedent(inspect.getsource(type(panel).restore_settings))))
    assert "other_obs" not in src, "restore_settings must not restore it"


def test_every_field_key_has_a_window_control_and_the_fix_sentences_name_it():
    """V3's window half. A Refusal names its field by a key and never a box, tab, button or flag;
    the yellow "Check your inputs" box gets its "how to fix here" line from ONE table, and the
    inference tabs build their static rows from the same table's labels (label(key)), so a control
    is named in one place and a renamed one cannot leave a stale sentence behind.

    (a) the table knows every registry key and no other -- an unmapped key would show a refusal
        with no way out, and an entry nobody raises is a sentence that can go stale unseen;
    (b) every tuple names a tab as InferenceScreen TITLES it (read off the built screen, not a
        copy of the tuple), a non-empty label, and renders the box sentence; a sentence entry is
        returned verbatim and has no label; a None entry says nothing and has no label; and none
        of the three ever raises from fix_sentence, which runs while a refusal is being shown;
    (c) the sentences the walkthrough rows C2 and C3 read, and the consents, verbatim -- each
        quotes the control's own text (posterior_tab.py:195, tsnpe_tab.py:88, infer_tab.py:112);
    (d) the keys with no window control are exactly the six the window never exposes plus the
        eleven tool-only diagnostics knobs, so a tool-only key renders no window sentence;
    (e) the three drive labels are built exactly as the Infer tab builds its rows
        (_rebuild_forcing_fields: labels.gui_forcing_label with config.FORCING_DISPLAY_UNITS, which
        cli.INFERENCE_PROMPT_UNITS aliases), so the read-back over the built Infer tab (Task 15)
        can find them.
    """
    from core import config
    import core.gui.fields as gui_fields
    from core.gui.screens.inference_screen import InferenceScreen
    from core.Helpers import labels
    from core.refusals import FIELDS

    # (a) the same key set, in both directions
    assert set(gui_fields.CONTROL) == set(FIELDS), (
        f"only in CONTROL: {sorted(set(gui_fields.CONTROL) - set(FIELDS))}; "
        f"only in FIELDS: {sorted(set(FIELDS) - set(gui_fields.CONTROL))}")

    # (b) the three shapes, against the screen's own tab titles
    qt_app()
    screen = InferenceScreen()
    tabs = [screen.tabs.tabText(i) for i in range(screen.tabs.count())]
    assert tabs == ["Config", "Prior", "Posterior", "Validate", "Infer", "TSNPE"]
    for key, entry in gui_fields.CONTROL.items():
        if isinstance(entry, tuple):
            tab, text = entry
            names = tab if isinstance(tab, tuple) else (tab,)
            assert names and all(t in tabs for t in names), f"{key}: {tab!r} is not a tab title"
            assert isinstance(text, str) and text, f"{key}: empty label"
            assert gui_fields.label(key) == text
            assert gui_fields.fix_sentence(key) == \
                f"Set it in the '{text}' box on the {' or '.join(names)} tab."
        elif isinstance(entry, str):
            assert entry.endswith("."), f"{key}: a fix sentence ends with a period: {entry!r}"
            assert gui_fields.fix_sentence(key) == entry
            with pytest.raises(KeyError):
                gui_fields.label(key)
        else:
            assert entry is None, f"{key}: {entry!r} is none of the three shapes"
            assert gui_fields.fix_sentence(key) == ""
            with pytest.raises(KeyError):
                gui_fields.label(key)
    assert gui_fields.fix_sentence(None) == ""
    assert gui_fields.fix_sentence("no_such_key") == ""
    with pytest.raises(KeyError):
        gui_fields.label("no_such_key")

    # (c) verbatim: the walkthrough sentences and the consents
    assert gui_fields.fix_sentence("t_obs") == "Set it in the 'T_obs (s)' box on the Infer tab."
    assert gui_fields.fix_sentence("n_directions") == \
        "Set it in the 'Directions truncated' box on the TSNPE tab."
    assert gui_fields.fix_sentence("accept_other_observation") == \
        "Tick 'Run on a different observation' on the Infer tab."
    assert gui_fields.fix_sentence("new_run") == (
        "Answer 'Start a new run anyway' in the dialog on the Posterior tab, or tick 'Start a new "
        "simulation even if a cache one setting away exists' on the TSNPE tab.")
    assert gui_fields.fix_sentence("accept_truncated") == "Confirm the load in the dialog on the Posterior tab."
    assert gui_fields.fix_sentence("recording_spont") == (
        "Select the passive recording on the Infer tab (the 'Spontaneous' box on the driven page, "
        "the 'Passive' box on the χ page).")
    assert gui_fields.fix_sentence("recording_probe") == \
        "Pick the probe's recording in the χ probe table on the Infer tab."
    assert gui_fields.fix_sentence("name") == "Choose another name in the Save box."
    assert (gui_fields.fix_sentence("chi_f0") == gui_fields.fix_sentence("chi_freq_bounds")
            == "Fixed by measurement: change it in config.py, deliberately.")
    # the budget boxes sit on two tabs, and a refusal raised on either must name the one the user is on
    assert gui_fields.CONTROL["num_runs"] == (("Posterior", "TSNPE"), "Batches")
    assert gui_fields.fix_sentence("run_size_cap") == \
        "Set it in the 'Max rows per batch (0 = auto)' box on the Posterior or TSNPE tab."

    # (d) no window control: the six the window never exposes, and the tool-only set
    assert {k for k, e in gui_fields.CONTROL.items() if e is None} == {
        "checkpoint_every", "resume", "device", "n_samples", "num_posterior_samples", "max_num_epochs",
        "repeats", "n_points", "n_worst", "top_n", "m", "m_noise", "rel", "min_valid", "rows",
        "n_sweep", "chi_k_fixed"}

    # (e) the drive labels, as the Infer tab builds them
    for key, name in (("drive_amplitude", "amp"), ("drive_frequency", "freq"), ("drive_phase", "phase")):
        assert gui_fields.CONTROL[key] == (
            "Infer", labels.gui_forcing_label(name, config.FORCING_DISPLAY_UNITS[name])), key
    assert gui_fields.label("drive_amplitude") == "A (N)"
    assert gui_fields.label("drive_frequency") == "f (Hz)"
    assert gui_fields.label("drive_phase") == "φ (rad)"


def test_a_rename_failure_reads_as_a_name_refusal(monkeypatch):
    """Save on the Prior and Posterior tabs is a rename, and a bad or taken name is a StoreError -- a
    Refusal with field="name" since piece 3 -- so it opens the yellow box with the core's sentence
    as its text and this front end's fix ("Choose another name in the Save box.") under it. It used
    to go through _config_error, whose box read "The configuration could not be built." over a
    sentence about a name: a lie for a rename. A rename that fails for any other reason is a bug and
    stays red, with its traceback. Either way the loaded artifact keeps its old name."""
    import types
    from PySide6.QtWidgets import QMessageBox
    from core.artifacts import StoreError
    from core.gui import fields as gui_fields
    from core.gui.panels.inference import posterior_tab as post_mod, prior_tab as prior_mod
    from core.gui.screens.inference_screen import InferenceScreen
    from core.gui.session import SbiSession
    from tests._fixtures import SHOWN, PaneCapture, qt_app

    qt_app()
    inf = InferenceScreen()
    inf.session = SbiSession(cfg=object(), inf_prior=_prior_stub(), posterior=_posterior_stub())

    def _taken(kind, ref, new_name):
        raise StoreError(f"a {kind} named {new_name!r} already exists; rename or delete it first",
                         field="name")

    def _bug(kind, ref, new_name):
        raise PermissionError("manifest.json is locked")

    for kind, mod, panel, name_box, save, loaded in (
            ("prior", prior_mod, inf.prior_panel, inf.prior_panel.prior_name,
             inf.prior_panel._save_prior, inf.session.inf_prior),
            ("posterior", post_mod, inf.posterior_panel, inf.posterior_panel.post_name,
             inf.posterior_panel._save_posterior, inf.session.posterior)):
        pane = PaneCapture(panel)
        name_box.setText("p")

        monkeypatch.setattr(mod, "default_store", lambda: types.SimpleNamespace(rename=_taken))
        SHOWN.clear()
        save()
        box = SHOWN[-1]
        assert box.windowTitle() == "Check your inputs" and box.icon() == QMessageBox.Warning, kind
        assert box.text() == f"a {kind} named 'p' already exists; rename or delete it first", kind
        assert box.informativeText() == gui_fields.fix_sentence("name") == \
            "Choose another name in the Save box.", kind
        assert box.detailedText() == "", kind
        assert loaded.name == "", "a refused rename must not relabel the loaded artifact"
        assert pane.lines[-1][0] == "warning" and "already exists" in pane.lines[-1][1], pane.lines

        monkeypatch.setattr(mod, "default_store", lambda: types.SimpleNamespace(rename=_bug))
        SHOWN.clear()
        save()
        box = SHOWN[-1]
        assert box.windowTitle() == "Error" and box.icon() == QMessageBox.Critical, kind
        assert box.text() == "manifest.json is locked", kind
        assert "Traceback" in box.detailedText() and "PermissionError" in box.detailedText(), kind
        assert loaded.name == "", kind
        assert pane.lines[-1] == ("error", "manifest.json is locked"), pane.lines


def test_the_inference_tabs_route_builder_failures_by_kind_and_no_longer_call_config_error(monkeypatch,
                                                                                            tmp_path):
    """"Build / Load prior" builds the SimConfig first. A Refusal from that builder (a bounds file
    the parser will not take, a units declaration that does not parse) opens the yellow box with the
    core's sentence as its TEXT -- not as the informative line under a generic "The configuration
    could not be built." -- and a bug in the builder opens the red box with its traceback. Nothing is
    installed on the session and nothing is dispatched either way.

    And a source pin, because the routing is a rule for all five inference tabs: none of
    core/gui/panels/inference/*.py calls _config_error any more. That method stays on BasePanel for
    the Simulate, Reduction, CrossVal and FDT panels until piece 5 retires it."""
    import types
    from PySide6.QtWidgets import QMessageBox
    import core.gui.panels.inference as inference_pkg
    from core.gui import fields as gui_fields
    from core.gui.screens.inference_screen import InferenceScreen
    from core.gui.session import SbiSession
    from core.refusals import Refusal
    from tests._fixtures import SHOWN, PaneCapture, qt_app

    qt_app()
    inf = InferenceScreen()
    pp = inf.prior_panel
    reached = []
    inf.install_config = lambda cfg: reached.append(("installed", cfg))
    pp.dispatch = lambda fn, *a, **k: reached.append(("dispatched", fn))
    pp.bounds_picker.selected_path = lambda: "Resources/Bounds/nadrowski/master.txt"
    pane = PaneCapture(pp)

    def _refusing(bounds_path=None, *, bounds_dicts=None):
        raise Refusal("The units are not parseable: 'furlong' is not a length unit.", field="units")

    def _buggy(bounds_path=None, *, bounds_dicts=None):
        raise ZeroDivisionError("division by zero")

    inf.session = SbiSession(draft=types.SimpleNamespace(make_config=_refusing))
    SHOWN.clear()
    pp._build_prior()
    box = SHOWN[-1]
    assert box.windowTitle() == "Check your inputs" and box.icon() == QMessageBox.Warning
    assert box.text() == "The units are not parseable: 'furlong' is not a length unit."
    assert box.informativeText() == gui_fields.fix_sentence("units")
    assert "'Units'" in box.informativeText(), box.informativeText()
    assert box.detailedText() == "" and reached == []
    assert pane.lines[-1][0] == "warning" and "'Units'" in pane.lines[-1][1], pane.lines

    inf.session = SbiSession(draft=types.SimpleNamespace(make_config=_buggy))
    SHOWN.clear()
    pp._build_prior()
    box = SHOWN[-1]
    assert box.windowTitle() == "Error" and box.icon() == QMessageBox.Critical
    assert box.text() == "division by zero" and "ZeroDivisionError" in box.detailedText()
    assert reached == [] and pane.lines[-1] == ("error", "division by zero")

    # A bounds file that vanished between the picker's refresh and the click, through the REAL
    # builder: cli.make_sim_config refuses it by the input kind (§3.3), so it is the yellow box naming
    # the Bounds picker -- not the parser's FileNotFoundError in the red one.
    from core import cli, registry
    from core.config import VALID_LABELS, VALID_MODELS
    gone = str(tmp_path / "gone.txt")
    pp.bounds_picker.selected_path = lambda: gone

    def _real(bounds_path=None, *, bounds_dicts=None):
        return cli.make_sim_config("NADROWSKI", VALID_LABELS[VALID_MODELS.index("NADROWSKI")],
                                   registry.state_dep_drift("NADROWSKI"), bounds_path,
                                   bounds_dicts=bounds_dicts)

    inf.session = SbiSession(draft=types.SimpleNamespace(make_config=_real))
    SHOWN.clear()
    pp._build_prior()
    box = SHOWN[-1]
    assert box.windowTitle() == "Check your inputs" and box.icon() == QMessageBox.Warning, box.text()
    assert box.text() == f"The bounds file was not found: {gone!r}."
    assert box.informativeText() == gui_fields.fix_sentence("bounds") and reached == []

    for path in sorted(Path(inference_pkg.__file__).parent.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Attribute) and n.attr == "_config_error"]
        assert not calls, f"{path.name} still routes a failure through _config_error"


def test_the_secondary_panels_still_show_a_bad_cell_as_check_your_inputs(monkeypatch, tmp_path):
    """The Simulate, Reduction, CrossVal and FDT panels are piece 5's. Until then their builder
    failures stay on BasePanel._config_error, whose box is titled "Check your inputs" and reads "The
    configuration could not be built." over the builder's own sentence. Pinned for BOTH shapes the
    builders raise across piece 3 -- the bare ValueError they raise today for a cell missing a
    parameter, and the Refusal(field="cell") they raise once cli is converted -- so the four panels
    keep the same box whichever lands first, and nothing is dispatched. The routing change on the
    inference tabs must not leak here."""
    from PySide6.QtWidgets import QMessageBox
    from core import cli
    from core.gui.panels import simulate_panel as sim_mod
    from core.gui.panels.crossval_panel import CrossValPanel
    from core.gui.panels.fdt_panel import FdtPanel
    from core.gui.panels.reduction_panel import ReductionPanel
    from core.gui.panels.simulate_panel import SimulatePanel
    from core.refusals import Refusal
    from tests._fixtures import SHOWN, qt_app

    qt_app()
    cell = tmp_path / "bad_cell.txt"
    cell.write_text("# a cell that will not build\n", encoding="utf-8")
    panels = {"fdt": FdtPanel(), "reduction": ReductionPanel(), "crossval": CrossValPanel(),
              "simulate": SimulatePanel()}
    for p in panels.values():
        p.cell_picker.selected_path = lambda: str(cell)
        p.dispatch = lambda *a, **k: pytest.fail("a panel dispatched with a config that did not build")
    clicks = {"fdt": panels["fdt"]._run, "reduction": panels["reduction"]._run,
              "crossval": panels["crossval"]._run, "simulate": panels["simulate"]._start}
    msg = f"Cell file {cell.name!r} does not define the parameter 'k_gs'."

    for make in (ValueError, lambda m: Refusal(m, field="cell")):
        def _bad(*a, **k):
            raise make(msg)

        monkeypatch.setattr(cli, "make_fdt_config", _bad)
        monkeypatch.setattr(cli, "make_reduction_config", _bad)
        monkeypatch.setattr(cli, "make_param_sweep_config", _bad)
        monkeypatch.setattr(sim_mod, "build_stream_config", _bad)
        for name, click in clicks.items():
            SHOWN.clear()
            click()
            box = SHOWN[-1]
            assert box.windowTitle() == "Check your inputs" and box.icon() == QMessageBox.Warning, name
            assert box.text() == "The configuration could not be built.", name
            assert box.informativeText() == msg, name
            assert box.detailedText() == "", name

def test_the_chi_drive_and_band_are_read_only_and_the_draft_carries_config():
    """V5 §5.2. The χ drive amplitude and band are MEASUREMENTS (config.py:541-573), not per-run
    choices: since D11 any other value is refused by build_prior seconds after Apply, so a box that
    accepts one only manufactures that refusal. The boxes stay as displays of config.py under a
    caption saying so; the draft carries None for both, so make_sim_config takes config.py's values
    exactly as the command-line tool does; and the "Model applied" line reports config.py, not the
    boxes. The help and the config.py comment stop inviting the edit.

    The proof is not the read-only flag (setText bypasses it, and so would a restore) but the draft:
    even after a programmatic write into the boxes, the config built from the draft passes the chi
    guard that would refuse any other band or amplitude.
    """
    from pathlib import Path
    from core import config
    from core.config import BOUNDS_PATH
    from core.gui.panels.inference.help_text import HELP
    from core.gui.screens.inference_screen import InferenceScreen
    from core.SBI.run_guards import _assert_chi_config_is_deliberate
    from tests._fixtures import PaneCapture, qt_app

    qt_app()
    screen = InferenceScreen()
    cfgp = screen.config_panel
    cfgp.model_combo.setCurrentText("NADROWSKI")
    cfgp._on_model_changed("NADROWSKI")
    cfgp.units_toggle.set_direct(False)
    for box in (cfgp.chi_f0, cfgp.chi_range.lo, cfgp.chi_range.hi):
        assert box.isReadOnly(), "the drive amplitude and band are displays of config.py"
    assert "config.py" in cfgp.chi_fixed_note.text(), cfgp.chi_fixed_note.text()
    assert cfgp.chi_f0.value() == config.CHI_F0
    assert cfgp.chi_range.value() == tuple(config.CHI_FREQ_BOUNDS)

    cap = PaneCapture(cfgp)
    cfgp.chi_check.setChecked(True)
    cfgp.chi_f0.setText("0.05")                          # a programmatic write: the draft must ignore it
    cfgp.chi_range.lo.setText("0.1")
    cfgp.chi_range.hi.setText("10")
    cfgp._build_config()
    draft = screen.session.draft
    assert draft is not None and draft.chi_mode is True
    assert draft.chi_f0 is None and draft.chi_freq_bounds is None, (draft.chi_f0, draft.chi_freq_bounds)
    lo, hi = config.CHI_FREQ_BOUNDS
    applied = [text for _level, text in cap.lines if text.startswith("Model applied")]
    assert len(applied) == 1, cap.lines
    assert f"over {lo:g}–{hi:g}×Ω₀ at ND amplitude {config.CHI_F0:g}" in applied[0], applied[0]

    bounds = BOUNDS_PATH / "nadrowski" / "master.txt"
    if bounds.exists():
        cfg = draft.make_config(str(bounds))
        assert cfg.chi_f0 == config.CHI_F0 and tuple(cfg.chi_freq_bounds) == tuple(config.CHI_FREQ_BOUNDS)
        _assert_chi_config_is_deliberate(cfg)             # raises on any other band or amplitude

    # The three texts that used to invite the edit now say the values are fixed by measurement.
    for key in ("chi_f0", "chi_range"):
        assert "config.py" in HELP[key] and "measurement" in HELP[key].lower(), (key, HELP[key])
    src = Path(config.__file__).read_text(encoding="utf-8")
    assert "TUNABLE per config in the Config tab" not in src, "config.py's CHI_F0 comment is stale"

def test_the_config_tab_refuses_bad_boxes_at_the_click_and_dispatches_nothing():
    """V2/V3 at Apply. Every bad box is a Refusal naming its field, shown through _refusal (the yellow
    box), and the session is left exactly as it was: no new draft, no re-gate. A BLANK box is refused,
    never read as zero -- IntField.value() gave 0 for "" and the old chain refused it only because
    0 < 2, while the pad was not checked here at all and surfaced one tab later as "The
    configuration could not be built." with a traceback. With χ mode off the three χ boxes are not
    read (they are disabled) and Apply goes through with None for them, so make_sim_config takes
    config.py's values, as the tool does."""
    from core import config
    from core.config import CHI_K_MAX
    from core.gui.screens.inference_screen import InferenceScreen
    from core.refusals import Refusal
    from tests._fixtures import qt_app

    qt_app()
    screen = InferenceScreen()
    cfgp = screen.config_panel
    cfgp.model_combo.setCurrentText("NADROWSKI")
    cfgp._on_model_changed("NADROWSKI")
    cfgp.units_toggle.set_direct(False)
    refused, drafts = [], []
    cfgp._refusal = lambda exc: refused.append(exc)      # the yellow box, recorded instead of shown
    screen.new_draft = lambda draft: drafts.append(draft)

    def apply(**boxes):
        """Apply with χ on, every χ box at config.py's value except the overrides; the new refusals."""
        cfgp.chi_check.setChecked(True)
        cfgp.chi_k.setText(str(config.CHI_N_FREQS))
        cfgp.chi_pad.setText(str(config.CHI_K_PAD))
        cfgp.chi_cycles.setText(str(config.CHI_MAX_CYCLES))
        for name, text in boxes.items():
            getattr(cfgp, name).setText(text)
        before = len(refused)
        cfgp._build_config()
        return refused[before:]

    cases = [
        ({"chi_k": ""}, "chi_n_freqs", "is blank"),
        ({"chi_k": "1"}, "chi_n_freqs", "at least 2"),
        ({"chi_k": str(CHI_K_MAX + 1)}, "chi_n_freqs", f"at most {CHI_K_MAX}"),
        ({"chi_k": "5", "chi_pad": "4"}, "chi_n_freqs", "slots"),
        ({"chi_pad": ""}, "chi_k_pad", "is blank"),
        ({"chi_pad": "1"}, "chi_k_pad", "at least 2"),
        ({"chi_pad": str(CHI_K_MAX + 1)}, "chi_k_pad", f"at most {CHI_K_MAX}"),
        ({"chi_cycles": ""}, "chi_max_cycles", "is blank"),
        ({"chi_cycles": str(config.CHI_MIN_CYCLES)}, "chi_max_cycles",
         f"greater than {config.CHI_MIN_CYCLES:g}"),
    ]
    for boxes, field, needle in cases:
        got = apply(**boxes)
        assert len(got) == 1 and isinstance(got[0], Refusal), (boxes, got)
        assert got[0].field == field and needle in str(got[0]), (boxes, str(got[0]))
        assert "(default " in str(got[0]), f"every message carries the default: {got[0]}"
    assert drafts == [], "a refused Apply must not touch the session"

    # Typed units: blank, then unusable. Both name "units"; neither names a box or a tab.
    cfgp.units_toggle.set_direct(True)
    cfgp.units_text.setText("")
    got = apply()
    assert len(got) == 1 and got[0].field == "units" and "blank" in str(got[0]), got
    cfgp.units_text.setText("notaunit")
    got = apply()
    assert len(got) == 1 and got[0].field == "units" and "notaunit" in str(got[0]), got
    assert drafts == []
    cfgp.units_toggle.set_direct(False)

    # χ mode OFF: the three boxes are not read. Blank them all and Apply goes through with None.
    n_refused = len(refused)
    cfgp.chi_check.setChecked(False)
    for box in (cfgp.chi_k, cfgp.chi_pad, cfgp.chi_cycles):
        box.setText("")
    cfgp._build_config()
    assert len(refused) == n_refused and len(drafts) == 1, (refused[n_refused:], drafts)
    off = drafts[0]
    assert off.chi_mode is False
    assert off.chi_n_freqs is None and off.chi_k_pad is None and off.chi_max_cycles is None

    # The happy path with χ on: the values travel typed, and the amplitude and band travel as None.
    assert apply() == []
    on = drafts[1]
    assert (on.chi_n_freqs, on.chi_k_pad, on.chi_max_cycles) == (
        config.CHI_N_FREQS, config.CHI_K_PAD, config.CHI_MAX_CYCLES)
    assert isinstance(on.chi_n_freqs, int) and isinstance(on.chi_k_pad, int)
    assert isinstance(on.chi_max_cycles, float)
    assert on.chi_f0 is None and on.chi_freq_bounds is None


def test_the_prior_tab_refuses_bad_boxes_at_the_click_and_dispatches_nothing():
    """V2 on the Prior tab. A blank, half-typed or out-of-rule knob box is REFUSED at the click --
    the yellow box, through _refusal -- and nothing is dispatched, the config is not built and the
    session's downstream is not reset. Until piece 3 the tab clamped (max(2, cluster_size.value()))
    and defaulted (walk_step or config.PRIOR_SWEEP_STEP), so a blank "Min cluster size" silently
    built a prior with a floor of 2 that nobody typed, and a blank "Random-walk step" one with the
    default that nobody chose.

    Three more things the same click must get right, because the stage does (spec §1.2, "a stage
    with a load branch"): a LOAD click reads none of the seven knobs, so a blank box does not stop
    it and no knob is forwarded (the stage resolves None to its default); a typed 0 in "Candidates
    per round" is the automatic value, not a blank; and the live sweep note under the boxes renders
    a bad box as "<label> is blank." / "<label> must be <rule>." from the SAME rule the click runs,
    so the note and the refusal can never name different limits."""
    from core import orchestrator
    from core.gui.fields import label
    from core.gui.screens.inference_screen import InferenceScreen
    from core.gui.session import SbiSession
    from core.refusals import Refusal
    from tests._fixtures import qt_app

    qt_app()
    inf = InferenceScreen()
    cfg = _spont_cfg()
    inf.session = SbiSession(draft=object(), cfg=None, inf_prior=_prior_stub())
    inf.session.draft = type("D", (), {"make_config": lambda self, **kw: cfg})()
    pp = inf.prior_panel
    pp.bounds_source.set_direct(False)
    pp.bounds_picker.combo.clear()
    pp.bounds_picker.combo.addItem("master.txt", userData="master.txt")
    pp.prior_picker.selected = lambda: (None, True)           # "(from scratch)": the BUILD branch
    sent, refused = [], []
    pp.dispatch = lambda fn, *a, **k: sent.append((fn, a, k))
    pp._refusal = lambda e: refused.append(e)

    def click():
        sent.clear()
        refused.clear()
        pp._build_prior()

    # (a) a blank box: refused with its field named; nothing dispatched, nothing built, nothing reset
    pp.cluster_size.setText("")
    click()
    assert sent == [], "a blank box must be refused at the click, not clamped and dispatched"
    assert isinstance(refused[-1], Refusal) and refused[-1].field == "min_cluster_size", refused
    assert "is blank" in refused[-1].message and "default" in refused[-1].message, refused[-1].message
    assert inf.session.cfg is None, "a refused click must not build the config"
    assert inf.session.inf_prior is not None, "a refused click must not reset the session's prior"

    # (b) half-typed and out-of-rule boxes, one per rule, each naming its field and its limit
    pp.cluster_size.setText("50")
    for attr, text, key, words in (("cluster_size", "-", "min_cluster_size", "is blank"),
                                   ("cluster_size", "1", "min_cluster_size", "must be at least 2"),
                                   ("cluster_samples", "0", "min_samples", "must be at least 1"),
                                   ("sweep_iters", "0", "num_iterations", "must be at least 1"),
                                   ("sweep_batch", "-1", "sweep_batch", "must be at least 0"),
                                   ("sweep_max_sets", "0", "max_sets", "must be at least 1"),
                                   ("sweep_step", "0", "walk_step", "must be greater than 0"),
                                   ("sweep_step", "", "walk_step", "is blank"),
                                   ("sweep_units", "-5", "stability_units", "must be greater than 0")):
        field = getattr(pp, attr)
        keep = field.text()
        field.setText(text)
        click()
        assert sent == [] and refused and refused[-1].field == key and words in refused[-1].message, \
            (attr, text, refused[-1].message if refused else None, sent)
        field.setText(keep)

    # (c) every box in rule: dispatched on the build branch with the seven knobs as numbers, and a
    # typed 0 in "Candidates per round" travels as 0 (= automatic), not as a refusal
    pp.sweep_iters.setText("3")
    pp.sweep_batch.setText("0")
    pp.sweep_max_sets.setText("40")
    pp.sweep_step.setText("0.02")
    pp.sweep_units.setText("250")
    pp.cluster_size.setText("9")
    pp.cluster_samples.setText("4")
    click()
    assert refused == [], refused
    fn, args, kw = sent[-1]
    assert fn is orchestrator.build_prior and args[0] is cfg and args[1] is None and args[2] is True
    assert (kw["num_iterations"], kw["sweep_batch"], kw["max_sets"]) == (3, 0, 40), kw
    assert (kw["walk_step"], kw["stability_units"]) == (0.02, 250.0), kw
    assert (kw["min_cluster_size"], kw["min_samples"]) == (9, 4), kw
    assert inf.session.cfg is cfg, "the build click installs the config"

    # (d) a LOAD click with a blank knob box still dispatches, and forwards no knob at all
    pp.prior_picker.selected = lambda: ("abc123def456", False)
    pp.sweep_iters.setText("")
    pp.cluster_size.setText("-")
    click()
    assert refused == [], "the load branch never reads the knobs, so a blank one must not refuse"
    fn, args, kw = sent[-1]
    assert fn is orchestrator.build_prior and args[1] == "abc123def456" and args[2] is False
    knob_keys = {"num_iterations", "sweep_batch", "max_sets", "walk_step", "stability_units",
                 "min_cluster_size", "min_samples"}
    assert not (knob_keys & set(kw)), f"a load must not forward the knobs: {sorted(knob_keys & set(kw))}"
    assert kw["provide_fig_sink"] is True and kw["on_result"] == pp._on_prior

    # (e) the live note: a bad box renders as the label and the rule -- nothing computed, nothing
    # raised, no dialog -- and a good set renders the census line again
    pp.sweep_iters.setText("")
    assert pp.sweep_note.text() == f"{label('num_iterations')} is blank."
    pp.sweep_iters.setText("0")
    assert pp.sweep_note.text() == f"{label('num_iterations')} must be at least 1."
    pp.sweep_iters.setText("3")
    pp.sweep_units.setText("-")
    assert pp.sweep_note.text() == f"{label('stability_units')} is blank."
    pp.sweep_units.setText("0")
    assert pp.sweep_note.text() == f"{label('stability_units')} must be greater than 0."
    pp.sweep_units.setText("250")
    pp.sweep_max_sets.setText("0")
    assert pp.sweep_note.text() == f"{label('max_sets')} must be at least 1."
    pp.sweep_max_sets.setText("40")
    pp.sweep_batch.setText("-1")
    assert pp.sweep_note.text() == f"{label('sweep_batch')} must be at least 0."
    pp.sweep_batch.setText("0")
    note = pp.sweep_note.text()
    assert note.startswith("Global census screens") and "(3 rounds x " in note and "40 sets" in note, note

    # (f) the note's words are the click's: for every box the note reads, its "must be" clause is a
    # substring of the refusal the click raises for the same value
    from core.gui.panels.inference.prior_tab import _KNOB_RULES, _NOTE_KEYS, _rule_words
    pp.prior_picker.selected = lambda: (None, True)
    for key, attr, minimum in _KNOB_RULES:
        if key not in _NOTE_KEYS:
            continue
        field = getattr(pp, attr)
        keep = field.text()
        field.setText("-1")
        assert pp.sweep_note.text() == f"{label(key)} must be {_rule_words(minimum)}.", (key, pp.sweep_note.text())
        click()
        assert refused[-1].field == key and f"must be {_rule_words(minimum)}" in refused[-1].message, \
            (key, refused[-1].message)
        field.setText(keep)


def test_the_prior_tab_rows_are_labelled_from_the_control_table():
    """Every registered field's row on the Prior tab takes its label from core.gui.fields.label(key),
    so the name in the yellow box ("Set it in the '<label>' box on the Prior tab.") and the name on
    the tab are ONE string (spec §3.2). A literal at an add_help_row call is the defect: the row can
    be renamed while the refusal keeps naming the old label. Task 15's read-back pin checks the
    rendered QLabels over every tab; this is the source-level half for this tab."""
    import re
    from core.gui.fields import CONTROL, label
    from core.gui.panels.inference.prior_tab import PriorPanel
    from tests._fixtures import code_only

    src = code_only(PriorPanel.__init__)
    assert src.count("add_help_row(") == 9, "the Prior tab has nine labelled rows"
    literal = re.findall(r"add_help_row\(\w+, (['\"][^'\"]*['\"])", src)
    assert literal == [], f"rows labelled by a literal instead of label(key): {literal}"
    keys = re.findall(r"add_help_row\(\w+, label\(['\"](\w+)['\"]\)", src)
    assert keys == ["bounds", "prior", "num_iterations", "sweep_batch", "max_sets", "walk_step",
                    "stability_units", "min_cluster_size", "min_samples"], keys
    for key in keys:
        assert CONTROL[key] == ("Prior", label(key)), (key, CONTROL[key])


def test_the_posterior_tab_refuses_bad_boxes_at_the_click_and_dispatches_nothing():
    """V2 at the Posterior tab's click. Every knob the TRAIN branch reads is validated before any
    worker starts, through the shared rules, and the refusal carries the box's field key so the
    yellow box can say where to fix it. Three shapes used to pass silently: a blank box read as 0
    and was clamped to 1 (`max(1, ...)`), a 0 in a `... or default` box became the default, and a
    negative Fisher step sailed through. Each is now a refusal, and NOTHING is dispatched.

    The LOAD branch is the other half of the rule, and the half that cuts the other way: a load
    reads the picker and the accept dialog, never the knob boxes, so a blank Hidden features box
    must not block loading a posterior -- and the knobs are not forwarded at all (the stage resolves
    None to its default and its load branch never reads them). Decided as the stage decides it:
    is_new from the picker mirrors `trains = train_new or ref is None`."""
    from core.artifacts import Accept
    from core.gui.screens.inference_screen import InferenceScreen
    from core.gui.session import SbiSession
    from core.refusals import Refusal
    from tests._fixtures import qt_app

    qt_app()
    inf = InferenceScreen()
    inf.session = SbiSession(cfg=object(), inf_prior=_prior_stub())
    pp = inf.posterior_panel
    pp.post_picker.selected = lambda: ("", True)              # "(from scratch)" -> the TRAIN branch
    pp._confirm_fresh_run = lambda cfg, n_runs, cap: (True, False)
    sent, refused = [], []
    pp.dispatch = lambda fn, *a, **k: sent.append(k)
    pp._refusal = lambda exc: refused.append(exc)

    def click(box, text):
        """One click with `box` holding `text`; the box is put back afterwards."""
        good = box.text()
        box.setText(text)
        sent.clear()
        refused.clear()
        pp._build_posterior()
        box.setText(good)
        return list(sent), list(refused)

    cases = [
        (pp.num_runs, "", "num_runs", "is blank"),
        (pp.num_runs, "0", "num_runs", "at least 1"),
        (pp.run_size_cap, "", "run_size_cap", "is blank"),
        (pp.run_size_cap, "-5", "run_size_cap", "at least 0"),
        (pp.flow_hidden, "", "hidden_features", "is blank"),
        (pp.flow_hidden, "0", "hidden_features", "at least 1"),
        (pp.flow_transforms, "0", "num_transforms", "at least 1"),
        (pp.flow_lr, "", "learning_rate", "is blank"),
        (pp.flow_lr, "0", "learning_rate", "greater than 0"),
        (pp.flow_patience, "0", "stop_after_epochs", "at least 1"),
        (pp.fisher_m, "0", "fisher_m", "at least 1"),
        (pp.fisher_dz, "", "fisher_dz", "is blank"),
        (pp.fisher_dz, "-0.01", "fisher_dz", "greater than 0"),
        (pp.fisher_points, "0", "fisher_points", "at least 1"),
    ]
    for box, text, field, rule in cases:
        got_sent, got_refused = click(box, text)
        assert got_sent == [], f"{field}={text!r} was dispatched: {got_sent}"
        assert len(got_refused) == 1 and isinstance(got_refused[0], Refusal), (field, got_refused)
        assert got_refused[0].field == field, (field, got_refused[0].field, str(got_refused[0]))
        assert rule in str(got_refused[0]), (field, str(got_refused[0]))

    # A typed 0 in the cap box is "automatic", not blank: the one 0 this tab accepts.
    got_sent, got_refused = click(pp.run_size_cap, "0")
    assert got_refused == [] and len(got_sent) == 1 and got_sent[0]["run_size_cap"] == 0, \
        (got_refused, got_sent)

    # The LOAD branch: a blank knob box does not block a load, and no knob is forwarded.
    pp.post_picker.selected = lambda: ("p1", False)
    pp._accept_for_load = lambda entry: Accept()
    pp.flow_hidden.setText("")
    pp.fisher_dz.setText("")
    sent.clear()
    refused.clear()
    pp._build_posterior()
    assert refused == [] and len(sent) == 1, ([str(e) for e in refused], sent)
    forwarded = set(sent[0]) & {"num_runs", "run_size_cap", "hidden_features", "num_transforms",
                                "learning_rate", "stop_after_epochs", "fisher_m", "fisher_dz",
                                "fisher_points"}
    assert forwarded == set(), f"a load forwarded knobs the load branch never reads: {forwarded}"
    assert sent[0]["new_run"] is False and isinstance(sent[0]["accept"], Accept), sent[0]


def test_the_validate_tab_refuses_bad_boxes_at_the_click_and_dispatches_nothing():
    """V2 on the Validate tab. Both boxes were clamped with max(1, ...) at the click, so a blank or a 0
    became a 1 nobody typed -- and for the operating points a 1 is a DIFFERENT measurement, not a
    smaller one: cal_n_scales is t_scale's effective sample size (trap X5), which is why the stage now
    refuses it instead of clamping (spec 3.3). A refused box is the yellow box (stubbed here as
    _refusal) naming the box, the Validate tab and the default, and NOTHING is dispatched; a typed
    value in rule reaches validate_calibration as an int, and 1 operating point is allowed -- it is a
    choice, not a typo."""
    from core import config, orchestrator
    from core.gui import fields as gui_fields
    from core.gui.screens.inference_screen import InferenceScreen
    from core.gui.session import SbiSession
    from core.refusals import Refusal
    from tests._fixtures import qt_app

    qt_app()
    inf = InferenceScreen()
    inf.session = SbiSession(cfg=object(), inf_prior=_prior_stub(), posterior=_posterior_stub())
    vp = inf.validate_panel
    sent, refused = {}, []
    vp.dispatch = lambda fn, *a, **k: sent.update(fn=fn, args=a, kwargs=k)
    vp._refusal = lambda exc: refused.append(exc)
    vp._on_error = lambda exc, tb: pytest.fail(f"a refused box reached _on_error: {exc!r}")

    def click(cal_n, cal_scales):
        vp.cal_n.setText(cal_n)
        vp.cal_scales.setText(cal_scales)
        sent.clear()
        refused.clear()
        vp._validate()

    for cal_n, cal_scales, field, rule in (("", "200", "n_cal", "is blank"),
                                          ("0", "200", "n_cal", "must be at least 1; got 0"),
                                          ("2000", "", "cal_n_scales", "is blank"),
                                          ("2000", "0", "cal_n_scales", "must be at least 1; got 0"),
                                          ("2000", "-3", "cal_n_scales", "must be at least 1; got -3")):
        click(cal_n, cal_scales)
        assert sent == {}, (
            f"a calibration was dispatched with cal_n={cal_n!r} cal_scales={cal_scales!r}: {sent}")
        assert len(refused) == 1 and isinstance(refused[0], Refusal), refused
        e = refused[0]
        assert e.field == field and rule in str(e), (e.field, str(e))
        default = config.SBC_N_CAL if field == "n_cal" else config.CAL_N_SCALES
        assert f"(default {default})" in str(e), str(e)
        fix = gui_fields.fix_sentence(e.field)
        assert gui_fields.label(field) in fix and "Validate" in fix, fix

    click("2000", "1")
    assert refused == [] and sent["fn"] is orchestrator.validate_calibration, (refused, sent)
    assert sent["kwargs"]["n_cal"] == 2000 and sent["kwargs"]["cal_n_scales"] == 1, sent["kwargs"]


def test_the_tsnpe_tab_refuses_bad_boxes_at_the_click_and_dispatches_nothing(monkeypatch):
    """V2 on the TSNPE tab, and walkthrough row C3 offscreen. A blank HPD box used to reach the stage
    as 0.0 (FloatField.value()) and a blank direction box as 0, and the stage refused each with a
    sentence naming neither the box nor the default; the batch count was clamped to 1 and a blank
    rows-per-batch became a 0 nobody typed. Now the click reads every box through the shared rules
    BEFORE the observation is re-hashed (a refused box must cost nothing), shows a refusal as the
    yellow box (stubbed here as _refusal) naming the box, the TSNPE tab and the default, and dispatches
    nothing. A TYPED 0 in the rows-per-batch box still means automatic. The latent-width ceiling on
    the direction count stays the STAGE's -- the click has no posterior to measure it against -- so an
    oversized count is dispatched, and tsnpe_round refuses it with the width and the default in the
    sentence (T7's pin)."""
    import types
    from core import config, orchestrator
    from core.gui import fields as gui_fields
    from core.gui.panels.inference import tsnpe_tab as tt
    from core.gui.screens.inference_screen import InferenceScreen
    from core.gui.session import SbiSession
    from core.refusals import Refusal
    from core.SBI import truncate as _tr
    from tests._fixtures import qt_app

    qt_app()
    inf = InferenceScreen()
    inf.session = SbiSession(cfg=object(), inf_prior=_prior_stub(), posterior=_posterior_stub())
    panel = inf.tsnpe_panel
    panel.obs_picker.key = lambda: "20260910T120000"

    def _never(cfg, ref):
        raise AssertionError("the observation was loaded before the boxes were read")

    monkeypatch.setattr(tt, "default_store", lambda: types.SimpleNamespace(load_observation=_never))
    sent, refused = {}, []
    panel.dispatch = lambda fn, *a, **k: sent.update(fn=fn, args=a, kwargs=k)
    panel._refusal = lambda exc: refused.append(exc)
    panel._on_error = lambda exc, tb: pytest.fail(f"a refused box reached _on_error: {exc!r}")

    def click(**boxes):
        panel.n_dirs.setText(str(_tr.DEFAULT_N_DIRECTIONS))
        panel.hpd.setText(str(_tr.DEFAULT_HPD))
        panel.num_runs.setText("3")
        panel.run_size_cap.setText("8")
        for name, text in boxes.items():
            getattr(panel, name).setText(text)
        sent.clear()
        refused.clear()
        panel._round()

    for boxes, field, rule, default in (
            ({"num_runs": "0"}, "num_runs", "must be at least 1; got 0", config.TRAINING_NUM_RUNS),
            ({"num_runs": ""}, "num_runs", "is blank", config.TRAINING_NUM_RUNS),
            ({"run_size_cap": ""}, "run_size_cap", "is blank", config.TRAINING_RUN_SIZE),
            ({"hpd": ""}, "hpd_level", "is blank", _tr.DEFAULT_HPD),
            ({"hpd": "1"}, "hpd_level", "between 0 and 1", _tr.DEFAULT_HPD),
            ({"n_dirs": ""}, "n_directions", "is blank", _tr.DEFAULT_N_DIRECTIONS),
            ({"n_dirs": "0"}, "n_directions", "must be at least 1; got 0", _tr.DEFAULT_N_DIRECTIONS)):
        click(**boxes)
        assert sent == {}, f"a round was dispatched with {boxes}: {sent}"
        assert len(refused) == 1 and isinstance(refused[0], Refusal), (boxes, refused)
        e = refused[0]
        assert e.field == field and rule in str(e), (boxes, e.field, str(e))
        assert f"(default {default})" in str(e), str(e)
        fix = gui_fields.fix_sentence(e.field)
        assert gui_fields.label(field) in fix and "TSNPE" in fix, fix

    # in rule: the loaded observation and the read values travel; a typed 0 cap is automatic; the
    # direction count is NOT checked against a width the tab cannot know
    sentinel = types.SimpleNamespace(id="20260910T120000", name="obs")
    monkeypatch.setattr(tt, "default_store",
                        lambda: types.SimpleNamespace(load_observation=lambda cfg, ref: sentinel))
    click(run_size_cap="0", n_dirs="99")
    assert refused == [] and sent["fn"] is orchestrator.tsnpe_round, (refused, sent)
    k = sent["kwargs"]
    assert k["run_size_cap"] == 0, "a TYPED 0 is automatic; only a blank is refused"
    assert k["n_directions"] == 99, "the latent-width ceiling is the stage's; the click must not guess P"
    assert k["level"] == _tr.DEFAULT_HPD and k["num_runs"] == 3 and sent["args"][3] is sentinel, sent


def test_the_validate_and_tsnpe_rows_are_named_from_the_control_table():
    """The rows' names are the CONTROL table's, through label(key), so the refusal's "Set it in the
    '<label>' box on the <tab> tab" and the row on screen can never disagree: renaming a control is
    one edit in core/gui/fields.py, and T15's read-back pins the rendered text. A literal here would be
    a second copy of the name -- the kind that goes stale the day the first one moves."""
    from core.gui import fields as gui_fields
    from core.gui.panels.inference.tsnpe_tab import TSNPEPanel
    from core.gui.panels.inference.validate_tab import ValidatePanel
    from tests._fixtures import code_only

    for panel, keys in ((ValidatePanel, ("n_cal", "cal_n_scales")),
                        (TSNPEPanel, ("observation", "hpd_level", "n_directions", "num_runs",
                                      "run_size_cap"))):
        src = code_only(panel.__init__)
        for key in keys:
            assert f"label({key!r})" in src, (
                f"{panel.__name__}.__init__ does not build its {key!r} row from label({key!r})")
            assert repr(gui_fields.label(key)) not in src, (
                f"{panel.__name__}.__init__ still names its {key!r} row by the literal "
                f"{gui_fields.label(key)!r}")


def test_a_blank_t_obs_is_refused_on_all_three_infer_branches(tmp_path):
    """V2 on the three observation-length boxes. Today FloatField.value() turns a blank or half-typed
    box into 0.0 and every branch forwards it: the simulated one spends the simulation and then dies
    in math.log(0.0) (statistics.py), the bench ones build a zero-length observation. Now the click
    reads the box through value_or_none() and require_positive("t_obs", ...): a blank, a lone "-" and
    a typed 0 are each refused through _refusal (the yellow box) with field "t_obs", and NOTHING is
    dispatched. One box per page: sim_tobs, exp_tobs (the passive and driven branches share it) and
    chi_tobs. Every file the click also checks is a real (empty) file, so the refusal seen is the
    box's and not a missing file's; the dispatch is stubbed, so nothing reads them."""
    from core import orchestrator
    from core.refusals import Refusal
    from core.gui.screens.inference_screen import InferenceScreen
    from tests._fixtures import qt_app

    qt_app()
    inf = InferenceScreen()
    inf.session.posterior = _posterior_stub()
    inf.session.inf_prior = _prior_stub()
    panel = inf.infer_panel
    sent, refused = {}, []
    panel.dispatch = lambda fn, *a, **k: sent.update(fn=fn, args=a, kwargs=k)
    panel._refusal = lambda exc: refused.append(exc)
    files = {}
    for stem in ("cell.txt", "passive.npy", "forced.npy", "probe0.npy", "probe1.npy"):
        files[stem] = tmp_path / stem
        files[stem].touch()

    def click(box, text):
        sent.clear()
        refused.clear()
        box.setText(text)
        panel._infer()

    def refused_t_obs(box):
        for text, why in (("", "is blank"), ("-", "is blank"), ("0", "greater than 0")):
            click(box, text)
            assert sent == {}, (text, sent)
            assert len(refused) == 1 and isinstance(refused[0], Refusal), (text, refused)
            assert refused[0].field == "t_obs" and why in refused[0].message, (text, refused[0].message)

    # simulated: sim_tobs
    inf.install_config(_forced_cfg())
    panel.infer_mode.setCurrentIndex(0)
    panel.cell_source.set_direct(False)
    panel.cell_picker.selected_path = lambda: str(files["cell.txt"])
    panel._cell_problems = []
    refused_t_obs(panel.sim_tobs)
    click(panel.sim_tobs, "3.5")
    assert refused == [] and sent["fn"] is orchestrator.simulated_inference and sent["args"][2] == 3.5

    # driven: exp_tobs, with every drive box and both recordings given
    panel.infer_mode.setCurrentIndex(1)
    panel.exp_spont.edit.setText(str(files["passive.npy"]))
    panel.exp_forced.edit.setText(str(files["forced.npy"]))
    for name, text in (("amp", "1.5"), ("freq", "5"), ("phase", "0"), ("offset", "0")):
        panel._forcing_fields[name].setText(text)
    refused_t_obs(panel.exp_tobs)
    click(panel.exp_tobs, "2.0")
    assert refused == [] and sent["fn"] is orchestrator.experimental_inference
    assert sent["args"][2].T_obs_s == 2.0 and sent["args"][2].forced == ((str(files["forced.npy"]), None),)

    # passive: the same exp_tobs box on the no-drive branch
    inf.install_config(_forced_cfg(names=()))
    refused_t_obs(panel.exp_tobs)
    click(panel.exp_tobs, "2.0")
    assert refused == [] and sent["fn"] is orchestrator.experimental_inference
    assert sent["args"][2].forced == () and sent["args"][2].forcing_params_si is None

    # chi: chi_tobs, with the passive recording, F0 and two complete probe rows given
    inf.install_config(_chi_cfg(k=2))
    panel.chi_spont.edit.setText(str(files["passive.npy"]))
    panel.chi_f0_si.setText("1.0")
    for i, row in enumerate(panel._chi_forced_fields):
        row.path.edit.setText(str(files[f"probe{i}.npy"]))
        row.freq.setText(str(10.0 + i))
    refused_t_obs(panel.chi_tobs)
    click(panel.chi_tobs, "2.0")
    assert refused == [] and sent["fn"] is orchestrator.experimental_inference
    assert sent["args"][2].F0_si == 1.0 and sent["args"][2].T_obs_s == 2.0


def test_the_driven_branch_refuses_a_zero_drive_and_a_missing_recording(tmp_path):
    """The driven page's boxes, at the click. A zero amplitude is the passive branch's job and a 0 Hz
    drive would reach the lock-in, so both are refused (require_positive); a blank phase is a blank
    (require_finite); each recording is refused by its own role -- "recording_spont" or
    "recording_forced" -- whether blank or pointing at a file that is not there (require_file).
    Today none of this is checked: FloatField(0.0)'s "0.0" dispatched a zero drive, and a wrong path
    surfaced as a FileNotFoundError inside the worker's red box. The offset row is a forcing name
    outside the registry: it is read as a finite number of either sign under its own row label, with
    no field key to point at. When every box passes, the recording set carries the SI drive keyed by
    forcing NAME (what build_experiment_obs takes) and the forced file with no frequency."""
    from core import orchestrator
    from core.refusals import Refusal
    from core.gui.screens.inference_screen import InferenceScreen
    from tests._fixtures import qt_app

    qt_app()
    inf = InferenceScreen()
    inf.install_config(_forced_cfg())
    inf.session.posterior = _posterior_stub()
    inf.session.inf_prior = _prior_stub()
    panel = inf.infer_panel
    sent, refused = {}, []
    panel.dispatch = lambda fn, *a, **k: sent.update(fn=fn, args=a, kwargs=k)
    panel._refusal = lambda exc: refused.append(exc)
    spont, forced = tmp_path / "spont.npy", tmp_path / "forced.npy"
    spont.touch()
    forced.touch()
    panel.infer_mode.setCurrentIndex(1)
    panel.exp_tobs.setText("2.0")
    panel.exp_spont.edit.setText(str(spont))
    panel.exp_forced.edit.setText(str(forced))
    amp, freq, phase, offset = (panel._forcing_fields[n] for n in ("amp", "freq", "phase", "offset"))
    for box, text in ((amp, "1.5"), (freq, "5"), (phase, "0"), (offset, "-3")):
        box.setText(text)

    def click():
        sent.clear()
        refused.clear()
        panel._infer()
        assert len(refused) <= 1, refused
        return refused[0] if refused else None

    def refuse(edit, text, field, why):
        keep = edit.text()
        edit.setText(text)
        e = click()
        assert sent == {}, (text, sent)
        assert isinstance(e, Refusal) and e.field == field and why in e.message, (text, e)
        edit.setText(keep)

    refuse(amp, "0", "drive_amplitude", "greater than 0")
    refuse(amp, "", "drive_amplitude", "is blank")
    refuse(freq, "0", "drive_frequency", "greater than 0")
    refuse(freq, "-1", "drive_frequency", "greater than 0")
    refuse(phase, "", "drive_phase", "is blank")
    refuse(offset, "", None, "offset (N) is blank")
    refuse(panel.exp_forced.edit, str(tmp_path / "gone.npy"), "recording_forced", "gone.npy")
    refuse(panel.exp_forced.edit, "", "recording_forced", "is blank")
    refuse(panel.exp_spont.edit, str(tmp_path / "gone.npy"), "recording_spont", "was not found")
    refuse(panel.exp_spont.edit, "", "recording_spont", "is blank")

    # every box given: dispatched once, with the drive by NAME and the forced file with no frequency
    assert click() is None
    assert sent["fn"] is orchestrator.experimental_inference, sent
    rec = sent["args"][2]
    assert rec.spont == str(spont) and rec.forced == ((str(forced), None),) and rec.T_obs_s == 2.0
    assert rec.forcing_params_si == {"amp": 1.5, "freq": 5.0, "phase": 0.0, "offset": -3.0}
    assert sent["kwargs"]["provide_fig_sink"] is True and sent["kwargs"]["accept"] is None


def test_the_simulated_branch_refuses_a_missing_or_misfitting_cell_at_the_click(tmp_path):
    """The simulated branch's Cell row, at the click, through _refusal with field "cell": no cell
    picked or a picked file that is not there (require_file), a picked cell the pick-time check found
    does not fit the bounds (a warning line today, a refusal now), and in direct entry a values grid
    with problems. Today the first dispatched a path the worker then failed on, and the other two
    were warning lines with no dialog. The bounds-fit check itself is the pick-time one
    (_on_cell_changed) and is unchanged; the grid's own validation is pinned at its widget tests."""
    from core.refusals import Refusal
    from core.gui.screens.inference_screen import InferenceScreen
    from tests._fixtures import qt_app

    qt_app()
    inf = InferenceScreen()
    inf.install_config(_forced_cfg())
    inf.session.posterior = _posterior_stub()
    panel = inf.infer_panel
    sent, refused = {}, []
    panel.dispatch = lambda fn, *a, **k: sent.update(fn=fn, args=a, kwargs=k)
    panel._refusal = lambda exc: refused.append(exc)
    panel.infer_mode.setCurrentIndex(0)
    panel.sim_tobs.setText("3.5")

    def click() -> str:
        sent.clear()
        refused.clear()
        panel._infer()
        assert sent == {}, sent
        assert len(refused) == 1 and isinstance(refused[0], Refusal), refused
        assert refused[0].field == "cell", refused[0]
        return refused[0].message

    panel.cell_source.set_direct(False)
    panel.cell_picker.selected_path = lambda: None
    panel._cell_problems = []
    assert "is blank" in click()
    panel.cell_picker.selected_path = lambda: str(tmp_path / "gone.txt")
    assert "was not found" in click()
    cell = tmp_path / "cell.txt"
    cell.touch()
    panel.cell_picker.selected_path = lambda: str(cell)
    panel._cell_problems = ["k = 9 is outside its bounds"]
    msg = click()
    assert "does not fit the bounds file" in msg and "k = 9" in msg, msg
    # direct entry: the grid's problems, verbatim, under the same key
    panel._cell_problems = []
    panel.cell_source.is_direct = lambda: True
    panel.values_grid.problems = lambda: ["k: must be a number"]
    assert click() == "Fix the values first: k: must be a number"


def test_the_probe_planner_refuses_a_blank_t_obs_through_the_yellow_box(tmp_path):
    """"Plan probes…" is a click, so it is refused like one (V2): a blank or non-positive T_obs and a
    blank or missing passive recording go to _refusal with their field, BEFORE the recording is
    loaded. Today a blank T_obs read as 0.0 and the planner ran on a one-sample window, and a blank
    recording was a warning line. A complete click passes the rules and goes on to load the
    recording; the stub config has no hardware, so on this test's path that load fails inside the
    planner's own guarded step, and its "Could not measure Ω₀" error line is the evidence the rules
    ran first and let the click through."""
    from core.refusals import Refusal
    from core.gui.screens.inference_screen import InferenceScreen
    from tests._fixtures import qt_app

    qt_app()
    inf = InferenceScreen()
    inf.install_config(_chi_cfg(k=2))
    panel = inf.infer_panel
    refused, lines = [], []
    panel._refusal = lambda exc: refused.append(exc)
    panel.log_pane.append_line = lambda text, kind="": lines.append((kind, text))
    passive = tmp_path / "passive.npy"
    passive.touch()

    def plan():
        refused.clear()
        lines.clear()
        panel._plan_chi_probes()

    panel.chi_spont.edit.setText(str(passive))
    panel.chi_tobs.setText("")
    plan()
    assert len(refused) == 1 and isinstance(refused[0], Refusal) and refused[0].field == "t_obs", refused
    assert "is blank" in refused[0].message and lines == [], lines
    panel.chi_tobs.setText("0")
    plan()
    assert refused[0].field == "t_obs" and "greater than 0" in refused[0].message and lines == []
    panel.chi_tobs.setText("4.5")
    panel.chi_spont.edit.setText("")
    plan()
    assert refused[0].field == "recording_spont" and "is blank" in refused[0].message and lines == []
    panel.chi_spont.edit.setText(str(tmp_path / "gone.npy"))
    plan()
    assert refused[0].field == "recording_spont" and "gone.npy" in refused[0].message and lines == []
    # a complete click passes the rules and reaches the load
    panel.chi_spont.edit.setText(str(passive))
    plan()
    assert refused == [], refused
    assert lines and lines[0][0] == "error" and lines[0][1].startswith("Could not measure Ω₀ from"), lines


def test_the_gui_control_table_matches_the_tabs_labels():
    """§3.2's pin. fields.CONTROL is where a refusal learns which box to name ("Set it in the
    'T_obs (s)' box on the Infer tab."), and the tabs build their rows FROM it (label(key)), so the two
    cannot drift -- this reads every tab's form rows back and checks that each (tab, label) entry is
    a label that tab actually shows. The rows are read the way Qt holds them: the QLabel inside each
    help_label holder (help_badge.py), or the plain QLabel of a row added without help text, compared
    against labels.pretty_gui(label), which is what the holder was given. The Infer tab is read after
    install_config with a FORCED stub config so its three drive rows exist; a chi config would build
    none (no force_params_dict) and the drive entries would pass vacuously.

    The second half is this task's own: the Infer tab's registered rows are built from label(key),
    not from a literal that happens to match today."""
    from PySide6.QtWidgets import QFormLayout, QLabel
    from core.gui import fields as gui_fields
    from core.gui.panels.inference.infer_tab import InferPanel
    from core.gui.screens.inference_screen import InferenceScreen
    from core.Helpers import labels
    from tests._fixtures import code_only, qt_app

    qt_app()
    inf = InferenceScreen()
    inf.install_config(_forced_cfg())
    tabs = {"Config": inf.config_panel, "Prior": inf.prior_panel, "Posterior": inf.posterior_panel,
            "Validate": inf.validate_panel, "Infer": inf.infer_panel, "TSNPE": inf.tsnpe_panel}

    def shown(panel) -> list:
        out = []
        for form in panel.findChildren(QFormLayout):
            for i in range(form.rowCount()):
                item = form.itemAt(i, QFormLayout.LabelRole)
                w = item.widget() if item is not None else None
                if w is None:
                    continue                      # a spanning row: a checkbox, a button, an anchor
                lab = w if isinstance(w, QLabel) else w.findChild(QLabel)
                if lab is not None:
                    out.append(lab.text())
        return out

    seen = {tab: shown(panel) for tab, panel in tabs.items()}
    tuples = {k: e for k, e in gui_fields.CONTROL.items() if isinstance(e, tuple)}
    assert tuples, "the control table has no (tab, label) entries"
    for key in ("drive_amplitude", "drive_frequency", "drive_phase"):
        assert key in tuples, f"{key} must be a (tab, label) entry so the read-back covers the drive rows"
    missing = []
    for key, (tab, text) in tuples.items():
        for name in (tab if isinstance(tab, tuple) else (tab,)):   # the budget boxes name two tabs
            assert name in seen, f"{key}: CONTROL names a tab that does not exist: {name!r}"
            if labels.pretty_gui(text) not in seen[name]:
                missing.append((key, name, text))
    assert not missing, f"CONTROL names labels no tab shows: {missing}\nshown: {seen}"

    src = code_only(InferPanel.__init__)
    for key in ("t_obs", "cell", "recording_forced", "chi_f0_si"):
        assert f"label({key!r})" in src, f"the Infer tab must build its {key} row from label({key!r})"
    for literal in ("'T_obs (s)'", "'Drive F₀ (N)'"):
        assert literal not in src, f"a literal row label {literal} survives in InferPanel.__init__"


def _pick_simulated(panel, cell: str, t_obs: str) -> None:
    """Put the Infer tab on its simulated page with a picked cell file and a typed T_obs. The picker is
    pointed at the file directly, as test_the_infer_tab_dispatches_the_compositions does: what these
    pins are about is the session config, not the picker's folder listing."""
    panel.infer_mode.setCurrentIndex(0)
    panel.cell_source.set_direct(False)
    panel.cell_picker.selected_path = lambda: cell
    panel._cell_problems = []
    panel.sim_tobs.setText(t_obs)


def _wait_for_run(app, panel, limit: float = 300.0) -> None:
    """Spin the event loop until the panel's run has finished, then deliver the queued signals.
    `finished` is emitted after `result`/`error`, and `_set_busy(False)` hangs off it, so once `_busy`
    drops the outcome has already been handled on this thread."""
    import time
    t0 = time.monotonic()
    while panel._busy and time.monotonic() - t0 < limit:
        app.processEvents()
        time.sleep(0.01)
    assert not panel._busy, f"the run did not finish within {limit:.0f} s"
    for _ in range(30):
        app.processEvents()
        time.sleep(0.01)


def test_a_failed_inference_leaves_the_session_config_pristine(screen_run, monkeypatch):
    """V1 at the window, on the FAILURE path: the composition injects the cell's truth, records the
    cell and sets T_obs on the config it holds, then the run dies inside generate_observations. Until
    piece 3 all three writes stayed on the session, so the NEXT training anchored its Fisher rotation on
    that cell and the next manifest named it as an input. The error reaches the red box (a bug, not a
    Refusal), and the session's config is untouched.

    (a) makes the pin non-vacuous: with SimConfig.copy_for_run switched off (public_entry then hands the
    composition the caller's own object), the same click leaves the truth on a throwaway session
    config -- the leak this test exists to catch."""
    from PySide6.QtWidgets import QMessageBox
    from core import orchestrator
    from core.sim_config import SimConfig
    from tests._fixtures import SHOWN, assert_cfg_unchanged, qt_app, snapshot_cfg

    app = qt_app()
    inf = screen_run.screen
    panel = inf.infer_panel
    pristine = inf.session.cfg

    def boom(*_a, **_kw):
        raise RuntimeError("stopped after the truth was injected")

    monkeypatch.setattr(orchestrator, "generate_observations", boom)
    _pick_simulated(panel, screen_run.cell, "1.5")

    # (a) the copy switched off: the leak happens, on a throwaway config
    probe = pristine.copy_for_run()
    inf.session.cfg = probe
    try:
        with monkeypatch.context() as m:
            m.setattr(SimConfig, "copy_for_run", lambda self: self)
            panel._infer()
            _wait_for_run(app, panel)
    finally:
        inf.session.cfg = pristine
    assert probe.has_ground_truth and "cell" in probe.sources, \
        "with the copy off the composition must write onto the session config; the pin below is vacuous"
    SHOWN.clear()

    # (b) the copy on: the same failure leaves nothing behind
    snap = snapshot_cfg(pristine)
    panel._infer()
    _wait_for_run(app, panel)
    assert_cfg_unchanged(pristine, snap)
    assert not pristine.has_ground_truth and "cell" not in pristine.sources
    assert len(SHOWN) == 1 and SHOWN[-1].icon() == QMessageBox.Critical, \
        [(b.windowTitle(), b.text()) for b in SHOWN]


def test_a_dispatched_inference_leaves_the_session_config_pristine(screen_run):
    """V1 at the window, on the SUCCESS path, with nothing stubbed: a real simulated inference on the
    tiny posterior, dispatched by this tab through the worker, writes its observation and inference
    and hands them to the session -- and leaves the session's config exactly as install_config set
    it: no truth, no cell among its sources, the T_obs it had (1.0 s from the fixture) although the
    run was typed 1.5 s. The box value differs from the session's on purpose, so the T_obs clause is
    not satisfied by coincidence."""
    from tests._fixtures import SHOWN, assert_cfg_unchanged, qt_app, snapshot_cfg

    app = qt_app()
    inf = screen_run.screen
    panel = inf.infer_panel
    cfg = inf.session.cfg
    _pick_simulated(panel, screen_run.cell, "1.5")
    before_obs, before_t_obs = inf.session.observation, cfg.T_obs
    snap = snapshot_cfg(cfg)

    panel._infer()
    _wait_for_run(app, panel)

    assert SHOWN == [], [(b.windowTitle(), b.text()) for b in SHOWN]
    assert inf.session.observation is not None and inf.session.observation is not before_obs, \
        "the run did not complete: _on_observation never installed an observation"
    assert_cfg_unchanged(cfg, snap)
    assert not cfg.has_ground_truth and "cell" not in cfg.sources
    assert cfg.T_obs == before_t_obs
