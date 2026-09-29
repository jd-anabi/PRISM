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
from PySide6.QtCore import Qt as _Qt                              # noqa: E402

_USER_ROLE = _Qt.UserRole

def _prior_stub(id_="p1", name=""):
    """A LoadedPrior-shaped stub: the tabs now read ``.prior``/``.force_prior`` off session.inf_prior
    rather than carrying the physical prior and forcing prior as separate session fields, so a bare
    ``object()`` no longer stands in wherever a dispatched call is actually reached."""
    return type("LoadedPriorStub", (), {"prior": object(), "force_prior": object(),
                                         "id": id_, "name": name})()
def _posterior_stub(id_="post1", name="", truncation=None, x_obs_digest=None):
    """A LoadedPosterior-shaped stub: the tabs now read ``.posterior`` (the TransformedPosterior) off
    session.posterior rather than carrying it directly, so a bare ``object()`` no longer stands in
    wherever a dispatched call actually reaches ``session.posterior.posterior``."""
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

    def boom(cfg, *, skip_sanity, confirm_production, writer, seed=None):
        raise FDTModelError("Observable 'x' has state-dependent (multiplicative) noise; FDT supports "
                            "additive-noise observables only.")

    def missing(cfg, *, skip_sanity, confirm_production, writer, seed=None):
        raise KeyError("k_gs")

    monkeypatch.setattr(fdt_panel, "run_fdt", boom)
    with pytest.raises(FDTModelError) as e:
        fdt_panel._run_fdt_guarded(Cfg(), skip_sanity=True, confirm_production=False, writer=None)
    assert isinstance(e.value, Refusal), "an FDTModelError must reach the worker as the Refusal it is"
    assert "multiplicative" in str(e.value), str(e.value)

    monkeypatch.setattr(fdt_panel, "run_fdt", missing)
    with pytest.raises(RuntimeError) as e:
        fdt_panel._run_fdt_guarded(Cfg(), skip_sanity=True, confirm_production=False, writer=None)
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

def test_dispatch_watches_every_directory_it_is_given(tmp_path):
    """The sweep study writes TWO records, so its figures land in two separate
    ``figures/`` directories -- and NewPngWatcher globs ONE directory and does not recurse
    (core/gui/plot_watcher.py: ``self._dir.glob("*.png")``). One watcher would therefore show the S
    sweep's plot and silently lose the T sweep's, which is the half of a long study you waited
    longest for.

    A single path still means one watcher, because every other caller passes one."""
    from pathlib import Path
    from core.gui.panels.base_panel import BasePanel
    from tests._fixtures import qt_app

    app = qt_app()
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(); b.mkdir()
    panel = BasePanel()
    seen = []
    panel.figure_stack.add_png = lambda title, path: seen.append((title, Path(path).name))
    panel.dispatch(lambda: None, watch_dir=[a, b])
    (a / "fdt3d_vs_S_20260922_120000.png").write_bytes(b"s")
    (b / "fdt3d_vs_T_20260922_130000.png").write_bytes(b"t")
    # `finished` is QUEUED to this thread; `_finished` stops BOTH watchers and each stop() does the
    # final scan that ignores the settle delay. Wait for it as the module's other dispatch tests do,
    # so BasePanel._running is False again before the next test dispatches anything.
    _wait_for_run(app, panel, limit=30.0)
    assert sorted(n for _t, n in seen) == ["fdt3d_vs_S_20260922_120000.png",
                                           "fdt3d_vs_T_20260922_130000.png"], seen

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
    """⚠ The narrowed-model rule -- a narrowed posterior is marked as such and never loads or infers
    as a broad one -- at the seam where it is easiest to lose.

    Every posterior is now WRITTEN at completion, so there is no deferred-save window for a region to
    fall out of on the way to disk -- but the SESSION must still carry the right LoadedPosterior after
    a TSNPE round, after its Save (a rename, which must not touch the body), and after an ordinary
    train replaces it. Three things, and the third is the one that is easy to miss: the round
    installs a NON-AMORTIZED posterior, the Save renames it in place, and training an ordinary
    posterior afterwards installs an AMORTIZED one.
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
    """⚠ The calibrate-on-the-region rule's GUI half, plus the question the Posterior tab asks before
    it loads a stored narrowed posterior. A non-amortized artifact is loaded through the Posterior
    tab, which now ASKS first and opts in with ``Accept(truncated=True)`` only on a yes; the
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
    """The probe table has variable rows. The core has always accepted 1..chi_k_pad probes at arbitrary
    frequencies; the GUI was the only thing forcing a fixed grid. Rows must be addable and removable,
    and the cap must be chi_k_pad -- which is FROZEN into the trained artifact, so exceeding it is not
    a soft limit."""
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

def test_chi_probe_table_survives_a_config_rebuild_and_rejects_a_blank_frequency(tmp_path):
    """Two constraints on the probe table's variable rows in one place, because both are about data
    the GUI cannot regenerate.

    PRESERVATION: rows carry hand-typed drive frequencies and browsed paths -- a record of a bench
    session that already happened. Rebuilding the config (to fix a bounds file, say) must not discard
    them, unlike the forcing rows, which ARE derivable from the config.

    BLANK FREQUENCY: a SEEDED row is blank and is said to be blank. It used to arrive holding "0.0"
    -- FloatField's own default -- and be refused as "must be a positive number (got 0)", a sentence
    about a value nobody entered, while 0 Hz is a genuine DC probe the lock-in would happily attempt.
    A TYPED zero keeps that sentence, because a zero somebody typed is a different state from a box
    nobody filled.
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
    """One state had two wordings a user could meet minutes apart: the tab's ``probe N: drive
    frequency is blank`` at the click, and "Plan probes…"'s ``no frequency entered``. The tab's is the
    one, which also makes the planner's own "Filled N blank frequency box(es)" line literally true.
    The TYPED-ZERO sentences are deliberately untouched at both layers -- the tab's "(got 0)" and
    chi.probe_verdict's "must be finite and positive, got 0.0 Hz".

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


def test_the_planner_fills_a_blank_frequency_box_and_never_a_typed_zero():
    """A typed zero is a DIFFERENT STATE from a box nobody filled, and the row's own ``problems()``
    keeps them apart -- but the planner's auto-fill selected the rows to overwrite with
    ``_probe_frequency(row) is None``, which maps a NON-POSITIVE box to None too. So "Plan probes…"
    silently replaced a typed ``0`` with a suggested grid frequency and counted it among the "blank
    boxes filled", while the comment directly above it reads "a typed frequency is a record of what
    the bench actually did".

    That was invisible while every seeded row held ``0``; making the seed blank is what turned it
    into a user-visible divergence -- the two layers now disagreed about the same box. The predicate
    asks the blank-aware accessor instead, so a typed zero is left alone and refused as a zero, as
    the row's own ``problems()`` refuses it.

    Tested on the predicate rather than through ``_plan_chi_probes``, which needs a built config and
    a measured Ω₀ from a real recording: what changed is which rows are SELECTED."""
    from core.gui.panels.inference.infer_tab import _blank_frequency
    from core.gui.panels.inference.rows import _ChiProbeRow
    from tests._fixtures import qt_app

    qt_app()
    blank = _ChiProbeRow(lambda _row: None)                     # a seeded row: an empty box
    typed_zero = _ChiProbeRow(lambda _row: None, 0.0)           # a DC probe, or a typo for 10
    real = _ChiProbeRow(lambda _row: None, 42.0)
    half_typed = _ChiProbeRow(lambda _row: None)
    half_typed.freq.setText("-")                                # mid-typing: parses as nothing

    assert _blank_frequency(blank) is True
    assert _blank_frequency(half_typed) is True, "a box that parses as nothing is still unfilled"
    assert _blank_frequency(typed_zero) is False, \
        "a typed zero is a record of what the bench did and must not be overwritten"
    assert _blank_frequency(real) is False
    assert typed_zero.freq.value_or_none() == 0.0, "the box still holds what was typed"

    # ... and the row still REFUSES that zero, instead of a silent fill.
    assert "probe 1: drive frequency must be a positive number (got 0)" in typed_zero.problems(0)


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

    The GUI runner this used to exercise is gone: the tab dispatches orchestrator.simulated_inference
    directly, so the wiring under test is the composition's."""
    import types
    import torch
    from core import cli, orchestrator

    qt_app()

    class Cfg:
        length_unit = "nm"                       # the trace figure's y-axis unit
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
        # T_obs=0.1s is below T_MIN_EXP_S on purpose: record the resulting PreflightWarning instead
        # of leaking it -- `match=` would re-emit any warning that does not match, and this call is
        # not asserted to emit exactly one.
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
    # recording before it dispatches, and the dispatch is stubbed so nothing reads them.
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

    # a NON-AMORTIZED posterior with the box ticked -- the consent travels on both branches
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
    from tests._fixtures import PaneCapture

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
    # single-process gate, silently disabling the near-miss check for every later test in the run.
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
    pane = PaneCapture(pp)              # both pane channels, as (level, text) -- the stub's own order
    lines = pane.lines
    pp._build_posterior()
    assert sent["kwargs"]["new_run"] is False
    assert any(k == "warning" and "unreadable header" in t for k, t in lines), lines


def test_the_fresh_cache_and_narrowed_posterior_dialogs_default_to_cancel(monkeypatch):
    """Enter on either dialog must do the SAFE thing, so Cancel is the default: on the load question
    the other button loads a NON-AMORTIZED posterior with Accept(truncated=True), and on the near-miss
    dialog it starts a fresh cache one setting away from a committed one, the accident the near-miss
    dialog exists to stop. Every other test replaces the _ask_* methods, so only this one sees the
    dialogs themselves."""
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
    """The near-miss consent is per-run. Persisting it would silence the near-miss refusal for every
    future round in every future session -- exactly the accident the refusal exists to catch."""
    from core.gui.panels.inference.tsnpe_tab import TSNPEPanel

    src = ast.unparse(ast.parse(textwrap.dedent(inspect.getsource(TSNPEPanel.save_settings))))
    assert "new_run" not in src, "save_settings must not persist the near-miss consent"
    src = ast.unparse(ast.parse(textwrap.dedent(inspect.getsource(TSNPEPanel.restore_settings))))
    assert "new_run" not in src, "restore_settings must not restore the near-miss consent"


def test_the_infer_tab_other_observation_box():
    """The narrowed-posterior consent on the Infer tab. The box is the GUI's only way to say "yes, run
    this TSNPE posterior on a different observation" -- and it must be UNREACHABLE for an amortized
    posterior, where it would mean nothing, and must never survive a change of posterior or a restart:
    a stale tick would silence the narrowed-model rule for a posterior the user never answered the
    question about."""
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
    """The window's half of the refusal contract. A Refusal names its field by a key and never a box,
    tab, button or flag; the yellow "Check your inputs" box gets its "how to fix here" line from ONE
    table, and the inference tabs build their static rows from the same table's labels (label(key)),
    so a control is named in one place and a renamed one cannot leave a stale sentence behind.

    (a) the table knows every registry key and no other -- an unmapped key would show a refusal
        with no way out, and an entry nobody raises is a sentence that can go stale unseen;
    (b) every tuple names a place: a tab title of any section, read off the built window, or one of
        SCREENS, a non-empty label, and renders the box sentence; a sentence entry is
        returned verbatim and has no label; a None entry says nothing and has no label; and none
        of the three ever raises from fix_sentence, which runs while a refusal is being shown;
    (c) the T_obs and direction-count sentences a user reads on the real screen, and the consents,
        verbatim -- the consents quote their controls' own text (posterior_tab.py:232,
        tsnpe_tab.py:102, infer_tab.py:150);
    (d) the keys with no window control are exactly the six the window never exposes, the eleven
        tool-only diagnostics knobs, and the five FDT settings neither front end exposes, so a
        tool-only key renders no window sentence;
    (e) the three drive labels are built exactly as the Infer tab builds its rows
        (_rebuild_forcing_fields: labels.gui_forcing_label with config.FORCING_DISPLAY_UNITS, which
        cli.INFERENCE_PROMPT_UNITS aliases), so the read-back over the built Infer tab can find them.
    """
    from core import config
    import core.gui.fields as gui_fields
    from core.Helpers import labels
    from core.refusals import FIELDS

    # (a) the same key set, in both directions
    assert set(gui_fields.CONTROL) == set(FIELDS), (
        f"only in CONTROL: {sorted(set(gui_fields.CONTROL) - set(FIELDS))}; "
        f"only in FIELDS: {sorted(set(FIELDS) - set(gui_fields.CONTROL))}")

    # (b) the three shapes, against the tab titles of EVERY section plus the two screens.
    # Read off the built window, not off a list written here: a renamed tab must fail this, and an
    # entry that names the FDT analysis tab is as real as one that names the Infer tab.
    qt_app()
    from core.gui.main_window import MainWindow
    window = MainWindow()
    tabs = []
    for section in (window.reduction_screen, window.fdt_screen, window.inference_screen,
                    window.simulate_screen):
        tabs += [section.tabs.tabText(i) for i in range(section.tabs.count())]
    assert tabs == ["NWK → Hopf reduction map", "FDT analysis", "Sweep study cross-validation",
                    "Config", "Prior", "Posterior", "Validate", "Infer", "TSNPE",
                    "Live simulation"]
    assert gui_fields.SCREENS == frozenset({"Artifacts", "Model Builder"}), \
        "a place is a tab title or one of these two screens, which host no tab widget"
    places = set(tabs) | set(gui_fields.SCREENS)
    for key, entry in gui_fields.CONTROL.items():
        if isinstance(entry, tuple):
            tab, text = entry
            names = tab if isinstance(tab, tuple) else (tab,)
            assert names and all(t in places for t in names), f"{key}: {tab!r} is not a place"
            assert isinstance(text, str) and text, f"{key}: empty label"
            assert gui_fields.label(key) == text
            # the place phrase through _where itself, pinned verbatim below (the mixed tab+screen case
            # included): one noun for the whole tuple would be a false red on the first mixed entry
            assert gui_fields.fix_sentence(key) == \
                f"Set it in the '{text}' box on {gui_fields._where(tab)}."
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

    # (c) verbatim: the sentences a user reads on the real screen, and the consents
    assert gui_fields.fix_sentence("t_obs") == \
        "Set it in the 'T_obs (s)' box on the Infer or Live simulation tab."
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
    assert gui_fields.fix_sentence("name") == (
        "Choose another name in the Save box, or in the 'Record name' or 'Comparison name' box "
        "on the FDT analysis or Sweep study cross-validation tab.")
    assert (gui_fields.fix_sentence("chi_f0") == gui_fields.fix_sentence("chi_freq_bounds")
            == "Fixed by measurement: change it in config.py, deliberately.")
    # the budget boxes sit on two tabs, and a refusal raised on either must name the one the user is on
    assert gui_fields.CONTROL["num_runs"] == (("Posterior", "TSNPE"), "Batches")
    assert gui_fields.fix_sentence("run_size_cap") == \
        "Set it in the 'Max rows per batch (0 = auto)' box on the Posterior or TSNPE tab."
    # An input that appears in several places lists them ALL. The cell picker is on five of
    # them, and a bad cell chosen on the measurement screen used to send the owner to the Infer tab.
    assert gui_fields.fix_sentence("cell") == (
        "Set it in the 'Cell' box on the Infer or FDT analysis or Sweep study cross-validation or "
        "NWK → Hopf reduction map or Live simulation tab.")
    assert gui_fields.fix_sentence("model") == (
        "Set it in the 'Model' box on the Config or FDT analysis or Live simulation tab.")
    assert gui_fields.fix_sentence("n_freqs") == (
        "Set it in the 'n_freqs' box on the FDT analysis or Sweep study cross-validation tab.")
    # the label is the word the row SHOWS, and the row renders it
    assert gui_fields.fix_sentence("ensemble_m") == (
        "Set it in the 'M_ensemble' box on the FDT analysis or Sweep study cross-validation tab.")
    assert labels.pretty_gui(gui_fields.label("ensemble_m")) == "M<sub>ensemble</sub>"
    assert gui_fields.label("freqs_per_batch") == "freqs / batch" == \
        labels.pretty_gui(gui_fields.label("freqs_per_batch"))
    assert gui_fields.label("s_grid") == "S grid  (T_a/T = 1)", \
        "a box on a SCREEN still has a label, so a row and its hint cannot drift apart"
    # the screen noun, on its own and beside a tab
    assert gui_fields._where(("Model Builder",)) == "the Model Builder screen"
    assert gui_fields._where(("Artifacts", "Model Builder")) == "the Artifacts or Model Builder screen"
    assert gui_fields._where(("Infer", "Model Builder")) == "the Infer tab or the Model Builder screen"
    assert gui_fields._where("Posterior") == "the Posterior tab"
    # the Artifacts screen's two, on a screen rather than a tab
    assert gui_fields.fix_sentence("artifact") == "Select an artifact in the list on the Artifacts screen."
    assert gui_fields.fix_sentence("note") == (
        "Edit it in the Note box on the Artifacts screen, or in the 'Note' or 'Comparison note' "
        "box on the FDT analysis or Sweep study cross-validation tab.")

    # (d) no window control: the six the window never exposes, and the tool-only set
    assert {k for k, e in gui_fields.CONTROL.items() if e is None} == {
        "checkpoint_every", "resume", "device", "n_samples", "num_posterior_samples", "max_num_epochs",
        "repeats", "n_points", "n_worst", "top_n", "m", "m_noise", "rel", "min_valid", "rows",
        "n_sweep", "chi_k_fixed",
        # the five FDT settings neither front end exposes
        "freq_bounds", "burn_in_nd", "t_obs_periods", "dt_nd", "psd_t_obs_nd"}

    # (e) the drive labels, as the Infer tab builds them
    for key, name in (("drive_amplitude", "amp"), ("drive_frequency", "freq"), ("drive_phase", "phase")):
        assert gui_fields.CONTROL[key] == (
            "Infer", labels.gui_forcing_label(name, config.FORCING_DISPLAY_UNITS[name])), key
    assert gui_fields.label("drive_amplitude") == "A (N)"
    assert gui_fields.label("drive_frequency") == "f (Hz)"
    assert gui_fields.label("drive_phase") == "φ (rad)"


def test_a_rename_failure_reads_as_a_name_refusal(monkeypatch):
    """Save on the Prior and Posterior tabs is a rename, and a bad or taken name is a StoreError -- a
    Refusal with field="name" -- so it opens the yellow box with the core's sentence as its text and
    this front end's fix ``fix_sentence("name")``, which starts "Choose another name in the Save box"
    and goes on to the record and comparison name boxes, under it. It used to go through
    _config_error, whose box read "The configuration could not be built." over a sentence about a
    name: a lie for a rename. A rename that fails for any other reason is a bug and stays red, with
    its traceback. Either way the loaded artifact keeps its old name."""
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
        assert box.informativeText() == gui_fields.fix_sentence("name") == (
            "Choose another name in the Save box, or in the 'Record name' or 'Comparison name' box "
            "on the FDT analysis or Sweep study cross-validation tab."), kind
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
    core/gui/panels/inference/*.py calls _config_error any more. That method was retired altogether
    once the four section panels were converted; the pin stays, because the scan is what stops a new
    tab reintroducing the call."""
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
    # builder: cli.make_sim_config refuses it by the input kind, so it is the yellow box naming the
    # Bounds picker -- not the parser's FileNotFoundError in the red one.
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


def test_the_four_secondary_panels_route_a_refusal_apart_from_a_bug(monkeypatch, tmp_path):
    """The FDT, CrossVal, Reduction and Simulate panels used to wrap their builder in a broad
    ``except`` ending at ``BasePanel._config_error``, whose box read "The configuration could not be
    built." over whatever sentence it had caught -- one box for a blank number and for a bug in the
    parser alike, and no way to tell which you were looking at. They now do what the six inference
    tabs already did: a ``Refusal`` opens the yellow "Check your inputs" box with the CORE's own
    sentence as its text and this front end's "where to fix it" under it, and anything else is a bug
    and keeps the red box with its traceback behind Details. Nothing is dispatched either way.

    This REPLACES test_the_secondary_panels_still_show_a_bad_cell_as_check_your_inputs, which pinned
    the unconverted behaviour across both exception shapes on purpose: a half-converted state cannot
    be expressed in it, so it is rewritten rather than extended.

    The Reduction panel's builder is stubbed like the other three, and the stub is enough: the real
    cli.make_reduction_config refuses through cli.parse_cell -- a cell missing a value the bounds file
    declares, and a legacy cell with no time unit -- and both are Refusal(field="cell"), so the stub's
    Refusal takes exactly their path through the panel, whose arm is chosen by the exception's type
    and never by its words. What the stub cannot stand in for is the one check the panel makes
    itself: its F0 box is read at the click, before the builder, because a blank box reads as 0 and
    make_reduction_config takes F0 as given. That case is asserted against a BLANK box and not only a
    typed zero.

    The source pin at the end is the point of the section: ``_config_error`` is gone from BasePanel
    and no module under core/gui names it, so no later panel can quietly route a failure back into a
    box that says nothing about what was wrong.
    """
    import ast
    from pathlib import Path
    from PySide6.QtWidgets import QMessageBox
    from core import cli
    from core.gui import fields as gui_fields
    from core.gui.panels import simulate_panel as sim_mod
    from core.gui.panels.base_panel import BasePanel
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
    msg = f"Cell file {cell.name!r} is missing value(s) the bounds file requires: k_gs."

    def _patch(raiser):
        monkeypatch.setattr(cli, "make_fdt_config", raiser)
        monkeypatch.setattr(cli, "make_reduction_config", raiser)
        monkeypatch.setattr(cli, "make_param_sweep_config", raiser)
        monkeypatch.setattr(sim_mod, "build_stream_config", raiser)

    # (a) a Refusal: the yellow box, the core's sentence, this front end's fix line, no traceback
    def _refusing(*a, **k):
        raise Refusal(msg, field="cell")

    _patch(_refusing)
    for name, click in clicks.items():
        SHOWN.clear()
        click()
        box = SHOWN[-1]
        assert box.windowTitle() == "Check your inputs" and box.icon() == QMessageBox.Warning, name
        assert box.text() == msg, name
        assert box.informativeText() == gui_fields.fix_sentence("cell"), name
        assert box.detailedText() == "", name

    # (b) anything else is a bug: the red box, the message, and the traceback behind Details
    def _booming(*a, **k):
        raise RuntimeError("the parser fell over")

    _patch(_booming)
    for name, click in clicks.items():
        SHOWN.clear()
        click()
        box = SHOWN[-1]
        assert box.windowTitle() == "Error" and box.icon() == QMessageBox.Critical, name
        assert box.text() == "the parser fell over", name
        assert "RuntimeError" in box.detailedText(), name

    # (c) the Reduction panel's F0 box, checked at the click and before the builder: a blank
    #     box -- which value() would have read as 0 -- and a typed 0 are both the yellow box naming
    #     the setting, and the builder is never reached
    monkeypatch.setattr(cli, "make_reduction_config",
                        lambda *a, **k: pytest.fail("the reduction builder ran on an F0 the panel refuses"))
    reduction = panels["reduction"]
    for typed, said in (
            ("", "The non-dimensional drive amplitude is blank (default 0.05)."),
            ("0", "The non-dimensional drive amplitude must be greater than 0; got 0 (default 0.05).")):
        reduction.f0.setText(typed)
        SHOWN.clear()
        reduction._run()
        box = SHOWN[-1]
        assert box.windowTitle() == "Check your inputs" and box.icon() == QMessageBox.Warning, typed
        assert box.text() == said, typed
        assert box.informativeText() == gui_fields.fix_sentence("f0"), typed
        assert "NWK → Hopf reduction map" in box.informativeText(), box.informativeText()

    # (d) a cell deleted after it was picked. The panel checks the file itself, beside the F0 check
    #     (make_reduction_config stays untouched), so it is one yellow box naming the cell -- not a
    #     FileNotFoundError out of the parser in the red one, the regression removing _config_error
    #     caused. The builder is never reached.
    gone = tmp_path / "deleted_after_the_pick.txt"
    reduction.cell_picker.selected_path = lambda: str(gone)
    reduction.f0.setText("0.05")
    monkeypatch.setattr(reduction, "_on_error", lambda *a, **k: pytest.fail("the red box opened"))
    SHOWN.clear()
    reduction._run()
    assert len(SHOWN) == 1, [b.text() for b in SHOWN]
    box = SHOWN[0]
    assert box.windowTitle() == "Check your inputs" and "was not found" in box.text(), box.text()
    assert box.informativeText() == gui_fields.fix_sentence("cell"), box.informativeText()

    # (e) the generic box is gone, and nothing under core/gui reaches for it
    assert not hasattr(BasePanel, "_config_error")
    gui_root = Path(sim_mod.__file__).resolve().parents[1]
    for path in sorted(gui_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        named = [n for n in ast.walk(tree)
                 if isinstance(n, ast.Attribute) and n.attr == "_config_error"]
        assert not named, f"{path.name} still routes a failure through _config_error"


def test_the_fdt_panel_leaves_an_unsupported_model_to_the_builders_refusal(monkeypatch, tmp_path):
    """The Run button is gated on registry.fdt_support when the model changes, but the registry can
    change under a selection: a user model re-saved with multiplicative noise while this tab still
    shows it. ``FdtPanel._run`` used to repeat the check as a "backstop" that wrote one warning line
    to the log pane and returned -- no box -- while cli.make_fdt_config refuses the same model with
    the same reason as Refusal(field="model"). The backstop is gone, so the builder's refusal is the
    ONE path to the operator: the yellow box, fdt_support's own sentence, and the fix line naming the
    Model box. Nothing is dispatched."""
    from PySide6.QtWidgets import QMessageBox
    from core import registry
    from core.gui import fields as gui_fields
    from core.gui.panels.fdt_panel import FdtPanel
    from tests._fixtures import SHOWN, qt_app

    qt_app()
    cell = tmp_path / "cell.txt"
    cell.write_text("# never parsed: the model is refused before the cell is read\n", encoding="utf-8")
    panel = FdtPanel()
    panel.cell_picker.selected_path = lambda: str(cell)
    panel.dispatch = lambda *a, **k: pytest.fail("an unsupported model was dispatched")
    reason = "FDT can't run 'NADROWSKI': a stand-in for fdt_support's per-model sentence."
    monkeypatch.setattr(registry, "fdt_support", lambda name: (False, reason))

    SHOWN.clear()
    panel._run()
    assert SHOWN, "an unsupported model reached the operator as a log line, not as the refusal box"
    box = SHOWN[-1]
    assert box.windowTitle() == "Check your inputs" and box.icon() == QMessageBox.Warning
    assert box.text() == reason
    assert box.informativeText() == gui_fields.fix_sentence("model")
    assert "FDT analysis" in box.informativeText(), box.informativeText()


def test_the_fdt_panel_offers_only_single_cell_records(monkeypatch):
    """One kind holds three studies, so the FDT screen's picker must filter to its own: a sweep
    record has no single-cell ratio curve to draw and a comparison is a picture of other records, and
    offering either would hand the panel's record viewer a body whose ``grid`` is null by design.

    The filter is the picker's row predicate, applied to the ``Summary`` rows, so nothing about the
    kind's other studies has to be known here.

    The tail pins the rows' labels: the seven registered rows are built from label(key), and the
    two literal ones -- "Record name" and "Note", whose keys are sentence entries with no label() --
    are read back off the form and must be the words the ``name`` and ``note`` fix sentences quote,
    because the control-table read-back skips sentence entries and nothing else would catch a drift."""
    import types
    from PySide6.QtWidgets import QFormLayout, QLabel
    from core.gui import fields as gui_fields
    from core.gui.panels.fdt_panel import FdtPanel
    from core.gui.widgets.artifact_picker import StorePicker
    from tests._fixtures import qt_app

    qt_app()
    rows = [types.SimpleNamespace(complete=True, finished=True, study=study, label=study, id=study,
                                  created="2026-09-22T12:00:00", mode=None, width=None,
                                  amortized=None)
            for study in ("single", "sweep", "comparison")]
    store = types.SimpleNamespace(list=lambda kind: list(rows) if kind == "fdt" else [])
    monkeypatch.setattr(StorePicker, "_resolved_store", lambda self: store)
    panel = FdtPanel()
    combo = panel.record_picker.combo
    assert [combo.itemData(i) for i in range(combo.count())] == ["single"], \
        [combo.itemText(i) for i in range(combo.count())]

    from tests._fixtures import code_only
    src = code_only(FdtPanel._build_controls)
    for key in ("model", "cell", "n_freqs", "ensemble_m", "freqs_per_batch", "f0", "seed"):
        assert f"label({key!r})" in src, f"the FDT panel must build its {key} row from label({key!r})"

    shown = []
    for form in panel.findChildren(QFormLayout):
        for i in range(form.rowCount()):
            item = form.itemAt(i, QFormLayout.LabelRole)
            w = item.widget() if item is not None else None
            lab = w if isinstance(w, QLabel) else (w.findChild(QLabel) if w is not None else None)
            if lab is not None:
                shown.append(lab.text())
    for key, text in (("name", "Record name"), ("note", "Note")):
        assert text in shown, f"the FDT panel shows no {text!r} row: {shown}"
        assert f"'{text}'" in gui_fields.fix_sentence(key), \
            f"the {key} fix sentence does not name the {text!r} box: {gui_fields.fix_sentence(key)!r}"


def test_the_fdt_panel_creates_the_record_before_it_dispatches(tmp_path, monkeypatch):
    """Forced by the ``log.txt`` every committed record carries. The panel must know the record's
    directory BEFORE the run starts, because that directory is what the figure watcher is pointed at
    -- and it must not enter the writer itself, because ``log.txt`` is written from
    ``runs.current_run_log()``, which is thread-local and is only populated inside ``capture_run()``
    on the WORKER thread. A writer entered on the window's thread would write no log at all.

    So: the front end CREATES (the id is minted and the name claimed, and nothing is on disk yet) and
    the stage ENTERS. This pins all four halves of that -- the writer travels as a keyword, the first
    body carries the facts the panel knows, the watch directory is the record's own ``figures/``, and
    ``create`` has written nothing, since ``__enter__`` is what does the mkdir.

    The Seed box is read ONCE and the same value reaches both halves: the builder (``cfg.seed``) and
    the run (the ``seed`` keyword). A BLANK box means no seed, and is the reason the box is read with
    value_or_none(): None reaches both halves -- value() would have sent a seed of 0 nobody typed --
    and the stage then draws one and records it. That last leg drives the dispatched call through the
    real run_fdt with only the measurement stubbed; the draw comes from a seeded Random patched in as
    the module's ``random``, never from reseeding Python's global stream (the one-process gate)."""
    import random
    from core.artifacts import ArtifactStore, use_store
    from core.FDT import fdt_pipeline
    from core.gui.panels.fdt_panel import FdtPanel, _run_fdt_guarded
    from tests._fixtures import qt_app

    qt_app()
    store = ArtifactStore(tmp_path)
    cap = {}
    with use_store(store):
        panel = FdtPanel()
        panel.dispatch = lambda fn, *a, **k: cap.update(fn=fn, args=a, kwargs=k)
        panel.seed.setText("4242")
        panel.record_name.setText("cell_a_first")
        panel.record_note.setText("the first look")
        panel._run()

    assert cap, "nothing was dispatched"
    assert cap["fn"] is _run_fdt_guarded, cap["fn"]
    writer = cap["kwargs"]["writer"]
    assert writer.kind == "fdt" and writer.name == "cell_a_first" and writer.note == "the first look"
    assert cap["kwargs"]["seed"] == 4242
    assert cap["args"][0].seed == 4242, "the builder's half: the dispatched config carries the seed"
    assert cap["kwargs"]["watch_dir"] == writer.dir / "figures", cap["kwargs"]["watch_dir"]
    assert not writer.dir.exists(), "create() must not touch the disk; __enter__ does the mkdir"
    assert writer.body["study"] == "single" and writer.body["seed"] == 4242
    assert writer.body["notices"] == [] and writer.body["complete"] is False
    assert all(writer.body[k] is None
               for k in ("grid", "points", "offgrid", "compared", "results")), writer.body

    # A BLANK Seed box: None to the builder and to the run, and the run draws one and records it
    cap.clear()
    with use_store(store):
        panel.seed.setText("")
        panel.record_name.setText("cell_a_drawn")
        panel._run()
    blank = cap["kwargs"]["writer"]
    assert cap["kwargs"]["seed"] is None, "a blank box must not reach the run as a seed of 0"
    assert cap["args"][0].seed is None, "a blank box must not reach the builder as a seed of 0"
    assert blank.body["seed"] is None, "the panel records what it knows; the stage records the draw"
    drawn = random.Random(4242).randrange(2 ** 31)
    monkeypatch.setattr(fdt_pipeline, "random", random.Random(4242))
    monkeypatch.setattr(fdt_pipeline, "_measure", lambda *a, **k: None)
    run_kwargs = {k: cap["kwargs"][k]
                  for k in ("writer", "seed", "skip_sanity", "confirm_production")}
    record = cap["fn"](*cap["args"], **run_kwargs)
    assert record.id == blank.id and record.body["seed"] == drawn, record.body["seed"]
    assert record.manifest.config["seed"] == drawn, "the config block names the drawn seed too"


def test_a_taken_fdt_record_name_is_refused_at_the_click(tmp_path):
    """A progressive record occupies its name from its first moment, and ``create`` runs
    ``assert_name_free``, so a second run started while an unfinished record of the same name sits on
    disk is refused BEFORE it spends anything -- rather than colliding hours later at a commit, which
    is how a finished run gets thrown away.

    The refusal is a StoreError, a Refusal subclass carrying ``field="name"``, so it belongs in the
    yellow "Check your inputs" box like every other input refusal on this screen. Raised out of the
    clicked slot instead, it would reach app.py's last-resort excepthook as a raw traceback with
    nothing in the panel's own log."""
    from PySide6.QtWidgets import QMessageBox
    from core.artifacts import ArtifactStore, use_store
    from core.gui.panels.fdt_panel import FdtPanel
    from tests._fixtures import SHOWN, qt_app

    qt_app()
    store = ArtifactStore(tmp_path)
    with use_store(store):
        taken = store.create("fdt", None, name="cell_a_first")
        taken.body = {"study": "single", "settings": {}, "seed": 1, "grid": None, "points": None,
                      "offgrid": None, "notices": [], "compared": None, "complete": False,
                      "results": None}
        taken.__enter__()                   # the record now exists, unfinished, holding its name

        panel = FdtPanel()
        panel.dispatch = lambda *a, **k: pytest.fail("a refused click dispatched a run anyway")
        panel.record_name.setText("cell_a_first")
        SHOWN.clear()
        panel._run()

    box = SHOWN[-1]
    assert box.windowTitle() == "Check your inputs" and box.icon() == QMessageBox.Warning
    assert "cell_a_first" in box.text(), box.text()
    assert box.detailedText() == "", "a refusal is not a crash and carries no traceback"
    assert len(list((tmp_path / "fdt").iterdir())) == 1, "the refused click left a second directory"


@pytest.mark.parametrize("key", ["n_freqs", "ensemble_m", "freqs_per_batch", "f0"])
@pytest.mark.parametrize("panel_name", ["fdt", "crossval"])
def test_a_blank_knob_box_is_refused_as_blank_on_both_measurement_tabs(tmp_path, monkeypatch,
                                                                       panel_name, key):
    """Both tabs read their four knob boxes with value(), which turns a blank into 0: the FDT tab
    then said "must be at least 1; got 0" about a value nobody typed, where the Reduction tab says the
    same key "is blank". And the CrossVal builder reads None as "use the preset", so reading the boxes
    with value_or_none() alone -- the "consistency" edit the grid rows invite -- would have run the
    preset's value in silence. The FDT tab hands the builder the blank (value_or_none), and the
    CrossVal tab refuses the blank itself before the builder (require_given): one yellow box, "is
    blank", the fix line naming the box, and nothing dispatched or minted. The fix line is read
    through ``fix_sentence``, never a copy of the box's label."""
    from PySide6.QtWidgets import QMessageBox
    from core.artifacts import ArtifactStore, use_store
    from core.gui import fields as gui_fields
    from core.gui.panels.crossval_panel import CrossValPanel
    from core.gui.panels.fdt_panel import FdtPanel
    from tests._fixtures import SHOWN, qt_app

    qt_app()
    store = ArtifactStore(tmp_path)
    monkeypatch.setattr(store, "create", lambda *a, **k: pytest.fail("a blank knob minted a record"))
    with use_store(store):
        panel = {"fdt": FdtPanel, "crossval": CrossValPanel}[panel_name]()
        panel.dispatch = lambda *a, **k: pytest.fail("a blank knob box dispatched a run anyway")
        assert panel.cell_picker.selected_path(), "the premise: a cell is selected"
        getattr(panel, key).setText("")
        SHOWN.clear()
        panel._run()

    assert len(SHOWN) == 1, [b.text() for b in SHOWN]
    box = SHOWN[0]
    assert box.windowTitle() == "Check your inputs" and box.icon() == QMessageBox.Warning
    assert "is blank" in box.text(), box.text()
    assert box.informativeText() == gui_fields.fix_sentence(key), box.informativeText()


@pytest.mark.parametrize("how", ["too_long", "two_lines"])
def test_an_fdt_record_note_is_judged_at_the_click(tmp_path, monkeypatch, how):
    """The store does not judge a note's text -- ``ArtifactStore.set_note``'s docstring says so --
    and every front end runs ``core.refusals.require_note`` before it writes one: ONE line, at most
    NOTE_MAX_CHARS. The FDT panel passed its Note box straight to ``create``, so a 201-character note
    or a pasted two-line one was stored as typed, and the Artifacts screen then showed a note its own
    Set button refuses to write back -- while ``fix_sentence("note")`` named this box for a refusal
    nothing here could raise.

    A line edit DOES hold a line break: ``insert`` (and a real paste) keeps it, so the two-line case
    is asserted to have one before the click, or this would pass on a note that never had it. The
    refusal is an input refusal: the yellow box, the rule's own sentence, the fix line naming the
    'Note' box, nothing dispatched and no id minted -- ``create`` is never reached, so nothing is on
    disk either."""
    from PySide6.QtWidgets import QMessageBox
    from core.artifacts import ArtifactStore, use_store
    from core.gui import fields as gui_fields
    from core.gui.panels.fdt_panel import FdtPanel
    from core.refusals import NOTE_MAX_CHARS
    from tests._fixtures import SHOWN, qt_app

    qt_app()
    store = ArtifactStore(tmp_path)
    monkeypatch.setattr(store, "create",
                        lambda *a, **k: pytest.fail("a refused note still minted a record"))
    with use_store(store):
        panel = FdtPanel()
        panel.dispatch = lambda *a, **k: pytest.fail("a refused note dispatched a run anyway")
        if how == "too_long":
            panel.record_note.setText("n" * (NOTE_MAX_CHARS + 1))
            said = f"must be at most {NOTE_MAX_CHARS} characters"
        else:
            panel.record_note.insert("first line\nsecond line")
            assert "\n" in panel.record_note.text(), "the line edit dropped the break: nothing is tested"
            said = "must be one line"
        SHOWN.clear()
        panel._run()

    assert SHOWN, "a refused note reached no box"
    box = SHOWN[-1]
    assert box.windowTitle() == "Check your inputs" and box.icon() == QMessageBox.Warning, how
    assert said in box.text(), box.text()
    assert box.informativeText() == gui_fields.fix_sentence("note")
    assert "'Note'" in box.informativeText() and "FDT analysis" in box.informativeText()
    assert box.detailedText() == "", "a refusal is not a crash and carries no traceback"


def test_the_fdt_panel_names_the_record_its_run_wrote(tmp_path):
    """run_fdt used to return None, so a finished run left the operator to find its output by hand;
    it now returns the LoadedFdt it wrote, and the panel's result slot must say which record that was
    -- by name AND id, since an unnamed record has only the id -- and move the saved-run picker onto
    it, so the selection matches the figures already in the stack.

    The record is written AFTER the panel is built, so the picker's first listing cannot hold it: a
    picker that ends up on it proves the slot re-listed the store rather than restoring a key into a
    listing that never had it (restore_key silently does nothing then)."""
    from core.artifacts import ArtifactStore, use_store
    from core.gui.panels.fdt_panel import FdtPanel
    from tests._fixtures import PaneCapture, qt_app

    qt_app()
    store = ArtifactStore(tmp_path)
    with use_store(store):
        panel = FdtPanel()
        pane = PaneCapture(panel)
        assert not panel.record_picker.has_entries(), "a fresh store offers no saved run"
        w = store.create("fdt", None, name="cell_a_first")
        w.body = {"study": "single", "settings": {}, "seed": 4242, "grid": None, "points": None,
                  "offgrid": None, "notices": [], "compared": None, "complete": False,
                  "results": None}
        with w:
            w.body["complete"] = True
        record = store.load_fdt(w.id)
        panel._on_record(record)

    said = [text for _level, text in pane.lines]
    assert any("cell_a_first" in text and record.id in text for text in said), said
    assert panel.record_picker.key() == record.id, "the picker must move onto the run just written"


def test_the_crossval_panel_creates_one_writer_per_swept_parameter(tmp_path):
    """The study sweeps two parameters and now writes a record for each, so the panel creates TWO
    writers and hands them over keyed by the same names ``run_fdt_param_sweep`` already uses for
    ``sweep_param`` -- "s" and "temp". One record would put an all-failed activity sweep and a good
    temperature sweep in one folder with one ``points`` block, which is precisely the coupling two
    records remove: an all-failed S sweep used to raise before the T sweep even started.

    The base name is suffixed per parameter (``-s``, ``-temp``), because two records cannot hold one
    name: a name is claimed at create() and a progressive record holds it from its first moment --
    and ``assert_name_free`` reads only the disk, so two creates of ONE name before either is entered
    would both pass; the distinct suffixes are what keep that unreachable from the window. Both
    ``figures/`` directories are watched (one watcher each), and neither directory exists yet --
    __enter__ on the worker thread is what creates them.

    The Seed box is read ONCE and the same value reaches the builder (``cfg.seed``), the run and both
    first bodies; a BLANK box is None in all three, never a seed of 0 nobody typed. A blank name
    gives two unnamed records, which must still be two ids and two folders (back-to-back creates used
    to mint one id, so the temperature sweep's folder collided after the whole activity sweep)."""
    from core.artifacts import ArtifactStore, use_store
    from core.FDT.cross_validation import run_param_study_cli
    from core.gui.panels.crossval_panel import CrossValPanel
    from tests._fixtures import qt_app

    qt_app()
    store = ArtifactStore(tmp_path)
    cap = {}
    with use_store(store):
        panel = CrossValPanel()
        panel.dispatch = lambda fn, *a, **k: cap.update(fn=fn, args=a, kwargs=k)
        panel.seed.setText("99")
        panel.record_name.setText("study_one")
        panel.record_note.setText("both halves")
        panel._run()

    assert cap, "nothing was dispatched"
    assert cap["fn"] is run_param_study_cli, cap["fn"]
    writers = cap["kwargs"]["writers"]
    assert sorted(writers) == ["s", "temp"], sorted(writers)
    assert [writers[k].name for k in ("s", "temp")] == ["study_one-s", "study_one-temp"]
    assert writers["s"].id != writers["temp"].id, "two records minted one id"
    assert all(w.kind == "fdt" and w.note == "both halves" for w in writers.values())
    assert cap["kwargs"]["seed"] == 99
    cfg = cap["args"][0]
    assert cfg.seed == 99, "the builder's half: the dispatched config carries the seed"
    assert cfg.preset_name == panel.preset_combo.currentText(), \
        "the preset's NAME must reach the record's settings"
    assert all(w.body["study"] == "sweep" and w.body["seed"] == 99 for w in writers.values())
    assert all(w.body["complete"] is False and w.body["notices"] == [] for w in writers.values())
    assert all(w.body[k] is None for w in writers.values()
               for k in ("settings", "grid", "points", "offgrid", "compared", "results")), \
        "the stage fills the rest"
    assert cap["kwargs"]["watch_dir"] == [writers["s"].dir / "figures",
                                          writers["temp"].dir / "figures"]
    assert not any(w.dir.exists() for w in writers.values()), \
        "create() must not touch the disk; the stage's __enter__ does the mkdir"
    assert "s_grid" in cap["kwargs"] and "t_grid" in cap["kwargs"], \
        f"the grids travel as keywords now: {sorted(cap['kwargs'])}"

    # A blank name and a blank Seed box: two unnamed records in two folders, and None everywhere
    cap.clear()
    with use_store(store):
        panel.record_name.setText("")
        panel.seed.setText("")
        panel._run()
    blank = cap["kwargs"]["writers"]
    assert [blank[k].name for k in ("s", "temp")] == ["", ""]
    assert blank["s"].id != blank["temp"].id and blank["s"].dir != blank["temp"].dir, \
        "two unnamed records share one folder"
    assert cap["kwargs"]["seed"] is None, "a blank box must not reach the run as a seed of 0"
    assert cap["args"][0].seed is None, "a blank box must not reach the builder as a seed of 0"
    assert all(w.body["seed"] is None for w in blank.values()), \
        "the panel records what it knows; the study records the seed it draws"


def test_a_taken_crossval_record_name_is_refused_at_the_click(tmp_path):
    """The taken-name refusal, for the study's SECOND name. The two creates sit inside the builder's
    ``try``, so a taken name -- here only the temperature sweep's, ``<base>-temp`` -- is the
    StoreError (field "name") of the yellow box rather than a raw traceback out of the clicked slot,
    and it is raised before anything is dispatched. The activity sweep's writer was already created
    by then, but create() writes nothing, so the refused click leaves no folder behind."""
    from PySide6.QtWidgets import QMessageBox
    from core.artifacts import ArtifactStore, use_store
    from core.gui import fields as gui_fields
    from core.gui.panels.crossval_panel import CrossValPanel
    from tests._fixtures import SHOWN, qt_app

    qt_app()
    store = ArtifactStore(tmp_path)
    with use_store(store):
        taken = store.create("fdt", None, name="study_one-temp")
        taken.body = {"study": "sweep", "settings": {}, "seed": 1, "grid": None, "points": None,
                      "offgrid": None, "notices": [], "compared": None, "complete": False,
                      "results": None}
        taken.__enter__()                   # the record now exists, unfinished, holding its name

        panel = CrossValPanel()
        panel.dispatch = lambda *a, **k: pytest.fail("a refused click dispatched a run anyway")
        panel.record_name.setText("study_one")
        SHOWN.clear()
        panel._run()

    box = SHOWN[-1]
    assert box.windowTitle() == "Check your inputs" and box.icon() == QMessageBox.Warning
    assert "study_one-temp" in box.text(), box.text()
    assert box.informativeText() == gui_fields.fix_sentence("name")
    assert box.detailedText() == "", "a refusal is not a crash and carries no traceback"
    assert len(list((tmp_path / "fdt").iterdir())) == 1, "the refused click left a second directory"


@pytest.mark.parametrize("grid", ["s_grid", "t_grid"])
@pytest.mark.parametrize("box", ["lo", "hi", "points"])
def test_a_blank_crossval_grid_box_is_refused_naming_its_grid(tmp_path, monkeypatch, grid, box):
    """The grid row used to read its three boxes with value(), which turns a blank into 0 -- and 0 is
    a legal END of a sweep, so no rule the builder could write refuses it: a blank min under a
    positive max arrived as (0.0, 1.5, n) and passed "min below max", running a sweep nobody typed
    (for the temperature grid, one starting below the bounds' own 0.05 floor). Each box is now read
    through value_or_none(), so a blank end reaches cli._check_grid as None and is refused as blank,
    and a blank count as fewer than 2 points -- either way the yellow box, the refusal naming THAT
    grid, and nothing dispatched or minted.

    A blank COUNT used to read "needs at least 2 points; got 0", a typed 0 nobody typed, against the
    house rule that a blank is refused as a blank; and every blank part is now named -- its minimum,
    its maximum or its point count. "Nothing minted" is asserted by ``create`` failing the test: a
    folder check could not fail, because ``create`` writes nothing."""
    from PySide6.QtWidgets import QMessageBox
    from core.artifacts import ArtifactStore, use_store
    from core.gui import fields as gui_fields
    from core.gui.panels.crossval_panel import CrossValPanel
    from core.refusals import describe
    from tests._fixtures import SHOWN, qt_app

    qt_app()
    store = ArtifactStore(tmp_path)
    monkeypatch.setattr(store, "create",
                        lambda *a, **k: pytest.fail("a blank grid box still minted a record"))
    with use_store(store):
        panel = CrossValPanel()
        panel.dispatch = lambda *a, **k: pytest.fail("a blank grid box dispatched a run anyway")
        row = getattr(panel, grid)
        assert all(v is not None for v in row.spec_or_none()), "the default grid is already blank"
        getattr(row, box).setText("")
        SHOWN.clear()
        panel._run()

    assert SHOWN, "a blank grid box reached no box"
    shown = SHOWN[-1]
    assert shown.windowTitle() == "Check your inputs" and shown.icon() == QMessageBox.Warning
    what = describe(grid)
    assert shown.text().startswith(what[0].upper() + what[1:]), shown.text()
    part = {"lo": "minimum", "hi": "maximum", "points": "point count"}[box]
    assert f"its {part} is blank" in shown.text(), shown.text()
    assert shown.informativeText() == gui_fields.fix_sentence(grid)


def test_the_crossval_panel_fills_each_grid_in_ascending_order(tmp_path):
    """Each sweep runs from its FDT-restoring limit to the cell's own value, and the panel used to
    fill the ends in THAT order: the temperature grid as lo=1, hi=the cell's T_a/T. The bounds allow
    T_a/T anywhere in (0.05, 10), so for a cell below 1 the panel's own untouched default was
    descending and "min below max" refused it -- a cell nobody could sweep without retyping two
    boxes. The ends are now filled in ascending order; the sweep covers the same points either way.

    The cell is a copy of a real one with only T_a/T changed, written into this test's own directory
    (never into Resources/); a cell outside the model folder resolves the model's master bounds file,
    as a new cell dropped in there would."""
    import numpy as np
    from core.artifacts import ArtifactStore, use_store
    from core.config import CELL_PATH
    from core.gui.panels.crossval_panel import CrossValPanel
    from tests._fixtures import SHOWN, qt_app

    qt_app()
    text = (CELL_PATH / "nadrowski" / "master_spont.txt").read_text(encoding="utf-8")
    assert "temp = 1.5" in text and "s = 0.95" in text, "the source cell changed: re-derive this test"
    cell = tmp_path / "nadrowski" / "cool_cell.txt"
    cell.parent.mkdir()
    cell.write_text(text.replace("temp = 1.5", "temp = 0.5"), encoding="utf-8")

    cap = {}
    with use_store(ArtifactStore(tmp_path / "store")):
        panel = CrossValPanel()
        panel.cell_picker.selected_path = lambda: str(cell)
        panel._on_cell_changed()
        assert "T_a/T = 0.5000" in panel.cell_values.text(), panel.cell_values.text()
        assert panel.t_grid.spec_or_none()[:2] == (0.5, 1.0), panel.t_grid.spec_or_none()
        assert panel.s_grid.spec_or_none()[:2] == (0.0, 0.95), panel.s_grid.spec_or_none()
        panel.dispatch = lambda fn, *a, **k: cap.update(kwargs=k)
        SHOWN.clear()
        panel._run()

    assert not SHOWN, [(b.windowTitle(), b.text()) for b in SHOWN]
    t_grid = cap["kwargs"]["t_grid"]
    assert t_grid[0] == 0.5 and t_grid[-1] == 1.0 and np.all(np.diff(t_grid) > 0), t_grid


@pytest.mark.parametrize("how", ["too_long", "two_lines"])
def test_a_crossval_record_note_is_judged_at_the_click(tmp_path, monkeypatch, how):
    """The store does not judge a note's text (``ArtifactStore.set_note``'s docstring) and every
    front end runs ``core.refusals.require_note`` before it writes one: ONE line, at most
    NOTE_MAX_CHARS. The CrossVal panel writes its note onto BOTH records, so an unjudged one would put
    two notes on disk that the Artifacts screen's own Set button refuses to write back.

    The refusal is an input refusal: the yellow box, the rule's own sentence, the fix line naming the
    'Note' box on this tab, nothing dispatched and no id minted -- the note is judged before either
    ``create``."""
    from PySide6.QtWidgets import QMessageBox
    from core.artifacts import ArtifactStore, use_store
    from core.gui import fields as gui_fields
    from core.gui.panels.crossval_panel import CrossValPanel
    from core.refusals import NOTE_MAX_CHARS
    from tests._fixtures import SHOWN, qt_app

    qt_app()
    store = ArtifactStore(tmp_path)
    monkeypatch.setattr(store, "create",
                        lambda *a, **k: pytest.fail("a refused note still minted a record"))
    with use_store(store):
        panel = CrossValPanel()
        panel.dispatch = lambda *a, **k: pytest.fail("a refused note dispatched a run anyway")
        if how == "too_long":
            panel.record_note.setText("n" * (NOTE_MAX_CHARS + 1))
            said = f"must be at most {NOTE_MAX_CHARS} characters"
        else:
            panel.record_note.insert("first line\nsecond line")
            assert "\n" in panel.record_note.text(), "the line edit dropped the break: nothing is tested"
            said = "must be one line"
        SHOWN.clear()
        panel._run()

    assert SHOWN, "a refused note reached no box"
    box = SHOWN[-1]
    assert box.windowTitle() == "Check your inputs" and box.icon() == QMessageBox.Warning, how
    assert said in box.text(), box.text()
    assert box.informativeText() == gui_fields.fix_sentence("note")
    assert "'Note'" in box.informativeText()
    assert "Sweep study cross-validation" in box.informativeText(), box.informativeText()
    assert box.detailedText() == "", "a refusal is not a crash and carries no traceback"


@pytest.mark.parametrize("s_refused", [False, True], ids=["both_finish", "s_measured_nothing"])
def test_the_crossval_window_shows_both_records_figures_and_names_them(tmp_path, monkeypatch,
                                                                       s_refused):
    """The study writes two records, and each sweep plots itself into its OWN record's ``figures/``
    when it finishes -- so a panel that watched one directory showed the S sweep's figure and never
    the T sweep's, the half of a long study that arrives last. One watcher per record is how both
    reach the figure stack, with no worker-to-window signal.

    The second case is the study two records exist for: the activity sweep measured nothing and
    refused, its record kept unfinished (a refusal after a figure or payload was handed out keeps the
    record), and the temperature sweep finished. The window then shows T's figure -- the S directory
    lists nothing -- and no box, because the study itself did not fail.

    Driven through the real dispatch, the real watchers and the real ``run_param_study_cli`` on the
    worker thread; only the per-sweep physics (``run_fdt_param_sweep``) is stood in for, by a stage
    that keeps its contract: it enters its writer on the worker thread and plots into the record's
    own ``figures/``. The panel must then NAME what the study wrote -- by name and id, since an
    unnamed record has only the id -- and move the saved-sweep picker onto the last finished record."""
    from pathlib import Path
    from core.artifacts import ArtifactStore, use_store
    from core.FDT import cross_validation
    from core.gui.panels.crossval_panel import CrossValPanel
    from core.refusals import Refusal
    from tests._fixtures import SHOWN, PaneCapture, qt_app

    app = qt_app()

    def sweep(cfg, sweep_param, sweep_grid, fixed_overrides=None, *, writer):
        writer.body.update(settings={}, points={"param": sweep_param, "planned": len(sweep_grid),
                                                "done": 0, "failed": 0})
        with writer:
            writer.payload("data.h5").write_bytes(b"spectra")
            if s_refused and sweep_param == "s":
                raise Refusal("The s sweep measured nothing.", field="s_grid")
            writer.figure_path(f"FDT ratio vs {sweep_param}").write_bytes(b"png")
            writer.body["complete"] = True
        return writer.store.load_fdt(writer.id)

    monkeypatch.setattr(cross_validation, "run_fdt_param_sweep", sweep)
    store = ArtifactStore(tmp_path)
    cap = {}
    with use_store(store):
        panel = CrossValPanel()
        pane = PaneCapture(panel)
        seen = []
        panel.figure_stack.add_png = lambda title, path: seen.append(Path(path))
        real_dispatch = panel.dispatch

        def spy(fn, *a, **k):
            cap.update(k)
            return real_dispatch(fn, *a, **k)

        panel.dispatch = spy
        panel.record_name.setText("study_one")
        panel.seed.setText("7")
        SHOWN.clear()
        panel._run()
        assert cap, "nothing was dispatched"
        _wait_for_run(app, panel, limit=60.0)

    assert SHOWN == [], [(b.windowTitle(), b.text()) for b in SHOWN]
    writers = cap["writers"]
    s_fig = writers["s"].dir / "figures" / "fdt_ratio_vs_s.png"
    t_fig = writers["temp"].dir / "figures" / "fdt_ratio_vs_temp.png"
    finished = [writers["temp"]] if s_refused else [writers["s"], writers["temp"]]
    if s_refused:
        assert seen == [t_fig], f"the window must show the T sweep's figure: {seen}"
    else:
        assert sorted(seen) == sorted([s_fig, t_fig]), f"both records' figures must arrive: {seen}"

    # Ids BRACKETED, as the line prints them: two ids minted in one second differ only by a suffix
    # ("...T145600" and "...T145600-2"), so a bare id is a substring of its sibling's.
    said = [text for _level, text in pane.lines]
    for w in finished:
        assert any(w.name in text and f"[{w.id}]" in text for text in said), (w.name, said)
    if s_refused:
        assert not any(f"[{writers['s'].id}]" in text for text in said), \
            f"the refused sweep's unfinished record was named as written: {said}"
    # ORDER-FREE (sorted, so a duplicate row still fails): the listing is newest first by `created`,
    # and the two creates in _run land on one clock tick or straddle one, which flipped a list compare.
    combo = panel.record_picker.combo
    assert sorted(combo.itemData(i) for i in range(combo.count())) == sorted(w.id for w in finished), \
        "the picker offers the FINISHED sweeps only"
    assert panel.record_picker.key() == writers["temp"].id, \
        "the picker must move onto the last record the study wrote"


def test_record_summary_names_the_cell_the_settings_the_seed_and_the_notices():
    """A saved run is only usable if you can see what produced it, and the old flat output carried no
    record of which cell or which settings produced it. Its cell, its settings, its seed and its
    notices are the four facts this renders, off the manifest alone.

    A record with no cell file -- the legacy inline-bounds branch records ``None``
    (core/artifacts/provenance.py) -- must say so rather than raise: this feeds a read-only label, and
    a formatter that raises inside a currentIndexChanged slot takes the panel down with it. The
    notices are the recorded "too thin to trust" sentences, and they are shown because a record that
    is a quick look must SAY it is a quick look wherever it is read."""
    from core.gui.panels.record_view import record_summary
    from core.artifacts.manifest import Manifest

    def _m(**over):
        d = dict(schema=1, kind="fdt", id="20260922T120000", name="cell_a", created="2026-09-22T12:00:00",
                 note="", prism={}, env={}, inputs={"cell": {"path": "Resources/Cells/nadrowski/x.txt",
                                                             "sha256": "ab"},
                                                    "bounds": None, "units": None, "model": "NADROWSKI"},
                 config={}, parents={}, fingerprints={}, payloads={}, figures=[],
                 body={"study": "single", "settings": {"n_freqs": 8, "F0": 0.05}, "seed": 4242,
                       "grid": None, "points": None, "offgrid": {"blanks": 1, "of": 8},
                       "notices": ["The frequency grid has 2 points, which is a quick look."],
                       "compared": None, "complete": True, "results": None})
        d.update(over)
        return Manifest(**d)

    text = record_summary(_m())
    assert "2026-09-22T12:00:00" in text and "seed 4242" in text, text
    assert "Resources/Cells/nadrowski/x.txt" in text, text
    assert "F0=0.05" in text and "n_freqs=8" in text, text
    assert "quick look" in text, "a notice must survive to the screen"
    assert "did not finish" not in text

    body = dict(_m().body, complete=False, seed=None, settings=None, notices=[])
    bare = record_summary(_m(inputs={"cell": None, "bounds": None, "units": None, "model": None},
                             body=body))
    assert "no cell file" in bare, bare
    assert "did not finish" in bare, "an unfinished record must say so wherever it is read"


def test_record_summary_reads_a_finished_sweep_and_survives_a_hand_edited_body():
    """The CrossVal half of the saved-run summary, and the "never raises" half of record_view's
    contract.

    A SWEEP's blank count is over every probe of every operating point it measured on its common
    grid, so the line must say so -- "5 of 240 probe frequencies came back blank" alone reads as one
    grid of 240 -- and its points block is the study's own progress, failures included.

    A HAND-EDITED body reaches this formatter: ``manifest.validate`` checks the body's key set and
    nothing inside it, so any value may be any JSON. Each shape below raised, or rendered garbage, in
    the formatter as first drafted -- a number where the notices list belongs was iterated, a bare
    string was iterated character by character, a non-dict body was asked for ``.get`` -- and this
    runs from each panel's __init__ at every launch, where one raise stops the application launching
    at all."""
    import types
    from core.gui.panels.record_view import record_summary

    sweep = types.SimpleNamespace(
        created="2026-09-22T12:00:00",
        inputs={"cell": {"path": "Resources/Cells/nadrowski/x.txt", "sha256": "ab"}},
        body={"study": "sweep", "seed": 7, "grid": None, "compared": None, "complete": True,
              "results": None,
              "settings": {"n_freqs": 30, "ensemble_M": 256, "F0": 0.05, "skip_sanity": None,
                           "preset": "exploratory", "sweep_grid": [0.0, 1.0, 8]},
              "points": {"param": "s", "planned": 8, "done": 7, "failed": 1},
              "offgrid": {"blanks": 5, "of": 240},
              "notices": ["The ensemble is 16 trajectories: read the result as a quick look."]})
    text = record_summary(sweep)
    assert "seed 7" in text and "preset=exploratory" in text, text
    assert "7 of 8 operating points of the S sweep measured (1 failed)." in text, text
    assert "5 of 240 probe frequencies across the operating points it measured came back blank." \
        in text, text
    assert "Notice: The ensemble is 16 trajectories" in text and "did not finish" not in text, text

    for body in ({"notices": 5, "settings": ["n_freqs", 8], "offgrid": "lots", "points": 3,
                  "seed": {"a": 1}},
                 {"notices": "one sentence, not a list", "settings": {1: "a", "b": 2}},
                 None, [1, 2], "a string"):
        text = record_summary(types.SimpleNamespace(created="2026-09-22T12:00:00",
                                                    inputs={"cell": "a/bare/string"}, body=body))
        assert text.startswith("Written 2026-09-22T12:00:00"), text
        assert "Cell: (no cell file recorded)" in text, text
        assert "Notice: o\n" not in text, f"a bare-string notice was iterated per character: {text}"
    assert "Notice: one sentence, not a list" in record_summary(types.SimpleNamespace(
        body={"notices": "one sentence, not a list"}))
    assert record_summary(object()).startswith("Written ?"), "a manifest with no attributes at all"


def test_record_figures_lists_a_records_pictures_by_the_watchers_own_title(tmp_path):
    """The live watcher and the re-opened view must name the same picture the same way, or a figure
    read off a saved record is a different thing from the one you watched land. Both go through
    plot_watcher._title, which is why record_view imports it rather than restating it.

    A record with no figures/ -- a run cancelled before it drew one, which keeps its folder -- is an
    empty list. It is not an error, and it must not be one: the folder surviving is the point."""
    from core.gui.panels.record_view import record_figures

    rec = tmp_path / "cell_a__20260922T120000"
    (rec / "figures").mkdir(parents=True)
    (rec / "figures" / "t_eff_ratio.png").write_bytes(b"png")
    (rec / "figures" / "spontaneous_psd.png").write_bytes(b"png")
    (rec / "data.h5").write_bytes(b"not a figure")

    assert record_figures(rec) == [
        ("spontaneous psd", str(rec / "figures" / "spontaneous_psd.png")),
        ("t eff ratio", str(rec / "figures" / "t_eff_ratio.png"))]
    assert record_figures(tmp_path / "cell_b__20260922T130000") == []


def test_selecting_a_saved_run_describes_it_and_re_opens_its_figures(tmp_path):
    """On both screens that carry a picker, selecting a record shows its cell, settings, seed and
    notices, and re-opens its figures from the record's own figures/ -- which is what makes a saved
    run readable at all, and what the old flat output could not do.

    Two guards are pinned with it, and both are defects rather than niceties. (1) The slot must not
    touch the figure stack while a run is live: the stack then holds the figures the watcher is
    landing, and clearing it would delete the live run's output to show an older run's. (2) A record
    the store cannot read must degrade to a line in the label, never raise -- this slot runs inside
    __init__ at every launch, and an exception there escapes CrossValPanel() -> MainWindow() ->
    build_app() before app.py has installed its excepthook, so a single unreadable record would leave
    the application unable to launch at all."""
    from core.artifacts import ArtifactStore, use_store
    from core.gui.panels.crossval_panel import CrossValPanel
    from core.gui.panels.fdt_panel import FdtPanel
    from tests._fixtures import qt_app

    qt_app()
    store = ArtifactStore(tmp_path)

    def _record(study, name):
        w = store.create("fdt", None, name=name)
        w.body = {"study": study, "settings": {"n_freqs": 8}, "seed": 4242, "grid": None,
                  "points": None, "offgrid": None,
                  "notices": ["The frequency grid has 2 points, which is a quick look."],
                  "compared": None, "complete": False, "results": None}
        with w:
            w.figure_path("ratio").write_bytes(b"png")
        return w

    with use_store(store):
        _record("single", "one_cell")
        _record("sweep", "a_sweep")
        for panel, key in ((FdtPanel(), "one_cell"), (CrossValPanel(), "a_sweep")):
            panel.record_picker.restore_key(store.get("fdt", key).id)
            panel._show_record()
            text = panel.record_line.text()
            assert "seed 4242" in text and "quick look" in text, (key, text)
            assert panel.figure_stack.count() == 1, key
            assert panel.figure_stack.tabText(0) == "ratio", panel.figure_stack.tabText(0)

            # (1) a live run owns the figure stack
            panel.figure_stack.clear_all()
            panel._busy = True
            try:
                panel._show_record()
            finally:
                panel._busy = False
            assert panel.figure_stack.count() == 0, f"{key}: a live run's figures were cleared"
            assert "seed 4242" in panel.record_line.text(), "the LINE still follows the selection"

            # (2) an unreadable record is a line, not a crash
            panel.record_picker.combo.setItemData(panel.record_picker.combo.currentIndex(),
                                                  "no_such_record")
            panel._show_record()
            assert "could not read" in panel.record_line.text(), panel.record_line.text()


def _saved_fdt_record(store, name, study, seed, titles):
    """A FINISHED ``fdt`` record of ``study`` whose ``figures/`` holds one PNG per title: what a run
    leaves behind, written through the real writer so the manifest and the folder are the store's."""
    w = store.create("fdt", None, name=name)
    w.body = {"study": study, "settings": {"n_freqs": 8}, "seed": seed, "grid": None,
              "points": None, "offgrid": None, "notices": [], "compared": None,
              "complete": False, "results": None}
    with w:
        for title in titles:
            w.figure_path(title).write_bytes(b"png")
    return w


def _choose(combo, row):
    """A USER's pick of ``row``, the way QComboBox makes one from its popup: the index moves
    (``currentIndexChanged``, only when it changes) and THEN ``activated`` is emitted. A programmatic
    move -- a refresh, restore_key, a run's result slot -- emits only the first, and that difference
    is what the saved-run viewer keys its figures on."""
    combo.setCurrentIndex(row)
    combo.activated.emit(row)


def _tabs(panel):
    return [panel.figure_stack.tabText(i) for i in range(panel.figure_stack.count())]


def _settle(app):
    """Give the event loop its turn: the viewer judges the SETTLED selection one turn after an index
    change, so a refresh's passing states are never mistaken for a choice."""
    for _ in range(3):
        app.processEvents()


@pytest.mark.parametrize("which", ["fdt", "crossval"])
def test_the_saved_run_viewer_opens_nothing_at_launch_and_closes_only_its_own_tabs(tmp_path, which):
    """The saved-run viewer's two rules on both screens, and the one thing that opens figures: a
    USER's pick.

    A launch fills the saved-run LINE for the restored selection and opens NONE of its figures. The
    saved run is deliberately not the first row, so restoring it CHANGES the combo's index.

    Only a pick opens figures. Qt emits ``activated`` for the keyboard and the popup and never for
    a programmatic index change; the keyboard leg below goes through Qt's own key handling, so the
    wiring is proved against Qt and not only against _choose. A programmatic move that SETTLES on
    another record closes the viewer's tabs -- they would describe a record no longer selected.

    A pick closes only the tabs the viewer itself opened. Anything else on the stack -- a comparison
    drawn there, a run's figures -- survives, and a tab the user already closed by hand is skipped
    rather than touched after Qt deleted it. A PNG another tab already shows is not opened a second
    time when its record is picked."""
    import shiboken6
    from PySide6.QtCore import QCoreApplication, QEvent, Qt
    from PySide6.QtTest import QTest
    from core.artifacts import ArtifactStore, use_store
    from core.gui import settings as st
    from core.gui.panels.crossval_panel import CrossValPanel
    from core.gui.panels.fdt_panel import FdtPanel
    from tests._fixtures import qt_app

    app = qt_app()
    store = ArtifactStore(tmp_path)
    cls, study = {"fdt": (FdtPanel, "single"), "crossval": (CrossValPanel, "sweep")}[which]

    with use_store(store):
        # The listing is newest first, so which record lands in which row is the store's business;
        # everything below is asserted by id, never by an assumed row.
        facts = {}
        for name, seed, titles in (("rec_a", 1, ["a one", "a two"]), ("rec_b", 2, ["b one"])):
            facts[_saved_fdt_record(store, name, study, seed, titles).id] = (seed, titles)
        combo = cls().record_picker.combo
        first, second = combo.itemData(0), combo.itemData(1)
        qs = st.settings()
        qs.beginGroup(which)
        qs.setValue("record", second)
        qs.endGroup()
        qs.sync()

        # a launch: the restored run is described, and nothing is opened
        panel = cls()
        combo = panel.record_picker.combo
        assert panel.record_picker.key() == second and combo.currentIndex() == 1
        assert f"seed {facts[second][0]}" in panel.record_line.text(), panel.record_line.text()
        assert _tabs(panel) == [], f"a launch re-opened the saved run's figures: {_tabs(panel)}"

        # a PROGRAMMATIC move describes the record and opens nothing
        combo.setCurrentIndex(0)
        _settle(app)
        assert f"seed {facts[first][0]}" in panel.record_line.text(), panel.record_line.text()
        assert _tabs(panel) == [], f"a programmatic move opened figures: {_tabs(panel)}"

        # a user's pick through Qt's own key handling: Down from row 0 is row 1
        QTest.keyClick(combo, Qt.Key_Down)
        assert combo.currentIndex() == 1 and _tabs(panel) == facts[second][1], _tabs(panel)

        # a tab the viewer did not open survives every pick
        panel.figure_stack.add_figure("comparison", b"")
        _choose(combo, 0)
        assert _tabs(panel) == ["comparison", *facts[first][1]], _tabs(panel)
        _choose(combo, 1)
        assert _tabs(panel) == ["comparison", *facts[second][1]], _tabs(panel)

        # a viewer tab the user closed by hand, deleted by Qt, is skipped on the next pick
        page = panel.figure_stack.widget(1)
        panel.figure_stack.tabCloseRequested.emit(1)
        # THIS page's deleteLater only: a flush for every receiver would also delete whatever earlier
        # tests left pending, out of their order, and left an uncollectable object at shutdown.
        QCoreApplication.sendPostedEvents(page, QEvent.DeferredDelete)
        assert not shiboken6.isValid(page), "the closed tab's page was never deleted: nothing is tested"
        _choose(combo, 0)
        assert _tabs(panel) == ["comparison", *facts[first][1]], _tabs(panel)

        # a programmatic move that SETTLES elsewhere closes the viewer's tabs, and only those
        panel.record_picker.restore_key(second)
        _settle(app)
        assert _tabs(panel) == ["comparison"], _tabs(panel)
        assert f"seed {facts[second][0]}" in panel.record_line.text(), panel.record_line.text()

        # no file twice: a PNG a watcher already put up is left to that tab when its record is picked
        c = _saved_fdt_record(store, "rec_c", study, 3, ["c one"])
        panel.record_picker.refresh()
        _settle(app)
        panel.figure_stack.add_png("c one", str(c.dir / "figures" / "c_one.png"))
        _choose(combo, combo.findData(c.id))
        assert _tabs(panel) == ["comparison", "c one"], _tabs(panel)
        assert "seed 3" in panel.record_line.text(), panel.record_line.text()


@pytest.mark.parametrize("how", ["store_changed", "rescan"])
def test_a_picker_refresh_reopens_no_figure_and_moves_no_tab(tmp_path, how):
    """``StorePicker.refresh`` -- run by MainWindow on every change the Artifacts screen makes (a
    note, a rename, a delete, a sweep) and by the picker's own Rescan button -- passes THREE index
    changes on its way back to the same selection: ``clear()`` to -1, the first ``addItem`` to row 0,
    and ``restore_key`` to the record. A viewer that took each for a choice opened row 0's figures (a
    record nobody chose), re-opened the tabs the user had closed, and moved the current tab -- all for
    a note set on another record. None of the three is a choice, and the selection settles where it
    was, so nothing on the figure stack may change.

    Driven through the real window: ``store_changed`` reaches ``MainWindow._refresh_store_pickers``,
    and Rescan is the picker's own button."""
    from PySide6.QtWidgets import QPushButton
    from core.artifacts import ArtifactStore, use_store
    from core.gui.main_window import MainWindow
    from core.gui.panels.fdt_panel import FdtPanel
    from tests._fixtures import qt_app

    app = qt_app()
    store = ArtifactStore(tmp_path)
    with use_store(store):
        for name, seed, titles in (("rec_a", 1, ["a one", "a two"]), ("rec_b", 2, ["b one", "b two"])):
            _saved_fdt_record(store, name, "single", seed, titles)
        w = MainWindow()
        panel = w.panel(FdtPanel)
        combo = panel.record_picker.combo
        chosen, other = combo.itemData(1), combo.itemData(0)   # NOT the row a refresh passes first
        _choose(combo, 1)
        assert len(_tabs(panel)) == 2, _tabs(panel)
        panel.figure_stack.tabCloseRequested.emit(0)           # the user closes one of them...
        panel.figure_stack.setCurrentIndex(0)                  # ...and is looking at the other
        before = (_tabs(panel), panel.figure_stack.currentIndex())
        line = panel.record_line.text()

        store.set_note("fdt", other, "an unrelated note")
        if how == "store_changed":
            w.artifact_screen.store_changed.emit()
        else:
            panel.record_picker.findChild(QPushButton).click()
        _settle(app)

        assert (_tabs(panel), panel.figure_stack.currentIndex()) == before, \
            f"a {how} refresh changed the figure stack: {before} -> {_tabs(panel)}"
        assert panel.record_picker.key() == chosen and panel.record_line.text() == line


def test_a_relaunch_opens_no_figure_even_after_an_unrelated_note_is_set(tmp_path):
    """The launch half of the refresh rule. A launch fills the saved run's LINE and opens none of its
    figures -- and that must survive the window's own wiring: the first change anyone makes in the
    Artifacts screen refreshes every picker, and a viewer that treated the refresh's passing index
    changes as a choice opened the restored run's figures then, undoing the launch rule one note set
    later."""
    from core.artifacts import ArtifactStore, use_store
    from core.gui import settings as st
    from core.gui.main_window import MainWindow
    from core.gui.panels.fdt_panel import FdtPanel
    from tests._fixtures import qt_app

    app = qt_app()
    store = ArtifactStore(tmp_path)
    with use_store(store):
        seeds = {}
        for name, seed in (("rec_a", 1), ("rec_b", 2)):
            seeds[_saved_fdt_record(store, name, "single", seed, [f"{name} fig"]).id] = seed
        combo = FdtPanel().record_picker.combo
        first, second = combo.itemData(0), combo.itemData(1)
        qs = st.settings()
        qs.beginGroup("fdt")
        qs.setValue("record", second)
        qs.endGroup()
        qs.sync()

        w = MainWindow()
        panel = w.panel(FdtPanel)
        assert panel.record_picker.key() == second and _tabs(panel) == []
        assert f"seed {seeds[second]}" in panel.record_line.text(), panel.record_line.text()

        store.set_note("fdt", first, "an unrelated note")
        w.artifact_screen.store_changed.emit()
        _settle(app)
        assert _tabs(panel) == [], f"a note set on another record opened figures: {_tabs(panel)}"
        assert panel.record_picker.key() == second
        assert f"seed {seeds[second]}" in panel.record_line.text(), panel.record_line.text()


@pytest.mark.parametrize("which", ["fdt", "crossval"])
def test_after_a_run_only_the_runs_own_figures_remain(tmp_path, monkeypatch, which):
    """The user had picked an earlier record, so its figures were open; then they ran the analysis.
    The run's watcher puts the new figures up as they land, and the result slot moves the picker onto
    the new record -- after which the stack held BOTH records' figures, under IDENTICAL titles (every
    run of one analysis draws the same figures), while the line described only the new one. When a
    run finishes the viewer closes its own tabs, and the move onto the new record opens nothing,
    because the watcher already shows its figures: what remains is the run's own figures, each once.

    Driven through the real dispatch, watchers and result slot; only the measurement is stood in for,
    by stages that keep their contract (enter the writer on the worker thread, plot into the record's
    own figures/)."""
    from core.artifacts import ArtifactStore, use_store
    from core.FDT import cross_validation
    from core.gui.panels import fdt_panel
    from core.gui.panels.crossval_panel import CrossValPanel
    from tests._fixtures import qt_app

    app = qt_app()

    def run_fdt(cfg, *, skip_sanity, confirm_production, writer, seed=None):
        with writer:
            for title in ("x one", "x two"):
                writer.figure_path(title).write_bytes(b"png")
        return writer.store.load_fdt(writer.id)

    def sweep(cfg, sweep_param, sweep_grid, fixed_overrides=None, *, writer):
        writer.body.update(settings={}, points={"param": sweep_param, "planned": len(sweep_grid),
                                                "done": 0, "failed": 0})
        with writer:
            writer.payload("data.h5").write_bytes(b"spectra")
            writer.figure_path(f"FDT ratio vs {sweep_param}").write_bytes(b"png")
        return writer.store.load_fdt(writer.id)

    monkeypatch.setattr(fdt_panel, "run_fdt", run_fdt)
    monkeypatch.setattr(cross_validation, "run_fdt_param_sweep", sweep)
    cls, study, titles = {
        "fdt": (fdt_panel.FdtPanel, "single", ["x one", "x two"]),
        "crossval": (CrossValPanel, "sweep", ["FDT ratio vs s", "FDT ratio vs temp"])}[which]
    store = ArtifactStore(tmp_path)
    with use_store(store):
        earlier = _saved_fdt_record(store, "earlier", study, 1, titles)   # the run's own titles
        panel = cls()
        combo = panel.record_picker.combo
        _choose(combo, combo.findData(earlier.id))
        assert len(_tabs(panel)) == len(titles), _tabs(panel)
        panel.seed.setText("77")
        panel._run()
        _wait_for_run(app, panel, limit=60.0)
        _settle(app)

        stack = panel.figure_stack
        shown = [stack.png_path(i) for i in range(stack.count())]
        assert not any(str(earlier.dir) in (p or "") for p in shown), \
            f"the earlier record's tabs outlived the run: {shown}"
        assert sorted(_tabs(panel)) == sorted(t.lower() for t in titles), _tabs(panel)
        assert len(set(shown)) == len(shown) == len(titles), f"a figure is shown twice: {shown}"
        assert panel.record_picker.key() != earlier.id
        assert "seed 77" in panel.record_line.text(), panel.record_line.text()


def test_after_a_failed_run_only_its_own_figures_remain(tmp_path, monkeypatch):
    """Only the run's own figures remain, on the path where the picker does NOT move. A run that
    fails hands back no record, so the selection stays on the earlier record and nothing about the
    picker says the run happened -- yet its figures landed through the watcher under the same titles
    as the earlier record's. The viewer's tabs close when the run finishes, whatever the outcome, so
    what remains is the failed run's own partial figures; the line still describes the selection,
    which is what the picker shows."""
    from core.artifacts import ArtifactStore, use_store
    from core.gui.panels import fdt_panel
    from tests._fixtures import SHOWN, qt_app

    app = qt_app()

    def run_fdt(cfg, *, skip_sanity, confirm_production, writer, seed=None):
        with writer:
            writer.figure_path("x one").write_bytes(b"png")
            raise RuntimeError("the measurement failed after its first figure")

    monkeypatch.setattr(fdt_panel, "run_fdt", run_fdt)
    store = ArtifactStore(tmp_path)
    with use_store(store):
        earlier = _saved_fdt_record(store, "earlier", "single", 1, ["x one", "x two"])
        panel = fdt_panel.FdtPanel()
        combo = panel.record_picker.combo
        _choose(combo, combo.findData(earlier.id))
        assert _tabs(panel) == ["x one", "x two"], _tabs(panel)
        SHOWN.clear()
        panel._run()
        _wait_for_run(app, panel, limit=60.0)
        _settle(app)

        assert SHOWN and "measurement failed" in SHOWN[-1].text(), "the failure reached no box"
        stack = panel.figure_stack
        shown = [stack.png_path(i) for i in range(stack.count())]
        assert _tabs(panel) == ["x one"] and str(earlier.dir) not in (shown[0] or ""), shown
        assert panel.record_picker.key() == earlier.id and "seed 1" in panel.record_line.text()


def test_record_summary_never_prints_none():
    """A null in a record means "not recorded" or "does not apply", and the line must say nothing
    rather than print Python's ``None``. A real SWEEP's settings carry two nulls on every record --
    ``skip_sanity`` and ``confirm_production`` are run_fdt's arguments, which a sweep does not take
    (``_settings_block``'s docstring) -- so they printed on every sweep. A body written part way can
    hold a count that is not known yet, and a count line with a missing count ("3 of None probe
    frequencies") is a sentence that is not true.

    The settings block below is built by the REAL ``_settings_block``, the one the sweep stage calls,
    so a change to what it records nulls for is seen here."""
    import types
    from core.FDT.fdt_pipeline import _settings_block
    from core.gui.panels.record_view import record_summary

    cfg = types.SimpleNamespace(n_freqs=30, ensemble_M=256, freqs_per_batch=1, F0=0.05,
                                freq_bounds=(0.1, 10.0), burn_in_nd=50.0, T_obs_periods=40,
                                dt_nd=0.01, psd_T_obs_nd=2000.0)
    settings = {**_settings_block(cfg), "preset": "exploratory", "sweep_grid": [0.0, 1.0, 8]}
    assert settings["skip_sanity"] is None and settings["confirm_production"] is None, \
        "the premise: a sweep's settings record both null"

    def _m(body):
        return types.SimpleNamespace(
            created="2026-09-22T12:00:00",
            inputs={"cell": {"path": "Resources/Cells/nadrowski/x.txt", "sha256": "ab"}},
            body={"study": "sweep", "grid": None, "compared": None, "results": None, **body})

    text = record_summary(_m({"settings": settings, "seed": 7, "complete": True, "notices": [],
                              "points": {"param": "s", "planned": 8, "done": 8, "failed": 0},
                              "offgrid": {"blanks": 5, "of": 240}}))
    assert "None" not in text, text
    assert "preset=exploratory" in text and "ensemble_M=256" in text, text
    assert "8 of 8 operating points of the S sweep measured (0 failed)." in text, text

    partial = record_summary(_m({"settings": {"n_freqs": 30, "F0": None}, "seed": 7,
                                 "complete": False, "notices": [None, "A real notice."],
                                 "points": {"param": "s", "planned": 8, "done": None,
                                            "failed": None},
                                 "offgrid": {"blanks": 3, "of": None}}))
    assert "None" not in partial, partial
    assert "n_freqs=30" in partial and "Notice: A real notice." in partial, partial
    assert "did not finish" in partial, partial


def test_the_chi_drive_and_band_are_read_only_and_the_draft_carries_config():
    """The χ drive amplitude and band are MEASUREMENTS (config.py's CHI_F0 and CHI_FREQ_BOUNDS), not
    per-run choices: there is no chi override, so build_prior refuses any other value seconds after
    Apply, and a box that accepts one only manufactures that refusal. The boxes stay as displays of
    config.py under a caption saying so; the draft carries None for both, so make_sim_config takes
    config.py's values exactly as the command-line tool does; and the "Model applied" line reports
    config.py, not the boxes. The help and the config.py comment stop inviting the edit.

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
    """At Apply, every bad box is a Refusal naming its field, shown through _refusal (the yellow box),
    and the session is left exactly as it was: no new draft, no re-gate. A BLANK box is refused,
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
    """On the Prior tab, a blank, half-typed or out-of-rule knob box is REFUSED at the click -- the
    yellow box, through _refusal -- and nothing is dispatched, the config is not built and the
    session's downstream is not reset. The tab used to clamp (max(2, cluster_size.value())) and
    default (walk_step or config.PRIOR_SWEEP_STEP), so a blank "Min cluster size" silently built a
    prior with a floor of 2 that nobody typed, and a blank "Random-walk step" one with the default
    that nobody chose.

    Three more things the same click must get right, because the stage, which has a load branch,
    does: a LOAD click reads none of the seven knobs, so a blank box does not stop it and no knob is
    forwarded (the stage resolves None to its default); a typed 0 in "Candidates per round" is the
    automatic value, not a blank; and the live sweep note under the boxes renders a bad box as
    "<label> is blank." / "<label> must be <rule>." from the SAME rule the click runs, so the note and
    the refusal can never name different limits."""
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
    the tab are ONE string. A literal at an add_help_row call is the defect: the row can be renamed
    while the refusal keeps naming the old label. The control table's read-back pin
    (test_the_gui_control_table_matches_the_tabs_labels) checks the rendered QLabels over every tab;
    this is the source-level half for this tab."""
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
    """At the Posterior tab's click, every knob the TRAIN branch reads is validated before any
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
    """On the Validate tab, both boxes were clamped with max(1, ...) at the click, so a blank or a 0
    became a 1 nobody typed -- and for the operating points a 1 is a DIFFERENT measurement, not a
    smaller one: the calibration's operating-point count, cal_n_scales, is t_scale's effective sample
    size, which is why the stage now refuses it instead of clamping. A refused box is the yellow box
    (stubbed here as _refusal) naming the box, the Validate tab and the default, and NOTHING is
    dispatched; a typed value in rule reaches validate_calibration as an int, and 1 operating point is
    allowed -- it is a choice, not a typo."""
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
    seed = sent["kwargs"]["seed"]
    assert type(seed) is int and 0 <= seed < 2 ** 31


def test_the_validate_tab_summary_names_the_verdict_and_the_seed():
    """The tab's closing line leads with the calibration verdict and the seed the run drew, so the
    log pane says whether the posterior passed and how to repeat the calibration."""
    from types import SimpleNamespace
    from core.gui.screens.inference_screen import InferenceScreen
    from core.gui.session import SbiSession
    from tests._fixtures import qt_app

    qt_app()
    inf = InferenceScreen()
    inf.session = SbiSession(cfg=object(), inf_prior=_prior_stub(), posterior=_posterior_stub())
    vp = inf.validate_panel
    lines = []
    vp.log_pane.append_line = lines.append
    vp._on_calibration(SimpleNamespace(name="c1", id="x", results={
        "tarp": {"atc": 0.01, "ks_p": 0.4}, "informativeness": None, "verdict": {"passed": False},
        "seed": 11}))
    assert lines[-1].startswith("Calibration verdict FAIL (seed 11), recorded as c1: TARP ATC=0.01, KS p=0.4")


def test_the_tsnpe_tab_refuses_bad_boxes_at_the_click_and_dispatches_nothing(monkeypatch):
    """On the TSNPE tab, offscreen, the refusals a user reads on the real screen. A blank HPD box used
    to reach the stage as 0.0 (FloatField.value()) and a blank direction box as 0, and the stage
    refused each with a sentence naming neither the box nor the default; the batch count was clamped
    to 1 and a blank rows-per-batch became a 0 nobody typed. Now the click reads every box through
    the shared rules BEFORE the observation is re-hashed (a refused box must cost nothing), shows a
    refusal as the yellow box (stubbed here as _refusal) naming the box, the TSNPE tab and the
    default, and dispatches nothing. A TYPED 0 in the rows-per-batch box still means automatic. The
    latent-width ceiling on the direction count stays the STAGE's -- the click has no posterior to
    measure it against -- so an oversized count is dispatched, and tsnpe_round refuses it with the
    width and the default in the sentence."""
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
    one edit in core/gui/fields.py, and the control table's read-back pins the rendered text. A
    literal here would be a second copy of the name -- the kind that goes stale the day the first one
    moves."""
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


def test_the_window_says_t_obs_where_it_means_the_recording_length():
    """On the tier-1 box T is also temperature, so the window's text names the recording length
    T_obs wherever it pairs it with t_scale: the calibration's operating-point row and both
    tooltips that describe those operating points."""
    from core.gui import fields as gui_fields
    from core.gui.panels.inference.help_text import HELP
    assert gui_fields.label("cal_n_scales") == "(t_scale, T_obs) operating points"
    assert HELP["cal_scales"].startswith("(t_scale, T_obs) operating points")
    assert "(t_scale, T_obs)" in HELP["num_runs"]
    assert "(t_scale, T)" not in HELP["cal_scales"] + HELP["num_runs"]


def test_a_blank_t_obs_is_refused_on_all_three_infer_branches(tmp_path):
    """The three observation-length boxes. FloatField.value() turns a blank or half-typed box into
    0.0, and every branch used to forward it: the simulated one spent the simulation and then died in
    math.log(0.0) (statistics.py), the bench ones built a zero-length observation. Now the click
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


def test_the_chi_branch_refuses_a_blank_drive_amplitude_and_a_missing_probe_at_the_click(tmp_path):
    """The chi page's boxes, at the click, each through _refusal with its own field and nothing
    dispatched: the physical drive amplitude blank or not positive (require_positive; a blank used to
    be a division by zero inside the worker), the passive recording blank or not there, zero probes,
    a problem in the probe table (a row with no recording), and a probe recording that names no file.
    When every box passes, the recording set carries the passive file, the (recording, frequency)
    pairs, T_obs and F0 as typed."""
    from core import orchestrator
    from core.refusals import Refusal
    from core.gui.screens.inference_screen import InferenceScreen
    from tests._fixtures import qt_app

    qt_app()
    inf = InferenceScreen()
    inf.install_config(_chi_cfg(k=2))
    inf.session.posterior = _posterior_stub()
    inf.session.inf_prior = _prior_stub()
    panel = inf.infer_panel
    sent, refused = {}, []
    panel.dispatch = lambda fn, *a, **k: sent.update(fn=fn, args=a, kwargs=k)
    panel._refusal = lambda exc: refused.append(exc)
    passive = tmp_path / "passive.npy"
    passive.touch()
    probes = [tmp_path / f"probe{i}.npy" for i in range(2)]
    for p in probes:
        p.touch()
    panel.infer_mode.setCurrentIndex(1)
    panel.chi_tobs.setText("2.0")
    panel.chi_f0_si.setText("1.5")
    panel.chi_spont.edit.setText(str(passive))
    assert len(panel._chi_forced_fields) == 2
    for i, row in enumerate(panel._chi_forced_fields):
        row.path.edit.setText(str(probes[i]))
        row.freq.setText(str(10.0 + i))

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

    refuse(panel.chi_f0_si, "", "chi_f0_si", "is blank")
    refuse(panel.chi_f0_si, "0", "chi_f0_si", "greater than 0")
    refuse(panel.chi_f0_si, "-1", "chi_f0_si", "greater than 0")
    refuse(panel.chi_spont.edit, "", "recording_spont", "is blank")
    refuse(panel.chi_spont.edit, str(tmp_path / "gone.npy"), "recording_spont", "was not found")
    refuse(panel._chi_forced_fields[1].path.edit, "", "recording_probe", "probe 2: no recording selected")
    refuse(panel._chi_forced_fields[0].path.edit, str(tmp_path / "gone.npy"), "recording_probe",
           "was not found")

    # every box given: dispatched once, with the pairs as typed
    assert click() is None
    assert sent["fn"] is orchestrator.experimental_inference, sent
    rec = sent["args"][2]
    assert rec.spont == str(passive) and rec.T_obs_s == 2.0 and rec.F0_si == 1.5
    assert rec.forced == ((str(probes[0]), 10.0), (str(probes[1]), 11.0)), rec.forced

    # zero probes: refused by the probe recording's field, before the (empty) table is read
    for row in list(panel._chi_forced_fields):
        panel._remove_chi_probe(row)
    e = click()
    assert sent == {} and isinstance(e, Refusal) and e.field == "recording_probe", (sent, e)
    assert "at least one forced probe" in e.message, e.message


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
    """"Plan probes…" is a click, so it is refused like one: a blank or non-positive T_obs and a
    blank or missing passive recording go to _refusal with their field, BEFORE the recording is
    loaded. A blank T_obs used to read as 0.0 and the planner ran on a one-sample window, and a blank
    recording was a warning line. A complete click passes the rules and goes on to load the
    recording; the stub config has no hardware, so on this test's path that load fails inside the
    planner's own guarded step, and its "Could not measure Ω₀" error line is the evidence the rules
    ran first and let the click through."""
    from core.refusals import Refusal
    from core.gui.screens.inference_screen import InferenceScreen
    from tests._fixtures import PaneCapture, qt_app

    qt_app()
    inf = InferenceScreen()
    inf.install_config(_chi_cfg(k=2))
    panel = inf.infer_panel
    refused = []
    panel._refusal = lambda exc: refused.append(exc)
    pane = PaneCapture(panel)           # both pane channels, as (level, text) -- the stub's own order
    lines = pane.lines                  # the list the capture appends to, so plan()'s clear() still works
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
    """The control table's pin. fields.CONTROL is where a refusal learns which box to name ("Set it
    in the 'T_obs (s)' box on the Infer or Live simulation tab."), and the tabs build their rows
    FROM it (label(key)), so the two cannot drift -- this reads every tab's form rows back and checks
    that each (tab, label) entry is a label that tab actually shows. The rows are read the way Qt
    holds them: the QLabel inside each help_label holder (help_badge.py), or the plain QLabel of a
    row added without help text, compared against labels.pretty_gui(label), which is what the holder
    was given. The Infer tab is read after install_config with a FORCED stub config so its three
    drive rows exist; a chi config would build none (no force_params_dict) and the drive entries
    would pass vacuously. The places are every section's tabs and the two screens, read off a built
    MainWindow -- the model builder with one variable declared and its parameter detected, because
    four of the boxes it names exist only then.

    The second half: the Infer tab's registered rows are built from label(key), not from a literal
    that happens to match today."""
    from PySide6.QtWidgets import QFormLayout, QLabel
    from core.gui import fields as gui_fields
    from core.gui.panels.inference.infer_tab import InferPanel
    from core.gui.screens.inference_screen import InferenceScreen
    from core.gui.widgets.field_row import LabeledFieldRow
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

    # A place may be any section's tab or one of the two screens, so read them all back off a built
    # window rather than the six inference tabs alone.
    from core.gui.main_window import MainWindow
    window = MainWindow()
    for section in (window.reduction_screen, window.fdt_screen, window.simulate_screen):
        for i in range(section.tabs.count()):
            tabs[section.tabs.tabText(i)] = section.tabs.widget(i)
    tabs["Artifacts"] = window.artifact_screen
    tabs["Model Builder"] = window.model_builder_screen
    # The model builder's init row exists once per declared variable and its value/min/max boxes once
    # per detected parameter (the control table names all four), so a freshly built screen shows
    # none of them: declare one variable and detect its parameter before reading the screen back. The
    # parameter row's boxes are captioned inside a LabeledFieldRow, not a form row, so read those
    # captions too.
    mb = window.model_builder_screen
    mb.vars_edit.setText("x")
    mb._set_variables()
    mb._var_rows[0].drift.setText("-k*x")
    mb._detect_params()
    seen = {tab: shown(panel) for tab, panel in tabs.items()}
    seen["Model Builder"] += [lab.text() for row in mb.findChildren(LabeledFieldRow)
                              for lab in row.findChildren(QLabel)]
    tuples = {k: e for k, e in gui_fields.CONTROL.items() if isinstance(e, tuple)}
    assert tuples, "the control table has no (tab, label) entries"
    for key in ("drive_amplitude", "drive_frequency", "drive_phase"):
        assert key in tuples, f"{key} must be a (tab, label) entry so the read-back covers the drive rows"
    # No exemptions: every entry is read back off its tab, the Seed rows of the FDT analysis and
    # Sweep study cross-validation tabs included.
    missing = []
    for key, (tab, text) in tuples.items():
        for name in (tab if isinstance(tab, tuple) else (tab,)):   # the budget boxes name two tabs
            assert name in seen, f"{key}: CONTROL names a place that does not exist: {name!r}"
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
    """A composition works on a private copy of the configuration and changes nothing it was handed
    -- pinned here at the window, on the FAILURE path: the composition injects the cell's truth,
    records the cell and sets T_obs on the config it holds, then the run dies inside
    generate_observations. All three writes used to stay on the session, so the NEXT training
    anchored its Fisher rotation on that cell and the next manifest named it as an input. The error
    reaches the red box (a bug, not a Refusal), and the session's config is untouched.

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
    """The private copy at the window, on the SUCCESS path, with nothing stubbed: a real simulated
    inference on the tiny posterior, dispatched by this tab through the worker, writes its
    observation and inference and hands them to the session -- and leaves the session's config
    exactly as install_config set it: no truth, no cell among its sources, the T_obs it had (1.0 s
    from the fixture) although the run was typed 1.5 s. The box value differs from the session's on
    purpose, so the T_obs clause is not satisfied by coincidence."""
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


# ── a live run is visible app-wide ───────────────────────────────────────────────────────────────
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
    this button sits on that path. `set_running` is the only writer: a string shows it, None hides
    and clears it. The objectName is what the global QSS keys on, so the two are pinned against each
    other here -- a renamed button would otherwise silently lose its styling."""
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


def test_the_window_fills_the_run_slot_while_a_panel_runs_and_clears_it_after():
    """The window's half of the run banner: the header says what is running and for how long, and
    stops saying it the moment the run ends. And LAUNCH IS QUIET -- the slot is empty, no clock is
    ticking and no panel is named until something actually runs. The QTimer is constructed in
    __init__ and never started there: an unstarted QTimer registers nothing with the OS, which is
    what keeps the 150 ms of layout the 2026-09-11 taskbar-icon incident was made of off the launch
    path."""
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
    # Start on a section that is NOT Home, so "stays where it is" is distinguishable from a wrong
    # fallback that navigates Home.
    start = w._section_index["Simulate"]
    assert start != 0
    w.nav.go_to(start)
    orphan._set_busy(True)
    try:
        assert not w.nav.btn_running.isHidden()
        assert w.nav.btn_running.text().startswith("Running: a task — ")
        assert w.nav.btn_running.toolTip() == ""
        w.nav.btn_running.click()
        assert w.nav.stack.currentIndex() == start, "an unknown panel must not navigate anywhere"
    finally:
        orphan._set_busy(False)


def test_the_running_section_tile_and_tab_carry_a_marker_and_nothing_is_disabled():
    """The run markers. The running section's Home tile takes a suffix and the running tab a leading
    mark, both cleared the moment the run ends -- and both rewritten from the labels the screens were
    BUILT with, so marking is idempotent and a cleared label is byte-identical to the original (the
    tab titles are read back verbatim by test_every_field_key_has_a_window_control..., which asserts
    ["Config", "Prior", "Posterior", "Validate", "Infer", "TSNPE"]).

    And nothing is disabled: the app-wide controls lock stands on its own, the Home tiles stay live,
    every tab stays selectable, and the marker survives refresh_gates -- which rewrites tab tooltips
    and enabled states after every stage, and would be running while a marked run is live. You must
    be able to look at another tab while a twenty-minute train runs."""
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


def test_the_config_tab_says_what_the_session_holds():
    """Apply's session line, the first half. ONE describer, ``InferenceScreen.session_contents()``,
    answers "what would a new draft throw away" -- in pipeline order, in plain phrases with no code
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


def test_apply_confirms_before_it_discards_the_session(monkeypatch):
    """Apply's confirmation, the second half. Apply is the destructive one of the screen's two entry
    points, so it asks -- but ONLY when there is something to lose.

    Four legs, and the first is the one that keeps the guard tolerable: an empty session is replaced
    in silence, with SHOWN empty, so nothing about the first Apply of a sitting changes. A non-empty
    one raises an INSTANCE QMessageBox (never QMessageBox.question -- the statics are C++ and escape
    tests/conftest.py's dialog guard, so a static call STALLS the offscreen suite instead of failing
    it) that lists exactly what ``session_contents()`` reports, says those artifacts stay on disk,
    and has "Keep this session" as its default BY NAME -- the lesson of the near-miss dialog and the
    load question, where buttons() orders by role and Enter landed on the destructive button.
    Anything but the destructive button leaves the session object identical.
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


def test_the_store_pickers_say_what_they_hold(monkeypatch):
    """The FIRST test of any kind on what a picker renders.

    Three things. The visible item text marks the EXCEPTION only -- a posterior with
    ``amortized is False`` reads "<label>  —  narrowed (TSNPE)", and amortized, being the norm,
    gets no suffix (a suffix on almost every row carries no information and pushes the name out of a
    combo sized to its first show). The per-item TOOLTIP keeps every fact it carried, and words the
    one fact it shares with the item text the SAME way: "narrowed (TSNPE)", in place of the old
    "NON-AMORTIZED (TSNPE)" -- one wording for one fact, in the item, the tooltip and the line. No
    test in the repository pinned either string before this one. And each of the three tabs that
    owns a StorePicker carries a read-only line beneath it, spelling the current selection out
    through ``selection_summary()`` -- refreshed on ``currentIndexChanged`` AND on ``refresh()``,
    because a stage that writes the store rescans the picker without anyone clicking.

    The store is stubbed at the picker's one seam, ``_resolved_store``, the way
    tests/test_settings_persistence.py already stubs it; each row carries exactly the ``Summary``
    fields refresh() reads and no more, so the test cannot pass on a field the real listing does not
    fill.
    """
    import types
    from PySide6.QtCore import Qt
    from core.gui.screens.inference_screen import InferenceScreen
    from core.gui.widgets.artifact_picker import NARROWED_SUFFIX, StorePicker
    from tests._fixtures import qt_app

    qt_app()

    def row(label, id_, created, *, mode=None, width=None, amortized=None, complete=True,
            finished=None):
        # For these six kinds ``complete`` and ``finished`` are the same fact, so the stub derives
        # one from the other rather than making every call site repeat it (store.Summary).
        return types.SimpleNamespace(complete=complete,
                                     finished=complete if finished is None else finished,
                                     label=label, id=id_, created=created,
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


def test_the_store_picker_offers_finished_rows_only_and_honours_a_row_filter(monkeypatch):
    """``StorePicker`` skipped rows whose ``complete`` was false, and ``complete`` means "has a valid
    manifest" -- NOT "the run finished" (store.Summary). For the six ordinary kinds the two coincide,
    because ``ArtifactWriter._commit`` writes the manifest last. For the two kinds written
    PROGRESSIVELY -- the training cache, and ``fdt`` -- they do not: a record carries a manifest from
    its first moment, so the picker would have offered a run that is still going, or one a cancel
    left half written, as if it were a result. It filters on ``finished`` instead, which is the same
    answer for every kind that is not progressive.

    And ``refresh`` listed every row of the kind with no hook, so the FDT and CrossVal screens could
    not show only their own study out of the one ``fdt`` kind (both analyses and their comparisons
    live in it). An optional row predicate is the whole addition -- no new widget, because every
    fact the picker shows is already a ``Summary`` field.

    The store is stubbed at the picker's one seam, ``_resolved_store``, as the picker tests already
    stub it; each row carries exactly the ``Summary`` fields ``refresh()`` reads and no more, so the
    test cannot pass on a field the real listing does not fill.
    """
    import types
    from core.gui.widgets.artifact_picker import StorePicker
    from tests._fixtures import qt_app

    qt_app()

    def row(label, id_, *, complete=True, finished=True, study=None):
        return types.SimpleNamespace(complete=complete, finished=finished, label=label, id=id_,
                                     created="2026-09-22T09:00:00", mode=None, width=None,
                                     amortized=None, study=study)

    rows = [row("done_single", "a", study="single"),
            row("running_single", "b", finished=False, study="single"),
            row("done_sweep", "c", study="sweep"),
            row("no_manifest", "d", complete=False, finished=False)]
    store = types.SimpleNamespace(list=lambda kind: list(rows) if kind == "fdt" else [])
    monkeypatch.setattr(StorePicker, "_resolved_store", lambda self: store)

    # (a) unfiltered: the unfinished run and the manifest-less leftover are both absent
    plain = StorePicker("fdt")
    assert [plain.combo.itemText(i) for i in range(plain.combo.count())] == \
        ["done_single", "done_sweep"]

    # (b) the predicate narrows it to one study, and is applied on top of the finished rule
    single = StorePicker("fdt", row_filter=lambda s: s.study == "single")
    assert [single.combo.itemText(i) for i in range(single.combo.count())] == ["done_single"]
    sweep = StorePicker("fdt", row_filter=lambda s: s.study == "sweep")
    assert [sweep.combo.itemText(i) for i in range(sweep.combo.count())] == ["done_sweep"]

    # (c) the predicate survives a refresh, which is how a screen sees a run that just finished
    rows.append(row("second_single", "e", study="single"))
    single.refresh()
    assert [single.combo.itemText(i) for i in range(single.combo.count())] == \
        ["done_single", "second_single"]

    # (d) the default is no predicate at all, so every existing caller is unchanged: the sentinel +
    # the THREE finished rows, because (c) appended one. BOUND to a name: a temporary picker can be
    # collected before .count() runs -- "Internal C++ object already deleted" in plain Python -- so
    # the one-liner passed only because pytest's assertion rewrite held it
    default = StorePicker("fdt", allow_new=True)
    assert default.combo.count() == 4


# ── the panel/tab counts, stale since the TSNPE tab arrived ──────────────────────────────────────
def test_the_panel_docstrings_no_longer_count_nine_panels_or_five_tabs():
    """Four sentences in base_panel.py and one in inference/base.py counted nine panels and five
    inference tabs, both stale from before the TSNPE tab: there are TEN BasePanel subclasses (the four
    section panels plus six inference tabs) and NINE save_settings overrides (ValidatePanel is the one
    subclass that does not override it). The Artifacts screen is not a panel at all -- it is a plain
    QWidget, so a run elsewhere cannot grey it out.

    Phrases, not a subclass count: the suites own throwaway BasePanel subclasses persist for the life
    of the process, so counting __subclasses__() would assert on test order.

    The source is whitespace-NORMALISED before every check. Two of these phrases are wrapped across a
    newline in the file ("there are nine independent / splitters"), so against raw source the literal
    would never appear, the assertion would pass before AND after the edit, and the stale sentence
    would survive behind a green test -- which is the one failure this test exists to prevent.
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
    and would pass before and after the rewrite."""
    from core.gui import main_window as mw

    src = " ".join(inspect.getsource(mw.MainWindow).split())
    assert "the only QSettings WRITE site" not in src
    assert "_persist_layout" in src and "1500" in src


def test_the_comparison_list_appends_from_the_single_selection_picker():
    """StorePicker is single-selection, so the comparison controls hold a LIST the picker appends to
    and no multi-select widget is built. Adding the same record twice is not an error and is not a
    second entry -- a curve drawn twice is a curve drawn once with a fatter line -- and the order the
    list keeps is the order the legend will read."""
    from core.gui.widgets.compare_list import CompareList
    from tests._fixtures import qt_app

    qt_app()

    class _Picker:
        def __init__(self):
            self.current = ("id_a", "run A")

        def selected(self):
            return (self.current[0], self.current[0] is None)

        def selection_text(self):
            return self.current[1]

    picker = _Picker()
    lst = CompareList(picker)
    assert lst.ids() == []
    assert lst.add_selected() is True and lst.ids() == ["id_a"]
    assert lst.add_selected() is False, "the same record twice is one curve, not two"
    picker.current = ("id_b", "run B")
    assert lst.add_selected() is True and lst.ids() == ["id_a", "id_b"]
    lst.list.setCurrentRow(0)
    lst.remove_selected()
    assert lst.ids() == ["id_b"]
    picker.current = (None, "")
    assert lst.add_selected() is False, "nothing selected adds nothing"


def test_the_fdt_and_crossval_screens_dispatch_their_comparison_modes():
    """The FDT screen carries cells, repeats and renormalise over its own picker, the CrossVal
    screen carries sweeps. Each button dispatches the ONE public entry with the mode, the
    ids the list holds and the mode's own setting -- the panel reimplements nothing, so the arity,
    the study and the unfinished-record refusals are the stage's and reach the yellow box through
    _on_error. The renormalise box is read with value_or_none(), so a BLANK box is refused as blank
    rather than read as the zero every numeric box returns for one."""
    from core.FDT.compare import compare
    from core.gui.panels.crossval_panel import CrossValPanel
    from core.gui.panels.fdt_panel import FdtPanel
    from tests._fixtures import qt_app

    qt_app()
    sent = {}
    panel = FdtPanel()
    panel.dispatch = lambda fn, *a, **k: sent.update(fn=fn, args=a, kwargs=k)
    panel.compare_list.list.addItem("run A")
    panel.compare_list.list.item(0).setData(_USER_ROLE, "id_a")
    panel.compare_list.list.addItem("run B")
    panel.compare_list.list.item(1).setData(_USER_ROLE, "id_b")

    panel.compare_mode.setCurrentText("cells")
    panel.btn_compare.click()
    assert sent["fn"] is compare and sent["args"] == ("cells", ["id_a", "id_b"]), sent
    assert sent["kwargs"]["provide_fig_sink"] is True and "prefactor" not in sent["kwargs"]
    assert (sent["kwargs"]["name"], sent["kwargs"]["note"]) == ("", ""), "blank boxes: unnamed, no note"

    sent.clear()
    panel.compare_mode.setCurrentText("renormalise")
    panel.renorm_prefactor.setText("3.5")
    panel.btn_compare.click()
    assert sent["args"][0] == "renormalise" and sent["kwargs"]["prefactor"] == 3.5, sent

    sent.clear()
    panel.renorm_prefactor.setText("")
    panel.btn_compare.click()
    assert sent["kwargs"]["prefactor"] is None, "a blank box travels as blank; the stage refuses it"

    sent.clear()
    xv = CrossValPanel()
    xv.dispatch = lambda fn, *a, **k: sent.update(fn=fn, args=a, kwargs=k)
    xv.compare_list.list.addItem("S sweep")
    xv.compare_list.list.item(0).setData(_USER_ROLE, "id_s")
    xv.compare_list.list.addItem("T sweep")
    xv.compare_list.list.item(1).setData(_USER_ROLE, "id_t")
    xv.slice_at.setText("0.4")
    xv.compare_name.setText("  s_vs_t  ")
    xv.compare_note.setText("  two sweeps  ")
    xv.btn_compare.click()
    assert sent["fn"] is compare and sent["args"] == ("sweeps", ["id_s", "id_t"]), sent
    assert sent["kwargs"]["at"] == 0.4
    assert (sent["kwargs"]["name"], sent["kwargs"]["note"]) == ("s_vs_t", "two sweeps"), sent
    sent.clear()
    xv.slice_at.setText("")
    xv.btn_compare.click()
    assert "at" not in sent["kwargs"], "a blank slice point means 'the middle of the shared range'"


def test_a_comparison_from_the_screen_names_its_record_and_outlives_the_next_pick(tmp_path):
    """A comparison, end to end on the FDT screen. Two saved runs are chosen the way every record in
    the window is chosen -- a user's pick in the picker, which also opens that run's figures -- and
    Add puts each on the list; the button runs the REAL comparison on a worker. Its figure reaches
    the stack through the dispatch's fig sink, and the pane names the record it wrote (a comparison
    says what it wrote, as a run does).

    Then the figure SURVIVES the next pick. Choosing the next record for another comparison is the
    very next thing a user does, and the saved-run viewer answers every pick by closing tabs -- only
    its own, which is what keeps the comparison just drawn on the stack.

    The normalisation constant is read by one mode, so its box is live in that mode alone. None of
    the controls is persisted: a remembered list would name records a later session may have
    deleted, and a remembered constant would renormalise by a number nobody typed this time."""
    from core.artifacts import ArtifactStore, use_store
    from core.gui.panels.crossval_panel import CrossValPanel
    from core.gui.panels.fdt_panel import FdtPanel
    from tests._fixtures import PaneCapture, build_fdt_record, code_only, qt_app

    app = qt_app()
    store = ArtifactStore(tmp_path)
    with use_store(store):
        a = build_fdt_record(store, name="cell_a")
        b = build_fdt_record(store, name="cell_b", ratio=(1.0, 2.0, 1.1, 1.0))
        c = build_fdt_record(store, name="cell_c")
        panel = FdtPanel()
        pane = PaneCapture(panel)
        combo = panel.record_picker.combo

        assert panel.renorm_prefactor.isEnabled() is False
        panel.compare_mode.setCurrentText("renormalise")
        assert panel.renorm_prefactor.isEnabled() is True
        panel.compare_mode.setCurrentText("cells")
        assert panel.renorm_prefactor.isEnabled() is False

        for ref in (a, b):
            _choose(combo, combo.findData(ref))
            panel.compare_list.btn_add.click()
        assert panel.compare_list.ids() == [a, b]
        assert [panel.compare_list.list.item(i).text() for i in range(2)] == ["cell_a", "cell_b"]
        viewer_tabs = _tabs(panel)
        assert viewer_tabs, "the pick opened none of the run's figures: the survival below is vacuous"

        panel.btn_compare.click()
        _wait_for_run(app, panel, limit=60.0)
        _settle(app)
        written = [s for s in store.list("fdt") if s.study == "comparison"]
        assert len(written) == 1, [(s.name, s.study) for s in store.list("fdt")]
        assert ("info", f"Comparison record written: (unnamed) [{written[0].id}].") in pane.lines, \
            pane.lines
        assert _tabs(panel) == [*viewer_tabs, "FDT ratio by cell"], _tabs(panel)

        _choose(combo, combo.findData(c))
        assert _tabs(panel)[0] == "FDT ratio by cell" and len(_tabs(panel)) == 1 + len(viewer_tabs), \
            f"the next pick closed the comparison, or kept the last run's tabs: {_tabs(panel)}"

    for cls in (FdtPanel, CrossValPanel):
        for method in (cls.save_settings, cls.restore_settings):
            src = code_only(method)
            for attr in ("compare_list", "compare_mode", "renorm_prefactor", "slice_at",
                         "compare_name", "compare_note"):
                assert attr not in src, f"{cls.__name__}.{method.__name__} persists {attr}"


def test_a_comparison_from_either_screen_is_named_and_noted_and_a_taken_name_names_its_box(
        tmp_path, monkeypatch):
    """The window's half of naming a comparison. Both "Compare saved ..." groups dispatched their
    comparison with neither a name nor a note, and nothing renames an fdt record afterwards -- so
    every comparison the window drew was "(unnamed)" for good, while the tool's ``compare`` took
    ``--name``. Each group now has a Comparison name and a Comparison note row of its own (the run's
    Record name box names the RUN), passed to the comparison.

    The note is judged at the click by the house rule (one line, at most NOTE_MAX_CHARS): a refused
    note is one yellow box and no dispatch. A taken name is the stage's refusal, raised on the worker
    by ``store.create``; its yellow box must name THIS box, so the ``name`` and ``note`` sentences
    list the comparison's boxes beside the run's. Neither box is ever written to PRISM.ini (a
    remembered name would be refused as taken at the next launch) -- the pin is in
    test_a_comparison_from_the_screen_names_its_record_and_outlives_the_next_pick."""
    import pytest
    from PySide6.QtWidgets import QListWidgetItem
    from core.artifacts import ArtifactStore, use_store
    from core.gui import fields as gui_fields
    from core.gui.panels.crossval_panel import CrossValPanel
    from core.gui.panels.fdt_panel import FdtPanel
    from core.refusals import NOTE_MAX_CHARS
    from tests._fixtures import SHOWN, build_fdt_record, qt_app

    assert "'Comparison name'" in gui_fields.fix_sentence("name"), gui_fields.fix_sentence("name")
    assert "'Comparison note'" in gui_fields.fix_sentence("note"), gui_fields.fix_sentence("note")

    app = qt_app()
    store = ArtifactStore(tmp_path)
    om = (0.5, 1.0, 2.0)
    with use_store(store):
        cells = [build_fdt_record(store, name="cell_a"),
                 build_fdt_record(store, name="cell_b", ratio=(1.0, 2.0, 1.1, 1.0))]
        sweeps = [build_fdt_record(store, name="s_low", study="sweep", omegas=om, sweep_param="s",
                                   points=[(0.0, (1.0, 1.0, 1.0)), (0.5, (1.0, 3.0, 1.1))]),
                  build_fdt_record(store, name="s_high", study="sweep", omegas=om, sweep_param="s",
                                   points=[(0.25, (1.0, 2.0, 1.0)), (0.75, (1.0, 6.0, 1.2))])]
        for panel, refs, name in ((FdtPanel(), cells, "ab_cells"), (CrossValPanel(), sweeps, "lo_hi")):
            for ref in refs:
                item = QListWidgetItem(ref)
                item.setData(_USER_ROLE, ref)
                panel.compare_list.list.addItem(item)
            panel.compare_name.setText(name)
            panel.compare_note.setText("  drawn from the screen  ")
            SHOWN.clear()
            panel.btn_compare.click()
            _wait_for_run(app, panel, limit=60.0)
            m = store.get("fdt", name)
            assert (m.name, m.note, m.body["study"]) == (name, "drawn from the screen", "comparison"), m
            assert SHOWN == [], [b.text() for b in SHOWN]

            # the same name again: refused by the stage, and the box names the comparison's box
            panel.btn_compare.click()
            _wait_for_run(app, panel, limit=60.0)
            box = SHOWN[-1]
            assert box.windowTitle() == "Check your inputs" and "already exists" in box.text(), box.text()
            assert box.informativeText() == gui_fields.fix_sentence("name"), box.informativeText()

            # a note the house rule refuses: one box at the click, nothing dispatched
            panel.compare_name.setText(f"{name}_again")
            panel.compare_note.setText("n" * (NOTE_MAX_CHARS + 1))
            monkeypatch.setattr(panel, "dispatch", lambda *a, **k: pytest.fail("a refused note ran"))
            SHOWN.clear()
            panel.btn_compare.click()
            assert len(SHOWN) == 1 and SHOWN[0].windowTitle() == "Check your inputs", \
                [b.text() for b in SHOWN]
            assert SHOWN[0].informativeText() == gui_fields.fix_sentence("note")
    assert len([s for s in store.list("fdt") if s.study == "comparison"]) == 2


def test_a_half_typed_number_is_refused_never_read_as_blank(tmp_path, monkeypatch):
    """A numeric box's validator accepts '-', '1e', '.' and '+' as text still being typed, and
    value_or_none() reads each as None -- which each of these boxes takes as "blank": the Slice at
    box then sliced at the MIDDLE of the shared range, a Seed box DREW a seed, and the renormalise
    constant was refused as "blank" although the box was not. Text that is not empty and does not
    parse is refused at the click, naming the box's setting: one yellow box, and nothing dispatched
    or written. The two list rows' help names the picker by its row."""
    from core.artifacts import ArtifactStore, use_store
    from core.gui import fields as gui_fields
    from core.gui.panels import crossval_panel as xv_mod
    from core.gui.panels import fdt_panel as fdt_mod
    from tests._fixtures import SHOWN, qt_app

    qt_app()
    store = ArtifactStore(tmp_path)
    monkeypatch.setattr(store, "create", lambda *a, **k: pytest.fail("a half-typed box minted a record"))
    with use_store(store):
        fdt, xv = fdt_mod.FdtPanel(), xv_mod.CrossValPanel()
        for p in (fdt, xv):
            p.dispatch = lambda *a, **k: pytest.fail("a half-typed box dispatched anyway")

        def refused(click, key, typed):
            SHOWN.clear()
            click()
            assert len(SHOWN) == 1, (key, [b.text() for b in SHOWN])
            box = SHOWN[0]
            assert box.windowTitle() == "Check your inputs", box.windowTitle()
            assert repr(typed) in box.text(), box.text()
            assert box.informativeText() == gui_fields.fix_sentence(key), box.informativeText()

        xv.slice_at.setText("-")
        refused(xv._compare, "slice_at", "-")
        fdt.compare_mode.setCurrentText("renormalise")
        fdt.renorm_prefactor.setText("1e")
        refused(fdt._compare, "prefactor", "1e")
        for panel in (fdt, xv):
            panel.seed.setText("-")
            refused(panel._run, "seed", "-")

    assert "'Saved run'" in fdt_mod.HELP["compare_list"] and "above" in fdt_mod.HELP["compare_list"]
    assert "'Saved sweep'" in xv_mod.HELP["compare_list"]


def test_every_comparison_refusal_reaches_the_yellow_box_naming_its_control(tmp_path):
    """Every refusal a comparison can make from what it was given is the STAGE's, raised on the
    worker -- the panel checks nothing -- and it must still reach the yellow "Check your inputs" box
    through BasePanel._on_error, with the line under it naming the control that answers it: the list
    for a choice of runs, the 'Normalisation constant' box, the 'Slice at' box. Driven through the
    real button, dispatch and worker, over every refusal of the per-mode pre-flight
    (``compare_preflight_refusals``) plus the two a screen reaches first: one run where two are
    needed, and a blank constant. None of them may leave a comparison record behind.

    The three sentences are pinned verbatim. ``prefactor`` and ``slice_at`` became (place, label)
    entries when their boxes were built, and the words the box shows did not change."""
    from PySide6.QtWidgets import QListWidgetItem
    from core.artifacts import ArtifactStore, use_store
    from core.gui import fields as gui_fields
    from core.gui.panels.crossval_panel import CrossValPanel
    from core.gui.panels.fdt_panel import FdtPanel
    from tests._fixtures import SHOWN, build_fdt_record, compare_preflight_refusals, qt_app

    fixes = {
        # the list quoted by its own row, and "Choose" -- the fix for "at most 2" or "named twice"
        # is a removal, never an "Add"
        "compare_records": ("Choose the records in the 'Runs to compare' list on the FDT analysis "
                            "tab, or in the 'Sweeps to compare' list on the Sweep study "
                            "cross-validation tab."),
        "prefactor": "Set it in the 'Normalisation constant' box on the FDT analysis tab.",
        "slice_at": "Set it in the 'Slice at' box on the Sweep study cross-validation tab.",
    }
    assert gui_fields.CONTROL["prefactor"] == ("FDT analysis", "Normalisation constant")
    assert gui_fields.CONTROL["slice_at"] == ("Sweep study cross-validation", "Slice at")

    app = qt_app()
    store = ArtifactStore(tmp_path)
    with use_store(store):
        a = build_fdt_record(store, name="a")
        cases = compare_preflight_refusals(store) + [
            ("one run for a cells comparison", "cells", [a], {}, "compare_records", "at least 2"),
            ("a blank constant", "renormalise", [a], {"prefactor": None}, "prefactor", "blank"),
        ]
        fdt, xv = FdtPanel(), CrossValPanel()
        for label, mode, refs, options, field, words in cases:
            panel = xv if mode == "sweeps" else fdt
            panel.compare_list.list.clear()
            for ref in refs:
                item = QListWidgetItem(ref)
                item.setData(_USER_ROLE, ref)
                panel.compare_list.list.addItem(item)
            if mode == "sweeps":
                at = options.get("at")
                panel.slice_at.setText("" if at is None else repr(at))
            else:
                panel.compare_mode.setCurrentText(mode)
                prefactor = options.get("prefactor")
                panel.renorm_prefactor.setText("" if prefactor is None else repr(prefactor))
            SHOWN.clear()
            panel.btn_compare.click()
            _wait_for_run(app, panel, limit=60.0)
            assert len(SHOWN) == 1, f"{label}: {len(SHOWN)} boxes, not one"
            box = SHOWN[-1]
            assert box.windowTitle() == "Check your inputs", (label, box.windowTitle(), box.text())
            assert words in box.text(), (label, box.text())
            assert box.informativeText() == fixes[field], (label, box.informativeText())
        written = [s.name for s in store.list("fdt") if s.study == "comparison"]
        assert written == [], f"a refused comparison left a record behind: {written}"
