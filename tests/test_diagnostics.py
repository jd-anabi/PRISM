"""Diagnostics: the ``diagnostic`` store kind, the shared feature-set and seeding helpers, and the
five diagnostic functions (piece 2 of the 2026-09-11 one-flow design, tasks T13-T18).

Everything here is CPU-sized and stubbed where a simulation would otherwise dominate: a diagnostic is
a measurement ABOUT a trained posterior, so the thing worth pinning is what it reads, what it refuses
and what it writes -- not the numbers a real training run would give it.
"""
import pytest

from core.artifacts import LoadedDiagnostic
from core.artifacts import store as st
from tests._fixtures import _nad_cfg, _posterior_artifact


class _LatentPriorStub:
    """The ``SBIPriorWrapper(latent)`` shape a trained posterior pickles: ``.gen_dist`` is the latent
    training prior, which is what the off-ground-truth points are drawn from."""
    def __init__(self, dim):
        import torch
        self.gen_dist = torch.distributions.Independent(
            torch.distributions.Normal(torch.zeros(dim), torch.ones(dim)), 1)


def _diagnostic(store, cfg, *, name="diag", parents=None, variant=None):
    """A diagnostic artifact written the way every diagnostic function writes one (spec 4.1)."""
    with store.create("diagnostic", cfg, name=name) as w:
        w.parents = dict(parents or {})
        w.config.update({"repeats": 2})
        w.body = {"diagnostic": "sbc", "variant": variant, "settings": {"repeats": 2},
                  "results": {"n_valid": 8, "accepted": []}}
    return w


def test_a_diagnostic_loads_without_a_config_and_blocks_deleting_what_it_names(store):
    """D5, the whole contract of the kind: its own directory and manifest body; a loader that takes a
    ref and NOTHING else, because there is no config to verify a measurement against and nothing is
    ever trained from one; and, because ``dependents`` walks every kind, the artifacts it names cannot
    be deleted out from under it without ``force``.
    """
    import inspect
    cfg = _nad_cfg()
    post = _posterior_artifact(store, cfg, name="p")
    w = _diagnostic(store, cfg, name="sbc_rep", parents={"posterior": post.id}, variant="k4")

    assert store.kind_dir("diagnostic") == store.root / "diagnostics"
    assert w.dir.parent.name == "diagnostics" and w.dir.name == f"sbc_rep__{w.id}"

    assert list(inspect.signature(store.load_diagnostic).parameters) == ["ref"], \
        "load_diagnostic takes a ref only: no cfg to check against, no Accept to pass"
    d = store.load_diagnostic("sbc_rep")
    assert isinstance(d, LoadedDiagnostic) and d.kind == "diagnostic" and d.id == w.id
    assert (d.diagnostic, d.variant) == ("sbc", "k4")
    assert d.settings == {"repeats": 2} and d.results == {"n_valid": 8, "accepted": []}
    assert store.load_diagnostic(w.id).id == w.id                 # by id as well as by name
    with pytest.raises(st.StoreError, match="no complete diagnostic"):
        store.load_diagnostic("nope")

    assert store.dependents("posterior", post.id) == [("diagnostic", w.id, "sbc_rep")]
    with pytest.raises(st.StoreError, match="diagnostic sbc_rep"):
        store.delete("posterior", post.id)
    store.delete("posterior", post.id, force=True)
    assert store.load_diagnostic("sbc_rep").id == w.id            # the measurement itself survives


def test_the_chi_feature_set_drops_group_g_and_adds_the_fisher_block(capsys):
    """The whole reason these helpers exist: a diagnostic built over the 41-feature single-frequency
    set while the posterior conditions on chi answers a question about a DIFFERENT experiment -- it
    reports the kappa~x_scale / lambda~t_scale aliases as strong as ever and falsely refutes the very
    hypothesis chi mode exists to test. In chi mode the rows are the 30 non-G summary features plus
    the three FISHER channels per probe, not the six CONDITIONING ones (whose u and mask columns are
    theta-independent in a central difference).
    """
    from core.SBI import chi as chi_mod
    from core.SBI.statistics import FEATURE_LABELS
    from core.diagnostics import feature_sets as fs

    plain = _nad_cfg()
    assert fs.feature_labels(plain) == list(FEATURE_LABELS)
    assert fs.n_features(plain) == len(FEATURE_LABELS) == 41

    keep = fs.summary_keep_idx()
    assert len(keep) == len(FEATURE_LABELS) - 11 == 30
    assert not any(FEATURE_LABELS[i].startswith("G") for i in keep)

    chi_cfg = _nad_cfg(chi_mode=True, chi_n_freqs=4)
    labels = fs.feature_labels(chi_cfg)
    assert labels[:30] == [FEATURE_LABELS[i] for i in keep]
    assert labels[30:] == chi_mod.chi_labels(4, chi_mod.CHI_FISHER_CHANNELS)
    assert labels[30] == "chi0_logmag" and len(labels) == 30 + 3 * 4 == fs.n_features(chi_cfg)

    fs.describe_features(chi_cfg)
    out = capsys.readouterr().out
    assert "[mode] CHI: feature rows = 30 spontaneous + 12 chi = 42" in out, out
    assert "f_scale is informative here" in out
    fs.describe_features(plain)
    assert f"[mode] {plain.observation_mode.upper()}: feature rows = 41" in capsys.readouterr().out


def test_the_diagnostic_guards_refuse_with_value_errors():
    """SystemExit was right for a script and wrong everywhere else. These guards now run inside the
    command-line tool, where a SystemExit would walk straight past main's exit-code table, and inside
    a GUI worker, where it would take the application down. They raise ValueError, and their messages
    name flags and subcommands, because there are no environment variables left to name.
    """
    import inspect
    from core import cli, config, registry
    from core.config import VALID_LABELS, VALID_MODELS
    from core.diagnostics import feature_sets as fs

    chi_cfg = _nad_cfg(chi_mode=True, chi_n_freqs=4)
    with pytest.raises(ValueError, match="has not been generalised to chi") as e:
        fs.assert_not_chi(chi_cfg, "identifiability laplace")
    assert "identifiability laplace" in str(e.value) and "--no-chi" in str(e.value)
    fs.assert_not_chi(_nad_cfg(), "identifiability laplace")            # not chi: no refusal

    other = _nad_cfg()
    other.model = "HOPF"
    with pytest.raises(ValueError, match="Nadrowski-specific") as e:
        fs.assert_nadrowski(other, "the printed ND parameter names are Nadrowski's")
    assert "HOPF" in str(e.value) and "--bounds" in str(e.value)
    fs.assert_nadrowski(_nad_cfg())                                     # NADROWSKI: no refusal

    spont = cli.make_sim_config("NADROWSKI", VALID_LABELS[VALID_MODELS.index("NADROWSKI")],
                                registry.state_dep_drift("NADROWSKI"),
                                str(config.BOUNDS_PATH / "nadrowski" / "master_spont.txt"))
    assert not spont.has_forcing, "master_spont.txt declares no Forcing section"
    with pytest.raises(ValueError, match="no amp/freq/phase to read") as e:
        fs.assert_forced(spont, "identifiability jacobian")
    assert "master_weak.txt" in str(e.value) and "--cell" in str(e.value)
    fs.assert_forced(_nad_cfg(), "identifiability jacobian")            # master.txt declares Forcing

    for guard in (fs.assert_not_chi, fs.assert_nadrowski, fs.assert_forced):
        assert "SystemExit" not in inspect.getsource(guard), guard.__name__


def test_seeded_restores_the_callers_rng():
    """Seed ONCE, let the stream run on inside the block, and hand the caller's RNG back on the way
    out. The tool's own tests call main(argv) in-process, so a leaked seed would make one test's
    numbers depend on which tests ran before it; and re-seeding per stage is what would make the
    calibration set replay the training strata (trap X5), which is why there is one context and not a
    seed argument on every stage.
    """
    import numpy as np
    import torch
    from core import config
    from core.diagnostics.rng import seeded

    dev = config.cpu_device().device
    torch.manual_seed(1234)
    np.random.seed(1234)
    before_t, before_n = torch.randn(3), np.random.rand(3)

    torch.manual_seed(1234)
    np.random.seed(1234)
    with seeded(7, dev):
        a_t, a_n = torch.randn(3), np.random.rand(3)
        onward_t = torch.randn(3)                     # the stream RUNS ON inside the block
    after_t, after_n = torch.randn(3), np.random.rand(3)
    assert torch.equal(after_t, before_t) and np.array_equal(after_n, before_n), \
        "seeded() did not hand the caller's RNG back"

    with seeded(7, dev):
        b_t, b_n = torch.randn(3), np.random.rand(3)
        assert torch.equal(torch.randn(3), onward_t), "the same seed must replay the same stream"
    assert torch.equal(a_t, b_t) and np.array_equal(a_n, b_n)
    assert not torch.equal(a_t, before_t), "the seed inside the block must actually take effect"


def test_the_calibration_draw_is_three_helpers_with_the_stratification_seam(store, monkeypatch):
    """T16's sbc_repeats draws its per-repeat calibration set through EXACTLY the code
    validate_calibration draws its own through. That is what gives the repeat-SBC run the four things
    scripts/sbc_characterize.py never had: check_basis, the t_scale-override mirror in the reference
    sample, the kept-fraction line and the LoadedPrior path. A second copy of the wrap is precisely how
    a repeat-SBC run comes to draw theta* from the FULL prior while the flow was trained on the region
    -- guardrail 8, silently inverted.

    So the draw is three named helpers, and the one thing sbc_repeats varies -- chi_k_fixed, which
    runs one probe-count stratum at a time -- is a keyword on the middle one. validate_calibration
    itself always passes None: its SBC is the POOLED one, over the same mixture of counts training saw.
    """
    import ast
    import contextlib
    import inspect
    import io
    import textwrap
    from types import SimpleNamespace

    import torch
    from core import orchestrator as orch
    from core.SBI import reparam as _rp, training_checkpoint as _tc, truncate as _tr
    from tests._fixtures import _prior_artifact

    cfg = _nad_cfg(reparam_rotate=False)
    lp = store.load_prior(cfg, _prior_artifact(store, cfg, name="p").id)
    P = len(cfg.params_dict) + len(cfg.rescale_params)
    i_t = len(cfg.params_dict) + cfg.rescale_idx["t_scale"]
    T = orch.build_inferred_bijection(cfg, log_params=orch._log_params_for(cfg))
    with torch.no_grad():
        z = orch._build_latent_prior_for_validation(cfg, lp.prior).sample((512,)).double()
    region = _tr.TruncationRegion([0, 1], [float(z[:, 0].quantile(0.25)), float(z[:, 1].quantile(0.25))],
                                  [float(z[:, 0].quantile(0.75)), float(z[:, 1].quantile(0.75))],
                                  n_latent=P, V=None, probe=_tc.bijection_probe(T, P))

    class _Lat:
        prior = None

        def sample(self, shape, x=None, **k):
            return torch.randn(int(torch.Size(shape).numel()), P)

        sample_batched = sample

    trunc_post = SimpleNamespace(posterior=_rp.TransformedPosterior(_Lat(), T, truncation=region))
    plain_post = SimpleNamespace(posterior=_rp.TransformedPosterior(_Lat(), T))

    # (a) check_basis must actually run for a truncated draw -- nothing else in this test would
    # notice if _calibration_prior silently dropped the call.
    basis_calls = []
    real_check_basis = region.check_basis

    def _recording_check_basis(*a, **k):
        basis_calls.append((a, k))
        return real_check_basis(*a, **k)

    monkeypatch.setattr(region, "check_basis", _recording_check_basis)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        vlp, T_out, truncation = orch._calibration_prior(cfg, trunc_post, lp)
    assert truncation is region and T_out is trunc_post.posterior.T
    assert isinstance(vlp, _tr.TruncatedLatentPrior) and vlp.region is region
    assert "PRIOR RESTRICTED" in buf.getvalue(), buf.getvalue()
    assert len(basis_calls) == 1 and basis_calls[0][0][0] is T_out, \
        "_calibration_prior must call check_basis exactly once, with the posterior's own T"

    vlp_plain, _, trunc_plain = orch._calibration_prior(cfg, plain_post, lp)
    assert trunc_plain is None and not isinstance(vlp_plain, _tr.TruncatedLatentPrior)

    # (b) the rotation wrap: with a stub V, the returned prior must come back wrapped in
    # RotatedLatentPrior. Plain posterior, so check_basis (and its call count above) is untouched.
    stub_V = torch.eye(P)
    monkeypatch.setattr(orch, "rotation_of", lambda _T: stub_V)
    vlp_rot, _, trunc_rot = orch._calibration_prior(cfg, plain_post, lp)
    assert trunc_rot is None
    assert isinstance(vlp_rot, _rp.RotatedLatentPrior) and vlp_rot.V is stub_V, \
        "_calibration_prior must wrap the calibration prior in RotatedLatentPrior when rotation_of(T) is not None"

    seen = {}

    def _gen_cal(**kw):
        seen.update(kw)
        return torch.zeros(6, 3), torch.randn(6, P)

    monkeypatch.setattr(orch.analysis, "gen_cal_data", _gen_cal)
    x_cal, theta_star = orch._draw_calibration_set(cfg, vlp, T, lp.force_prior, n_cal=6,
                                                   cal_n_scales=1, chi_k_fixed=4)
    assert seen["prior"] is vlp and seen["theta_transform"] is T and seen["n_cal"] == 6
    assert seen["cal_n_scales"] == 1 and seen["chi_k_fixed"] == 4
    assert x_cal.shape == (6, 3) and theta_star.shape == (6, P)
    assert inspect.signature(orch._draw_calibration_set).parameters["chi_k_fixed"].default is None

    # (c) validate_calibration must actually CALL the three helpers, not merely mention their names
    # in a comment -- an AST check over calls, not a text search over the source.
    tree = ast.parse(textwrap.dedent(inspect.getsource(orch.validate_calibration)))
    calls_by_name = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            calls_by_name.setdefault(node.func.id, []).append(node)
    for name in ("_calibration_prior", "_draw_calibration_set", "_sbc_reference_sample"):
        assert name in calls_by_name and len(calls_by_name[name]) == 1, \
            f"validate_calibration must call {name} exactly once"
    draw_call = calls_by_name["_draw_calibration_set"][0]
    chi_kw = {kw.arg: kw.value for kw in draw_call.keywords}
    assert "chi_k_fixed" in chi_kw and isinstance(chi_kw["chi_k_fixed"], ast.Constant) \
        and chi_kw["chi_k_fixed"].value is None, \
        "validate_calibration's SBC is the pooled one and must pass chi_k_fixed=None explicitly"

    ref = orch._sbc_reference_sample(cfg, vlp, T, region, lp.prior, theta_star)
    assert ref.shape == theta_star.shape
    assert torch.equal(ref[:, i_t].sort().values, theta_star[:, i_t].sort().values), \
        "the reference sample must mirror the per-batch t_scale override"
    plain_ref = orch._sbc_reference_sample(cfg, vlp_plain, T, None, lp.prior, theta_star)
    assert plain_ref.shape[0] == theta_star.shape[0]
    # (d) the plain branch draws straight from the prior; without a region there is no override
    # to mirror, so its t_scale column must NOT be a permutation of theta*'s.
    assert not torch.equal(plain_ref[:, i_t].sort().values, theta_star[:, i_t].sort().values), \
        "the plain branch must not mirror theta*'s t_scale column -- there is no region to mirror it against"


def test_sbc_writes_one_diagnostic_naming_its_posterior_and_prior(tiny_run):
    """The repeat study is one artifact: K x n_cal calibration sets, the per-repeat KS/C2ST tables in a
    payload, one pooled figure, and a manifest naming the posterior and the prior it was trained from.
    Seeded per repeat (seed + r) under core.diagnostics.rng.seeded, so a second run at the same seed
    reproduces the table exactly -- which is the whole point of a repeat study: the spread across
    repeats is the measurement, and it is worthless if the spread is partly the caller's RNG state."""
    import numpy as np
    from matplotlib import pyplot as plt
    from core.diagnostics import sbc_repeats
    r = tiny_run
    # tiny_run.sink is a no-op (it does not close figures); the writer's own fig_sink saves the PNG
    # and then forwards to whichever sink was given, so a no-op sink here would leak one matplotlib
    # figure per call -- two, across this test's two calls.
    close = lambda title, fig: plt.close(fig)                       # noqa: E731
    d = sbc_repeats(r.cfg, r.posterior, r.prior, repeats=2, n_cal=8, num_posterior_samples=40,
                    cal_n_scales=1, seed=0, fig_sink=close, name="sbc_two")
    keys = list(r.cfg.params_dict) + list(r.cfg.rescale_params)
    m = d.manifest
    assert d.diagnostic == "sbc" and d.variant == "pooled"
    assert m.parents == {"posterior": r.posterior.id, "prior": r.prior.id}
    assert m.fingerprints["gmm"] == r.prior.fingerprint
    assert m.config["repeats"] == 2 and m.config["n_cal"] == 8 and m.config["seed"] == 0
    assert [p["name"] for p in d.results["per_param"]] == keys
    assert set(d.results["per_param"][0]) == {"name", "ks_p_median", "ks_p_min", "frac_ks_below_05",
                                              "c2st_ranks_median"}
    # n_cal is an UPPER bound: gen_cal_data drops rows whose simulation was invalid, which is why the
    # diagnostic records n_valid at all. Assert the invariant, not the count.
    assert len(d.results["n_valid"]) == 2 and all(0 < n <= 8 for n in d.results["n_valid"])
    assert d.results["kept_fraction"] is None
    assert d.results["accepted"] == []
    assert set(m.payloads) == {"sbc_repeats.npz"}
    assert m.figures == ["figures/sbc_ranks_pooled_over_repeats_histogram.png"]
    z = np.load(d.path / "sbc_repeats.npz")
    assert z["ks"].shape == (2, len(keys)) and z["c2st_ranks"].shape == (2, len(keys))
    assert z["c2st_dap"].shape == (2, len(keys))
    assert z["ranks"].shape == (sum(d.results["n_valid"]), len(keys))
    assert z["repeat"].tolist() == ([0] * d.results["n_valid"][0] + [1] * d.results["n_valid"][1])
    assert int(z["nps"]) == 40
    assert [str(s) for s in z["labels"]] == keys
    assert r.store.load_diagnostic(d.id).results == d.results
    # The spread across repeats IS the measurement: a loop that reused seeded(seed) for every repeat
    # (zero spread) or that restored the RNG but never reseeded it would still pass every assertion
    # above -- shapes, counts and keys are all identical either way. Only comparing the repeats'
    # actual rank rows catches it. n_valid can differ between repeats (gen_cal_data drops invalid
    # rows independently each time), so the comparison is truncated to the shorter of the two rather
    # than assuming equal lengths.
    n0, n1 = d.results["n_valid"]
    ranks0, ranks1 = z["ranks"][:n0], z["ranks"][n0:n0 + n1]
    n_common = min(ranks0.shape[0], ranks1.shape[0])
    assert not np.array_equal(ranks0[:n_common], ranks1[:n_common]), \
        "the two repeats produced identical rank rows -- seeded() must reseed EACH repeat at " \
        "seed + r, not reuse one seed, or the repeat spread this diagnostic exists to measure is gone"
    again = sbc_repeats(r.cfg, r.posterior, r.prior, repeats=2, n_cal=8, num_posterior_samples=40,
                        cal_n_scales=1, seed=0, fig_sink=close)
    assert np.allclose(np.load(again.path / "sbc_repeats.npz")["ks"], z["ks"], equal_nan=True), \
        "the same seed must reproduce the KS table; the repeat spread is the measurement"


def test_sbc_refuses_before_the_spend(tiny_run, monkeypatch):
    """Every refusal fires before a single calibration set is simulated -- n_cal x repeats simulations
    is the spend this diagnostic exists to characterise, and learning that the name is taken (or that
    CHI_K_FIXED means nothing outside chi mode) afterwards throws all of it away.

    Independent of test order (its own taken name, not the previous test's ``sbc_two``), and proves
    the refusals fire before ``_calibration_prior`` -- not merely before the simulation ``gen_cal_data``
    would run -- which is what actually distinguishes ``sbc_repeats``' own early ``assert_name_free``
    from the refusal ``store.create`` would raise much later, after the whole calibration set."""
    import pytest
    from core import orchestrator
    from core.artifacts import StoreError
    from core.diagnostics import sbc_repeats
    r = tiny_run
    before = len(r.store.list("diagnostic"))
    _diagnostic(r.store, r.cfg, name="sbc_taken")          # our own taken name, not the prior test's
    drawn, calibrated = [], []
    monkeypatch.setattr(orchestrator.analysis, "gen_cal_data", lambda **k: drawn.append(1))
    monkeypatch.setattr(orchestrator, "_calibration_prior", lambda *a, **k: calibrated.append(1))
    with pytest.raises(ValueError, match="repeats must be at least 1"):
        sbc_repeats(r.cfg, r.posterior, r.prior, repeats=0, n_cal=8, fig_sink=r.sink)
    with pytest.raises(ValueError, match="n_cal must be at least 1"):
        sbc_repeats(r.cfg, r.posterior, r.prior, repeats=1, n_cal=0, fig_sink=r.sink)
    with pytest.raises(ValueError, match="num_posterior_samples must be at least 1"):
        sbc_repeats(r.cfg, r.posterior, r.prior, repeats=1, n_cal=8, num_posterior_samples=0,
                    fig_sink=r.sink)
    with pytest.raises(ValueError, match="chi_k_fixed"):
        sbc_repeats(r.cfg, r.posterior, r.prior, repeats=1, n_cal=8, chi_k_fixed=2, fig_sink=r.sink)
    with pytest.raises(StoreError, match="already exists"):
        sbc_repeats(r.cfg, r.posterior, r.prior, repeats=1, n_cal=8, fig_sink=r.sink, name="sbc_taken")
    with pytest.raises(ValueError, match="not the one this posterior was trained with"):
        sbc_repeats(r.cfg, r.posterior, r.other_prior(), repeats=1, n_cal=8, fig_sink=r.sink)
    assert drawn == [], "a calibration set was simulated before the refusals"
    assert calibrated == [], \
        "_calibration_prior ran before a refusal fired -- every guard above must precede it"
    assert len(r.store.list("diagnostic")) == before + 1, \
        "a refused sbc_repeats call wrote a diagnostic of its own (only the name taken above should exist)"


def _rotation_posterior(store, cfg, *, V, evals):
    """A posterior artifact whose transform block records V (columns) and its eigenvalues."""
    from core.artifacts import manifest as mf
    from tests._fixtures import _posterior_artifact
    over = {("transform", "fisher_eigenvalues"): None if evals is None else [float(v) for v in evals]}
    return _posterior_artifact(store, cfg, name="rot", V=V, over=over)


def test_identifiability_rotation_decomposes_a_stored_basis(store):
    """What did this artifact MEASURE? The Fisher eigenbasis is on disk already, so the answer costs
    no simulation: which directions carry the information, what each is made of, and which parameters
    are their OWN near-null direction (flat, not aliased -- no reparameterisation reaches those)."""
    import math
    import torch
    from core.diagnostics import identifiability_rotation
    from tests._fixtures import _nad_cfg
    cfg = _nad_cfg()
    P = len(cfg.params_dict) + len(cfg.rescale_params)
    names = list(cfg.params_dict) + list(cfg.rescale_params)
    # A permutation basis: direction j is exactly parameter order[j], so every share is 0 or 1 and the
    # arithmetic below is checkable by hand. order[-1] is the WORST direction and its parameter is flat.
    order = list(range(P))
    V = torch.zeros(P, P, dtype=torch.float64)
    for j, i in enumerate(order):
        V[i, j] = 1.0
    evals = [10.0 ** (1 - k) for k in range(P)]      # descending, and the top one IS 10.0
    w = _rotation_posterior(store, cfg, V=V, evals=evals)
    lp = store.load_posterior(cfg, w.id)
    d = identifiability_rotation(cfg, lp, n_worst=1, top_n=2, name="rot1")
    res = d.results
    assert d.diagnostic == "identifiability" and d.variant == "rotation"
    assert d.manifest.parents == {"posterior": lp.id} and res["P"] == P
    assert res["orthogonality"] < 1e-12
    assert res["eigenvalues"][0] == 10.0 and len(res["directions"]) == P
    assert res["directions"][0]["index"] == 0 and res["directions"][0]["eigenvalue"] == 10.0
    assert res["directions"][0]["loadings"][0] == {"name": names[order[0]], "loading": 1.0}
    assert [p["name"] for p in res["per_param"]] == names
    worst = next(p for p in res["per_param"] if p["name"] == names[order[-1]])
    assert worst["bottom_share"] == 1.0 and worst["peak_dir"] == P - 1 and worst["verdict"] == "UNMEASURED"
    assert [a["name"] for a in res["flat_axes"]] == [names[order[-1]]]
    assert res["flat_axes"][0]["partners"] == [] and res["accepted"] == []
    assert d.manifest.config["n_worst"] == 1 and d.manifest.config["top_n"] == 2
    assert d.manifest.payloads == {} and d.manifest.figures == []


def test_identifiability_rotation_reports_absent_eigenvalues_and_refuses_an_absent_rotation(store, capsys):
    """Eigenvalues absent is a REPORT: every TSNPE round carries None (it reuses the parent's V and
    never runs a Fisher), and the loadings still answer "which direction is worst". V absent is a
    REFUSAL: there is no basis to decompose at all."""
    import pytest
    import torch
    from core.diagnostics import identifiability_rotation
    from tests._fixtures import _nad_cfg
    cfg = _nad_cfg()
    P = len(cfg.params_dict) + len(cfg.rescale_params)
    w = _rotation_posterior(store, cfg, V=torch.eye(P, dtype=torch.float64), evals=None)
    d = identifiability_rotation(cfg, store.load_posterior(cfg, w.id), name="rot_noev")
    assert d.results["eigenvalues"] is None and len(d.results["directions"]) == P
    assert d.results["directions"][0]["eigenvalue"] is None
    assert "NOT STORED" in capsys.readouterr().out
    from tests._fixtures import _posterior_artifact
    flat = _posterior_artifact(store, cfg, name="norot", V=None)
    with pytest.raises(ValueError, match="records no Fisher rotation"):
        identifiability_rotation(cfg, store.load_posterior(cfg, flat.id), name="rot_none")
    assert [s.name for s in store.list("diagnostic") if s.name == "rot_none"] == []


def _forced_nad_cfg():
    """A FORCED Nadrowski config off master_weak -- the pairing the simulating identifiability
    diagnostics need: whether a drive EXISTS is a property of the bounds file, and the spontaneous
    master resolves to a box with no amp/freq/phase to read."""
    from core import cli, config, registry
    from core.config import VALID_LABELS, VALID_MODELS
    labels = VALID_LABELS[VALID_MODELS.index("NADROWSKI")]
    cell = str(config.CELL_PATH / "nadrowski" / "master_weak.txt")
    # master.txt is the FORCED box (it declares amp/freq/phase/offset, which is what assert_forced
    # reads); there is no Bounds/nadrowski/master_weak.txt -- master_weak is a CELL only, and every
    # existing suite pairs that cell with these bounds.
    cfg = cli.make_sim_config("NADROWSKI", labels, registry.state_dep_drift("NADROWSKI"),
                              str(config.BOUNDS_PATH / "nadrowski" / "master.txt"))
    cfg.hw = config.cpu_device()
    cli.load_and_validate_gt(cfg, cell)
    return cfg


def test_identifiability_laplace_reports_sd_per_point_with_its_unit(store, monkeypatch):
    """The Laplace marginal SD is in PRIOR-RANGE units, and which unit (log-range or linear range) is
    chosen by the parameter's name, not by the posterior's box -- so each record has to say which, or
    two numbers in one table mean different things. The simulation is replaced: what is under test is
    the point construction, the per-point table and the artifact, not the solver."""
    import numpy as np
    import torch
    from core.diagnostics import identifiability, identifiability_laplace
    from tests._fixtures import _posterior_artifact
    cfg = _forced_nad_cfg()
    P = len(cfg.params_dict) + len(cfg.rescale_params)
    V = torch.eye(P, dtype=torch.float64)
    w = _posterior_artifact(store, cfg, name="lap_post", V=V,
                            prior=_LatentPriorStub(P))
    lp = store.load_posterior(cfg, w.id)
    calls = []

    def _fake_raw(cfg_, nd, res, force, m, crn, n_obs):
        calls.append((int(m), bool(crn), int(n_obs)))
        n_feat = identifiability.feature_sets.n_features(cfg_)
        g = torch.zeros(int(m), 8, dtype=torch.float64)
        feats = np.tile(np.arange(n_feat, dtype=float) + float(nd.sum()), (int(m), 1))
        return feats, g + 1.0, g + 1.0

    monkeypatch.setattr(identifiability, "_laplace_raw", _fake_raw)
    d = identifiability_laplace(cfg, lp, n_points=2, m=4, m_noise=16, t_obs_s=2.0, seed=3,
                                name="lap1")
    res = d.results
    assert d.variant == "laplace" and d.manifest.parents == {"posterior": lp.id}
    assert [p["tag"] for p in res["points"]] == ["GT", "prior1"]
    names = list(cfg.params_dict) + list(cfg.rescale_params)
    assert [p["name"] for p in res["per_param"]] == names
    units = {p["name"]: p["unit"] for p in res["per_param"]}
    assert units["x_scale"] == "log-range" and units["k"] == "range"
    assert d.manifest.config["log_range_params"] == [n for n in names if "scale" in n]
    assert d.manifest.config["t_obs_s"] == 2.0 and d.manifest.config["seed"] == 3
    assert set(d.manifest.payloads) == {"laplace_sd.npz"}
    z = np.load(d.path / "laplace_sd.npz")
    assert z["SD"].shape == (2, len(names)) and z["points"].shape[0] == 2
    assert cfg.T_obs is None or cfg.T_obs != 2.0, "the diagnostic must never write cfg.T_obs"
    assert {m for m, _, _ in calls} == {4, 16} and {n for _, _, n in calls} == {int(2.0 * cfg.get_unit_conversion_factor("s") / cfg.dt_exp)}


def test_the_laplace_guards_refuse_before_anything_is_created(store, monkeypatch):
    """A chi posterior conditions on a different feature set entirely, so the single-frequency
    41-feature arithmetic would produce a confident, meaningless 'identified / not identified'. Both
    guards fire before store.create, so no directory is written."""
    import pytest
    import torch
    from core.diagnostics import identifiability, identifiability_laplace
    from tests._fixtures import _nad_cfg, _posterior_artifact
    chi_cfg = _nad_cfg(chi_mode=True)
    P = len(chi_cfg.params_dict) + len(chi_cfg.rescale_params)
    w = _posterior_artifact(store, chi_cfg, name="chi_post", V=torch.eye(P, dtype=torch.float64),
                            prior=_LatentPriorStub(P))
    monkeypatch.setattr(identifiability, "_laplace_raw",
                        lambda *a, **k: pytest.fail("simulated before the guards"))
    with pytest.raises(ValueError, match="chi"):
        identifiability_laplace(chi_cfg, store.load_posterior(chi_cfg, w.id), name="lap_chi")
    assert [s for s in store.list("diagnostic") if s.name == "lap_chi"] == []


def test_identifiability_jacobian_maps_degeneracy_over_the_mode_s_own_features(store, monkeypatch):
    """The map must be built from the features the posterior CONDITIONS on. Under chi, Group G's 11
    columns are zeroed and 3K chi columns take their place; a map that kept Group G and omitted chi
    would be literally independent of the chi toggle -- it would report kappa~x_scale as strong as
    ever and falsely refute the hypothesis chi mode exists to test."""
    import numpy as np
    from core.diagnostics import identifiability, identifiability_jacobian
    cfg = _forced_nad_cfg()
    names = list(cfg.params_dict) + list(cfg.rescale_params)
    n_feat = identifiability.feature_sets.n_features(cfg)
    rng = np.random.default_rng(0)

    def _fake_feats(ctx, pvec, rescale_vec, m, crn):
        import torch
        # A smooth, invertible response plus tiny noise: every parameter measurable, no dead channels.
        # R-J: the noise is PROPORTIONAL TO EACH CHANNEL'S OWN SCALE (relative noise), not a fixed
        # absolute magnitude -- an absolute 1e-3 against a scale that grows with the feature index
        # (`base = arange(1, n_feat+1) * (1+theta.sum())`) falls below `noise_eps * fscale` for the
        # later, larger-scale channels and flags them dead BY CONSTRUCTION, which is not what this
        # fake is for (it exists to prove the "no dead channels" premise on a well-behaved response).
        theta = np.concatenate([pvec.detach().cpu().numpy(), rescale_vec.detach().cpu().numpy()])
        base = np.arange(1, n_feat + 1, dtype=float) * (1.0 + theta.sum())
        feats = base[None, :] * (1.0 + rng.normal(0.0, 1e-3, size=(int(m), n_feat)))
        good = torch.ones(int(m), 4, dtype=torch.float64)
        return feats, good, good

    monkeypatch.setattr(identifiability, "_jacobian_features", _fake_feats)
    d = identifiability_jacobian(cfg, m=4, m_noise=16, t_obs_s=2.0, seed=1, name="jac1")
    res = d.results
    assert d.variant == "jacobian" and d.manifest.parents == {}
    assert res["observation_mode"] == "forced" and res["T_obs_s"] == 2.0
    assert res["n_features"] == n_feat and res["dead_channels"] == []
    assert res["unmeasurable"] == [] and res["condition_number"] > 0
    assert all(a in names and b in names for a, b, _ in
               [(p["a"], p["b"], p["cos"]) for p in res["degenerate_pairs"]])
    assert set(d.manifest.payloads) == {"degeneracy_map.npz"}
    assert sorted(d.manifest.figures) == ["figures/jacobian_cosine_matrix.png",
                                          "figures/jacobian_singular_spectrum.png"]
    z = np.load(d.path / "degeneracy_map.npz")
    assert z["J"].shape == (n_feat, len(names))
    assert [str(s) for s in z["param_names"]] == names
    assert d.manifest.inputs["cell"]["path"].endswith("master_weak.txt")
    assert d.manifest.inputs["bounds"]["sha256"]


def test_the_probe_budget_refuses_a_miswired_cos_sin_pair():
    """cos^2 + sin^2 == 1 is the strongest invariant available and it costs nothing: channels 1 and 2
    of each probe are the cosine and sine of ONE angle, so any mis-wiring that puts something else in
    either slot breaks it. This is the standing version of the check that caught gen_chi_raw's [:2]
    unpack binding `u` into `logcyc`."""
    import numpy as np
    import pytest
    import torch
    from core.diagnostics import identifiability
    from tests._fixtures import _nad_cfg
    cfg = _nad_cfg(chi_mode=True)
    keep = identifiability.feature_sets.summary_keep_idx()
    labels = identifiability.feature_sets.feature_labels(cfg)
    ctx = identifiability._JacCtx(cfg=cfg, n_obs=100, mults=torch.tensor([0.1, 0.2]),
                                  forcing_gt=None, keep_idx=keep, feat_labels=labels, n_force_ch=1)
    n_sp = len(keep)
    feats = np.zeros((4, len(labels)))
    for j in range(2):
        feats[:, n_sp + 3 * j + 1] = 0.6          # cos
        feats[:, n_sp + 3 * j + 2] = 0.6          # sin: 0.72, not 1
    keep0 = np.ones(4, dtype=bool)
    xs0 = torch.randn(4, 256, dtype=cfg.hw.dtype)
    with pytest.raises(ValueError, match=r"cos\^2 \+ sin\^2"):
        identifiability._probe_budget(ctx, feats, keep0, xs0, 1.0)
