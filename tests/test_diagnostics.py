"""Diagnostics: the ``diagnostic`` store kind, the shared feature-set and seeding helpers, and the
five diagnostic functions (piece 2 of the 2026-09-11 one-flow design, tasks T13-T18).

Everything here is CPU-sized and stubbed where a simulation would otherwise dominate: a diagnostic is
a measurement ABOUT a trained posterior, so the thing worth pinning is what it reads, what it refuses
and what it writes -- not the numbers a real training run would give it.
"""
import pytest
import torch
from sbi.inference import DirectPosterior

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
    # the refusal depends on --bounds, which the tool never resolves from the cell: name a bounds file
    # with a Forcing section
    assert "--bounds" in str(e.value) and "master.txt" in str(e.value), str(e.value)
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


def test_sbc_prints_small_p_values_as_numbers_and_bins_ranks_at_least_ten_wide(tiny_run, monkeypatch,
                                                                             capsys):
    """Two display defects on the rows that matter most.

    The KS table formatted str(float) truncated to 8 characters, so 3.212345646893978e-20 printed as
    3.212345 -- a p-value between 1 and 10 on exactly the miscalibrated rows the table sorts to the top.

    The pooled histogram took one bin per 20 pooled rows. Pooled N is repeats x n_cal, so at the defaults
    that is 950 bins over 1001 integer ranks: some bins hold two ranks and spike above the band on a
    well-calibrated posterior. Each bin must span at least ~10 integer ranks."""
    import numpy as np
    from matplotlib import pyplot as plt
    from core import orchestrator as orch
    from core.diagnostics import sbc_repeats
    r = tiny_run
    P = len(r.cfg.params_dict) + len(r.cfg.rescale_params)
    N, nps = 2000, 1000
    tiny_p = 3.212345646893978e-20
    assert len(str(tiny_p)) > 8
    seen = {}

    def _plot(**k):
        seen.update(k)
        return plt.figure(), None

    monkeypatch.setattr(orch, "_draw_calibration_set",
                        lambda *a, **k: (torch.zeros(N, 3), torch.zeros(N, P)))
    monkeypatch.setattr(orch, "run_sbc", lambda **k: (
        (torch.arange(N * P).reshape(N, P) % (nps + 1)), torch.zeros(N, P)))
    monkeypatch.setattr(orch, "_sbc_reference_sample", lambda *a, **k: torch.zeros(N, P))
    monkeypatch.setattr(orch, "check_sbc", lambda **k: {"ks_pvals": [tiny_p] * P,
                                                        "c2st_ranks": [0.5] * P, "c2st_dap": [0.5] * P})
    monkeypatch.setattr(orch, "sbc_rank_plot", _plot)
    sbc_repeats(r.cfg, r.posterior, r.prior, repeats=10, n_cal=N, num_posterior_samples=nps,
                fig_sink=lambda title, fig: plt.close(fig))
    out = capsys.readouterr().out
    assert seen["num_bins"] <= (nps + 1) // 10, seen["num_bins"]
    assert "3.21e-20" in out and "3.212345" not in out, out


def _rotation_posterior(store, cfg, *, V, evals):
    """A posterior artifact whose transform block records V (columns) and its eigenvalues."""
    over = {("transform", "fisher_eigenvalues"): None if evals is None else [float(v) for v in evals]}
    return _posterior_artifact(store, cfg, name="rot", V=V, over=over)


def test_identifiability_rotation_decomposes_a_stored_basis(store):
    """What did this artifact MEASURE? The Fisher eigenbasis is on disk already, so the answer costs
    no simulation: which directions carry the information, what each is made of, and which parameters
    are their OWN near-null direction (flat, not aliased -- no reparameterisation reaches those)."""
    import torch
    from core.diagnostics import identifiability_rotation
    cfg = _nad_cfg()
    P = len(cfg.params_dict) + len(cfg.rescale_params)
    names = list(cfg.params_dict) + list(cfg.rescale_params)
    # A permutation basis: direction j is exactly parameter order[j], so every share is 0 or 1 and the
    # arithmetic below is checkable by hand. order[-1] is the WORST direction and its parameter is flat.
    # I1: a CYCLIC SHIFT, not the identity -- V(order=range(P)) is the identity matrix, which is
    # SYMMETRIC, so reading it transposed (the D6 defect: rows instead of columns) is indistinguishable
    # from reading it correctly and every assertion below would still pass. A cyclic shift of P>2
    # elements is not an involution, so its permutation matrix is genuinely non-symmetric.
    order = [(i + 1) % P for i in range(P)]
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
    cfg = _nad_cfg()
    P = len(cfg.params_dict) + len(cfg.rescale_params)
    w = _rotation_posterior(store, cfg, V=torch.eye(P, dtype=torch.float64), evals=None)
    d = identifiability_rotation(cfg, store.load_posterior(cfg, w.id), name="rot_noev")
    assert d.results["eigenvalues"] is None and len(d.results["directions"]) == P
    assert d.results["directions"][0]["eigenvalue"] is None
    assert "NOT STORED" in capsys.readouterr().out
    flat = _posterior_artifact(store, cfg, name="norot", V=None)
    with pytest.raises(ValueError, match="records no Fisher rotation"):
        identifiability_rotation(cfg, store.load_posterior(cfg, flat.id), name="rot_none")
    assert [s.name for s in store.list("diagnostic") if s.name == "rot_none"] == []


def test_identifiability_rotation_refuses_n_worst_over_p(store):
    """M8: ``W[i, P-n_worst:]`` is a NEGATIVE slice once ``n_worst > P`` -- Python reads it from the
    end instead of raising, so ``bottom_share`` would silently sum fewer than n_worst directions.
    Refused before store.create, so no directory is written."""
    import pytest
    import torch
    from core.diagnostics import identifiability_rotation
    cfg = _nad_cfg()
    P = len(cfg.params_dict) + len(cfg.rescale_params)
    w = _rotation_posterior(store, cfg, V=torch.eye(P, dtype=torch.float64), evals=None)
    with pytest.raises(ValueError, match=f"n_worst \\({P + 1}\\) cannot exceed"):
        identifiability_rotation(cfg, store.load_posterior(cfg, w.id), n_worst=P + 1, name="rot_nw")
    assert [s.name for s in store.list("diagnostic") if s.name == "rot_nw"] == []


def test_identifiability_rotation_writes_a_non_finite_eigenvalue_as_none(store):
    """N1b / S1 (spec 4.1): a non-finite eigenvalue reaching identifiability_rotation's results must
    become None, not raise at the manifest write (allow_nan=False) -- or worse, silently succeed with
    a NaN embedded in a JSON field no downstream reader expects.

    A REAL posterior artifact cannot carry a NaN eigenvalue in the first place: store.create's own
    manifest validator refuses it outright, well before identifiability_rotation ever runs -- proven
    directly below. So this uses the rotation tests' own synthetic-manifest pattern
    (_rotation_posterior/_posterior_artifact), but skips the real store WRITE for the input posterior:
    identifiability_rotation reads only posterior.manifest.body/.config/.name/.id/.accepted, never the
    pickled payload, so a plain in-memory stand-in is enough -- exactly as cheap as a real one, and the
    only way to get a non-finite value in front of this function at all.
    """
    import math
    from types import SimpleNamespace

    import pytest
    import torch
    from core.artifacts import manifest as mf
    from core.diagnostics import identifiability_rotation
    cfg = _nad_cfg()
    P = len(cfg.params_dict) + len(cfg.rescale_params)

    # Proof a REAL artifact refuses this outright -- the reason a synthetic stand-in is needed at all.
    with pytest.raises(Exception, match="non-finite"):
        _posterior_artifact(store, cfg, name="nan_real", V=torch.eye(P, dtype=torch.float64),
                            over={("transform", "fisher_eigenvalues"): [float("nan")] * P})

    keys = list(cfg.params_dict) + list(cfg.rescale_params)
    evals = [float("nan")] + [10.0 ** (1 - k) for k in range(1, P)]
    body = {"mode": cfg.observation_mode, "conditioning": mf.conditioning_block(cfg),
            "transform": {"param_keys": keys, "V": torch.eye(P, dtype=torch.float64).tolist(),
                          "fisher_eigenvalues": evals}}
    posterior = SimpleNamespace(manifest=SimpleNamespace(body=body, config={"model": cfg.model}),
                                name="synthetic_rot", id="synthrotid", accepted=[])
    d = identifiability_rotation(cfg, posterior, n_worst=1, top_n=2, name="rot_nan")
    res = d.results
    assert res["eigenvalues"][0] is None
    assert all(v is None or math.isfinite(v) for v in res["eigenvalues"])
    assert res["directions"][0]["eigenvalue"] is None
    assert all(dr["eigenvalue"] is None or math.isfinite(dr["eigenvalue"]) for dr in res["directions"])
    # the manifest was actually written and loads back clean -- proof no NaN literal reached
    # json.dumps(allow_nan=False).
    reloaded = store.load_diagnostic(d.id)
    assert reloaded.results == res


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
        row = np.arange(n_feat, dtype=float) + float(nd.sum())
        # M12: an alternating +-0.01 "wobble" over the ensemble AXIS (same for every feature column),
        # not the exactly-constant-row fake this replaces. An EXACTLY constant ensemble makes fnoise
        # clamp to the 1e-9 floor, which -- after dividing a raw gradient of ~1 by it -- puts every ND
        # column of J at ~1e9-1e12 in magnitude; J^T@J then lands at ~1e21-1e24, where adding
        # `np.eye(P)` is a complete float64 NO-OP (bit-lost), turning a well-posed ridge inversion into
        # a numerically singular one -- the covariance diagonal comes back as SIGN-NOISE near zero,
        # not the true (well-conditioned, closed-form) answer. `m` and `m_noise` are both EVEN here
        # (4, 16), so `sum((-1)**i for i in range(m)) == 0` EXACTLY (IEEE754 negation is exact, and
        # summing exact +/-0.01 pairs cancels to exactly 0.0) -- the ensemble MEAN, which is all
        # measure() ever reads, is therefore untouched, while the ensemble STD (fnoise) becomes a
        # clean, well-scaled 0.01 instead of the pathological 1e-9 floor.
        wobble = 0.01 * ((-1.0) ** np.arange(int(m)))[:, None]
        feats = np.tile(row, (int(m), 1)) + wobble
        return feats, g + 1.0, g + 1.0

    monkeypatch.setattr(identifiability, "_laplace_raw", _fake_raw)
    t_obs_before = cfg.T_obs                          # I3: captured BEFORE the call, not guessed after
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
    # I3: EQUALS its pre-call value, not merely "not 2.0" -- 2.0 is SECONDS while the script's own
    # (never-ported) write was `cfg.T_obs = t_obs_s * hz` in CELL units, which is essentially never
    # exactly 2.0, so the old check would not have noticed that write happening at all.
    assert cfg.T_obs == t_obs_before, "the diagnostic must never write cfg.T_obs"
    assert {m for m, _, _ in calls} == {4, 16} and {n for _, _, n in calls} == {int(2.0 * cfg.get_unit_conversion_factor("s") / cfg.dt_exp)}
    # F15: the noise-floor ensemble draws INDEPENDENT noise (crn False); the +-d arms share common
    # random numbers (crn True), or the finite difference is noise over 2d
    assert {c for mm, c, _ in calls if mm == 16} == {False}, calls
    assert {c for mm, c, _ in calls if mm == 4} == {True}, calls

    # M12: this fake makes the numbers a closed form, derived here and pinned so a units/arithmetic
    # slip in _analyze_point's covariance inversion fails LOUDLY rather than merely changing a number.
    # The fake's feats depend on nd.sum() (`arange(n_feat) + nd.sum() + wobble`), never on res/force:
    #  - perturbing a RESCALE parameter leaves the fake's MEAN output totally unchanged (the wobble is
    #    independent of nd/res/force) -> raw gradient EXACTLY 0; fnoise is the wobble's own std, 0.01,
    #    uniformly over every feature -> 0/0.01 stays exactly 0. J's rescale COLUMNS are therefore
    #    all-zero, so J^T@J is BLOCK-DIAGONAL with a plain identity rescale block, and I^-1 = I: every
    #    rescale SD is EXACTLY 1.0 at every point, never < sd_identified (0.3) -> frac_identified==0.0.
    #  - perturbing ND param p changes nd.sum() by +-d, so fp-fm = 2d for EVERY feature row (the
    #    arange offset and the wobble both cancel identically) -> raw gradient is the CONSTANT vector
    #    1.0 (=2d/2d), scaled by fac=(hi-lo) (is_log is False for every ND index) and 1/fnoise=100.
    #    So each ND column of J is the constant vector w_p=100*(hi_p-lo_p) at every row; J^T@J's nd-nd
    #    block is then EXACTLY n_feat * outer(w, w), rank 1, and cov = (n_feat*outer(w,w) + I)^-1 --
    #    computed here EXACTLY as the code computes it (no asymptotic approximation: at this w-scale
    #    the "+1" stays representable, unlike the 1e9-scale fake this replaces, where it silently
    #    vanished into float64 rounding and turned a well-posed ridge into a numerically singular one).
    #    Both points give the SAME value (the additive nd.sum() shift cancels in every finite
    #    difference), so median_sd equals this closed form and measurable == P at every point.
    n_nd = len(cfg.params_dict)
    n_feat = identifiability.feature_sets.n_features(cfg)
    nd_w = np.array([hi - lo for _, (_v, (lo, hi)) in cfg.params_dict.items()])
    w_full = np.zeros(len(names))
    w_full[:n_nd] = nd_w / 0.01                        # fac / fnoise, fnoise == 0.01 by construction
    cov = np.linalg.inv(n_feat * np.outer(w_full, w_full) + np.eye(len(names)))
    expected_sd = np.sqrt(np.clip(np.diag(cov), 0, None))
    assert np.allclose(z["SD"][:, n_nd:], 1.0, atol=1e-8), "every rescale SD must be exactly 1.0"
    assert np.allclose(z["SD"], expected_sd[None, :], atol=1e-3), \
        "the SDs must match the closed form derived from the fake's linear response"
    nd_records, rescale_records = res["per_param"][:n_nd], res["per_param"][n_nd:]
    assert all(r["frac_identified"] == 0.0 for r in rescale_records)
    assert [r["median_sd"] for r in nd_records] == pytest.approx(list(expected_sd[:n_nd]), abs=1e-3)
    assert all(pt["measurable"] == len(names) for pt in res["points"])


def test_the_laplace_guards_refuse_before_anything_is_created(store, monkeypatch):
    """A chi posterior conditions on a different feature set entirely, so the single-frequency
    41-feature arithmetic would produce a confident, meaningless 'identified / not identified'.
    M11: THREE guards fire before store.create, so no directory is written -- chi (a chi posterior
    conditions on a different feature set), forced (a spontaneous cell has no drive to read), and
    n_points (there must be at least the ground truth)."""
    import pytest
    import torch
    from core import cli, config, registry
    from core.config import VALID_LABELS, VALID_MODELS
    from core.diagnostics import identifiability, identifiability_laplace
    chi_cfg = _nad_cfg(chi_mode=True)
    P = len(chi_cfg.params_dict) + len(chi_cfg.rescale_params)
    w = _posterior_artifact(store, chi_cfg, name="chi_post", V=torch.eye(P, dtype=torch.float64),
                            prior=_LatentPriorStub(P))
    monkeypatch.setattr(identifiability, "_laplace_raw",
                        lambda *a, **k: pytest.fail("simulated before the guards"))
    with pytest.raises(ValueError, match="chi"):
        identifiability_laplace(chi_cfg, store.load_posterior(chi_cfg, w.id), name="lap_chi")
    assert [s for s in store.list("diagnostic") if s.name == "lap_chi"] == []

    spont = cli.make_sim_config("NADROWSKI", VALID_LABELS[VALID_MODELS.index("NADROWSKI")],
                                registry.state_dep_drift("NADROWSKI"),
                                str(config.BOUNDS_PATH / "nadrowski" / "master_spont.txt"))
    spont.hw = config.cpu_device()
    Ps = len(spont.params_dict) + len(spont.rescale_params)
    ws = _posterior_artifact(store, spont, name="spont_post", V=torch.eye(Ps, dtype=torch.float64),
                             prior=_LatentPriorStub(Ps))
    with pytest.raises(ValueError, match="no amp/freq/phase to read"):
        identifiability_laplace(spont, store.load_posterior(spont, ws.id), name="lap_spont")
    assert [s for s in store.list("diagnostic") if s.name == "lap_spont"] == []

    forced_cfg = _forced_nad_cfg()
    Pf = len(forced_cfg.params_dict) + len(forced_cfg.rescale_params)
    wf = _posterior_artifact(store, forced_cfg, name="npts_post", V=torch.eye(Pf, dtype=torch.float64),
                             prior=_LatentPriorStub(Pf))
    with pytest.raises(ValueError, match="n_points must be at least 1"):
        identifiability_laplace(forced_cfg, store.load_posterior(forced_cfg, wf.id), n_points=0,
                                name="lap_npts")
    assert [s for s in store.list("diagnostic") if s.name == "lap_npts"] == []


def test_laplace_and_jacobian_refuse_bad_probe_settings_before_any_simulation(store, monkeypatch):
    """M9/R3: a t_obs_s that computes a non-positive OR sub-one n_obs (zero, negative, or a tiny
    positive value that still floors to 0 samples), or a non-finite t_obs_s (NaN would otherwise
    crash at int(nan) rather than refuse), and an m_noise below 10 (which cannot estimate a
    feature-noise floor at all) -- all fire before store.create, for both simulating diagnostics."""
    import pytest
    import torch
    from core.diagnostics import identifiability, identifiability_jacobian, identifiability_laplace
    cfg = _forced_nad_cfg()
    P = len(cfg.params_dict) + len(cfg.rescale_params)
    w = _posterior_artifact(store, cfg, name="pb_post", V=torch.eye(P, dtype=torch.float64),
                            prior=_LatentPriorStub(P))
    lp = store.load_posterior(cfg, w.id)
    monkeypatch.setattr(identifiability, "_laplace_raw",
                        lambda *a, **k: pytest.fail("simulated before the guard"))
    monkeypatch.setattr(identifiability, "_jacobian_features",
                        lambda *a, **k: pytest.fail("simulated before the guard"))

    with pytest.raises(ValueError, match="t_obs_s"):
        identifiability_laplace(cfg, lp, t_obs_s=0.0, name="lap_bad_t")
    with pytest.raises(ValueError, match="t_obs_s"):
        # R3: a TINY positive value must also refuse -- n_obs floors to 0 samples, not a valid
        # recording, and the old `t_obs_s <= 0` guard let it straight through.
        identifiability_laplace(cfg, lp, t_obs_s=1e-12, name="lap_tiny_t")
    with pytest.raises(ValueError, match="m_noise"):
        identifiability_laplace(cfg, lp, m_noise=4, name="lap_bad_m")
    with pytest.raises(ValueError, match="t_obs_s"):
        identifiability_jacobian(cfg, t_obs_s=-1.0, name="jac_bad_t")
    with pytest.raises(ValueError, match="t_obs_s"):
        identifiability_jacobian(cfg, t_obs_s=1e-12, name="jac_tiny_t")
    with pytest.raises(ValueError, match="m_noise"):
        identifiability_jacobian(cfg, m_noise=4, name="jac_bad_m")
    for nm in ("lap_bad_t", "lap_tiny_t", "lap_bad_m", "jac_bad_t", "jac_tiny_t", "jac_bad_m"):
        assert [s for s in store.list("diagnostic") if s.name == nm] == []

    # F10: the arm ensemble, the relative step and the validity floor. --m 0 used to run the whole
    # noise ensemble and then every arm at batch 0; --rel 0 on a zero-valued truth divided by zero and
    # reached lstsq after the spend; --min-valid outside (0, 1] silently accepted or refused every arm.
    bad = [("m", {"m": 0}), ("rel", {"rel": 0.0}), ("rel", {"rel": float("nan")}),
           ("rel", {"rel": -0.02}), ("min_valid", {"min_valid": 0.0}), ("min_valid", {"min_valid": 1.5})]
    for knob, kw in bad:
        with pytest.raises(ValueError, match=knob):
            identifiability_laplace(cfg, lp, name=f"lap_bad_{knob}", **kw)
        with pytest.raises(ValueError, match=knob):
            identifiability_jacobian(cfg, name=f"jac_bad_{knob}", **kw)
        for nm in (f"lap_bad_{knob}", f"jac_bad_{knob}"):
            assert [s for s in store.list("diagnostic") if s.name == nm] == []

    # rotation used to CLAMP n_worst / top_n to 1: a --n-worst 0 quietly became 1 (the D6 trap)
    from core.diagnostics import identifiability_rotation
    rot = store.load_posterior(cfg, _rotation_posterior(store, cfg, V=torch.eye(P, dtype=torch.float64),
                                                        evals=None).id)
    with pytest.raises(ValueError, match="n_worst"):
        identifiability_rotation(cfg, rot, n_worst=0, name="rot_nw0")
    with pytest.raises(ValueError, match="top_n"):
        identifiability_rotation(cfg, rot, top_n=0, name="rot_top0")
    assert [s for s in store.list("diagnostic") if s.name in ("rot_nw0", "rot_top0")] == []


def test_laplace_raw_does_not_leak_its_crn_seed(monkeypatch):
    """M5: ``_laplace_raw``'s CRN reseeds (``_SF``/``_SS``) must not escape the call, exactly as
    ``_jacobian_features`` already guards with ``fork_rng`` -- otherwise every subsequent measurement
    (including the noise floor at points 2..K, which is NOT itself reseeded) is pinned downstream of
    the CRN constants and silently stops depending on ``--seed`` at all.

    Verified directly and cheaply (CPU, no real simulation): the global torch stream must be
    bit-identical before and after a crn=True call. torch.manual_seed alone leaks its effect out;
    only fork_rng restores the incoming state. pipeline.gen_obs is stubbed to something that still
    consumes the RNG (so a leak has something to leak), replacing the real (expensive) simulator.
    """
    import torch
    from core.diagnostics import identifiability
    from core.SBI import pipeline

    def _stub_gen_obs(*, model, params, t, inits, force, n_segs, steady_idx, state_dep_drift,
                      batch_size, dtype, device, **_kw):
        return (torch.randn(batch_size, t.shape[0], dtype=dtype, device=device),)

    monkeypatch.setattr(pipeline, "gen_obs", _stub_gen_obs)
    cfg = _forced_nad_cfg()
    nd = cfg.params_tensor[0].clone()
    res = torch.tensor([v for v, _ in cfg.rescale_params.values()], dtype=cfg.hw.dtype)
    force = torch.tensor([v for v, _ in cfg.force_params_dict.values()], dtype=cfg.hw.dtype)
    # R4: torch.manual_seed is process-global and NEVER restores on its own -- fork_rng here is not
    # the thing under test (that is _laplace_raw's OWN fork_rng), it is this TEST keeping its own
    # seeding from leaking into whichever test runs next.
    with torch.random.fork_rng():
        torch.manual_seed(123)
        before = torch.get_rng_state()
        identifiability._laplace_raw(cfg, nd, res, force, 4, True, 20)
        after = torch.get_rng_state()
    assert torch.equal(before, after), "the CRN reseed leaked into the caller's RNG stream"


def test_laplace_points_draw_independent_noise_but_the_whole_run_reproduces(monkeypatch):
    """R1 (a regression introduced by the M5 fix above): fork_rng must wrap ONLY the CRN-seeded
    (``crn=True``) arms, not the ``crn=False`` noise-floor ensemble too. An earlier version wrapped
    the whole function unconditionally, so fork_rng ALSO captured-and-discarded whatever the
    crn=False branch drew -- every Laplace POINT's m_noise ensemble then replayed the exact same
    frozen incoming state, and "independent" noise floors across the ground truth and the prior
    draws were not independent at all: the ORIGINAL SCRIPT's points each drew fresh noise.

    Verified directly and cheaply (CPU, no real simulation, same stub as the leak test above): two
    crn=False calls made back-to-back inside one seeded(...) run must draw DIFFERENT noise (the
    stream genuinely advances between them), while replaying the same two-call sequence from the
    same seed reproduces both calls bit-for-bit -- the whole measurement stays reproducible even
    though each point's own draw is not a repeat of the last.
    """
    import numpy as np
    import torch
    from core.diagnostics import identifiability
    from core.diagnostics.rng import seeded
    from core.SBI import pipeline

    def _stub_gen_obs(*, model, params, t, inits, force, n_segs, steady_idx, state_dep_drift,
                      batch_size, dtype, device, **_kw):
        return (torch.randn(batch_size, t.shape[0], dtype=dtype, device=device),)

    monkeypatch.setattr(pipeline, "gen_obs", _stub_gen_obs)
    cfg = _forced_nad_cfg()
    nd = cfg.params_tensor[0].clone()
    res = torch.tensor([v for v, _ in cfg.rescale_params.values()], dtype=cfg.hw.dtype)
    force = torch.tensor([v for v, _ in cfg.force_params_dict.values()], dtype=cfg.hw.dtype)

    def _two_points():
        with seeded(7, cfg.hw.device):
            feats1, _, _ = identifiability._laplace_raw(cfg, nd, res, force, 4, False, 20)
            feats2, _, _ = identifiability._laplace_raw(cfg, nd, res, force, 4, False, 20)
        return feats1, feats2

    f1a, f2a = _two_points()
    assert not np.allclose(f1a, f2a), \
        "two crn=False calls in the same run must draw DIFFERENT noise -- fork_rng is discarding " \
        "the stream's progression between them, exactly the R1 regression"
    f1b, f2b = _two_points()
    assert np.allclose(f1a, f1b) and np.allclose(f2a, f2b), \
        "the same seed must reproduce BOTH points' noise exactly"


def test_identifiability_jacobian_maps_degeneracy_over_the_mode_s_own_features(store, monkeypatch):
    """The map must be built from the features the posterior CONDITIONS on. Under chi, Group G's 11
    columns are zeroed and 3K chi columns take their place; a map that kept Group G and omitted chi
    would be literally independent of the chi toggle -- it would report kappa~x_scale as strong as
    ever and falsely refute the hypothesis chi mode exists to test."""
    import math

    import numpy as np
    from core.diagnostics import identifiability, identifiability_jacobian
    cfg = _forced_nad_cfg()
    names = list(cfg.params_dict) + list(cfg.rescale_params)
    n_feat = identifiability.feature_sets.n_features(cfg)
    n_theta = len(names)
    rng = np.random.default_rng(0)
    # I2: a per-(feature, parameter) weight matrix, deterministic and with NO shared structure across
    # parameters -- unlike the R-J fake this replaces, whose response depended ONLY on theta.sum(),
    # making every raw gradient the SAME constant vector (=2d/2d) for EVERY parameter and every pair
    # "degenerate" (|cos|==1) by construction: a rank-1 Jacobian the assertions below could not tell
    # apart from a real, non-degenerate one.
    W = ((np.arange(n_feat)[:, None] * 7 + np.arange(n_theta)[None, :] * 3) % 5 + 1).astype(float)
    crns = []

    def _fake_feats(ctx, pvec, rescale_vec, m, crn):
        import torch
        crns.append((int(m), bool(crn)))
        theta = np.concatenate([pvec.detach().cpu().numpy(), rescale_vec.detach().cpu().numpy()])
        idx = np.arange(n_feat)
        base = (idx + 1).astype(float)
        # R-J: relative noise (proportional to each channel's own scale), not a fixed absolute
        # magnitude -- see the module's own comment on the same choice in _dead_channels' docstring.
        lean = 0.001 * (W @ theta)
        response = base * (1.0 + lean)
        feats = response[None, :] * (1.0 + rng.normal(0.0, 1e-3, size=(int(m), n_feat)))
        # I2: ONE genuinely dead channel -- ~5.0 plus INDEPENDENT (theta-blind) noise 5 orders of
        # magnitude quieter than its own scale, well under noise_eps*fscale. Its noise is NOT exactly
        # zero (unlike simply hardcoding a constant): a literal constant would already give an
        # all-zero raw gradient on its own (0 divided by anything is 0), so deleting the dead-channel
        # zeroing step downstream would change NOTHING observable. This tiny independent noise instead
        # leaks a small but genuinely NONZERO difference into the +-d central-difference arms, which
        # divided by an even smaller fnoise is an AMPLIFIER (see _dead_channels): with the zeroing
        # applied the row is exactly 0; without it, it is a spurious, non-negligible row of J.
        feats[:, -1] = 5.0 + rng.normal(0.0, 1e-7, size=int(m))
        good = torch.ones(int(m), 4, dtype=torch.float64)
        return feats, good, good

    monkeypatch.setattr(identifiability, "_jacobian_features", _fake_feats)
    d = identifiability_jacobian(cfg, m=4, m_noise=16, t_obs_s=2.0, seed=1, name="jac1")
    res = d.results
    # F15: the m_noise ensemble without common random numbers, every +-d arm with them
    assert {c for mm, c in crns if mm == 16} == {False}, crns
    assert {c for mm, c in crns if mm == 4} == {True}, crns
    assert d.variant == "jacobian" and d.manifest.parents == {}
    assert res["observation_mode"] == "forced" and res["T_obs_s"] == 2.0
    assert res["n_features"] == n_feat
    dead_label = identifiability.feature_sets.feature_labels(cfg)[-1]
    assert res["dead_channels"] == [dead_label]
    assert res["unmeasurable"] == [] and res["condition_number"] > 0
    assert all(a in names and b in names for a, b, _ in
               [(p["a"], p["b"], p["cos"]) for p in res["degenerate_pairs"]])
    # I2: a rank-1 J (the old fake) makes EVERY pair degenerate -- this full-rank response must not.
    assert 0 <= len(res["degenerate_pairs"]) < math.comb(n_theta, 2)
    assert set(d.manifest.payloads) == {"degeneracy_map.npz"}
    assert sorted(d.manifest.figures) == ["figures/jacobian_cosine_matrix.png",
                                          "figures/jacobian_singular_spectrum.png"]
    z = np.load(d.path / "degeneracy_map.npz")
    assert z["J"].shape == (n_feat, len(names))
    assert [str(s) for s in z["param_names"]] == names
    assert np.all(z["J"][-1, :] == 0.0), "the dead row must be ZEROED, not merely coincidentally 0"
    assert d.manifest.inputs["cell"]["path"].endswith("master_weak.txt")
    assert d.manifest.inputs["bounds"]["sha256"]

    # S1: the restored npz arrays -- shapes only (the numbers are the fake's, not a fixed science
    # result): S/C describe the SVD/cosine structure over the measurable and stiff subsets, the pairs
    # are parallel arrays (no pickle needed to load them), and the top-features table is (P, k).
    n_meas = int(z["measurable_mask"].sum())
    n_stiff = int(z["stiff_mask"].sum())
    assert z["measurable_mask"].shape == (n_theta,) and z["stiff_mask"].shape == (n_theta,)
    assert z["C"].shape == (n_meas, n_meas) and z["S"].shape == (n_stiff,)
    assert z["sloppiest_loadings"].shape == (n_stiff,) and z["unique_frac"].shape == (n_meas,)
    n_pairs = len(res["degenerate_pairs"])
    assert z["pair_a"].shape == z["pair_b"].shape == z["pair_cos"].shape == (n_pairs,)
    top_k = min(5, n_feat)
    assert z["top_feat_idx"].shape == z["top_feat_val"].shape == (n_theta, top_k)
    assert z["row_dominant_idx"].shape == z["row_dominant_fnoise"].shape == (min(8, n_feat),)


def test_identifiability_jacobian_is_chi_aware(store, monkeypatch):
    """I2: the map's row count and payload must track the MODE's own feature set -- 30 spontaneous +
    3K chi under chi, not the 41-feature forced set -- so hard-coding the forced width anywhere in
    this path would be caught here."""
    import numpy as np
    from core import cli, config, registry
    from core.config import VALID_LABELS, VALID_MODELS
    from core.diagnostics import identifiability, identifiability_jacobian
    labels = VALID_LABELS[VALID_MODELS.index("NADROWSKI")]
    cell = str(config.CELL_PATH / "nadrowski" / "master_weak.txt")
    # I2: the smallest HONEST setup -- chi mode ignores the cell's own drive entirely (assert_forced
    # is skipped for chi in identifiability_jacobian), but cfg.ground_truth still needs a loaded cell
    # for the pre-spend refusal to pass, so this reuses the SAME forced cell/bounds pairing every
    # other test here uses rather than inventing a new bounds/cell file just for this one assertion.
    chi_cfg = cli.make_sim_config("NADROWSKI", labels, registry.state_dep_drift("NADROWSKI"),
                                  str(config.BOUNDS_PATH / "nadrowski" / "master.txt"), chi_mode=True)
    chi_cfg.hw = config.cpu_device()
    cli.load_and_validate_gt(chi_cfg, cell)
    names = list(chi_cfg.params_dict) + list(chi_cfg.rescale_params)
    n_feat = identifiability.feature_sets.n_features(chi_cfg)
    n_sp = len(identifiability.feature_sets.summary_keep_idx())
    rng = np.random.default_rng(0)

    def _fake_feats(ctx, pvec, rescale_vec, m, crn):
        import torch
        theta = np.concatenate([pvec.detach().cpu().numpy(), rescale_vec.detach().cpu().numpy()])
        base = (np.arange(n_feat, dtype=float) + 1.0) * (1.0 + 0.001 * theta.sum())
        feats = base[None, :] * (1.0 + rng.normal(0.0, 1e-3, size=(int(m), n_feat)))
        for j in range(len(ctx.mults)):                # a genuine cos/sin pair per probe, unit norm
            feats[:, n_sp + 3 * j + 1] = 0.6
            feats[:, n_sp + 3 * j + 2] = 0.8
        good = torch.ones(int(m), 4, dtype=torch.float64)
        return feats, good, good

    monkeypatch.setattr(identifiability, "_jacobian_features", _fake_feats)
    d = identifiability_jacobian(chi_cfg, m=4, m_noise=16, t_obs_s=2.0, seed=1, name="jacchi")
    assert d.results["observation_mode"] == "chi"
    assert d.results["n_features"] == n_feat == identifiability.feature_sets.n_features(chi_cfg)
    z = np.load(d.path / "degeneracy_map.npz")
    assert z["J"].shape == (n_feat, len(names))


def test_identifiability_jacobian_refuses_wrong_model_or_spontaneous_cfg(store):
    """assert_nadrowski and assert_forced both fire before store.create -- both guards run before
    cfg.ground_truth is even touched, so neither needs a loaded cell to demonstrate the refusal."""
    import pytest
    from core import cli, config, registry
    from core.config import VALID_LABELS, VALID_MODELS
    from core.diagnostics import identifiability_jacobian
    other = _nad_cfg()
    other.model = "HOPF"
    with pytest.raises(ValueError, match="Nadrowski-specific"):
        identifiability_jacobian(other, t_obs_s=2.0, name="jac_wrongmodel")
    assert [s for s in store.list("diagnostic") if s.name == "jac_wrongmodel"] == []

    spont = cli.make_sim_config("NADROWSKI", VALID_LABELS[VALID_MODELS.index("NADROWSKI")],
                                registry.state_dep_drift("NADROWSKI"),
                                str(config.BOUNDS_PATH / "nadrowski" / "master_spont.txt"))
    spont.hw = config.cpu_device()
    with pytest.raises(ValueError, match="no amp/freq/phase to read"):
        identifiability_jacobian(spont, t_obs_s=2.0, name="jac_spont")
    assert [s for s in store.list("diagnostic") if s.name == "jac_spont"] == []


def test_the_probe_budget_refuses_a_miswired_cos_sin_pair():
    """cos^2 + sin^2 == 1 is the strongest invariant available and it costs nothing: channels 1 and 2
    of each probe are the cosine and sine of ONE angle, so any mis-wiring that puts something else in
    either slot breaks it. This is the standing version of the check that caught gen_chi_raw's [:2]
    unpack binding `u` into `logcyc`."""
    import numpy as np
    import pytest
    import torch
    from core.diagnostics import identifiability
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


def test_the_probe_budget_accepts_a_correctly_wired_pair():
    """M10: the miswired test above sets every other slot to ZERO, so a channel-OFFSET regression
    (reading cos/sin from the wrong pair of columns) could still coincidentally break cos^2+sin^2==1
    and pass for the wrong reason. A genuinely correct (0.6, 0.8) pair (0.36+0.64==1) must NOT raise,
    which a wrong-offset read (landing on a zeroed slot) generally would."""
    import numpy as np
    import torch
    from core.diagnostics import identifiability
    cfg = _nad_cfg(chi_mode=True)
    keep = identifiability.feature_sets.summary_keep_idx()
    labels = identifiability.feature_sets.feature_labels(cfg)
    ctx = identifiability._JacCtx(cfg=cfg, n_obs=100, mults=torch.tensor([0.1, 0.2]),
                                  forcing_gt=None, keep_idx=keep, feat_labels=labels, n_force_ch=1)
    n_sp = len(keep)
    feats = np.zeros((4, len(labels)))
    for j in range(2):
        feats[:, n_sp + 3 * j + 1] = 0.6          # cos
        feats[:, n_sp + 3 * j + 2] = 0.8          # sin: 0.36+0.64 == 1, genuinely valid
    keep0 = np.ones(4, dtype=bool)
    xs0 = torch.randn(4, 256, dtype=cfg.hw.dtype)
    identifiability._probe_budget(ctx, feats, keep0, xs0, 1.0)          # must not raise


def test_load_rows_x_only_stops_at_max_rows(tmp_path):
    """The conditioning-channel diagnostics read x and never the targets, and they want a sample, not
    the cache: at the production shape the th_ shards are half the bytes on disk and the whole cache
    is tens of GiB. x_only must not open a th_ shard at all -- pinned by deleting them -- and max_rows
    must stop at the shard that reaches the count rather than walking to the end."""
    import pytest
    import torch
    from core.SBI import training_checkpoint as tc
    d = tmp_path / "cache"
    (d / "shards").mkdir(parents=True)
    for a, b in ((0, 2), (2, 4), (4, 6)):
        torch.save(torch.full((2 * 5, 3), float(a)), d / "shards" / f"x_{a:06d}_{b:06d}.pt")
        torch.save(torch.full((2 * 5, 2), float(a)), d / "shards" / f"th_{a:06d}_{b:06d}.pt")
    x, th = tc.load_rows(d, 6, 5)
    assert tuple(x.shape) == (30, 3) and tuple(th.shape) == (30, 2)
    for f in (d / "shards").glob("th_*.pt"):
        f.unlink()
    x, th = tc.load_rows(d, 6, 5, x_only=True)
    assert tuple(x.shape) == (30, 3) and th is None
    x, th = tc.load_rows(d, 6, 5, x_only=True, max_rows=12)
    assert tuple(x.shape) == (12, 3) and th is None
    assert x[:10].eq(0.0).all() and x[10:].eq(2.0).all(), "max_rows must keep batch order"
    # The completeness check is SKIPPED under max_rows -- a partial read is the point -- but still
    # fires without it, because a cache that commits 6 batches and holds 2 is corrupt.
    assert tuple(tc.load_rows(d, 99, 5, x_only=True, max_rows=4)[0].shape) == (4, 3)
    with pytest.raises(ValueError, match="commits 99 batches but only 6"):
        tc.load_rows(d, 99, 5, x_only=True)


def test_ablation_sweeps_the_summary_columns_through_the_whole_conditioning_path():
    """Two things at once, because they are one defect.

    (a) The sweep runs over the SUMMARY columns only: the forcing / chi block stays at the base row's
    values, so a forced or chi posterior -- whose width includes that block -- is measured at a real
    observation rather than at a row no recording can produce.

    (b) It runs through the whole conditioning path, standardizer included. For spontaneous and forced
    posteriors sbi puts a per-column affine ahead of the net (z_score_x="independent"), so a bare-net
    sweep fed raw values to a net trained on z-scores. A standardizer that zeroes its input must make
    every channel read as doing nothing -- which can only happen if it is actually in the path.
    """
    import torch
    from core import orchestrator
    from core.diagnostics import ablation
    from core.SBI.statistics import FEATURE_LABELS, SUMMARY_WIDTH, VALID_FLAG_LABELS

    class _Zero(torch.nn.Module):
        """A stand-in standardizer that maps everything to zero -- if it is in the path, nothing moves."""
        def forward(self, x):
            return torch.zeros_like(x)

    cfg = _forced_nad_cfg()
    n_sum = SUMMARY_WIDTH + 1
    fdim = orchestrator.expected_forcing_dim(cfg)
    labels = FEATURE_LABELS + VALID_FLAG_LABELS + ["logT"]
    seen = []

    class _Record(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            seen.append(x.detach().clone())
            return self.inner(x)

    # I2: the net's own weight init AND the data draw are BOTH seeded inside one fork_rng, so
    # healthy_max no longer depends on what ran before this test in the same process -- fork_rng
    # restores the caller's RNG on the way out, exactly as core.diagnostics.rng.seeded does.
    with torch.random.fork_rng():
        torch.manual_seed(0)
        net = orchestrator.build_embedding_net(cfg).eval()
        data = torch.randn(64, n_sum + fdim)
    # Element 0 of the return is the base-row marker (its r[0] is the row INDEX, not a displacement),
    # so every assertion below reads the summary records only.
    rows = ablation._sweep_channels(_Record(net), data, n_sum, 5, labels)[1:]
    assert len(rows) == n_sum and [r[1] for r in rows] == labels
    base = seen[0]
    assert base.shape == (1, n_sum + fdim)
    # M5: the base point must be a REAL row of data (the docstring's whole point), not a column-wise
    # median vector -- which, for i.i.d. Gaussian data, essentially never matches any actual row.
    assert any(torch.equal(base[0], data[i]) for i in range(data.shape[0])), \
        "the base point must be a real row of data, not a synthesized median vector"
    for j, call in enumerate(seen[1:]):
        assert torch.equal(call[:, n_sum:], base[:, n_sum:].expand(call.shape[0], -1)), \
            "the forcing block moved during a summary sweep"
        # M5: EXACTLY the j-th column, not merely "at most one" -- the data is random normal, so a
        # column/label mix-up (sweeping column k while labelling and counting it as column j) would
        # still pass a "<= 1" check but fails this one.
        varying = [k for k in range(n_sum) if call[:, k].min() != call[:, k].max()]
        assert varying == [j], "the swept column did not match the sweep's own column index"
    healthy_max = max(r[0] for r in rows)
    # I2: guards against a near-dead random init shrinking the bound (below) into the noise floor --
    # a healthy sweep on an untrained net is still order 1 here, not order 1e-6.
    assert healthy_max > 1e-2, "an unstandardized sweep must move the embedding by more than noise"

    zeroing = torch.nn.Sequential(_Zero(), net)
    rows0 = ablation._sweep_channels(zeroing, data, n_sum, 5, labels)[1:]
    # N3: R-K(3) as given (== 0.0 -> <= 1e-6) still under-tolerates. Since I2 the net's init is SEEDED
    # (built inside the same fork_rng as the data draw above), so this is not init noise: _Zero maps
    # every input to the identical zero tensor, but _sweep_channels calls emb() once on a 1-ROW batch
    # (e0 = emb(base)) and once on a 5-ROW batch (emb(v), n_sweep=5) -- the same logical computation,
    # batched differently. A LayerNorm/matmul kernel is not guaranteed bit-identical across batch
    # shapes (a different vectorization or summation order for 1 row vs 5 rows), so the "zeroed" leg
    # still lands a few ulps of FLOAT32_EPS (1.19e-7) above exactly 0.0 -- ~9e-7 to ~1.4e-6, measured
    # across five runs while writing this test -- comfortably distinct from a HEALTHY channel's
    # response (2-4 here) by 6+ orders of magnitude. The bound is therefore relative to this very
    # call's own healthy max, not a hardcoded absolute: the intent -- a zeroing standardizer makes the
    # sweep read as machine noise, not as a real signal -- survives whatever scale a given net's
    # (now seeded, reproducible) init happens to produce.
    assert max(r[0] for r in rows0) <= healthy_max * 1e-4, \
        "the standardizer was not in the path: the sweep bypassed it and reached the bare net"


def test_ablation_reads_the_cache_its_posterior_names(tiny_run, monkeypatch):
    """The rows come from the simulation cache the POSTERIOR names, not from whatever checkpoint is
    newest: the ranges a channel is swept over are the ranges that network was trained on, and any
    other cache's quantiles describe a different experiment. A posterior trained with checkpointing
    off has no rows left at all, and that is a refusal rather than a silent substitution."""
    import pytest
    import torch
    from matplotlib import pyplot as plt
    from core import orchestrator
    from core.diagnostics import ablation, channel_ablation
    r = tiny_run
    close = lambda title, fig: plt.close(fig)                       # noqa: E731
    with pytest.raises(ValueError, match="checkpointing was off"):
        channel_ablation(r.cfg, r.posterior, rows=8, n_sweep=3, name="abl_nocache")
    post = orchestrator.build_posterior(r.cfg, r.prior, None, True, fig_sink=close, num_runs=2,
                                        run_size_cap=8, hidden_features=8, num_transforms=1,
                                        stop_after_epochs=1, checkpoint_every=1, new_run=True,
                                        name="abl_post")
    used = {}
    real = ablation._sweep_channels

    def _recording_sweep(emb, *a, **k):
        # R-K(2): a NAMED function, not a lambda short-circuiting on `used.setdefault(...) or real(...)`
        # -- setdefault returns the stored value, and `emb` (an nn.Module) is truthy, so `or` would
        # never call `real` at all.
        used["emb"] = emb
        return real(emb, *a, **k)

    monkeypatch.setattr(ablation, "_sweep_channels", _recording_sweep)
    d = channel_ablation(r.cfg, post, rows=12, n_sweep=5, name="abl1")
    res = d.results
    assert d.diagnostic == "ablation" and d.variant is None
    assert d.manifest.parents == {"posterior": post.id,
                                  "simulation": post.manifest.parents["simulation"]}
    assert used["emb"] is post.latent.posterior_estimator.embedding_net
    assert res["n_rows"] == 12 and 0 <= res["base_row"] < 12 and res["live_probes"] is None
    from core.SBI.statistics import SUMMARY_WIDTH
    assert len(res["channels"]) == SUMMARY_WIDTH + 1
    assert set(res["channels"][0]) == {"label", "max_disp", "rel_median", "p1", "p99", "verdict"}
    assert res["counts"]["total"] == SUMMARY_WIDTH + 1
    # N2: the invariant must cover ALL FOUR counted categories, "nonfinite" (S1) included -- a real
    # SBITEST net never diverges here, so this stays a no-op today (nonfinite == 0), but the sum would
    # silently undercount total the day it legitimately is not.
    assert (res["counts"]["constant"] + res["counts"]["invisible"] + res["counts"]["usable"]
            + res["counts"]["nonfinite"] == res["counts"]["total"])
    assert res["accepted"] == [] and d.manifest.payloads == {} and d.manifest.figures == []
    assert d.manifest.config["rows"] == 12 and d.manifest.config["n_sweep"] == 5
    # M4: matched on the DIAGNOSTIC's own name/id, not merely the generic "name it as a parent" text --
    # post itself also names the simulation cache as a parent, so a generic match would pass even if
    # the diagnostic's OWN parent link were silently dropped. dependents() lists each blocker as
    # "{kind} {name or '(unnamed)'} [{id}]" (store.py); re.escape because the id may contain regex
    # metacharacters and the brackets are literal here, not a character class.
    import re
    with pytest.raises(Exception, match=re.escape(f"diagnostic {d.name} [{d.id}]")):
        r.store.delete("simulation", post.manifest.parents["simulation"])


def test_channel_ablation_refuses_bad_rows_and_n_sweep_before_the_row_read(store):
    """I1: --rows < 1 and --n-sweep < 2 are refused before ANYTHING about the posterior or its cache
    is even touched. Before this guard, a negative --rows silently sliced ``x[:-5]``, --n-sweep 1 wrote
    a table measured at p1 only (no range at all), and 0 for either crashed deep inside
    ``training_checkpoint.load_rows``/``store.create`` well after the read. ``object()`` stands in for
    the posterior: if either guard did not fire FIRST, this would blow up on ``posterior.name`` with an
    AttributeError, not the ValueError under test -- so the test is self-checking on ordering too."""
    import pytest
    from core.diagnostics import channel_ablation
    cfg = _nad_cfg()
    before = len(store.list("diagnostic"))
    for bad_rows in (-5, 0):
        with pytest.raises(ValueError, match="--rows"):
            channel_ablation(cfg, object(), rows=bad_rows, n_sweep=5, name=f"bad_rows_{bad_rows}")
    for bad_sweep in (0, 1):
        with pytest.raises(ValueError, match="--n-sweep"):
            channel_ablation(cfg, object(), rows=10, n_sweep=bad_sweep, name=f"bad_sweep_{bad_sweep}")
    assert len(store.list("diagnostic")) == before, "a refused call must write no diagnostic directory"


def test_load_rows_max_rows_still_checks_completeness_if_the_cap_is_never_reached(tmp_path):
    """M3: max_rows only waives the batches_done completeness check when the CAP actually stopped the
    walk early -- a partial read is the point THEN. If the cache holds fewer committed batches than
    batches_done claims, asking for far more rows than the (incomplete) cache actually holds must still
    refuse: the cap can never fire, so silently returning whatever partial data exists on disk would
    hide the very corruption the completeness check exists to catch."""
    import pytest
    import torch
    from core.SBI import training_checkpoint as tc
    d = tmp_path / "cache"
    (d / "shards").mkdir(parents=True)
    # Only batches [0, 2) are committed on disk, but the caller claims 6 are done -- and max_rows asks
    # for far more rows than this (incomplete) cache holds, so the cap is never reached.
    torch.save(torch.full((2 * 5, 3), 0.0), d / "shards" / "x_000000_000002.pt")
    with pytest.raises(ValueError, match="commits 6 batches but only 2"):
        tc.load_rows(d, 6, 5, x_only=True, max_rows=1000)


def test_load_rows_max_rows_stops_before_a_later_corrupt_shard(tmp_path):
    """M6: the cap stops the walk AT the shard that reaches the count -- a corrupted LATER shard must
    never even be opened. Also covers x_only=False together with max_rows, which no earlier test did."""
    import pytest
    import torch
    from core.SBI import training_checkpoint as tc
    d = tmp_path / "cache"
    (d / "shards").mkdir(parents=True)
    for a, b in ((0, 2), (2, 4)):
        torch.save(torch.full((2 * 5, 3), float(a)), d / "shards" / f"x_{a:06d}_{b:06d}.pt")
        torch.save(torch.full((2 * 5, 2), float(a)), d / "shards" / f"th_{a:06d}_{b:06d}.pt")
    # A corrupted/truncated LATER shard: its name claims 2 batches (10 rows) but it holds only 1.
    torch.save(torch.full((1, 3), 99.0), d / "shards" / "x_000004_000006.pt")
    torch.save(torch.full((1, 2), 99.0), d / "shards" / "th_000004_000006.pt")
    # max_rows satisfied by the first two shards alone (20 rows): the walk must stop there and never
    # touch the corrupt third shard.
    x, th = tc.load_rows(d, 6, 5, x_only=True, max_rows=15)
    assert tuple(x.shape) == (15, 3) and th is None
    # x_only=False + max_rows together must also stop before the corrupt shard.
    x, th = tc.load_rows(d, 6, 5, max_rows=15)
    assert tuple(x.shape) == (15, 3) and tuple(th.shape) == (15, 2)
    # Without max_rows the walk DOES reach the corrupt shard, and its row-count mismatch is caught.
    with pytest.raises(ValueError, match=r"holds 1 rows, not the 10"):
        tc.load_rows(d, 6, 5, x_only=True)


class _FakeDPWithEst(DirectPosterior):
    """A DirectPosterior by type only (the loader's isinstance check accepts it), carrying a tiny REAL
    ``EmbeddedNet`` as its ``posterior_estimator`` so ``ablation._find_net`` has something to find --
    module-level so it pickles. M7: this is what makes the two width refusals cheap to pin without a
    real training run."""
    def __init__(self, net):
        self.posterior_estimator = net


class _EstWithEmbedding(torch.nn.Module):
    """``posterior_estimator`` with its net BEHIND a real ``.embedding_net`` attribute -- module-level
    so it pickles (a local class inside a function is not picklable at all: ``torch.save`` fails with
    ``Can't get local object``). N1a: this is what lets ``channel_ablation``'s own
    ``emb = est.embedding_net`` resolve when the caller goes on to monkeypatch ``_sweep_channels``."""
    def __init__(self, inner):
        super().__init__()
        self.embedding_net = inner


def _ablation_posterior_with_cache(store, cfg, *, name, net_input_dim, sim_cols,
                                   batches_done=1, run_size=3, wrap_embedding=False):
    """A posterior naming a real (tiny) simulation cache on disk, without a real training run (M7).
    ``net_input_dim`` controls the trained net's OWN ``input_dim`` (independent of the cache); ``sim_cols``
    controls the cache's OWN row width (independent of the net) -- so the two guards in
    ``channel_ablation`` (the net's summary width against ``SUMMARY_WIDTH + 1``, and the cache's row
    width against the posterior's ``conditioning.width``) can each be pinned in isolation. Both guards
    fire before ``est.embedding_net`` (the whole conditioning path) is ever touched, so a bare,
    forcing_dim=0 EmbeddedNet is enough -- the sweep path itself is never exercised by this stub.

    ``wrap_embedding=True`` (N1a) instead puts the net BEHIND a ``.embedding_net`` attribute, so
    ``channel_ablation``'s own ``emb = est.embedding_net`` resolves and the sweep call is actually
    reached -- needed only when the caller goes on to monkeypatch ``_sweep_channels`` itself, since a
    real forcing_dim=0 net makes ``reparam.posterior_mode``'s tier-2 detection read "spontaneous",
    which only agrees with the manifest for a genuinely SPONTANEOUS ``cfg`` (forcing_dim=0 there too).
    """
    import torch
    from core.artifacts import manifest as mf
    from core.artifacts import store as artifact_store
    from core.SBI import embedded_network
    from core.SBI.training_checkpoint import identity_digest
    net = embedded_network.EmbeddedNet(net_input_dim, 3, (4, 4), forcing_dim=0)
    est = _EstWithEmbedding(net) if wrap_embedding else net
    keys = list(cfg.params_dict) + list(cfg.rescale_params)
    ident = {"format": "training-rows/2", "prior_fingerprint": None, "n_runs": 1,
             "run_size": int(run_size), "truncation": None}
    digest = identity_digest(ident)
    with store.create("posterior", cfg, name=name) as w:
        torch.save(_FakeDPWithEst(est), str(w.payload("posterior.pt")))
        w.parents = {"prior": "20260910T100000", "simulation": digest}
        w.body = {
            "mode": cfg.observation_mode, "conditioning": mf.conditioning_block(cfg),
            "transform": {"param_keys": keys, "log_params": [],
                          "nd_lows": [float(b[0]) for _, b in cfg.params_dict.values()],
                          "nd_highs": [float(b[1]) for _, b in cfg.params_dict.values()],
                          "rescale_lows": [float(b[0]) for _, b in cfg.rescale_params.values()],
                          "rescale_highs": [float(b[1]) for _, b in cfg.rescale_params.values()],
                          "V": None, "V_orientation": "columns",
                          "fisher_eigenvalues": None, "V_digest": mf.tensor_digest(None)},
            "amortized": True, "truncation": None, "training": {}}
    sim_dir = store.kind_dir("simulation") / f"fake_{digest}"
    artifact_store.write_simulation_manifest(sim_dir, ident, batches_done=batches_done, complete=False)
    (sim_dir / "shards").mkdir(parents=True, exist_ok=True)
    torch.save(torch.zeros(batches_done * run_size, sim_cols),
              sim_dir / "shards" / f"x_{0:06d}_{batches_done:06d}.pt")
    return store.load_posterior(cfg, w.id)


def test_channel_ablation_refuses_a_mismatched_summary_width(store):
    """M7: pins the guard at ablation.py's ``n_sum != SUMMARY_WIDTH + 1`` -- a stub posterior/cache
    pair, no real training run, since the guard fires before the sweep path is ever built."""
    import pytest
    from core.diagnostics import channel_ablation
    from core.SBI.statistics import SUMMARY_WIDTH
    cfg = _nad_cfg()
    post = _ablation_posterior_with_cache(store, cfg, name="bad_width_net",
                                          net_input_dim=SUMMARY_WIDTH, sim_cols=10)
    before = len(store.list("diagnostic"))
    with pytest.raises(ValueError, match="wide summary block"):
        channel_ablation(cfg, post, rows=50, n_sweep=5, name="abl_bad_net")
    assert len(store.list("diagnostic")) == before


def test_channel_ablation_refuses_a_cache_whose_width_disagrees_with_the_posterior(store):
    """M7: pins the OTHER width guard -- the cache's own row width against
    ``posterior.manifest.body['conditioning']['width']`` -- again via a stub, no training run.

    The chi branch (``blk[:, -1]``, not ``blk[:, 0]``) is NOT pinned here: reaching it needs a full,
    successful sweep (``est.embedding_net`` actually built and called), which needs a REAL,
    internally-consistent chi net -- one whose ``chi_layout``/``chi_k_pad`` agree with the manifest,
    or ``store.load_posterior``'s own tier-2 mode detection (``reparam.posterior_mode``) raises a
    StoreError before ``channel_ablation`` is ever reached. Building that (a correctly-shaped chi
    ``EmbeddedNet`` wrapped so ``.embedding_net`` resolves, PLUS fabricated probe rows in the real chi
    column layout) is materially more scaffolding than the two width guards, which fail before any of
    it is touched -- so per M7's own escape clause, this is skipped rather than forced.
    """
    import pytest
    from core import orchestrator
    from core.diagnostics import channel_ablation
    from core.SBI.statistics import SUMMARY_WIDTH
    cfg = _nad_cfg()
    width = SUMMARY_WIDTH + 1 + orchestrator.expected_forcing_dim(cfg)
    post = _ablation_posterior_with_cache(store, cfg, name="bad_width_cache",
                                          net_input_dim=SUMMARY_WIDTH + 1, sim_cols=width - 1)
    before = len(store.list("diagnostic"))
    with pytest.raises(ValueError, match="do not describe the same measurement"):
        channel_ablation(cfg, post, rows=50, n_sweep=5, name="abl_bad_cache")
    assert len(store.list("diagnostic")) == before


def test_channel_ablation_writes_a_non_finite_displacement_as_an_explicit_verdict(store, monkeypatch):
    """N1a / S1 (spec 4.1): a channel whose sweep drove the network to NaN/Inf gets its OWN verdict --
    every numeric comparison against NaN is False, so before the fix it fell all the way through to
    "healthy" instead -- and the float fields that DERIVE from that NaN must be None in the manifest,
    because the writer refuses a non-finite float outright (``allow_nan=False``) and would otherwise
    fail well after the whole sweep had already run.

    ``_sweep_channels`` is monkeypatched WHOLESALE to a fixed record list (one channel poisoned to NaN,
    the rest finite fillers): the point is not to reproduce a real divergence, but to drive the REAL,
    unmonkeypatched code that runs AFTER it returns -- the verdict loop, the counts, the results dict
    and the manifest write -- with the least scaffolding. That still needs ``est.embedding_net`` to
    resolve (``channel_ablation`` builds it before calling ``_sweep_channels``), so this reuses M7's
    stub-posterior helper with ``wrap_embedding=True`` rather than a real training run.
    """
    import math
    from core import cli, config, registry
    from core.config import VALID_LABELS, VALID_MODELS
    from core.diagnostics import ablation, channel_ablation
    from core.SBI.statistics import SUMMARY_WIDTH
    labels = VALID_LABELS[VALID_MODELS.index("NADROWSKI")]
    # SPONTANEOUS, not the module's usual forced master.txt: wrap_embedding's real (forcing_dim=0) net
    # is tier-2 mode-detected as "spontaneous" by reparam.posterior_mode, which store.load_posterior
    # then checks against the manifest's own mode -- only a genuinely spontaneous cfg agrees.
    spont = cli.make_sim_config("NADROWSKI", labels, registry.state_dep_drift("NADROWSKI"),
                                str(config.BOUNDS_PATH / "nadrowski" / "master_spont.txt"))
    spont.hw = config.cpu_device()
    width = SUMMARY_WIDTH + 1
    post = _ablation_posterior_with_cache(store, spont, name="abl_nan_post", net_input_dim=width,
                                          sim_cols=width, wrap_embedding=True)
    poisoned = {}

    def _fake_sweep(emb, data, n_sum_, n_sweep_, sweep_labels):
        poisoned["label"] = sweep_labels[0]
        out = [(0, "__base__", 0.0, 0.0, "__base__"), (float("nan"), sweep_labels[0], -1.0, 1.0, "")]
        out += [(1.0 + 0.01 * j, sweep_labels[j], -1.0, 1.0, "") for j in range(1, n_sum_)]
        return out

    monkeypatch.setattr(ablation, "_sweep_channels", _fake_sweep)
    d = channel_ablation(spont, post, rows=5, n_sweep=5, name="abl_nan")
    res = d.results
    poisoned_rec = next(c for c in res["channels"] if c["label"] == poisoned["label"])
    assert "NON-FINITE" in poisoned_rec["verdict"]
    assert "healthy" not in poisoned_rec["verdict"].lower()
    # the fields that DERIVE from the NaN displacement (d itself, and d/med) must be None; p1/p99 are
    # independent quantile bounds computed before the sweep and stay finite.
    assert poisoned_rec["max_disp"] is None and poisoned_rec["rel_median"] is None
    assert math.isfinite(poisoned_rec["p1"]) and math.isfinite(poisoned_rec["p99"])
    assert res["counts"]["nonfinite"] == 1
    # the manifest was actually written and loads back clean -- proof no NaN literal reached
    # json.dumps(allow_nan=False), which would have raised mid-write rather than merely mislabelling.
    reloaded = store.load_diagnostic(d.id)
    assert reloaded.results == res
