"""Diagnostics: the ``diagnostic`` store kind, the shared feature-set and seeding helpers, and the
five diagnostic functions.

Everything here is CPU-sized and stubbed where a simulation would otherwise dominate: a diagnostic is
a measurement ABOUT a trained posterior, so the thing worth pinning is what it reads, what it refuses
and what it writes -- not the numbers a real training run would give it.
"""
import pytest
import torch
from sbi.inference import DirectPosterior

from core.artifacts import LoadedDiagnostic
from core.artifacts import store as st
from core.refusals import Refusal
from tests._fixtures import _nad_cfg, _posterior_artifact


class _LatentPriorStub:
    """The ``SBIPriorWrapper(latent)`` shape a trained posterior pickles: ``.gen_dist`` is the latent
    training prior, which is what the off-ground-truth points are drawn from."""
    def __init__(self, dim):
        import torch
        self.gen_dist = torch.distributions.Independent(
            torch.distributions.Normal(torch.zeros(dim), torch.ones(dim)), 1)


def _diagnostic(store, cfg, *, name="diag", parents=None, variant=None):
    """A diagnostic artifact written the way every diagnostic function writes one."""
    with store.create("diagnostic", cfg, name=name) as w:
        w.parents = dict(parents or {})
        w.config.update({"repeats": 2})
        w.body = {"diagnostic": "sbc", "variant": variant, "settings": {"repeats": 2},
                  "results": {"n_valid": 8, "accepted": []}}
    return w


def test_a_diagnostic_loads_without_a_config_and_blocks_deleting_what_it_names(store):
    """The whole contract of the diagnostic kind: its own directory and manifest body; a loader that
    takes a ref and NOTHING else, because there is no config to verify a measurement against and
    nothing is ever trained from one; and, because ``dependents`` walks every kind, the artifacts it
    names cannot be deleted out from under it without ``force``.
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


def test_the_chi_feature_set_drops_group_g_and_adds_the_fisher_block(caplog):
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

    # The banner is a record at INFO from the feature-set module: same words, but the
    # window shows it plain, the tool prints it on stdout and the run's log.txt keeps it.
    caplog.clear()
    fs.describe_features(chi_cfg)
    got = [(r.name, r.levelname, r.getMessage()) for r in caplog.records]
    assert [(n, lvl) for n, lvl, _ in got] == [("core.diagnostics.feature_sets", "INFO")] * 2, got
    assert got[0][2].startswith("[mode] CHI: feature rows = 30 spontaneous + 12 chi = 42"), got
    assert "f_scale is informative here" in got[1][2], got
    caplog.clear()
    fs.describe_features(plain)
    got = [(r.levelname, r.getMessage()) for r in caplog.records]
    assert len(got) == 1 and got[0][0] == "INFO", got
    assert got[0][1].startswith(f"[mode] {plain.observation_mode.upper()}: feature rows = 41"), got


def test_the_diagnostic_guards_refuse_naming_no_flag_or_file():
    """SystemExit was right for a script and wrong everywhere else; a bare ValueError that named
    `--no-chi` and `--bounds` was right for the tool and wrong for the window, whose user has no flag
    to pass. The guards now raise Refusal: one refusal kind, a neutral message that names the
    diagnostic, the mode and the model but no flag, subcommand or file, and `field=None`, because no
    single control answers them -- each front end appends its own "how to fix here" from its table,
    and for None that is nothing.
    """
    import inspect
    from core import cli, config, registry
    from core.config import VALID_LABELS, VALID_MODELS
    from core.diagnostics import feature_sets as fs
    banned = ("--no-chi", "--bounds", "--cell", "python -m core", "master.txt", "master_weak.txt",
              str(config.BOUNDS_PATH), str(config.CELL_PATH))

    chi_cfg = _nad_cfg(chi_mode=True, chi_n_freqs=4)
    with pytest.raises(Refusal, match="has not been generalised to chi") as e:
        fs.assert_not_chi(chi_cfg, "identifiability laplace")
    assert e.value.field is None and "identifiability laplace" in str(e.value)
    assert "jacobian" in str(e.value), "the chi-aware alternative is still named, as a diagnostic"
    assert not any(b in str(e.value) for b in banned), str(e.value)
    fs.assert_not_chi(_nad_cfg(), "identifiability laplace")            # not chi: no refusal

    other = _nad_cfg()
    other.model = "HOPF"
    with pytest.raises(Refusal, match="Nadrowski-specific") as e:
        fs.assert_nadrowski(other, "the printed ND parameter names are Nadrowski's")
    assert e.value.field is None and "HOPF" in str(e.value) and "nadrowski" in str(e.value)
    assert not any(b in str(e.value) for b in banned), str(e.value)
    fs.assert_nadrowski(_nad_cfg())                                     # NADROWSKI: no refusal

    spont = cli.make_sim_config("NADROWSKI", VALID_LABELS[VALID_MODELS.index("NADROWSKI")],
                                registry.state_dep_drift("NADROWSKI"),
                                str(config.BOUNDS_PATH / "nadrowski" / "master_spont.txt"))
    assert not spont.has_forcing, "master_spont.txt declares no Forcing section"
    with pytest.raises(Refusal, match="no amp/freq/phase to read") as e:
        fs.assert_forced(spont, "identifiability jacobian")
    assert e.value.field is None and "SPONTANEOUS" in str(e.value)
    # the refusal depends on the bounds file, which is never resolved from the cell: say so, without
    # naming a flag or a file
    assert "Forcing section" in str(e.value) and "bounds file" in str(e.value), str(e.value)
    assert not any(b in str(e.value) for b in banned), str(e.value)
    fs.assert_forced(_nad_cfg(), "identifiability jacobian")            # master.txt declares Forcing

    for guard in (fs.assert_not_chi, fs.assert_nadrowski, fs.assert_forced):
        src = inspect.getsource(guard)
        assert "SystemExit" not in src and "raise ValueError" not in src, guard.__name__


def test_seeded_restores_the_callers_rng():
    """Seed ONCE, let the stream run on inside the block, and hand the caller's RNG back on the way
    out. The tool's own tests call main(argv) in-process, so a leaked seed would make one test's
    numbers depend on which tests ran before it; and re-seeding per stage is what would make the
    calibration set replay the training strata -- and the calibration's operating-point count is
    t_scale's effective sample size -- which is why there is one context and not a seed argument on
    every stage.
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


@pytest.mark.gpu
def test_a_cpu_seeded_block_leaves_every_cuda_generator_alone():
    """``torch.manual_seed`` reseeds EVERY CUDA device as well as the CPU, and a CPU block forks no
    CUDA generator -- ``fork_rng(devices=[cpu])`` raises -- so the old ``seeded(seed, cpu)`` left each
    card's generator pinned at the block's seed: two CPU FDT runs at seed 5, each followed by a CUDA
    draw, drew identical CUDA numbers. That is the hazard core/SBI/decorrelate.py documents, reached
    through the context whose whole job is to hand the caller's streams back. A CPU block now seeds
    the CPU generator and numpy only.

    The context lives in core/rng.py, and core.diagnostics.rng re-exports the SAME object, so the
    diagnostics and `smoke` -- which a tool test patches through core.diagnostics.rng -- keep their
    import."""
    import core.diagnostics.rng
    import core.rng
    from core import config
    from core.rng import seeded

    assert core.diagnostics.rng.seeded is core.rng.seeded
    cpu = config.cpu_device().device
    before = [torch.cuda.get_rng_state(i) for i in range(torch.cuda.device_count())]
    with seeded(5, cpu):
        first = torch.randn(3)
    with seeded(5, cpu):
        assert torch.equal(torch.randn(3), first), "the CPU seed must still take effect"
    after = [torch.cuda.get_rng_state(i) for i in range(torch.cuda.device_count())]
    assert all(torch.equal(a, b) for a, b in zip(before, after)), \
        "a CPU seeded() block reseeded a CUDA generator and did not restore it"


def test_the_calibration_draw_is_three_helpers_with_the_stratification_seam(store, monkeypatch, caplog):
    """sbc_repeats draws its per-repeat calibration set through EXACTLY the code validate_calibration
    draws its own through. That is what gives the repeat-SBC run the four things the retired
    repeat-SBC script never had: check_basis, the t_scale-override mirror in the reference sample,
    the kept-fraction line and the LoadedPrior path. A second copy of the wrap is precisely how a
    repeat-SBC run comes to draw theta* from the FULL prior while the flow was trained on the region
    -- the calibrate-on-the-region rule, silently inverted.

    So the draw is three named helpers, and the one thing sbc_repeats varies -- chi_k_fixed, which
    runs one probe-count stratum at a time -- is a keyword on the middle one. validate_calibration
    itself always passes None: its SBC is the POOLED one, over the same mixture of counts training saw.
    """
    import ast
    import inspect
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

    caplog.clear()
    vlp, T_out, truncation = orch._calibration_prior(cfg, trunc_post, lp)
    assert truncation is region and T_out is trunc_post.posterior.T
    assert isinstance(vlp, _tr.TruncatedLatentPrior) and vlp.region is region
    said = [(r.name, r.levelname, r.getMessage()) for r in caplog.records]
    assert any(n == "core.orchestrator" and lv == "INFO"
               and m.startswith("[tsnpe] calibration draws theta* from the PRIOR RESTRICTED to ")
               for n, lv, m in said), said
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


def test_sbc_records_the_rank_uniformity_half_of_the_verdict_per_repeat(store, monkeypatch, caplog):
    """The repeated SBC computes no joint coverage, so it can judge only the rank half of the
    calibration verdict: per repeat, whether every parameter's KS p reaches 0.05 / (the number of
    parameters), a p exactly at the threshold passing; and the fraction of repeats that pass. The
    record and its line say it is that half only."""
    from matplotlib import pyplot as plt
    from core.diagnostics import sbc_repeats
    from tests._fixtures import stub_calibration_battery
    good, bad, edge = [0.5] * 13, [0.5] * 12 + [0.003], [0.05 / 13] * 13
    seen = stub_calibration_battery(monkeypatch, ks_pvals=[good, bad, good, edge])
    caplog.set_level("INFO", logger="core")
    d = sbc_repeats(_nad_cfg(), seen["posterior"], seen["prior"], repeats=4, n_cal=8,
                    num_posterior_samples=10, fig_sink=lambda t, f: plt.close(f), store=store)
    assert d.results["rank_verdict"] == {"per_repeat": [True, False, True, True], "fraction_passed": 0.75,
                                         "scope": "rank-uniformity half only"}
    lines = [r.getMessage() for r in caplog.records if r.name == "core.diagnostics.sbc"]
    assert any("rank-uniformity half" in m and "3/4 repeats pass" in m for m in lines), lines


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
    # The writer's own fig_sink saves the PNG and then forwards to whichever sink was given, so a no-op
    # sink here would leak one matplotlib figure per call -- two, across this test's two calls.
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
                                              "c2st_ranks_median", "assumed"}
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
    # Each is a Refusal carrying the knob's field key, and the message names no flag -- the tool
    # appends `(--repeats)` and friends from its own table.
    with pytest.raises(Refusal, match="at least 1") as e:
        sbc_repeats(r.cfg, r.posterior, r.prior, repeats=0, n_cal=8, fig_sink=r.sink)
    assert e.value.field == "repeats"
    with pytest.raises(Refusal, match="at least 1") as e:
        sbc_repeats(r.cfg, r.posterior, r.prior, repeats=1, n_cal=0, fig_sink=r.sink)
    assert e.value.field == "n_cal"
    with pytest.raises(Refusal, match="at least 1") as e:
        sbc_repeats(r.cfg, r.posterior, r.prior, repeats=1, n_cal=8, num_posterior_samples=0,
                    fig_sink=r.sink)
    assert e.value.field == "num_posterior_samples"
    with pytest.raises(Refusal, match=r"chi\(omega\) mode") as e:
        sbc_repeats(r.cfg, r.posterior, r.prior, repeats=1, n_cal=8, chi_k_fixed=2, fig_sink=r.sink)
    assert e.value.field == "chi_k_fixed" and "--chi-k-fixed" not in str(e.value), str(e.value)
    # ...and in chi mode, a count outside 1..chi_k_pad is the same field's refusal, here -- not a bare
    # ValueError from gen_training_data inside the first calibration draw, after store.create
    chi_cfg = r.cfg.copy_for_run()
    chi_cfg.chi_mode = True
    for bad in (chi_cfg.chi_k_pad + 1, 0):
        with pytest.raises(Refusal, match="fixed chi probe count must be between 1 and") as e:
            sbc_repeats(chi_cfg, r.posterior, r.prior, repeats=1, n_cal=8, chi_k_fixed=bad,
                        fig_sink=r.sink)
        assert e.value.field == "chi_k_fixed" and "--chi-k-fixed" not in str(e.value), str(e.value)
        assert f"got {bad}" in str(e.value), str(e.value)
    with pytest.raises(StoreError, match="already exists"):
        sbc_repeats(r.cfg, r.posterior, r.prior, repeats=1, n_cal=8, fig_sink=r.sink, name="sbc_taken")
    with pytest.raises(ValueError, match="not the one this posterior was trained with"):
        sbc_repeats(r.cfg, r.posterior, r.other_prior(), repeats=1, n_cal=8, fig_sink=r.sink)
    assert drawn == [], "a calibration set was simulated before the refusals"
    assert calibrated == [], \
        "_calibration_prior ran before a refusal fired -- every guard above must precede it"
    assert len(r.store.list("diagnostic")) == before + 1, \
        "a refused sbc_repeats call wrote a diagnostic of its own (only the name taken above should exist)"


@pytest.mark.parametrize("n_pooled, nps, want", [
    (20000, 1000, 91),    # cap 100; 91 divides 1001, every bin holds 11 ranks
    (20000, 500, None),   # cap 50; 501 = 3 x 167 has no divisor in [25, 50]: the cap, never 3
    (20000, 40, 4),       # 41 is prime: the cap (4), never 1
    (20000, 100, 10),     # 101 is prime: the cap (10), never 1
    (200, 1000, 7),       # N // 20 = 10 binds; 7 divides 1001 and lies in [5, 10]
    (100, 1000, 5),       # N // 20 = 5 binds; no divisor of 1001 in [2, 5]: the cap
    (10, 1000, 1),        # N // 20 = 0: the floor of 1
])
def test_the_sbc_rank_histogram_bin_count_stays_near_its_cap(n_pooled, nps, want):
    """The pooled histogram's bin count: the largest divisor of nps + 1 in [cap // 2, cap], so every bin
    holds the same number of integer ranks, and otherwise the cap itself. A pure largest-divisor rule
    collapsed to 1 bin when nps + 1 is prime and to 3 bins at nps = 500 -- a useless rank histogram."""
    import numpy as np
    from core.diagnostics.sbc import _rank_hist_bins
    cap = max(1, min(n_pooled // 20, (nps + 1) // 10))
    got = _rank_hist_bins(n_pooled, nps)
    assert max(1, cap // 2) <= got <= cap, (got, cap)
    if want is not None:
        assert got == want, (got, want)
    divisors_in_range = [d for d in range(max(1, cap // 2), cap + 1) if (nps + 1) % d == 0]
    if divisors_in_range:
        assert got == max(divisors_in_range)
        per_bin, _ = np.histogram(np.arange(nps + 1), bins=got)
        assert len(set(per_bin.tolist())) == 1, per_bin
    else:
        assert got == cap


def test_sbc_prints_small_p_values_as_numbers_and_bins_ranks_at_least_ten_wide(tiny_run, monkeypatch,
                                                                             caplog):
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
    caplog.clear()
    sbc_repeats(r.cfg, r.posterior, r.prior, repeats=10, n_cal=N, num_posterior_samples=nps,
                fig_sink=lambda title, fig: plt.close(fig))
    # The report table is records at INFO from the sbc module, one per line as the prints were; the
    # blank line the header's print opened with went with the print.
    sbc_lines = [(r.levelname, r.getMessage()) for r in caplog.records if r.name == "core.diagnostics.sbc"]
    assert ("INFO", "=== KS p-value distribution over repeats (sorted by median; low = miscalibrated) ===") \
        in sbc_lines, sbc_lines
    assert {lvl for lvl, _ in sbc_lines} == {"INFO"}, sbc_lines
    out = "\n".join(msg for _, msg in sbc_lines)
    assert seen["num_bins"] <= (nps + 1) // 10, seen["num_bins"]
    # sbi draws plt.hist(ranks, bins=<int>): edges linspace(min, max, bins + 1) with the last bin
    # closed, so over ranks 0..nps each bin holds the same number of integer ranks only when the bin
    # count divides nps + 1. At 100 bins the closed last bin holds 11 ranks against 10 elsewhere.
    assert (nps + 1) % seen["num_bins"] == 0, seen["num_bins"]
    per_bin, _ = np.histogram(np.arange(nps + 1), bins=seen["num_bins"])
    assert len(set(per_bin.tolist())) == 1, per_bin
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
    # A CYCLIC SHIFT, not the identity -- V(order=range(P)) is the identity matrix, which is
    # SYMMETRIC, so reading it transposed (a rotation saved transposed: rows instead of columns) is
    # indistinguishable from reading it correctly and every assertion below would still pass. A
    # cyclic shift of P>2 elements is not an involution, so its permutation matrix is genuinely
    # non-symmetric.
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


def test_identifiability_rotation_reports_absent_eigenvalues_and_refuses_an_absent_rotation(store, caplog):
    """Eigenvalues absent is a REPORT: a rotation whose eigenvalues never reached the record -- a run
    resumed from a checkpoint written before they were kept beside the rotation, or a narrowing round
    that inherited none -- still has loadings that answer "which direction is worst". The warning
    names the records that can share the rotation and still hold the scale (a narrowing round's
    amortized ancestor, the posterior of the run that started a resumed run's simulation cache), and
    ends on what is left when none does: a new amortized run computes a rotation of its own. V
    absent is a REFUSAL: there is no basis to decompose at all."""
    import pytest
    import torch
    from core.diagnostics import identifiability_rotation
    cfg = _nad_cfg()
    P = len(cfg.params_dict) + len(cfg.rescale_params)
    w = _rotation_posterior(store, cfg, V=torch.eye(P, dtype=torch.float64), evals=None)
    d = identifiability_rotation(cfg, store.load_posterior(cfg, w.id), name="rot_noev")
    assert d.results["eigenvalues"] is None and len(d.results["directions"]) == P
    assert d.results["directions"][0]["eigenvalue"] is None
    # A WARNING, said once: the explanatory lines travel in one record, so the pane's triangle and
    # the tool's "warning: " prefix mark the block once and log.txt stamps it once.
    warned = [r.getMessage() for r in caplog.records
              if r.name == "core.diagnostics.identifiability" and r.levelname == "WARNING"]
    assert len(warned) == 1 and warned[0].startswith("[eigenvalues] NOT STORED for this artifact.\n"), warned
    assert warned[0].endswith("\n  amortized training run that does not resume a cache computes a rotation, with\n"
                              "  eigenvalues of its own."), warned
    flat = _posterior_artifact(store, cfg, name="norot", V=None)
    with pytest.raises(Refusal, match="records no Fisher rotation") as e:
        identifiability_rotation(cfg, store.load_posterior(cfg, flat.id), name="rot_none")
    assert e.value.field == "posterior", "one control answers it: point at another posterior"
    assert [s.name for s in store.list("diagnostic") if s.name == "rot_none"] == []


def test_identifiability_rotation_refuses_n_worst_over_p(store):
    """``W[i, P-n_worst:]`` is a NEGATIVE slice once ``n_worst > P`` -- Python reads it from the end
    instead of raising, so ``bottom_share`` would silently sum fewer than n_worst directions.
    Refused before store.create, so no directory is written."""
    import pytest
    import torch
    from core.diagnostics import identifiability_rotation
    cfg = _nad_cfg()
    P = len(cfg.params_dict) + len(cfg.rescale_params)
    w = _rotation_posterior(store, cfg, V=torch.eye(P, dtype=torch.float64), evals=None)
    with pytest.raises(Refusal, match=f"at most {P}") as e:
        identifiability_rotation(cfg, store.load_posterior(cfg, w.id), n_worst=P + 1, name="rot_nw")
    assert e.value.field == "n_worst" and f"got {P + 1}" in str(e.value), str(e.value)
    assert "(default 3)" in str(e.value), "the width AND the default, as the direction refusal does"
    assert [s.name for s in store.list("diagnostic") if s.name == "rot_nw"] == []


def test_identifiability_rotation_writes_a_non_finite_eigenvalue_as_none(store):
    """A non-finite eigenvalue reaching identifiability_rotation's results must become None, not
    raise at the manifest write (allow_nan=False) -- or worse, silently succeed with a NaN embedded
    in a JSON field no downstream reader expects.

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
        # An alternating +-0.01 "wobble" over the ensemble AXIS (same for every feature column),
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
    t_obs_before = cfg.T_obs                          # captured BEFORE the call, not guessed after
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
    # EQUALS its pre-call value, not merely "not 2.0" -- 2.0 is SECONDS while the script's own
    # (never-ported) write was `cfg.T_obs = t_obs_s * hz` in CELL units, which is essentially never
    # exactly 2.0, so the old check would not have noticed that write happening at all.
    assert cfg.T_obs == t_obs_before, "the diagnostic must never write cfg.T_obs"
    assert {m for m, _, _ in calls} == {4, 16} and {n for _, _, n in calls} == {int(2.0 * cfg.get_unit_conversion_factor("s") / cfg.dt_exp)}
    # The noise-floor ensemble draws INDEPENDENT noise (crn False); the +-d arms share common
    # random numbers (crn True), or the finite difference is noise over 2d
    assert {c for mm, c, _ in calls if mm == 16} == {False}, calls
    assert {c for mm, c, _ in calls if mm == 4} == {True}, calls

    # This fake makes the numbers a closed form, derived here and pinned so a units/arithmetic
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
    THREE guards fire before store.create, so no directory is written -- chi (a chi posterior
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
    with pytest.raises(Refusal, match="chi") as e:
        identifiability_laplace(chi_cfg, store.load_posterior(chi_cfg, w.id), name="lap_chi")
    assert e.value.field is None and "--no-chi" not in str(e.value), str(e.value)
    assert [s for s in store.list("diagnostic") if s.name == "lap_chi"] == []

    spont = cli.make_sim_config("NADROWSKI", VALID_LABELS[VALID_MODELS.index("NADROWSKI")],
                                registry.state_dep_drift("NADROWSKI"),
                                str(config.BOUNDS_PATH / "nadrowski" / "master_spont.txt"))
    spont.hw = config.cpu_device()
    Ps = len(spont.params_dict) + len(spont.rescale_params)
    ws = _posterior_artifact(store, spont, name="spont_post", V=torch.eye(Ps, dtype=torch.float64),
                             prior=_LatentPriorStub(Ps))
    with pytest.raises(Refusal, match="no amp/freq/phase to read") as e:
        identifiability_laplace(spont, store.load_posterior(spont, ws.id), name="lap_spont")
    assert e.value.field is None
    assert [s for s in store.list("diagnostic") if s.name == "lap_spont"] == []

    forced_cfg = _forced_nad_cfg()
    Pf = len(forced_cfg.params_dict) + len(forced_cfg.rescale_params)
    wf = _posterior_artifact(store, forced_cfg, name="npts_post", V=torch.eye(Pf, dtype=torch.float64),
                             prior=_LatentPriorStub(Pf))
    with pytest.raises(Refusal, match="at least 1") as e:
        identifiability_laplace(forced_cfg, store.load_posterior(forced_cfg, wf.id), n_points=0,
                                name="lap_npts")
    assert e.value.field == "n_points"
    assert [s for s in store.list("diagnostic") if s.name == "lap_npts"] == []


def test_laplace_and_jacobian_refuse_bad_probe_settings_before_any_simulation(store, monkeypatch):
    """A t_obs_s that computes a non-positive OR sub-one n_obs (zero, negative, or a tiny positive
    value that still floors to 0 samples), or a non-finite t_obs_s (NaN would otherwise crash at
    int(nan) rather than refuse), and an m_noise below 10 (which cannot estimate a feature-noise floor
    at all) -- all fire before store.create, for both simulating diagnostics."""
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

    def _refused(field, fn, *a, **kw) -> str:
        """The call raises a Refusal carrying `field`, and its message names no flag: the tool
        appends `(--t-obs)` and friends itself, from its own table."""
        with pytest.raises(Refusal) as e:
            fn(*a, **kw)
        assert e.value.field == field, (field, str(e.value))
        assert "--" not in str(e.value), str(e.value)
        return str(e.value)

    assert "greater than 0" in _refused("t_obs", identifiability_laplace, cfg, lp, t_obs_s=0.0,
                                        name="lap_bad_t")
    # A TINY positive value must also refuse -- n_obs floors to 0 samples, not a valid
    # recording, and the old `t_obs_s <= 0` guard let it straight through.
    assert "at least one sample" in _refused("t_obs", identifiability_laplace, cfg, lp,
                                             t_obs_s=1e-12, name="lap_tiny_t")
    assert "at least 10" in _refused("m_noise", identifiability_laplace, cfg, lp, m_noise=4,
                                     name="lap_bad_m")
    assert "greater than 0" in _refused("t_obs", identifiability_jacobian, cfg, t_obs_s=-1.0,
                                        name="jac_bad_t")
    assert "at least one sample" in _refused("t_obs", identifiability_jacobian, cfg, t_obs_s=1e-12,
                                             name="jac_tiny_t")
    assert "at least 10" in _refused("m_noise", identifiability_jacobian, cfg, m_noise=4,
                                     name="jac_bad_m")
    for nm in ("lap_bad_t", "lap_tiny_t", "lap_bad_m", "jac_bad_t", "jac_tiny_t", "jac_bad_m"):
        assert [s for s in store.list("diagnostic") if s.name == nm] == []

    # The arm ensemble, the relative step and the validity floor. --m 0 used to run the whole
    # noise ensemble and then every arm at batch 0; --rel 0 on a zero-valued truth divided by zero and
    # reached lstsq after the spend; --min-valid outside (0, 1] silently accepted or refused every arm.
    bad = [("m", {"m": 0}), ("rel", {"rel": 0.0}), ("rel", {"rel": float("nan")}),
           ("rel", {"rel": -0.02}), ("min_valid", {"min_valid": 0.0}), ("min_valid", {"min_valid": 1.5})]
    for knob, kw in bad:
        _refused(knob, identifiability_laplace, cfg, lp, name=f"lap_bad_{knob}", **kw)
        _refused(knob, identifiability_jacobian, cfg, name=f"jac_bad_{knob}", **kw)
        for nm in (f"lap_bad_{knob}", f"jac_bad_{knob}"):
            assert [s for s in store.list("diagnostic") if s.name == nm] == []

    # rotation used to CLAMP n_worst / top_n to 1: a --n-worst 0 quietly became 1, where a setting
    # must be refused, never silently clamped
    from core.diagnostics import identifiability_rotation
    rot = store.load_posterior(cfg, _rotation_posterior(store, cfg, V=torch.eye(P, dtype=torch.float64),
                                                        evals=None).id)
    assert "at least 1" in _refused("n_worst", identifiability_rotation, cfg, rot, n_worst=0,
                                    name="rot_nw0")
    assert "at least 1" in _refused("top_n", identifiability_rotation, cfg, rot, top_n=0,
                                    name="rot_top0")
    assert [s.name for s in store.list("diagnostic") if s.name in ("rot_nw0", "rot_top0")] == []


def test_laplace_raw_does_not_leak_its_crn_seed(monkeypatch):
    """``_laplace_raw``'s CRN reseeds (``_SF``/``_SS``) must not escape the call, exactly as
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
    # torch.manual_seed is process-global and NEVER restores on its own -- fork_rng here is not
    # the thing under test (that is _laplace_raw's OWN fork_rng), it is this TEST keeping its own
    # seeding from leaking into whichever test runs next.
    with torch.random.fork_rng():
        torch.manual_seed(123)
        before = torch.get_rng_state()
        identifiability._laplace_raw(cfg, nd, res, force, 4, True, 20)
        after = torch.get_rng_state()
    assert torch.equal(before, after), "the CRN reseed leaked into the caller's RNG stream"


def test_laplace_points_draw_independent_noise_but_the_whole_run_reproduces(monkeypatch):
    """A regression the seed-leak fix above once introduced: fork_rng must wrap ONLY the CRN-seeded
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
        "the stream's progression between them, exactly what the seed-leak fix once broke"
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
    # A per-(feature, parameter) weight matrix, deterministic and with NO shared structure across
    # parameters -- unlike the fake this replaces, whose response depended ONLY on theta.sum(),
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
        # Relative noise (proportional to each channel's own scale), not a fixed absolute
        # magnitude -- see the module's own comment on the same choice in _dead_channels' docstring.
        lean = 0.001 * (W @ theta)
        response = base * (1.0 + lean)
        feats = response[None, :] * (1.0 + rng.normal(0.0, 1e-3, size=(int(m), n_feat)))
        # ONE genuinely dead channel -- ~5.0 plus INDEPENDENT (theta-blind) noise 5 orders of
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
    # the m_noise ensemble without common random numbers, every +-d arm with them
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
    # A rank-1 J (the old fake) makes EVERY pair degenerate -- this full-rank response must not.
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

    # The restored npz arrays -- shapes only (the numbers are the fake's, not a fixed science
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
    """The map's row count and payload must track the MODE's own feature set -- 30 spontaneous + 3K
    chi under chi, not the 41-feature forced set -- so hard-coding the forced width anywhere in this
    path would be caught here."""
    import numpy as np
    from core import cli, config, registry
    from core.config import VALID_LABELS, VALID_MODELS
    from core.diagnostics import identifiability, identifiability_jacobian
    labels = VALID_LABELS[VALID_MODELS.index("NADROWSKI")]
    cell = str(config.CELL_PATH / "nadrowski" / "master_weak.txt")
    # The smallest HONEST setup -- chi mode ignores the cell's own drive entirely (assert_forced
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
    """The miswired test above sets every other slot to ZERO, so a channel-OFFSET regression (reading
    cos/sin from the wrong pair of columns) could still coincidentally break cos^2+sin^2==1 and pass
    for the wrong reason. A genuinely correct (0.6, 0.8) pair (0.36+0.64==1) must NOT raise, which a
    wrong-offset read (landing on a zeroed slot) generally would."""
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

    # The net's own weight init AND the data draw are BOTH seeded inside one fork_rng, so
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
    # The base point must be a REAL row of data (the docstring's whole point), not a column-wise
    # median vector -- which, for i.i.d. Gaussian data, essentially never matches any actual row.
    assert any(torch.equal(base[0], data[i]) for i in range(data.shape[0])), \
        "the base point must be a real row of data, not a synthesized median vector"
    for j, call in enumerate(seen[1:]):
        assert torch.equal(call[:, n_sum:], base[:, n_sum:].expand(call.shape[0], -1)), \
            "the forcing block moved during a summary sweep"
        # EXACTLY the j-th column, not merely "at most one" -- the data is random normal, so a
        # column/label mix-up (sweeping column k while labelling and counting it as column j) would
        # still pass a "<= 1" check but fails this one.
        varying = [k for k in range(n_sum) if call[:, k].min() != call[:, k].max()]
        assert varying == [j], "the swept column did not match the sweep's own column index"
    healthy_max = max(r[0] for r in rows)
    # Guards against a near-dead random init shrinking the bound (below) into the noise floor --
    # a healthy sweep on an untrained net is still order 1 here, not order 1e-6.
    assert healthy_max > 1e-2, "an unstandardized sweep must move the embedding by more than noise"

    zeroing = torch.nn.Sequential(_Zero(), net)
    rows0 = ablation._sweep_channels(zeroing, data, n_sum, 5, labels)[1:]
    # An absolute bound (== 0.0, or even <= 1e-6) still under-tolerates. The net's init is SEEDED
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
        # A NAMED function, not a lambda short-circuiting on `used.setdefault(...) or real(...)`
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
    # The invariant must cover ALL FOUR counted categories, "nonfinite" included -- a real
    # SBITEST net never diverges here, so this stays a no-op today (nonfinite == 0), but the sum would
    # silently undercount total the day it legitimately is not.
    assert (res["counts"]["constant"] + res["counts"]["invisible"] + res["counts"]["usable"]
            + res["counts"]["nonfinite"] == res["counts"]["total"])
    assert res["accepted"] == [] and d.manifest.payloads == {} and d.manifest.figures == []
    assert d.manifest.config["rows"] == 12 and d.manifest.config["n_sweep"] == 5
    # Matched on the DIAGNOSTIC's own name/id, not merely the generic "name it as a parent" text --
    # post itself also names the simulation cache as a parent, so a generic match would pass even if
    # the diagnostic's OWN parent link were silently dropped. dependents() lists each blocker as
    # "{kind} {name or '(unnamed)'} [{id}]" (store.py); re.escape because the id may contain regex
    # metacharacters and the brackets are literal here, not a character class.
    import re
    with pytest.raises(Exception, match=re.escape(f"diagnostic {d.name} [{d.id}]")):
        r.store.delete("simulation", post.manifest.parents["simulation"])


def test_channel_ablation_refuses_bad_rows_and_n_sweep_before_the_row_read(store):
    """rows < 1 and n_sweep < 2 are refused before ANYTHING about the posterior or its cache is even
    touched. Before this guard, a negative --rows silently sliced ``x[:-5]``, --n-sweep 1 wrote
    a table measured at p1 only (no range at all), and 0 for either crashed deep inside
    ``training_checkpoint.load_rows``/``store.create`` well after the read. Each is a Refusal
    carrying the knob's field key and naming no flag. ``object()`` stands in for the posterior:
    if either guard did not fire FIRST, this would blow up on ``posterior.name`` with an
    AttributeError, not the Refusal under test -- so the test is self-checking on ordering too."""
    from core.diagnostics import channel_ablation
    cfg = _nad_cfg()
    before = len(store.list("diagnostic"))
    for bad_rows in (-5, 0):
        with pytest.raises(Refusal, match="at least 1") as e:
            channel_ablation(cfg, object(), rows=bad_rows, n_sweep=5, name=f"bad_rows_{bad_rows}")
        assert e.value.field == "rows" and "--rows" not in str(e.value), str(e.value)
    for bad_sweep in (0, 1):
        with pytest.raises(Refusal, match="at least 2") as e:
            channel_ablation(cfg, object(), rows=10, n_sweep=bad_sweep, name=f"bad_sweep_{bad_sweep}")
        assert e.value.field == "n_sweep" and "--n-sweep" not in str(e.value), str(e.value)
    assert len(store.list("diagnostic")) == before, "a refused call must write no diagnostic directory"


def test_load_rows_max_rows_still_checks_completeness_if_the_cap_is_never_reached(tmp_path):
    """max_rows only waives the batches_done completeness check when the CAP actually stopped the walk
    early -- a partial read is the point THEN. If the cache holds fewer committed batches than
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
    """The cap stops the walk AT the shard that reaches the count -- a corrupted LATER shard must never
    even be opened. Also covers x_only=False together with max_rows, which no earlier test did."""
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
    module-level so it pickles. This is what makes the two width refusals cheap to pin without a real
    training run."""
    def __init__(self, net):
        self.posterior_estimator = net


class _EstWithEmbedding(torch.nn.Module):
    """``posterior_estimator`` with its net BEHIND a real ``.embedding_net`` attribute -- module-level
    so it pickles (a local class inside a function is not picklable at all: ``torch.save`` fails with
    ``Can't get local object``). This is what lets ``channel_ablation``'s own
    ``emb = est.embedding_net`` resolve when the caller goes on to monkeypatch ``_sweep_channels``."""
    def __init__(self, inner):
        super().__init__()
        self.embedding_net = inner


def _ablation_posterior_with_cache(store, cfg, *, name, net_input_dim, sim_cols,
                                   batches_done=1, run_size=3, wrap_embedding=False):
    """A posterior naming a real (tiny) simulation cache on disk, without a real training run.
    ``net_input_dim`` controls the trained net's OWN ``input_dim`` (independent of the cache); ``sim_cols``
    controls the cache's OWN row width (independent of the net) -- so the two guards in
    ``channel_ablation`` (the net's summary width against ``SUMMARY_WIDTH + 1``, and the cache's row
    width against the posterior's ``conditioning.width``) can each be pinned in isolation. Both guards
    fire before ``est.embedding_net`` (the whole conditioning path) is ever touched, so a bare,
    forcing_dim=0 EmbeddedNet is enough -- the sweep path itself is never exercised by this stub.

    ``wrap_embedding=True`` instead puts the net BEHIND a ``.embedding_net`` attribute, so
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
    """Pins the guard at ablation.py's ``n_sum != SUMMARY_WIDTH + 1`` -- a stub posterior/cache pair,
    no real training run, since the guard fires before the sweep path is ever built."""
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
    """Pins the OTHER width guard -- the cache's own row width against
    ``posterior.manifest.body['conditioning']['width']`` -- again via a stub, no training run.

    The chi branch (``blk[:, -1]``, not ``blk[:, 0]``) is NOT pinned here: reaching it needs a full,
    successful sweep (``est.embedding_net`` actually built and called), which needs a REAL,
    internally-consistent chi net -- one whose ``chi_layout``/``chi_k_pad`` agree with the manifest,
    or ``store.load_posterior``'s own tier-2 mode detection (``reparam.posterior_mode``) raises a
    StoreError before ``channel_ablation`` is ever reached. Building that (a correctly-shaped chi
    ``EmbeddedNet`` wrapped so ``.embedding_net`` resolves, PLUS fabricated probe rows in the real chi
    column layout) is materially more scaffolding than the two width guards, which fail before any of
    it is touched -- so this is skipped rather than forced.
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
    """A channel whose sweep drove the network to NaN/Inf gets its OWN verdict --
    every numeric comparison against NaN is False, so before the fix it fell all the way through to
    "healthy" instead -- and the float fields that DERIVE from that NaN must be None in the manifest,
    because the writer refuses a non-finite float outright (``allow_nan=False``) and would otherwise
    fail well after the whole sweep had already run.

    ``_sweep_channels`` is monkeypatched WHOLESALE to a fixed record list (one channel poisoned to NaN,
    the rest finite fillers): the point is not to reproduce a real divergence, but to drive the REAL,
    unmonkeypatched code that runs AFTER it returns -- the verdict loop, the counts, the results dict
    and the manifest write -- with the least scaffolding. That still needs ``est.embedding_net`` to
    resolve (``channel_ablation`` builds it before calling ``_sweep_channels``), so this reuses the
    width tests' stub-posterior helper with ``wrap_embedding=True`` rather than a real training run.
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


def test_the_diagnostic_warnings_are_records_and_the_reports_are_information(caplog, capsys):
    """Standard logging in the diagnostics and the plot helpers. A diagnostic's report -- its tables,
    banners and per-point progress -- is information; what the operator must act on is a warning:
    feature channels whose Jacobian rows were ZEROED as noise, and a loss plot with nothing to draw.
    Every one of them used to be print(), so on the command line the dead-channel block sat in stdout
    between two tables, and in the window it wore no triangle.

    A warning printed as several lines is ONE record: the header and its per-channel rows travel
    together, so the pane's triangle and the tool's "warning: " prefix mark the block once and log.txt
    puts one timestamp over the rows it explains. Nothing reaches stdout any more -- on the command line
    the tool's information handler puts the report there, not a print."""
    import logging
    from types import SimpleNamespace

    import numpy as np

    from core.diagnostics import identifiability
    from core.Helpers import visualizers

    assert logging.getLogger("core").level == logging.INFO, "core/runs.py sets it at import"
    ctx = SimpleNamespace(feat_labels=["A1_mean", "B2_std"])
    feats0 = np.array([[1.0, 5.0], [1.0, 6.0]])
    keep0 = np.array([True, True])

    # a dead channel: one WARNING record, header first, the channel's row inside it
    caplog.clear()
    dead = identifiability._dead_channels(ctx, feats0, keep0, np.array([1e-12, 0.5]), 1e-6)
    assert dead.tolist() == [True, False]
    got = [(r.name, r.levelname, r.getMessage()) for r in caplog.records]
    assert len(got) == 1, got
    name, level, msg = got[0]
    assert (name, level) == ("core.diagnostics.identifiability", "WARNING"), got
    head, *rows = msg.split("\n")
    assert head.startswith("!! DEAD FEATURE CHANNELS: 1/2 rows ZEROED in J -- "), msg
    assert len(rows) == 1 and rows[0].startswith("     A1_mean ") and "ratio=1e-12" in rows[0], msg

    # no dead channel: one INFO record
    caplog.clear()
    dead = identifiability._dead_channels(ctx, feats0, keep0, np.array([0.5, 0.5]), 1e-6)
    assert not dead.any()
    got = [(r.levelname, r.getMessage()) for r in caplog.records]
    assert len(got) == 1 and got[0][0] == "INFO", got
    assert got[0][1].startswith("[noise] no dead channels (min std/|feat| = "), got

    # a loss plot with no validation curve: a WARNING from the plot helpers, and no figure
    caplog.clear()
    assert visualizers.plot_training_loss({"training_loss": [1.0, 0.9]}) is None
    got = [(r.name, r.levelname, r.getMessage()) for r in caplog.records]
    assert got == [("core.Helpers.visualizers", "WARNING",
                    "plot_training_loss: no validation_loss curve in diagnostics; nothing to plot.")], got

    out, err = capsys.readouterr()
    assert out == "" and err == "", (out, err)


_GEOMETRY_NAMES = ("T_nd_k", "dt_nd_k", "subsample_factor", "N_points_k", "n_fine_total")


def _geometry_bindings(source: str) -> list:
    """The five geometry statements of a function's ``source``, in line order.

    Every binding of the five names anywhere in the function is counted: assignment targets, including
    names nested in tuple and list targets, augmented and annotated assignments, walrus expressions,
    loop targets and ``with ... as`` targets. Each name must be bound exactly once, by a plain
    single-target assignment; any other binding means the statement run here is not the value the
    loop goes on to use, so the name is refused rather than trusted."""
    import ast
    import textwrap

    def bound(target):
        if isinstance(target, ast.Name):
            yield target.id
        elif isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                yield from bound(element)
        elif isinstance(target, ast.Starred):
            yield from bound(target.value)

    binders = {name: [] for name in _GEOMETRY_NAMES}
    for node in ast.walk(ast.parse(textwrap.dedent(source))):
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign, ast.NamedExpr, ast.For, ast.AsyncFor)):
            targets = [node.target]
        elif isinstance(node, ast.withitem) and node.optional_vars is not None:
            targets = [node.optional_vars]
        else:
            continue
        for target in targets:
            for name in bound(target):
                if name in binders:
                    binders[name].append(node)
    refused = [f"{name}: bound {len(nodes)} time(s) -- " + "; ".join(ast.unparse(n).splitlines()[0] for n in nodes)
               for name, nodes in binders.items()
               if not (len(nodes) == 1 and isinstance(nodes[0], ast.Assign) and len(nodes[0].targets) == 1
                       and isinstance(nodes[0].targets[0], ast.Name))]
    assert not refused, ("each geometry name must be bound exactly once, by a plain single-target assignment:\n"
                         + "\n".join(refused))
    return sorted((nodes[0] for nodes in binders.values()), key=lambda n: n.lineno)


def _batch_loop_geometry(cfg, t_obs_s, t_scale):
    """(N_points_k, n_fine_total, subsample_factor) as the training batch loop computes them: its own
    five geometry statements, lifted from gen_training_data's source and run on one (length, t_scale).

    Running production's statements, rather than a copy of them written here, is what makes the check
    below a pin: a change to the batch loop's arithmetic changes the expected values with it. Each
    name must be assigned exactly once, and the statements may read nothing but the loop's own inputs,
    so a reshaped loop fails here by name instead of passing against a stale restatement."""
    import ast
    import inspect
    from core.SBI import pipeline
    found = _geometry_bindings(inspect.getsource(pipeline.gen_training_data))
    scope = {"T_k": t_obs_s * cfg.get_unit_conversion_factor("s"), "t_scale_k": t_scale,
             "dt_exp": cfg.dt_exp, "dt_nd_min": cfg.dt_nd_min, "steady_idx": cfg.steady_idx}
    exec(compile(ast.Module(body=found, type_ignores=[]), "gen_training_data", "exec"), scope)
    return scope["N_points_k"], scope["n_fine_total"], scope["subsample_factor"]


_GEOMETRY_SOURCE = """
def gen(T_k, t_scale_k, dt_exp, dt_nd_min, steady_idx):
    T_nd_k = T_k / t_scale_k
    dt_nd_k = dt_exp / t_scale_k
    subsample_factor = max(1, int(dt_nd_k / dt_nd_min))
    N_points_k = int(T_nd_k / dt_nd_k)
    n_fine_total = steady_idx + N_points_k * subsample_factor
"""


@pytest.mark.parametrize("rebinding", ["N_points_k += 0", "N_points_k: int = 0", "print(N_points_k := 0)",
                                       "N_points_k, _ = 0, 0", "[N_points_k] = [0]",
                                       "for N_points_k in ():\n        pass"])
def test_the_geometry_pin_refuses_a_name_bound_twice_or_not_plainly(rebinding):
    """The geometry pin runs the batch loop's own statements, so it holds only while each of the five
    names is bound once, by a plain assignment. Any other binding -- augmented, annotated, a walrus, a
    tuple or list target, a loop target -- would leave the pin running a statement whose value the loop
    then changes, and passing against a stale one. The pin names the name it refuses."""
    assert [ast_node.targets[0].id for ast_node in _geometry_bindings(_GEOMETRY_SOURCE)] == [
        "T_nd_k", "dt_nd_k", "subsample_factor", "N_points_k", "n_fine_total"]
    with pytest.raises(AssertionError, match="N_points_k"):
        _geometry_bindings(_GEOMETRY_SOURCE.replace("    n_fine_total",
                                                    f"    {rebinding}\n    n_fine_total"))


def test_the_recording_geometry_agrees_with_the_training_batch_formula():
    """A probe length is sized with the training loop's own arithmetic, float truncation included, so
    it describes the trace a training batch of that length holds. The expected values are what the
    batch loop's own statements compute, not a restatement of them."""
    from core.diagnostics import probe_math
    cfg = _nad_cfg()
    for t_obs_s, t_scale in ((1.0, 3.73), (1.0, 3.7300000190734863), (5.0, 3.73),
                             (26.909, 3.7300000190734863), (2.5, 1.0), (7.3, 40.0)):
        want = _batch_loop_geometry(cfg, t_obs_s, t_scale)
        assert tuple(probe_math.recording_geometry(cfg, t_obs_s, t_scale)) == want, (t_obs_s, t_scale)
    assert probe_math.recording_geometry(cfg, 5.0, 3.73) == probe_math.Geometry(
        n_obs=5000, n_fine=59000, subsample=11)
    # the truncation is training's too, and is not corrected: one second at 3.73 holds 999 samples
    assert probe_math.recording_geometry(cfg, 1.0, 3.73).n_obs == 999


def test_the_training_ceiling_is_the_bound_the_sobol_prefilter_enforces():
    """The longest recording training can draw at a t_scale: the pre-filter admits it and refuses two
    samples more."""
    from core.diagnostics import probe_math
    from core.SBI import pipeline
    cfg = _nad_cfg()
    assert probe_math.training_ceiling_s(cfg, 3.73) == pytest.approx(26.909, abs=1e-9)
    top = probe_math.training_ceiling_s(cfg, 3.73) * cfg.get_unit_conversion_factor("s")
    args = (cfg.dt_exp, cfg.dt_nd_min, cfg.steady_idx)
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(0)
        pipeline._batch_schedule(4, cfg.t, (3.73, 3.73), top, top, *args)
        beyond = top + 2 * cfg.dt_exp
        with pytest.raises(ValueError, match="fine-grid ceiling"):
            pipeline._batch_schedule(4, cfg.t, (3.73, 3.73), beyond, beyond, *args)


def test_the_default_probe_grids_follow_the_configured_band_and_the_cells_ceiling():
    from core import config
    from core.diagnostics import probe_math
    cfg = _nad_cfg()
    assert config.CHI_FREQ_BOUNDS == (0.03, 0.3)
    assert probe_math.default_lengths(cfg, 3.73) == [1.0, 2.27, 5.18, 11.81, 26.9]
    assert probe_math.default_lengths(cfg, 3.73, n=2) == [1.0, 26.9]
    assert probe_math.default_multipliers() == [0.03, 0.0646, 0.1392, 0.3, 0.6]


def test_the_own_peak_ratio_counts_only_the_power_inside_the_window():
    import math
    from core.diagnostics import probe_math
    t = torch.arange(5000, dtype=torch.float64)                 # dt 1, bins of 1/5000
    rows = torch.linspace(0.0, 2.0, 4, dtype=torch.float64).unsqueeze(1)
    ref = torch.sin(2 * math.pi * 0.02 * t + rows)               # bin 100
    assert probe_math.own_peak_ratio(0.5 * ref, ref, 0.02, 0.10, 1.0) == pytest.approx(0.25, rel=1e-9)
    near = torch.sin(2 * math.pi * 0.0216 * t + rows)            # bin 108
    assert probe_math.own_peak_ratio(near, ref, 0.02, 0.10, 1.0) == pytest.approx(1.0, rel=1e-6)
    assert probe_math.own_peak_ratio(near, ref, 0.02, 0.05, 1.0) < 1e-12
    two_off = torch.sin(2 * math.pi * 0.0204 * t + rows)         # bin 102: inside the two-bin floor
    assert probe_math.own_peak_ratio(two_off, ref, 0.02, 1e-6, 1.0) == pytest.approx(1.0, rel=1e-6)
    assert math.isnan(probe_math.own_peak_ratio(ref, ref, float("nan"), 0.10, 1.0))
    with pytest.raises(ValueError):
        probe_math.own_peak_ratio(ref[:, :4000], ref, 0.02, 0.10, 1.0)


def test_the_own_peak_window_keeps_its_floor_and_stays_inside_the_spectrum():
    """At its two edges the window is cut, never shifted. A peak on bin 1 takes the two-bin floor but
    never the zero-frequency bin, whose power is the mean every trace has removed; a peak on the last
    bin stops at the last bin rather than running past the spectrum."""
    from core.diagnostics import probe_math
    assert probe_math.own_peak_window(100, 0.01, 0.1, 1.0) == (1, 4)
    assert probe_math.own_peak_window(100, 0.5, 0.1, 1.0) == (45, 51)


def test_the_circular_spread_is_zero_for_one_phase_and_ignores_the_wrap():
    import math
    from core.diagnostics import probe_math
    one = torch.polar(torch.ones(5, dtype=torch.float64), torch.full((5,), 1.3, dtype=torch.float64))
    assert probe_math.circular_spread(one) == pytest.approx(0.0, abs=1e-7)
    across = torch.polar(torch.ones(2, dtype=torch.float64),
                         torch.tensor([math.pi - 0.05, -math.pi + 0.05], dtype=torch.float64))
    assert probe_math.circular_spread(across) == pytest.approx(math.sqrt(-2 * math.log(math.cos(0.05))), rel=1e-9)
    spread = torch.polar(torch.ones(4, dtype=torch.float64),
                         torch.tensor([0.0, math.pi / 2, math.pi, 3 * math.pi / 2], dtype=torch.float64))
    assert probe_math.circular_spread(spread) > 5.0


def test_the_harmonic_flag_marks_a_probe_whose_low_harmonic_lands_in_the_own_peak_window():
    from core.diagnostics import probe_math
    assert probe_math.harmonic_flags([0.03, 0.0646, 0.1392, 0.3, 0.6], 0.10) == [False, False, False, True, False]
    assert probe_math.harmonic_flags([0.189, 0.25, 0.5, 1.0, 1.4], 0.10) == [True, True, True, True, False]
    assert probe_math.harmonic_flags([0.189], 0.10, max_order=4) == [False]
    assert probe_math.harmonic_flags([1.4], 0.02) == [False]


def _probe_cfg(bounds="master.txt", cell="master_spont.txt", *, chi=True):
    """A CPU Nadrowski config with a cell loaded, built the way the probe checks build one."""
    from core import cli, config, registry
    from core.config import VALID_LABELS, VALID_MODELS
    cfg = cli.make_sim_config("NADROWSKI", VALID_LABELS[VALID_MODELS.index("NADROWSKI")],
                              registry.state_dep_drift("NADROWSKI"),
                              str(config.BOUNDS_PATH / "nadrowski" / bounds), chi_mode=chi)
    cfg.hw = config.cpu_device()
    cli.load_and_validate_gt(cfg, str(config.CELL_PATH / "nadrowski" / cell))
    return cfg


def _probe_stand_in(*, f_own=0.025, own_factor=None, gain=2.0, noise=0.01, seen=None, nan_driven=False,
                    phase=None, background=None):
    """In place of probes._simulate, a cell of known response. Its own oscillation sits at f_own (cell
    frequency units, one value or one per row) with a random phase per row. A drive adds a response of
    |chi| = gain in phase with it, and scales the own oscillation by own_factor(strength in model units).
    ``gain`` may also be one value per row, ``phase`` (radians, one per row) delays each row's response
    so that arg chi = phase, and ``background`` = (frequency, amplitude) adds a fixed cosine to every
    run, driven or not -- what the undriven lock-in then reads at that frequency."""
    import math
    own_factor = own_factor or (lambda a: torch.where(a >= 1.0, 0.1, 1.0))

    def _sim(cfg, geom, nd, res_sim, sim_idx, inits, *, amp_dim=None, freq=None):
        B, n = nd.shape[0], geom.n_obs
        t = torch.arange(n, dtype=torch.float64) * cfg.dt_exp
        f = torch.as_tensor(f_own, dtype=torch.float64).reshape(-1, 1)
        own = torch.sin(2 * math.pi * f * t + 2 * math.pi * torch.rand(B, 1, dtype=torch.float64))
        x = own + noise * torch.randn(B, n, dtype=torch.float64)
        if background is not None:
            x = x + background[1] * torch.cos(2 * math.pi * background[0] * t)
        if amp_dim is not None:
            amp = amp_dim.double().reshape(-1, 1)
            strength = amp / res_sim[:, sim_idx["f_scale"]].double().reshape(-1, 1)
            if seen is not None:
                seen.append((strength.flatten().tolist(), freq.double().flatten().tolist()))
            g = torch.as_tensor(gain, dtype=torch.float64).reshape(-1, 1)
            lag = torch.as_tensor(0.0 if phase is None else phase, dtype=torch.float64).reshape(-1, 1)
            x = x + (own_factor(strength) - 1.0) * own + g * amp * torch.cos(2 * math.pi * freq.double().reshape(-1, 1) * t - lag)
            x = torch.full_like(x, float("nan")) if nan_driven else x
        return x.to(cfg.hw.dtype)
    return _sim


def _diagnostic_dirs(store):
    d = store.kind_dir("diagnostic")
    return sorted(p.name for p in d.iterdir()) if d.is_dir() else []


def test_probes_band_judges_the_configured_band_and_drive_on_a_cell_of_known_response(store, monkeypatch, caplog):
    import math
    import numpy as np
    from core.diagnostics import probes, probe_band
    monkeypatch.setattr(probes, "_simulate", _probe_stand_in())
    d = probe_band(_probe_cfg(), lengths=[1.0, 2.0], multipliers=[0.05, 0.3, 0.6], repeats=4, name="band1")
    s, r = d.settings, d.results
    assert (d.diagnostic, d.variant, d.manifest.parents) == ("probes", "band", {})
    assert s["drives"] == [0.15] and s["cycle_caps"] == [20.0] and s["phase_max"] is None
    assert s["drive_phase"] == pytest.approx(math.pi / 2)
    assert s["configured"] == {"chi_f0": 0.15, "chi_freq_bounds": [0.03, 0.3], "chi_k_pad": 12,
                               "chi_min_cycles": 2.0, "chi_max_cycles": 20.0, "probe_count": 6}
    assert r["band_holds"] is True and r["drive_holds"] is True
    assert [(f["multiplier"], f["in_band"], f["harmonic"], f["passes_capped"]) for f in r["frequencies"]] == [
        (0.05, True, False, True), (0.3, True, True, True), (0.6, False, False, True)]
    assert len(r["points"]) == 6 and all(p["n_valid"] == 4 and p["nyquist_masked"] == 0 for p in r["points"])
    slow = r["points"][0]                      # 0.05 x 25 Hz over 1 s is 1.25 drive cycles
    assert (slow["length_s"], slow["multiplier"], slow["drive"]) == (1.0, 0.05, 0.15)
    assert slow["full"]["floor_masked"] == 1.0 and slow["full"]["verdict"] == "pass"   # reported, not judged
    assert r["lengths"][0]["omega0_hz"] == pytest.approx(25.0) and r["training_ceiling_s"] == pytest.approx(26.909)
    assert len(r["caveats"]) == 2
    z = np.load(d.path / "probe_band.npz")
    assert z["cv_capped"].shape == z["snr_full"].shape == (2, 3, 1) and d.manifest.figures
    said = [m.getMessage() for m in caplog.records if m.name == "core.diagnostics.probes" and m.levelname == "INFO"]
    assert any(m.startswith("[band] the configured band (0.03, 0.3) holds for this cell") for m in said), said
    assert any(m.startswith("[band] the configured drive 0.15 holds for this cell") for m in said), said


def test_probes_band_fails_a_frequency_captured_at_any_drive_and_judges_the_configured_drive_alone(store, monkeypatch, caplog):
    from core.diagnostics import probes, probe_band
    kw = dict(lengths=[1.0], multipliers=[0.12, 0.2], repeats=4)        # drives on exact bins: no leakage
    monkeypatch.setattr(probes, "_simulate", _probe_stand_in())        # captures from strength 1.0
    r = probe_band(_probe_cfg(), drives=[0.15, 2.0], name="band2", **kw).results
    assert r["band_holds"] is False and r["drive_holds"] is True
    assert all(not f["passes_capped"] and any("captured" in why for why in f["reasons"]) for f in r["frequencies"])
    assert [(x["drive"], x["configured"], len(x["captured_in_band"])) for x in r["drives"]] == [(0.15, True, 0), (2.0, False, 2)]
    monkeypatch.setattr(probes, "_simulate", _probe_stand_in(own_factor=lambda a: torch.where(a >= 0.1, 0.1, 1.0)))
    r = probe_band(_probe_cfg(), name="band3", **kw).results
    assert r["band_holds"] is False and r["drive_holds"] is False
    said = [m.getMessage() for m in caplog.records if m.name == "core.diagnostics.probes"]
    assert any(m.startswith("[band] the configured drive 0.15 does not hold for this cell") for m in said), said
    assert any("core/config.py" in m for m in said)


def test_probes_band_flags_a_harmonic_against_the_own_peak_window_it_measured(store, monkeypatch):
    """The capture share sums the own-peak window as measured at each length, never narrower than two
    frequency bins either side, so the harmonic flag asks the same question of that window, the way
    the drive check does. A 5 Hz peak over 1 s sits on bin 5: the window is bins 3 to 7, 0.6 to 1.4
    times the peak, and the x0.6 probe's own tone on bin 3 lands inside it, though it is far outside
    the nominal +/-10 %. Over 10 s the peak sits on bin 50 and the window is 10 % wide, clear of it."""
    import argparse

    from core.diagnostics import probes, probe_band
    from core.tool import build_parser
    monkeypatch.setattr(probes, "_simulate", _probe_stand_in(f_own=0.005))
    r = probe_band(_probe_cfg(), lengths=[1.0, 10.0], multipliers=[0.6], repeats=4, name="slow_peak").results
    by_length = {p["length_s"]: p for p in r["points"]}
    assert by_length[1.0]["harmonic"] is True, by_length[1.0]
    assert by_length[10.0]["harmonic"] is False, by_length[10.0]
    assert r["frequencies"][0]["harmonic"] is True
    assert r["lengths"][0]["own_peak_window_x"] == pytest.approx([0.6, 1.4])
    assert r["lengths"][1]["own_peak_window_x"] == pytest.approx([0.9, 1.1])
    p = build_parser().subcommands["probes"]
    band = next(sub for a in p._actions if isinstance(a, argparse._SubParsersAction)
                for name, sub in a.choices.items() if name == "band")
    assert "never narrower than two frequency bins" in band._option_string_actions["--peak-window"].help


def test_probes_band_reports_a_probe_past_the_sampling_limit_as_masked_and_never_clamps_it(store, monkeypatch,
                                                                                         caplog):
    from core.diagnostics import probes, probe_band
    seen = []
    monkeypatch.setattr(probes, "_simulate", _probe_stand_in(f_own=[0.2, 0.2, 0.4, 0.4], seen=seen))
    d = probe_band(_probe_cfg(), lengths=[1.0], multipliers=[0.05, 1.5, 3.0], repeats=4, name="fast")
    pts = {p["multiplier"]: p for p in d.results["points"]}
    assert (pts[0.05]["n_valid"], pts[0.05]["nyquist_masked"]) == (4, 0)
    assert (pts[1.5]["n_valid"], pts[1.5]["nyquist_masked"]) == (2, 2) and pts[1.5]["full"]["cv"] is not None
    assert (pts[3.0]["n_valid"], pts[3.0]["nyquist_masked"]) == (0, 4)
    assert pts[3.0]["full"]["verdict"] == pts[3.0]["capped"]["verdict"] == "masked"
    assert pts[3.0]["full"]["cv"] is None and pts[3.0]["sup"] is None
    fast = next(f for f in d.results["frequencies"] if f["multiplier"] == 3.0)
    assert not fast["passes_capped"] and any("masked" in why for why in fast["reasons"])
    assert max(f for _, freqs in seen for f in freqs) == pytest.approx(0.6, rel=1e-6)   # driven where asked
    assert store.load_diagnostic(d.id).results == d.results
    # a point measured over fewer runs than were simulated says so in the report a user reads
    said = [m.getMessage() for m in caplog.records if m.name == "core.diagnostics.probes"]
    assert any("x1.5" in m and m.endswith("2 of 4 runs masked at the sampling limit") for m in said), said
    assert not any("x0.05" in m and "masked" in m for m in said), said


def test_probes_band_masks_at_nine_tenths_of_the_sampling_limit_and_needs_two_runs(store, monkeypatch):
    """The two edges of training's masking rule, where the obvious wrong rules part from it: a probe
    between 0.9 x Nyquist and Nyquist itself is masked, and one run left below the limit is too few to
    measure a point. At 1000 samples of 1 ms the peaks 0.1 and 0.2 cycles/ms sit on exact bins, so x2.3
    puts one run at 0.23 (kept) and three at 0.46 -- above 0.9 x Nyquist (0.45), below Nyquist (0.5)."""
    from core.diagnostics import probes, probe_band
    monkeypatch.setattr(probes, "_simulate", _probe_stand_in(f_own=[0.1, 0.2, 0.2, 0.2]))
    d = probe_band(_probe_cfg(), lengths=[1.0], multipliers=[2.3], repeats=4, name="edge")
    point, = d.results["points"]
    assert (point["n_valid"], point["nyquist_masked"]) == (1, 3)
    assert point["full"]["verdict"] == point["capped"]["verdict"] == "masked"
    freq, = d.results["frequencies"]
    assert freq["reasons"] == ["masked at the sampling limit at 1.00 s"] and not freq["passes_capped"]


def test_probes_band_judges_the_phase_only_against_a_given_threshold(store, monkeypatch):
    """Two runs whose response lags by 0 and pi/2 scatter in phase by sqrt(ln 2) = 0.8326 rad (the
    circular standard deviation of two orthogonal unit vectors). Without a threshold the phase is
    reported and not judged; a threshold below the scatter fails the point on phase, one above passes."""
    import math
    from core.diagnostics import probes, probe_band
    monkeypatch.setattr(probes, "_simulate", _probe_stand_in(phase=[0.0, math.pi / 2]))
    kw = dict(lengths=[1.0], multipliers=[0.12], repeats=2)            # 3 drive cycles on an exact bin
    free = probe_band(_probe_cfg(), name="phase_free", **kw).results["points"][0]
    assert free["full"]["phase"] == pytest.approx(math.sqrt(math.log(2)), abs=2e-3)
    assert free["full"]["verdict"] == free["capped"]["verdict"] == "pass"
    tight = probe_band(_probe_cfg(), phase_max=0.5, name="phase_tight", **kw).results["points"][0]
    assert tight["full"]["verdict"] == tight["capped"]["verdict"] == "phase"
    loose = probe_band(_probe_cfg(), phase_max=1.0, name="phase_loose", **kw)
    assert loose.results["points"][0]["full"]["verdict"] == "pass" and loose.settings["phase_max"] == 1.0


def test_probes_band_measures_the_amplitude_spread_and_the_signal_over_the_floor(store, monkeypatch):
    """The two lock-in measures by value. Runs whose |chi| is 1, 3, 1 and 3 spread by the unbiased std
    over the mean, sqrt(4/3) / 2 = 0.5774: noisy. A fixed background cosine of amplitude 0.5 at the probe
    frequency is what the undriven lock-in reads, so a response of 3 on top of it gives a signal over the
    floor of (3 + 0.5) / 0.5 = 7, which passes a threshold of 3 and fails one of 10."""
    import math
    from core.diagnostics import probes, probe_band
    kw = dict(lengths=[1.0], multipliers=[0.12], repeats=4)            # 3 drive cycles on an exact bin
    monkeypatch.setattr(probes, "_simulate", _probe_stand_in(gain=[1.0, 3.0, 1.0, 3.0]))
    p = probe_band(_probe_cfg(), name="spread", **kw).results["points"][0]
    assert p["full"]["cv"] == pytest.approx(math.sqrt(4 / 3) / 2, abs=2e-3)
    assert p["full"]["chi_mag"] == pytest.approx(2.0, rel=1e-3) and p["full"]["verdict"] == "noisy"
    monkeypatch.setattr(probes, "_simulate", _probe_stand_in(background=(0.003, 0.5)))
    p = probe_band(_probe_cfg(), name="floor", **kw).results["points"][0]
    assert p["full"]["snr"] == pytest.approx(7.0, rel=1e-2) and p["full"]["verdict"] == "pass"
    p = probe_band(_probe_cfg(), snr_min=10.0, name="floor_high", **kw).results["points"][0]
    assert p["full"]["verdict"] == p["capped"]["verdict"] == "low signal"


def test_probes_band_repeats_at_one_seed_and_differs_at_another(store, monkeypatch):
    from core.diagnostics import probes, probe_band
    monkeypatch.setattr(probes, "_simulate", _probe_stand_in())
    kw = dict(lengths=[1.0], multipliers=[0.3], repeats=3)
    a = probe_band(_probe_cfg(), seed=5, name="seed_a", **kw)
    b = probe_band(_probe_cfg(), seed=5, name="seed_b", **kw)
    c = probe_band(_probe_cfg(), seed=6, name="seed_c", **kw)
    assert a.results == b.results and a.settings == b.settings
    assert c.results["points"] != a.results["points"] and c.settings["seed"] == 6


def test_probes_band_brackets_the_wall_only_when_caps_are_given(store, monkeypatch):
    from core.diagnostics import probes, probe_band
    monkeypatch.setattr(probes, "_simulate", _probe_stand_in())
    kw = dict(lengths=[2.0], multipliers=[0.3], repeats=2)             # 0.3 x 25 Hz over 2 s is 15 cycles
    assert probe_band(_probe_cfg(), name="nowall", **kw).results["wall"] is None
    d = probe_band(_probe_cfg(), cycle_caps=[8, 12], name="wall", **kw)
    wall = d.results["wall"]
    assert d.settings["cycle_caps"] == [8.0, 12.0, 20.0]
    assert [c["cap"] for c in wall["caps"]] == [8.0, 12.0, 20.0]
    assert [c["n_truncated"] for c in wall["caps"]] == [1, 1, 0]
    assert all(c["n_fail"] == 0 for c in wall["caps"] if c["n_truncated"]), wall["caps"]
    assert "above 12" in wall["reading"], wall["reading"]
    assert d.results["points"][0]["full"]["cycles"] == pytest.approx(15.0)
    # the per-run ceiling rounds DOWN, as training's does: floor(8 / 0.0075) = 1066 samples of 1 ms
    cap8 = d.results["points"][0]["caps"][0]
    assert cap8["cap"] == 8.0 and cap8["cycles"] == pytest.approx(7.995) and cap8["cycles"] < 8.0


def test_the_wall_reading_brackets_the_first_failing_cap_from_below():
    """The four readings of the cap grid: nothing binds; nothing fails; the smallest binding cap fails;
    and a clean cap below the first failing one, which brackets the wall."""
    from core.diagnostics import probes

    def caps(*rows):
        return [{"cap": c, "n_truncated": t, "n_fail": f} for c, t, f in rows]

    assert probes._wall(caps((8.0, 0, 0), (20.0, 0, 0)))["reading"] == "no cap binds any point"
    got = probes._wall(caps((8.0, 2, 0), (12.0, 1, 0), (20.0, 0, 0)))
    assert got["reading"] == "no cap in the grid fails: the wall lies above 12 cycles"
    assert got["first_failing"] is None and got["largest_clean_below"] is None
    got = probes._wall(caps((8.0, 2, 1), (12.0, 1, 1), (20.0, 0, 0)))
    assert got["reading"] == ("the smallest cap that binds (8 cycles) already fails: the grid does not "
                              "bracket the wall from below")
    assert got["first_failing"] == 8.0 and got["largest_clean_below"] is None
    got = probes._wall(caps((8.0, 3, 0), (12.0, 2, 0), (28.0, 2, 1), (36.0, 1, 1)))
    assert got["reading"] == "the wall lies between 12 and 28 cycles"
    assert (got["first_failing"], got["largest_clean_below"]) == (28.0, 12.0)


def test_probes_band_turns_non_finite_measures_into_null(store, monkeypatch):
    """A lock-in that returns NaN is not measured: a NaN compares false against every threshold and
    would otherwise read as a pass, and a manifest refuses it."""
    from core.diagnostics import probes, probe_band
    monkeypatch.setattr(probes, "_simulate", _probe_stand_in(nan_driven=True))
    d = probe_band(_probe_cfg(), lengths=[1.0], multipliers=[0.12], repeats=2, name="nan")
    p = d.results["points"][0]
    assert p["full"]["cv"] is None and p["full"]["snr"] is None and p["sup"] is None
    assert p["full"]["verdict"] == p["capped"]["verdict"] == "not measured"
    assert d.results["band_holds"] is False and d.results["drive_holds"] is None
    assert store.load_diagnostic(d.id).results == d.results


_BAND_BASE = dict(lengths=[1.0], multipliers=[0.1], repeats=2)


@pytest.mark.parametrize("kw, field", [
    ({"repeats": 1}, "band_repeats"),
    ({"lengths": [0.0]}, "probe_lengths"),
    ({"lengths": [1e-7]}, "probe_lengths"),
    ({"lengths": []}, "probe_lengths"),
    ({"multipliers": [0.0]}, "probe_multipliers"),
    ({"multipliers": [float("nan")]}, "probe_multipliers"),
    ({"drives": [-1.0]}, "probe_drives"),
    ({"cycle_caps": [0.0]}, "probe_cycle_caps"),
    ({"cv_max": float("nan")}, "cv_max"),
    ({"snr_min": float("inf")}, "snr_min"),
    ({"sup_min": 0.0}, "sup_min"),
    ({"sup_min": 1.5}, "sup_min"),
    ({"phase_max": float("nan")}, "phase_max"),
    ({"peak_window": 0.0}, "band_peak_window"),
    ({"peak_window": 1.0}, "band_peak_window"),
    ({"seed": -1}, "probe_seed"),
    ({"name": "taken"}, "name"),
])
def test_probes_band_refuses_before_the_writer_opens(store, monkeypatch, kw, field):
    from core.diagnostics import probes, probe_band
    cfg = _probe_cfg()
    _diagnostic(store, cfg, name="taken")
    monkeypatch.setattr(probes, "_simulate", lambda *a, **k: pytest.fail("simulated before the refusal"))
    before = _diagnostic_dirs(store)
    with pytest.raises(Refusal) as e:
        probe_band(cfg, **{**_BAND_BASE, **kw})
    assert e.value.field == field, (kw, e.value.field, str(e.value))
    assert _diagnostic_dirs(store) == before


def test_probes_band_refuses_a_config_without_chi_mode(store, monkeypatch):
    from core.diagnostics import probes, probe_band
    monkeypatch.setattr(probes, "_simulate", lambda *a, **k: pytest.fail("simulated before the refusal"))
    with pytest.raises(Refusal) as e:
        probe_band(_probe_cfg(chi=False), **_BAND_BASE)
    assert e.value.field is None and "chi" in str(e.value), str(e.value)
    assert _diagnostic_dirs(store) == []


def test_a_probes_run_leaves_a_training_configuration_and_config_py_untouched(store, monkeypatch):
    """What the check measures with reaches the simulator and nothing else: a training configuration
    built in the same process is unchanged, so are config.py's chi constants, a fresh one still opens at
    config.py's band and drive, and every drive the stand-in saw is the one the grid asked for."""
    from core import config
    from core.diagnostics import probes, probe_band
    from tests._fixtures import assert_cfg_unchanged, snapshot_cfg
    training = _nad_cfg(chi_mode=True)
    snap = snapshot_cfg(training)
    constants = {k: getattr(config, k) for k in dir(config) if k.startswith("CHI_")}
    seen = []
    monkeypatch.setattr(probes, "_simulate", _probe_stand_in(seen=seen))
    d = probe_band(_probe_cfg(), lengths=[1.0], multipliers=[0.05, 0.6], drives=[0.5], repeats=2,
                   name="untouched")
    assert d.results["drive_holds"] is None            # the configured drive was not among those measured
    assert_cfg_unchanged(training, snap)
    assert {k: getattr(config, k) for k in dir(config) if k.startswith("CHI_")} == constants
    fresh = _nad_cfg(chi_mode=True)
    assert (fresh.chi_f0, fresh.chi_freq_bounds) == (config.CHI_F0, config.CHI_FREQ_BOUNDS)
    assert seen and all(v == pytest.approx(0.5) for strengths, _ in seen for v in strengths), seen


def test_the_probe_simulator_drives_a_tier1_cell_at_its_derived_force_scale(monkeypatch):
    """A real two-row driven run of the tier-1 twin: the drive is built against the simulator's index,
    whose force scale is the one derived from the temperature, never the box's temperature column."""
    from core.diagnostics import probe_math, probes
    from core.SBI import derived, pipeline
    cfg = _probe_cfg("master_tier1.txt", "master_spont_tier1.txt")
    seen = []
    real = pipeline.build_nondim_sin_force_tensor

    def _spy(fp, t, rescale, fidx, ridx):
        seen.append((rescale.detach().clone(), dict(ridx)))
        return real(fp, t, rescale, fidx, ridx)

    monkeypatch.setattr(pipeline, "build_nondim_sin_force_tensor", _spy)
    res = torch.tensor([[v for v, _ in cfg.rescale_params.values()]], dtype=cfg.hw.dtype)
    res_sim, sim_idx = derived.for_simulation(cfg, cfg.params_tensor, res)
    geom = probe_math.recording_geometry(cfg, 0.2, float(res_sim[0, sim_idx["t_scale"]]))
    nd, rs, inits = (v.expand(2, -1).contiguous() for v in (cfg.params_tensor, res_sim, cfg.inits_tensor))
    amp = (cfg.chi_f0 * rs[:, sim_idx["f_scale"]]).contiguous()
    freq = torch.full((2,), 0.005, dtype=torch.float64)
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(0)
        x = probes._simulate(cfg, geom, nd, rs, sim_idx, inits, amp_dim=amp, freq=freq)
    assert x.shape == (2, geom.n_obs) and bool(torch.isfinite(x).all())
    assert len(seen) == 1
    rescale, idx = seen[0]
    assert "f_scale" in idx and "T" not in idx, idx
    assert float(rescale[0, idx["f_scale"]]) == pytest.approx(50 * 14.1 * cfg.k_b_cell * 300 / 62.14, rel=1e-5)


@pytest.mark.slow
def test_probes_band_gives_the_same_criteria_on_the_master_cell_and_its_tier1_twin(store):
    """The master cell and its tier-1 twin are one cell, declared two ways: the same seed gives the same
    reproducibility, phase, signal and capture, and |chi| scales by the ratio of their force scales."""
    from core.diagnostics import probe_band
    kw = dict(lengths=[1.0], multipliers=[0.1392], repeats=8, seed=0)
    a = probe_band(_probe_cfg(), name="twin_master", **kw).results["points"][0]
    b = probe_band(_probe_cfg("master_tier1.txt", "master_spont_tier1.txt"), name="twin_tier1",
                   **kw).results["points"][0]
    for col in ("full", "capped"):
        for m in ("cv", "phase", "snr"):
            assert a[col][m] == pytest.approx(b[col][m], rel=1e-3, abs=1e-6), (col, m, a[col], b[col])
    assert a["sup"] == pytest.approx(b["sup"], rel=1e-3)
    assert a["full"]["chi_mag"] / b["full"]["chi_mag"] == pytest.approx(
        50 * 14.1 * 1.380649e-2 * 300 / 62.14 / 10.0, rel=1e-3)


#: A 5000-unit recording sampled every half unit: a sampling interval of 1 would hide a length or a
#: Nyquist limit computed the wrong way round.
_MASK_DT, _MASK_POINTS = 0.5, 10_000
#: The duration fractions of the three probes, shared by every row as training shares them.
_MASK_FRAC = (1.0, 0.2, 1.0)
#: Row kinds: (the row's own peak, its three multipliers after placement, the probes the packer keeps).
#:   full    every probe clears the floor.
#:   short   between the band's ends -- its bottom would miss the floor at full length, so placement lifted
#:           its probes -- and it loses the probe the duration draw shortened.
#:   slow    too slow even at the band's top, its probes parked there: all three miss the floor.
#:   packer  as full, but the packer drops one lock-in.
#:   fast    one probe at 0.9 x Nyquist and one non-finite, neither possible in training.
#:   wide    one valid probe outside the band, which the packer drops -- not possible in training either.
_MASK_ROWS = {"full": (0.02, (0.1, 0.2, 0.3), 3), "short": (0.004, (0.15, 0.2, 0.3), 2),
              "slow": (0.0008, (0.3, 0.3, 0.3), 0), "packer": (0.02, (0.1, 0.2, 0.3), 2),
              "fast": (0.8, (0.3, 1.2, float("nan")), 1), "wide": (0.02, (0.1, 0.2, 0.6), 2)}


def _mask_record(tag, lo, hi, kinds=("full", "short", "slow", "packer"), *, k_pad=12):
    """Rows lo..hi of a batch whose rows are ``kinds``, built as the training generator builds them: each
    probe driven at its row's peak times its multiplier, locked in over its duration fraction of the
    recording under the 20-cycle ceiling, and masked when not finite, at 0.9 x Nyquist or below 2 cycles.
    The default batch masks 5 of its 12 probes: 3 too slow, 1 shortened, 1 at the packer."""
    from core.SBI.chi_probes import ProbeRecord
    rows = [_MASK_ROWS[k] for k in kinds][lo:hi]
    f_peak = torch.tensor([r[0] for r in rows], dtype=torch.float64)
    mult = torch.tensor([r[1] for r in rows], dtype=torch.float64)
    freq = f_peak.unsqueeze(1) * mult
    n_k = torch.tensor([max(1, round(f * _MASK_POINTS)) for f in _MASK_FRAC], dtype=torch.float64)
    usable = torch.isfinite(freq) & (freq > 0) & (freq < 0.9 * 0.5 / _MASK_DT)
    ceiling = torch.floor(20.0 / freq.clamp(min=1e-30) / _MASK_DT).clamp(min=1.0)
    cycles = freq * torch.where(usable, torch.minimum(ceiling, n_k), n_k) * _MASK_DT
    live = torch.tensor([r[2] for r in rows])
    return ProbeRecord(batch_tag=tag, lo=lo, hi=hi, f_peak=f_peak,
                       duration_frac=torch.tensor(_MASK_FRAC, dtype=torch.float64), k=3, u=torch.log(mult),
                       logcyc=torch.log(cycles.clamp(min=1e-30)), valid=usable & (cycles >= 2.0),
                       packed_mask=torch.arange(k_pad).unsqueeze(0) < live.unsqueeze(1),
                       dt_exp=_MASK_DT, n_points=_MASK_POINTS)


def test_a_probe_record_compares_by_identity():
    """A record's fields are mostly tensors, and a field-by-field comparison of tensors has no single
    truth value: the generated ``==`` raised on any two records. A record is equal to itself and to no
    copy of it, and the comparison always gives a plain bool."""
    import dataclasses
    r = _mask_record("b0", 0, 4)
    assert r == r and (r == dataclasses.replace(r)) is False


def test_the_probe_checks_default_settings_are_the_agreed_numbers():
    """The refusal table and the tool's help read each probe check's defaults off its signature, so a
    changed default moves all three together and nothing fails. The numbers themselves are pinned here,
    so that changing what a check judges by default is a deliberate edit of this test too."""
    import inspect
    from core.diagnostics import probes

    def defaults(fn, *names):
        sig = inspect.signature(fn).parameters
        return tuple(sig[n].default for n in names)

    assert defaults(probes.probe_band, "repeats", "cv_max", "snr_min", "sup_min", "peak_window", "phase_max",
                    "seed") == (24, 0.20, 3.0, 0.50, 0.10, None, 0)
    assert defaults(probes.probe_mask, "num_runs", "run_size", "chi_k_fixed", "seed") == (12, 32, None, 0)
    assert defaults(probes.probe_drive, "t_obs_s", "repeats", "detune", "free_min", "captured_max", "peak_window",
                    "clarity_min", "seed") == (5.0, 16, 1.4, 0.70, 0.10, 0.02, 3.0, 0)


def _mask_generator(script, calls):
    """In place of pipeline.gen_training_data: per batch, the warnings an abandoned attempt left, then
    each committed range's own masked-probe warning, then the records, handed over at commit. Each call's
    arguments are recorded bound to the real generator's signature, read before the patch."""
    import inspect
    import warnings
    from core.SBI import pipeline
    signature = inspect.signature(pipeline.gen_training_data)

    def _gen(*a, probe_observer=None, **kw):
        calls.append(signature.bind(*a, probe_observer=probe_observer, **kw).arguments)
        for tag, stale, records in script:
            for text in stale:
                warnings.warn(text)
            for rec in records:
                n = int((~rec.packed_mask[:, :rec.k]).sum())
                if n:
                    warnings.warn(f"{tag}: chi: {n}/{(rec.hi - rec.lo) * rec.k} probes masked (below 2.0 drive cycles).")
            for rec in records:
                probe_observer(rec)
        return torch.zeros(1, 1), torch.zeros(1, 1)
    return _gen


def _mask_prior(store, cfg):
    from tests._fixtures import _prior_artifact
    _prior_artifact(store, cfg, name="mp")
    return store.load_prior(cfg, "mp")


def test_probes_mask_splits_the_thrown_out_probes_by_cause_over_the_committed_row_ranges(store, monkeypatch):
    import numpy as np
    from core import config
    from core.diagnostics import probe_mask
    from core.SBI import pipeline
    from tests._fixtures import only_masked_probe_warnings
    cfg = _nad_cfg(chi_mode=True)
    prior = _mask_prior(store, cfg)
    t1, t2 = (f"training batch {k}/2 [t_scale=3.73, T=5000, n_fine=59000, N_points=5000, rows=4]" for k in (1, 2))
    script = [(t1, [], [_mask_record(t1, 0, 4)]),
              (t2, [f"{t2}: chi: 5/12 probes masked (an attempt the row halving abandoned)"],
               [_mask_record(t2, 0, 2), _mask_record(t2, 2, 4)])]
    calls = []
    monkeypatch.setattr(pipeline, "gen_training_data", _mask_generator(script, calls))
    with only_masked_probe_warnings() as caught:
        d = probe_mask(cfg, prior, num_runs=2, run_size=4, name="mask1")
    assert len(caught) == 4, "every warning still reaches the hook that was in force"
    r, kw = d.results, calls[0]
    assert kw["prior"] is prior.prior and kw["forcing_prior"] is prior.force_prior
    assert kw["checkpoint"] is None and kw["chi_mode"] is True and kw.get("theta_transform") is None
    assert (kw["chi_f0"], kw["chi_freq_bounds"]) == (config.CHI_F0, config.CHI_FREQ_BOUNDS)
    assert (kw["n_runs"], kw["run_size"], kw["chi_k_fixed"]) == (2, 4, None)
    assert store.list("simulation") == []
    assert (d.variant, d.manifest.parents, d.manifest.fingerprints["gmm"]) == ("mask", {"prior": prior.id}, prior.fingerprint)
    assert (r["probes"], r["live"], r["masked"]) == (24, 14, 10)
    c = r["causes"]
    assert (c["cycle_floor"]["count"], c["too_slow_at_band_top"]["count"],
            c["shortened_by_duration_draw"]["count"], c["non_finite_lock_in"]["count"]) == (8, 6, 2, 2)
    assert r["invariants"] == {"non_finite_frequency": 0, "at_or_above_nyquist": 0, "out_of_band": 0, "hold": True}
    assert r["attribution"] == {"floor_label_at_or_above_the_floor": 0, "shortened_label_at_full_length": 0,
                                "holds": True}
    assert r["per_batch"]["fractions"] == [pytest.approx(5 / 12)] * 2 and r["per_batch"]["sd"] == 0.0
    assert r["per_batch"]["k"] == [3, 3]
    assert r["omega0"]["hz"][2] == pytest.approx(12.0)             # the median of 0.8, 4 and 20 Hz rows
    assert r["rows"] == {"total": 8, "zero_live": 2, "one_live": 0}
    assert r["span"]["values"][1] == pytest.approx(3.0) and r["span"]["single_probe_rows"] == 0
    assert r["dominant_cause"] == "too_slow_at_band_top"
    assert r["cross_check"] == {"row_ranges": 3, "batches": {"expected": 2, "committed": 2}, "coverage": [],
                                "mismatches": [], "abandoned_attempt_warnings": 1, "holds": True}
    assert "fixed probe-generator seed" in d.settings["probe_layout"]
    assert d.settings["configured"]["probe_count"] == {"drawn_per_batch_from": [config.CHI_K_MIN_TRAIN, 12]}
    assert np.load(d.path / "probe_mask.npz")["batch_k"].tolist() == [3, 3]

    with only_masked_probe_warnings():
        d = probe_mask(cfg, prior, num_runs=2, run_size=4, chi_k_fixed=3, name="mask1k3")
    assert calls[1]["chi_k_fixed"] == 3
    assert d.settings["chi_k_fixed"] == d.settings["configured"]["probe_count"] == 3


def test_probes_mask_records_and_warns_a_count_the_production_warning_does_not_confirm(store, monkeypatch,
                                                                                     caplog):
    import warnings
    from core.diagnostics import probe_mask
    from core.SBI import pipeline
    from tests._fixtures import only_masked_probe_warnings
    cfg = _nad_cfg(chi_mode=True)
    prior = _mask_prior(store, cfg)
    t1 = "training batch 1/1 [t_scale=3.73, T=5000, n_fine=59000, N_points=5000, rows=4]"

    def _gen(*a, probe_observer=None, **kw):
        warnings.warn(f"{t1}: chi: 4/12 probes masked (below 2.0 drive cycles).")
        probe_observer(_mask_record(t1, 0, 4))
        return torch.zeros(1, 1), torch.zeros(1, 1)

    monkeypatch.setattr(pipeline, "gen_training_data", _gen)
    with only_masked_probe_warnings():
        d = probe_mask(cfg, prior, num_runs=1, run_size=4, name="mask2")
    check = d.results["cross_check"]
    assert check["mismatches"] == [{"batch_tag": t1, "lo": 0, "hi": 4, "audit": {"masked": 5, "total": 12},
                                    "production": {"masked": 4, "total": 12}}]
    assert (check["row_ranges"], check["abandoned_attempt_warnings"], check["holds"]) == (1, 0, False)
    said = [m.getMessage() for m in caplog.records
            if m.name == "core.diagnostics.probes" and m.levelname == "WARNING"]
    assert sum("does not match" in m for m in said) == 1, said
    assert any("5 of 12" in m and "4 of 12" in m for m in said), said


def test_the_mask_audit_agrees_with_the_production_warnings_on_real_training_rows(monkeypatch):
    """The training generator itself -- its schedule, probe draw, placement, lock-in, packer and
    masked-probe warning -- on the real cell with a stand-in solver, audited from what its observer
    handed over. The split of the cycle floor is checked against a plain per-probe count."""
    import math
    from core import cli, config
    from core.diagnostics import probes
    from core.rng import seeded
    from core.SBI import pipeline
    from tests._fixtures import only_masked_probe_warnings, stand_in_gen_obs
    cfg = _nad_cfg(chi_mode=True)
    cli.load_and_validate_gt(cfg, str(config.CELL_PATH / "nadrowski" / "master_weak.txt"))
    monkeypatch.setattr(pipeline, "gen_obs", stand_in_gen_obs)
    truth = cfg.ground_truth_tensor.reshape(1, -1)

    class _Fixed:
        def sample(self, shape):
            return truth.expand(shape[0], -1).clone()

    n_grid = 12_000
    t = torch.linspace(0, n_grid * cfg.dt_nd_min, n_grid, dtype=cfg.hw.dtype)
    records = []
    with only_masked_probe_warnings(), seeded(0, cfg.hw.device), probes._capture_masked_warnings() as warned:
        pipeline.gen_training_data(
            cfg.model, _Fixed(), None, t, run_size=4, n_runs=2, steady_idx=500, dt_nd_min=cfg.dt_nd_min,
            nd_dim=len(cfg.params_dict), forcing_idx=cfg.forcing_idx, rescale_idx=cfg.rescale_idx,
            dt_exp=cfg.dt_exp, t_min_exp=cfg.t_min_exp, t_max_exp=cfg.t_max_exp,
            t_scale_bounds=cfg.t_scale_bounds, state_dep_drift=cfg.state_dep_drift, chi_mode=True,
            chi_f0=config.CHI_F0, chi_freq_bounds=config.CHI_FREQ_BOUNDS, chi_k_pad=4,
            chi_max_cycles=config.CHI_MAX_CYCLES, n_vars=cfg.inits_tensor.shape[-1],
            dtype=cfg.hw.dtype, device=cfg.hw.device, probe_observer=records.append)
    res = probes._audit(cfg, records, warned, num_runs=2, run_size=4)
    assert res["probes"] > 0 and res["masked"] > 0, res
    check = res["cross_check"]
    assert check["holds"] and check["mismatches"] == [] and check["row_ranges"] == len(records), check
    assert res["masked"] == sum(int((~r.packed_mask[:, :r.k]).sum()) for r in records)
    inv, c = res["invariants"], res["causes"]
    assert (c["cycle_floor"]["count"] + c["non_finite_lock_in"]["count"] + inv["non_finite_frequency"]
            + inv["at_or_above_nyquist"] + inv["out_of_band"]) == res["masked"]
    assert res["attribution"]["holds"], res["attribution"]
    too_slow = shortened = 0
    for rec in records:
        full_cycles_at_top = rec.f_peak * (rec.n_points * rec.dt_exp) * config.CHI_FREQ_BOUNDS[1]
        for b in range(rec.hi - rec.lo):
            for j in range(rec.k):
                freq = float(rec.f_peak[b]) * math.exp(float(rec.u[b, j]))
                if bool(rec.valid[b, j]) or not 0 < freq < 0.9 * 0.5 / rec.dt_exp:
                    continue
                if float(full_cycles_at_top[b]) < config.CHI_MIN_CYCLES:
                    too_slow += 1
                else:
                    shortened += 1
                    assert float(rec.duration_frac[j]) < 1.0
    assert (c["too_slow_at_band_top"]["count"], c["shortened_by_duration_draw"]["count"]) == (too_slow, shortened)
    assert too_slow + shortened == c["cycle_floor"]["count"] > 0


@pytest.mark.parametrize("kw, field", [
    ({"num_runs": 0}, "mask_num_runs"),
    ({"run_size": 0}, "mask_run_size"),
    ({"chi_k_fixed": 1}, "chi_k_fixed"),
    ({"chi_k_fixed": 13}, "chi_k_fixed"),
    ({"seed": -1}, "probe_seed"),
    ({"name": "taken"}, "name"),
])
def test_probes_mask_refuses_before_the_writer_opens(store, monkeypatch, kw, field):
    from core.diagnostics import probe_mask
    from core.SBI import pipeline
    cfg = _nad_cfg(chi_mode=True)
    prior = _mask_prior(store, cfg)
    _diagnostic(store, cfg, name="taken")
    monkeypatch.setattr(pipeline, "gen_training_data", lambda *a, **k: pytest.fail("simulated before the refusal"))
    before = _diagnostic_dirs(store)
    with pytest.raises(Refusal) as e:
        probe_mask(cfg, prior, **{"num_runs": 1, "run_size": 1, **kw})
    assert e.value.field == field, (kw, e.value.field, str(e.value))
    assert _diagnostic_dirs(store) == before


def test_probes_mask_refuses_a_config_that_is_not_chi_or_not_config_pys_band(store, monkeypatch):
    """A non-chi config, and a chi one whose band is not config.py's: training would refuse the second
    before its first simulation, so the audit does too, and neither refusal names a setting to change."""
    from core.diagnostics import probe_mask
    from core.SBI import pipeline
    prior = _mask_prior(store, _nad_cfg(chi_mode=True))
    monkeypatch.setattr(pipeline, "gen_training_data", lambda *a, **k: pytest.fail("simulated before the refusal"))
    for cfg, words in ((_nad_cfg(chi_mode=False), "chi"),
                       (_nad_cfg(chi_mode=True, chi_freq_bounds=(0.1, 10.0)), "does not match config.py")):
        with pytest.raises(Refusal) as e:
            probe_mask(cfg, prior, num_runs=1, run_size=1)
        assert e.value.field is None and words in str(e.value), str(e.value)
    assert _diagnostic_dirs(store) == []


def test_probes_mask_refuses_a_user_model_out_of_sync_with_its_bounds_as_a_prior_load_does(store, monkeypatch):
    """The simulator binds parameter columns by position, so a user model whose definition no longer
    lists its parameters in the bounds file's order is refused before the spend -- by the same check,
    in the same words, as loading a prior for training."""
    from types import SimpleNamespace
    from core import orchestrator, registry
    from core.diagnostics import probe_mask
    from core.SBI import pipeline
    cfg = _nad_cfg(chi_mode=True)
    prior = _mask_prior(store, cfg)
    stale = SimpleNamespace(compiled=SimpleNamespace(param_names=list(reversed(list(cfg.params_dict)))))
    monkeypatch.setattr(registry, "is_user_model", lambda name: True)
    monkeypatch.setattr(registry, "get", lambda name: stale)
    monkeypatch.setattr(pipeline, "gen_training_data", lambda *a, **k: pytest.fail("simulated before the refusal"))
    said = []
    for call in (lambda: probe_mask(cfg, prior, num_runs=1, run_size=1),
                 lambda: orchestrator.build_prior(cfg, "mp", False, store=store)):
        with pytest.raises(Refusal) as e:
            call()
        said.append((e.value.field, str(e.value)))
    assert said[0] == said[1] and said[0][0] is None and "out of sync" in said[0][1], said
    assert _diagnostic_dirs(store) == []


def test_the_mask_cross_check_pairs_each_warning_with_its_own_row_range():
    """Under nested row halving a batch warns for attempts it then abandons. Here [0,8) fails without a
    warning; [0,4) commits masking 2; [4,8) warns 3 and fails; its halves [4,6) and [6,8) commit masking
    1 and nothing. Each committed range is paired with its own warning and the abandoned one is set
    aside. A range the audit counts as masking nothing, when the generator warned about it, is a
    mismatch, never an abandoned attempt."""
    from core.diagnostics import probes
    cfg = _nad_cfg(chi_mode=True)
    tag = "training batch 1/1 [t_scale=3.73, T=5000, n_fine=59000, N_points=10000, rows=8]"
    kinds = ("packer", "full", "short", "full", "short", "full", "full", "full")
    records = [_mask_record(tag, lo, hi, kinds) for lo, hi in ((0, 4), (4, 6), (6, 8))]
    check = probes._audit(cfg, records, [(tag, 2, 12), (tag, 3, 12), (tag, 1, 6)],
                          num_runs=1, run_size=8)["cross_check"]
    assert (check["mismatches"], check["abandoned_attempt_warnings"], check["holds"]) == ([], 1, True), check
    quiet = [_mask_record(tag, 0, 4, ("full",) * 4)]
    check = probes._audit(cfg, quiet, [(tag, 3, 12)], num_runs=1, run_size=4)["cross_check"]
    assert check["mismatches"] == [{"batch_tag": tag, "lo": 0, "hi": 4, "audit": {"masked": 0, "total": 12},
                                    "production": {"masked": 3, "total": 12}}], check
    assert check["abandoned_attempt_warnings"] == 0 and not check["holds"]


def test_probes_mask_finds_a_batch_and_rows_the_audit_never_saw(store, monkeypatch, caplog):
    """Three batches asked for, two reached the audit, and the second of those only for half its rows:
    both are findings, recorded and warned in the one record that says the audit does not match."""
    from core.diagnostics import probe_mask
    from core.SBI import pipeline
    from tests._fixtures import only_masked_probe_warnings
    cfg = _nad_cfg(chi_mode=True)
    prior = _mask_prior(store, cfg)
    t1, t2 = (f"training batch {k}/3 [t_scale=3.73, T=5000, n_fine=59000, N_points=10000, rows=4]" for k in (1, 2))
    script = [(t1, [], [_mask_record(t1, 0, 4)]), (t2, [], [_mask_record(t2, 0, 2)])]
    monkeypatch.setattr(pipeline, "gen_training_data", _mask_generator(script, []))
    with only_masked_probe_warnings():
        check = probe_mask(cfg, prior, num_runs=3, run_size=4, name="seen").results["cross_check"]
    assert check["batches"] == {"expected": 3, "committed": 2} and not check["holds"]
    assert [(c["batch_tag"], c["ranges"], c["rows"]) for c in check["coverage"]] == [(t2, [[0, 2]], 4)]
    said = [m.getMessage() for m in caplog.records
            if m.name == "core.diagnostics.probes" and m.levelname == "WARNING"]
    found = [m for m in said if "does not match" in m]
    assert len(found) == 1 and "2 of the 3 batches" in found[0] and t2 in found[0], said


def test_probes_mask_reports_a_cause_that_cannot_fire_in_training_as_a_broken_invariant(store, monkeypatch,
                                                                                      caplog):
    """A probe at 0.9 x Nyquist, one whose frequency is not finite and a valid one outside the band are
    each counted under their own reason -- never as the cycle floor -- and a warning says a cause that
    cannot fire in training fired. At a half-unit sampling interval the limit is 0.9 cycles per unit, so
    the valid probe at 0.24 is below it."""
    from core.diagnostics import probe_mask
    from core.SBI import pipeline
    from tests._fixtures import only_masked_probe_warnings
    cfg = _nad_cfg(chi_mode=True)
    prior = _mask_prior(store, cfg)
    t1 = "training batch 1/1 [t_scale=3.73, T=5000, n_fine=59000, N_points=10000, rows=2]"
    script = [(t1, [], [_mask_record(t1, 0, 2, ("fast", "wide"))])]
    monkeypatch.setattr(pipeline, "gen_training_data", _mask_generator(script, []))
    with only_masked_probe_warnings():
        r = probe_mask(cfg, prior, num_runs=1, run_size=2, name="broken").results
    assert r["invariants"] == {"non_finite_frequency": 1, "at_or_above_nyquist": 1, "out_of_band": 1, "hold": False}
    c = r["causes"]
    assert (r["masked"], c["cycle_floor"]["count"], c["non_finite_lock_in"]["count"]) == (3, 0, 0)
    assert r["dominant_cause"] is None and r["cross_check"]["holds"]
    said = [m.getMessage() for m in caplog.records
            if m.name == "core.diagnostics.probes" and m.levelname == "WARNING"]
    assert sum("cannot fire in training" in m for m in said) == 1, said


def test_the_mask_audit_flags_a_masked_probe_its_causes_do_not_explain(caplog):
    """A probe masked for a reason the audit does not know -- valid by every rule it checks, locked in
    over the whole recording -- would otherwise be counted under the cycle floor as shortened by the
    duration draw. The labels are checked against the record, and a label it contradicts is a finding."""
    import dataclasses
    from core.diagnostics import probes
    cfg = _nad_cfg(chi_mode=True)
    tag = "training batch 1/1 [t_scale=3.73, T=5000, n_fine=59000, N_points=10000, rows=1]"
    rec = _mask_record(tag, 0, 1, ("full",))
    odd = dataclasses.replace(rec, valid=torch.tensor([[False, True, True]]),
                              packed_mask=torch.arange(12).unsqueeze(0) < 2)
    res = probes._audit(cfg, [odd], [(tag, 1, 3)], num_runs=1, run_size=1)
    assert res["causes"]["shortened_by_duration_draw"]["count"] == 1
    assert res["attribution"] == {"floor_label_at_or_above_the_floor": 1, "shortened_label_at_full_length": 1,
                                  "holds": False}
    probes._log_mask(res)
    said = [m.getMessage() for m in caplog.records
            if m.name == "core.diagnostics.probes" and m.levelname == "WARNING"]
    assert sum("a masking rule the audit does not know" in m for m in said) == 1, said


def _probe_said(caplog, level):
    return [m.getMessage() for m in caplog.records if m.name == "core.diagnostics.probes" and m.levelname == level]


def test_probes_drive_picks_the_leading_free_running_run_and_the_weakest_capture(store, monkeypatch, caplog):
    import numpy as np
    from core.diagnostics import probes, probe_drive
    refill = lambda a: torch.where(a < 0.1, 1.0, torch.where(a < 0.2, 0.5, torch.where(a < 5.0, 0.1, 1.0)))
    monkeypatch.setattr(probes, "_simulate", _probe_stand_in(own_factor=refill))   # own peak on bin 125 at 5 s
    d = probe_drive(_probe_cfg(chi=False), repeats=2, strengths=[0.01, 0.05, 0.1, 0.2, 1.0, 8.0], name="drive1")
    r = d.results
    assert [s["verdict"] for s in r["strengths"]] == ["free-running", "free-running", "in between",
                                                    "captured", "captured", "free-running"]
    assert r["strongest_free_running"]["strength"] == 0.05                    # never the refilled 8.0
    assert r["strongest_free_running"]["cell_force"] == pytest.approx(0.5)
    assert (r["weakest_captured"]["strength"], r["weakest_captured"]["cell_force"]) == (0.2, pytest.approx(2.0))
    assert r["omega0"]["hz"] == pytest.approx(25.0) and r["drive_frequency"]["hz"] == pytest.approx(35.0)
    assert r["oscillating"] is True and r["clarity"] > 3.0 and r["chi_f0_row"] is None
    sf = r["suggested_forcing"]
    assert sf["free_running"]["amp"] == pytest.approx(0.5) and sf["free_running"]["freq"] == pytest.approx(0.035)
    assert sf["captured"]["amp"] == pytest.approx(2.0)
    assert sf["box"]["amp"] == [0.0, pytest.approx(6.0)] and sf["box"]["freq"] == [pytest.approx(0.0025), pytest.approx(0.25)]
    assert np.load(d.path / "probe_drive.npz")["own_peak_ratio"].shape == (6,)
    said = [m.getMessage() for m in caplog.records if m.name == "core.diagnostics.probes" and m.levelname == "INFO"]
    assert any(m.startswith("[drive] strongest free-running strength 0.05") for m in said), said
    assert any(m.startswith("[drive] weakest captured strength 0.2") for m in said), said
    assert any(m.startswith("[drive] suggested Forcing section") for m in said), said
    # the record's shape and its values: the own oscillation keeps its amplitude times own_factor, so
    # the own-peak ratio is that factor squared, and the response is |chi| = 2 at every strength
    assert (d.diagnostic, d.variant, d.manifest.parents) == ("probes", "drive", {})
    assert set(r) == {"omega0", "peak_per_trace_hz", "clarity", "noise_clarity", "oscillating", "peak_to_peak",
                      "cycles_in_recording", "n_obs", "drive_frequency", "force_scale", "strengths",
                      "strongest_free_running", "weakest_captured", "chi_f0_row", "suggested_forcing", "notes"}
    assert [s["own_peak_ratio"] for s in r["strengths"]] == pytest.approx([1.0, 1.0, 0.25, 0.01, 0.01, 1.0], abs=2e-3)
    assert [s["chi_median"] for s in r["strengths"]] == pytest.approx([2.0] * 6, rel=1e-2)
    assert [s["cell_force"] for s in r["strengths"]] == pytest.approx([0.1, 0.5, 1.0, 2.0, 10.0, 80.0])
    assert r["strengths"][-1]["plv_median"] > 0.99 and all(0.0 <= s["plv_median"] <= 1.0 for s in r["strengths"])
    assert (r["n_obs"], r["cycles_in_recording"], r["force_scale"]) == (5000, pytest.approx(125.0), pytest.approx(10.0))
    assert r["omega0"]["cell_units"] == pytest.approx(0.025)
    assert r["peak_per_trace_hz"] == {"median": pytest.approx(25.0), "spread": pytest.approx(0.0, abs=1e-9)}
    assert r["drive_frequency"] == {"cell_units": pytest.approx(0.035), "hz": pytest.approx(35.0), "detune": 1.4,
                                    "harmonic": None}
    assert r["peak_to_peak"] == pytest.approx(2.0, abs=0.1)
    assert sf["free_running"] == {"amp": pytest.approx(0.5), "freq": pytest.approx(0.035), "phase": 0.0, "offset": 0.0}
    assert sf["captured"] == {"amp": pytest.approx(2.0), "freq": pytest.approx(0.035), "phase": 0.0, "offset": 0.0}
    assert sf["box"]["phase"] == [0.0, 6.283185307] and sf["box"]["offset"] == [-50.0, 50.0]
    z = np.load(d.path / "probe_drive.npz")
    assert set(z.files) == {"strengths", "own_peak_ratio", "chi_median", "plv_median", "freqs", "power"}
    assert z["strengths"].tolist() == [0.01, 0.05, 0.1, 0.2, 1.0, 8.0] and z["freqs"].shape == z["power"].shape
    assert d.manifest.figures == ["figures/probe_drive_own_peak_ratio.png"]
    assert store.load_diagnostic(d.id).results == r
    # at two runs pure noise reaches a clarity of about 9 in 1 of 100 ensembles, which the default
    # threshold of 3 does not clear: that is the run's one warning, and the record states the level
    from core.diagnostics import probe_math
    assert r["noise_clarity"] == {"median": pytest.approx(probe_math.noise_clarity(2, 2500, 0.5)),
                                  "p99": pytest.approx(probe_math.noise_clarity(2, 2500, 0.99))}
    warned = _probe_said(caplog, "WARNING")
    assert len(warned) == 1 and warned[0].startswith("[drive] the clarity threshold 3 does not clear"), warned


def test_probes_drive_measures_the_configured_chi_drive_explicitly_on_its_default_grid(store, monkeypatch):
    import math
    from core.diagnostics import probes, probe_drive
    monkeypatch.setattr(probes, "_simulate", _probe_stand_in())        # captures from strength 1.0
    d = probe_drive(_probe_cfg(chi=False), repeats=2, name="drive_grid")
    s, r = d.settings, d.results
    assert s["strengths"] == [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0,
                              12.0, 20.0]
    row = r["chi_f0_row"]
    assert set(row) == {"strength", "own_peak_ratio", "verdict", "note"}
    assert row["strength"] == 0.15 and row["verdict"] == "free-running" and "detune" in row["note"]
    assert row["own_peak_ratio"] == pytest.approx(1.0, abs=2e-3)
    assert any("no linearity test" in n for n in r["notes"])
    assert s["drive_phase"] == pytest.approx(math.pi / 2)
    assert set(s) == {"t_obs_s", "repeats", "detune", "strengths", "free_min", "captured_max", "peak_window",
                      "clarity_min", "seed", "drive_phase", "configured"}
    assert (s["t_obs_s"], s["repeats"], s["detune"], s["free_min"], s["captured_max"], s["peak_window"],
            s["clarity_min"], s["seed"]) == (5.0, 2, 1.4, 0.7, 0.1, 0.02, 3.0, 0)
    assert s["configured"] == {"chi_f0": 0.15, "chi_freq_bounds": [0.03, 0.3], "chi_k_pad": 12,
                               "chi_min_cycles": 2.0, "chi_max_cycles": 20.0, "probe_count": 6}
    assert (r["strongest_free_running"]["strength"], r["weakest_captured"]["strength"]) == (0.75, 1.0)


def test_probes_drive_reports_what_it_cannot_judge_as_results_not_refusals(store, monkeypatch, caplog):
    """Each of these is known only after the undriven spend: no clear oscillation, nothing captured in
    the grid, and a drive at 0.9 x Nyquist or above. Each writes its record and says so in one warning.
    The two where nothing is judged drive nothing at all. Sixteen runs, so that the clarity threshold
    clears what pure noise reaches and no warning about it joins the one being pinned."""
    from core.diagnostics import probes, probe_drive
    warned, ids = [], []

    def run(stand_in, **kw):
        caplog.clear()
        monkeypatch.setattr(probes, "_simulate", stand_in)
        d = probe_drive(_probe_cfg(chi=False), repeats=16, **kw)
        warned.append(_probe_said(caplog, "WARNING"))
        ids.append(d.id)
        return d.results

    seen = []
    r = run(_probe_stand_in(seen=seen), clarity_min=1e9, strengths=[0.1, 2.0], name="quiet")
    assert r["oscillating"] is False and seen == []
    assert all(s["verdict"] is None and s["own_peak_ratio"] is None and s["chi_median"] is None
               for s in r["strengths"])
    assert r["strongest_free_running"] is None and r["weakest_captured"] is None
    assert r["suggested_forcing"] == {"free_running": None, "captured": None, "box": None}
    assert r["clarity"] > 3.0                                      # measured, below the threshold asked for
    r = run(_probe_stand_in(), strengths=[0.1, 0.5], name="uncaptured")
    assert r["weakest_captured"] is None and r["strongest_free_running"]["strength"] == 0.5
    assert r["suggested_forcing"]["captured"] is None and r["suggested_forcing"]["box"] is None
    assert r["suggested_forcing"]["free_running"]["amp"] == pytest.approx(5.0)
    seen = []
    r = run(_probe_stand_in(f_own=0.4, seen=seen), strengths=[0.1], name="fast")
    assert r["drive_frequency"]["cell_units"] == pytest.approx(0.56) and r["strengths"][0]["verdict"] is None
    assert r["oscillating"] is True and seen == []
    assert [len(w) for w in warned] == [1, 1, 1], warned
    assert "no clear oscillation" in warned[0][0]
    assert "nothing in the grid captured" in warned[1][0]
    assert "Nyquist" in warned[2][0]
    assert [store.load_diagnostic(i).variant for i in ids] == ["drive"] * 3


def test_probes_drive_masks_a_drive_between_nine_tenths_of_nyquist_and_nyquist(store, monkeypatch, caplog):
    """The edge of training's masking rule, where "at or above Nyquist" parts from it: at 1 ms sampling
    Nyquist is 0.5 cycles per ms, and a cell at 0.33 is driven at 0.462 -- below Nyquist, above 0.45."""
    from core.diagnostics import probes, probe_drive
    seen = []
    monkeypatch.setattr(probes, "_simulate", _probe_stand_in(f_own=0.33, seen=seen))
    r = probe_drive(_probe_cfg(chi=False), repeats=2, strengths=[0.1], name="edge").results
    assert r["drive_frequency"]["cell_units"] == pytest.approx(0.462) and r["strengths"][0]["verdict"] is None
    assert seen == [] and any("Nyquist" in m for m in _probe_said(caplog, "WARNING"))


def test_probes_drive_turns_non_finite_measures_into_null(store, monkeypatch):
    """A driven ensemble that returns NaN is not measured: a NaN compares false against every threshold
    and would otherwise read as captured or in between, and a manifest refuses it."""
    from core.diagnostics import probes, probe_drive
    monkeypatch.setattr(probes, "_simulate", _probe_stand_in(nan_driven=True))
    d = probe_drive(_probe_cfg(chi=False), repeats=2, strengths=[0.1], name="drive_nan")
    row, = d.results["strengths"]
    assert row["own_peak_ratio"] is None and row["verdict"] is None
    assert row["chi_median"] is None and row["plv_median"] is None
    assert d.results["strongest_free_running"] is None and d.results["weakest_captured"] is None
    assert store.load_diagnostic(d.id).results == d.results


def test_probes_drive_judges_each_strength_against_the_thresholds_it_is_given(store, monkeypatch, caplog):
    """Own-peak ratios of 1, 0.25 and 0.01: free-running at or above free_min, captured at or below
    captured_max, in between otherwise -- so moving either threshold past 0.25 moves that one strength,
    and a grid whose weakest strength is not free-running names no free-running strength at all."""
    from core.diagnostics import probes, probe_drive
    refill = lambda a: torch.where(a < 0.1, 1.0, torch.where(a < 0.2, 0.5, 0.1))
    monkeypatch.setattr(probes, "_simulate", _probe_stand_in(own_factor=refill))
    kw = dict(repeats=2, strengths=[0.05, 0.1, 0.2])

    def verdicts(r):
        return [s["verdict"] for s in r["strengths"]]

    r = probe_drive(_probe_cfg(chi=False), name="t_default", **kw).results
    assert verdicts(r) == ["free-running", "in between", "captured"]
    r = probe_drive(_probe_cfg(chi=False), free_min=0.2, name="t_free", **kw).results
    assert verdicts(r) == ["free-running", "free-running", "captured"]
    assert r["strongest_free_running"]["strength"] == 0.1 and r["weakest_captured"]["strength"] == 0.2
    r = probe_drive(_probe_cfg(chi=False), free_min=0.5, captured_max=0.3, name="t_captured", **kw).results
    assert verdicts(r) == ["free-running", "captured", "captured"]
    assert r["strongest_free_running"]["strength"] == 0.05 and r["weakest_captured"]["strength"] == 0.1
    caplog.clear()
    r = probe_drive(_probe_cfg(chi=False), repeats=16, strengths=[0.1, 0.2], name="t_strong").results
    assert verdicts(r) == ["in between", "captured"]
    assert r["strongest_free_running"] is None and r["weakest_captured"]["strength"] == 0.2
    assert r["suggested_forcing"]["free_running"] is None and r["suggested_forcing"]["captured"] is not None
    warned = _probe_said(caplog, "WARNING")
    assert len(warned) == 1 and "no free-running strength" in warned[0], warned


def test_probes_drive_states_every_frequency_and_length_in_the_cells_own_units(store, monkeypatch):
    """At a half-millisecond sampling interval, a sampling interval of 1 would hide a units inversion: a
    5 s recording holds 10000 samples, the own peak stays at 25 Hz and 125 cycles, the lock-in reads
    |chi| = 2 only on the true time axis, and 0.9 x Nyquist is 0.9 cycles per ms -- so a drive at 0.56,
    masked at 1 ms, is driven here."""
    from core.diagnostics import probes, probe_drive
    refill = lambda a: torch.where(a < 0.1, 1.0, 0.1)
    monkeypatch.setattr(probes, "_simulate", _probe_stand_in(own_factor=refill))
    cfg = _probe_cfg(chi=False)
    cfg.dt_exp = 0.5
    r = probe_drive(cfg, repeats=2, strengths=[0.05, 0.2], name="half_ms").results
    assert (r["n_obs"], r["cycles_in_recording"]) == (10000, pytest.approx(125.0))
    assert r["omega0"] == {"cell_units": pytest.approx(0.025), "hz": pytest.approx(25.0)}
    assert r["peak_per_trace_hz"]["median"] == pytest.approx(25.0)
    assert r["drive_frequency"]["hz"] == pytest.approx(35.0)
    assert [s["verdict"] for s in r["strengths"]] == ["free-running", "captured"]
    assert [s["chi_median"] for s in r["strengths"]] == pytest.approx([2.0, 2.0], rel=1e-2)
    seen = []
    monkeypatch.setattr(probes, "_simulate", _probe_stand_in(f_own=0.4, seen=seen))
    r = probe_drive(cfg, repeats=2, strengths=[0.1], name="half_ms_fast").results
    assert r["drive_frequency"]["cell_units"] == pytest.approx(0.56) and len(seen) == 1
    assert r["strengths"][0]["verdict"] == "free-running"


def _pulled_stand_in(shift_bins):
    """In place of probes._simulate, a cell whose own oscillation, at 0.025 cycles per ms, keeps its
    amplitude under a drive but is pulled ``shift_bins`` frequency bins up the axis; no other response."""
    import math

    def _sim(cfg, geom, nd, res_sim, sim_idx, inits, *, amp_dim=None, freq=None):
        B, n = nd.shape[0], geom.n_obs
        t = torch.arange(n, dtype=torch.float64) * cfg.dt_exp
        f = 0.025 + (0.0 if amp_dim is None else shift_bins / (n * cfg.dt_exp))
        x = torch.sin(2 * math.pi * f * t + 2 * math.pi * torch.rand(B, 1, dtype=torch.float64))
        return (x + 0.01 * torch.randn(B, n, dtype=torch.float64)).to(cfg.hw.dtype)
    return _sim


def test_probes_drive_reads_the_own_peak_window_it_is_given_and_never_judges_one_the_drive_reaches(
        store, monkeypatch, caplog):
    """The window's width is the one asked for: an oscillation pulled four bins up leaves a 2 % window
    (two bins either side of bin 125) and stays inside a 5 % one (six bins). And the window is never
    narrower than two bins, so over 1 s (bins of 1 Hz, the peak on bin 25) a drive at 1.05 x the peak
    lands on bin 26, inside it: that strength is not judged, while 1.12 x -- bin 28 -- is."""
    from core.diagnostics import probes, probe_drive
    monkeypatch.setattr(probes, "_simulate", _pulled_stand_in(4))
    kw = dict(repeats=2, strengths=[0.1])
    r = probe_drive(_probe_cfg(chi=False), name="w_narrow", **kw).results
    assert r["strengths"][0]["verdict"] == "captured" and r["strengths"][0]["own_peak_ratio"] < 0.01
    r = probe_drive(_probe_cfg(chi=False), peak_window=0.05, name="w_wide", **kw).results
    assert r["strengths"][0]["verdict"] == "free-running"
    assert r["strengths"][0]["own_peak_ratio"] == pytest.approx(1.0, abs=2e-3)
    seen = []
    monkeypatch.setattr(probes, "_simulate", _probe_stand_in(seen=seen))
    caplog.clear()
    r = probe_drive(_probe_cfg(chi=False), t_obs_s=1.0, detune=1.05, name="w_reached", repeats=16,
                    strengths=[0.1]).results
    assert r["strengths"][0]["verdict"] is None and seen == []
    warned = _probe_said(caplog, "WARNING")
    assert len(warned) == 1 and "own-peak window" in warned[0], warned
    r = probe_drive(_probe_cfg(chi=False), t_obs_s=1.0, detune=1.12, name="w_clear", **kw).results
    assert r["strengths"][0]["verdict"] == "free-running" and len(seen) == 1


def test_probes_drive_repeats_at_one_seed_and_differs_at_another(store, monkeypatch):
    from core.diagnostics import probes, probe_drive
    monkeypatch.setattr(probes, "_simulate", _probe_stand_in())
    kw = dict(repeats=2, strengths=[0.1, 2.0])
    a = probe_drive(_probe_cfg(chi=False), seed=5, name="dseed_a", **kw)
    b = probe_drive(_probe_cfg(chi=False), seed=5, name="dseed_b", **kw)
    c = probe_drive(_probe_cfg(chi=False), seed=6, name="dseed_c", **kw)
    assert a.results == b.results and a.settings == b.settings
    assert c.results != a.results and c.settings["seed"] == 6


def test_a_drive_check_leaves_its_configuration_a_training_one_and_config_py_untouched(store, monkeypatch):
    """The drive strengths and the detuned frequency reach the simulator and nothing else."""
    from core import config
    from core.diagnostics import probes, probe_drive
    from tests._fixtures import assert_cfg_unchanged, snapshot_cfg
    training = _nad_cfg(chi_mode=True)
    cfg = _probe_cfg(chi=False)
    snaps = [(training, snapshot_cfg(training)), (cfg, snapshot_cfg(cfg))]
    constants = {k: getattr(config, k) for k in dir(config) if k.startswith("CHI_")}
    seen = []
    monkeypatch.setattr(probes, "_simulate", _probe_stand_in(seen=seen))
    probe_drive(cfg, repeats=2, strengths=[0.5], name="drive_untouched")
    for watched, snap in snaps:
        assert_cfg_unchanged(watched, snap)
    assert {k: getattr(config, k) for k in dir(config) if k.startswith("CHI_")} == constants
    assert seen and all(v == pytest.approx(0.5) for strengths, _ in seen for v in strengths), seen
    assert all(f == pytest.approx(0.035) for _, freqs in seen for f in freqs), seen


_DRIVE_BASE = dict(repeats=2, strengths=[0.1])


@pytest.mark.parametrize("kw, field", [
    ({"t_obs_s": 0.0}, "drive_t_obs"),
    ({"t_obs_s": 1e-7}, "drive_t_obs"),
    ({"t_obs_s": 1e4}, "drive_t_obs"),
    ({"repeats": 1}, "drive_repeats"),
    ({"detune": 0.0}, "drive_detune"),
    ({"detune": 1.01}, "drive_detune"),
    ({"detune": 0.99}, "drive_detune"),
    ({"strengths": []}, "drive_strengths"),
    ({"strengths": [0.1, 0.05]}, "drive_strengths"),
    ({"strengths": [0.1, 0.1]}, "drive_strengths"),
    ({"strengths": [0.1, float("nan")]}, "drive_strengths"),
    ({"strengths": [-0.1]}, "drive_strengths"),
    ({"free_min": 1.2}, "free_min"),
    ({"free_min": 0.0}, "free_min"),
    ({"captured_max": 0.0}, "captured_max"),
    ({"captured_max": 0.8}, "captured_max"),
    ({"captured_max": 0.7}, "captured_max"),
    ({"peak_window": 0.0}, "drive_peak_window"),
    ({"peak_window": 1.0}, "drive_peak_window"),
    ({"clarity_min": 0.0}, "clarity_min"),
    ({"clarity_min": float("nan")}, "clarity_min"),
    ({"seed": -1}, "probe_seed"),
    ({"name": "taken"}, "name"),
])
def test_probes_drive_refuses_before_the_writer_opens(store, monkeypatch, kw, field):
    from core.diagnostics import probes, probe_drive
    cfg = _probe_cfg(chi=False)
    _diagnostic(store, cfg, name="taken")
    monkeypatch.setattr(probes, "_simulate", lambda *a, **k: pytest.fail("simulated before the refusal"))
    before = _diagnostic_dirs(store)
    with pytest.raises(Refusal) as e:
        probe_drive(cfg, **{**_DRIVE_BASE, **kw})
    assert e.value.field == field, (kw, e.value.field, str(e.value))
    assert _diagnostic_dirs(store) == before


def test_probes_drive_refuses_a_box_with_no_forcing_section(store, monkeypatch):
    """The strengths and the suggested Forcing lines are stated in the force unit a Forcing box carries;
    the spontaneous box declares no Forcing section, so the check refuses it before anything is spent."""
    from core.diagnostics import probes, probe_drive
    monkeypatch.setattr(probes, "_simulate", lambda *a, **k: pytest.fail("simulated before the refusal"))
    with pytest.raises(Refusal) as e:
        probe_drive(_probe_cfg("master_spont.txt", "master_spont.txt", chi=False), **_DRIVE_BASE)
    assert e.value.field is None and "Forcing section" in str(e.value), str(e.value)
    assert _diagnostic_dirs(store) == []


def test_probes_drive_refuses_a_recording_the_time_grid_cannot_hold_and_names_the_longest(store, monkeypatch):
    """A recording longer than the pre-simulated time grid holds is refused before anything is spent --
    never clipped to what fits, which would record one length and measure another -- and the refusal
    names the longest the grid holds for this cell, rounded down to the millisecond, which fits."""
    import math
    from core.diagnostics import probe_math, probes, probe_drive
    from core.SBI import derived
    cfg = _probe_cfg(chi=False)
    res = torch.tensor([[v for v, _ in cfg.rescale_params.values()]], dtype=cfg.hw.dtype)
    res_sim, idx = derived.for_simulation(cfg, cfg.params_tensor, res)
    t_scale = float(res_sim[0, idx["t_scale"]])
    n_grid, sub = cfg.t.shape[0], probe_math.recording_geometry(cfg, 1.0, t_scale).subsample
    fit = (n_grid - cfg.steady_idx) // sub
    longest = math.floor(fit * cfg.dt_exp / cfg.get_unit_conversion_factor("s") * 1000) / 1000
    assert probe_math.recording_geometry(cfg, longest, t_scale).n_fine <= n_grid
    assert probe_math.recording_geometry(cfg, longest + 0.01, t_scale).n_fine > n_grid
    monkeypatch.setattr(probes, "_simulate", lambda *a, **k: pytest.fail("simulated before the refusal"))
    with pytest.raises(Refusal) as e:
        probe_drive(cfg, **{**_DRIVE_BASE, "t_obs_s": longest + 0.01})
    assert e.value.field == "drive_t_obs" and f"at most {longest:.3f} s" in str(e.value), str(e.value)
    assert _diagnostic_dirs(store) == []


def test_the_drive_checks_peak_is_the_highest_bin_and_its_clarity_that_bin_over_the_median_bin():
    """The undriven measures on a built ensemble at a half-unit sampling interval, against a spectrum
    computed here independently: the peak is the ensemble spectrum's highest bin above zero frequency,
    and the clarity is that one bin's power over the median power above zero frequency -- not over the
    mean power, and not a window's summed power, each of which differs here by several per cent."""
    import math
    import numpy as np
    from core.diagnostics import probes
    g = torch.Generator().manual_seed(0)
    n, dt = 4000, 0.5
    t = torch.arange(n, dtype=torch.float64) * dt
    x0 = (0.3 * torch.sin(2 * math.pi * 0.0125 * t + 6 * torch.rand(6, 1, generator=g, dtype=torch.float64))
          + torch.randn(6, n, generator=g, dtype=torch.float64))
    u = probes._undriven(x0, dt)
    x = x0.numpy() - x0.numpy().mean(axis=-1, keepdims=True)
    power = (np.abs(np.fft.rfft(x, axis=-1)) ** 2).mean(axis=0)
    k = 1 + int(np.argmax(power[1:]))
    assert k == 25 and u.omega0 == float(torch.fft.rfftfreq(n, d=dt, dtype=torch.float64)[k])
    assert u.omega0 == pytest.approx(0.0125)
    assert u.clarity == pytest.approx(power[k] / np.median(power[1:]), rel=1e-12)
    assert u.clarity != pytest.approx(power[k] / power[1:].mean(), rel=0.02)
    assert u.clarity != pytest.approx(power[k - 2:k + 3].sum() / np.median(power[1:]), rel=0.02)


def _noise_stand_in(seen=None):
    """In place of probes._simulate, pure white noise, driven or not: a cell with no oscillation."""
    def _sim(cfg, geom, nd, res_sim, sim_idx, inits, *, amp_dim=None, freq=None):
        if amp_dim is not None and seen is not None:
            seen.append(freq.double().flatten().tolist())
        return torch.randn(nd.shape[0], geom.n_obs, dtype=torch.float64).to(cfg.hw.dtype)
    return _sim


def test_probes_drive_reads_pure_noise_over_sixteen_runs_as_no_clear_oscillation(store, monkeypatch, caplog):
    """Over 16 runs of 5000 samples white noise scores a single-bin clarity of about 2.1 -- below the
    default 3, so nothing is judged and nothing driven. Its power summed over a five-bin window would
    score about 6.5 and call the noise an oscillation."""
    from core.diagnostics import probe_math, probes, probe_drive
    seen = []
    monkeypatch.setattr(probes, "_simulate", _noise_stand_in(seen))
    r = probe_drive(_probe_cfg(chi=False), repeats=16, strengths=[0.1], name="noise16").results
    assert r["oscillating"] is False and r["clarity"] < 3.0 and seen == []
    assert r["clarity"] == pytest.approx(probe_math.noise_clarity(16, 2500, 0.5), rel=0.15)
    warned = _probe_said(caplog, "WARNING")
    assert len(warned) == 1 and "no clear oscillation" in warned[0], warned


def test_the_noise_clarity_is_the_clarity_white_noise_scores_at_that_size():
    """Analytic, from the Gamma law of one bin of an ensemble-mean white-noise spectrum, and checked here
    against simulated white noise measured by the drive check's own helper: at 2 and 16 runs of 2000
    samples the median of 200 simulated clarities lies within 3 % of it, and its 99th percentile is
    passed by no more than 4 % of them. Fewer runs, a higher level; no bins, not measured."""
    import math
    from core.diagnostics import probe_math, probes
    assert probe_math.noise_clarity(16, 2500, 0.5) == pytest.approx(2.141, abs=1e-3)
    assert probe_math.noise_clarity(2, 2500, 0.99) == pytest.approx(9.062, abs=1e-3)
    assert probe_math.noise_clarity(4, 2500, 0.5) > probe_math.noise_clarity(8, 2500, 0.5)
    assert math.isnan(probe_math.noise_clarity(4, 0, 0.5))
    g = torch.Generator().manual_seed(1)
    for runs in (2, 16):
        scores = torch.tensor([probes._undriven(torch.randn(runs, 2000, generator=g, dtype=torch.float64),
                                                1.0).clarity for _ in range(200)])
        assert float(scores.median()) == pytest.approx(probe_math.noise_clarity(runs, 1000, 0.5), rel=0.03)
        assert float((scores >= probe_math.noise_clarity(runs, 1000, 0.99)).double().mean()) <= 0.04


def test_probes_drive_states_the_clarity_noise_reaches_and_warns_a_threshold_that_does_not_clear_it(
        store, monkeypatch, caplog):
    """Over 4 runs of 5000 samples white noise scores a clarity of about 4 and reaches about 5.4 in 1 of
    100 ensembles: the default threshold of 3 lets it pass as an oscillation. The record states both
    levels, a note says what they mean, and one warning says the threshold does not clear them. A
    threshold of 6 clears them: no such warning, and the noise is no clear oscillation."""
    from core.diagnostics import probe_math, probes, probe_drive
    monkeypatch.setattr(probes, "_simulate", _noise_stand_in())
    r = probe_drive(_probe_cfg(chi=False), repeats=4, strengths=[0.1], name="noise4").results
    assert r["noise_clarity"] == {"median": pytest.approx(probe_math.noise_clarity(4, 2500, 0.5)),
                                  "p99": pytest.approx(probe_math.noise_clarity(4, 2500, 0.99))}
    assert r["noise_clarity"]["median"] == pytest.approx(3.99, abs=0.01) and r["oscillating"] is True
    assert any("noise_clarity" in n for n in r["notes"])
    warned = _probe_said(caplog, "WARNING")
    assert sum(m.startswith("[drive] the clarity threshold 3 does not clear") for m in warned) == 1, warned
    caplog.clear()
    r = probe_drive(_probe_cfg(chi=False), repeats=4, clarity_min=6.0, strengths=[0.1], name="noise4_high").results
    assert r["oscillating"] is False
    assert not any("does not clear" in m for m in _probe_said(caplog, "WARNING"))


def test_probes_drive_names_the_top_of_the_leading_free_running_run_not_the_strongest_below_the_first_capture(
        store, monkeypatch):
    """Own-peak ratios 1, 0.25, 1 and 0.01: free-running, in between, free-running, captured. The strongest
    free-running strength is the first -- the leading run ends at the in-between strength -- where "the
    strongest free-running strength below the first capture" would name the third. With no capture at
    all (1, 0.25, 1) it is still the first, never the refilled third."""
    from core.diagnostics import probes, probe_drive
    factor = lambda a: torch.where(a < 0.1, 1.0, torch.where(a < 0.2, 0.5, torch.where(a < 0.5, 1.0, 0.1)))
    monkeypatch.setattr(probes, "_simulate", _probe_stand_in(own_factor=factor))
    r = probe_drive(_probe_cfg(chi=False), repeats=2, strengths=[0.05, 0.1, 0.2, 0.5], name="lead1").results
    assert [s["verdict"] for s in r["strengths"]] == ["free-running", "in between", "free-running", "captured"]
    assert r["strongest_free_running"]["strength"] == 0.05 and r["weakest_captured"]["strength"] == 0.5
    assert r["suggested_forcing"]["free_running"]["amp"] == pytest.approx(0.5)
    r = probe_drive(_probe_cfg(chi=False), repeats=2, strengths=[0.05, 0.1, 0.2], name="lead2").results
    assert [s["verdict"] for s in r["strengths"]] == ["free-running", "in between", "free-running"]
    assert r["strongest_free_running"]["strength"] == 0.05 and r["weakest_captured"] is None


def test_probes_drive_warns_and_notes_a_recording_with_few_cycles_of_the_peak(store, monkeypatch, caplog):
    """Over 1 s the 25 Hz peak fits 25 cycles, fewer than 30: the drive then lies few bins from the
    own-peak window, and an off-bin drive's leakage can make a captured cell read in between or
    free-running. The strength is still judged; a warning and a note say so. Over 5 s, 125 cycles,
    neither."""
    from core.diagnostics import probes, probe_drive
    monkeypatch.setattr(probes, "_simulate", _probe_stand_in())        # captures from strength 1.0
    kw = dict(repeats=16, strengths=[0.1, 2.0])
    r = probe_drive(_probe_cfg(chi=False), t_obs_s=1.0, name="few", **kw).results
    warned = _probe_said(caplog, "WARNING")
    assert len(warned) == 1 and "25 cycles" in warned[0], warned
    assert [s["verdict"] for s in r["strengths"]] == ["free-running", "captured"]
    assert len(r["notes"]) == len(probes.DRIVE_NOTES) + 1 and "fewer than 30 cycles" in r["notes"][-1]
    caplog.clear()
    r = probe_drive(_probe_cfg(chi=False), name="many", **kw).results
    assert _probe_said(caplog, "WARNING") == [] and r["notes"] == list(probes.DRIVE_NOTES)


def test_probes_drive_flags_a_sub_harmonic_drive_whose_harmonic_lands_on_the_peak(store, monkeypatch, caplog):
    """At half the peak frequency the drive's second harmonic lands on the peak itself, at a third its
    third: a nonlinear cell's response there can refill the own-peak window and hide capture. The
    record names the order and a warning says so; the strengths are still judged. At 1.4 x no harmonic
    comes near (the first test pins that as None). The peak sits at 24 Hz, bin 120 of a 5 s recording,
    so both drives fall on exact bins."""
    from core.diagnostics import probes, probe_drive
    monkeypatch.setattr(probes, "_simulate", _probe_stand_in(f_own=0.024))    # captures from strength 1.0
    for detune, order in ((0.5, 2), (1 / 3, 3)):
        caplog.clear()
        r = probe_drive(_probe_cfg(chi=False), repeats=16, detune=detune, strengths=[0.1, 2.0],
                        name=f"sub{order}").results
        assert r["drive_frequency"]["harmonic"] == order
        assert [s["verdict"] for s in r["strengths"]] == ["free-running", "captured"]
        warned = _probe_said(caplog, "WARNING")
        assert len(warned) == 1 and f"harmonic {order}" in warned[0], warned


def test_the_drive_verdict_and_the_clarity_threshold_are_inclusive():
    """free-running AT free_min, captured AT captured_max, an oscillation AT clarity_min; a measure that
    was not taken is judged nothing."""
    from core.diagnostics import probes
    assert probes._drive_verdict(0.7, 0.7, 0.1) == "free-running"
    assert probes._drive_verdict(0.1, 0.7, 0.1) == "captured"
    assert probes._drive_verdict(0.6999, 0.7, 0.1) == probes._drive_verdict(0.1001, 0.7, 0.1) == "in between"
    assert probes._drive_verdict(None, 0.7, 0.1) is None
    assert probes._oscillating(3.0, 3.0) is True and probes._oscillating(2.9999, 3.0) is False
    assert probes._oscillating(None, 3.0) is False


def test_probes_drive_states_a_tier1_cell_in_its_derived_force_scale(store, monkeypatch):
    """On the box that declares temperature in place of the force scale, every strength is driven and
    stated at the force scale derived from it -- 46.99 for the master cell's tier-1 twin."""
    from core.diagnostics import probes, probe_drive
    seen = []
    monkeypatch.setattr(probes, "_simulate", _probe_stand_in(seen=seen))
    cfg = _probe_cfg("master_tier1.txt", "master_spont_tier1.txt", chi=False)
    r = probe_drive(cfg, repeats=2, strengths=[0.1, 2.0], name="tier1").results
    scale = 50 * 14.1 * cfg.k_b_cell * 300 / 62.14
    assert r["force_scale"] == pytest.approx(scale, rel=1e-5) and r["force_scale"] == pytest.approx(46.99, abs=0.01)
    assert [s["cell_force"] for s in r["strengths"]] == pytest.approx([0.1 * scale, 2.0 * scale], rel=1e-5)
    assert r["suggested_forcing"]["free_running"]["amp"] == pytest.approx(0.1 * scale, rel=1e-5)
    assert r["suggested_forcing"]["captured"]["amp"] == pytest.approx(2.0 * scale, rel=1e-5)
    assert [strengths[0] for strengths, _ in seen] == pytest.approx([0.1, 2.0], rel=1e-5)
