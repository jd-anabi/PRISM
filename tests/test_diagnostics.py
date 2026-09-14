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
