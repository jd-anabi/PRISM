"""Artifact-identity tests: the master Bounds/Cells triple, and the guards that stop a prior or a
posterior being used against a configuration it does not belong to.

WHAT THESE LOCK DOWN
    A posterior is only meaningful against the exact (model, parameter set + ORDER, box) it was
    trained in -- the flow learns a density over the LATENT coordinate, so the box is what turns its
    output back into physical values. None of that used to be checked. `build_prior` validated
    NOTHING on its load path, `build_posterior` checked only the log-mask, the old width-computing
    mode guard checked only the observation mode and the conditioning WIDTH, and the old sidecar
    loader rebuilt the box from whatever config happened to be loaded rather than from the posterior.
    (The load-side checks now live in ``core.artifacts.store.ArtifactStore.load_prior/load_posterior``,
    pinned in ``tests/test_artifact_store.py``.)

    The cost of that was paid once already: a chi posterior trained on one cell's bounds could be
    loaded, validated and inferred against another's, with every reported parameter silently decoded
    through the wrong box edges, and the only way to establish what had actually happened was to
    compare GMM component counts against the prior files on disk after the fact.

    Also pinned here: the master Bounds/Cells triple that replaced the five per-cell boxes. Its whole
    point is that ONE box serves every cell, so "the two bounds files agree" and "every cell sits
    strictly inside" are invariants, not incidental facts.

Run:  pytest tests/test_artifact_consistency.py
"""
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import pytest

from core import cli, config, orchestrator, registry
from core.config import BOUNDS_PATH, CELL_PATH, VALID_LABELS, VALID_MODELS
from core.Helpers import file_manager
from core.SBI import reparam, run_guards
from core.refusals import Refusal

from tests._fixtures import _WriteFailed, _failing

_NAD = "nadrowski"
_LABELS = VALID_LABELS[VALID_MODELS.index("NADROWSKI")]
_MASTER_CELLS = ("master_spont", "master_weak", "master_entrained")


def _cfg(bounds="master.txt", **kw):
    cfg = cli.make_sim_config("NADROWSKI", _LABELS, registry.state_dep_drift("NADROWSKI"),
                              str(BOUNDS_PATH / _NAD / bounds), **kw)
    cfg.hw = config.cpu_device()
    return cfg


# ── the master Bounds/Cells triple ────────────────────────────────────────────────────────────────
def test_master_bounds_pair_share_one_nd_section():
    """master_spont.txt exists ONLY to drop f_scale and the Forcing section (mode 1 drops f_scale
    because it only ever divides a force). Its ND block must be byte-for-byte the same box, or the
    two modes are quietly inferring over different parameter spaces."""
    forced = file_manager.parse_bounds_file(str(BOUNDS_PATH / _NAD / "master.txt"))
    spont = file_manager.parse_bounds_file(str(BOUNDS_PATH / _NAD / "master_spont.txt"))
    assert list(forced[0].items()) == list(spont[0].items()), "ND sections differ"
    assert list(forced[1]) == ["x_scale", "t_scale", "f_scale"], list(forced[1])
    assert list(spont[1]) == ["x_scale", "t_scale"], list(spont[1])
    assert spont[2] == {} and forced[2], "only the forced file may declare a Forcing section"


def test_every_master_cell_is_strictly_interior_to_the_master_box():
    """A ground truth ON a bound is the hazard the archived cell_2 shipped with (tau_c = 0, temp = 0
    sat exactly on their lower edges), which is why those marginals could only ever be one-sided.
    Strictly interior, with no exceptions, is the property the master cells were built to have."""
    for cell in _MASTER_CELLS:
        cfg = _cfg()
        cli.load_and_validate_gt(cfg, str(CELL_PATH / _NAD / f"{cell}.txt"))
        for name, (val, (lo, hi)) in list(cfg.params_dict.items()) + list(cfg.rescale_params.items()):
            assert lo < val < hi, f"{cell}: {name}={val} is not strictly inside ({lo}, {hi})"


def test_every_master_cell_injects_under_every_mode():
    """One cell set, three observation modes. The spontaneous cell deliberately keeps f_scale and a
    zeroed Forcing section so chi mode -- which needs f_scale inferred -- can use it too; extras are
    ignored, never fatal."""
    seen = set()
    for bounds, chi in (("master.txt", False), ("master.txt", True), ("master_spont.txt", False)):
        for cell in _MASTER_CELLS:
            cfg = _cfg(bounds, chi_mode=chi, chi_n_freqs=4)
            problems = cli.validate_gt_file(cfg, str(CELL_PATH / _NAD / f"{cell}.txt"))
            assert not problems, f"{bounds} + {cell}: {problems}"
            cli.load_and_validate_gt(cfg, str(CELL_PATH / _NAD / f"{cell}.txt"))
            seen.add(cfg.observation_mode)
    assert seen == {"spontaneous", "forced", "chi"}, seen


def test_chi_probe_band_is_sub_resonance():
    """CHI_FREQ_BOUNDS was retargeted to the band where |chi| is actually reproducible: measured on
    the master cell, everything above ~0.25x Omega_0 has CV 0.2-0.7 at EVERY drive amplitude, and
    does not improve from T_obs 5 s to 25 s (so it is systematic, not statistical). The old
    (0.1, 10.0) put 8 of 10 probes at K=10 in that regime. This pins the band against a well-meaning
    revert to 'cover more frequency'."""
    lo, hi = config.CHI_FREQ_BOUNDS
    assert 0 < lo < hi <= 0.3, f"chi probes must stay sub-resonance, got {config.CHI_FREQ_BOUNDS}"
    assert 0 < config.CHI_F0 < 0.2, (
        f"CHI_F0={config.CHI_F0}: 0.2 is the measured entrainment onset -- at or above it the drive "
        f"captures the bundle and chi reports the drive back to itself")


# ── bounds resolution ─────────────────────────────────────────────────────────────────────────────
def test_bounds_resolution_prefers_a_sibling_then_falls_back_to_master():
    """Three cells now share one bounds file, which the same-named-sibling rule cannot express. The
    sibling still WINS where it exists, so every pre-existing cell resolves exactly as before."""
    assert cli.resolve_bounds_for_cell(str(CELL_PATH / _NAD / "master_spont.txt")).name == "master_spont.txt"
    for cell in ("master_weak", "master_entrained"):
        got = cli.resolve_bounds_for_cell(str(CELL_PATH / _NAD / f"{cell}.txt"))
        assert got.name == cli.MASTER_BOUNDS_NAME, f"{cell} -> {got}"
    # a cell that exists in neither place still resolves, because master.txt governs the folder
    assert cli.resolve_bounds_for_cell(str(CELL_PATH / _NAD / "no_such_cell.txt")).name == "master.txt"
    # ...but a model folder with no master.txt and no sibling resolves to nothing, not to a guess
    assert cli.resolve_bounds_for_cell(str(CELL_PATH / "hopf" / "no_such_cell.txt")) is None


# ── prior identity ────────────────────────────────────────────────────────────────────────────────
def _write_prior(path, lows, highs, keys, model="NADROWSKI"):
    """A minimal on-disk prior file, still used by the atomic-write test below (build_prior's load
    path validation moved to store.load_prior; this helper is just a convenient writer)."""
    d = len(lows)
    base = torch.distributions.MixtureSameFamily(
        torch.distributions.Categorical(probs=torch.ones(2)),
        torch.distributions.MultivariateNormal(torch.zeros(2, d),
                                               covariance_matrix=torch.eye(d).expand(2, d, d)))
    T = reparam.build_box_bijection(torch.tensor(lows), torch.tensor(highs),
                                    torch.zeros(d, dtype=torch.bool))
    file_manager.save_mix_dist(torch.distributions.TransformedDistribution(base, T), str(path),
                               model=model, param_keys=keys)


def test_a_prior_the_posterior_was_not_trained_with_is_refused():
    """SBC draws theta* from the TRAINING prior. Run against a different one it is not a calibration
    measurement of that posterior at all. The posterior carries its own training prior, so this needs
    no sidecar and works for a posterior trained moments ago and never saved."""
    d = 4
    def gmm(seed):
        torch.manual_seed(seed)
        return torch.distributions.MixtureSameFamily(
            torch.distributions.Categorical(probs=torch.rand(3)),
            torch.distributions.MultivariateNormal(torch.randn(3, d),
                                                   covariance_matrix=torch.eye(d).expand(3, d, d)))
    g1, g2 = gmm(1), gmm(2)
    T = reparam.build_box_bijection(torch.zeros(d), torch.ones(d), torch.zeros(d, dtype=torch.bool))
    phys1 = torch.distributions.TransformedDistribution(g1, T)
    phys2 = torch.distributions.TransformedDistribution(g2, T)

    class _Product:                      # stands in for ProductPrior
        def __init__(self, ds): self.distributions = ds

    class _Wrapper:                      # stands in for SBIPriorWrapper
        def __init__(self, g): self.gen_dist = g

    class _Post:
        def __init__(self, p): self.prior = p

    post = _Post(_Wrapper(_Product([g1])))
    # the fingerprint must see the SAME gmm through every wrapper shape it can arrive in
    assert len({orchestrator._gmm_fingerprint(x)
                for x in (g1, phys1, _Product([phys1]), _Wrapper(_Product([g1])))}) == 1
    orchestrator._assert_prior_used_matches_posterior(post, _Product([phys1]), "t")   # must not raise
    try:
        orchestrator._assert_prior_used_matches_posterior(post, _Product([phys2]), "t")
        raise AssertionError("a foreign prior was accepted")
    except Refusal as e:
        assert e.field == "prior", e.field
    # unverifiable on either side => silence, not a false alarm (legacy artifacts land here)
    orchestrator._assert_prior_used_matches_posterior(_Post(None), _Product([phys1]), "t")


# ── the end-of-run artifact writes are atomic ─────────────────────────────────────────────────────
def test_a_torn_prior_write_leaves_the_previous_prior_intact(tmp_path):
    """A prior is not just a file: it is what the training checkpoint's identity fingerprints and what
    SBC draws theta* from. Half-replacing one does not produce a broken run, it produces a run that
    resumes against a distribution nobody can name (2026-08-12: prior_fingerprint is in the checkpoint
    identity for exactly this reason)."""
    path = tmp_path / "_ptest_atomic.pt"
    real_save = torch.save
    try:
        _write_prior(path, [0.0, 0.0], [1.0, 1.0], ["a", "b"])
        first = file_manager.read_prior_metadata(str(path))["param_keys"]
        assert first == ["a", "b"], first

        torch.save = _failing(real_save)
        try:
            _write_prior(path, [0.0, 0.0], [1.0, 1.0], ["c", "d"])
            raise AssertionError("the injected failure did not propagate")
        except _WriteFailed:
            pass
        finally:
            torch.save = real_save

        assert file_manager.read_prior_metadata(str(path))["param_keys"] == ["a", "b"], \
            "a torn write clobbered the prior it was replacing"
        assert not (tmp_path / "_ptest_atomic.pt.tmp").exists(), "a failed write left its temp behind"
    finally:
        torch.save = real_save


def test_atomic_savez_round_trips_and_cannot_be_torn(tmp_path):
    """The .loss.npz is a zip, so a truncated one raises BadZipFile rather than reading short -- and it
    is the file scripts/retrain_convergence.py at 7433ced^ read back for its convergence verdict.

    The round-trip half is load-bearing on its own: np.savez appends '.npz' when handed a NAME but not
    when handed a HANDLE, which is the difference between landing on <name>.loss.npz and on
    <name>.loss.npz.tmp.npz."""
    import numpy as np
    path = tmp_path / "_ptest_atomic.npz"
    real_savez = np.savez
    try:
        file_manager.atomic_savez(path, dict(validation_loss=np.arange(3.0), epochs_trained=7))
        assert path.exists(), f"nothing landed at {path} -- np.savez rewrote the name"
        with np.load(str(path)) as z:
            assert list(z["validation_loss"]) == [0.0, 1.0, 2.0] and int(z["epochs_trained"]) == 7

        np.savez = _failing(real_savez)
        try:
            file_manager.atomic_savez(path, dict(validation_loss=np.arange(99.0), epochs_trained=99))
            raise AssertionError("the injected failure did not propagate")
        except _WriteFailed:
            pass
        finally:
            np.savez = real_savez

        with np.load(str(path)) as z:
            assert int(z["epochs_trained"]) == 7, "a torn write clobbered the previous curve"
        assert not (tmp_path / "_ptest_atomic.npz.tmp").exists(), "a failed write left its temp behind"
    finally:
        np.savez = real_savez


# ── the chi band/drive preflight (2026-08-19 regression) ─────────────────────────────────────────
def test_a_chi_run_at_a_non_default_band_is_refused_before_the_simulation_spend(monkeypatch):
    """A ~5-day retrain was spent at the RETIRED band (0.1, 10.0) because QSettings restored a value
    saved before C-5 changed it. ``store.load_posterior`` catches that disagreement only when a
    posterior is LOADED, i.e. after the days are gone.

    The subtle half is the LOAD path: it compares the posterior against cfg, so a stale cfg loading
    the posterior trained under that same stale cfg agrees with itself and stays silent. This guard
    compares against config.py, the one party that cannot go stale.

    Scope matters as much as existence: chi_n_freqs must NOT be an error. It is the count an
    OBSERVATION supplies, training draws its own K per batch, and failing on it would refuse a
    perfectly good 7-recording experiment.
    """
    cfg = _cfg(chi_mode=True, chi_n_freqs=4)

    orchestrator._assert_chi_config_is_deliberate(cfg)          # at the defaults: must not raise

    for field, bad in (("chi_freq_bounds", (0.1, 10.0)), ("chi_f0", cfg.chi_f0 * 2)):
        stale = _cfg(chi_mode=True, chi_n_freqs=4)
        setattr(stale, field, bad)
        try:
            orchestrator._assert_chi_config_is_deliberate(stale)
            raise AssertionError(f"a chi run with a non-default {field} was accepted")
        except Refusal as e:
            assert field in str(e), f"the message must name {field}, got: {e}"
            assert e.field is None, "two knobs and a science file; no single control answers it"
            # After V5 no front end can produce the mismatch: the Config tab shows config.py's band
            # and drive read-only, and the keys are neither written nor read. Sending an operator to
            # PRISM.ini or the Config tab would send them to a cause that no longer exists.
            for banned in ("PRISM.ini", "QSettings", "Config tab"):
                assert banned not in str(e), f"the refusal still points at '{banned}': {e}"
            assert "config.py" in str(e), f"the refusal must say what to do instead, got: {e}"

    # K alone is legitimate -- one posterior serves any probe count (build_posterior omits it from
    # training_params on purpose). Refusing it would break the K-agnosticism the set encoder buys.
    k_only = _cfg(chi_mode=True, chi_n_freqs=4)
    k_only.chi_n_freqs = int(config.CHI_N_FREQS) + 3
    orchestrator._assert_chi_config_is_deliberate(k_only)       # must not raise

    # A non-chi run is never gated by a chi knob.
    spont = _cfg("master_spont.txt")
    spont.chi_freq_bounds = (0.1, 10.0)
    orchestrator._assert_chi_config_is_deliberate(spont)        # must not raise

    # THERE IS NO ESCAPE HATCH ANY MORE (D11). PRISM_CHI_OVERRIDE=1 used to let a deliberate band
    # sweep through; its one sanctioned caller is archived, every knob now travels as an argument,
    # and a non-default band means editing config.py deliberately. The refusal must not advertise a
    # variable that does nothing -- an operator who sets it and sees the run proceed learns the wrong
    # lesson, and an operator who sets it and sees it ignored deserves to be told why.
    monkeypatch.setenv("PRISM_CHI_OVERRIDE", "1")
    override = _cfg(chi_mode=True, chi_n_freqs=4)
    override.chi_freq_bounds = (0.1, 10.0)
    try:
        orchestrator._assert_chi_config_is_deliberate(override)
        raise AssertionError("PRISM_CHI_OVERRIDE=1 still let a non-default band through")
    except ValueError as e:
        assert "PRISM_CHI_OVERRIDE" not in str(e), \
            f"the refusal still advertises a hatch that no longer exists: {e}"
        assert "config.py" in str(e), f"the refusal must say what to do instead, got: {e}"
    assert not hasattr(run_guards, "CHI_OVERRIDE_ENV"), "run_guards still defines the override name"
    assert not hasattr(orchestrator, "CHI_OVERRIDE_ENV"), "orchestrator still re-exports the override name"


def test_config_build_refusals_name_their_field():
    """V3 at the config build, the first thing every subcommand and the Prior tab do. Each refusal a
    bad input can provoke there is a Refusal carrying the key of the ONE control that answers it --
    the chi knob, the cell, the units -- or None for the exactly-one-of check, which is a caller's
    mistake with no control behind it. The texts are unchanged (the pins on them elsewhere hold);
    what changes is the class and the key, so T10's worker can route each to the yellow box and the
    tool can append the flag from its own table instead of the message naming one."""
    import dataclasses
    cfg = _cfg(chi_mode=True, chi_n_freqs=4)
    for kw, key in ((dict(chi_k_pad=1), "chi_k_pad"),
                    (dict(chi_n_freqs=cfg.chi_k_pad + 1), "chi_n_freqs"),
                    (dict(chi_max_cycles=config.CHI_MIN_CYCLES), "chi_max_cycles")):
        with pytest.raises(Refusal) as e:
            dataclasses.replace(cfg, **kw)                       # re-runs __post_init__
        assert e.value.field == key, (kw, str(e.value))
    # the cell's CONTENT: a bounds-built config handed a cell with nothing in it
    with pytest.raises(Refusal, match="missing") as e:
        _cfg().inject_ground_truth({}, {}, {}, {})
    assert e.value.field == "cell"
    with pytest.raises(Refusal, match="exactly one of") as e:
        cli.make_sim_config("NADROWSKI", _LABELS, registry.state_dep_drift("NADROWSKI"))
    assert e.value.field is None
    # the bounds FILE itself (§3.3): a path that names no file is refused by the input kind at the
    # build, never the parser's bare "File not found" from inside file_manager
    with pytest.raises(Refusal, match="The bounds file was not found") as e:
        _cfg("no_such_bounds.txt")
    assert e.value.field == "bounds"
    with pytest.raises(Refusal, match="time unit") as e:
        cli.units_to_factors(("nm", "pN"))                       # no time unit among them
    assert e.value.field == "units"
    with pytest.raises(Refusal, match="units file") as e:
        cli.resolve_units_file("no_such_model_for_this_test")
    assert e.value.field == "units"
