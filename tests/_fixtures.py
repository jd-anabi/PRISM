"""Shared test helpers for the artifact-store suites (piece 1 of the 2026-09-10 hardening programme).

A plain module -- importing it has NO side effects (no simulation, no file I/O, no torch seeding);
everything here is a function or class definition, imported lazily inside each helper exactly as it
was before the move so the import graph a bare ``import tests._fixtures`` pulls in stays light.
Moved out of ``tests/test_artifact_store.py`` (Task 7) so ``tests/test_user_sbi.py``,
``tests/test_nav_and_gating.py`` and others can share them without importing a whole other test
module (which pytest would then also collect a second time under a different name).
"""
import torch

from core.artifacts import manifest as mf


def _nad_cfg(**over):
    """A real NADROWSKI SimConfig off the master bounds file, CPU device."""
    from core import cli, registry
    from core.config import BOUNDS_PATH, VALID_LABELS, VALID_MODELS
    from core import config
    labels = VALID_LABELS[VALID_MODELS.index("NADROWSKI")]
    cfg = cli.make_sim_config("NADROWSKI", labels, registry.state_dep_drift("NADROWSKI"),
                              str(BOUNDS_PATH / "nadrowski" / "master.txt"), **over)
    cfg.hw = config.cpu_device()
    return cfg


def _gmm_in_box(lows, highs, mask, seed=0):
    """A tiny 2-component MixtureSameFamily pushed through a box bijection -- a real GMM prior payload,
    small enough to build and pickle in a test."""
    from core.SBI import reparam
    torch.manual_seed(seed)
    d = len(lows)
    base = torch.distributions.MixtureSameFamily(
        torch.distributions.Categorical(probs=torch.tensor([0.5, 0.5])),
        torch.distributions.MultivariateNormal(torch.randn(2, d), covariance_matrix=torch.eye(d).expand(2, d, d)))
    T = reparam.build_box_bijection(torch.tensor(lows, dtype=torch.float32), torch.tensor(highs, dtype=torch.float32), mask)
    return torch.distributions.TransformedDistribution(base, T)


def _prior_artifact(store, cfg, *, name="p", lows=None, highs=None, keys=None, model=None, seed=0):
    """A prior artifact with a real 2-component GMM payload, laid out exactly as build_prior writes it."""
    from core.Helpers import file_manager
    from core.SBI.reparam import nd_log_mask
    from core.SBI.run_guards import _gmm_fingerprint, _log_params_for
    keys = keys or list(cfg.params_dict)
    lows = lows or [b[0] for _, b in cfg.params_dict.values()]
    highs = highs or [b[1] for _, b in cfg.params_dict.values()]
    mask = nd_log_mask(cfg, log_params=_log_params_for(cfg))
    dist = _gmm_in_box(lows, highs, mask, seed)
    with store.create("prior", cfg, name=name) as w:
        file_manager.save_mix_dist(dist, str(w.payload("prior.pt")), model=model or cfg.model, param_keys=keys)
        if model:
            w.config["model"] = model
        w.fingerprints["gmm"] = _gmm_fingerprint(dist)
        w.body = {"gmm": {"n_components": 2, "param_keys": list(keys),
                          "box": {"nd_lows": [float(v) for v in lows], "nd_highs": [float(v) for v in highs],
                                  "log_mask": [bool(v) for v in mask.tolist()]}},
                  "sweep": {}, "stability": {"accepted_sets": None, "iterations": 1}}
    return w


from sbi.inference import DirectPosterior


class _FakeDP(DirectPosterior):
    """A DirectPosterior by type only (the loader's isinstance accepts it); module-level so it pickles."""
    def __init__(self):
        pass


def _set_path(w, path, value):
    target = w.config if path[0] == "config" else w.body
    for key in path[1 if path[0] == "config" else 0:-1]:
        target = target[key]
    target[path[-1]] = value


def _posterior_artifact(store, cfg, *, name="post", amortized=True, region=None, V=None, prior=None, over=None):
    """A posterior artifact with a _FakeDP payload, laid out exactly as build_posterior writes it."""
    keys = list(cfg.params_dict) + list(cfg.rescale_params)
    with store.create("posterior", cfg, name=name) as w:
        dp = _FakeDP()
        dp.prior = prior
        torch.save(dp, str(w.payload("posterior.pt")))
        w.parents = {"prior": "20260910T100000"}
        w.body = {
            "mode": cfg.observation_mode, "conditioning": mf.conditioning_block(cfg),
            "transform": {"param_keys": keys, "log_params": [],
                          "nd_lows": [float(b[0]) for _, b in cfg.params_dict.values()],
                          "nd_highs": [float(b[1]) for _, b in cfg.params_dict.values()],
                          "rescale_lows": [float(b[0]) for _, b in cfg.rescale_params.values()],
                          "rescale_highs": [float(b[1]) for _, b in cfg.rescale_params.values()],
                          "V": mf.tensor_to_json(V), "V_orientation": "columns",
                          "fisher_eigenvalues": None, "V_digest": mf.tensor_digest(V)},
            "amortized": amortized, "truncation": None if region is None else mf.region_to_json(region),
            "training": {}}
        for path, value in (over or {}).items():
            _set_path(w, path, value)
    return w


class _LoadedStub:
    """The LoadedPrior shape with no GMM: fails open through every fingerprint check."""
    id, name, fingerprint, force_prior = None, "", None, None
    def __init__(self, prior=None): self.prior = prior if prior is not None else object()


class _WriteFailed(RuntimeError):
    """Injected mid-write failure. Not OSError, so a handler that swallows disk errors cannot hide it."""


def _failing(real):
    """Wrap a serializer so it writes its bytes and THEN fails -- the tear that atomicity must absorb.

    Failing before writing anything would pass against a plain `torch.save` too: the destination is
    only clobbered once the writer has begun. The bytes have to land first for the test to mean
    anything.

    Signature-agnostic (`*a, **k`) because the two serialisers order their arguments differently --
    ``torch.save(obj, file)`` against ``np.savez(file, **arrays)``.
    """
    def _boom(*a, **k):
        real(*a, **k)
        raise _WriteFailed("disk full")
    return _boom
