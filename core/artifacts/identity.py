"""The simulation cache's identity: every config field a checkpoint of training rows must agree
with before it can be resumed. Successor of ``orchestrator.training_identity`` (training-rows/1),
which the clean break retires along with every directory it named.

``truncation`` is ALWAYS present (None for an amortized run) -- the old dict omitted the key so that
existing digests would not move; there are no existing digests any more. ``feature_set_version``
is new: the old identity digested only the valid-flag LABELS, so a changed feature definition behind
an unchanged label re-keyed nothing and a cache could be resumed onto rows that meant something else.

Everything in the identity is known BEFORE the Fisher rotation runs, and it has to be: the digest
names the directory the rotation's own V is stored in (``header.pt``), so including V would make
the naming circular. A resumed run reuses the stored V precisely because V is not reproducible
across processes (its operating points come from the unseeded global RNG), so the identity must be
computable without it. The region a TSNPE round trains under DOES enter (``truncation``): it
carries the PARENT's V, copied and never recomputed, so its digest is stable by construction.
"""
from __future__ import annotations

from dataclasses import dataclass

FORMAT = "training-rows/2"


@dataclass(frozen=True)
class SimulationIdentity:
    values: dict

    @classmethod
    def from_cfg(cls, cfg, prior, run_size: int, n_runs: int, *, truncation=None) -> "SimulationIdentity":
        """``prior`` is a LoadedPrior (its ``.fingerprint`` is used), a bare prior distribution (its GMM
        is fingerprinted), or anything else (None -- the GUI's pre-Train status line passes stubs and
        must fail open)."""
        from core import config as _config
        from core.SBI.reparam import resolved_log_params
        from core.SBI.run_guards import _gmm_fingerprint, _log_params_for
        from core.SBI.statistics import FEATURE_SET_VERSION, VALID_FLAG_LABELS
        fp = getattr(prior, "fingerprint", None)
        if fp is None:
            fp = _gmm_fingerprint(prior)
        chi = bool(cfg.chi_mode)
        vals = {
            "format": FORMAT,
            "model": cfg.model,
            "prior_fingerprint": fp,
            "mode": cfg.observation_mode,
            "param_keys": list(cfg.params_dict) + list(cfg.rescale_params),
            "nd_lows": [b[0] for _, b in cfg.params_dict.values()],
            "nd_highs": [b[1] for _, b in cfg.params_dict.values()],
            "rescale_lows": [b[0] for _, b in cfg.rescale_params.values()],
            "rescale_highs": [b[1] for _, b in cfg.rescale_params.values()],
            "log_params": resolved_log_params(cfg, log_params=_log_params_for(cfg)),
            "reparam_rotate": bool(cfg.reparam_rotate),
            "run_size": int(run_size),
            "n_runs": int(n_runs),
            "steady_idx": int(cfg.steady_idx),
            "dt_nd_min": float(cfg.dt_nd_min),
            "dt_exp": float(cfg.dt_exp),
            "t_min_exp": float(cfg.t_min_exp),
            "t_max_exp": float(cfg.t_max_exp),
            "t_scale_bounds": list(cfg.t_scale_bounds),
            "n_grid": int(cfg.t.shape[0]),
            "spontaneous_only": not cfg.has_forcing,
            "summary_flags": list(VALID_FLAG_LABELS),
            "feature_set_version": int(FEATURE_SET_VERSION),
            "chi_mode": chi,
            "chi_layout": _config.CHI_LAYOUT if chi else None,
            "chi_k_pad": cfg.chi_k_pad if chi else None,
            "chi_elem_w": _config.CHI_ELEM_W if chi else None,
            "chi_f0": cfg.chi_f0 if chi else None,
            "chi_freq_bounds": list(cfg.chi_freq_bounds) if chi else None,
            "chi_max_cycles": float(cfg.chi_max_cycles) if chi else None,
            "device": cfg.hw.device.type,
            "dtype": str(cfg.hw.dtype),
            "truncation": None if truncation is None else truncation.identity_fields(),
        }
        return cls(vals)

    def to_dict(self) -> dict:
        return dict(self.values)

    @property
    def digest(self) -> str:
        from core.SBI.training_checkpoint import identity_digest
        return identity_digest(self.values)

    @property
    def truncated(self) -> bool:
        return self.values.get("truncation") is not None
