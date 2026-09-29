"""The simulation cache's identity: every config field a checkpoint of training rows must agree
with before it can be resumed. Successor of the training-rows/1 identity, whose directories were
deleted; ``orchestrator.training_identity`` survives as a delegate to this class.

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

``units_sha256`` is the units declaration's fingerprint, and it is what moved the format from
training-rows/2 to training-rows/3. On a box that declares temperature in place of the force scale,
every simulation is driven at a force scale derived through Boltzmann's constant in the cell's
units, whose force and length factors nothing else in the identity carries (only the time factor
reaches it, through the experiment's time grid), so the same bounds read under other units simulate
other rows. The declaration comes from one of two places, fingerprinted differently:

  * a units FILE, named by ``cfg.sources["units"]``: the sha256 of its bytes with CRLF read as LF, so
    a checkout that rewrites line endings does not re-key a cache. The file, never
    ``cfg.units_dict``, because that tuple is parsed through a set and its order is not stable
    between processes.
  * unit TOKENS typed in place of a file (the window's typed units, or a token override given to
    ``make_sim_config``), which name no file: the sha256 of the UTF-8 text ``"units-tokens:"``
    followed by the distinct tokens, sorted and joined with newlines. Sorted, because the order
    carries no meaning; prefixed, so it is never the fingerprint of a usable units file: a file
    holding exactly that text has no ``# Units`` section and so declares no units at all.

A named file that cannot be read, and a config with neither a file nor tokens (a stand-in), fail open
to None, as the GUI's pre-Train status line computes identities from stand-ins. A cache written
under the older format keys a different directory, so it is never found for a resume, and a
directory moved into place by hand is refused field by field (``training_checkpoint.verify``).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

FORMAT = "training-rows/3"


_TOKENS_PREFIX = "units-tokens:"


def _units_sha256(cfg) -> str | None:
    """The units declaration's fingerprint, as the module docstring describes: sha256 hex of the units
    file ``cfg.sources["units"]`` names, over its bytes with CRLF replaced by LF, or None when that
    file cannot be read; with no file named, sha256 hex of ``"units-tokens:"`` plus the sorted
    distinct tokens of ``cfg.units_dict`` joined by newlines; None when there are no tokens either."""
    path = (getattr(cfg, "sources", None) or {}).get("units")
    if path:
        try:
            data = Path(path).read_bytes()
        except OSError:
            return None
        return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()
    tokens = getattr(cfg, "units_dict", None)
    if not isinstance(tokens, (list, tuple)) or not tokens or not all(isinstance(u, str) for u in tokens):
        return None
    text = _TOKENS_PREFIX + "\n".join(sorted(set(tokens)))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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
            "units_sha256": _units_sha256(cfg),
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
