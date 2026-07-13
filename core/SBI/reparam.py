"""
Reparameterization layer for bounded posterior inference.

CONVENTION:
  - T.__call__(z):     latent z ∈ ℝ^d  ->  physical θ ∈ Π_i (lo_i, hi_i)
  - T.inv.__call__(θ): physical θ      ->  latent z
  - Names "*_prior" refer to PHYSICAL-space priors unless prefixed with "latent_".
  - Flow internals operate on latent z exclusively.
  - Only gen_prior.construct_prior and TransformedPosterior.sample apply T forward.
    Everywhere else that touches T uses T.inv.
"""
from __future__ import annotations

import torch
from torch.distributions import constraints
from torch.distributions.transforms import (
    AffineTransform, SigmoidTransform, ComposeTransform, Transform,
)

class UnitToBoxTransform(Transform):
    """
    Elementwise map from the unit interval (the SigmoidTransform output s ∈ (0,1)) into the
    per-parameter box (lo_i, hi_i). Replaces the plain AffineTransform so individual params can
    be placed in LOG (geometric) coordinates:

        linear param:  θ_i = lo_i + s_i (hi_i - lo_i)
        log param:     θ_i = exp(log lo_i + s_i (log hi_i - log lo_i)) = lo_i (hi_i/lo_i)^{s_i}

    Putting the multiplicative params (kappa·x_scale amplitude, lambda·t_scale timescale) in log
    space turns those *product* degeneracies into *additive* ones in the flow's coordinate, so a
    single global Fisher rotation can decorrelate them across the whole prior, not just at GT.
    Log dims require lo_i > 0. log_mask all-False reproduces the legacy linear box exactly.

    Elementwise (event_dim 0): log_abs_det_jacobian returns a tensor shaped like the input;
    callers reduce over the last dim for a per-sample scalar (matches the old AffineTransform).
    """
    domain = constraints.real
    codomain = constraints.real
    bijective = True

    def __init__(self, lows: torch.Tensor, highs: torch.Tensor, log_mask: torch.Tensor, cache_size: int = 0):
        super().__init__(cache_size=cache_size)
        self.lows = lows
        self.highs = highs
        self.log_mask = log_mask
        tiny = torch.finfo(lows.dtype).tiny
        # log-bounds are only used on masked (lo>0) dims; zero elsewhere keeps the unused branch finite.
        self.loglo = torch.where(log_mask, torch.log(lows.clamp_min(tiny)), torch.zeros_like(lows))
        self.loghi = torch.where(log_mask, torch.log(highs.clamp_min(tiny)), torch.zeros_like(highs))

    def __eq__(self, other):
        return (isinstance(other, UnitToBoxTransform)
                and torch.equal(self.lows, other.lows) and torch.equal(self.highs, other.highs)
                and torch.equal(self.log_mask, other.log_mask))

    def _call(self, s):                                     # s ∈ (0,1) -> θ ∈ (lo,hi)
        lin = self.lows + s * (self.highs - self.lows)
        log = torch.exp(self.loglo + s * (self.loghi - self.loglo))
        return torch.where(self.log_mask, log, lin)

    def _inverse(self, theta):                              # θ -> s ∈ (0,1)
        tiny = torch.finfo(self.lows.dtype).tiny
        lin = (theta - self.lows) / (self.highs - self.lows)
        log = (torch.log(theta.clamp_min(tiny)) - self.loglo) / (self.loghi - self.loglo)
        return torch.where(self.log_mask, log, lin)

    def log_abs_det_jacobian(self, s, theta):               # |dθ/ds|: linear (hi-lo); log θ·(loghi-loglo)
        span_lin = (self.highs - self.lows)
        span_log = (self.loghi - self.loglo)
        jac = torch.where(self.log_mask, theta.abs() * span_log, span_lin.expand_as(theta))
        return torch.log(jac)


def build_box_bijection(lows: torch.Tensor, highs: torch.Tensor,
                        log_mask: torch.Tensor | None = None) -> ComposeTransform:
    """
    Per-parameter box bijection from ℝ^d to prod_i (lo_i, hi_i), elementwise.
      linear dims: z_i -> lo_i + sigmoid(z_i) (hi_i - lo_i)
      log dims (log_mask True, lo_i > 0): z_i -> lo_i (hi_i/lo_i)^{sigmoid(z_i)}  (geometric)

    log_mask=None (or all-False) reproduces the legacy linear box exactly. Both log_abs_det terms
    (sigmoid + box) are elementwise; callers reduce over the last dim for a per-sample scalar.
    """
    if lows.shape != highs.shape:
        raise ValueError(f"lows and highs must share shape, got {lows.shape} vs {highs.shape}")
    if (highs <= lows).any():
        raise ValueError("highs must be strictly > lows per-element")
    if log_mask is None:
        log_mask = torch.zeros_like(lows, dtype=torch.bool)
    if bool((log_mask & (lows <= 0)).any()):
        raise ValueError("log-space box requires strictly positive lower bounds on log dims")
    return ComposeTransform([
        SigmoidTransform(),                                 # ℝ -> (0, 1)
        UnitToBoxTransform(lows, highs, log_mask),          # (0, 1) -> (lo, hi), per-dim linear/log
    ])


def _resolve_log_params(log_params) -> set:
    """The set of param NAMES to place in log space: the explicit override, or config default."""
    if log_params is None:
        from core.config import REPARAM_LOG_PARAMS
        log_params = REPARAM_LOG_PARAMS
    return set(log_params or [])


def _log_mask(names: list[str], lows: torch.Tensor, log_params) -> torch.Tensor:
    """
    Bool mask (len == #params) True where a param is requested log-space AND has lo > 0.
    Requesting log on a non-positive-lower-bound param is downgraded to linear with a warning.
    """
    want = _resolve_log_params(log_params)
    requested = torch.tensor([n in want for n in names], dtype=torch.bool, device=lows.device)
    valid = lows > 0
    if bool((requested & ~valid).any()):
        import warnings
        bad = [n for n, r, v in zip(names, requested.tolist(), valid.tolist()) if r and not v]
        warnings.warn(f"log-space requested for non-positive-lower-bound params {bad}; using linear there.")
    return requested & valid


def resolved_log_params(cfg, log_params=None) -> list[str]:
    """The param names that build_inferred_bijection will actually place in log space (lo>0)."""
    want = _resolve_log_params(log_params)
    out = []
    for d in (cfg.params_dict, cfg.rescale_params):
        for n, (_, b) in d.items():
            if n in want and b[0] > 0:
                out.append(n)
    return out


def nd_log_mask(cfg, log_params=None) -> torch.Tensor:
    """Per-ND-param log mask (on cfg.hw.device), aligned with cfg.params_dict order. For gen_prior."""
    names = list(cfg.params_dict.keys())
    lows = torch.tensor([b[0] for _, b in cfg.params_dict.values()], dtype=cfg.hw.dtype, device=cfg.hw.device)
    return _log_mask(names, lows, log_params)


def build_nd_bijection(cfg, log_params=None) -> ComposeTransform:
    """Bijection for the ND params only (used inside gen_prior). log_params=None => config default."""
    names = list(cfg.params_dict.keys())
    lows  = torch.tensor([b[0] for _, b in cfg.params_dict.values()],
                         dtype=cfg.hw.dtype, device=cfg.hw.device)
    highs = torch.tensor([b[1] for _, b in cfg.params_dict.values()],
                         dtype=cfg.hw.dtype, device=cfg.hw.device)
    return build_box_bijection(lows, highs, _log_mask(names, lows, log_params))


def build_rescale_bijection(cfg, log_params=None) -> ComposeTransform:
    """Bijection for the rescale params only (used to build the latent rescale prior)."""
    names = list(cfg.rescale_params.keys())
    lows  = torch.tensor([b[0] for _, b in cfg.rescale_params.values()],
                         dtype=cfg.hw.dtype, device="cpu")
    highs = torch.tensor([b[1] for _, b in cfg.rescale_params.values()],
                         dtype=cfg.hw.dtype, device="cpu")
    return build_box_bijection(lows, highs, _log_mask(names, lows, log_params))


def build_inferred_bijection(cfg, log_params=None) -> ComposeTransform:
    """
    Bijection for the full inferred parameter space (ND + rescale), in the same order
    ProductPrior stacks them: [nd_params, rescale_params]. Used inside gen_training_data
    and to wrap the trained DirectPosterior. log_params=None => config REPARAM_LOG_PARAMS;
    pass an explicit list (e.g. the value saved beside a posterior) to reproduce a past box.
    """
    names = list(cfg.params_dict.keys()) + list(cfg.rescale_params.keys())
    nd_lows  = torch.tensor([b[0] for _, b in cfg.params_dict.values()],
                            dtype=cfg.hw.dtype, device=cfg.hw.device)
    nd_highs = torch.tensor([b[1] for _, b in cfg.params_dict.values()],
                            dtype=cfg.hw.dtype, device=cfg.hw.device)
    rs_lows  = torch.tensor([b[0] for _, b in cfg.rescale_params.values()],
                            dtype=cfg.hw.dtype, device=cfg.hw.device)
    rs_highs = torch.tensor([b[1] for _, b in cfg.rescale_params.values()],
                            dtype=cfg.hw.dtype, device=cfg.hw.device)
    lows = torch.cat([nd_lows, rs_lows])
    return build_box_bijection(lows, torch.cat([nd_highs, rs_highs]), _log_mask(names, lows, log_params))


def _transform_device(transform: ComposeTransform) -> torch.device:
    """Extract the device the transform's tensors live on by peeking at the box's bound tensors."""
    for t in transform.parts:
        if isinstance(t, UnitToBoxTransform):
            return t.lows.device
        if isinstance(t, AffineTransform):
            return t.loc.device
    return torch.device("cpu")


class TransformedPosterior:
    """
    Adapter that turns a latent-space DirectPosterior into a physical-space one
    via a bijection T: z -> θ_phys. Preserves the x-conditional API of DirectPosterior.
    """
    def __init__(self, latent_posterior, transform: ComposeTransform):
        self.latent = latent_posterior
        self.T = transform

    def sample(self, sample_shape, x=None, **kwargs):
        z = self.latent.sample(sample_shape, x=x, **kwargs)
        return self.T(z)

    def sample_batched(self, sample_shape, x, **kwargs):
        z = self.latent.sample_batched(sample_shape, x=x, **kwargs)
        return self.T(z)

    def log_prob(self, theta_phys, x=None, **kwargs):
        z = self.T.inv(theta_phys)
        latent_lp = self.latent.log_prob(z, x=x, **kwargs)
        log_det = self.T.log_abs_det_jacobian(z, theta_phys)
        # Elementwise T: log_det has shape (..., d). Collapse rightmost dims until it matches latent_lp.
        while log_det.dim() > latent_lp.dim():
            log_det = log_det.sum(dim=-1)
        return latent_lp - log_det

    def log_prob_batched(self, theta_phys, x, **kwargs):
        z = self.T.inv(theta_phys)
        latent_lp = self.latent.log_prob_batched(z, x=x, **kwargs)
        log_det = self.T.log_abs_det_jacobian(z, theta_phys)
        while log_det.dim() > latent_lp.dim():
            log_det = log_det.sum(dim=-1)
        return latent_lp - log_det

    def set_default_x(self, x):
        self.latent.set_default_x(x)
        return self


# ── Decorrelating reparameterization (optional Fisher-eigenbasis rotation) ────────
# The flow struggles to calibrate the near-degenerate (e.g. 0.95-correlated kappa~x_scale)
# posterior. Rotating the flow's latent coordinate into the Fisher eigenbasis makes that
# posterior axis-aligned, so the flow calibrates it -> tighter, correct marginals, with NO
# loss of information (the rotation is orthogonal/invertible) and NO model change. V = I
# recovers the exact current pipeline, so the rotation is fully optional.
class OrthogonalTransform(Transform):
    """
    Fixed orthogonal rotation in the flow's latent space (row-vector convention):
        forward:  w -> w @ M      inverse:  z -> z @ M^T
    Orthogonal M => volume-preserving => log|det J| = 0.
    """
    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True

    def __init__(self, M: torch.Tensor, cache_size: int = 0):
        super().__init__(cache_size=cache_size)
        self.M = M

    def __eq__(self, other):
        return (isinstance(other, OrthogonalTransform) and self.M.shape == other.M.shape
                and bool(torch.equal(self.M, other.M)))

    def _call(self, x):
        return x @ self.M

    def _inverse(self, y):
        return y @ self.M.transpose(-1, -2)

    def log_abs_det_jacobian(self, x, y):
        return torch.zeros(x.shape[:-1], dtype=x.dtype, device=x.device)


def fisher_eigenbasis(F: torch.Tensor) -> torch.Tensor:
    """
    Eigenvectors V (columns) of a symmetric Fisher matrix F, descending by eigenvalue.
    The map w = V^T z sends the (Laplace) posterior covariance F^{-1} to a diagonal in w,
    i.e. decorrelates the flow's coordinates. F is computed from SIMULATIONS (the standardized
    feature-Jacobian J: F = J^T J) — no trained posterior is needed.
    """
    F = 0.5 * (F + F.transpose(-1, -2))
    evals, evecs = torch.linalg.eigh(F)
    return evecs[:, torch.argsort(evals, descending=True)]


def build_rotated_bijection(box_transform: ComposeTransform, V: torch.Tensor) -> ComposeTransform:
    """
    Compose the decorrelating rotation with the per-parameter box bijection:
        T_new(w) = box_transform(w @ V^T)      # flow coord w -> physical θ
    so the flow operates on the rotated coordinate w = z @ V, z = box_transform.inv(θ).
    V orthogonal => the rotation adds 0 to the log-det. V = I recovers box_transform.
    """
    return ComposeTransform([OrthogonalTransform(V.transpose(-1, -2))] + list(box_transform.parts))


def load_eval_bijection(cfg, choice: str, posterior_dir) -> ComposeTransform:
    """
    Reconstruct the EXACT physical-space bijection (log box + optional rotation) for a SAVED
    posterior, self-describing from its sidecar so eval never depends on the current config
    matching the training config. Single source of truth shared by the orchestrator load path
    and the offline diagnostic scripts — keep in sync with build_posterior's save side.

    Sidecar '<name>.rot.pt' (written by build_posterior):
      - dict {"V": tensor|None, "log_params": [names]}   (current format)
      - bare tensor V                                     (legacy: rotation only, linear box)
    No sidecar => posterior predates the reparam work => plain LINEAR box (backward compatible).

    :param cfg:           SimConfig (provides param bounds/order to rebuild the box).
    :param choice:        posterior filename (e.g. "posterior_new.pt").
    :param posterior_dir: directory holding the posterior + its .rot.pt sidecar (a Path).
    """
    base = choice[:-3] if choice.endswith(".pt") else choice
    rot_path = posterior_dir / (base + ".rot.pt")
    if not rot_path.exists():
        return build_inferred_bijection(cfg, log_params=[])        # legacy: linear box, no rotation
    obj = torch.load(str(rot_path), weights_only=False)
    if isinstance(obj, dict):
        V, log_params = obj.get("V", None), obj.get("log_params", [])
    else:                                                          # legacy bare-tensor V (linear box)
        V, log_params = obj, []
    T = build_inferred_bijection(cfg, log_params=log_params)
    return build_rotated_bijection(T, V) if V is not None else T


class RotatedLatentPrior:
    """
    Prior over the rotated flow coordinate w = z @ V, with z ~ base (the latent inferred
    prior) and V orthogonal. sample()/log_prob() only — the minimal interface the SBI training
    path and SBIPriorWrapper need. log|det V| = 0, so no Jacobian correction.
    """
    def __init__(self, base, V: torch.Tensor):
        self.base = base
        self.V = V

    @property
    def batch_shape(self):
        return self.base.batch_shape

    @property
    def event_shape(self):
        return self.base.event_shape

    def sample(self, sample_shape=torch.Size()):
        return self.base.sample(sample_shape) @ self.V

    def log_prob(self, w):
        return self.base.log_prob(w @ self.V.transpose(-1, -2))