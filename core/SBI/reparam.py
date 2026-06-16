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

def build_box_bijection(lows: torch.Tensor, highs: torch.Tensor) -> ComposeTransform:
    """
    Per-parameter scaled sigmoid: z_i -> lo_i + sigmoid(z_i) * (hi_i - lo_i).
    Bijection from ℝ^d to prod_i (lo_i, hi_i), elementwise.

    Both log_abs_det_jacobian terms (sigmoid + affine) are elementwise and returned
    as tensors of the same shape as the input; callers must reduce over the last dim
    to get a per-sample scalar.
    """
    if lows.shape != highs.shape:
        raise ValueError(f"lows and highs must share shape, got {lows.shape} vs {highs.shape}")
    if (highs <= lows).any():
        raise ValueError("highs must be strictly > lows per-element")
    return ComposeTransform([
        SigmoidTransform(),                                 # ℝ -> (0, 1)
        AffineTransform(loc=lows, scale=highs - lows),      # (0, 1) -> (lo, hi)
    ])


def build_nd_bijection(cfg) -> ComposeTransform:
    """Bijection for the ND params only (used inside gen_prior)."""
    lows  = torch.tensor([b[0] for _, b in cfg.params_dict.values()],
                         dtype=cfg.hw.dtype, device=cfg.hw.device)
    highs = torch.tensor([b[1] for _, b in cfg.params_dict.values()],
                         dtype=cfg.hw.dtype, device=cfg.hw.device)
    return build_box_bijection(lows, highs)


def build_rescale_bijection(cfg) -> ComposeTransform:
    """Bijection for the rescale params only (used to build the latent rescale prior)."""
    lows  = torch.tensor([b[0] for _, b in cfg.rescale_params.values()],
                         dtype=cfg.hw.dtype, device="cpu")
    highs = torch.tensor([b[1] for _, b in cfg.rescale_params.values()],
                         dtype=cfg.hw.dtype, device="cpu")
    return build_box_bijection(lows, highs)


def build_inferred_bijection(cfg) -> ComposeTransform:
    """
    Bijection for the full inferred parameter space (ND + rescale), in the same order
    ProductPrior stacks them: [nd_params, rescale_params]. Used inside gen_training_data
    and to wrap the trained DirectPosterior.
    """
    nd_lows  = torch.tensor([b[0] for _, b in cfg.params_dict.values()],
                            dtype=cfg.hw.dtype, device=cfg.hw.device)
    nd_highs = torch.tensor([b[1] for _, b in cfg.params_dict.values()],
                            dtype=cfg.hw.dtype, device=cfg.hw.device)
    rs_lows  = torch.tensor([b[0] for _, b in cfg.rescale_params.values()],
                            dtype=cfg.hw.dtype, device=cfg.hw.device)
    rs_highs = torch.tensor([b[1] for _, b in cfg.rescale_params.values()],
                            dtype=cfg.hw.dtype, device=cfg.hw.device)
    return build_box_bijection(torch.cat([nd_lows, rs_lows]),
                               torch.cat([nd_highs, rs_highs]))


def _transform_device(transform: ComposeTransform) -> torch.device:
    """Extract the device the transform's tensors live on by peeking at the AffineTransform's loc."""
    for t in transform.parts:
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