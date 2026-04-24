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
from torch.distributions.transforms import (
    AffineTransform, SigmoidTransform, ComposeTransform,
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