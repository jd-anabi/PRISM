"""
Pipeline orchestration for the SBI pipeline.

No input() calls live here -- all user interaction is delegated to cli.py.
This module owns the pipeline flow: observe -> prior -> posterior -> validate.
"""
import importlib
import math
import time

import torch
import numpy as np
from matplotlib import pyplot as plt
from sbi.analysis import pairplot
from sbi.inference import DirectPosterior
from torch.distributions import Distribution, MixtureSameFamily

from .config import SimConfig, PRIOR_PATH, POSTERIOR_PATH, PLOT_PATH
from . import cli
from .Helpers import helpers, visualizers, file_manager
from .SBI import embedded_network, pipeline, analysis
from .SBI.Priors import sbi_prior_wrapper

# Directories have spaces in their names, so use importlib for these imports
_scaling_mod = importlib.import_module("core.SBI.Priors.Scaling Priors.scaling_prior")
ScalingPrior = _scaling_mod.ScalingPrior

_forcing_mod = importlib.import_module("core.SBI.Priors.Forcing Priors.forcing_prior")
ForcingPrior = _forcing_mod.ForcingPrior

_product_mod = importlib.import_module("core.SBI.Priors.Product Prior.product_prior")
ProductPrior = _product_mod.ProductPrior


# ── Pipeline entry point ────────────────────────────────────────────────────
def run(cfg: SimConfig):
    """
    Execute the full SBI pipeline:
      1. Generate synthetic observations
      2. Build or load the prior (product prior: ND x rescale x forcing)
      3. Train or load the posterior
      4. Validate (PPC, SBC, coverage, eye test)
    """
    # 1. Synthetic observations
    x_dim, obs_stats, t_dim = generate_observations(cfg)
    visualizers.plot(
        t_dim.squeeze(0).cpu().detach().numpy(),
        x_dim[0, :].cpu().detach().numpy(),
    )

    # 2. Prior
    prior_choice, build_new = cli.select_or_build_prior()
    inf_prior, force_prior = build_prior(cfg, prior_choice, build_new)

    # 3. Posterior
    pos_choice, train_new = cli.select_or_train_posterior()
    posterior, pos_diagnostics = build_posterior(cfg, inf_prior, force_prior, obs_stats, pos_choice, train_new)
    helpers.clear_screen()

    # 4. Validate
    validate(cfg, posterior, x_dim, obs_stats, force_prior, t_dim)


# ── Step 1: Synthetic data ──────────────────────────────────────────────────
def generate_observations(cfg: SimConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Simulate ground-truth time series and compute summary statistics.

    :return: (obs_data, obs_stats) where obs_data has shape (n_vars, T_steady)
             and obs_stats has shape (1, n_stats).
    """
    t = cfg.t

    # Stack ground-truth forcing and rescale params as (1, n) tensors
    forcing_gt = torch.tensor([[val for val, _ in cfg.force_params_dict.values()]], dtype=cfg.hw.dtype, device=cfg.hw.device)
    rescale_gt = torch.tensor([[val for val, _ in cfg.rescale_params.values()]], dtype=cfg.hw.dtype, device=cfg.hw.device)

    # Build ND force
    force = pipeline.build_nondim_sin_force_tensor(forcing_gt, cfg.t, rescale_gt, cfg.forcing_idx, cfg.rescale_idx)

    x_nd = pipeline.gen_obs(
        model=cfg.model, params=cfg.params_tensor, t=t, inits=cfg.inits_tensor,
        force=force, n_segs=cfg.n_segs, steady_idx=cfg.steady_idx,
        state_dep_drift=cfg.state_dep_drift,
        dtype=cfg.hw.dtype, device=cfg.hw.device,
    )[0, :, :]

    x_scale = rescale_gt[:, cfg.rescale_idx["x_scale"]].unsqueeze(1)
    x_offset = rescale_gt[:, cfg.rescale_idx["x_offset"]].unsqueeze(1)
    t_scale = rescale_gt[:, cfg.rescale_idx["t_scale"]]
    t_offset = rescale_gt[:, cfg.rescale_idx["t_offset"]]
    x_dim = helpers.rescale(x_nd, x_scale, x_offset)
    t_dim = helpers.rescale(t[cfg.steady_idx:], t_scale, t_offset)
    dt_dim = t_scale * cfg.dt

    obs_stats = pipeline.gen_stats(x_dim, dt_dim)
    obs_stats = torch.cat([obs_stats, forcing_gt.cpu()], dim=-1)
    return x_dim, obs_stats, t_dim


# ── Step 2: Prior construction ──────────────────────────────────────────────
def build_prior(cfg: SimConfig, choice: str | None, build_new: bool) -> tuple[Distribution, Distribution]:
    """
    Load an existing prior from disk, or construct a new product prior:
        ProductPrior = ND parameter prior x rescaling prior x forcing prior

    :param cfg: Pipeline configuration.
    :param choice: Filename of a saved prior, or None to build from scratch.
    :param build_new: True to construct from scratch.
    :return: A Distribution that can be sampled and scored.
    """
    # 1. Forcing prior
    force_prior = _build_forcing_prior(cfg)

    # 2. Rescaling prior
    rescale_prior = _build_rescale_prior(cfg)

    if not build_new and choice is not None:
        nd_prior = file_manager.load_mix_dist(str(PRIOR_PATH / choice), device=cfg.hw.device)
        visualizers.visualize_dist(nd_prior, labels=cfg.labels)
        nd_dim = len(cfg.params_dict)
        rescale_dim = len(cfg.rescale_params)
        inferred_prior = ProductPrior(
            distributions=[nd_prior, rescale_prior],
            dims=[nd_dim, rescale_dim],
        )
        return inferred_prior, force_prior

    # --- Build from scratch ---
    print("No prior found. Going to construct prior from scratch.")
    time.sleep(5)
    helpers.clear_screen()

    # 3. ND parameter prior (stability-filtered GMM)
    nd_prior = pipeline.gen_prior(
        model=cfg.model, t=cfg.t,
        global_batch_size=cfg.hw.batch_size,
        local_batch_size=(cfg.hw.batch_size // 2),
        segs=math.ceil(cfg.n_segs / 2),
        prior_bounds=cfg.nd_params_bounds,
        state_dep_drift=cfg.state_dep_drift,
        num_iterations=50,
        dtype=cfg.hw.dtype, device=cfg.hw.device,
    )

    # Save the ND prior (GMM) with the existing serializer
    nd_name = cli.prompt_save_name("ND parameter prior")
    file_manager.save_mix_dist(nd_prior, str(PRIOR_PATH / (nd_name + ".pt")))
    visualizers.visualize_dist(nd_prior, labels=cfg.labels, save_path=str(PLOT_PATH / (nd_name + ".png")))

    # 4. Compose into product prior
    nd_dim = len(cfg.params_dict)
    rescale_dim = len(cfg.rescale_params)

    inferred_prior = ProductPrior(
        distributions=[nd_prior, rescale_prior],
        dims=[nd_dim, rescale_dim],
    )

    return inferred_prior, force_prior


def _build_rescale_prior(cfg: SimConfig) -> Distribution:
    """Construct the rescaling-parameter prior from cell file bounds."""
    # rescale_params format: {name: (val, (lo, hi))}
    bounds = [row[1] for row in cfg.rescale_params.values()]
    types = tuple("uniform" for _ in cfg.rescale_params)

    scaling = ScalingPrior(cfg.hw.dtype, cfg.hw.device)
    return scaling.construct_prior(bounds, types)


def _build_forcing_prior(cfg: SimConfig) -> Distribution:
    """Construct the forcing-parameter prior from cell file bounds."""
    # force_params_dict format: {name: (val, (lo, hi))}
    bounds = [row[1] for row in cfg.force_params_dict.values()]
    types = tuple("uniform" for _ in cfg.force_params_dict)

    forcing = ForcingPrior(cfg.hw.dtype, cfg.hw.device)
    return forcing.construct_prior(bounds, types)


# ── Step 3: Posterior construction ──────────────────────────────────────────
def build_posterior(
    cfg: SimConfig,
    prior: Distribution,
    force_prior: Distribution,
    obs_stats: torch.Tensor,
    choice: str | None,
    train_new: bool,
) -> tuple[DirectPosterior, dict | None]:
    """
    Load an existing posterior from disk, or train a new one with SNPE.

    :return: (posterior, diagnostics_dict_or_None)
    """
    if not train_new and choice is not None:
        posterior = torch.load(str(POSTERIOR_PATH / choice), weights_only=False)
        assert isinstance(posterior, DirectPosterior)
        return posterior, None

    # --- Train from scratch ---
    # Build fixed_dict for parameter groups that are NOT inferred
    fixed_dict = _build_fixed_dict(cfg)

    training_params = {
        "model": cfg.model,
        "prior": prior,
        "t": cfg.t,
        "run_size": cfg.hw.batch_size,
        "num_runs": 300,
        "n_segs": cfg.n_segs,
        "steady_idx": cfg.steady_idx,
        "dt": cfg.dt,
        "state_dep_drift": cfg.state_dep_drift,
        "dtype": cfg.hw.dtype,
        "device": cfg.hw.device,
    }

    # Set up embedded network (with optional forcing conditioning)
    force_dim = len(cfg.force_params_dict) if "forcing" not in cfg.inferred_groups else 0
    input_dim = obs_stats.shape[1] - force_dim

    if force_dim > 0:
        embedded_net = embedded_network.EmbeddedNet(
            input_dim, 3 * input_dim // 2,
            (5 * input_dim // 2, 2 * input_dim),
            forcing_dim=force_dim,
            forcing_layer_dims=(force_dim * 4, force_dim * 2),
            merge_layer_dim=2 * input_dim,
        )
    else:
        embedded_net = embedded_network.EmbeddedNet(
            input_dim, 3 * input_dim // 2,
            (5 * input_dim // 2, 2 * input_dim),
        )

    # Wrap prior for SBI compatibility
    sbi_prior = sbi_prior_wrapper.SBIPriorWrapper(prior)

    posterior, pos_diagnostics = pipeline.train_nn(
        training_params, model="maf", prior=sbi_prior,
        embedding_net=embedded_net, forcing_prior=force_prior,
        nd_dim=len(cfg.params_dict), forcing_idx=cfg.forcing_idx, rescale_idx=cfg.rescale_idx,
        x_obs=obs_stats, theta_obs=cfg.ground_truth_tensor, num_rounds=1,
        return_diagnostics=True, fixed_dict=fixed_dict,
        batch_size=int(2 ** 7), device=cfg.hw.device,
    )

    # Save
    name = cli.prompt_save_name("posterior")
    torch.save(posterior, str(POSTERIOR_PATH / (name + ".pt")))

    assert isinstance(posterior, DirectPosterior)
    return posterior, pos_diagnostics


def _build_fixed_dict(cfg: SimConfig) -> dict | None:
    """
    Build a dict mapping parameter indices to fixed ground-truth values
    for parameter groups that are NOT being inferred.

    The product prior concatenates parameters as: [nd | rescale].
    """
    nd_dim = len(cfg.params_dict)
    fixed = {}

    # If rescale params are NOT inferred, fix them at ground-truth values
    if "rescale" not in cfg.inferred_groups:
        for i, (val, _bounds) in enumerate(cfg.rescale_params.values()):
            fixed[nd_dim + i] = val

    return fixed if fixed else None


# ── Step 4: Validation ──────────────────────────────────────────────────────
def validate(cfg: SimConfig, posterior: DirectPosterior, obs_data: torch.Tensor,
    obs_stats: torch.Tensor, force_prior: Distribution, t_dim: torch.Tensor) -> None:
    """
    Run all posterior validation steps:
      - Corner plot
      - Posterior predictive check (PPC)
      - Simulation-based calibration (SBC)
      - Expected coverage
      - Eye test (MAP vs ground truth vs posterior samples)
    """
    t = cfg.t
    device = cfg.hw.device
    dtype = cfg.hw.dtype

    # Corner plot
    samples = posterior.sample((1000,), x=obs_stats.to(device))
    fig, ax = pairplot(
        samples.cpu().numpy(),
        points=np.array([cfg.ground_truth]),
        labels=cfg.inferred_labels,
    )
    plt.show()

    # PPC
    nd_dim = len(cfg.params_dict)
    samples_nd = samples[:, :nd_dim]
    samples_rescale = samples[:, nd_dim:]

    forcing_gt = torch.tensor([[val for val, _ in cfg.force_params_dict.values()]], dtype=dtype, device=device)
    forcing_gt_expanded = forcing_gt.expand(samples.shape[0], -1)  # (n_samples, n_forcing)
    force = pipeline.build_nondim_sin_force_tensor(
        forcing_gt_expanded, t, samples_rescale, cfg.forcing_idx, cfg.rescale_idx
    )

    x_nd = pipeline.gen_obs(
        model=cfg.model, params=samples_nd, t=t, inits=cfg.inits_tensor.expand(samples.shape[0], -1),
        force=force, n_segs=cfg.n_segs, steady_idx=cfg.steady_idx,
        state_dep_drift=cfg.state_dep_drift,
        batch_size=samples.shape[0], dtype=dtype, device=device,
    )[0, :, :]

    x_scale = samples_rescale[:, cfg.rescale_idx["x_scale"]].unsqueeze(1)
    x_offset = samples_rescale[:, cfg.rescale_idx["x_offset"]].unsqueeze(1)
    t_scale = samples_rescale[:, cfg.rescale_idx["t_scale"]].unsqueeze(1)
    x_dim = helpers.rescale(x_nd, x_scale, x_offset)
    dt_dim = (t_scale * cfg.dt).squeeze(1)  # (n_samples,)
    del x_nd, force

    sim_stats = pipeline.gen_stats(x_dim, dt_dim, device=device)
    sim_stats = torch.cat([sim_stats, forcing_gt_expanded.cpu()], dim=-1)
    results = analysis.posterior_predictive_check(obs_stats.squeeze(), sim_stats)

    # SBC + coverage
    x_cal, theta_star = analysis.gen_cal_data(
        model=cfg.model, prior=posterior, forcing_prior=force_prior,
        t=t, n_segs=cfg.n_segs, steady_idx=cfg.steady_idx, dt=cfg.dt, n_cal=1000,
        nd_dim=len(cfg.params_dict), forcing_idx=cfg.forcing_idx, rescale_idx=cfg.rescale_idx,
        state_dep_drift=cfg.state_dep_drift, dtype=dtype, device=device,
    )
    ranks = analysis.compute_sbc_ranks(posterior, theta_star, x_cal, m=1000, device=device)
    alphas = analysis.compute_expected_coverage(posterior, theta_star, x_cal, m=1000, dtype=dtype, device=device)

    sbc_plot = visualizers.plot_sbc(ranks, param_names=cfg.inferred_labels, m=1000, fig_size=(7, 12))
    expected_cov_plot = visualizers.plot_expected_coverage(alphas, fig_size=(7, 20))
    plt.show()

    # Eye test: MAP vs ground truth vs posterior samples
    log_probs = posterior.log_prob(samples, x=obs_stats.to(device))
    map_params = samples[log_probs.argmax()].unsqueeze(0)  # (1, nd_dim + rescale_dim)
    map_nd = map_params[:, :nd_dim]  # (1, nd_dim)
    map_rescale = map_params[:, nd_dim:]  # (1, rescale_dim)

    # Build force from MAP rescale params and ground-truth forcing
    forcing_gt_single = forcing_gt_expanded[:1]  # (1, n_forcing) — reuse from PPC
    force_map = pipeline.build_nondim_sin_force_tensor(
        forcing_gt_single, t, map_rescale, cfg.forcing_idx, cfg.rescale_idx
    )

    x_map_nd = pipeline.gen_obs(
        model=cfg.model, params=map_nd, t=t, inits=cfg.inits_tensor,
        force=force_map, n_segs=cfg.n_segs, steady_idx=cfg.steady_idx,
        state_dep_drift=cfg.state_dep_drift,
        dtype=dtype, device=device,
    )[0, 0, :]  # (T_steady,)

    # Redimensionalize
    map_x_scale = map_rescale[:, cfg.rescale_idx["x_scale"]]
    map_x_offset = map_rescale[:, cfg.rescale_idx["x_offset"]]
    x_map = helpers.rescale(x_map_nd, map_x_scale, map_x_offset)  # (T_steady,)

    t_plot = t_dim.squeeze(0).cpu().numpy()
    fig = visualizers.plot_posterior_vs_truth(
        t=t_plot,
        x_true=obs_data[0, :].cpu().numpy(),
        x_map=x_map.cpu().numpy(),
        x_samples=x_dim.cpu().numpy(),
        n_show=10,
    )
    plt.show()
