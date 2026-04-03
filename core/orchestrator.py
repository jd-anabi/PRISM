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

_forcing_mod = importlib.import_module("core.SBI.Priors.Forcing Priors.sin_prior")
SinPrior = _forcing_mod.SinPrior

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
    obs_data, obs_stats = generate_observations(cfg)
    visualizers.plot(
        cfg.t[cfg.steady_idx:].cpu().detach().numpy(),
        obs_data[0, :].cpu().detach().numpy(),
    )

    # 2. Prior
    prior_choice, build_new = cli.select_or_build_prior()
    prior = build_prior(cfg, prior_choice, build_new)

    # 3. Posterior
    pos_choice, train_new = cli.select_or_train_posterior()
    posterior, pos_diagnostics = build_posterior(cfg, prior, obs_stats, pos_choice, train_new)
    helpers.clear_screen()

    # 4. Validate
    validate(cfg, posterior, obs_data, obs_stats)


# ── Step 1: Synthetic data ──────────────────────────────────────────────────
def generate_observations(cfg: SimConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Simulate ground-truth time series and compute summary statistics.

    :return: (obs_data, obs_stats) where obs_data has shape (n_vars, T_steady)
             and obs_stats has shape (1, n_stats).
    """
    t = cfg.t
    force = torch.zeros((cfg.hw.batch_size, t.shape[0]), dtype=cfg.hw.dtype, device=cfg.hw.device)

    obs_data = pipeline.gen_obs(
        model=cfg.model, params=cfg.params_tensor, t=t, inits=cfg.inits_tensor,
        force=force[0].unsqueeze(0), n_segs=cfg.n_segs, steady_idx=cfg.steady_idx,
        state_dep_drift=cfg.state_dep_drift,
    )[0, :, :]
    obs_stats = pipeline.gen_stats(obs_data, cfg.dt)
    return obs_data, obs_stats


# ── Step 2: Prior construction ──────────────────────────────────────────────
def build_prior(cfg: SimConfig, choice: str | None, build_new: bool) -> Distribution:
    """
    Load an existing prior from disk, or construct a new product prior:
        ProductPrior = ND parameter prior x rescaling prior x forcing prior

    :param cfg: Pipeline configuration.
    :param choice: Filename of a saved prior, or None to build from scratch.
    :param build_new: True to construct from scratch.
    :return: A Distribution that can be sampled and scored.
    """
    if not build_new and choice is not None:
        prior = file_manager.load_mix_dist(str(PRIOR_PATH / choice), device=cfg.hw.device)
        visualizers.visualize_dist(prior, labels=cfg.labels)
        return prior

    # --- Build from scratch ---
    print("No prior found. Going to construct prior from scratch.")
    time.sleep(5)
    helpers.clear_screen()

    # 1. ND parameter prior (stability-filtered GMM)
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

    # 2. Rescaling prior
    rescale_prior = _build_rescale_prior(cfg)

    # 3. Forcing prior
    force_prior = _build_forcing_prior(cfg)

    # 4. Compose into product prior
    nd_dim = len(cfg.params_dict)
    rescale_dim = len(cfg.rescale_params)
    force_dim = len(cfg.force_params_dict)

    product = ProductPrior(
        distributions=[nd_prior, rescale_prior, force_prior],
        dims=[nd_dim, rescale_dim, force_dim],
    )

    return product


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

    forcing = SinPrior(cfg.hw.dtype, cfg.hw.device)
    return forcing.construct_prior(bounds, types)


# ── Step 3: Posterior construction ──────────────────────────────────────────
def build_posterior(
    cfg: SimConfig,
    prior: Distribution,
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
    input_dim = obs_stats.shape[1]
    force_dim = len(cfg.force_params_dict) if "forcing" not in cfg.inferred_groups else 0

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
        embedding_net=embedded_net, x_obs=obs_stats,
        theta_obs=cfg.ground_truth_tensor, num_rounds=1,
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

    The product prior concatenates parameters as: [nd | rescale | forcing].
    """
    nd_dim = len(cfg.params_dict)
    rescale_dim = len(cfg.rescale_params)
    force_dim = len(cfg.force_params_dict)

    fixed = {}

    # If rescale params are NOT inferred, fix them at ground-truth values
    if "rescale" not in cfg.inferred_groups:
        for i, (val, _bounds) in enumerate(cfg.rescale_params.values()):
            fixed[nd_dim + i] = val

    # If forcing params are NOT inferred, fix them at ground-truth values
    if "forcing" not in cfg.inferred_groups:
        for i, (val, _bounds) in enumerate(cfg.force_params_dict.values()):
            fixed[nd_dim + rescale_dim + i] = val

    return fixed if fixed else None


# ── Step 4: Validation ──────────────────────────────────────────────────────
def validate(
    cfg: SimConfig,
    posterior: DirectPosterior,
    obs_data: torch.Tensor,
    obs_stats: torch.Tensor,
):
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
        labels=cfg.labels,
    )
    plt.show()

    # PPC
    x_sims = pipeline.gen_obs(
        model=cfg.model, params=samples, t=t, inits=cfg.inits_tensor.expand(samples.shape[0], -1),
        force=torch.zeros((samples.shape[0], t.shape[0]), dtype=dtype, device=device),
        n_segs=cfg.n_segs, steady_idx=cfg.steady_idx,
        state_dep_drift=cfg.state_dep_drift,
        batch_size=samples.shape[0], dtype=dtype, device=device,
    )[0, :, :]
    sim_stats = pipeline.gen_stats(x_sims, cfg.dt, device=device)
    results = analysis.posterior_predictive_check(obs_stats.squeeze(), sim_stats)

    # SBC + coverage
    x_cal, theta_star = analysis.gen_cal_data(
        model=cfg.model, prior=posterior, t=t, n_segs=cfg.n_segs,
        steady_idx=cfg.steady_idx, dt=cfg.dt, n_cal=1000,
        state_dep_drift=cfg.state_dep_drift, dtype=dtype, device=device,
    )
    ranks = analysis.compute_sbc_ranks(posterior, theta_star, x_cal, m=1000, device=device)
    alphas = analysis.compute_expected_coverage(posterior, theta_star, x_cal, m=1000, dtype=dtype, device=device)

    sbc_plot = visualizers.plot_sbc(ranks, param_names=cfg.labels, m=1000, fig_size=(7, 12))
    expected_cov_plot = visualizers.plot_expected_coverage(alphas, fig_size=(7, 20))
    plt.show()

    # Eye test: MAP vs ground truth vs posterior samples
    log_probs = posterior.log_prob(samples, x=obs_stats.to(device))
    map_params = samples[log_probs.argmax()].unsqueeze(0)

    x_map = pipeline.gen_obs(
        model=cfg.model, params=map_params, t=t, inits=cfg.inits_tensor,
        force=torch.zeros(1, t.shape[0], dtype=dtype, device=device),
        n_segs=cfg.n_segs, steady_idx=cfg.steady_idx,
        state_dep_drift=cfg.state_dep_drift,
    )[0, 0, :]

    t_plot = t[cfg.steady_idx:].cpu().numpy()
    fig = visualizers.plot_posterior_vs_truth(
        t=t_plot,
        x_true=obs_data[0, :].cpu().numpy(),
        x_map=x_map.cpu().numpy(),
        x_samples=x_sims.cpu().numpy(),
        n_show=10,
    )
    plt.show()
