"""
Pipeline orchestration for the SBI pipeline.

No input() calls live here -- all user interaction is delegated to cli.py.
This module owns the pipeline flow: observe -> prior -> posterior -> validate.
"""
import importlib
import math
import time
import warnings

import torch
import numpy as np
from matplotlib import pyplot as plt
from sbi.analysis import pairplot, sbc_rank_plot, plot_tarp
from sbi.diagnostics import run_sbc, check_sbc, run_tarp, check_tarp
from sbi.inference import DirectPosterior
from torch.distributions import Distribution, MixtureSameFamily
from tqdm import tqdm

from .config import (
    SimConfig, PRIOR_PATH, POSTERIOR_PATH, PLOT_PATH,
    CHUNK_LEN, N_ND_MAX, PPC_BIN_SIZE, STABILITY_SWEEP_ND_UNITS, TRAINING_NUM_RUNS,
)
from . import cli
from .Helpers import helpers, visualizers, file_manager
from .SBI import embedded_network, pipeline, analysis
from .SBI.Priors import sbi_prior_wrapper
from .SBI.reparam import build_inferred_bijection, TransformedPosterior, build_rescale_bijection, _transform_device

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
      5. Optionally infer parameters from a real experimental recording
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
    validate(cfg, posterior, inf_prior, x_dim, obs_stats, force_prior, t_dim)

    # 5. Inference on real experimental data (optional)
    if cli.select_or_skip_inference():
        data_path, T_obs_s, forcing_params_si = cli.get_inference_inputs(list(cfg.force_params_dict.keys()))
        X_obs = file_manager.load_experimental_data(data_path, dtype=cfg.hw.dtype)
        samples = infer_from_experiment(
            cfg, posterior, X_obs, T_obs_s, forcing_params_si, n_samples=1000,
        )
        # Corner plot of inferred parameters
        fig, ax = pairplot(
            samples.cpu().numpy(),
            labels=cfg.inferred_labels,
        )
        plt.show()


# ── Step 1: Synthetic data ──────────────────────────────────────────────────
def generate_observations(cfg: SimConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Simulate a ground-truth observation matching experimental conditions.

    Simulates at fine ND resolution (dt_nd_min, stable for EM), then downsamples
    to match the physical sampling rate dt_exp and duration T_obs — exactly
    mirroring what the training loop produces.

    :return: (obs_data, obs_stats, t_dim) where obs_data has shape (1, N_obs),
             obs_stats has shape (1, n_stats + n_forcing + 1), and t_dim is the
             dimensional time vector.
    """
    t = cfg.t  # full pre-simulated ND time vector at dt_nd_min

    # Ground-truth rescale and forcing params as (1, n) tensors
    forcing_gt = torch.tensor([[val for val, _ in cfg.force_params_dict.values()]], dtype=cfg.hw.dtype, device=cfg.hw.device)
    rescale_gt = torch.tensor([[val for val, _ in cfg.rescale_params.values()]], dtype=cfg.hw.dtype, device=cfg.hw.device)

    # Ground-truth t_scale for this observation
    t_scale_gt = rescale_gt[:, cfg.rescale_idx["t_scale"]].item()

    # Compute ND quantities for this observation (same logic as training loop)
    dt_nd_gt = cfg.dt_exp / t_scale_gt
    T_nd_obs = cfg.T_obs / t_scale_gt
    subsample_factor = max(1, round(dt_nd_gt / cfg.dt_nd_min))
    N_obs = int(T_nd_obs / dt_nd_gt)

    # Fine-resolution time vector: transient + enough to downsample into N_obs points
    n_fine_total = cfg.steady_idx + N_obs * subsample_factor

    # OOD warning: NN was only trained on combinations with n_fine_total <= N_ND_MAX
    if n_fine_total > N_ND_MAX:
        warnings.warn(
            f"Synthetic GT observation out-of-distribution: n_fine_total={n_fine_total} "
            f"> N_ND_MAX={N_ND_MAX}. Network was trained only on combinations with "
            f"n_fine_total <= {N_ND_MAX}. Posterior may extrapolate poorly.",
            stacklevel=2,
        )

    # Cost ceiling: if simulation exceeds the pre-simulated grid, clip and update T_obs
    # so that log(T_obs) conditioning matches the actual trajectory length downstream.
    if n_fine_total > len(t):
        N_obs = (len(t) - cfg.steady_idx) // subsample_factor
        n_fine_total = cfg.steady_idx + N_obs * subsample_factor
        actual_T_obs = N_obs * cfg.dt_exp
        warnings.warn(
            f"Observation cost ceiling hit: requested T_obs={cfg.T_obs:.4f} exceeds "
            f"pre-simulated grid. Clipping N_obs to {N_obs} (actual T_obs={actual_T_obs:.4f}). "
            f"cfg.T_obs updated so downstream code sees the consistent value.",
            stacklevel=2,
        )
        cfg.T_obs = actual_T_obs  # keep log(T) conditioning consistent across pipeline

    t_fine = t[:n_fine_total]

    # Auto-derive n_segs based on CHUNK_LEN (per-chunk memory cap)
    n_segs_gt = max(1, math.ceil(n_fine_total / CHUNK_LEN))

    # Build ND force at fine resolution
    force = pipeline.build_nondim_sin_force_tensor(forcing_gt, t_fine, rescale_gt, cfg.forcing_idx, cfg.rescale_idx)

    # Simulate at fine dt_nd_min (stable for EM), then downsample
    x_nd_fine = pipeline.gen_obs(
        model=cfg.model, params=cfg.params_tensor, t=t_fine, inits=cfg.inits_tensor,
        force=force, n_segs=n_segs_gt, steady_idx=cfg.steady_idx,
        state_dep_drift=cfg.state_dep_drift,
        dtype=cfg.hw.dtype, device=cfg.hw.device,
    )[0, :, :]
    x_nd = x_nd_fine[:, ::subsample_factor][:, :N_obs]
    del x_nd_fine, force

    # Redimensionalize
    x_scale = rescale_gt[:, cfg.rescale_idx["x_scale"]].unsqueeze(1)
    x_offset = rescale_gt[:, cfg.rescale_idx["x_offset"]].unsqueeze(1)
    t_offset = rescale_gt[:, cfg.rescale_idx["t_offset"]].item()
    x_dim = helpers.rescale(x_nd, x_scale, x_offset)

    # Dimensional time vector for plotting (N_obs points at dt_exp spacing)
    t_dim = torch.arange(N_obs, dtype=cfg.hw.dtype) * cfg.dt_exp + t_offset
    t_dim = t_dim.unsqueeze(0)  # (1, N_obs)

    # Summary statistics + conditioning vector
    obs_stats = pipeline.gen_stats(x_dim, cfg.dt_exp, device=cfg.hw.device)
    log_T_obs = torch.tensor([[math.log(cfg.T_obs)]], dtype=cfg.hw.dtype)
    obs_stats = torch.cat([obs_stats, forcing_gt.cpu(), log_T_obs], dim=-1)
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
    # Stability is a per-parameter property — screen on a short fixed-length trajectory
    # (STABILITY_SWEEP_ND_UNITS) rather than the full master grid. Global sweep uses
    # half this (t_global_scale=2 inside gen_prior), local sweep uses the full t_stab.
    n_stab_fine = int(STABILITY_SWEEP_ND_UNITS / cfg.dt_nd_min)
    t_stab = cfg.t[:n_stab_fine]
    prior_segs = max(1, math.ceil(n_stab_fine / CHUNK_LEN))
    nd_prior = pipeline.gen_prior(
        model=cfg.model, t=t_stab,
        global_batch_size=cfg.hw.batch_size,
        local_batch_size=(cfg.hw.batch_size // 2),
        segs=prior_segs,
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
    """
    Construct the rescaling-parameter prior from cell file bounds.

    Scale parameters (names containing 'scale') use log-uniform — they're positive
    and span orders of magnitude, so uniform would over-weight the high end.
    Offset parameters use uniform — they can be negative or zero.
    """
    bounds = [row[1] for row in cfg.rescale_params.values()]
    types = tuple(
        "log-uni" if "scale" in name else "uniform"
        for name in cfg.rescale_params.keys()
    )
    scaling = ScalingPrior(cfg.hw.dtype, cfg.hw.device)
    return scaling.construct_prior(bounds, types)


def _build_forcing_prior(cfg: SimConfig) -> Distribution:
    """
    Construct the forcing-parameter prior from cell file bounds.

    'freq' uses log-uniform — hair bundle resonances span decades of Hz, and uniform
    over-weights the high end. All other forcing params (amp, phase, offset) use
    uniform — amp bound can include 0 (log-uniform would fail), phase is a bounded
    angle, offset can be negative.
    """
    bounds = [row[1] for row in cfg.force_params_dict.values()]
    types = tuple(
        "log-uni" if name == "freq" else "uniform"
        for name in cfg.force_params_dict.keys()
    )
    forcing = ForcingPrior(cfg.hw.dtype, cfg.hw.device)
    return forcing.construct_prior(bounds, types)


# ── Step 3: Posterior construction ──────────────────────────────────────────
def build_posterior(
    cfg: SimConfig,
    prior: Distribution,                 # physical inferred prior from build_prior
    force_prior: Distribution,
    obs_stats: torch.Tensor,
    choice: str | None,
    train_new: bool,
) -> tuple[TransformedPosterior, dict | None]:
    """
    Load an existing latent DirectPosterior from disk and wrap with T, or train a new one
    via NPE in latent space. Returns a TransformedPosterior whose .sample/.log_prob operate
    in physical-parameter coordinates for downstream code.
    """
    T = build_inferred_bijection(cfg)

    if not train_new and choice is not None:
        posterior_latent = torch.load(str(POSTERIOR_PATH / choice), weights_only=False)
        assert isinstance(posterior_latent, DirectPosterior)
        return TransformedPosterior(posterior_latent, T), None

    # --- Build a LATENT product prior for SBI to train on ---
    # Physical prior layout: ProductPrior([nd_prior_physical, rescale_prior_physical]).
    # Extract latent ND (the MixtureSameFamily inside the TransformedDistribution):
    nd_prior_physical      = prior.distributions[0]      # TransformedDistribution(latent_gmm, T_nd)
    if not isinstance(nd_prior_physical, torch.distributions.TransformedDistribution):
        raise ValueError(
            "Loaded ND prior is not a TransformedDistribution — it was saved with the pre-reparameterization "
            "pipeline. Regenerate the prior with the current `gen_prior` before training a new posterior."
        )
    rescale_prior_physical = prior.distributions[1]      # MultipleIndependent
    latent_nd = nd_prior_physical.base_dist              # the raw latent MixtureSameFamily

    # Pushforward the physical rescale prior through T_rescale.inv (Issue 2a).
    T_rescale = build_rescale_bijection(cfg)
    latent_rescale = torch.distributions.TransformedDistribution(rescale_prior_physical, T_rescale.inv)

    latent_inferred_prior = ProductPrior(
        distributions=[latent_nd, latent_rescale],
        dims=[len(cfg.params_dict), len(cfg.rescale_params)],
    )

    training_params = {
        "model": cfg.model,
        "prior": latent_inferred_prior,               # <-- latent
        "t": cfg.t,
        "run_size": cfg.hw.batch_size,
        "num_runs": TRAINING_NUM_RUNS,
        "steady_idx": cfg.steady_idx,
        "dt_nd_min": cfg.dt_nd_min,
        "dt_exp": cfg.dt_exp,
        "t_min_exp": cfg.t_min_exp,
        "t_max_exp": cfg.t_max_exp,
        "t_scale_bounds": cfg.t_scale_bounds,
        "state_dep_drift": cfg.state_dep_drift,
        "dtype": cfg.hw.dtype,
        "device": cfg.hw.device,
    }

    force_dim = len(cfg.force_params_dict) + 1
    input_dim = obs_stats.shape[1] - force_dim

    embedded_net = embedded_network.EmbeddedNet(
        input_dim, 3 * input_dim // 2,
        (5 * input_dim // 2, 2 * input_dim),
        forcing_dim=force_dim,
        forcing_layer_dims=(force_dim * 4, force_dim * 2),
        merge_layer_dim=2 * input_dim,
    )

    sbi_prior = sbi_prior_wrapper.SBIPriorWrapper(latent_inferred_prior)

    # theta_obs is the ground-truth PHYSICAL theta; for diagnostics we pass its latent form.
    theta_obs_latent = T.inv(cfg.ground_truth_tensor.to(_transform_device(T)))

    posterior_latent, pos_diagnostics = pipeline.train_nn(
        training_params, model="nsf", prior=sbi_prior,
        embedding_net=embedded_net, forcing_prior=force_prior,
        nd_dim=len(cfg.params_dict), forcing_idx=cfg.forcing_idx, rescale_idx=cfg.rescale_idx,
        x_obs=obs_stats, theta_obs=theta_obs_latent, num_rounds=1,
        return_diagnostics=True,
        theta_transform=T,
        batch_size=int(2 ** 7), device=cfg.hw.device,
    )

    name = cli.prompt_save_name("posterior")
    # Save the RAW latent DirectPosterior (not the wrapped one).
    torch.save(posterior_latent, str(POSTERIOR_PATH / (name + ".pt")))

    assert isinstance(posterior_latent, DirectPosterior)
    return TransformedPosterior(posterior_latent, T), pos_diagnostics


# ── Step 4: Validation ──────────────────────────────────────────────────────
def validate(cfg: SimConfig, posterior: DirectPosterior | TransformedPosterior, inferred_prior: Distribution,
    obs_data: torch.Tensor, obs_stats: torch.Tensor, force_prior: Distribution,
    t_dim: torch.Tensor) -> None:
    """
    Run all posterior validation steps:
      - Corner plot
      - Posterior predictive check (PPC)
      - Simulation-based calibration (SBC)
      - Expected coverage
      - Eye test (MAP vs ground truth vs posterior samples)

    :param inferred_prior: The actual prior used to train the posterior (ND x rescale
                           product prior). SBC requires drawing theta_star from the
                           prior, not from the posterior itself.
    """
    t = cfg.t
    device = cfg.hw.device
    dtype = cfg.hw.dtype
    T_obs = cfg.T_obs
    T = build_inferred_bijection(cfg)

    # Corner plot
    samples = posterior.sample((1000,), x=obs_stats.to(device))
    fig, ax = pairplot(
        samples.cpu().numpy(),
        points=np.array([cfg.ground_truth]),
        labels=cfg.inferred_labels,
    )
    plt.show()

    # PPC - Option B: sort posterior samples by t_scale, process in mini-batches
    # Each sample gets its own subsample_factor based on its t_scale; all samples
    # share physical duration T_obs at dt_exp sampling (matching the observation).
    nd_dim = len(cfg.params_dict)
    samples_nd = samples[:, :nd_dim]
    samples_rescale = samples[:, nd_dim:]
    n_samples = samples.shape[0]
    N_points_obs = int(cfg.T_obs / cfg.dt_exp)  # same for all samples

    forcing_gt = torch.tensor([[val for val, _ in cfg.force_params_dict.values()]], dtype=dtype, device=device)
    forcing_gt_expanded = forcing_gt.expand(n_samples, -1)  # (n_samples, n_forcing)

    # Sort by t_scale (ascending) so each bin contains similar-scale samples
    t_scales_all = samples_rescale[:, cfg.rescale_idx["t_scale"]]
    sort_idx = torch.argsort(t_scales_all)
    inv_sort_idx = torch.argsort(sort_idx)
    samples_nd_sorted = samples_nd[sort_idx]
    samples_rescale_sorted = samples_rescale[sort_idx]

    x_dim_sorted = torch.empty((n_samples, N_points_obs), dtype=dtype, device=device)
    arange_out = torch.arange(N_points_obs, device=device, dtype=torch.long)
    n_bins = math.ceil(n_samples / PPC_BIN_SIZE)

    with torch.no_grad():
        for b in tqdm(range(n_bins), desc="PPC simulations", leave=False):
            start = b * PPC_BIN_SIZE
            end = min(start + PPC_BIN_SIZE, n_samples)
            bs = end - start

            bin_nd = samples_nd_sorted[start:end]
            bin_rescale = samples_rescale_sorted[start:end]
            bin_t_scales = bin_rescale[:, cfg.rescale_idx["t_scale"]]

            # Smallest t_scale in the bin determines the finest resolution needed
            # (largest subsample_factor, hence largest n_fine_total)
            bin_t_scale_min = bin_t_scales.min().item()
            max_subsample_bin = max(1, round((cfg.dt_exp / bin_t_scale_min) / cfg.dt_nd_min))
            n_fine_bin = min(cfg.steady_idx + N_points_obs * max_subsample_bin, len(t))
            t_fine_bin = t[:n_fine_bin]
            n_segs_bin = max(1, math.ceil(n_fine_bin / CHUNK_LEN))

            # Build ND force for this bin
            force_bin = pipeline.build_nondim_sin_force_tensor(
                forcing_gt.expand(bs, -1), t_fine_bin, bin_rescale, cfg.forcing_idx, cfg.rescale_idx
            )

            # Simulate the bin at fine resolution
            x_nd_bin = pipeline.gen_obs(
                model=cfg.model, params=bin_nd, t=t_fine_bin,
                inits=cfg.inits_tensor.expand(bs, -1),
                force=force_bin, n_segs=n_segs_bin, steady_idx=cfg.steady_idx,
                state_dep_drift=cfg.state_dep_drift,
                batch_size=bs, dtype=dtype, device=device,
            )[0, :, :]  # (bs, n_fine_bin - steady_idx)

            # Per-sample downsample via gather: each row uses its own subsample_factor
            subsample_factors = torch.clamp(
                torch.round((cfg.dt_exp / bin_t_scales) / cfg.dt_nd_min), min=1
            ).long()  # (bs,)
            idx = subsample_factors.unsqueeze(1) * arange_out.unsqueeze(0)  # (bs, N_points_obs)
            idx = torch.clamp(idx, max=x_nd_bin.shape[1] - 1)  # safety for OOD samples
            x_nd_ds = torch.gather(x_nd_bin, dim=1, index=idx)  # (bs, N_points_obs)

            # Rescale per sample to dimensional units
            x_scale_col = bin_rescale[:, cfg.rescale_idx["x_scale"]].unsqueeze(1)
            x_offset_col = bin_rescale[:, cfg.rescale_idx["x_offset"]].unsqueeze(1)
            x_dim_sorted[start:end] = x_scale_col * x_nd_ds + x_offset_col

            del x_nd_bin, force_bin, x_nd_ds
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # Restore original sample order
    x_dim = x_dim_sorted[inv_sort_idx]

    sim_stats = pipeline.gen_stats(x_dim, cfg.dt_exp, device=device)
    log_T_obs = torch.full((sim_stats.shape[0], 1), math.log(T_obs), dtype=dtype)
    sim_stats = torch.cat([sim_stats, forcing_gt_expanded.cpu(), log_T_obs], dim=-1)
    results = analysis.posterior_predictive_check(obs_stats.squeeze(), sim_stats)
    visualizers.plot_ppc(
        results,
        ground_truth=cfg.ground_truth,
        param_names=cfg.inferred_labels,
        n_samples=n_samples,
    )
    plt.show()

    # SBC + coverage
    # Critical: draw theta_star from the PRIOR (not the posterior) for valid SBC.
    # SBC tests whether posterior(theta | x_cal) correctly ranks theta_true when
    # (theta_true, x_cal) is drawn from the joint used to train the posterior.
    x_cal, theta_star = analysis.gen_cal_data(
        model=cfg.model, prior=_build_latent_prior_for_validation(cfg, inferred_prior),  # see below
        forcing_prior=force_prior,
        t=t, steady_idx=cfg.steady_idx, dt_nd_min=cfg.dt_nd_min, n_cal=1000,
        nd_dim=len(cfg.params_dict), forcing_idx=cfg.forcing_idx, rescale_idx=cfg.rescale_idx,
        dt_exp=cfg.dt_exp, t_min_exp=cfg.t_min_exp, t_max_exp=cfg.t_max_exp,
        t_scale_bounds=cfg.t_scale_bounds,
        theta_transform=T,
        state_dep_drift=cfg.state_dep_drift, dtype=dtype, device=device,
    )

    # `gen_cal_data` returns CPU tensors; the posterior's embedding net is on
    # CUDA after training. Sample_batched does not move x for us (see
    # TransformedPosterior.sample_batched in reparam.py), so we need to match
    # devices manually — mirroring the obs_stats.to(device) call above.
    x_cal_dev = x_cal.to(device)
    theta_star_dev = theta_star.to(device)

    # --- SBC (Talts 2018, marginals) via sbi.diagnostics ---
    ranks, dap_samples = run_sbc(
        thetas=theta_star_dev, xs=x_cal_dev, posterior=posterior,
        num_posterior_samples=1000, reduce_fns="marginals",
        use_batched_sampling=True, show_progress_bar=True,
    )
    prior_samples = inferred_prior.sample((theta_star.shape[0],)).cpu()
    sbc_stats = check_sbc(
        ranks=ranks.cpu(), prior_samples=prior_samples, dap_samples=dap_samples.cpu(),
        num_posterior_samples=1000,
    )
    print("SBC uniformity checks:")
    for j, label in enumerate(cfg.inferred_labels):
        print(f"  {label}: KS p={sbc_stats['ks_pvals'][j]:.3f}  "
              f"c2st_ranks={sbc_stats['c2st_ranks'][j]:.3f}  "
              f"c2st_dap={sbc_stats['c2st_dap'][j]:.3f}")

    sbc_rank_plot(ranks=ranks, num_posterior_samples=1000, plot_type="cdf",
                  parameter_labels=cfg.inferred_labels)
    sbc_rank_plot(ranks=ranks, num_posterior_samples=1000, plot_type="hist",
                  parameter_labels=cfg.inferred_labels)
    plt.show()

    # --- Expected coverage (TARP, Lemos 2023) via sbi.diagnostics ---
    ecp, alpha_grid = run_tarp(
        thetas=theta_star_dev, xs=x_cal_dev, posterior=posterior,
        num_posterior_samples=1000, use_batched_sampling=True,
        z_score_theta=True, show_progress_bar=True,
    )
    atc, tarp_kspval = check_tarp(ecp.cpu(), alpha_grid.cpu())
    print(f"TARP: ATC={atc:.3f}  KS p={tarp_kspval:.3f}")
    plot_tarp(ecp.cpu(), alpha_grid.cpu(),
              title=f"TARP (ATC={atc:.3f}, KS p={tarp_kspval:.3f})")
    plt.show()

    # Eye test: MAP vs ground truth vs posterior samples
    # MAP is just the posterior sample with highest log-prob — its simulated
    # trajectory is already in x_dim (from the PPC step above), no need to re-simulate.
    log_probs = posterior.log_prob(samples, x=obs_stats.to(device))
    map_idx = log_probs.argmax().item()
    x_map = x_dim[map_idx]  # (N_points_obs,)

    t_plot = t_dim.squeeze(0).cpu().numpy()
    fig = visualizers.plot_posterior_vs_truth(
        t=t_plot,
        x_true=obs_data[0, :].cpu().numpy(),
        x_map=x_map.cpu().numpy(),
        x_samples=x_dim.cpu().numpy(),
        n_show=10,
    )
    plt.show()

def _build_latent_prior_for_validation(cfg, inferred_prior):
    """Mirror of the latent-prior construction in build_posterior, for gen_cal_data in validate."""
    nd_prior_physical = inferred_prior.distributions[0]
    if not isinstance(nd_prior_physical, torch.distributions.TransformedDistribution):
        raise ValueError(
            "Loaded ND prior is not a TransformedDistribution — it was saved with the pre-reparameterization "
            "pipeline. Regenerate the prior with the current `gen_prior` before running validate."
        )
    latent_nd = nd_prior_physical.base_dist
    T_rescale = build_rescale_bijection(cfg)
    latent_rescale = torch.distributions.TransformedDistribution(inferred_prior.distributions[1], T_rescale.inv)
    return ProductPrior(
        distributions=[latent_nd, latent_rescale],
        dims=[len(cfg.params_dict), len(cfg.rescale_params)],
    )


# ── Step 5: Inference on real experimental data ────────────────────────────
def infer_from_experiment(
    cfg: SimConfig,
    posterior: DirectPosterior | TransformedPosterior,
    X_obs: torch.Tensor,
    T_obs_s: float,
    forcing_params_si: dict,
    n_samples: int = 1000,
) -> torch.Tensor:
    """
    Infer posterior over [ND params, rescale params] from a real experimental recording.

    The recording must be sampled at 1/cfg.dt_exp (the camera frame rate the network
    was trained on). The user provides T_obs and forcing parameters in SI units;
    this function converts them to cell-file units, builds the conditioning vector
    [S(X_obs), F_dim, log(T_obs)], and samples from the trained posterior.

    :param cfg: Pipeline configuration (provides dt_exp, unit conversion, device).
    :param posterior: Trained DirectPosterior loaded from disk or freshly trained.
    :param X_obs: 1D experimental recording, shape (N_obs,), at 1/cfg.dt_exp sampling.
    :param T_obs_s: Observation duration in SECONDS.
    :param forcing_params_si: Dict with keys "amp" (N), "freq" (Hz), "phase" (rad), "offset" (N).
    :param n_samples: Number of posterior samples to draw. Defaults to 1000.
    :return: Posterior samples, shape (n_samples, nd_dim + rescale_dim), in cell-file units.
    """
    device = cfg.hw.device
    dtype = cfg.hw.dtype

    # Unit conversions: SI -> cell file units.
    # Known forcing param SI units; fall back to no conversion (dimensionless) if unknown.
    s_to_cell = cfg.get_unit_conversion_factor("s")
    T_obs = T_obs_s * s_to_cell

    # Consistency check: X_obs must be sampled at 1/dt_exp with duration T_obs.
    expected_N = int(T_obs / cfg.dt_exp)
    if abs(X_obs.shape[-1] - expected_N) > 1:
        warnings.warn(
            f"X_obs length ({X_obs.shape[-1]}) doesn't match expected from T_obs_s={T_obs_s:.4f}s "
            f"at 1/dt_exp sampling (expected ~{expected_N} points). "
            f"Check that X_obs is sampled at dt_exp={cfg.dt_exp:.6f} (cell units).",
            stacklevel=2,
        )

    # Out-of-distribution warning: compute the minimum feasible t_scale for this T_obs.
    # The NN was only trained on batches where n_fine_total <= N_ND_MAX, i.e.
    #   steady_idx + (T_k / dt_exp) * (t_scale_hi / t_scale_k) <= N_ND_MAX
    # Solving for t_scale_k given T_k = T_obs:
    t_scale_lo_prior, t_scale_hi = cfg.t_scale_bounds
    budget = N_ND_MAX - cfg.steady_idx
    if budget > 0:
        t_scale_min_feasible = (T_obs / cfg.dt_exp) * t_scale_hi / budget
        if t_scale_min_feasible > t_scale_lo_prior:
            warnings.warn(
                f"Inference out-of-distribution risk: for T_obs={T_obs_s:.2f}s, the NN was "
                f"only trained on t_scale >= {t_scale_min_feasible:.3f} (in cell file units). "
                f"If the true t_scale is below this, the posterior may extrapolate poorly.",
                stacklevel=2,
            )

    # Build forcing tensor generically: iterate cfg.force_params_dict and apply
    # appropriate SI->cell conversion per parameter name. Unknown names raise.
    _FORCING_SI_UNITS = {"amp": "N", "amp_y": "N", "freq": "Hz", "phase": None, "offset": "N"}
    forcing_t = torch.empty((1, len(cfg.force_params_dict)), dtype=dtype)
    for name in cfg.force_params_dict.keys():
        if name not in forcing_params_si:
            raise KeyError(f"forcing_params_si missing required key '{name}' "
                           f"(cell file expects: {list(cfg.force_params_dict.keys())})")
        if name not in _FORCING_SI_UNITS:
            raise ValueError(f"Unknown forcing parameter '{name}'. Known: {list(_FORCING_SI_UNITS)}. "
                             f"Add an entry to _FORCING_SI_UNITS in infer_from_experiment.")
        si_unit = _FORCING_SI_UNITS[name]
        si_val = forcing_params_si[name]
        cell_val = si_val if si_unit is None else si_val * cfg.get_unit_conversion_factor(si_unit)
        forcing_t[0, cfg.forcing_idx[name]] = cell_val

    # Reshape X_obs to (1, N_obs) and compute summary statistics with dt_exp
    X_obs_batched = X_obs.to(dtype=dtype).unsqueeze(0)
    obs_stats = pipeline.gen_stats(X_obs_batched, cfg.dt_exp, device=cfg.hw.device)

    # Build conditioning vector: [S(X_obs), forcing, log(T_obs)]
    log_T_obs = torch.tensor([[math.log(T_obs)]], dtype=dtype)
    obs_stats = torch.cat([obs_stats, forcing_t, log_T_obs], dim=-1)

    # Sample from the trained posterior
    samples = posterior.sample((n_samples,), x=obs_stats.to(device))
    return samples.cpu()
