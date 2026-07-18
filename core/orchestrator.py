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
    SimConfig, PRIOR_PATH, POSTERIOR_PATH, PLOT_PATH, T_MIN_EXP_S, T_MAX_EXP_S,
    CHUNK_LEN, N_ND_MAX, PPC_BIN_SIZE, SBC_N_CAL, STABILITY_SWEEP_ND_UNITS, TRAINING_NUM_RUNS,
    DENSITY_ESTIMATOR, NSF_HIDDEN_FEATURES, NSF_NUM_TRANSFORMS, NSF_NUM_BINS,
    TRAINING_NUM_ROUNDS, TRAINING_BATCH_SIZE, TRAINING_LEARNING_RATE,
    TRAINING_STOP_AFTER_EPOCHS, TRAINING_MAX_NUM_EPOCHS, TRAINING_SHOW_SUMMARY, REPARAM_ROTATE,
)
from . import cli
from .Helpers import helpers, visualizers, file_manager, labels
from .SBI import embedded_network, pipeline, analysis, decorrelate
from .SBI.Priors import sbi_prior_wrapper
from .SBI.reparam import (
    build_inferred_bijection, TransformedPosterior, build_rescale_bijection, _transform_device,
    build_rotated_bijection, RotatedLatentPrior, OrthogonalTransform, load_eval_bijection,
    nd_log_mask, resolved_log_params,
)

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
    Execute the SBI pipeline:
      1. Build or load the prior (ND x rescale x forcing product prior).
      2. Train or load the posterior (amortized NPE — ground-truth-free).
      3. Calibration diagnostics (SBC + expected coverage) — no chosen observation needed.
      4. Optionally infer on a chosen observation: a simulated cell (ground truth), experimental
         data, or neither. Only this step shows observation-dependent plots (GT trace, corner, PPC,
         eye test).
    """
    # 1. Prior
    prior_choice, build_new = cli.select_or_build_prior()
    inf_prior, force_prior = build_prior(cfg, prior_choice, build_new)

    # 2. Posterior (training is amortized and observation-independent)
    pos_choice, train_new = cli.select_or_train_posterior()
    posterior, pos_diagnostics = build_posterior(cfg, inf_prior, force_prior, pos_choice, train_new)
    helpers.clear_screen()

    # 3. Calibration (data-free): SBC + expected coverage
    validate_calibration(cfg, posterior, inf_prior, force_prior)

    # 4. Optional inference on a chosen observation
    mode = cli.select_inference_mode()
    if mode == "simulated":
        cell_file = cli.select_cell_file()
        cli.load_and_validate_gt(cfg, cell_file)        # inject GT values + inits, validated vs bounds
        T_obs_s = cli.get_time_params()
        cfg.T_obs = T_obs_s * cfg.get_unit_conversion_factor("s")
        if T_obs_s < T_MIN_EXP_S:
            warnings.warn(
                f"T_obs={T_obs_s:.2f}s is below the training range minimum T_MIN_EXP_S="
                f"{T_MIN_EXP_S:.2f}s; the posterior may extrapolate poorly.", stacklevel=2)
        elif T_obs_s > T_MAX_EXP_S:
            warnings.warn(
                f"T_obs={T_obs_s:.2f}s exceeds the training range maximum T_MAX_EXP_S="
                f"{T_MAX_EXP_S:.2f}s; the posterior may extrapolate poorly.", stacklevel=2)
        x_dim, obs_stats, t_dim = generate_observations(cfg)
        visualizers.plot(
            t_dim.squeeze(0).cpu().detach().numpy(),
            x_dim[0, :].cpu().detach().numpy(),
            title="Ground-truth trace",
            labels=(labels.axis_label("t", "s"), labels.axis_label("x", cfg.length_unit)),
        )
        infer_and_visualize(cfg, posterior, obs_stats, x_dim, t_dim, show_truth=True)
    elif mode == "experimental" and not cfg.has_forcing:
        # Passive recording: a single unforced trace, no drive.
        path, T_obs_s = cli.get_inference_inputs_spontaneous()
        X_obs = file_manager.load_experimental_data(path, dtype=cfg.hw.dtype)
        obs_stats, obs_data, t_dim = build_experiment_obs_spontaneous(cfg, X_obs, T_obs_s)
        infer_and_visualize(cfg, posterior, obs_stats, obs_data, t_dim, show_truth=False)
    elif mode == "experimental":
        spont_path, forced_path, T_obs_s, forcing_params_si = cli.get_inference_inputs(
            list(cfg.force_params_dict.keys()))
        X_obs_spont = file_manager.load_experimental_data(spont_path, dtype=cfg.hw.dtype)
        X_obs_forced = file_manager.load_experimental_data(forced_path, dtype=cfg.hw.dtype)
        obs_stats, obs_data, t_dim = build_experiment_obs(
            cfg, X_obs_spont, X_obs_forced, T_obs_s, forcing_params_si)
        infer_and_visualize(cfg, posterior, obs_stats, obs_data, t_dim, show_truth=False)
    # mode == "none": stop after calibration


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
    n_vars = cfg.inits_tensor.shape[-1]

    # Auto-derive n_segs based on CHUNK_LEN (per-chunk memory cap)
    n_segs_gt = max(1, math.ceil(n_fine_total / CHUNK_LEN))

    x_scale = rescale_gt[:, cfg.rescale_idx["x_scale"]].unsqueeze(1)
    x_offset = rescale_gt[:, cfg.rescale_idx["x_offset"]].unsqueeze(1) if "x_offset" in cfg.rescale_idx else 0.0
    t_offset = rescale_gt[:, cfg.rescale_idx["t_offset"]].item() if "t_offset" in cfg.rescale_idx else 0.0

    def _spont_run(force_tensor):
        x_fine = pipeline.gen_obs(
            model=cfg.model, params=cfg.params_tensor, t=t_fine, inits=cfg.inits_tensor,
            force=force_tensor, n_segs=n_segs_gt, steady_idx=cfg.steady_idx,
            state_dep_drift=cfg.state_dep_drift, dtype=cfg.hw.dtype, device=cfg.hw.device,
        )[0, :, :]
        return x_fine[:, ::subsample_factor][:, :N_obs]

    if cfg.has_forcing:
        force = pipeline.build_nondim_sin_force_tensor(forcing_gt, t_fine, rescale_gt, cfg.forcing_idx, cfg.rescale_idx)
        x_nd = _spont_run(force)                                 # forced run -> Group G
        x_nd_spont = _spont_run(torch.zeros_like(force))         # spontaneous -> Groups A-F
        x_dim = helpers.rescale(x_nd, x_scale, x_offset)
        x_spont_dim = helpers.rescale(x_nd_spont, x_scale, x_offset)
        del x_nd, x_nd_spont, force
    else:
        # No drive: a single spontaneous run; Group G is zero-padded, no forcing conditioning block.
        zero_force = torch.zeros((1, n_vars, n_fine_total), dtype=cfg.hw.dtype, device=cfg.hw.device)
        x_spont_dim = helpers.rescale(_spont_run(zero_force), x_scale, x_offset)
        x_dim = x_spont_dim                                      # the observation IS the passive trace

    # Dimensional time vector for plotting, in SECONDS (N_obs points at dt_exp spacing). t_dim is
    # display-only (never fed to gen_stats), so converting cell-time-units -> s here makes every
    # downstream trace plot seconds without per-site conversion.
    s_per_cell = 1.0 / cfg.get_unit_conversion_factor("s")   # cell time unit (e.g. ms) -> seconds
    t_dim = (torch.arange(N_obs, dtype=cfg.hw.dtype) * cfg.dt_exp + t_offset) * s_per_cell
    t_dim = t_dim.unsqueeze(0)  # (1, N_obs), seconds

    # Summary statistics + conditioning vector. Layout: [S | log(T) | forcing]; log(T) is grouped
    # with the summary pathway. Keep this order in sync with gen_training_data and build_posterior.
    log_T_obs = torch.tensor([[math.log(cfg.T_obs)]], dtype=cfg.hw.dtype)
    if cfg.has_forcing:
        obs_stats = pipeline.gen_stats(
            x_spont_dim, x_dim, cfg.dt_exp,
            forcing_gt[:, cfg.forcing_idx["amp"]], forcing_gt[:, cfg.forcing_idx["freq"]],
            forcing_gt[:, cfg.forcing_idx["phase"]], device=cfg.hw.device,
        )
        obs_stats = torch.cat([obs_stats, log_T_obs, forcing_gt.cpu()], dim=-1)
    else:
        obs_stats = pipeline.gen_stats(x_spont_dim, None, cfg.dt_exp, None, None, None,
                                       device=cfg.hw.device, spontaneous_only=True)
        obs_stats = torch.cat([obs_stats, log_T_obs], dim=-1)
    return x_dim, obs_stats, t_dim


# ── Step 2: Prior construction ──────────────────────────────────────────────
def build_prior(cfg: SimConfig, choice: str | None, build_new: bool,
                *, save: bool = True, save_name: str | None = None, fig_sink=None) -> tuple[Distribution, Distribution]:
    """
    Load an existing prior from disk, or construct a new product prior:
        ProductPrior = ND parameter prior x rescaling prior x forcing prior

    :param cfg: Pipeline configuration.
    :param choice: Filename of a saved prior, or None to build from scratch.
    :param build_new: True to construct from scratch.
    :param save: When building new, persist the ND prior (+corner PNG). Defaults True (CLI behavior).
                 Pass False to defer saving (e.g. a GUI that saves via an explicit control).
    :param save_name: Name to save under; when None (and save=True) the CLI prompt is used.
    :param fig_sink: Optional (title, fig) -> None display callback for the corner plot; None => plt.show().
    :return: A Distribution that can be sampled and scored.
    """
    # User-model guard: the bounds ND section order MUST equal the compiled param order (torch.unbind
    # binds columns positionally). A hand-edited JSON over a stale Bounds file would mis-bind silently.
    from core import registry
    if registry.is_user_model(cfg.model):
        spec = registry.get(cfg.model)
        expected = list(spec.compiled.param_names)
        actual = list(cfg.params_dict.keys())
        if actual != expected:
            raise ValueError(
                f"Model '{cfg.model}' is out of sync with its bounds file: definition uses {expected}, "
                f"bounds file lists {actual}. Re-save the model from the Settings model builder.")

    # 1. Forcing prior
    force_prior = _build_forcing_prior(cfg)

    # 2. Rescaling prior
    rescale_prior = _build_rescale_prior(cfg)

    if not build_new and choice is not None:
        nd_prior = file_manager.load_mix_dist(str(PRIOR_PATH / choice), device=cfg.hw.device)
        visualizers.visualize_dist(nd_prior, labels=cfg.labels, title="Prior (loaded)", sink=fig_sink)
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
        log_mask=nd_log_mask(cfg),   # geometric/log box on the configured ND scale params (REPARAM_LOG_PARAMS)
        dtype=cfg.hw.dtype, device=cfg.hw.device,
    )

    # Save the ND prior (GMM) with the existing serializer -- or defer (GUI) and just display it.
    if save:
        nd_name = save_name if save_name is not None else cli.prompt_save_name("ND parameter prior")
        save_prior_artifacts(nd_name, nd_prior, cfg, fig_sink=fig_sink)
    else:
        visualizers.visualize_dist(nd_prior, labels=cfg.labels, title="Prior", sink=fig_sink)

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
    if not cfg.has_forcing:
        return None                                   # no drive -> no forcing prior (spontaneous model)
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
    choice: str | None,
    train_new: bool,
    *, save: bool = True, save_name: str | None = None, fig_sink=None,
) -> tuple[TransformedPosterior, dict | None]:
    """
    Load an existing latent DirectPosterior from disk and wrap with T, or train a new one
    via NPE in latent space. Returns a TransformedPosterior whose .sample/.log_prob operate
    in physical-parameter coordinates for downstream code.

    :param save: When training new, persist <name>.pt / .rot.pt / .loss.npz / _loss.png. Defaults
                 True (CLI behavior). Pass False to defer saving (GUI saves via an explicit control).
    :param save_name: Name to save under; when None (and save=True) the CLI prompt is used.
    :param fig_sink: Optional (title, fig) -> None display callback for the training-loss curve
                     (a GUI embeds it); None keeps the CLI behavior (loss saved to PNG, not shown).
    """
    T = build_inferred_bijection(cfg)

    if not train_new and choice is not None:
        # map_location rehomes every stored tensor onto this machine's device, so a posterior trained
        # on a CUDA box (e.g. a Windows GPU) loads on a CPU/MPS-only Mac instead of raising
        # "Attempting to deserialize object on a CUDA device". sbi caches the training device in two
        # scalar attributes it does NOT refresh on load, so repoint both: .device drives sampling and
        # ._device drives log_prob (sbi DirectPosterior.log_prob builds tensors on ._device).
        posterior_latent = torch.load(str(POSTERIOR_PATH / choice),
                                      map_location=cfg.hw.device, weights_only=False)
        assert isinstance(posterior_latent, DirectPosterior)
        posterior_latent.device = posterior_latent._device = cfg.hw.device
        # Reconstruct the exact training box (+ rotation) from the <name>.rot.pt sidecar — log-mask
        # and V are self-describing, so eval is correct regardless of the current config (single
        # source of truth shared with the offline diagnostic scripts).
        T_load = load_eval_bijection(cfg, choice, POSTERIOR_PATH)
        return TransformedPosterior(posterior_latent, T_load), None

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

    # The latent ND GMM was fit in its box's coordinate. If we now train with a different ND log
    # box (REPARAM_LOG_PARAMS changed since this prior was built), physical training samples would
    # be drawn from the wrong prior. Require the loaded prior's box mask to match the config mask.
    from torch.distributions.transforms import ComposeTransform as _Compose
    from .SBI.reparam import UnitToBoxTransform as _Box
    _nd_box = next((inner for tr in nd_prior_physical.transforms
                    for inner in (tr.parts if isinstance(tr, _Compose) else [tr])
                    if isinstance(inner, _Box)), None)
    if _nd_box is not None:
        _want = nd_log_mask(cfg).to(_nd_box.log_mask.device)
        if not torch.equal(_nd_box.log_mask, _want):
            raise ValueError(
                "Loaded ND prior's log-box mask does not match config.REPARAM_LOG_PARAMS "
                f"(prior log dims={_nd_box.log_mask.tolist()}, config wants={_want.tolist()}). "
                "The latent GMM was fit in a different coordinate — REBUILD the ND prior "
                "(construct a new prior) before training a new posterior."
            )

    # Pushforward the physical rescale prior through T_rescale.inv (Issue 2a).
    T_rescale = build_rescale_bijection(cfg)
    latent_rescale = torch.distributions.TransformedDistribution(rescale_prior_physical, T_rescale.inv)

    latent_inferred_prior = ProductPrior(
        distributions=[latent_nd, latent_rescale],
        dims=[len(cfg.params_dict), len(cfg.rescale_params)],
    )

    # Optional decorrelating reparameterization (Track A): rotate the flow's latent coordinate
    # into the simulation-based Fisher eigenbasis so the well-identified-but-correlated posterior
    # is axis-aligned and the flow can calibrate it. REPARAM_ROTATE=False => V=None => plain.
    # The Fisher rotation probes a representative drive (decorrelate reads forcing_idx["amp"/…]); a
    # no-forcing model has no such params, so rotation is disabled for it. V=None is the plain pipeline.
    rotate = REPARAM_ROTATE and cfg.has_forcing
    if rotate:
        print("Computing decorrelating Fisher rotation (REPARAM_ROTATE=True)...")
        # Average the Fisher over the prior (not just GT) so the linear rotation is valid prior-wide.
        # GT-free: the rotation anchors on the prior median with a representative drive (force_prior).
        V = decorrelate.build_latent_fisher_rotation(
            cfg, T, latent_prior=latent_inferred_prior, force_prior=force_prior)
        T_train = build_rotated_bijection(T, V)
        train_prior = RotatedLatentPrior(latent_inferred_prior, V)
    else:
        V, T_train, train_prior = None, T, latent_inferred_prior

    training_params = {
        "model": cfg.model,
        "prior": train_prior,                         # <-- latent (rotated if REPARAM_ROTATE)
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
        "spontaneous_only": not cfg.has_forcing,
        "n_vars": cfg.inits_tensor.shape[-1],
        "dtype": cfg.hw.dtype,
        "device": cfg.hw.device,
    }

    # Conditioning layout is [S(x) | log(T) | forcing]. log(T) rides with the summary
    # pathway, so input_dim (the leading summary block) includes it; only the forcing
    # params form the separate forcing pathway.
    forcing_dim = len(cfg.force_params_dict)
    from .SBI.statistics import FEATURE_LABELS
    input_dim = len(FEATURE_LABELS) + 1            # n_summary_stats + log(T); observation-independent

    embedded_net = embedded_network.EmbeddedNet(
        input_dim, 3 * input_dim // 2,
        (5 * input_dim // 2, 2 * input_dim),
        forcing_dim=forcing_dim,
        forcing_layer_dims=(forcing_dim * 4, forcing_dim * 2),
        merge_layer_dim=2 * input_dim,
    )

    sbi_prior = sbi_prior_wrapper.SBIPriorWrapper(train_prior)

    # Training is amortized (TRAINING_NUM_ROUNDS=1) and observation-independent: x_obs/theta_obs only
    # feed training-time diagnostics, so we pass None — no ground-truth observation needed to train.
    theta_obs_latent = None

    posterior_latent, pos_diagnostics = pipeline.train_nn(
        training_params, model=DENSITY_ESTIMATOR, prior=sbi_prior,
        embedding_net=embedded_net, forcing_prior=force_prior,
        nd_dim=len(cfg.params_dict), forcing_idx=cfg.forcing_idx, rescale_idx=cfg.rescale_idx,
        x_obs=None, theta_obs=theta_obs_latent, num_rounds=TRAINING_NUM_ROUNDS,
        return_diagnostics=True,
        theta_transform=T_train,
        hidden_features=NSF_HIDDEN_FEATURES, num_transforms=NSF_NUM_TRANSFORMS, num_bins=NSF_NUM_BINS,
        learning_rate=TRAINING_LEARNING_RATE, stop_after_epochs=TRAINING_STOP_AFTER_EPOCHS,
        max_num_epochs=TRAINING_MAX_NUM_EPOCHS, show_train_summary=TRAINING_SHOW_SUMMARY,
        batch_size=TRAINING_BATCH_SIZE, device=cfg.hw.device,
    )

    if save:
        name = save_name if save_name is not None else cli.prompt_save_name("posterior")
        save_posterior_artifacts(name, posterior_latent, V, pos_diagnostics, cfg)

    # Display the training-loss curve (a GUI embeds it via the sink; the CLI historically saved it to
    # PNG without showing, so with no sink we do nothing here to preserve that behavior).
    if fig_sink is not None and pos_diagnostics is not None and pos_diagnostics.get("validation_loss"):
        fig_loss = visualizers.plot_training_loss(pos_diagnostics)
        if fig_loss is not None:
            fig_sink("Training loss", fig_loss)

    assert isinstance(posterior_latent, DirectPosterior)
    return TransformedPosterior(posterior_latent, T_train), pos_diagnostics


def save_prior_artifacts(name: str, nd_prior, cfg: SimConfig, *, fig_sink=None) -> None:
    """
    Persist an ND prior GMM to Resources/Priors/<name>.pt and its corner PNG to Resources/Plots.
    Shared by build_prior (CLI, save=True) and a GUI's explicit "Save prior" control. With no
    fig_sink the corner plot falls back to plt.show() (a no-op under the GUI's Agg backend).
    """
    file_manager.save_mix_dist(nd_prior, str(PRIOR_PATH / (name + ".pt")))
    visualizers.visualize_dist(nd_prior, labels=cfg.labels,
                               save_path=str(PLOT_PATH / (name + ".png")), title="Prior", sink=fig_sink)


def save_posterior_artifacts(name: str, posterior_latent, V, diagnostics: dict | None, cfg: SimConfig) -> None:
    """
    Persist a trained posterior and its companions: <name>.pt (raw latent DirectPosterior), the
    <name>.rot.pt reparam sidecar (rotation V + log params, when either is active), and the
    <name>.loss.npz curve + <name>_loss.png. Shared by build_posterior (CLI) and a GUI's explicit
    "Save posterior" control.
    """
    torch.save(posterior_latent, str(POSTERIOR_PATH / (name + ".pt")))
    # Self-describing sidecar so eval reconstructs the exact training box (log-mask + rotation V);
    # absence => legacy linear, no rotation. Shared single source of truth with the diagnostic scripts.
    log_params_used = resolved_log_params(cfg)
    if V is not None or log_params_used:
        torch.save({"V": V, "log_params": log_params_used}, str(POSTERIOR_PATH / (name + ".rot.pt")))
    # Loss curve: persisted so the convergence check is reproducible (sbi keeps it only in the trainer).
    if diagnostics is not None and diagnostics.get("validation_loss"):
        np.savez(
            str(PLOT_PATH / (name + ".loss.npz")),
            training_loss=np.asarray(diagnostics.get("training_loss", []), dtype=float),
            validation_loss=np.asarray(diagnostics.get("validation_loss", []), dtype=float),
            best_validation_loss=float(diagnostics.get("best_validation_loss") or float("nan")),
            epochs_trained=int(diagnostics.get("epochs_trained") or -1),
            stop_after_epochs=int(diagnostics.get("stop_after_epochs") or -1),
        )
        fig_loss = visualizers.plot_training_loss(diagnostics, save_path=str(PLOT_PATH / (name + "_loss.png")))
        if fig_loss is not None:
            plt.close(fig_loss)


def _observation_inits(cfg: SimConfig) -> torch.Tensor:
    """
    (1, n_vars) initial conditions for observation-side simulation (PPC / eye-test): the loaded cell's
    inits when present (simulated branch), else the model-default the training loop synthesizes
    (experimental branch has no cell; the transient washes these out).
    """
    if cfg.inits_dict:
        return cfg.inits_tensor
    from core import registry
    if registry.is_user_model(cfg.model):
        from core.SBI.Priors.user_prior import declared_inits
        return declared_inits(registry.get(cfg.model)).to(dtype=cfg.hw.dtype, device=cfg.hw.device)
    n_pos, n_prob = pipeline.INIT_SHAPES[cfg.model.lower()]
    rng = np.random.RandomState(0)
    arr = np.concatenate([rng.randint(0, 10, size=(1, n_pos)), np.zeros((1, n_prob))], axis=1)
    return torch.tensor(arr, dtype=cfg.hw.dtype, device=cfg.hw.device)


# ── Step 4a: Calibration diagnostics (data-free — no chosen observation) ─────
def _emit(fig_sink, title: str, fig) -> None:
    """Display a figure: hand it to fig_sink (a GUI canvas) when given, else fall back to the legacy
    blocking plt.show() (CLI). This keeps orchestrator.run's CLI behavior unchanged when fig_sink is None."""
    if fig_sink is not None:
        fig_sink(title, fig)
    else:
        plt.show()


def validate_calibration(cfg: SimConfig, posterior: DirectPosterior | TransformedPosterior,
                         inferred_prior: Distribution, force_prior: Distribution,
                         *, fig_sink=None) -> None:
    """
    Data-free posterior calibration: SBC (Talts 2018, marginals) + expected coverage (TARP, Lemos
    2023). Both draw their calibration set from the PRIOR (theta_star ~ prior, x_cal simulated), so
    this runs right after training with no chosen observation.

    :param inferred_prior: the actual training prior (ND x rescale product prior) — SBC draws
                           theta_star from it, not from the posterior.
    """
    t = cfg.t
    device = cfg.hw.device
    dtype = cfg.hw.dtype
    # Posterior's actual transform (rotated if REPARAM_ROTATE) so the cal prior + theta_transform match.
    T = posterior.T if isinstance(posterior, TransformedPosterior) else build_inferred_bijection(cfg)

    # Critical: draw theta_star from the PRIOR (not the posterior) for valid SBC.
    val_latent_prior = _build_latent_prior_for_validation(cfg, inferred_prior)
    # If the posterior uses a decorrelating rotation, rotate the calibration prior to match it.
    if hasattr(T, "parts") and len(T.parts) and isinstance(T.parts[0], OrthogonalTransform):
        val_latent_prior = RotatedLatentPrior(val_latent_prior, T.parts[0].M.transpose(-1, -2))
    x_cal, theta_star = analysis.gen_cal_data(
        model=cfg.model, prior=val_latent_prior,
        forcing_prior=force_prior,
        t=t, steady_idx=cfg.steady_idx, dt_nd_min=cfg.dt_nd_min, n_cal=SBC_N_CAL,
        nd_dim=len(cfg.params_dict), forcing_idx=cfg.forcing_idx, rescale_idx=cfg.rescale_idx,
        dt_exp=cfg.dt_exp, t_min_exp=cfg.t_min_exp, t_max_exp=cfg.t_max_exp,
        t_scale_bounds=cfg.t_scale_bounds,
        theta_transform=T,
        state_dep_drift=cfg.state_dep_drift,
        spontaneous_only=not cfg.has_forcing, n_vars=cfg.inits_tensor.shape[-1],
        dtype=dtype, device=device,
    )
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

    f_cdf, _ = sbc_rank_plot(ranks=ranks, num_posterior_samples=1000, plot_type="cdf",
                             parameter_labels=cfg.inferred_labels)
    f_cdf.tight_layout()
    n_sbc_rows = math.ceil(len(cfg.inferred_labels) / 4)
    f_hist, _ = sbc_rank_plot(ranks=ranks, num_posterior_samples=1000, plot_type="hist",
                              parameter_labels=cfg.inferred_labels, figsize=(16, 2.75 * n_sbc_rows))
    f_hist.tight_layout()
    if fig_sink is not None:
        fig_sink("SBC ranks (CDF)", f_cdf)
        fig_sink("SBC ranks (histogram)", f_hist)
    else:
        plt.show()   # CLI: a single blocking show for both open SBC figures (unchanged)

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
    _emit(fig_sink, "TARP coverage", plt.gcf())


# ── Step 4b: Inference visualization (requires a chosen observation) ─────────
def infer_and_visualize(cfg: SimConfig, posterior: DirectPosterior | TransformedPosterior,
                        obs_stats: torch.Tensor, obs_data: torch.Tensor, t_dim: torch.Tensor,
                        show_truth: bool, *, fig_sink=None) -> None:
    """
    Observation-dependent posterior plots for a chosen observation (a simulated ground-truth cell or
    experimental data): corner plot, posterior-predictive check (PPC), and the eye test. show_truth
    overlays the ground truth (simulated branch) or omits it (experimental branch).

    :param fig_sink: Optional (title, fig) -> None display callback (a GUI embeds the figures); when
                     None each plot falls back to the legacy blocking plt.show() (CLI unchanged).
    """
    t = cfg.t
    device = cfg.hw.device
    dtype = cfg.hw.dtype
    T_obs = cfg.T_obs
    inits = _observation_inits(cfg)

    # Corner plot
    samples = posterior.sample((1000,), x=obs_stats.to(device))
    fig, ax = pairplot(
        samples.cpu().numpy(),
        points=(np.array([cfg.ground_truth]) if show_truth else None),
        labels=cfg.inferred_labels,
    )
    _emit(fig_sink, "Posterior corner", fig)

    # PPC - Option B: sort posterior samples by t_scale, process in mini-batches
    # Each sample gets its own subsample_factor based on its t_scale; all samples
    # share physical duration T_obs at dt_exp sampling (matching the observation).
    nd_dim = len(cfg.params_dict)
    samples_nd = samples[:, :nd_dim]
    samples_rescale = samples[:, nd_dim:]
    n_samples = samples.shape[0]
    N_points_obs = int(cfg.T_obs / cfg.dt_exp)  # same for all samples

    forcing_gt = torch.tensor([[val for val, _ in cfg.force_params_dict.values()]], dtype=dtype, device=device)
    forcing_gt_expanded = forcing_gt.expand(n_samples, -1)  # (n_samples, n_forcing); empty if no forcing
    n_vars = inits.shape[-1]

    # Sort by t_scale (ascending) so each bin contains similar-scale samples
    t_scales_all = samples_rescale[:, cfg.rescale_idx["t_scale"]]
    sort_idx = torch.argsort(t_scales_all)
    inv_sort_idx = torch.argsort(sort_idx)
    samples_nd_sorted = samples_nd[sort_idx]
    samples_rescale_sorted = samples_rescale[sort_idx]

    x_dim_sorted = torch.empty((n_samples, N_points_obs), dtype=dtype, device=device)
    x_spont_sorted = torch.empty((n_samples, N_points_obs), dtype=dtype, device=device)
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

            # Per-sample downsample indices (each row uses its own subsample_factor)
            subsample_factors = torch.clamp(
                torch.round((cfg.dt_exp / bin_t_scales) / cfg.dt_nd_min), min=1
            ).long()  # (bs,)
            idx = subsample_factors.unsqueeze(1) * arange_out.unsqueeze(0)  # (bs, N_points_obs)

            x_scale_col = bin_rescale[:, cfg.rescale_idx["x_scale"]].unsqueeze(1)
            x_offset_col = bin_rescale[:, cfg.rescale_idx["x_offset"]].unsqueeze(1) if "x_offset" in cfg.rescale_idx else 0.0

            # Forced run (Group G) then spontaneous run (Groups A-F); no-forcing = spontaneous only.
            if cfg.has_forcing:
                force_bin = pipeline.build_nondim_sin_force_tensor(
                    forcing_gt.expand(bs, -1), t_fine_bin, bin_rescale, cfg.forcing_idx, cfg.rescale_idx)
                run_specs = ((force_bin, x_dim_sorted), (torch.zeros_like(force_bin), x_spont_sorted))
            else:
                force_bin = torch.zeros((bs, n_vars, t_fine_bin.shape[0]), dtype=dtype, device=device)
                run_specs = ((force_bin, x_spont_sorted),)
            for force_run, dest in run_specs:
                x_nd_bin = pipeline.gen_obs(
                    model=cfg.model, params=bin_nd, t=t_fine_bin,
                    inits=inits.expand(bs, -1),
                    force=force_run, n_segs=n_segs_bin, steady_idx=cfg.steady_idx,
                    state_dep_drift=cfg.state_dep_drift,
                    batch_size=bs, dtype=dtype, device=device,
                )[0, :, :]  # (bs, n_fine_bin - steady_idx)
                idx_c = torch.clamp(idx, max=x_nd_bin.shape[1] - 1)  # safety for OOD samples
                x_nd_ds = torch.gather(x_nd_bin, dim=1, index=idx_c)  # (bs, N_points_obs)
                dest[start:end] = x_scale_col * x_nd_ds + x_offset_col
                del x_nd_bin, x_nd_ds

            del force_bin
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # Restore original sample order
    x_spont = x_spont_sorted[inv_sort_idx]
    log_T_obs = torch.full((n_samples, 1), math.log(T_obs), dtype=dtype)
    # Layout [S | log(T) | forcing] — must match the observation in generate_observations.
    if cfg.has_forcing:
        x_dim = x_dim_sorted[inv_sort_idx]
        n_drive = x_dim.shape[0]
        sim_stats = pipeline.gen_stats(
            x_spont, x_dim, cfg.dt_exp,
            forcing_gt[:, cfg.forcing_idx["amp"]].expand(n_drive),
            forcing_gt[:, cfg.forcing_idx["freq"]].expand(n_drive),
            forcing_gt[:, cfg.forcing_idx["phase"]].expand(n_drive),
            device=device,
        )
        sim_stats = torch.cat([sim_stats, log_T_obs, forcing_gt_expanded.cpu()], dim=-1)
    else:
        x_dim = x_spont                                 # the PPC "sample trajectories" are spontaneous
        sim_stats = pipeline.gen_stats(x_spont, None, cfg.dt_exp, None, None, None,
                                       device=device, spontaneous_only=True)
        sim_stats = torch.cat([sim_stats, log_T_obs], dim=-1)
    results = analysis.posterior_predictive_check(obs_stats.squeeze(), sim_stats)
    fig_ppc = visualizers.plot_ppc(
        results,
        ground_truth=(cfg.ground_truth if show_truth else None),
        param_names=cfg.inferred_labels,
        n_samples=n_samples,
    )
    _emit(fig_sink, "Posterior predictive check", fig_ppc)

    # Eye test: central-estimate trajectories (posterior mean & median) vs ground truth.
    # The MAP (argmax-log-prob sample) is a poor summary of a wide posterior, so instead we
    # simulate the trajectories of the posterior MEAN and MEDIAN parameter vectors. Averaging
    # the sample trajectories pointwise would destructively cancel the oscillation (samples
    # differ in freq/phase), so we simulate the central PARAMETERS and keep a coherent drive
    # response. Each central vector is simulated on the same physical grid as the observation
    # (T_obs at dt_exp), mirroring one row of the per-sample PPC path above.
    def _simulate_central_trajectory(theta_central: torch.Tensor) -> np.ndarray:
        """Forced-run trajectory of a single (nd + rescale) param vector, on the obs grid."""
        theta_central = theta_central.unsqueeze(0)                       # (1, n_inferred)
        central_nd = theta_central[:, :nd_dim]
        central_rescale = theta_central[:, nd_dim:]
        t_scale_c = central_rescale[0, cfg.rescale_idx["t_scale"]].item()
        subsample_c = max(1, round((cfg.dt_exp / t_scale_c) / cfg.dt_nd_min))
        n_fine_c = min(cfg.steady_idx + N_points_obs * subsample_c, len(t))
        t_fine_c = t[:n_fine_c]
        n_segs_c = max(1, math.ceil(n_fine_c / CHUNK_LEN))
        if cfg.has_forcing:
            force_c = pipeline.build_nondim_sin_force_tensor(
                forcing_gt, t_fine_c, central_rescale, cfg.forcing_idx, cfg.rescale_idx)
        else:
            force_c = torch.zeros((1, n_vars, t_fine_c.shape[0]), dtype=dtype, device=device)
        x_nd_c = pipeline.gen_obs(
            model=cfg.model, params=central_nd, t=t_fine_c, inits=inits,
            force=force_c, n_segs=n_segs_c, steady_idx=cfg.steady_idx,
            state_dep_drift=cfg.state_dep_drift, dtype=dtype, device=device,
        )[0, :, :]                                                       # (1, n_fine_c - steady_idx)
        idx_c = torch.clamp(
            torch.arange(N_points_obs, device=device) * subsample_c, max=x_nd_c.shape[1] - 1
        )
        x_nd_c_ds = x_nd_c[:, idx_c]                                     # (1, N_points_obs)
        x_scale_c = central_rescale[:, cfg.rescale_idx["x_scale"]].unsqueeze(1)
        x_offset_c = central_rescale[:, cfg.rescale_idx["x_offset"]].unsqueeze(1) if "x_offset" in cfg.rescale_idx else 0.0
        return (x_scale_c * x_nd_c_ds + x_offset_c)[0].cpu().numpy()     # (N_points_obs,)

    with torch.no_grad():
        x_mean = _simulate_central_trajectory(samples.mean(dim=0))
        x_median = _simulate_central_trajectory(samples.median(dim=0).values)

    t_plot = t_dim.squeeze(0).cpu().numpy()
    fig = visualizers.plot_posterior_vs_truth(
        t=t_plot,
        x_true=obs_data[0, :].cpu().numpy(),
        x_mean=x_mean,
        x_median=x_median,
        x_samples=x_dim.cpu().numpy(),
        n_show=10,
        xlabel=labels.axis_label("t", "s"),
        ylabel=labels.axis_label("x", cfg.length_unit),
    )
    _emit(fig_sink, "Eye test", fig)

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
def build_experiment_obs(
    cfg: SimConfig,
    X_obs_spont: torch.Tensor,
    X_obs_forced: torch.Tensor,
    T_obs_s: float,
    forcing_params_si: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build the conditioning vector + observed trajectory from a real experimental recording, and set the
    observation context on cfg (T_obs + forcing values, in cell-file units) so infer_and_visualize's
    PPC / eye-test can simulate. Posterior sampling + the corner plot are done by infer_and_visualize.

    The recording must be sampled at 1/cfg.dt_exp (the camera frame rate the network was trained on).
    T_obs and the forcing params are given in SI units and converted to cell-file units here.

    :param X_obs_spont: 1D spontaneous (unforced) recording, shape (N_obs,), at 1/cfg.dt_exp.
    :param X_obs_forced: 1D forced (driven) recording, shape (N_obs,), at 1/cfg.dt_exp.
    :param T_obs_s: Observation duration in SECONDS.
    :param forcing_params_si: Dict with keys "amp" (N), "freq" (Hz), "phase" (rad), "offset" (N).
    :return: (obs_stats, obs_data, t_dim): the [S | log(T) | forcing] conditioning vector (1, D), the
             forced recording (1, N_obs) for the eye-test, and the dimensional time axis (1, N_obs).
    """
    dtype = cfg.hw.dtype

    # Unit conversions: SI -> cell file units.
    # Known forcing param SI units; fall back to no conversion (dimensionless) if unknown.
    s_to_cell = cfg.get_unit_conversion_factor("s")
    T_obs = T_obs_s * s_to_cell

    # Consistency check: X_obs must be sampled at 1/dt_exp with duration T_obs.
    expected_N = int(T_obs / cfg.dt_exp)
    if X_obs_spont.shape[-1] != X_obs_forced.shape[-1]:
        raise ValueError(
            f"Spontaneous and forced recordings must be the same length "
            f"({X_obs_spont.shape[-1]} vs {X_obs_forced.shape[-1]})."
        )
    if abs(X_obs_forced.shape[-1] - expected_N) > 1:
        warnings.warn(
            f"Recording length ({X_obs_forced.shape[-1]}) doesn't match expected from T_obs_s={T_obs_s:.4f}s "
            f"at 1/dt_exp sampling (expected ~{expected_N} points). "
            f"Check that both recordings are sampled at dt_exp={cfg.dt_exp:.6f} (cell units).",
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

    # Reshape both recordings to (1, N_obs) and compute summary statistics with dt_exp
    X_spont_batched = X_obs_spont.to(dtype=dtype).unsqueeze(0)
    X_forced_batched = X_obs_forced.to(dtype=dtype).unsqueeze(0)
    obs_stats = pipeline.gen_stats(
        X_spont_batched, X_forced_batched, cfg.dt_exp,
        forcing_t[:, cfg.forcing_idx["amp"]], forcing_t[:, cfg.forcing_idx["freq"]],
        forcing_t[:, cfg.forcing_idx["phase"]], device=cfg.hw.device,
    )

    # Build conditioning vector: [S(X_obs), log(T_obs), forcing]
    log_T_obs = torch.tensor([[math.log(T_obs)]], dtype=dtype)
    obs_stats = torch.cat([obs_stats, log_T_obs, forcing_t], dim=-1)

    # Record the observation context so infer_and_visualize's PPC / eye-test can simulate.
    forcing_vals = {name: float(forcing_t[0, cfg.forcing_idx[name]]) for name in cfg.force_params_dict}
    cfg.set_observation_context(T_obs, forcing_vals)

    # Dimensional time axis (in SECONDS) + observed (forced) trajectory for the eye-test.
    N_obs = X_forced_batched.shape[-1]
    s_per_cell = 1.0 / cfg.get_unit_conversion_factor("s")   # cell time unit -> seconds
    t_dim = ((torch.arange(N_obs, dtype=dtype) * cfg.dt_exp) * s_per_cell).unsqueeze(0)
    return obs_stats, X_forced_batched, t_dim


def build_experiment_obs_spontaneous(
    cfg: SimConfig, X_obs: torch.Tensor, T_obs_s: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Passive-recording variant of build_experiment_obs for a NO-FORCING model: a SINGLE unforced
    recording, no forced file, and no forcing/frequency SI units (the drive machinery is not entered).
    Conditioning is [S(A-F, Group G zeroed) | log(T_obs)], matching generate_observations' no-forcing path.

    :param X_obs: 1D passive recording, shape (N_obs,), sampled at 1/cfg.dt_exp.
    :param T_obs_s: observation duration in SECONDS.
    :return: (obs_stats, obs_data, t_dim): the [S | log(T)] conditioning vector, the recording (1, N_obs),
             and the dimensional (seconds) time axis (1, N_obs).
    """
    dtype = cfg.hw.dtype
    s_to_cell = cfg.get_unit_conversion_factor("s")
    T_obs = T_obs_s * s_to_cell

    expected_N = int(T_obs / cfg.dt_exp)
    if abs(X_obs.shape[-1] - expected_N) > 1:
        warnings.warn(
            f"Recording length ({X_obs.shape[-1]}) doesn't match expected from T_obs_s={T_obs_s:.4f}s "
            f"at 1/dt_exp sampling (expected ~{expected_N} points).", stacklevel=2)

    X_batched = X_obs.to(dtype=dtype).unsqueeze(0)
    obs_stats = pipeline.gen_stats(X_batched, None, cfg.dt_exp, None, None, None,
                                   device=cfg.hw.device, spontaneous_only=True)
    log_T_obs = torch.tensor([[math.log(T_obs)]], dtype=dtype)
    obs_stats = torch.cat([obs_stats, log_T_obs], dim=-1)      # [S | log(T)], no forcing block

    cfg.set_observation_context(T_obs, {})
    N_obs = X_batched.shape[-1]
    s_per_cell = 1.0 / cfg.get_unit_conversion_factor("s")
    t_dim = ((torch.arange(N_obs, dtype=dtype) * cfg.dt_exp) * s_per_cell).unsqueeze(0)
    return obs_stats, X_batched, t_dim
