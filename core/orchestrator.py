"""
Pipeline orchestration for the SBI pipeline.

No input() anywhere; the front ends (core/gui, core/tool) call these stages.
This module owns the pipeline flow: observe -> prior -> posterior -> validate.
"""
import importlib
import json
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
    SimConfig,
    T_MIN_EXP_S, T_MAX_EXP_S,
    CHUNK_LEN, N_ND_MAX, SBC_N_CAL, STABILITY_SWEEP_ND_UNITS, TRAINING_NUM_RUNS,
    PRIOR_SWEEP_ITERATIONS, PRIOR_SWEEP_BATCH, TRAINING_RUN_SIZE, TRAINING_CHECKPOINT_EVERY,
    PRIOR_SWEEP_MAX_SETS, PRIOR_SWEEP_STEP, PRIOR_CLUSTER_MIN_SIZE, PRIOR_CLUSTER_MIN_SAMPLES,
    REPARAM_FISHER_M, REPARAM_FISHER_DZ, REPARAM_FISHER_POINTS, CAL_N_SCALES,
    DENSITY_ESTIMATOR, NSF_HIDDEN_FEATURES, NSF_NUM_TRANSFORMS, NSF_NUM_BINS,
    TRAINING_NUM_ROUNDS, TRAINING_BATCH_SIZE, TRAINING_LEARNING_RATE,
    TRAINING_STOP_AFTER_EPOCHS, TRAINING_MAX_NUM_EPOCHS, TRAINING_SHOW_SUMMARY, FORCING_SI_UNITS,
    EYE_TEST_CYCLES,
)
from . import cli, config, forcing
from .refusals import (Refusal, require_at_least, require_between, require_choice, require_file,
                       require_positive)
from .Helpers import helpers, visualizers, file_manager, labels
from .Helpers.visualizers import thin_ticks as _thin_ticks
from .artifacts import (LoadedPrior, LoadedPosterior, LoadedObservation, LoadedCalibration,
                        LoadedInference, resolve_store)
from .artifacts.provenance import inputs_from_cfg as _inputs_from_cfg
from .artifacts.provenance import file_ref as _file_ref
from .runs import public_entry          # V1: every public stage below works on a private copy of its config
from .SBI.overlay import emit_overlay_figures as _emit_overlay_figures
from .SBI.run_guards import (_find_nd_gmm, _gmm_fingerprint,  # noqa: E402
                             _assert_prior_used_matches_posterior, _assert_prior_matches_region,
                             _assert_chi_config_is_deliberate, _log_params_for)
from .SBI import (embedded_network, pipeline, analysis, decorrelate, chi, derived, overlay, ppc,
                  truncate,
                  statistics, training_checkpoint)
from .SBI.Priors import sbi_prior_wrapper
from .SBI.reparam import (
    build_inferred_bijection, TransformedPosterior, build_rescale_bijection,
    build_rotated_bijection, RotatedLatentPrior,
    nd_log_mask, resolved_log_params,
    rotation_of, rotation_of_prior, transform_device, UnitToBoxTransform,
)

# Directories have spaces in their names, so use importlib for these imports
_scaling_mod = importlib.import_module("core.SBI.Priors.Scaling Priors.scaling_prior")
ScalingPrior = _scaling_mod.ScalingPrior

_forcing_mod = importlib.import_module("core.SBI.Priors.Forcing Priors.forcing_prior")
ForcingPrior = _forcing_mod.ForcingPrior

_product_mod = importlib.import_module("core.SBI.Priors.Product Prior.product_prior")
ProductPrior = _product_mod.ProductPrior


class PreflightWarning(UserWarning):
    """A judgement reported before or instead of refusing: out-of-distribution truth, a truth outside a
    non-amortized posterior's region, T_obs outside the training range, an HPD tighter than recommended,
    cell values the bounds ignore."""


# The GUI routes warnings.showwarning to the log pane at WARNING severity, while a print lands at info
# (core/gui/streams.py:244-247), and pytest.warns can assert one. The "always" filter defeats Python's
# once-per-location registry: without it a second inference in the same session would stay silent about
# a repeated out-of-distribution truth, which is exactly the case an operator needs told twice.
warnings.filterwarnings("always", category=PreflightWarning)


def _preflight_warn(msg: str) -> None:
    """The ONE channel a composition reports a judgement through. stacklevel=3 points the warning at the
    front end that called the composition, not at this helper."""
    warnings.warn(msg, PreflightWarning, stacklevel=3)


def _truth_outside_region(T_train, region, truth) -> list:
    """Per-direction messages for a truth outside a truncation region; [] when it is inside.

    ONE wording, two callers: build_posterior's guardrail-5 block (which prints and warns in its own
    words around these) and simulated_inference's region check. The latent inverse is taken under
    no_grad and compared on the CPU in float64, which is the dtype a region's bounds are stored in.
    """
    with torch.no_grad():
        z = T_train.inv(truth.reshape(1, -1)).detach().cpu().to(torch.float64)
    if bool(region.contains(z)[0]):
        return []
    sel = z[0, region.dims]
    bad = [f"direction {d}: truth {float(v):+.3g} outside [{float(a):.3g}, {float(b):.3g}]"
           for d, v, a, b in zip(region.dims, sel, region.lo, region.hi)
           if v < a or v > b]
    # contains() is False for a non-finite coordinate too, which neither comparison above names
    return bad or [f"direction {d}: truth {float(v)!r} not inside [{float(a):.3g}, {float(b):.3g}]"
                   for d, v, a, b in zip(region.dims, sel, region.lo, region.hi)]


# ── Step 1: Synthetic data ──────────────────────────────────────────────────
@public_entry
def generate_observations(cfg: SimConfig, *, name: str = "", note: str = "", fig_sink=None,
                          store=None) -> LoadedObservation:
    """
    Simulate a ground-truth observation matching experimental conditions, and WRITE it as an
    observation artifact (the trace figure, the payload, and a manifest carrying the source cell,
    its ground truth and its inits) so a fresh session can put itself back where inference ran.

    Simulates at fine ND resolution (dt_nd_min, stable for EM), then downsamples
    to match the physical sampling rate dt_exp and duration T_obs — exactly
    mirroring what the training loop produces.
    """
    store = resolve_store(store)
    store.assert_name_free("observation", name)      # before the simulation, not at the write
    t = cfg.t  # full pre-simulated ND time vector at dt_nd_min

    # Ground-truth rescale and forcing params as (1, n) tensors
    forcing_gt = torch.tensor([[val for val, _ in cfg.force_params_dict.values()]], dtype=cfg.hw.dtype, device=cfg.hw.device)
    rescale_gt = torch.tensor([[val for val, _ in cfg.rescale_params.values()]], dtype=cfg.hw.dtype, device=cfg.hw.device)
    # TIER 1 (a box that declares T instead of f_scale): substitute the DERIVED f_scale into T's column before anything
    # simulates. A no-op for a box that declares f_scale. `sim_rescale_idx` is what the force
    # builders and gen_chi_raw must then be given -- handed the INFERRED index they would not
    # find 'f_scale', would fall into the Hopf-style x_scale/t_scale branch, and would drive
    # at a silently wrong amplitude.
    rescale_gt = derived.to_sim_rescale(cfg.params_tensor, rescale_gt, cfg.rescale_idx,
                                       *cfg.tier1_args)

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

    # Publish the RESOLVED length (post-clipping) so downstream stages use this exact count instead
    # of re-deriving it. infer_and_visualize used to recompute int(cfg.T_obs / cfg.dt_exp), which is
    # algebraically the same but can differ by one sample -- and after the clip above it re-truncates
    # cfg.T_obs = N_obs*dt_exp back to N_obs-1. The observation and the PPC traces then had different
    # widths, and _emit_overlay_figures' shape guard dropped all five overlay figures without a word.
    cfg.n_obs = N_obs

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
            state_dep_drift=cfg.state_dep_drift, var_idx=0,
            dtype=cfg.hw.dtype, device=cfg.hw.device,
        )[0, :, :]
        return x_fine[:, ::subsample_factor][:, :N_obs]

    if cfg.has_forcing and not cfg.chi_mode:
        force = pipeline.build_nondim_sin_force_tensor(forcing_gt, t_fine, rescale_gt,
                                                      cfg.forcing_idx, cfg.sim_rescale_idx)
        x_nd = _spont_run(force)                                 # forced run -> Group G
        x_nd_spont = _spont_run(torch.zeros_like(force))         # spontaneous -> Groups A-F
        x_dim = helpers.rescale(x_nd, x_scale, x_offset)
        x_spont_dim = helpers.rescale(x_nd_spont, x_scale, x_offset)
        del x_nd, x_nd_spont, force
    else:
        # No single-frequency drive (spontaneous, or chi-mode which drives its own K-freq probe):
        # a single spontaneous run; the base observation IS the passive trace.
        zero_force = torch.zeros((1, forcing.n_force_channels(cfg.model, cfg.forcing_idx, n_vars),
                                  n_fine_total), dtype=cfg.hw.dtype, device=cfg.hw.device)
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
    if cfg.chi_mode:
        # [S(41, Group G zeroed) | log(T) | padded probe SET] -- probes at mult_k * Omega_0.
        # An OBSERVATION uses the deterministic grid, not the training sampler's jitter: this is a
        # specific measurement, and the PPC has to be able to reproduce its exact drive frequencies.
        obs_stats = pipeline.gen_stats(x_spont_dim, None, cfg.dt_exp, None, None, None,
                                       device=cfg.hw.device, spontaneous_only=True)
        obs_mults = chi.chi_multipliers_for(cfg)
        chi_block, _chi_mask = pipeline.gen_chi_block(
            cfg.model, cfg.params_tensor, rescale_gt, x_spont_dim, t_fine, cfg.inits_tensor,
            cfg.sim_rescale_idx, n_segs_gt, cfg.steady_idx, subsample_factor, N_obs, cfg.dt_exp,
            obs_mults, cfg.chi_f0, k_pad=cfg.chi_k_pad, bounds=cfg.chi_freq_bounds,
            max_cycles=cfg.chi_max_cycles,
            state_dep_drift=cfg.state_dep_drift, dtype=cfg.hw.dtype, device=cfg.hw.device)
        # Record the ABSOLUTE probe frequencies this observation was measured at, so the PPC drives
        # the same experiment rather than re-deriving frequencies from each posterior sample's own
        # f_peak -- which would simulate a different experiment and make the PPC agree for the wrong
        # reason.
        cfg.chi_obs_freqs = (obs_mults.to(cfg.hw.device)
                             * chi.peak_freq(x_spont_dim, cfg.dt_exp).median()).detach()
        obs_stats = statistics.conditioning_rows(obs_stats, cfg.T_obs, chi_block.cpu())
    elif cfg.has_forcing:
        obs_stats = pipeline.gen_stats(
            x_spont_dim, x_dim, cfg.dt_exp,
            forcing_gt[:, cfg.forcing_idx["amp"]], forcing_gt[:, cfg.forcing_idx["freq"]],
            forcing_gt[:, cfg.forcing_idx["phase"]], device=cfg.hw.device,
        )
        obs_stats = statistics.conditioning_rows(obs_stats, cfg.T_obs, forcing_gt.cpu())
    else:
        obs_stats = pipeline.gen_stats(x_spont_dim, None, cfg.dt_exp, None, None, None,
                                       device=cfg.hw.device, spontaneous_only=True)
        obs_stats = statistics.conditioning_rows(obs_stats, cfg.T_obs)

    src = getattr(cfg, "sources", {}).get("cell")
    source = {
        "kind": "simulated",
        "cell": _file_ref(src, relative_to=config.RESOURCES_ROOT) if src else None,
        "params": {k: float(v) for k, (v, _) in cfg.params_dict.items()},
        "rescale": {k: float(v) for k, (v, _) in cfg.rescale_params.items()},
        "forcing": {k: float(v) for k, (v, _) in cfg.force_params_dict.items()},
        "inits": {k: float(v) for k, v in cfg.inits_dict.items()},
        "T_obs_s": float(cfg.T_obs / cfg.get_unit_conversion_factor("s")),
    }
    forcing_vals = ({k: float(v) for k, (v, _) in cfg.force_params_dict.items()}
                    if (cfg.has_forcing and not cfg.chi_mode) else {})
    return _write_observation(store, cfg, name, note, fig_sink, obs_stats, x_dim, t_dim,
                              title="Ground-truth trace", source=source, forcing_vals=forcing_vals)


@public_entry
def build_experiment_observation(cfg: SimConfig, rec: "RecordingSet", *, name: str = "", note: str = "",
                                 fig_sink=None, store=None) -> LoadedObservation:
    """A bench recording set -> the observation artifact. Every file is checked and hashed BEFORE any
    compute (a blank or missing recording is a Refusal naming that recording's field here, not a
    traceback inside a worker), then the mode's builder runs -- refusing a bad drive before any lock-in
    -- and the artifact records the recordings, the drive and the context."""
    store = resolve_store(store)
    store.assert_name_free("observation", name)      # with the file checks, before any compute
    named = [(rec.spont, "spont", None)] + [(p, "forced", f) for p, f in rec.forced]
    # The field each role answers to, for the front ends' tables: the passive recording, forced mode's
    # one driven recording, or a chi probe's recording (picked in the probe table).
    role_field = {"spont": "recording_spont",
                  "forced": "recording_probe" if cfg.observation_mode == "chi" else "recording_forced"}
    refs = []
    for p, role, f in named:
        # "is blank" for an empty path, "was not found" for a path with no file behind it: the same rule
        # and the same sentence the Infer tab runs at the click (spec §3.3, §3.4).
        require_file(role_field[role], p, "recording")
        # D9. Role-scoped on purpose: `named`'s FIRST element is the passive recording, whose frequency
        # is None BY CONSTRUCTION, so an unscoped check would refuse every chi observation. A lock-in
        # aimed at a guessed mult_k * Omega_0 instead of the drive the bench applied decays like a sinc
        # -- a fraction of 1/T_obs off destroys the estimate, and nothing says so.
        if cfg.observation_mode == "chi" and role == "forced" and f is None:
            raise Refusal(f"chi mode: forced recording {p!r} has no drive frequency; every driven "
                          f"chi recording must state the frequency (Hz) it was driven at",
                          field="recording_probe")
        r = _file_ref(p)
        r.update({"role": role, "freq_Hz": None if f is None else float(f)})
        refs.append(r)
    X_spont = file_manager.load_experimental_data(rec.spont, dtype=cfg.hw.dtype)
    if cfg.observation_mode == "chi":
        # Every `f` is a real frequency here: the file-check loop above refused a None one (D9).
        forced = [(file_manager.load_experimental_data(p, dtype=cfg.hw.dtype), float(f))
                  for p, f in rec.forced]
        obs_stats, obs_data, t_dim = build_experiment_obs_chi(cfg, X_spont, forced, rec.T_obs_s, rec.F0_si)
    elif cfg.has_forcing:
        if len(rec.forced) != 1:
            raise Refusal(f"forced mode takes exactly one forced recording, got {len(rec.forced)}",
                          field="recording_forced")
        X_forced = file_manager.load_experimental_data(rec.forced[0][0], dtype=cfg.hw.dtype)
        obs_stats, obs_data, t_dim = build_experiment_obs(cfg, X_spont, X_forced, rec.T_obs_s,
                                                          rec.forcing_params_si or {})
    else:
        obs_stats, obs_data, t_dim = build_experiment_obs_spontaneous(cfg, X_spont, rec.T_obs_s)
    # The context _write_observation records is THIS recording's, not what the session's cfg last held:
    # the length of the trace that is plotted and eye-tested (the passive trace for spontaneous and chi,
    # the forced one for forced), which a simulated observation before it would otherwise have left.
    cfg.n_obs = int(obs_data.shape[-1])
    if cfg.observation_mode == "chi":
        # The frequencies the bench drove at, in CELL units and in rec.forced order: the units
        # generate_observations records and the lock-in above used. Without them the PPC falls back to
        # re-deriving mult_k * f_peak per posterior sample, which simulates a different experiment.
        cfg.chi_obs_freqs = torch.tensor([float(f) * cfg.freq_si_to_cell for _, f in rec.forced],
                                         dtype=cfg.hw.dtype, device=cfg.hw.device)
        cfg.chi_n_freqs = len(rec.forced)
    else:
        cfg.chi_obs_freqs = None
    forcing_vals = ({k: float(v) for k, (v, _) in cfg.force_params_dict.items()}
                    if (cfg.has_forcing and not cfg.chi_mode) else {})
    source = {"kind": "experimental", "recordings": refs, "T_obs_s": float(rec.T_obs_s),
              "F0_si": None if rec.F0_si is None else float(rec.F0_si),
              "forcing_params_si": None if rec.forcing_params_si is None else
              {k: float(v) for k, v in rec.forcing_params_si.items()}}
    return _write_observation(store, cfg, name, note, fig_sink, obs_stats, obs_data, t_dim,
                              title="Observed trace", source=source, forcing_vals=forcing_vals)


def _write_observation(store, cfg, name, note, fig_sink, x_obs, obs_data, t_dim, *, title, source, forcing_vals):
    """The one write site for both observation stages: the payload, the trace figure, and the manifest
    carrying the mode, the conditioning geometry, the digest, the context and the source. Read back
    through the loader, exactly as a later load would verify it."""
    from .artifacts.manifest import conditioning_block, tensor_digest, tensor_to_json
    digest = tensor_digest(x_obs)
    with store.create("observation", cfg, name=name, note=note) as w:
        file_manager.atomic_torch_save({"x_obs": x_obs.detach().cpu(), "obs_data": obs_data.detach().cpu(),
                                        "t_dim": t_dim.detach().cpu()}, w.payload("observation.pt"))
        visualizers.plot(t_dim.squeeze(0).cpu().detach().numpy(), obs_data[0, :].cpu().detach().numpy(),
                         title=title, labels=(labels.axis_label("t", "s"), labels.axis_label("x", cfg.length_unit)),
                         sink=w.fig_sink(fig_sink))
        w.fingerprints["x_obs"] = digest
        w.body = {"mode": cfg.observation_mode, "conditioning": conditioning_block(cfg), "x_obs_digest": digest,
                  "T_obs_cell": float(cfg.T_obs),
                  "n_obs": int(cfg.n_obs) if cfg.n_obs is not None else int(obs_data.shape[-1]),
                  "forcing_vals": forcing_vals,
                  "chi_obs_freqs": tensor_to_json(getattr(cfg, "chi_obs_freqs", None)),
                  "source": source}
    return store.load_observation(cfg, w.id)


# ── Step 2: Prior construction ──────────────────────────────────────────────




def training_identity(cfg: SimConfig, prior, run_size: int, n_runs: int, truncation=None) -> dict:
    """The simulation identity, as the dict training_checkpoint digests. A delegate to
    core.artifacts.identity.SimulationIdentity, kept under this name for the GUI's Posterior tab,
    which computes it before Train is pressed (the checkpoint line and the near-miss modal) and passes
    whatever it holds for a prior -- a LoadedPrior, or a stub that must fail open."""
    from .artifacts.identity import SimulationIdentity
    return SimulationIdentity.from_cfg(cfg, prior, run_size, n_runs, truncation=truncation).to_dict()


def _training_run_size(cfg: SimConfig, size_cap: int) -> int:
    """Rows per training batch: the hardware batch, capped. A CEILING, never a replacement.

    THE one resolution, because build_posterior and fresh_run_near_misses must agree to the row: the
    run size is part of the simulation identity, so a detector that resolved it differently would look
    in a directory the run never touches. The Posterior tab did exactly that -- `cap or _hw_batch(cfg)`
    against the stage's `min(hw, cap)` -- and with a cap above the hardware batch it asked about the
    wrong cache. Pure; the announcements stay in build_posterior.
    """
    run_size = cfg.hw.batch_size
    return min(size_cap, run_size) if size_cap else run_size


def fresh_run_near_misses(cfg: SimConfig, prior, *, num_runs=None, run_size_cap=None, truncation=None,
                          checkpoint_every=None, store=None) -> list:
    """training_checkpoint.near_miss_siblings for the identity build_posterior WOULD use. [] when
    checkpointing resolves to off, when the run's own cache already holds batches (a resume), or when
    no committed sibling is one field away. Pure; fails open for a stub prior (identity.py:30-32).

    ONE DETECTOR, shared by the training stage and the Posterior tab's dialog, so the question the
    user is asked and the refusal that follows can never disagree about which directory this run will
    touch. Its answer is advisory for the tab and binding inside the stage.
    """
    ck_every = TRAINING_CHECKPOINT_EVERY if checkpoint_every is None else int(checkpoint_every)
    if not ck_every:
        return []                            # nothing is read or written, so nothing is near
    store = resolve_store(store)
    n_runs = TRAINING_NUM_RUNS if num_runs is None else int(num_runs)
    size_cap = TRAINING_RUN_SIZE if run_size_cap is None else int(run_size_cap)
    ident = training_identity(cfg, prior, _training_run_size(cfg, size_cap), n_runs,
                              truncation=truncation)
    root = store.kind_dir("simulation")
    if (training_checkpoint.peek(training_checkpoint.resolve_dir(ident, root)) or {}).get("batches_done"):
        return []                            # this IS a resume; there is nothing to warn about
    return training_checkpoint.near_miss_siblings(ident, root)


def _near_miss_lines(ckpt_dir, near) -> str:
    """The three richest near misses, one line each: what they hold, and the single field that differs."""
    out = []
    for r in near[:3]:
        st = training_checkpoint.peek(ckpt_dir.parent / r["name"]) or {}
        out.append(f"  {r['name']}: {r['batches']:,} batches "
                   f"({'complete' if st.get('complete') else 'partial'}) -- differs only in "
                   f"{r['field']}: this run {r['mine']!r}, that cache {r['theirs']!r}")
    return "\n".join(out)


def _near_miss_message(ckpt_dir, near) -> str:
    """D7's refusal text. The spend it prevents is days of simulation, so it names the field, BOTH
    values, and the two ways out -- continue that cache, or say explicitly that this is a new run.
    Only the keyword is named: the button and the flag come from the front-end tables
    (core/gui/fields.py, core/tool/fields.py), keyed by the Refusal's field."""
    return (f"This run would start a NEW simulation cache at {ckpt_dir} from zero, but a committed "
            f"cache ONE setting away exists:\n" + _near_miss_lines(ckpt_dir, near) +
            f"\nIf you meant to continue that cache, set {near[0]['field']} back to "
            f"{near[0]['theirs']!r}. To start a new cache anyway, pass new_run=True.")


def _no_cache_message(ckpt_dir, near) -> str:
    msg = f"resume='require' but there is no resumable cache at {ckpt_dir}."
    if near:
        msg += "\nA committed cache ONE setting away exists:\n" + _near_miss_lines(ckpt_dir, near)
    return msg


@public_entry
def build_prior(cfg: SimConfig, ref: str | None, build_new: bool,
                *, name: str = "", note: str = "", fig_sink=None, store=None,
                num_iterations: int | None = None, sweep_batch: int | None = None,
                max_sets: int | None = None, walk_step: float | None = None,
                stability_units: float | None = None,
                min_cluster_size: int | None = None,
                min_samples: int | None = None) -> LoadedPrior:
    """
    Load a prior artifact by name or id, or construct a new product prior and WRITE it at once:
        ProductPrior = ND parameter prior x rescaling prior x forcing prior

    Returns a LoadedPrior -- the artifact's id and manifest travel with the prior, so a posterior
    trained from it can name its parent. A built prior is unnamed until store.rename gives it a
    name (the GUI's Save button); nothing is ever in memory only, which is what makes the
    checkpoint identity's prior fingerprint reproducible after this process ends.

    :param cfg: Pipeline configuration.
    :param ref: a prior artifact's name or id; None when building.
    :param build_new: True to construct from scratch.
    :param name: the artifact's name ("" = unnamed) when building. A name already taken is refused
                     HERE, at the entry, not by the write at the end: the stability sweep is ~9
                     minutes and there is no reason to spend it to learn the name is in use.
    :param note: free-text note recorded on the artifact when building.
    :param fig_sink: Optional (title, fig) -> None display callback for the corner plot; None => plt.show().
    :param store: the ArtifactStore to read/write; None = the process default.
    :param num_iterations: GLOBAL sweep rounds; None = config.PRIOR_SWEEP_ITERATIONS.
    :param sweep_batch: candidates per global round; None = config.PRIOR_SWEEP_BATCH (0 = follow the
                     hardware batch). ⚠ NOT a speed knob -- see the note at the constant (527 s at
                     batch 2048 against >70 min unfinished at 32; the sweep is iteration-bounded).
    :param max_sets: accepted sets that stop the LOCAL flood-fill; None = config.PRIOR_SWEEP_MAX_SETS.
    :param walk_step: flood-fill random-walk stride; None = config.PRIOR_SWEEP_STEP.
    :param stability_units: ND time units the stability screen integrates over; None =
                     config.STABILITY_SWEEP_ND_UNITS. This defines what "stable" MEANS, so changing
                     it changes the prior's support, not just how long the sweep takes.
    :param min_cluster_size: HDBSCAN floor on an island; None = config.PRIOR_CLUSTER_MIN_SIZE.
    :param min_samples: HDBSCAN density conservatism; None = config.PRIOR_CLUSTER_MIN_SAMPLES.
                     ⚠ These two are the CLUSTERING stage, not the sweep: HDBSCAN's label count is
                     handed straight to the GMM's n_components, so they set how many modes the
                     prior has. A prior with a different component count is a different prior.
    :return: A LoadedPrior wrapping the artifact's ProductPrior (ND x rescale x forcing).

    ⚠ WHY THESE ARE PARAMETERS AND NOT "JUST SET THE CONFIG CONSTANT" -- the same reason
    build_posterior's budget is: this module does `from .config import PRIOR_SWEEP_ITERATIONS, ...`,
    which SNAPSHOTS them at import, so a caller writing `config.PRIOR_SWEEP_ITERATIONS = 10` is a
    silent no-op and the sweep runs at 50 anyway with nothing to say otherwise.
    """
    store = resolve_store(store)
    # BEFORE THE SPEND, not at the write: create() refuses a duplicate name, and it runs after the
    # sweep. Uniform across every stage (see build_posterior, the observation stages, validate and
    # infer), and a no-op for the unnamed default.
    store.assert_name_free("prior", name)
    # FIRST, before the ~9-minute stability sweep: is this chi configuration the one you meant? The
    # prior itself is chi-independent, so this is here purely to fail at the START of a session
    # rather than after its first expensive stage.
    _assert_chi_config_is_deliberate(cfg)

    # User-model guard: the bounds ND section order MUST equal the compiled param order (torch.unbind
    # binds columns positionally). A hand-edited JSON over a stale Bounds file would mis-bind silently.
    from core import registry
    if registry.is_user_model(cfg.model):
        spec = registry.get(cfg.model)
        expected = list(spec.compiled.param_names)
        actual = list(cfg.params_dict.keys())
        if actual != expected:
            # field=None: the fix is a re-save in the model builder, which no inference control names.
            raise Refusal(
                f"Model '{cfg.model}' is out of sync with its bounds file: definition uses {expected}, "
                f"bounds file lists {actual}. Re-save the model from the Settings model builder.")

    if not build_new and ref is not None:
        loaded = store.load_prior(cfg, ref)
        visualizers.visualize_dist(loaded.nd_prior, labels=cfg.labels, title="Prior (loaded)", sink=fig_sink)
        return loaded

    # --- Build from scratch ---
    print("Constructing the prior from scratch.")

    # 3. ND parameter prior (stability-filtered GMM)
    # Stability is a per-parameter property — screen on a short fixed-length trajectory
    # (STABILITY_SWEEP_ND_UNITS) rather than the full master grid. Global sweep uses
    # half this (t_global_scale=2 inside gen_prior), local sweep uses the full t_stab.
    # THE SEVEN SWEEP AND CLUSTERING KNOBS, resolved here (None -> the config constant) and refused here,
    # before cfg.t is sliced and before the ~9-minute sweep (spec §3.4). Four of them used to travel as
    # None and be resolved -- min_cluster_size and min_samples CLAMPED, max(2, ...) -- inside gen_prior,
    # so a 1 became a 2 nobody chose and the manifest's knobs recorded None for values the sweep did
    # use. gen_prior and the manifest now get the resolved numbers. The load branch above returned
    # before this line, so a load never reads -- and never refuses -- any of them.
    n_iter = require_at_least("num_iterations",
                              PRIOR_SWEEP_ITERATIONS if num_iterations is None else num_iterations, 1)
    # The sweep's batch is its OWN knob (C-7), not the training batch. They were the same number,
    # and because the sweep is ITERATION-bounded rather than accept-bounded, shrinking that number for
    # a cheap run made the prior worse WITHOUT making it faster -- 527 s at batch 2048 against >70 min
    # and unfinished at batch 32. PRIOR_SWEEP_BATCH = 0 keeps the historical behaviour (follow the
    # hardware batch), which is still what a real run wants; see config for when to set it.
    sweep_batch_r = require_at_least("sweep_batch", PRIOR_SWEEP_BATCH if sweep_batch is None else sweep_batch, 0)
    max_sets_r = require_at_least("max_sets", PRIOR_SWEEP_MAX_SETS if max_sets is None else max_sets, 1)
    walk_step_r = require_positive("walk_step", PRIOR_SWEEP_STEP if walk_step is None else walk_step)
    stab_units = require_positive("stability_units",
                                  STABILITY_SWEEP_ND_UNITS if stability_units is None else stability_units)
    min_cluster_size_r = require_at_least(
        "min_cluster_size", PRIOR_CLUSTER_MIN_SIZE if min_cluster_size is None else min_cluster_size, 2)
    min_samples_r = require_at_least(
        "min_samples", PRIOR_CLUSTER_MIN_SAMPLES if min_samples is None else min_samples, 1)
    sweep_batch = sweep_batch_r or cfg.hw.batch_size          # a TYPED 0 = follow the hardware batch
    n_stab_fine = int(stab_units / cfg.dt_nd_min)
    t_stab = cfg.t[:n_stab_fine]
    prior_segs = max(1, math.ceil(n_stab_fine / CHUNK_LEN))
    nd_prior = pipeline.gen_prior(
        model=cfg.model, t=t_stab,
        global_batch_size=sweep_batch,
        local_batch_size=(sweep_batch // 2),
        segs=prior_segs,
        prior_bounds=cfg.nd_params_bounds,
        state_dep_drift=cfg.state_dep_drift,
        num_iterations=n_iter,
        n_max=max_sets_r, step=walk_step_r,
        min_cluster_size=min_cluster_size_r, min_samples=min_samples_r,
        # geometric/log box on the ND params that asked for one: a user model's own per-parameter
        # choice, else config.REPARAM_LOG_PARAMS. See _log_params_for.
        log_mask=nd_log_mask(cfg, log_params=_log_params_for(cfg)),
        dtype=cfg.hw.dtype, device=cfg.hw.device,
    )

    nd_log = nd_log_mask(cfg, log_params=_log_params_for(cfg))
    gmm = _find_nd_gmm(nd_prior)
    with store.create("prior", cfg, name=name, note=note) as w:
        file_manager.save_mix_dist(nd_prior, str(w.payload("prior.pt")),
                                   model=cfg.model, param_keys=list(cfg.params_dict.keys()))
        visualizers.visualize_dist(nd_prior, labels=cfg.labels, title="Prior", sink=w.fig_sink(fig_sink))
        w.fingerprints["gmm"] = _gmm_fingerprint(nd_prior)
        knobs = {"num_iterations": n_iter, "sweep_batch": sweep_batch, "max_sets": max_sets_r,
                 "walk_step": walk_step_r, "stability_units": stab_units,
                 "min_cluster_size": min_cluster_size_r, "min_samples": min_samples_r}
        w.config.update(knobs)
        w.body = {
            "gmm": {"n_components": int(gmm.mixture_distribution.probs.numel()) if gmm is not None else None,
                    "param_keys": list(cfg.params_dict),
                    "box": {"nd_lows": [float(b[0]) for _, b in cfg.params_dict.values()],
                            "nd_highs": [float(b[1]) for _, b in cfg.params_dict.values()],
                            "log_mask": [bool(v) for v in nd_log.tolist()]}},
            "sweep": dict(knobs),
            # gen_prior does not return the accepted count; recorded honestly as unknown.
            "stability": {"accepted_sets": None, "iterations": n_iter},
        }
    return store.load_prior(cfg, w.id)


def build_rescale_prior(cfg: SimConfig) -> Distribution:
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


def build_forcing_prior(cfg: SimConfig) -> Distribution:
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
@public_entry
def build_posterior(
    cfg: SimConfig,
    prior: LoadedPrior,                  # the built/loaded prior wrapper; .prior is the physical inferred prior
    ref: str | None,
    train_new: bool,
    *, name: str = "", note: str = "", fig_sink=None, store=None,
    num_runs: int | None = None, run_size_cap: int | None = None,
    truncation=None,
    observation=None, parent_posterior=None, accept=None,
    hidden_features: int | None = None, num_transforms: int | None = None,
    learning_rate: float | None = None, stop_after_epochs: int | None = None,
    max_num_epochs: int | None = None, checkpoint_every: int | None = None,
    resume: str = "auto", new_run: bool = False,
    fisher_m: int | None = None, fisher_dz: float | None = None,
    fisher_points: int | None = None,
) -> LoadedPosterior:
    """
    Load an existing posterior artifact through the store, or train a new one via NPE in latent
    space -- writing it to the store at completion, the moment it is read back. Either way the
    return is a LoadedPosterior whose ``.posterior`` (a TransformedPosterior) is what every
    downstream stage samples/log_probs in physical-parameter coordinates.

    :param name / note: the artifact's name ("" = unnamed) and note. Every training run is WRITTEN at
                     completion -- the Save button is a rename -- so a multi-day run can no longer be
                     lost to a forgotten click, and every posterior has an id its children can name.
    :param fig_sink: Optional (title, fig) -> None display callback for the training-loss curve
                     (a GUI embeds it); None saves the loss curve to PNG instead of showing it.
    :param store: the ArtifactStore the simulation cache and the region-fingerprint lookup read/write
                     under; None = the process default.
    :param num_runs: Training BATCHES to simulate; None (the default) = config.TRAINING_NUM_RUNS;
                     every caller passes it as an argument (the GUI, `python -m core train/smoke`).
    :param run_size_cap: CEILING on simulations per batch, 0 = follow the hardware default; None =
                     config.TRAINING_RUN_SIZE.
    :param truncation: a ``SBI.truncate.TruncationRegion`` to restrict the PRIOR to (TSNPE round 2+).
                     None = ordinary amortized NPE. The resulting artifact is marked NON-AMORTIZED in
                     its manifest and the load path refuses it for general inference.
    :param observation / parent_posterior: for a TSNPE round, the LoadedObservation the region was
                     drawn around (its ``.digest`` is what the artifact records as what it is valid
                     near) and the LoadedPosterior it was drawn from; recorded as parents.
    :param accept: LOAD path only. An ``artifacts.Accept``; a NON-AMORTIZED artifact is refused unless
                     ``accept.truncated`` (the Posterior tab passes it), and the flag is recorded.
    :param hidden_features: flow width per transform; None = config.NSF_HIDDEN_FEATURES.
    :param num_transforms: flow depth; None = config.NSF_NUM_TRANSFORMS.
    :param learning_rate: Adam LR; None = config.TRAINING_LEARNING_RATE.
    :param stop_after_epochs: early-stopping patience; None = config.TRAINING_STOP_AFTER_EPOCHS.
    :param max_num_epochs: hard epoch cap; None = config.TRAINING_MAX_NUM_EPOCHS.
    :param checkpoint_every: training batches between checkpoint writes, 0 = checkpointing OFF (no
                     simulation cache is read or written); None = config.TRAINING_CHECKPOINT_EVERY.
                     NOT part of the simulation identity -- the checkpoint header stores no cadence
                     either, so changing it never re-keys a cache.
    :param resume: "auto" (resume this run's own cache when it holds batches), "require" (refuse if
                     there is none) or "never" (refuse if there is one). Recorded as an OUTCOME in
                     the manifest (``training.resumed_from_batch``), not as a setting.
    :param new_run: consent (D7). Silences the near-miss refusal below and NOTHING else -- it never
                     forces a fresh start over a resumable cache of this run's own.
    :param fisher_m: ensemble per latent perturbation for the rotation; None =
                     config.REPARAM_FISHER_M.
    :param fisher_dz: latent central-difference step; None = config.REPARAM_FISHER_DZ.
    :param fisher_points: operating points the Fisher is averaged over; None =
                     config.REPARAM_FISHER_POINTS. n_points=1 is GT-only, which re-correlates
                     off-GT -- averaging is what makes ONE linear rotation valid prior-wide.
                     ⚠ A RESUMED run reuses the checkpoint's stored V and ignores all three
                     : V is not reproducible across processes, so a fresh one would
                     put the reused rows in a different coordinate than their stored targets.
                     ⚠ So does a TRUNCATED run (``truncation=``): it reuses the region's V and never
                     runs the Fisher at all -- the box is measured in the parent's basis and is
                     meaningless in any other (guardrail 7).
                     Either way, and on an unrotated run, the manifest records all three as None:
                     they are written only when the Fisher ran in this process (V7).
    :return: a LoadedPosterior; ``.posterior`` is the TransformedPosterior every downstream stage samples.

    ⚠ THESE FOUR ARE WHAT A COMPLETE C-11 CHECKPOINT IS FOR. Its own docstring says a finished
    checkpoint "is a cache of the whole simulation run, so you can retrain the flow at a different
    capacity/learning rate without re-simulating" -- and until 2026-08-27 there was no way to do that
    without editing config.py. Re-trying capacity costs ~46 h against ~57 h for a full run.
    ⚠ And note the 2026-08-25 characterisation ruled out more capacity on the BROKEN conditioning
    (a clean loss plateau well before the best epoch); that verdict is worth re-testing on the
    repaired conditioning, not inherited.

    ⚠ WHY THESE ARE PARAMETERS AND NOT "JUST SET THE CONFIG CONSTANT". This module does
    `from .config import TRAINING_NUM_RUNS, TRAINING_RUN_SIZE`, which SNAPSHOTS both at import -- so a
    caller writing `config.TRAINING_NUM_RUNS = 2000` is a silent no-op and the run uses 5000 anyway,
    with nothing to say otherwise. (`python -m core train` and `smoke` pass them as arguments, which
    is why the tool rebinds no module global; a GUI mutating one per run would leak across runs.)
    Passing them keeps every front end byte-identical and makes the override explicit and testable.

    ⚠ AND THEY ARE NOT INTERCHANGEABLE BUDGET KNOBS. Each batch shares ONE Sobol (t_scale_k, T_k)
    pair, overridden for every row in it -- so `num_runs` is the (t_scale, T) DIVERSITY count and the
    run size is rows per operating point. 5000x2048 and 10000x1024 have equal totals and different
    statistics (calibration has the same property). Batch WIDTH is also nearly free in
    wall-clock -- the solver is kernel-launch-bound; measured 7.37 s at 2048 against 7.74 s at 1024 --
    so narrowing it does not speed anything up, it trades training rows for peak VRAM about 1:1.
    """
    store = resolve_store(store)
    # Before anything is spent, on BOTH branches. A LOAD with a name is not a thing the GUI does, but
    # the check is cheap and uniform, and a load that would end in a refused rename is worth refusing
    # up front too. Days of training, in the training branch's case.
    store.assert_name_free("posterior", name)
    from .artifacts import Accept
    from .artifacts.manifest import conditioning_block, region_to_json, tensor_digest, tensor_to_json
    accept = accept or Accept()
    # The observation now carries its own digest (piece 1): a separately-supplied one is no longer
    # taken as a parameter, it is read straight off the LoadedObservation guardrail 2 records against.
    x_obs_digest = observation.digest if observation is not None else None
    _t0 = time.time()
    _spread = None
    inferred, force_prior = prior.prior, prior.force_prior
    # Tier 1: announce the DERIVED force scale before the first simulation, for the
    # same reason the chi banner exists -- a training distribution that changed silently is what cost
    # the 2026-08-19 run. Reports rather than refuses: whether ~1e4 pN is reasonable is a judgement
    # about the preparation, not something a threshold in this file should decide.
    if derived.uses_derived_f_scale(cfg.rescale_idx):
        try:
            _s = inferred.sample((4096,)).to("cpu")
            print(derived.describe_derived_f_scale(
                _s[:, :len(cfg.params_dict)], _s[:, len(cfg.params_dict):],
                cfg.rescale_idx, cfg.nd_idx, cfg.k_b_cell,
                chi_f0=cfg.chi_f0 if cfg.chi_mode else None), flush=True)
        except Exception as _e:                  # noqa: BLE001 -- a banner must never stop a run
            print(f"[tier1] could not describe the derived f_scale: {_e}", flush=True)

    # EVERY KNOB IS RESOLVED HERE (None -> the config constant) and refused here, before the Fisher and
    # every simulation: sbi only objects to a bad flow knob after the whole budget is spent, a zero
    # patience or learning rate does not object at all -- it writes an untrained posterior -- and
    # decorrelate's `m or REPARAM_FISHER_M` turned a Fisher 0 into the default without a word. The
    # manifest records these same resolved names.
    # Refused only on a call that TRAINS (spec §1.2, "a stage with a load branch"): the branch below
    # loads exactly when `not train_new and ref is not None`, decided from the arguments alone, and a
    # load reads none of the budget, cadence, flow or Fisher knobs. The Posterior tab sends its boxes on
    # a load too, and refusing a load over a box the load never reads is the defect the flow knobs'
    # exemption fixed first; the budget and the cadence now follow it. The resume POLICY is checked on
    # both branches.
    trains = train_new or ref is None
    # The budget, resolved before anything reads it: both are part of the checkpoint identity below.
    n_runs = TRAINING_NUM_RUNS if num_runs is None else int(num_runs)
    size_cap = TRAINING_RUN_SIZE if run_size_cap is None else int(run_size_cap)
    # The cadence, resolved beside them for the same reason: everything that decides WHETHER a
    # simulation cache is touched has to be settled before the identity is built.
    ck_every = TRAINING_CHECKPOINT_EVERY if checkpoint_every is None else int(checkpoint_every)
    max_ep = TRAINING_MAX_NUM_EPOCHS if max_num_epochs is None else int(max_num_epochs)
    hf = NSF_HIDDEN_FEATURES if hidden_features is None else int(hidden_features)
    nt = NSF_NUM_TRANSFORMS if num_transforms is None else int(num_transforms)
    lr = TRAINING_LEARNING_RATE if learning_rate is None else float(learning_rate)
    patience = TRAINING_STOP_AFTER_EPOCHS if stop_after_epochs is None else int(stop_after_epochs)
    # The rotation's three, resolved HERE and passed down resolved; the record below names them only
    # when the Fisher actually ran with them (V7).
    fm = REPARAM_FISHER_M if fisher_m is None else int(fisher_m)
    fdz = REPARAM_FISHER_DZ if fisher_dz is None else float(fisher_dz)
    fp = REPARAM_FISHER_POINTS if fisher_points is None else int(fisher_points)
    if trains:
        n_runs = require_at_least("num_runs", n_runs, 1)
        size_cap = require_at_least("run_size_cap", size_cap, 0)        # 0 = follow the hardware batch
        ck_every = require_at_least("checkpoint_every", ck_every, 0)    # 0 = checkpointing off
        hf = require_at_least("hidden_features", hf, 1)
        nt = require_at_least("num_transforms", nt, 1)
        lr = require_positive("learning_rate", lr)
        patience = require_at_least("stop_after_epochs", patience, 1)
        max_ep = require_at_least("max_num_epochs", max_ep, 1)
        fm = require_at_least("fisher_m", fm, 1)
        fdz = require_positive("fisher_dz", fdz)
        fp = require_at_least("fisher_points", fp, 1)
    resume = require_choice("resume", resume, ("auto", "require", "never"))
    if resume != "auto" and not ck_every and train_new:
        # Without this, a `resume='require'` drill with checkpointing off exits 0 having tested
        # nothing: there is no cache for the policy to act on, so the policy is silently ignored.
        # field=None: two controls answer it (the cadence and the policy), so no front end appends one.
        raise Refusal(
            f"resume={resume!r} needs checkpointing on: with checkpoint_every=0 no simulation cache "
            f"is read or written, so the policy has nothing to act on. Pass checkpoint_every=N, or "
            f"leave resume at 'auto'.")
    # Above BOTH branches, and the load branch is the subtle half. store.load_posterior compares the
    # posterior against cfg -- so a STALE cfg loading the posterior trained under that same stale cfg
    # agrees with itself and says nothing, while every inference it serves is at a retired band. This
    # check is against config.py, which is the only party to the comparison that cannot go stale.
    _assert_chi_config_is_deliberate(cfg)

    # The TRAINING bijection. Its log box must be the one gen_prior fitted the latent GMM in, hence
    # the same resolver the guard below and the manifest use (_log_params_for).
    T = build_inferred_bijection(cfg, log_params=_log_params_for(cfg))

    if not train_new and ref is not None:
        if truncation is not None:
            raise Refusal(
                "build_posterior(truncation=...) restricts the prior for a NEW training run; a loaded "
                "posterior already carries whatever region it was trained under. Load it without a "
                "region, or train a new round from it.")
        loaded = store.load_posterior(cfg, ref, accept=accept)
        region = loaded.posterior.truncation
        if region is not None:
            # The region names the base prior its parent restricted; the prior loaded beside this
            # artifact must be that one (silent when either side is unverifiable, as in training).
            _assert_prior_matches_region(region, inferred, f"Posterior '{loaded.name or loaded.id}'")
            print(f"[tsnpe] loaded a NON-AMORTIZED posterior: valid only near the observation with "
                  f"digest {loaded.posterior.x_obs_digest or '(not recorded)'} ({region!r}). "
                  f"Calibration restricts its prior to that region; inference on any other observation "
                  f"refuses unless told to accept it.", flush=True)
        return loaded

    # --- Build a LATENT product prior for SBI to train on ---
    # Physical prior layout: ProductPrior([nd_prior_physical, rescale_prior_physical]).
    # Extract latent ND (the MixtureSameFamily inside the TransformedDistribution):
    nd_prior_physical      = inferred.distributions[0]   # TransformedDistribution(latent_gmm, T_nd)
    if not isinstance(nd_prior_physical, torch.distributions.TransformedDistribution):
        raise Refusal(
            "Loaded ND prior is not a TransformedDistribution — it was saved with the pre-reparameterization "
            "pipeline. Regenerate the prior with the current `gen_prior` before training a new posterior.",
            field="prior")
    rescale_prior_physical = inferred.distributions[1]   # MultipleIndependent
    latent_nd = nd_prior_physical.base_dist              # the raw latent MixtureSameFamily

    # The latent ND GMM was fit in its box's coordinate. If we now train with a different ND log
    # box (REPARAM_LOG_PARAMS changed since this prior was built), physical training samples would
    # be drawn from the wrong prior. Require the loaded prior's box mask to match the config mask.
    from torch.distributions.transforms import ComposeTransform as _Compose
    _nd_box = next((inner for tr in nd_prior_physical.transforms
                    for inner in (tr.parts if isinstance(tr, _Compose) else [tr])
                    if isinstance(inner, UnitToBoxTransform)), None)
    if _nd_box is not None:
        _want = nd_log_mask(cfg, log_params=_log_params_for(cfg)).to(_nd_box.log_mask.device)
        if not torch.equal(_nd_box.log_mask, _want):
            _src = ("this user model's per-parameter box settings"
                    if _log_params_for(cfg) is not None else "config.REPARAM_LOG_PARAMS")
            raise Refusal(
                f"Loaded ND prior's log-box mask does not match {_src} "
                f"(prior log dims={_nd_box.log_mask.tolist()}, config wants={_want.tolist()}). "
                "The latent GMM was fit in a different coordinate — REBUILD the ND prior "
                "(construct a new prior) before training a new posterior.",
                field="prior"
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
    # Read the flag off the CONFIG, not the module: `from .config import REPARAM_ROTATE` snapshots at
    # import, so a GUI toggle could never have taken effect. Works in ALL THREE observation modes --
    # decorrelate.feats builds its Jacobian over whichever feature set the mode conditions on.
    #
    # chi mode used to be excluded here, because "chi(omega) already attacks the degeneracy the
    # rotation targets". That was never measured, and it is false: on the master cell k~x_scale is
    # 0.98 forced vs 0.95 chi (`python -m core identifiability jacobian`, 2026-08-05), i.e. chi leaves the dominant
    # alias essentially intact while improving nearly everything else. The rotation exists for that
    # alias, so chi gets one too. Cost note: the Fisher pays (1 + K) simulations per evaluation in chi
    # mode instead of 2, so a rotation costs ~(K+1)/2 x what it does in forced mode -- REPARAM_FISHER_M
    # and REPARAM_FISHER_POINTS are the knobs if that is too slow.
    # The training batch's OWN ceiling -- not hw.batch_size, and deliberately not PRIOR_SWEEP_BATCH's
    # twin. A CEILING rather than a replacement, because three pipeline tests shrink runs by writing
    # cfg.hw.batch_size directly and a replacing knob would silently override them.
    # Announced when it binds: a cap that changes the shape of a multi-day run is not allowed to be
    # silent, and the printed row count is also the check that TRAINING_NUM_RUNS was moved to match.
    # Resolved HERE, above the rotation, because the checkpoint's identity includes it.
    run_size = _training_run_size(cfg, size_cap)
    if size_cap and size_cap < cfg.hw.batch_size:
        print(f"Training batch capped at {size_cap} (hardware default {cfg.hw.batch_size}) — "
              f"{n_runs} batches x {size_cap} = {n_runs * size_cap:,} training rows.")
    if n_runs != TRAINING_NUM_RUNS:
        # Announced for the same reason the cap is: a batch count that changes the shape (and the
        # (t_scale, T) diversity) of a multi-day run is not allowed to be silent.
        print(f"Training batch COUNT overridden: {n_runs} batches (config default "
              f"{TRAINING_NUM_RUNS}) — {n_runs * run_size:,} training rows.")

    # UNCONDITIONAL, unlike the two announcements above, which fire only when a knob was overridden.
    # This is guardrail 6 ("show the cost") on the command line: the GUI has a budget group, and the
    # tool has this line. It is the one place a run states what it is about to simulate.
    print(f"[budget] {n_runs:,} batches x {run_size:,} rows = {n_runs * run_size:,} training rows",
          flush=True)

    # --- training-data checkpoint (C-11): resolved BEFORE the rotation, because a resume REUSES V ---
    # This ordering is the whole reason the resume works with rotation ON, which is how the retrain is
    # specified to run. `build_latent_fisher_rotation` seeds its noise under fork_rng but draws its
    # OPERATING POINTS from the caller's global RNG (decorrelate's z_med/z_samp), which nothing seeds
    # -- so a restarted process computes a DIFFERENT V, hence a different T_train, hence different
    # LATENT targets for every batch after the seam, silently mixed with the pre-crash rows. Trap X10.
    # Reusing the stored V also skips the Fisher entirely on a resume, which is the single largest
    # pre-training cost.
    ckpt_dir = ckpt_resumed = None
    if ck_every and train_new:
        # The region is part of the identity (omitted for an amortized run), so a TSNPE round has its
        # OWN directory and can never resume the amortized run's rows, nor the other way round (D3).
        from .artifacts.identity import SimulationIdentity
        ident = SimulationIdentity.from_cfg(cfg, prior, run_size, n_runs, truncation=truncation).to_dict()
        ckpt_dir = training_checkpoint.resolve_dir(ident, store.kind_dir("simulation"))
        _st = training_checkpoint.peek(ckpt_dir)
        # A COMPLETE checkpoint counts too, and deliberately so. Its rows are already expressed in the
        # V they were generated under, so recomputing V here would make gen_training_data's probe
        # check refuse the very rows it is about to reuse -- turning the "died during NN training,
        # don't re-simulate for days" path into a hard failure. Any committed batch pins V.
        if _st and _st.get("batches_done"):
            ckpt_resumed = training_checkpoint.read_header(ckpt_dir)
        else:
            # D7, plus the hoisted resume='require'. HERE: after the taken-name refusal (:700) and
            # before the truncation-branch refusals (:898-933) and the Fisher rotation (:999) -- before
            # any simulation, and before the cache's own create() (pipeline.py:1414), so every refusal
            # still lands before any spend. One consequence of the order: a caller who consents with
            # new_run=True can still be refused moments later by an unrelated truncation check (a
            # region with no observation digest, a rotation mismatch) -- harmless, since nothing has
            # been simulated yet. That refusal's own pin
            # (test_a_round_whose_region_names_no_observation_is_refused_before_the_spend) runs with
            # checkpointing off, so it never reaches this block. The pipeline's copy
            # of the require refusal fires only after a freshly computed rotation, which is the most
            # expensive thing this run does before it simulates; it stays there as a second line.
            near = training_checkpoint.near_miss_siblings(ident, store.kind_dir("simulation"))
            if resume == "require":
                raise Refusal(_no_cache_message(ckpt_dir, near))
            if near and not new_run:
                raise Refusal(_near_miss_message(ckpt_dir, near), field="new_run")

    rotate = cfg.reparam_rotate
    # Only the freshly-computed branch below knows the eigenvalues; a resumed checkpoint carries V but
    # not them, and an unrotated run has no Fisher at all. None is recorded honestly in the sidecar.
    fisher_evals = None
    if truncation is not None:
        # ── TSNPE: THE BASIS COMES WITH THE REGION, AND THE FISHER IS NEVER RUN (guardrail 7) ────
        # The region's dims are columns of the PARENT posterior's V, and V is not reproducible across
        # processes (trap X10: the operating points come from the unseeded global RNG). The
        # 2026-09-02 round computed a fresh V' here and enforced the parent's box in it: 0.01% of
        # the parent posterior survived, and the ground truth did not (Appendix A 2026-09-09, D1).
        # So a truncated round trains in the region's own V, refuses a checkpoint stored under any
        # other, and refuses a config whose rotation flag disagrees with the region -- a rotated
        # box has no meaning in an unrotated latent and vice versa.
        if bool(rotate) != (truncation.V is not None):
            raise Refusal(
                f"cfg.reparam_rotate is {bool(rotate)} but the truncation region was measured "
                f"{'WITH' if truncation.V is not None else 'WITHOUT'} a Fisher rotation. A truncated "
                f"round trains in the parent posterior's basis: load the config the parent was trained "
                f"with, or rebuild the region from a posterior trained under this one.")
        # GUARDRAIL 2's binding: the region already knows which observation it was drawn around
        # (build_truncation_region records it from the same observation artifact it checked the digest
        # of), so the artifact's digest is that one -- a separately supplied one must agree, and None
        # is filled in rather than recorded as "valid near observation None".
        if truncation.x_obs_digest is not None:
            if x_obs_digest is not None and x_obs_digest != truncation.x_obs_digest:
                raise Refusal(
                    f"x_obs_digest={x_obs_digest!r} does not match the observation the truncation "
                    f"region was drawn around ({truncation.x_obs_digest}); a non-amortized artifact "
                    f"must name the observation its region came from.")
            x_obs_digest = truncation.x_obs_digest
        # x_obs_digest is now final. Refuse HERE, before any simulation or training: a region built by
        # hand (build_truncation_region always sets the digest) that names no observation would
        # otherwise train and simulate the whole round only to have store.load_posterior refuse the
        # finished artifact at read-back, because guardrail 2 could never fire for it.
        if x_obs_digest is None:
            raise Refusal(
                "A non-amortized round's region must name the observation it was drawn around "
                "(x_obs_digest is None): guardrail 2 could never fire for the posterior it would "
                "produce, and the store would refuse to load it after the whole spend. Build the "
                "region through build_truncation_region, or pass the observation.")
        # The box restricts the PARENT's training prior, and check_basis sees V and the box but not
        # the GMM: supplied another prior, the round would train that prior restricted to a box
        # nobody measured on it, with every basis check green. The region carries the parent's GMM
        # fingerprint; silence only when one side is unverifiable (a pre-2026-09-10 region, a
        # stand-in prior), the same policy as validate_calibration's prior check.
        _want = getattr(truncation, "prior_fingerprint", None)
        try:
            _assert_prior_matches_region(truncation, inferred, "A truncated round")
        except ValueError as _e:
            _hit = store.find_prior_by_fingerprint(_want)
            raise Refusal(f"{_e} The region's fingerprint is that of prior '{_hit.label}' [{_hit.id}]."
                          if _hit else str(_e), field="prior") from None
        if _want is not None and _gmm_fingerprint(inferred) is not None:
            print(f"[tsnpe] prior: the parent's training prior ({_want}), "
                  f"verified against the loaded one.", flush=True)
        else:
            print("[tsnpe] prior: NOT verifiable against the parent's (the region carries no prior "
                  "fingerprint, or the loaded prior has no GMM) -- make sure the loaded prior is the "
                  "one the parent posterior was trained with.", flush=True)
        V = None if truncation.V is None else truncation.V.to(device=cfg.hw.device, dtype=cfg.hw.dtype)
        if ckpt_resumed is not None:
            # Rows may be resumed only if they were DRAWN UNDER THIS REGION: the stored identity must
            # record it (the amortized parent's never does -- its rows are the full prior's, and
            # resuming them is a silent no-op of the whole round, defect D3), and the stored V must
            # be the region's. Both checks; the identity is what keeps the parent's own V from
            # passing the second one.
            _where = f"The training checkpoint at {ckpt_dir} ({_st['batches_done']} batches)"
            _stored_region = (ckpt_resumed.get("identity") or {}).get("truncation")
            if _stored_region != truncation.identity_fields():
                raise Refusal(
                    f"{_where} was not generated under this truncation region -- its identity records "
                    f"{'no region at all (an amortized run)' if _stored_region is None else 'a different region'}"
                    f". Resuming would train a 'truncated' round on rows drawn from another proposal "
                    f"while printing that it is restricted. A truncated round resumes only its own "
                    f"checkpoint; rename that directory or change the budget so this round starts one.")
            truncation.check_checkpoint_V(ckpt_resumed.get("V"), where=_where)
        print(f"[tsnpe] basis: reusing the PARENT posterior's rotation carried with the region "
              f"({'V ' + truncate.rotation_digest(truncation.V) if V is not None else 'unrotated'}); "
              f"the Fisher is NOT recomputed for a truncated round -- a fresh V would put the box on "
              f"axes it was never measured on.", flush=True)
        T_train = build_rotated_bijection(T, V) if V is not None else T
        train_prior = RotatedLatentPrior(latent_inferred_prior, V) if V is not None else latent_inferred_prior
    elif ckpt_resumed is not None and rotate:
        # Rehomed onto this run's device/dtype. The checkpoint stores V on the CPU so it is portable,
        # but build_latent_fisher_rotation returns it on cfg.hw.device -- and OrthogonalTransform does
        # `x @ M`, which is a hard device error, not a silent promotion. Without this the FIRST GPU
        # resume with rotation ON would crash, i.e. exactly the run this feature exists to rescue.
        # Same defect the bijection probe had; found by looking for its siblings after the smoke train
        # surfaced the first one.
        V = ckpt_resumed.get("V")
        if V is not None:
            V = V.to(device=cfg.hw.device, dtype=cfg.hw.dtype)
        _done = _st["batches_done"]
        print(f"Reusing the Fisher rotation stored with the training checkpoint "
              f"({_done}/{n_runs} batches"
              f"{' — COMPLETE, so generation will be skipped' if _st.get('complete') else ''}) — NOT "
              f"recomputing it: the rotation's operating points are not reproducible across "
              f"processes, so a fresh V would put the reused rows in a different coordinate than the "
              f"targets stored beside them.")
        T_train = build_rotated_bijection(T, V) if V is not None else T
        train_prior = RotatedLatentPrior(latent_inferred_prior, V) if V is not None else latent_inferred_prior
    elif rotate:
        print("Computing decorrelating Fisher rotation (REPARAM_ROTATE=True)...")
        # Average the Fisher over the prior (not just GT) so the linear rotation is valid prior-wide.
        # GT-free: the rotation anchors on the prior median with a representative drive (force_prior).
        V, fisher_evals = decorrelate.build_latent_fisher_rotation(
            cfg, T, latent_prior=latent_inferred_prior, force_prior=force_prior, with_values=True,
            m=fm, dz=fdz, n_points=fp)
        # The eigenvalues ride into the sidecar with V. Without them the saved rotation only says
        # WHICH direction is least constrained, never BY HOW MUCH -- and recovering them afterwards
        # costs a full Fisher re-run. See `python -m core identifiability rotation`.
        _spread = float(fisher_evals[0] / fisher_evals[-1]) if float(fisher_evals[-1]) > 0 else float("inf")
        print(f"[fisher] eigenvalue spread (best/worst direction): {_spread:.3g}", flush=True)
        T_train = build_rotated_bijection(T, V)
        train_prior = RotatedLatentPrior(latent_inferred_prior, V)
    else:
        V, T_train, train_prior = None, T, latent_inferred_prior

    # ── TSNPE ─────────────────────────────────────────────────────────────────────────────────────
    # ⚠ THE PROPOSAL IS THE TRUNCATED PRIOR, NEVER THE POSTERIOR. Wrapped around `train_prior`, which
    # is the ROTATED latent prior when the rotation is on -- so the region's axes are the flow's own
    # latent axes, i.e. V's columns, i.e. the Fisher directions. Wrapping the UNROTATED prior instead
    # would silently truncate along physical-ish axes and cut the flat directions on noise, which is
    # exactly what guardrail 3 exists to prevent.
    #
    # No proposal correction is applied, and that is correct rather than an omission: truncation is a
    # RESTRICTION, not a reweighting, which is the property that distinguishes TSNPE from SNPE-A/B/C.
    if truncation is not None:
        _P = len(cfg.params_dict) + len(cfg.rescale_params)
        # The region's own guard: the bijection this round trains in must be the one the box was
        # measured in -- V compared directly (a probe cannot see a column permutation) and the probe
        # for the box. Refuses; a region in the wrong basis is the whole of defect D1.
        truncation.check_basis(T_train, dim=_P, device=cfg.hw.device)
        if cfg.has_ground_truth:
            # GUARDRAIL 5's honest failure rate, per run: does the box even contain the loaded cell's
            # truth? Warn, never refuse -- an experimental observation has no truth, and a simulated
            # one's lying outside is a finding about the parent posterior, not a reason to stop.
            # The per-direction wording is _truth_outside_region's, shared with simulated_inference:
            # one sentence for one fact, wherever the operator meets it.
            _bad = _truth_outside_region(T_train, truncation, cfg.ground_truth_tensor)
            if _bad:
                _msg = ("[tsnpe] WARNING: the loaded cell's GROUND TRUTH lies OUTSIDE the truncation "
                        "region -- " + "; ".join(_bad) + ". This round will never see a training row "
                        "near the truth. Continuing, because the truth is a simulated cell's, not the "
                        "data's -- but the parent posterior disagreed with it, and this round inherits "
                        "that.")
                print(_msg, flush=True)
                warnings.warn(_msg, stacklevel=2)
        train_prior = truncate.TruncatedLatentPrior(train_prior, truncation)
        print(f"[tsnpe] training on the PRIOR RESTRICTED to {truncation!r}", flush=True)
        print(f"[tsnpe] this artifact will be marked NON-AMORTIZED; it is valid only near the "
              f"observation its region was drawn around (digest {x_obs_digest}).", flush=True)

    training_params = pipeline.TrainingPlan(
        model=cfg.model,
        prior=train_prior,                         # <-- latent (rotated if REPARAM_ROTATE)
        t=cfg.t,
        run_size=run_size,
        num_runs=n_runs,
        steady_idx=cfg.steady_idx,
        dt_nd_min=cfg.dt_nd_min,
        dt_exp=cfg.dt_exp,
        t_min_exp=cfg.t_min_exp,
        t_max_exp=cfg.t_max_exp,
        t_scale_bounds=cfg.t_scale_bounds,
        state_dep_drift=cfg.state_dep_drift,
        spontaneous_only=not cfg.has_forcing,
        chi_mode=cfg.chi_mode,
        # No "chi_n_freqs" here on purpose. It is the count an OBSERVATION supplies; training draws K
        # per batch over [CHI_K_MIN_TRAIN, chi_k_pad] and subsets again per row, which is what makes
        # one posterior serve any probe count. It used to be threaded in and silently ignored -- an
        # invitation to "fix" gen_training_data into honouring it and destroy exactly that property.
        chi_f0=cfg.chi_f0,
        chi_freq_bounds=cfg.chi_freq_bounds,
        chi_k_pad=cfg.chi_k_pad,
        chi_max_cycles=cfg.chi_max_cycles,
        # _observation_inits, NOT cfg.inits_tensor: training is ground-truth-free, so a config built from
        # bounds alone (no cell loaded) has an empty inits_dict and cfg.inits_tensor would RAISE. The
        # fallback synthesizes the same model-default inits the training loop itself uses.
        n_vars=_observation_inits(cfg).shape[-1],
        # Tier 1. Both are None-safe downstream and are simply ignored by a box
        # that declares f_scale, so this costs pre-tier-1 runs nothing.
        nd_idx=cfg.tier1_args[0],
        k_b_cell=cfg.tier1_args[1],
        dtype=cfg.hw.dtype,
        device=cfg.hw.device,
    )
    if ckpt_dir is not None:
        # V and the probe go in AFTER the rotation, so a fresh run stores the V it just computed and
        # a resumed one stores nothing new (create() is not called on a resume).
        training_params.checkpoint = {
            "dir": ckpt_dir, "identity": ident,
            # device= is load-bearing: T_train holds the rotation V on cfg.hw.device, and a CPU grid
            # into a CUDA matmul is a hard RuntimeError inside build_posterior.
            "probe": training_checkpoint.bijection_probe(
                T_train, len(cfg.params_dict) + len(cfg.rescale_params), device=cfg.hw.device),
            "V": V, "every": ck_every, "resume": resume,
            "parents": {"prior": prior.id},
            "inputs": _inputs_from_cfg(cfg), "hw": cfg.hw,
        }

    # Conditioning layout is [S(x) | log(T) | forcing]. log(T) rides with the summary
    # pathway, so input_dim (the leading summary block) includes it; only the forcing
    # params form the separate forcing pathway.
    # chi-mode routes the padded probe SET through the EmbeddedNet's second pathway in place of the
    # single-frequency forcing block, as a permutation-invariant set encoder.
    forcing_dim = expected_forcing_dim(cfg)        # shared with the manifest + store.load_posterior's mode guard
    from .SBI.statistics import SUMMARY_WIDTH
    input_dim = SUMMARY_WIDTH + 1            # n_summary_stats + log(T); observation-independent

    embedded_net = build_embedding_net(cfg, input_dim, forcing_dim)

    sbi_prior = sbi_prior_wrapper.SBIPriorWrapper(train_prior)

    # Training is amortized (TRAINING_NUM_ROUNDS=1) and observation-independent: x_obs/theta_obs only
    # feed training-time diagnostics, so we pass None — no ground-truth observation needed to train.
    theta_obs_latent = None

    # hf, nt, lr, patience and max_ep were resolved (and range-checked) above, before any spend -- not
    # inlined into this call -- so the manifest records what was actually used.
    posterior_latent, pos_diagnostics = pipeline.train_nn(
        training_params, model=DENSITY_ESTIMATOR, prior=sbi_prior,
        embedding_net=embedded_net, forcing_prior=force_prior,
        nd_dim=len(cfg.params_dict), forcing_idx=cfg.forcing_idx, rescale_idx=cfg.rescale_idx,
        x_obs=None, theta_obs=theta_obs_latent, num_rounds=TRAINING_NUM_ROUNDS,
        return_diagnostics=True,
        theta_transform=T_train,
        hidden_features=hf, num_transforms=nt, num_bins=NSF_NUM_BINS,
        learning_rate=lr, stop_after_epochs=patience,
        max_num_epochs=max_ep, show_train_summary=TRAINING_SHOW_SUMMARY,
        batch_size=TRAINING_BATCH_SIZE, device=cfg.hw.device,
    )

    # The tsnpe kept-fraction lines stay exactly as they were (guardrail 5); they run before the write.
    _acc = _in = _tot = None
    if truncation is not None and hasattr(train_prior, "acceptance_rate"):
        # GUARDRAIL 5: the fraction of prior mass this round threw away, measured rather than
        # assumed. Deleted support is a one-way ratchet -- no later round can recover it -- so the
        # number belongs in the run log next to the artifact it produced. TWO numbers, and they
        # differ: the rejection sampler's acceptance is P(A) of the draws BEFORE gen_training_data's
        # per-batch t_scale override; the recorded containment is what the training set holds after
        # it, and only the second says how much of the set actually lies in the region (D4).
        _acc = float(train_prior.acceptance_rate)
        _in, _tot = getattr(train_prior, "recorded_counts", (0, 0))
        print(f"[tsnpe] the truncation accepted {_acc:.3%} of prior draws at the rejection sampler "
              f"(P(A), PRE-override); {1 - _acc:.3%} of the prior's mass along the truncated "
              f"directions is unavailable to later rounds.", flush=True)
        print(f"[tsnpe] post-override containment of the recorded training targets: "
              + ("not measured -- no rows were generated or loaded in this process"
                 if not _tot else f"{_in:,}/{_tot:,} = {_in / _tot:.3%}")
              + " (this is the number that says how much of the training set lies in the region).",
              flush=True)

    assert isinstance(posterior_latent, DirectPosterior)

    keys = list(cfg.params_dict) + list(cfg.rescale_params)
    V_rec = rotation_of(T_train)             # THE one decoder of the convention (D6); None when unrotated
    probe = training_checkpoint.bijection_probe(T_train, len(keys), device=cfg.hw.device)
    diag = pos_diagnostics or {}
    _best = diag.get("best_validation_loss")
    with store.create("posterior", cfg, name=name, note=note) as w:
        file_manager.atomic_torch_save(posterior_latent, w.payload("posterior.pt"))
        if diag.get("validation_loss"):
            file_manager.atomic_savez(w.payload("loss.npz"), dict(
                training_loss=np.asarray(diag.get("training_loss", []), dtype=float),
                validation_loss=np.asarray(diag.get("validation_loss", []), dtype=float),
                best_validation_loss=float(_best if _best is not None else float("nan")),
                epochs_trained=int(diag.get("epochs_trained") or -1),
                stop_after_epochs=int(diag.get("stop_after_epochs") or -1)))
            fig_loss = visualizers.plot_training_loss(diag)
            if fig_loss is not None:
                w.fig_sink(fig_sink)("Training loss", fig_loss)
        w.parents = {"prior": prior.id}
        if ckpt_dir is not None:
            w.parents["simulation"] = ckpt_dir.name
        if parent_posterior is not None:
            w.parents["parent_posterior"] = parent_posterior.id
        if observation is not None:
            w.parents["observation"] = observation.id
        w.fingerprints = {"gmm": prior.fingerprint, "V": tensor_digest(V_rec), "probe": tensor_digest(probe)}
        # V7 (spec §6.2): the three Fisher settings are recorded only when the rotation RAN in this
        # process, and fisher_evals is its one witness (only the freshly-computed branch sets it). A
        # resumed run reuses the checkpoint's V -- whose header carries no m, dz or points -- a truncated
        # round reuses the region's, and an unrotated run has no Fisher at all: each records None, never
        # the settings of a Fisher nobody computed. When it ran, the values are the ones it was called
        # with, resolved at entry.
        _fisher_ran = fisher_evals is not None
        w.config.update({"num_runs": n_runs, "run_size": run_size, "hidden_features": hf, "num_transforms": nt,
                         "learning_rate": lr, "stop_after_epochs": patience,
                         "fisher_m": fm if _fisher_ran else None,
                         "fisher_dz": fdz if _fisher_ran else None,
                         "fisher_points": fp if _fisher_ran else None,
                         "checkpoint_every": ck_every, "max_num_epochs": max_ep})
        w.body = {
            "mode": cfg.observation_mode,
            "conditioning": conditioning_block(cfg),
            "transform": {"param_keys": keys,
                          "log_params": list(resolved_log_params(cfg, log_params=_log_params_for(cfg))),
                          "nd_lows": [float(b[0]) for _, b in cfg.params_dict.values()],
                          "nd_highs": [float(b[1]) for _, b in cfg.params_dict.values()],
                          "rescale_lows": [float(b[0]) for _, b in cfg.rescale_params.values()],
                          "rescale_highs": [float(b[1]) for _, b in cfg.rescale_params.values()],
                          "V": tensor_to_json(V_rec), "V_orientation": "columns",
                          "fisher_eigenvalues": tensor_to_json(fisher_evals), "V_digest": tensor_digest(V_rec)},
            "amortized": truncation is None,
            "truncation": None if truncation is None else region_to_json(truncation),
            "training": {"n_runs": n_runs, "run_size": run_size,
                         "resumed_from_batch": int(_st["batches_done"]) if ckpt_resumed is not None else None,
                         "hidden_features": hf, "num_transforms": nt, "learning_rate": lr,
                         "stop_after_epochs": patience,
                         "epochs_trained": diag.get("epochs_trained"),
                         "best_validation_loss": (float(_best) if _best is not None and math.isfinite(float(_best)) else None),
                         "fisher_spread": (_spread if _spread is not None and math.isfinite(_spread) else None),
                         "tsnpe_acceptance": _acc,
                         "tsnpe_containment": None if _tot is None else {"inside": int(_in), "total": int(_tot)},
                         "wall_seconds": time.time() - _t0},
        }
    # Read back through the loader: the freshly written artifact is verified exactly as a later load
    # would verify it (box, width, the rotation against the pickled prior, the region's basis).
    loaded = store.load_posterior(cfg, w.id, accept=Accept(truncated=True))
    loaded.diagnostics = pos_diagnostics
    loaded.accepted = []                     # a round just trained is not an "accepted" load
    return loaded


def observation_digest(x_obs: torch.Tensor) -> str:
    """16-hex digest of a conditioning vector -- the store's tensor_digest, kept under this name."""
    from .artifacts.manifest import tensor_digest
    return tensor_digest(x_obs)


@public_entry
def build_truncation_region(posterior, observation, *,
                            n_directions: int = None, level: float = None,
                            t_scale_idx: int | None = None):
    """The TSNPE region for the NEXT round, drawn from ``posterior`` around ``observation``.

    GUARDRAIL 1 is now structural: the observation is an artifact whose payload the loader hashed
    against its manifest, so the region can only ever be drawn around a recorded observation -- the
    digest check below is a second, cheap look at the same payload the loader already checked, not a
    substitute for it (an in-session ``LoadedObservation`` cannot have been tampered with in between).
    The region also records the parent's basis (its V and a probe of its whole training transform,
    guardrail 7) and the GMM fingerprint of the parent's TRAINING prior (the prior pickled inside the
    posterior) -- the base prior its box restricts -- so ``build_posterior`` can refuse a round started
    with another prior loaded; ``check_basis`` sees V and the box, not the GMM.

    :param posterior: the parent LoadedPosterior. Its ``.posterior.T`` is the only carrier of the
                      latent basis the region is measured in, so a parent whose transform is missing
                      is refused: it cannot say which coordinate its samples are in.
    :param observation: the LoadedObservation the region is drawn around.
    :param t_scale_idx: t_scale's index in the latent, ``len(cfg.params_dict) + cfg.rescale_idx[
                        "t_scale"]``; taken from the observation's recorded parameter order when not
                        given. REQUIRED at this level: a direction that loads on t_scale is not
                        truncated, because the per-batch t_scale override would turn its box into a
                        reweighting (D4).
    """
    x_obs = observation.x_obs
    if observation_digest(x_obs) != observation.digest:
        raise Refusal(
            f"Observation '{observation.name or observation.id}' does not hash to its own digest; "
            f"a region drawn from it would delete prior support on the strength of data nobody "
            f"recorded -- and truncation is one-way, so a later round cannot undo it.",
            field="observation")
    if t_scale_idx is None:
        keys = list(observation.manifest.config.get("param_keys") or [])
        if "t_scale" not in keys:
            raise Refusal(
                "build_truncation_region needs t_scale's latent index (t_scale_idx=len(cfg.params_dict) "
                "+ cfg.rescale_idx['t_scale']) -- a direction that loads on t_scale must not be "
                "truncated (D4) -- and "
                + ("the observation carries no param_keys to derive it from." if not keys
                   else f"its recorded parameter order {keys} contains no t_scale."))
        t_scale_idx = keys.index("t_scale")
    # GUARDRAIL 7: the region records the PARENT's basis -- its rotation V and the bijection probe of
    # its whole training transform -- so the retrain can reuse that V and refuse any other. Without
    # this the box's "direction j" is a number with no coordinate attached (defect D1).
    T_parent = getattr(posterior.posterior, "T", None)
    if T_parent is None:
        raise Refusal(
            "build_truncation_region needs the parent's TransformedPosterior: its .T is the only "
            "carrier of the latent basis the region is measured in. A bare DirectPosterior cannot say "
            "which coordinate its samples are in, so a region drawn from it cannot be applied safely.",
            field="posterior")
    V = rotation_of(T_parent)
    latent = posterior.latent
    # Belt and braces against D6: the parent's transform must rotate by the rotation its own network
    # was trained under (the one pickled in its prior). A posterior loaded through build_posterior
    # has been reconciled already; an in-session one is consistent by construction; anything else
    # would hand the whole round a self-consistent wrong basis.
    _V_net = rotation_of_prior(getattr(latent, "prior", None))
    if _V_net is not None and (
            V is None or V.shape != _V_net.shape or not torch.allclose(
                V.detach().cpu().to(torch.float64), _V_net.detach().cpu().to(torch.float64), atol=1e-6)):
        raise Refusal(
            f"The parent posterior's transform "
            f"{'has no rotation' if V is None else 'does not rotate by the rotation'} "
            f"{'although' if V is None else 'that'} its network was trained under (the one inside its "
            f"training prior{'' if V is None else f', {tuple(_V_net.shape)} against {tuple(V.shape)}'}) "
            f"-- a transposed, rotation-less or foreign sidecar (D6). Reload the posterior through "
            f"build_posterior: it reconciles a transposed or missing sidecar rotation against the prior, "
            f"and refuses one that is neither.", field="posterior")
    _box = next((p for p in T_parent.parts if isinstance(p, UnitToBoxTransform)), None)
    if _box is None:
        raise Refusal("The parent posterior's transform has no parameter box; the region's probe "
                      "cannot be sized from it.")
    probe = training_checkpoint.bijection_probe(T_parent, int(_box.lows.numel()),
                                                device=transform_device(T_parent))
    # The base prior the box restricts, by fingerprint; None when the parent carries no GMM (a
    # legacy or stand-in posterior), which build_posterior treats as unverifiable, not as wrong.
    # RECORDED from the prior pickled inside the parent, exactly as before R7 -- not from
    # ``posterior.fingerprint`` -- with the two compared here: for every real wrapper they are the
    # same value by construction (store.py's load_posterior computes .fingerprint by this same walk),
    # so a mismatch means the artifact on disk is not what its own loader described.
    _walked_fp = _gmm_fingerprint(getattr(latent, "prior", None))
    _claimed_fp = posterior.fingerprint
    if _claimed_fp is not None and _walked_fp is not None and _claimed_fp != _walked_fp:
        raise Refusal(
            f"the parent wrapper claims training prior {_claimed_fp} but pickles {_walked_fp}: reload "
            f"the parent through the store (ArtifactStore.load_posterior, i.e. build_posterior's load "
            f"branch), which computes the fingerprint by this same walk. A wrapper whose fingerprint "
            f"disagrees with the prior it pickles is not a current PRISM artifact, and the region would "
            f"record the wrong base prior for the next round to be checked against.", field="posterior")
    return truncate.region_from_posterior(
        latent, x_obs.to(transform_device(T_parent)),
        n_directions=truncate.DEFAULT_N_DIRECTIONS if n_directions is None else int(n_directions),
        level=truncate.DEFAULT_HPD if level is None else float(level),
        V=V, probe=probe, x_obs_digest=observation.digest, t_scale_idx=int(t_scale_idx),
        prior_fingerprint=_walked_fp)


def expected_forcing_dim(cfg: SimConfig) -> int:
    """Width of the conditioning vector's forcing/chi block for this config. Single source of truth,
    shared by build_posterior's EmbeddedNet, the save-side sidecar and the load-side mode guard.

    Under chi this is a function of the PAD, not of the probe count -- which is the one line that buys
    K-agnosticism. A posterior trained with K drawn over 2..K_PAD loads against a config declaring any
    other probe count with NO width guard loosened anywhere.
    """
    return chi.n_chi_features(cfg.chi_k_pad) if cfg.chi_mode else len(cfg.force_params_dict)


def build_embedding_net(cfg: SimConfig, input_dim: int = None, forcing_dim: int = None):
    """The ONE construction site for the conditioning network.

    The sizing arithmetic used to be duplicated in two scripts as well as here, so a layout change
    had three places to be wrong in. Under chi the forcing pathway's hidden dims are CONSTANTS
    (the set encoder's geometry is independent of the pad, or no two pads could share a checkpoint);
    everything else keeps the original forcing_dim-derived sizing byte-for-byte.
    """
    from .SBI.statistics import SUMMARY_WIDTH
    input_dim = (SUMMARY_WIDTH + 1) if input_dim is None else input_dim
    forcing_dim = expected_forcing_dim(cfg) if forcing_dim is None else forcing_dim
    if cfg.chi_mode:
        return embedded_network.EmbeddedNet(
            input_dim, 3 * input_dim // 2, (5 * input_dim // 2, 2 * input_dim),
            forcing_dim=forcing_dim,
            forcing_layer_dims=(config.CHI_PHI_DIM, config.CHI_SET_OUT),
            merge_layer_dim=2 * input_dim,
            chi_k_pad=cfg.chi_k_pad, chi_band=cfg.chi_freq_bounds,
        )
    return embedded_network.EmbeddedNet(
        input_dim, 3 * input_dim // 2, (5 * input_dim // 2, 2 * input_dim),
        forcing_dim=forcing_dim,
        forcing_layer_dims=(forcing_dim * 4, forcing_dim * 2),
        merge_layer_dim=2 * input_dim,
    )


def check_observation_in_distribution(cfg: SimConfig, inferred_prior, force_prior,
                                      n_samples: int = 2000,
                                      lo_pct: float = 1.0, hi_pct: float = 99.0) -> list:
    """Warn when the chosen ground truth / drive sits outside the region the network was TRAINED on.

    Bounds-checking is NOT enough, and this is the check that would have caught the 2026-07-27 retrain:
      * the ND prior is a stability-SCREENED GMM, so a value can sit inside the box yet in a near-empty
        corner the training data never visited; and
      * the forcing block is deliberately not range-checked at all, so a cell with amp=freq=0 was being
        conditioned against a log-uniform drive prior that *cannot produce 0* -- a point of zero
        probability under training, where the flow can only revert to the prior.

    SAMPLING-based on purpose: the latent prior is mixed-device (cpu rescale bijection + cuda ND GMM) and
    the pipeline rule is sample-only, never ``.log_prob``.

    :return: human-readable warnings; empty when everything is comfortably in-distribution.
    """
    if not cfg.has_ground_truth:
        return []
    msgs = []

    def _flag(kind, names, values, samples):
        if samples is None or samples.numel() == 0:
            return
        s = samples.detach().to("cpu", torch.float64)
        lo = torch.quantile(s, lo_pct / 100.0, dim=0)
        hi = torch.quantile(s, hi_pct / 100.0, dim=0)
        for i, name in enumerate(names):
            # PHASE is circular: its prior is uniform over a full turn, so 0 and 2*pi are the same point
            # and a percentile band says nothing about being in-distribution. Every phase is reachable.
            if name.split("_")[0] == "phase":
                continue
            v = float(values[i])
            if not (float(lo[i]) <= v <= float(hi[i])):
                msgs.append(
                    f"{kind} '{name}' = {v:g} lies outside the training prior's {lo_pct:g}-{hi_pct:g}% "
                    f"range [{float(lo[i]):g}, {float(hi[i]):g}]. The posterior will extrapolate here "
                    f"and may simply revert to the prior for this parameter.")

    with torch.no_grad():
        try:
            theta = inferred_prior.sample((n_samples,))
        except Exception:                              # noqa: BLE001 -- a diagnostic must never break a run
            theta = None
        _flag("Ground-truth parameter", list(cfg.params_dict) + list(cfg.rescale_params),
              cfg.ground_truth, theta)

        if cfg.has_forcing and not cfg.chi_mode:
            # chi-mode ignores the cell's own drive entirely, so it cannot be out-of-distribution there.
            try:
                drive = force_prior.sample((n_samples,)) if force_prior is not None else None
            except Exception:                          # noqa: BLE001
                drive = None
            _flag("Drive parameter", list(cfg.force_params_dict),
                  [v for v, _ in cfg.force_params_dict.values()], drive)
    return msgs


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


def _calibration_prior(cfg: SimConfig, posterior: LoadedPosterior, prior: LoadedPrior):
    """The proposal a calibration set must be drawn from, and the bijection it is drawn through:
    ``(val_latent_prior, T, truncation)``.

    Factored out so ``core.diagnostics.sbc_repeats`` draws through THE SAME code. The rotation wrap,
    the basis check and the truncation wrap ARE guardrail 8, and a second copy of them is how a
    repeat-SBC run comes to draw theta* from the full prior while the flow was trained on the region.
    """
    post, inferred_prior = posterior.posterior, prior.prior
    truncation = post.truncation
    # Posterior's actual transform (rotated if REPARAM_ROTATE) so the cal prior + theta_transform match.
    T = (post.T if isinstance(post, TransformedPosterior)
         else build_inferred_bijection(cfg, log_params=_log_params_for(cfg)))

    # Critical: draw theta_star from the PRIOR (not the posterior) for valid SBC.
    val_latent_prior = _build_latent_prior_for_validation(cfg, inferred_prior)
    # If the posterior uses a decorrelating rotation, rotate the calibration prior to match it.
    # rotation_of is the ONE decoder of parts[0].M == V^T; reading the attribute here directly is
    # how the GUI's deferred save came to write V transposed (defect D6).
    _V_post = rotation_of(T)
    if _V_post is not None:
        val_latent_prior = RotatedLatentPrior(val_latent_prior, _V_post)
    if truncation is not None:
        # AFTER the rotation wrap: the region's dims index the rotated latent w = z @ V, and its basis
        # must be the one this posterior evaluates in -- the same check a training round makes.
        truncation.check_basis(T, dim=len(cfg.params_dict) + len(cfg.rescale_params), device=cfg.hw.device)
        val_latent_prior = truncate.TruncatedLatentPrior(val_latent_prior, truncation)
        print(f"[tsnpe] calibration draws theta* from the PRIOR RESTRICTED to {truncation!r}: SBC and "
              f"TARP certify calibration ON THE REGION -- they cannot tell whether the region cut real "
              f"mass at x_obs.", flush=True)
    return val_latent_prior, T, truncation


def _draw_calibration_set(cfg: SimConfig, val_latent_prior, T, force_prior, *, n_cal: int,
                          cal_n_scales: "int | None", chi_k_fixed: "int | None" = None):
    """Simulate the calibration set: ``(x_cal, theta_star)``, theta* PHYSICAL. ``T`` is always given
    here, and ``gen_cal_data`` applies it to the latent draw before returning --
    ``theta_transform(theta_star_latent)`` (core/SBI/analysis.py:197-202) -- so the caller must NOT
    push theta* through ``T`` again; a caller that trusted "latent" here would transform it twice.

    ``n_cal`` is already RESOLVED -- validate_calibration applies its SBC_N_CAL default before calling,
    and sbc_repeats has a default of its own -- so this helper never reads a module constant.

    ``chi_k_fixed`` stratifies the draw by probe count. None is the POOLED draw, over the same mixture
    of counts training saw, and is what validate_calibration always passes; ``core.diagnostics.sbc_repeats``
    passes a count to run one stratum at a time, because a pooled SBC over a mixture of counts can be
    flat while each count is miscalibrated in compensating directions.
    """
    return analysis.gen_cal_data(
        model=cfg.model, prior=val_latent_prior,
        forcing_prior=force_prior,
        t=cfg.t, steady_idx=cfg.steady_idx, dt_nd_min=cfg.dt_nd_min,
        n_cal=n_cal,
        cal_n_scales=cal_n_scales,
        nd_dim=len(cfg.params_dict), forcing_idx=cfg.forcing_idx, rescale_idx=cfg.rescale_idx,
        dt_exp=cfg.dt_exp, t_min_exp=cfg.t_min_exp, t_max_exp=cfg.t_max_exp,
        t_scale_bounds=cfg.t_scale_bounds,
        theta_transform=T,
        state_dep_drift=cfg.state_dep_drift,
        # _observation_inits: SBC/TARP draw theta from the PRIOR and need no ground truth, so this must
        # work on a cell-free config (cfg.inits_tensor would raise). See build_posterior.
        spontaneous_only=not cfg.has_forcing, chi_mode=cfg.chi_mode,
        chi_f0=cfg.chi_f0, chi_freq_bounds=cfg.chi_freq_bounds,
        chi_k_pad=cfg.chi_k_pad, chi_max_cycles=cfg.chi_max_cycles,
        chi_k_fixed=chi_k_fixed,
        n_vars=_observation_inits(cfg).shape[-1],
        nd_idx=cfg.tier1_args[0], k_b_cell=cfg.tier1_args[1],
        dtype=cfg.hw.dtype, device=cfg.hw.device,
    )


def _sbc_reference_sample(cfg: SimConfig, val_latent_prior, T, truncation, inferred_prior, theta_star):
    """check_sbc's reference sample: the draw its data-averaged-posterior test compares ranks against.

    It must come from the SAME proposal theta* did -- the data-averaged posterior converges to that
    proposal, and against the full prior c2st_dap reports a miscalibration that is not one. That
    proposal is the restricted prior mapped through the posterior's own bijection AND THEN the
    per-batch t_scale override theta* went through in gen_training_data: without mirroring it, a
    region that constrains a t_scale-loaded direction pins the reference's t_scale while theta*'s
    spans the whole schedule, and c2st_dap[t_scale] reads ~1 by construction. A permutation of
    theta*'s own t_scale column IS the schedule's marginal, drawn independently of the other
    coordinates -- exactly the override's effect.
    """
    if truncation is None:
        return inferred_prior.sample((theta_star.shape[0],)).cpu()
    with torch.no_grad():
        _z_ref = val_latent_prior.sample((theta_star.shape[0],)).to(device=cfg.hw.device, dtype=cfg.hw.dtype)
        _ref = T(_z_ref).detach().cpu()
    _i_t = len(cfg.params_dict) + cfg.rescale_idx["t_scale"]
    _ref[:, _i_t] = theta_star[torch.randperm(theta_star.shape[0]), _i_t].to(_ref)
    return _ref


# ── Step 4a: Calibration diagnostics (data-free — no chosen observation) ─────
@public_entry
def validate_calibration(cfg: SimConfig, posterior: LoadedPosterior, prior: LoadedPrior,
                         *, name: str = "", note: str = "", fig_sink=None, store=None,
                         n_cal: int | None = None, cal_n_scales: int | None = None,
                         num_posterior_samples: int = 1000) -> LoadedCalibration:
    """
    Data-free posterior calibration: SBC (Talts 2018, marginals) + expected coverage (TARP, Lemos
    2023). Both draw their calibration set from the PRIOR (theta_star ~ prior, x_cal simulated), so
    this runs right after training with no chosen observation. WRITES a calibration artifact
    (results.json, ranks.npz, the three figures, a manifest naming the posterior and prior as
    parents) inside ``store.create``, so a failure partway through leaves no half-artifact.

    ⚠ FOR A TSNPE POSTERIOR THE PRIOR IS THE TRUNCATED ONE (guardrail 8). With pt = p·1_A/P(A), NPE
    trained on pt(θ)p(x|θ) converges to pt(θ|x) = p(θ|x)·1_A(θ)/P(A|x) -- exact, but equal to p(θ|x)
    only on the set of x whose posterior mass lies in A (which the round intends to contain x_obs and
    does not verify). The absence of a proposal correction is a property of the LOSS and holds for
    every x; what is x_obs-specific is only that P(A|x_obs) ≈ 1. So SBC/TARP must draw θ* from pt:
    drawn from the full prior they measure extrapolation where the flow saw no row, and check_sbc's
    data-averaged-posterior test reports a false miscalibration by construction. What a flat result
    then certifies is calibration ON THE REGION; it cannot tell whether the region cut real mass at
    x_obs. (The failure is NOT a pair of rank spikes: the ranks here are per PHYSICAL parameter, and a
    latent box is unbounded in the untruncated directions, so ranks pile up continuously toward the
    ends in proportion to Σ_{j<k} V[i,j]².)

    THE t_scale OVERRIDE. The calibration draws pass through gen_training_data's per-batch t_scale
    override exactly as the training rows did, so the calibration proposal IS the training proposal
    -- and that proposal is the truncated prior only along directions that carry no t_scale loading.
    Along a direction that does, the override carries θ* off the box (a restriction becomes a
    reweighting by P(A | θ_-t), or a no-op when the direction IS the t_scale axis, as it is for the
    round-0 rotation's direction 0). check_sbc's reference sample therefore mirrors the override --
    its t_scale column is a permutation of θ*'s own -- so it is the proposal the flow was trained on,
    not the pure region. region_from_posterior SKIPS any direction whose |V[t_scale, j]| exceeds
    truncate.t_scale_loading_max, so the residual tilt is bounded by the summed squared loadings it
    prints; the kept fraction below is the pure region's P(A) at the rejection sampler, and the
    containment of the recorded calibration targets printed beside it is what the override left.

    :param posterior / prior: the LoadedPosterior and the LoadedPrior it was trained from; the region
                     comes off the posterior.
    :param n_cal: calibration datasets for SBC/TARP; None = config.SBC_N_CAL.
    :param cal_n_scales: (t_scale, T) operating points the calibration set is spread over; None =
                     config.CAL_N_SCALES.
                     ⚠ TRAP X5: this is `t_scale`'s EFFECTIVE SAMPLE SIZE, not a speed dial. Lowering
                     it is a DIFFERENT measurement, not a faster one -- "SBC flat on all 13" is
                     strong for 11 of them and materially weaker for `t_scale` and anything the probe
                     design controls, and this number is why.
    :param num_posterior_samples: draws per calibration point for SBC and TARP (1000 = the historical
                     constant).
    """
    store = resolve_store(store)
    store.assert_name_free("calibration", name)   # before the calibration set is simulated
    post, inferred_prior, force_prior = posterior.posterior, prior.prior, prior.force_prior
    # Resolved and refused before the calibration set is simulated: run_sbc only objects to zero draws
    # after all of it. cal_n_scales is REFUSED below 1 here, where gen_cal_data used to CLAMP it to 1:
    # it is t_scale's effective sample size (trap X5), so a count nobody typed is a different
    # measurement, not a cheaper one. The resolved count is what the set is drawn over and what the
    # artifact records.
    n_cal_used = require_at_least("n_cal", SBC_N_CAL if n_cal is None else n_cal, 1)
    nps = require_at_least("num_posterior_samples", num_posterior_samples, 1)
    n_scales_used = require_at_least("cal_n_scales", CAL_N_SCALES if cal_n_scales is None else cal_n_scales, 1)
    _assert_prior_used_matches_posterior(post, inferred_prior, "SBC/TARP calibration")
    with store.create("calibration", cfg, name=name, note=note) as w:
        device = cfg.hw.device
        dtype = cfg.hw.dtype
        # The draw is three helpers so that core.diagnostics.sbc_repeats runs through the same code --
        # the rotation wrap, the basis check, the truncation wrap and the reference sample's t_scale
        # mirror are guardrail 8, and a second copy of them is how a repeat-SBC run silently inverts it.
        val_latent_prior, T, truncation = _calibration_prior(cfg, posterior, prior)
        # chi_k_fixed stays None here: validate_calibration's SBC is the POOLED one, over the same
        # mixture of probe counts training saw. Stratifying by count is `python -m core sbc`'s
        # --chi-k-fixed (core.diagnostics.sbc_repeats), run per stratum -- a pooled SBC over a mixture
        # of counts can be flat while each count is miscalibrated in compensating directions.
        x_cal, theta_star = _draw_calibration_set(cfg, val_latent_prior, T, force_prior,
                                                  n_cal=n_cal_used, cal_n_scales=n_scales_used,
                                                  chi_k_fixed=None)
        x_cal_dev = x_cal.to(device)
        theta_star_dev = theta_star.to(device)

        # --- SBC (Talts 2018, marginals) via sbi.diagnostics ---
        ranks, dap_samples = run_sbc(
            thetas=theta_star_dev, xs=x_cal_dev, posterior=post,
            num_posterior_samples=nps, reduce_fns="marginals",
            use_batched_sampling=True, show_progress_bar=True,
        )
        prior_samples = _sbc_reference_sample(cfg, val_latent_prior, T, truncation, inferred_prior,
                                              theta_star)
        sbc_stats = check_sbc(
            ranks=ranks.cpu(), prior_samples=prior_samples, dap_samples=dap_samples.cpu(),
            num_posterior_samples=nps,
        )
        print("SBC uniformity checks:")
        for j, label in enumerate(cfg.inferred_labels):
            print(f"  {label}: KS p={sbc_stats['ks_pvals'][j]:.3f}  "
                  f"c2st_ranks={sbc_stats['c2st_ranks'][j]:.3f}  "
                  f"c2st_dap={sbc_stats['c2st_dap'][j]:.3f}")
        if truncation is not None:
            _acc = val_latent_prior.acceptance_rate
            _rec = val_latent_prior.recorded_containment
            print(f"[tsnpe] kept fraction: the region accepted {_acc:.3%} of prior draws at the rejection "
                  f"sampler (P(A), before the per-batch t_scale override, which re-opens any t_scale-loaded "
                  f"direction); {'' if _rec is None else f'{_rec:.3%} of the recorded calibration targets lie inside it after the override. '}"
                  f"The JOINT KL in the informativeness block below is measured against the "
                  f"FULL prior, so for this truncated posterior it is inflated by -log P(A) = "
                  f"{-math.log(max(_acc, 1e-300)):.2f} nats; its per-parameter entropy reductions are "
                  f"against the full prior too, each by its own offset.", flush=True)

        # Both SBC figures are grids of small panels, so give each ROW enough height for its own x-label --
        # at the previous 2.75 in/row the per-panel "posterior rank <param>" label was clipped by the row
        # beneath it -- and add explicit vertical spacing rather than relying on tight_layout alone.
        n_sbc_rows = math.ceil(len(cfg.inferred_labels) / 4)
        # sbc_rank_plot's own default is num_sbc_runs // 20 (Talts et al.'s recommendation) -- 0 for a
        # calibration set smaller than 20 (a tiny test run), which numpy/matplotlib then refuse outright.
        # Passed explicitly so it floors at 1 and is otherwise IDENTICAL to sbi's default at any
        # n_cal >= 20 (every real run: SBC_N_CAL defaults to 2000).
        num_bins = max(1, n_cal_used // 20)
        f_cdf, _ = sbc_rank_plot(ranks=ranks, num_posterior_samples=nps, plot_type="cdf", num_bins=num_bins,
                                 parameter_labels=cfg.inferred_labels, figsize=(16, 3.4 * n_sbc_rows))
        f_cdf.subplots_adjust(hspace=0.75, wspace=0.3)
        _thin_ticks(f_cdf, max_ticks=4, rotation=0)
        f_hist, _ = sbc_rank_plot(ranks=ranks, num_posterior_samples=nps, plot_type="hist", num_bins=num_bins,
                                  parameter_labels=cfg.inferred_labels, figsize=(16, 3.4 * n_sbc_rows))
        f_hist.subplots_adjust(hspace=0.75, wspace=0.3)
        _thin_ticks(f_hist, max_ticks=4, rotation=0)
        sink = w.fig_sink(fig_sink)
        sink("SBC ranks (CDF)", f_cdf)
        sink("SBC ranks (histogram)", f_hist)

        # --- Expected coverage (TARP, Lemos 2023) via sbi.diagnostics ---
        ecp, alpha_grid = run_tarp(
            thetas=theta_star_dev, xs=x_cal_dev, posterior=post,
            num_posterior_samples=nps, use_batched_sampling=True,
            z_score_theta=True, show_progress_bar=True,
        )
        atc, tarp_kspval = check_tarp(ecp.cpu(), alpha_grid.cpu())
        print(f"TARP: ATC={atc:.3f}  KS p={tarp_kspval:.3f}")
        plot_tarp(ecp.cpu(), alpha_grid.cpu(),
                  title=f"TARP (ATC={atc:.3f}, KS p={tarp_kspval:.3f})")
        sink("TARP coverage", plt.gcf())

        # --- Informativeness --------------------------------------------------------------------------
        # Everything above measures CALIBRATION, and a posterior that simply returns the prior passes all
        # of it. This is the scalar that says whether the run learned anything, on the calibration set
        # just simulated, so it costs nothing extra. Reported alongside rather than instead: a run wants
        # both numbers, and the pair is what distinguishes "honest and useful" from "honest and vacuous".
        try:
            info = analysis.informativeness(
                post, theta_star_dev, x_cal_dev, inferred_prior,
                param_names=list(cfg.params_dict) + list(cfg.rescale_params))
            print(analysis.describe_informativeness(info))
        except Exception as _e:                      # noqa: BLE001 -- a diagnostic must never lose a multi-day run's other results
            # A diagnostic must never be the thing that loses a multi-day run's other results. The
            # sample-based decomposition in particular reaches into the posterior's transform stack.
            warnings.warn(f"informativeness could not be computed ({type(_e).__name__}: {_e}); the "
                          f"calibration results above are unaffected.", stacklevel=2)
            info = None

        file_manager.atomic_savez(w.payload("ranks.npz"), {
            "ranks": ranks.detach().cpu().numpy(), "theta_star": theta_star.detach().cpu().numpy(),
            "ecp": ecp.detach().cpu().numpy(), "alpha_grid": alpha_grid.detach().cpu().numpy()})
        keys = list(cfg.params_dict) + list(cfg.rescale_params)
        results = {
            "sbc": {"per_param": [{"name": k, "ks_p": _num(sbc_stats["ks_pvals"][j]),
                                   "c2st_ranks": _num(sbc_stats["c2st_ranks"][j]),
                                   "c2st_dap": _num(sbc_stats["c2st_dap"][j])} for j, k in enumerate(keys)]},
            "tarp": {"atc": _num(atc), "ks_p": _num(tarp_kspval)},
            "informativeness": None if info is None else {
                "total_nats": _num(info["total_nats"]), "sem_nats": _num(info["sem_nats"]),
                "per_param": None if info["per_param"] is None else [_num(v) for v in info["per_param"]],
                "per_direction": None if info["per_direction"] is None else [_num(v) for v in info["per_direction"]],
                "n_used": int(info["n_used"]), "n_dropped": int(info["n_dropped"]),
                "description": analysis.describe_informativeness(info)},
            "kept_fraction": None if truncation is None else {
                "acceptance": _num(val_latent_prior.acceptance_rate),
                "containment": _num(val_latent_prior.recorded_containment)},
            "n_cal": int(n_cal_used), "cal_n_scales": n_scales_used,
            "num_posterior_samples": nps,
        }
        _write_results_json(w.payload("results.json"), results)
        w.parents = {"posterior": posterior.id, "prior": prior.id}
        w.fingerprints["gmm"] = prior.fingerprint
        w.config.update({"n_cal": int(n_cal_used), "cal_n_scales": results["cal_n_scales"], "num_posterior_samples": nps})
        w.body = {"results": results}
    return store.load_calibration(w.id)


def _write_results_json(path, payload: dict) -> None:
    """The human-readable copy of a calibration/inference body, written ATOMICALLY.

    Through file_manager's one mechanism like every other write in the store: a plain write_text can
    leave a truncated file if the process dies mid-write, and the writer would then hash that
    truncation into the manifest as the payload's sha256 -- an artifact that is permanently,
    verifiably inconsistent with itself.
    """
    text = json.dumps(payload, indent=2, allow_nan=False).encode("utf-8")
    file_manager._atomic_write(path, lambda fh: fh.write(text))


def _num(x):
    """A finite float, or None -- manifests refuse NaN/inf and a diagnostic may legitimately produce one."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


# ── Step 4b: Inference visualization (requires a chosen observation) ─────────
@public_entry
def infer_and_visualize(cfg: SimConfig, posterior: LoadedPosterior, observation: LoadedObservation,
                        *, name: str = "", note: str = "", fig_sink=None, store=None, accept=None,
                        n_samples: int = 1000) -> LoadedInference:
    """
    Observation-dependent posterior plots for a chosen observation (a simulated ground-truth cell or
    experimental data): corner plot, posterior-predictive check (PPC), and the eye test. WRITES an
    inference artifact (samples.pt, results.json, every figure) inside ``store.create``, so a failure
    partway through leaves no half-artifact, naming the posterior and observation as parents.

    :param posterior / observation: the wrappers; the observation's context (T_obs, n_obs, the drive,
                     a simulated cell's truth) is put back on cfg by observation.install, so this runs
                     in a fresh session too. show_truth = the observation is a simulated cell.
    :param accept: Accept(other_observation=True) lets a NON-AMORTIZED posterior run on an observation
                     other than its region's; the flag is recorded in the artifact.
    :param n_samples: posterior draws for the corner, the PPC and the summary (1000 = the historical
                     constant).
    :param fig_sink: Optional (title, fig) -> None display callback (a GUI embeds the figures); every
                     front end passes one, and None is the bare-library fallback to plt.show().
    """
    from .artifacts import Accept
    store = resolve_store(store)
    store.assert_name_free("inference", name)     # before the PPC's simulations
    _refuse_no_samples(n_samples)
    accept = accept or Accept()
    post = posterior.posterior
    # NOT installed yet: both refusals below come first. show_truth reads the MANIFEST, so it does not
    # need the install either.
    show_truth = observation.manifest.body["source"]["kind"] == "simulated"
    p_body = posterior.manifest.body
    if observation.mode != p_body["mode"] or observation.width != int(p_body["conditioning"]["width"]):
        raise Refusal(
            f"Observation '{observation.name or observation.id}' is {observation.mode} / {observation.width} wide, "
            f"but posterior '{posterior.name or posterior.id}' conditions on {p_body['mode']} / "
            f"{p_body['conditioning']['width']}. They do not describe the same measurement.",
            field="observation")
    # GUARDRAIL 2 at the one place every inference passes through -- a REFUSAL now, not a warning:
    # outside its region a truncated flow extrapolates confidently. Accept(other_observation=True)
    # is the recorded exception (a simulated cell re-drawn with new noise is the legitimate case).
    _want, accepted = post.x_obs_digest, []
    if _want is not None and observation.digest != _want:
        _msg = (f"[tsnpe] this posterior is NOT AMORTIZED. TSNPE trained it on a prior restricted to a "
                f"region drawn around the observation with digest {_want}; the observation supplied has digest "
                f"{observation.digest}. Near that observation (the same cell re-simulated with new noise, say) it is "
                f"valid; anywhere else the flow has never seen a training row and extrapolates confidently rather "
                f"than returning the prior.")
        if not accept.other_observation:
            raise Refusal(_msg + " Use the recorded observation or an amortized posterior. To run anyway, "
                          "pass Accept(other_observation=True); the inference will record it.",
                          field="accept_other_observation")
        print(_msg + " Running anyway (accepted).", flush=True)
        warnings.warn(_msg, stacklevel=2)
        accepted = accept.used()
    # ONLY NOW: a refused inference must not leave the rejected observation's T_obs, probe frequencies
    # or ground truth on the session's cfg, where the next stage would silently run against them.
    observation.install(cfg)
    t, device, dtype, T_obs = cfg.t, cfg.hw.device, cfg.hw.dtype, cfg.T_obs
    inits = _observation_inits(cfg)
    # obs_stats stays on the CPU: it is a conditioning row, and statistics.conditioning_rows builds
    # every row on the CPU by contract -- the PPC below (analysis.posterior_predictive_check) and the
    # overlay ranking (overlay.rank_by_stats) compare it against sim_stats assembled there. Only the
    # flow's sample() needs it on the device, and it gets its own copy. Binding a device copy here put
    # a CUDA row against CPU rows in the PPC; no CPU suite can see that (the 2026-09-11 GPU gate did,
    # and tests/test_gpu_paths.py now pins it).
    obs_stats, obs_data, t_dim = observation.x_obs, observation.obs_data, observation.t_dim
    keys = list(cfg.params_dict) + list(cfg.rescale_params)
    with store.create("inference", cfg, name=name, note=note) as w:
        sink = w.fig_sink(fig_sink)
        samples = post.sample((int(n_samples),), x=obs_stats.to(device))
        # Corner plot
        # Size the corner by the PARAMETER COUNT: a 13x13 grid at pairplot's default is cramped enough that
        # tick labels overlap and axis titles clip. Thin the ticks for the same reason.
        n_p = len(cfg.inferred_labels)
        fig, ax = pairplot(
            samples.cpu().numpy(),
            points=(np.array([cfg.ground_truth]) if show_truth else None),
            labels=cfg.inferred_labels,
            figsize=(min(24, max(8, 1.35 * n_p)), min(24, max(8, 1.35 * n_p))),
        )
        _thin_ticks(fig)
        sink("Posterior corner", fig)

        # PPC - Option B: sort posterior samples by t_scale, process in mini-batches
        # Each sample gets its own subsample_factor based on its t_scale; all samples
        # share physical duration T_obs at dt_exp sampling (matching the observation).
        nd_dim = len(cfg.params_dict)
        samples_nd = samples[:, :nd_dim]
        samples_rescale = samples[:, nd_dim:]
        # TIER 1 (a box that declares T instead of f_scale): substitute the DERIVED f_scale into T's column before anything
        # simulates. A no-op for a box that declares f_scale. `sim_rescale_idx` is what the force
        # builders and gen_chi_raw must then be given -- handed the INFERRED index they would not
        # find 'f_scale', would fall into the Hopf-style x_scale/t_scale branch, and would drive
        # at a silently wrong amplitude.
        samples_rescale = derived.to_sim_rescale(samples_nd, samples_rescale, cfg.rescale_idx,
                                                *cfg.tier1_args)
        n_drawn = samples.shape[0]
        # Same for all samples. Prefer the length generate_observations actually resolved (post
        # cost-ceiling clip); fall back to the formula only on the experimental paths, which never call
        # generate_observations and take their length from the recording itself.
        N_points_obs = cfg.n_obs if cfg.n_obs is not None else int(cfg.T_obs / cfg.dt_exp)

        forcing_gt = torch.tensor([[val for val, _ in cfg.force_params_dict.values()]], dtype=dtype, device=device)
        forcing_gt_expanded = forcing_gt.expand(n_drawn, -1)  # (n_drawn, n_forcing); empty if no forcing
        n_vars = inits.shape[-1]
        n_force_ch = forcing.n_force_channels(cfg.model, cfg.forcing_idx, n_vars)

        (x_dim_sorted, x_spont_sorted, chi_block_sorted,
         inv_sort_idx) = ppc.simulate_ppc_bins(cfg, t, inits, samples_nd, samples_rescale,
                                               forcing_gt, N_points_obs, expected_forcing_dim(cfg),
                                               dtype, device)

        # Restore original sample order
        x_spont = x_spont_sorted[inv_sort_idx]
        # Layout [S | log(T) | forcing|chi] — must match the observation in generate_observations.
        if cfg.chi_mode:
            x_dim = x_spont                                 # PPC "sample trajectories" = passive spontaneous trace
            sim_stats = pipeline.gen_stats(x_spont, None, cfg.dt_exp, None, None, None,
                                           device=device, spontaneous_only=True)
            sim_stats = statistics.conditioning_rows(sim_stats, T_obs, chi_block_sorted[inv_sort_idx].cpu())
        elif cfg.has_forcing:
            x_dim = x_dim_sorted[inv_sort_idx]
            n_drive = x_dim.shape[0]
            sim_stats = pipeline.gen_stats(
                x_spont, x_dim, cfg.dt_exp,
                forcing_gt[:, cfg.forcing_idx["amp"]].expand(n_drive),
                forcing_gt[:, cfg.forcing_idx["freq"]].expand(n_drive),
                forcing_gt[:, cfg.forcing_idx["phase"]].expand(n_drive),
                device=device,
            )
            sim_stats = statistics.conditioning_rows(sim_stats, T_obs, forcing_gt_expanded.cpu())
        else:
            x_dim = x_spont                                 # the PPC "sample trajectories" are spontaneous
            sim_stats = pipeline.gen_stats(x_spont, None, cfg.dt_exp, None, None, None,
                                           device=device, spontaneous_only=True)
            sim_stats = statistics.conditioning_rows(sim_stats, T_obs)
        # Conditioning layout, so the zero-variance count can be split by origin rather than reported as
        # one number. See analysis.invalid_breakdown: most of a big "invalid" count is normally empty chi
        # probe slots, which is a fact about the run's K, not a defect.
        from .SBI.statistics import SUMMARY_WIDTH
        ppc_layout = {"input_dim": SUMMARY_WIDTH + 1,
                      "chi_k_pad": cfg.chi_k_pad if cfg.chi_mode else None,
                      "chi_elem_w": config.CHI_ELEM_W if cfg.chi_mode else None,
                      "chi_n_freqs": cfg.chi_n_freqs if cfg.chi_mode else None}
        results = analysis.posterior_predictive_check(obs_stats.squeeze(), sim_stats, layout=ppc_layout)
        _note = analysis.describe_invalid(results.get("invalid_breakdown"))
        if _note:
            print(f"[ppc] {_note}", flush=True)
        fig_ppc = visualizers.plot_ppc(
            results,
            ground_truth=(cfg.ground_truth if show_truth else None),
            param_names=cfg.inferred_labels,
            n_samples=n_drawn,
        )
        sink("Posterior predictive check", fig_ppc)

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
            central_rescale = derived.to_sim_rescale(central_nd, central_rescale, cfg.rescale_idx,
                                                    *cfg.tier1_args)              # tier 1, as above
            t_scale_c = central_rescale[0, cfg.rescale_idx["t_scale"]].item()
            subsample_c = max(1, round((cfg.dt_exp / t_scale_c) / cfg.dt_nd_min))
            n_fine_c = min(cfg.steady_idx + N_points_obs * subsample_c, len(t))
            t_fine_c = t[:n_fine_c]
            n_segs_c = max(1, math.ceil(n_fine_c / CHUNK_LEN))
            if cfg.has_forcing and not cfg.chi_mode:
                force_c = pipeline.build_nondim_sin_force_tensor(
                    forcing_gt, t_fine_c, central_rescale, cfg.forcing_idx, cfg.sim_rescale_idx)
            else:
                force_c = torch.zeros((1, n_force_ch, t_fine_c.shape[0]), dtype=dtype, device=device)
            x_nd_c = pipeline.gen_obs(
                model=cfg.model, params=central_nd, t=t_fine_c, inits=inits,
                force=force_c, n_segs=n_segs_c, steady_idx=cfg.steady_idx,
                state_dep_drift=cfg.state_dep_drift, var_idx=0, dtype=dtype, device=device,
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

        # ── Posterior-overlay figures ────────────────────────────────────────────────────────────────
        # Phase is set by the noise realisation, not by theta, so a draw can never match the observation
        # pointwise; these figures either align that away explicitly or avoid depending on it. See
        # core/SBI/overlay.py.
        _emit_overlay_figures(cfg, obs_data, x_dim, sim_stats, obs_stats, samples, show_truth, sink)

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
        sink("Eye test", fig)

        file_manager.atomic_torch_save(samples.detach().cpu(), w.payload("samples.pt"))
        q = torch.quantile(samples.detach().cpu().to(torch.float64),
                           torch.tensor([0.05, 0.5, 0.95], dtype=torch.float64), dim=0)
        out = {
            "ppc": {"mean_abs_z": _num(results["mean_abs_z"]), "max_abs_z": _num(results["max_abs_z"]),
                    "coverage_90": _num(results["coverage_90"]), "num_outside": int(results["num_outside"]),
                    "num_invalid": int(results["num_invalid"]),
                    "invalid_breakdown": results.get("invalid_breakdown"), "note": _note or ""},
            # A LIST of records, not a dict keyed by name: manifest JSON sorts keys, so a dict would lose
            # the parameter order (Task 9's review; the same rule as sbc.per_param).
            "posterior_summary": [{"name": k, "q05": _num(q[0, i]), "median": _num(q[1, i]), "q95": _num(q[2, i])}
                                  for i, k in enumerate(keys)],
            # _num like every other float in this body: the manifest refuses a non-finite number, and
            # a cell file is hand-edited -- a bare float() would make the WRITE the place that fails.
            "ground_truth": {k: _num(v) for k, v in zip(keys, cfg.ground_truth)} if show_truth else None,
            "n_samples": int(n_drawn), "accepted": accepted,
        }
        _write_results_json(w.payload("results.json"), out)
        w.parents = {"posterior": posterior.id, "observation": observation.id}
        w.fingerprints["x_obs"] = observation.digest
        w.config.update({"n_samples": int(n_drawn)})
        w.body = {"results": out}
    return store.load_inference(w.id)


def _build_latent_prior_for_validation(cfg, inferred_prior):
    """Mirror of the latent-prior construction in build_posterior, for gen_cal_data in validate."""
    nd_prior_physical = inferred_prior.distributions[0]
    if not isinstance(nd_prior_physical, torch.distributions.TransformedDistribution):
        raise Refusal(
            "Loaded ND prior is not a TransformedDistribution — it was saved with the pre-reparameterization "
            "pipeline. Regenerate the prior with the current `gen_prior` before running validate.",
            field="prior")
    latent_nd = nd_prior_physical.base_dist
    T_rescale = build_rescale_bijection(cfg)
    latent_rescale = torch.distributions.TransformedDistribution(inferred_prior.distributions[1], T_rescale.inv)
    return ProductPrior(
        distributions=[latent_nd, latent_rescale],
        dims=[len(cfg.params_dict), len(cfg.rescale_params)],
    )


# ── The compositions: ONE flow, two front ends ──────────────────────────────
# Defined HERE, in the module the stages live in, so that generate_observations, infer_and_visualize
# and cli.* resolve through the module globals at CALL time -- which is what the suites patch
# (tests/_fixtures.py, tests/test_nav_and_gating.py, tests/test_user_sbi.py). A re-export from another
# module would be a second binding and every one of those patches would miss it.
def _refuse_no_samples(n_samples) -> None:
    """Zero posterior draws, refused before the observation is simulated or written: otherwise the
    inference fails at samples.median after the spend, leaving an observation nothing names. None is
    "not given" (infer_and_visualize's own default applies), never a blank to refuse."""
    if n_samples is not None:
        require_at_least("n_samples", n_samples, 1)


def _t_obs_outside_training(T_obs_s: float) -> "str | None":
    """The observation length against the range training drew T from, as the judgement's sentence, or
    None inside it. A judgement, so the compositions WARN with it (PreflightWarning, walkthrough B6)
    and never refuse. One wording for both paths: until piece 3 only the simulated path said it, though
    a bench recording outside the range extrapolates exactly as a simulated cell does (spec §3.3).
    Returns the sentence rather than warning, so _preflight_warn's stacklevel is unchanged."""
    if T_obs_s < T_MIN_EXP_S:
        return (f"T_obs={T_obs_s:.2f}s is below the training range minimum T_MIN_EXP_S="
                f"{T_MIN_EXP_S:.2f}s; the posterior may extrapolate poorly.")
    if T_obs_s > T_MAX_EXP_S:
        return (f"T_obs={T_obs_s:.2f}s exceeds the training range maximum T_MAX_EXP_S="
                f"{T_MAX_EXP_S:.2f}s; the posterior may extrapolate poorly.")
    return None


@public_entry
def simulated_inference(cfg: SimConfig, posterior: LoadedPosterior, T_obs_s: float, *,
                        cell=None, gt_values=None, prior: "LoadedPrior | None" = None,
                        accept=None, n_samples: int | None = None,
                        name: str = "", note: str = "", fig_sink=None, store=None):
    """Simulate a ground-truth observation for a cell and infer on it: the whole simulated path.

    Every check that used to live in one front end only lives here now, and every one of them runs
    BEFORE the simulation -- so a refusal costs nothing and leaves no orphan observation behind.

    :param cell: the cell file to take the truth from. Give EXACTLY one of ``cell`` and ``gt_values``.
    :param gt_values: hand-entered values in parse_values_file's ``(inits, params, rescale, forcing)``
                      shape, validated against the bounds exactly as a file's are.
    :param prior: the LoadedPrior the posterior was trained with. When given, the truth is checked
                      against the distribution the network actually saw; omitted, the check is skipped
                      silently (a stand-in prior cannot answer the question).
    :param accept: an ``artifacts.Accept``. ``other_observation=True`` is REQUIRED for a NON-AMORTIZED
                      posterior: a cell simulated now draws new noise, so it can never be the
                      observation that posterior's region was drawn around.
    :param n_samples: posterior draws; None leaves infer_and_visualize's own default, so that number
                      stays written down in exactly one place.
    :return: ``(observation, inference)`` -- the payload shape the Infer tab unpacks.
    """
    from .artifacts import Accept
    store = resolve_store(store)
    # THE PRE-SPEND BLOCK, in the order spec §3.4 fixes and the "refused before any spend" pins rely
    # on: the numeric knobs, then the name, then the files -- all before the cell is parsed, its truth
    # injected or anything simulated.
    _refuse_no_samples(n_samples)
    # A blank or zero length used to pass here, simulate, write the observation and then die in
    # math.log(0.0) inside the conditioning row: after the spend, with an orphan left behind.
    T_obs_s = require_positive("t_obs", T_obs_s)
    # Before the simulation, not inside infer_and_visualize: that check used to run after the
    # observation had been simulated AND written, so a taken name cost a simulation and left an
    # observation artifact nothing would ever name.
    store.assert_name_free("inference", name)
    if (cell is None) == (gt_values is None):
        raise Refusal("simulated_inference takes exactly one of cell= (a cell file) and gt_values= "
                      "(hand-entered values in parse_values_file's shape).")
    if cell is not None:
        # The file, last: a missing cell surfaced as file_manager's bare FileNotFoundError from inside the
        # parse. _read_lines keeps that raise for a race (spec §3.6); this is the refusal.
        require_file("cell", cell, "cell")
    accept = accept or Accept()
    # GUARDRAIL 2, hoisted ahead of the spend. x_obs_digest, truncation and amortized=False are written
    # together by build_posterior and never set independently, so this predicate, the Infer tab's gate
    # and the store's load refusal all pick out the same posteriors.
    if posterior.posterior.x_obs_digest is not None and not accept.other_observation:
        raise Refusal(
            f"[tsnpe] this posterior is NOT AMORTIZED: it was trained on a prior restricted to a region "
            f"drawn around the observation with digest {posterior.posterior.x_obs_digest}. A cell "
            f"simulated now draws NEW noise, so it can never be that observation. Use an amortized "
            f"posterior. To run anyway, pass Accept(other_observation=True); the inference will "
            f"record it.", field="accept_other_observation")
    ignored = (cli.load_and_validate_gt(cfg, cell) if cell is not None
               else cfg.inject_ground_truth(*gt_values))
    if gt_values is not None:
        # inject_ground_truth never touches cfg.sources, so without this the provenance of a
        # hand-entered truth would name -- and content-hash -- whatever cell was loaded before it.
        cfg.sources.pop("cell", None)
    if ignored:
        _preflight_warn(f"the bounds file does not declare {', '.join(ignored)}; those cell values were "
                        f"ignored (the bounds file defines the inferred set)")
    cfg.T_obs = T_obs_s * cfg.get_unit_conversion_factor("s")
    _outside = _t_obs_outside_training(T_obs_s)
    if _outside:
        _preflight_warn(_outside)
    if prior is not None:
        # Bounds-checking cannot answer this: the ND prior is a stability-SCREENED GMM, so a value can
        # sit inside the box and in a corner the training data never visited.
        for msg in check_observation_in_distribution(cfg, prior.prior, prior.force_prior):
            _preflight_warn(msg)
    region = posterior.posterior.truncation
    if region is not None:
        # Reachable only past an ACCEPTED refusal above. check_observation_in_distribution samples the
        # FULL base prior, while a non-amortized flow was trained on that prior restricted to this box:
        # being in-distribution there says nothing about being inside the region.
        for msg in _truth_outside_region(posterior.posterior.T, region, cfg.ground_truth_tensor):
            _preflight_warn("this truth lies outside the region the NON-AMORTIZED posterior was trained "
                            "on; the flow extrapolates there: " + msg)
    obs = generate_observations(cfg, fig_sink=fig_sink, store=store)
    inf = infer_and_visualize(cfg, posterior, obs, name=name, note=note, fig_sink=fig_sink, store=store,
                              accept=accept,
                              **({"n_samples": n_samples} if n_samples is not None else {}))
    return obs, inf


@public_entry
def experimental_inference(cfg: SimConfig, posterior: LoadedPosterior, rec: "RecordingSet", *,
                           accept=None, n_samples: int | None = None,
                           name: str = "", note: str = "", fig_sink=None, store=None):
    """Any bench recording set (passive, driven, chi) -> the observation artifact -> the inference.

    Its own checks are the inputs' alone, before any compute: the sample count, the observation length
    (a Refusal at or below 0; outside the training range the same PreflightWarning the simulated path
    gives, spec §3.3) and the name. The recordings are checked by build_experiment_observation's file
    loop and the drive by the builder, so ``rec`` is forwarded unchecked. A recording carries no truth,
    so there is nothing to compare against the training distribution. And rebuilding the same
    recordings reproduces a region's digest, which is the LEGITIMATE match -- so the non-amortized
    refusal stays in infer_and_visualize, where it can tell that case apart from a foreign observation.
    Nothing is lost by waiting: building an experimental observation costs no simulation.
    """
    store = resolve_store(store)
    # The pre-spend block, in the order spec §3.4 fixes: the numeric knobs, then the name.
    _refuse_no_samples(n_samples)
    T_obs_s = require_positive("t_obs", rec.T_obs_s)
    store.assert_name_free("inference", name)   # with the recordings' own file checks, before any compute
    _outside = _t_obs_outside_training(T_obs_s)
    if _outside:
        _preflight_warn(_outside)
    obs = build_experiment_observation(cfg, rec, fig_sink=fig_sink, store=store)
    inf = infer_and_visualize(cfg, posterior, obs, name=name, note=note, fig_sink=fig_sink, store=store,
                              accept=accept,
                              **({"n_samples": n_samples} if n_samples is not None else {}))
    return obs, inf


@public_entry
def tsnpe_round(cfg: SimConfig, posterior: LoadedPosterior, prior: LoadedPrior,
                observation: LoadedObservation, *,
                n_directions: int | None = None, level: float | None = None,
                num_runs: int | None = None, run_size_cap: int | None = None,
                hidden_features: int | None = None, num_transforms: int | None = None,
                learning_rate: float | None = None, stop_after_epochs: int | None = None,
                max_num_epochs: int | None = None, checkpoint_every: int | None = None,
                resume: str = "auto", new_run: bool = False,
                name: str = "", note: str = "", fig_sink=None, store=None) -> LoadedPosterior:
    """One TSNPE round: the region drawn from ``posterior`` around ``observation``, then the PRIOR
    RESTRICTED to it, simulated and retrained.

    ⚠ THE PROPOSAL IS THE TRUNCATED PRIOR, NEVER THE POSTERIOR. Nothing here reimplements that rule:
    the region travels as ``truncation=``, and build_posterior is what wraps the latent prior in a
    TruncatedLatentPrior. This function only carries a front end's choices into the stages, and refuses
    before any of them is spent.

    ⚠ NO FISHER KNOBS, BY DESIGN. With a region, build_posterior reuses the region's own V and never
    reaches the Fisher at all: the box is measured in the parent's basis and is meaningless in any
    other (guardrail 7).

    :param posterior: the parent LoadedPosterior the region is drawn from.
    :param prior: the parent's own base prior. The region records that prior's GMM fingerprint and
                      build_posterior refuses any other -- a box measured on one prior selects a slab
                      of another that nobody measured.
    :param observation: the LoadedObservation the region is drawn around. Its context is installed on
                      ``cfg`` before training, so the round's ground-truth check reads THIS
                      observation's truth, or none for a bench recording.
    :param n_directions / level: how many of the best-constrained directions to truncate, and the
                      per-direction interval; None = ``truncate.DEFAULT_N_DIRECTIONS`` / ``DEFAULT_HPD``.
    :param resume / new_run: the cache policy and the D7 consent, forwarded to build_posterior always.
    :return: the LoadedPosterior build_posterior wrote and read back. It carries its own region and
                      observation digest, so nothing here has to return them separately.
    """
    store = resolve_store(store)
    store.assert_name_free("posterior", name)      # before the region draw and before days of training
    # region_from_posterior CLAMPS both ends silently (max(1, min(n, p))). Clamping a 0 up to 1 hides a
    # miskeyed setting; clamping down to P truncates EVERY direction, including the flat ones guardrail
    # 3 exists to leave at full width. Both are refusals here instead, with the default in the sentence;
    # the width's is built here because only the stage knows the width (the TSNPE tab has no posterior
    # to measure it against at the click).
    n = require_at_least("n_directions",
                         truncate.DEFAULT_N_DIRECTIONS if n_directions is None else n_directions, 1)
    P = len(cfg.params_dict) + len(cfg.rescale_params)
    if n > P:
        raise Refusal(f"The number of directions to truncate must be at most {P}, this posterior's latent "
                      f"width; got {n} (default {truncate.DEFAULT_N_DIRECTIONS}). Truncating every "
                      f"direction deletes support along the flat ones too.", field="n_directions")
    q = require_between("hpd_level", truncate.DEFAULT_HPD if level is None else level, 0.0, 1.0,
                        open_lo=True, open_hi=True)
    if q < truncate.DEFAULT_HPD - 0.009:           # i.e. below 0.99
        # A judgement, so it warns rather than refuses -- but deleted support is a ONE-WAY ratchet, and
        # a region that is too TIGHT is the expensive mistake, not the cheap one.
        _preflight_warn(f"HPD {q:g} is tighter than the recommended {truncate.DEFAULT_HPD:g}. Truncation "
                        f"permanently deletes prior support; no later round can recover it.")
    # D12, WITH NO ESCAPE HATCH. A non-amortized parent is valid only near its own observation, so a
    # region drawn from it around another one is drawn where the flow extrapolates rather than where it
    # was trained. Reachable today by loading a round's posterior and then picking another observation;
    # build_truncation_region checks the observation's digest against itself, not against the parent.
    parent_digest = posterior.posterior.x_obs_digest
    if parent_digest is not None and parent_digest != observation.digest:
        raise Refusal(
            f"[tsnpe] the parent posterior is itself NON-AMORTIZED: it is valid only near the "
            f"observation with digest {parent_digest}, and the observation supplied "
            f"('{observation.name or observation.id}') has digest {observation.digest}. A region drawn "
            f"from it around another observation would be drawn where the flow extrapolates. There is "
            f"no override: draw the round around the parent's own observation, or start from an "
            f"amortized posterior.")
    # THIS observation's context, and only this one's: a simulated observation puts back its own truth,
    # an experimental one clears whatever an earlier inference left behind. Training reads nothing
    # install sets -- the simulation identity holds no T_obs, n_obs or chi_n_freqs.
    observation.install(cfg)
    # t_scale_idx: a direction that loads on t_scale is not truncated -- the per-batch t_scale override
    # would turn its box into a reweighting (D4) -- and the region records which ones were skipped.
    t_scale_idx = len(cfg.params_dict) + cfg.rescale_idx["t_scale"]
    region = build_truncation_region(posterior, observation, n_directions=n, level=q,
                                     t_scale_idx=t_scale_idx)
    print(f"[tsnpe] region from observation {observation.name or observation.id}: {region!r}", flush=True)
    # Forwarded only when set, so every stage default stays written down in exactly one place; resume
    # and new_run always travel, because their defaults are this function's own.
    knobs = {k: v for k, v in (("num_runs", num_runs), ("run_size_cap", run_size_cap),
                               ("hidden_features", hidden_features), ("num_transforms", num_transforms),
                               ("learning_rate", learning_rate), ("stop_after_epochs", stop_after_epochs),
                               ("max_num_epochs", max_num_epochs), ("checkpoint_every", checkpoint_every))
             if v is not None}
    return build_posterior(cfg, prior, None, True, truncation=region, observation=observation,
                           parent_posterior=posterior, resume=resume, new_run=new_run,
                           name=name, note=note, fig_sink=fig_sink, store=store, **knobs)


# The experimental observation builders (and RecordingSet) live in SBI/observations.py; re-exported
# here because the GUI runners and the diagnostic scripts call them as orchestrator.build_experiment_obs*
# / orchestrator.RecordingSet.
from .SBI.observations import (build_experiment_obs, build_experiment_obs_spontaneous,  # noqa: E402
                               build_experiment_obs_chi, RecordingSet)
