"""
Configuration constants, device detection, and data carriers for the SBI pipeline.
"""
import os
from dataclasses import dataclass, field, replace
from collections import OrderedDict
from functools import cached_property
from pathlib import Path

import torch

# === DEVICE DETECTION ===
@dataclass
class DeviceConfig:
    """Hardware configuration: device, dtype, and batch size."""
    device: torch.device
    dtype: torch.dtype
    batch_size: int

def detect_device() -> DeviceConfig:
    """Detect the best available compute device and set dtype / batch size accordingly."""
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        major, minor = torch.cuda.get_device_properties(dev).major, torch.cuda.get_device_properties(dev).minor
        if (major, minor) < (8, 0):
            dev = torch.device("cpu")
    elif torch.backends.mps.is_available():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")

    dtype = torch.float32

    if dev.type == "cuda" and dtype == torch.float32:
        batch_size = 2 ** 11
    elif dev.type == "cuda" and dtype == torch.float64:
        batch_size = 2 ** 10
    else:
        batch_size = 2 ** 6

    return DeviceConfig(device=dev, dtype=dtype, batch_size=batch_size)


def cpu_device() -> DeviceConfig:
    """
    Force a CPU DeviceConfig.

    Used by the FDT and parameter-sweep branches. Their Euler-Maruyama solver is a
    sequential Python time loop over small ensembles (M ~ 256, state dim 3-5), so
    each step is a handful of tiny tensor ops. On GPU this is kernel-launch-bound
    (per-step time is ~constant regardless of M) and benchmarks ~3.4x SLOWER than
    CPU at M=256; the CPU<->GPU crossover is near M ~ 4096, far above FDT ensemble
    sizes. SBI (large batch_size, huge simulation volume) is left on detect_device().
    """
    return DeviceConfig(device=torch.device("cpu"), dtype=torch.float32, batch_size=2 ** 6)

# === PATHS ===
# Resources live at <repo-root>/Resources. The run scripts (run.bat/run.sh) cd to the repo root, so the
# cwd-relative form is correct in normal use; the __file__ fallback keeps paths valid if the app is ever
# launched from another directory (config.py is core/config.py, so parent.parent is the repo root).
_ROOT = Path(os.getcwd()) / "Resources"
if not _ROOT.exists():
    _ROOT = Path(__file__).resolve().parent.parent / "Resources"
CELL_PATH    = _ROOT / "Cells"
BOUNDS_PATH  = _ROOT / "Bounds"
UNITS_PATH   = _ROOT / "Units"
PRIOR_PATH   = _ROOT / "Priors"
POSTERIOR_PATH = _ROOT / "Posteriors"
PLOT_PATH    = _ROOT / "Plots"

# === PARAMETER LABELS (for plotting) ===
HOPF_LABELS = [r"$\mu$", r"$\beta$", r"$\sigma_x$", r"$\sigma_y$"]
BP_LABELS = [r"$\tau_{hb}$", r"$\tau_m$", r"$\tau_{gs}$", r"$\tau_t$",
             r"$C_{min}$", r"$S_{min}$", r"$S_{max}$", r"$Ca^2_m$", r"$Ca^2_{gs}$",
             r"$U_{gs,\ max}$", r"$\Delta G$", r"$k_{gs, \text{ ratio}}$",
             r"$\chi_{hb}$", r"$\chi_a$", r"$x_c$", r"$\eta_{hb}$", r"$\eta_{a}$"]
NADROWSKI_LABELS = [r"$\kappa$", r"$\tilde{\lambda}$", r"$\varphi$", r"$\tilde{\tau}$", r"$\tilde{\tau}_c$",
                    r"$S$", r"$\Delta \tilde{G}$", r"$\beta$", r"$N$", r"$\tilde{T}$"]

VALID_MODELS = ["BP", "NADROWSKI", "HOPF"]
VALID_LABELS = [BP_LABELS, NADROWSKI_LABELS, HOPF_LABELS]

# === ENSEMBLE CONSTANTS ===
UNIQUE_FREQS = 2 ** 6
K_B = 1.380649e-23  # m^2 kg s^-2 K^-1

# === EXPERIMENTAL CONSTANTS (in seconds, converted to cell file units during setup) ===
DT_EXP_S = 1e-3        # 1000 FPS camera frame interval
T_MIN_EXP_S = 1.0      # shortest expected recording (1 s)
T_MAX_EXP_S = 60.0     # longest expected recording (1 min)

# === SIMULATION COST CONSTANTS ===
CHUNK_LEN = 100_000    # fine integration steps per segment (per-chunk memory cap)
N_ND_MAX = 300_000     # max total fine integration steps per batch (pre-filter ceiling)
PPC_BIN_SIZE = 50      # samples per mini-batch for posterior-predictive-check simulation
CAL_RUN_SIZE = 10      # samples per (t_scale, T) pair in SBC calibration data
SBC_N_CAL = 2000       # calibration datasets for SBC in validate(). n_cal=1000 was under-powered:
                       # the K=10 repeat study (scripts/sbc_characterize.py) showed mild marginal
                       # miscalibration only surfaces reliably at n_cal>=2000 (KS power grows with n_cal).
TRAINING_NUM_RUNS = 5000  # number of (t_scale_k, T_k) batches per training round (data budget)

# === NEURAL POSTERIOR & TRAINING HYPERPARAMETERS ===
# Capacity / convergence knobs for the SBI posterior. Raise the flow capacity and/or the
# training budget to address broad SBC under-calibration; defaults match sbi's own.
DENSITY_ESTIMATOR = "nsf"                # flow family: "nsf" (neural spline flow) or "maf"
NSF_HIDDEN_FEATURES = 128                 # hidden units per flow transform (sbi default 50)
NSF_NUM_TRANSFORMS = 8                   # number of flow transforms (sbi default 5)
NSF_NUM_BINS = 10                        # spline bins per transform, NSF only (sbi default 10)
TRAINING_NUM_ROUNDS = 1                  # 1 = amortized NPE; >1 = sequential NPE near the observation
TRAINING_BATCH_SIZE = 512                # density-estimator minibatch size
TRAINING_LEARNING_RATE = 1e-3            # Adam learning rate (sbi default)
TRAINING_STOP_AFTER_EPOCHS = 20          # early-stopping patience in epochs (sbi default)
TRAINING_MAX_NUM_EPOCHS = 2_147_483_647  # hard epoch cap (sbi default: effectively unbounded)
TRAINING_SHOW_SUMMARY = True             # print sbi's train/validation-loss summary (check convergence)

# === PROGRESS BARS ===
# The per-time-segment bar (core/Simulator/simulator.py) wraps segs in {1,2,3} -- a three-step bar that
# tells a user nothing, while nesting a whole extra level under the training-data bar.
# core.gui.app.build_app() sets this True; the CLI and scripts/ never touch it, so `python -m core`
# renders exactly the bars it always has.
# Read this through the MODULE (`from core import config; config.QUIET_SEGMENT_BAR`) -- a
# `from core.config import QUIET_SEGMENT_BAR` snapshots the value at import and would freeze it False.
QUIET_SEGMENT_BAR = False

# The SDE solver's per-step bar (core/Solvers/sdeint.py) is f"{SOLVER_BAR_DESC} (batch={batch_size})".
# It stays ON under the GUI: its it/s IS the "Solver Performance" meter, and its percentage is the only
# thing that moves during a ~10s training iteration. The GUI does NOT render it as a progress row (a
# posterior build constructs 10k-30k of these bars, one per time segment) -- it feeds a dedicated widget,
# found by this desc prefix. Keyed on the DESC, never on the row: the bar's tqdm `pos` is 0, 1 or 2
# depending on which phase and which panel is running. See core/gui/widgets/progress_pane.py.
SOLVER_BAR_DESC = "step"

# === DECORRELATING REPARAMETERIZATION (Track A: flow calibration via latent rotation) ===
# When the inferred params are well-identified but strongly correlated (e.g. kappa~x_scale at
# |cos|=0.95), the flow mis-calibrates the thin diagonal ridge. Rotating the flow's latent
# coordinate into the simulation-based Fisher eigenbasis makes that posterior axis-aligned so the
# flow can calibrate it -- no information loss, no model/stats change. REPARAM_ROTATE=False (V=I)
# is exactly the current pipeline, so the rotation is fully optional and model-agnostic.
REPARAM_ROTATE = True   # True = rotate into the Fisher eigenbasis; False = plain pipeline.
REPARAM_FISHER_M = 48    # ensemble per latent-perturbation for the simulation-based Fisher estimate.
REPARAM_FISHER_DZ = 0.1  # latent-space central-difference step for the Fisher Jacobian.
# Operating points (GT + prior draws) over which the simulation Fisher is AVERAGED to build the
# rotation V. >1 makes the single linear rotation valid prior-wide, not just at GT (a GT-only V
# re-correlates the curved degeneracies off-GT). 1 = GT-only (the original behavior).
REPARAM_FISHER_POINTS = 8

# === LOG-SPACE BOX (linearize the multiplicative degeneracies before rotating) ===
# ND/rescale params (by cell-file key) whose box bijection is GEOMETRIC (log) instead of linear.
# In log coords the products kappa*x_scale (amplitude) and lambda*t_scale (timescale) become SUMS,
# so the single linear Fisher rotation can decorrelate them across the whole prior. Only params with
# a strictly positive lower bound are eligible (others fall back to linear with a warning). Empty
# list = pure linear box (legacy). The chosen mask is persisted beside each posterior (<name>.rot.pt)
# so eval reconstructs the exact training box regardless of this setting. REBUILD the ND prior after
# changing this (the latent GMM is fit in the box's coordinate).
REPARAM_LOG_PARAMS = []   # ALL-LINEAR box (the keeper posterior_07012026's coordinate). Log-scaling
                          # f_scale (REPARAM_LOG_PARAMS=["f_scale"]) was TRIED as a fix for its mild
                          # linear-box SBC tilt (GT=10 at box-fraction 0.009 = flat sigmoid tail; see
                          # scripts/diagnose_fscale.py), but the posterior trained under it was WORSE --
                          # bad TARP / expected-coverage and a worse f_scale SBC rank -- so it was
                          # discarded and this was reverted to []. Keep the DEGENERACY params
                          # (k, lam, x_scale, t_scale) LINEAR too (log OVER-MIXED those in posterior_6302026).
                          # f_scale is a RESCALE param, so toggling it here does NOT rebuild the ND prior:
                          # nd_log_mask stays all-False, and the existing linear ND prior
                          # (prior_forcing_no_forcing.pt) + posterior_07012026 already match this box.

# === TRANSIENT (Case A: clip initial conditions settling) ===
TRANSIENT_ND_UNITS = 100  # ND time units of transient to discard; ~20 e-folds of the slowest
                          # bounded mode (tau_c up to ~5.0) in ND Nadrowski cell files.

# === PRIOR STABILITY SCREENING ===
STABILITY_SWEEP_ND_UNITS = 1000  # ND time units used to screen parameter stability during
                                # prior construction (global + local sweeps). Short enough
                                # to be cheap, long enough for instabilities to manifest.

# === SIMULATION CONFIG DATACLASS ===
@dataclass
class SimConfig:
    """
    Carries all state needed to run the SBI pipeline.
    Replaces the 9-element tuple that was threaded through setup() -> run().
    """
    # Model selection
    model: str
    labels: list[str]
    state_dep_drift: bool

    # Parsed from cell file
    inits_dict: OrderedDict # {name: val}
    params_dict: OrderedDict # {name: (val, (lo, hi))}
    rescale_params: OrderedDict # {name: (val, (lo, hi))}
    force_params_dict: OrderedDict # {name: (val, (lo, hi))}
    units_dict: tuple
    si_factors: list[float]

    # Time / segmentation (legacy fallback fields; primary time setup uses dt_exp + T_obs)
    t_max: float = None
    dt: float = None

    # Experimental observation parameters (in cell file time units, set during setup)
    dt_exp: float = None          # camera frame interval
    t_min_exp: float = None       # shortest expected recording
    t_max_exp: float = None       # longest expected recording
    T_obs: float = None           # ground-truth observation duration (user input)

    # Hardware
    hw: DeviceConfig = field(default_factory=detect_device)

    # --- Derived properties ---
    @property
    def t_scale_bounds(self) -> tuple[float, float]:
        """(lo, hi) bounds on the t_scale rescaling parameter (λ/K_gs)."""
        _, (lo, hi) = self.rescale_params["t_scale"]
        return lo, hi

    @property
    def dt_nd_min(self) -> float:
        """Finest ND time step needed: dt_exp / t_scale_max."""
        _, t_scale_hi = self.t_scale_bounds
        return self.dt_exp / t_scale_hi

    @property
    def t_nd_max(self) -> float:
        """Longest ND duration needed: t_max_exp / t_scale_min."""
        t_scale_lo, _ = self.t_scale_bounds
        return self.t_max_exp / t_scale_lo

    @cached_property
    def t(self) -> torch.Tensor:
        """
        Pre-simulated ND time vector at finest resolution and longest duration.

        Cached: SimConfig is effectively immutable after build_sim_config(), so we
        allocate the 2.4M-point tensor once per config lifetime.
        """
        if self.dt_exp is not None:
            n_steps = int(self.t_nd_max / self.dt_nd_min)
            return torch.linspace(0, self.t_nd_max, n_steps,
                                  dtype=self.hw.dtype, device=self.hw.device)
        # fallback for legacy usage
        return torch.linspace(0, self.t_max, int(self.t_max / self.dt),
                              dtype=self.hw.dtype, device=self.hw.device)

    @property
    def steady_idx(self) -> int:
        """
        Index where transient ends and steady-state begins.

        Fixed number of fine integration steps corresponding to TRANSIENT_ND_UNITS
        ND time units — model-intrinsic, independent of prior bounds on T or t_scale.
        """
        steady_idx = int(TRANSIENT_ND_UNITS / self.dt_nd_min)
        # Safety check: transient must leave budget for at least the minimum output batch
        assert steady_idx < N_ND_MAX, (
            f"TRANSIENT_ND_UNITS={TRANSIENT_ND_UNITS} produces steady_idx={steady_idx} "
            f">= N_ND_MAX={N_ND_MAX}. Reduce TRANSIENT_ND_UNITS or raise N_ND_MAX."
        )
        return steady_idx

    @property
    def has_ground_truth(self) -> bool:
        """True once ground-truth VALUES are loaded (a bounds-built config has None value slots)."""
        rows = list(self.params_dict.values()) + list(self.rescale_params.values())
        return len(rows) > 0 and all(row[0] is not None for row in rows)

    def _require_ground_truth(self) -> None:
        if not self.has_ground_truth:
            raise ValueError(
                "SimConfig was built from a bounds file (no ground-truth values). Load a cell file via "
                "inject_ground_truth(...) before generating a synthetic observation, or use experimental data."
            )

    @property
    def ground_truth(self) -> list[float]:
        """Ground-truth values for all inferred params (ND + rescale). Requires a loaded cell."""
        self._require_ground_truth()
        nd = [row[0] for row in self.params_dict.values()]
        rescale = [row[0] for row in self.rescale_params.values()]
        return nd + rescale

    @property
    def ground_truth_tensor(self) -> torch.Tensor:
        return torch.tensor(self.ground_truth, dtype=self.hw.dtype, device=self.hw.device)

    @property
    def nd_params_bounds(self) -> list[tuple]:
        """Parameter bounds for prior construction."""
        return [row[1] for row in self.params_dict.values()]

    @property
    def inits_tensor(self) -> torch.Tensor:
        """Initial conditions as a (1, n_vars) tensor. A bounds-built config has no inits until a cell loads."""
        if not self.inits_dict:
            raise ValueError(
                "SimConfig has no initial conditions (built from bounds + units only). Load a ground-truth "
                "cell file via inject_ground_truth(...) before generating an observation."
            )
        return torch.tensor(list(self.inits_dict.values()), dtype=self.hw.dtype, device=self.hw.device).unsqueeze(0)

    @property
    def params_tensor(self) -> torch.Tensor:
        """ND-only ground-truth parameters as a (1, n_params) tensor for the simulator. Requires a loaded cell."""
        self._require_ground_truth()
        nd = [row[0] for row in self.params_dict.values()]
        return torch.tensor(nd, dtype=self.hw.dtype, device=self.hw.device).unsqueeze(0)

    @staticmethod
    def _fill_checked(label: str, cell_vals: dict, cfg_dict: OrderedDict, check_bounds: bool) -> None:
        """Validate a cell values dict against a config (val,(lo,hi)) dict, then fill in the values."""
        if set(cell_vals.keys()) != set(cfg_dict.keys()):
            missing = sorted(set(cfg_dict) - set(cell_vals))
            extra = sorted(set(cell_vals) - set(cfg_dict))
            raise ValueError(
                f"Cell file {label} do not match the bounds file: missing={missing}, unexpected={extra}."
            )
        if check_bounds:
            oob = [f"{n}={cell_vals[n]} not in ({lo}, {hi})"
                   for n, (_, (lo, hi)) in cfg_dict.items() if not (lo <= cell_vals[n] <= hi)]
            if oob:
                raise ValueError(f"Cell file {label} outside the bounds file's bounds: " + "; ".join(oob))
        for n in cfg_dict:
            cfg_dict[n] = (cell_vals[n], cfg_dict[n][1])

    def inject_ground_truth(self, inits: dict, param_vals: dict,
                            rescale_vals: dict, forcing_vals: dict) -> None:
        """
        Fill ground-truth VALUES + initial conditions from a cell file into a bounds-built config.

        SAFEGUARD: the ND and rescale (inferred) param sets must match the bounds file and every value
        must lie within its bounds — else a clear ValueError listing the offenders. Forcing is the known
        DRIVE (conditioning, not an inferred param): its set must match, but its range is not enforced
        (a spontaneous cell legitimately uses amp=0/freq=0 outside the drive prior's range).
        """
        self._fill_checked("ND parameters", param_vals, self.params_dict, check_bounds=True)
        self._fill_checked("rescale parameters", rescale_vals, self.rescale_params, check_bounds=True)
        self._fill_checked("forcing parameters", forcing_vals, self.force_params_dict, check_bounds=False)
        self.inits_dict = OrderedDict(inits)

    def set_observation_context(self, T_obs: float, forcing_vals: dict | None = None) -> None:
        """
        Set the observation duration (and optionally forcing VALUES) for the experimental-data branch,
        so PPC / eye-test simulators that read cfg.T_obs and cfg.force_params_dict values work.
        """
        self.T_obs = T_obs
        if forcing_vals is not None:
            for name, v in forcing_vals.items():
                if name in self.force_params_dict:
                    self.force_params_dict[name] = (v, self.force_params_dict[name][1])

    @property
    def inferred_labels(self) -> list[str]:
        """LaTeX labels (with units) for all inferred params (ND + rescale) for plotting.

        ND params keep their model LaTeX (self.labels); rescale params are rendered via
        Helpers.labels.rescale_axis_label so a corner/SBC plot shows e.g. ``$x_{\\mathrm{scale}}$ (nm/ND)``
        instead of the raw string ``x_scale``."""
        from .Helpers import labels as _labels
        rescale_labels = [
            _labels.rescale_axis_label(name, length_unit=self.length_unit,
                                       time_unit=self.time_unit, force_unit=self.force_unit)
            for name in self.rescale_params
        ]
        return self.labels + rescale_labels

    @property
    def forcing_idx(self) -> dict[str, int]:
        """Maps forcing param names to column indices, e.g. {"amp": 0, "freq": 1, ...}."""
        return {name: i for i, name in enumerate(self.force_params_dict.keys())}

    @property
    def rescale_idx(self) -> dict[str, int]:
        """Maps rescale param names to column indices, e.g. {"x_offset": 0, "x_scale": 1, ...}."""
        return {name: i for i, name in enumerate(self.rescale_params.keys())}

    def get_unit_conversion_factor(self, si_unit: str) -> float:
        """
        SI unit -> cell file equivalent unit conversion factor.

        Finds which unit in the cell file has the same dimensionality as si_unit,
        and returns the multiplicative factor to convert from SI value to cell value.

        Examples:
          - get_unit_conversion_factor("s")  -> 1000.0 if cell uses ms
          - get_unit_conversion_factor("N")  -> 1e12 if cell uses pN
          - get_unit_conversion_factor("Hz") -> 1.0 if cell uses Hz

        :param si_unit: SI unit string (e.g. "s", "N", "Hz", "rad").
        :return: Conversion factor: cell_value = si_value * factor.
        :raises ValueError: If no unit in the cell file matches the given dimensionality.
        """
        import pint
        ureg = pint.UnitRegistry()
        target_dim = ureg.Quantity(1, si_unit).dimensionality
        for unit_str in self.units_dict:
            try:
                if ureg.Quantity(1, unit_str).dimensionality == target_dim:
                    return ureg.Quantity(1, si_unit).to(unit_str).magnitude
            except pint.UndefinedUnitError:
                continue
        raise ValueError(f"No unit with dimensionality {target_dim} found in cell file.")

    @cached_property
    def _ureg(self):
        import pint
        return pint.UnitRegistry()

    def _resolve_unit(self, si_unit: str) -> "str | None":
        """The cell's unit TOKEN whose dimensionality matches ``si_unit`` (e.g. "s" -> "ms"), or None.

        units_dict is set-derived (unordered), so match by DIMENSIONALITY, never by index."""
        ureg = self._ureg
        try:
            target = ureg.Quantity(1, si_unit).dimensionality
        except Exception:                      # noqa: BLE001
            return None
        for tok in self.units_dict:
            try:
                if ureg.Quantity(1, tok).dimensionality == target:
                    return tok
            except Exception:                  # noqa: BLE001 -- undefined token; skip
                continue
        return None

    @cached_property
    def length_unit(self) -> "str | None":
        """Cell length unit token (e.g. "nm") for displacement axis labels."""
        return self._resolve_unit("m")

    @cached_property
    def time_unit(self) -> "str | None":
        """Cell time unit token (e.g. "ms"). Note: trace TIME axes are shown in seconds; this is only for
        the rescale-param label t_scale (ms/ND)."""
        return self._resolve_unit("s")

    @cached_property
    def force_unit(self) -> "str | None":
        """Cell force unit token (e.g. "pN"); None for BP, which declares no force unit."""
        return self._resolve_unit("N")

    @cached_property
    def freq_unit(self) -> "str | None":
        """Cell frequency unit token (e.g. "Hz"); None for BP."""
        return self._resolve_unit("Hz")


# === FDT CONFIG DATACLASS ===
@dataclass
class FDTConfig:
    """
    Carries all state needed to run the FDT analysis pipeline.
    Parallel to SimConfig but minimal: no prior/posterior/inference plumbing.
    """
    # Shared with SimConfig (model identity + parsed cell file)
    model: str
    state_dep_drift: bool
    inits_dict: OrderedDict           # {name: val}
    params_dict: OrderedDict          # {name: (val, (lo, hi))}
    rescale_params: OrderedDict       # {name: (val, (lo, hi))}
    force_params_dict: OrderedDict    # {name: (val, (lo, hi))}
    units_dict: tuple
    si_factors: list[float]

    # FDT-specific knobs (sensible defaults; overrideable in build_fdt_config)
    n_freqs: int = 60
    # Multipliers of omega_0 for the Campaign-2 production grid.
    # Asymmetric in log space by design: below = 1 decade, above = 1.5 decades
    # (=> ~50% more drive frequencies above omega_0, to capture FDT recovery
    # at the high-frequency end while still resolving the active band below).
    freq_bounds: tuple = (0.1, 30.0)
    ensemble_M: int = 256              # trajectories per Campaign-2 frequency
    freqs_per_batch: int = 1           # frequencies packed per simulator call in Campaign 2
    F0: float = 0.05                   # ND forcing amplitude (within linear regime)
    burn_in_nd: float = 100.0
    T_obs_periods: int = 30
    dt_nd: float = 0.01
    psd_T_obs_nd: float = 8000.0       # Campaign-1 steady-state duration

    # Filled in by run_fdt after cfg is built (from params_dict["k"])
    omega_0: float = None

    # Hardware
    hw: DeviceConfig = field(default_factory=detect_device)

    # --- Derived ---
    @property
    def inits_tensor(self) -> torch.Tensor:
        """(1, n_vars) tensor of initial conditions."""
        return torch.tensor(list(self.inits_dict.values()),
                            dtype=self.hw.dtype, device=self.hw.device).unsqueeze(0)

    @property
    def params_tensor(self) -> torch.Tensor:
        """(1, n_params) Nadrowski ND params."""
        nd = [row[0] for row in self.params_dict.values()]
        return torch.tensor(nd, dtype=self.hw.dtype, device=self.hw.device).unsqueeze(0)

    def params_for_M(self, M: int) -> torch.Tensor:
        """Tile ND params to shape (M, n_params) for ensemble batching."""
        return self.params_tensor.expand(M, -1).contiguous()

    def inits_for_M(self, M: int) -> torch.Tensor:
        """Tile initial conditions to shape (M, n_vars)."""
        return self.inits_tensor.expand(M, -1).contiguous()

    def with_overrides(self, **kwargs) -> "FDTConfig":
        """
        Return a shallow copy with overridden values.

        Keys may be:
          - ND parameter names from params_dict (overrides value, preserves bounds);
            used by passive-baseline sanity check (temp=1.0, tau_c=0.0)
          - any top-level FDTConfig field (n_freqs, F0, ensemble_M, ...)
        """
        nd_keys = set(self.params_dict.keys())
        top_kwargs = {k: v for k, v in kwargs.items() if k not in nd_keys}
        nd_kwargs = {k: v for k, v in kwargs.items() if k in nd_keys}

        new_params = OrderedDict(self.params_dict)
        for k, v in nd_kwargs.items():
            _, bounds = new_params[k]
            new_params[k] = (v, bounds)

        return replace(self, params_dict=new_params, **top_kwargs)
