"""
Configuration constants, device detection, and data carriers for the SBI pipeline.
"""
import os
from dataclasses import dataclass, field
from collections import OrderedDict
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

# === PATHS ===
_ROOT = Path(os.getcwd()) / "Resources"
CELL_PATH    = _ROOT / "Cells"
PRIOR_PATH   = _ROOT / "Priors"
POSTERIOR_PATH = _ROOT / "Posteriors"
PLOT_PATH    = _ROOT / "Plots"

# === PARAMETER LABELS (for plotting) ===
HOPF_LABELS = [r"$\mu$", r"$\omega$", r"$\alpha$", r"$\beta$", r"$\epsilon_x$", r"$\epsilon_y$"]
DIM_LABELS = [r"$\lambda_x$", r"$\lambda_y$", r"$\lambda_{sf}$", r"$k_{sf}", r"k_{sp}",
              r"$k_{gs, min}$", r"$k_{gs, max}$", r"$k_{es}", r"$x_{sf}$", r"$x_{es}$", r"$x_{sp}$", r"$x_c$",
              r"$d$", r"$n$", r"$\gamma$", r"$c_{min}$", r"$s_{min}$", r"$c_{max}$", r"$s_{max}$",
              r"$k_{m, +}$", r"$k_{r, +}", r"$k_{m, -}$", r"$k_{r, -}$", r"$Ca2_{x, in}$", r"$ca2_{x, ex}$",
              r"$v_m$", r"$v_{ref}$", r"$z$", r"$r_m$", r"$r_r$", r"$\Delta_e$", r"$\tau_0$", r"$T$", r"$\epsilon$"]
ND_LABELS = [r"$\tau_{hb}$", r"$\tau_m$", r"$\tau_{gs}$", r"$\tau_t$",
             r"$C_{min}$", r"$S_{min}$", r"$S_{max}$", r"$Ca^2_m$", r"$Ca^2_{gs}$",
             r"$U_{gs,\ max}$", r"$\Delta E$", r"$k_{gs, \text{ ratio}}$",
             r"$\chi_{hb}$", r"$\chi_a$", r"$x_c$", r"$\eta_{hb}$", r"$\eta_{a}$"]
NADROWSKI_LABELS = [r"$\lambda$", r"$\lambda_y$", r"$\tau$", r"$k_{gs}$", r"$k_{sp}$",
                    r"$d$", r"$f_{max}$", r"$c_0$", r"$c_m$", r"$S$",
                    r"$n$", r"$\Delta E$", r"$T$", r"$T_{eff}$", r"$\tau_c$"]
ND_NADROWSKI_LABELS = [r"$\kappa$", r"$\lambda$", r"$f_{\text{max}}$", r"$\tau$", r"$\tau_c$",
                       r"$c_0$", r"$S$", r"$\Delta E$", r"$\beta", r"$n$", r"$T$"]

VALID_MODELS = ["DIMENSIONAL", "NON-DIMENSIONAL", "NADROWSKI", "ND NADROWSKI", "HOPF"]
VALID_LABELS = [DIM_LABELS, ND_LABELS, NADROWSKI_LABELS, ND_NADROWSKI_LABELS, HOPF_LABELS]


# === ENSEMBLE CONSTANTS ===
UNIQUE_FREQS = 2 ** 6
K_B = 1.380649e-23  # m^2 kg s^-2 K^-1

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

    # Time / segmentation (set by CLI after model selection)
    t_max: float = None
    dt: float = None
    steady_pct: float = None
    n_segs: int = None

    # Hardware
    hw: DeviceConfig = field(default_factory=detect_device)

    # Which parameter groups are inferred vs. fixed during posterior sampling
    # Options: "nd", "rescale", "forcing"
    inferred_groups: list[str] = field(default_factory=lambda: ["nd", "rescale"])

    # --- Derived properties ---
    @property
    def t(self) -> torch.Tensor:
        """Time tensor on the configured device."""
        return torch.linspace(0, self.t_max, int(self.t_max / self.dt),
                              dtype=self.hw.dtype, device=self.hw.device)

    @property
    def steady_idx(self) -> int:
        """Index where transient ends and steady-state begins."""
        return int(self.steady_pct * int(self.t_max / self.dt))

    @property
    def ground_truth(self) -> list[float]:
        """Ground-truth parameter values from the cell file."""
        return [row[0] for row in self.params_dict.values()]

    @property
    def ground_truth_tensor(self) -> torch.Tensor:
        return torch.tensor(self.ground_truth, dtype=self.hw.dtype, device=self.hw.device)

    @property
    def nd_params_bounds(self) -> list[tuple]:
        """Parameter bounds for prior construction."""
        return [row[1] for row in self.params_dict.values()]

    @property
    def inits_tensor(self) -> torch.Tensor:
        """Initial conditions as a (1, n_vars) tensor."""
        return torch.tensor(list(self.inits_dict.values()), dtype=self.hw.dtype).unsqueeze(0)

    @property
    def params_tensor(self) -> torch.Tensor:
        """Ground-truth parameters as a (1, n_params) tensor."""
        return torch.tensor(self.ground_truth, dtype=self.hw.dtype).unsqueeze(0)
