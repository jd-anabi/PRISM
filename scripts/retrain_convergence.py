"""
Step 3 part B (sbi_calibration_handoff.txt): convergence retrain.

Trains a NEW posterior non-interactively and saves its per-epoch train/validation loss
curve (via the part-A capture in train_nn -> build_posterior), so we can read whether the
posterior is UNDER-FIT (validation loss still descending near the best epoch -> train
longer / raise capacity) or CONVERGED (clean plateau -> the remaining wide/biased marginals
are a data/identifiability limit, not under-fit).

It inherits the EXACT training prior from a base posterior (posterior.prior =
SBIPriorWrapper(latent ProductPrior)), so the retrain is apples-to-apples with the
existing run and there is no prior-file ambiguity / no expensive stability rebuild.

Workflow:
  1. Baseline read: run with defaults (current config capacity/patience) -> inspect the
     saved <NAME>_loss.png.
  2. If still descending: re-run with STOP_AFTER=40 (or 60) and/or HIDDEN=192 / TRANSFORMS=10,
     then re-check SBC (scripts/sbc_characterize.py POST=<NAME>.pt) on {kappa,lambda,beta,...}.

Env knobs:
  CELL       cell file                              (default Resources/Cells/nadrowski_cell_2.txt)
  BASE_POST  posterior to inherit the prior from    (default posterior_3d.pt)
  NAME       output posterior name (no extension)   (default posterior_convergence)
  HIDDEN     NSF hidden features                    (default config.NSF_HIDDEN_FEATURES=128)
  TRANSFORMS NSF num transforms                     (default config.NSF_NUM_TRANSFORMS=8)
  STOP_AFTER early-stopping patience (epochs)       (default config.TRAINING_STOP_AFTER_EPOCHS=20)
  NUM_RUNS   training batches (data budget)         (default config.TRAINING_NUM_RUNS)
  SEED       RNG seed                               (default 0)
  DRY        if "1", build everything but skip train_nn (fast wiring check)

Run (baseline):
  & "C:\\Users\\J\\anaconda3\\envs\\biophys-env\\python.exe" scripts/retrain_convergence.py
Run (more patience + capacity):
  $env:STOP_AFTER=40; $env:HIDDEN=192; $env:NAME="posterior_conv_big"; & "...python.exe" scripts/retrain_convergence.py
"""
import os
import sys
import warnings; warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
from matplotlib import pyplot as plt

from core import cli, orchestrator
from core.config import (
    SimConfig, DT_EXP_S, T_MIN_EXP_S, T_MAX_EXP_S, detect_device, NADROWSKI_LABELS,
    POSTERIOR_PATH, PLOT_PATH,
    DENSITY_ESTIMATOR, NSF_HIDDEN_FEATURES, NSF_NUM_TRANSFORMS, NSF_NUM_BINS,
    TRAINING_NUM_ROUNDS, TRAINING_BATCH_SIZE, TRAINING_LEARNING_RATE,
    TRAINING_STOP_AFTER_EPOCHS, TRAINING_MAX_NUM_EPOCHS, TRAINING_NUM_RUNS,
)
from core.SBI import pipeline, embedded_network, statistics
from core.SBI.Priors import sbi_prior_wrapper
from core.SBI.reparam import build_inferred_bijection
from core.Helpers import visualizers

# ---- knobs ----
CELL = os.environ.get("CELL", "Resources/Cells/nadrowski_cell_2.txt")
BASE_POST = os.environ.get("BASE_POST", "posterior_3d.pt")
NAME = os.environ.get("NAME", "posterior_convergence")
HIDDEN = int(os.environ.get("HIDDEN", str(NSF_HIDDEN_FEATURES)))
TRANSFORMS = int(os.environ.get("TRANSFORMS", str(NSF_NUM_TRANSFORMS)))
STOP_AFTER = int(os.environ.get("STOP_AFTER", str(TRAINING_STOP_AFTER_EPOCHS)))
NUM_RUNS = int(os.environ.get("NUM_RUNS", str(TRAINING_NUM_RUNS)))
SEED = int(os.environ.get("SEED", "0"))
DRY = os.environ.get("DRY", "0") == "1"
torch.manual_seed(SEED)
print(f"[cfg] CELL={CELL} BASE_POST={BASE_POST} NAME={NAME} HIDDEN={HIDDEN} "
      f"TRANSFORMS={TRANSFORMS} STOP_AFTER={STOP_AFTER} NUM_RUNS={NUM_RUNS} SEED={SEED} DRY={DRY}",
      flush=True)

# ---- build cfg non-interactively; T_obs is irrelevant for training (T sampled per batch) ----
inits, params, rescale, forcing, units, si, s2c = cli._parse_cell(CELL)
cfg = SimConfig(model="NADROWSKI", labels=NADROWSKI_LABELS, state_dep_drift=True,
                inits_dict=inits, params_dict=params, rescale_params=rescale,
                force_params_dict=forcing, units_dict=units, si_factors=si,
                dt_exp=DT_EXP_S * s2c, t_min_exp=T_MIN_EXP_S * s2c,
                t_max_exp=T_MAX_EXP_S * s2c, T_obs=T_MIN_EXP_S * s2c, hw=detect_device())
dtype, device = cfg.hw.dtype, cfg.hw.device
nd_dim = len(cfg.params_dict)

# ---- inherit the EXACT training (latent) prior from the base posterior ----
base = torch.load(str(POSTERIOR_PATH / BASE_POST), weights_only=False)
latent_inferred_prior = base.prior.gen_dist            # ProductPrior([latent_nd_gmm, latent_rescale])
_z = latent_inferred_prior.sample((2,))
assert _z.shape[-1] == nd_dim + len(cfg.rescale_params), \
    f"latent prior dim {_z.shape[-1]} != {nd_dim + len(cfg.rescale_params)}"
del base

T = build_inferred_bijection(cfg)
force_prior = orchestrator._build_forcing_prior(cfg)
sbi_prior = sbi_prior_wrapper.SBIPriorWrapper(latent_inferred_prior)

# ---- embedding net (mirror orchestrator.build_posterior) ----
forcing_dim = len(cfg.force_params_dict)
input_dim = len(statistics.FEATURE_LABELS) + 1          # 41 summary stats + log(T)
embedded_net = embedded_network.EmbeddedNet(
    input_dim, 3 * input_dim // 2,
    (5 * input_dim // 2, 2 * input_dim),
    forcing_dim=forcing_dim,
    forcing_layer_dims=(forcing_dim * 4, forcing_dim * 2),
    merge_layer_dim=2 * input_dim,
)

training_params = {
    "model": cfg.model, "prior": latent_inferred_prior, "t": cfg.t,
    "run_size": cfg.hw.batch_size, "num_runs": NUM_RUNS,
    "steady_idx": cfg.steady_idx, "dt_nd_min": cfg.dt_nd_min,
    "dt_exp": cfg.dt_exp, "t_min_exp": cfg.t_min_exp, "t_max_exp": cfg.t_max_exp,
    "t_scale_bounds": cfg.t_scale_bounds, "state_dep_drift": cfg.state_dep_drift,
    "dtype": cfg.hw.dtype, "device": cfg.hw.device,
}

print(f"[wiring] device={device} input_dim={input_dim} forcing_dim={forcing_dim} nd_dim={nd_dim} "
      f"run_size={cfg.hw.batch_size} latent_prior_dim={_z.shape[-1]}", flush=True)
if DRY:
    print("[DRY] setup OK; skipping train_nn.", flush=True)
    sys.exit(0)

# ---- train (loss curve captured via the part-A change in train_nn) ----
posterior_latent, diag = pipeline.train_nn(
    training_params, model=DENSITY_ESTIMATOR, prior=sbi_prior,
    embedding_net=embedded_net, forcing_prior=force_prior,
    nd_dim=nd_dim, forcing_idx=cfg.forcing_idx, rescale_idx=cfg.rescale_idx,
    x_obs=None, theta_obs=None, num_rounds=TRAINING_NUM_ROUNDS,
    return_diagnostics=True, theta_transform=T,
    hidden_features=HIDDEN, num_transforms=TRANSFORMS, num_bins=NSF_NUM_BINS,
    learning_rate=TRAINING_LEARNING_RATE, stop_after_epochs=STOP_AFTER,
    max_num_epochs=TRAINING_MAX_NUM_EPOCHS, show_train_summary=True,
    batch_size=TRAINING_BATCH_SIZE, device=device,
)

# ---- persist posterior + loss curve ----
torch.save(posterior_latent, str(POSTERIOR_PATH / (NAME + ".pt")))
val = diag.get("validation_loss") or []
train = diag.get("training_loss") or []
np.savez(str(POSTERIOR_PATH / (NAME + ".loss.npz")),
         training_loss=np.asarray(train, dtype=float),
         validation_loss=np.asarray(val, dtype=float),
         best_validation_loss=float(diag.get("best_validation_loss") or float("nan")),
         epochs_trained=int(diag.get("epochs_trained") or -1),
         stop_after_epochs=int(diag.get("stop_after_epochs") or STOP_AFTER))
visualizers.plot_training_loss(diag, save_path=str(PLOT_PATH / (NAME + "_loss.png")))
print("saved:", str(POSTERIOR_PATH / (NAME + ".pt")))
print("saved:", str(PLOT_PATH / (NAME + "_loss.png")))

# ---- convergence read ----
print("\n=== CONVERGENCE READ ===")
if len(val):
    best_epoch = int(np.argmin(val)) + 1
    et = diag.get("epochs_trained")
    best = diag.get("best_validation_loss")
    # improvement over the last 5 epochs leading up to the best epoch (mild = converged).
    lo = max(0, best_epoch - 5)
    recent_drop = (val[lo] - val[best_epoch - 1]) if best_epoch - 1 < len(val) else float("nan")
    print(f"  epochs_trained={et}  best_epoch={best_epoch}  best_val={best:.4f}  "
          f"(patience={STOP_AFTER})")
    print(f"  val-loss drop over the 5 epochs before best: {recent_drop:+.4f}")
    print("  Read the curve: still descending at the best epoch => UNDER-FIT (raise STOP_AFTER /")
    print("  capacity, re-run). Clean plateau => CONVERGED (=> the wide/biased marginals are a")
    print("  data/identifiability limit; go to steps 4-5).")
else:
    print("  No validation curve captured (unexpected).")
print("\nNext: re-check calibration on the new posterior with")
print(f'  $env:POST="{NAME}.pt"; & "<biophys-env python>" scripts/sbc_characterize.py')
print("RETRAIN_CONVERGENCE_DONE", flush=True)
