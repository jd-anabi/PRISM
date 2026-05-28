"""
Dev-only profiling harness for the Euler-Maruyama SDE solver.

Usage:
    python scripts/profile_sde.py [--steps 100000] [--batch 2048] [--trials 3] [--no-trace]

Reports median wall-clock per simulator call (CUDA-synced), then runs one extra
call under torch.profiler and prints the top-CUDA-time op table. With --trace,
also dumps a chrome trace to Resources/profiles/sde_trace_<ts>.json.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

# Allow running from repo root: `python scripts/profile_sde.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.profiler import ProfilerActivity, profile

from core.Simulator.hopf_simulator import HopfSimulator


def _build_hopf(batch: int, n_steps: int, device: torch.device, dtype: torch.dtype,
                use_compile: bool | None = None) -> HopfSimulator:
    """Build a Hopf simulator with reasonable parameters and zero forcing for profiling."""
    # Hopf params: (mu, beta, sigma_x, sigma_y) per ensemble member
    mu = torch.full((batch,), -0.1, dtype=dtype)
    beta = torch.full((batch,), 0.5, dtype=dtype)
    sigma_x = torch.full((batch,), 0.05, dtype=dtype)
    sigma_y = torch.full((batch,), 0.05, dtype=dtype)
    params = torch.stack((mu, beta, sigma_x, sigma_y), dim=1).to(device)

    # Zero forcing — profiling the solver, not the forcing tensor.
    # Shape (batch, n_channels=2, n_steps). The SDE indexes [:, 0, t] / [:, 1, t].
    force = torch.zeros((batch, 2, n_steps), dtype=dtype, device=device)

    inits = torch.zeros((batch, 2), dtype=dtype, device=device)
    t = torch.linspace(0.0, n_steps * 0.01, n_steps, dtype=dtype, device=device)

    return HopfSimulator(
        params=params, force=force, inits=inits, t=t,
        freqs_per_batch=1, segs=1, batch_size=batch, device=device,
        use_compile=use_compile,
    )


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _time_call(sim: HopfSimulator, device: torch.device) -> float:
    _sync(device)
    t0 = time.perf_counter()
    sim.simulate(state_dep_drift=False)
    _sync(device)
    return time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=100_000,
                    help="Fine integration steps per call (default 100000).")
    ap.add_argument("--batch", type=int, default=2048,
                    help="Ensemble size (default 2048).")
    ap.add_argument("--trials", type=int, default=3,
                    help="Number of timed trials (default 3).")
    ap.add_argument("--warmup", type=int, default=2,
                    help="Untimed warmup calls before measurement (default 2 — first includes torch.compile).")
    ap.add_argument("--device", type=str, default=None,
                    help="Force device (cuda / cpu). Default: auto-detect.")
    ap.add_argument("--no-trace", action="store_true",
                    help="Skip the torch.profiler pass.")
    ap.add_argument("--no-compile", action="store_true",
                    help="Force the eager path (skip torch.compile + CUDA Graphs).")
    args = ap.parse_args()

    if args.device is not None:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    dtype = torch.float32

    use_compile = False if args.no_compile else None  # None = auto
    print(f"=== SDE profile ===")
    print(f"device={device}  dtype={dtype}  batch={args.batch}  steps={args.steps}")
    if device.type == "cuda":
        print(f"  gpu={torch.cuda.get_device_name(device)}")
    print(f"  path={'eager (--no-compile)' if args.no_compile else 'auto (compile if CUDA)'}")
    print()

    sim = _build_hopf(args.batch, args.steps, device, dtype, use_compile=use_compile)

    # Warmup (CUDA kernels, allocator, etc.)
    for _ in range(args.warmup):
        sim.simulate(state_dep_drift=False)
    _sync(device)

    # Timed trials
    timings = [_time_call(sim, device) for _ in range(args.trials)]
    med = statistics.median(timings)
    spread = (max(timings) - min(timings)) / 2
    per_step_us = med / args.steps * 1e6

    print("Wall-clock per simulate() call:")
    for i, t in enumerate(timings):
        print(f"  trial {i}: {t*1000:.1f} ms")
    print(f"  median: {med*1000:.1f} ms  (±{spread*1000:.1f} ms)")
    print(f"  per-step: {per_step_us:.2f} µs   (batch={args.batch})")
    print()

    if args.no_trace:
        return

    # One profiler pass
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    print("Profiler pass (one call)...")
    with profile(activities=activities, record_shapes=False) as prof:
        sim.simulate(state_dep_drift=False)
        _sync(device)

    sort_key = "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
    print(prof.key_averages().table(sort_by=sort_key, row_limit=20))

    # Chrome trace
    out_dir = Path(__file__).resolve().parents[1] / "Resources" / "profiles"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    trace_path = out_dir / f"sde_trace_{stamp}.json"
    prof.export_chrome_trace(str(trace_path))
    print(f"\nChrome trace: {trace_path}")
    print("Open in chrome://tracing or https://ui.perfetto.dev")


if __name__ == "__main__":
    main()
