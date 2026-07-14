import math
from typing import Any

import torch
from tqdm import tqdm

from core import config


def _step_iter(n: int, batch_size: int):
    """tqdm wrapper for the per-step loop with overhead-minimizing settings.

    `miniters` controls how often tqdm checks for a display refresh.
    With ~1% of total per check, the per-iteration cost is just a counter
    increment, which is essentially free even at 2.4M iterations.

    This bar runs in the GUI too, not just the CLI: its rendered `it/s` is what drives the GUI's
    "Solver Performance" meter, and its percentage is the only thing that moves during a training
    iteration (which takes ~10s, so the top-level bar looks frozen between ticks). The GUI finds it by
    its desc -- hence config.SOLVER_BAR_DESC rather than a literal -- and shows it in a dedicated widget
    rather than as a progress row. See core/gui/widgets/progress_pane.py.
    """
    return tqdm(
        range(n - 1),
        desc=f"{config.SOLVER_BAR_DESC} (batch={batch_size})",
        leave=False,
        mininterval=1.0,
        miniters=max(1000, n // 100),
    )

class Solver:
    def __init__(self):
        def euler(sde: Any, x0: torch.Tensor, ts: tuple[float, float], n: int, state_dep_drift: bool = False) -> torch.Tensor:
            """
            Explicit Euler-Maruyama SDE solver.

            Noise is assumed diagonal: `sde.g(...)` returns a (batch, d) vector of
            per-channel amplitudes, and the update is `x + f*dt + g*dW` elementwise.
            """
            x0 = x0.to(sde.device)

            t = torch.linspace(*ts, n, device=x0.device)
            dt = t[1].item() - t[0].item()
            sqrt_dt = math.sqrt(dt)

            batch_size, d = x0.shape

            xs = torch.zeros((n, batch_size, d), dtype=x0.dtype, device=x0.device)
            xs[0, :, :] = x0

            # Pre-allocated dW buffer reused every step.
            dW_buf = torch.empty((batch_size, d), dtype=x0.dtype, device=x0.device)

            if not state_dep_drift:
                g = sde.g()  # (batch, d)
                for i in _step_iter(n, batch_size):
                    x_curr = xs[i, :, :]
                    dW_buf.normal_()
                    eta = g * dW_buf * sqrt_dt
                    xs[i + 1, :, :] = x_curr + sde.f(x_curr, i) * dt + eta
            else:
                for i in _step_iter(n, batch_size):
                    x_curr = xs[i, :, :]
                    g = sde.g(x_curr)  # (batch, d)
                    dW_buf.normal_()
                    eta = g * dW_buf * sqrt_dt
                    xs[i + 1, :, :] = x_curr + sde.f(x_curr, i) * dt + eta

            return xs

        def implicit_euler(sde: Any, x0: torch.Tensor, ts: tuple[float, float], n: int, max_iter: int = 10, tol: float = 1e-6) -> torch.Tensor:
            """
            Implicit Euler-Maruyama SDE solver
            :param sde: SDE class containing drift and diffusion functions
            :param x0: initial conditions; shape: (batch size, d)
            :param ts: times span to solve SDEs for
            :param n: number of time steps
            :param max_iter: maximum number of iterations when performing the update step x_{n+1} = x_n + f(t_{n+1}, x_{n+1})dt + g(t_n, x_n)dW
            :param tol: tolerance to check for convergence
            :return: tensor of the solution; shape: (n, batch size, d)
            """
            x0 = x0.to(sde.device)

            # time info
            t = torch.linspace(*ts, n)
            dt = t[1].item() - t[0].item() # fixed time step

            # create tensor of the solution
            batch_size, d = x0.shape
            xs = torch.zeros((n, batch_size, d), dtype=x0.dtype, device=x0.device)
            xs[0, :, :] = x0

            # time and state independent drift
            g = sde.g()

            # pre-compute constants
            sqrt_dt = math.sqrt(dt)

            # Pre-allocated dW buffer reused every step.
            dW_buf = torch.empty((batch_size, d), dtype=x0.dtype, device=x0.device)

            for i in _step_iter(n, batch_size):
                x_curr = xs[i, :, :]
                dW_buf.normal_()
                eta = g * dW_buf * sqrt_dt
                x_next = x_curr.clone()
                for _ in range(max_iter):
                    x_temp = x_curr + sde.f(x_next, i) * dt + eta
                    if torch.norm(x_temp - x_next) < tol:
                        break
                    x_next = x_temp
                xs[i + 1, :, :] = x_next

            return xs

        def euler_compiled(sde: Any, x0: torch.Tensor, ts: tuple[float, float], n: int, state_dep_drift: bool = False) -> torch.Tensor:
            """
            Euler-Maruyama via the model's `compiled_step` (torch.compile + CUDA Graphs).

            The model must expose:
              - `compiled_step(x, force_step, dW, *params, dt, sqrt_dt) -> next_x`
                wrapped with `@torch.compile(mode='reduce-overhead')`
              - `f_pure(x, force_step)` (used by the compiled step when traced)
              - `compiled_params()` returning the params tuple

            Diagonal-noise assumption: the compiled step is responsible for computing
            its own g (constant or state-dependent) internally.
            """
            x0 = x0.to(sde.device)

            t = torch.linspace(*ts, n, device=x0.device)
            dt = t[1].item() - t[0].item()
            sqrt_dt = math.sqrt(dt)

            batch_size, d = x0.shape

            xs = torch.zeros((n, batch_size, d), dtype=x0.dtype, device=x0.device)
            xs[0, :, :] = x0

            step = sde.compiled_step
            params = sde.compiled_params()
            dW_buf = torch.empty((batch_size, d), dtype=x0.dtype, device=x0.device)

            x = x0
            for i in _step_iter(n, batch_size):
                dW_buf.normal_()
                force_step = sde.force[:, :, i]
                x = step(x, force_step, dW_buf, *params, dt, sqrt_dt)
                xs[i + 1, :, :] = x

            return xs

        self.euler = euler
        self.implicit_euler = implicit_euler
        self.euler_compiled = euler_compiled