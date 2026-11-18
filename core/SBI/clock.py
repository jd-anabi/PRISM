"""The record clock: how a simulated cell is put on a recording's frames.

A bench camera records a cell at one frame interval, ``dt_exp``; a cell of time scale ``t_scale`` covers
``Delta = dt_exp / t_scale`` of its own (model) time between frames. A simulation settles first, for
``steady_idx`` steps at exactly ``dt_nd_min``, the step the prior's stability screen certified, with the drive
already running; that settling is discarded, and the recording starts at its end. From there the solver takes
``s`` steps per frame, each ``h = Delta / s``, so every kept point is exactly one frame after the last in the
cell's own time and the cell runs at exactly the time scale its row is labelled with.

THE STRIDE. ``s = ceil((1 - 1e-6) r)`` with ``r = Delta / dt_nd_min``, between 1 and 40 on the built-in box:
the smallest number of steps per frame that keeps the recorded step at or below the certified one. An explicit
Euler-Maruyama step coarser than the screen's can blow up on a parameter set the prior admitted; a finer one
stays inside the region it tested. The factor below 1 absorbs a float32 time scale on a whole ``r``. So
``dt_nd_min / 2 < h <= dt_nd_min``, and ``h = dt_nd_min`` exactly where ``r`` is whole; above the box's top
(``r < 1``, an arm of a central difference) ``s = 1`` and ``h = Delta``, finer still. Never round: rounding
would make the step up to half again coarser than the certified one.

THE DRIVE'S CLOCK. The recording and the drive share one origin, the first kept frame, as a bench drive and its
camera do: a probe's cosine stands at its top there and a forced drive at its drawn phase. Through the transient
the drive runs at negative times, from the transient's own step, so the cell is in its driven steady state at
the first frame.

WHAT A ROW HOLDS. A kept trace is ``gen_obs(...)[..., ::s][..., :n_points]`` (gen_obs already drops the
transient), one frame apart in the cell's time, read with ``dt_exp`` as its true spacing.

``RECORD_CLOCK`` names this clock in a simulation set's identity, so that no set simulated on another is ever
continued or trained on as one.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

RECORD_CLOCK = "frame-exact/1"

_BELOW_ONE = 1.0 - 1e-6


def stride_for(r: float) -> int:
    """Fine steps per frame for a frame of ``r`` certified steps: ``ceil((1 - 1e-6) r)``, at least 1."""
    return max(1, math.ceil(_BELOW_ONE * float(r)))


@dataclass(frozen=True)
class FrameClock:
    """One batch's clock: the transient, then the recorded part, as the solver integrates them.

    :param stride: fine steps per frame, ``s``.
    :param step: the recorded step ``h`` in model time: one float, or a (B,) float64 tensor, one per row.
    :param t_scale: the time scale it was built for: one float, or a (B,) float64 tensor.
    :param n_transient: the transient's steps, ``steady_idx``; the first kept frame is that fine index.
    :param transient_step: the transient's step, ``dt_nd_min``.
    :param n_points: frames kept.
    :param dt_exp: the frame interval in physical time.
    :param chunk_len: the most steps one recorded segment integrates (the caller's ``CHUNK_LEN``).
    """
    stride: int
    step: object
    t_scale: object
    n_transient: int
    transient_step: float
    n_points: int
    dt_exp: float
    chunk_len: int

    @property
    def per_row(self) -> bool:
        return torch.is_tensor(self.step)

    @property
    def n_fine(self) -> int:
        """Fine points of the whole run: the transient's steps plus ``stride`` per frame."""
        return self.n_transient + self.stride * self.n_points

    def segments(self) -> list:
        """The solver's plan (``Simulator.simulate``): the transient as one segment at ``transient_step``,
        then the recorded part at ``step``, split into at most ``chunk_len`` steps each."""
        last = self.n_fine - 1
        out = []
        if self.n_transient > 0:
            out.append((0, self.n_transient, self.transient_step))
        a = self.n_transient
        while a < last:
            b = min(a + self.chunk_len, last)
            out.append((a, b, self.step))
            a = b
        return out

    @property
    def n_segments(self) -> int:
        return len(self.segments())

    def nd_grid(self, dtype=torch.float32, device=None) -> torch.Tensor:
        """The model time of every fine point, built in float64 and cast. Defined for a one-step clock only:
        a per-row clock has a grid per row, and the solver reads only a grid's length (:meth:`grid`)."""
        if self.per_row:
            raise ValueError("a clock with a step per row has no single model-time grid; use grid()")
        g = torch.arange(self.n_fine, dtype=torch.float64)
        t = torch.where(g <= self.n_transient, g * self.transient_step,
                        self.n_transient * self.transient_step + (g - self.n_transient) * float(self.step))
        return t.to(dtype=dtype, device=device)

    def grid(self, dtype=torch.float32, device=None) -> torch.Tensor:
        """The grid handed to gen_obs: the model time for a one-step clock, a length-only grid (zeros) for a
        per-row one. The solver reads only its length; the steps come from :meth:`segments`."""
        if self.per_row:
            return torch.zeros(self.n_fine, dtype=dtype, device=device)
        return self.nd_grid(dtype, device)

    def drive_times(self, dtype=torch.float32, device=None) -> torch.Tensor:
        """The physical time of every fine point, counted from the first kept frame (a row's ``t_offset`` is
        added by the force builder): through the transient ``(j - n_transient) * transient_step * t_scale``,
        negative and reaching 0 exactly at the first kept frame; from there ``(j - n_transient) * dt_exp /
        stride``, which does not depend on the time scale. Built in float64 and cast once. ``(n_fine,)`` for
        a one-step clock, ``(B, n_fine)`` for a per-row one (only the transient's part differs by row)."""
        n_tr = self.n_transient
        j_tr = torch.arange(n_tr + 1, dtype=torch.float64) - n_tr                     # .., -1, 0
        rec = (torch.arange(1, self.n_fine - n_tr, dtype=torch.float64) * (self.dt_exp / self.stride))
        if self.per_row:
            ts = self.t_scale.to(dtype=torch.float64).reshape(-1, 1)
            tr = (j_tr.unsqueeze(0) * self.transient_step) * ts
            return torch.cat([tr.to(dtype=dtype, device=device),
                              rec.to(dtype=dtype, device=device).unsqueeze(0).expand(ts.shape[0], -1)], dim=1)
        tr = (j_tr * self.transient_step) * float(self.t_scale)
        return torch.cat([tr, rec]).to(dtype=dtype, device=device)

    def frames(self, x_past_transient: torch.Tensor) -> torch.Tensor:
        """The kept frames of a trace gen_obs returned (its fine points past the transient)."""
        return x_past_transient[..., ::self.stride][..., :self.n_points]


def frame_clock(dt_exp, t_scale, n_points, dt_nd_min, steady_idx, chunk_len, *, stride=None) -> FrameClock:
    """The clock of a batch at one time scale (the training batch, a simulated observation, a probe check)."""
    delta = float(dt_exp) / float(t_scale)
    s = stride_for(delta / float(dt_nd_min)) if stride is None else int(stride)
    return FrameClock(stride=s, step=delta / s, t_scale=float(t_scale), n_transient=int(steady_idx),
                      transient_step=float(dt_nd_min), n_points=int(n_points), dt_exp=float(dt_exp),
                      chunk_len=int(chunk_len))


def rows_clock(dt_exp, t_scales, n_points, dt_nd_min, steady_idx, chunk_len) -> FrameClock:
    """The clock of rows at their own time scales (a predictive-check bin; the arms of a central difference):
    ONE stride, from the smallest time scale, and one step per row, so the rows share their step count and
    their draws. Every row's step is then at or below the certified one."""
    ts = torch.as_tensor(t_scales, dtype=torch.float64).reshape(-1)
    delta = float(dt_exp) / ts
    s = stride_for(float(delta.max()) / float(dt_nd_min))
    return FrameClock(stride=s, step=delta / s, t_scale=ts, n_transient=int(steady_idx),
                      transient_step=float(dt_nd_min), n_points=int(n_points), dt_exp=float(dt_exp),
                      chunk_len=int(chunk_len))


def training_n_fine(dt_exp, t_scales, Ts, dt_nd_min, steady_idx) -> torch.Tensor:
    """:class:`FrameClock` ``n_fine`` for every (t_scale, T) pair of a schedule, vectorised, in float64 from the
    schedule's own (float32) values: the arithmetic of the batch loop, which reads each pair back as a Python
    float, so the schedule's filter and the loop share one."""
    ts = torch.as_tensor(t_scales).to(torch.float64)
    T = torch.as_tensor(Ts).to(torch.float64)
    delta = float(dt_exp) / ts
    s = torch.clamp(torch.ceil(_BELOW_ONE * (delta / float(dt_nd_min))), min=1)
    n_points = torch.floor((T / ts) / delta)
    return (int(steady_idx) + s * n_points).to(torch.long)
