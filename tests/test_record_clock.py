"""The record clock (core/SBI/clock.py): how a simulated cell is put on a recording's frames. The stride and the
recorded step at every time scale; the transient at the step the prior's screen certified, as a segment of its own;
the drive's time counted from the first kept frame; the schedule's filter and the batch loop on one arithmetic; and
a response equal to its drive, read through the clock and the lock-in, reading 1 at phase 0."""
import math

import numpy as np
import torch

from core import config, forcing
from core.Helpers import helpers
from core.SBI import chi, clock

# the built-in box: frames 1 ms apart, t_scale up to 40 ms, so dt_nd_min = 0.025 and the transient 4000 steps
DT_EXP, DT_MIN, STEADY = 1.0, 0.025, 4000


def _clk(t_scale, n_points=10000, **kw):
    return clock.frame_clock(DT_EXP, t_scale, n_points, DT_MIN, STEADY, config.CHUNK_LEN, **kw)


def test_the_stride_and_the_recorded_step_at_every_time_scale():
    """The stride is ceil((1 - 1e-6) r), r = Delta / dt_nd_min = 40 / t_scale: the smallest whole number of
    steps per frame whose step is no coarser than the certified one, never the nearest (which would make the step
    up to half again coarser). A 10-s record's fine steps follow: the transient's 4000 plus the stride per frame."""
    table = {3.0: (14, 0.02381, 144), 3.73: (11, 0.02437, 114), 6.0: (7, 0.02381, 74), 7.215: (6, 0.0231, 64),
             12.0: (4, 0.02083, 44), 13.9: (3, 0.02398, 34), 16.0: (3, 0.02083, 34), 22.0: (2, 0.02273, 24),
             25.0: (2, 0.02, 24), 27.0: (2, 0.01852, 24), 30.0: (2, 0.01667, 24), 37.13: (2, 0.01347, 24)}
    for ts, (s, h, k_steps) in table.items():
        c = _clk(ts)
        assert (c.stride, round(c.step, 5), c.n_fine) == (s, h, k_steps * 1000), (ts, c.stride, c.step, c.n_fine)
    for ts, s in ((1.0, 40), (2.0, 20), (5.0, 8), (40.0, 1)):       # a whole r: the certified step exactly
        c = _clk(ts)
        assert c.stride == s and c.step == DT_MIN, (ts, c.stride, c.step)
    # the factor below 1 absorbs a float32 time scale on a whole r: one float32 step below 5 ms is still 8
    assert _clk(float(np.nextafter(np.float32(5.0), np.float32(0.0)))).stride == 8


def test_the_recorded_step_is_never_coarser_than_the_certified_one():
    """Over 10,000 time scales of the box, dt_nd_min / 2 < h <= dt_nd_min (to one part in a million); above the
    box's top (r < 1, where an arm of a central difference may reach) the stride is 1 and the step is the frame
    itself, finer still."""
    g = torch.Generator().manual_seed(17)
    ts = torch.exp(torch.rand(10000, generator=g, dtype=torch.float64) * math.log(40.0))
    for t in ts.tolist():
        h = _clk(t).step
        assert DT_MIN / 2 < h <= DT_MIN * (1 + 1e-6), (t, h)
    c = _clk(40.12)
    assert c.stride == 1 and c.step == DT_EXP / 40.12 < DT_MIN


def test_the_transient_runs_at_the_certified_step_and_the_drive_starts_at_the_first_frame():
    """The transient is today's: steady_idx steps at exactly dt_nd_min, a segment of its own; the step switches to
    Delta / s at the first kept frame, which is the transient's end. The drive's time is 0 exactly there, negative
    through the transient (from the transient's own step), and one frame per stride after it."""
    for ts in (3.73, 13.9, 30.0):
        c = _clk(ts, n_points=25000)
        seg = c.segments()
        assert seg[0] == (0, STEADY, DT_MIN)
        assert seg[1][0] == STEADY and all(st == c.step for _, _, st in seg[1:])
        assert all(b - a <= config.CHUNK_LEN for a, b, _ in seg[1:]) and seg[-1][1] == c.n_fine - 1
        assert c.n_segments == 1 + math.ceil((c.n_fine - 1 - STEADY) / config.CHUNK_LEN)
        t = c.drive_times(dtype=torch.float64)
        assert t.shape == (c.n_fine,) and float(t[STEADY]) == 0.0
        assert torch.allclose(t[1:STEADY + 1] - t[:STEADY], torch.full((STEADY,), DT_MIN * ts, dtype=torch.float64))
        assert torch.allclose(t[STEADY + 1:] - t[STEADY:-1],
                              torch.full((c.n_fine - 1 - STEADY,), DT_EXP / c.stride, dtype=torch.float64))
        frames = c.frames(t[STEADY:].unsqueeze(0))[0]
        assert torch.allclose(frames, torch.arange(c.n_points, dtype=torch.float64) * DT_EXP, rtol=0, atol=1e-9)
        nd = c.nd_grid(dtype=torch.float64)
        assert float(nd[STEADY]) == STEADY * DT_MIN
        assert torch.allclose(nd[STEADY:][::c.stride][:c.n_points] - STEADY * DT_MIN,
                              torch.arange(c.n_points, dtype=torch.float64) * (DT_EXP / ts), rtol=1e-12, atol=1e-9)


def test_a_clock_with_a_step_per_row_shares_one_stride():
    """A predictive-check bin's rows, or a central difference's arms, sit at their own time scales: one stride,
    from the smallest, and a step per row, all at or below the certified one; the drive's recorded part is the
    same for every row, its transient part each row's own."""
    ts = torch.tensor([3.73, 5.0, 12.0, 39.0], dtype=torch.float64)
    c = clock.rows_clock(DT_EXP, ts, 300, DT_MIN, STEADY, config.CHUNK_LEN)
    assert c.per_row and c.stride == 11
    assert torch.allclose(c.step, (DT_EXP / ts) / 11) and bool((c.step <= DT_MIN * (1 + 1e-6)).all())
    t = c.drive_times(dtype=torch.float64)
    assert t.shape == (4, c.n_fine) and bool((t[:, STEADY] == 0).all())
    assert torch.equal(t[0, STEADY:], t[3, STEADY:])
    assert torch.allclose(t[:, 0], -STEADY * DT_MIN * ts)
    assert c.grid().shape == (c.n_fine,)
    try:
        c.nd_grid()
        raise AssertionError("a per-row clock gave one model-time grid")
    except ValueError:
        pass


def test_the_schedule_filter_and_the_batch_loop_share_one_arithmetic():
    """training_n_fine, vectorised in float64 from the schedule's float32 values, equals the clock the batch loop
    builds from each pair read back as a Python float, over a whole Sobol draw of the built-in box and of a model
    builder's box (t_scale in (v/2, 2v), whose transient is long)."""
    for lo, hi, steady, dt_min in ((1.0, 40.0, STEADY, DT_MIN), (150.0, 600.0, 120_000, DT_EXP / 600.0)):
        with torch.random.fork_rng(devices=[]):                     # the scrambled engine draws at construction
            torch.manual_seed(3)
            pts = torch.quasirandom.SobolEngine(dimension=2, scramble=True).draw(3000)
        ts32 = torch.exp(math.log(lo) + pts[:, 0] * (math.log(hi) - math.log(lo)))
        T32 = torch.exp(math.log(1000.0) + pts[:, 1] * (math.log(60000.0) - math.log(1000.0)))
        got = clock.training_n_fine(DT_EXP, ts32, T32, dt_min, steady)
        want = []
        for a, b in zip(ts32.tolist(), T32.tolist()):
            n_points = int((b / a) / (DT_EXP / a))
            want.append(clock.frame_clock(DT_EXP, a, n_points, dt_min, steady, config.CHUNK_LEN).n_fine)
        assert got.tolist() == want


def test_the_schedule_keeps_the_first_pairs_that_fit_however_it_draws():
    """The schedule is the first n_runs pairs of the scrambled Sobol sequence that fit, whatever the size of its
    draws, and an empty redraw is taken to mean that nothing fits only when the redraw is large: a two-batch run on a
    12,000-step grid, where a frame takes at least two fine steps, used to redraw six candidates at a time and give
    up when all six missed (as it does at this seed)."""
    from core.SBI import pipeline
    n_grid, steady = 12_000, 500
    t = torch.zeros(n_grid)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(9)
        ts, Ts = pipeline._batch_schedule(2, t, (1.0, 40.0), 1000.0, 60000.0, DT_EXP, DT_MIN, steady)
        torch.manual_seed(9)
        pts = torch.quasirandom.SobolEngine(dimension=2, scramble=True).draw(4096)
    want_ts = torch.exp(pts[:, 0] * math.log(40.0))
    want_Ts = torch.exp(math.log(1000.0) + pts[:, 1] * (math.log(60000.0) - math.log(1000.0)))
    fits = clock.training_n_fine(DT_EXP, want_ts, want_Ts, DT_MIN, steady) <= n_grid
    assert torch.equal(ts, want_ts[fits][:2]) and torch.equal(Ts, want_Ts[fits][:2])


def _ideal_reading(t_scale, f_khz, n_cycles):
    """A response equal to its drive, through the clock and the drive builder, locked in over whole cycles."""
    c = _clk(t_scale, n_points=int(round(n_cycles / (f_khz * DT_EXP))))
    ridx = {"x_scale": 0, "t_scale": 1, "f_scale": 2}
    resc = torch.tensor([[62.14, t_scale, 10.0]])
    amp = 0.15 * 10.0
    fp = torch.tensor([[amp, f_khz, math.pi / 2.0, 0.0]])
    drive = forcing.build_nondim_sin_force_tensor(fp, None, resc, {"amp": 0, "freq": 1, "phase": 2, "offset": 3},
                                                  ridx, t_dim=c.drive_times())
    x = helpers.rescale(c.frames(drive[:, 0, STEADY:]), 10.0)
    return complex(chi.lock_in_batched(x, torch.tensor([2 * math.pi * f_khz]), amp, c.n_points * DT_EXP, DT_EXP)[0])


def test_an_ideal_response_reads_one_at_phase_zero_on_the_clock():
    """A response equal to its drive reads |chi| = 1 at phase 0 at every time scale: within the lock-in's
    finite-window term (0.006 at 2 cycles, 0.001 at 20 over whole cycles), and |arg chi| <= 1e-3 rad. Over 220
    log-uniform time scales the worst 20-cycle reading is within 1e-3 of 1. (On the old clock a whole-number stride
    put the trace's cell at another time scale: an ideal response read 0.23 at 20 cycles in the median.)"""
    special = [1.0, 2.0, 3.0, 3.73, 4.0, 5.0, 6.0, 7.215, 8.0, 10.0, 12.0, 13.9, 16.0, 18.0, 20.0, 22.0, 25.0,
               27.0, 30.0, 33.0, 37.13, 40.0]
    for ts in special:
        for n_cycles, tol in ((2, 0.006), (5, 0.003), (10, 0.002), (20, 0.001)):
            r = _ideal_reading(ts, 0.01, n_cycles)
            assert abs(abs(r) - 1.0) <= tol and abs(math.atan2(r.imag, r.real)) <= 1e-3, (ts, n_cycles, r)
    grid = [math.exp(i * math.log(40.0) / 199) for i in range(200)]
    worst = max(abs(abs(_ideal_reading(ts, 0.01, 20)) - 1.0) for ts in grid)
    assert worst <= 1e-3, worst


def test_the_drive_builder_takes_the_clocks_times():
    """Handed the times (``t_dim``), the builder adds each row's t_offset and builds exactly what it builds from
    a model-time grid scaled by the row's time scale; t_scale is still read for a Hopf-style force scale."""
    ridx = {"x_scale": 0, "t_scale": 1, "t_offset": 2, "f_scale": 3}
    resc = torch.tensor([[62.0, 3.73, 0.4, 10.0], [40.0, 12.0, 0.0, 7.0]])
    fp = torch.tensor([[1.0, 0.05, 0.3, 2.0], [2.0, 0.02, 1.1, -1.0]])
    fidx = {"amp": 0, "freq": 1, "phase": 2, "offset": 3}
    t_nd = torch.linspace(0, 30, 500)
    old = forcing.build_nondim_sin_force_tensor(fp, t_nd, resc, fidx, ridx)
    t_dim = helpers.rescale(t_nd.unsqueeze(0), resc[:, 1:2], 0.0)
    new = forcing.build_nondim_sin_force_tensor(fp, None, resc, fidx, ridx, t_dim=t_dim)
    assert torch.equal(old, new)
    hopf = {"x_scale": 0, "t_scale": 1}
    a = forcing.build_nondim_force_tensor(fp, t_nd, resc[:, :2], fidx, hopf)
    b = forcing.build_nondim_force_tensor(fp, None, resc[:, :2], fidx, hopf,
                                          t_dim=helpers.rescale(t_nd.unsqueeze(0), resc[:, 1:2], 0.0))
    assert torch.equal(a, b)


def _ideal_gen_obs(model, params, t, inits, force, n_segs, steady_idx, **kw):
    """The simulator replaced by an ideal system: the response is the drive, past the transient."""
    return force[:, 0:1, steady_idx:].permute(1, 0, 2).clone()


def test_the_probe_loop_reads_an_ideal_response_on_the_clock(monkeypatch):
    """gen_chi_raw itself -- its placement, Nyquist test, cycle floor and ceiling, and lock-in -- with the simulator
    replaced by "response = drive", at ten time scales between whole strides and at the whole strides: the mean
    |chi| over the live probes is within 1 % of the truth and the median |phase| at most 0.01 rad (on the old clock
    the means ran 0.09-0.85 between whole strides, the phases scrambled). Every live probe reads the closed-form
    lock-in of a cosine from its top at the first frame over that probe's own window, |chi| within 1e-6."""
    from core.SBI import chi_probes, pipeline
    monkeypatch.setattr(pipeline, "gen_obs", _ideal_gen_obs)
    ridx = {"x_scale": 0, "t_scale": 1, "f_scale": 2}
    N, f_peak, X, F = 5000, 0.02, 62.14, 10.0
    x_spont = (X * torch.sin(2 * math.pi * f_peak * torch.arange(N, dtype=torch.float32) * DT_EXP)).unsqueeze(0)
    peak = float(chi.peak_freq(x_spont, DT_EXP)[0])
    mults = chi.sample_multipliers(12, config.CHI_FREQ_BOUNDS, generator=torch.Generator().manual_seed(20260805))
    for ts in (3.73, 6.0, 7.215, 12.0, 13.9, 16.0, 22.0, 25.0, 30.0, 37.13, 1.0, 2.0, 5.0, 8.0, 20.0, 40.0):
        c = clock.frame_clock(DT_EXP, ts, N, DT_MIN, STEADY, config.CHUNK_LEN)
        chis, u, logcyc, valid = chi_probes.gen_chi_raw(
            "NADROWSKI", torch.zeros(1, 10), torch.tensor([[X, ts, F]]), x_spont, inits=torch.zeros(1, 3),
            rescale_idx=ridx, multipliers=mults, f0_nd=0.15, adapt_placement=True, bounds=config.CHI_FREQ_BOUNDS,
            clock=c)
        live = valid[0]
        assert int(live.sum()) >= 6, (ts, int(live.sum()))
        r = chis[0][live] / (X / F)
        assert 0.99 <= float(r.abs().mean()) <= 1.01 and float(torch.angle(r).abs().median()) <= 0.01, ts
        for k in live.nonzero().flatten().tolist():
            f = peak * math.exp(float(u[0, k]))
            n = int(round(math.exp(float(logcyc[0, k])) / (f * DT_EXP)))
            ref = complex(chi.lock_in_batched(torch.cos(2 * math.pi * f * torch.arange(n, dtype=torch.float64)
                                                        * DT_EXP).unsqueeze(0),
                                              torch.tensor([2 * math.pi * f], dtype=torch.float64), 1.0, n * DT_EXP,
                                              DT_EXP)[0])
            got = complex(chis[0, k]) / (X / F)
            assert abs(abs(got) - abs(ref)) <= 1e-6 and abs(got - ref) <= 1e-4, (ts, k, got, ref)


def test_a_forced_drive_stands_at_its_drawn_phase_at_the_first_frame(monkeypatch):
    """A forced row's drive is built on the clock's times, so its drawn phase is its phase at the first kept frame,
    where Group G's lock-in starts: for a response equal to its drive, the first harmonic's phase against the drive's
    reads the same at every time scale and every drawn phase. (It used to carry 2 pi f t_scale times the transient's
    length besides, a scramble.)"""
    from core.SBI import pipeline, statistics
    monkeypatch.setattr(pipeline, "gen_obs", _ideal_gen_obs)
    fidx = {"amp": 0, "freq": 1, "phase": 2, "offset": 3}
    sim_ridx = {"x_scale": 0, "t_scale": 1, "f_scale": 2}
    fp = torch.tensor([[2.0, 0.05, 0.3, 0.0], [2.0, 0.05, 2.1, 0.0], [1.0, 0.11, 4.0, 0.0]])
    g2 = []
    for ts in (5.0, 13.9, 30.0):
        c = clock.frame_clock(DT_EXP, ts, 3000, DT_MIN, STEADY, config.CHUNK_LEN)
        resc = torch.tensor([[62.14, ts, 10.0]]).repeat(3, 1)
        g = pipeline._BatchGeometry(model="NADROWSKI", clock=c, t_fine=c.grid(), T_k=3000.0, n_force_ch=1,
                                    forcing_idx=fidx, sim_ridx=sim_ridx, fixed_dict=None, state_dep_drift=True,
                                    dtype=torch.float32, device=torch.device("cpu"))
        patho = dict.fromkeys(("rows", "nonfinite", "constant", "overflow"), 0)
        rows = pipeline._forced_rows(g, torch.zeros(3, 10), resc, torch.zeros(3, 3), fp,
                                     resc[:, 0:1], 0.0, patho)
        lab = statistics.FEATURE_LABELS
        g2.append(rows[:, [lab.index("G2_cos"), lab.index("G2_sin")]])
    g2 = torch.cat(g2)
    assert float((g2 - g2[0]).abs().max()) <= 1e-3, g2
