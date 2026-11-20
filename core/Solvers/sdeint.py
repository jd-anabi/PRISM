import logging
from typing import Any

import torch
from tqdm import tqdm

from core import config
from core.progress import SOLVER

log = logging.getLogger(__name__)


def _bar_kwargs(n: int, batch_size: int) -> dict:
    """The solver bar's settings, shared by both constructions so they cannot drift apart.

    `miniters` controls how often tqdm checks for a display refresh. At ~1% of total per check the
    per-iteration cost is a counter increment, essentially free even at 2.4M iterations.

    ⚠ `mininterval` IS 0.1 (tqdm's own default) AND MUST NOT GO BACK UP TO 1.0. It was 1.0, with a
    docstring explaining that a top-level iteration takes ~10 s. Since CUDA graphs landed a
    100k-step call takes ~0.7 s -- SHORTER than a 1.0 s mininterval -- so tqdm painted the "?it/s"
    opening frame and never a rate at all: measured 123 chars of stderr with no rate, against 670
    chars containing `14152.92it/s` with graphs off. The CLI was left watching a bar that never
    reports a speed. `miniters` gates refreshes as well and BOTH conditions must be met, but at
    max(1000, n // 100) it is not the binding one -- 1000 steps is ~14 ms at the graphed rate, far
    inside 100 ms -- which is why lowering mininterval alone is sufficient. AFTER, on the same call
    (RTX 5070 Ti, batch 2048, T=100k, 0.811 s, 123 349 steps/s): 678 chars carrying 7 rate frames,
    the first `128420.87it/s`. Pinned by
    `test_the_solver_bar_paints_a_rate_even_when_the_call_is_under_a_second`, which asserts the
    counterfactual too -- put mininterval back to 1.0 and that test fails rather than going quiet.
    """
    return dict(
        desc=f"{config.SOLVER_BAR_DESC} (batch={batch_size})",
        leave=False,
        mininterval=0.1,
        miniters=max(1000, n // 100),
    )


def _step_bar(n: int, batch_size: int):
    """The solver's per-step bar, driven manually via ``.update()``.

    The graphed path advances the time loop a CHUNK at a time, so it cannot wrap an iterable -- use
    :func:`_advance`, never ``bar.update()`` alone, or the step counter and the bar disagree.

    THE GUI NO LONGER READS THIS BAR'S TEXT. It used to: the "Solver Performance" meter was scraped
    from the rendered ``it/s``, which is exactly the coupling the speedup above broke. The rate now
    comes from ``core.progress.SOLVER``. What the GUI still does with this bar is EXCLUDE it -- by
    ``config.SOLVER_BAR_DESC`` -- from the election that drives the overall bar, since its total is
    in the tens of thousands and it would win every time, sweeping the bar 0->100% every second.

    It also stays ENABLED under the GUI, deliberately, rather than being quieted the way
    ``config.QUIET_SEGMENT_BAR`` quiets the segment bar: its redraws are the cooperative cancel's
    most frequent checkpoint through a long batch, because each redraw is a write to the run's stream
    and that write checks for a cancel (core/gui/streams.py).
    """
    return tqdm(total=n - 1, **_bar_kwargs(n, batch_size))


def _step_iter(n: int, batch_size: int):
    """The same bar as :func:`_step_bar`, as an ITERABLE over step indices, publishing as it goes.

    A generator rather than the bare tqdm object, so the count is published once here instead of in
    three separate caller loop bodies. Cost is one generator resume plus one integer add per step --
    ~50 ns against a MEASURED 54.87 us/step for the eager loop this path serves, i.e. under 0.1%.

    ⚠ ITERATE this; never hold it to call ``.close()``. The bar's close is tqdm's own ``finally``
    inside ``__iter__``, and that is what the cancel's tqdm-lock unwind relies on. Every call site is
    ``for i in _step_iter(...)``, which unwinds correctly: a GeneratorExit here propagates into that
    inner ``for``, so tqdm's finally still runs.
    """
    for i in tqdm(range(n - 1), **_bar_kwargs(n, batch_size)):
        SOLVER.add(1)
        yield i


def _advance(bar, k: int) -> None:
    """Advance the graphed path's bar AND the step counter by the same k.

    Kept together on purpose: they are two views of one quantity, and a caller that updates only one
    is a silent bug -- a graphed run that ticked once per replay would read 1/50th of the truth.
    """
    bar.update(k)
    SOLVER.add(k)


# --- CUDA Graph step capture ---------------------------------------------------------------------
# Keyed on everything the capture bakes in. NOT hung off the Solver class: the solver-failure test
# patches the class, which requires
# `sdeint.Solver` to be resolvable (and patchable) at call time, and a cache on a Solver instance
# would be useless anyway -- Solver is constructed once per TIME SEGMENT, so it would recapture
# constantly. A module-level dict is orthogonal to that seam.
#
# The PARAMETERS are static buffers copied into per call rather than captured by reference, which is
# what lets one graph serve every training batch at a given geometry. Capturing `params` directly
# would bake the address of one model's tensors and silently replay stale physics for the next. The
# STEP and its root are static buffers too, copied in the same way: every training batch integrates at
# its own step, so a step baked into the capture (or into its key) would capture once per batch.
_GRAPH_CACHE: dict = {}
_GRAPH_DISABLED = False          # set once if capture fails, so we do not retry every segment


def _graph_key(step, x0, force, chunk):
    return (id(step), tuple(x0.shape), int(force.shape[1]), chunk,
            str(x0.dtype), str(x0.device))


def step_tensors(dt, batch_size: int, dtype: torch.dtype, device) -> tuple[torch.Tensor, torch.Tensor]:
    """The step and its square root as ``(batch, 1)`` tensors of the state's dtype, one value per row.

    ``dt`` is a float or a tensor of one value or of one value per row. It is taken in double, its root is
    taken in double, and each is rounded ONCE to the state's dtype -- exactly what a step handed to the
    arithmetic as two Python floats computes, so the tensor form changes no number. A pair of times, the
    argument this replaced, is refused: the solver never derives a step from times.
    """
    if isinstance(dt, (tuple, list)):
        raise TypeError("the solver takes its step, not a pair of times; pass the step itself")
    d = (dt.to(dtype=torch.float64) if torch.is_tensor(dt)
         else torch.tensor(float(dt), dtype=torch.float64)).reshape(-1, 1)
    if d.shape[0] not in (1, batch_size):
        raise ValueError(f"a step for {d.shape[0]} rows cannot serve a batch of {batch_size}")
    d = d.expand(batch_size, 1)
    return (d.to(dtype=dtype, device=device).contiguous(),
            torch.sqrt(d).to(dtype=dtype, device=device).contiguous())


def drop_graph_cache() -> int:
    """Release every captured graph and its private pool. Returns how many entries were dropped.

    WHY THIS IS WORTH CALLING ON AN OOM PATH. A captured graph's memory lives in a PRIVATE pool that
    ``torch.cuda.empty_cache()`` cannot reclaim -- which is why SOLVER_GRAPH_CACHE_MAX bounds this
    cache rather than letting it grow. At an out-of-memory up to that many pools are pinned, and the
    halving retry that follows is about to capture ANOTHER one: ``tuple(x0.shape)`` is part of
    _graph_key, so 2048 -> 1024 -> 512 -> 256 is four distinct keys. Dropping the cache is therefore
    the one release that acts on the retry's OWN next allocation rather than on the failed attempt's.

    THE DROP IS A REQUEST, NOT A GUARANTEE. Clearing the dict removes our reference; the pool comes
    back when the CUDAGraph object is actually collected, so an entry still pinned by a live
    traceback frame is freed later rather than now. That is one more reason a release belongs OUTSIDE
    an ``except`` block, where the failed attempt's frames have already been dropped.

    Recapture costs three warm-up passes of SOLVER_GRAPH_CHUNK steps plus one device synchronize --
    nothing against the batch that just failed, and only paid if the run continues.
    """
    n = len(_GRAPH_CACHE)
    _GRAPH_CACHE.clear()
    return n


def _acquire_graph(step, params, x0, force, dt, sqrt_dt, chunk, ends: bool = False):
    """Capture (or fetch) a graph replaying ``chunk`` steps of ``step``. None => caller runs eager.

    ``dt`` and ``sqrt_dt`` are this call's ``(batch, 1)`` step tensors; the capture reads static copies of
    them, which the caller fills per call like the parameters. ``ends`` is the corrected step's: each of
    its steps reads the drive at its start AND at its end, so the drive buffer holds ``chunk + 1`` points,
    step j reading points j and j + 1, and the caller fills it with one point more than the chunk's steps
    (the last point of one replay is the first of the next)."""
    global _GRAPH_DISABLED
    if _GRAPH_DISABLED or not getattr(config, "SOLVER_CUDA_GRAPHS", True):
        return None
    if x0.device.type != "cuda" or not torch.cuda.is_available():
        return None
    key = _graph_key(step, x0, force, chunk)
    ent = _GRAPH_CACHE.get(key)
    if ent is not None:
        return ent
    try:
        B, d = x0.shape
        n_ch = int(force.shape[1])
        kw = dict(dtype=x0.dtype, device=x0.device)
        x_s = torch.zeros((B, d), **kw)
        dW_s = torch.empty((B, d), **kw)
        f_s = torch.zeros((B, n_ch, chunk + 1 if ends else chunk), **kw)
        out_s = torch.empty((chunk, B, d), **kw)
        p_s = tuple(p.clone() for p in params)
        dt_s, sq_s = dt.clone(), sqrt_dt.clone()

        def one(x, j):
            if ends:
                return step(x, f_s[:, :, j], f_s[:, :, j + 1], dW_s, *p_s, dt_s, sq_s)
            return step(x, f_s[:, :, j], dW_s, *p_s, dt_s, sq_s)

        # Warm-up on a side stream is required before capture; three passes is the documented
        # minimum that lets lazy allocations and any autotuning settle.
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                xx = x_s.clone()
                for j in range(chunk):
                    dW_s.normal_()
                    xx = one(xx, j)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            cur = x_s
            for j in range(chunk):
                dW_s.normal_()
                cur = one(cur, j)
                out_s[j] = cur
            # Chain the state forward INSIDE the graph, so consecutive chunks need no copy between
            # replays -- x_s is both the graph's input and its output.
            x_s.copy_(cur)

        if len(_GRAPH_CACHE) >= getattr(config, "SOLVER_GRAPH_CACHE_MAX", 8):
            _GRAPH_CACHE.pop(next(iter(_GRAPH_CACHE)))
        ent = {"graph": g, "x": x_s, "dW": dW_s, "force": f_s, "out": out_s,
               "params": p_s, "dt": dt_s, "sqrt_dt": sq_s, "chunk": chunk}
        _GRAPH_CACHE[key] = ent
        return ent
    except Exception as e:                     # noqa: BLE001 -- a solver must never die on a speedup
        _GRAPH_DISABLED = True
        log.warning(f"[solver] CUDA graph capture unavailable ({type(e).__name__}: {e}); "
                    f"falling back to the eager step loop for the rest of this process.")
        return None


class Solver:
    def __init__(self):
        def euler(sde: Any, x0: torch.Tensor, dt, n: int, state_dep_drift: bool = False) -> torch.Tensor:
            """
            Explicit Euler-Maruyama SDE solver: ``n`` points, ``n - 1`` steps of ``dt``.

            ``dt`` is the step, a float or one value per row (see :func:`step_tensors`); the caller hands
            it, and nothing here derives one from times. Noise is assumed diagonal: `sde.g(...)` returns a
            (batch, d) vector of per-channel amplitudes, and the update is `x + f*dt + g*dW` elementwise.
            """
            x0 = x0.to(sde.device)
            batch_size, d = x0.shape
            dt, sqrt_dt = step_tensors(dt, batch_size, x0.dtype, x0.device)

            # empty, not zeros: row 0 is assigned immediately below and _step_iter is range(n-1), so
            # the loop writes every remaining row. The zero-fill was a full memset of a buffer that
            # is overwritten in its entirety -- ~2.5 GB per segment, x segments x runs x 5000
            # training batches.
            xs = torch.empty((n, batch_size, d), dtype=x0.dtype, device=x0.device)
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

        def euler_compiled(sde: Any, x0: torch.Tensor, dt, n: int, state_dep_drift: bool = False) -> torch.Tensor:
            """
            Euler-Maruyama via the model's `compiled_step`, replayed from a CUDA Graph where possible:
            ``n`` points, ``n - 1`` steps of ``dt``, the step handed in as in :func:`euler`.

            The model must expose:
              - `compiled_step(x, force_step, dW, *params, dt, sqrt_dt) -> next_x`, a
                `@torch.jit.script` function whose step and root are ``(batch, 1)`` tensors
              - `compiled_params()` returning the params tuple, POSITIONALLY load-bearing
              - `f_pure(x, force_step)`

            Diagonal-noise assumption: the compiled step is responsible for computing
            its own g (constant or state-dependent) internally. That is why ``state_dep_drift`` is
            accepted and unused here: it tells the eager loop whether to evaluate ``sde.g`` once or on
            every step, while each model's compiled step carries its noise law inline.
            ``test_each_compiled_step_follows_its_models_registered_noise_flag`` runs both solvers
            with the registry's flag and live noise, so the flag and the compiled step cannot drift
            apart unseen.

            WHY THE GRAPH. TorchScript removes PYTHON overhead; it does not remove kernel-LAUNCH
            overhead, and that is what this loop is bound by -- measured 54.87 us/step eager against
            6.65 us/step replayed at batch 2048, i.e. ~88% of the time was the CPU submitting ~3
            kernels per step to a GPU that was ~6% utilised. The docstring here used to claim
            "torch.compile + CUDA Graphs"; neither existed anywhere in the repo, so the fast path it
            described was never built. It is built now.

            NOT bit-identical to the eager loop, because the noise is drawn inside the captured
            region and the draw order differs. That is the same licence ``gen_obs`` already takes when
            it splits a batch. The replay is pinned bitwise against this function's own un-graphed
            loop by ``test_the_cuda_graph_step_matches_the_eager_step_bitwise`` (noise zeroed, so the
            comparison isolates the chunking from the RNG); the compiled step's arithmetic is pinned
            against the eager model's drift and noise by
            ``test_nadrowski_compiled_step_matches_eager_and_keeps_its_arity``.
            """
            return _compiled_run(sde, x0, dt, n, sde.compiled_step, ends=False)

        def heun(sde: Any, x0: torch.Tensor, dt, n: int, state_dep_drift: bool = False) -> torch.Tensor:
            """
            The stochastic Heun predictor-corrector: ``n`` points, ``n - 1`` steps of ``dt``, the step handed
            in as in :func:`euler`. Each step takes the plain step as a prediction, evaluates the drift and the
            noise again where it lands, and averages the two, on the same increment:

                prediction  x* = x + f(x, F_i) dt + g(x) dW
                step        x' = x + (f(x, F_i) + f(x*, F_i+1)) dt / 2 + (g(x) + g(x*)) dW / 2

            The prediction is the plain step's arithmetic exactly, so it is that step bit for bit; the drive is
            read at the step's start (index i) and at its end (i + 1), so ``sde.f`` must accept every index up
            to ``n - 1``; and the step draws the one unit normal per variable the plain step draws, so the
            random stream is the plain step's. The drift is corrected to second order. Averaging the noise
            integrates the equation the plain step integrates only where no variable's noise depends on that
            variable itself (the Ito-Stratonovich correction, half of g dg/dx in each component, is then
            zero); a model says so by ``admits_corrected_step``, and ``Simulator.simulate`` hands this method
            only such a model. The stable region of this step contains the plain step's.
            """
            x0 = x0.to(sde.device)
            batch_size, d = x0.shape
            dt, sqrt_dt = step_tensors(dt, batch_size, x0.dtype, x0.device)
            # empty, not zeros -- same reasoning as the eager euler above.
            xs = torch.empty((n, batch_size, d), dtype=x0.dtype, device=x0.device)
            xs[0, :, :] = x0
            dW_buf = torch.empty((batch_size, d), dtype=x0.dtype, device=x0.device)

            if not state_dep_drift:
                g = sde.g()  # (batch, d), the same at the step's end
                for i in _step_iter(n, batch_size):
                    x_curr = xs[i, :, :]
                    dW_buf.normal_()
                    eta = g * dW_buf * sqrt_dt
                    f_now = sde.f(x_curr, i)
                    x_pred = x_curr + f_now * dt + eta
                    xs[i + 1, :, :] = x_curr + 0.5 * (f_now + sde.f(x_pred, i + 1)) * dt + 0.5 * (eta + eta)
            else:
                for i in _step_iter(n, batch_size):
                    x_curr = xs[i, :, :]
                    g = sde.g(x_curr)  # (batch, d)
                    dW_buf.normal_()
                    eta = g * dW_buf * sqrt_dt
                    f_now = sde.f(x_curr, i)
                    x_pred = x_curr + f_now * dt + eta
                    f_end = sde.f(x_pred, i + 1)
                    eta_end = sde.g(x_pred) * dW_buf * sqrt_dt
                    xs[i + 1, :, :] = x_curr + 0.5 * (f_now + f_end) * dt + 0.5 * (eta + eta_end)

            return xs

        def heun_compiled(sde: Any, x0: torch.Tensor, dt, n: int, state_dep_drift: bool = False) -> torch.Tensor:
            """
            :func:`heun` via the model's ``compiled_corrected_step``, replayed from a CUDA Graph where possible,
            as :func:`euler_compiled` replays the plain step. The model exposes
            ``compiled_corrected_step(x, force_now, force_next, dW, *params, dt, sqrt_dt) -> next_x``, a
            ``@torch.jit.script`` function taking the plain compiled step's ``compiled_params()``, with the
            drive at the step's end after the drive at its start. Each replay reads one drive point more than
            its steps (:func:`_acquire_graph`'s ``ends``). ``state_dep_drift`` is accepted and unused, as in
            :func:`euler_compiled`: the compiled step carries its noise law inline.
            """
            return _compiled_run(sde, x0, dt, n, sde.compiled_corrected_step, ends=True)

        self.euler = euler
        self.euler_compiled = euler_compiled
        self.heun = heun
        self.heun_compiled = heun_compiled


def _compiled_run(sde: Any, x0: torch.Tensor, dt, n: int, step, ends: bool) -> torch.Tensor:
    """The compiled loop both compiled methods share: ``n`` points, ``n - 1`` steps of ``step``, replayed from a
    CUDA Graph where possible, eager otherwise. ``ends`` is the corrected step's: each step reads the drive at
    its start and at its end (``sde.force`` indices i and i + 1), and each replay's drive buffer is one point
    longer than its steps."""
    x0 = x0.to(sde.device)
    batch_size, d = x0.shape
    dt, sqrt_dt = step_tensors(dt, batch_size, x0.dtype, x0.device)

    # empty, not zeros -- same reasoning as the eager euler.
    xs = torch.empty((n, batch_size, d), dtype=x0.dtype, device=x0.device)
    xs[0, :, :] = x0

    params = sde.compiled_params()

    def one(x, i, dW):
        if ends:
            return step(x, sde.force[:, :, i], sde.force[:, :, i + 1], dW, *params, dt, sqrt_dt)
        return step(x, sde.force[:, :, i], dW, *params, dt, sqrt_dt)

    chunk = int(getattr(config, "SOLVER_GRAPH_CHUNK", 50))
    n_steps = n - 1
    ent = (_acquire_graph(step, params, x0, sde.force, dt, sqrt_dt, chunk, ends=ends)
           if n_steps >= chunk else None)

    if ent is not None:
        # Copy THIS call's parameters and step into the captured statics. The graph replays
        # whatever is in these buffers, so this is what makes one capture serve every batch.
        for ps, p in zip(ent["params"], params):
            ps.copy_(p)
        ent["dt"].copy_(dt)
        ent["sqrt_dt"].copy_(sqrt_dt)
        ent["x"].copy_(x0)
        f_s, out_s, g = ent["force"], ent["out"], ent["graph"]
        width = chunk + 1 if ends else chunk
        n_full = n_steps // chunk
        bar = _step_bar(n, batch_size)
        try:
            for c in range(n_full):
                lo = c * chunk
                f_s.copy_(sde.force[:, :, lo:lo + width])
                g.replay()
                xs[lo + 1:lo + 1 + chunk, :, :].copy_(out_s)
                _advance(bar, chunk)
            # Tail: the last n_steps % chunk steps, eager, continuing from the graph's state.
            x = ent["x"].clone()
            dW_buf = torch.empty((batch_size, d), dtype=x0.dtype, device=x0.device)
            for i in range(n_full * chunk, n_steps):
                dW_buf.normal_()
                x = one(x, i, dW_buf)
                xs[i + 1, :, :] = x
                _advance(bar, 1)
        finally:
            bar.close()
        return xs

    dW_buf = torch.empty((batch_size, d), dtype=x0.dtype, device=x0.device)
    x = x0
    for i in _step_iter(n, batch_size):
        dW_buf.normal_()
        x = one(x, i, dW_buf)
        xs[i + 1, :, :] = x

    return xs