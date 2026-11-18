import torch
from tqdm import tqdm
from abc import ABC, abstractmethod

from core import config
from core.Helpers import helpers
from core.Solvers import sdeint


class SimulationError(RuntimeError):
    """A solver failure inside ``Simulator.simulate``, carrying the geometry that produced it.

    Always raised ``from`` the underlying error, so the original traceback survives as ``__cause__``
    and Python prints both. Callers that need to branch on the cause can test it directly, e.g.
    ``isinstance(err.__cause__, torch.OutOfMemoryError)``.

    This used to be ``print(...); exit()``, which hard-killed the interpreter: a CUDA OOM arrived with
    no traceback (so the failing call site was unknowable), and any caller -- the test runners, the
    GUI's QThreadPool worker, the scripts -- died mid-run with nothing reported. Note ``exit()`` is the
    ``site`` builtin and is not guaranteed to exist at all under ``python -O`` or an embedded
    interpreter.
    """


def segment_bounds(n_points: int, segs: int) -> list:
    """The state indices at which ``segs`` even segments of an ``n_points`` grid meet: ``[0, ..., n_points - 1]``,
    segment k taking the steps from index ``bounds[k]`` to ``bounds[k + 1]``. The split is the one the solver
    has always used (``helpers.get_even_ids``), its last bound the grid's last index."""
    ids = helpers.get_even_ids(n_points, max(1, int(segs)) + 1)
    bounds = [0]
    for b in ids[1:-1] + [n_points - 1]:
        if b > bounds[-1]:
            bounds.append(b)
    return bounds


def _check_plan(plan, n_points: int) -> None:
    """A plan's segments are contiguous, each takes at least one step, and they run from index 0 to the grid's
    last: anything else would leave states unwritten or written twice."""
    at = 0
    for a, b, _dt in plan:
        if a != at or b <= a:
            raise ValueError(f"a simulation plan's segments must be contiguous steps from index 0: got {plan!r}")
        at = b
    if at != n_points - 1:
        raise ValueError(f"a simulation plan ends at index {at}, but the grid's last index is {n_points - 1}")


class Simulator(ABC):
    def __init__(self, params: torch.Tensor, force: torch.Tensor, inits: torch.Tensor, t: torch.Tensor,
                 freqs_per_batch: int = 1, segs: int = 1, batch_size: int = 1, device: torch.device = torch.device('cpu'),
                 use_compile: bool | None = None):
        # device initialization
        self._device = device
        self._dtype = inits.dtype
        self._batch_size = batch_size

        # rest of the constructor
        self._params = params
        self._force = force
        self.inits = inits
        self.t = t
        self.freqs_per_batch = freqs_per_batch
        self.segs = segs

        # Auto-enable the compiled path on CUDA when the model exposes `compiled_step`: sdeint's
        # euler_compiled, the model's TorchScript step replayed from a CUDA graph where capture
        # works. Explicitly pass False to force the eager loop.
        self._use_compile = use_compile

        # check if we are using the steady-state solution (all zeros for the 4th parameter)
        self._set_up_model()

    # --- PUBLIC METHODS --- #
    def simulate(self, state_dep_drift: bool = False, plan=None) -> torch.Tensor:
        """
        Simulates the model with the given constructor parameters.

        :param plan: the segments to integrate, ``[(a, b, dt), ...]``: segment k takes ``b - a`` steps of
            ``dt`` from the state at index ``a`` to index ``b``, the segments contiguous from index 0 to the
            grid's last; ``dt`` a float or one value per row (``sdeint.step_tensors``). A caller that puts
            a run on a recording's frames hands one: the transient at its step, the recorded part at another.
            None (the prior's screens) splits the grid into ``segs`` segments and reads the step ONCE, from
            the grid's first interval exactly as the first segment always has (:func:`legacy_plan`).
        :return: simulated solution with shape (N, FPB, B / FPB, T)

        THE SEAM. Segment k starts from the state at the previous segment's last index and is handed the
        drive from that index; it integrates its own steps and writes only its new states. So index g
        holds the state after g steps in every segment, drive index m serves step m -> m+1, and a run
        cut into segments equals one long run, bit for bit on the CPU with the same draws in the same
        order. (Each seam used to repeat a sample, the drive's index then leading the state's by one.)
        """
        ensemble_size = self._batch_size // self.freqs_per_batch
        n_t = self.t.shape[0]
        if plan is None:
            plan = self.legacy_plan()
        _check_plan(plan, n_t)

        n_vars = self.inits.shape[-1]
        curr_inits = self.inits
        # empty, not zeros -- the same reasoning sdeint.euler already applies to its own buffer, at
        # the LARGER of the two: this is (n_vars, batch, n_fine), ~6.95 GiB at the production
        # geometry against the solver buffer's ~2.3 GiB. Index 0 is written below and every segment
        # writes its new states, so every element is overwritten and the memset is pure waste -- x
        # segs x runs x 5000 training batches.
        sol = torch.empty((n_vars, self._batch_size, n_t), dtype=self.t.dtype, device=self.t.device)
        sol[:, :, 0] = torch.transpose(self.inits, 0, 1).to(dtype=sol.dtype, device=sol.device)

        # The SDE model indexes force with the solver's local step index (0..n_steps-1), not the
        # absolute step across the full simulation, so each segment is handed the drive from its own
        # first index; the full reference is restored after.
        full_force = self._force
        # `disable` is passed only under the GUI (config.QUIET_SEGMENT_BAR). It must be a conditional
        # splat, not `disable=config.QUIET_SEGMENT_BAR`: tqdm.__init__ is @envwrap("TQDM_")-decorated
        # (tqdm/std.py:951) and a call kwarg outranks the environment, so an explicit False would shadow
        # a TQDM_DISABLE override that reaches this call today. Omitting it keeps the CLI byte-identical.
        for a, b, dt in tqdm(plan, desc="Running time segments", leave=False,
                             **({"disable": True} if config.QUIET_SEGMENT_BAR else {})):
            self.sde.force = full_force[:, :, a:b]
            results = self.__sols(b - a + 1, curr_inits, dt, state_dep_drift)  # (b - a + 1, batch, n_vars)

            # update initial conditions. The .clone() is load-bearing for MEMORY, not correctness:
            # results[-1, :, :] is a VIEW into this segment's (n, batch, n_vars) solver buffer, so
            # carrying it into the next iteration would pin that whole buffer -- 2.5 GB at
            # CHUNK_LEN=100k x batch=2048 x 3 vars -- alive for the whole of the next segment's
            # integration. Copying one (batch, n_vars) row costs ~24 KB and lets the buffer go.
            curr_inits = results[-1, :, :].clone()
            # the segment's NEW states only: results[0] is the state at index a, already written
            sol[:, :, a + 1:b + 1] = torch.transpose(results[1:], 0, 2)  # (n_vars, batch, b - a)
            del results          # release the segment buffer BEFORE the next __sols allocates its own
        self.sde.force = full_force
        sol = sol.reshape(n_vars, self.freqs_per_batch, ensemble_size, n_t)  # shape: (number of variables, frequencies per batch, ensemble size, length of time series)
        return sol

    def legacy_plan(self) -> list:
        """The plan of a call that hands none: ``segs`` even segments over the grid at ONE step, read from the
        grid's first interval as the first segment's float32 linspace over its end points gives it -- the
        step the prior's stability screen has always integrated at, so its one-segment screen is bitwise
        as before. Every later segment keeps that step."""
        bounds = segment_bounds(self.t.shape[0], self.segs)
        first = self.t[bounds[0]:bounds[1] + (1 if len(bounds) == 2 else 0)]
        grid = torch.linspace(first[0], first[-1], first.shape[0], device=self._device)
        dt = grid[1].item() - grid[0].item()
        return [(bounds[k], bounds[k + 1], dt) for k in range(len(bounds) - 1)]

    # --- GETTERS AND SETTERS --- #
    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device: torch.device):
        self._device = device
        self._set_up_model()

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, dtype: torch.dtype):
        self._dtype = dtype
        self._set_up_model()

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, batch_size: int):
        self._batch_size = batch_size
        self._set_up_model()

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, params: torch.Tensor):
        self._params = params
        self._set_up_model()

    @property
    def force(self):
        return self._force

    @force.setter
    def force(self, force: torch.Tensor):
        self._force = force
        self._set_up_model()

    # --- PRIVATE METHODS --- #
    def __sols(self, n: int, inits: torch.Tensor, dt, state_dep_drift: bool) -> torch.Tensor:
        """
        Returns the sde solution over one segment: ``n`` points, ``n - 1`` steps of ``dt``, from ``inits``.
        :return: a (n, batch, n_vars) tensor
        """

        # Solving a system of SDEs. Constructed per call ON PURPOSE, despite this running once per
        # time segment: Solver builds its two methods (euler and euler_compiled) as closures in
        # __init__, so resolving sdeint.Solver at CALL time is the seam that lets a caller swap the
        # solver out (see
        # tests/test_user_sbi.py::test_solver_failure_raises_instead_of_killing_the_process, which
        # patches the class to make every method raise). Hoisting this to a module-level singleton
        # saves two closure allocations per ~10-second segment -- far too little to be worth losing
        # the seam over.
        solver = sdeint.Solver()

        # Pick eager vs compiled. Auto: CUDA + model exposes compiled_step.
        use_compile = self._use_compile
        if use_compile is None:
            use_compile = (self._device.type == "cuda"
                           and hasattr(self.sde, "compiled_step"))

        with torch.no_grad():
            try:
                if use_compile:
                    sol = solver.euler_compiled(self.sde, inits, dt, n, state_dep_drift=state_dep_drift)
                else:
                    sol = solver.euler(self.sde, inits, dt, n, state_dep_drift=state_dep_drift)
            # NOT BaseException: streams.WorkerCancelled derives from BaseException precisely so a
            # cooperative cancel sails through handlers like this one to reach Worker.run. Widening
            # here would turn every GUI cancel into a spurious simulation failure.
            except Exception as e:
                method = "euler_compiled" if use_compile else "euler"
                raise SimulationError(
                    f"{type(self.sde).__name__} {method} failed after {n} steps "
                    f"(batch={self._batch_size}, segs={self.segs}, device={self._device}, "
                    f"dtype={self._dtype}): {type(e).__name__}: {e}"
                ) from e
        return sol

    @abstractmethod
    def _set_up_model(self):
        self.sde = None