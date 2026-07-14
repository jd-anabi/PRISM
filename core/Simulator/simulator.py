import torch
from tqdm import tqdm
from abc import ABC, abstractmethod

from core import config
from core.Helpers import helpers
from core.Solvers import sdeint

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

        # Auto-enable the torch.compile path on CUDA when the model exposes
        # `compiled_step`. Explicitly pass False to force the eager loop.
        self._use_compile = use_compile

        # check if we are using the steady-state solution (all zeros for the 4th parameter)
        self._set_up_model()

    # --- PUBLIC METHODS --- #
    def simulate(self, state_dep_drift: bool = False) -> torch.Tensor:
        """
        Simulates the model with the given constructor parameters
        :return: simulated solution with shape (N, FPB, B / FPB, T)
        """
        ensemble_size = self._batch_size // self.freqs_per_batch
        time_seg_ids = helpers.get_even_ids(self.t.shape[0], self.segs + 1)

        n_vars = self.inits.shape[-1]
        curr_inits = self.inits
        sol = torch.zeros((n_vars, self._batch_size, self.t.shape[0]), dtype=self.t.dtype, device=self.t.device)

        # The SDE model indexes force with the solver's local step index (0..n_seg-1),
        # not the absolute step across the full simulation. When segs > 1 with
        # non-constant forcing, we slice force to the current segment so the local
        # index lookup picks up the right values; restore the full reference after.
        full_force = self._force
        # `disable` is passed only under the GUI (config.QUIET_SEGMENT_BAR). It must be a conditional
        # splat, not `disable=config.QUIET_SEGMENT_BAR`: tqdm.__init__ is @envwrap("TQDM_")-decorated
        # (tqdm/std.py:951) and a call kwarg outranks the environment, so an explicit False would shadow
        # a TQDM_DISABLE override that reaches this call today. Omitting it keeps the CLI byte-identical.
        for tid in tqdm(range(len(time_seg_ids) - 1), desc="Running time segments", leave=False,
                        **({"disable": True} if config.QUIET_SEGMENT_BAR else {})):
            curr_time = self.t[time_seg_ids[tid]:time_seg_ids[tid + 1]]
            self.sde.force = full_force[:, :, time_seg_ids[tid]:time_seg_ids[tid + 1]]
            results = self.__sols(curr_time, curr_inits, state_dep_drift)  # shape: (len(curr_time), BATCH_SIZE, number of variables)

            # update initial conditions
            curr_inits = results[-1, :, :]

            # extract position data
            sol[:, :, time_seg_ids[tid]:time_seg_ids[tid + 1]] = torch.transpose(results, 0, 2)  # shape: (number of variables, BATCH_SIZE, len(curr_time))
        self.sde.force = full_force
        sol = sol.reshape(n_vars, self.freqs_per_batch, ensemble_size, self.t.shape[0])  # shape: (number of variables, frequencies per batch, ensemble size, length of time series)
        return sol

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
    def __sols(self, t: torch.Tensor, inits: torch.Tensor, state_dep_drift: bool, explicit: bool = True) -> torch.Tensor:
        """
        Returns sde solution for a hair bundle given a set of parameters and initial conditions
        :param t: time array
        :param explicit: whether to use the explicit Euler-Maruyama method
        :return: a 2D array of length len(t) x num_vars; num_vars is 5 if pt_steady_state is False and 4 otherwise
        """
        # time array
        n = t.shape[0]
        ts = (t[0], t[-1])

        # solving a system of SDEs
        solver = sdeint.Solver()

        # Pick eager vs compiled. Auto: CUDA + model exposes compiled_step.
        use_compile = self._use_compile
        if use_compile is None:
            use_compile = (self._device.type == "cuda"
                           and hasattr(self.sde, "compiled_step"))

        with torch.no_grad():
            try:
                if not explicit:
                    sol = solver.implicit_euler(self.sde, inits, ts, n)
                elif use_compile:
                    sol = solver.euler_compiled(self.sde, inits, ts, n, state_dep_drift=state_dep_drift)
                else:
                    sol = solver.euler(self.sde, inits, ts, n, state_dep_drift=state_dep_drift)
            except (Warning, Exception) as e:
                print(f'Warning or Exception occurred: {e}')
                exit()
        return sol

    @abstractmethod
    def _set_up_model(self):
        self.sde = None