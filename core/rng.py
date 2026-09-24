"""The one seeding context ``smoke``, every diagnostic and the FDT measurement run inside.

Here, at the top of ``core``, and not in ``core/diagnostics`` where it started (piece 2): importing ANY
submodule of a package runs the package's ``__init__``, and ``core/diagnostics/__init__.py`` imports
the five diagnostics and through them ``core.orchestrator``, sbi's inference modules and
``pytensor``. An FDT run
needs none of that, and paid about two seconds and two false pytensor "g++" lines on stderr for it at
the head of every ``python -m core fdt`` (Task 17, fix round 1). This module imports only
``contextlib``; torch and numpy are imported when the context is entered. ``core.diagnostics.rng``
re-exports the same object, so every diagnostics import keeps working.

Seed ONCE and let the streams run on, exactly as ``scripts/smoke_train.py`` did. Training and
calibration both draw their initial conditions from numpy's global RNG and their Sobol (t_scale, T)
scramble from torch's, and on a TSNPE round -- or with the rotation off -- nothing consumes either
stream between a seed and the pipeline. Seeding per stage would therefore start draws that must be
independent from identical states: the calibration set would replay the training strata, and trap X5
makes those operating points t_scale's effective SBC sample size. That is why there is a context
here and no ``seed`` argument on any stage.

It restores what it borrowed, so an in-process ``main(argv)`` never changes the calling process's
streams -- the tool's tests run in-process, and a leaked seed makes one test's numbers depend on
which tests ran before it.

Seeded runs are NOT bitwise-reproducible on CUDA or across devices: a kernel's reduction order is not
fixed. The seed buys a repeatable EXPERIMENT on one device, not a repeatable number.
"""
import contextlib


@contextlib.contextmanager
def seeded(seed: int, device):
    """Run the block with torch and numpy seeded from ``seed``, restoring both afterwards.

    :param seed: the seed. numpy takes ``seed % 2**32``, which is its whole accepted range.
    :param device: the ``torch.device`` the block draws on. A CUDA device is forked as well as the
                   CPU, and ``torch.manual_seed`` seeds the CPU and every CUDA device, so the fork is
                   what makes the restore complete. A CPU device forks no CUDA generator --
                   ``fork_rng(devices=[cpu])`` raises -- so it seeds the CPU generator ALONE: under
                   ``torch.manual_seed`` a CPU run (every FDT run is one) left each card's generator
                   pinned at its seed after the block, and two runs at one seed made the next CUDA
                   draws identical -- the hazard core/SBI/decorrelate.py documents (spec §3.7; Task 17,
                   fix round 1).
    """
    import numpy as np
    import torch
    devices = [device] if getattr(device, "type", None) == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        state = np.random.get_state()
        try:
            if devices:
                torch.manual_seed(int(seed))
            else:
                torch.default_generator.manual_seed(int(seed))
            np.random.seed(int(seed) % 2**32)
            yield
        finally:
            np.random.set_state(state)
