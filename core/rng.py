"""The one seeding context ``smoke``, every diagnostic and the FDT measurement run inside.

Here, at the top of ``core``, and not in ``core/diagnostics`` where it started (piece 2): importing ANY
submodule of a package runs the package's ``__init__``, and ``core/diagnostics/__init__.py`` imports
the five diagnostics and through them ``core.orchestrator``, sbi's inference modules and
``pytensor``. An FDT run
needs none of that, and paid about two seconds and two false pytensor "g++" lines on stderr for it at
the head of every ``python -m core fdt`` (Task 17, fix round 1). This module imports only
``contextlib`` and the stdlib-only ``core.refusals``; torch and numpy are imported when the context is
entered. ``core.diagnostics.rng`` re-exports the same object, so every diagnostics import keeps
working.

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

from core.refusals import describe, refuse, require_at_least

#: The largest seed ``seeded`` can take: ``torch.manual_seed``'s documented ceiling,
#: 0xffff_ffff_ffff_ffff. One above it raises "Overflow when unpacking long long", a bare ValueError,
#: from inside the block -- that is, after an FDT run has opened its record.
SEED_MAX = 2 ** 64 - 1


def require_seed(seed) -> int:
    """The one rule for a seed an FDT run is handed: a whole number from 0 to ``SEED_MAX``, refused
    under the field key ``seed`` before anything is spent. Both FDT builders, the sweep study and
    each sweep apply it, so the four cannot disagree; ``--seed`` is ``type=int`` and takes any integer.

    The floor is ``require_at_least``'s, sentence and all: 0 is ``SeedSequence``'s floor, and the
    sweep derives every per-point stream through one. The ceiling is its own sentence through
    ``refuse`` rather than ``require_between``, the one rule with an upper end, because that rule
    compares and prints in float: it renders both ends of this range as "1.84467e+19" ("must be
    between 0 and 1.84467e+19; got 1.84467e+19"), and ``float(2**64 - 1)`` rounds up to 2**64, so it
    would refuse ``SEED_MAX`` itself (Task 28's ruling asked for a readable sentence).
    """
    seed = require_at_least("seed", seed, 0)
    if seed > SEED_MAX:
        what = describe("seed")
        refuse("seed", f"{what[0].upper()}{what[1:]} must be at most {SEED_MAX}, the largest the "
                       f"random number generator accepts; got {seed}.")
    return seed


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
