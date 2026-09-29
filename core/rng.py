"""The one seeding context ``smoke``, every diagnostic, the FDT measurement and a seeded calibration
run inside.

Here, at the top of ``core``, and not in ``core/diagnostics`` where it started: importing ANY
submodule of a package runs the package's ``__init__``, and ``core/diagnostics/__init__.py`` imports
the five diagnostics and through them ``core.orchestrator``, sbi's inference modules and
``pytensor``. An FDT run needs none of that, and paid about two seconds and two false pytensor "g++"
lines on stderr for it at the head of every ``python -m core fdt``. This module imports only
``contextlib`` and the stdlib-only ``core.refusals``; torch and numpy are imported when the context
is entered, and numpy when ``calibration_seed`` is called. ``core.diagnostics.rng`` re-exports the
same object, so every diagnostics import keeps working.

Seed ONCE and let the streams run on, exactly as ``smoke`` does. Training and calibration both draw
their initial conditions from numpy's global RNG and their Sobol (t_scale, T) scramble from torch's,
and on a TSNPE round -- or with the rotation off -- nothing consumes either stream between a seed and
the pipeline. Seeding per stage would therefore start draws that must be independent from identical
states: the calibration set would replay the training strata, and the calibration's operating-point
count is t_scale's effective SBC sample size. That is why there is a context here and no ``seed``
argument on any stage but one.

``validate`` with a seed is the one stage that seeds itself. It draws from a stream derived from the
seed and a fixed calibration tag (``calibration_seed``), so it never starts where a training run
seeded with the same number started, and its calibration set cannot replay that run's strata. With
no seed it takes none, and runs on in the caller's streams, so one seed given to a whole chain of
stages still covers its calibration.

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

#: The fixed tag a calibration's stream is derived with: the ASCII bytes of "CAL".
CALIBRATION_TAG = 0x43414C


def calibration_seed(seed: int) -> int:
    """The seed a calibration seeded with ``seed`` actually runs at: one 32-bit word of numpy's
    ``SeedSequence([seed, CALIBRATION_TAG])``, so it is never ``seed`` itself and a calibration never
    starts on the stream a training run seeded with ``seed`` started on. The same ``seed`` always
    gives the same number, from 0 to 2**32 - 1, inside ``SEED_MAX``. numpy is imported here, not at
    the top, so importing this module stays free of numpy and torch."""
    import numpy as np
    return int(np.random.SeedSequence([int(seed), CALIBRATION_TAG]).generate_state(1)[0])


def require_seed(seed, *, key: str = "seed") -> int:
    """The one rule for a seed a run is handed: a whole number from 0 to ``SEED_MAX``, refused under
    the field key ``key`` before anything is spent. Both FDT builders, the sweep study, each sweep, a
    seeded calibration and the probe checks apply it, so none of them can disagree; ``--seed`` is
    ``type=int`` and takes any integer. The key is ``seed`` for every caller but the probe checks,
    whose seed has its own key because its default is 0 rather than one drawn and recorded -- so the
    refusal states the default of the seed actually refused.

    The floor is ``require_at_least``'s, sentence and all: 0 is ``SeedSequence``'s floor, and the
    sweep derives every per-point stream through one. The ceiling is its own sentence through
    ``refuse`` rather than ``require_between``, the one rule with an upper end, because that rule
    compares and prints in float: it renders both ends of this range as "1.84467e+19" ("must be
    between 0 and 1.84467e+19; got 1.84467e+19"), and ``float(2**64 - 1)`` rounds up to 2**64, so it
    would refuse ``SEED_MAX`` itself.
    """
    seed = require_at_least(key, seed, 0)
    if seed > SEED_MAX:
        what = describe(key)
        refuse(key, f"{what[0].upper()}{what[1:]} must be at most {SEED_MAX}, the largest the "
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
                   draws identical -- the hazard core/SBI/decorrelate.py documents.
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
