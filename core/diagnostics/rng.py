"""The one seeding context ``smoke`` and every diagnostic runs inside.

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
    :param device: the ``torch.device`` whose CUDA generator to fork as well. A CPU device forks
                   none -- ``fork_rng(devices=[cpu])`` raises -- and ``torch.manual_seed`` seeds every
                   CUDA device anyway, so the fork is what makes the restore complete.
    """
    import numpy as np
    import torch
    devices = [device] if getattr(device, "type", None) == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        state = np.random.get_state()
        try:
            torch.manual_seed(int(seed))
            np.random.seed(int(seed) % 2**32)
            yield
        finally:
            np.random.set_state(state)
