"""``python -m core probes <mode>``: the probe checks, from the command line only.

The family re-measures the chi probe settings -- the band, the drive strength, the lock-in ceiling --
on a lab's own cell. It MEASURES and never changes a setting: each mode writes one ordinary diagnostic
record that states the configured settings it judged, and nothing it measures with reaches a training
run's configuration. A deliberate change to a setting is made in core/config.py.

``band`` builds its configuration in chi mode itself -- the chi probes are what it measures -- so it
takes no chi flag at all: ``config_args.add_config_flags(..., chi=False)`` leaves out ``--chi``,
``--no-chi`` and ``--chi-k``, and chi mode and the configured probe count come from this parser's
defaults instead. Its grid flags have their own names and field keys, never the training drive's.

Every parser in the family is built with ``allow_abbrev=False``. argparse otherwise accepts any
unambiguous prefix of an option, so an option typed out of habit on a mode that happens to have a
longer one starting the same way would silently set that longer one.

Torch-free at import, like every parser module: the stage module is imported inside the handler, so
``--help`` costs no torch import.
"""
import argparse

from . import config_args, help_defaults
from .config_args import add_name_flags, knobs, report

#: What Ctrl-C leaves behind. A probe check writes its record only when it finishes, so an interrupted
#: run leaves nothing to clear and nothing to resume -- main's generic advice, written for a
#: checkpointed training cache, would be wrong here.
PROBES_INTERRUPT_NOTE = ("a probe check writes its record only when it finishes, so the interrupted one "
                         "was removed and there is nothing to clear; nothing resumes, so the same "
                         "command starts the measurement again.")

_BAND = "probes band"
_BAND_KNOBS = ("lengths", "multipliers", "drives", "repeats", "cycle_caps", "cv_max", "phase_max",
               "snr_min", "sup_min", "peak_window", "seed")

BAND_EPILOG = """\
At every recording length, probe frequency and drive strength, an ensemble of
the cell's own noisy runs is driven and locked in on, at full length and under
the configured lock-in ceiling, and four criteria are judged: the spread of
|chi| over the runs (--cv-max), the phase scatter (--phase-max: reported, and
judged only when given), the signal over the same lock-in on the undriven runs
(--snr-min), and capture -- how much of the cell's own peak power survives the
drive (--sup-min). Only capture is physical evidence; the other thresholds are
conventions. A probe at or above 0.9 x Nyquist is masked, never moved.

The check changes nothing: its record states the configured band, drive,
cycle floor, ceiling and probe count it judged, and a deliberate change to any
of them is made in core/config.py.
"""


def _value_flag(p, flag, *, help, **kw):
    """One value flag of ``probes band``, its help ending with the default clause the help table gives
    it -- so the table, not this module, states every default."""
    action = p.add_argument(flag, help=help, **kw)
    clause = help_defaults.default_for(_BAND, action)
    if clause:
        action.help = help + clause
    return action


def _band(args, store) -> None:
    import core.diagnostics as diag
    cfg, _ = config_args.build_cfg(args, load_gt=True)
    return report(diag.probe_band(cfg, name=args.name, note=args.note, fig_sink=config_args.close_sink,
                                  store=store, **knobs(args, *_BAND_KNOBS)))


def _register_band(modes) -> None:
    text = "do the configured chi band and drive hold for this cell?"
    band = modes.add_parser("band", help=text, description=text, epilog=BAND_EPILOG, allow_abbrev=False,
                            formatter_class=argparse.RawDescriptionHelpFormatter)
    config_args.add_config_flags(band, chi=False)
    add_name_flags(band)
    band.add_argument("--cell", required=True, metavar="PATH",
                      help="the cell file whose ground truth the probes are measured on")
    _value_flag(band, "--lengths", type=float, nargs="+", metavar="S",
                help="recording lengths, in seconds, each measured on an ensemble of its own; one "
                     "beyond the cell's training ceiling is measured and marked")
    _value_flag(band, "--multipliers", type=float, nargs="+", metavar="X",
                help="probe frequencies, as multiples of each run's own peak frequency")
    _value_flag(band, "--drives", type=float, nargs="+", metavar="X",
                help="non-dimensional drive amplitudes; a probe frequency captured at any of them fails "
                     "the band verdict, while the drive verdict judges the configured drive alone")
    _value_flag(band, "--repeats", type=int, metavar="N",
                help="noise repeats per point, the ensemble every measure is taken over")
    _value_flag(band, "--cycle-caps", type=float, nargs="+", metavar="N",
                help="lock-in ceilings, in drive cycles, re-measured to bracket where a longer lock-in "
                     "stops helping; the configured ceiling is always among them")
    _value_flag(band, "--cv-max", type=float, metavar="X",
                help="largest spread of |chi| over the runs, as a fraction of its mean, that passes")
    _value_flag(band, "--phase-max", type=float, metavar="X",
                help="largest circular spread of the phase of chi, in radians, that passes")
    _value_flag(band, "--snr-min", type=float, metavar="X",
                help="smallest mean |chi| driven, over the same lock-in on the undriven runs, that passes")
    _value_flag(band, "--sup-min", type=float, metavar="X",
                help="smallest share of the cell's own peak power that must survive the drive, in "
                     "(0, 1]; below it the probe captured the cell")
    _value_flag(band, "--peak-window", type=float, metavar="X",
                help="half-width of the own-peak window, as a fraction of the peak frequency, in (0, 1)")
    _value_flag(band, "--seed", type=int, metavar="N", help="the random seed for the whole run")
    band.set_defaults(handler=_band, chi_mode=True, chi_n_freqs=None, interrupt_note=PROBES_INTERRUPT_NOTE)


def register(sub) -> dict:
    """``{"probes": parent}``: the family's parent parser, its modes under ``variant``."""
    text = "check the chi probe settings on a cell or a prior"
    p = sub.add_parser("probes", help=text, description=text, allow_abbrev=False)
    # the metavar names the modes in a bare `probes` usage error, never the internal dest; it grows with
    # the family
    modes = p.add_subparsers(dest="variant", required=True, metavar="{band}")
    _register_band(modes)
    return {"probes": p}
