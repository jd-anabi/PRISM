"""The FEATURE SET a config's diagnostics must be built over, and the guards that refuse a config
whose science they do not cover.

Moved from ``scripts/_common.py`` (piece 2 of the 2026-09-11 one-flow design). The computation is
unchanged; only the guards changed, from ``SystemExit`` to ``ValueError`` naming ``python -m core``
subcommands and flags rather than environment variables -- these run inside the command-line tool and
the GUI now, where a SystemExit would walk past the tool's exit-code table or take the app down.

Every Jacobian / identifiability diagnostic must be built from the features the posterior ACTUALLY
conditions on, or it answers a question about a different experiment:

  spontaneous / forced : the full 41 summary features
  chi                  : 30 summary features (Group G is ZEROED in this mode, so its 11 columns
                         carry no information) + 3K chi features

Left on the 41-feature assumption, these diagnostics kept Group G and omitted chi entirely, so their
results were literally independent of the chi toggle -- they would report the kappa~x_scale /
lambda~t_scale aliases as strong as ever and falsely refute the hypothesis chi mode exists to test.
"""
from core import config
from core.config import SimConfig

_GROUP_G_PREFIX = "G"


def summary_keep_idx() -> list:
    """Indices of the summary features that survive in chi mode (everything outside Group G)."""
    from core.SBI.statistics import FEATURE_LABELS
    keep = [i for i, lbl in enumerate(FEATURE_LABELS) if not lbl.startswith(_GROUP_G_PREFIX)]
    n_dropped = len(FEATURE_LABELS) - len(keep)
    assert n_dropped == 11, f"expected 11 Group-G features, found {n_dropped}"
    return keep


def feature_labels(cfg: SimConfig) -> list:
    """Ordered labels for the Jacobian ROWS this config's diagnostics should be built over."""
    from core.SBI import chi as _chi
    from core.SBI.statistics import FEATURE_LABELS
    if not cfg.chi_mode:
        return list(FEATURE_LABELS)
    # The FISHER channel set (log|chi|, cos, sin), not the conditioning one: these diagnostics
    # build a Jacobian, and the conditioning block's `u` and `mask` columns are theta-independent
    # there -- see chi.CHI_FISHER_CHANNELS for what that does to a central difference.
    return ([FEATURE_LABELS[i] for i in summary_keep_idx()]
            + _chi.chi_labels(cfg.chi_n_freqs, _chi.CHI_FISHER_CHANNELS))


def n_features(cfg: SimConfig) -> int:
    """Row count of the mode's feature vector. Replaces the hardcoded 41 these diagnostics carried."""
    return len(feature_labels(cfg))


def describe_features(cfg: SimConfig) -> None:
    """One banner line so a result can never be read without knowing which information set made it."""
    if cfg.chi_mode:
        from core.SBI import chi as _chi
        n_sp, n_ch = len(summary_keep_idx()), len(_chi.CHI_FISHER_CHANNELS)
        print(f"[mode] CHI: feature rows = {n_sp} spontaneous + {n_ch * cfg.chi_n_freqs} chi "
              f"= {n_features(cfg)}  (Group G dropped: it is zeroed in this mode)", flush=True)
        print("[mode] NOTE f_scale is informative here (chi drives at amp = F0 * f_scale).", flush=True)
    else:
        print(f"[mode] {cfg.observation_mode.upper()}: feature rows = {n_features(cfg)} "
              f"(the full single-frequency feature set)", flush=True)


def assert_not_chi(cfg: SimConfig, what: str) -> None:
    """Refuse to run a diagnostic that has not been generalised past the single-frequency layout.

    Better a loud refusal than a plausible-looking number computed over the wrong feature set --
    which is precisely how these diagnostics failed before.

    :raises ValueError: in chi mode.
    """
    if cfg.chi_mode:
        from core.SBI import chi as _chi
        raise ValueError(
            f"{what} has not been generalised to chi(omega) mode: it measures the single-frequency "
            f"41-feature information set, while a chi posterior conditions on "
            f"{n_features(cfg)} different features (Group G zeroed, "
            f"{len(_chi.CHI_FISHER_CHANNELS) * cfg.chi_n_freqs} chi features added). Running it here "
            f"would produce a confident, meaningless answer.\n"
            f"  -> pass --no-chi to analyse the forced information set, or run "
            f"`python -m core identifiability jacobian`, which is chi-aware.")


def assert_nadrowski(cfg: SimConfig, why: str = "") -> None:
    """Guard for diagnostics whose SCIENCE is Nadrowski-specific (hardcoded parameter roles, fixed
    column indices). Better a loud refusal than a plausible-looking wrong answer for another model.

    :raises ValueError: for any other model.
    """
    if cfg.model != "NADROWSKI":
        raise ValueError(
            f"This diagnostic is Nadrowski-specific{(' (' + why + ')') if why else ''}, but the "
            f"config is for {cfg.model}. It would run and produce meaningless numbers.\n"
            f"  -> point --bounds at a file under Bounds/nadrowski/ (the bounds file's parent folder "
            f"is what names the model).")


def assert_forced(cfg: SimConfig, what: str) -> None:
    """Guard for diagnostics that read the cell's own drive (``cfg.forcing_idx["amp"]`` and friends).

    Whether a drive EXISTS is a property of the bounds file, not the cell values, so pointing one of
    these at a spontaneous cell used to surface as a bare ``KeyError: 'amp'`` twenty lines below the
    config banner. Say which file to point at instead.

    :raises ValueError: when the bounds file declares no Forcing section.
    """
    if not cfg.has_forcing:
        raise ValueError(
            f"{what} measures the response to the cell's OWN drive, but this config is "
            f"{cfg.observation_mode.upper()}: its bounds file declares no Forcing section, so there "
            f"is no amp/freq/phase to read.\n"
            f"  -> point it at a forced cell, e.g. "
            f"--cell {config.CELL_PATH / 'nadrowski' / 'master_weak.txt'} , whose sibling bounds file "
            f"is the forced box --bounds then resolves to.")
