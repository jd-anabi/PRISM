"""A kept training after a reopening: what the window reads to offer it, from the store's kept trainings and the run
history, never while a run is live.
"""
from pathlib import Path

from ... import continuing
from ...run import history


def read_offers(store, folder) -> tuple:
    """What a reopened window can offer, read now: the kept trainings of ``store`` (none read when the store lists no
    kept training), the run history in ``folder``, each simulation set's own count, read from the set, and each kept
    training's prior's label. The one reading of these; its caller never asks while a run is live. What cannot be read
    offers nothing, and nothing raises."""
    from core.SBI import fit_state, training_checkpoint
    from ... import record_words
    try:
        kept = fit_state.kept_fits(store) if store.list("fit") else []
    except Exception:                        # noqa: BLE001 -- a store that cannot be read keeps nothing to offer
        kept = []
    try:
        runs = history.runs_of(Path(folder), records_root=str(store.root))
    except OSError:
        runs = []

    def set_batches(name: str):
        try:
            state = training_checkpoint.peek(store.kind_dir("simulation") / name)
        except Exception:                    # noqa: BLE001 -- a set that cannot be read holds nothing to continue
            return None
        done = (state or {}).get("batches_done")
        return done if isinstance(done, int) and not isinstance(done, bool) else None

    labels: dict = {}

    def prior_label(ident: str):
        if ident not in labels:
            try:
                manifest = store.get("prior", ident)
                labels[ident] = manifest.name or record_words.record_label("prior", "", None)
            except Exception:                # noqa: BLE001 -- a prior no longer in Records is said so
                labels[ident] = None
        return labels[ident]

    return continuing.offers(runs, kept, records_root=str(store.root), set_batches=set_batches,
                             prior_label=prior_label)
