"""The folder that holds the real settings file, read as a listing, and the check a test run makes of it as it
ends.

tests/conftest.py's ``_settings_home`` lists that folder as the run begins and hands the listing to
``check_kept`` as the run ends: whatever the window writes beside its settings file goes beside the test's own
during a test, and a file it left beside the real one would be read by the next launch. Two plain functions, so
that a test can try them on a folder of its own; this file imports the standard library and nothing else.
"""
from pathlib import Path


def listing(folder: Path) -> list[str] | None:
    """Every path under ``folder``, at any depth, sorted as text; None when there is no such folder."""
    return sorted(str(path) for path in folder.rglob("*")) if folder.exists() else None


def check_kept(folder: Path, held: list[str] | None) -> None:
    """Fail, by raising, when ``folder`` does not hold exactly what ``held``, its listing at the start, says,
    naming the first five paths that came or went."""
    holds = listing(folder)
    if holds != held:
        raise AssertionError(f"the suite changed what the real settings folder {folder} holds: "
                             f"{sorted(set(holds or ()) ^ set(held or ()))[:5]}")
