"""The artifact store: directories under one root, a manifest in each, and (from Task 4 on) the
loaders that refuse a mismatch before anything is spent.

Every generated artifact is ``<root>/<kind dir>/<name>__<id>/`` -- ``_unnamed__<id>`` until it is
named -- with ``manifest.json`` written LAST, atomically, by ``ArtifactWriter``. The store indexes by
READING manifests, so a hand-renamed folder still resolves and parent references (by id) survive.
The simulation cache is the exception: its directory is the identity digest and its manifest is
written by ``training_checkpoint`` (``write_simulation_manifest``), because a resumable cache must
survive an interrupted run, which a writer's remove-on-exception would delete. An fdt record needs
the same survival but keeps the writer, through its progressive mode (``PROGRESSIVE_KINDS``): its
manifest is written FIRST and refreshed as the run proceeds.
"""
from __future__ import annotations

import contextlib
import math
import os
import re
import shutil
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from core import config, runs
from core.Helpers import file_manager
from core.refusals import Refusal

from . import manifest as mf
from . import provenance as prov

KIND_DIRS = {"prior": "priors", "simulation": "simulations", "posterior": "posteriors",
             "observation": "observations", "calibration": "calibrations", "inference": "inferences",
             "diagnostic": "diagnostics",
             # NOT "crossval": core/Reduction/plots.py already writes reduction_crossval_<stamp>.png
             # into the reduction directory, and a second meaning for the word would collide with
             # existing vocabulary (spec §2.1). Reusing the EXISTING ``fdt`` directory is chosen, not
             # accidental: the owner's stray pictures then sit as loose files inside a kind
             # directory, which is where ``loose_files`` can reach them.
             "fdt": "fdt"}
MANIFEST = "manifest.json"
# The run's records, written beside the manifest by ArtifactWriter (piece 3, V4). NOT a payload: not
# hashed, not in ``payloads``, never read by a loader. The simulation cache never gets one (§1.2).
LOG_FILE = "log.txt"
# The ONE reason ``_entries`` gives a directory that carries no manifest.json AT ALL, and therefore
# the only state a sweep may remove (piece 4 whole-piece review, R1). A directory whose manifest
# EXISTS but this build will not parse, or that declares another kind, is something nobody here
# understands -- it is reported in the listing with its own reason and left alone. Both front ends
# read this constant to decide what to offer, so neither restates the rule.
NO_MANIFEST_REASON = "no manifest.json (incomplete or interrupted)"
# How recently a directory's TREE must have been touched for a removal to refuse it as possibly
# live (R3). ``ArtifactWriter`` creates the directory first and writes the manifest LAST, so for the
# whole of a run a live directory looks exactly like a leftover -- and a sweep from a SECOND process
# (`python -m core artifacts sweep` beside a training in the window) has no ``BasePanel._running`` to
# consult. Not a cross-process lock, which is out of scope: a recency guard plus an honest report
# cannot itself destroy anything, and it costs a just-abandoned leftover a few minutes' wait.
RECENT_WRITE_SECONDS = 300.0
# Kinds whose record is written PROGRESSIVELY (piece 5, E2): the directory and a first manifest
# exist from ``__enter__``, ``refresh()`` rewrites the manifest and the log as the run proceeds, and
# an exception KEEPS the directory instead of removing it. The training cache has the same property
# by another mechanism -- it has no writer at all -- and for the same reason: a run that takes hours
# and is interrupted must leave behind what it measured.
#
# OPT-IN PER KIND, never a default: the six ordinary kinds keep today's behaviour exactly (directory
# created at entry, manifest last, directory removed on any exception) and a test pins that they do.
#
# One consequence, stated where the constant is: a progressive record carries a manifest from its
# first moment, so ``remove_incomplete`` -- which removes only a directory with NO manifest at all --
# can never remove one, finished or not. That is the right answer (spec §2.5), and it is why
# ``__exit__`` removes the directory for a refusal raised before anything was written: an empty
# record nothing can ever clear would otherwise accumulate.
PROGRESSIVE_KINDS: frozenset = frozenset({"fdt"})
# Directories that may sit BESIDE the kind directories because an older build wrote them. A CLOSED
# literal, never a scan: the store never walks its own root, and offering to remove whatever happens
# to be under it is how a tidy-up destroys something nobody meant it to. ``legacy_dirs`` and
# ``remove_legacy`` are the only way either front end can see or clear one (piece 5, E10), and
# nothing is removed on the owner's behalf. A name here may never also be a KIND_DIRS value.
LEGACY_DIRS: tuple = ("crossval",)
# Which ``parents`` keys can name an artifact of a given kind.
_PARENT_KEYS = {"prior": ("prior",), "simulation": ("simulation",),
                "posterior": ("posterior", "parent_posterior"), "observation": ("observation",),
                "calibration": (), "inference": (), "diagnostic": (),
                # Explicit, though a MISSING entry means the same thing to ``dependents``: an fdt
                # record measures a CELL and depends on no artifact, and nothing can depend on one.
                # A comparison names the records it drew in its BODY, not here -- ``parents`` is a
                # flat {key: id} map read as ``m.parents.get(pk) == id_`` (spec §1.2), so it cannot
                # carry an arbitrary number of ids without widening the contract for every kind.
                "fdt": ()}


class StoreError(Refusal):
    """The store refused: a missing, incomplete, duplicate or depended-upon artifact."""


@dataclass(frozen=True)
class Accept:
    """The ONLY escape hatches on the load path. An INFERENCE writes the flags used into its own
    manifest's ``results.accepted``, so a number produced under one is marked; a calibration or a
    TSNPE round built with one does not carry that mark itself -- there, the flag only ever unlocks
    the load, and what is on record is the non-amortized posterior's own manifest, which already says
    ``amortized: false``."""
    truncated: bool = False            # load a NON-AMORTIZED (TSNPE) posterior
    other_observation: bool = False    # infer with it on an observation other than its region's

    def used(self) -> list:
        return [n for n in ("truncated", "other_observation") if getattr(self, n)]


@dataclass(frozen=True)
class LooseFile:
    """A file sitting DIRECTLY inside a kind directory, where only artifact directories belong.

    An older build wrote its output as bare files under ``Artifacts/fdt`` (spec §1), and no listing
    can see one: ``_entries`` iterates directories only. Frozen, because it is a report about the
    disk and nothing downstream may edit it into an instruction.
    """
    name: str            # the file's own name, never a path
    size: int            # bytes
    mtime: float         # POSIX seconds


@dataclass
class Summary:
    kind: str
    id: str
    name: str
    created: str
    note: str
    path: Path
    # "has a valid manifest", i.e. the directory describes a real artifact -- NOT "the run finished".
    # For the kinds whose body carries ``complete`` those differ: a cache is manifested from its first
    # batch on, an fdt record from its first moment (piece 5, E2), and ``body["complete"]`` is the
    # field that says whether the run got to the end. ``finished`` below is that honest question,
    # asked the same way for every kind -- ask it, not this one, when what you mean is "did the run
    # get to the end" (piece 4, B3).
    complete: bool
    reason: "str | None"
    mode: "str | None" = None
    width: "int | None" = None
    amortized: "bool | None" = None
    parents: dict = field(default_factory=dict)
    # Piece 4 (B2): everything else a browser row shows, off the ONE manifest read ``list`` already
    # did. Keyword with defaults, so no positional construction anywhere breaks.
    dir_name: str = ""                      # the directory's own name, ALWAYS; remove_incomplete's handle
    finished: bool = False                  # did the RUN finish; not ``complete`` for a cache or fdt record
    batches_done: "int | None" = None       # simulation only
    batches_planned: "int | None" = None    # simulation only: body["identity"]["n_runs"], the PLANNED total
    rows: "tuple[int, ...] | None" = None   # simulation only: rows per batch, written at completion ONLY
    variant: "str | None" = None            # diagnostic only: body["variant"]
    # Piece 5: fdt only. ``study`` is "single", "sweep" or "comparison"; the three counts are a
    # SWEEP's operating points (body["points"]), None for a single run and for a comparison, which
    # have none. Declared here and FILLED by ArtifactStore.list.
    study: "str | None" = None
    points_done: "int | None" = None
    points_planned: "int | None" = None
    points_failed: "int | None" = None

    @property
    def label(self) -> str:
        return self.name or f"(unnamed {self.id})"


@dataclass
class Loaded:
    kind: str
    id: str
    name: str
    manifest: mf.Manifest
    path: Path


@dataclass
class LoadedPrior(Loaded):
    prior: object = None            # the physical ProductPrior (ND x rescale)
    force_prior: object = None      # None for a no-forcing model
    nd_prior: object = None         # the TransformedDistribution over the latent GMM
    fingerprint: "str | None" = None


@dataclass
class LoadedPosterior(Loaded):
    posterior: object = None        # reparam.TransformedPosterior (carries .T, .truncation, .x_obs_digest)
    latent: object = None           # the sbi DirectPosterior
    fingerprint: "str | None" = None
    diagnostics: "dict | None" = None
    accepted: list = field(default_factory=list)


@dataclass
class LoadedObservation(Loaded):
    x_obs: object = None            # (1, W) conditioning row
    obs_data: object = None         # (1, N) the observed trace
    t_dim: object = None            # (1, N) seconds
    digest: str = ""
    mode: str = ""
    width: int = 0

    def install(self, cfg) -> None:
        """Put the observation's context back on ``cfg`` -- what infer_and_visualize reads from it:
        T_obs, n_obs, the chi probe frequencies (and their COUNT), the forcing values, and for a
        simulated observation the ground truth and initial conditions (so show_truth works in a fresh
        session)."""
        import torch
        body = self.manifest.body
        cfg.T_obs = float(body["T_obs_cell"])
        cfg.n_obs = int(body["n_obs"]) if body["n_obs"] is not None else None
        cfg.chi_obs_freqs = (None if body["chi_obs_freqs"] is None else
                             torch.tensor(body["chi_obs_freqs"], dtype=cfg.hw.dtype, device=cfg.hw.device))
        if body["chi_obs_freqs"] is not None:
            # cfg.chi_n_freqs is "how many probes THIS OBSERVATION supplies" (it sets no width -- the
            # pad does), and the PPC re-drives the experiment from it, so a fresh session must take it
            # from the observation rather than from config.CHI_N_FREQS.
            cfg.chi_n_freqs = len(body["chi_obs_freqs"])
        src = body["source"]
        if src["kind"] == "simulated":
            cfg.inject_ground_truth(dict(src["inits"]), dict(src["params"]), dict(src["rescale"]),
                                    dict(src["forcing"]))
            # The cell is part of the context: a round drawn around this observation must not record
            # the cell a later inference loaded. Restored only when the file still hashes to the
            # record -- build_posterior hashes sources after the Fisher, so an unchecked path that has
            # moved or changed would turn into a refusal after the spend.
            ref = src.get("cell")
            p = config.RESOURCES_ROOT / ref["path"] if ref else None
            if p is not None and p.is_file() and prov.sha256_file(p) == ref.get("sha256"):
                cfg.sources["cell"] = str(p)
            else:
                cfg.sources.pop("cell", None)
        else:
            # "put back THIS observation's context" cuts both ways: a bench recording has no truth, and
            # one left over from an earlier simulated inference would be read as this observation's by
            # the round's ground-truth check and by the PPC's initial conditions.
            cfg.clear_ground_truth()
        cfg.set_observation_context(cfg.T_obs, dict(body["forcing_vals"] or {}))


@dataclass
class LoadedCalibration(Loaded):
    results: dict = field(default_factory=dict)


@dataclass
class LoadedInference(Loaded):
    results: dict = field(default_factory=dict)
    samples_path: "Path | None" = None


@dataclass
class LoadedDiagnostic(Loaded):
    diagnostic: str = ""            # the diagnostic that wrote it ("sbc", "identifiability", ...)
    variant: "str | None" = None    # its mode where it has more than one ("rotation"/"laplace"/"jacobian")
    settings: dict = field(default_factory=dict)
    results: dict = field(default_factory=dict)


@dataclass
class LoadedFdt(Loaded):
    """One effective-temperature measurement, one parameter sweep, or one comparison of them.

    NOT frozen (planning ruling P1): ``Loaded`` is a plain dataclass, and Python refuses
    ``@dataclass(frozen=True)`` on a subclass of a non-frozen one (TypeError at import). Every
    other ``Loaded*`` in this module is a plain dataclass for the same reason.

    ``body`` is the manifest's own body, copied; ``data_path`` is ``<dir>/data.h5`` when the run got
    as far as writing numbers and None when it did not -- which an INTERRUPTED record (E2) very
    often has not.
    """
    body: dict = field(default_factory=dict)
    data_path: "Path | None" = None


def slug(title: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", title).strip("_").lower()
    return s or "figure"


def _rmtree_retry(path, retries: int = 3, backoff_s: float = 0.1) -> None:
    """shutil.rmtree with the same PermissionError patience as file_manager._atomic_write: on
    Windows a virus scanner or an Explorer preview holds files open for a moment."""
    for attempt in range(retries):
        try:
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            return
        except PermissionError:
            if attempt == retries - 1:
                raise
            time.sleep(backoff_s * (attempt + 1))


def _newest_mtime(path) -> float:
    """The newest modification time anywhere in ``path``'s tree, as a POSIX timestamp.

    The DIRECTORY's own mtime is not the question a removal has to ask: creating or deleting an entry
    touches it, but a long write to a file already inside it does not -- and that is precisely the
    shape of a training that is writing shard after shard into a directory it created minutes ago.
    An entry that cannot be stat'd (it vanished mid-walk) is skipped rather than fatal; the caller is
    about to try to remove the tree anyway.
    """
    newest = 0.0
    for root, dirs, files in os.walk(str(path)):
        for p in (root, *(os.path.join(root, n) for n in (*dirs, *files))):
            try:
                newest = max(newest, os.stat(p).st_mtime)
            except OSError:
                continue
    return newest


def _is_link(path: Path) -> bool:
    """A symbolic link OR a Windows directory junction. ``Path.is_symlink()`` is False for a
    junction, so a guard written with it alone lets one through -- and the resolved-parent check
    beside it cannot catch a junction that points at a directory INSIDE the root, a kind directory
    included (piece 5, E10)."""
    return path.is_symlink() or path.is_junction()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _write_manifest(path: Path, m: mf.Manifest) -> None:
    text = mf.to_json_text(m).encode("utf-8")
    file_manager._atomic_write(path / MANIFEST, lambda fh: fh.write(text))


class ArtifactWriter:
    """Context manager handed out by ``ArtifactStore.create``: creates the directory, hands out
    payload and figure paths, and on a clean exit hashes the payloads, writes the run's ``log.txt``
    and writes the manifest LAST. On ANY exception (a cancel included) the directory is removed and
    the exception re-raised, so a half-artifact never looks real.

    A kind in ``PROGRESSIVE_KINDS`` is written the other way round (piece 5, E2): ``__enter__``
    writes a first manifest carrying every body key with the unknown ones null, ``refresh()``
    re-writes it and the log as the run proceeds, and an exception KEEPS the directory with
    ``complete`` still False -- except a ``Refusal`` raised before the first payload or figure, which
    is a pre-spend refusal and must not leave a permanent empty record behind.
    """

    def __init__(self, store, kind, cfg, *, name, note, id, created):
        self.store, self.kind, self.cfg = store, kind, cfg
        self.name, self.note, self.id, self.created = name, note, id, created
        self.dir = store.kind_dir(kind) / f"{name or mf.UNNAMED_DIR}__{id}"
        self.config = mf.config_from_cfg(cfg) if cfg is not None else {}
        self.parents, self.fingerprints, self.body = {}, {}, {}
        self._payloads, self._figures = [], []
        self.manifest = None
        self.progressive = kind in PROGRESSIVE_KINDS
        # True once a payload path or a figure path has been HANDED OUT. The question ``__exit__``
        # asks of a refusal is "was anything spent", and handing out the path is the last moment
        # this class can observe -- what the caller then does with it is out of sight.
        self._wrote_anything = False
        # True once ``_commit`` has written the final manifest. ``refresh`` refuses from then on: a
        # refresh writes ``complete: false`` and null payload hashes, so a late one would turn a
        # finished record back into an unfinished one that ``load_fdt`` never verifies again.
        self._committed = False
        # The header blocks, computed once and reused by every refresh. git_info runs three git
        # subprocesses and env_info imports torch; a sweep refreshes after every operating point, and
        # neither block can change while one run is in flight.
        self._prism = None
        self._env = None

    def payload(self, filename: str) -> Path:
        if "/" in filename or "\\" in filename:
            raise StoreError(f"a payload is a file directly inside the artifact directory: {filename!r}")
        if filename not in self._payloads:
            self._payloads.append(filename)
        self._wrote_anything = True
        return self.dir / filename

    def figure_path(self, title: str) -> Path:
        figs = self.dir / "figures"
        figs.mkdir(exist_ok=True)
        base = slug(title)
        p, n = figs / f"{base}.png", 2
        while p.exists():
            p = figs / f"{base}-{n}.png"
            n += 1
        self._figures.append(f"figures/{p.name}")
        self._wrote_anything = True
        return p

    def fig_sink(self, forward=None):
        """A ``(title, fig) -> None`` sink that saves the PNG here and THEN forwards to the GUI's sink
        (which closes the figure after rendering, base_panel._png_fig_sink) or closes it itself."""
        def _sink(title, fig):
            from matplotlib import pyplot as plt
            from core.Helpers.visualizers import save_figure
            save_figure(fig, self.figure_path(title), dpi=110, bbox_inches="tight")
            if forward is not None:
                forward(title, fig)
            else:
                plt.close(fig)
        return _sink

    def __enter__(self):
        self.dir.mkdir(parents=True, exist_ok=False)
        if self.progressive:
            # The FIRST manifest, before a single number is computed (spec §2.2 step 1). ``validate``
            # refuses a partial body key set, so this carries every key: what the caller set between
            # create() and here, and null for the rest.
            try:
                self._write(hashed=False, complete=False)
            except BaseException:
                # ``__exit__`` never runs when ``__enter__`` raises, so this is the one place that can
                # take the folder back (fix round 1, finding 1). A first body the manifest cannot hold
                # -- a NaN ``validate`` refuses, or a numpy scalar that passes it and then fails in
                # ``json.dumps`` after the log is on disk -- would otherwise leave a manifest-less
                # folder for a run that never started, and every refused click would add one. The
                # manifest is written last and atomically, so a failure here means it never landed.
                self._remove_dir("it has no manifest, so the store ignores it")
                raise
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            if self.progressive and not (isinstance(exc, Refusal) and not self._wrote_anything):
                # E2. The run was interrupted or it failed AFTER spending something: the folder stays,
                # ``complete`` stays False, and one last refresh puts the log up to the failure on
                # disk -- that log is the document that says why it stopped.
                #
                # The log FIRST, and in an attempt of its own (fix round 1, finding 3): a body the
                # final manifest cannot validate -- a NaN the stage stored as it failed -- must not
                # take the records since the last good refresh down with it. Neither attempt may
                # REPLACE the exception that ended the run, so each failure is a warning.
                try:
                    self._write_log()
                except Exception as e:        # noqa: BLE001 -- the in-flight cause must propagate
                    warnings.warn(f"could not write the log of the unfinished {self.kind} record "
                                  f"{self.dir} ({type(e).__name__}: {e}); it keeps the log it had.",
                                  stacklevel=2)
                try:
                    self._write(hashed=False, complete=False, log=False)
                except Exception as e:        # noqa: BLE001 -- the in-flight cause must propagate
                    warnings.warn(f"could not refresh the unfinished {self.kind} record {self.dir} "
                                  f"({type(e).__name__}: {e}); it keeps the manifest it had.",
                                  stacklevel=2)
                return False
            if self.progressive:
                # Refused before it wrote anything, so it DOES carry its first manifest (ruling F30).
                # Conditional, because the removal may have got as far as the manifest before it
                # failed: what is left is then a manifest-less leftover, not an unfinished record.
                self._remove_dir("if its manifest is still there it is listed as unfinished and can be "
                                 "removed with the delete action; if not, the leftover sweep clears it")
            else:
                self._remove_dir("it has no manifest, so the store ignores it")
            return False
        self._commit()
        return False

    def _remove_dir(self, left: str) -> None:
        """Remove this record's directory on a failure path; ``left`` says what remains if it will not go.

        The cleanup must never REPLACE the exception that ended the run -- the one the operator needs
        -- with a PermissionError about a directory the index already ignores (it has no manifest, so
        it is incomplete by definition and nothing will ever load it), so a failure here is a warning.
        Shared by ``__enter__`` and ``__exit__``: stacklevel 3 is this frame, then theirs, then the
        caller's ``with`` line, which is the one the warning names.
        """
        try:
            _rmtree_retry(self.dir)
        except OSError as e:              # noqa: BLE001 -- the in-flight cause must propagate, not this
            warnings.warn(f"could not remove the incomplete artifact directory {self.dir} "
                          f"({type(e).__name__}: {e}); {left}.",
                          stacklevel=3)

    def refresh(self) -> None:
        """Re-write the manifest and ``log.txt`` for a record still being written (spec §2.2 step 2).

        Called by the stage at points it chooses -- after the spontaneous campaign, after each
        operating point, after each figure -- so a browser row, a listing and the log all follow a
        run that may take hours. The write is the same atomic one ``_commit`` performs.

        Payloads are NOT hashed here: a file still being appended to would hash to a value that is
        wrong the moment it is written, so an unfinished record lists its payloads with a null hash
        and the real hashes land at the commit. ``load_fdt`` treats a null hash as "not yet".

        Refused once the record is committed (fix round 1, finding 2): for the same reason, a refresh
        then would write ``complete: false`` and null hashes over a finished record.
        """
        if not self.progressive:
            raise StoreError(f"{self.kind} artifacts are written in one step, so there is nothing to "
                             f"refresh; only {sorted(PROGRESSIVE_KINDS)} are written progressively")
        if self._committed:
            raise StoreError(f"the {self.kind} record {self.name or self.id} is already committed; a "
                             f"refresh now would mark a finished record unfinished and drop the hashes "
                             f"of its payloads")
        self._write(hashed=False, complete=False)

    def _write(self, *, hashed: bool, complete: bool, log: bool = True) -> None:
        """Build, validate and write the manifest (and, unless ``log`` is False, the run's log beside
        it, first). ``log=False`` is the failure path's, which has written the log on its own."""
        self.manifest = mf.validate(self._manifest_dict(hashed=hashed, complete=complete))
        if log:
            self._write_log()
        _write_manifest(self.dir, self.manifest)

    def _write_log(self) -> None:
        # The run's log so far, BEFORE the manifest: every ``core`` record and every Python warning
        # since the outermost public entry began (core/runs.py). A composition's first artifact
        # therefore holds the records up to its own commit and its last one the whole run. Written
        # whenever a run is active, empty when it said nothing: every committed artifact but the
        # simulation cache (which has no writer) carries the file. Outside any entry, no file.
        run_log = runs.current_run_log()
        if run_log is not None:
            # newline="\n" explicitly: write_text's default would make every record CRLF on Windows,
            # and this file is read back by read_log, shown in the browser's detail pane and saved
            # verbatim to a report whose bytes spec §5 says both front ends must agree on.
            (self.dir / LOG_FILE).write_text(run_log.text(), encoding="utf-8", newline="\n")

    def _inputs(self, *, warn: bool) -> dict:
        # missing_ok: this runs at the END of a run that may have taken days, and an input file moved
        # or edited meanwhile must be recorded as unhashed rather than raise here and lose everything
        # the run produced. Named in a warning so the gap is not silent -- ONCE, at the commit: a
        # progressive record refreshes many times and would otherwise repeat it on every one.
        inputs = (prov.inputs_from_cfg(self.cfg, missing_ok=True) if self.cfg is not None
                  else {"bounds": None, "cell": None, "units": None, "model": None})
        gone = [k for k, v in inputs.items() if isinstance(v, dict) and v.get("sha256") is None]
        if gone and warn:
            warnings.warn(
                f"{self.kind} artifact {self.id}: input file(s) "
                + ", ".join(f"{k} ({inputs[k]['path']})" for k in gone)
                + " could not be read at commit, so the manifest records them unhashed. The artifact "
                  "is written anyway -- losing a finished run to a moved input file would be worse.",
                # _inputs <- _manifest_dict <- _write <- _commit <- __exit__ <- the caller's ``with``:
                # the frame the old _commit's stacklevel=3 named, so the six ordinary kinds' warning
                # still points at the caller's code and not at this module.
                stacklevel=6)
        return inputs

    def _manifest_dict(self, *, hashed: bool, complete: bool) -> dict:
        hw = getattr(self.cfg, "hw", None)
        if self._prism is None:
            self._prism = prov.git_info(config.REPO_ROOT)
        if self._env is None:
            self._env = prov.env_info(hw if hw is not None else config.cpu_device())
        body = self.body
        if self.progressive:
            # ``complete`` is the WRITER's field: it is what says whether this record's run reached
            # the end, and the writer is the only party that knows. Filling the rest with null is
            # what lets the caller set a key when it learns it rather than up front. A COPY, so the
            # caller's own dict never gains keys it did not set; an ordinary kind's body goes in as
            # the very object it always did.
            body = dict(body)
            body["complete"] = bool(complete)
            for k in mf.BODY_KEYS[self.kind]:
                body.setdefault(k, None)
        return dict(
            schema=mf.SCHEMA, kind=self.kind, id=self.id, name=self.name, created=self.created.isoformat(),
            note=self.note, prism=self._prism, env=self._env,
            inputs=self._inputs(warn=hashed),
            config=self.config, parents=dict(self.parents), fingerprints=dict(self.fingerprints),
            payloads={f: (prov.sha256_file(self.dir / f) if hashed else None) for f in self._payloads},
            figures=list(self._figures), body=body,
        )

    def _commit(self) -> None:
        self._write(hashed=True, complete=True)
        self._committed = True


def _as_int(v) -> "int | None":
    """``manifest.validate`` checks body key-sets and top-level types only, never what is INSIDE a
    count -- so a hand-edited or partially-written manifest can hold a non-numeric value where
    ``list`` expects one. Coerce leniently and answer None rather than raise, so one corrupt
    directory degrades a single field instead of taking the whole listing down with it."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return int(f) if math.isfinite(f) else None


class ArtifactStore:
    def __init__(self, root, *, clock=None):
        self.root = Path(root)
        self._clock = clock or _utc_now

    def kind_dir(self, kind: str) -> Path:
        if kind not in KIND_DIRS:
            raise StoreError(f"unknown artifact kind {kind!r}")
        return self.root / KIND_DIRS[kind]

    # ── index ────────────────────────────────────────────────────────────────────────────────────
    def _entries(self, kind: str) -> list:
        """``(dir, Manifest | None, reason)`` for every directory under the kind."""
        d = self.kind_dir(kind)
        if not d.is_dir():
            return []
        out = []
        for sub in sorted(p for p in d.iterdir() if p.is_dir()):
            mpath = sub / MANIFEST
            if not mpath.exists():
                out.append((sub, None, NO_MANIFEST_REASON))
                continue
            try:
                m = mf.from_json_text(mpath.read_text(encoding="utf-8"))
            except (mf.ManifestError, OSError) as e:
                out.append((sub, None, f"unreadable manifest: {e}"))
                continue
            if m.kind != kind:
                out.append((sub, None, f"manifest declares kind {m.kind!r}"))
                continue
            out.append((sub, m, None))
        return out

    def list(self, kind: str) -> list:
        """Every directory under ``kind`` as a ``Summary``: complete rows first, newest first.

        Reads each manifest ONCE and KEEPS what it read, so a browser row needs no second read
        (piece 4, B2). ``complete`` and ``finished`` are different questions -- see ``Summary``.
        """
        out = []
        for sub, m, reason in self._entries(kind):
            if m is None:
                out.append(Summary(kind, sub.name, "", "", "", sub, False, reason, dir_name=sub.name))
                continue
            body = m.body
            # Derived from the SCHEMA, not from a hand-written kind name: a kind whose body carries
            # a ``complete`` flag is one whose manifest exists before the run has finished (the
            # cache, because it is resumable; an fdt record, because E2 keeps an interrupted one),
            # and for every other kind a manifest IS the finish -- ``_commit`` writes it last. The
            # old form named "simulation" alone and would have silently called every half-written
            # fdt record finished.
            finished = (bool(body.get("complete")) if "complete" in mf.BODY_KEYS[kind] else True)
            # A manifest body is not type-validated below its key set (manifest.validate checks
            # key-sets and top-level types only), so a hand-edited or partially-written body can hold
            # a non-numeric count here. Coerce leniently -- a malformed FIELD degrades to None, never
            # a raise that would take the whole listing down with it.
            per_batch = body.get("rows")
            try:
                row_ints = None if per_batch is None else [_as_int(r) for r in per_batch]
            except TypeError:
                row_ints = None
            coerced_rows = None if row_ints is None or any(r is None for r in row_ints) else tuple(row_ints)
            # The PLANNED batch count lives only in the cache's identity -- the dict the naming digest
            # is taken over -- and a row that did not carry it could not render "3/4 batches" without
            # reading this manifest again (piece 4, B2).
            ident = (body.get("identity") or {}) if kind == "simulation" else {}
            # An fdt sweep's operating points, off the SAME manifest read (B2). A single run and a
            # comparison carry ``points: null``, so the counts stay None and the cell stays blank --
            # "0/0" would be a claim neither ever makes. ``_as_int`` is what keeps a hand-edited or
            # partially written body from taking the whole listing down.
            pts = body.get("points")
            pts = pts if isinstance(pts, dict) else {}
            out.append(Summary(kind, m.id, m.name, m.created, m.note, sub, True, None,
                               mode=body.get("mode"), width=(body.get("conditioning") or {}).get("width"),
                               amortized=body.get("amortized"), parents=dict(m.parents),
                               dir_name=sub.name, finished=finished,
                               batches_done=_as_int(body.get("batches_done")),
                               batches_planned=_as_int(ident.get("n_runs")),
                               rows=coerced_rows,
                               variant=body.get("variant"),
                               study=body.get("study"),
                               points_done=_as_int(pts.get("done")),
                               points_planned=_as_int(pts.get("planned")),
                               points_failed=_as_int(pts.get("failed"))))
        out.sort(key=lambda s: s.created, reverse=True)       # newest first ...
        out.sort(key=lambda s: not s.complete)                # ... complete first (stable)
        return out

    def _find(self, kind: str, ref: str):
        for sub, m, _ in self._entries(kind):
            if m is not None and (m.id == ref or (m.name and m.name == ref)):
                return sub, m
        return None, None

    def get(self, kind: str, ref: str) -> mf.Manifest:
        _, m = self._find(kind, ref)
        if m is None:
            incomplete = [s.name for s, mm, _ in self._entries(kind) if mm is None]
            raise StoreError(f"no complete {kind} artifact named or id'd {ref!r} under {self.kind_dir(kind)}"
                             + (f" (incomplete directories present: {incomplete})" if incomplete else ""))
        return m

    def path(self, kind: str, ref: str) -> Path:
        sub, m = self._find(kind, ref)
        if m is None:
            raise StoreError(f"no complete {kind} artifact named or id'd {ref!r}")
        return sub

    def read_log(self, kind: str, ref: str, *, max_bytes: "int | None" = None) -> "tuple[str | None, bool]":
        """``(text, truncated)`` of the artifact's ``log.txt``; ``(None, False)`` when there is no file.

        None and "" are DIFFERENT answers, deliberately. The simulation cache never gets a log: it has
        no writer (``create`` refuses the kind), its manifest is refreshed batch by batch across
        resumes, and one cache is shared by every posterior that names it -- so no single commit holds
        one entry's records. An artifact written outside any public entry gets none either. But a run
        that said NOTHING writes an EMPTY one: silence is a record too (piece 3, V4).

        With ``max_bytes`` the TAIL is returned and ``truncated`` is True -- the end is where the
        failure is -- and its first line may be a partial one.

        Newlines come back as LF. ``_commit`` writes the file with ``newline="\\n"``, but a log.txt an
        OLDER build wrote went through ``write_text``'s default ``newline=None`` and is CRLF on
        Windows; normalising here means one convention reaches the pane, the saved report and both
        front ends whoever wrote the file.

        THE one place that joins ``LOG_FILE``, so "is there a log, and what does it say" is answered
        once and the cache's absence is explained here rather than in each front end (piece 4, B4).

        Resolves with ``_find`` rather than through ``path()`` so that both refusals can carry
        ``field="artifact"``: ``path()``'s is field-less, and an inherited field-less refusal leaves
        each front end with no control to name.
        """
        if kind not in KIND_DIRS:
            raise StoreError(f"unknown artifact kind {kind!r}", field="artifact")
        sub, m = self._find(kind, ref)
        if m is None:
            raise StoreError(f"no complete {kind} artifact named or id'd {ref!r}", field="artifact")
        f = sub / LOG_FILE
        try:
            data = f.read_bytes()
        except (FileNotFoundError, NotADirectoryError):
            return None, False
        truncated = max_bytes is not None and len(data) > max_bytes
        if truncated:
            # len(data) - max_bytes, NOT -max_bytes: ``data[-0:]`` is the whole file, so a caller
            # asking for nothing would be handed everything.
            data = data[len(data) - max_bytes:]
        # errors="replace" is the identity over a whole, valid file; it matters only for a tail cut
        # in the middle of a multi-byte character, which must render rather than raise.
        text = data.decode("utf-8", errors="replace")
        return text.replace("\r\n", "\n"), truncated

    # ── create / rename / delete ─────────────────────────────────────────────────────────────────
    def _new_id(self, kind: str) -> str:
        stamp = self._clock().strftime("%Y%m%dT%H%M%S")
        taken = {m.id for _, m, _ in self._entries(kind) if m is not None}
        d = self.kind_dir(kind)
        if d.is_dir():
            taken |= {p.name.rsplit("__", 1)[-1] for p in d.iterdir() if p.is_dir()}
        cand, n = stamp, 2
        while cand in taken:
            cand, n = f"{stamp}-{n}", n + 1
        return cand

    def assert_name_free(self, kind: str, name: str, *, allow: "str | None" = None) -> None:
        """Refuse a bad or taken artifact name for ``kind``, raising the same StoreError ``create``
        would raise at the END of the stage.

        PUBLIC, and called at every stage's entry, because ``create`` runs after the expensive part:
        a prior sweep is ~9 minutes and a training run is days, and learning that the name is taken
        only once the writer opens is how a finished run gets thrown away. Cheap (it reads the kind's
        manifests) and idempotent, so a stage may call it and ``create`` may check again.

        An empty name is always free -- unnamed is the default, and a kind may hold any number of
        unnamed artifacts. ``allow`` is the id already entitled to the name (``rename``'s own
        artifact), which is not a collision with itself.

        Every refusal here carries ``field="name"`` (piece 3, V3): the message names no box and no
        flag; each front end's table maps the key to its own control.
        """
        if not name:
            return
        if not mf.NAME_RE.match(name):
            raise StoreError(f"bad artifact name {name!r}: letters, digits, _ . - and at most 64 characters",
                             field="name")
        if mf.ID_RE.match(name):
            # get()/path()/_find resolve a ref as an id OR a name, and the id is tried first: a name
            # shaped like an id would shadow the artifact whose id it is, which could then never be
            # addressed at all.
            raise StoreError(f"bad artifact name {name!r}: it is shaped like an artifact id, which would "
                             f"shadow the artifact whose id it is in get() and path()", field="name")
        other = self._find(kind, name)[1]
        if other is not None and (allow is None or other.id != allow):
            raise StoreError(f"a {kind} named {name!r} already exists; rename or delete it first", field="name")

    def create(self, kind: str, cfg=None, *, name: str = "", note: str = "") -> ArtifactWriter:
        if kind == "simulation":
            raise StoreError("simulation manifests are written by training_checkpoint (write_simulation_manifest)")
        self.assert_name_free(kind, name)
        return ArtifactWriter(self, kind, cfg, name=name, note=note, id=self._new_id(kind), created=self._clock())

    def rename(self, kind: str, ref: str, new_name: str) -> mf.Manifest:
        if not new_name:
            raise StoreError(f"bad artifact name {new_name!r}: letters, digits, _ . - and at most 64 characters",
                             field="name")
        sub, m = self._find(kind, ref)
        if m is None:
            raise StoreError(f"no complete {kind} artifact named or id'd {ref!r}")
        self.assert_name_free(kind, new_name, allow=m.id)      # the same rules as create's
        m.name = new_name
        _write_manifest(sub, m)                       # the index first; the directory name is convenience
        target = sub.with_name(m.dir_name)
        if target != sub:
            for attempt in range(3):
                try:
                    sub.rename(target)
                    break
                except PermissionError:
                    if attempt == 2:
                        break                         # tolerated: the manifest is what resolves
                    time.sleep(0.1 * (attempt + 1))
        return m

    def set_note(self, kind: str, ref: str, note: str) -> mf.Manifest:
        """Rewrite one artifact's note. ``field="artifact"`` (piece 4, design §2.4): the front ends
        name the control or the flag themselves, and this was the one store mutation that named
        nothing. The key is the ARTIFACT and not the note: what is wrong is the ref, so the fix is to
        select an artifact that exists, not to edit the note box. ``field="note"`` is
        ``core.refusals.require_note``'s, for a note whose TEXT is refused.

        The text is not judged here -- ``require_note`` is that rule, and both front ends run it
        before they call this -- so a caller that reaches past them writes what it passes.
        """
        sub, m = self._find(kind, ref)
        if m is None:
            raise StoreError(f"no complete {kind} artifact named or id'd {ref!r}", field="artifact")
        m.note = str(note)
        _write_manifest(sub, m)
        return m

    def dependents(self, kind: str, id_: str) -> list:
        """``[(kind, id, name)]`` of every artifact whose parents name this one.

        For a PRIOR, also every simulation cache GENERATED AGAINST it -- its identity's
        ``prior_fingerprint`` is this prior's GMM -- whether or not it names the prior as a parent
        (a cache created before the parent was recorded, or by a script that passed a stand-in). The
        cache directory is keyed on that fingerprint and its rows are meaningless without the prior
        they were drawn from, so deleting the prior would orphan days of simulation. Either side
        being None (a pre-fingerprint manifest, a stand-in prior) is not a match: unverifiable is
        not the same as equal.
        """
        out, seen = [], set()
        for k in KIND_DIRS:
            for _, m, _ in self._entries(k):
                if m is not None and any(m.parents.get(pk) == id_ for pk in _PARENT_KEYS.get(kind, ())):
                    out.append((k, m.id, m.name))
                    seen.add((k, m.id))
        if kind == "prior":
            owner = self._find("prior", id_)[1]
            want = owner.fingerprints.get("gmm") if owner is not None else None
            if want is not None:
                for _, m, _ in self._entries("simulation"):
                    if m is not None and ("simulation", m.id) not in seen \
                            and m.fingerprints.get("gmm") == want:
                        out.append(("simulation", m.id, m.name))
        return out

    def delete(self, kind: str, ref: str, *, force: bool = False) -> None:
        sub, m = self._find(kind, ref)
        if m is None:
            raise StoreError(f"no complete {kind} artifact named or id'd {ref!r}", field="artifact")
        deps = self.dependents(kind, m.id)
        if deps and not force:
            listed = "; ".join(f"{k} {n or '(unnamed)'} [{i}]" for k, i, n in deps)
            # The wording does not move, ``force=True`` included: core/refusals.py's own rule is that
            # a message names no box, tab, flag or button but MAY name a Python keyword -- "it is the
            # core API's own word". No front end offers force (B6); the log's reader still needs the
            # escape hatch's name. Both refusals take field="artifact": either is answered by picking
            # a different artifact in the list (piece 4, P1/P20).
            raise StoreError(
                f"refusing to delete {kind} {m.name or m.id}: {len(deps)} artifact(s) name it as a "
                f"parent -- {listed}. Delete those first, or pass force=True to orphan them.",
                field="artifact")
        # The same two guards ``remove_incomplete`` and ``sweep_incomplete`` were given in fix
        # round 2, and this is the only one of the three that removes a REAL artifact (R7).
        try:
            _rmtree_retry(sub)
        except OSError as e:
            # A held handle (an Explorer preview, a scanner) is not a bug: without this wrapper it
            # escapes the tool's Refusal rung as a raw traceback for "try again in a moment".
            raise StoreError(f"could not remove {kind} {m.name or m.id} at {sub}: "
                             f"{type(e).__name__}: {e}", field="artifact") from e
        if sub.exists():
            # _rmtree_retry treats ANY FileNotFoundError as "already gone" -- including one raised
            # from inside the walk, after part of the tree has been removed. Verified, not assumed.
            raise StoreError(f"{kind} {m.name or m.id} at {sub} was not removed", field="artifact")

    def remove_incomplete(self, kind: str, dir_name: str) -> Path:
        """Remove one directory under ``kind`` that has NO manifest file at all. Returns the path
        removed.

        THE COMPLEMENT OF ``delete``. That one resolves through ``_find``, which only ever returns
        manifest-bearing entries, so it can only remove a real artifact -- and therefore always runs
        the dependency check. This one asks ``_entries`` (the same classifier ``list`` shows) and
        removes only what came back WITHOUT a manifest. Neither call can do the other's job, which is
        the safety property: nothing can reach a leftover folder by accident, and nothing can reach a
        real artifact without the dependency check (piece 4, B7).

        Refuses, as a StoreError with ``field="artifact"``: an unknown kind; a ``dir_name`` that is
        not a direct child (any separator, ``..``, an absolute path); a name that resolves to no
        directory at all; a directory ``_entries`` classifies as COMPLETE; a directory that carries a
        manifest.json this build cannot use or that declares another kind (R1 -- see
        ``NO_MANIFEST_REASON``); and one whose tree was written within ``RECENT_WRITE_SECONDS``, which
        may be a run in flight in another process (R3).

        Fix round 1 (CRITICAL): the completeness check does NOT compare ``dir_name`` as a string
        against each entry's name. Windows addresses one directory through many spellings -- a case
        variant, an 8.3 short name, a trailing dot it silently strips -- so a string compare against
        ``d / dir_name`` could resolve to a REAL artifact's directory while failing to match that
        same artifact's name in ``_entries``'s listing, and the manifest check would then see no
        match and treat a real, possibly dependency-bearing artifact as a nameless leftover. Instead,
        every directory in ``_entries(kind)`` is checked with ``os.path.samefile`` AND a resolved-path
        comparison against the candidate path (fix round 2): ``samefile`` catches an alias no string
        compare would (its identity check is ``(st_dev, st_ino)``), but on a volume where ``st_ino``
        is 0 for every entry -- FAT/exFAT, some network shares, and the artifacts root is
        relocatable to exactly such a drive -- ``samefile`` would call EVERY entry a match, matching
        whichever one ``_entries`` happens to list first. ``os.path.realpath`` catches that
        degenerate case because it compares the resolved path STRING, which is not built from the
        (possibly all-zero) inode. Neither check alone is sufficient; both must hold.
        """
        if kind not in KIND_DIRS:
            # Not kind_dir()'s own refusal, which carries no field key: every refusal on this path
            # names the artifact, so a front end can say where to pick another one.
            raise StoreError(f"unknown artifact kind {kind!r}", field="artifact")
        d = self.kind_dir(kind)
        if (not dir_name or dir_name in (".", "..") or "/" in dir_name or "\\" in dir_name
                or dir_name != Path(dir_name).name):
            # The separator checks are not redundant with the Path comparison: on POSIX a backslash
            # is an ordinary character, so "sub\\x" would pass it. Necessary, but -- per the review --
            # NOT sufficient: they reject an obvious path, not an alias of a single real name.
            raise StoreError(f"{dir_name!r} is not the name of a directory directly under {d}; a "
                             f"leftover is removed by its own folder name, never by a path",
                             field="artifact")
        candidate = d / dir_name
        if not candidate.is_dir():
            raise StoreError(f"no directory named {dir_name!r} under {d}", field="artifact")
        sub = m = reason = None
        for s, mm, why in self._entries(kind):
            try:
                if os.path.samefile(candidate, s) and os.path.realpath(candidate) == os.path.realpath(s):
                    sub, m, reason = s, mm, why
                    break
            except OSError:
                continue        # an entry that vanished mid-scan is not a match, not a crash
        if sub is None:
            raise StoreError(f"no directory named {dir_name!r} under {d}", field="artifact")
        if m is not None:
            raise StoreError(
                f"{dir_name!r} holds a valid {kind} manifest, so it is a real artifact and not a "
                f"leftover; remove it with delete(), which refuses it while anything depends on it",
                field="artifact")
        if reason != NO_MANIFEST_REASON:
            # R1. "This build cannot parse it" is NOT "there is no artifact here": a manifest written
            # under a different SCHEMA, or by a newer build, or read through a transient Windows
            # failure (the very class _rmtree_retry retries for, one function away) all land here --
            # a reviewer probed the old rule deleting a real calibration with its payload. A
            # directory nobody understands is reported, never removed.
            raise StoreError(
                f"{dir_name!r} under {d} carries a manifest.json, so it is not a leftover: {reason}. "
                f"A sweep removes only a directory with NO manifest at all; this one keeps its row "
                f"in the listing, with that reason, and is removed by hand if it really is dead",
                field="artifact")
        age = time.time() - _newest_mtime(sub)
        if age < RECENT_WRITE_SECONDS:
            # R3. The manifest is written LAST, so a run in flight -- possibly in ANOTHER process,
            # which no _running flag can see -- is indistinguishable from a leftover by content
            # alone. Age is the one signal available without a cross-process lock.
            raise StoreError(
                f"{dir_name!r} under {d} was written {age:.0f} s ago, so something may still be "
                f"writing it: an artifact's manifest is written last, and a run in flight looks "
                f"exactly like a leftover until it commits. Leave it at least "
                f"{RECENT_WRITE_SECONDS:.0f} s and sweep again",
                field="artifact")
        try:
            _rmtree_retry(sub)
        except OSError as e:
            # A junction or symlink can make rmtree fail in a way _rmtree_retry's own
            # FileNotFoundError/PermissionError handling does not absorb. That exception is not a
            # Refusal and carries no field key, so a front end catching Refusal would see it as an
            # unhandled crash rather than a reported refusal.
            raise StoreError(f"could not remove {dir_name!r} under {d}: {type(e).__name__}: {e}",
                             field="artifact") from e
        if sub.exists():
            # A path trick (a trailing space or "..." that Win32 quietly resolves to nothing) can
            # make shutil.rmtree raise FileNotFoundError, which _rmtree_retry treats as "already
            # gone" -- so trust is verified here rather than assumed from a call that returned.
            raise StoreError(f"{dir_name!r} under {d} was not removed", field="artifact")
        return sub

    def loose_files(self, kind: str) -> "list[LooseFile]":
        """Every FILE sitting directly inside ``kind``'s directory, sorted by name: a kind directory
        holds artifact directories and nothing else, so each of these breaks the store's oldest rule.

        The complement of ``_entries``, which iterates directories only -- so between them the two
        account for everything under a kind directory, and neither can see what the other owns. A
        valid record's payloads are never here: they are inside the record's own directory.

        A kind directory that does not exist has no loose files; that is not an error, because a
        store is allowed to be empty.
        """
        if kind not in KIND_DIRS:
            # Not kind_dir()'s own refusal, which carries no field key -- remove_incomplete's rule.
            raise StoreError(f"unknown artifact kind {kind!r}", field="artifact")
        d = self.kind_dir(kind)
        if not d.is_dir():
            return []
        out = []
        for p in sorted(d.iterdir()):
            try:
                if not p.is_file():
                    continue
                st_ = p.stat()
            except OSError:
                continue          # an entry that vanished mid-scan is not reported, and not a crash
            out.append(LooseFile(p.name, int(st_.st_size), float(st_.st_mtime)))
        return out

    def remove_loose(self, kind: str, filename: str) -> None:
        """Remove ONE loose file from ``kind``'s directory, by its own name.

        The file counterpart of ``remove_incomplete``, and hardened the same way. It resolves the
        name against ``loose_files``'s own entries with ``os.path.samefile`` AND a resolved-path
        comparison, for the reason that call spells out at length: Windows addresses one file through
        many spellings, and on a volume where ``st_ino`` is 0 for every entry ``samefile`` alone
        would call every entry a match. Because those entries are FILES, this call can never reach a
        directory -- so it can never reach an artifact, and ``remove_incomplete`` can never reach a
        loose file. Neither can do the other's job, which is the safety property.

        Refuses, as a StoreError with ``field="artifact"``: an unknown kind; a name that is not a
        direct child (any separator, ``..``, an absolute path); a name that resolves to no file (a
        directory included); and one written within ``RECENT_WRITE_SECONDS``, which may be a run in
        flight in another process -- the same guard, for the same reason (R3).
        """
        if kind not in KIND_DIRS:
            raise StoreError(f"unknown artifact kind {kind!r}", field="artifact")
        d = self.kind_dir(kind)
        if (not filename or filename in (".", "..") or "/" in filename or "\\" in filename
                or filename != Path(filename).name):
            raise StoreError(f"{filename!r} is not the name of a file directly under {d}; a loose "
                             f"file is removed by its own name, never by a path", field="artifact")
        candidate = d / filename
        chosen = None
        for lf in self.loose_files(kind):
            p = d / lf.name
            try:
                if os.path.samefile(candidate, p) and os.path.realpath(candidate) == os.path.realpath(p):
                    chosen = p
                    break
            except OSError:
                continue
        if chosen is None:
            raise StoreError(f"no loose file named {filename!r} under {d}; an artifact's own "
                             f"directory is removed by delete() or by remove_incomplete(), never here",
                             field="artifact")
        try:
            age = time.time() - chosen.stat().st_mtime
        except OSError as e:
            raise StoreError(f"could not read {filename!r} under {d}: {type(e).__name__}: {e}",
                             field="artifact") from e
        if age < RECENT_WRITE_SECONDS:
            raise StoreError(
                f"{filename!r} under {d} was written {age:.0f} s ago, so something may still be "
                f"writing it. Leave it at least {RECENT_WRITE_SECONDS:.0f} s and sweep again",
                field="artifact")
        try:
            chosen.unlink()
        except OSError as e:
            raise StoreError(f"could not remove {filename!r} under {d}: {type(e).__name__}: {e}",
                             field="artifact") from e
        if chosen.exists():
            raise StoreError(f"{filename!r} under {d} was not removed", field="artifact")

    def sweep_incomplete(self, entries) -> "tuple[list, list]":
        """Remove EXACTLY the ``(kind, dir_name)`` pairs handed in. ``(removed, failed)``, where
        ``removed`` is ``[(kind, dir_name)]`` and ``failed`` is ``[(kind, dir_name, reason)]``.

        R2: this call no longer RE-SCANS. It used to take a kind and remove whatever was incomplete
        at the moment it ran, which meant the confirmation the operator answered did not bind the
        action -- a reviewer probed it removing a directory created while the dialog sat open, and
        reporting "Removed 2 of 1 leftover directories". The caller now computes the candidates,
        shows them, and hands that list here; anything that appeared since is simply not in it.

        Each entry goes BY NAME through ``remove_incomplete``, which is the one call hardened for
        this (a Windows name alias cannot reach a real artifact, a directory that became complete
        since the listing is refused, a manifest this build cannot read is refused, a tree something
        may still be writing is refused, and the removal is verified). Its refusal becomes this
        entry's ``failed`` reason, so no removal rule lives in two places and the caller's count
        sentence is true by construction.

        A directory that will not delete is REPORTED, never fatal -- the sweep finishes the rest. On
        Windows a held handle (an Explorer preview, a virus scanner, a file this process still has
        open) makes ``shutil.rmtree`` raise PermissionError; ``_rmtree_retry`` waits 0.1 s and then
        0.2 s and then gives up, and one such directory must not cost the operator the other six.
        """
        removed, failed = [], []
        for kind, dir_name in entries:
            try:
                self.remove_incomplete(kind, dir_name)
            except StoreError as e:            # every refusal on that path is one, and fielded
                failed.append((kind, dir_name, e.message))
            except OSError as e:               # noqa: BLE001 -- one held handle must not stop the sweep
                failed.append((kind, dir_name, f"{type(e).__name__}: {e}"))
            else:
                removed.append((kind, dir_name))
        return removed, failed

    def legacy_dirs(self) -> "list[str]":
        """The names in ``LEGACY_DIRS`` that actually exist as directories under this root.

        A name that is also a kind directory is skipped whatever the literal says: that is the
        structural guarantee that this family can never reach a real artifact, rather than a promise
        the literal is trusted to keep. A link is skipped too -- a junction included, which
        ``is_symlink()`` does not report on Windows -- because ``remove_legacy`` refuses one, and
        offering what the removal will refuse is a list the operator cannot act on.
        """
        taken = set(KIND_DIRS.values())
        return [n for n in LEGACY_DIRS
                if n not in taken and (self.root / n).is_dir() and not _is_link(self.root / n)]

    def remove_legacy(self, name: str) -> None:
        """Remove one legacy directory under this root, by its own name.

        Refuses, as a StoreError with ``field="artifact"``: a name not in ``LEGACY_DIRS``; a name
        that is a KIND directory (belt beside the braces in ``legacy_dirs``); a name that is not a
        direct child; a name that is not a directory or is a link (a junction called ``crossval``
        pointing at C:\\ must not be followed, and ``_is_link`` is what sees a junction); and a tree
        written within ``RECENT_WRITE_SECONDS``, the same guard ``remove_incomplete`` applies, for
        the same reason.

        The resolved path's parent is compared against the resolved root, which is what a string
        compare cannot do: it is the realpath half of ``remove_incomplete``'s hardening, applied to a
        directory that carries no manifest to identify it by.
        """
        if name not in LEGACY_DIRS or name in set(KIND_DIRS.values()):
            raise StoreError(f"{name!r} is not a legacy directory this build knows; the ones it "
                             f"offers to clear are {list(LEGACY_DIRS)}", field="artifact")
        if not name or name in (".", "..") or "/" in name or "\\" in name or name != Path(name).name:
            raise StoreError(f"{name!r} is not the name of a directory directly under {self.root}",
                             field="artifact")
        candidate = self.root / name
        if _is_link(candidate) or not candidate.is_dir():
            raise StoreError(f"no legacy directory named {name!r} under {self.root}", field="artifact")
        if os.path.dirname(os.path.realpath(candidate)) != os.path.realpath(self.root):
            raise StoreError(f"{name!r} under {self.root} resolves outside the store root, so it is "
                             f"not this store's to remove", field="artifact")
        age = time.time() - _newest_mtime(candidate)
        if age < RECENT_WRITE_SECONDS:
            raise StoreError(
                f"{name!r} under {self.root} was written {age:.0f} s ago, so something may still be "
                f"writing it. Leave it at least {RECENT_WRITE_SECONDS:.0f} s and sweep again",
                field="artifact")
        try:
            _rmtree_retry(candidate)
        except OSError as e:
            raise StoreError(f"could not remove {name!r} under {self.root}: {type(e).__name__}: {e}",
                             field="artifact") from e
        if candidate.exists():
            raise StoreError(f"{name!r} under {self.root} was not removed", field="artifact")

    def unnamed(self, kind: str, *, older_than: "timedelta | None" = None) -> list:
        now = self._clock()
        out = []
        for s in self.list(kind):
            if s.name or not s.complete:
                continue
            if older_than is not None and datetime.fromisoformat(s.created) > now - older_than:
                continue
            out.append(s)
        return out

    def find_prior_by_fingerprint(self, fingerprint: "str | None"):
        if not fingerprint:
            return None
        for s in self.list("prior"):
            if s.complete and self.get("prior", s.id).fingerprints.get("gmm") == fingerprint:
                return s
        return None

    def simulation_for(self, identity: dict):
        from core.SBI.training_checkpoint import identity_digest, peek
        d = self.kind_dir("simulation") / identity_digest(identity)
        return d if peek(d) else None

    # ── loaders ──────────────────────────────────────────────────────────────────────────────────
    def load_prior(self, cfg, ref: str) -> LoadedPrior:
        """Refuses model, ND parameter set/order, box and log-mask mismatches BEFORE reading the
        payload -- the GMM is fit in its box's own coordinate, so a prior is meaningful only against
        the exact (model, parameter set + ORDER, box) it was built for -- then asserts the payload's
        GMM is the one the manifest fingerprinted."""
        import torch
        from core import orchestrator as _orch
        from core.SBI.reparam import nd_log_mask
        from core.SBI.run_guards import _gmm_fingerprint, _log_params_for
        sub, m = self._find("prior", ref)
        if m is None:
            raise StoreError(f"no complete prior named or id'd {ref!r} under {self.kind_dir('prior')}")
        label = m.name or m.id
        gmm = m.body["gmm"]

        def _bad(what, got, want):
            raise Refusal(
                f"Prior '{label}' does not match this configuration: {what} differs.\n"
                f"  prior:  {got}\n  config: {want}\n"
                f"A prior's GMM is fit in its own box coordinate, so loading it here would train the "
                f"flow against a different distribution than the one the samples came from. Build a new "
                f"prior for this bounds file, or pick the prior that belongs to it.", field="prior")

        if m.config.get("model") != cfg.model:
            _bad("the model", m.config.get("model"), cfg.model)
        keys = list(cfg.params_dict)
        if list(gmm["param_keys"]) != keys:
            _bad("the ND parameter set or ORDER", list(gmm["param_keys"]), keys)
        want_lo = torch.tensor([b[0] for _, b in cfg.params_dict.values()], dtype=torch.float64)
        want_hi = torch.tensor([b[1] for _, b in cfg.params_dict.values()], dtype=torch.float64)
        got_lo = torch.tensor(gmm["box"]["nd_lows"], dtype=torch.float64)
        got_hi = torch.tensor(gmm["box"]["nd_highs"], dtype=torch.float64)
        if not (torch.allclose(got_lo, want_lo) and torch.allclose(got_hi, want_hi)):
            diff = [f"{n}: prior ({lo:g}, {hi:g}) vs config ({wl:g}, {wh:g})"
                    for n, lo, hi, wl, wh in zip(keys, got_lo.tolist(), got_hi.tolist(),
                                                 want_lo.tolist(), want_hi.tolist()) if lo != wl or hi != wh]
            _bad("the ND box", "; ".join(diff), "the bounds file in use")
        want_mask = [bool(v) for v in nd_log_mask(cfg, log_params=_log_params_for(cfg)).tolist()]
        if [bool(v) for v in gmm["box"]["log_mask"]] != want_mask:
            _bad("the log-box mask (which ND parameters use a geometric box)", gmm["box"]["log_mask"], want_mask)
        nd_prior = file_manager.load_mix_dist(str(sub / "prior.pt"), device=cfg.hw.device)
        fp = _gmm_fingerprint(nd_prior)
        if fp != m.fingerprints.get("gmm"):
            raise StoreError(f"prior '{label}': prior.pt holds GMM {fp}, not the one the manifest records "
                             f"({m.fingerprints.get('gmm')}); the artifact is inconsistent")
        prior = _orch.ProductPrior(distributions=[nd_prior, _orch.build_rescale_prior(cfg)],
                                   dims=[len(cfg.params_dict), len(cfg.rescale_params)])
        return LoadedPrior("prior", m.id, m.name, m, sub, prior=prior,
                           force_prior=_orch.build_forcing_prior(cfg), nd_prior=nd_prior, fingerprint=fp)

    def load_posterior(self, cfg, ref: str, *, accept: "Accept | None" = None) -> LoadedPosterior:
        """Every check the old width-computing mode guard and build_posterior's load branch made, read
        from the manifest instead of a '.rot.pt' sidecar, BEFORE the payload is unpickled; then the
        payload's own view of its mode, the bijection rebuilt from the manifest, its rotation checked
        against the prior pickled inside the posterior, and the amortization gate."""
        import torch
        from sbi.inference import DirectPosterior
        from core import config as _config
        from core.SBI import reparam, truncate
        from core.SBI.run_guards import _assert_chi_config_is_deliberate, _gmm_fingerprint
        from core.SBI.statistics import SUMMARY_WIDTH
        from core.orchestrator import expected_forcing_dim
        accept = accept or Accept()
        sub, m = self._find("posterior", ref)
        if m is None:
            raise StoreError(f"no complete posterior named or id'd {ref!r} under {self.kind_dir('posterior')}")
        label = m.name or m.id
        # config.py is the one party that cannot go stale: a stale cfg agrees with the posterior
        # trained under that same stale cfg (the 2026-08-19 band).
        _assert_chi_config_is_deliberate(cfg)
        body, cond, tr = m.body, m.body["conditioning"], m.body["transform"]

        def _bad(msg):
            raise Refusal(f"Posterior '{label}' {msg}", field="posterior")

        if m.config.get("model") != cfg.model:
            _bad(f"was trained for model {m.config.get('model')}, but this config is for {cfg.model}.")
        want_keys = list(cfg.params_dict) + list(cfg.rescale_params)
        if list(tr["param_keys"]) != want_keys:
            _bad(f"was trained over a different inferred parameter set or ORDER.\n  posterior: {list(tr['param_keys'])}"
                 f"\n  config:    {want_keys}\nColumns bind positionally, so every reported value would refer "
                 f"to the wrong parameter. Pick the bounds file this posterior was trained with.")
        want_lo = [float(b[0]) for _, b in cfg.params_dict.values()] + [float(b[0]) for _, b in cfg.rescale_params.values()]
        want_hi = [float(b[1]) for _, b in cfg.params_dict.values()] + [float(b[1]) for _, b in cfg.rescale_params.values()]
        got_lo, got_hi = list(tr["nd_lows"]) + list(tr["rescale_lows"]), list(tr["nd_highs"]) + list(tr["rescale_highs"])
        if not (torch.allclose(torch.tensor(got_lo), torch.tensor(want_lo)) and
                torch.allclose(torch.tensor(got_hi), torch.tensor(want_hi))):
            diff = [f"{n}: posterior ({lo:g}, {hi:g}) vs config ({wl:g}, {wh:g})"
                    for n, lo, hi, wl, wh in zip(want_keys, got_lo, got_hi, want_lo, want_hi) if lo != wl or hi != wh]
            _bad(f"was trained in a different box than this config declares: {'; '.join(diff)}. The flow "
                 f"decodes every latent sample through its training box, so pick the bounds file it was trained with.")
        want_mode, want_dim = cfg.observation_mode, int(expected_forcing_dim(cfg))
        if body["mode"] != want_mode:
            _bad(f"was trained in {str(body['mode']).upper()} mode, but this config is {want_mode.upper()} mode "
                 f"(the chi toggle and the bounds file's Forcing section are what select the mode).")
        if body["mode"] == "chi":
            if cond["chi_layout"] != _config.CHI_LAYOUT:
                _bad(f"was trained under chi layout {cond['chi_layout']}; this build writes layout "
                     f"{_config.CHI_LAYOUT} (a padded probe set, {_config.CHI_ELEM_W} channels per slot). Retrain.")
            for key, want, what in (("chi_k_pad", int(cfg.chi_k_pad), "probe-slot capacity"),
                                    ("chi_elem_w", int(_config.CHI_ELEM_W), "channels per slot")):
                if int(cond[key]) != want:
                    _bad(f"has {what} {cond[key]}, but this config declares {want}. It is frozen into the "
                         f"trained network's input shape, so retrain or set {key} back to {cond[key]}.")
            if [float(v) for v in cond["chi_freq_bounds"]] != [float(v) for v in cfg.chi_freq_bounds]:
                _bad(f"was trained over chi band {tuple(cond['chi_freq_bounds'])}, but this config declares "
                     f"{tuple(cfg.chi_freq_bounds)}. The band fixes the encoder's frequency normalization.")
            if abs(float(cond["chi_max_cycles"]) - float(cfg.chi_max_cycles)) > 1e-9:
                _bad(f"was trained with a {float(cond['chi_max_cycles']):g}-cycle lock-in ceiling, but this "
                     f"config declares {float(cfg.chi_max_cycles):g}; logcyc is how the encoder weighs a probe.")
        if int(cond["forcing_dim"]) != want_dim or int(cond["width"]) != SUMMARY_WIDTH + 1 + want_dim:
            _bad(f"conditions on {cond['width']} features (forcing/chi block {cond['forcing_dim']}), but this "
                 f"config expects {SUMMARY_WIDTH + 1 + want_dim} (block {want_dim}). Conditioning widths are "
                 f"incompatible.")

        latent = torch.load(str(sub / "posterior.pt"), map_location=cfg.hw.device, weights_only=False)
        if not isinstance(latent, DirectPosterior):
            raise StoreError(f"posterior '{label}': posterior.pt is a {type(latent).__name__}, not a DirectPosterior")
        latent.device = latent._device = cfg.hw.device     # sbi caches the training device in both
        try:                                                # the trained net's own view must agree
            net_mode, net_dim, _ = reparam.posterior_mode(latent, None)
        except ValueError:
            net_mode = net_dim = None                       # a payload with no estimator (a stub): the manifest rules
        if net_mode is not None and (net_mode != body["mode"] or int(net_dim) != int(cond["forcing_dim"])):
            raise StoreError(f"posterior '{label}': the trained network is {net_mode} with a {net_dim}-wide block "
                             f"but the manifest says {body['mode']} / {cond['forcing_dim']}; the artifact is inconsistent")
        T = reparam.build_eval_bijection(cfg, tr)
        reparam.assert_rotation_consistent(T, getattr(latent, "prior", None), name=label)
        region = digest = None
        if not body["amortized"]:
            trd = body["truncation"] or {}
            if not accept.truncated:
                # V3: neutral. Accept(truncated=True) is the core API's own name and stays; the dialog
                # and the flag that answer this live in the front-end tables under
                # field="accept_truncated", so a renamed control cannot go stale here.
                raise StoreError(
                    f"Posterior '{label}' is NOT AMORTIZED: it was trained by TSNPE on a prior truncated to a "
                    f"{trd.get('level', '?')}-HPD region along Fisher direction(s) {trd.get('dims')}, drawn around "
                    f"the observation with digest {trd.get('x_obs_digest')}. It is only valid for observations in "
                    f"that region -- outside it the flow has never seen a training row and will extrapolate "
                    f"confidently rather than return the prior. To load it anyway, pass Accept(truncated=True); "
                    f"the front ends offer their own consent. Its region then restricts calibration, and "
                    f"inference refuses any other observation unless told to accept it.",
                    field="accept_truncated")
            region = truncate.TruncationRegion.from_dict(mf.region_from_json(trd)) if trd else None
            # The digest is refused alongside the basis and for the same class of reason: without a
            # probe the coordinate the box refers to cannot be verified, and without a digest the
            # region does not say which observation it was drawn around -- so GUARDRAIL 2 (the G2
            # refusal in infer_and_visualize) could never fire and the artifact would serve any
            # observation as if it were the one it is valid near.
            if region is None:
                _bad("declares itself NON-AMORTIZED but its manifest carries no truncation region, so "
                     "the coordinate its box refers to cannot be verified. Run the round again.")
            if region.probe is None:
                _bad("declares itself NON-AMORTIZED but its region carries no basis, so the coordinate "
                     "its box refers to cannot be verified. Run the round again.")
            if region.x_obs_digest is None:
                _bad("declares itself NON-AMORTIZED but its region does not name the observation it was "
                     "drawn around, so the one observation it is valid near cannot be verified and no "
                     "inference could ever be refused for being somewhere else. Run the round again.")
            region.check_basis(T, dim=len(want_keys), device=cfg.hw.device)
            digest = trd.get("x_obs_digest")
        post = reparam.TransformedPosterior(latent, T, truncation=region, x_obs_digest=digest)
        return LoadedPosterior("posterior", m.id, m.name, m, sub, posterior=post, latent=latent,
                               fingerprint=_gmm_fingerprint(getattr(latent, "prior", None)),
                               diagnostics=None, accepted=accept.used() if region is not None else [])

    def load_observation(self, cfg, ref: str) -> LoadedObservation:
        """Refuses an observation whose model, parameter order, mode, conditioning width or chi
        layout/pad is not this config's; asserts the payload's digest is the manifest's, and that the
        payload's own width is the one the manifest declares. Casts the payload to the SESSION's
        dtype -- the artifact is written in whatever dtype the run that made it used, and a float32
        row meeting a float64 network is a hard RuntimeError, not a promotion -- but leaves it on the
        CPU, where conditioning rows live by contract (statistics.conditioning_rows): the PPC and the
        overlay ranking compare it against simulated rows assembled there, and the one consumer that
        needs it on cfg.hw.device, the flow's sample(), moves its own copy. Handing it back on the
        device put a CUDA row against CPU rows in the PPC (the 2026-09-11 GPU gate, invisible to every
        CPU suite; tests/test_gpu_paths.py pins it)."""
        import torch
        from core import config as _config
        from core.SBI.statistics import SUMMARY_WIDTH
        from core.orchestrator import expected_forcing_dim
        sub, m = self._find("observation", ref)
        if m is None:
            raise StoreError(f"no complete observation named or id'd {ref!r} under {self.kind_dir('observation')}")
        label = m.name or m.id
        body, cond = m.body, m.body["conditioning"]
        want_keys = list(cfg.params_dict) + list(cfg.rescale_params)

        def _bad(msg):
            raise Refusal(f"Observation '{label}' {msg}", field="observation")

        if m.config.get("model") != cfg.model:
            _bad(f"was recorded for model {m.config.get('model')}, not {cfg.model}.")
        if list(m.config.get("param_keys") or []) != want_keys:
            _bad(f"was recorded over parameters {m.config.get('param_keys')}, not this config's {want_keys}.")
        if body["mode"] != cfg.observation_mode:
            _bad(f"is a {str(body['mode']).upper()}-mode observation, but this config is "
                 f"{cfg.observation_mode.upper()} mode.")
        want_dim = int(expected_forcing_dim(cfg))
        if int(cond["forcing_dim"]) != want_dim or int(cond["width"]) != SUMMARY_WIDTH + 1 + want_dim:
            _bad(f"is {cond['width']} wide (block {cond['forcing_dim']}); this config conditions on "
                 f"{SUMMARY_WIDTH + 1 + want_dim} (block {want_dim}).")
        if body["mode"] == "chi" and (cond["chi_layout"] != _config.CHI_LAYOUT
                                      or int(cond["chi_k_pad"]) != int(cfg.chi_k_pad)):
            _bad(f"was packed under chi layout {cond['chi_layout']} with {cond['chi_k_pad']} slots; this "
                 f"config is layout {_config.CHI_LAYOUT} / {cfg.chi_k_pad}.")
        payload = torch.load(str(sub / "observation.pt"), map_location="cpu", weights_only=False)
        x_obs = payload["x_obs"]
        if mf.tensor_digest(x_obs) != body["x_obs_digest"]:
            raise StoreError(f"observation '{label}': observation.pt does not hash to the manifest's digest; "
                             f"the artifact is inconsistent")
        if int(x_obs.shape[-1]) != int(cond["width"]):
            # field= like the width Refusal twelve lines up, and for the same reason: whichever of the
            # two fires, the operator answers it by picking another observation, and each front end's
            # table is what names that control (piece 4, §8.1).
            raise StoreError(f"observation '{label}': observation.pt holds a {int(x_obs.shape[-1])}-wide "
                             f"conditioning row but its manifest declares {int(cond['width'])}; the artifact "
                             f"is inconsistent (every width guard above compared the MANIFEST, not this row)",
                             field="observation")
        # The digest is over the float64 bytes, so it is computed on the payload as written and the
        # cast below cannot change it. dtype only -- see the docstring for why the row stays on the CPU.
        x_obs = x_obs.to(dtype=cfg.hw.dtype)
        return LoadedObservation("observation", m.id, m.name, m, sub, x_obs=x_obs,
                                 obs_data=payload["obs_data"].to(cfg.hw.dtype),
                                 t_dim=payload["t_dim"].to(cfg.hw.dtype), digest=body["x_obs_digest"],
                                 mode=body["mode"], width=int(cond["width"]))

    def load_calibration(self, ref: str) -> LoadedCalibration:
        sub, m = self._find("calibration", ref)
        if m is None:
            raise StoreError(f"no complete calibration named or id'd {ref!r}")
        # From the manifest body, like load_inference: the manifest is the authoritative description
        # of the artifact, and results.json is only the human-readable copy (never read back here).
        # to_json_text writes the manifest with sort_keys=True (so the file diffs deterministically),
        # which -- being recursive -- would alphabetize a dict keyed by parameter name on every load
        # and silently decouple it from cfg.params_dict's order; sbc.per_param is therefore a list of
        # {name, ks_p, c2st_ranks, c2st_dap} records, not a dict, so order survives the round trip.
        return LoadedCalibration("calibration", m.id, m.name, m, sub, results=dict(m.body["results"]))

    def load_inference(self, ref: str) -> LoadedInference:
        sub, m = self._find("inference", ref)
        if m is None:
            raise StoreError(f"no complete inference named or id'd {ref!r}")
        return LoadedInference("inference", m.id, m.name, m, sub, results=dict(m.body["results"]),
                               samples_path=sub / "samples.pt")

    def load_diagnostic(self, ref: str) -> LoadedDiagnostic:
        """A diagnostic is a MEASUREMENT about other artifacts, so this verifies nothing and refuses
        nothing (D5): there is no configuration it has to match, and nothing is ever trained from it.
        The one failure is a ref that names no complete diagnostic.

        From the manifest body, like load_calibration and load_inference: the manifest is the
        authoritative description of the artifact, and any results file beside it is a
        human-readable copy. Per-parameter records are LISTS of {"name": ...} entries for the same
        reason they are there -- to_json_text sorts keys recursively, which would alphabetize a dict
        keyed by parameter name and silently decouple it from cfg.params_dict's order.
        """
        sub, m = self._find("diagnostic", ref)
        if m is None:
            raise StoreError(f"no complete diagnostic named or id'd {ref!r}")
        body = m.body
        return LoadedDiagnostic("diagnostic", m.id, m.name, m, sub, diagnostic=body["diagnostic"],
                                variant=body["variant"], settings=dict(body["settings"]),
                                results=dict(body["results"]))

    def load_fdt(self, ref: str) -> LoadedFdt:
        """A measurement of a CELL, read back: its body and the path to its numbers.

        Like ``load_diagnostic``, this constrains nothing (D5): an fdt record describes an
        experiment that has already happened, there is no configuration it has to match, and nothing
        is ever trained from it. The ONE thing it verifies is every payload's own sha256, because
        that is the one claim the manifest makes about a file this call is about to hand out. A
        recorded hash whose file is GONE fails that claim as surely as one that no longer matches
        (controller ruling F29): skipping it would hand back ``data_path=None`` and pass a finished
        record off as one that never wrote numbers.

        A payload whose recorded hash is NULL is not verified, present or not. A progressive record
        hashes at the final commit only (spec §2.2), so every unfinished record lists its payloads
        unhashed -- and reading what an interrupted run did manage to write is exactly what E2 keeps
        the folder for (P47).
        """
        sub, m = self._find("fdt", ref)
        if m is None:
            raise StoreError(f"no complete fdt artifact named or id'd {ref!r} under "
                             f"{self.kind_dir('fdt')}", field="artifact")
        for name, want in sorted(m.payloads.items()):
            if want is None:
                continue
            p = sub / name
            if not p.is_file():
                raise StoreError(f"fdt {m.name or m.id}: {name} is missing, but its manifest records "
                                 f"its sha256 {want}; the artifact is inconsistent", field="artifact")
            got = prov.sha256_file(p)
            if got != want:
                raise StoreError(f"fdt {m.name or m.id}: {name} hashes to {got}, not the {want} its "
                                 f"manifest records; the artifact is inconsistent", field="artifact")
        data = sub / "data.h5"
        return LoadedFdt("fdt", m.id, m.name, m, sub, body=dict(m.body),
                         data_path=data if data.is_file() else None)


def write_simulation_manifest(path, identity: dict, *, parents=None, inputs=None, hw=None,
                              batches_done: int = 0, complete: bool = False, rows=None, V=None) -> mf.Manifest:
    """The simulation cache's manifest, written by training_checkpoint at create and refreshed on every
    save and at completion. Keeps id, created, note, prism, env, inputs and parents from an existing
    manifest (a resume must not rewrite who started the run); ``wall_seconds`` is the span from
    ``created`` to completion, across resumes."""
    from core.SBI.training_checkpoint import identity_digest
    path = Path(path)
    digest = identity_digest(identity)
    old = None
    mpath = path / MANIFEST
    if mpath.exists():
        with contextlib.suppress(mf.ManifestError, OSError):
            old = mf.from_json_text(mpath.read_text(encoding="utf-8"))
    created = old.created if old else _utc_now().isoformat()
    if complete:
        wall = (_utc_now() - datetime.fromisoformat(created)).total_seconds()
    else:
        wall = old.body.get("wall_seconds") if old else None
    v_digest = mf.tensor_digest(V) if V is not None else (old.fingerprints.get("V") if old else None)
    d = dict(
        schema=mf.SCHEMA, kind="simulation", id=digest, name="", created=created,
        note=old.note if old else "",
        prism=old.prism if old else prov.git_info(config.REPO_ROOT),
        env=old.env if old else prov.env_info(hw if hw is not None else config.cpu_device()),
        inputs=old.inputs if old else (inputs or {"bounds": None, "cell": None, "units": None,
                                                    "model": identity.get("model")}),
        config=dict(identity), parents=old.parents if old else dict(parents or {}),
        fingerprints={"gmm": identity.get("prior_fingerprint"), "V": v_digest},
        payloads={}, figures=[],
        body={"identity": dict(identity), "digest": digest, "batches_done": int(batches_done),
              "complete": bool(complete), "rows": None if rows is None else [int(r) for r in rows],
              "V_digest": v_digest, "wall_seconds": wall},
    )
    m = mf.validate(d)
    path.mkdir(parents=True, exist_ok=True)
    _write_manifest(path, m)
    return m


# ── the process default ──────────────────────────────────────────────────────────────────────────
_DEFAULT: "ArtifactStore | None" = None


def default_store() -> ArtifactStore:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = ArtifactStore(config.artifacts_root())
    return _DEFAULT


def set_default_store(store: "ArtifactStore | None") -> None:
    global _DEFAULT
    _DEFAULT = store


@contextlib.contextmanager
def use_store(store: ArtifactStore):
    global _DEFAULT
    prev = _DEFAULT
    _DEFAULT = store
    try:
        yield store
    finally:
        _DEFAULT = prev


def resolve_store(store) -> ArtifactStore:
    """``store`` when given, else the process default -- the one line every stage function starts with."""
    return store if store is not None else default_store()
