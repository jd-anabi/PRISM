"""``python -m core artifacts <mode>``: read, annotate and tidy the artifact store.

The command-line twin of the Artifacts screen (piece 4, B9; design §4). ONE subcommand with a
positional mode -- the ``identifiability {rotation,laplace,jacobian}`` precedent
(core/tool/diagnostics.py:60-105), where each mode is a parser of its own, so a flag that means
nothing to a mode is an argparse error rather than a setting that is silently ignored.

WHY THIS MODULE IS NOT ``artifacts.py``. ``core/tool/__init__.py`` does ``from . import ...``, so a
sibling of that name would sit one mistaken relative import away from shadowing the
``core.artifacts`` package this module is a front end for.

NO CONFIGURATION FLAGS. ``add_config_flags`` is deliberately NOT called (design §4.1): a listing must
not be able to fail on a bounds file it does not need, and ``python -m core artifacts --help`` must
stay torch-free. The consequence is stated in EPILOG: this family cannot say "this posterior does not
match your bounds file" -- that is what LOADING it does. ``fdt``/``crossval`` are the precedent for a
subcommand that builds no SimConfig.

NO ``--store-root``, EITHER. The root is ``config.artifacts_root()`` (PRISM_ARTIFACTS), like every
subcommand but ``smoke``, and ``main`` has already opened the store by the time a handler runs. The
name is not reused because ``main`` keys three of smoke's behaviours on the FLAG's presence:
``has_store_root = hasattr(args, "store_root")`` (:95) decides whether a fresh ``mkdtemp`` root is
created (:96-98), whether an empty auto-created root is removed afterwards (:156-157), and which
Ctrl-C advice is printed (:121-122). Declaring ``--store-root`` here would change all three for this
family, for a root it does not want.

OUTPUT IS ``print``. The tool's framing prints are deliberately outside the no-print pin -- "The
tool's framing prints (core/tool) stay prints and are deliberately outside this set"
(tests/test_tool.py:1701) -- and a browse command's output IS framing: it is the result a script
reads, not the pipeline's own voice, and it must never reach an artifact's log.txt.

HEAVY IMPORTS INSIDE THE HANDLERS, per core/tool/stages.py's rule, so building the parser costs no
torch import.

AND NOTHING FROM ``core/gui``, EVER -- not even the browser's ``columns_for``, however tempting it is
to read the column titles from the one place that owns them: importing it would pull PySide6 into
``python -m core --help``. This module therefore keeps its own ``COLUMNS``, and a test in
tests/test_tool.py imports BOTH front ends and pins the two column sets against each other.
"""
import argparse
import sys
from pathlib import Path

from core.refusals import NOTE_MAX_CHARS

# The seven kinds in KIND_DIRS order, restated as a literal: reading core.artifacts.store.KIND_DIRS
# at parser-build time would import torch (core/artifacts/__init__.py imports .store, which imports
# core.config). tests/test_tool.py pins the two in sync, both order and membership -- the same shape
# as --preset's hard-coded choices pinned against cli.SWEEP_PRESETS.
KINDS = ("prior", "simulation", "posterior", "observation", "calibration", "inference", "diagnostic")

# The columns each kind shows, exactly design §3.2's table: the browser's own tuples
# (core/gui/widgets/artifact_table.columns_for), LOWER-CASED (P11). A SECOND literal on purpose --
# that module is a Qt module, and importing it here would pull PySide6 into `python -m core --help`.
# So this module imports NOTHING from core/gui, and the two column sets are kept in step by a test
# that imports both and lower-cases the GUI's (tests/test_tool.py).
COLUMNS = {
    "prior": ("name", "created", "note"),
    "simulation": ("name", "created", "progress", "finished", "note"),
    "posterior": ("name", "created", "mode", "width", "amortized", "note"),
    "observation": ("name", "created", "mode", "width", "note"),
    "calibration": ("name", "created", "note"),
    "inference": ("name", "created", "note"),
    "diagnostic": ("name", "created", "variant", "note"),
}

# The tail of log.txt ``show`` prints. A DISPLAY cap, not a knob: it bounds what one screenful of
# records can cost. The browser's detail pane has its own cap (artifact_screen.LOG_MAX_BYTES),
# chosen independently and today at the same 1 MiB; nothing pins the two equal, because each front
# end bounds its own surface -- a terminal and a text box are not the same screenful.
LOG_TAIL_BYTES = 1 << 20

# The one dependent that names nothing (fix round 1, IMPORTANT 1). A SECOND literal on purpose, the
# same shape as COLUMNS above: core/gui/screens/artifact_screen.py's own _FINGERPRINT_DEPENDENT,
# restated here because this module imports NOTHING from core/gui. A cache's directory is keyed on
# the prior's GMM, so ArtifactStore.dependents reports it whether or not the manifest records the
# parent link, and a bare id gives an operator no way to tell that one from a child that named it.
_FINGERPRINT_DEPENDENT = ("a training cache was generated against this prior and its rows are "
                          "meaningless without it")

EPILOG = """\
Reads PRISM_ARTIFACTS (the artifacts root, default <repo>/Artifacts) -- the same root as every
subcommand but `smoke`. There is no --store-root, and there are no configuration flags: a listing
must not be able to fail on a bounds file it does not need, and --help here costs no torch import.
The consequence is that this family cannot tell you whether a posterior matches your bounds file --
that is what LOADING it does (`python -m core validate --posterior ...`).

  list [<kind>]         one line per artifact; with no kind, all seven under headings
  show <kind> <ref>     the manifest, then the records of the run that wrote it
  note <kind> <ref> --note TEXT
                        set the note: one line, at most 200 characters; '' clears it
  rm <kind> <ref>       delete one artifact. There is no --force: an artifact anything depends on
                        cannot be deleted at all, so delete its children first
  sweep [<kind>]        remove every directory with no usable manifest; with no kind, all seven
  summary <kind> <ref> [--out PATH]
                        the lineage report: this artifact, then its parents, oldest last

<kind> is one of prior, simulation, posterior, observation, calibration, inference, diagnostic.
<ref> is an artifact's name or its id. An empty listing exits 0: a script must be able to tell
"nothing on disk" from "you asked for something wrong".
"""


def _flat(text) -> str:
    """One line, whitespace collapsed: a cell in an aligned table cannot carry a newline or a tab."""
    return " ".join(str(text or "").split())


def _width(s) -> str:
    return "?" if s.width is None else str(s.width)


def _progress(s) -> str:
    """"3/4 batches" while a cache runs, "4/4 batches, 96 rows" once it has finished -- from
    ``Summary`` ALONE (B2: a row needs no second manifest read). The total is ``batches_planned``, off
    the body's ``identity["n_runs"]`` (spec §12 row 2), so the cell is a FRACTION. The row count is
    written only by ``mark_complete`` -- ``save`` passes none -- so a cache mid-run has no rows-so-far
    to show and the cell stops at the batches. The separate ``finished`` column still answers whether
    the batches are all there (B3): a manifest marked complete is what makes 4/4 mean finished.
    A comma and plain ASCII, not the browser's middle dot: a script may read this line."""
    done = "?" if s.batches_done is None else str(s.batches_done)
    planned = "?" if s.batches_planned is None else str(s.batches_planned)
    text = f"{done}/{planned} batches"
    if s.rows:
        text += f", {sum(int(r) for r in s.rows)} rows"
    return text


def _cells(kind: str, s) -> tuple:
    """One complete row's cells, in ``COLUMNS[kind]`` order. ``label`` is the name, or
    ``(unnamed <id>)`` -- as the pickers show it, and as a simulation cache always reads, because
    write_simulation_manifest always writes ``name=""``."""
    head = (_flat(s.label), _flat(s.created))
    if kind == "simulation":
        return head + (_progress(s), "yes" if s.finished else "no", _flat(s.note))
    if kind == "posterior":
        # The exception is what is worth a word: amortized is the norm (B13's rule for the pickers,
        # applied to the table's own cell).
        amortized = "amortized" if s.amortized else ("narrowed (TSNPE)" if s.amortized is False
                                                     else "?")
        return head + (_flat(s.mode), _width(s), amortized, _flat(s.note))
    if kind == "observation":
        return head + (_flat(s.mode), _width(s), _flat(s.note))
    if kind == "diagnostic":
        # A diagnostic records its kind under ``variant``, never ``mode``: the mode column would
        # otherwise report a conditioning geometry that does not exist (design §1.3).
        return head + (_flat(s.variant) or "-", _flat(s.note))
    return head + (_flat(s.note),)


def _print_kind(kind: str, rows) -> None:
    """One kind's heading, its aligned table, and its unusable directories last."""
    print(f"\n== {kind} ==")
    good = [s for s in rows if s.complete]
    broken = [s for s in rows if not s.complete]
    if not good:
        print("  nothing here yet" if not broken else "  no artifact with a usable manifest here")
    else:
        columns = COLUMNS[kind]
        cells = [_cells(kind, s) for s in good]
        widths = [max([len(columns[i])] + [len(c[i]) for c in cells]) for i in range(len(columns))]
        for row in (columns, *cells):
            print("  " + "  ".join(v.ljust(w) for v, w in zip(row, widths)).rstrip())
    for s in broken:
        # LAST, with the reason in place of the kind's own columns, and identified by the directory
        # name -- the handle ``sweep`` takes (B7). ``list`` already sorts them last; this split is
        # what keeps the table above aligned on real rows only.
        print(f"  incomplete  {_flat(s.dir_name)}  -- {_flat(s.reason)}")


def _list(args, store) -> int:
    """``list [<kind>]``. An empty listing is EXIT 0 with a line saying so (§4.3); an unknown kind is
    the store's own refusal, through the ladder's ``refused:`` rung at exit 1. With no kind, all seven
    are walked and a kind with nothing in it is NAMED in a trailing line rather than silently
    skipped, so "I did not look" and "there is nothing" stay distinguishable."""
    kinds = [args.kind] if args.kind else list(KINDS)
    empty = []
    for kind in kinds:
        rows = store.list(kind)                  # an unknown kind: StoreError -> the ladder's exit 1
        if not rows and not args.kind:
            empty.append(kind)
            continue
        _print_kind(kind, rows)
    if empty:
        print(f"\nnothing in: {', '.join(empty)}")
        if len(empty) == len(kinds):
            print(f"the store under {store.root} holds no artifacts yet.")
    return 0


def _show(args, store) -> int:
    """``show <kind> <ref>``: the manifest as design §3.3 renders it, then the run's records, then the
    same honest gaps the browser's detail pane states. A ref that names no complete artifact is the
    store's own refusal (exit 1); its message already quotes the ref, which is why
    core/tool/fields.py maps the ``artifact`` key to None -- the tool names the artifact
    positionally, so there is no option string to print."""
    from core.artifacts import render_manifest
    m = store.get(args.kind, args.ref)
    print(render_manifest(m))
    sub = store.path(args.kind, args.ref)
    if sub.name != m.dir_name:
        # A rename whose directory move was refused leaves exactly this, and no front end has ever
        # shown it (design §3.3). The manifest is what resolves an artifact, so nothing is broken.
        print(f"\nthe directory is named {sub.name!r} while its manifest says {m.dir_name!r}: a "
              f"rename whose directory move was refused leaves that. The manifest is what resolves "
              f"the artifact, so nothing here is broken.")
    print("\n== records ==")
    text, truncated = store.read_log(args.kind, args.ref, max_bytes=LOG_TAIL_BYTES)
    if text is None:
        # TWO honest gaps behind one answer: read_log says (None, False) for both, so the KIND is
        # what tells them apart. Both sentences are the browser's OWN CONSTANTS, word for word
        # (core/gui/screens/artifact_screen.py's _CACHE_NO_LOG and _NO_RUN_LOG) -- one operator
        # reads both front ends.
        print("  a training cache keeps no log: it is written batch by batch across resumes and "
              "shared by every posterior that names it"
              if args.kind == "simulation" else
              "  written outside a run")
        return 0
    if truncated:
        print(f"  (only the last {LOG_TAIL_BYTES} bytes are shown; the log is longer)")
    if not text:
        print("  the run wrote no records. Silence is a record too: the file is there and empty.")
    else:
        print(text, end="" if text.endswith("\n") else "\n")
    return 0


def _leftover_hint(store, kind: str, ref: str) -> str:
    """``""`` ordinarily, or a suffix naming ``sweep`` when ``ref`` is exactly the ``dir_name`` of
    one of ``kind``'s own incomplete directories (fix round 1, IMPORTANT 3).

    ``note`` and ``rm`` resolve through ``_find``, which never returns a manifest-less entry, so a
    ref copied straight off ``list``'s own "incomplete ..." row cannot resolve there -- and the
    store's "no complete artifact" refusal says nothing about why, or what removes it. ``show``
    already tells this story (the honest-gaps sentences); this closes the same gap for the two modes
    that write.

    A read failure here is swallowed: an unrelated problem listing the kind is not this ref's
    business, and the plain refusal -- itself still correct -- is left to stand."""
    try:
        rows = store.list(kind)
    except Exception:                              # noqa: BLE001 -- the hint is a courtesy, not a check
        return ""
    if any(not s.complete and s.dir_name == ref for s in rows):
        return (f" {ref!r} is one of the leftovers `sweep` removes, not an artifact with a note or "
                f"a delete of its own; run `python -m core artifacts sweep {kind}` to clear it.")
    return ""


def _rm_refusal(store, kind: str, m, deps) -> "Refusal":
    """The reworded dependents refusal (fix round 1, IMPORTANT 1): the store's own sentence ends
    "pass force=True to orphan them", a step nothing in either front end offers (B6), so its raw
    wording must never reach an operator. Same shape as the window's own reword
    (core/gui/screens/artifact_screen.py's ``_dependents_refusal``, which exists for the WORDING
    alone) -- every dependent, WITH WHY, then what to do -- flattened to this tool's one-line
    convention rather than the window's multi-line box."""
    from core.refusals import Refusal
    parents = {}
    for k in sorted({k for k, _, _ in deps}):
        for row in store.list(k):
            parents[(k, row.id)] = row.parents
    clauses = []
    for dep_kind, dep_id, dep_name in deps:
        held = parents.get((dep_kind, dep_id)) or {}
        # dependents()'s first pass matches parents[<kind key>] == id; its second adds simulation
        # caches by fingerprint alone -- the same distinction the window's reword draws.
        fingerprint_only = (kind == "prior" and dep_kind == "simulation"
                            and held.get("prior") != m.id)
        why = _FINGERPRINT_DEPENDENT if fingerprint_only else f"it names this {kind} as a parent"
        clauses.append(f"{dep_kind} {dep_name or '(unnamed)'} [{dep_id}] -- {why}")
    return Refusal(
        f"refusing to delete {kind} {m.name or m.id} [{m.id}]: {len(deps)} artifact(s) depend on "
        f"it: " + "; ".join(clauses) + ". Delete those first.", field="artifact")


def _note(args, store) -> int:
    """``note <kind> <ref> --note TEXT``. B5's rule runs BEFORE the store is touched, so a bad note
    rewrites no manifest; ``""`` clears the note, which is why ``--note`` is required rather than
    defaulted -- a note must never be cleared by leaving a flag off.

    Two refusals, two field keys, both from elsewhere: ``require_note``'s (the over-long note, the
    newline) carries field="note", so the ladder prints ``(--note)``, while ``set_note``'s "no
    complete artifact" carries field="artifact", whose entry in core/tool/fields.py is None -- the
    tool names the artifact positionally -- so that line simply ends at the message. When the ref
    names a leftover directory, ``_leftover_hint`` appends the missing next step (IMPORTANT 3)."""
    from core.refusals import Refusal, require_note
    text = require_note("note", args.note)
    try:
        m = store.set_note(args.kind, args.ref, text)
    except Refusal as exc:
        hint = _leftover_hint(store, args.kind, args.ref)
        if hint:
            raise Refusal(exc.message + hint, field=exc.field) from None
        raise
    label = m.name or m.id
    if m.note:
        print(f"[prism] {args.kind} {label}: note = {m.note!r}")
    else:
        print(f"[prism] {args.kind} {label}: note cleared")
    return 0


def _rm(args, store) -> int:
    """``rm <kind> <ref>``. NO ``--force`` (B6): dependents are read BEFORE anything is deleted and
    refused in this tool's own words, never the store's raw "pass force=True to orphan them" (fix
    round 1, IMPORTANT 1 -- no front end offers that). A missing ref that names a leftover directory
    gets ``sweep`` as its next step (IMPORTANT 3). ``delete`` still reads dependents again and stays
    the last word for a race between the two reads -- the window's own precedent."""
    from core.refusals import Refusal
    try:
        m = store.get(args.kind, args.ref)
    except Refusal as exc:                        # a missing ref: StoreError -> the ladder's exit 1
        hint = _leftover_hint(store, args.kind, args.ref)
        if hint:
            raise Refusal(exc.message + hint, field=exc.field) from None
        raise
    sub = store.path(args.kind, args.ref)
    deps = store.dependents(args.kind, m.id)
    if deps:
        raise _rm_refusal(store, args.kind, m, deps)
    store.delete(args.kind, args.ref)
    print(f"[prism] removed {args.kind} {args.ref}: {sub}")
    return 0


def _sweep(args, store) -> int:
    """``sweep [<kind>]``: remove every directory of a kind (or of all seven) with no usable
    manifest. §4.3: nothing to remove is 0; a directory that would not delete is 1, naming each. A
    failure never stops the sweep -- ``sweep_incomplete`` finishes the rest and reports it.

    ``sweep_incomplete``'s own return carries no reason (fix round 1, RULED IN 5): the reasons are
    read off ``list`` BEFORE the removal -- the same rows the table calls incomplete -- and matched
    back by ``(kind, dir_name)`` once the removal is done, so a removed line says WHY without a
    second manifest read. The window reads them the same way, earlier, for its confirmation; there is
    no confirmation here, so this prints the reason AFTER. A kind ``list`` cannot read (an unknown
    kind, a permissions problem) is not fatal here either: ``sweep_incomplete`` is the authority on
    what gets removed and raises its own refusal for a bad kind regardless."""
    reasons = {}
    for kind in ((args.kind,) if args.kind else KINDS):
        try:
            rows = store.list(kind)
        except Exception:                          # noqa: BLE001 -- sweep_incomplete is the authority
            continue
        for row in rows:
            if not row.complete:
                reasons[(kind, row.dir_name)] = row.reason
    removed, failed = store.sweep_incomplete(args.kind)
    for kind, dir_name in removed:
        why = reasons.get((kind, dir_name))
        print(f"[prism] removed {kind} leftover {dir_name}" + (f" -- {why}" if why else ""))
    for kind, dir_name, reason in failed:
        print(f"prism artifacts: could not remove {kind} leftover {dir_name}: {reason}",
              file=sys.stderr)
    if not removed and not failed:
        print("[prism] nothing to sweep: every directory here carries a usable manifest.")
    return 1 if failed else 0


def _summary(args, store) -> int:
    """``summary <kind> <ref> [--out PATH]``: the lineage report -- the artifact, then its parents
    transitively, a parent absent from the store printed as MISSING rather than skipped (design §5).
    A file or stdout, never an eighth store kind (B10).

    newline="\\n" explicitly (P23): write_text's default newline=None translates every "\\n" to
    "\\r\\n" on Windows, and the browser's own Lineage report button writes the same text with
    newline="\\n" -- so without it §5's "the same bytes whichever front end made it" is quietly
    false.

    ``--out`` naming a directory, or a read-only file, is an ``OSError`` the window's identical write
    (``artifact_screen._lineage_report``) catches and reports -- an operator's typo must not escape
    as an unhandled traceback here either (fix round 1, IMPORTANT 2)."""
    from core.artifacts import render_lineage
    from core.refusals import Refusal
    text = render_lineage(store, args.kind, args.ref)
    if args.out:
        try:
            Path(args.out).write_text(text, encoding="utf-8", newline="\n")
        except OSError as e:
            raise Refusal(f"could not write the lineage report to {args.out!r}: "
                          f"{type(e).__name__}: {e}") from e
        print(f"[prism] lineage report: {args.out}")
    else:
        print(text, end="" if text.endswith("\n") else "\n")
    return 0


_KIND = "one of " + ", ".join(KINDS)
_REF = "the artifact's name, or its id"


def register(subparsers) -> dict:
    """``{name: subparser}`` -- the contract ``build_parser``'s ``p.subcommands.update(...)`` loop
    needs. One subcommand, six modes; Task 14 adds four of them."""
    p = subparsers.add_parser(
        "artifacts", help="read, annotate and tidy the artifact store (loads nothing, simulates "
                          "nothing)",
        epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    modes = p.add_subparsers(dest="mode", required=True,
                             metavar="{list,show,note,rm,sweep,summary}")

    ls = modes.add_parser("list", help="one line per artifact of a kind, or of all seven")
    ls.add_argument("kind", nargs="?", default=None, metavar="<kind>", help=_KIND)
    ls.set_defaults(handler=_list)

    show = modes.add_parser("show", help="one artifact's manifest and the records of the run that "
                                         "wrote it")
    show.add_argument("kind", metavar="<kind>", help=_KIND)
    show.add_argument("ref", metavar="<ref>", help=_REF)
    show.set_defaults(handler=_show)

    note = modes.add_parser("note", help="set or clear one artifact's note")
    note.add_argument("kind", metavar="<kind>", help=_KIND)
    note.add_argument("ref", metavar="<ref>", help=_REF)
    note.add_argument("--note", required=True, metavar="TEXT",
                      help=f"one line, at most {NOTE_MAX_CHARS} characters; '' clears it. A note "
                           f"with a newline, or a longer one, is refused rather than trimmed to fit")
    note.set_defaults(handler=_note)

    rm = modes.add_parser("rm", help="delete one artifact; refused when anything depends on it")
    rm.add_argument("kind", metavar="<kind>", help=_KIND)
    rm.add_argument("ref", metavar="<ref>", help=_REF)
    rm.set_defaults(handler=_rm)

    sweep = modes.add_parser("sweep", help="remove every directory with no usable manifest")
    sweep.add_argument("kind", nargs="?", default=None, metavar="<kind>",
                       help=f"{_KIND}; with no kind, all seven")
    sweep.set_defaults(handler=_sweep)

    summary = modes.add_parser("summary", help="the lineage report: this artifact, then its parents")
    summary.add_argument("kind", metavar="<kind>", help=_KIND)
    summary.add_argument("ref", metavar="<ref>", help=_REF)
    summary.add_argument("--out", default=None, metavar="PATH",
                         help="write the report here instead of to stdout")
    summary.set_defaults(handler=_summary)
    return {"artifacts": p}
