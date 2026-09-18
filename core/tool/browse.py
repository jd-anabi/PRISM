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

EPILOG = """\
Reads PRISM_ARTIFACTS (the artifacts root, default <repo>/Artifacts) -- the same root as every
subcommand but `smoke`. There is no --store-root, and there are no configuration flags: a listing
must not be able to fail on a bounds file it does not need, and --help here costs no torch import.
The consequence is that this family cannot tell you whether a posterior matches your bounds file --
that is what LOADING it does (`python -m core validate --posterior ...`).

  list [<kind>]         one line per artifact; with no kind, all seven under headings
  show <kind> <ref>     the manifest, then the records of the run that wrote it

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
        # what tells them apart. Both sentences are the browser's, word for word (design §3.3) --
        # one operator reads both front ends.
        print("  a training cache keeps no log: it is written batch by batch across resumes and "
              "shared by every posterior that names it."
              if args.kind == "simulation" else
              "  written outside a run: nothing captured this artifact's records.")
        return 0
    if truncated:
        print(f"  (only the last {LOG_TAIL_BYTES} bytes are shown; the log is longer)")
    if not text:
        print("  the run wrote no records. Silence is a record too: the file is there and empty.")
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
    modes = p.add_subparsers(dest="mode", required=True, metavar="{list,show}")

    ls = modes.add_parser("list", help="one line per artifact of a kind, or of all seven")
    ls.add_argument("kind", nargs="?", default=None, metavar="<kind>", help=_KIND)
    ls.set_defaults(handler=_list)

    show = modes.add_parser("show", help="one artifact's manifest and the records of the run that "
                                         "wrote it")
    show.add_argument("kind", metavar="<kind>", help=_KIND)
    show.add_argument("ref", metavar="<ref>", help=_REF)
    show.set_defaults(handler=_show)
    return {"artifacts": p}
