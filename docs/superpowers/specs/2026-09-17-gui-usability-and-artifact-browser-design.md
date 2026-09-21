# PRISM piece 4: GUI usability and the artifact browser (design)

**Status:** approved 2026-09-17 (brainstormed with the owner the same day).
**Piece:** 4 of the pre-retrain hardening programme
(`docs/superpowers/specs/2026-09-10-pre-retrain-hardening-decomposition.md` §4), which runs in
parallel with piece 5 and after piece 3.
**Predecessors it stands on:** piece 1's artifact store (`core/artifacts/`), piece 2's one stage API
and command-line tool (`core/tool/`), piece 3's refusals, logging and private copy
(`core/refusals.py`, `core/runs.py`).

## 1. Purpose and scope

The store has been complete since piece 1 and almost none of it is reachable. `list`, `get`, `path`,
`rename`, `set_note`, `dependents`, `delete` and `unnamed` all exist and are tested; outside
`core/artifacts/store.py` the only production callers are the two Save buttons' `rename` calls and
`StorePicker.refresh`'s `list`. `set_note`, `delete`, `dependents` and `unnamed` have no caller at
all but the suites. Since piece 3 every committed artifact also carries a `log.txt` of its run's
records, and nothing reads it. Four kinds — the simulation cache, calibrations, inferences and
diagnostics — have no front-end surface whatever.

Piece 4 is the front end over that engine, plus the four usability items the decomposition's piece-4
row named and the four defects piece 3 handed on.

The decomposition's row reads:

> list/inspect/delete artifacts with their manifests; picker shows mode, width,
> amortized/truncated; the two-entry-point trap (M1b) resolved; one-task-at-a-time made visible;
> the never-seen list walked on a display with the checklist; a saveable run summary

Two of those six are already answered and are re-stated here so nothing is built twice:

- **"Picker shows mode, width, amortized/truncated"** landed in piece 1 (`86328e4`), as the
  `StorePicker` per-item **tooltip**. The closed combo still shows only `Summary.label`, and no test
  pins any of it. §6.3 finishes the job.
- **"The never-seen list walked on a display"** was run by the owner on 2026-09-11 at `df7491e`:
  rows 1–20 of `docs/checklists/display-walkthrough.md` all pass. Piece 4 re-checks only what it
  changes, plus row 1 (§11).

"A saveable run summary" and "one-task-at-a-time made visible" had dropped out of `docs/STATE.md`
altogether, and "the two-entry-point trap resolved" with them. The owner put all three back in scope
on 2026-09-17. This is therefore the largest piece since the store itself.

### 1.1 Decisions (binding)

| # | decision |
|---|---|
| **B1** | The browser is a **fifth Home tile**, "Artifacts", a peer of the four sections — not a stage tab and **not a `BasePanel`**. A `BasePanel` enrols in `BasePanel._instances`, so every run anywhere would grey the browser's controls and you could not read a log while training. It is a plain screen with its own status line (`ModelBuilderScreen`'s pattern) and is added **by name** to `MainWindow._save_state`, because only panel-typed screens get that sweep for free. |
| **B2** | All **seven** kinds are listed, one kind at a time, in a real sortable table. `Summary` grows so that a row needs no second manifest read. |
| **B3** | `Summary.complete` keeps its meaning — "has a valid manifest". A separate, explicit **`finished`** answers "did the run finish", and a cache's row carries its own progress (`batches_done`, `batches_planned`, `rows`). No caller may confuse the two again. **Amended at planning** (§12 row 2): `batches_planned` was added, because the planned total lives only in the body's identity, and `rows` is written **only** at completion — `save` passes none — so an unfinished cache has no rows-so-far to show. |
| **B4** | The detail view is **text only**: the manifest rendered, and the run's `log.txt`. No figure rendering, no open-the-folder button, no "use this" jump into a stage tab. |
| **B5** | A note is **one line**, trimmed, at most **200 characters**; blank clears it. A note containing a newline or over the limit is **refused** (V2: no clamp, no silent default). |
| **B6** | Delete is **one artifact at a time**. **No `force=True` in either front end** — so an artifact anything depends on cannot be deleted at all, and the browser therefore **refuses** it, naming every dependent (including a training cache that holds a prior only by fingerprint, with no recorded parent link), rather than offering a confirmation that could only fail. With no dependents it asks for confirmation, and an unfinished cache's confirmation names its **committed batches** (§12 row 2: rows-so-far do not exist on disk mid-run, and writing them would mean touching the checkpoint commit protocol for a display nicety). Note, delete and sweep are all **refused while any run is live**; reading never is. |
| **B7** | **Sweep** removes every incomplete directory of a kind in one action, through a new store call that can only ever remove a directory with no usable manifest. |
| **B8** | Any store change made in the browser **refreshes the three `StorePicker`s**, the way saving a user model already refreshes the model combos. |
| **B9** | The tool gets an **`artifacts`** subcommand family with modes `list`, `show`, `note`, `rm`, `sweep`, `summary`. It takes **no configuration flags** and reads `config.artifacts_root()`; heavy imports stay inside the handlers. An empty listing **exits 0**. |
| **B10** | "A saveable run summary" is **two** things: **Save** on the detail view writes what is on screen, and a **lineage report** walks the selected artifact's parents back through the chain. Both are **files**, never an eighth store kind. |
| **B11** | A live run is **visible app-wide**: a clickable line in the shell header naming what is running, where and for how long, plus a marker on the running section's Home tile and its tab. |
| **B12** | **Apply** shows a permanent line naming what the session holds, and **confirms** before discarding a non-empty session, defaulting to not applying. Silent when there is nothing to lose. |
| **B13** | A posterior's amortization is in the **visible** picker text, and a **read-only line under each store picker** spells the selection out in full. |
| **B14** | Both front ends install a **root-logger handler at start-up**. `core` records pass through untouched; every other logger's record is shown **once**, prefixed with the logger that said it. Because a handler exists from start-up, `logging.basicConfig` never fires. |
| **B15** | The training rescue save runs inside a **deferred-cancel critical section**, so a pending cancel can no longer skip it; the window reports the **original exception** with the cancel noted. **Corrected during the whole-piece review** (§12 row 3): the mechanism is **raise-time deferral**, not a catch-time walk of a caught exception's `__context__`. A front-end cancel checkpoint does not fire while `sys.exc_info()` is set anywhere on the worker's stack — inside any `except`/`finally`/`__exit__`/generator teardown on the training or inference paths — so the question of whether a chained exception is "real" is never asked after the fact. That delivers the original promise whole, for the direct collision **and** the residual race alike: a genuine crash with a cancel pending is always reported as the crash, at error level, with a line noting the cancel; a cancel that arrives only once an exception has been fully recovered is a plain cancel, exactly as before. |
| **B16** | A newly added chi probe row's frequency box is **blank**, and the wordings for a blank frequency are unified across the tab and the planner. |
| **B17** | `PaneCapture` wraps **both** pane channels into one ordered list, so a test sees what a user sees. |

### 1.2 How the design reads the decisions

**The store is extended, not reshaped.** Every addition in §2 is additive: new keyword fields on
`Summary` with defaults, two new methods, one new rule. No existing signature changes and no existing
behaviour changes, so piece 5 can build on `core/artifacts/` unchanged while piece 4 runs.

**One formatter per fact.** The browser, the tool's listing and the pickers' new line all describe the
same artifacts. Each fact is formatted in exactly one place — a kind-aware describer in the store's
front-end-neutral vocabulary is *not* built (the store stays presentation-free); instead the *fields*
are structured on `Summary` (`batches_done`, `rows`, `variant`, `finished`) and each front end formats
them, with the GUI's own formatter shared between the table and the pickers' line. What must not
happen is two front ends disagreeing about whether a cache is finished.

**Refusals, not dialogs, carry the rules.** The note rule and the delete refusals are
`core.refusals.Refusal` / `StoreError` with a field key; the two front-end tables name the control or
the flag. Two keys are added (`artifact`, `note`) and the registry's count pin moves with them.

**Every new dialog is an instance dialog.** `tests/conftest.py::_no_modal_dialogs` patches the
*instance* method `QMessageBox.exec`; the static helpers (`QMessageBox.question/warning/information`)
are C++ statics and escape it, so a static call **hangs the offscreen suite** instead of failing. Every
dialog piece 4 adds is built as `QMessageBox(...)` and shown with `.exec()`. This is a global
constraint, not a style note.

**Start-up work is the icon's known failure mode.** The 2026-09-11 taskbar-icon incident was caused by
~150 ms of layout between the native show and the first idle turn. The header's run slot is therefore
built **empty**: no icon load, no timer, nothing to lay out. The browser lists from its `showEvent`
rather than its constructor, so it adds no store scan either — and note that start-up is not scan-free
today, because each of the three `StorePicker`s calls `refresh()` in its own `__init__`. Walkthrough
row 1 is re-run.

### 1.3 Out of scope, and who owns it

- **Figures in the browser, an open-the-folder button, and "use this" jumps.** Considered and declined
  by the owner on 2026-09-17. If wanted later they are additive on top of §3.3.
- **A session summary at quit** ("what this sitting did"). Declined in favour of B10.
- **`rename` from the browser.** Save-as-rename stays on the Prior and Posterior tabs. `rename` on a
  simulation cache is additionally unsound today (it writes a name into the manifest and renames
  nothing, because `Manifest.dir_name` is the bare digest for that kind, after which the cache
  resolves by a name its directory does not carry). The browser never calls it; the unsoundness is
  recorded here and owned by nobody yet.
- **`unnamed()` stays uncalled.** It would return every simulation cache, because
  `write_simulation_manifest` always writes `name=""`, so a "tidy unnamed artifacts" action built on
  it would offer to delete committed training rows. Left as it is.
- **The listing's cost.** `find_prior_by_fingerprint` is quadratic and `delete` costs roughly ten full
  directory scans. Correct and irrelevant at this scale; no index is built. Recorded, unowned.
- **`Summary.mode` for a diagnostic.** A diagnostic deliberately records its kind under `variant`, not
  `mode`, so the mode column cannot show a conditioning geometry that does not exist. §2.1 surfaces
  `variant` rather than changing that.
- **The handoff's stale trap group M** (its header names a module that is now a re-export shim; M1
  omits the TSNPE tab; M3 says five sibling tabs and there are six; M2b's `FloatField.value()` advice
  is superseded by `value_or_none()`). Piece 6 owns the `docs/` split. **Exception:** M1b, which this
  piece resolves, and which §6.2 therefore rewrites.
- **FDT/CrossVal outputs.** Still outside the store (`artifacts_root()/fdt`, `/crossval`); piece 5's.
  The browser lists the seven store kinds and says nothing about those trees.
- **The rest of `docs/STATE.md`'s unowned list** (the Jacobian `LinAlgError`, chi repeat-SBC's fixed
  probe seed, the calibration/TSNPE `accepted` records, the exactly-as-many-directions rule, the
  multi-GPU seed restore, `--chi-k 1`, a resumed run's Fisher record). Untouched.

## 2. The store's additions

All in `core/artifacts/store.py` unless said otherwise.

### 2.1 `Summary` grows (B2, B3)

Five new keyword fields, each defaulted, so no positional construction breaks:

```python
    dir_name: str = ""            # the directory's own name, ALWAYS; the handle remove_incomplete takes
    finished: bool = False        # did the run finish -- NOT the same as `complete` for a cache
    batches_done: "int | None" = None      # simulation only
    rows: "tuple[int, ...] | None" = None  # simulation only: rows per committed batch
    variant: "str | None" = None           # diagnostic only: body["variant"]
```

`list` fills them:

- `dir_name = sub.name` on every row, complete or not. `id` keeps today's meaning exactly (the
  manifest id for a complete row, the directory name for an incomplete one), because `get`'s error
  message already lists incomplete directory names and the GUI picker keys on `id`.
- `finished`: for every kind but `simulation`, `finished = complete` — a committed artifact is
  finished by construction, since `ArtifactWriter._commit` writes the manifest last. For `simulation`,
  `finished = bool(body.get("complete"))`.
- `batches_done` / `rows` / `variant` come straight off the body; None for the kinds that have none.

`Summary.complete`'s docstring gains one sentence pointing at `finished`, so the distinction is stated
where it is read and not only where it is written.

**Why a `finished` field and not a fixed `complete`.** `complete` is load-bearing in three places
(`StorePicker.refresh` skips incomplete rows; `list`'s secondary sort puts them last; the store suite
asserts on it). Redefining it would change all three for the sake of one word. Adding the honest
question beside it costs nothing and closes the trap the state file names.

### 2.2 Reading a run's records (B4)

```python
    def read_log(self, kind: str, ref: str, *, max_bytes: "int | None" = None) -> "tuple[str | None, bool]":
        """``(text, truncated)`` of the artifact's ``log.txt``; ``(None, False)`` when there is no file.

        None and "" are different answers, deliberately: the simulation cache never gets a log (§1.2 of
        the piece-3 design) and an artifact written outside any run gets none either, while a run that
        said nothing writes an EMPTY one -- silence is a record too. With ``max_bytes`` the TAIL is
        returned and ``truncated`` is True.
        """
```

The front ends never join `LOG_FILE` themselves: one place answers "is there a log, and what does it
say", so the cache's absence is explained once.

### 2.3 Removing an incomplete directory (B7)

Nothing today can remove a directory with no manifest: `delete` and `set_note` both resolve through
`_find`, which only ever returns manifest-bearing entries. That is a safety property worth keeping —
neither call can touch a leftover folder by accident — so the new call is a separate one that can
touch **only** those:

```python
    def remove_incomplete(self, kind: str, dir_name: str) -> Path:
        """Remove one directory under ``kind`` that has no usable manifest. Returns the path removed.

        Refuses, as a fielded StoreError: an unknown kind; a ``dir_name`` that is not a direct child
        (any separator, ``..``, or an absolute path); a name that is not a directory; and -- the one
        that matters -- a directory ``_entries`` classifies as COMPLETE. The complement of ``delete``:
        that one can only remove a real artifact, this one can only remove a leftover.
        """

    def sweep_incomplete(self, kind: "str | None" = None) -> "tuple[list, list]":
        """``(removed, failed)`` over one kind or all seven: ``removed`` is ``[(kind, dir_name)]`` and
        ``failed`` is ``[(kind, dir_name, reason)]``. A directory that will not delete (a held handle on
        Windows) is reported, never fatal -- the sweep finishes the rest."""
```

Both raise `StoreError(..., field="artifact")`.

### 2.4 The note rule (B5)

In `core/refusals.py` (torch-free), beside the other rules:

```python
def require_note(key: str, text: str) -> str:
    """The trimmed note, or a Refusal. One LINE, at most NOTE_MAX_CHARS characters; "" clears it.

    Surrounding whitespace is trimmed -- the one transformation this module allows, because it changes
    no meaning -- and everything else is refused rather than fixed (V2): a newline or a tab refuses,
    and so does a note over the limit, whose message gives the limit and the length given.
    """
```

`NOTE_MAX_CHARS = 200` lives in `core/refusals.py` itself, not `core/config.py`: the module is
torch-free on purpose and its defaults are already literals pinned against config by
`tests/test_refusals.py`; a note limit is not a science constant.

`ArtifactStore.set_note` gains `field="note"` on its "no complete artifact" refusal — today it carries
no field key, the only store mutation without one — and both front ends run `require_note` before
calling it.

### 2.5 The two new field keys

`core.refusals.FIELDS` grows by two, so the count pin in `tests/test_refusals.py` moves from 61 to 63:

```python
    Field("artifact", "the artifact", None),
    Field("note", "the note", None),
```

Both descriptions obey the wording ban (no "tab", "box", "flag", "button", "click", "tick", "dialog").

- `core/gui/fields.py`: sentence-shaped entries, because the browser is not an inference tab and the
  `(tab, label)` shape is pinned against `InferenceScreen`'s tab titles.
  `"artifact": "Select an artifact in the list on the Artifacts screen."`,
  `"note": "Edit it in the Note box on the Artifacts screen."`
- `core/tool/fields.py`: `"note": "--note"` (§4.1 spells the note text as a flag precisely so this
  sentence works), and `"artifact": None` — the tool names the artifact positionally, so there is no
  option string to print, and the refusal's own message already quotes the ref. The table's `None`
  branch and its documented reason already exist for the units and the four chi constants.

### 2.6 Pins

- `finished` is `complete` for six kinds and `body["complete"]` for the seventh, over a store holding
  one of each.
- A cache manifested from its first batch lists `complete=True, finished=False` with its
  `batches_done` and `rows`; after `mark_complete` it lists both True.
- `dir_name` is the folder for both a complete and an incomplete row; the complete row's `dir_name`
  disagrees with `Manifest.dir_name` after a rename whose directory move was refused, and the listing
  shows that (§3.3).
- `read_log`: text for a committed artifact, `("", False)` for a silent run, `(None, False)` for a
  cache and for an artifact written with no run active, a tail plus `truncated=True` over `max_bytes`.
- `remove_incomplete` refuses a complete artifact's directory, `..`, an absolute path, a name with a
  separator, an unknown kind, and a plain file; and removes a manifest-less directory and one whose
  manifest is unreadable or declares the wrong kind.
- `sweep_incomplete` over an empty kind returns two empty lists; with one directory it cannot remove,
  it still removes the others and reports that one.
- `require_note`: trims, accepts 200, refuses 201 naming both numbers, refuses `"a\nb"` and `"a\tb"`,
  and returns `""` for `"   "`.
- `set_note`'s refusal carries `field="note"`.

## 3. The artifact browser

New: `core/gui/screens/artifact_screen.py`, plus a table widget in `core/gui/widgets/`.

### 3.1 Where it lives and what it is (B1)

Five registration points, all in the files the piece-4 readers identified:

1. `HomeScreen.SECTIONS` gains `"Artifacts"` (last, after `"Simulate"`).
2. `MainWindow`'s `live_sections` literal gains it — the same name, a third time, as the existing four
   already are; the duplication is left as it is rather than refactored mid-piece.
3. `MainWindow.__init__` builds the screen and `nav.add_screen`s it, **after** the four sections and
   **before** the Settings screen, so the back-arrow slide direction stays monotone with the tile
   order.
4. `MainWindow._section_index` gains `"Artifacts": idx_artifacts`.
5. `MainWindow._save_state` calls the screen's `save_settings(qs)` **by name**. `_all_panels()` is
   panel-typed and stays untouched, which also keeps the browser out of `_refresh_model_combos`.

It is a `QWidget`, not a `BasePanel` (B1). It gets: the design tokens, `make_form`, the icon font, a
status line, and one refusal surface of its own — a small helper that shows the same yellow
"Check your inputs" box `BasePanel._refusal` shows, factored out of `base_panel.py` into
`core/gui/widgets/refusal_box.py` so the two callers share one box rather than two lookalikes. That
extraction is the only change to `BasePanel`'s refusal path, and `BasePanel._refusal` keeps its
signature.

### 3.2 The listing (B2)

A kind selector (the seven kinds, in `KIND_DIRS` order) over a flat, sortable table. **There is no
item view anywhere in the repository today** and `design.build_qss` styles none, so this piece adds
both: a `QTreeWidget` used flat (`setRootIsDecorated(False)`, `setSortingEnabled(True)`,
`setSelectionBehavior(SelectRows)`) and the QSS for `QTreeView`/`QTreeView::item`/`QHeaderView` in
`core/gui/design.py`, in the same token vocabulary as the rest. Building rows by hand — the chi probe
table's and `SettingsScreen.refresh_models`' idiom — neither sorts nor scales past a few dozen rows.

Columns, per kind, from `Summary` alone (no second read):

| kind | columns after `name` and `created` |
|---|---|
| prior | note |
| simulation | progress (`3/4 batches · 96 rows`), finished, note |
| posterior | mode, width, `amortized` / `narrowed (TSNPE)`, note |
| observation | mode, width, note |
| calibration | note |
| inference | note |
| diagnostic | variant, note |

An unnamed artifact shows `Summary.label` (`(unnamed <id>)`), as the pickers do. Incomplete rows sort
last (`list` already does that) and show their `reason` in place of the kind's own columns, with the
`dir_name` as their identifier.

**An unreadable root is not an empty one.** `StorePicker.refresh` swallows the exception and lists
nothing, so today the two look identical. The browser distinguishes them: a kind directory that does
not exist yet reads "nothing here yet"; a root that cannot be read puts the error on the status line.

### 3.3 The detail view (B4)

For the selected row, in a read-only pane beside the table:

- **The manifest, rendered** — `store.get(kind, id)`: schema, id, name, created, note, the git
  revision and environment, the input files with their hashes, the parents, the payloads with their
  digests, the figures by name, and the body's knobs. One scan per selection, which is fine.
- **The run's records** — `read_log(..., max_bytes=1 MiB)`, verbatim, with its `HH:MM:SS level`
  stamps. Two honest gaps are stated in the pane rather than left blank: a cache says *"a training
  cache keeps no log: it is written batch by batch across resumes and shared by every posterior that
  names it"*, and an artifact with no file says *"written outside a run"*. A truncated log says so.
- **A stale folder name.** When `dir_name` disagrees with the manifest's own `dir_name`, the pane says
  so — a rename whose directory move was refused leaves exactly that, and the browser is the first
  place it is visible.
- **Save** (B10) writes what the pane shows to a file chosen through `QFileDialog`.

An incomplete row's detail pane shows its `reason`, its path and its `dir_name`, and offers Sweep.

### 3.4 The actions (B5, B6, B7, B8)

**Note.** A single-line box plus Set. `require_note` runs at the click; a refusal opens the yellow box
with the note key's fix sentence. Success rewrites the manifest through `set_note` and refreshes the
row.

**Delete.** One row at a time, and **two outcomes, not one**. Because no `force` is offered (B6),
`store.delete` will refuse anything with dependents — so a confirmation for such an artifact could
only ever be followed by a failure. The browser therefore asks `store.dependents` **first**:

- **Dependents exist →** the yellow "Check your inputs" box, with no Yes: the kind, name and id, and
  every dependent, including the prior-by-fingerprint case stated in its own words (*"a training cache
  was generated against this prior and its rows are meaningless without it"*). The fix sentence names
  the list on the Artifacts screen, and the operator deletes the children first.
- **No dependents →** an instance `QMessageBox` (§1.2) naming what is about to go, with **No** as the
  default button, and for an unfinished cache its committed rows. On Yes: `store.delete(kind, ref)`.

The store's own refusal stays the last word in both branches — `dependents` is read twice (once by the
browser, once inside `delete`), which is two directory scans and irrelevant at this scale.

**Sweep.** One action per kind (and one for all kinds), behind a confirmation listing what it will
remove, calling `sweep_incomplete`. Failures are reported on the status line.

**Refused while a run is live.** Both Note and Delete and Sweep check `BasePanel._running` and refuse
with the same wording the three existing sites use (`MainWindow`'s model delete, the close event, the
model builder). The browser's *reading* is never blocked — that is the whole reason it is not a
`BasePanel`.

**After any change (B8).** The screen emits `store_changed`; `MainWindow` connects it to a new
`_refresh_store_pickers()` that walks the three `StorePicker`s, mirroring `_refresh_model_combos`.
Without it a picker keeps pointing at a deleted artifact: `StorePicker.restore_key` silently does
nothing when the saved id has vanished, leaving whatever item happens to be current selected. That
silence is deliberate for a picker and becomes a defect the moment a browser can delete.

### 3.5 What it remembers

Selections only (V5): the kind last viewed, and the sort column and order. **Not** the selected
artifact — a remembered id that has since been deleted is exactly the dangling state §3.4 is
protecting the pickers from. Keys are namespaced under the screen's own group by
`MainWindow._save_state`.

## 4. The command-line twin (B9)

New: `core/tool/browse.py`, registered in `core/tool/__init__.py::build_parser` beside `stages`,
`diagnostics`, `smoke` and `fdt`. The module is **not** called `artifacts.py` although its subcommand
is: `core/tool/__init__.py` already does `from . import ...`, and a sibling of that name would sit one
mistaken relative import away from shadowing the `core.artifacts` package it is a front end for.

### 4.1 The subcommand family

One subcommand, `artifacts`, with a positional mode — the `identifiability
{rotation,laplace,jacobian}` precedent:

```
python -m core artifacts list [<kind>]
python -m core artifacts show <kind> <ref>
python -m core artifacts note <kind> <ref> --note TEXT
python -m core artifacts rm <kind> <ref>
python -m core artifacts sweep [<kind>]
python -m core artifacts summary <kind> <ref> [--out PATH]
```

- **No configuration flags.** `add_config_flags` is not called: listing must not be able to fail on a
  bounds file it does not need, and `--help` must stay torch-free. The consequence is stated in the
  subcommand's epilog: the tool cannot say "this posterior does not match your bounds file" — that is
  what loading it does.
- **The root is `config.artifacts_root()`** (`PRISM_ARTIFACTS`), like every subcommand but `smoke`.
  It is **not** spelled `--store-root`: `main` keys `smoke`'s temporary-root behaviour, its empty-root
  cleanup and its Ctrl-C advice on `hasattr(args, "store_root")`, so reusing that name would change
  `main`'s behaviour for this family.
- **Heavy imports inside the handlers**, per `core/tool/stages.py`'s rule.
- **No `force`** (B6).
- `--note TEXT` rather than a positional note text, so `core/tool/fields.py` can map the `note` key to
  a real option string (§2.5).

### 4.2 Output

Prose columns, aligned, in the tool's existing voice — `print`, not `logging`, because
`core/tool`'s framing prints are deliberately excluded from the no-print pin, and a browse command's
output is framing. No JSON: the tool has none anywhere, and inventing a machine-readable format for
one family would be the odd one out.

`list` prints one line per artifact with the same facts as the GUI's columns for that kind, incomplete
rows last with their reason. `list` with no kind prints all seven, each under a heading, and skips a
kind with nothing in it (naming it as empty in a trailing line, so "I didn't look" and "there is
nothing" stay distinguishable).

`show` prints the manifest exactly as §3.3 renders it, then the records, then the two honest gaps.
`summary` prints the lineage report (§5) or writes it to `--out`.

### 4.3 Exit codes

- A missing or ambiguous ref, a bad kind, a bad note: **1**, through the existing `refused:` rung, with
  the fix sentence from `core/tool/fields.py`.
  The kind is a plain positional with NO argparse `choices=`, precisely so a bad one is a refusal at
  1 and not argparse's usage error at 2; the seven kinds are listed in that argument's help text.
- **An empty listing is 0**, with a line saying there is nothing there. A script must be able to tell
  "nothing on disk" from "you asked for something wrong"; exiting 1 on an empty store would make the
  two indistinguishable.
- `sweep` with nothing to remove: **0**. `sweep` that failed to remove something: **1**, naming each.
- A bug: the existing unhandled rung, which sets **1** with a traceback. There is no exit 3 anywhere
  in the tool, and `main`'s own docstring says "1 a refusal or a bug"; no task touches the ladder.

## 5. The run summary and the lineage report (B10)

Two deliverables, one shared renderer. It lives in `core/artifacts/report.py` — in the artifacts
package, beside the manifest code it reads, but **not** in `store.py` and **not** an eighth kind: the
report *describes* the store and must never be mistaken for something the store holds. Two pure
functions over manifests, called by both front ends:

```python
def render_manifest(m) -> str:          # what §3.3 shows and `artifacts show` prints
def render_lineage(store, kind, ref) -> str:
    """The selected artifact, then its parents transitively, oldest last.

    Each step: kind, name, id, created, the input files with their hashes, the knobs that decided it,
    and any Accept it recorded. A parent named in a manifest but absent from the store is printed as
    MISSING with its id -- skipping it would make a broken chain read as a complete one. A cycle is
    impossible by construction (parents are older ids) and is guarded anyway by a visited set.
    """
```

The GUI's Save writes `render_manifest` output plus the records; the GUI's "Lineage report…" writes
`render_lineage`. The tool's `show` and `summary` print the same two strings. One renderer, so the
document a reviewer receives is the same whichever front end made it.

`render_lineage` is torch-free in the same qualified sense as `manifest.py`: its own imports are
stdlib, and a caller reaches it through `core.artifacts`, which imports torch. No claim is made that
it can be imported without torch.

## 6. The three front-end behaviours

### 6.1 A run in progress, visible (B11)

Three parts.

**A run-state broadcaster.** `BasePanel._set_busy` already sets the class flag; it now also publishes
through a module-level singleton in `core/gui/panels/base_panel.py`:

```python
class _RunState(QObject):
    changed = Signal(object)       # the running BasePanel, or None when idle
RUN_STATE = _RunState()
```

A module-level singleton rather than a class signal, because `BasePanel._running` is already
class-level for the same reason (the process-wide stream swap) and a second app-wide fact belongs
beside it.

**The shell's slot.** `NavShell`'s `header_row` gains a flat `QToolButton` named `navRunning`, hidden
by default, on the right of the row before the stretch. `MainWindow` connects `RUN_STATE.changed`:
on a run it sets the text, shows the button and starts a 1 s `QTimer`; on idle it stops the timer and
hides the button. The text comes from a pure helper so a test can assert it without waiting:

```python
def running_banner(panel_title: str, seconds: int) -> str:   # "Running: Posterior — 4:07"
```

Clicking it navigates to the running panel. A `BasePanel` carries no title of its own, so
`MainWindow` builds the map once, in `__init__`, from what it already has: each `SectionScreen`'s
`(label, panel)` pairs and `InferenceScreen.panels()` against its tab titles, giving
`panel -> (section name, tab label, screen index, tab index)`. One place, built from the same literals
the screens are built from, so a renamed tab cannot leave the banner naming a tab that is gone. A
panel the map does not know (the model builder, which is not a `BasePanel`, or a future screen) shows
the banner without a destination rather than raising.

**The section markers.** The running section's Home tile and the running tab carry a marker (the tile
gets a suffix on its label, the tab a leading glyph from the bundled icon font), cleared on idle.

Nothing is disabled that is not already disabled: the existing app-wide control lock stands, and
navigation stays free — you must be able to look at another tab while a twenty-minute train runs.

**Start-up:** the button is created hidden with no icon and no timer (§1.2).

### 6.2 Apply: the session line and the confirmation (B12)

`InferenceScreen` gains a describer:

```python
    def session_contents(self) -> list[str]:
        """Plain phrases for what the session holds -- the config, the prior, the posterior, the
        observation -- in pipeline order; empty when a new_draft would lose nothing."""
```

**The line.** The Config tab grows a read-only derived status line (`_StagePanel`'s existing pattern,
`core/gui/panels/inference/base.py`) naming those contents, refreshed from `refresh_local_gates`,
which `InferenceScreen.refresh_gates` already calls on every panel after every stage.

**The confirmation.** `ConfigPanel._build_config` — the method holding the `new_draft` call; there is
no `_apply` in that module, and it is NOT renamed, because six test sites call it — asks
`session_contents()` first. When it is non-empty it shows an
instance `QMessageBox` listing them, stating that **they stay on disk and can be selected again**,
with No as the default; anything but Yes returns without touching the session. When it is empty,
nothing is shown and today's behaviour is unchanged.

**M1b's other half.** `InferenceScreen.new_draft`'s and `install_config`'s docstrings are rewritten to
say that the confirmation is now the guard and that `install_config` is the in-place one *because* the
prior stage installs the config as its own first step. `PRISM_HANDOFF.md`'s M1b bullet gains one
sentence recording that the user-facing half is answered here; the rest of group M stays piece 6's.

### 6.3 The pickers (B13)

In `core/gui/widgets/artifact_picker.py`:

- **The visible item text** appends a marker for the exception only: a posterior with
  `amortized is False` reads `<label>  —  narrowed (TSNPE)`. Amortized is the norm and gets no suffix.
  The tooltip keeps everything it has.
- **A read-only line under each store picker**, built by one new formatter so the line, the item text
  and the browser's columns cannot drift:

```python
    def selection_summary(self) -> str:
        """"chi · width 18 · amortized · 2026-09-14T10:22:31" for the current item; "" when none."""
```

  The three tabs that own a `StorePicker` (Prior, Posterior, TSNPE) place it directly beneath, and
  refresh it on `currentIndexChanged` and on `refresh()`.
- **Tests, at last.** Nothing pins any picker text today. The item text, the tooltip and the new line
  each get one.

## 7. The four handed-on defects

### 7.1 The front-end root handler (B14)

**What is actually wrong.** Piece 3's spec §4.6 says a library's module-level warning reaches the
operator through `logging.lastResort`. It does not: `logging.warning(...)` calls
`basicConfig()` when the root logger has no handlers, which installs a real `StreamHandler` on the
root. `core.propagate` is True (pinned, because `caplog` reads off root), so from that moment every
`core` record is emitted twice — once by the front end's own handler, once by the root's.

Piece 3 recorded that no trigger exists on today's paths. **That is wrong.** sbi has eleven
module-level `logging.warning` sites, and the leakage warnings in `accept_reject_sample`
(`sbi/samplers/rejection/rejection.py:336,359`, threshold `warn_acceptance=0.01`) sit on the path of
every posterior draw PRISM makes — `DirectPosterior.sample` and sbi's own `RestrictedPrior` both go
through it. A leaky posterior triggers it with no NaN involved. Two more are reachable
(`sbiutils.py:344`, a one-row standardizing batch; `rejection.py:250`, unused kwargs). Only
`npe_msg_on_invalid_x` is genuinely closed, and closed by core's own pre-filter in
`core/SBI/train.py`, not by chance.

**And in the window it is worse than doubling.** `basicConfig`'s `StreamHandler` binds `sys.stderr`
**at construction**, which under a run is that run's `_SignalStream`. So the duplicate arrives in the
pane at the error stream's `warning` level (a ⚠ on an INFO record), and once the run ends the handler
keeps writing into a stopped pump, where lines are appended and never published. After one trigger,
library records go **missing** rather than doubling.

**The fix.** A new torch-free module `core/logging_root.py`:

```python
def install(sink) -> None:
    """Install THE root handler for this process. ``sink(record)`` is the front end's own output, called
    with the LogRecord so it can split by level and resolve its streams itself (§7.1 of this design).

    Two jobs. (1) Existing from start-up, it is why ``logging.basicConfig`` never fires, so no second
    copy of a ``core`` record can ever appear. (2) A record from the ``core`` tree is DROPPED here --
    it already has the front end's own handler and the artifact's log.txt, both of which read off the
    ``core`` logger -- and every other logger's record is handed to ``sink`` once.
    """
def remove() -> None: ...
```

The drop is a `logging.Filter` on the handler, not a change to `core.propagate`: `caplog` reads off
root and `tests/test_refusals.py:423` pins that propagation, and `RunLog` and both front-end handlers
sit on the `core` logger itself, so a filter leaves all three untouched.

**The GUI's sink** (installed in `core/gui/app.py::build_app`, before `MainWindow`): write to
`sys.stdout` below WARNING and `sys.stderr` at WARNING and above, **both resolved at emit time**, with
the line `library: <logger name>: <message>`.

Resolving at emit is the fix, not the bug: what broke was `basicConfig`'s handler binding the stream
at **construction** and then writing into a stopped pump for the rest of the process.
`core/tool/logging_console.py` already states the rule ("resolve `sys.stdout`/`sys.stderr` AT EMIT
TIME, never at construction") for exactly this reason. Resolving late also solves three problems at
once and adds no plumbing: during a run those two names **are** the run's `_SignalStream`s, which are
thread-safe by construction and feed the pump — so a library record arriving on the **worker thread**
never touches a widget directly, which a `log_pane.append_line` from the handler would have done — and
outside a run they are the real console. Splitting by level matters because `_SignalStream` on err is
hard-wired to the pane's `warning` level: routing an INFO library record through stderr would put a ⚠
on it.

One accepted consequence: a library record written during a run goes through `_SignalStream.write`,
which is a cancel checkpoint like every other print. That is correct — it is one more place a cancel
can take effect — and the commit and unwind paths are covered by §7.2's critical section.

**The tool's sink** (installed in `core/tool/__init__.py::main`, around the whole body, removed in a
`finally` because `main` runs repeatedly in one process under the suite): every non-`core` record to
**stderr**, resolved at emit, with the same prefix, whatever its level. Deliberately unlike the GUI:
the tool's stdout carries results a script reads, so a library's chatter may never land there.

`core/logging_root.py` is a new top-level file under `core/`, so it is inside `CODE_ROOTS`' `"core"`
and needs no `tests/_fixtures.py` edit. It imports only `logging`, which keeps `python -m core --help`
torch-free.

**Why no suite has ever seen this.** `_pytest/logging.py:349-355` attaches a handler to the root logger
for every test phase, so `len(root.handlers) == 0` is never true during a test and `basicConfig`
cannot fire. The test for this therefore removes pytest's root handlers, asserts the defect's
precondition, and restores them (§9.2).

### 7.2 The rescue save and the cancel latch (B15)

**The mechanism.** `core/SBI/pipeline.py`'s `except BaseException` rescue block logs before it saves.
`gui.streams._PumpLogHandler.emit` calls `CancelToken.check()` before its own try, `check` raises
`WorkerCancelled` (a `BaseException`, deliberately) when the token is requested-but-not-yet-fired on
the worker thread, and stdlib logging does not catch it. So a Cancel pressed just before a crash makes
the rescue block's own announcement raise: `_tc.save` is skipped, the re-raise at the end of the block
is never reached, and up to `TRAINING_CHECKPOINT_EVERY - 1` batches (default cadence 50, more if an
earlier write deferred on a failed RNG snapshot) are lost. The block's comment asserts this is safe
because the latch is one-shot and "the raise has already happened" — true only when the cancel is what
unwound the run. Two calls can raise, not one: the `log.warning` at `:1784` is outside the inner try
altogether, and the `log.info` at `:1792` is inside a try whose only handler is `except Exception`.

**Second consequence, which the hand-off note missed.** `Worker.run` catches `WorkerCancelled` **by
name** and reports a cancellation: no error signal, no dialog, no traceback. The real failure — the
OOM, the device error, the bug — is discarded along with the rows. **You are told you cancelled a run
that crashed.**

**The fix, two parts.**

1. **A named critical section.** `core/runs.py` gains a thread-local context manager:

```python
@contextmanager
def cancel_deferred():
    """Inside this block a front end's cancel checkpoint does not fire: records still flow, the token
    stays REQUESTED, and the next check outside the block raises as usual.

    It exists for the unwind path and the checkpoint commit -- the two places where raising between
    two writes loses committed work. It defers a cancel; it never discards one.
    """
```

   `gui.streams._PumpLogHandler.emit` and `_SignalStream.write` consult it before calling
   `CancelToken.check()`. The rescue block runs inside it, and so does `training_checkpoint.save`'s
   commit — which turns the "do not print or log between steps 1 and 3" rule from a convention a
   source-reading test polices into a mechanism. The rule and the test both stay: the section is the
   guard, the test is the proof that the guard is where it is claimed to be.

2. **The window reports the crash.** `Worker.run`'s `WorkerCancelled` branch walks `__context__` /
   `__cause__`; when it finds a non-cancel exception it reports **that**, as an error with its
   traceback, and the pane and the dialog note that a cancel was pending. A clean cancel — nothing
   chained — is reported as a cancel exactly as today.

**What this does not claim.** No gate can provoke the collision; the smoke gate does not crash. It is
pinned by a test that injects a failure with a requested-but-unfired token (§9.2).

`tests/test_user_sbi.py:3782` pins two source lines of the rescue block and `:4006` pins the level of
each of its three messages; both move with the change, in the same commit, as part of the task.

### 7.3 The blank probe row (B16)

`FloatField.__init__` accepts `None` for "empty box" (`"" if default is None else str(default)`), and
`_ChiProbeRow.__init__` and `InferTab._add_chi_probe` default `freq_hz=None`. Nothing persists a probe
row, so this touches only the in-session seed.

**The wordings.** Today one state has three sentences: `_ChiProbeRow.problems` says
`probe N: drive frequency must be a positive number (got 0)`, "Plan probes…" says `no frequency
entered` — its `_probe_frequency` already maps a non-positive box to None, so it *already* treats the
seeded zero as blank — and `core/SBI/chi.py` says `drive frequency must be finite and positive, got
0.0 Hz`. With a blank seed the seeded state becomes genuinely blank, and:

- `problems()`'s blank branch (`probe N: drive frequency is blank`) is the one wording, unchanged.
- The planner's `no frequency entered` becomes the same sentence, which also makes its own
  "Filled N blank frequency box(es)" line literally true.
- The **typed-zero** sentences stay as they are at both layers. A zero somebody typed is a different
  state from a box nobody filled, and V2's whole point is that a message describes what happened.

`tests/test_nav_and_gating.py:662` asserts a *seeded* row reports "positive" and "got 0"; it is
rewritten in the same commit to assert the seeded row reports blank and a **typed** zero reports the
zero.

The key stays `recording_probe`. A new key would cost five coordinated edits for a sentence the
existing key already answers; the table maps it to `--forced`, which is the flag that carries a
probe's frequency.

### 7.4 `PaneCapture` (B17)

`tests/_fixtures.py::PaneCapture` replaces `append_line` only. Because `BasePanel.dispatch` resolves
`self.log_pane.append_line` at connect time, a capture installed before the dispatch **does** see the
`signals.log` channel; the gap is exactly `signals.log_batch` → `append_lines`, which is the pump's
channel and therefore carries everything the worker produces: every print, every retired tqdm bar,
every `core` record via `_PumpLogHandler`, and every Python warning via the redirect's `showwarning`.

The helper wraps both, appending `(level, text)` for each pair of a batch — flipping the payload's
`(text, level)` into the helper's existing order, which its own test
(`tests/test_worker_dispatch.py:359`) pins. One ordered list, so a test reads the lines in the order a
user sees them; a test that wants one channel alone filters. Its docstring states the two orders and
why they differ.

The four tests that hand-roll a `log_batch` slot are revisited: where one builds its own signals only
to read pane lines, it moves to the helper; where one deliberately inspects the raw payload, it stays
and gains a comment naming the order. A batch arrives from the pump's daemon thread at 15 Hz as a
queued cross-thread signal, so such a test pumps the Qt loop before asserting, exactly as the existing
worker tests do.

## 8. The small carried items

### 8.1 The two untested guards in `load_observation` (from `docs/STATE.md`)

The state file names one gap; there are two, side by side, and they differ in class, key and timing:

- `store.py:714-717` — config width vs the manifest's: a `Refusal(field="observation")`, **before** the
  payload is read. This one is tested.
- `store.py:727-730` — the payload's own width vs the manifest's: a `StoreError` with **no** field key,
  **after** `torch.load`. Untested — the gap the state file names.
- `store.py:724-726` — the payload's digest vs the manifest's. **Also untested**, and not recorded
  anywhere. A second gap the documents miss.

Both payload guards get isolated tests: write a real observation, corrupt the payload (a wrong width;
then a right width with wrong bytes), and assert the sentence and the class. The width guard also
gains `field="observation"`, so it matches its sibling four lines up and the front ends can name the
control.

### 8.2 Stale counts, in the files this piece touches

`core/gui/panels/base_panel.py` says "Nine of these exist (Reduction, FDT, CrossVal, Simulate + the
five inference tabs)" and counts nine three more times; `core/gui/panels/inference/base.py` says "the
five inference tabs"; `core/gui/screens/inference_screen.py`'s module docstring says "five tabs" while
its class docstring twelve lines below says six. There are **ten** `BasePanel` subclasses (four section
panels plus six inference tabs). The behaviour is fine — `layout_key` is the class name, so all ten
differ — and every one of those files is touched by this piece, so the counts are corrected as part of
the task that touches them. With the browser registered, the Home tiles go from four to five and
`base_panel`'s count stays ten (the browser is not a panel); the corrected sentences say so.

## 9. Tests

### 9.1 Fixtures

No new session-wide guard. Three additions to `tests/_fixtures.py`:

- `PaneCapture` wraps both channels (§7.4).
- A cheap store builder for browser tests: one artifact of each of the seven kinds written through the
  real writer at minimum size, plus one manifest-less directory, one unreadable-manifest directory and
  one wrong-kind directory. **Seconds, not minutes** — it must not reach for `tiny_run`, whose cost is
  why `screen_run` is module-scoped.
- A helper that builds the browser screen against a given store, monkeypatching the same
  `_resolved_store` seam the picker tests already use.

Note for every task: `tests/conftest.py::_no_modal_dialogs` patches the **instance** `QMessageBox.exec`
only. A test that needs the Yes branch monkeypatches on top, the way
`tests/test_nav_and_gating.py:1180` already does. A piece-4 dialog written as a static helper would
**hang** the offscreen suite (§1.2).

### 9.2 New tests, by area (names indicative)

**The store (§2).** `finished` per kind and across a cache's life; `dir_name` on both row shapes and
after a refused directory rename; `read_log`'s four answers and its tail; `remove_incomplete`'s six
refusals and its two successes; `sweep_incomplete`'s partial failure; `require_note`'s five legs;
`set_note`'s field key.

**The browser (§3), a new suite `tests/test_artifact_browser.py` — the eighteenth.** The kind selector
lists seven kinds; each kind's columns for a known store; an unnamed artifact's label; incomplete rows
last with their reason; "nothing here yet" versus an unreadable root; the detail pane's manifest
rendering, its records, the cache's no-log sentence, the written-outside-a-run sentence, the truncation
notice and the stale-folder notice; Note's success and its refusal (with the fix sentence in the box);
Delete's confirmation text including a fingerprint-held cache and an unfinished cache's rows; Delete
refused by the store; Sweep; all three refused while a run is live; `store_changed` reaching the
pickers; the remembered kind and sort across a relaunch; the selection deliberately **not** remembered.

**The tool (§4), in `tests/test_tool.py`.** Each of the six modes in process through `main(argv)`; an
empty listing exits 0 and says so; a missing ref exits 1 with the `refused:` line and the fix sentence;
`rm` has no `--force`; `sweep`'s partial failure exits 1 naming each; `--help` for the family imports
no torch (the existing fresh-interpreter pattern); the family takes no configuration flags.

**The report (§5).** `render_lineage` over a full chain (prior → cache → posterior → observation →
inference); a missing parent printed as MISSING; `--out` writes the same bytes the GUI's Save writes.

**The run indicator (§6.1).** `RUN_STATE` fires on busy and idle; `running_banner`'s text at 0 s, 59 s
and over an hour; the button hidden at launch with no timer running; the tile and tab markers set and
cleared; clicking it navigates to the running panel.

**Apply (§6.2).** `session_contents` for each stage's state; the line's text; the dialog appears only
when non-empty and its text names what is held; No leaves the session untouched; Yes replaces it;
an empty session shows nothing (`SHOWN == []`).

**The pickers (§6.3).** The item text for an amortized and a narrowed posterior; the tooltip unchanged;
`selection_summary` for each kind and for no selection; the line refreshed on change and on refresh.

**The root handler (§7.1).** With pytest's root handlers temporarily removed: the precondition
(`logging.warning` installs a handler when none exists) and that `install()` prevents it; a `core`
record reaches the sink **not at all** and the front-end handler **once**; a library record reaches the
sink once with its prefix and level; `caplog` still sees `core` records; the GUI sink routes to the live
pane and otherwise to the captured real stderr; the tool sink writes to stderr and never stdout; both
remove cleanly, and `main` leaves no handler behind across repeated calls.

**The rescue save (§7.2).** `cancel_deferred` defers and does not discard (the next check outside
raises); a run that crashes with a requested-but-unfired token **saves** its rows and re-raises the
original; `Worker.run` reports that original as an error with the cancel noted; a clean cancel is still
a cancel; the commit runs inside the section; the source-reading pin extended to the rescue block.

**The probe row (§7.3).** A new row's box is empty and `value_or_none()` is None; a seeded row reports
blank; a typed zero still reports the zero; the planner's sentence matches the tab's; `FloatField(None)`.

**The shared refusal box (§3.1).** The extracted `refusal_box` shows the same title, text and fix
sentence for a given `Refusal` whichever caller shows it, and `BasePanel._refusal`'s own behaviour is
unchanged (its existing tests stand as the pin; one new test asserts the two callers agree).

**The carried items (§8).** The two payload guards; the width guard's new field key.

### 9.3 Tests that change

`tests/test_nav_and_gating.py:662` (the seeded zero, §7.3); `tests/test_user_sbi.py:3782` and `:4006`
(the rescue block's source and levels, §7.2); `tests/test_worker_dispatch.py:359` (extended for the
second channel, §7.4); `tests/test_refusals.py:89` (`len(FIELDS)` 61 → 63) and its two table-closure
tests; the four hand-rolled `log_batch` tests (§7.4);
`tests/test_artifact_store.py`'s `Summary` assertions.

### 9.4 Count and budget

Piece 2 added 99 against a budget of about 42 ± 8, and piece 3 added 143 against about 80 ± 20 with
the overrun unrecorded in its deviations. Stating a number honestly is therefore part of this spec:
**130 ± 30 added tests**, taking the collected total from 603 to roughly **733**. §12 records the
actual number and, if it falls outside the band, why — high or low.

## 10. Gates and verification

- **Per task:** tests first, then the implementation, then a code review and a fix loop, then the
  one-process fast gate `pytest -m "not slow"` in the background, logging to a file, **with no source
  edited while it runs**. Target 16 minutes (12 min 13 s at `9e2f7ef` with 600 tests); ~130 cheap GUI
  and store tests should add two to three minutes. A gate over 16 minutes is a deviation row, not a
  silent new normal.
- **The gates are run by the session that owns the piece, never by a subagent**, and no subagent starts
  a background test run. (Piece 3's lesson.)
- **Before the piece is called done:** the slow set (`pytest -m slow`), and the **GPU smoke gate** —
  the four command lines of `CLAUDE.md` — because §7.2 changes the training unwind path and §7.1
  installs a handler the training run logs through. The diagnostic card is **not** required: no task
  touches a line under `core/diagnostics` that creates or moves a tensor, and §12 records that
  judgement so it can be checked.
- **The display walkthrough's D rows** (§11) are the owner's, on a real screen, at the end. **No task
  may claim them.**
- `docs/STATE.md` is updated at the end of every session, and the execution ledger lives under
  `.superpowers/sdd/2026-09-17-gui-usability-and-artifact-browser/`. Every ruling made during
  execution goes in the ledger **with what it costs if it is wrong**.

## 11. Display walkthrough rows (section "Piece-4 GUI checks")

Added to `docs/checklists/display-walkthrough.md` as rows **D1–D16**. Rows 1–20, A1–A9, B1–B8 and
C1–C11 stand; piece 4 re-checks only what it changes, plus row 1.

| # | surface | why |
|---|---|---|
| D1 | The Artifacts tile opens the browser and lists each of the seven kinds | B1, B2 |
| D2 | A kind with nothing in it reads "nothing here yet" | §3.2 |
| D3 | A training cache's row: its progress, and finished versus has-a-description | B3 |
| D4 | The detail pane: the description, the records with their stamps, and the cache's no-log sentence | B4 |
| D5 | A note set, cleared, and refused (over 200 characters, and with a newline) | B5 |
| D6 | Delete refused by a dependent — including a prior held by a cache that never named it | B6 |
| D7 | Delete an unfinished cache: the confirmation names its rows; No leaves it | B6 |
| D8 | Sweep removes the leftover folders and nothing else | B7 |
| D9 | After a delete, the Posterior tab's dropdown no longer offers it | B8 |
| D10 | Note, Delete and Sweep refused while a run is live; reading still works | B6 |
| D11 | The header line during a train: what, where, elapsed; the tile and tab markers; clicking it jumps | B11 |
| D12 | Apply: the session line, the confirmation naming what is held, and No keeping the session | B12 |
| D13 | A narrowed posterior says so in the closed dropdown, and the line beneath spells it out | B13 |
| D14 | A newly added probe row is blank, and Run reports it as blank | B16 |
| D15 | A full train shows no line twice in the pane, and a library warning carries its prefix | B14 |
| D16 | Row 1 re-run: the taskbar button still carries the mark after the header's new slot | §1.2 |

## 12. Deviations made during execution (piece 4)

*(Filled in as the piece runs; every row states what was found, what was decided, and what it costs if
the ruling is wrong. Empty at approval.)*

Rows 1–12 were ruled at **planning** time, before any code was written, from 60 objections raised by the
ten plan drafters against the interface contract and against this spec. The full text of each, with what
it costs if the ruling is wrong, is in the ledger's rulings P1–P43
(`.superpowers/sdd/2026-09-17-gui-usability-and-artifact-browser/progress.md`). Four outright factual
errors in this spec were corrected **inline** rather than recorded as deviations: an invented exit code 3
(the tool's unhandled rung sets 1), a claim that start-up performs no store read (the three pickers each
list in their constructor), two test-change entries for assertions that do not exist, and a reference to
`config_tab._apply`, a method of that name not existing.

| # | spec section | deviation and why |
|---|---|---|
| 1 | §2.4, §2.5 | `set_note`'s missing-artifact refusal carries `field="artifact"`, not `field="note"`: its sentence is about an artifact that is not there, so the note key would send the operator to the note box when the fix is to select an artifact that exists. `field="note"` is kept for `require_note`'s own refusals. `read_log`'s and `delete`'s missing-ref refusals take the same key (P1, P20) |
| 2 | §2.1, §3.2, B3, B6 | `Summary` gains a **sixth** field, `batches_planned`, from the body's `identity["n_runs"]`, so `3/4 batches` can be rendered. `rows` is written only by `mark_complete` — `save` passes none — so an unfinished cache shows batches and no row count, and the delete confirmation names batches. The alternative was to make the checkpoint commit write rows per batch, which this piece will not do for a display nicety (P2) |
| 3 | §7.2, B15 | **Superseded during the whole-piece review; the text below replaces P26.** P26's premise (a blind `__context__` walk would open the red box for a *recovered* exception, so the residual race must report a plain cancel plus the in-flight traceback instead) was factually wrong on two counts, both found by running the real code: the out-of-memory ladders log **outside** their `except` handlers (`pipeline.py:707,797,909,1659`), so they were never the hazard P26 was written to avoid; and the actual common false-positive site is `sdeint.py`'s `finally: bar.close()`, which sits on the graphed CUDA solver — the most likely GPU crash site — so a catch-time judgement would have routinely reported a real GPU crash as "Run cancelled," the outcome the owner named worst. The fix moves the question from catch time to **raise time**: `sys.exc_info()` is set during every `except`/`finally`/`__exit__`/generator teardown and `None` in normal flow, so deferring a front-end cancel checkpoint (`_PumpLogHandler.emit`, `_SignalStream.write`) whenever it is set — after an audit of every such block on the worker's training and inference paths for long-running work — makes the direct collision **and** the residual race report the same thing: the real crash, at error level, with a line noting the cancel was pending. A recovered guard followed by a cancel is still a plain cancel. This restores the owner's original B15 choice in full rather than narrowing it. Cost if wrong: a Cancel can wait while the worker is inside a long exception handler — bounded by the audit (the longest deliberate hold measured was 1.2 s) — delayed, never lost |
| 4 | §3.2 | "Incomplete rows sort last" is made a property of the table — a `QTreeWidgetItem.__lt__` comparing `(not complete, then the column)` — rather than the accident it would otherwise be (an incomplete row's `created` is `""`, so it only trails while the sort is by date). `int(Qt.SortOrder)` also raises in PySide6 6.9.3; `.value` is the spelling (P12, P13) |
| 5 | §3.3 | The stale-folder notice is **suppressed for the simulation kind**: `Manifest.dir_name` is the bare digest there and `write_simulation_manifest` writes wherever it is handed, so a folder name differing from the id is legitimate and the notice would fire on any hand-placed cache (P6) |
| 6 | §3.3 | `read_log`'s "a run that said nothing" answer gets its own sentence, "the run recorded nothing" — piece 3's invariant is that silence is a record, and a blank pane under a Records heading reads as a bug (P18) |
| 7 | §4.2 | The tool prints `,` where the GUI prints `·`: `core/tool`'s printed strings are ASCII-only today and this piece does not make the first exception (P10) |
| 8 | §1.2, §4.2 | The tool cannot import the GUI's `columns_for` — that would pull PySide6 into `python -m core --help` — so it keeps its own column list and a **test** pins the two in step, with the GUI's spelling canonical. §1.2's "one formatter per fact" therefore holds per front end, not across them (P11) |
| 9 | §3.1 | `KIND_DIRS` is re-exported from `core/artifacts/__init__.py` so the GUI makes no submodule import (P17) |
| 10 | §3.4 | `core/gui/main_window.py`'s three **static** `QMessageBox` calls are converted to instance dialogs by the task that already edits that file: they are the exact form §1.2 says hangs the offscreen suite, sitting in a file this piece touches (P21) |
| 11 | §5, §9.2 | Both front ends write their report with `newline="
"`: `Path.write_text` defaults to translating `
` to `

` on Windows, which would have made the "same bytes" claim quietly false (P23) |
| 12 | §6.3 | The picker's tooltip adopts `narrowed (TSNPE)` too, rather than keeping `NON-AMORTIZED (TSNPE)` beside the new item text — one fact, one wording (P25) |
| 13 | §1.2 | "the GUI's own formatter shared between the table and the pickers' line" does not hold: the table needs a per-kind tuple and the picker one line, so the GUI has two formatters (`artifact_table.cells_for` and `artifact_picker._summary_line`). What §1.2 actually protects — that no two surfaces disagree about whether a cache is finished — holds, because both read the same `Summary` fields (P44) |
| 14 | §9.2 | The table widget's own tests (the pure `columns_for`/`cells_for` cases and the widget cases) live in `tests/test_worker_dispatch.py`, beside the other widget tests, because the browser's suite does not exist yet at that task. `tests/test_artifact_browser.py` — the eighteenth suite — is created by the screen task and holds the screen's tests (P45) |

Rows 15–22 were ruled at the **whole-piece review** (2026-09-21), after all 26 tasks landed. The full
text, with what each costs if wrong, is in the ledger's rulings R1–R9
(`.superpowers/sdd/2026-09-17-gui-usability-and-artifact-browser/progress.md`, "Whole-piece review").
Row 15 records a mechanism change that Task 20's own ruling claimed had already been written here and
had not been — the controller's error, caught only at this final review.

| 15 | §7.1, B14 | B14's guarantee — a `core` record is shown once, everything else is shown once with its logger named — is delivered by a **different mechanism** than this section describes. Not the name-based rule (drop anything from the `core` tree): the root sink drops a record only when a logger **below** the root, on the path to it, already has a handler that emitted it (a bare `NullHandler` does not count). The name-based rule, as written, would have reintroduced three of the very losses B14 exists to close: a `core` WARNING logged with no `core` handler attached (the window between runs) would reach nothing instead of `stderr`; a library that installs its own handler at import (pytensor does) would be printed twice; and a library's `log.exception` traceback would be dropped. The walk-up rule keeps B14's guarantee exactly during a run and in the tool, where `core` always has a handler, so the sink stays silent there (T20's ruling) |
| 16 | §2.3, §3.4, B7 | Sweep's safety rule is **narrower** than approved, and its removal is **bound** to what the confirmation showed. A review probe deleted a real calibration together with its payload: the approved rule treated any manifest this build could not parse — including one written under a later schema — as "no artifact here," and a manifest declaring another kind was treated the same way. Sweep now removes **only** a directory with no manifest file at all; an unparsable or wrong-kind manifest is reported on the status line and left untouched. A second probe showed the confirmation did not bind the action: `_sweep` re-scanned the directory at removal time, so an entry that appeared after the dialog opened was removed although the operator never saw it ("Removed 2 of 1 leftover directories"). The list shown is now exactly what is removed, each entry going by name through the single-directory `remove_incomplete` call — the same call Task 5 spent two fix rounds hardening against case, alias and junction tricks, and which no front end had ever called before this. A third probe showed a sweep from a second process could delete a directory a **live** run is still writing into, because the manifest commits last; a directory whose tree was modified in the last few minutes is now skipped and named in the report instead (R1, R2, R3) |
| 17 | §4.1, §4.3, B9 | `python -m core artifacts sweep` defaults to a **dry run** — it prints exactly what it would remove and removes nothing — with `--yes` performing it. This is not a configuration flag (B9 still takes none): it is a confirmation, matching the window's dialog, and a dry run refuses nothing so it needs no new refusal key (R4) |
| 18 | §3.4 | `_sweep`'s call into the store was unguarded: a kind that could not be listed (an unreadable directory, a permissions error) threw out of the click **after** earlier kinds' leftovers had already been removed — the application's last-resort red box, no status line, `store_changed` never emitted, and the three pickers left showing rows that no longer existed. Guarded like every other entry point on this screen, reporting the unreadable kind on the status line instead (R5) |
| 19 | §7.2, B15 | `Worker.run` never consulted the cancel token on the **success** path, so a Cancel requested inside a deferral window that the run finishes before checking again left the run reported as a plain completed success — the request simply vanished. The run really did finish, so it is still reported as success, with an added line saying it finished before the cancel could take effect: the missing counterpart to the failure-path fix in row 3 (R6) |
| 20 | §2.3, B6 | `delete()` could report success on a removal that did not happen — the retry helper silently swallowed a `FileNotFoundError` raised inside its own walk — and, alone among the store's three destructive calls, wrapped no `OSError` from a held file handle, so a live handle escaped `artifacts rm` as a raw traceback instead of a refusal. `delete()` is given the same existence check and `StoreError` wrapping its two siblings already had, since it is the only one of the three that removes a real artifact and was the least guarded (R7) |
| 21 | §1.2 | No source scan enforced §1.2's "every new dialog is an instance dialog" rule; `tests/conftest.py` asserted as fact that no static `QMessageBox` call remained under `core/`, but nothing would have caught a regression, and a regression **hangs** the offscreen gate rather than failing it — the worst failure mode a suite can have. A scan is added, in the repository's existing source-scan idiom (R8) |
| 22 | §7.3, B16 | B16's typed-zero-versus-blank distinction did not hold end to end: the probe planner's auto-fill mapped a **typed** zero to `None` the same as a genuinely blank box, so a value someone typed was silently overwritten by a suggested frequency instead of being refused as a zero. The predicate is fixed (`value_or_none() is None`) rather than the claim withdrawn, so a typed `0` now refuses, which is the decision B16 made (R9) |

**Test count and the diagnostic card, recorded per §9.4 and §10.** The final fast gate at `80be144`
collected **732** tests against **603** before the piece began — **+129 added**, inside §9.4's stated
band of 130 ± 30. The diagnostic card was judged unnecessary at planning because no task touches a
line under `core/diagnostics` that creates or moves a tensor (§10); that judgement was checked again
at the end against the piece's whole diff and **held** — the diagnostic card was not run.
