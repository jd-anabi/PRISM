# PRISM piece 5: the secondary analyses — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring the effective-temperature measurement and the parameter sweep under everything
pieces 1–4 built — records with provenance, input checking that refuses before the spend, one
refusal wording across five screens and two subcommands — fix the two science defects on that path,
and add a facility for comparing saved runs.

**Architecture:** The two analyses are the last code in the tree outside the store and outside the
refusal machinery. An eighth artifact kind, `fdt`, holds both of them and their comparisons; unlike
the six ordinary kinds it is written **progressively** — its directory and a first manifest exist
from the start and survive a cancel — which is the training cache's rule, adopted because these runs
take hours. The front end creates the writer (so it knows the folder before it dispatches) and the
stage enters it (so the run log, which is thread-local, reaches `log.txt`). Checks live in the two
config builders, so both front ends inherit one wording. The comparison facility is last and
nothing before it depends on it.

**Tech Stack:** Python 3.12 (conda env `biophys-env`), PySide6 6.9.3, torch 2.9.0+cu130, h5py 3.15.1,
sbi 0.25.0, pytest. Windows 11; PowerShell and Git Bash both available.

**Spec:** `docs/superpowers/specs/2026-09-22-secondary-analyses-design.md` — binding. Its §1.1 holds
decisions **E1–E12** and its §1.2 the design rulings; every task below cites what authorises it.
Read the spec and this plan together. The spec was reviewed against the code by four readers and a
judge before this plan was written; their reports are in the ledger.

## Global Constraints

Every task's requirements implicitly include this section.

**Environment**
- Interpreter: `C:\Users\J\anaconda3\envs\biophys-env\python.exe` (Python 3.12). The `python` on PATH
  is NOT this environment and has none of the dependencies.
- Anything importing torch needs `KMP_DUPLICATE_LIB_OK=TRUE`; any headless check that imports PySide6
  also needs `QT_QPA_PLATFORM=offscreen`. The root `conftest.py` defaults both; `run.bat`/`run.sh`
  set only the first (the GUI needs a real platform plugin); a bare `python -c` sets neither.
- CUDA is available (RTX 5070 Ti, 16 GB shared with the desktop). Read free VRAM with
  `nvidia-smi --query-gpu=memory.used --format=csv`, never `torch.cuda.mem_get_info()`.
- Paths never depend on the working directory: `core/config.py` resolves `RESOURCES_ROOT`
  (`PRISM_RESOURCES` overrides) and `artifacts_root()` (`PRISM_ARTIFACTS` overrides) from its own
  location. Point `PRISM_ARTIFACTS` at a scratch directory, or use `core.artifacts.use_store`, to
  keep a check away from the real `Artifacts/`.
- **The FDT path is pinned to the CPU.** `cli.make_fdt_config` forces `cpu_device()` and
  `cli.make_param_sweep_config` does the same; only `cli.make_reduction_config` uses
  `detect_device()`. No task changes that.
- `core/refusals.py`, `core/runs.py`, `core/tool/fields.py`, `core/tool/logging_console.py` and
  `core/logging_root.py` are torch-free and stay so: `python -m core --help` must not import torch
  (piece-2 D10). `core/tool/browse.py` keeps its heavy imports inside its handler for the same
  reason, and so must `core/tool/fdt.py`'s additions.

**Tests and gates**
- Fast gate, after EVERY task, ONE process, in the background, logging to a file:
  `pytest -m "not slow"`. Target **16 minutes** (13 min 57 s at `80be144` with 729 tests; this piece
  adds about 170). Never run two pytest processes at once, and check for a live `python` process
  before starting one.
- **The gate is run by the session that owns the piece, never by a task's implementer, and no
  implementer starts a background test run.** A task's own steps run FOCUSED tests only (named files
  or node ids), in the foreground.
- **Do not edit any source file while a suite is running.** Several tests assert on
  `inspect.getsource`, which reads the file as it is now with the line numbers the function was
  loaded with; an edit mid-run produces a failure that is not real.
- Markers: `slow` (today the chi full-pipeline test in `tests/test_user_sbi.py` and the FDT/crossval
  tiny-size run in `tests/test_tool.py`; this piece adds to it), `gpu` (skipped without CUDA),
  `display` (skipped offscreen). **Every test that runs a real simulation campaign is
  `slow`-marked** — that is what keeps the fast gate under target.
- The suites run against a temp artifact root installed once per process; the teardown asserts the
  real `Artifacts/` gained nothing, that the real `PRISM.ini` is unchanged, and that no
  `<repo>/sbi-logs` appeared. **The FDT and sweep output paths resolve through
  `config.artifacts_root()` directly**, so any test that runs one must set `PRISM_ARTIFACTS` (the
  `tool_env` fixture does) or write through a store the test owns — otherwise it writes into the
  real `Artifacts/` and fails the session teardown.
- Every `QMessageBox.exec` a test did not fake itself is recorded in `tests/_fixtures.py::SHOWN` and
  returns 0. **The guard patches the INSTANCE method only** — the statics
  (`QMessageBox.question/warning/information`) are C++ statics and escape it, so a static call
  **hangs the offscreen suite** instead of failing. Every dialog this piece adds is `QMessageBox(...)`
  shown with `.exec()`. A source scan under `core/` forbids the statics (piece 4).
- GUI tests call `tests/_fixtures.py::qt_app()`, not the unused session `qapp` fixture.
- A green suite does not certify the GPU path. The smoke gate's need is judged against the diff at
  the end (spec §9) and recorded either way.
- Do not pipe large Python or Markdown through a bash heredoc; write the file with the Write tool
  and run it. A foreground `python` check that imports torch and touches the prior or checkpoint
  machinery can hang a tool call for good — write it to a script and run it with a timeout.

**Load-bearing rules**
- Every knob is an ARGUMENT, never a config write: `core/orchestrator.py` binds config constants at
  import (`from .config import X`), so assigning `config.X` at runtime changes nothing and the run
  silently uses the default.
- No public stage, composition or diagnostic mutates the configuration it is handed
  (`core.runs.public_entry` copies on entry — piece 3, V1). **`public_entry` is duck-typed on
  `copy_for_run()`** (`core/runs.py:243-247`): a config class without that method gets the run log
  and no copy. That is why Task 7 adds it to `FDTConfig`.
- Refuse before the spend. Every pre-spend refusal is a `core.refusals.Refusal` (or a `StoreError`,
  its subclass) with a field key. In the converted modules **no message names a box, tab, flag or
  button** — `core/gui/fields.py` and `core/tool/fields.py` do that. A `Field`'s description may not
  contain the words tab, box, flag, button, click, tick or dialog (`tests/test_refusals.py` pins it
  with a regex), and `len(FIELDS)` is pinned by an exact count.
- A refusal about a setting **no front end exposes** is registered in `FIELDS` like any other and
  carries `None` in BOTH front-end tables, so `fix_sentence` returns `""` and the message names the
  setting and offers no fix. Five settings are in that class: `freq_bounds`, `burn_in_nd`,
  `t_obs_periods`, `dt_nd`, `psd_t_obs_nd`. **They are NOT unregistered**: every `require_*` rule
  builds its sentence through `describe(key)`, which raises a bare `KeyError` for a key `FIELDS`
  does not hold (P2, which corrects spec §1.2 and §5.3).
- Messages are `logging` records: `log.info` for banners, timings, tables and progress notices,
  `log.warning` for anything the operator should act on, `log.error` for a failure reported rather
  than raised. `core/tool`'s framing `print`s stay prints (its no-print pin excludes `core/tool`).
- `Resources/` holds the hand-edited inputs; everything generated lives under
  `Artifacts/<kind>/<name>__<id>/` with a `manifest.json` and a `log.txt`. **No code outside
  `core/config.py` builds a literal `Resources/` or `Artifacts/` path** — a test pins it.
- Loading refuses any verifiable mismatch, and `Accept(truncated, other_observation)` are the only
  escape hatches. D11 (no chi override) and D12 (no hatch for a round on a non-amortized parent with
  another observation) are standing refusals no task may re-open.
- The GUI runs ONE task at a time app-wide (`BasePanel._running`, `streams._REDIRECT`), because
  `redirect_streams` swaps `sys.stdout`/`sys.stderr` process-wide.
- The science guardrails are `PRISM_HANDOFF.md` §11.6 (TSNPE) and the traps in §5. It is the science
  reference until piece 6 and is **not** edited by this piece.

**Line numbers move.**
- Every number in a Files block or a step was correct against the tree at `a0d85da` and is a hint,
  not an address. Every task re-derives its line numbers from the file as it reads it, because an
  earlier task in the same file has usually moved them.
- The QUOTED text in a step is the anchor, never the bare number. Find the quoted text; edit what it
  names; ignore the number if the two disagree.
- If the quoted text is not found, re-read the file and **say so in the task report** rather than
  guessing at the line the step meant.

**Git**
- Work directly on the local `main` branch. No feature branches, no worktrees.
- One commit per task, never amended. Subjects are SHORT: one line, a brief body only when the
  subject cannot carry the why. The orchestrator appends the `Co-Authored-By` trailer; a task's own
  commit command does not write one.
- The user pushes and handles every other remote operation.

**Documents**
- `docs/STATE.md` is the moving state: read it first, update it at the end of every session.
- The execution ledger lives under `.superpowers/sdd/2026-09-22-secondary-analyses/` (gitignored).
  Every ruling made during execution goes in it **with what it costs if it is wrong**. Its
  `recon/` and `spec-review/` folders hold the fourteen reports the spec was built and checked
  against; `progress.md` holds rulings R1–R10 from the design pass.
- The display-walkthrough rows this piece adds (**E rows**) are run by the USER on a real screen, at
  the end. **No task may claim them as done.**

## Review Focus

Five input classes the spec implies but that no task's own happy-path tests would exercise, most
likely to bite first. Each line names the task whose tests pin it.

1. **A cancel arriving between the first manifest and the first payload.** The record exists, nothing
   has been written into it, and the cancel is a `BaseException`. The folder must survive (E2) — but
   a *refusal* at the same point must remove it (spec §2.2 step 3), and the two arrive through the
   same `__exit__`. Pinned in **Task 3**.
2. **A second run started while an unfinished record of the same name sits on disk.**
   `assert_name_free` runs at `create()`, and a progressive record occupies its name from its first
   moment — so the second run is refused by name, before it spends anything, rather than colliding.
   Pinned in **Task 3**.
3. **A blank box that reads as zero reaching a check that only refuses negatives.** Every window
   numeric box returns `0` for a blank, so a rule written as "reject below zero" accepts every blank
   box in the application. Each floor is asserted against a blank box, not only against a typed
   zero. Pinned in **Task 10** and **Task 11**. *(Corrected at pre-flight, F34: the draft named Tasks 12 and 13.)*
4. **A cell whose bounds file resolves to the folder's master rather than a same-named sibling.**
   The two resolve differently and only one is recorded; a record that names the wrong bounds file
   is worse than one that names none. Pinned in **Task 7**.
5. **A comparison asked for one record, or for records of different studies.** Every mode's arity is
   a real input class — one cell is not a comparison, and a sweep drawn as a single-cell run reads a
   layout that is not there. Pinned in **Task 36**.

---

## Interface contract

**Every name below is fixed.** A task's implementer sees only their own task; this section is how
they learn the names their neighbours use. If a task's body disagrees with this section, this
section wins, and the task report says so.

### `core/artifacts/manifest.py`

```python
KINDS = ("prior", "simulation", "posterior", "observation", "calibration", "inference",
         "diagnostic", "fdt")

BODY_KEYS = {
    ...,
    # A measurement of a CELL (not of another artifact), or a comparison of such measurements.
    # ``study`` is "single", "sweep" or "comparison". Written PROGRESSIVELY: every key is present
    # from the first manifest, with the not-yet-known ones null.
    "fdt": ("study", "settings", "seed", "grid", "points", "offgrid", "notices", "compared",
            "complete", "results"),
}
```

`config_from_cfg(cfg)` gains a branch, taken when `cfg` has no `observation_mode`, that returns the
FDT settings block.

### `core/artifacts/store.py`

```python
KIND_DIRS = {..., "fdt": "fdt"}
_PARENT_KEYS = {..., "fdt": ()}

#: Kinds whose record is written progressively: the directory and a first manifest exist from
#: ``__enter__``, ``refresh()`` rewrites the manifest and the log as the run proceeds, and an
#: exception KEEPS the directory (E2). The simulation cache has the same property by another
#: mechanism (it has no writer at all).
PROGRESSIVE_KINDS: frozenset = frozenset({"fdt"})

#: Directories that may sit beside the kind directories because an older build wrote them.
#: ``legacy_dirs``/``remove_legacy`` are the only way either front end can see or clear one.
LEGACY_DIRS: tuple = ("crossval",)


@dataclass(frozen=True)
class LooseFile:
    name: str            # the file's own name, never a path
    size: int            # bytes
    mtime: float         # POSIX seconds


@dataclass                       # NOT frozen: Loaded is not, and Python refuses a frozen
class LoadedFdt(Loaded):         # subclass of a non-frozen dataclass at import time (P1).
    body: dict = field(default_factory=dict)
    data_path: "Path | None" = None      # <dir>/data.h5, or None when the run never wrote one


class Summary:                   # gains four fields, all defaulting to None
    study: "str | None" = None
    points_done: "int | None" = None
    points_planned: "int | None" = None
    points_failed: "int | None" = None


class ArtifactWriter:
    progressive: bool            # kind in PROGRESSIVE_KINDS
    _wrote_anything: bool        # True once payload() or figure_path() has been called

    def refresh(self) -> None: ...


class ArtifactStore:
    def load_fdt(self, ref: str) -> LoadedFdt: ...
    def loose_files(self, kind: str) -> "list[LooseFile]": ...
    def remove_loose(self, kind: str, filename: str) -> None: ...
    def legacy_dirs(self) -> "list[str]": ...
    def remove_legacy(self, name: str) -> None: ...
```

`core/artifacts/__init__.py` re-exports `LoadedFdt`, `LooseFile`, `PROGRESSIVE_KINDS` and
`LEGACY_DIRS` beside the existing names.

### `data.h5`'s layout (P5, P6)

Fixed here because T18 writes it and T36-T39 read it, and they are different tasks. The names are
the ones `cross_validation._fdt_measure` and the sweep file already use, so the one function in the
tree written to consume this data needs no translation.

```python
# root attributes, on every data.h5: study ("single" | "sweep" | "comparison"), omega_0, prefactor
# a single-cell record's datasets:
"omega_grid"        # the Campaign-2 probe grid, (n_freqs,)
"T_eff_over_T"      # the ratio at each probe, (n_freqs,), NaN where off-grid
"chi_prime"         # Re chi, (n_freqs,)
"chi_double_prime"  # Im chi, (n_freqs,)
"PSD_omegas"        # the Welch grid, its OWN axis
"PSD_G"             # the spontaneous spectrum on that axis
# a sweep record keeps the per-operating-point group layout load_param_sweep already reads.
# a comparison record: "omega_common" (the common grid) and a "curves" group whose members are
#   zero-padded ordinals ("000", "001", ...), each carrying a `label` attribute. NOT per-id
#   datasets: an id is not a name to rely on for ordering, and the drawers need the label (P71).
```

### `body["compared"]`'s shape (P5)

```python
{"mode": "cells" | "repeats" | "renormalise" | "sweeps",
 "records": [{"kind": "fdt", "id": "<id>", "name": "<name>"}, ...]}
```

The per-record `kind` is carried deliberately, so `render_lineage` can print `MISSING fdt [<id>]`
without assuming one.

### `core/gui/widgets/artifact_picker.py` (P9)

```python
class StorePicker(QWidget):
    def __init__(self, kind: str, allow_new: bool = False, store=None,
                 row_filter=None, parent=None):
        """``row_filter``: Callable[[Summary], bool] | None, applied AFTER the finished rule.
        Before ``parent`` so existing positional calls are unaffected."""
```

T25 passes `row_filter=lambda s: s.study == "single"`, T26 `"sweep"`. Both expose the widget as
`self.record_picker` (P61).

### `core/tool/__init__.py` (P10)

```python
#: Set by smoke.register alone, through set_defaults(temp_store_root=True). Read with
#: getattr(args, "temp_store_root", False). It replaces hasattr(args, "store_root") for the
#: throwaway-root and the empty-root-cleanup behaviours, so fdt/crossval can declare the flag
#: without inheriting either (spec 6.1, E11).
args.temp_store_root: bool
```

### `core/FDT/sanity.py` (P7)

```python
def _resolved_span(omegas) -> tuple:
    """A spectrum's first strictly POSITIVE frequency and its last. (inf, -inf) for an empty grid.
    The leading underscore matches _interp_log, which two other modules already import."""
```

### `core/refusals.py` (P8)

```python
def missing_values_phrase(label: str, missing) -> str:
    """The one wording for "the cell does not carry what the bounds file requires": a lower-case
    fragment, no trailing period, so a caller may capitalise it or join it into a problem list.
    Here and not in core/cli.py because core/sim_config.py cannot import core/cli.py."""
```

### `core/FDT/compare.py` (P11)

```python
@public_entry
def compare(mode: str, refs: list, *, name: str = "", note: str = "", store,
            fig_sink=None, **options) -> "LoadedFdt":
    """ONE public entry for all four modes; the per-mode drawers live in ``_DRAWERS``. One entry,
    because the public_entry scan's three sets are closed by EQUALITY and four entries would mean
    four tasks editing them (P11)."""

_DRAWERS: dict          # mode -> drawer; a mode with no drawer is refused, naming what this build draws
```

### Seeds (P12, P13, P14)

- `run_fdt(seed=…)` and `run_param_study_cli(seed=…)` **override** `cfg.seed`; `None` falls back to
  `cfg.seed`; `None` in both means draw one.
- A drawn seed is drawn from `[0, 2**31)`, so it is typeable into the house `IntField`.
- The sweep's seed travels on `cfg.seed`. `run_param_study_cli` resolves it once on its private
  copy; `run_fdt_param_sweep` reads it and writes nothing. Phase A point *k* runs under `seed + k`,
  Phase B point *k* under `seed + n_points + k`.

### The first body (P15)

The **caller** sets `study`, `seed`, `notices: []`, `complete: False` and the not-yet-known keys as
`None`. The **stage** fills `settings` from its private copy before `__enter__`, and **updates
`writer.body` in place** rather than replacing it — replacing it would drop the study and the seed.

### `core/artifacts/report.py`

`render_lineage` gains a `compared` branch: for a record whose `body["compared"]` is not null, each
listed id is resolved through `store.get` and printed `MISSING <kind> [<id>]` when the store no
longer holds it — the same wording the parents walk already uses.

### `core/sim_config.py`

```python
@dataclass
class FDTConfig:
    ...
    sources: dict = field(default_factory=dict)   # {"cell": str, "bounds": str|None,
                                                  #  "units": str|None, "model": str}
    seed: "int | None" = None
    preset_name: "str | None" = None              # the sweep's preset name (P72); None for a single run

    def copy_for_run(self) -> "FDTConfig":
        """A private deep copy. ``public_entry`` is duck-typed on this method
        (core/runs.py:243-247): without it the decorator gives the run log and NO copy."""
```

### `core/cli.py`

```python
def cell_sources(cell_file: str, model: str) -> dict:
    """The ``sources`` block for a cell: its own path, the bounds file that RESOLVES for it
    (``resolve_bounds_for_cell``; None on the legacy inline-bounds branch), the units file, and the
    model NAME. Beside ``parse_cell``, which keeps its 7-tuple."""

def make_fdt_config(model, state_dep_drift, cell_file, *, n_freqs=60, ensemble_M=256,
                    freqs_per_batch=1, F0=0.05, seed=None) -> FDTConfig: ...

def make_param_sweep_config(cell_file, *, preset, preset_name, s_spec, t_spec,
                            n_freqs=None, ensemble_M=None, freqs_per_batch=None, F0=None,
                            seed=None): ...
```

### `core/refusals.py`

```python
def require_below(key: str, lo, hi) -> tuple:
    """A blank, a non-finite value, or ``lo >= hi`` is refused; the pair is returned as floats.
    Message: "<what> must have its lower bound below its upper bound (got <lo> and <hi>)"."""
```

New `FIELDS` keys, in this order: `n_freqs`, `ensemble_m`, `freqs_per_batch`, `f0`, `preset`,
`s_grid`, `t_grid`, `seed`, then the Simulate, Reduction and model-builder keys each task names.

### `core/gui/fields.py`

```python
#: Place strings that name a SCREEN rather than an inference tab. ``fix_sentence`` renders
#: "on the <place> screen" for these and "on the <place> tab" for everything else. The tab titles
#: outside Parameter Inference are ordinary tab entries: "FDT analysis",
#: "Sweep study cross-validation", "Live simulation", "NWK → Hopf reduction map".
SCREENS: frozenset = frozenset({"Artifacts", "Model Builder"})
```

`CONTROL`'s tuple entry keeps its `(place, label)` shape and `place` may still be a tuple of several
places. `label(key)` is unchanged.

### `core/FDT/fdt_pipeline.py`

```python
@public_entry
def run_fdt(cfg: FDTConfig, *, skip_sanity: bool, confirm_production: bool,
            writer: "ArtifactWriter", seed: "int | None" = None) -> "LoadedFdt":
    """Takes the OPEN-BUT-NOT-ENTERED writer and does ``with writer:`` itself, so __enter__, every
    refresh() and __exit__ run on the thread whose RunLog will become log.txt."""
```

### `core/FDT/cross_validation.py`

```python
@public_entry
def run_param_study_cli(cfg, *, s_grid, t_grid, writers: dict, seed=None) -> "list[LoadedFdt]":
    """``writers`` is {"s": ArtifactWriter, "temp": ArtifactWriter} — one record per swept
    parameter (spec §4.1). One seed is drawn once and recorded on both."""

def run_fdt_param_sweep(cfg, sweep_param: str, sweep_grid, fixed_overrides=None, *,
                        writer: "ArtifactWriter") -> "LoadedFdt": ...
```

### `core/FDT/compare.py` (new, Task 36)

```python
def common_grid(curves: list) -> "np.ndarray":
    """Log-spaced over the INTERSECTION of the curves' spans, with N the smallest ``n_freqs``
    among them (spec §7.2)."""

def interpolate_onto(grid, omegas, values) -> "np.ndarray":
    """Linear in log-omega. A target point whose bracketing samples include a blank is blank."""
```

---

## Task order and dependencies

```
  THE STORE
 T1  the kind's declarations + the three silent pins ───────┐
 T2  Summary grows, the two column tables, numeric sort  ── T1
 T3  the writer's progressive mode + refresh()            ── T1 T2  (P81)
 T4  loose files and legacy directories                   ── T1
 T5  render_lineage's compared branch                     ── T1
 T6  the delete confirmation names what is half-written   ── T2 T3

  THE SETTINGS OBJECT AND THE RUN ENTRIES
 T7  FDTConfig: sources, seed, copy_for_run; cell_sources; config_from_cfg ── T1
     (the TWO FDT builders fill sources; make_reduction_config is NOT touched — P19)
 T8  require_below + the new field keys + the tool table ──┐
 T9  core/gui/fields.py widened to screens                 ── T8
 T10 make_fdt_config's checks                              ── T7 T8
 T11 make_param_sweep_config's checks + preset_name        ── T7 T8 T10
 T12 the thin-setting notices + the config.py thresholds   ── T10 T11

  THE SCIENCE FIXES
 T13 the prefactor moves to the start + field keys         ── (independent: its keys exist — P18)
 T14 the off-grid range test + the blanks count            ── (independent)
 T15 the band refusal before the driven campaign           ── T13 T14
 T16 nothing measurable is a refusal (single-cell)         ── T14 T15

  THE TWO RUNS BECOME RECORDS
 T17 run_fdt as a public entry writing a record  ── T3 T7 T10 T13 T15 T16
     (AND the public_entry source scan's three sets — same task, or the gate reds)
 T18 the numbers into data.h5 (single)                     ── T17
 T19 the sweep: two records, one seed, public entry        ── T3 T7 T11 T17
     (AND the source scan again)
 T20 the sweep's failure counts, the all-failed refusal, the nanmax defect, _fdt_measure removed ── T19

  THE FIVE SURFACES
 T21 the four panels' refusal routing; _config_error removed; parse_cell's two ValueErrors ── T9
 T22 the model-builder screen through show_refusal         ── T9 T21
 T23 Simulate's clamps and its observation length          ── T8 T9 T21
 T24 StorePicker: a row predicate, and `finished` not `complete` ── T2
 T25 the FDT panel: picker, Seed box, creates the writer, watches figures/ ── T17 T24
 T26 the CrossVal panel: the same                          ── T19 T24
 T27 showing an earlier run: settings, seed, notices, figures ── T25 T26

  THE COMMAND LINE
 T28 --store-root on both, and the three behaviours        ── T17 T19
 T29 refusals with field keys, the interrupt notes, the folder branch ── T28
 T30 artifacts sweep: loose files and the legacy directory ── T4
 T31 the Artifacts screen's sweep: the same categories     ── T4 T30

  THE GAPS AND THE ROUND TRIPS
 T32 the three plots.py functions' figure properties ──────┐ (independent)
 T33 the Nadrowski sanity end-to-end (slow)                 ── T17
 T34 the tiny-size test rewritten                           ── T17 T19 T28
 T35 Simulate and model-builder round trips                 ── T22 T23

  COMPARING (last: nothing above depends on it)
 T36 core/FDT/compare.py: the record, the common grid       ── T18 T20
 T37 compare cells and compare repeats                      ── T36
 T38 compare renormalise                                    ── T36
 T39 compare sweeps                                         ── T36
 T40 the two screens' comparison controls                   ── T25 T26 T37 T38 T39

  THE CLOSE
 T41 the documents, the walkthrough rows, the gates of record ── everything
```

**Corrected edges (pre-flight F5, F8 and P81).** The graph above omits dependencies the tasks' own anchors and tests rely on. The complete set is: T3 ── T1 T2; T4 ── T1 T3; T15 ── T8 T13 T14; T17 ── T3 T7 T10 T12 T13 T15 T16; T19 ── T3 T7 T11 T12 T17; T21 ── T9 T10 T17 T19; T25 ── T9 T17 T21 T24; T26 ── T9 T19 T21 T24; T33 ── T17 T28; T34 ── T17 T18 T19 T20 T28; T36 ── T8 T9 T18 T20 T22 T23; T38 ── T36 T37; T39 ── T36 T37. **The tasks execute strictly in number order, one at a time**, so every edge above is satisfied by construction; the list documents what each task consumes.

**T17 and T19 each carry the `public_entry` source scan.** `tests/test_artifact_store.py`'s scan
walks `CODE_ROOTS = ("core",)` and asserts `found == want` by equality, with `_UNTOUCHED_LEGS` and
`_REFUSAL_FIELDS` closed the same way. The moment a decorator lands without the scan being updated
in the same commit, the fast gate reds. Neither task may defer it.

---
## Rulings made at planning time

Before any code was written this plan was read twice over, and **eighty-two rulings** came out of it.

First, ten drafters read the code for their own tasks and raised **69 objections** to the interface
contract, to the spec and to the task order — 6 blocking, 31 important, 32 minor. Every one is ruled
on below as **P1–P69**. Their texts, each anchored in the real code, are in the ledger
(`.superpowers/sdd/2026-09-22-secondary-analyses/objections.md`).

Then the assembled plan was read by three more lenses — spec coverage, cross-task name and type
consistency, and "would the gate be green after every task" — and a judge that re-checked each
finding against the real code before promoting it. Thirteen survived, and are **P70–P82** in the
next section. They sit on the seams between the drafters' groups, which is what a parallel draft
costs and what that pass buys back. Those reports are in the ledger's `plan-review/`.

Every ruling carries what it costs if it is wrong.

**If a task's text disagrees with a ruling, the ruling wins** — and say so in the task's report, so
the plan can be corrected rather than quietly diverged from.

**Ten of these correct the spec itself** rather than deviating from it: P2, P4, P9, P10, P17, P26,
P32, P47, P48 and P53. They are applied to the spec inline, each carrying its P-number, and named
again in its §12's preamble. **Nine more extend the design beyond what the spec says** and are
§12's rows 1–9: P3, P11, P22, P23, P27, P28, P30/P31, P55 and P60.

### Corrections after the plan review (P70–P82)

The assembled plan was then read by three more independent lenses — spec coverage, cross-task name
and type consistency, and "would the gate be green after every task" — and a judge that re-checked
every finding against the real code before promoting it. Verdict: **ready with fixes**, thirteen of
them. All thirteen are below. They were found because ten drafters wrote in parallel, and they sit
exactly on the seams between their groups — which is what a parallel draft costs and what this pass
buys back.

**These supersede the task bodies wherever the two disagree**, on the same rule as P1–P69.

- **P70 — `body["settings"]` carries `skip_sanity` and `confirm_production`, so `_settings_block`
  must be given them.** They are arguments of `run_fdt`, not attributes of `FDTConfig`, so the
  config-only helper T17 writes can never produce them — while T17's own test, T33's and T34's all
  assert them. T17 Step 6's signature becomes
  `def _settings_block(cfg, *, skip_sanity=None, confirm_production=None) -> dict:`, appending
  `"skip_sanity"` and `"confirm_production"` as `None`-or-`bool`. T19 calls it with neither, so a
  sweep record records both as null. *Cost if wrong: two body keys read null on a single-cell run.*

- **P71 — the single-cell payload follows the CONTRACT (`omega_grid`, `chi_prime`,
  `chi_double_prime`), and the comparison payload follows T36 (`omega_common` + `curves/<NNN>` with
  a `label` attribute).** T18 wrote `omegas` plus one complex `chi`, which no drawer can read; the
  contract's names are the ones `_fdt_measure` and the sweep file already use, which is why P5
  ratified them. The comparison layout goes the other way, because an id is not a name to rely on
  for ordering and the drawers need the label. **The contract is corrected for the second; T18 is
  corrected for the first.** T34's tiny-size run gains one assertion that `compare` can read the
  record a REAL single-cell run wrote — without it the only reader is the fixture, and the fast gate
  cannot see the mismatch at all. *Cost if wrong: every comparison of a real record refuses.*

- **P72 — `FDTConfig` gains a THIRD defaulted field, `preset_name: "str | None" = None`.** T11
  accepts `preset_name` and uses it only to validate; nothing carries it to the record, so
  `body.settings["preset"]` — the one thing §4.4 requires — would always be null and both T19's and
  T34's assertions would fail. T11 passes it into the construction and pins it; T19 reads
  `cfg.preset_name` directly and drops its hedging `getattr`. A defaulted field, so §1.3's
  reduction-map constraint holds. *Cost if wrong: one more field on a shared dataclass.*

- **P73 — T17 owns `tests/test_nav_and_gating.py`'s FDT panel test.** Making `writer` a required
  keyword of `_run_fdt_guarded` breaks the existing direct caller and its two two-keyword stubs.
  T16's step correctly says that test stubs `run_fdt` wholesale and is untouched — true for T13–T16,
  false for T17, which changes the GUARD. The file joins T17's Files block, its run command and its
  `git add`, and a numbered step gives `boom` and `missing` the `writer` and `seed` keywords.
  *Cost if wrong: the fast gate reds at T17, which the Global Constraints forbid.*

- **P74 — T17's `fdt_pipeline.py` anchors are against the PRE-T16 file and must be rewritten.**
  T16 (P22) already moved `plot_psd` before Campaign 2 and shortened the closing block to two paths.
  T17 then deletes the `timestamp` binding while an early `psd_path = _out_dir() / f"psd_{timestamp}.png"`
  still references it — a `NameError` on every run — and quotes a three-line closing block and a
  three-path log line that no longer exist. T17 Step 8 gains a find/replace for the early site
  (`psd_path = writer.figure_path("Spontaneous PSD")`), its closing-block anchor becomes the
  two-line `ratio_path`/`chi_path` pair, and Step 9's anchor becomes the two-path record. T17's brief
  gains: **"T16 has already moved `plot_psd`; there is no `psd_path` in the closing block when you
  arrive."** *Cost if wrong: two anchors stall the implementer and the spontaneous spectrum lands
  outside its record.*

- **P75 — P2 STANDS, and the tasks are corrected to it.** The judge proposed withdrawing P2 on the
  ground that its premise never arises, because the tasks as drafted build their own sentences and
  never call a `require_*` rule with an unregistered key. That is true of the drafts and is exactly
  why they must change: **building the sentences by hand in `core/cli.py` is the duplicated wording
  three pieces of this programme were spent removing**, and it is what P2 overruled in the first
  place. So: T8 registers all five (`FIELDS`, `BASE_KEYS`, `CONTROL = None`, `FLAG = None`), drops
  its "deliberately absent" comment, and reads its count literal off the failing pin rather than
  writing 71 blind; T10 deletes its `_positive`/`_at_least` helpers and calls
  `require_positive("dt_nd", …)`, `require_at_least("burn_in_nd", …, 0)` and
  `require_below("freq_bounds", lo, hi)`, asserting the keys instead of `None`; T15's band refusal
  carries `field="freq_bounds"`; T41's `docs/STATE.md` text follows P2's wording, not its first
  draft's. **These five are the first keys with neither a control nor a flag, and `fix_sentence`
  returning the empty string for them is the intended behaviour** — the message names the setting
  and offers no fix, because there is nothing to name. *Cost if wrong: five registry entries and two
  table entries to remove, and one count literal. Nothing outside the refusal text moves. The reason
  to accept that cost is that `freq_bounds` is the subject of E9's own refusal and is the likeliest
  of the five to gain a control, at which point a registered key already behaves.*

- **P76 — T29 does NOT produce a message builder; T21 does.** T29 as drafted introduces
  `cell_missing_message` and rewrites the two sites T21 has already converted to
  `missing_values_phrase`, against anchors T21 has already replaced — which would either add a
  second builder for one rule or revert T21, and would red T21's source-scan pin. P4 already
  overruled the name. **T29's Steps 14–17, its `cell_missing_message` interface line, and
  `core/refusals.py`, `core/sim_config.py` and `core/cli.py` from its Files and `git add` lines are
  deleted.** T29 keeps the model gate, the interrupt notes and the two "no bounds file" sentences.
  The cross-site equality T29 wanted to assert is wrong in either design and is replaced, in T21, by
  `assert exc.value.message == f"Cell file is {problems[0]}."` — the dry run's fragment is
  deliberately lower-case and the refusal a capitalised sentence. *Cost if wrong: one rule worded
  twice, which is the defect §6.2 exists to close.*

- **P77 — an all-failed first sweep must not cost the second.** The spec says so twice and demands a
  test, and no task delivered either: T20's all-failed `Refusal` propagates straight out of
  `run_param_study_cli`, so the temperature sweep never starts — today's behaviour with a calmer
  message — while T34's rewritten docstring claims the opposite, so the piece would ship a false
  statement in a test. T20 gains a step: each `run_fdt_param_sweep` call is wrapped, a `Refusal` is
  logged at error and recorded, the second sweep runs regardless, and only if **both** measured
  nothing does the study refuse. Its unfinished record stays on disk (E2). T20 also gains §8.2's
  test: fail every point of the S grid only, then assert the T record is finished and the S record is
  on disk and unfinished. *Cost if wrong: the piece's headline sweep fix does not exist.*

- **P78 — a finished sweep record fills `results` and `offgrid`.** Both are set to `None` in the
  first body and nothing ever writes them, so §2.3's "null only until the run finishes" is false for
  a sweep for ever. T20 gains a step at the clean end of `run_fdt_param_sweep`: aggregate the usable
  points into `results` in T17's `_results_block` shape, and accumulate the per-point blank counts of
  the common grid into `offgrid = {"blanks": <total>, "of": n_points * len(omegas_common)}`. Its
  failure-count test asserts both are non-null. *Cost if wrong: a sweep record is less informative
  than its own body table promises.*

- **P79 — `fdt` and `crossval` actually gain `--seed`.** E7's command-line half had no owner: T8
  registers the flag name, the builders and entries accept `seed=`, both panels grew a Seed box — but
  no task adds `add_argument("--seed", …)` and no handler passes `args.seed`, and the tool-table pin
  stays green only because `smoke` and the diagnostics already define that option string. T28's
  `_add_fdt_knobs` gains the argument; T17's and T19's handler edits pass `seed=args.seed`; T11's
  tool call site passes it too; and T28 or T29 pins it with a real `--seed 7` run whose recorder sees
  7. *Cost if wrong: a recorded seed cannot be supplied back, which is the exact defect E7 names.*

- **P80 — T25's and T26's panel anchors are the POST-T17 text.** P3 amended their briefs but not
  their bodies: they still quote the pre-T17 dispatch block, guard signature and `run_fdt` call, and
  ask for replacements byte-identical to T17's. Their anchors become
  `writer = default_store().create("fdt", cfg)` and the `writer=writer, watch_dir=writer.dir / "figures"`
  dispatch, and they add only what is new — `name=`/`note=`, the Seed read, `seed=seed`,
  `on_result=self._on_record`, and the record picker. Their duplicate guard-signature and `run_fdt`
  edits are deleted. Each brief gains: **"the writer creation and the guard's `writer`/`seed`
  keywords already exist (P3); if they do not, that is a T17 defect and goes in the report."**
  *Cost if wrong: three anchors stall the implementer and the new work is re-derived by hand.*

- **P81 — T3 depends on T2, not on T1 alone.** T3's tests assert `Summary.finished` for the `fdt`
  kind, which stays `True` until T2 replaces `store.list`'s `kind == "simulation"` branch; with both
  declaring only T1, a dispatcher may run T3 first and three of its assertions fail (one of them
  vacuously, which is worse). The task-order line is corrected above.

- **P82 — the seed must be shown to DETERMINE the numbers.** Two of §8.2's four seed tests had no
  owner: "two runs with the same seed agree; two runs with different seeds differ", and "a point is
  reproducible from the seed and its index". Everything drafted asserts only that a seed is recorded
  — which is not E7's claim, and is the premise `compare repeats` rests on when E8 calls the spread
  across repeats the measurement error. T17 gains a step whose stub draws from the ambient generator,
  runs twice at one seed and once at another, and asserts equal then unequal. T20 gains the
  per-point version: point *k* under `seed` draws what a single point under `seed + k` draws.
  *Cost if wrong: the piece records a seed that might mean nothing.*

### The six blocking ones

- **P1 (O1, O5) — `LoadedFdt` is a plain `@dataclass`, not frozen.** `Loaded` is not frozen, and
  Python raises `TypeError: cannot inherit frozen dataclass from a non-frozen one` at import, which
  would kill every import in the tree. The contract is corrected; `body: dict = field(default_factory=dict)`
  and `data_path: "Path | None" = None`, matching its six siblings. `LooseFile` inherits nothing and
  stays frozen. *Cost if wrong: none — this is a language rule, verified.*

- **P2 (O2) — the five non-knob settings ARE registered in `FIELDS`, with `None` in BOTH front-end
  tables.** The drafter is right that `require_positive("dt_nd", …)` raises a bare `KeyError` from
  inside `describe()` for an unregistered key, so the spec's "checked with `field=None` and no table
  entry" is unsatisfiable. **But its fix — a helper in `core/cli.py` reproducing the rules' wordings
  — is overruled:** duplicated wording is the thing this programme spent three pieces removing. The
  registry already has the shape needed. `core/gui/fields.py` documents `None` as "a key the window
  has no control for" and `core/tool/fields.py` carries six `None` flags today. So `freq_bounds`,
  `burn_in_nd`, `t_obs_periods`, `dt_nd` and `psd_t_obs_nd` join `FIELDS` with `None` in both
  tables, the ordinary `require_*` rules are used, and `fix_sentence` returns `""` for them — which
  is exactly the behaviour the spec wanted, through the existing mechanism. **Spec §1.2 and §5.3 are
  corrected.** *Cost if wrong: five registry entries and two table entries to remove; no behaviour
  outside the refusal text.*

- **P3 (O3) — T17 and T19 each carry the minimal bridging edits to their own call sites.** A
  required `writer` keyword with the two call sites left behind reds the fast gate, which the Global
  Constraints forbid after any task. The bridging shape is the spec's own (§1.2: the front end
  creates, the stage enters), so T25/T26 add the Seed box, the pickers and the name and note
  controls **on top of** working code rather than undoing a temporary wrong shape. **T25's and T26's
  briefs are amended: the writer creation already exists when you arrive.** *Cost if wrong: T25/T26
  find the code already there and say so; nothing is lost.*

- **P4 (O4, O30) — the spec is WRONG that `cli.parse_cell` raises bare `ValueError`s.** Verified:
  `grep -n "raise ValueError" core/cli.py` returns nothing, and both mistakes
  `BasePanel._config_error`'s docstring names already raise `Refusal(field="cell")` — piece 3
  converted them. So T21 has nothing to convert, and the justification §5.1 attaches to the
  conversion is already satisfied: those two reach the yellow box the moment T21's routing lands.
  **What the conversion was there to buy is still owed**, and T21 delivers it directly:
  `core.refusals.missing_values_phrase(label: str, missing) -> str`, a lower-case fragment with no
  trailing period, spliced at the two sites §6.2 names. It lives in `core/refusals.py`, not
  `core/cli.py`, because `core/sim_config.py` cannot import `core/cli.py` (the import runs the other
  way). **O30's alternative name `cell_missing_message` is overruled** — one name, first proposed,
  and T29 consumes it. **Spec §5.1 and §6.2 are corrected.** *Cost if wrong: a rename across three
  call sites.*

- **P5 (O6, O35, O45) — the payload layout and the `compared` shape are fixed in the contract.**
  Both were genuinely undefined and would have met only at the walkthrough. The drafters' choices
  are ratified because they reuse the vocabulary the one would-be reader already expects
  (`_fdt_measure` and the sweep file both already write `omega_grid`, `T_eff_over_T`, `chi_prime`,
  `chi_double_prime`, `PSD_omegas`, `PSD_G`). See the contract addendum below. *Cost if wrong: a
  rename inside `core/FDT/compare.py`'s constants and `build_fdt_record`.*

- **P6 (O6) — `data.h5` carries root attributes `study`, `omega_0` and `prefactor`**, so a reader
  can check the layout it holds before reading a dataset. Ratified.

### The interface contract, corrected and extended

These are additions to the contract; every task inherits them.

- **P7 (O14)** — `core.FDT.sanity._resolved_span(omegas) -> tuple[float, float]`: a spectrum's first
  strictly positive frequency and its last. Empty grid returns `(inf, -inf)`. Consumed by T14, T15,
  T16 and T17.
- **P8 (O21, O30)** — `core.refusals.missing_values_phrase(label: str, missing) -> str`. See P4.
- **P9 (O22, O25)** — `StorePicker(kind, allow_new=False, store=None, row_filter=None, parent=None)`
  with `row_filter: Callable[[Summary], bool] | None`, applied **after** the `finished` rule. T25
  passes `row_filter=lambda s: s.study == "single"`, T26 `"sweep"`. The parameter goes before
  `parent` so existing positional calls are unaffected. **The spec's §5.4 gains the name.**
- **P10 (O29)** — `args.temp_store_root: bool`, set by `smoke.register` alone through
  `set_defaults(temp_store_root=True)` and read with `getattr(args, "temp_store_root", False)`. It
  replaces `hasattr(args, "store_root")` for the throwaway-root and the empty-root-cleanup
  behaviours. **The spec's §6.1 gains the name.**
- **P11 (O36, O37)** — the comparison facility is **ONE** public entry,
  `core.FDT.compare.compare(mode, refs, *, name, note, store, fig_sink=None, **options)`, with
  per-mode drawers in `compare._DRAWERS`. T36 registers the whole `compare` subcommand — all four
  modes and every flag — and T37–T39 add drawers only. This is forced: the tool-table pin walks the
  real parser, so a `FLAG` entry whose option string no subcommand defines fails immediately; and
  the `public_entry` source scan's three equality-closed sets would otherwise be edited by four
  tasks. *Cost if wrong: T36 is larger than its neighbours; the alternative reds the gate three
  times.*
- **P12 (O24)** — **seed precedence: `run_fdt(seed=…)` and `run_param_study_cli(seed=…)` override
  `cfg.seed`; `None` falls back to `cfg.seed`; `None` in both means draw one.** Both panels read
  their box once and pass the same integer to the builder and to the run, so either honouring gives
  the owner what they typed.
- **P13 (O28)** — **a drawn seed is drawn from `[0, 2**31)`**, so it is typeable into the house
  `IntField`, whose validator is `QIntValidator()`. A recorded seed the owner cannot type back would
  defeat E7's stated purpose. The Seed box is read through a blank-preserving accessor: blank means
  draw one.
- **P14 (O18)** — the sweep's seed travels on `cfg.seed`, which T7 adds for exactly this.
  `run_param_study_cli` (a public entry, working on its private copy) resolves it once;
  `run_fdt_param_sweep` **reads** it and writes nothing, because it is not decorated and gets no
  copy. Phase A point *k* runs under `seed + k`, Phase B point *k* under `seed + n_points + k`.
- **P15 (O27)** — **the STAGE fills `body["settings"]`, not the front end**, and it updates
  `writer.body` **in place** rather than replacing it (replacing it would drop the study and the
  seed the caller set). The caller sets `study`, `seed`, `notices: []`, `complete: False` and the
  not-yet-known keys as `None`. The stage fills `settings` from its private copy before `__enter__`.
  Five of the settings are not front-end knobs (P2), so the front end could not fill them.

### The task order, corrected

- **P16 (O8)** — **T1 carries `Summary`'s four new fields as declarations and both column tables and
  both cell renderers**; T2 makes `store.list` fill them, changes the `finished` branch and adds the
  numeric sort key. Three closed-set pins go red the instant `KIND_DIRS` gains a key, so T1 as
  originally scoped would leave the gate red. Merging T2 into T1 would make one thirty-step task and
  lose the independently reviewable sort fix. *Cost if wrong: T1 is a long task.*
- **P17 (O32)** — **T32 is a TESTS-ONLY task.** All four drawing functions already close a saved
  figure (`bb22ac4` touched all four); what is missing is three tests. The spec's §8.2 bullet
  compresses `docs/STATE.md`'s correct statement misleadingly and **is corrected**. T32's step 2
  predicts PASS with the reason, and the task proves the tests have teeth by deleting
  `plt.close(fig)` from one function, watching the test fail, then restoring with
  `git checkout --` and verifying a clean tree. `core/FDT/plots.py` is excluded from its commit.
- **P18 (O50)** — T13 depends on nothing in T8: its two keys, `cell` and `model`, are already
  registered. The dependency arrow is dropped.
- **P19 (O47)** — **only the two FDT builders fill `sources`; `cli.make_reduction_config` is not
  touched.** The task order's "the three builders" was the plan author's error against the spec's
  own §1.2 and §1.3, which say twice that the reduction builder is left alone. **The task order line
  is corrected.** T7 still pins that the reduction builder builds a working settings object.
- **P20 (O23)** — **T22 and T23 each add their own field keys in full** — `core/refusals.py`, both
  front-end tables, and both test literals — so each task is independently green. The exact registry
  count is never written blind: each task runs the pin once and reads the real number off the
  failure. Whichever task reaches `tests/test_refusals.py:805` first converts the equality to a
  subset assertion; the second finds the quoted text absent and **reports it rather than adding a
  second conversion**.
- **P21 (O41)** — **T41 is two commits, and that is deliberate**, the one exception to
  one-commit-per-task: the documents and docstrings first, then the gates of record, whose measured
  numbers are written into `docs/STATE.md` afterwards. A gate cannot measure a tree that does not
  yet hold the edits it is measuring.

### The science path

- **P22 (O15)** — **`plot_psd` moves to just before Campaign 2.** Spec §3.6 promises the unfinished
  folder holds the spontaneous spectrum, and today that figure is drawn at step 9, after both
  refusals could fire. Without the move the promise is unreachable and the folder holds only a time
  series. Verbatim move, including the continuation indentation; the closing block loses it and the
  "Saved plots to:" record shortens to two paths. *Cost if wrong: one figure is drawn earlier in
  every run — no numbers change.*
- **P23 (O16)** — `check_passive_baseline`'s bare `ValueError` becomes reachable from the low end for
  the first time once T14 lands. **It is converted only if T33 goes red on it**, and then in T33,
  as `Refusal(..., field=None)` — not deferred to a later piece. *Cost if wrong: one test fails
  loudly in the piece that caused it, which is the point.*
- **P24 (O13)** — T13 makes `run_fdt` read `cfg.params_dict` before its first log record, which reds
  `tests/test_fdt_user.py`'s record test through a stub that carries only a model name. The stub
  gains `params_dict`, as a numbered step of T13, and **the new prefactor call logs nothing**, so
  that test's exact-equality record list stays a pin on the pipeline rather than on this task.
- **P25 (O51)** — the field keys go on **every** `FDTModelError` raise site §1 lists, including
  `_make_simulator`'s, not only `observable_noise_prefactor`'s. A refusal with no key is the state
  V3 exists to remove.
- **P26 (O48)** — **the core model refusal carries `fdt_support`'s reason and nothing else; the
  "where the name came from" hint moves to `core/tool/fields.py`'s entry for `model`.** A core
  message may not name a flag — the source scan pins it — and the spec §3.3 row that told it to keep
  the hint **is corrected**.
- **P27 (O19)** — **`run_param_study_cli` gets §3.4's normalisation check at its top**, and
  `_REFUSAL_FIELDS["run_param_study_cli"] = "cell"`. The scan's three sets are closed by equality and
  demand a refusal leg; the justification is §3.4's own — a cell that cannot supply the constant
  would otherwise cost the whole first phase. This goes beyond the spec's text and into its §12.
- **P28 (O53)** — **`plot_fdt_3d_vs_param` must take its destination**, not build one under
  `<artifacts root>/crossval`. Otherwise the sweep keeps writing into the very directory E10 offers
  for tidy-up, and a "legacy" directory refills itself. Added to T19's scope. *Cost if wrong: a
  figure lands outside its record and the tidy-up offers it — the exact confusion E10 exists to
  end.*

### The screens

- **P29 (O20)** — **the Simulate panel's two bare `ValueError`s are converted too**, to
  `Refusal(field="cell")` and `Refusal(field="model")`. The spec carves out the Reduction panel by
  name but the identical hazard sits on Simulate, and removing `_config_error` without converting
  them moves both to the red crash box. Both keys already exist; `Refusal` subclasses `ValueError`,
  so the existing `except ValueError` in `tests/test_user_models.py` still catches. The second
  message also stops naming the Settings screen, which a core message may not do.
- **P30 (O26)** — **both panels gain a "Record name" and a "Note" box**, neither persisted (a
  remembered name is refused by `assert_name_free` at the next launch's first click), and **T9
  widens `CONTROL["name"]` and `CONTROL["note"]`** to list the two new tabs beside the places they
  already name. Without this, `store.create(name=…)` has no source and Review Focus item 2 has
  nothing to test. *Cost if wrong: two controls to remove.*
- **P31 (O60)** — **a two-record study's name box gives a STEM**: the records are `<stem>-s` and
  `<stem>-temp`. Two records cannot share one name. A blank stem leaves both unnamed, as elsewhere.
- **P32 (O10)** — **`units` does not collide and is not widened.** There is exactly one units
  control in the application; on the four other screens the units file is resolved from the model
  with no control at all, so widening would produce a fix sentence naming a box that does not exist.
  `cell` (five places) and `model` (three) are widened. **Spec §5.2 is corrected** — it names three.
- **P33 (O55)** — the Simulate panel's observation length is the **existing** `t_obs` key, so
  checking it there makes a fourth colliding input. T9 widens `t_obs` with the others.
- **P34 (O58)** — the FDT and CrossVal panels **do** build their row labels from
  `core.gui.fields.label(key)`, as the inference tabs do. §5.2's stated payoff — that a label and its
  hint sentence cannot drift apart — lands only if they do.
- **P35 (O59)** — the sweep writes two records, so the watcher is pointed at the **first** record's
  `figures/` and re-pointed when the second opens. T26 owns the re-point and states it.
- **P36 (O54)** — "each forcing field" is **one** registry key, `forcing_value`, whose control entry
  is a sentence naming the forcing table, not one key per parameter: the rows' labels are the
  parameter names, built at run time from the model, so a tuple entry per parameter cannot exist.
- **P37 (O56)** — the two existing picker tests stub rows as namespaces carrying only what
  `refresh()` reads today; T24 adds `finished` to both stubs as a numbered step, so the change fails
  as an assertion rather than as an `AttributeError`.
- **P38 (O57)** — `_fill_checked`'s `label` is spliced into two messages and its callers pass plural
  labels; `missing_values_phrase` takes the label as given and does not pluralise. The out-of-bounds
  message is left alone — it is a different rule.

### The command line and the documents

- **P39 (O7, O61)** — **T41 owns both falsified texts** that no earlier task's tests would red:
  `core/tool/fdt.py`'s two "no bounds file" sentences and `core/tool/browse.py`'s `--store-root`
  docstring. T7 and T29 do not carry them. They are steps of T41 because T41 is the only task that
  cannot forget them. **Superseded in part (P76, pre-flight F50):** Task 29 corrects `core/tool/fdt.py`'s two sentences, because it already rewrites that file; Task 41 keeps `browse.py`'s docstring only.
- **P40 (O38)** — `tests/test_tool.py`'s artifacts-family test repeats `browse.py`'s old reason
  verbatim, with line numbers already stale at `a0d85da`. Its assertions still pass after T28, so
  nothing reds and the false sentence would survive the piece. T41 corrects it, docstring only, and
  proves the assertions are untouched.
- **P41 (O39)** — the piece-2 spec states the soon-to-be-false fact **twice**: at `:425` and in its
  out-of-scope table's piece-5 row. T41 corrects both.
- **P42 (O40)** — **older walkthrough rows are never edited.** The established practice is that each
  new section states its supersessions in its own preamble, because the old rows are a dated record
  of what a human saw. T41 writes the supersession into the E-row preamble and leaves rows 15, 16
  and B8 untouched. **The spec's §10 wording is read that way.**
- **P43 (O43)** — the gate table's display-walkthrough row reads "Not yet run. Owed by the owner, on
  a real screen", exactly as piece 4's did at the same moment. The piece's record is complete
  without it.
- **P44 (O42)** — the legacy-directory half of the tidy-up cannot be exercised on the owner's
  machine without creating `Artifacts/crossval/` by hand. The E-row says so and tells the owner to
  create an empty one; the loose-files half needs no setup, because two stray PNGs are already
  there.
- **P45 (O62)** — T30 rewrites the `sweep` block of the tool's epilog, which contains one of the
  counts T1 corrected. T30 re-derives the text it finds and does not assume T1's wording.
- **P46 (O63)** — `crossval` has no bare refusal of its own to convert; its pre-spend raises are
  already `UsageError`s, reported at exit 2 by design. T29 says so rather than inventing one.

### The rest

- **P47 (O9)** — **`load_fdt` verifies every recorded payload hash that is not null and refuses a
  mismatch with `field="artifact"`; a null hash means "not yet", never a mismatch; nothing else is
  checked.** The spec's §2.4 item 6 said both "verifies the payload hash" and "like `load_diagnostic`",
  which verifies nothing — **corrected** to the stricter reading, because a progressive record's
  hashes are written at the final commit only and a loader that refused a null hash would refuse
  exactly the records E2 keeps.
- **P48 (O17)** — **figures go through `writer.figure_path(title)`, not `w.fig_sink`.** None of the
  four drawing functions ever yields a live Figure — each takes a `save_path` and calls `savefig`
  itself — so `fig_sink` would mean rewriting all four, which is in no task's mandate. `figure_path`
  delivers every property §2.3 needs: the PNG in the record's `figures/` and the name in the
  manifest. The window still receives them through the watcher. **Spec §2.3 is corrected.**
- **P49 (O44)** — `_wrote_anything` flips in `payload()` and `figure_path()`, which is the last
  moment the writer can observe anything; a stage that takes a path and then refuses without writing
  keeps an empty record. **The mitigation is in the stages: call `payload()` at the moment of
  writing, not at the top of the run.** Stated in T17's and T19's briefs.
- **P50 (O11)** — the preset fallback moves into `make_param_sweep_config` and the tool's call-site
  fallback is **left in place**, so the tool's exact-set pin only has to gain `preset_name`. A later
  piece may delete the now-redundant one; nothing depends on it.
- **P51 (O12)** — **the thin-setting judgement is in the run, not the builder.** The run buffer tees
  warnings only for the thread that opened it, and the builders run on the window's thread before
  any run exists, so a warning raised there could never reach `log.txt` or `body.notices`. Spec §3.3
  already says "the run raises"; the task brief was the looser reading.
- **P52 (O46)** — `require_below`'s message uses the module's own "; got …" idiom rather than
  parentheses, so it matches its eight siblings and the shape helper.
- **P53 (O31)** — `_merge_vals_bounds`'s third wording of the same rule is **left alone and handed
  on**: it carries the cell path, which the FDT builders need and the other two do not. Recorded in
  the spec's open-items list, not fixed here. **Spec §6.2 is corrected** to say two wordings are
  unified and a third is known.
- **P54 (O49)** — `n_freqs` and `ensemble_m` have two effective defaults (the dataclass's and the
  preset's). Their `Field` default clause names the dataclass default and says the sweep's comes
  from the preset.
- **P55 (O33)** — **`run_fdt` on the `confirm_production=False` branch returns its record, finished,
  with `body.grid` and `body.offgrid` null**: the sanity checks ran and nothing else did, which is a
  complete answer to what was asked. T17 states it; T33 runs the full path anyway and records the
  measured seconds.
- **P56 (O34)** — every record assertion addresses records by `Summary.study` and `store.list`'s
  documented order, never by name, and T33 takes its own `--store-root`.
- **P57 (O64)** — `body.offgrid` counts the **Campaign-2 probe grid**, the points the ratio is
  reported at. The spec's "grid frequencies" is read that way throughout.
- **P58 (O65)** — a defect a round trip exposes and does not fix is recorded in the **spec's
  open-items list**, not §12; §12 is for deviations from this plan. T35 says so.
- **P59 (O66)** — half the Simulate round trip exists already (`tests/test_simulate.py`'s GIF
  round trip). T35 adds precision to it rather than claiming new ground, and says which half is new.
- **P60 (O67)** — `compare repeats` does not enforce "the same cell": a record's cell lives in the
  manifest's inputs block and nothing stops a caller passing two cells. The mode **reports** the
  cells it drew in the drawing and in `body.notices`, and refuses only on arity. *Cost if wrong: a
  misleading band nobody asked for; the drawing names the cells, so it is visible.*
- **P61 (O68)** — T25 and T26 expose their picker as `self.record_picker`, and T40 reads it.
- **P62 (O69)** — **a nineteenth suite, `tests/test_fdt_compare.py`, is correct.** The comparison
  facility is a new module, a new subcommand and new controls, and its siblings do not exist.
  `CLAUDE.md`'s "eighteen suites" is updated by T41.
- **P63–P69** — the remaining minors are ratified as each drafter resolved them, with no change to
  the contract: the `--help` epilog's count wording (T1 re-derives), the `UsageError` rungs, the
  `_EntryWriter` stand-in's unsuitability for the comparison leg, the `IntField` accessor, the
  docstring corrections, and the two stale line-number ranges in existing test docstrings.

---
# Piece 5 — Tasks 1–6: the store

The eighth artifact kind, its progressive writer, the tidy-up's two new categories, the lineage
report's `compared` branch, and the delete confirmation. Nothing here runs a simulation, so every
test in these six tasks is fast-gate work: no `slow` marker anywhere below.

Read `docs/superpowers/specs/2026-09-22-secondary-analyses-design.md` §2 and the plan's Interface
Contract before starting. **Line numbers below are hints; the QUOTED text is the anchor.**

---

### Task 1: the new kind's declarations, and the three silent pins

**Why:** Spec §2.1 and §2.4 items 1–7, 13, 16, 18, 21–22, 25 (decision **E3**: one kind covers both
analyses). Three of the checklist's items — `KIND_LABELS`, the hand-written kind counts in the
tool's `--help`, and `_bodies` — fail at run time rather than in the suite, so each gains a test
here, written first (spec §11's first risk row).

**Files:**
- Modify: `core/artifacts/manifest.py:19` (`KINDS`), `:27-44` (`BODY_KEYS`)
- Modify: `core/artifacts/store.py:31-33` (`KIND_DIRS`), `:52-54` (`_PARENT_KEYS`), `:75-105`
  (`Summary`), `:192-198` (beside `LoadedDiagnostic`), `:1056-1073` (beside `load_diagnostic`)
- Modify: `core/artifacts/__init__.py:6-9` (the re-export)
- Modify: `core/gui/screens/artifact_screen.py:40-48` (`KIND_LABELS`)
- Modify: `core/gui/widgets/artifact_table.py:24-32` (`_EXTRA_COLUMNS`), `:89-112` (`cells_for`)
- Modify: `core/tool/browse.py:45-49` (`KINDS`), `:56-64` (`COLUMNS`), `:80-104` (`EPILOG`),
  `:132-151` (`_cells`), `:442`, `:468` (the two mode helps)
- Test: `tests/_fixtures.py:385-456` (`build_browse_store`), `tests/test_artifact_store.py:152-172`
  (`_bodies`), `tests/test_artifact_browser.py`, `tests/test_tool.py`

**Interfaces:**
- Consumes: nothing from an earlier task — this is the first.
- Produces:
  - `manifest.KINDS` gains `"fdt"` (eighth, last); `manifest.BODY_KEYS["fdt"] = ("study",
    "settings", "seed", "grid", "points", "offgrid", "notices", "compared", "complete", "results")`
  - `store.KIND_DIRS["fdt"] = "fdt"`; `store._PARENT_KEYS["fdt"] = ()`
  - `store.LoadedFdt(Loaded)` with `body: dict` and `data_path: "Path | None"`
  - `ArtifactStore.load_fdt(self, ref: str) -> LoadedFdt`
  - `store.Summary` gains `study: "str | None" = None`, `points_done: "int | None" = None`,
    `points_planned: "int | None" = None`, `points_failed: "int | None" = None` — **declared here,
    FILLED by Task 2**
  - `core.artifacts.LoadedFdt` (the package re-export)
  - `artifact_table._EXTRA_COLUMNS["fdt"] = ("Study", "Points", "Finished")` and the matching
    `cells_for` branch; `browse.COLUMNS["fdt"] = ("name", "created", "study", "points", "finished",
    "note")` and the matching `_cells` branch
  - `tests/_fixtures.build_browse_store` returns an `"fdt"` id beside the other seven

**Note on scope.** `Summary`'s four fields and both column tables are spec item 9's and item
14's, which the plan's task order gives to Task 2. They are carried **here** because three existing
closed-set pins go red the moment `KIND_DIRS` gains a key — `tests/test_worker_dispatch.py:699`
(`set(at._EXTRA_COLUMNS) == set(KIND_DIRS)`), `tests/test_tool.py:1753-1758` (`browse.KINDS ==
tuple(KIND_DIRS)` and a column set per kind) and `tests/test_worker_dispatch.py:714-724`
(`len(cells_for(kind, s)) == len(columns_for(kind))` for every kind) — and the fast gate is run
after every task. Task 2 makes `store.list` fill them. This is recorded as an objection.

- [ ] **Step 1: Give `_bodies` the new kind and close it against `BODY_KEYS`**

In `tests/test_artifact_store.py`, find:

```python
def _bodies():
    """One valid body per WRITER kind -- the six ``store.create`` accepts. A FRESH dict per call, so a
    test that hands one to the writer cannot leave a mutation behind for the next.

    The simulation kind is deliberately absent: its manifest has no writer at all (``store.create``
    refuses it outright) and ``write_simulation_manifest`` builds its body itself.
    """
```

Replace with:

```python
def _bodies():
    """One valid body per WRITER kind -- the seven ``store.create`` accepts. A FRESH dict per call, so
    a test that hands one to the writer cannot leave a mutation behind for the next.

    The simulation kind is deliberately absent: its manifest has no writer at all (``store.create``
    refuses it outright) and ``write_simulation_manifest`` builds its body itself. That absence is
    why ``test_bodies_covers_every_writer_kind`` closes this dict against ``BODY_KEYS`` MINUS that
    one kind rather than against ``BODY_KEYS`` itself (piece 5, checklist 22).
    """
```

and add the new entry to the returned dict, after the `diagnostic` one:

```python
        "diagnostic": {"diagnostic": "sbc", "variant": None, "settings": {"repeats": 2},
                       "results": {"n_valid": 8}},
        # A finished single-cell measurement at its smallest: the five keys a run knows before it
        # starts, the three a finished single run fills, and the two that are null for a single run
        # (``points``) and for anything but a comparison (``compared``). ``complete`` is True
        # because _make exits its writer cleanly, and the writer is what sets that flag (Task 3).
        "fdt": {"study": "single", "settings": {"n_freqs": 2, "ensemble_M": 8}, "seed": 11,
                "grid": {"omega_0": 1.0, "n_freqs": 2}, "points": None, "offgrid": {"blanks": 0,
                "of": 2}, "notices": [], "compared": None, "complete": True,
                "results": {"ratio_at_resonance": 1.5}},
```

Then add, directly under `_bodies`:

```python
def test_bodies_covers_every_writer_kind():
    """Checklist 22, a SILENT pin. ``_bodies`` feeds the per-kind round trip and the finished-per-kind
    test, and neither is closed against the schema -- so a kind added to ``BODY_KEYS`` and forgotten
    here is simply never round-tripped, and nothing says so. The simulation kind is subtracted rather
    than listed: it has no writer at all (``store.create`` refuses it), which is why ``_bodies``
    omits it on purpose.
    """
    assert set(_bodies()) == set(mf.BODY_KEYS) - {"simulation"}, \
        "a kind in BODY_KEYS with no body here is silently never written by any test"
    for kind, body in _bodies().items():
        assert set(body) == set(mf.BODY_KEYS[kind]), kind
```

- [ ] **Step 2: Run them and watch them fail**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_artifact_store.py::test_bodies_covers_every_writer_kind tests/test_artifact_store.py::test_create_list_get_round_trip_per_kind -v`

Expected: FAIL. `test_bodies_covers_every_writer_kind` fails on the first assert —
`AssertionError: a kind in BODY_KEYS with no body here is silently never written by any test` with
`{'fdt'} != set()` in the diff (the dict has `fdt`, `BODY_KEYS` does not).
`test_create_list_get_round_trip_per_kind` fails with
`core.artifacts.store.StoreError: unknown artifact kind 'fdt'` from `kind_dir`.

- [ ] **Step 3: Declare the kind in the manifest schema**

In `core/artifacts/manifest.py`, find:

```python
KINDS = ("prior", "simulation", "posterior", "observation", "calibration", "inference", "diagnostic")
```

Replace with:

```python
KINDS = ("prior", "simulation", "posterior", "observation", "calibration", "inference", "diagnostic",
         "fdt")
```

Then find the end of `BODY_KEYS`:

```python
    "diagnostic": ("diagnostic", "variant", "settings", "results"),
}
```

Replace with:

```python
    "diagnostic": ("diagnostic", "variant", "settings", "results"),
    # A measurement of a CELL (not of another artifact), or a comparison of such measurements
    # (piece 5, E3). ``study`` is "single", "sweep" or "comparison" and is what tells the three
    # apart; there is deliberately no second kind and no registry of study names here.
    # Written PROGRESSIVELY (store.PROGRESSIVE_KINDS): every key is present from the FIRST manifest,
    # with the not-yet-known ones null, because ``validate`` below refuses a partial key set. So the
    # nullable keys are the schema's way of saying "this run has not got there yet", and ``complete``
    # -- False until the writer commits -- is the one that says whether it ever did.
    "fdt": ("study", "settings", "seed", "grid", "points", "offgrid", "notices", "compared",
            "complete", "results"),
}
```

- [ ] **Step 4: Declare the kind in the store**

In `core/artifacts/store.py`, find:

```python
KIND_DIRS = {"prior": "priors", "simulation": "simulations", "posterior": "posteriors",
             "observation": "observations", "calibration": "calibrations", "inference": "inferences",
             "diagnostic": "diagnostics"}
```

Replace with:

```python
KIND_DIRS = {"prior": "priors", "simulation": "simulations", "posterior": "posteriors",
             "observation": "observations", "calibration": "calibrations", "inference": "inferences",
             "diagnostic": "diagnostics",
             # NOT "crossval": core/Reduction/plots.py already writes reduction_crossval_<stamp>.png
             # into the reduction directory, and a second meaning for the word would collide with
             # existing vocabulary (spec §2.1). Reusing the EXISTING ``fdt`` directory is chosen, not
             # accidental: the owner's stray pictures then sit as loose files inside a kind
             # directory, which is where ``loose_files`` can reach them.
             "fdt": "fdt"}
```

Then find:

```python
_PARENT_KEYS = {"prior": ("prior",), "simulation": ("simulation",),
                "posterior": ("posterior", "parent_posterior"), "observation": ("observation",),
                "calibration": (), "inference": (), "diagnostic": ()}
```

Replace with:

```python
_PARENT_KEYS = {"prior": ("prior",), "simulation": ("simulation",),
                "posterior": ("posterior", "parent_posterior"), "observation": ("observation",),
                "calibration": (), "inference": (), "diagnostic": (),
                # Explicit, though a MISSING entry means the same thing to ``dependents``: an fdt
                # record measures a CELL and depends on no artifact, and nothing can depend on one.
                # A comparison names the records it drew in its BODY, not here -- ``parents`` is a
                # flat {key: id} map read as ``m.parents.get(pk) == id_`` (spec §1.2), so it cannot
                # carry an arbitrary number of ids without widening the contract for every kind.
                "fdt": ()}
```

- [ ] **Step 5: Run the two tests and watch them pass**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_artifact_store.py::test_bodies_covers_every_writer_kind tests/test_artifact_store.py::test_create_list_get_round_trip_per_kind -v`

Expected: PASS (2 passed).

- [ ] **Step 6: Write the failing test for `LoadedFdt` and `load_fdt`**

In `tests/test_artifact_store.py`, add after `test_bodies_covers_every_writer_kind`:

```python
def test_load_fdt_verifies_the_payload_hash_and_nothing_else(store):
    """Checklist 6. An fdt record is a MEASUREMENT of a cell, not a constraint on a later run, so
    there is no configuration it has to match and nothing is ever trained from it -- ``load_diagnostic``
    is the precedent (D5). What it DOES check is the one thing the manifest can be checked against:
    the payload's own sha256, so a data.h5 edited or truncated since the commit is refused rather
    than read as the numbers the record claims.

    A NULL recorded hash is not a mismatch. A progressive record hashes its payloads at the final
    commit only (spec §2.2), so an unfinished record lists data.h5 with a null hash and must still
    load -- reading what an interrupted run managed to write is the whole point of E2.
    """
    from core.artifacts import LoadedFdt
    # "measured" never wrote numbers: _make gives it results.json and nothing else.
    _make(store, "fdt", name="measured", body=_bodies()["fdt"])
    # "measured2" did. A payload is registered on the WRITER, so it is written inside the with.
    with store.create("fdt", None, name="measured2") as w2:
        w2.body = _bodies()["fdt"]
        w2.payload("data.h5").write_bytes(b"\x89HDF\r\n\x1a\n" + b"0" * 64)

    loaded = store.load_fdt("measured2")
    assert isinstance(loaded, LoadedFdt) and loaded.kind == "fdt" and loaded.id == w2.id
    assert loaded.body["study"] == "single" and loaded.body["seed"] == 11
    assert loaded.data_path == w2.dir / "data.h5" and loaded.data_path.is_file()
    assert loaded.manifest.payloads["data.h5"] == prov.sha256_file(w2.dir / "data.h5")

    # a record that never wrote numbers: no data.h5, and the loader says so rather than pointing at
    # a path that is not there
    assert store.load_fdt("measured").data_path is None

    # a null recorded hash (an unfinished record) loads
    mpath = w2.dir / st.MANIFEST
    d = json.loads(mpath.read_text(encoding="utf-8"))
    d["payloads"]["data.h5"] = None
    mpath.write_text(json.dumps(d), encoding="utf-8")
    assert store.load_fdt("measured2").data_path is not None, "a null hash is 'not yet', not 'wrong'"

    # a payload edited since the commit IS refused
    d["payloads"]["data.h5"] = "0" * 64
    mpath.write_text(json.dumps(d), encoding="utf-8")
    with pytest.raises(st.StoreError, match="data.h5"):
        store.load_fdt("measured2")

    with pytest.raises(st.StoreError, match="no complete fdt"):
        store.load_fdt("nosuch")
```

- [ ] **Step 7: Run it and watch it fail**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_artifact_store.py::test_load_fdt_verifies_the_payload_hash_and_nothing_else -v`

Expected: FAIL with `ImportError: cannot import name 'LoadedFdt' from 'core.artifacts'`.

- [ ] **Step 8: Add `LoadedFdt` and `load_fdt`**

In `core/artifacts/store.py`, find:

```python
@dataclass
class LoadedDiagnostic(Loaded):
    diagnostic: str = ""            # the diagnostic that wrote it ("sbc", "identifiability", ...)
    variant: "str | None" = None    # its mode where it has more than one ("rotation"/"laplace"/"jacobian")
    settings: dict = field(default_factory=dict)
    results: dict = field(default_factory=dict)
```

Replace with:

```python
@dataclass
class LoadedDiagnostic(Loaded):
    diagnostic: str = ""            # the diagnostic that wrote it ("sbc", "identifiability", ...)
    variant: "str | None" = None    # its mode where it has more than one ("rotation"/"laplace"/"jacobian")
    settings: dict = field(default_factory=dict)
    results: dict = field(default_factory=dict)


@dataclass
class LoadedFdt(Loaded):
    """One effective-temperature measurement, one parameter sweep, or one comparison of them.

    NOT frozen, unlike the plan's interface contract: ``Loaded`` is a plain dataclass, and Python
    refuses ``@dataclass(frozen=True)`` on a subclass of a non-frozen one (TypeError at import).
    Every other ``Loaded*`` in this module is a plain dataclass for the same reason.

    ``body`` is the manifest's own body, copied; ``data_path`` is ``<dir>/data.h5`` when the run got
    as far as writing numbers and None when it did not -- which an INTERRUPTED record (E2) very
    often has not.
    """
    body: dict = field(default_factory=dict)
    data_path: "Path | None" = None
```

Then find:

```python
        sub, m = self._find("diagnostic", ref)
        if m is None:
            raise StoreError(f"no complete diagnostic named or id'd {ref!r}")
        body = m.body
        return LoadedDiagnostic("diagnostic", m.id, m.name, m, sub, diagnostic=body["diagnostic"],
                                variant=body["variant"], settings=dict(body["settings"]),
                                results=dict(body["results"]))
```

Replace with:

```python
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
        is ever trained from it. The ONE thing it verifies is the payload's own sha256, because that
        is the one claim the manifest makes about a file this call is about to hand out.

        A payload whose recorded hash is NULL is not verified. A progressive record hashes at the
        final commit only (spec §2.2), so every unfinished record lists its payloads unhashed -- and
        reading what an interrupted run did manage to write is exactly what E2 keeps the folder for.
        """
        sub, m = self._find("fdt", ref)
        if m is None:
            raise StoreError(f"no complete fdt artifact named or id'd {ref!r} under "
                             f"{self.kind_dir('fdt')}", field="artifact")
        for name, want in sorted(m.payloads.items()):
            p = sub / name
            if want is None or not p.is_file():
                continue
            got = prov.sha256_file(p)
            if got != want:
                raise StoreError(f"fdt {m.name or m.id}: {name} hashes to {got}, not the {want} its "
                                 f"manifest records; the artifact is inconsistent", field="artifact")
        data = sub / "data.h5"
        return LoadedFdt("fdt", m.id, m.name, m, sub, body=dict(m.body),
                         data_path=data if data.is_file() else None)
```

Then, in `core/artifacts/__init__.py`, find:

```python
from .store import (ArtifactStore, ArtifactWriter, Accept, Summary, StoreError, KIND_DIRS,  # noqa: F401
                    Loaded, LoadedPrior, LoadedPosterior, LoadedObservation, LoadedCalibration,
                    LoadedInference, LoadedDiagnostic, write_simulation_manifest,
                    default_store, set_default_store, use_store, resolve_store)
```

Replace with:

```python
from .store import (ArtifactStore, ArtifactWriter, Accept, Summary, StoreError, KIND_DIRS,  # noqa: F401
                    Loaded, LoadedPrior, LoadedPosterior, LoadedObservation, LoadedCalibration,
                    LoadedInference, LoadedDiagnostic, LoadedFdt, write_simulation_manifest,
                    default_store, set_default_store, use_store, resolve_store)
```

- [ ] **Step 9: Run it and watch it pass**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_artifact_store.py::test_load_fdt_verifies_the_payload_hash_and_nothing_else -v`

Expected: PASS.

- [ ] **Step 10: Give `Summary` its four new fields and both tables their fdt entry**

In `core/artifacts/store.py`, find:

```python
    rows: "tuple[int, ...] | None" = None   # simulation only: rows per batch, written at completion ONLY
    variant: "str | None" = None            # diagnostic only: body["variant"]
```

Replace with:

```python
    rows: "tuple[int, ...] | None" = None   # simulation only: rows per batch, written at completion ONLY
    variant: "str | None" = None            # diagnostic only: body["variant"]
    # Piece 5: fdt only. ``study`` is "single", "sweep" or "comparison"; the three counts are a
    # SWEEP's operating points (body["points"]), None for a single run and for a comparison, which
    # have none. Declared here and FILLED by ArtifactStore.list.
    study: "str | None" = None
    points_done: "int | None" = None
    points_planned: "int | None" = None
    points_failed: "int | None" = None
```

In `core/gui/widgets/artifact_table.py`, find:

```python
    "diagnostic": ("Variant",),
}
```

Replace with:

```python
    "diagnostic": ("Variant",),
    # An fdt record shows WHICH of the three studies it is, a sweep's operating points, and whether
    # the run finished -- ``finished`` is a real question for this kind (E2: an interrupted record
    # keeps its folder), which is why it has the cache's column and the other six do not.
    "fdt": ("Study", "Points", "Finished"),
}
```

Then find:

```python
    elif kind == "diagnostic":
        middle = (s.variant or "",)
    return (s.label, s.created) + middle + (s.note or "",)
```

Replace with:

```python
    elif kind == "diagnostic":
        middle = (s.variant or "",)
    elif kind == "fdt":
        middle = (s.study or "", _points(s), "yes" if s.finished else "no")
    return (s.label, s.created) + middle + (s.note or "",)
```

and add, directly above `def cells_for(`:

```python
def _points(s) -> str:
    """A sweep's operating points: ``"10/12"``, or ``"10/12 · 2 failed"`` when some failed (E4).

    Blank for a single run and for a comparison, which have no points at all -- ``points`` is null in
    their bodies, so ``points_done`` is None and the cell says nothing rather than "0/0". A planned
    total the manifest does not carry (a hand-edited body) falls back to the count alone rather than
    printing "10/None", exactly as ``_progress`` does above.
    """
    if s.points_done is None:
        return ""
    head = (f"{int(s.points_done)}" if s.points_planned is None
            else f"{int(s.points_done)}/{int(s.points_planned)}")
    if not s.points_failed:
        return head
    return f"{head} · {int(s.points_failed)} failed"
```

- [ ] **Step 11: Give the tool its kind, its columns and its cells**

In `core/tool/browse.py`, find:

```python
# The seven kinds in KIND_DIRS order, restated as a literal: reading core.artifacts.store.KIND_DIRS
# at parser-build time would import torch (core/artifacts/__init__.py imports .store, which imports
# core.config). tests/test_tool.py pins the two in sync, both order and membership -- the same shape
# as --preset's hard-coded choices pinned against cli.SWEEP_PRESETS.
KINDS = ("prior", "simulation", "posterior", "observation", "calibration", "inference", "diagnostic")
```

Replace with:

```python
# The eight kinds in KIND_DIRS order, restated as a literal: reading core.artifacts.store.KIND_DIRS
# at parser-build time would import torch (core/artifacts/__init__.py imports .store, which imports
# core.config). tests/test_tool.py pins the two in sync, both order and membership -- the same shape
# as --preset's hard-coded choices pinned against cli.SWEEP_PRESETS.
KINDS = ("prior", "simulation", "posterior", "observation", "calibration", "inference", "diagnostic",
         "fdt")
```

Then find:

```python
    "diagnostic": ("name", "created", "variant", "note"),
}
```

Replace with:

```python
    "diagnostic": ("name", "created", "variant", "note"),
    "fdt": ("name", "created", "study", "points", "finished", "note"),
}
```

Then find:

```python
    if kind == "diagnostic":
        # A diagnostic records its kind under ``variant``, never ``mode``: the mode column would
        # otherwise report a conditioning geometry that does not exist (design §1.3).
        return head + (_flat(s.variant) or "-", _flat(s.note))
    return head + (_flat(s.note),)
```

Replace with:

```python
    if kind == "diagnostic":
        # A diagnostic records its kind under ``variant``, never ``mode``: the mode column would
        # otherwise report a conditioning geometry that does not exist (design §1.3).
        return head + (_flat(s.variant) or "-", _flat(s.note))
    if kind == "fdt":
        return head + (_flat(s.study) or "-", _fdt_points(s), "yes" if s.finished else "no",
                       _flat(s.note))
    return head + (_flat(s.note),)
```

and add, directly above `def _cells(`:

```python
def _fdt_points(s) -> str:
    """"10/12" for a sweep, "10/12, 2 failed" when some points failed, "-" for a single run or a
    comparison, which have no operating points at all. A comma and plain ASCII, not the browser's
    middle dot: a script may read this line -- ``_progress`` above takes the same care."""
    if s.points_done is None:
        return "-"
    planned = "?" if s.points_planned is None else str(int(s.points_planned))
    text = f"{int(s.points_done)}/{planned}"
    if s.points_failed:
        text += f", {int(s.points_failed)} failed"
    return text
```

- [ ] **Step 12: Run the two closed-set pins and watch them pass**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_worker_dispatch.py::test_columns_for_names_all_seven_kinds_after_name_and_created tests/test_worker_dispatch.py::test_cells_for_renders_each_kind_from_the_summary_alone "tests/test_tool.py::test_the_tools_kinds_and_columns_match_the_browsers" -v`

Expected: PASS. (If the last node id does not resolve, find the test that holds
`assert browse.KINDS == tuple(KIND_DIRS), "the seven kinds, in KIND_DIRS order"` and run that one;
its name is the anchor's owner.) Fix its docstring's "seven" → "eight" in the same step if the
assertion message reads "the seven kinds".

- [ ] **Step 13: SILENT PIN 1 — `KIND_LABELS` closed against `KIND_DIRS`**

In `tests/test_artifact_browser.py`, add after
`test_the_browser_lists_the_seven_kinds_with_the_incomplete_directories_last`:

```python
def test_every_kind_dir_has_a_selector_label(tmp_path):
    """Checklist 13, a SILENT pin. ``ArtifactScreen.__init__`` does ``KIND_LABELS[kind]`` for every
    key in ``KIND_DIRS``, so a kind added to the store without a label here is a KeyError AT WINDOW
    LAUNCH -- not in a listing, not at a click, but before the app draws anything -- and nothing
    pinned it. Closed both ways: a label for a kind the store does not have would put an entry in the
    selector that resolves to no directory.
    """
    from core.gui.screens.artifact_screen import KIND_LABELS
    assert set(KIND_LABELS) == set(KIND_DIRS), \
        "a kind with no selector label is a KeyError at window launch"
    assert list(KIND_LABELS) == list(KIND_DIRS), "the selector is built in KIND_DIRS order"
    assert all(v and not v.startswith(" ") for v in KIND_LABELS.values())
    # and the screen really does build from it, so the pin cannot rot into a test of a dead dict
    screen = artifact_screen(ArtifactStore(tmp_path))
    assert screen.kind_combo.count() == len(KIND_DIRS)
```

- [ ] **Step 14: Run it and watch it fail**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_artifact_browser.py::test_every_kind_dir_has_a_selector_label -v`

Expected: FAIL with `AssertionError: a kind with no selector label is a KeyError at window launch`
(the sets differ by `{'fdt'}`). Note that the construction line at the end would raise
`KeyError: 'fdt'` — which is the defect the pin exists for.

- [ ] **Step 15: Add the label**

In `core/gui/screens/artifact_screen.py`, find:

```python
    "diagnostic": "Diagnostics",
}
```

Replace with:

```python
    "diagnostic": "Diagnostics",
    "fdt": "FDT measurements and sweeps",
}
```

- [ ] **Step 16: Run it and watch it pass**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_artifact_browser.py::test_every_kind_dir_has_a_selector_label -v`

Expected: PASS.

- [ ] **Step 17: SILENT PIN 2 — every hand-written kind count in the RENDERED `--help`**

In `tests/test_tool.py`, add after the test that holds
`assert browse.KINDS == tuple(KIND_DIRS), "the seven kinds, in KIND_DIRS order"`:

```python
def test_the_rendered_help_counts_the_kinds_correctly():
    """Checklist 18, a SILENT pin. Five phrases in core/tool/browse.py hand-write how many kinds
    there are -- two "all seven"s in EPILOG, the "<kind> is one of ..." line under them, and the
    ``kind`` help on ``list`` and on ``sweep``. None is generated, none was pinned, and a stale one
    prints a wrong help page to an operator with no failure anywhere.

    The scan is over the RENDERED help and not over the EPILOG constant, deliberately: two of the
    five live on argparse arguments and never appear in that string, so a test that read the
    constant would pass while `python -m core artifacts sweep --help` lied.
    """
    import argparse
    import re
    from core.tool import browse

    words = {6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}
    want = words[len(browse.KINDS)]

    p = build_parser().subcommands["artifacts"]
    modes = {name: sub for a in p._actions if isinstance(a, argparse._SubParsersAction)
             for name, sub in a.choices.items()}
    rendered = {"artifacts": p.format_help(),
                **{name: sub.format_help() for name, sub in modes.items()}}

    for name, text in rendered.items():
        for found in re.findall(r"\ball (\w+)", text):
            assert found == want, f"{name} --help says 'all {found}' for {len(browse.KINDS)} kinds"
    assert f"all {want}" in rendered["artifacts"], "the epilog names the count at all"
    assert f"all {want}" in rendered["sweep"], "sweep's --help names the count at all"
    assert "<kind> is one of " + ", ".join(browse.KINDS) in rendered["artifacts"], \
        "the epilog lists the kinds by name, in KINDS order"
```

- [ ] **Step 18: Run it and watch it fail**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_tool.py::test_the_rendered_help_counts_the_kinds_correctly -v`

Expected: FAIL with
`AssertionError: artifacts --help says 'all seven' for 8 kinds` (the first `all seven` in EPILOG).

- [ ] **Step 19: Correct all five counts**

In `core/tool/browse.py`'s `EPILOG`, find:

```
  list [<kind>]         one line per artifact; with no kind, all seven under headings
```

Replace with:

```
  list [<kind>]         one line per artifact; with no kind, all eight under headings
```

Find:

```
                        remove every directory with NO manifest at all; with no kind, all seven. A
```

Replace with:

```
                        remove every directory with NO manifest at all; with no kind, all eight. A
```

Find:

```
<kind> is one of prior, simulation, posterior, observation, calibration, inference, diagnostic.
```

Replace with:

```
<kind> is one of prior, simulation, posterior, observation, calibration, inference, diagnostic, fdt.
```

Find:

```python
    ls = modes.add_parser("list", help="one line per artifact of a kind, or of all seven")
```

Replace with:

```python
    ls = modes.add_parser("list", help="one line per artifact of a kind, or of all eight")
```

Find:

```python
    sweep.add_argument("kind", nargs="?", default=None, metavar="<kind>",
                       help=f"{_KIND}; with no kind, all seven")
```

Replace with:

```python
    sweep.add_argument("kind", nargs="?", default=None, metavar="<kind>",
                       help=f"{_KIND}; with no kind, all eight")
```

Finally, correct the two stale docstrings in the same file (not rendered, but they are the prose a
maintainer reads): in `_list`, `With no kind, all seven` → `all eight`; in `_sweep`,
`(or of all seven)` → `(or of all eight)`.

- [ ] **Step 20: Run it and watch it pass**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_tool.py::test_the_rendered_help_counts_the_kinds_correctly -v`

Expected: PASS.

- [ ] **Step 21: Give `build_browse_store` an fdt record**

In `tests/_fixtures.py`, find:

```python
def build_browse_store(root):
    """One artifact of each of the SEVEN kinds plus the three bad-directory shapes, in a store at
    ``root``. Returns ``{kind: id, ..., "bad": (dir_name, dir_name, dir_name)}``.
```

Replace with:

```python
def build_browse_store(root):
    """One artifact of each of the EIGHT kinds plus the three bad-directory shapes, in a store at
    ``root``. Returns ``{kind: id, ..., "bad": (dir_name, dir_name, dir_name)}``.
```

Then find:

```python
        "diagnostic": {"diagnostic": "identifiability", "variant": "laplace",
                       "settings": {}, "results": {}},
    }
```

Replace with:

```python
        "diagnostic": {"diagnostic": "identifiability", "variant": "laplace",
                       "settings": {}, "results": {}},
        # A FINISHED sweep, so the fdt row carries content in both of its own columns: a study name
        # and a point fraction with a failure in it. ``complete`` True because the writer commits
        # cleanly here; an UNFINISHED record is a different shape and the tests that need one build
        # it themselves rather than making this fixture carry two.
        "fdt": {"study": "sweep", "settings": {"n_freqs": 2, "preset": "fast"}, "seed": 7,
                "grid": None, "points": {"param": "S", "planned": 3, "done": 2, "failed": 1},
                "offgrid": {"blanks": 0, "of": 6}, "notices": [], "compared": None,
                "complete": True, "results": {"peak_ratio": 2.0}},
    }
```

- [ ] **Step 22: Run the browser and tool listings and watch them pass**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_artifact_browser.py tests/test_tool.py -k "browse or kinds or help" -v`

Expected: PASS. The browser's `test_the_browser_lists_the_seven_kinds_...` now walks eight kinds and
finds `browse_fdt` with `the fdt row`; the tool's listing tests find the same row.

- [ ] **Step 23: Run all four touched suites**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_artifact_store.py tests/test_artifact_browser.py tests/test_worker_dispatch.py tests/test_tool.py -m "not slow" -q`

Expected: PASS.

- [ ] **Step 24: Commit**

```bash
git add core/artifacts/manifest.py core/artifacts/store.py core/artifacts/__init__.py core/gui/screens/artifact_screen.py core/gui/widgets/artifact_table.py core/tool/browse.py tests/_fixtures.py tests/test_artifact_store.py tests/test_artifact_browser.py tests/test_tool.py
git commit -m "store: declare the fdt kind, with the three silent pins"
```

#### Amendments (binding — these supersede the text above)

From the pre-flight scan against HEAD `673868d`. No ruling in P70–P82 changes this task. P1 and P16 are already reflected in the steps above. The corrections below fix one stale test anchor and one stale docstring sentence.

1. **Step 12, the third node id.** `tests/test_tool.py::test_the_tools_kinds_and_columns_match_the_browsers` does not exist. The test that holds `assert browse.KINDS == tuple(KIND_DIRS), "the seven kinds, in KIND_DIRS order"` is `test_the_artifacts_listing_shows_the_browsers_own_columns`. Run:

   `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_worker_dispatch.py::test_columns_for_names_all_seven_kinds_after_name_and_created tests/test_worker_dispatch.py::test_cells_for_renders_each_kind_from_the_summary_alone tests/test_tool.py::test_the_artifacts_listing_shows_the_browsers_own_columns -v`

   In that test, change the assertion message `"the seven kinds, in KIND_DIRS order"` to `"the eight kinds, in KIND_DIRS order"`, and the docstring's `The seven kinds are restated in the tool as a literal` to `The eight kinds are restated in the tool as a literal`.

2. **Step 17's placement.** Add `test_the_rendered_help_counts_the_kinds_correctly` directly after `test_the_artifacts_listing_shows_the_browsers_own_columns` in `tests/test_tool.py`.

3. **Step 8, the `LoadedFdt` docstring (P1).** The interface contract was corrected to a plain dataclass, so replace the paragraph

   ```
       NOT frozen, unlike the plan's interface contract: ``Loaded`` is a plain dataclass, and Python
       refuses ``@dataclass(frozen=True)`` on a subclass of a non-frozen one (TypeError at import).
       Every other ``Loaded*`` in this module is a plain dataclass for the same reason.
   ```

   with

   ```
       NOT frozen (planning ruling P1): ``Loaded`` is a plain dataclass, and Python refuses
       ``@dataclass(frozen=True)`` on a subclass of a non-frozen one (TypeError at import). Every
       other ``Loaded*`` in this module is a plain dataclass for the same reason.
   ```

4. **Expected until Task 2 lands:** `ArtifactStore.list` does not fill `Summary.study` or `points_*` yet, so an fdt row's Study and Points cells are blank in the browser and `-` in the tool. That is Task 2's job (P16). Do not fill them here.

#### Controller rulings (pre-flight, binding — these supersede the amendments above and the task text)

- **F28** — also add, in `tests/test_worker_dispatch.py` beside the per-kind `columns_for("<kind>") == (...)` assertions, one for the new kind: `assert columns_for("fdt") == ("Name", "Created", "Study", "Points", "Finished", "Note")` — or whatever exact tuple your `_EXTRA_COLUMNS["fdt"]` and the table's leading/trailing columns produce; read the real `_LEADING` and write the real tuple. Add the file to the Files block and the `git add` line. The spec names this assertion (checklist item 23) and it is the only pin on the GUI column titles.
- **F29** — `load_fdt` REFUSES a payload whose recorded hash is non-null but whose file is missing, with `field="artifact"`, instead of skipping it and returning `data_path=None`. A hash the manifest recorded is a claim that can be checked; a deleted payload fails it. Only a NULL recorded hash means "not yet" (P47). Add the case to the load_fdt test: finish a record, delete its `data.h5`, assert the refusal.

---

### Task 2: `Summary` filled, the finished branch, and a numeric sort key

**Why:** Spec §2.4 items 8, 9, 14, 17, 23, 24 and §1.2's last-but-two ruling. `store.list`'s
`finished` branch names one kind by hand and must name the two that carry `complete`; the artifact
table's numeric columns sort as text (`10/12 batches` before `9/12`), and this piece adds a `Points`
column of exactly that shape, so it would manufacture a fresh instance of the defect it inherits.

**Files:**
- Modify: `core/artifacts/store.py:397-437` (`ArtifactStore.list`)
- Modify: `core/gui/widgets/artifact_table.py:36-39` (beside `DEFAULT_SORT`), `:115-151` (`_Row`)
- Test: `tests/test_artifact_store.py`, `tests/test_worker_dispatch.py`

**Interfaces:**
- Consumes: `Summary.study / points_done / points_planned / points_failed` and
  `_EXTRA_COLUMNS["fdt"]` / `COLUMNS["fdt"]` (Task 1).
- Produces:
  - `ArtifactStore.list` fills the four fields from `body["study"]` and `body["points"]`, and
    computes `finished` as `bool(body.get("complete")) if "complete" in mf.BODY_KEYS[kind] else True`
  - `artifact_table.NUMERIC_COLUMNS: frozenset` and
    `artifact_table.cell_key(column: str, text: str) -> tuple[float, str]`

- [ ] **Step 1: Write the failing test for the listing**

In `tests/test_artifact_store.py`, add after `test_a_summary_says_whether_the_run_finished_for_every_kind`:

```python
def test_an_fdt_row_carries_its_study_its_points_and_whether_it_finished(store):
    """Checklist 8 and 9. ``finished`` used to be "True unless this is the simulation kind" -- one
    kind named by hand. Two kinds now carry a ``complete`` flag in their body, and the branch is
    derived from BODY_KEYS rather than restated, so a third can never be forgotten.

    The point counts come off ``body["points"]`` in ONE manifest read, like every other Summary
    field (B2): a listing costs one directory scan and a browser row costs nothing. A single run and
    a comparison have no points at all -- ``body["points"]`` is null -- and their counts stay None
    rather than becoming a confident, false 0/0.
    """
    sweep = _bodies()["fdt"]
    sweep.update(study="sweep", points={"param": "S", "planned": 12, "done": 10, "failed": 2})
    w = _make(store, "fdt", name="a_sweep", body=sweep)
    row = [r for r in store.list("fdt") if r.id == w.id][0]
    assert (row.study, row.points_done, row.points_planned, row.points_failed) == ("sweep", 10, 12, 2)
    assert row.complete and row.finished, "a committed record is finished"

    single = _bodies()["fdt"]
    w2 = _make(store, "fdt", name="a_single", body=single)
    one = [r for r in store.list("fdt") if r.id == w2.id][0]
    assert one.study == "single"
    assert (one.points_done, one.points_planned, one.points_failed) == (None, None, None), \
        "a single run has no operating points; it must not report 0 of 0"

    # every other kind leaves all four alone
    for kind in ("prior", "calibration", "diagnostic"):
        other = _make(store, kind, name=f"o_{kind}", body=_bodies()[kind])
        s = [r for r in store.list(kind) if r.id == other.id][0]
        assert (s.study, s.points_done) == (None, None), kind
        assert s.finished is True, "a kind with no ``complete`` key is finished by construction"

    # the branch is DERIVED, not a hand-written pair of names
    assert {k for k, keys in mf.BODY_KEYS.items() if "complete" in keys} == {"simulation", "fdt"}, \
        "the two kinds whose manifest exists before the run has finished"
```

- [ ] **Step 2: Run it and watch it fail**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_artifact_store.py::test_an_fdt_row_carries_its_study_its_points_and_whether_it_finished -v`

Expected: FAIL with
`AssertionError: assert (None, None, None, None) == ('sweep', 10, 12, 2)` — `list` does not fill
the four fields yet.

- [ ] **Step 3: Fill them in `ArtifactStore.list`**

In `core/artifacts/store.py`, find:

```python
            finished = bool(body.get("complete")) if kind == "simulation" else True
```

Replace with:

```python
            # Derived from the SCHEMA, not from a hand-written kind name: a kind whose body carries
            # a ``complete`` flag is one whose manifest exists before the run has finished (the
            # cache, because it is resumable; an fdt record, because E2 keeps an interrupted one),
            # and for every other kind a manifest IS the finish -- ``_commit`` writes it last. The
            # old form named "simulation" alone and would have silently called every half-written
            # fdt record finished.
            finished = (bool(body.get("complete")) if "complete" in mf.BODY_KEYS[kind] else True)
```

Then find:

```python
            ident = (body.get("identity") or {}) if kind == "simulation" else {}
```

Replace with:

```python
            ident = (body.get("identity") or {}) if kind == "simulation" else {}
            # An fdt sweep's operating points, off the SAME manifest read (B2). A single run and a
            # comparison carry ``points: null``, so the counts stay None and the cell stays blank --
            # "0/0" would be a claim neither ever makes. ``_as_int`` is what keeps a hand-edited or
            # partially written body from taking the whole listing down.
            pts = body.get("points")
            pts = pts if isinstance(pts, dict) else {}
```

Then find:

```python
                               rows=coerced_rows,
                               variant=body.get("variant")))
```

Replace with:

```python
                               rows=coerced_rows,
                               variant=body.get("variant"),
                               study=body.get("study"),
                               points_done=_as_int(pts.get("done")),
                               points_planned=_as_int(pts.get("planned")),
                               points_failed=_as_int(pts.get("failed"))))
```

- [ ] **Step 4: Run it and watch it pass**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_artifact_store.py::test_an_fdt_row_carries_its_study_its_points_and_whether_it_finished tests/test_artifact_store.py::test_a_summary_says_whether_the_run_finished_for_every_kind -v`

Expected: PASS (2 passed).

- [ ] **Step 5: Write the failing test for the numeric sort**

In `tests/test_worker_dispatch.py`, add after
`test_cells_for_renders_each_kind_from_the_summary_alone`:

```python
def test_a_numeric_column_sorts_by_its_number_and_not_as_text():
    """Spec §1.2. ``Progress`` and ``Points`` are cells like "10/12 batches" and "9/12", and a text
    sort puts 10 before 9 -- the carry-forward docs/STATE.md recorded from piece 4, which this piece
    would have doubled by adding a second column of exactly that shape.

    The completeness rank still comes FIRST and still survives a descending sort (that is _Row's own
    property, pinned beside this), so what is asserted here is only the order WITHIN the complete
    rows, ascending and descending alike.
    """
    from core.gui.widgets.artifact_table import ArtifactTable, NUMERIC_COLUMNS, cell_key
    from tests._fixtures import qt_app
    qt_app()

    assert cell_key("Points", "9/12") < cell_key("Points", "10/12"), "9 sorts before 10"
    assert cell_key("Note", "9") < cell_key("Note", "10"), "a text column is unchanged: '10' < '9'?"
    assert cell_key("Note", "10") < cell_key("Note", "9"), "... yes: text order, as today"
    assert "Points" in NUMERIC_COLUMNS and "Progress" in NUMERIC_COLUMNS

    rows = [_row("fdt", id="s9", name="nine", points_done=9, points_planned=12, study="sweep"),
            _row("fdt", id="s10", name="ten", points_done=10, points_planned=12, study="sweep"),
            _row("fdt", id="s2", name="two", points_done=2, points_planned=12, study="sweep")]
    table = ArtifactTable()
    table.set_rows("fdt", rows)
    col = list(columns_for("fdt")).index("Points")

    table.apply_sort_state(col, 0)
    got = [table.topLevelItem(i).text(col) for i in range(table.topLevelItemCount())]
    assert got == ["2/12", "9/12", "10/12"], got

    table.apply_sort_state(col, 1)
    got = [table.topLevelItem(i).text(col) for i in range(table.topLevelItemCount())]
    assert got == ["10/12", "9/12", "2/12"], got
```

- [ ] **Step 6: Run it and watch it fail**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_worker_dispatch.py::test_a_numeric_column_sorts_by_its_number_and_not_as_text -v`

Expected: FAIL with
`ImportError: cannot import name 'NUMERIC_COLUMNS' from 'core.gui.widgets.artifact_table'`.

- [ ] **Step 7: Add the sort key**

In `core/gui/widgets/artifact_table.py`, find:

```python
DEFAULT_SORT = (1, 1)
```

Replace with:

```python
DEFAULT_SORT = (1, 1)

# The columns whose cell STARTS with a number and must sort by that number. Sorting them as text is
# the defect piece 4 handed forward ("10/12 batches" above "9/12"), and this piece adds "Points", a
# column of exactly that shape -- so the key lives here, named by column title, rather than in a
# per-kind branch that the next numeric column would have to remember to join. "Width" is here for
# the same reason: "100" sorts before "50" as text.
NUMERIC_COLUMNS: frozenset = frozenset({"Progress", "Points", "Width"})

_LEADING_NUMBER = re.compile(r"^\s*(-?\d+(?:\.\d+)?)")


def cell_key(column: str, text: str) -> tuple:
    """What one cell sorts on: ``(number, text)``.

    For a column that is not numeric the number is a constant, so the text decides and the order is
    exactly today's. For a numeric one the LEADING number decides and the text breaks ties, which is
    what makes "10/12" follow "9/12" and "10/12 batches" follow "9/12 batches" without this function
    having to know either cell's shape.

    A numeric cell with no leading number -- a blank ``Points`` for a single run -- sorts as ``-inf``,
    i.e. before every number ascending, which is exactly where the empty string it holds sorts today.
    The type is the same two-tuple for every column, so a comparison can never meet a float on one
    side and a string on the other.
    """
    if column not in NUMERIC_COLUMNS:
        return (0.0, text)
    m = _LEADING_NUMBER.match(text)
    return ((float(m.group(1)) if m is not None else float("-inf")), text)
```

Add `import re` to the module's imports, above the PySide6 ones:

```python
import re

from PySide6.QtCore import Qt, Signal
```

Then find:

```python
    def _key(self, col: int, descending: bool):
        """``(rank, the column's own text)``. ``!= descending`` is the XOR that cancels Qt's
        reversal, so the incomplete rows sit at the bottom under an ascending and a descending sort
        alike."""
        return ((not self._complete) != descending, self.text(col))

    def __lt__(self, other):
        tree = self.treeWidget()
        col = tree.sortColumn() if tree is not None else 0
        if col < 0:                                 # nothing sorted yet: rank by the first column
            col = 0
        if tree is None or not isinstance(other, _Row):     # neither happens; cheap to survive
            return self.text(col) < other.text(col)
        # .value, not int(): int(Qt.SortOrder) raises TypeError in PySide6 6.9.3.
        descending = bool(tree.header().sortIndicatorOrder().value)
        return self._key(col, descending) < other._key(col, descending)
```

Replace with:

```python
    def _key(self, col: int, descending: bool, column: str = ""):
        """``(rank, cell_key(column, text))``. ``!= descending`` is the XOR that cancels Qt's
        reversal, so the incomplete rows sit at the bottom under an ascending and a descending sort
        alike; ``cell_key`` is what makes a numeric column sort by its number (§1.2)."""
        return ((not self._complete) != descending, cell_key(column, self.text(col)))

    def __lt__(self, other):
        tree = self.treeWidget()
        col = tree.sortColumn() if tree is not None else 0
        if col < 0:                                 # nothing sorted yet: rank by the first column
            col = 0
        if tree is None or not isinstance(other, _Row):     # neither happens; cheap to survive
            return self.text(col) < other.text(col)
        # The column's TITLE, read off the header item, is what says whether this column is numeric.
        # Read here rather than stored on the row: one row is shown under one kind's header at a
        # time, and a stored title would go stale the moment set_rows re-labelled the tree.
        head = tree.headerItem()
        column = head.text(col) if head is not None else ""
        # .value, not int(): int(Qt.SortOrder) raises TypeError in PySide6 6.9.3.
        descending = bool(tree.header().sortIndicatorOrder().value)
        return self._key(col, descending, column) < other._key(col, descending, column)
```

- [ ] **Step 8: Run it and watch it pass**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_worker_dispatch.py::test_a_numeric_column_sorts_by_its_number_and_not_as_text -v`

Expected: PASS.

- [ ] **Step 9: Run the suites that own the table and the listing**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_worker_dispatch.py tests/test_artifact_browser.py tests/test_artifact_store.py -m "not slow" -q`

Expected: PASS. (If a piece-4 test asserted an order that only held because a numeric column sorted
as text, it names the column in its failure — report it rather than weakening the key.)

- [ ] **Step 10: Commit**

```bash
git add core/artifacts/store.py core/gui/widgets/artifact_table.py tests/test_artifact_store.py tests/test_worker_dispatch.py
git commit -m "store: fill an fdt row's study and points, and sort numeric columns by number"
```

#### Amendments (binding — these supersede the text above)

From the pre-flight scan against HEAD `673868d`. No ruling in P1–P82 overrides this task's design. Task 3 depends on this task (P81), so land this one first. One test line is wrong and is corrected here.

1. **Step 5, the test body: delete one false assertion.** `cell_key("Note", "9")` is `(0.0, "9")` and `cell_key("Note", "10")` is `(0.0, "10")`. `"9" < "10"` is False as text, so the line below fails whatever the implementation does, and it contradicts the line after it. Delete exactly this line:

   ```python
       assert cell_key("Note", "9") < cell_key("Note", "10"), "a text column is unchanged: '10' < '9'?"
   ```

   Keep the next line, which is correct:

   ```python
       assert cell_key("Note", "10") < cell_key("Note", "9"), "... yes: text order, as today"
   ```

   Everything else in Steps 5–8 stands.

---

### Task 3: `ArtifactWriter`'s progressive mode

**Why:** Spec §2.2 and decision **E2** — an interrupted run keeps its folder, marked unfinished.
§1.2's first ruling makes it a MODE on the existing writer rather than a second writer, because this
kind needs the payload, figure, `fig_sink` and `log.txt` handling the training cache's own manifest
function does without. §11's second risk row requires the six ordinary kinds' behaviour to be pinned
**before** the mode exists.

**Files:**
- Modify: `core/artifacts/store.py:34-54` (beside `RECENT_WRITE_SECONDS`), `:248-349`
  (`ArtifactWriter`)
- Modify: `core/artifacts/__init__.py:6-9` (re-export `PROGRESSIVE_KINDS`)
- Test: `tests/test_artifact_store.py`

**Interfaces:**
- Consumes: `manifest.BODY_KEYS["fdt"]`, `store.KIND_DIRS["fdt"]` (Task 1).
- Produces:
  - `store.PROGRESSIVE_KINDS: frozenset = frozenset({"fdt"})`
  - `ArtifactWriter.progressive: bool` (`kind in PROGRESSIVE_KINDS`)
  - `ArtifactWriter._wrote_anything: bool` — True once `payload()` or `figure_path()` has been called
  - `ArtifactWriter.refresh() -> None`
  - `core.artifacts.PROGRESSIVE_KINDS` (the package re-export)
- Guarantees later tasks rely on: `__enter__` writes a first manifest carrying every body key, with
  `complete` False; `__exit__` on an exception KEEPS the directory unless the exception is a
  `Refusal` raised before the first `payload()`/`figure_path()` call; `_commit` sets
  `body["complete"] = True` for a progressive kind.

- [ ] **Step 1: Pin the SIX ordinary kinds first — before the mode exists**

In `tests/test_artifact_store.py`, add after `test_writer_removes_the_directory_on_exception`:

```python
# The kinds ``store.create`` accepts that are NOT written progressively. Spelled out, and checked
# against _bodies below, so the pin that follows cannot quietly shrink as kinds are added.
ORDINARY_KINDS = ("prior", "posterior", "observation", "calibration", "inference", "diagnostic")


def test_every_ordinary_kind_still_removes_its_directory_on_a_failure(store):
    """Spec §11, risk row 2, and §8.2. The progressive mode touches ``ArtifactWriter``, which every
    kind uses, and piece 4's review caught a delete that destroyed a finished artifact -- so the six
    ordinary kinds' remove-on-exception is pinned HERE, before the mode exists, and this test is what
    says the mode changed nothing for them.

    Both exception shapes, because they arrive through the same ``__exit__``: a plain Exception, and
    a BaseException (the shape of gui.streams.WorkerCancelled, which is what a Cancel click raises).
    Each writer has written a payload first, so "nothing was written" cannot be what saves the
    directory.
    """
    class _Cancel(BaseException):
        pass

    assert set(ORDINARY_KINDS) == set(_bodies()) - {"fdt"}, \
        "a kind added to _bodies must be classified here: progressive, or removed on failure"

    for kind in ORDINARY_KINDS:
        for boom in (RuntimeError, _Cancel):
            with pytest.raises(boom):
                with store.create(kind, None, name=f"{kind}_{boom.__name__}") as w:
                    w.body = _bodies()[kind]
                    w.payload("results.json").write_text("{}", encoding="utf-8")
                    raise boom("the run failed")
            assert not w.dir.exists(), f"{kind} kept a half-written directory after {boom.__name__}"
        assert store.list(kind) == [], f"{kind} left a row behind"
```

- [ ] **Step 2: Run it and watch it PASS**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_artifact_store.py::test_every_ordinary_kind_still_removes_its_directory_on_a_failure -v`

Expected: PASS. This is the one test in the task that is green before the change: it records
today's behaviour so the mode can be shown not to have altered it. (If it fails now, stop — the
premise of §2.2 is wrong and the task report says so.)

- [ ] **Step 3: Write the failing tests for the mode**

In `tests/test_artifact_store.py`, add directly after it:

```python
def test_a_progressive_record_has_a_valid_manifest_from_its_first_moment(store):
    """Spec §2.2 step 1 and E2. ``validate`` refuses a PARTIAL body key set, so the first manifest
    cannot carry only what is known -- it carries every key of BODY_KEYS["fdt"] with the unknown ones
    null. Five are known before the run starts (study, settings, seed, notices, complete) and five
    are not (grid, points, offgrid, compared, results), which is what makes an in-flight record
    readable in the browser while it runs.

    ``complete`` is the WRITER's field, not the stage's: it is False here because the writer wrote it
    so, and it is the writer that sets it True at the commit. A stage that forgot to touch it cannot
    therefore produce a record that claims to have finished.
    """
    w = store.create("fdt", None, name="inflight")
    assert w.progressive is True and w._wrote_anything is False
    assert not w.dir.exists(), "create() mints the id and the path and creates NOTHING (spec §1.2)"
    w.body = {"study": "single", "settings": {"n_freqs": 2}, "seed": 5, "notices": []}
    with w:
        assert w.dir.is_dir()
        m = mf.from_json_text((w.dir / st.MANIFEST).read_text(encoding="utf-8"))
        assert set(m.body) == set(mf.BODY_KEYS["fdt"])
        assert m.body["complete"] is False and m.body["study"] == "single"
        assert [k for k in mf.BODY_KEYS["fdt"] if m.body[k] is None] == \
            ["grid", "points", "offgrid", "compared", "results"]
        row = store.list("fdt")[0]
        assert row.complete and not row.finished, "listed while it runs, and honestly unfinished"
        w.body["grid"] = {"omega_0": 1.0}
        w.refresh()
        assert mf.from_json_text((w.dir / st.MANIFEST).read_text(encoding="utf-8")).body["grid"] \
            == {"omega_0": 1.0}, "refresh re-writes the manifest as the run proceeds"
        w.payload("data.h5").write_bytes(b"numbers")
        assert w._wrote_anything is True
        assert mf.from_json_text((w.dir / st.MANIFEST).read_text(encoding="utf-8")) is not None
    assert store.list("fdt")[0].finished, "the commit is what makes it finished"
    assert store.get("fdt", w.id).body["complete"] is True
    assert store.get("fdt", w.id).payloads["data.h5"] == prov.sha256_file(w.dir / "data.h5"), \
        "payloads are hashed at the COMMIT, and the commit is where the hash lands"


def test_a_cancel_between_the_first_manifest_and_the_first_payload_keeps_the_record(store):
    """Review Focus 1, and E2. The record exists, nothing has been written into it, and the cancel is
    a BaseException -- the shape ``core/gui/streams.py``'s WorkerCancelled has. The folder must
    survive: a run that took hours and was stopped is exactly what E2 keeps a folder for, and the
    spontaneous spectrum inside it is what diagnoses why it was stopped.

    A REFUSAL at the same point must do the opposite (spec §2.2 step 3) and the two arrive through
    the same ``__exit__``, which is why both are asserted here and not in two places.
    """
    class _Cancel(BaseException):
        pass

    w = store.create("fdt", None, name="cancelled")
    w.body = {"study": "single", "settings": {}, "seed": 1, "notices": []}
    with pytest.raises(_Cancel):
        with w:
            raise _Cancel()
    assert w.dir.is_dir(), "E2: an interrupted record keeps its folder"
    row = [r for r in store.list("fdt") if r.id == w.id][0]
    assert row.complete and not row.finished, "a manifest, and an honest 'did not finish'"
    assert store.get("fdt", w.id).body["complete"] is False

    # the same point, a REFUSAL, nothing written: the directory goes
    w2 = store.create("fdt", None, name="refused")
    w2.body = {"study": "single", "settings": {}, "seed": 1, "notices": []}
    with pytest.raises(Refusal):
        with w2:
            raise Refusal("the band reaches below what the spectrum resolves", field="freq_bounds")
    assert not w2.dir.exists(), \
        "a pre-spend refusal must not leave a permanent empty record: the sweep can never clear one"

    # a refusal AFTER something was written is an interrupted run like any other
    w3 = store.create("fdt", None, name="refused_late")
    w3.body = {"study": "single", "settings": {}, "seed": 1, "notices": []}
    with pytest.raises(Refusal):
        with w3:
            w3.payload("data.h5").write_bytes(b"the spontaneous spectrum")
            raise Refusal("every grid frequency came back blank", field=None)
    assert w3.dir.is_dir() and (w3.dir / "data.h5").is_file(), \
        "what the run did measure is what the message tells the reader to look at"


def test_a_second_run_is_refused_by_name_while_an_unfinished_record_holds_it(store):
    """Review Focus 2. ``assert_name_free`` runs at ``create()``, before anything is spent, and a
    progressive record occupies its name from its first moment -- so a second run under the same name
    is refused at the click rather than colliding with a directory halfway through an hours-long
    measurement. The refusal carries ``field="name"``, which is what lets each front end name its own
    control.
    """
    w = store.create("fdt", None, name="repeat")
    w.body = {"study": "single", "settings": {}, "seed": 1, "notices": []}
    w.__enter__()             # entered and NEVER exited: the record is live on disk, mid-run
    assert (w.dir / st.MANIFEST).is_file() and not store.list("fdt")[0].finished

    with pytest.raises(st.StoreError, match="already exists") as exc:
        store.create("fdt", None, name="repeat")
    assert exc.value.field == "name"


def test_a_failure_inside_a_progressive_record_writes_the_log_up_to_it(store):
    """Spec §2.2 step 3: the final refresh on the failure path is what puts the run's records on
    disk. Without it the one document that says WHY the run stopped would exist only for runs that
    did not stop -- which is the opposite of when it is needed. The log is written from
    ``runs.current_run_log()``, which is thread-local, so it is written here from inside a real run.
    """
    import logging
    from core import runs

    def _boom():
        w = store.create("fdt", None, name="logged")
        w.body = {"study": "single", "settings": {}, "seed": 1, "notices": []}
        with w:
            logging.getLogger("core.test").info("the spontaneous campaign finished")
            raise RuntimeError("the driven campaign failed")

    entry = runs.public_entry(lambda: _boom())
    with pytest.raises(RuntimeError):
        entry()
    text, truncated = store.read_log("fdt", "logged")
    assert text is not None and "the spontaneous campaign finished" in text, text
    assert not truncated
```

- [ ] **Step 4: Run them and watch them fail**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_artifact_store.py -k "progressive or cancel_between or unfinished_record_holds or writes_the_log_up_to" -v`

Expected: FAIL, all four. The first is
`AttributeError: 'ArtifactWriter' object has no attribute 'progressive'`; the second fails at
`assert w.dir.is_dir()` (`__exit__` removes it today); the third at
`assert (w.dir / st.MANIFEST).is_file()` (no manifest until the commit); the fourth at
`assert text is not None` (`read_log` returns `(None, False)` — the directory was removed).

- [ ] **Step 5: Declare `PROGRESSIVE_KINDS`**

In `core/artifacts/store.py`, find:

```python
RECENT_WRITE_SECONDS = 300.0
```

Replace with:

```python
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
```

- [ ] **Step 6: Give the writer its mode**

In `core/artifacts/store.py`, find:

```python
class ArtifactWriter:
    """Context manager handed out by ``ArtifactStore.create``: creates the directory, hands out
    payload and figure paths, and on a clean exit hashes the payloads, writes the run's ``log.txt``
    and writes the manifest LAST. On ANY exception (a cancel included) the directory is removed and
    the exception re-raised, so a half-artifact never looks real."""

    def __init__(self, store, kind, cfg, *, name, note, id, created):
        self.store, self.kind, self.cfg = store, kind, cfg
        self.name, self.note, self.id, self.created = name, note, id, created
        self.dir = store.kind_dir(kind) / f"{name or mf.UNNAMED_DIR}__{id}"
        self.config = mf.config_from_cfg(cfg) if cfg is not None else {}
        self.parents, self.fingerprints, self.body = {}, {}, {}
        self._payloads, self._figures = [], []
        self.manifest = None

    def payload(self, filename: str) -> Path:
        if "/" in filename or "\\" in filename:
            raise StoreError(f"a payload is a file directly inside the artifact directory: {filename!r}")
        if filename not in self._payloads:
            self._payloads.append(filename)
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
        return p
```

Replace with:

```python
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
```

- [ ] **Step 7: Split the manifest build out of `_commit`, and add `refresh`**

In `core/artifacts/store.py`, find:

```python
    def __enter__(self):
        self.dir.mkdir(parents=True, exist_ok=False)
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            # The cleanup must never REPLACE the exception that ended the run -- the one the operator
            # needs -- with a PermissionError about a directory the index already ignores (it has no
            # manifest, so it is incomplete by definition and nothing will ever load it).
            try:
                _rmtree_retry(self.dir)
            except OSError as e:              # noqa: BLE001 -- the in-flight cause must propagate, not this
                warnings.warn(f"could not remove the incomplete artifact directory {self.dir} "
                              f"({type(e).__name__}: {e}); it has no manifest, so the store ignores it.",
                              stacklevel=2)
            return False
        self._commit()
        return False

    def _commit(self) -> None:
        hw = getattr(self.cfg, "hw", None)
        # missing_ok: this runs at the END of a run that may have taken days, and an input file moved
        # or edited meanwhile must be recorded as unhashed rather than raise here and lose everything
        # the run produced. Named in a warning so the gap is not silent.
        inputs = (prov.inputs_from_cfg(self.cfg, missing_ok=True) if self.cfg is not None
                  else {"bounds": None, "cell": None, "units": None, "model": None})
        gone = [k for k, v in inputs.items() if isinstance(v, dict) and v.get("sha256") is None]
        if gone:
            warnings.warn(
                f"{self.kind} artifact {self.id}: input file(s) "
                + ", ".join(f"{k} ({inputs[k]['path']})" for k in gone)
                + " could not be read at commit, so the manifest records them unhashed. The artifact "
                  "is written anyway -- losing a finished run to a moved input file would be worse.",
                stacklevel=3)
        d = dict(
            schema=mf.SCHEMA, kind=self.kind, id=self.id, name=self.name, created=self.created.isoformat(),
            note=self.note, prism=prov.git_info(config.REPO_ROOT),
            env=prov.env_info(hw if hw is not None else config.cpu_device()),
            inputs=inputs,
            config=self.config, parents=dict(self.parents), fingerprints=dict(self.fingerprints),
            payloads={f: prov.sha256_file(self.dir / f) for f in self._payloads},
            figures=list(self._figures), body=self.body,
        )
        self.manifest = mf.validate(d)
```

Replace with:

```python
    def __enter__(self):
        self.dir.mkdir(parents=True, exist_ok=False)
        if self.progressive:
            # The FIRST manifest, before a single number is computed (spec §2.2 step 1). ``validate``
            # refuses a partial body key set, so this carries every key: what the caller set between
            # create() and here, and null for the rest.
            self._write(hashed=False, complete=False)
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            if self.progressive and not (isinstance(exc, Refusal) and not self._wrote_anything):
                # E2. The run was interrupted or it failed AFTER spending something: the folder stays,
                # ``complete`` stays False, and one last refresh puts the log up to the failure on
                # disk -- that log is the document that says why it stopped.
                try:
                    self._write(hashed=False, complete=False)
                except Exception as e:        # noqa: BLE001 -- the in-flight cause must propagate
                    warnings.warn(f"could not refresh the unfinished {self.kind} record {self.dir} "
                                  f"({type(e).__name__}: {e}); it keeps the manifest it had.",
                                  stacklevel=2)
                return False
            # The cleanup must never REPLACE the exception that ended the run -- the one the operator
            # needs -- with a PermissionError about a directory the index already ignores (it has no
            # manifest, so it is incomplete by definition and nothing will ever load it).
            try:
                _rmtree_retry(self.dir)
            except OSError as e:              # noqa: BLE001 -- the in-flight cause must propagate, not this
                warnings.warn(f"could not remove the incomplete artifact directory {self.dir} "
                              f"({type(e).__name__}: {e}); it has no manifest, so the store ignores it.",
                              stacklevel=2)
            return False
        self._commit()
        return False

    def refresh(self) -> None:
        """Re-write the manifest and ``log.txt`` for a record still being written (spec §2.2 step 2).

        Called by the stage at points it chooses -- after the spontaneous campaign, after each
        operating point, after each figure -- so a browser row, a listing and the log all follow a
        run that may take hours. The write is the same atomic one ``_commit`` performs.

        Payloads are NOT hashed here: a file still being appended to would hash to a value that is
        wrong the moment it is written, so an unfinished record lists its payloads with a null hash
        and the real hashes land at the commit. ``load_fdt`` treats a null hash as "not yet".
        """
        if not self.progressive:
            raise StoreError(f"{self.kind} artifacts are written in one step, so there is nothing to "
                             f"refresh; only {sorted(PROGRESSIVE_KINDS)} are written progressively")
        self._write(hashed=False, complete=False)

    def _write(self, *, hashed: bool, complete: bool) -> None:
        """Build, validate and write the manifest (and the run's log beside it)."""
        self.manifest = mf.validate(self._manifest_dict(hashed=hashed, complete=complete))
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
                stacklevel=4)
        return inputs

    def _manifest_dict(self, *, hashed: bool, complete: bool) -> dict:
        hw = getattr(self.cfg, "hw", None)
        if self._prism is None:
            self._prism = prov.git_info(config.REPO_ROOT)
        if self._env is None:
            self._env = prov.env_info(hw if hw is not None else config.cpu_device())
        body = dict(self.body)
        if self.progressive:
            # ``complete`` is the WRITER's field: it is what says whether this record's run reached
            # the end, and the writer is the only party that knows. Filling the rest with null is
            # what lets the caller set a key when it learns it rather than up front.
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
```

Then delete what is left of the old `_commit` body — find and remove:

```python
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
        _write_manifest(self.dir, self.manifest)
```

(it has moved into `_write` and `_write_log`).

Finally, in `core/artifacts/__init__.py`, find:

```python
                    LoadedInference, LoadedDiagnostic, LoadedFdt, write_simulation_manifest,
```

Replace with:

```python
                    LoadedInference, LoadedDiagnostic, LoadedFdt, PROGRESSIVE_KINDS,
                    write_simulation_manifest,
```

- [ ] **Step 8: Run the four new tests and the ordinary-kind pin**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_artifact_store.py -k "progressive or cancel_between or unfinished_record_holds or writes_the_log_up_to or every_ordinary_kind" -v`

Expected: PASS (5 passed). If `_manifest_dict`'s `complete` handling has upset
`test_create_list_get_round_trip_per_kind` (which asserts `m.body == body`), the fdt body in
`_bodies()` carries `complete: True` and the commit writes True — they agree; a failure there means
the fixture was changed, not this code.

- [ ] **Step 9: Run the whole store suite**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_artifact_store.py -m "not slow" -q`

Expected: PASS.

- [ ] **Step 10: Commit**

```bash
git add core/artifacts/store.py core/artifacts/__init__.py tests/test_artifact_store.py
git commit -m "store: a progressive writer mode, so an interrupted fdt record keeps its folder"
```

#### Amendments (binding — these supersede the text above)

From the pre-flight scan against HEAD `673868d`.

1. **Order (P81).** This task depends on Task 2 as well as Task 1: three of its assertions (`row.complete and not row.finished`, and `store.list("fdt")[0].finished` after the commit) need Task 2's `finished = (bool(body.get("complete")) if "complete" in mf.BODY_KEYS[kind] else True)`. Before starting, confirm that line is in `core/artifacts/store.py`. If it is not, Task 2 has not landed: stop and say so in the report. In **Interfaces → Consumes**, read: `manifest.BODY_KEYS["fdt"]`, `store.KIND_DIRS["fdt"]` (Task 1), and `ArtifactStore.list`'s BODY_KEYS-derived `finished` (Task 2).

2. **Step 4, the fourth prediction is corrected.** Replace "the fourth at `assert text is not None` (`read_log` returns `(None, False)` — the directory was removed)" with: the fourth fails at the `store.read_log("fdt", "logged")` call, with `core.artifacts.store.StoreError: no complete fdt artifact named or id'd 'logged'`. The directory was removed, so `_find` finds no manifest and `read_log` refuses rather than returning `(None, False)`.

3. **Step 7, `_inputs`: keep today's warning attribution.** The old `_commit` warned with `stacklevel=3` (`_commit` → `__exit__` → the caller's `with` line). The new call chain is `_inputs` → `_manifest_dict` → `_write` → `_commit` → `__exit__` → caller, so the same attribution needs `stacklevel=6`. In the `_inputs` you add, replace

   ```python
                   stacklevel=4)
   ```

   with

   ```python
                   # _inputs <- _manifest_dict <- _write <- _commit <- __exit__ <- the caller's ``with``:
                   # the frame the old _commit's stacklevel=3 named, so the six ordinary kinds' warning
                   # still points at the caller's code and not at this module.
                   stacklevel=6)
   ```

Nothing else in P1–P82 overrides this task. P15, P47, P48 and P49 are already what Steps 5–7 build.

#### Controller rulings (pre-flight, binding — these supersede the amendments above and the task text)

- **F7** — at the end of `test_a_cancel_between_the_first_manifest_and_the_first_payload_keeps_the_record`, add the store half of spec §8.2's "the leftover sweep never offers it": `backdate_tree(w.dir)` (already imported in that suite), then `with pytest.raises(st.StoreError): store.remove_incomplete("fdt", w.dir.name)`, then assert no `store.list("fdt")` row has `complete` false. The tool half is Task 30's.
- **F30** — where a progressive record falls through to the ordinary removal (the refused-before-anything-was-written branch) and `_rmtree_retry` then fails, word the warning by mode: for a progressive record it must NOT say "it has no manifest, so the store ignores it" — it has one. Say it keeps a manifest, is listed as unfinished, and can be removed with the delete action.
- **P81** — this task runs after Task 2, not only Task 1: its assertions on `Summary.finished` for `fdt` need T2's `store.list` branch.

---

### Task 4: loose files and legacy directories

**Why:** Spec §2.4 item 11 and §6.3, decision **E10** — the tidy-up command learns to see the legacy
loose files so the owner can clear them; nothing is deleted on their behalf. `_entries` iterates
directories only, so a file inside a kind directory is invisible to every listing, and the store
never walks its own root, so a `crossval/` beside the kind directories is invisible too. The owner
has a live instance: two PNGs under `Artifacts/fdt/` from a run stamped `20260915_153042` (spec §1).

**Read `remove_incomplete` first** (`core/artifacts/store.py:648-748`): both new removers share its
recency guard and its samefile+realpath hardening, and neither may reach what the other owns.

**Files:**
- Modify: `core/artifacts/store.py:34-54` (beside `PROGRESSIVE_KINDS`), `:61-73` (beside `Accept`),
  `:648-748` (after `remove_incomplete`)
- Modify: `core/artifacts/__init__.py:6-9` (re-export `LooseFile`, `LEGACY_DIRS`)
- Test: `tests/test_artifact_store.py`

**Interfaces:**
- Consumes: `KIND_DIRS`, `RECENT_WRITE_SECONDS`, `_newest_mtime`, `_rmtree_retry`, `StoreError`.
- Produces:
  - `store.LEGACY_DIRS: tuple = ("crossval",)`
  - `store.LooseFile` — a frozen dataclass with `name: str`, `size: int`, `mtime: float`
  - `ArtifactStore.loose_files(kind) -> list[LooseFile]` (sorted by name)
  - `ArtifactStore.remove_loose(kind, filename) -> None`
  - `ArtifactStore.legacy_dirs() -> list[str]`
  - `ArtifactStore.remove_legacy(name) -> None`
  - `core.artifacts.LooseFile`, `core.artifacts.LEGACY_DIRS`
- Every refusal here is a `StoreError` with `field="artifact"`, like `remove_incomplete`'s.

- [ ] **Step 1: Write the failing test for the loose files**

In `tests/test_artifact_store.py`, add after `test_every_ordinary_kind_still_removes_its_directory_on_a_failure`:

```python
def test_loose_files_sees_what_no_listing_can_and_removes_one_by_name(store):
    """E10 and spec §6.3. ``_entries`` iterates DIRECTORIES only, so a file sitting inside a kind
    directory is invisible to every listing in both front ends -- and the owner has two of them, the
    PNGs an older FDT run dropped into ``Artifacts/fdt``. This is the only route by which either
    front end can see or clear one; nothing is removed on the owner's behalf.

    It can never reach a directory: the candidate is matched against ``loose_files``'s own entries,
    which are files. ``remove_incomplete`` owns the directories and refuses a non-directory, so the
    two calls cannot do each other's job -- the same safety property piece 4 gave ``delete`` and
    ``remove_incomplete`` (B7).
    """
    d = store.kind_dir("fdt")
    d.mkdir(parents=True, exist_ok=True)
    (d / "fdt_ratio_20260915_153042.png").write_bytes(b"\x89PNG stray")
    (d / "fdt_spont_20260915_153042.png").write_bytes(b"\x89PNG stray too")
    w = _make(store, "fdt", name="real", body=_bodies()["fdt"])   # writes results.json INSIDE its dir

    loose = store.loose_files("fdt")
    assert [f.name for f in loose] == ["fdt_ratio_20260915_153042.png",
                                       "fdt_spont_20260915_153042.png"], \
        "a valid record's payloads are never offered: they are inside its own directory"
    assert loose[0].size == len(b"\x89PNG stray") and loose[0].mtime > 0

    with pytest.raises(st.StoreError, match="was written") as recent:
        store.remove_loose("fdt", "fdt_ratio_20260915_153042.png")
    assert recent.value.field == "artifact", "the recency guard, shared with remove_incomplete"

    backdate_tree(d)
    store.remove_loose("fdt", "fdt_ratio_20260915_153042.png")
    assert [f.name for f in store.loose_files("fdt")] == ["fdt_spont_20260915_153042.png"]
    assert store.get("fdt", w.id).id == w.id, "a real record is untouched"

    # what it refuses
    for bad in ("real__" + w.id, "sub/x.png", "..", "", "nosuch.png"):
        with pytest.raises(st.StoreError):
            store.remove_loose("fdt", bad)
    assert (store.path("fdt", w.id)).is_dir(), "the record's directory is remove_incomplete's, not this call's"
    with pytest.raises(st.StoreError, match="unknown artifact kind"):
        store.loose_files("plot")
    assert store.loose_files("prior") == [], "a kind directory that does not exist has no loose files"
```

- [ ] **Step 2: Run it and watch it fail**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_artifact_store.py::test_loose_files_sees_what_no_listing_can_and_removes_one_by_name -v`

Expected: FAIL with `AttributeError: 'ArtifactStore' object has no attribute 'loose_files'`.

- [ ] **Step 3: Add `LooseFile` and `loose_files`**

In `core/artifacts/store.py`, find:

```python
@dataclass
class Summary:
```

Replace with:

```python
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
```

Then find:

```python
    def sweep_incomplete(self, entries) -> "tuple[list, list]":
```

Replace with:

```python
    def loose_files(self, kind: str) -> list:
        """Every FILE sitting directly inside ``kind``'s directory, oldest rule first: a kind
        directory holds artifact directories and nothing else.

        The complement of ``_entries``, which iterates directories only -- so between them the two
        account for everything under a kind directory, and neither can see what the other owns. A
        valid record's payloads are never here: they are inside the record's own directory.

        A kind directory that does not exist has no loose files; that is not an error, because a
        store is allowed to be empty.
        """
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
```

- [ ] **Step 4: Run it and watch it pass**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_artifact_store.py::test_loose_files_sees_what_no_listing_can_and_removes_one_by_name -v`

Expected: PASS.

- [ ] **Step 5: Write the failing test for the legacy directories**

In `tests/test_artifact_store.py`, add directly after it:

```python
def test_a_legacy_directory_beside_the_kind_directories_is_seen_and_cleared(store):
    """E10's second half. The store never walks its own ROOT, so a ``crossval/`` written by an older
    build sits beside the kind directories and no command in either front end can see it. The list of
    such names is a CLOSED literal: a store root is not a place to guess at, and offering to remove
    whatever happens to be there is how a sweep destroys something nobody meant it to.

    It can never reach a kind directory: a name that is one is refused even if it were listed, which
    is the mirror of ``remove_loose`` never reaching a directory.
    """
    assert st.LEGACY_DIRS == ("crossval",), \
        "the one directory an older build wrote beside the kind directories (spec §2.1)"
    assert not set(st.LEGACY_DIRS) & set(KIND_DIRS.values()), \
        "a legacy name that is also a kind directory would make this call reach a real artifact"

    assert store.legacy_dirs() == [], "nothing on disk, nothing offered"
    old = store.root / "crossval"
    old.mkdir(parents=True)
    (old / "sweep_S.h5").write_bytes(b"an older build's numbers")
    assert store.legacy_dirs() == ["crossval"]

    with pytest.raises(st.StoreError, match="was written"):
        store.remove_legacy("crossval")
    backdate_tree(old)
    store.remove_legacy("crossval")
    assert not old.exists() and store.legacy_dirs() == []

    for bad in ("priors", "nosuch", "../etc", "", "."):
        with pytest.raises(st.StoreError):
            store.remove_legacy(bad)
    assert (store.root / "priors").exists() or True, "a kind directory is never this call's to remove"
```

Note: import `KIND_DIRS` at the top of the test with
`from core.artifacts.store import KIND_DIRS` if the module does not already bind it (`st.KIND_DIRS`
is available either way — use `st.KIND_DIRS` and drop the import if so).

- [ ] **Step 6: Run it and watch it fail**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_artifact_store.py::test_a_legacy_directory_beside_the_kind_directories_is_seen_and_cleared -v`

Expected: FAIL with `AttributeError: module 'core.artifacts.store' has no attribute 'LEGACY_DIRS'`.

- [ ] **Step 7: Add `LEGACY_DIRS`, `legacy_dirs` and `remove_legacy`**

In `core/artifacts/store.py`, find:

```python
PROGRESSIVE_KINDS: frozenset = frozenset({"fdt"})
```

Replace with:

```python
PROGRESSIVE_KINDS: frozenset = frozenset({"fdt"})
# Directories that may sit BESIDE the kind directories because an older build wrote them. A CLOSED
# literal, never a scan: the store never walks its own root, and offering to remove whatever happens
# to be under it is how a tidy-up destroys something nobody meant it to. ``legacy_dirs`` and
# ``remove_legacy`` are the only way either front end can see or clear one (piece 5, E10), and
# nothing is removed on the owner's behalf. A name here may never also be a KIND_DIRS value.
LEGACY_DIRS: tuple = ("crossval",)
```

Then find:

```python
    def unnamed(self, kind: str, *, older_than: "timedelta | None" = None) -> list:
```

Replace with:

```python
    def legacy_dirs(self) -> list:
        """The names in ``LEGACY_DIRS`` that actually exist as directories under this root.

        A name that is also a kind directory is skipped whatever the literal says: that is the
        structural guarantee that this family can never reach a real artifact, rather than a promise
        the literal is trusted to keep.
        """
        taken = set(KIND_DIRS.values())
        return [n for n in LEGACY_DIRS
                if n not in taken and (self.root / n).is_dir() and not (self.root / n).is_symlink()]

    def remove_legacy(self, name: str) -> None:
        """Remove one legacy directory under this root, by its own name.

        Refuses, as a StoreError with ``field="artifact"``: a name not in ``LEGACY_DIRS``; a name
        that is a KIND directory (belt beside the braces in ``legacy_dirs``); a name that is not a
        direct child; a name that is not a directory or is a link (a junction called ``crossval``
        pointing at C:\\ must not be followed); and a tree written within ``RECENT_WRITE_SECONDS``,
        the same guard ``remove_incomplete`` applies, for the same reason.

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
        if candidate.is_symlink() or not candidate.is_dir():
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
```

Finally, in `core/artifacts/__init__.py`, find:

```python
                    LoadedInference, LoadedDiagnostic, LoadedFdt, PROGRESSIVE_KINDS,
                    write_simulation_manifest,
```

Replace with:

```python
                    LoadedInference, LoadedDiagnostic, LoadedFdt, LooseFile, PROGRESSIVE_KINDS,
                    LEGACY_DIRS, write_simulation_manifest,
```

- [ ] **Step 8: Run both tests and watch them pass**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_artifact_store.py -k "loose_files or legacy_directory" -v`

Expected: PASS (2 passed).

- [ ] **Step 9: Run the store suite**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_artifact_store.py -m "not slow" -q`

Expected: PASS.

- [ ] **Step 10: Commit**

```bash
git add core/artifacts/store.py core/artifacts/__init__.py tests/test_artifact_store.py
git commit -m "store: see and clear loose files and a legacy directory"
```

#### Amendments (binding — these supersede the text above)

From the pre-flight scan against HEAD `673868d`. No ruling in P1–P82 overrides this task's design. The corrections below fix its dependency and three defects in its own test and code.

1. **Precondition: Task 3 must have landed.** Three anchors here are Task 3's text, not Task 1's: the Step 1 placement (`test_every_ordinary_kind_still_removes_its_directory_on_a_failure`), the Step 7 anchor `PROGRESSIVE_KINDS: frozenset = frozenset({"fdt"})`, and the Step 7 `__init__.py` anchor `LoadedInference, LoadedDiagnostic, LoadedFdt, PROGRESSIVE_KINDS,` / `write_simulation_manifest,`. If `PROGRESSIVE_KINDS` is not in `core/artifacts/store.py`, stop and report that Task 3 has not landed. Do not re-derive the anchors.

2. **Step 3, `loose_files`: an unknown kind carries `field="artifact"`, as this task's Interfaces promise for every refusal.** `kind_dir()`'s own refusal has no field key. In `loose_files`, replace the first line of the body

   ```python
           d = self.kind_dir(kind)
           if not d.is_dir():
               return []
   ```

   with

   ```python
           if kind not in KIND_DIRS:
               # Not kind_dir()'s own refusal, which carries no field key -- remove_incomplete's rule.
               raise StoreError(f"unknown artifact kind {kind!r}", field="artifact")
           d = self.kind_dir(kind)
           if not d.is_dir():
               return []
   ```

   The Step 1 test's `match="unknown artifact kind"` still holds.

3. **Step 5, the legacy-directory test.** `KIND_DIRS` is not bound in `tests/test_artifact_store.py`, which imports the module as `st`. Replace

   ```python
       assert not set(st.LEGACY_DIRS) & set(KIND_DIRS.values()), \
   ```

   with

   ```python
       assert not set(st.LEGACY_DIRS) & set(st.KIND_DIRS.values()), \
   ```

   and delete the "Note: import `KIND_DIRS` at the top of the test ..." paragraph. Add no import.

   Also replace the vacuous tail

   ```python
       for bad in ("priors", "nosuch", "../etc", "", "."):
           with pytest.raises(st.StoreError):
               store.remove_legacy(bad)
       assert (store.root / "priors").exists() or True, "a kind directory is never this call's to remove"
   ```

   with

   ```python
       kind_dir = store.root / st.KIND_DIRS["prior"]
       kind_dir.mkdir(parents=True, exist_ok=True)
       backdate_tree(kind_dir)                  # old enough that only the name rule can protect it
       for bad in ("priors", "nosuch", "../etc", "", "."):
           with pytest.raises(st.StoreError) as exc:
               store.remove_legacy(bad)
           assert exc.value.field == "artifact", bad
       assert kind_dir.is_dir(), "a kind directory is never this call's to remove"
   ```

   (`backdate_tree` is already imported at the top of the module.)

#### Controller rulings (pre-flight, binding — these supersede the amendments above and the task text)

- **F8** — this task runs after Task 3: its `PROGRESSIVE_KINDS` and `__init__.py` anchors are text Task 3 produces.

---

### Task 5: `render_lineage`'s `compared` branch

**Why:** Spec §1.2, §2.4 item 12 and §7.3. A comparison names the runs it drew **in its body, not in
`parents`**, because the parents block is a flat `{key: id}` map and cannot carry an arbitrary number
of ids without widening the store's contract for every kind. Deleting a run a comparison used is
therefore not refused — and without this branch the promise that its lineage still says what it drew
is unreachable, because `render_lineage` walks `m.parents` only and never touches the body.

**Files:**
- Modify: `core/artifacts/report.py:192-254` (`render_lineage`)
- Test: `tests/test_artifact_store.py`

**Interfaces:**
- Consumes: `manifest.BODY_KEYS["fdt"]`'s `compared` key, `store.KIND_DIRS` (Task 1).
- Produces: the **shape of `body["compared"]`**, which Tasks 36–39 write:

  ```python
  body["compared"] = {"mode": "cells",            # the comparison mode
                      "records": [{"kind": "fdt", "id": "<id>", "name": "<name>"}, ...]}
  ```

  `render_lineage` reads `compared["mode"]` and `compared["records"]`; a record entry with no
  `kind` is read as `"fdt"`. Anything else in the dict is ignored by the renderer.

- [ ] **Step 1: Write the failing test**

In `tests/test_artifact_store.py`, add after
`test_render_lineage_walks_the_chain_and_prints_a_missing_parent`:

```python
def test_render_lineage_resolves_what_a_comparison_compared(store):
    """§7.3 and checklist 12. A comparison names the runs it drew in its BODY, because ``parents`` is
    a flat {key: id} map that cannot carry an arbitrary number of ids without widening the store's
    contract for every kind -- so deleting a record a comparison used is NOT refused, and this branch
    is the only thing that keeps the comparison's own provenance readable afterwards.

    A record the store no longer holds prints MISSING with its kind and id, the same wording the
    parents walk already uses: skipping it would make a broken chain read as a complete one, which is
    the one thing a provenance report must never do.
    """
    from core.artifacts import render_lineage
    a = _make(store, "fdt", name="cell_a", body=_bodies()["fdt"])
    b = _make(store, "fdt", name="cell_b", body=_bodies()["fdt"])
    comp_body = _bodies()["fdt"]
    comp_body.update(study="comparison", seed=None, grid=None, offgrid=None, results=None,
                     compared={"mode": "cells",
                               "records": [{"kind": "fdt", "id": a.id, "name": "cell_a"},
                                           {"kind": "fdt", "id": b.id, "name": "cell_b"}]})
    c = _make(store, "fdt", name="two_cells", body=comp_body)

    text = render_lineage(store, "fdt", c.id)
    assert "  compared (cells):" in text, text
    assert f"    fdt cell_a [{a.id}]" in text, text
    assert f"    fdt cell_b [{b.id}]" in text, text
    assert "MISSING" not in text, "both are on disk"

    store.delete("fdt", a.id)
    text = render_lineage(store, "fdt", c.id)
    assert f"    MISSING fdt [{a.id}]" in text, text
    assert str(store.kind_dir("fdt")) in text, "where it was looked for"
    assert f"    fdt cell_b [{b.id}]" in text, "the one still there is unaffected"

    # a record with no comparison says nothing extra
    assert "compared" not in render_lineage(store, "fdt", b.id)
```

- [ ] **Step 2: Run it and watch it fail**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_artifact_store.py::test_render_lineage_resolves_what_a_comparison_compared -v`

Expected: FAIL with `AssertionError` on `assert "  compared (cells):" in text` — the body is never
read, so nothing about the comparison appears.

- [ ] **Step 3: Add the branch**

In `core/artifacts/report.py`, find:

```python
        pairs = [(pk, m.parents[pk]) for pk in sorted(m.parents) if m.parents[pk]]
        out.append(f"{INDENT}parents: " + (", ".join(f"{pk}={pid}" for pk, pid in pairs) or NONE))
        for pk, pid in pairs:
```

Replace with:

```python
        pairs = [(pk, m.parents[pk]) for pk in sorted(m.parents) if m.parents[pk]]
        out.append(f"{INDENT}parents: " + (", ".join(f"{pk}={pid}" for pk, pid in pairs) or NONE))
        out += _compared_lines(store, m)
        for pk, pid in pairs:
```

and add, directly above `def render_lineage(`:

```python
def _compared_lines(store, m) -> list:
    """What a comparison DREW, resolved (spec §7.3).

    A comparison names its sources in the body and not in ``parents``: that block is a flat
    ``{key: id}`` map read as ``m.parents.get(pk) == id_``, so it cannot carry an arbitrary number of
    ids without widening the store's contract for every kind. The consequence is deliberate --
    deleting a record a comparison used is NOT refused -- and these lines are what keep the
    comparison honest about it, printing ``MISSING <kind> [<id>]`` in the wording the parents walk
    above already uses.

    Reads ``body["compared"]`` with ``.get``: every other kind's body has no such key, and a body
    that has it null is a record of this kind that is not a comparison.
    """
    from .store import KIND_DIRS, StoreError
    comp = m.body.get("compared") if isinstance(m.body, dict) else None
    if not isinstance(comp, dict):
        return []
    out = [f"{INDENT}compared ({comp.get('mode') or NONE}):"]
    for rec in comp.get("records") or []:
        rec = rec if isinstance(rec, dict) else {}
        # "fdt" is the default because a comparison of measurements is what this block exists for;
        # a kind this build does not know is named rather than skipped, like an unknown parent key.
        ck, cid = str(rec.get("kind") or "fdt"), str(rec.get("id") or "")
        if ck not in KIND_DIRS:
            out.append(f"{INDENT}{INDENT}(compared entry names kind {ck!r}, which is no artifact "
                       f"kind this build knows)")
            continue
        try:
            cm = store.get(ck, cid)
        except StoreError:
            cm = None
        if cm is None:
            out.append(f"{INDENT}{INDENT}MISSING {ck} [{cid}]  -- no complete {ck} with that id "
                       f"under {store.kind_dir(ck)}")
        else:
            out.append(f"{INDENT}{INDENT}{ck} {cm.name or '(unnamed)'} [{cm.id}]")
    return out
```

- [ ] **Step 4: Run it and watch it pass**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_artifact_store.py -k "render_lineage" -v`

Expected: PASS (2 passed — the new one and the existing chain walk, which must be unchanged).

- [ ] **Step 5: Run the two suites that render the report**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_artifact_store.py tests/test_artifact_browser.py -m "not slow" -q`

Expected: PASS. The browser's `test_the_lineage_report_writes_exactly_what_render_lineage_returns`
compares bytes against `render_lineage` itself, so it follows any wording change automatically.

- [ ] **Step 6: Commit**

```bash
git add core/artifacts/report.py tests/test_artifact_store.py
git commit -m "report: a lineage names what a comparison drew, and what has since gone"
```

#### Amendments (binding — these supersede the text above)

None. The pre-flight scan against HEAD `673868d` found every anchor verbatim, and no ruling in P1–P82 changes this task. Its `body["compared"]` shape is the interface contract's (P5).

---

### Task 6: the delete confirmation names what is half-written

**Why:** Spec §2.4 item 15. `_delete_prompt` hard-codes the one kind that can be unfinished
(`if s.kind == "simulation" and not s.finished:`). An unfinished `fdt` record — the whole point of
**E2** — would otherwise be deleted behind a prompt that says nothing about what is half-written.

**Files:**
- Modify: `core/gui/screens/artifact_screen.py:86-104` (`_delete_prompt`)
- Test: `tests/test_artifact_browser.py`

**Interfaces:**
- Consumes: `Summary.finished`, `Summary.study`, `Summary.points_done`, `Summary.points_planned`
  (Tasks 1–2), and the progressive writer that produces an unfinished record on disk (Task 3).
- Produces: `artifact_screen._unfinished_note(s) -> str` — the one sentence that says what deleting
  an unfinished record of that kind destroys. Task 31 does not use it; nothing else does.

- [ ] **Step 1: Write the failing test**

In `tests/test_artifact_browser.py`, add after
`test_deleting_an_unfinished_cache_names_its_batches_and_defaults_to_no`:

```python
def test_deleting_an_unfinished_fdt_record_names_what_is_half_written(store, monkeypatch):
    """Checklist 15. The confirmation hard-coded the ONE kind that could be unfinished, and E2 makes
    a second one -- an fdt record whose run was cancelled or failed keeps its folder with whatever it
    measured inside. Deleting that behind a prompt that says only "this cannot be undone" throws away
    hours of measurement without saying so.

    There is no resume for these runs (spec §1.3), so the sentence says that too: unlike a cache,
    whose batches a later run continues from, this one starts again from the beginning.
    """
    from PySide6.QtWidgets import QMessageBox
    from core.gui.screens.artifact_screen import _delete_prompt
    from tests._fixtures import SHOWN, artifact_screen, qt_app
    qt_app()

    w = store.create("fdt", None, name="stopped")
    w.body = {"study": "sweep", "settings": {}, "seed": 3, "notices": [],
              "points": {"param": "S", "planned": 12, "done": 7, "failed": 1}}
    with pytest.raises(RuntimeError):
        with w:
            w.payload("data.h5").write_bytes(b"seven points' worth")
            raise RuntimeError("the eighth point failed")

    scr = artifact_screen(store)
    _show_kind(scr, "fdt")
    s = _select_ref(scr.table, w.id)
    assert s.complete and not s.finished and s.points_done == 7

    text, detail = _delete_prompt(s)
    assert text == f"Delete fdt {s.label}?"
    assert "UNFINISHED" in detail, detail
    assert "7 of 12" in detail, "what it had measured when it stopped"
    assert "no resume" in detail or "from the start" in detail, detail
    assert "cannot be undone" in detail

    scr._delete()                                     # exec() returns 0 -> not Yes
    box = SHOWN[-1]
    assert box.button(QMessageBox.No) is box.defaultButton(), "No must be the default"
    assert "UNFINISHED" in box.informativeText(), box.informativeText()
    assert store.get("fdt", w.id).id == w.id, "No must leave it on disk"

    # a FINISHED record says nothing of the sort
    done = store.create("fdt", None, name="finished_one")
    with done as d:
        d.body = {"study": "single", "settings": {}, "seed": 4, "notices": []}
    scr.refresh()
    s2 = _select_ref(scr.table, done.id)
    assert s2.finished
    assert "UNFINISHED" not in _delete_prompt(s2)[1], _delete_prompt(s2)[1]
```

- [ ] **Step 2: Run it and watch it fail**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_artifact_browser.py::test_deleting_an_unfinished_fdt_record_names_what_is_half_written -v`

Expected: FAIL with `AssertionError: id 20260922T...\nThis removes ...` on
`assert "UNFINISHED" in detail` — the prompt's only conditional branch names the simulation kind.

- [ ] **Step 3: Generalise the prompt**

In `core/gui/screens/artifact_screen.py`, find:

```python
def _delete_prompt(s) -> tuple:
    """``(text, informative)`` for the confirmation: what goes, and what cannot come back.

    An UNFINISHED cache names its committed BATCHES: ``finished`` is not ``complete`` (B3), and those
    batches are the only thing deleting one destroys that a later run could not simply remake.

    Batches and not rows, deliberately (P2): ``rows`` is written by ``mark_complete`` alone --
    ``training_checkpoint.save`` passes none -- so every real mid-run cache has ``rows is None``, and
    a ``sum(())`` here would print a confident, false "0 rows". Where the row counts come from is
    said instead.
    """
    lines = [f"id {s.id}"]
    if s.kind == "simulation" and not s.finished:
        lines.append(f"This training cache is UNFINISHED: {s.batches_done} committed batch(es). "
                     "Deleting it throws those batches away and a later run starts from zero. "
                     "(Only the batch count is known while a cache is running: the rows are "
                     "recorded when the cache finishes.)")
    lines.append(f"This removes {s.path} and everything in it, and cannot be undone.")
    return f"Delete {s.kind} {s.label}?", "\n".join(lines)
```

Replace with:

```python
def _unfinished_note(s) -> str:
    """The one sentence that says what deleting an UNFINISHED record of this kind destroys.

    TWO kinds can be unfinished, not one (piece 5, E2): a training cache, whose manifest exists from
    its first batch because it is resumable, and an fdt record, whose folder survives a cancel or a
    crash so that what it measured is still readable. The old form named the cache by hand and would
    have deleted a half-measured sweep behind a prompt that said nothing about it.

    The cache names its committed BATCHES and not its rows, deliberately (P2): ``rows`` is written by
    ``mark_complete`` alone -- ``training_checkpoint.save`` passes none -- so every real mid-run
    cache has ``rows is None``, and a ``sum(())`` here would print a confident, false "0 rows".

    The fdt record names its operating points where it has them, and says there is no resume: unlike
    a cache, whose batches a later run continues from, this one starts again from the beginning
    (spec §1.3).
    """
    if s.kind == "simulation":
        return (f"This training cache is UNFINISHED: {s.batches_done} committed batch(es). "
                "Deleting it throws those batches away and a later run starts from zero. "
                "(Only the batch count is known while a cache is running: the rows are "
                "recorded when the cache finishes.)")
    if s.kind == "fdt":
        if s.points_done is None:
            measured = "it holds whatever it had written when it stopped"
        else:
            planned = "?" if s.points_planned is None else s.points_planned
            measured = f"it holds {s.points_done} of {planned} operating point(s)"
        return (f"This {s.study or 'measurement'} record is UNFINISHED: the run was interrupted or "
                f"it failed, and {measured}, its figures and its log. There is no resume for these "
                "runs -- deleting it means measuring again from the start.")
    return ("This record is UNFINISHED: its folder holds only what the run had written when it "
            "stopped.")


def _delete_prompt(s) -> tuple:
    """``(text, informative)`` for the confirmation: what goes, and what cannot come back.

    An UNFINISHED artifact names what is half-written, whichever kind it is -- ``finished`` is not
    ``complete`` (B3), and what a half-written record holds is the only thing deleting it destroys
    that a later run could not simply remake. ``_unfinished_note`` is where each kind's sentence is.

    ``s.complete and not s.finished`` rather than ``not s.finished`` alone: an INCOMPLETE row (a
    directory with no usable manifest) is not an artifact at all, ``finished`` is False for it, and
    it is removed by the sweep rather than by this button.
    """
    lines = [f"id {s.id}"]
    if s.complete and not s.finished:
        lines.append(_unfinished_note(s))
    lines.append(f"This removes {s.path} and everything in it, and cannot be undone.")
    return f"Delete {s.kind} {s.label}?", "\n".join(lines)
```

- [ ] **Step 4: Run the new test and the cache's**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_artifact_browser.py -k "unfinished" -v`

Expected: PASS (2 passed). The cache's existing assertions —
`"2 committed batch(es)" in box.informativeText()` and `"0 rows" not in ...` — hold unchanged: the
sentence moved, its words did not.

- [ ] **Step 5: Run the browser suite**

Run: `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_artifact_browser.py -m "not slow" -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add core/gui/screens/artifact_screen.py tests/test_artifact_browser.py
git commit -m "browser: the delete prompt names what is half-written, for both unfinished kinds"
```
## THE SETTINGS OBJECT AND THE CHECKS (T7–T12)

#### Amendments (binding — these supersede the text above)

None. The pre-flight scan against HEAD `673868d` found every anchor verbatim, and no ruling in P1–P82 changes this task. Tasks 2 and 3 must have landed first, as its Interfaces already say.

---

### Task 7: `FDTConfig` gains `sources`, `seed` and `copy_for_run`; `cli.cell_sources`; the manifest's FDT branch

**Why:** `public_entry` is duck-typed on `copy_for_run()` (`core/runs.py:243-247`), so without it the
decorator gives `run_fdt` the run log and **no** private copy and `cfg.omega_0 = …` keeps writing on
the caller's object — the defect spec §1 names (spec §1.2, "FDTConfig gains two fields and one
method"). `sources` is what makes `provenance.inputs_from_cfg` record the cell, the resolved bounds
file and the units file with no new provenance code (spec §3.2), and `manifest.config_from_cfg`
needs its FDT branch or `store.create("fdt", cfg, …)` raises `AttributeError` on
`cfg.observation_mode` (spec §1.2). This task owns **Review Focus item 4** — a cell whose bounds file
resolves to the folder's `master.txt` rather than to a same-named sibling — and pins that
`make_reduction_config` still builds a working settings object (spec §1.3).

**Files:**
- Modify: `core/sim_config.py:562-636` (the `FDTConfig` dataclass; anchors below)
- Modify: `core/cli.py:128-148` (beside `resolve_bounds_for_cell`) and `core/cli.py:319-405` (the
  three builders)
- Modify: `core/artifacts/manifest.py:211-238` (`config_from_cfg`)
- Test: `tests/test_artifact_store.py` (beside
  `test_sim_config_records_its_sources_and_the_feature_set_is_versioned` and
  `test_copy_for_run_drops_the_caches_first_and_keeps_chi_obs_freqs`)
- Test: `tests/test_artifact_consistency.py` (beside
  `test_bounds_resolution_prefers_a_sibling_then_falls_back_to_master`)

**Interfaces:**
- Consumes: `store.KINDS`/`BODY_KEYS` already carry `"fdt"` (Task 1);
  `cli.resolve_bounds_for_cell(cell_file, model) -> Path | None`;
  `provenance.inputs_from_cfg(cfg, *, missing_ok=False) -> dict`.
- Produces:
  - `FDTConfig.sources: dict` (`{"cell": str, "bounds": str|None, "units": str|None, "model": str}`),
    `FDTConfig.seed: int | None = None`, `FDTConfig.copy_for_run() -> FDTConfig`.
  - `cli.cell_sources(cell_file: str, model: str | None = None) -> dict` — the same four keys.
  - `cli.make_fdt_config(..., seed=None)`, `cli.make_param_sweep_config(..., seed=None)`: both fill
    `sources` and carry `seed` onto the config. `cli.make_reduction_config` fills `sources` only.
  - `manifest.config_from_cfg(cfg)` returns the FDT settings block for a `cfg` with no
    `observation_mode`.

- [ ] **Step 1: Write the failing test for the two fields and the copy**

In `tests/test_artifact_store.py`, after
`test_copy_for_run_drops_the_caches_first_and_keeps_chi_obs_freqs`:

```python
def test_fdt_config_carries_sources_and_a_seed_and_copies_itself_for_a_run():
    """``public_entry`` copies the config it is handed ONLY when that config has ``copy_for_run``
    (core/runs.py:243-247, ``if hasattr(kwargs["cfg"], "copy_for_run")``). FDTConfig had none, so
    decorating run_fdt would have delivered the run log and silently NOT V1's private copy -- and
    ``cfg.omega_0 = ...`` at the top of run_fdt would have kept writing on the caller's object, which
    is the defect spec §1 names. The copy is a plain deep copy: FDTConfig carries no cached_property,
    so it needs none of SimConfig.copy_for_run's _CACHED popping, and this asserts that (a new cached
    property on FDTConfig must come with the popping, as SimConfig's did).

    ``sources`` and ``seed`` are DEFAULTED fields, because the reduction map shares this dataclass and
    is outside the programme (spec §1.3): a required field would break it at every construction site.
    Everything a run writes on -- omega_0, the four OrderedDicts, sources -- is an independent equal
    object on the copy."""
    from collections import OrderedDict
    from functools import cached_property

    from core.config import FDTConfig, cpu_device

    assert not [n for n, v in vars(FDTConfig).items() if isinstance(v, cached_property)], \
        "FDTConfig gained a cached_property: copy_for_run must pop it first, as SimConfig's does"

    cfg = FDTConfig(model="HOPF", state_dep_drift=False, inits_dict=OrderedDict(x=0.0),
                    params_dict=OrderedDict(sigma_x=(0.1, (0.0, 1.0))),
                    rescale_params=OrderedDict(), force_params_dict=OrderedDict(),
                    units_dict=("nm", "ms"), hw=cpu_device())
    assert cfg.sources == {} and cfg.seed is None, "both default, for the reduction map's sake"

    cfg.sources["cell"] = "Cells/hopf/cell.txt"
    cfg.seed = 7
    c = cfg.copy_for_run()
    assert isinstance(c, FDTConfig) and c is not cfg
    assert c.sources == cfg.sources and c.sources is not cfg.sources
    assert c.params_dict == cfg.params_dict and c.params_dict is not cfg.params_dict
    assert c.seed == 7 and c.hw is not None

    c.omega_0 = 3.5
    c.sources["cell"] = "elsewhere.txt"
    assert cfg.omega_0 is None, "the caller's resonance is untouched: this is what V1 buys run_fdt"
    assert cfg.sources["cell"] == "Cells/hopf/cell.txt"
```

- [ ] **Step 2: Run it and watch it fail**

Run: `pytest tests/test_artifact_store.py::test_fdt_config_carries_sources_and_a_seed_and_copies_itself_for_a_run -v`
Expected: FAIL with `TypeError: FDTConfig.__init__() got an unexpected keyword argument` — no:
the call passes no new keyword, so the first failure is
`AssertionError: assert {} == {}` … in fact it is `AttributeError: 'FDTConfig' object has no attribute 'sources'` at
`assert cfg.sources == {} and cfg.seed is None`.

- [ ] **Step 3: Add the two fields and the method**

In `core/sim_config.py`, find:

```python
    # Filled in by run_fdt after cfg is built (from params_dict["k"])
    omega_0: float = None

    # Hardware
    hw: DeviceConfig = field(default_factory=detect_device)
```

Replace with:

```python
    # Filled in by run_fdt after cfg is built (from params_dict["k"])
    omega_0: float = None

    # Hardware
    hw: DeviceConfig = field(default_factory=detect_device)

    # The files this config was built from, as paths, plus the model NAME:
    # {"cell": ..., "bounds": ..., "units": ..., "model": ...}. Filled by cli.cell_sources through
    # the three builders; read by core.artifacts.provenance.inputs_from_cfg, so an fdt record names
    # its cell, its RESOLVED bounds file and its units file by path AND content hash (piece 5, §3.2).
    # Defaulted, like `seed`: the reduction map shares this dataclass and is out of scope (§1.3).
    sources: dict = field(default_factory=dict)

    # The seed the run used, drawn when none was supplied (E7). None means "not chosen yet".
    seed: "int | None" = None
```

Then, at the end of the class, after `with_overrides`'s `return replace(...)` line, add:

```python
    def copy_for_run(self) -> "FDTConfig":
        """A private deep copy for one run: what `core.runs.public_entry` hands a public entry in
        place of the caller's config (V1 of piece 3).

        NOT optional, and not cosmetic: `public_entry` is duck-typed on this method
        (core/runs.py:243-247, `if hasattr(kwargs["cfg"], "copy_for_run")`), so a config class
        without it gets the run log and silently NO copy -- and run_fdt writes `cfg.omega_0` twice.
        A plain deep copy suffices, unlike SimConfig's: this class carries no cached_property, so
        there is no 2.4M-point grid and no pint registry to pop off a shallow copy first (a test
        pins that it still carries none). Everything a run writes on -- omega_0, seed, sources and
        the four OrderedDicts through with_overrides -- is an independent equal object on the copy.
        """
        return copy.deepcopy(self)
```

- [ ] **Step 4: Run it and watch it pass**

Run: `pytest tests/test_artifact_store.py::test_fdt_config_carries_sources_and_a_seed_and_copies_itself_for_a_run -v`
Expected: PASS

- [ ] **Step 5: Write the failing test for `cell_sources` and the master/sibling split**

In `tests/test_artifact_consistency.py`, directly after
`test_bounds_resolution_prefers_a_sibling_then_falls_back_to_master`:

```python
def test_cell_sources_records_the_bounds_file_that_actually_resolved():
    """Review Focus 4. Three Nadrowski cells share one box and one does not: master_spont.txt has a
    same-named sibling in Bounds/, master_weak.txt falls back to the folder's master.txt
    (cli.resolve_bounds_for_cell). Only one of the two is the box a result was measured under, and a
    record that names the WRONG bounds file is worse than one that names none -- on the decoupled
    path "the bounds file defines the param set + order" (cli.parse_cell's docstring), so the
    parameter set itself is not recoverable without it.

    The legacy inline-bounds branch has no bounds file and no units file, and cell_sources reports
    None for both rather than a path that is not read: parse_cell takes that branch when EITHER is
    absent, so units must follow bounds and not stand alone."""
    nad = str(CELL_PATH / _NAD / "master_spont.txt")
    weak = str(CELL_PATH / _NAD / "master_weak.txt")

    src = cli.cell_sources(nad, "NADROWSKI")
    assert set(src) == {"cell", "bounds", "units", "model"}
    assert src["cell"] == nad and src["model"] == "NADROWSKI"
    assert Path(src["bounds"]).name == "master_spont.txt", "the sibling wins where it exists"
    assert Path(src["units"]).name == "units.txt"

    assert Path(cli.cell_sources(weak, "NADROWSKI")["bounds"]).name == cli.MASTER_BOUNDS_NAME

    # the model defaults to the cell's parent folder, exactly as parse_cell derives it
    assert cli.cell_sources(weak)["model"] == "NADROWSKI"

    # a cell in a folder with neither a sibling nor a master: the legacy branch, and no files
    ghost = str(CELL_PATH / "hopf" / "no_such_cell.txt")
    assert cli.resolve_bounds_for_cell(ghost) is None
    assert cli.cell_sources(ghost, "HOPF") == {"cell": ghost, "bounds": None, "units": None,
                                               "model": "HOPF"}
```

- [ ] **Step 6: Run it and watch it fail**

Run: `pytest tests/test_artifact_consistency.py::test_cell_sources_records_the_bounds_file_that_actually_resolved -v`
Expected: FAIL with `AttributeError: module 'core.cli' has no attribute 'cell_sources'`

- [ ] **Step 7: Add `cell_sources` beside `resolve_bounds_for_cell`**

In `core/cli.py`, find:

```python
def parse_cell(cell_file: str, model: str | None = None):
```

and insert immediately **above** it:

```python
def cell_sources(cell_file: str, model: str | None = None) -> dict:
    """The ``sources`` block for a cell: its own path, the bounds file that RESOLVES for it, the
    units file, and the model NAME. Beside :func:`parse_cell`, which keeps its 7-tuple.

    parse_cell resolves both paths and throws them away (``bounds_path`` and ``units_path`` are
    local to it), so an FDT record had no way to say which box its parameter set came from. The two
    candidates resolve DIFFERENTLY -- a same-named sibling, else the folder's master.txt -- and only
    one of them governed the run.

    Both come back None on the LEGACY inline-bounds branch, and the condition is parse_cell's own
    (``bounds_path is not None and units_path.exists()``): the units file is only read when the
    decoupled path is taken, so reporting it alone would name a file the run never opened.
    """
    p = Path(cell_file)
    name = (model or p.parent.name)
    bounds_path = resolve_bounds_for_cell(cell_file, name.lower())
    units_path = UNITS_PATH / name.lower() / "units.txt"
    decoupled = bounds_path is not None and units_path.exists()
    return {"cell": str(cell_file),
            "bounds": str(bounds_path) if decoupled else None,
            "units": str(units_path) if decoupled else None,
            "model": name.upper()}
```

- [ ] **Step 8: Run it and watch it pass**

Run: `pytest tests/test_artifact_consistency.py::test_cell_sources_records_the_bounds_file_that_actually_resolved -v`
Expected: PASS

- [ ] **Step 9: Write the failing test for the three builders and the manifest branch**

In `tests/test_artifact_store.py`, after the test added in step 1:

```python
def test_the_three_fdt_builders_fill_sources_and_the_manifest_has_an_fdt_branch():
    """Every artifact names its inputs by path and hash, and an fdt record is no exception (spec
    §3.2): the three FDTConfig builders fill ``sources`` so provenance.inputs_from_cfg works with no
    new provenance code. make_reduction_config fills it too and is otherwise UNTOUCHED -- it is the
    reduction map's builder, out of scope, and the only FDTConfig in the tree not pinned to the CPU
    (§1.3), so this also pins that it still builds a working settings object.

    config_from_cfg needed a branch or store.create("fdt", cfg, ...) would raise AttributeError on
    cfg.observation_mode, the SECOND key it reads (manifest.py:211-238). The branch is taken on the
    ABSENCE of observation_mode, and it records the settings AS GIVEN TO THE BUILDER: omega_0 is
    deliberately not in it, because the writer computes this block at create() time from the
    caller's object while the run refines the resonance on its private copy (§1.2). Every value is
    finite, which validate() requires of the whole config block (manifest.py:158)."""
    from core import cli, config, registry
    from core.artifacts import manifest as mfm
    from core.artifacts import provenance as provm

    cell = str(config.CELL_PATH / "nadrowski" / "master_weak.txt")
    cfg = cli.make_fdt_config("NADROWSKI", registry.state_dep_drift("NADROWSKI"), cell)
    assert cfg.sources == cli.cell_sources(cell, "NADROWSKI")
    assert cfg.hw.device.type == "cpu", "the FDT path stays pinned to the CPU"

    inputs = provm.inputs_from_cfg(cfg)
    assert inputs["model"] == "NADROWSKI"
    assert inputs["cell"]["path"] == "Cells/nadrowski/master_weak.txt"      # relative to Resources/
    assert inputs["bounds"]["path"] == "Bounds/nadrowski/master.txt"
    assert len(inputs["cell"]["sha256"]) == 64 and len(inputs["bounds"]["sha256"]) == 64

    block = mfm.config_from_cfg(cfg)
    assert block["model"] == "NADROWSKI" and block["n_freqs"] == 60 and block["ensemble_M"] == 256
    assert block["freq_bounds"] == [0.1, 30.0] and block["device"] == "cpu"
    assert block["seed"] is None and "omega_0" not in block, \
        "the resonance the RUN discovers belongs in body.grid, not in the config block"
    mfm._check_finite(block, "config")          # what validate() does to it on every write

    red = cli.make_reduction_config(str(config.CELL_PATH / "nadrowski" / "master_spont.txt"))
    assert red.model == "NADROWSKI" and red.params_dict, "the reduction builder still builds"
    assert Path(red.sources["cell"]).name == "master_spont.txt"
```

- [ ] **Step 10: Run it and watch it fail**

Run: `pytest tests/test_artifact_store.py::test_the_three_fdt_builders_fill_sources_and_the_manifest_has_an_fdt_branch -v`
Expected: FAIL with `AssertionError: assert {} == {'cell': ..., 'bounds': ..., 'units': ..., 'model': 'NADROWSKI'}`
at `assert cfg.sources == cli.cell_sources(cell, "NADROWSKI")`

- [ ] **Step 11: Fill `sources` (and carry `seed`) in the three builders**

In `core/cli.py`, find:

```python
def make_fdt_config(model: str, state_dep_drift: bool, cell_file: str, *,
                    n_freqs: int = 60, ensemble_M: int = 256, freqs_per_batch: int = 1,
                    F0: float = 0.05) -> FDTConfig:
```

Replace the signature with:

```python
def make_fdt_config(model: str, state_dep_drift: bool, cell_file: str, *,
                    n_freqs: int = 60, ensemble_M: int = 256, freqs_per_batch: int = 1,
                    F0: float = 0.05, seed: "int | None" = None) -> FDTConfig:
```

and in the same function find:

```python
        F0=F0,
        hw=cpu_device(),  # FDT: sequential SDE loop at M~256 is ~3.4x faster on CPU than GPU
    )
```

Replace with:

```python
        F0=F0,
        hw=cpu_device(),  # FDT: sequential SDE loop at M~256 is ~3.4x faster on CPU than GPU
        sources=cell_sources(cell_file, model),
        seed=seed,
    )
```

In `make_reduction_config`, find:

```python
        F0=F0,
        hw=detect_device(),
    )
```

Replace with:

```python
        F0=F0,
        hw=detect_device(),
        sources=cell_sources(cell_file, "NADROWSKI"),
    )
```

In `make_param_sweep_config`, find:

```python
        psd_T_obs_nd=preset["psd_T_obs_nd"],
        hw=cpu_device(),  # sweep: sequential SDE loop at M~256 is ~3.4x faster on CPU than GPU
    )
```

Replace with:

```python
        psd_T_obs_nd=preset["psd_T_obs_nd"],
        hw=cpu_device(),  # sweep: sequential SDE loop at M~256 is ~3.4x faster on CPU than GPU
        sources=cell_sources(cell_file, "NADROWSKI"),
        seed=seed,
    )
```

and widen its signature's tail from `F0: float = 0.05) -> tuple[...]` to
`F0: float = 0.05, seed: "int | None" = None) -> tuple[...]` (Task 11 rewrites this signature
further; the `seed` keyword lands here because the field lands here).

- [ ] **Step 12: Add the FDT branch to `config_from_cfg`**

In `core/artifacts/manifest.py`, find:

```python
def config_from_cfg(cfg) -> dict:
    """The SimConfig-derived identity vocabulary every manifest's ``config`` starts from. Stages
    ``.update()`` their own resolved knobs on top."""
    from core.SBI.reparam import resolved_log_params
    from core.SBI.run_guards import _log_params_for
    return {
```

Replace with:

```python
def config_from_cfg(cfg) -> dict:
    """The SimConfig-derived identity vocabulary every manifest's ``config`` starts from. Stages
    ``.update()`` their own resolved knobs on top.

    An FDTConfig takes the second branch, chosen on the ABSENCE of ``observation_mode``: the fdt
    kind measures a CELL rather than conditioning a network, so it has no mode, no chi geometry and
    no training grid, and reading ``cfg.observation_mode`` on one is an AttributeError two keys into
    the dict below (piece 5, §1.2). That branch records the settings AS GIVEN TO THE BUILDER --
    ``omega_0`` is deliberately absent, because the writer computes this block at ``create()`` time
    from the caller's object while the run refines the resonance on its own private copy, and the
    refined value belongs in ``body.grid``.
    """
    if not hasattr(cfg, "observation_mode"):
        return _fdt_config_from_cfg(cfg)
    from core.SBI.reparam import resolved_log_params
    from core.SBI.run_guards import _log_params_for
    return {
```

and add, immediately **above** `def config_from_cfg`:

```python
def _fdt_config_from_cfg(cfg) -> dict:
    """``config_from_cfg``'s branch for an FDTConfig: the cell's parameter values and every
    resolution knob the run was built with. Floats only and all finite -- ``validate`` refuses a
    non-finite number anywhere in the config block (``_check_finite(d["config"], "config")``)."""
    return {
        "model": cfg.model,
        "state_dep_drift": bool(cfg.state_dep_drift),
        "param_keys": list(cfg.params_dict) + list(cfg.rescale_params),
        "param_values": ([float(v[0]) for v in cfg.params_dict.values()]
                         + [float(v[0]) for v in cfg.rescale_params.values()]),
        "n_freqs": int(cfg.n_freqs),
        "freq_bounds": [float(v) for v in cfg.freq_bounds],
        "ensemble_M": int(cfg.ensemble_M),
        "freqs_per_batch": int(cfg.freqs_per_batch),
        "F0": float(cfg.F0),
        "burn_in_nd": float(cfg.burn_in_nd),
        "T_obs_periods": int(cfg.T_obs_periods),
        "dt_nd": float(cfg.dt_nd),
        "psd_T_obs_nd": float(cfg.psd_T_obs_nd),
        "seed": None if cfg.seed is None else int(cfg.seed),
        "units": list(cfg.units_dict) if isinstance(cfg.units_dict, (list, tuple)) else None,
        "device": cfg.hw.device.type, "dtype": str(cfg.hw.dtype),
    }
```

- [ ] **Step 13: Run the three tests and watch them pass**

Run: `pytest tests/test_artifact_store.py -k "fdt_config or fdt_builders or copy_for_run" tests/test_artifact_consistency.py::test_cell_sources_records_the_bounds_file_that_actually_resolved -v`
Expected: PASS

- [ ] **Step 14: Commit**

```bash
git add core/sim_config.py core/cli.py core/artifacts/manifest.py tests/test_artifact_store.py tests/test_artifact_consistency.py
git commit -m "fdt: the settings object gets sources, a seed and copy_for_run"
```

#### Amendments (binding — these supersede the text above)

**A1 — P19: `cli.make_reduction_config` is NOT touched.** Only the two FDT builders fill `sources`.
- **Step 11:** delete the whole middle edit, the one that begins "In `make_reduction_config`, find:" and adds `sources=cell_sources(cell_file, "NADROWSKI"),` after `hw=detect_device(),`. Leave `make_reduction_config` byte-identical.
- **Step 3:** in the new `sources` comment, replace
  `# {"cell": ..., "bounds": ..., "units": ..., "model": ...}. Filled by cli.cell_sources through`
  `# the three builders; read by ...`
  with
  `# {"cell": ..., "bounds": ..., "units": ..., "model": ...}. Filled by cli.cell_sources in the two`
  `# FDT builders (make_reduction_config leaves it empty, P19); read by ...`
  and keep the rest of the comment.
- **Step 9:** rename the test to `test_the_two_fdt_builders_fill_sources_and_the_manifest_has_an_fdt_branch`. In its docstring, replace "the three FDTConfig builders fill ``sources`` so provenance.inputs_from_cfg works with no new provenance code. make_reduction_config fills it too and is otherwise UNTOUCHED -- it is" with "the two FDT builders fill ``sources`` so provenance.inputs_from_cfg works with no new provenance code. make_reduction_config is UNTOUCHED (P19; spec §1.2, §1.3) -- it is". Replace its last line
  ```python
      assert Path(red.sources["cell"]).name == "master_spont.txt"
  ```
  with
  ```python
      assert red.sources == {} and red.seed is None, "make_reduction_config is left alone (P19)"
  ```
- **Files / Interfaces:** read "(the three builders)" as "(the two FDT builders)", and delete "`cli.make_reduction_config` fills `sources` only."

**A2 — Step 2's expected failure** is `AttributeError: 'FDTConfig' object has no attribute 'sources'`, raised at `assert cfg.sources == {} and cfg.seed is None`. Ignore the struck-through alternatives in that paragraph.

**A3 — Step 13's command.** `-k` also filters the explicit node id and would deselect the consistency test. Run instead:
`pytest "tests/test_artifact_store.py::test_fdt_config_carries_sources_and_a_seed_and_copies_itself_for_a_run" "tests/test_artifact_store.py::test_the_two_fdt_builders_fill_sources_and_the_manifest_has_an_fdt_branch" tests/test_artifact_store.py::test_copy_for_run_drops_the_caches_first_and_keeps_chi_obs_freqs tests/test_artifact_consistency.py::test_cell_sources_records_the_bounds_file_that_actually_resolved -v`. Expected: 4 passed.

**A4 — P72:** `FDTConfig.preset_name` is added by Task 11, not here. Do not add it.

**A5 — contract note:** keep `cell_sources(cell_file: str, model: str | None = None)` as written. The contract's `model: str` is satisfied by every caller. Say so in the task report; a finding asks the owner to confirm.

#### Controller rulings (pre-flight, binding — these supersede the amendments above and the task text)

- **P72** — `FDTConfig` gains THREE defaulted fields, not two: `sources`, `seed`, and `preset_name: "str | None" = None  # the preset this sweep's knobs came from; None for a single-cell run`. Task 11 fills it; Task 19 reads it.
- **F31** — `cell_sources(cell_file, model=None)` keeps a default for `model`, mirroring `parse_cell`'s own signature, although the contract lists it as required. Say so in your report.

---

### Task 8: `require_below`, the eight new field keys, and the tool's flag table

**Why:** Spec §3.3 ("One new rule is added to `core/refusals.py`: `require_below(key, lo, hi, …)`
for an ordered pair … It is the only rule the existing eight do not cover") and §5.3 (`FIELDS` gains
one key per new front-end setting). `tests/test_refusals.py` hard-codes `len(FIELDS)` and asserts the
None-flag set by **equality**; both move here, and the equality becomes a subset assertion with its
reason (spec §8.3).

**Files:**
- Modify: `core/refusals.py:79-126` (the `FIELDS` tuple) and `core/refusals.py:215-226` (after
  `require_between`)
- Modify: `core/tool/fields.py:24-100` (`FLAG`)
- Modify: `core/gui/fields.py:44-110` (`CONTROL` — sentence entries here; Task 9 converts them)
- Test: `tests/test_refusals.py` (`BASE_KEYS`, the registry count, the rule tests, the FLAG pins)
- Test: `tests/test_nav_and_gating.py:1447` (the key set is asserted both ways, so the new keys need
  a `CONTROL` entry in the same commit)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `refusals.require_below(key: str, lo, hi) -> tuple[float, float]`; the keys
  `n_freqs`, `ensemble_m`, `freqs_per_batch`, `f0`, `preset`, `s_grid`, `t_grid`, `seed` in
  `refusals.FIELDS`, in `tool.fields.FLAG` and in `gui.fields.CONTROL`. Tasks 10 and 11 call the
  rules with these keys; Task 9 converts the `CONTROL` entries to place tuples.

- [ ] **Step 1: Write the failing test for `require_below`**

In `tests/test_refusals.py`, after
`test_require_between_honours_open_and_closed_ends_and_treats_nan_as_outside`:

```python
def test_require_below_refuses_an_inverted_or_blank_pair_and_returns_two_floats():
    """The sweep grids and the FDT frequency band are ORDERED PAIRS, and none of the existing eight
    rules covers that shape: require_between judges one value against fixed bounds, and a pair whose
    own two ends are the thing being judged has no bound to compare to. An inverted pair is not
    hypothetical -- every window numeric box returns 0 for a blank (FloatField.value), so a grid
    whose 'max' was left empty arrives as (0.0, 0.0) and np.linspace would happily produce a sweep
    of one repeated value rather than refusing.

    Both ends are refused blank and non-finite first, so the message never reads "0 and nan"; the
    pair comes back as floats, which is what the caller binds."""
    assert require_below("s_grid", 0.0, 1.0) == (0.0, 1.0)
    assert all(isinstance(v, float) for v in require_below("s_grid", 0, 1))
    for lo, hi, shown in ((1.0, 0.0, "1 and 0"), (0.0, 0.0, "0 and 0"), (2.5, 2.5, "2.5 and 2.5")):
        with pytest.raises(Refusal) as e:
            require_below("s_grid", lo, hi)
        assert _shape(e.value, "s_grid") == (
            f"The activity sweep grid must have its lower bound below its upper bound "
            f"(got {shown}).")
    with pytest.raises(Refusal) as e:
        require_below("t_grid", None, 2.0)
    assert _shape(e.value, "t_grid") == "The temperature sweep grid is blank."
    with pytest.raises(Refusal) as e:
        require_below("t_grid", 1.0, float("nan"))
    assert _shape(e.value, "t_grid") == "The temperature sweep grid must be a finite number; got nan."
```

Add `require_below` to the module's `from core.refusals import (...)` list at the top of the file.

- [ ] **Step 2: Run it and watch it fail**

Run: `pytest tests/test_refusals.py::test_require_below_refuses_an_inverted_or_blank_pair_and_returns_two_floats -v`
Expected: collection error — `ImportError: cannot import name 'require_below' from 'core.refusals'`

- [ ] **Step 3: Add `require_below`**

In `core/refusals.py`, find:

```python
def require_choice(key: str, value, choices: tuple) -> str:
```

and insert immediately **above** it:

```python
def require_below(key: str, lo, hi) -> tuple:
    """A blank, a non-finite end, or ``lo >= hi`` is refused; the pair is returned as two floats.

    The ordered-pair rule the other eight do not cover: ``require_between`` judges ONE value against
    fixed bounds, and here the two ends are themselves what is being judged. Each end goes through
    ``require_finite`` first, so a blank box (which every window numeric field reads as 0) and a NaN
    are refused by their own sentences rather than appearing inside this one.
    """
    a, b = require_finite(key, lo), require_finite(key, hi)
    if not a < b:
        raise Refusal(f"{_what(key)} must have its lower bound below its upper bound "
                      f"(got {a:g} and {b:g}){_default_clause(key)}.", field=key)
    return a, b
```

- [ ] **Step 4: Run it and watch it fail on the unregistered key**

Run: `pytest tests/test_refusals.py::test_require_below_refuses_an_inverted_or_blank_pair_and_returns_two_floats -v`
Expected: FAIL with `KeyError: 's_grid'` raised from `describe` inside `_what`

- [ ] **Step 5: Register the eight new keys**

In `core/refusals.py`, find:

```python
    # the artifact browser (piece 4): the artifact a browse action acts on, and its note
    Field("artifact", "the artifact", None),
    Field("note", "the note", None),
```

Replace with:

```python
    # the artifact browser (piece 4): the artifact a browse action acts on, and its note
    Field("artifact", "the artifact", None),
    Field("note", "the note", None),
    # the two secondary analyses (piece 5, §5.3): the knobs both front ends expose. The five the
    # front ends do NOT expose -- the frequency band, the burn-in, the two durations and the step --
    # are deliberately absent: their refusals carry field=None, because there is no control and no
    # flag a fix sentence could name (§1.2).
    Field("n_freqs", "the number of drive frequencies", "60"),                         # FDTConfig.n_freqs
    Field("ensemble_m", "the number of trajectories per frequency", "256"),            # FDTConfig.ensemble_M
    Field("freqs_per_batch", "the number of frequencies per simulator call", "1"),     # FDTConfig.freqs_per_batch
    Field("f0", "the non-dimensional drive amplitude", "0.05"),                        # FDTConfig.F0
    Field("preset", "the resolution preset", "exploratory"),                           # cli.SWEEP_PRESETS
    Field("s_grid", "the activity sweep grid", None),
    Field("t_grid", "the temperature sweep grid", None),
    Field("seed", "the random seed", "none: one is drawn and recorded"),
```

- [ ] **Step 6: Run it and watch it pass, then watch two other tests go red**

Run: `pytest tests/test_refusals.py -v`
Expected: the new test PASSES;
`test_the_registry_holds_exactly_the_initial_keys_with_neutral_descriptions` FAILS with
`AssertionError: assert {…} == {…}` on `set(FIELDS) == set(BASE_KEYS) | set(TOOL_ONLY_KEYS)`, and
`test_the_tool_flag_table_...` FAILS on `set(tool_fields.FLAG) == set(FIELDS)`.

- [ ] **Step 7: Widen the registry pin**

In `tests/test_refusals.py`, find:

```python
    "chi_max_cycles", "chi_f0", "chi_freq_bounds", "device", "model", "observation", "posterior", "prior",
    "artifact", "note",
)
```

Replace with:

```python
    "chi_max_cycles", "chi_f0", "chi_freq_bounds", "device", "model", "observation", "posterior", "prior",
    "artifact", "note",
    # piece 5's secondary analyses: the knobs the FDT and sweep screens and their two subcommands
    # expose. The five neither front end exposes are NOT here -- see core/refusals.py's comment.
    "n_freqs", "ensemble_m", "freqs_per_batch", "f0", "preset", "s_grid", "t_grid", "seed",
)
```

and find:

```python
    assert len(FIELDS) == len(BASE_KEYS) + len(TOOL_ONLY_KEYS) == 63, "a key is listed twice above"
```

Replace with:

```python
    assert len(FIELDS) == len(BASE_KEYS) + len(TOOL_ONLY_KEYS) == 71, "a key is listed twice above"
```

- [ ] **Step 8: Add the eight flags and soften the None-flag equality**

In `core/tool/fields.py`, find:

```python
    "artifact": None,
    "note": "--note",                                   # add_name_flags defines it beside --name
```

Replace with:

```python
    "artifact": None,
    "note": "--note",                                   # add_name_flags defines it beside --name
    # the two secondary analyses (piece 5): `fdt` and `crossval` share the four resolution knobs,
    # and the grids and the preset are the sweep's alone. --seed exists on smoke and the
    # diagnostics today; piece 5 adds it to these two as well (E7).
    "n_freqs": "--n-freqs",
    "ensemble_m": "--ensemble-m",
    "freqs_per_batch": "--freqs-per-batch",
    "f0": "--f0",
    "preset": "--preset",
    "s_grid": "--s-grid",
    "t_grid": "--t-grid",
    "seed": "--seed",
```

In `tests/test_refusals.py`, find:

```python
    assert {k for k, f in tool_fields.FLAG.items() if f is None} == {
        "units", "chi_k_pad", "chi_max_cycles", "chi_f0", "chi_freq_bounds", "artifact"}
```

Replace with:

```python
    # A SUBSET assertion, not an equality (spec §8.3): these six answer to no option string today
    # and must keep doing so -- the units are declared per model, the four chi constants are
    # config.py's, and the artifact is positional. A later key with no flag is a new fact about that
    # key, not a regression in these six, and an equality here turns every such addition into a
    # false red in a file that has nothing to do with it.
    assert {"units", "chi_k_pad", "chi_max_cycles", "chi_f0", "chi_freq_bounds", "artifact"} <= \
        {k for k, f in tool_fields.FLAG.items() if f is None}
    assert tool_fields.FLAG["s_grid"] == "--s-grid" and tool_fields.FLAG["seed"] == "--seed"
```

- [ ] **Step 9: Give the eight a window entry so the both-ways key pin stays green**

In `core/gui/fields.py`, find:

```python
    # tool-only: the diagnostics' knobs; no window sentence
```

and insert immediately **above** it:

```python
    # the two secondary analyses (piece 5, §5.3). SENTENCES for now: the tuple shape's first element
    # must be an inference tab title, which none of these places is, and widening it to understand a
    # screen is the next task's work. Each sentence is written exactly as the widened tuple will
    # render it, so that conversion is a pure refactor.
    "n_freqs": "Set it in the 'n_freqs' box on the FDT analysis or Sweep study cross-validation tab.",
    "ensemble_m": ("Set it in the 'ensemble_M' box on the FDT analysis or Sweep study "
                   "cross-validation tab."),
    "freqs_per_batch": ("Set it in the 'freqs_per_batch' box on the FDT analysis or Sweep study "
                        "cross-validation tab."),
    "f0": ("Set it in the 'F0 (ND forcing amplitude)' box on the FDT analysis or Sweep study "
           "cross-validation or NWK → Hopf reduction map tab."),
    "preset": "Set it in the 'Preset' box on the Sweep study cross-validation tab.",
    "s_grid": "Set it in the 'S grid  (T_a/T = 1)' box on the Sweep study cross-validation tab.",
    "t_grid": "Set it in the 'T_a/T grid  (S = 0)' box on the Sweep study cross-validation tab.",
    "seed": "Set it in the 'Seed' box on the FDT analysis or Sweep study cross-validation tab.",
```

- [ ] **Step 10: Run the two table suites and watch them pass**

Run: `pytest tests/test_refusals.py tests/test_nav_and_gating.py::test_every_field_key_has_a_window_control_and_the_fix_sentences_name_it -v`
Expected: PASS

- [ ] **Step 11: Commit**

```bash
git add core/refusals.py core/tool/fields.py core/gui/fields.py tests/test_refusals.py
git commit -m "refusals: require_below and the eight secondary-analysis field keys"
```

#### Amendments (binding — these supersede the text above)

**A1 — P52: `require_below`'s message uses "; got …", not parentheses** (this also supersedes the contract's `(got <lo> and <hi>)`).
- **Step 3:** the raise becomes
  ```python
      a, b = require_finite(key, lo), require_finite(key, hi)
      if not a < b:
          raise Refusal(f"{_what(key)} must have its lower bound below its upper bound; got {a:g} and "
                        f"{b:g}{_default_clause(key)}.", field=key)
      return a, b
  ```
  In its docstring, replace "so a blank box (which every window numeric field reads as 0) and a NaN are refused by their own sentences rather than appearing inside this one" with "so None (a blank read through value_or_none) and a NaN are refused by their own sentences; a blank that value() reads as 0 reaches the ordering test as (0, 0) and is refused there".
- **Step 1:** the expected sentence becomes
  ```python
          assert _shape(e.value, "s_grid") == (
              f"The activity sweep grid must have its lower bound below its upper bound; got {shown}.")
  ```

**A2 — P2/P75/P54: Step 5's replacement is this block** (it replaces the whole of Step 5's replacement text, the "deliberately absent" comment included):
```python
    # the artifact browser (piece 4): the artifact a browse action acts on, and its note
    Field("artifact", "the artifact", None),
    Field("note", "the note", None),
    # the two secondary analyses (piece 5, §5.3): the knobs both front ends expose. n_freqs and
    # ensemble_m have two effective defaults, the dataclass's and the sweep preset's (P54).
    Field("n_freqs", "the number of drive frequencies", "60, or the preset's in a sweep"),        # FDTConfig.n_freqs
    Field("ensemble_m", "the number of trajectories per frequency", "256, or the preset's in a sweep"),  # FDTConfig.ensemble_M
    Field("freqs_per_batch", "the number of frequencies per simulator call", "1"),     # FDTConfig.freqs_per_batch
    Field("f0", "the non-dimensional drive amplitude", "0.05"),                        # FDTConfig.F0
    Field("preset", "the resolution preset", "exploratory"),                           # crossval --preset default
    Field("s_grid", "the activity sweep grid", None),
    Field("t_grid", "the temperature sweep grid", None),
    Field("seed", "the random seed", "none: one is drawn and recorded"),
    # the five FDT settings NEITHER front end exposes (§1.2, P2, P75). Registered like any other key,
    # because every require_* rule builds its sentence through describe(key); both front-end tables
    # map them to None, so a refusal names the setting and offers no fix -- there is nothing to name.
    Field("freq_bounds", "the drive frequency band, in multiples of the resonance", "0.1 to 30.0"),  # FDTConfig.freq_bounds
    Field("burn_in_nd", "the burn-in, in ND units", "100.0"),                         # FDTConfig.burn_in_nd
    Field("t_obs_periods", "the drive window, in periods", "30"),                     # FDTConfig.T_obs_periods
    Field("dt_nd", "the integration step, in ND units", "0.01"),                      # FDTConfig.dt_nd
    Field("psd_t_obs_nd", "the spontaneous recording length, in ND units", "8000.0"), # FDTConfig.psd_T_obs_nd
```

**A3 — Step 7's BASE_KEYS replacement** (the new keys go ABOVE `"artifact", "note",`, so Task 22's anchor `"artifact", "note",\n)` survives):
```python
    "chi_max_cycles", "chi_f0", "chi_freq_bounds", "device", "model", "observation", "posterior", "prior",
    # piece 5's secondary analyses: the eight knobs the FDT and sweep screens and their two subcommands
    # expose, then the five FDT settings neither front end exposes -- registered all the same (P2, P75).
    "n_freqs", "ensemble_m", "freqs_per_batch", "f0", "preset", "s_grid", "t_grid", "seed",
    "freq_bounds", "burn_in_nd", "t_obs_periods", "dt_nd", "psd_t_obs_nd",
    "artifact", "note",
)
```
The count literal: do NOT write 71. Run `pytest tests/test_refusals.py::test_the_registry_holds_exactly_the_initial_keys_with_neutral_descriptions -v`, read the real count off the failure (76 expected: 63 + 13), and write that number.

**A4 — Step 8's FLAG replacement gains, after `"seed": "--seed",`:**
```python
    # the five FDT settings neither front end exposes (P2, P75): no option string answers them
    "freq_bounds": None, "burn_in_nd": None, "t_obs_periods": None, "dt_nd": None, "psd_t_obs_nd": None,
```
In core/tool/fields.py's module docstring, extend the sentence beginning "``None`` marks a key no option string answers:" to also name "the five FDT settings neither front end exposes (the frequency band, the burn-in, the two durations and the step)". The subset assertion Step 8 writes already accommodates the five.

**A5 — Step 9: add the five window entries as a SEPARATE group, directly after Step 9's `"seed": ...` sentence line** (Task 9 replaces the sentence block and must not take these with it):
```python
    # the five FDT settings NEITHER front end exposes (§1.2, P2, P75): no control, so fix_sentence
    # returns "" and the refusal names the setting and offers no fix
    "freq_bounds": None, "burn_in_nd": None, "t_obs_periods": None, "dt_nd": None, "psd_t_obs_nd": None,
```
In core/gui/fields.py's module docstring, extend "``None`` for a key the window has no control for: the tool-only diagnostics knobs, and the six settings the window never exposes (...)" with "and the five FDT settings neither front end exposes (P2)".

**A6 — new Step 9b: the window's None-set pin.** In `tests/test_nav_and_gating.py`, `test_every_field_key_has_a_window_control_and_the_fix_sentences_name_it`, find
```python
    assert {k for k, e in gui_fields.CONTROL.items() if e is None} == {
        "checkpoint_every", "resume", "device", "n_samples", "num_posterior_samples", "max_num_epochs",
        "repeats", "n_points", "n_worst", "top_n", "m", "m_noise", "rel", "min_valid", "rows",
        "n_sweep", "chi_k_fixed"}
```
and replace with
```python
    assert {k for k, e in gui_fields.CONTROL.items() if e is None} == {
        "checkpoint_every", "resume", "device", "n_samples", "num_posterior_samples", "max_num_epochs",
        "repeats", "n_points", "n_worst", "top_n", "m", "m_noise", "rel", "min_valid", "rows",
        "n_sweep", "chi_k_fixed",
        # piece 5 (P2, P75): the five FDT settings neither front end exposes
        "freq_bounds", "burn_in_nd", "t_obs_periods", "dt_nd", "psd_t_obs_nd"}
```
In the same test's docstring, item (d) becomes "the keys with no window control are exactly the six the window never exposes, the eleven tool-only diagnostics knobs, and the five FDT settings neither front end exposes (P2)".

**A7 — new Step 9c: the default pin closes over the new keys.** In `tests/test_refusals.py`, `test_every_registry_default_is_the_trees_own_default`, find
```python
    looked_at = set(owned_by_config) | set(owned_by_a_signature) | {"device"}
    rest = {k: f.default for k, f in FIELDS.items() if k not in looked_at}
    assert rest == {k: ("none: it must be given" if k == "t_obs" else None) for k in rest}, rest
```
and replace with
```python
    # piece 5 (P2, P54, P75): the FDT knobs' defaults are FDTConfig's own -- make_fdt_config's keyword
    # defaults equal them -- and n_freqs / ensemble_m also say a sweep takes its preset's value.
    import dataclasses
    from core import cli
    from core.config import FDTConfig
    from core.tool import build_parser
    fdt = {f.name: f.default for f in dataclasses.fields(FDTConfig)}
    owned_by_fdt_config = {"freqs_per_batch": "freqs_per_batch", "f0": "F0", "burn_in_nd": "burn_in_nd",
                           "t_obs_periods": "T_obs_periods", "dt_nd": "dt_nd",
                           "psd_t_obs_nd": "psd_T_obs_nd"}
    for key, name in owned_by_fdt_config.items():
        assert FIELDS[key].default == str(fdt[name]), (key, name, FIELDS[key].default)
    for key, name in (("n_freqs", "n_freqs"), ("ensemble_m", "ensemble_M")):
        assert FIELDS[key].default == f"{fdt[name]}, or the preset's in a sweep", (key, FIELDS[key].default)
    for name in ("n_freqs", "ensemble_M", "freqs_per_batch", "F0"):
        assert _default(cli.make_fdt_config, name) == fdt[name], name
    lo, hi = fdt["freq_bounds"]
    assert FIELDS["freq_bounds"].default == f"{lo} to {hi}", FIELDS["freq_bounds"].default
    assert (FIELDS["preset"].default == build_parser().subcommands["crossval"].get_default("preset")
            == next(iter(cli.SWEEP_PRESETS)))

    looked_at = (set(owned_by_config) | set(owned_by_a_signature) | {"device"} | set(owned_by_fdt_config)
                 | {"n_freqs", "ensemble_m", "freq_bounds", "preset"})
    rest = {k: f.default for k, f in FIELDS.items() if k not in looked_at}
    no_constant = {"t_obs": "none: it must be given", "seed": "none: one is drawn and recorded"}
    assert rest == {k: no_constant.get(k) for k in rest}, rest
```

**A8 — Step 6's expected result:** the new require_below test PASSES. Three tests FAIL: `test_the_registry_holds_exactly_the_initial_keys_with_neutral_descriptions` (key set and count), `test_every_field_key_has_a_flag_and_every_key_literal_under_core_is_registered` (`set(tool_fields.FLAG) == set(FIELDS)`), and `test_every_registry_default_is_the_trees_own_default` (the closing `rest` assertion). A3, A4 and A7 turn them green.

**A9 — Step 10 and Step 11.** Run: `pytest tests/test_refusals.py tests/test_nav_and_gating.py::test_every_field_key_has_a_window_control_and_the_fix_sentences_name_it tests/test_nav_and_gating.py::test_the_gui_control_table_matches_the_tabs_labels -v`. Expected: PASS. The commit adds the test file the Files block already names:
```bash
git add core/refusals.py core/tool/fields.py core/gui/fields.py tests/test_refusals.py tests/test_nav_and_gating.py
```

#### Controller rulings (pre-flight, binding — these supersede the amendments above and the task text)

- **F32** — P54's two-defaults sentence in the `Field` default clause applies to every key whose sweep value comes from the preset, not only `n_freqs` and `ensemble_m`: also `freq_bounds`, `t_obs_periods` and `psd_t_obs_nd`. Extend the suffix to those three and their pins.

---

### Task 9: the fix-hint table widened so a place may be a screen

**Why:** E6 — "The fix-hint table is widened to understand screens, not only the six inference tabs,
and an input that appears in several places lists them all" (spec §5.2). Today `CONTROL`'s tuple
entry must name one of the six inference tab titles, so a bad cell chosen on the measurement screen
sends the owner to the Infer tab.

**Files:**
- Modify: `core/gui/fields.py:1-32` (the module docstring's "Three shapes of entry"),
  `:44-110` (`CONTROL`), `:128-142` (`fix_sentence`)
- Test: `tests/test_nav_and_gating.py:1447-1547`
  (`test_every_field_key_has_a_window_control_and_the_fix_sentences_name_it`)

**Interfaces:**
- Consumes: Task 8's eight `CONTROL` sentence entries.
- Produces: `gui.fields.SCREENS: frozenset` (`{"Artifacts", "Model Builder"}`); a `CONTROL` tuple
  entry whose place may be any section tab title or any member of `SCREENS`, singly or as a tuple;
  `fix_sentence` rendering "on the … screen" for a screen and "on the … tab" for a tab.
  Tasks 22 and 23 add `("Model Builder", …)` and `("Live simulation", …)` entries against this.

- [ ] **Step 1: Write the failing test — the widened pin over every section**

In `tests/test_nav_and_gating.py`, replace the `(b)` block of
`test_every_field_key_has_a_window_control_and_the_fix_sentences_name_it`. Find:

```python
    # (b) the three shapes, against the screen's own tab titles
    qt_app()
    screen = InferenceScreen()
    tabs = [screen.tabs.tabText(i) for i in range(screen.tabs.count())]
    assert tabs == ["Config", "Prior", "Posterior", "Validate", "Infer", "TSNPE"]
    for key, entry in gui_fields.CONTROL.items():
        if isinstance(entry, tuple):
            tab, text = entry
            names = tab if isinstance(tab, tuple) else (tab,)
            assert names and all(t in tabs for t in names), f"{key}: {tab!r} is not a tab title"
```

Replace with:

```python
    # (b) the three shapes, against the tab titles of EVERY section plus the two screens.
    # Read off the built window, not off a list written here: a renamed tab must fail this, and an
    # entry that names the FDT analysis tab is as real as one that names the Infer tab (E6).
    qt_app()
    from core.gui.main_window import MainWindow
    window = MainWindow()
    tabs = []
    for section in (window.reduction_screen, window.fdt_screen, window.inference_screen,
                    window.simulate_screen):
        tabs += [section.tabs.tabText(i) for i in range(section.tabs.count())]
    assert tabs == ["NWK → Hopf reduction map", "FDT analysis", "Sweep study cross-validation",
                    "Config", "Prior", "Posterior", "Validate", "Infer", "TSNPE",
                    "Live simulation"]
    assert gui_fields.SCREENS == frozenset({"Artifacts", "Model Builder"}), \
        "a place is a tab title or one of these two screens, which host no tab widget"
    places = set(tabs) | set(gui_fields.SCREENS)
    for key, entry in gui_fields.CONTROL.items():
        if isinstance(entry, tuple):
            tab, text = entry
            names = tab if isinstance(tab, tuple) else (tab,)
            assert names and all(t in places for t in names), f"{key}: {tab!r} is not a place"
```

and, further down in the same test, find:

```python
            assert gui_fields.label(key) == text
            assert gui_fields.fix_sentence(key) == \
                f"Set it in the '{text}' box on the {' or '.join(names)} tab."
```

Replace with:

```python
            assert gui_fields.label(key) == text
            noun = "screen" if all(n in gui_fields.SCREENS for n in names) else "tab"
            assert gui_fields.fix_sentence(key) == \
                f"Set it in the '{text}' box on the {' or '.join(names)} {noun}."
```

- [ ] **Step 2: Add the collision and screen pins to the same test**

In the same test, find:

```python
    # piece 4's two, on the Artifacts screen rather than a tab (B5, design §2.5)
    assert gui_fields.fix_sentence("artifact") == "Select an artifact in the list on the Artifacts screen."
```

and insert immediately **above** it:

```python
    # E6: an input that appears in several places lists them ALL. The cell picker is on five of
    # them, and a bad cell chosen on the measurement screen used to send the owner to the Infer tab.
    assert gui_fields.fix_sentence("cell") == (
        "Set it in the 'Cell' box on the Infer or FDT analysis or Sweep study cross-validation or "
        "NWK → Hopf reduction map or Live simulation tab.")
    assert gui_fields.fix_sentence("model") == (
        "Set it in the 'Model' box on the Config or FDT analysis or Live simulation tab.")
    assert gui_fields.fix_sentence("n_freqs") == (
        "Set it in the 'n_freqs' box on the FDT analysis or Sweep study cross-validation tab.")
    assert gui_fields.label("s_grid") == "S grid  (T_a/T = 1)", \
        "a box on a SCREEN still has a label, so a row and its hint cannot drift apart (§5.2)"
    # the screen noun, exercised on a place no key claims yet: Task 22's model-builder keys will.
    assert gui_fields._where(("Model Builder",)) == "the Model Builder screen"
    assert gui_fields._where(("Artifacts", "Model Builder")) == "the Artifacts or Model Builder screen"
    assert gui_fields._where(("Infer", "Model Builder")) == "the Infer tab or the Model Builder screen"
    assert gui_fields._where("Posterior") == "the Posterior tab"
```

- [ ] **Step 3: Run it and watch it fail**

Run: `pytest tests/test_nav_and_gating.py::test_every_field_key_has_a_window_control_and_the_fix_sentences_name_it -v`
Expected: FAIL with `AttributeError: module 'core.gui.fields' has no attribute 'SCREENS'`

- [ ] **Step 4: Add `SCREENS` and `_where`, and widen `fix_sentence`**

In `core/gui/fields.py`, find:

```python
_FIXED = "Fixed by measurement: change it in config.py, deliberately."
```

and insert immediately **above** it:

```python
#: Place strings that name a SCREEN rather than a tab. ``fix_sentence`` renders "on the <place>
#: screen" for these and "on the <place> tab" for everything else. The Artifacts browser and the
#: model builder are whole screens with no tab widget; the tab titles outside Parameter Inference --
#: "NWK → Hopf reduction map", "FDT analysis", "Sweep study cross-validation", "Live simulation" --
#: are ORDINARY tab entries, because those sections do host a tab widget (E6, piece 5 §5.2).
SCREENS: frozenset = frozenset({"Artifacts", "Model Builder"})
```

Then find:

```python
def fix_sentence(key: str | None) -> str:
    """What the yellow box says under the message: ``"Set it in the 'T_obs (s)' box on the Infer
    tab."`` for a box entry, the sentence itself for a sentence entry, ``""`` for ``field=None``, a
    None entry or an unknown key. Never raises: it runs while a refusal is being shown."""
    if key is None:
        return ""
    entry = CONTROL.get(key)
    if isinstance(entry, tuple):
        tab, text = entry
        where = " or ".join(tab) if isinstance(tab, tuple) else tab
        return f"Set it in the '{text}' box on the {where} tab."
    return entry or ""
```

Replace with:

```python
def _where(place) -> str:
    """The place phrase: ``"the Infer tab"``, ``"the Posterior or TSNPE tab"``, ``"the Artifacts
    screen"``, and for a mixed tuple ``"the Infer tab or the Model Builder screen"``.

    An all-tab or all-screen tuple shares ONE noun, which is what keeps the sentences the
    walkthrough quotes byte-identical ("on the Posterior or TSNPE tab"); only a tuple that genuinely
    mixes the two spells the noun out per place, because "the Infer or Model Builder tab" would be
    a lie about one of them."""
    names = place if isinstance(place, tuple) else (place,)
    screens = [n in SCREENS for n in names]
    if all(screens):
        return f"the {' or '.join(names)} screen"
    if not any(screens):
        return f"the {' or '.join(names)} tab"
    return " or ".join(f"the {n} {'screen' if n in SCREENS else 'tab'}" for n in names)


def fix_sentence(key: str | None) -> str:
    """What the yellow box says under the message: ``"Set it in the 'T_obs (s)' box on the Infer
    tab."`` for a box entry, the sentence itself for a sentence entry, ``""`` for ``field=None``, a
    None entry or an unknown key. Never raises: it runs while a refusal is being shown."""
    if key is None:
        return ""
    entry = CONTROL.get(key)
    if isinstance(entry, tuple):
        place, text = entry
        return f"Set it in the '{text}' box on {_where(place)}."
    return entry or ""
```

- [ ] **Step 5: Convert the colliding inputs and Task 8's eight to place tuples**

In `core/gui/fields.py`, find:

```python
    # inputs
    "bounds": ("Prior", "Bounds"),
    "cell": ("Infer", "Cell"),
    "units": ("Config", "Units"),
    "model": ("Config", "Model"),
```

Replace with:

```python
    # inputs. The cell picker and the model combo appear on several places at once and each entry
    # names them ALL (E6): a bad cell chosen on the FDT analysis tab used to be answered with "the
    # Infer tab". ``units`` is NOT widened -- one units control exists in the whole application (the
    # Config tab's toggle); everywhere else the units file is resolved from the model and there is
    # no control to name.
    "bounds": ("Prior", "Bounds"),
    "cell": (("Infer", "FDT analysis", "Sweep study cross-validation",
              "NWK → Hopf reduction map", "Live simulation"), "Cell"),
    "units": ("Config", "Units"),
    "model": (("Config", "FDT analysis", "Live simulation"), "Model"),
```

and find the eight sentence entries added in Task 8 (the block beginning
`# the two secondary analyses (piece 5, §5.3). SENTENCES for now:`) and replace the whole block
with:

```python
    # the two secondary analyses (piece 5, §5.3). Ordinary tuple entries now that a place may be any
    # section's tab title: the label is the row the panel builds, so label(key) keeps a box and its
    # hint sentence from drifting apart. The Seed rows arrive with the panels (Tasks 25, 26).
    "n_freqs": (("FDT analysis", "Sweep study cross-validation"), "n_freqs"),
    "ensemble_m": (("FDT analysis", "Sweep study cross-validation"), "ensemble_M"),
    "freqs_per_batch": (("FDT analysis", "Sweep study cross-validation"), "freqs_per_batch"),
    "f0": (("FDT analysis", "Sweep study cross-validation", "NWK → Hopf reduction map"),
           "F0 (ND forcing amplitude)"),
    "preset": ("Sweep study cross-validation", "Preset"),
    "s_grid": ("Sweep study cross-validation", "S grid  (T_a/T = 1)"),
    "t_grid": ("Sweep study cross-validation", "T_a/T grid  (S = 0)"),
    "seed": (("FDT analysis", "Sweep study cross-validation"), "Seed"),
```

- [ ] **Step 6: Update the module docstring's account of the three shapes**

In `core/gui/fields.py`, find:

```python
* ``(tab, label)`` for a box or a picker: the tab title exactly as ``InferenceScreen`` shows it
  (Config, Prior, Posterior, Validate, Infer, TSNPE) and the row label as the tab passes it to
  ``add_help_row``.
```

Replace with:

```python
* ``(place, label)`` for a box or a picker: ``place`` is a tab title exactly as its section shows it
  -- the six inference tabs (Config, Prior, Posterior, Validate, Infer, TSNPE), "FDT analysis",
  "Sweep study cross-validation", "NWK → Hopf reduction map", "Live simulation" -- or one of
  ``SCREENS``, and the label is the row as the panel passes it to ``add_help_row``. A tuple of
  places is how one input names every place it appears (E6): the cell picker is on five of them.
```

- [ ] **Step 7: Run the widened pin and watch it pass**

Run: `pytest tests/test_nav_and_gating.py::test_every_field_key_has_a_window_control_and_the_fix_sentences_name_it -v`
Expected: PASS

- [ ] **Step 8: Run the two table suites together**

Run: `pytest tests/test_refusals.py tests/test_nav_and_gating.py -k "control or field or fix_sentence or table" -v`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add core/gui/fields.py tests/test_nav_and_gating.py
git commit -m "gui: a fix-hint place may be a screen, and a shared input names them all"
```

#### Amendments (binding — these supersede the text above)

**A1 — P33: `t_obs` is widened HERE, to the Simulate panel's box too.**
- **Step 5, add:** in core/gui/fields.py find `    "t_obs": ("Infer", "T_obs (s)"),` and replace with
  ```python
      "t_obs": (("Infer", "Live simulation"), "T_obs (s)"),   # E6/P33: the Simulate panel's box too
  ```
- **Step 2, add:** in the field test's block (c), find
  `    assert gui_fields.fix_sentence("t_obs") == "Set it in the 'T_obs (s)' box on the Infer tab."`
  and replace with
  ```python
      assert gui_fields.fix_sentence("t_obs") == \
          "Set it in the 'T_obs (s)' box on the Infer or Live simulation tab."
  ```
- **New step 5b:** in `tests/test_worker_dispatch.py`, replace each of the three occurrences of the string `"Set it in the 'T_obs (s)' box on the Infer tab."` (two `assert box.informativeText() == ...` lines and one `fix = ...` assignment) with `"Set it in the 'T_obs (s)' box on the Infer or Live simulation tab."`.

**A2 — P30: `name` and `note` name the two new tabs' boxes too** (T25/T26 build them as 'Record name' and 'Note').
- **Step 5, add:** find `    "name": "Choose another name in the Save box.",` and replace with
  ```python
      "name": ("Choose another name in the Save box, or in the 'Record name' box on the FDT analysis or "
               "Sweep study cross-validation tab."),
  ```
  Find `    "note": "Edit it in the Note box on the Artifacts screen.",` and replace with
  ```python
      "note": ("Edit it in the Note box on the Artifacts screen, or in the 'Note' box on the FDT "
               "analysis or Sweep study cross-validation tab."),
  ```
- **Tests:** in tests/test_nav_and_gating.py's field test (c), set both pins to the new sentences:
  `assert gui_fields.fix_sentence("name") == ("Choose another name in the Save box, or in the 'Record name' box on the FDT analysis or Sweep study cross-validation tab.")`
  `assert gui_fields.fix_sentence("note") == ("Edit it in the Note box on the Artifacts screen, or in the 'Note' box on the FDT analysis or Sweep study cross-validation tab.")`.
  In `test_a_rename_failure_reads_as_a_name_refusal`, find
  ```python
          assert box.informativeText() == gui_fields.fix_sentence("name") == \
              "Choose another name in the Save box.", kind
  ```
  and replace the literal with the new `name` sentence. In `tests/test_artifact_browser.py`, find
  `    assert box.informativeText() == "Edit it in the Note box on the Artifacts screen."` and replace the literal with the new `note` sentence. The comment at that file's `# The control core/gui/fields.py names ("Edit it in the Note box on the Artifacts screen.") has` may keep quoting the sentence's first half.

**A3 — the read-back pin must follow the widening (it reds otherwise). New step 5c**, in `tests/test_nav_and_gating.py::test_the_gui_control_table_matches_the_tabs_labels`. Find
```python
    seen = {tab: shown(panel) for tab, panel in tabs.items()}
```
and insert immediately ABOVE it:
```python
    # E6 (piece 5): a place may be any section's tab or one of the two screens, so read them all back
    # off a built window rather than the six inference tabs alone.
    from core.gui.main_window import MainWindow
    window = MainWindow()
    for section in (window.reduction_screen, window.fdt_screen, window.simulate_screen):
        for i in range(section.tabs.count()):
            tabs[section.tabs.tabText(i)] = section.tabs.widget(i)
    tabs["Artifacts"] = window.artifact_screen
    tabs["Model Builder"] = window.model_builder_screen
```
Then find
```python
    missing = []
    for key, (tab, text) in tuples.items():
        for name in (tab if isinstance(tab, tuple) else (tab,)):   # the budget boxes name two tabs
            assert name in seen, f"{key}: CONTROL names a tab that does not exist: {name!r}"
            if labels.pretty_gui(text) not in seen[name]:
                missing.append((key, name, text))
```
and replace with
```python
    # A row a CONTROL entry names before the panel that shows it is built. Each entry must be ABSENT
    # from its tab, so the task that builds the row turns this red and deletes its own line (T25: the
    # FDT analysis Seed row; T26: the Sweep study cross-validation one). An exemption cannot outlive
    # its row.
    _NOT_BUILT_YET = {("seed", "FDT analysis"), ("seed", "Sweep study cross-validation")}
    for key, name in _NOT_BUILT_YET:
        assert labels.pretty_gui(gui_fields.label(key)) not in seen[name], \
            f"{key} now has its row on {name}: delete its _NOT_BUILT_YET entry"
    missing = []
    for key, (tab, text) in tuples.items():
        for name in (tab if isinstance(tab, tuple) else (tab,)):   # the budget boxes name two tabs
            assert name in seen, f"{key}: CONTROL names a place that does not exist: {name!r}"
            if (key, name) in _NOT_BUILT_YET:
                continue
            if labels.pretty_gui(text) not in seen[name]:
                missing.append((key, name, text))
```
Add one sentence to that test's docstring: "Since piece 5 the places are every section's tabs and the two screens (E6), read off a built MainWindow." (This is Option B of this task's open finding. If the owner rules Option A, keep `seed` as Task 8's sentence instead, and drop `_NOT_BUILT_YET`.)

**A4 — Step 5's second edit:** Task 8's sentence block ends at its `"seed": "Set it in the 'Seed' box ..."` line. The five `None` entries Task 8 placed after it, under the comment `# the five FDT settings NEITHER front end exposes`, are NOT part of the block; leave them.

**A5 — Step 1's docstring:** in the field test's docstring, item (b) "every tuple names a tab as InferenceScreen TITLES it" becomes "every tuple names a place: a tab title of any section, read off the built window, or one of SCREENS".

**A6 — Step 8 and Step 9.** Run: `pytest tests/test_refusals.py tests/test_nav_and_gating.py tests/test_worker_dispatch.py tests/test_artifact_browser.py -m "not slow" -q`. Expected: PASS. Commit:
```bash
git add core/gui/fields.py tests/test_nav_and_gating.py tests/test_worker_dispatch.py tests/test_artifact_browser.py
git commit -m "gui: a fix-hint place may be a screen, and a shared input names them all"
```

---

### Task 10: `make_fdt_config`'s checks

**Why:** Spec §3.3's table. Nothing is checked anywhere on this path today: `freqs_per_batch = 0`
makes `_plan_adaptive_batches` append `(start, 0)` for ever, `ensemble_M = 0` raises
`ZeroDivisionError` in `_pick_n_segs`, `F0 = 0` divides by zero in `lock_in_chi`, and `n_freqs = 0`
yields an empty grid, an empty figure and exit 0. This task owns **Review Focus item 3**: every
floor is asserted against a **blank box**, not only against a typed zero, because
`IntField.value()`/`FloatField.value()` return `0` for a blank
(`core/gui/widgets/labeled_inputs.py:47-51,22-26`). The model row is **not** `require_choice`:
`registry.fdt_support` returns `(ok, reason)` with a tailored per-model sentence, so it becomes
`refuse("model", reason)`.

**Files:**
- Modify: `core/cli.py:9-21` (imports), `core/cli.py:319-339` (`make_fdt_config`), and a new
  `_check_unexposed` helper beside it
- Test: `tests/test_fdt_user.py` (the FDT suite; its module docstring is updated in step 7)

**Interfaces:**
- Consumes: Task 7's `make_fdt_config(..., seed=None)` and `cell_sources`; Task 8's field keys
  `n_freqs`, `ensemble_m`, `freqs_per_batch`, `f0`, `seed`.
- Produces: `make_fdt_config` raises `core.refusals.Refusal` before `parse_cell` for every bad knob,
  with `field` set for the five exposed ones and `field=None` for the five settings neither front
  end exposes. Task 12 adds the thin-setting notices on top; Task 21 routes these to the yellow box.

- [ ] **Step 1: Write the failing test for the exposed floors, against blank boxes**

In `tests/test_fdt_user.py`, at the end of the file:

```python
def test_make_fdt_config_refuses_every_zero_knob_a_blank_box_produces():
    """Review Focus 3, and spec §3.3's table. Nothing on this path was checked: freqs_per_batch=0
    makes campaigns._plan_adaptive_batches append (start, 0) for ever -- the application LOOKS HUNG
    rather than failed -- ensemble_M=0 raises ZeroDivisionError inside _pick_n_segs
    (FDT_MAX_ELEMENTS_PER_SEG // batch_size), F0=0 divides by zero in spectral.lock_in_chi
    (2.0 / (F0 * T_obs)), and n_freqs=0 produces an empty grid, an empty figure and exit 0.

    Each floor is asserted against the value a BLANK BOX produces and not only against a typed zero,
    because IntField.value() and FloatField.value() both return 0 for an empty field: a rule written
    as "reject below zero" would accept every blank field in the application. The refusals are
    raised BEFORE parse_cell, so a bad knob costs no file parsing, and each carries the field key
    its front-end table maps to a control or a flag."""
    import pytest

    from core import cli, config
    from core.refusals import Refusal

    cell = str(config.CELL_PATH / "nadrowski" / "master_spont.txt")
    ok = dict(n_freqs=4, ensemble_M=8, freqs_per_batch=1, F0=0.05)

    assert cli.make_fdt_config("NADROWSKI", True, cell, **ok).n_freqs == 4

    for knob, field, sentence in (
            ("n_freqs", "n_freqs",
             "The number of drive frequencies must be at least 1; got 0 (default 60)."),
            ("ensemble_M", "ensemble_m",
             "The number of trajectories per frequency must be at least 1; got 0 (default 256)."),
            ("freqs_per_batch", "freqs_per_batch",
             "The number of frequencies per simulator call must be at least 1; got 0 (default 1)."),
            ("F0", "f0",
             "The non-dimensional drive amplitude must be greater than 0; got 0 (default 0.05).")):
        with pytest.raises(Refusal) as e:
            cli.make_fdt_config("NADROWSKI", True, cell, **{**ok, knob: 0})
        assert e.value.field == field and str(e.value) == sentence, str(e.value)

    # ...and the value a blank box really produces, read off the widgets themselves
    from core.gui.widgets.labeled_inputs import FloatField, IntField
    from tests._fixtures import qt_app
    qt_app()
    blank_int, blank_float = IntField(60), FloatField(0.05)
    blank_int.setText("")
    blank_float.setText("")
    assert blank_int.value() == 0 and blank_float.value() == 0.0, "the premise of this test"
    with pytest.raises(Refusal, match="at least 1"):
        cli.make_fdt_config("NADROWSKI", True, cell, **{**ok, "n_freqs": blank_int.value()})
    with pytest.raises(Refusal, match="greater than 0"):
        cli.make_fdt_config("NADROWSKI", True, cell, **{**ok, "F0": blank_float.value()})
```

- [ ] **Step 2: Run it and watch it fail**

Run: `pytest tests/test_fdt_user.py::test_make_fdt_config_refuses_every_zero_knob_a_blank_box_produces -v`
Expected: FAIL with `DID NOT RAISE <class 'core.refusals.Refusal'>` at the `n_freqs=0` case

- [ ] **Step 3: Write the failing test for the cell, the model and the five unexposed settings**

In `tests/test_fdt_user.py`, after the test from step 1:

```python
def test_make_fdt_config_refuses_a_missing_cell_an_unsupported_model_and_a_broken_band():
    """The rest of spec §3.3's table. A missing cell surfaced as FileNotFoundError from the parser;
    it is now refused by its input kind first, with field="cell", exactly as make_sim_config refuses
    a missing bounds file. The model row is NOT require_choice: registry.fdt_support is a predicate
    that returns a TAILORED diagnostic sentence per model (intrinsic forcing, multiplicative noise,
    a deterministic observable), not a list of choices, so the rule is refuse("model", reason) and
    fdt_support's own words are kept verbatim.

    The frequency band, the burn-in, the two durations and the step are parameters of neither
    builder and are exposed by neither front end (§1.2), so they are checked DEFENSIVELY and their
    refusals carry field=None: there is no control and no flag, and inventing a fix sentence for one
    would be a lie. They are reached through with_overrides, which is how a hand-edited preset or a
    caller can produce one."""
    import pytest

    from core import cli, config, registry
    from core.refusals import Refusal

    cell = str(config.CELL_PATH / "nadrowski" / "master_spont.txt")

    with pytest.raises(Refusal) as e:
        cli.make_fdt_config("NADROWSKI", True, str(config.CELL_PATH / "nadrowski" / "nope.txt"))
    assert e.value.field == "cell" and "was not found" in str(e.value)
    with pytest.raises(Refusal) as e:
        cli.make_fdt_config("NADROWSKI", True, "")
    assert e.value.field == "cell" and "is blank" in str(e.value)

    forced = _register_user("FDT_FORCED_CHECK", [
        {"name": "x", "drift": "-x + F", "noise": "0.1", "forcing": "sin(t)"}])
    assert forced is not None
    ok_, reason = registry.fdt_support("FDT_FORCED_CHECK")
    assert not ok_ and reason
    with pytest.raises(Refusal) as e:
        cli.make_fdt_config("FDT_FORCED_CHECK", False, cell)
    assert e.value.field == "model" and str(e.value) == reason, \
        "fdt_support's own per-model reason, kept verbatim and given a field key"

    good = cli.make_fdt_config("NADROWSKI", True, cell, n_freqs=4, ensemble_M=8)
    for bad, needle in ((dict(dt_nd=0.0), "must be greater than 0"),
                        (dict(psd_T_obs_nd=0.0), "must be greater than 0"),
                        (dict(T_obs_periods=0), "must be greater than 0"),
                        (dict(burn_in_nd=-1.0), "must be at least 0"),
                        (dict(freq_bounds=(0.0, 30.0)), "must be greater than 0"),
                        (dict(freq_bounds=(30.0, 0.1)), "lower bound below its upper bound")):
        with pytest.raises(Refusal) as e:
            cli.check_fdt_settings(good.with_overrides(**bad))
        assert needle in str(e.value), str(e.value)
        assert e.value.field is None, "no control and no flag names it, so no field key"
    assert cli.check_fdt_settings(good) is None, "the built config passes its own check"
```

- [ ] **Step 4: Run it and watch it fail**

Run: `pytest tests/test_fdt_user.py::test_make_fdt_config_refuses_a_missing_cell_an_unsupported_model_and_a_broken_band -v`
Expected: FAIL with `Failed: DID NOT RAISE <class 'core.refusals.Refusal'>` — the first
`make_fdt_config` call raises `FileNotFoundError` from `file_manager.parse_values_file`, which
`pytest.raises(Refusal)` does not catch

- [ ] **Step 5: Add the checks to `make_fdt_config`**

In `core/cli.py`, find:

```python
from collections import OrderedDict
from pathlib import Path
```

Replace with:

```python
import math
from collections import OrderedDict
from pathlib import Path
```

and find:

```python
from .refusals import Refusal, require_file
```

Replace with:

```python
from .refusals import (Refusal, describe, refuse, require_at_least, require_below, require_choice,
                       require_file, require_finite, require_positive)
```

Then find:

```python
def make_fdt_config(model: str, state_dep_drift: bool, cell_file: str, *,
                    n_freqs: int = 60, ensemble_M: int = 256, freqs_per_batch: int = 1,
                    F0: float = 0.05, seed: "int | None" = None) -> FDTConfig:
    """Build an FDTConfig (no prompts) from a model + cell file + FDT knobs. Shared by the command-line
    tool (core/tool) and the GUI's FDT form."""
    (inits_dict, params_dict, rescale_params, force_params_dict,
     units_dict, _, _) = parse_cell(cell_file, model=model)
```

Replace with:

```python
def check_fdt_settings(cfg: FDTConfig) -> None:
    """Refuse the five FDT settings neither front end exposes: the frequency band, the burn-in, the
    two durations and the integration step (§1.2, §3.3).

    They are parameters of neither builder -- they arrive from the dataclass defaults, from the
    closed ``--preset``, or from ``with_overrides`` -- so a bad one is a hand-edited preset or a
    caller's bug, not a mistyped control. That is exactly why these refusals carry NO field key:
    there is no box and no flag for a fix sentence to name, and the front-end tables deliberately
    hold no entry for them. The sentences are the ``require_*`` wordings, minus the field.
    """
    def _positive(what: str, value):
        v = float(value)
        if not math.isfinite(v) or v <= 0:
            raise Refusal(f"{what} must be greater than 0; got {v:g}.")

    _positive("The integration step, in ND units", cfg.dt_nd)
    _positive("The spontaneous recording length, in ND units", cfg.psd_T_obs_nd)
    _positive("The drive window, in periods", cfg.T_obs_periods)
    burn = float(cfg.burn_in_nd)
    if not math.isfinite(burn) or burn < 0:
        # A zero burn-in is a well-defined setting, not a broken one: E5 forbids the over-floor.
        raise Refusal(f"The burn-in, in ND units, must be at least 0; got {burn:g}.")
    lo, hi = (float(v) for v in cfg.freq_bounds)
    for edge, value in (("lower", lo), ("upper", hi)):
        if not math.isfinite(value) or value <= 0:
            raise Refusal(f"The frequency band's {edge} bound must be greater than 0; "
                          f"got {value:g}.")
    if not lo < hi:
        raise Refusal(f"The frequency band must have its lower bound below its upper bound "
                      f"(got {lo:g} and {hi:g}).")


def make_fdt_config(model: str, state_dep_drift: bool, cell_file: str, *,
                    n_freqs: int = 60, ensemble_M: int = 256, freqs_per_batch: int = 1,
                    F0: float = 0.05, seed: "int | None" = None) -> FDTConfig:
    """Build an FDTConfig (no prompts) from a model + cell file + FDT knobs. Shared by the command-line
    tool (core/tool) and the GUI's FDT form.

    The checks live HERE and not on FDTConfig and not in the screens (§1.2), so both front ends
    inherit one wording. They run before ``parse_cell``: every one of them is free, and the four
    knobs are what a blank field turns into a zero -- 0 frequencies is an empty figure and exit 0,
    0 trajectories is a ZeroDivisionError, 0 frequencies per call is an unbounded loop that looks
    like a hang, and F0 = 0 divides by zero in the lock-in.
    """
    n_freqs = require_at_least("n_freqs", n_freqs, 1)
    ensemble_M = require_at_least("ensemble_m", ensemble_M, 1)
    freqs_per_batch = require_at_least("freqs_per_batch", freqs_per_batch, 1)
    F0 = require_positive("f0", F0)
    if seed is not None:
        seed = require_at_least("seed", seed, 0)
    require_file("cell", cell_file, "cell")
    # Local import: core.registry pulls the user-model machinery, and core.cli is imported by
    # core.orchestrator at module scope.
    from core import registry
    ok, reason = registry.fdt_support(model)
    if not ok:
        # NOT require_choice: fdt_support is a predicate returning a tailored diagnostic sentence
        # per model, not a list of choices, and that sentence is the useful half of the refusal.
        refuse("model", reason)
    (inits_dict, params_dict, rescale_params, force_params_dict,
     units_dict, _, _) = parse_cell(cell_file, model=model)
```

and, at the end of the same function, find:

```python
        sources=cell_sources(cell_file, model),
        seed=seed,
    )
```

Replace with:

```python
        sources=cell_sources(cell_file, model),
        seed=seed,
    )
    check_fdt_settings(cfg)
    return cfg
```

renaming the `return FDTConfig(` line that opens the block to `cfg = FDTConfig(`.

- [ ] **Step 6: Run both new tests and watch them pass**

Run: `pytest tests/test_fdt_user.py -k make_fdt_config -v`
Expected: PASS

- [ ] **Step 7: Update the FDT suite's module docstring**

In `tests/test_fdt_user.py`, find:

```python
No cell files / QApplication needed: a tiny fake cfg supplies only .model / .params_dict (and, for the
force-channel test, .inits_tensor / .force_params_dict), which is all these functions read.
```

Replace with:

```python
The normalisation and gate tests need no cell file and no QApplication: a tiny fake cfg supplies only
.model / .params_dict (and, for the force-channel test, .inits_tensor / .force_params_dict), which is
all those functions read. The FDT config-builder tests added by piece 5 DO read the real Nadrowski
cell, because what they pin is the builder refusing a knob before it parses one.
```

- [ ] **Step 8: Run the whole FDT suite and the panel guard beside it**

Run: `pytest tests/test_fdt_user.py tests/test_nav_and_gating.py::test_fdt_panel_guard_translates_model_error_and_gate_admits_builtins -v`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add core/cli.py tests/test_fdt_user.py
git commit -m "fdt: make_fdt_config refuses its knobs before it parses a cell"
```

#### Amendments (binding — these supersede the text above)

**A1 — P75/P2: `check_fdt_settings` uses the registered keys and the ordinary rules** (Task 8 registered `freq_bounds`, `burn_in_nd`, `t_obs_periods`, `dt_nd` and `psd_t_obs_nd`, with `None` in both front-end tables). In Step 5, replace the whole `def check_fdt_settings(...)` with:
```python
def check_fdt_settings(cfg: FDTConfig) -> None:
    """Refuse the five FDT settings neither front end exposes: the frequency band, the burn-in, the
    two durations and the integration step (§1.2, §3.3).

    They are parameters of neither builder -- they arrive from the dataclass defaults, from the
    closed preset, or from ``with_overrides`` -- so a bad one is a hand-edited preset or a caller's
    bug, not a mistyped control. They are registered in ``core.refusals.FIELDS`` like every key and
    map to ``None`` in BOTH front-end tables (P2, P75): each refusal names the setting and offers no
    fix, because there is no box and no flag to name.
    """
    require_positive("dt_nd", cfg.dt_nd)
    require_positive("psd_t_obs_nd", cfg.psd_T_obs_nd)
    require_positive("t_obs_periods", cfg.T_obs_periods)
    # A zero burn-in is a well-defined setting, not a broken one: E5 forbids the over-floor. Finite
    # first, because require_at_least coerces with int() and int(nan) is a bare ValueError.
    require_finite("burn_in_nd", cfg.burn_in_nd)
    require_at_least("burn_in_nd", cfg.burn_in_nd, 0)
    lo, hi = cfg.freq_bounds
    require_positive("freq_bounds", lo)
    require_below("freq_bounds", lo, hi)
```
Do NOT add `import math` (Step 5's first edit is dropped; nothing uses it). Keep Step 5's parenthesised `from .refusals import (...)` line exactly as written: Task 11 uses `describe`, `require_choice` and `require_below` from it.

**A2 — Step 3's test asserts the keys.** Replace its closing loop (from `good = cli.make_fdt_config(...)` to the end) with:
```python
    from core.gui import fields as gui_fields
    from core.tool import fields as tool_fields

    good = cli.make_fdt_config("NADROWSKI", True, cell, n_freqs=4, ensemble_M=8)
    for bad, key, needle in ((dict(dt_nd=0.0), "dt_nd", "must be greater than 0"),
                             (dict(psd_T_obs_nd=0.0), "psd_t_obs_nd", "must be greater than 0"),
                             (dict(T_obs_periods=0), "t_obs_periods", "must be greater than 0"),
                             (dict(burn_in_nd=-1.0), "burn_in_nd", "must be at least 0"),
                             (dict(freq_bounds=(0.0, 30.0)), "freq_bounds", "must be greater than 0"),
                             (dict(freq_bounds=(30.0, 0.1)), "freq_bounds",
                              "lower bound below its upper bound")):
        with pytest.raises(Refusal) as e:
            cli.check_fdt_settings(good.with_overrides(**bad))
        assert needle in str(e.value), str(e.value)
        assert e.value.field == key, (key, e.value.field)
        assert gui_fields.fix_sentence(key) == "" and tool_fields.fix_sentence(key) == "", \
            "no control and no flag: the message names the setting and offers no fix (P2)"
    assert cli.check_fdt_settings(good.with_overrides(burn_in_nd=0.0)) is None, "E5: a zero burn-in is legal"
    assert cli.check_fdt_settings(good) is None, "the built config passes its own check"
```
In the same test's docstring, replace "so they are checked DEFENSIVELY and their refusals carry field=None: there is no control and no flag, and inventing a fix sentence for one would be a lie" with "so they are checked DEFENSIVELY under their own registered keys, which map to None in both front-end tables: the message names the setting and fix_sentence adds nothing, because there is no control and no flag (P2, P75)".

**A3 — Step 3's user model is registered and REMOVED, in the file's own schema.** Replace
```python
    forced = _register_user("FDT_FORCED_CHECK", [
        {"name": "x", "drift": "-x + F", "noise": "0.1", "forcing": "sin(t)"}])
    assert forced is not None
    ok_, reason = registry.fdt_support("FDT_FORCED_CHECK")
    assert not ok_ and reason
    with pytest.raises(Refusal) as e:
        cli.make_fdt_config("FDT_FORCED_CHECK", False, cell)
    assert e.value.field == "model" and str(e.value) == reason, \
        "fdt_support's own per-model reason, kept verbatim and given a field key"
```
with
```python
    try:
        _register_user("FDT_FORCED_CHECK", [
            {"name": "x", "drift": "-k*x", "D": "d0", "forcing": {"kind": "sin", "params": {}}}])
        ok_, reason = registry.fdt_support("FDT_FORCED_CHECK")
        assert not ok_ and "forcing" in reason, reason
        with pytest.raises(Refusal) as e:
            cli.make_fdt_config("FDT_FORCED_CHECK", False, cell)
        assert e.value.field == "model" and str(e.value) == reason, \
            "fdt_support's own per-model reason, kept verbatim and given a field key"
    finally:
        registry.unregister("FDT_FORCED_CHECK")
```

**A4 — P54: Step 1's two expected sentences** become
```python
            ("n_freqs", "n_freqs",
             "The number of drive frequencies must be at least 1; got 0 "
             "(default 60, or the preset's in a sweep)."),
            ("ensemble_M", "ensemble_m",
             "The number of trajectories per frequency must be at least 1; got 0 "
             "(default 256, or the preset's in a sweep)."),
```
(the freqs_per_batch and F0 sentences are unchanged).

**A5 — Step 4's expected failure** is the uncaught `FileNotFoundError` from `file_manager.parse_values_file` on `nope.txt`, which `pytest.raises(Refusal)` does not catch. It is not "DID NOT RAISE".

**A6 — Step 7's docstring** additionally says: "One of them also builds two numeric widgets (offscreen) to read the value a blank box really produces." Replace the file's first line phrase "Qt-free" with "Qt-free except for that one widget read".

**A7 — Interfaces, "Produces":** read "with `field` set for the five exposed ones and `field=None` for the five settings neither front end exposes" as "with `field` set for every refusal: the five exposed knobs and (P75) `dt_nd`, `psd_t_obs_nd`, `t_obs_periods`, `burn_in_nd`, `freq_bounds`, whose table entries are `None`".

#### Controller rulings (pre-flight, binding — these supersede the amendments above and the task text)

- **F33** — for `burn_in_nd` use `require_between("burn_in_nd", value, 0.0, math.inf, open_hi=True)`, not `require_at_least(..., 0)`. `require_at_least` coerces with `int()`, so a fractional negative such as -0.5 becomes 0 and passes, and NaN raises a bare `ValueError` from `int(nan)`; `require_between` refuses NaN, any negative and infinity, and it is an existing rule, so P75's principle (ordinary rules, no hand-built sentences) holds. Read the message it renders for this key: if it reads badly with an infinite upper bound, fall back to `require_finite` followed by `require_at_least(..., 0)` and say so in your report.

---

### Task 11: `make_param_sweep_config`'s checks and its `preset_name` keyword

**Why:** Spec §4.4 — the builder applies §3.3's rules and, in addition, each grid is `(min, max, N)`
with `N` at least 2, `min` and `max` finite, and `min` below `max`. The tool checks the point count
already (`core/tool/fdt.py:73-77`); **the window checks nothing**. The builder takes `preset` as an
already-resolved dict and the call site drops the NAME (`core/tool/fdt.py:151-153`), which
`body.settings["preset"]` must hold — hence the new `preset_name` keyword. Review Focus item 3
applies to the grid fields too: `_GridRow.spec()` is three `value()` calls, so a blank grid arrives
as `(0.0, 0.0, 0)`.

**Files:**
- Modify: `core/cli.py:377-405` (`make_param_sweep_config`)
- Modify: `core/tool/fdt.py:148-157` (`run_crossval`) and
  `core/gui/panels/crossval_panel.py:140-150` (`CrossValPanel._run`) — both call sites pass the name
- Test: `tests/test_fdt_user.py` (beside Task 10's builder tests)
- Test: `tests/test_tool.py:1226-1250` — the exact-set `kw` pins go red the moment the keyword lands

**Interfaces:**
- Consumes: Task 7's `sources`/`seed` fill; Task 8's `preset`, `s_grid`, `t_grid` keys and
  `require_below`; Task 10's `check_fdt_settings`.
- Produces: `cli.make_param_sweep_config(cell_file, *, preset, preset_name, s_spec, t_spec,
  n_freqs=None, ensemble_M=None, freqs_per_batch=None, F0=None, seed=None) -> (FDTConfig, s_grid,
  temp_grid)`. Task 19 reads `cfg.sources` and the resolved knobs; Task 26 passes `preset_name` from
  the combo.

- [ ] **Step 1: Write the failing test**

In `tests/test_fdt_user.py`, after Task 10's tests:

```python
def test_make_param_sweep_config_refuses_a_blank_grid_and_records_the_preset_name():
    """Spec §4.4. The window checks NOTHING about its two grids today, and _GridRow.spec() is three
    value() calls -- so a grid whose 'max' was left empty arrives as (0.0, 0.0, 0) and np.linspace
    produces a sweep of zero points without a word. Each grid is therefore checked as a whole: both
    ends finite, at least 2 points, and the minimum below the maximum.

    ``preset_name`` exists because the builder takes ``preset`` as an already-RESOLVED dict and both
    call sites drop the name (core/tool/fdt.py's ``dict(cli.SWEEP_PRESETS[args.preset])``), while
    body.settings["preset"] has to hold the name a reader can act on. It is a closed choice in both
    front ends and is checked as one."""
    import numpy as np
    import pytest

    from core import cli, config
    from core.refusals import Refusal

    cell = str(config.CELL_PATH / "nadrowski" / "master_spont.txt")
    preset = dict(cli.SWEEP_PRESETS["exploratory"])
    ok = dict(preset=preset, preset_name="exploratory", s_spec=(0.0, 0.5, 3), t_spec=(1.0, 1.5, 3))

    cfg, s_grid, t_grid = cli.make_param_sweep_config(cell, **ok)
    assert cfg.model == "NADROWSKI" and cfg.hw.device.type == "cpu"
    assert cfg.n_freqs == preset["n_freqs"] and cfg.ensemble_M == preset["ensemble_M"], \
        "an unset knob falls back to the preset, in the builder rather than at each call site"
    assert np.allclose(s_grid, np.linspace(0.0, 0.5, 3)) and len(t_grid) == 3
    assert cfg.sources["cell"] == cell

    for spec, field, needle in (
            ((0.0, 0.0, 0), "s_grid", "at least 2 points"),
            ((0.0, 0.5, 1), "s_grid", "at least 2 points"),
            ((0.5, 0.1, 3), "s_grid", "lower bound below its upper bound"),
            ((float("nan"), 0.5, 3), "s_grid", "must be a finite number")):
        with pytest.raises(Refusal) as e:
            cli.make_param_sweep_config(cell, **{**ok, "s_spec": spec})
        assert e.value.field == field and needle in str(e.value), str(e.value)

    with pytest.raises(Refusal) as e:
        cli.make_param_sweep_config(cell, **{**ok, "t_spec": (1.5, 1.0, 3)})
    assert e.value.field == "t_grid", "the refusal names the grid the user actually broke"

    with pytest.raises(Refusal) as e:
        cli.make_param_sweep_config(cell, **{**ok, "preset_name": "overnight"})
    assert e.value.field == "preset" and str(e.value) == (
        "The resolution preset must be one of exploratory, production; got 'overnight' "
        "(default exploratory).")

    # §3.3's four shared knobs are checked here too, with the same wording the single-cell builder
    # uses: one rule set, two builders.
    with pytest.raises(Refusal) as e:
        cli.make_param_sweep_config(cell, **ok, ensemble_M=0)
    assert e.value.field == "ensemble_m" and "at least 1" in str(e.value)
```

- [ ] **Step 2: Run it and watch it fail**

Run: `pytest tests/test_fdt_user.py::test_make_param_sweep_config_refuses_a_blank_grid_and_records_the_preset_name -v`
Expected: FAIL with `TypeError: make_param_sweep_config() got an unexpected keyword argument 'preset_name'`

- [ ] **Step 3: Rewrite the builder**

In `core/cli.py`, find:

```python
def make_param_sweep_config(cell_file: str, *, preset: dict, s_spec: tuple, t_spec: tuple,
                            n_freqs: int, ensemble_M: int, freqs_per_batch: int = 1,
                            F0: float = 0.05, seed: "int | None" = None) -> tuple["FDTConfig", "np.ndarray", "np.ndarray"]:
    """Build (FDTConfig, s_grid, temp_grid) for the sweep study (no prompts). ``preset`` supplies the
    advanced resolution levers (freq_bounds / T_obs_periods / psd_T_obs_nd); ``s_spec``/``t_spec`` are
    (min, max, n_points). Model fixed to NADROWSKI. Shared by the command-line tool (core/tool) + the GUI."""
    import numpy as np  # local import — keep top-of-file lean
    (inits_dict, params_dict, rescale_params, force_params_dict,
     units_dict, _, _) = parse_cell(cell_file, model="NADROWSKI")
    s_grid = np.linspace(*s_spec)
    temp_grid = np.linspace(*t_spec)
```

Replace with:

```python
def _check_grid(key: str, spec: tuple) -> tuple:
    """One sweep axis, ``(min, max, N)``, as ``np.linspace`` needs it: both ends finite, ``N`` a
    whole number of at least 2, and the minimum below the maximum (§4.4).

    Checked as a WHOLE, because a blank field in the window's grid row reads as 0 -- a grid whose
    'max' was never filled in arrives as ``(0.0, 0.0, 0)``, and neither end on its own is wrong.
    """
    lo, hi, n = spec
    if n is None:
        n = 0
    n = int(n)
    if n < 2:
        what = describe(key)
        refuse(key, f"{what[0].upper()}{what[1:]} needs at least 2 points; got {n}.")
    lo, hi = require_below(key, lo, hi)
    return lo, hi, n


def make_param_sweep_config(cell_file: str, *, preset: dict, preset_name: str,
                            s_spec: tuple, t_spec: tuple,
                            n_freqs: int | None = None, ensemble_M: int | None = None,
                            freqs_per_batch: int | None = None, F0: float | None = None,
                            seed: "int | None" = None) -> tuple["FDTConfig", "np.ndarray", "np.ndarray"]:
    """Build (FDTConfig, s_grid, temp_grid) for the sweep study (no prompts). ``preset`` supplies the
    advanced resolution levers (freq_bounds / T_obs_periods / psd_T_obs_nd); ``s_spec``/``t_spec`` are
    (min, max, n_points). Model fixed to NADROWSKI. Shared by the command-line tool (core/tool) + the GUI.

    ``preset_name`` is the name of the preset ``preset`` was resolved from. Both front ends pick the
    preset from a closed list and then pass the resolved DICT, dropping the name -- and the record's
    ``settings["preset"]`` has to hold the name, because the dict alone does not say which of the two
    a reader is looking at (§4.4).

    Each unset knob falls back to the preset (or to FDTConfig's own default), here rather than at
    each call site, so the window and the tool cannot fall back differently.
    """
    import numpy as np  # local import — keep top-of-file lean
    require_choice("preset", preset_name, tuple(SWEEP_PRESETS))
    n_freqs = require_at_least("n_freqs", preset["n_freqs"] if n_freqs is None else n_freqs, 1)
    ensemble_M = require_at_least(
        "ensemble_m", preset["ensemble_M"] if ensemble_M is None else ensemble_M, 1)
    freqs_per_batch = require_at_least(
        "freqs_per_batch", 1 if freqs_per_batch is None else freqs_per_batch, 1)
    F0 = require_positive("f0", 0.05 if F0 is None else F0)
    if seed is not None:
        seed = require_at_least("seed", seed, 0)
    s_spec = _check_grid("s_grid", s_spec)
    t_spec = _check_grid("t_grid", t_spec)
    require_file("cell", cell_file, "cell")
    (inits_dict, params_dict, rescale_params, force_params_dict,
     units_dict, _, _) = parse_cell(cell_file, model="NADROWSKI")
    s_grid = np.linspace(*s_spec)
    temp_grid = np.linspace(*t_spec)
```

and at the end of the same function, find:

```python
        sources=cell_sources(cell_file, "NADROWSKI"),
        seed=seed,
    )
    return cfg, s_grid, temp_grid
```

Replace with:

```python
        sources=cell_sources(cell_file, "NADROWSKI"),
        seed=seed,
    )
    check_fdt_settings(cfg)
    return cfg, s_grid, temp_grid
```

- [ ] **Step 4: Pass the name from both call sites**

In `core/tool/fdt.py`, find:

```python
    cfg, s_grid, temp_grid = cli.make_param_sweep_config(
        args.cell, preset=preset,
        s_spec=_grid("--s-grid", args.s_grid), t_spec=_grid("--t-grid", args.t_grid),
```

Replace with:

```python
    cfg, s_grid, temp_grid = cli.make_param_sweep_config(
        args.cell, preset=preset, preset_name=args.preset,
        s_spec=_grid("--s-grid", args.s_grid), t_spec=_grid("--t-grid", args.t_grid),
```

In `core/gui/panels/crossval_panel.py`, find:

```python
            cfg, s_grid, temp_grid = cli.make_param_sweep_config(
                cell, preset=preset, s_spec=self.s_grid.spec(), t_spec=self.t_grid.spec(),
```

Replace with:

```python
            cfg, s_grid, temp_grid = cli.make_param_sweep_config(
                cell, preset=preset, preset_name=self.preset_combo.currentText(),
                s_spec=self.s_grid.spec(), t_spec=self.t_grid.spec(),
```

- [ ] **Step 5: Run the tool's forwarding test and watch it go red**

Run: `pytest tests/test_tool.py -k forward -v`
Expected: FAIL with
`AssertionError: assert {'preset', 'preset_name', 's_spec', 't_spec', 'n_freqs', 'ensemble_M', 'freqs_per_batch', 'F0'} == {'preset', 's_spec', 't_spec', 'n_freqs', 'ensemble_M', 'freqs_per_batch', 'F0'}`

- [ ] **Step 6: Widen that test's two exact-set pins**

In `tests/test_tool.py`, find:

```python
    assert set(kw) == {"preset", "s_spec", "t_spec", "n_freqs", "ensemble_M", "freqs_per_batch", "F0"}
```

Replace with:

```python
    assert set(kw) == {"preset", "preset_name", "s_spec", "t_spec", "n_freqs", "ensemble_M",
                       "freqs_per_batch", "F0"}
    assert kw["preset_name"] == "exploratory", \
        "the resolved dict does not say which preset it is; body.settings must hold the name (§4.4)"
```

and find:

```python
    assert set(kw) == {"preset", "s_spec", "t_spec", "n_freqs", "ensemble_M"}, \
        "freqs_per_batch and F0 were left unset -- they must not be forwarded"
```

Replace with:

```python
    assert set(kw) == {"preset", "preset_name", "s_spec", "t_spec", "n_freqs", "ensemble_M"}, \
        "freqs_per_batch and F0 were left unset -- they must not be forwarded"
    assert kw["preset_name"] == "production"
```

- [ ] **Step 7: Run the three suites and watch them pass**

Run: `pytest tests/test_fdt_user.py tests/test_tool.py -k "sweep or forward or crossval" -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add core/cli.py core/tool/fdt.py core/gui/panels/crossval_panel.py tests/test_fdt_user.py tests/test_tool.py
git commit -m "crossval: the sweep builder checks its grids and keeps the preset's name"
```

#### Amendments (binding — these supersede the text above)

**A1 — P72: the config carries the preset's NAME.** New Step 3a: in `core/sim_config.py` find (Task 7's text)
```python
    # The seed the run used, drawn when none was supplied (E7). None means "not chosen yet".
    seed: "int | None" = None
```
and replace with
```python
    # The seed the run used, drawn when none was supplied (E7). None means "not chosen yet".
    seed: "int | None" = None

    # The sweep preset's NAME ("exploratory" | "production"), set by cli.make_param_sweep_config;
    # None for a single-cell run and for the reduction map. The resolved dict alone does not say
    # which preset it was, and body.settings["preset"] must (§4.4, P72). Defaulted, for §1.3.
    preset_name: "str | None" = None
```
In Step 3's tail replacement, the construction ends
```python
        sources=cell_sources(cell_file, "NADROWSKI"),
        seed=seed,
        preset_name=preset_name,
    )
    check_fdt_settings(cfg)
    return cfg, s_grid, temp_grid
```
In Step 1's test, after `assert cfg.sources["cell"] == cell`, add
```python
    assert cfg.preset_name == "exploratory", "body.settings['preset'] is read off the config (P72)"
    assert cli.make_fdt_config("NADROWSKI", True, cell, n_freqs=4, ensemble_M=8).preset_name is None
```

**A2 — P79: the tool forwards the seed.** In Step 4, also find in `core/tool/fdt.py`
```python
        **knobs(args, "freqs_per_batch", "F0"))
```
and replace with
```python
        **knobs(args, "freqs_per_batch", "F0", "seed"))
```
`knobs` reads `getattr(args, n, None)`, so this is inert until Task 28 declares `--seed` on crossval, and it is not forwarded when unset. Step 6's exact-set pins therefore do NOT gain "seed".

**A3 — Step 5's command** (the pins are in `test_fdt_and_crossval_flags_reach_their_builders`, which `-k forward` does not select): `pytest "tests/test_tool.py::test_fdt_and_crossval_flags_reach_their_builders" -v`. The expected failure is as written.

**A4 — Step 7's command** (keep the slow tiny-size run and the unrelated `artifacts sweep` tests out): `pytest tests/test_fdt_user.py tests/test_tool.py::test_fdt_and_crossval_flags_reach_their_builders tests/test_tool.py::test_fdt_and_crossval_usage_errors tests/test_tool.py::test_crossval_preset_choices_match_sweep_presets tests/test_artifact_store.py::test_fdt_config_carries_sources_and_a_seed_and_copies_itself_for_a_run -m "not slow" -v`. Expected: PASS.

**A5 — Files and commit:** add `core/sim_config.py` (A1).
```bash
git add core/sim_config.py core/cli.py core/tool/fdt.py core/gui/panels/crossval_panel.py tests/test_fdt_user.py tests/test_tool.py
git commit -m "crossval: the sweep builder checks its grids and keeps the preset's name"
```

---

### Task 12: the thin-setting notices and their thresholds

**Why:** E5 and spec §3.3's last paragraph — "Checks refuse what breaks; a setting too thin to trust
warns, and the warning is recorded … Below a documented threshold — fewer than two grid frequencies,
fewer than eight trajectories — the run raises a `PreflightWarning` whose sentence goes into
`body.notices`, so the record says the answer is a quick look. The thresholds are named constants in
`config.py`. Eight is chosen because the existing end-to-end test runs at eight and must keep
passing." A deliberately tiny quick look stays possible and is visibly marked as one.

**Files:**
- Modify: `core/config.py:221-225` (after the experimental constants)
- Modify: `core/FDT/fdt_pipeline.py:39-62` (beside `_estimate_omega_0`) and `:63-75` (the top of
  `run_fdt`)
- Modify: `core/FDT/cross_validation.py:193` (the top of `run_fdt_param_sweep`)
- Test: `tests/test_fdt_user.py` (beside Tasks 10 and 11's builder tests)

**Interfaces:**
- Consumes: Task 10's and Task 11's builders (the configs whose knobs are judged).
- Produces: `config.FDT_THIN_N_FREQS: int`, `config.FDT_THIN_ENSEMBLE_M: int`;
  `fdt_pipeline.thin_notices(cfg) -> list[str]` (pure) and
  `fdt_pipeline.warn_thin_settings(cfg) -> list[str]` (warns, then returns the same list).
  Task 17 puts the returned list into `body.notices`; Task 19 does the same for each sweep record.

- [ ] **Step 1: Write the failing test**

In `tests/test_fdt_user.py`, after Task 11's test:

```python
def test_a_thin_setting_warns_and_hands_back_the_sentence_for_the_record():
    """E5: checks refuse what BREAKS; a setting too thin to trust warns instead, and the warning is
    recorded. A one-frequency grid and a two-trajectory ensemble both produce a real number -- the
    computation is defined -- but the number is a quick look, and a record that does not say so
    reads later as a measurement. A floor here would forbid the quick look, which E5 explicitly
    does not.

    The channel is PreflightWarning, the same one every other judgement in the tree uses
    (core/orchestrator.py:81), so the window shows it at warning severity, the tool sends it to
    stderr, and the run buffer copies it into the record's log.txt -- and the SENTENCES come back so
    the stage can put them in body.notices, which is the half a warning alone cannot do.

    Eight trajectories is the threshold because the existing end-to-end test runs at eight and must
    keep passing without a notice; two frequencies because one frequency is not a spectrum."""
    import warnings

    import pytest

    from core import cli, config
    from core.FDT import fdt_pipeline
    from core.orchestrator import PreflightWarning

    assert (config.FDT_THIN_N_FREQS, config.FDT_THIN_ENSEMBLE_M) == (2, 8)

    cell = str(config.CELL_PATH / "nadrowski" / "master_spont.txt")
    fat = cli.make_fdt_config("NADROWSKI", True, cell, n_freqs=2, ensemble_M=8)
    assert fdt_pipeline.thin_notices(fat) == [], "at the thresholds exactly: nothing to say"

    thin = cli.make_fdt_config("NADROWSKI", True, cell, n_freqs=1, ensemble_M=2)
    said = fdt_pipeline.thin_notices(thin)
    assert said == [
        "The frequency grid has 1 point, below the 2 this measurement is trusted at: read the "
        "result as a quick look, not as a measurement.",
        "The ensemble is 2 trajectories, below the 8 this measurement is trusted at: read the "
        "result as a quick look, not as a measurement.",
    ], said

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        returned = fdt_pipeline.warn_thin_settings(thin)
    assert returned == said, "the sentences come back for body.notices, not only to the warning hook"
    assert [str(w.message) for w in rec] == said
    assert all(issubclass(w.category, PreflightWarning) for w in rec), [w.category for w in rec]

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        assert fdt_pipeline.warn_thin_settings(fat) == []
    assert rec == [], "a run at the thresholds is not annotated"

    # the run entries raise it themselves: a notice the operator never sees is not a notice
    import inspect
    for fn in (fdt_pipeline.run_fdt,):
        assert "warn_thin_settings" in inspect.getsource(fn), fn.__name__
    from core.FDT import cross_validation
    assert "warn_thin_settings" in inspect.getsource(cross_validation.run_fdt_param_sweep)
```

- [ ] **Step 2: Run it and watch it fail**

Run: `pytest tests/test_fdt_user.py::test_a_thin_setting_warns_and_hands_back_the_sentence_for_the_record -v`
Expected: FAIL with `AttributeError: module 'core.config' has no attribute 'FDT_THIN_N_FREQS'`

- [ ] **Step 3: Add the two thresholds to `config.py`**

In `core/config.py`, find:

```python
# === EXPERIMENTAL CONSTANTS (in seconds, converted to cell file units during setup) ===
DT_EXP_S = 1e-3        # 1000 FPS camera frame interval
T_MIN_EXP_S = 1.0      # shortest expected recording (1 s)
T_MAX_EXP_S = 60.0     # longest expected recording (1 min)
```

Replace with:

```python
# === EXPERIMENTAL CONSTANTS (in seconds, converted to cell file units during setup) ===
DT_EXP_S = 1e-3        # 1000 FPS camera frame interval
T_MIN_EXP_S = 1.0      # shortest expected recording (1 s)
T_MAX_EXP_S = 60.0     # longest expected recording (1 min)

# === FDT "TOO THIN TO TRUST" THRESHOLDS (piece 5, E5) ===
# NOT floors. Below either of these the effective-temperature measurement is still well defined and
# still runs -- a deliberately tiny quick look is a thing the owner does on purpose -- but the run
# raises a PreflightWarning and the sentence is recorded in the record's `notices`, so the answer
# cannot later be read as a measurement. Floors live in the config builders and exist only where the
# computation is otherwise undefined (core/cli.py, spec §3.3).
FDT_THIN_N_FREQS = 2    # one frequency is not a spectrum
FDT_THIN_ENSEMBLE_M = 8 # the existing tiny-size end-to-end run uses exactly 8 and must stay unmarked
```

- [ ] **Step 4: Add the two helpers to `fdt_pipeline.py`**

In `core/FDT/fdt_pipeline.py`, find:

```python
import logging
import math
from datetime import datetime
```

Replace with:

```python
import logging
import math
import warnings
from datetime import datetime
```

Then find:

```python
def run_fdt(cfg: FDTConfig, *, skip_sanity: bool, confirm_production: bool) -> None:
```

and insert immediately **above** it:

```python
def thin_notices(cfg: FDTConfig) -> list:
    """The "too thin to trust" sentences for this run's settings; ``[]`` when there are none (E5).

    Pure, and shared by the single-cell run and the sweep, so both mark a quick look the same way
    and ``body.notices`` carries the same words the operator was shown. The thresholds are
    ``config.FDT_THIN_*`` and are read LIVE, never from-imported, so a session can move one.
    """
    notices = []
    if int(cfg.n_freqs) < config.FDT_THIN_N_FREQS:
        notices.append(
            f"The frequency grid has {int(cfg.n_freqs)} point"
            f"{'' if int(cfg.n_freqs) == 1 else 's'}, below the {config.FDT_THIN_N_FREQS} this "
            f"measurement is trusted at: read the result as a quick look, not as a measurement.")
    if int(cfg.ensemble_M) < config.FDT_THIN_ENSEMBLE_M:
        notices.append(
            f"The ensemble is {int(cfg.ensemble_M)} trajector"
            f"{'y' if int(cfg.ensemble_M) == 1 else 'ies'}, below the "
            f"{config.FDT_THIN_ENSEMBLE_M} this measurement is trusted at: read the result as a "
            f"quick look, not as a measurement.")
    return notices


def warn_thin_settings(cfg: FDTConfig) -> list:
    """Raise one ``PreflightWarning`` per thin setting and return the sentences for ``body.notices``.

    The warning is what the operator sees while the run is going (the window's pane at warning
    severity, the tool's stderr) and what the run buffer copies into ``log.txt``; the returned list
    is what the record keeps, which a warning alone cannot do. The import is LOCAL because
    ``core.orchestrator`` imports ``core.cli`` at module scope and this module is reached from both
    front ends before the SBI stack is needed.
    """
    from core.orchestrator import PreflightWarning
    from core.runs import RUN_BOUNDARY_FILES
    notices = thin_notices(cfg)
    for sentence in notices:
        warnings.warn(sentence, PreflightWarning, stacklevel=2,
                      skip_file_prefixes=RUN_BOUNDARY_FILES)
    return notices
```

- [ ] **Step 5: Raise them from the two run entries**

In `core/FDT/fdt_pipeline.py`, find:

```python
    # 1. Model-specific natural-frequency starting estimate; the production omega_0
    #    is refined from the Campaign 1 PSD peak below.
    cfg.omega_0, omega_0_desc = _estimate_omega_0(cfg)
```

Replace with:

```python
    # 0. The settings too thin to trust: not a refusal (E5 keeps the quick look possible), a
    #    judgement the operator sees now and the record keeps afterwards (Task 17 stores it).
    notices = warn_thin_settings(cfg)

    # 1. Model-specific natural-frequency starting estimate; the production omega_0
    #    is refined from the Campaign 1 PSD peak below.
    cfg.omega_0, omega_0_desc = _estimate_omega_0(cfg)
```

In `core/FDT/cross_validation.py`, find:

```python
    fixed_overrides = fixed_overrides or {}
```

Replace with:

```python
    from .fdt_pipeline import warn_thin_settings
    notices = warn_thin_settings(cfg)          # E5; Task 19 stores these on the sweep's record
    fixed_overrides = fixed_overrides or {}
```

(`cross_validation` already imports `_estimate_omega_0` from `fdt_pipeline` at module scope; the
local import here keeps the two names that the sweep uses side by side at their call sites — replace
it with a module-scope import if the implementer prefers, the behaviour is identical.)

- [ ] **Step 6: Run the new test and watch it pass**

Run: `pytest tests/test_fdt_user.py::test_a_thin_setting_warns_and_hands_back_the_sentence_for_the_record -v`
Expected: PASS

- [ ] **Step 7: Run the FDT suite's existing stubbed sweep beside it**

Run: `pytest tests/test_fdt_user.py -v`
Expected: PASS — the stubbed sweep runs at `ensemble_M = 2`, so it now also emits one
`PreflightWarning`; its `pytest.warns(UserWarning, match="1/2 operating points failed")` still
passes, because `pytest.warns` requires one match and tolerates the others.

- [ ] **Step 8: Commit**

```bash
git add core/config.py core/FDT/fdt_pipeline.py core/FDT/cross_validation.py tests/test_fdt_user.py
git commit -m "fdt: a setting too thin to trust warns and the sentence is kept"
```
## THE SCIENCE FIXES (T13–T16)

These four tasks land **before** T17, so `run_fdt`'s signature is still
`run_fdt(cfg, *, skip_sanity, confirm_production) -> None` throughout and its figures still go to
`_out_dir()` (`<artifacts root>/fdt`). Nothing here creates a writer, a record or a body key; T17
and T18 lift what these tasks compute (`prefactor`, `lo_res`/`hi_res`, `blanks`) into `body.grid`
and `body.offgrid`.

All four tasks' tests go in `tests/test_fdt_user.py`. That file is the Qt-free FDT unit suite: it
already holds `test_the_fdt_messages_are_records_with_their_own_levels`, which tests the pipeline
and the sweep rather than user models, so the pipeline's own unit tests have siblings there. None of
these tests simulates anything — every campaign seam is stubbed, exactly as that test stubs it — so
none of them is `slow`-marked.

#### Amendments (binding — these supersede the text above)

**A1 — new Step 5b: the existing record test's stub must carry the two knobs step 0 reads.** `test_the_fdt_messages_are_records_with_their_own_levels` drives `run_fdt` with a stub that has only a model name, and step 0 now reads `cfg.n_freqs` and `cfg.ensemble_M` first. In `tests/test_fdt_user.py`, find
```python
    class _Hopf:
        model = "HOPF"
```
and replace with
```python
    class _Hopf:
        model = "HOPF"
        # run_fdt judges the thin settings first (E5) and reads both knobs to do it. At these values
        # nothing is said, so the exact record list below is unchanged.
        n_freqs, ensemble_M = 60, 256
```
(Task 13 later finds `class _Hopf:\n        model = "HOPF"` together with the comment line above it and appends `params_dict` after the model line. These lines stay.)

**A2 — Step 7's expected result** holds only with A1 applied: PASS. The stubbed sweep (`ensemble_M = 2`) emits one extra PreflightWarning, which `pytest.warns(UserWarning, match="1/2 operating points failed")` tolerates. The `_Hopf` run (60, 256) emits none.

**A3 — scope note:** `notices` stays an unused local in both entries after this task, by design: Task 17 stores the single-cell list and Task 19 the sweep's. Do not store them here. Keep the literal call `warn_thin_settings(` in `run_fdt`'s and `run_fdt_param_sweep`'s own bodies, because this task's test pins it with `inspect.getsource`.

**A4 — commit:** unchanged (A1 edits `tests/test_fdt_user.py`, already in the git add line).

---

### Task 13: the normalisation prefactor moves to the top of `run_fdt`

**Why:** Spec §3.4, first bullet: `observable_noise_prefactor(cfg)` is called at step 8 of
`run_fdt` (`fdt_pipeline.py:128-134` per §1, "The cell's normalisation constant is checked after
both campaigns"), so a cell missing `n` or `beta` costs the whole run before it is refused. It moves
to the top and its result is carried to step 8. The six `FDTModelError`s §1 names
(`campaigns.py:103,127,133,136,141,145`) gain field keys, so `Refusal.field` is no longer `None` and
both front-end tables can name a control or a flag. Authorised by E5 ("checks refuse what breaks")
and the global constraint "refuse before the spend".

**Files:**
- Modify: `core/FDT/campaigns.py:103` (`_make_simulator`'s invalid-model raise)
- Modify: `core/FDT/campaigns.py:127,133,136,141,145` (`observable_noise_prefactor`'s five raises)
- Modify: `core/FDT/fdt_pipeline.py:71-74` (the head of `run_fdt`) and `:133-135` (step 8)
- Test: `tests/test_fdt_user.py`

**Interfaces:**
- Consumes: `core.refusals.FIELDS` — the keys `"cell"` (`refusals.py:93`,
  `Field("cell", "the cell file", None)`) and `"model"` (`refusals.py:107`,
  `Field("model", "the model", None)`). **Both already exist**; T13 adds no key, so T8's new keys
  are not in fact a prerequisite (see objections).
- Produces: `observable_noise_prefactor(cfg) -> float` unchanged in signature; every
  `FDTModelError` it and `_make_simulator` raise now carries `field="cell"` (the cell lacks a
  parameter) or `field="model"` (the model itself cannot be run). `run_fdt` binds a local
  `prefactor` before anything is simulated and reuses it at step 8; T17 puts that value in
  `data.h5`.

- [ ] **Step 1: Write the failing test**

In `tests/test_fdt_user.py`, after `test_observable_noise_prefactor_user_rejects_multiplicative_zero_negative`:

```python
def test_the_prefactor_is_refused_before_anything_is_simulated(tmp_path, monkeypatch):
    """Spec §3.4, first bullet. The per-model normalisation prefactor was resolved at step 8 of
    run_fdt -- AFTER both campaigns -- so a cell missing `n` or `beta` cost the entire run, hours of
    it, before the pipeline said the one thing it could have said in a second. It is resolved first
    now, and carried down to step 8.

    The assertion that carries the point is that NO CAMPAIGN RAN. A refusal merely moved a few lines
    up in the source but still sitting behind a campaign would pass a message-only test and buy the
    operator nothing. Both entry shapes are checked: the sanity branch spends a campaign of its own
    before the production one, so a check placed after the sanity gate would still be too late.

    The field key is the second half. An FDTModelError is a Refusal (tests/test_refusals.py's
    "every domain error is a refusal and carries a field"), but until now every FDT one carried
    field=None, so neither front-end table could name the control or the flag that answers it.
    """
    import pytest

    from core.FDT import fdt_pipeline
    from core.FDT.campaigns import FDTModelError

    spent = []

    def _never(*a, **kw):
        spent.append(a)
        raise RuntimeError("a campaign ran: the prefactor must be refused before anything is spent")

    monkeypatch.setattr(fdt_pipeline, "_out_dir", lambda: tmp_path)
    monkeypatch.setattr(fdt_pipeline, "run_campaign1_psd", _never)
    monkeypatch.setattr(fdt_pipeline, "run_all_sanity", _never)

    class _NoN:
        """A Nadrowski cell carrying k -- so the omega_0 estimate would have succeeded -- but no n,
        which is half of the Nadrowski prefactor n*beta."""
        model = "NADROWSKI"
        params_dict = {"k": (1.0, None), "beta": (14.1, None)}

    for skip_sanity in (True, False):
        with pytest.raises(FDTModelError) as e:
            fdt_pipeline.run_fdt(_NoN(), skip_sanity=skip_sanity, confirm_production=True)
        assert "'n'" in str(e.value), str(e.value)
        assert e.value.field == "cell", f"skip_sanity={skip_sanity}: field={e.value.field!r}"
    assert spent == [], "the prefactor refusal arrived only after a campaign had been spent"
```

- [ ] **Step 2: Run it and watch it fail**

Run: `pytest tests/test_fdt_user.py::test_the_prefactor_is_refused_before_anything_is_simulated -v`

Expected: FAIL with
`RuntimeError: a campaign ran: the prefactor must be refused before anything is spent`
— raised from the `run_campaign1_psd` stub at step 3 of `run_fdt`, so the `FDTModelError` the
`pytest.raises` block waits for never arrives.

- [ ] **Step 3: Give `observable_noise_prefactor`'s refusals their field keys**

In `core/FDT/campaigns.py`, find:

```python
    except KeyError as e:
        raise FDTModelError(f"The {cfg.model} cell is missing the FDT parameter {e}.") from e
```

Replace with:

```python
    except KeyError as e:
        raise FDTModelError(f"The {cfg.model} cell is missing the FDT parameter {e}.",
                            field="cell") from e
```

Find:

```python
    spec = registry.get(cfg.model)
    if spec is None or spec.compiled is None:
        raise FDTModelError(f"User model '{cfg.model}' has no compiled definition for FDT.")
    c = spec.compiled
    if {str(s) for s in c.diff_exprs[0].free_symbols} & set(c.var_names):
        raise FDTModelError(f"Observable '{c.var_names[0]}' has state-dependent (multiplicative) noise; "
                            "FDT supports additive-noise observables only.")
    try:
        param_vals = [pd[name][0] for name in c.param_names]
    except KeyError as e:
        raise FDTModelError(f"The {cfg.model} cell is missing parameter {e}.") from e
```

Replace with:

```python
    spec = registry.get(cfg.model)
    if spec is None or spec.compiled is None:
        raise FDTModelError(f"User model '{cfg.model}' has no compiled definition for FDT.",
                            field="model")
    c = spec.compiled
    if {str(s) for s in c.diff_exprs[0].free_symbols} & set(c.var_names):
        raise FDTModelError(f"Observable '{c.var_names[0]}' has state-dependent (multiplicative) noise; "
                            "FDT supports additive-noise observables only.", field="model")
    try:
        param_vals = [pd[name][0] for name in c.param_names]
    except KeyError as e:
        raise FDTModelError(f"The {cfg.model} cell is missing parameter {e}.", field="cell") from e
```

Find:

```python
    if not math.isfinite(D0) or D0 <= 0.0:
        raise FDTModelError(f"Observable '{c.var_names[0]}' has non-positive/zero noise (D0={D0}); "
                            "FDT requires a stochastic observable.")
```

Replace with:

```python
    if not math.isfinite(D0) or D0 <= 0.0:
        raise FDTModelError(f"Observable '{c.var_names[0]}' has non-positive/zero noise (D0={D0}); "
                            "FDT requires a stochastic observable.", field="model")
```

- [ ] **Step 4: Give `_make_simulator`'s refusal its field key**

Spec §1 names `campaigns.py:103` among the field-less refusals. In `core/FDT/campaigns.py`, find:

```python
    cls = VALID_SIMS.get(cfg.model.lower())
    if cls is None:
        raise FDTModelError(f"Invalid model for FDT: {cfg.model}. Valid: {list(VALID_SIMS)}.")
```

Replace with:

```python
    cls = VALID_SIMS.get(cfg.model.lower())
    if cls is None:
        raise FDTModelError(f"Invalid model for FDT: {cfg.model}. Valid: {list(VALID_SIMS)}.",
                            field="model")
```

- [ ] **Step 5: Move the prefactor to the top of `run_fdt`**

In `core/FDT/fdt_pipeline.py`, find:

```python
    # 1. Model-specific natural-frequency starting estimate; the production omega_0
    #    is refined from the Campaign 1 PSD peak below.
    cfg.omega_0, omega_0_desc = _estimate_omega_0(cfg)
```

Replace with:

```python
    # The per-model normalisation prefactor FIRST, before anything is simulated. It reads the cell's
    # parameters and nothing else, and it used to sit at step 8 -- so a cell missing `n` or `beta`
    # was refused only after BOTH campaigns had been spent (spec §3.4). Carried to step 8 below.
    prefactor = observable_noise_prefactor(cfg)

    # 1. Model-specific natural-frequency starting estimate; the production omega_0
    #    is refined from the Campaign 1 PSD peak below.
    cfg.omega_0, omega_0_desc = _estimate_omega_0(cfg)
```

- [ ] **Step 6: Use the carried value at step 8**

In `core/FDT/fdt_pipeline.py`, find:

```python
    # 8. T_eff/T -- the normalization prefactor is per-model (Nadrowski n*beta, else 1/D_x).
    prefactor = observable_noise_prefactor(cfg)
    ratio = eff_temp_ratio(G_at_omegas, chis.imag, omegas.to(torch.float64), prefactor)
```

Replace with:

```python
    # 8. T_eff/T -- the per-model normalization prefactor (Nadrowski n*beta, else 1/D_x) was
    #    resolved at the top of this function, before anything was spent.
    ratio = eff_temp_ratio(G_at_omegas, chis.imag, omegas.to(torch.float64), prefactor)
```

- [ ] **Step 7: Fix the existing test the move reds**

`test_the_fdt_messages_are_records_with_their_own_levels` drives `run_fdt` with a stub cfg that
carries only a model name; the prefactor now reads `cfg.params_dict` before the first log record, so
that stub raises `AttributeError`. In `tests/test_fdt_user.py`, find:

```python
    # ── the FDT run: a failed sanity verdict is a WARNING, without the hand-typed word ─────────────
    class _Hopf:
        model = "HOPF"
```

Replace with:

```python
    # ── the FDT run: a failed sanity verdict is a WARNING, without the hand-typed word ─────────────
    class _Hopf:
        model = "HOPF"
        # run_fdt resolves the normalisation prefactor before its first record (spec §3.4), and the
        # HOPF prefactor is 2/sigma_x^2 -- so this stub has to carry the one parameter it reads.
        params_dict = {"sigma_x": (0.1, None)}
```

- [ ] **Step 8: Run the tests and watch them pass**

Run: `pytest tests/test_fdt_user.py -v`
Expected: PASS (7 tests: the 5 existing, the changed logging test, the new one). The logging test's
exact-equality record list is unchanged, because the prefactor call logs nothing — deliberately, so
that assertion stays a pin on the pipeline's records rather than on this task.

- [ ] **Step 9: Run the refusal-key scan, which walks `core/`**

Run: `pytest tests/test_refusals.py -v`
Expected: PASS. `_field_key_literals` collects every `field="…"` literal under `CODE_ROOTS` and
asserts each is in `FIELDS`; `"cell"` and `"model"` are registered (`refusals.py:93,107`), so the
four new literals are already covered.

- [ ] **Step 10: Commit**

```bash
git add core/FDT/campaigns.py core/FDT/fdt_pipeline.py tests/test_fdt_user.py
git commit -m "fdt: resolve the normalisation prefactor before the campaigns"
```

#### Amendments (binding — these supersede the text above)

Rulings already folded into this task, no further change: **P18** (no dependency on T8: `cell` and `model` are registered today), **P24** (the `_Hopf` stub gains `params_dict`; the prefactor call logs nothing), **P25** (every `FDTModelError` raise site, `_make_simulator`'s included, gets a key). **P26** applies as a constraint: none of the six messages may name a flag, box or tab. They do not; keep it so.

**A. T12 has landed, and `run_fdt` now reads two config fields BEFORE your prefactor.** T12 inserted this line directly above the `# 1. Model-specific ...` block your Step 5 anchors on:

```python
    notices = warn_thin_settings(cfg)
```

It reads `cfg.n_freqs` and `cfg.ensemble_M`. Your Step 5 anchor is still found, because T12's replacement ends with those three lines. Put the prefactor between T12's `# 0.` block and `# 1.` exactly as Step 5 says. Do not move T12's line. Every stub config you hand to `run_fdt` must now carry both fields, at or above T12's thresholds (`FDT_THIN_N_FREQS = 2`, `FDT_THIN_ENSEMBLE_M = 8`), so that no `PreflightWarning` is raised.

*Step 1*: in the new test, the `_NoN` stub becomes (docstring unchanged):

```python
    class _NoN:
        """A Nadrowski cell carrying k -- so the omega_0 estimate would have succeeded -- but no n,
        which is half of the Nadrowski prefactor n*beta."""
        model = "NADROWSKI"
        params_dict = {"k": (1.0, None), "beta": (14.1, None)}
        # read by the thin-setting check, which runs before the prefactor; at or above its
        # thresholds, so that check says nothing
        n_freqs, ensemble_M = 60, 256
```

Without these fields, Step 2 fails with `AttributeError: '_NoN' object has no attribute 'n_freqs'` instead of the predicted `RuntimeError: a campaign ran: ...`, and Step 8 fails the same way.

*Step 7*: the `_Hopf` replacement becomes:

```python
    # ── the FDT run: a failed sanity verdict is a WARNING, without the hand-typed word ─────────────
    class _Hopf:
        model = "HOPF"
        # run_fdt resolves the normalisation prefactor before its first record (spec §3.4), and the
        # HOPF prefactor is 2/sigma_x^2 -- so this stub has to carry the one parameter it reads.
        params_dict = {"sigma_x": (0.1, None)}
        # the thin-setting check reads these two first; at or above its thresholds, so no
        # PreflightWarning joins the records this test pins by exact equality
        n_freqs, ensemble_M = 60, 256
```

If T12 already gave `_Hopf` `n_freqs`/`ensemble_M`, keep T12's line and add only `params_dict` and its two comment lines. If T12 rewrote the `model = "HOPF"` line so that the Step 7 anchor is not found, add `params_dict` inside the class body anyway and say so in the task report.

**B. Step 8's count is stale.** "PASS (7 tests: the 5 existing, the changed logging test, the new one)" was counted before T10–T12. T10 adds two tests, T11 one and T12 one, so after this task the file holds **11** tests. Read the count off the run; every test must pass.

**C. Step 9's count is stale.** Steps 3 and 4 add **six** `field="…"` literals (five in `observable_noise_prefactor`, one in `_make_simulator`), not four. Both keys are registered, so `tests/test_refusals.py` still passes.

#### Controller rulings (pre-flight, binding — these supersede the amendments above and the task text)

- **F10** — insert `prefactor = observable_noise_prefactor(cfg)` ABOVE Task 12's `# 0. The settings too thin to trust` block, not below it: a refused run must not first print a quick-look warning. Your anchor is that comment line. Your `_NoN` stub then needs no `n_freqs`/`ensemble_M`.
- **F34** — no blank-box test belongs in this task; the plan's Review Focus item 3 named the wrong tasks and is corrected to Tasks 10 and 11.

---

### Task 14: the off-grid range test, and the callers' honest band

**Why:** Spec §3.5 and E9. `_interp_log` promises "NaN outside the grid", but the Welch grid's first
entry is exactly `0.0` (`spectral.py:56-57`, `omegas = 2.0 * math.pi * freqs_hz` from `rfftfreq`),
the helper clamps it to `1e-30`, and the in-range test `log_x_new >= log_x_old[0]` is then true for
**every** positive frequency. The low end never blanks: it returns a value blended off the
zero-frequency bin, and the callers' count of excluded points reports zero. The test's first job is
to demonstrate that empirically.

**Files:**
- Modify: `core/FDT/sanity.py:28-48` (`_interp_log`) — a new `_resolved_span` goes directly above it
- Modify: `core/FDT/sanity.py:98-102` (`check_passive_baseline`'s warning) and `:166-171`
  (`check_high_freq_fdt`'s warning)
- Test: `tests/test_fdt_user.py`

**Interfaces:**
- Consumes: nothing from an earlier task (the dependency graph marks T14 independent).
- Produces: `core.FDT.sanity._resolved_span(omegas: torch.Tensor) -> tuple[float, float]` — the
  first strictly positive frequency of a spectrum's grid and its last, as plain floats; `(inf,
  -inf)` for a grid with no positive bin, which puts every probe out of range without raising. T15
  and T16 import it into `core/FDT/fdt_pipeline.py`; T17 records its two values under `body.grid`.
  `_interp_log`'s signature is unchanged.

- [ ] **Step 1: Write the two failing `_interp_log` tests**

In `tests/test_fdt_user.py`, after the Task 13 test:

```python
def test_a_probe_below_the_first_real_bin_comes_back_blank_not_blended():
    """Spec §3.5, E9 -- and the demonstration of the defect, which is the whole point of the fix.

    _interp_log's docstring promises NaN outside the grid, and PRISM_HANDOFF names the consequence of
    not having it: "widen freq_bounds past the PSD resolution and T_eff/T acquires a smooth,
    plausible-looking, entirely fabricated tail". A Welch grid starts at exactly 0.0, the helper
    clamps that bin to 1e-30 before taking logarithms, and the in-range test compares against the
    CLAMPED bin -- so every positive frequency passed it and the low end returned a value blended off
    the zero-frequency bin. Here the zero bin holds 100 and the first real bin holds 1: a probe at
    half the first real bin used to come back at about 1.993, a number with no measurement behind it.
    It is a blank now, and the assertion message prints what came back instead so the defect is
    legible in the failure rather than only in this docstring."""
    import math

    from core.FDT.sanity import _interp_log

    x_old = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)   # a Welch grid: the DC bin is 0.0
    y_old = torch.tensor([100.0, 1.0, 2.0], dtype=torch.float64)
    got = float(_interp_log(torch.tensor([0.5], dtype=torch.float64), x_old, y_old)[0])
    assert math.isnan(got), (f"a probe at 0.5, below the spectrum's first real bin at 1.0, came back "
                             f"blended off the zero bin instead of blank: {got:.4f}")


def test_every_probe_outside_the_resolved_span_is_blank_and_the_covered_ones_are_exact():
    """Spec §3.5. The count of excluded points is what reaches the record as body.offgrid (spec
    §2.3) and what the sweep's per-point summary reports, so it has to be honest at BOTH ends: the
    upper end already blanked, the lower end never did. The covered points are asserted too, because
    a range test that blanked too much would also make the count "honest" and would silently throw
    away measured frequencies -- 2**0.5 sits at exactly half a log-decade between the 1.0 and 2.0
    bins, so its interpolated value is exactly halfway between their values."""
    from core.FDT.sanity import _interp_log

    x_old = torch.tensor([0.0, 1.0, 2.0, 4.0], dtype=torch.float64)
    y_old = torch.tensor([100.0, 1.0, 2.0, 4.0], dtype=torch.float64)
    probes = torch.tensor([0.5, 1.0, 2.0 ** 0.5, 2.0, 4.0, 8.0], dtype=torch.float64)

    got = _interp_log(probes, x_old, y_old)
    blanks = torch.isnan(got)
    assert int(blanks.sum()) == 2, (f"expected 2 blanks -- 0.5 below the first real bin and 8.0 above "
                                    f"the last -- got {int(blanks.sum())}: {got.tolist()}")
    assert bool(blanks[0]) and bool(blanks[-1]), got.tolist()
    assert torch.allclose(got[1:5], torch.tensor([1.0, 1.5, 2.0, 4.0], dtype=torch.float64),
                          atol=1e-12), got.tolist()
```

- [ ] **Step 2: Run them and watch them fail**

Run: `pytest tests/test_fdt_user.py -k "blank" -v`

Expected: both FAIL.
- `test_a_probe_below_the_first_real_bin_comes_back_blank_not_blended`:
  `AssertionError: a probe at 0.5, below the spectrum's first real bin at 1.0, came back blended off the zero bin instead of blank: 1.9934`
- `test_every_probe_outside_the_resolved_span_is_blank_and_the_covered_ones_are_exact`:
  `AssertionError: expected 2 blanks -- 0.5 below the first real bin and 8.0 above the last -- got 1: [1.9933992..., 1.0, 1.5, 2.0, 4.0, nan]`

- [ ] **Step 3: Add `_resolved_span`**

In `core/FDT/sanity.py`, find:

```python
def _interp_log(x_new: torch.Tensor, x_old: torch.Tensor, y_old: torch.Tensor) -> torch.Tensor:
```

Insert directly above it:

```python
def _resolved_span(omegas: torch.Tensor) -> tuple[float, float]:
    """The band a spectrum on ``omegas`` actually resolves: its first STRICTLY POSITIVE frequency to
    its last.

    A Welch grid's first entry is exactly 0.0 (``spectral.psd_welch``: ``rfftfreq``), and the zero
    bin is the segment mean -- it is not a measurement at any positive frequency, so nothing may
    interpolate from it. The lower end is therefore the SECOND entry of a Welch grid, and it is the
    resolution the spontaneous recording bought: 2*pi/(nperseg*dt).

    A grid with no positive bin resolves nothing, and returns an EMPTY band (+inf, -inf) rather than
    raising -- every probe is then out of range, which is the truthful answer.
    """
    pos = omegas[omegas > 0]
    if pos.numel() == 0:
        return float("inf"), float("-inf")
    return float(pos[0]), float(pos[-1])


def _interp_log(x_new: torch.Tensor, x_old: torch.Tensor, y_old: torch.Tensor) -> torch.Tensor:
```

- [ ] **Step 4: Compare against the resolved span, not the clamped zero bin**

In `core/FDT/sanity.py`, find:

```python
    """1-D linear INTERPOLATION in log-x, NaN outside the grid. Aligns PSD onto the chi grid.

    The clamp below bounds the INDEX, not the VALUE. Left as it was, a frequency outside the Welch
    grid's span got ``frac < 0`` or ``frac > 1`` -- a linear EXTRAPOLATION off the two edge bins,
    returned silently as if it were data. ``eff_temp_ratio`` then divides by it, so widening
    ``freq_bounds`` past the PSD resolution grew a smooth, plausible-looking, entirely fabricated
    T_eff/T tail. Out-of-range now yields NaN and the callers report how many points that cost.
    """
    x_new = x_new.to(torch.float64)
    x_old = x_old.to(torch.float64).clamp(min=1e-30)
    y_old = y_old.to(torch.float64)
    log_x_old = torch.log(x_old)
    log_x_new = torch.log(x_new.clamp(min=1e-30))
    idx = torch.searchsorted(log_x_old, log_x_new).clamp(1, len(log_x_old) - 1)
    x0, x1 = log_x_old[idx - 1], log_x_old[idx]
    y0, y1 = y_old[idx - 1], y_old[idx]
    frac = (log_x_new - x0) / (x1 - x0)
    out = y0 + frac * (y1 - y0)
    in_range = (log_x_new >= log_x_old[0]) & (log_x_new <= log_x_old[-1])
    return torch.where(in_range, out, torch.full_like(out, float("nan")))
```

Replace with:

```python
    """1-D linear INTERPOLATION in log-x, NaN outside the RESOLVED span. Aligns PSD onto the chi grid.

    The clamp below bounds the INDEX, not the VALUE. Left as it was, a frequency outside the Welch
    grid's span got ``frac < 0`` or ``frac > 1`` -- a linear EXTRAPOLATION off the two edge bins,
    returned silently as if it were data. ``eff_temp_ratio`` then divides by it, so widening
    ``freq_bounds`` past the PSD resolution grew a smooth, plausible-looking, entirely fabricated
    T_eff/T tail. Out-of-range yields NaN and the callers report how many points that cost.

    The in-range test is against ``_resolved_span``, NOT against ``x_old[0]``, and that is the whole
    of the piece-5 fix (spec §3.5, E9). A Welch grid's first entry is exactly 0.0, the clamp turns it
    into 1e-30, and ``log_x_new >= log_x_old[0]`` was therefore true for EVERY positive frequency --
    so the promise above held at the top of the grid and was false at the bottom, where a probe
    returned a value blended off the zero-frequency bin and the callers counted no exclusions at all.
    """
    x_new = x_new.to(torch.float64)
    x_old = x_old.to(torch.float64)
    y_old = y_old.to(torch.float64)
    lo, hi = _resolved_span(x_old)
    log_x_old = torch.log(x_old.clamp(min=1e-30))
    log_x_new = torch.log(x_new.clamp(min=1e-30))
    idx = torch.searchsorted(log_x_old, log_x_new).clamp(1, len(log_x_old) - 1)
    x0, x1 = log_x_old[idx - 1], log_x_old[idx]
    y0, y1 = y_old[idx - 1], y_old[idx]
    frac = (log_x_new - x0) / (x1 - x0)
    out = y0 + frac * (y1 - y0)
    in_range = (x_new >= lo) & (x_new <= hi)
    return torch.where(in_range, out, torch.full_like(out, float("nan")))
```

- [ ] **Step 5: Run the two tests and watch them pass**

Run: `pytest tests/test_fdt_user.py -k "blank" -v`
Expected: PASS (2 passed).

- [ ] **Step 6: Write the failing test for the callers' band**

The two sanity checks print the excluded band as `freqs_psd.min()..freqs_psd.max()`, which is
`0..<max>` for every Welch grid — a sentence that tells the reader a probe at 0.5 fell outside
`0..2`. In `tests/test_fdt_user.py`, after the two tests above:

```python
def test_the_sanity_checks_name_the_band_the_spectrum_actually_resolves(monkeypatch):
    """Spec §3.5: "the callers' count of excluded points becomes honest". The count is only half of
    it. Both sanity checks print the excluded band as freqs_psd.min()..freqs_psd.max(), and a Welch
    grid's min is exactly 0.0 -- so the sentence read "fall outside the Welch PSD grid (0..2)" while
    excluding a probe at 0.5, which is inside 0..2. The band the reader is asked to narrow towards
    has to be the band the spectrum resolves.

    check_high_freq_fdt is the one to pin: it reads the TOP three frequencies of the probe grid,
    precisely where the PSD runs out, so it was the check most exposed to the old extrapolation. Its
    campaign seams are stubbed, so nothing is simulated."""
    import pytest

    from core.FDT import sanity

    class _Cfg:
        ensemble_M, psd_T_obs_nd, omega_0, freq_bounds = 8, 100.0, 1.0, (0.1, 30.0)

        class hw:
            device = torch.device("cpu")
            dtype = torch.float64

        def with_overrides(self, **kw):
            return self

    # A spectrum resolving 0.5..2.0; the top three of the 7-point probe grid are 4.48, 11.59 and 30.
    freqs_psd = torch.tensor([0.0, 0.5, 1.0, 2.0], dtype=torch.float64)
    G = torch.tensor([9.0, 1.0, 2.0, 1.0], dtype=torch.float64)
    monkeypatch.setattr(sanity, "run_campaign1_psd", lambda c: (freqs_psd, G))
    monkeypatch.setattr(sanity, "run_campaign2_chi",
                        lambda c, om, **kw: torch.full((len(om),), 1 + 1j, dtype=torch.complex128))
    monkeypatch.setattr(sanity, "observable_noise_prefactor", lambda c: 1.0)

    with pytest.warns(UserWarning, match="3/3") as rec:
        passed, metrics = sanity.check_high_freq_fdt(_Cfg())

    msg = str(rec[0].message)
    assert "(0.5..2)" in msg, msg
    assert "(0..2)" not in msg, f"the warning named the zero bin as the band's lower end: {msg}"
    assert passed is False and metrics["n_off_grid"] == 3, metrics
```

- [ ] **Step 7: Run it and watch it fail**

Run: `pytest tests/test_fdt_user.py -k "resolves" -v`

Expected: FAIL with
`AssertionError: check_high_freq_fdt: 3/3 of the top probe frequencies fall outside the Welch PSD grid (0..2) and were EXCLUDED. ...`
(the first assertion, `"(0.5..2)" in msg`, fails and prints the message it got).

- [ ] **Step 8: Name the resolved band in both warnings**

In `core/FDT/sanity.py`, find:

```python
    devs = np.abs(ratio - 1.0)
    n_off = int(np.isnan(devs).sum())
    if n_off:
        warnings.warn(
            f"check_passive_baseline: {n_off}/{devs.size} probe frequencies fall outside the Welch "
            f"PSD grid ({float(freqs_psd.min()):g}..{float(freqs_psd.max()):g}) and were EXCLUDED. "
```

Replace with:

```python
    devs = np.abs(ratio - 1.0)
    n_off = int(np.isnan(devs).sum())
    if n_off:
        lo_res, hi_res = _resolved_span(freqs_psd)
        warnings.warn(
            f"check_passive_baseline: {n_off}/{devs.size} probe frequencies fall outside the Welch "
            f"PSD grid ({lo_res:g}..{hi_res:g}) and were EXCLUDED. "
```

Find:

```python
    devs = np.abs(ratio - 1.0)
    n_off = int(np.isnan(devs).sum())
    if n_off:
        warnings.warn(
            f"check_high_freq_fdt: {n_off}/{devs.size} of the top probe frequencies fall outside the "
            f"Welch PSD grid ({float(freqs_psd.min()):g}..{float(freqs_psd.max()):g}) and were "
            f"EXCLUDED. This check previously extrapolated there, so a past pass at these "
```

Replace with:

```python
    devs = np.abs(ratio - 1.0)
    n_off = int(np.isnan(devs).sum())
    if n_off:
        lo_res, hi_res = _resolved_span(freqs_psd)
        warnings.warn(
            f"check_high_freq_fdt: {n_off}/{devs.size} of the top probe frequencies fall outside the "
            f"Welch PSD grid ({lo_res:g}..{hi_res:g}) and were "
            f"EXCLUDED. This check previously extrapolated there, so a past pass at these "
```

- [ ] **Step 9: Run the whole file and watch it pass**

Run: `pytest tests/test_fdt_user.py -v`
Expected: PASS (10 tests).

- [ ] **Step 10: Commit**

```bash
git add core/FDT/sanity.py tests/test_fdt_user.py
git commit -m "fdt: blank probes below the spectrum's first real bin"
```

#### Amendments (binding — these supersede the text above)

Rulings already satisfied by the body, no change: **P7** (`_resolved_span` as drafted is the contract's function), **P57** (`body.offgrid` counts the Campaign-2 probe grid; this task makes that count honest at the low end).

**A. P23: do not touch `check_passive_baseline`'s bare `ValueError`.** Once this task lands, the block below in `core/FDT/sanity.py` can fire from the LOW end of the band for the first time:

```python
    if covered.size == 0:
        raise ValueError(
            "check_passive_baseline: every probe frequency lies outside the PSD grid, so the FDT "
            "ratio is unmeasurable here. Narrow cfg.freq_bounds or lengthen the passive run.")
```

Leave it exactly as it is. P23 assigns its conversion to T33, and only if T33 goes red on it.

**B. Pin the contract's empty-grid case (P7).** At the end of `test_every_probe_outside_the_resolved_span_is_blank_and_the_covered_ones_are_exact` (Step 1), append:

```python
    from core.FDT.sanity import _resolved_span
    assert _resolved_span(x_old) == (1.0, 4.0), "the zero bin is not the band's lower end"
    assert _resolved_span(torch.tensor([0.0], dtype=torch.float64)) == (float("inf"), float("-inf")), \
        "a grid with no positive bin resolves nothing: every probe is then out of range (P7)"
```

Step 2's predicted failure for this test is unchanged, because its first assertion fails before these lines run. Step 5 still expects `2 passed`.

**C. Add a step between Step 5 and Step 6: run the older `_interp_log` pin too.**

Run: `pytest tests/test_user_sbi.py::test_interp_log_returns_nan_off_the_psd_grid -v`
Expected: PASS. Its grid is `torch.logspace(0, 2, 32)` and has no zero bin, so `_resolved_span` returns `(1.0, 100.0)` and the probes 0.1 and 500 are still the only blanks. Run only that node id; the file also holds a `slow` test.

**D. Step 9's count is stale.** "PASS (10 tests)" was counted before T10–T12 added four tests to this file. Expect **14**, and read the count off the run.

**E. An Interfaces correction, with no code change.** Drop "T17 records its two values under `body.grid`". T17's grid block (spec §2.3) has no resolved-span key. `lo_res`/`hi_res` are used only in the two sanity warnings here and in the refusal and warning text of T15 and T16.

#### Controller rulings (pre-flight, binding — these supersede the amendments above and the task text)

- **F35/F39** — delete the Interfaces sentence saying Task 17 records the resolved span under `body.grid`. It does not, and spec §2.3's `grid` list does not include it. `_resolved_span` is consumed by Tasks 15-17 in code only.

---

### Task 15: the band refusal, before the driven campaign

**Why:** Spec §3.4, second bullet, and E9: "once the grid is built — the first moment it is knowable
— a band reaching below that resolution refuses before the driven campaign is spent". After T14 such
a band no longer fabricates a tail; it produces blanks. Refusing is better than reporting a curve
that is blank where the operator asked to look, and the driven campaign is the expensive half of the
run. `freq_bounds` and `psd_T_obs_nd` are settings no front end exposes, so the refusal carries
`field=None` and gets no table entry (spec §1.2, "Five settings are checked but are not front-end
knobs").

**Files:**
- Modify: `core/FDT/fdt_pipeline.py:22` (the `sanity` import) and the imports block
- Modify: `core/FDT/fdt_pipeline.py:120-128` (between step 5 and step 6 of `run_fdt`)
- Test: `tests/test_fdt_user.py`

**Interfaces:**
- Consumes: `core.FDT.sanity._resolved_span(omegas) -> tuple[float, float]` (T14);
  `observable_noise_prefactor` already resolved at the top of `run_fdt` (T13).
- Produces: in `run_fdt`, the locals `lo_res, _hi_res = _resolved_span(freqs_psd)` bound after the
  grid is built and before Campaign 2, and a `core.refusals.Refusal` with `field=None` when
  `float(omegas[0]) < lo_res`. T16 renames `_hi_res` to `hi_res` and uses both; T17 records both
  under `body.grid`.

- [ ] **Step 1: Write the failing test**

In `tests/test_fdt_user.py`, after the Task 14 tests:

```python
def test_a_band_below_the_spectrums_resolution_refuses_before_the_driven_campaign(tmp_path, monkeypatch):
    """Spec §3.4, second bullet, and E9. The spontaneous recording's length sets the lowest frequency
    the spectrum resolves; the probe grid is built around a resonance the spontaneous campaign found,
    so the two are only comparable once Campaign 1 is done. That is the first moment the condition is
    knowable, and it sits directly before the expensive half of the run -- which is the only reason
    the check is worth anything. Before T14 this band silently produced a fabricated low-frequency
    tail; after T14 it produces blanks; refusing says so before the drive is spent.

    The assertion that carries the point is that Campaign 2 NEVER RAN. The message is checked too:
    it names the band that was asked for, the band that exists, and the spontaneous duration that
    sets it -- and it names no box, tab or flag, because neither setting is exposed by either front
    end (spec §1.2), so its field is None and neither table can offer a fix sentence."""
    import pytest

    from core.FDT import fdt_pipeline
    from core.refusals import Refusal

    driven = []

    class _Cfg:
        """The subset of FDTConfig run_fdt reads, at a size nothing simulates."""
        model = "HOPF"
        params_dict = {"sigma_x": (0.1, None)}
        n_freqs, freq_bounds = 5, (0.1, 30.0)
        burn_in_nd, dt_nd, psd_T_obs_nd, omega_0 = 0.0, 0.01, 200.0, 1.0

        class hw:
            device = torch.device("cpu")
            dtype = torch.float64

    # A spectrum resolving 1.0..3.0 whose peak is at 2.0, so the grid runs 0.2 .. 60 -- its lowest
    # probe sits a factor of five below the lowest frequency the recording resolves.
    freqs_psd = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64)
    G = torch.tensor([9.0, 1.0, 5.0, 1.0], dtype=torch.float64)
    monkeypatch.setattr(fdt_pipeline, "_out_dir", lambda: tmp_path)
    monkeypatch.setattr(fdt_pipeline, "run_campaign1_psd",
                        lambda cfg, return_trajectory=False: (
                            freqs_psd, G, torch.arange(4, dtype=torch.float64),
                            torch.zeros(4, dtype=torch.float64)))
    monkeypatch.setattr(fdt_pipeline, "plot_spontaneous_trajectory", lambda *a, **kw: None)

    def _no_drive(*a, **kw):
        driven.append(a)
        raise AssertionError("the driven campaign ran: the band refusal must come first")

    monkeypatch.setattr(fdt_pipeline, "run_campaign2_chi", _no_drive)

    with pytest.raises(Refusal) as e:
        fdt_pipeline.run_fdt(_Cfg(), skip_sanity=True, confirm_production=True)
    msg = str(e.value)
    assert "0.2" in msg and "psd_T_obs_nd = 200" in msg, msg
    assert e.value.field is None, e.value.field
    assert driven == [], "Campaign 2 was entered before the band was checked"
```

- [ ] **Step 2: Run it and watch it fail**

Run: `pytest "tests/test_fdt_user.py::test_a_band_below_the_spectrums_resolution_refuses_before_the_driven_campaign" -v`

Expected: FAIL with
`AssertionError: the driven campaign ran: the band refusal must come first`
— raised inside the `pytest.raises(Refusal)` block by the Campaign-2 stub, which is not a `Refusal`,
so it propagates and fails the test.

- [ ] **Step 3: Import the two names the check needs**

In `core/FDT/fdt_pipeline.py`, find:

```python
from core.FDT.spectral import gen_freqs_log, eff_temp_ratio, find_spectral_peak
from core.FDT.sanity import run_all_sanity, _interp_log
```

Replace with:

```python
from core.FDT.spectral import gen_freqs_log, eff_temp_ratio, find_spectral_peak
from core.FDT.sanity import run_all_sanity, _interp_log, _resolved_span
from core.refusals import Refusal
```

- [ ] **Step 4: Refuse the band between the grid and the drive**

In `core/FDT/fdt_pipeline.py`, find:

```python
    omegas = gen_freqs_log(cfg.omega_0, cfg.n_freqs, cfg.freq_bounds,
                            cfg.hw.device, cfg.hw.dtype)

    # 6. Campaign 2: forced chi via lock-in
```

Replace with:

```python
    omegas = gen_freqs_log(cfg.omega_0, cfg.n_freqs, cfg.freq_bounds,
                            cfg.hw.device, cfg.hw.dtype)

    # The band, checked the first moment it is knowable and BEFORE the driven campaign -- the
    # expensive half of the run (spec §3.4). The grid's lowest frequency is known only now, because
    # it is built around the resonance Campaign 1 found; the spectrum's lowest RESOLVED frequency is
    # its first non-zero bin, which the spontaneous duration sets. A grid reaching below it comes
    # back blank there (spec §3.5), and used to come back with a fabricated tail instead.
    # field=None: neither the band nor the spontaneous duration is exposed by a front end (spec
    # §1.2), so no table can offer a fix sentence and none pretends to.
    lo_res, _hi_res = _resolved_span(freqs_psd)
    omega_lo = float(omegas[0])
    if omega_lo < lo_res:
        raise Refusal(
            f"The frequency band reaches below what the spontaneous spectrum resolves: the lowest "
            f"probe frequency is {omega_lo:g} (ND) but the lowest frequency the spectrum resolves "
            f"is {lo_res:g} (ND). Raise freq_bounds' lower multiplier above "
            f"{lo_res / cfg.omega_0:g}, or lengthen the spontaneous recording "
            f"(psd_T_obs_nd = {cfg.psd_T_obs_nd:g}), which is what sets the resolution.")

    # 6. Campaign 2: forced chi via lock-in
```

- [ ] **Step 5: Run the test and watch it pass**

Run: `pytest "tests/test_fdt_user.py::test_a_band_below_the_spectrums_resolution_refuses_before_the_driven_campaign" -v`
Expected: PASS.

- [ ] **Step 6: Run the whole file**

Run: `pytest tests/test_fdt_user.py -v`
Expected: PASS (11 tests). In particular the Task 13 test still passes: its cfg never reaches the
grid, because the prefactor refuses first.

- [ ] **Step 7: Commit**

```bash
git add core/FDT/fdt_pipeline.py tests/test_fdt_user.py
git commit -m "fdt: refuse a band below the spectrum's resolution before the drive"
```

#### Amendments (binding — these supersede the text above)

**A. P75: the band refusal carries `field="freq_bounds"`, not `field=None`.** `freq_bounds` is registered in `core.refusals.FIELDS`, and BOTH front-end tables map it to `None` (P2). `fix_sentence` therefore returns `""`, and the refusal names the setting and offers no fix, which is the intended behaviour.

*Why paragraph*: "so the refusal carries `field=None` and gets no table entry (spec §1.2, ...)" is superseded. It now reads: the refusal carries `field="freq_bounds"`, a registered key that both tables map to `None`.

*Interfaces, Produces*: "a `core.refusals.Refusal` with `field=None`" becomes "a `core.refusals.Refusal` with `field="freq_bounds"`". Also drop "T17 records both under `body.grid`": T17's grid block has no resolved-span key, so `lo_res`/`hi_res` feed only this refusal and T16's text.

*Step 1*: in the test's docstring, replace

```
    it names the band that was asked for, the band that exists, and the spontaneous duration that
    sets it -- and it names no box, tab or flag, because neither setting is exposed by either front
    end (spec §1.2), so its field is None and neither table can offer a fix sentence."""
```

with

```
    it names the band that was asked for, the band that exists, and the spontaneous duration that
    sets it -- and it names no box, tab or flag. Its field is "freq_bounds" (P75): the key is
    registered, and both front-end tables map it to None because no control and no flag exposes the
    band, so neither table offers a fix sentence and neither pretends to."""
```

Then replace `    assert e.value.field is None, e.value.field` with:

```python
    assert e.value.field == "freq_bounds", e.value.field
    from core.tool.fields import fix_sentence
    assert fix_sentence("freq_bounds") == "", "no flag exposes the band, so no fix is offered (P2, P75)"
```

*Step 4*: the replacement block becomes the following. Keep the first comment line and the `lo_res, _hi_res = ...` line **verbatim**, because T16 anchors on both.

```python
    omegas = gen_freqs_log(cfg.omega_0, cfg.n_freqs, cfg.freq_bounds,
                            cfg.hw.device, cfg.hw.dtype)

    # The band, checked the first moment it is knowable and BEFORE the driven campaign -- the
    # expensive half of the run (spec §3.4). The grid's lowest frequency is known only now, because
    # it is built around the resonance Campaign 1 found; the spectrum's lowest RESOLVED frequency is
    # its first non-zero bin, which the spontaneous duration sets. A grid reaching below it comes
    # back blank there (spec §3.5), and used to come back with a fabricated tail instead.
    # field="freq_bounds" (P2, P75): the key is registered and BOTH front-end tables map it to None,
    # because neither the band nor the spontaneous duration is exposed by a front end -- so no table
    # offers a fix sentence and none pretends to.
    lo_res, _hi_res = _resolved_span(freqs_psd)
    omega_lo = float(omegas[0])
    if omega_lo < lo_res:
        raise Refusal(
            f"The frequency band reaches below what the spontaneous spectrum resolves: the lowest "
            f"probe frequency is {omega_lo:g} (ND) but the lowest frequency the spectrum resolves "
            f"is {lo_res:g} (ND). Raise freq_bounds' lower multiplier above "
            f"{lo_res / cfg.omega_0:g}, or lengthen the spontaneous recording "
            f"(psd_T_obs_nd = {cfg.psd_T_obs_nd:g}), which is what sets the resolution.",
            field="freq_bounds")

    # 6. Campaign 2: forced chi via lock-in
```

**B. T8 must have landed WITH P75 before this task.** T8, as corrected by P75, registers `freq_bounds` (`FIELDS`, `BASE_KEYS`, `CONTROL["freq_bounds"] = None`, `FLAG["freq_bounds"] = None`). The task-order graph does not show this dependency. Before Step 1, confirm that `core/refusals.py` holds `Field("freq_bounds", ...)`. If it does not, stop and report it: a `field="freq_bounds"` literal would red `tests/test_refusals.py::test_every_field_key_has_a_flag_and_every_key_literal_under_core_is_registered`.

**C. T12 has landed.** `run_fdt` now calls `warn_thin_settings(cfg)` before anything else, and it reads `cfg.ensemble_M`. In Step 1's `_Cfg`, directly under `n_freqs, freq_bounds = 5, (0.1, 30.0)`, add:

```python
        ensemble_M = 8        # read first by the thin-setting check; at its threshold, so it says nothing
```

Without it, Step 2 fails with `AttributeError: '_Cfg' object has no attribute 'ensemble_M'` instead of the predicted `AssertionError: the driven campaign ran: ...`.

**D. Step 6.** The count is **15** (not 11) once T10–T12 have added their four tests; read it off the run. Also run `pytest tests/test_refusals.py -v` and expect PASS: the key-literal scan must accept `"freq_bounds"`.

#### Controller rulings (pre-flight, binding — these supersede the amendments above and the task text)

- **F35/F39** — as for Task 14: `body.grid` does not carry the resolved span; delete that Interfaces sentence.

---

### Task 16: nothing measurable is a refusal

**Why:** Spec §3.6 and E4: "If every grid frequency comes back blank, the run refuses — a calm
one-line refusal naming the band and the spontaneous duration, not a crash — and its unfinished
folder stays behind holding the spontaneous spectrum… Today this case saves an empty figure and
reports success." Spec §3.5 says which blanks stay reachable once T15 gates the low end: the
**upper** end for a single-cell run, because the test is two-sided and the chi grid tops out at
`bounds[1]*omega_0` against a spectrum Nyquist of `pi/dt_nd`, which an active cell can exceed. (For
a sweep, either end stays reachable — its common grid is not behind T15's gate; that is T20's.)

**Files:**
- Modify: `core/FDT/fdt_pipeline.py` — the `lo_res, _hi_res` binding T15 added; the block after
  step 7; the step-9 plot block (`:137-156`)
- Test: `tests/test_fdt_user.py`

**Interfaces:**
- Consumes: `_resolved_span` (T14) and T15's `lo_res, _hi_res` binding.
- Produces: in `run_fdt`, `hi_res` is now bound (T15's `_hi_res` renamed) and a local
  `blanks = int(torch.isnan(G_at_omegas).sum())` is computed after step 7. `blanks` and
  `G_at_omegas.numel()` are exactly the pair T17 writes as `body.offgrid = {"blanks": …, "of": …}`.
  `run_fdt` raises `core.refusals.Refusal` (`field=None`) when every probe is blank, and logs one
  warning record when only some are. The spontaneous PSD figure is written **before** Campaign 2,
  so `psd_<stamp>.png` is on disk whenever this refusal fires.

- [ ] **Step 1: Write the failing test**

In `tests/test_fdt_user.py`, after the Task 15 test:

```python
def test_a_run_that_can_measure_nothing_refuses_and_leaves_the_spectrum_behind(tmp_path, monkeypatch):
    """Spec §3.6, E4. A run whose every probe frequency is blank has measured nothing: today it
    divides blanks by blanks, saves a figure with no points on it, and reports success -- an answer
    indistinguishable from a real one until someone opens the picture. It refuses instead, naming the
    band and the two settings that set it.

    E4's other half is asserted here too: the run leaves behind what diagnoses the failure. The
    spontaneous spectrum's picture is written BEFORE the driven campaign, so it is on disk when this
    refusal fires, while the ratio and susceptibility figures -- which would have been empty -- are
    not. T17 turns that directory into the record's unfinished folder; the ordering is what makes the
    promise keepable.

    The low end is gated by T15, so the reachable case is the UPPER end (spec §3.5): here the probe
    grid runs 1.0..6.0 against a spectrum that resolves 0.1..0.3. The four plot helpers are replaced
    by recorders that write their save_path, so the test asserts WHICH figures a run leaves without
    paying for matplotlib."""
    import pytest

    from core.FDT import fdt_pipeline
    from core.refusals import Refusal

    class _Cfg:
        model = "HOPF"
        params_dict = {"sigma_x": (0.1, None)}
        n_freqs, freq_bounds = 5, (5.0, 30.0)
        burn_in_nd, dt_nd, psd_T_obs_nd, omega_0 = 0.0, 0.01, 50.0, 1.0

        class hw:
            device = torch.device("cpu")
            dtype = torch.float64

    # Resolves 0.1..0.3, peak at 0.2 -> the grid runs 1.0 .. 6.0: its LOWEST probe is above the
    # spectrum's highest bin, so T15's low-end gate passes and every probe is blank anyway.
    freqs_psd = torch.tensor([0.0, 0.1, 0.2, 0.3], dtype=torch.float64)
    G = torch.tensor([9.0, 1.0, 5.0, 1.0], dtype=torch.float64)
    monkeypatch.setattr(fdt_pipeline, "_out_dir", lambda: tmp_path)
    monkeypatch.setattr(fdt_pipeline, "run_campaign1_psd",
                        lambda cfg, return_trajectory=False: (
                            freqs_psd, G, torch.arange(4, dtype=torch.float64),
                            torch.zeros(4, dtype=torch.float64)))
    monkeypatch.setattr(fdt_pipeline, "run_campaign2_chi",
                        lambda cfg, om, **kw: torch.full((len(om),), 1 + 1j, dtype=torch.complex128))
    for name in ("plot_spontaneous_trajectory", "plot_psd", "plot_eff_temp_ratio",
                 "plot_chi_components"):
        monkeypatch.setattr(fdt_pipeline, name,
                            lambda *a, save_path=None, **kw: save_path.write_bytes(b"png"))

    with pytest.raises(Refusal) as e:
        fdt_pipeline.run_fdt(_Cfg(), skip_sanity=True, confirm_production=True)
    msg = str(e.value)
    assert "0.1..0.3" in msg and "psd_T_obs_nd = 50" in msg, msg
    assert e.value.field is None, e.value.field

    written = sorted(p.name.rsplit("_", 2)[0] for p in tmp_path.glob("*.png"))
    assert written == ["psd", "spontaneous_trajectory"], written
```

- [ ] **Step 2: Run it and watch it fail**

Run: `pytest "tests/test_fdt_user.py::test_a_run_that_can_measure_nothing_refuses_and_leaves_the_spectrum_behind" -v`

Expected: FAIL with `Failed: DID NOT RAISE <class 'core.refusals.Refusal'>` — the run completes,
divides an all-NaN `G_at_omegas` by an all-real `chi''`, and saves all four figures.

- [ ] **Step 3: Bind the upper end of the resolved span**

In `core/FDT/fdt_pipeline.py`, find:

```python
    lo_res, _hi_res = _resolved_span(freqs_psd)
```

Replace with:

```python
    lo_res, hi_res = _resolved_span(freqs_psd)
```

- [ ] **Step 4: Write the spectrum's picture before the driven campaign**

In `core/FDT/fdt_pipeline.py`, find:

```python
    # 6. Campaign 2: forced chi via lock-in
    log.info("Campaign 2: forced response -> chi via lock-in")
```

Replace with:

```python
    # The spectrum's own picture goes to disk BEFORE the driven campaign, because it is what
    # diagnoses the two refusals that can follow -- the band above and nothing-measurable below --
    # and E2 keeps the folder it is written into (spec §3.6). It used to be written at the very end,
    # with the other two, where neither refusal could ever reach it.
    psd_path = _out_dir() / f"psd_{timestamp}.png"
    plot_psd(freqs_psd.cpu().numpy(), G.cpu().numpy(),
              save_path=psd_path,
              title=f"Spontaneous PSD (Campaign 1): ND {cfg.model}",
              omega_natural=omega_natural,
              plot_band=(cfg.omega_0 * cfg.freq_bounds[0],
                          cfg.omega_0 * cfg.freq_bounds[1]))
    log.info(f"Saved spontaneous PSD plot to: {psd_path}")

    # 6. Campaign 2: forced chi via lock-in
    log.info("Campaign 2: forced response -> chi via lock-in")
```

- [ ] **Step 5: Refuse when nothing was measured**

In `core/FDT/fdt_pipeline.py`, find:

```python
    # 7. Interpolate Welch G onto the chi frequency grid (log-omega, linear-y)
    G_at_omegas = _interp_log(omegas, freqs_psd, G)
```

Replace with:

```python
    # 7. Interpolate Welch G onto the chi frequency grid (log-omega, linear-y)
    G_at_omegas = _interp_log(omegas, freqs_psd, G)

    # Nothing measurable is a refusal, not an empty picture (spec §3.6, E4). A probe the spontaneous
    # spectrum does not resolve comes back blank; when EVERY probe is blank there is no ratio, and
    # this used to divide blanks by blanks, save a figure with no points on it and report success.
    # The low end is gated above, so what stays reachable here is the UPPER end: the grid tops out at
    # freq_bounds[1]*omega_0 against a spectrum Nyquist of pi/dt_nd, which an active cell can exceed
    # (spec §3.5). `blanks` and `of` are what T17 records as body.offgrid.
    blanks, of = int(torch.isnan(G_at_omegas).sum()), G_at_omegas.numel()
    if blanks == of:
        raise Refusal(
            f"None of the {of} probe frequencies is resolved by the spontaneous spectrum, which "
            f"covers {lo_res:g}..{hi_res:g} (ND), so the effective-temperature ratio is unmeasurable "
            f"here. Narrow freq_bounds towards that span, lengthen the spontaneous recording "
            f"(psd_T_obs_nd = {cfg.psd_T_obs_nd:g}) to reach lower, or shorten the step "
            f"(dt_nd = {cfg.dt_nd:g}) to reach higher.")
    if blanks:
        log.warning(f"{blanks}/{of} probe frequencies fall outside the band the spontaneous spectrum "
                    f"resolves ({lo_res:g}..{hi_res:g} ND) and are blank in the ratio.")
```

- [ ] **Step 6: Drop the PSD plot from the closing block**

In `core/FDT/fdt_pipeline.py`, find:

```python
    # 9. Plot + save (timestamp set at the top of run_fdt)
    ratio_path = _out_dir() / f"fdt_ratio_{timestamp}.png"
    chi_path = _out_dir() / f"chi_components_{timestamp}.png"
    psd_path = _out_dir() / f"psd_{timestamp}.png"

    plot_psd(freqs_psd.cpu().numpy(), G.cpu().numpy(),
              save_path=psd_path,
              title=f"Spontaneous PSD (Campaign 1): ND {cfg.model}",
              omega_natural=omega_natural,
              plot_band=(cfg.omega_0 * cfg.freq_bounds[0],
                          cfg.omega_0 * cfg.freq_bounds[1]))
    plot_eff_temp_ratio(omegas.cpu().numpy(), ratio.cpu().numpy(),
```

Replace with:

```python
    # 9. Plot + save (timestamp set at the top of run_fdt; the PSD went to disk before Campaign 2)
    ratio_path = _out_dir() / f"fdt_ratio_{timestamp}.png"
    chi_path = _out_dir() / f"chi_components_{timestamp}.png"

    plot_eff_temp_ratio(omegas.cpu().numpy(), ratio.cpu().numpy(),
```

Find:

```python
    log.info(f"Saved plots to:\n  {psd_path}\n  {ratio_path}\n  {chi_path}")
```

Replace with:

```python
    log.info(f"Saved plots to:\n  {ratio_path}\n  {chi_path}")
```

- [ ] **Step 7: Run the test and watch it pass**

Run: `pytest "tests/test_fdt_user.py::test_a_run_that_can_measure_nothing_refuses_and_leaves_the_spectrum_behind" -v`
Expected: PASS.

- [ ] **Step 8: Run every suite that touches this path**

Run: `pytest tests/test_fdt_user.py tests/test_refusals.py tests/test_nav_and_gating.py -v`
Expected: PASS. `test_nav_and_gating.py`'s FDT panel test stubs `run_fdt` wholesale
(`monkeypatch.setattr(fdt_panel, "run_fdt", boom)`), so it is untouched by any of T13–T16; it is run
here because it is the other suite that imports `FDTModelError`.

- [ ] **Step 9: Commit**

```bash
git add core/FDT/fdt_pipeline.py tests/test_fdt_user.py
git commit -m "fdt: refuse a run that measured nothing instead of drawing an empty figure"
```
## THE TWO RUNS BECOME RECORDS (T17–T20)

> These four tasks are the heart of the piece. T17 and T19 each carry the `public_entry` source scan
> and its two companion sets in the SAME commit — `tests/test_artifact_store.py` asserts
> `found == want` by equality over `CODE_ROOTS = ("core",)`, so the fast gate reds the moment a
> decorator lands without it. Neither may defer it.
>
> **Line numbers are hints; the quoted text is the anchor.** T13–T16 have already edited
> `core/FDT/fdt_pipeline.py` before T17 runs (the prefactor moved to the top of `run_fdt`, the band
> refusal landed after the grid is built, the all-blank refusal landed after the interpolation), so
> every number quoted from that file is stale by design. Find the quoted text; if it is not there,
> re-read the file and say so in the task report.

#### Amendments (binding — these supersede the text above)

Rulings already satisfied by the body, no change: **P57** (`of = G_at_omegas.numel()` is the Campaign-2 probe grid), **P73** (Step 8 is right that `tests/test_nav_and_gating.py`'s FDT test is untouched by T13–T16).

**A. T12 has landed.** `run_fdt` now calls `warn_thin_settings(cfg)` before anything else, and it reads `cfg.ensemble_M`. In Step 1's `_Cfg`, directly under `n_freqs, freq_bounds = 5, (5.0, 30.0)`, add:

```python
        ensemble_M = 8        # read first by the thin-setting check; at its threshold, so it says nothing
```

Without it, Step 2 fails with `AttributeError` instead of `DID NOT RAISE`.

**B. P22 (as ratified from O15, which states the move puts the figure "on disk for BOTH refusals"): the PSD figure goes above T15's band check, not below it.** Step 4's anchor (`# 6. Campaign 2`) sits BELOW T15's band refusal, so the band refusal would leave a folder with no spectrum. That contradicts the drafted comment "the two refusals that can follow -- the band above ...".

*Step 4 is replaced by this.* In `core/FDT/fdt_pipeline.py`, find the first line of the comment T15 added:

```python
    # The band, checked the first moment it is knowable and BEFORE the driven campaign -- the
```

and insert directly ABOVE it, after the `omegas = gen_freqs_log(...)` statement and its blank line:

```python
    # The spectrum's own picture goes to disk BEFORE the band check and the driven campaign, because
    # it is what diagnoses both refusals that can follow -- a band reaching below what the spectrum
    # resolves (just below) and nothing measurable (after the drive) -- and E2 keeps the folder it is
    # written into (spec §3.6, P22). It used to be written at the very end, where neither refusal
    # could ever reach it.
    psd_path = _out_dir() / f"psd_{timestamp}.png"
    plot_psd(freqs_psd.cpu().numpy(), G.cpu().numpy(),
              save_path=psd_path,
              title=f"Spontaneous PSD (Campaign 1): ND {cfg.model}",
              omega_natural=omega_natural,
              plot_band=(cfg.omega_0 * cfg.freq_bounds[0],
                          cfg.omega_0 * cfg.freq_bounds[1]))
    log.info(f"Saved spontaneous PSD plot to: {psd_path}")

```

Keep `psd_path = _out_dir() / f"psd_{timestamp}.png"` **verbatim**: T17 (P74) finds and replaces exactly that line. Do NOT insert anything before `# 6. Campaign 2: forced chi via lock-in`; that line and the `log.info("Campaign 2: ...")` below it stay as they are. `omega_natural` and `cfg.omega_0` are both bound by step 4 of `run_fdt`, above this point.

*Interfaces, Produces*: "The spontaneous PSD figure is written **before** Campaign 2, so `psd_<stamp>.png` is on disk whenever this refusal fires" becomes: "... is written before T15's band check (and so before Campaign 2), so `psd_<stamp>.png` is on disk whenever EITHER refusal fires."

**C. New Step 4b: pin the band case in T15's test.** In `tests/test_fdt_user.py`, inside `test_a_band_below_the_spectrums_resolution_refuses_before_the_driven_campaign`, find:

```python
    monkeypatch.setattr(fdt_pipeline, "plot_spontaneous_trajectory", lambda *a, **kw: None)
```

and replace it with:

```python
    monkeypatch.setattr(fdt_pipeline, "plot_spontaneous_trajectory", lambda *a, **kw: None)
    monkeypatch.setattr(fdt_pipeline, "plot_psd",
                        lambda *a, save_path=None, **kw: save_path.write_bytes(b"png"))
```

Then find:

```python
    assert driven == [], "Campaign 2 was entered before the band was checked"
```

and append after it:

```python
    written = [p.name.rsplit("_", 2)[0] for p in tmp_path.glob("*.png")]
    assert written == ["psd"], ("the spontaneous spectrum's picture must be on disk when the band "
                                f"refusal fires -- it is what diagnoses it (P22): {written}")
```

(The trajectory stub in that test writes nothing, so the PSD is the only PNG.)

**D. Step 7 runs both tests.** Run: `pytest "tests/test_fdt_user.py::test_a_run_that_can_measure_nothing_refuses_and_leaves_the_spectrum_behind" "tests/test_fdt_user.py::test_a_band_below_the_spectrums_resolution_refuses_before_the_driven_campaign" -v`. Expected: 2 passed. Before Step 4 (B) lands, the band test's new assertion fails with `written == []`.

**E. The refusal's field is pending a ruling.** P75 keys T15's band refusal `"freq_bounds"` and does not name this one. Build Step 5 as drafted (`field=None`, test asserts `is None`) unless the plan owner has ruled otherwise. If the ruling is `"freq_bounds"`, add `field="freq_bounds"` as the last argument of Step 5's `raise Refusal(...)` and change the test's `assert e.value.field is None, e.value.field` to `assert e.value.field == "freq_bounds", e.value.field`.

**F. Step 6 is unchanged.** The closing block still loses `psd_path` and its `plot_psd` call, and the "Saved plots to:" record still shortens to the two paths. T17's P74 anchors depend on exactly that result.

#### Controller rulings (pre-flight, binding — these supersede the amendments above and the task text)

- **F37** — the nothing-measurable refusal carries `field="freq_bounds"`, as Task 15's band refusal does under P75: same subject (the band against the spectrum's span). Change the raise and the test's assertion to `assert e.value.field == "freq_bounds"`. The fix sentence is empty either way.
- **F12** — confirmed as the amendment applied it: the spontaneous-spectrum figure is drawn ABOVE Task 15's band check, still before Campaign 2, so the folder holds it for BOTH refusals.

---

### Task 17: `run_fdt` becomes a public entry that writes an `fdt` record

**Why:** Spec §3.1 — `run_fdt` is decorated with `core.runs.public_entry`, takes the
OPEN-BUT-NOT-ENTERED writer and does `with writer:` itself, so `__enter__`, every `refresh()` and
`__exit__` run on the thread whose `RunLog` becomes `log.txt` (spec §1.2, forced by the fact that
`runs._active` is a `threading.local`). It returns `LoadedFdt` instead of `None` (E1). It draws and
records a seed when none is given, confining the reseeding the way `core/diagnostics/rng.py` confines
its own (spec §3.7, E7). With `FDTConfig.copy_for_run()` in place (T7) the decorator delivers V1: the
two `cfg.omega_0 = …` writes stop reaching the caller's object.

**Files:**
- Modify: `core/FDT/fdt_pipeline.py:32-37` (`def _out_dir():`), `:62-70` (`def run_fdt(...)` and its
  docstring), `:76-77` (the timestamp), `:84` / `:104` / `:138-140` (the five figure paths)
- Modify: `core/tool/fdt.py:122-144` (`run_fdt_cmd`) — the bridging call site
- Modify: `core/gui/panels/fdt_panel.py:37-51` (`_run_fdt_guarded`) and `:140-143` (the dispatch)
- Test: `tests/test_artifact_store.py:2497-2568` (the scan, `_UNTOUCHED_LEGS`, `_REFUSAL_FIELDS`)
- Test: `tests/test_fdt_user.py` (the record round trip; the module's existing `run_fdt` call at `:212`)
- Test: `tests/test_tool.py:1181-1223` (the `_run_fdt` recorder and its three assertions)

**Interfaces:**
- Consumes, from T1: `manifest.BODY_KEYS["fdt"] = ("study", "settings", "seed", "grid", "points",
  "offgrid", "notices", "compared", "complete", "results")`, `store.KIND_DIRS["fdt"] = "fdt"`,
  `store.LoadedFdt(Loaded)` with `body: dict` and `data_path: "Path | None"`,
  `ArtifactStore.load_fdt(ref) -> LoadedFdt`.
- Consumes, from T3: `ArtifactWriter.progressive` (true for `"fdt"`), `ArtifactWriter.refresh()`, and
  the progressive `__exit__` — an exception KEEPS the directory unless it is a `Refusal` raised before
  the first payload or figure.
- Consumes, from T7: `FDTConfig.sources`, `FDTConfig.seed`, `FDTConfig.copy_for_run()`, and
  `manifest.config_from_cfg`'s FDT branch.
- Consumes, from T13: `observable_noise_prefactor(cfg)` is called at the top of `run_fdt`'s body and
  raises `FDTModelError(..., field="cell")` for a cell missing an FDT parameter. **If T13 chose a
  different field key, `_REFUSAL_FIELDS["run_fdt"]` in Step 1 takes that key instead** — the pin is an
  equality, so it must name whatever T13 actually raises.
- Consumes, from T16: `run_fdt` already counts the blank (NaN) entries of the interpolated spectrum.
  If T16 left no named count, compute it here as `int(torch.isnan(G_at_omegas).sum())`.
- Produces, for T18/T19/T25/T28:
  - `run_fdt(cfg, *, skip_sanity, confirm_production, writer, seed=None) -> LoadedFdt`
  - `core.FDT.fdt_pipeline._resolve_seed(seed, cfg) -> int`
  - `core.FDT.fdt_pipeline._settings_block(cfg) -> dict` (the knob half of `body.settings`; T19
    updates it with the preset name and the grids)
  - `core.FDT.fdt_pipeline._measure(cfg, *, skip_sanity, confirm_production, writer)` — the old body,
    run inside the writer; T18 adds the `data.h5` write to it.
  - `FdtPanel._run` creates the writer on the GUI thread and passes it; **T25** adds the Seed box, the
    record picker, the name/note controls and the `figures/` watch directory on top of it.

- [ ] **Step 1: Add `run_fdt` to the `public_entry` scan and its two companion sets**

In `tests/test_artifact_store.py`, find:

```python
_UNTOUCHED_LEGS = {
    "generate_observations": _leg_generate_observations,
```

and add one entry at the END of that dict (after `"channel_ablation": _leg_channel_ablation,`):

```python
    "run_fdt": _leg_run_fdt,
}
```

Find:

```python
_REFUSAL_FIELDS = {
    "generate_observations": "name",                 # _EntryStore refuses "taken" as the store does
```

and add at the END of that dict:

```python
    "run_fdt": "cell",                               # a cell with no FDT normalisation constant (§3.4)
}
```

Find:

```python
def test_the_fifteen_public_entries_carry_public_entry_and_nothing_else_does():
    """V1 (spec §2.2). The private copy is kept by ONE decorator on exactly fifteen functions: the ten
    stages and compositions of core/orchestrator.py and the five diagnostics. Read off the source
```

Replace with (the name loses its count so T19 does not have to rename it again):

```python
def test_the_public_entries_carry_public_entry_and_nothing_else_does():
    """V1 (spec §2.2). The private copy is kept by ONE decorator on exactly the functions named below:
    the ten stages and compositions of core/orchestrator.py, the five diagnostics, and -- since piece
    5 -- core/FDT's single-cell measurement. Read off the source
```

Find:

```python
    want |= {("core/diagnostics/identifiability.py", n) for n in (
        "identifiability_rotation", "identifiability_laplace", "identifiability_jacobian")}
```

Replace with:

```python
    want |= {("core/diagnostics/identifiability.py", n) for n in (
        "identifiability_rotation", "identifiability_laplace", "identifiability_jacobian")}
    # Piece 5: the FDT measurement writes a record and must not write on the caller's FDTConfig --
    # `cfg.omega_0 = ...` twice in its own body is exactly the V1 defect, and copy_for_run (T7) is
    # what stops it reaching the panel's settings object.
    want |= {("core/FDT/fdt_pipeline.py", "run_fdt")}
```

- [ ] **Step 2: Write the leg the parametrised V1 pin runs**

In `tests/test_artifact_store.py`, immediately after `_leg_channel_ablation` (the last leg, which ends
`return cfg, lambda: ablation.channel_ablation(...)`), add:

```python
class _FdtWriter:
    """The writer surface run_fdt touches, over a real temp directory but no store.

    PROGRESSIVE, like the real one (spec §2.2): __exit__ KEEPS the directory on an exception, so a leg
    that refuses or booms leaves its folder behind exactly as E2 requires. It absorbs _BodyDone and
    nothing else, so a leg can end the measurement at its first campaign and still take run_fdt's
    `return writer.store.load_fdt(writer.id)` line -- a success from the caller's side, with no
    solver, no figure and no h5py."""

    def __init__(self, tmp_path):
        self.id = "20260922T000000"
        self.dir = Path(tmp_path) / "fdt" / f"_unnamed__{self.id}"
        self.body, self.parents, self.fingerprints = {}, {}, {}
        self.store = SimpleNamespace(load_fdt=lambda ref: SimpleNamespace(id=ref, body=self.body))
        self.refreshed = 0

    def __enter__(self):
        self.dir.mkdir(parents=True, exist_ok=True)
        return self

    def __exit__(self, exc_type, exc, tb):
        return exc_type is not None and issubclass(exc_type, _BodyDone)

    def refresh(self):
        self.refreshed += 1

    def payload(self, filename):
        return self.dir / filename

    def figure_path(self, title):
        (self.dir / "figures").mkdir(exist_ok=True)
        return self.dir / "figures" / f"{title}.png"


def _fdt_cfg_for_leg(case):
    """A real HOPF FDTConfig at the smallest size. The "refusal" case deletes the parameter
    observable_noise_prefactor needs, which is the run's own pre-spend refusal (spec §3.4)."""
    from core import cli, config as _cfgmod
    cfg = cli.make_fdt_config("HOPF", False, str(_cfgmod.CELL_PATH / "hopf" / "cell.txt"),
                              n_freqs=2, ensemble_M=2)
    if case == "refusal":
        cfg.params_dict.pop("sigma_x")
    return cfg


def _leg_run_fdt(case, monkeypatch, tmp_path):
    """run_fdt's three endings. Its own write on its working config is `cfg.omega_0 = ...` (step 1 of
    its body) and `cfg.seed = ...`, both made before Campaign 1 -- so the seam that raises _Injected is
    Campaign 1 itself, and "boom" therefore lands AFTER a write, which is what the pin is for."""
    from core.FDT import fdt_pipeline
    cfg = _fdt_cfg_for_leg(case)
    monkeypatch.setattr(fdt_pipeline, "run_campaign1_psd",
                        _raise_injected if case == "boom" else _body_done)
    w = _FdtWriter(tmp_path)
    return cfg, lambda: fdt_pipeline.run_fdt(cfg, skip_sanity=True, confirm_production=True,
                                             writer=w, seed=11)
```

- [ ] **Step 3: Run the two pins and watch them fail**

Run: `pytest tests/test_artifact_store.py -k "public_entry or callers_config" -v`

Expected: FAIL. `test_the_public_entries_carry_public_entry_and_nothing_else_does` fails on
`assert found == want` with `missing [('core/FDT/fdt_pipeline.py', 'run_fdt')]; unexpected []`, and
each of the three `test_every_public_entry_leaves_the_callers_config_untouched[run_fdt-*]` cases fails
with `TypeError: run_fdt() got an unexpected keyword argument 'writer'`.

- [ ] **Step 4: Write the record round-trip test**

In `tests/test_fdt_user.py`, add at the end of the file:

```python
def _stub_campaigns(monkeypatch, n_psd=16):
    """Campaign 1 and Campaign 2 replaced by arithmetic: a single-peaked spectrum on a positive grid
    and a flat susceptibility. The record's SHAPE is what these tests are about; a real campaign is
    slow-marked and lives in tests/test_tool.py."""
    import torch
    from core.FDT import fdt_pipeline

    omegas_psd = torch.linspace(0.0, 4.0, n_psd, dtype=torch.float64)
    G = torch.exp(-((omegas_psd - 1.0) ** 2) / 0.02) + 1e-6
    t = torch.linspace(0.0, 1.0, 8, dtype=torch.float64)

    def _c1(cfg, return_trajectory=False):
        if return_trajectory:
            return omegas_psd, G, t, torch.zeros_like(t)
        return omegas_psd, G

    monkeypatch.setattr(fdt_pipeline, "run_campaign1_psd", _c1)
    monkeypatch.setattr(fdt_pipeline, "run_campaign2_chi",
                        lambda cfg, omegas: torch.full(omegas.shape, 1 + 1j, dtype=torch.complex128))
    for name in ("plot_psd", "plot_eff_temp_ratio", "plot_chi_components",
                 "plot_spontaneous_trajectory"):
        monkeypatch.setattr(fdt_pipeline, name,
                            lambda *a, save_path=None, **k: Path(save_path).write_bytes(b"\x89PNG"))


def test_run_fdt_writes_a_record_and_leaves_the_callers_config_alone(store, monkeypatch):
    """E1 and V1 together, which is the whole point of making this a public entry. Before piece 5 the
    measurement returned None, wrote five timestamped PNGs into a flat <artifacts root>/fdt that no
    command could list or delete, and wrote ``cfg.omega_0`` on the panel's own settings object twice
    (core/FDT/fdt_pipeline.py's steps 1 and 4) -- so the frequency grid of the NEXT run started from
    the last run's resonance. The record answers the first; copy_for_run, which public_entry is
    duck-typed on, answers the second.

    The body must carry EVERY key of BODY_KEYS["fdt"] (manifest.validate compares the key set by
    equality), with the keys a single-cell run cannot fill left null."""
    from core import cli, config
    from core.artifacts import manifest as mf
    from core.FDT import fdt_pipeline
    from tests._fixtures import assert_cfg_unchanged, snapshot_cfg

    _stub_campaigns(monkeypatch)
    cfg = cli.make_fdt_config("HOPF", False, str(config.CELL_PATH / "hopf" / "cell.txt"),
                              n_freqs=5, ensemble_M=2)
    snap = snapshot_cfg(cfg)

    w = store.create("fdt", cfg, name="run1", note="a stubbed measurement")
    rec = fdt_pipeline.run_fdt(cfg, skip_sanity=True, confirm_production=True, writer=w, seed=11)

    assert rec.name == "run1" and rec.body["study"] == "single"
    assert set(rec.body) == set(mf.BODY_KEYS["fdt"]), "validate compares the key set by equality"
    assert rec.body["seed"] == 11 and rec.body["complete"] is True
    assert rec.body["points"] is None and rec.body["compared"] is None
    assert rec.body["notices"] == [], "no notices is an empty list, never null (spec §2.3)"
    assert rec.body["grid"]["n_freqs"] == 5 and rec.body["grid"]["omega_0"] > 0.0
    assert rec.body["offgrid"]["of"] == 5
    assert rec.body["settings"]["ensemble_M"] == 2 and rec.body["settings"]["skip_sanity"] is True
    assert sorted(rec.manifest.figures) == ["figures/chi-components.png",
                                            "figures/effective-temperature-ratio.png",
                                            "figures/spontaneous-psd.png",
                                            "figures/spontaneous-trajectory.png"]
    assert (rec.path / "log.txt").read_text(encoding="utf-8").strip(), \
        "log.txt is written from runs.current_run_log(), which only exists on the thread that " \
        "entered the writer -- an empty file means the front end entered it instead"
    assert_cfg_unchanged(cfg, snap)


def test_a_failed_fdt_run_keeps_its_record_marked_unfinished(store, monkeypatch):
    """E2: an interrupted or crashed measurement keeps its folder, plainly marked unfinished, and the
    spontaneous spectrum it did collect is what diagnoses the failure. The six ordinary kinds still
    delete theirs -- tests/test_artifact_store.py pins that -- so this is the one place the
    progressive mode is visible from a stage."""
    import pytest
    from core import cli, config
    from core.FDT import fdt_pipeline

    _stub_campaigns(monkeypatch)
    monkeypatch.setattr(fdt_pipeline, "run_campaign2_chi",
                        lambda cfg, omegas: (_ for _ in ()).throw(RuntimeError("stub out of memory")))
    cfg = cli.make_fdt_config("HOPF", False, str(config.CELL_PATH / "hopf" / "cell.txt"),
                              n_freqs=3, ensemble_M=2)
    w = store.create("fdt", cfg, name="halfway")
    with pytest.raises(RuntimeError, match="stub out of memory"):
        fdt_pipeline.run_fdt(cfg, skip_sanity=True, confirm_production=True, writer=w, seed=3)

    (summary,) = store.list("fdt")
    assert summary.name == "halfway" and summary.complete and not summary.finished
    assert (w.dir / "figures" / "spontaneous-trajectory.png").exists(), \
        "the figures drawn before the failure stay on disk"


def test_run_fdt_draws_and_records_a_seed_when_none_is_given(store, monkeypatch):
    """E7: every run records the seed it used, so repeats of one cell can be told apart and their
    spread read as the measurement error. A blank Seed box and an absent --seed both mean "draw one
    and record it" -- a run whose seed were simply unset could never be repeated."""
    import random
    from core import cli, config
    from core.FDT import fdt_pipeline

    _stub_campaigns(monkeypatch)
    cfg = cli.make_fdt_config("HOPF", False, str(config.CELL_PATH / "hopf" / "cell.txt"),
                              n_freqs=3, ensemble_M=2)
    random.seed(4242)
    drawn = random.randrange(2 ** 31)
    random.seed(4242)
    rec = fdt_pipeline.run_fdt(cfg, skip_sanity=True, confirm_production=True,
                               writer=store.create("fdt", cfg, name="drawn"))
    assert rec.body["seed"] == drawn, "the drawn seed is the recorded seed"
    assert cfg.seed is None, "the draw lands on the private copy, never on the caller's config"
```

At the top of `tests/test_fdt_user.py` the module already imports `Path`; no new import is needed
there. Widen the module docstring's first line, which currently reads:

```python
"""FDT-for-user-models unit tests (FEATURE 1 v3 + B-d), Qt-free.
```

Replace with:

```python
"""FDT unit tests, Qt-free: the user-model normalization gate (FEATURE 1 v3 + B-d) and, since piece 5,
the records the two measurements write.
```

- [ ] **Step 5: Run the new tests and watch them fail**

Run: `pytest tests/test_fdt_user.py -k "record or seed or unfinished" -v`

Expected: FAIL — all three with `TypeError: run_fdt() got an unexpected keyword argument 'writer'`.

- [ ] **Step 6: Give `run_fdt` the decorator, the writer and the seed**

In `core/FDT/fdt_pipeline.py`, find:

```python
import logging
import math
from datetime import datetime

import torch

from core import config
from core.config import FDTConfig
```

Replace with:

```python
import logging
import math
import random

import torch

from core.config import FDTConfig
from core.diagnostics.rng import seeded
from core.runs import public_entry
```

(`core.diagnostics.rng` imports only `contextlib` at module level, so this adds nothing heavy;
`core.runs` is torch-free by contract. `config` and `datetime` are left behind by Step 7.)

Find and DELETE:

```python
def _out_dir():
    """Where FDT saves its plots: <artifacts root>/fdt, created on demand (piece 5 wraps FDT in the
    store; until then this is a plain directory)."""
    d = config.artifacts_root() / "fdt"
    d.mkdir(parents=True, exist_ok=True)
    return d
```

In its place put:

```python
def _resolve_seed(seed, cfg) -> int:
    """The seed this run used (E7): the one given, else the one the config was built with, else one
    DRAWN and recorded. Drawn from Python's own ``random``, which is the one global stream neither
    the solver nor the spectrum reads -- ``torch.seed()`` would reseed every CUDA device as a side
    effect of being asked a question (the hazard core/SBI/decorrelate.py documents)."""
    if seed is not None:
        return int(seed)
    if getattr(cfg, "seed", None) is not None:
        return int(cfg.seed)
    return random.randrange(2 ** 31)


def _settings_block(cfg) -> dict:
    """Every knob the run resolved, for ``body.settings`` (spec §2.3). The five that no front end
    exposes -- freq_bounds, burn_in_nd, T_obs_periods, dt_nd, psd_T_obs_nd -- are recorded here even
    though no control and no flag names them, because without them a number is not reproducible."""
    return {"n_freqs": int(cfg.n_freqs), "ensemble_M": int(cfg.ensemble_M),
            "freqs_per_batch": int(cfg.freqs_per_batch), "F0": float(cfg.F0),
            "freq_bounds": [float(v) for v in cfg.freq_bounds],
            "burn_in_nd": float(cfg.burn_in_nd), "T_obs_periods": int(cfg.T_obs_periods),
            "dt_nd": float(cfg.dt_nd), "psd_T_obs_nd": float(cfg.psd_T_obs_nd)}
```

- [ ] **Step 7: Split `run_fdt` into the decorated wrapper and the measurement**

In `core/FDT/fdt_pipeline.py`, find:

```python
def run_fdt(cfg: FDTConfig, *, skip_sanity: bool, confirm_production: bool) -> None:
    """End-to-end FDT analysis. Runs sanity checks first; gates on the caller's answer before the
    production sweep.

    :param skip_sanity: skip the sanity checks. REQUIRED: there is no prompt to fall back to (D1
                        retired the CLI), and the old None default meant an input() that a GUI worker
                        thread could never answer.
    :param confirm_production: proceed to the production sweep after sanity. REQUIRED, for the same
                        reason; only consulted when the sanity checks run."""
```

Replace with:

```python
@public_entry
def run_fdt(cfg: FDTConfig, *, skip_sanity: bool, confirm_production: bool,
            writer, seed: "int | None" = None):
    """End-to-end FDT analysis, written into one ``fdt`` record. Runs sanity checks first; gates on
    the caller's answer before the production sweep.

    :param skip_sanity: skip the sanity checks. REQUIRED: there is no prompt to fall back to (D1
                        retired the CLI), and the old None default meant an input() that a GUI worker
                        thread could never answer.
    :param confirm_production: proceed to the production sweep after sanity. REQUIRED, for the same
                        reason; only consulted when the sanity checks run.
    :param writer: the OPEN-BUT-NOT-ENTERED ArtifactWriter the front end minted with
                        ``store.create("fdt", cfg, ...)``. THIS function enters it, and that is not a
                        detail: ``log.txt`` is written from ``runs.current_run_log()``, which is
                        thread-local and only populated inside ``capture_run()`` on the worker thread
                        -- a writer entered on the window's own thread would write no log at all
                        (spec §1.2). The front end needs ``writer.dir`` before it dispatches, to point
                        the figure watcher at the record's ``figures/``; hence the split.
    :param seed: the seed, or None to draw one and record it (E7).
    :returns: the LoadedFdt for the record just written.

    The decorator hands the body ``cfg.copy_for_run()``, so the two ``cfg.omega_0 = ...`` writes below
    land on a private copy and the caller's settings object is exactly what it was, whether this
    returned, refused or crashed (V1).
    """
    seed = _resolve_seed(seed, cfg)
    cfg.seed = seed                                  # on the PRIVATE copy; recorded in the body below
    writer.body = {"study": "single", "settings": _settings_block(cfg), "seed": seed,
                   "grid": None, "points": None, "offgrid": None, "notices": [],
                   "compared": None, "complete": False, "results": None}
    with writer:
        with seeded(seed, cfg.hw.device):
            _measure(cfg, skip_sanity=skip_sanity, confirm_production=confirm_production,
                     writer=writer)
        writer.body["complete"] = True
    return writer.store.load_fdt(writer.id)


def _measure(cfg: FDTConfig, *, skip_sanity: bool, confirm_production: bool, writer) -> None:
    """The measurement itself, inside the entered writer and inside one seeded context."""
```

The rest of the old `run_fdt` body follows unchanged in `_measure`, **except** for the edits in
Steps 8 and 9. Keep its comments and its step numbering.

- [ ] **Step 8: Send the five figures into the record**

In `core/FDT/fdt_pipeline.py`, find:

```python
    # Single plot dir + timestamp for all outputs from this run (incl. sanity plots).
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
```

Replace with (nothing: delete both lines — the record's directory is the run's identity now, and a
wall-clock stamp inside it would say less than its `created` field does).

Find:

```python
        passive_plot_path = _out_dir() / f"fdt_ratio_passive_{timestamp}.png"
```

Replace with:

```python
        passive_plot_path = writer.figure_path("Passive baseline ratio")
```

Find:

```python
    traj_path = _out_dir() / f"spontaneous_trajectory_{timestamp}.png"
```

Replace with:

```python
    traj_path = writer.figure_path("Spontaneous trajectory")
```

Find:

```python
    ratio_path = _out_dir() / f"fdt_ratio_{timestamp}.png"
    chi_path = _out_dir() / f"chi_components_{timestamp}.png"
    psd_path = _out_dir() / f"psd_{timestamp}.png"
```

Replace with:

```python
    ratio_path = writer.figure_path("Effective temperature ratio")
    chi_path = writer.figure_path("Chi components")
    psd_path = writer.figure_path("Spontaneous PSD")
```

Also delete the now-stale parenthetical `# (timestamp set at the top of run_fdt.)` above the
trajectory plot and `(timestamp set at the top of run_fdt)` in the `# 9. Plot + save` comment.

- [ ] **Step 9: Fill the body as the run proceeds, and refresh**

In `core/FDT/fdt_pipeline.py`, find (the line T13/T16 left after the peak search):

```python
    log.info(f"Spontaneous-oscillation frequency from PSD peak: {omega_natural:.4f} (ND)")
```

Add immediately after it:

```python
    writer.body["grid"] = {"omega_0": float(omega_natural), "omega_0_source": "spectrum peak",
                           "n_freqs": int(cfg.n_freqs),
                           "bounds": [float(v) for v in cfg.freq_bounds],
                           "omega_min": float(cfg.freq_bounds[0] * omega_natural),
                           "omega_max": float(cfg.freq_bounds[1] * omega_natural)}
    writer.refresh()          # the spontaneous half is on disk before the driven campaign is spent
```

Find:

```python
    G_at_omegas = _interp_log(omegas, freqs_psd, G)
```

Add immediately after it (reuse T16's own count if it named one rather than adding a second):

```python
    blanks = int(torch.isnan(G_at_omegas).sum())
    writer.body["offgrid"] = {"blanks": blanks, "of": int(omegas.numel())}
```

Find:

```python
    log.info(f"Saved plots to:\n  {psd_path}\n  {ratio_path}\n  {chi_path}")
```

Replace with:

```python
    writer.body["results"] = _results_block(omegas, ratio, omega_natural, blanks)
    writer.refresh()
    log.info(f"Saved plots to:\n  {psd_path}\n  {ratio_path}\n  {chi_path}")
```

And add, beside `_settings_block`:

```python
def _results_block(omegas, ratio, omega_natural: float, blanks: int) -> dict:
    """The short summary the listing and the detail pane show. FINITE NUMBERS ONLY: manifest.validate
    refuses a non-finite float anywhere in the body, and T_eff/T legitimately carries NaN wherever the
    spectrum came back blank or chi'' crossed zero -- so a summary that would be NaN is recorded as
    null (spec §2.3)."""
    r = ratio.detach().cpu().to(torch.float64)
    w = omegas.detach().cpu().to(torch.float64)
    usable = torch.isfinite(r)

    def _f(v):
        v = float(v)
        return v if math.isfinite(v) else None

    if not bool(usable.any()):
        return {"peak_ratio": None, "peak_omega": None, "ratio_at_resonance": None,
                "usable_fraction": 0.0, "offgrid_blanks": int(blanks)}
    idx = int(torch.argmax(torch.where(usable, r, torch.full_like(r, -math.inf))))
    at_res = int(torch.argmin(torch.abs(w - float(omega_natural))))
    return {"peak_ratio": _f(r[idx]), "peak_omega": _f(w[idx]),
            "ratio_at_resonance": _f(r[at_res]),
            "usable_fraction": float(usable.sum()) / float(r.numel()),
            "offgrid_blanks": int(blanks)}
```

- [ ] **Step 10: Bridge the two call sites so the tree stays green**

In `core/tool/fdt.py`, find:

```python
def run_fdt_cmd(args, store):
    """``store`` is unused: FDT writes plots, not artifacts (piece 5 wraps it)."""
```

Replace with:

```python
def run_fdt_cmd(args, store):
    """One ``fdt`` record per run. The record is created HERE -- ``store.create`` mints the id and runs
    ``assert_name_free`` before anything is spent -- and ENTERED by ``run_fdt`` (spec §1.2)."""
```

Find:

```python
    fdt_pipeline.run_fdt(cfg, skip_sanity=args.skip_sanity,
                         confirm_production=not args.no_production)
    print(f"[prism fdt] plots under {config.artifacts_root() / 'fdt'}")
```

Replace with:

```python
    writer = store.create("fdt", cfg)
    rec = fdt_pipeline.run_fdt(cfg, skip_sanity=args.skip_sanity,
                               confirm_production=not args.no_production, writer=writer)
    print(f"[prism fdt] record {rec.id} at {rec.path}")
```

(`config` stays imported: `run_crossval` still uses it until T19.)

In `core/gui/panels/fdt_panel.py`, find:

```python
    try:
        return run_fdt(cfg, skip_sanity=skip_sanity, confirm_production=confirm_production)
    except KeyError as e:
```

Replace with:

```python
    try:
        return run_fdt(cfg, skip_sanity=skip_sanity, confirm_production=confirm_production,
                       writer=writer, seed=seed)
    except KeyError as e:
```

and change that function's signature line, which reads:

```python
def _run_fdt_guarded(cfg, *, skip_sanity, confirm_production):
```

to:

```python
def _run_fdt_guarded(cfg, *, skip_sanity, confirm_production, writer, seed=None):
```

Find, in `FdtPanel._run`:

```python
        # Explicit bools, never None -- see the module docstring.
        self.dispatch(_run_fdt_guarded, cfg, watch_dir=config.artifacts_root() / "fdt",
                      skip_sanity=self.skip_sanity.isChecked(),
                      confirm_production=self.confirm_production.isChecked(),
                      on_finished=lambda: self.log_pane.append_line("FDT run finished."))
```

Replace with:

```python
        # The RECORD is created here, on the GUI thread: store.create mints the id and refuses a taken
        # name before anything is spent, and it fills writer.dir WITHOUT creating the directory, so
        # the watcher can be pointed at the record's figures/ before the run is dispatched. The stage
        # enters the writer on the worker thread, where the run log lives (spec §1.2). T25 adds the
        # Seed box, the name/note controls and the record picker on top of this.
        writer = default_store().create("fdt", cfg)
        self.dispatch(_run_fdt_guarded, cfg, watch_dir=writer.dir / "figures",
                      writer=writer,
                      skip_sanity=self.skip_sanity.isChecked(),
                      confirm_production=self.confirm_production.isChecked(),
                      on_finished=lambda: self.log_pane.append_line("FDT run finished."))
```

and add `default_store` to the panel's imports — find:

```python
from core import cli, config, registry
```

Replace with:

```python
from core import cli, registry
from core.artifacts import default_store
```

(`config` is no longer read by this module once the watch directory comes from the writer; drop it if
no other line in the file uses it, and keep it if one does.)

- [ ] **Step 11: Update the two tests that would otherwise go red**

In `tests/test_fdt_user.py`, find:

```python
    monkeypatch.setattr(fdt_pipeline, "_out_dir", lambda: tmp_path)
    monkeypatch.setattr(fdt_pipeline, "run_all_sanity",
                        lambda cfg, passive_plot_path=None: {"linearity": (False, {"ratio": 0.5})})
    caplog.clear()
    fdt_pipeline.run_fdt(_Hopf(), skip_sanity=False, confirm_production=False)
```

Replace with:

```python
    monkeypatch.setattr(fdt_pipeline, "run_all_sanity",
                        lambda cfg, passive_plot_path=None: {"linearity": (False, {"ratio": 0.5})})
    caplog.clear()
    fdt_pipeline.run_fdt(_Hopf(), skip_sanity=False, confirm_production=False,
                         writer=_LogWriter(tmp_path), seed=1)
```

and widen the `_Hopf` stub, which reads:

```python
    class _Hopf:
        model = "HOPF"
```

to:

```python
    class _Hopf:
        model, n_freqs, ensemble_M, freqs_per_batch, F0 = "HOPF", 2, 2, 1, 0.05
        freq_bounds, burn_in_nd, T_obs_periods = (0.1, 30.0), 100.0, 30
        dt_nd, psd_T_obs_nd, seed = 0.01, 8000.0, None
        hw = config.cpu_device()
```

with `from core import config` added to that test's local imports, and a `_LogWriter` beside `_Hopf`:

```python
    class _LogWriter:
        """Enough writer for a run that stops after the sanity verdict: a real directory, a body, and
        a store that hands back what it was given. A PLAIN stub is deliberate -- this test is about the
        records the pipeline LOGS, and a real ArtifactWriter would drag a manifest into it."""
        def __init__(self, d):
            self.id, self.dir, self.body = "x", d, {}
            self.store = SimpleNamespace(load_fdt=lambda ref: ref)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def refresh(self):
            pass

        def figure_path(self, title):
            return self.dir / f"{title}.png"
```

(`SimpleNamespace` is imported from `types` at the top of that test function.)

In `tests/test_tool.py`, find:

```python
    def _run_fdt(cfg, *, skip_sanity, confirm_production):
        seen["run_fdt"] = (cfg, skip_sanity, confirm_production)
```

Replace with:

```python
    def _run_fdt(cfg, *, skip_sanity, confirm_production, writer, seed=None):
        seen["run_fdt"] = (cfg, skip_sanity, confirm_production)
        seen["fdt_writer_kind"] = writer.kind
        return SimpleNamespace(id="rec", path=writer.dir)
```

and, after the last `assert seen["run_fdt"] == ("CFG", False, True), \` block, add:

```python
    assert seen["fdt_writer_kind"] == "fdt", \
        "the record is created by the front end and ENTERED by the stage (spec §1.2)"
```

(`SimpleNamespace` is already imported by `tests/test_tool.py`; if it is not, add
`from types import SimpleNamespace` beside the module's other imports.)

- [ ] **Step 12: Run the tests and watch them pass**

Run: `pytest tests/test_artifact_store.py -k "public_entry or callers_config" tests/test_fdt_user.py -v`
then `pytest tests/test_tool.py -k "fdt or crossval" -v`

Expected: PASS.

- [ ] **Step 13: Commit**

```bash
git add core/FDT/fdt_pipeline.py core/tool/fdt.py core/gui/panels/fdt_panel.py tests/test_artifact_store.py tests/test_fdt_user.py tests/test_tool.py
git commit -m "fdt: run_fdt is a public entry that writes a record"
```

#### Amendments (binding — these supersede the text above)

**A0 — read first.** T16 has already moved `plot_psd`; there is no `psd_path` in the closing block when you arrive (P74). When you arrive, `core/FDT/fdt_pipeline.py` also carries:
- T12's `import warnings`, its `thin_notices`/`warn_thin_settings` above `def run_fdt`, and a `# 0.` block (`notices = warn_thin_settings(cfg)`) at the top of the body;
- T13's `prefactor = observable_noise_prefactor(cfg)` just above `# 1.`;
- T15's `_resolved_span`/`Refusal` imports and band refusal;
- T16's `lo_res, hi_res`, its early `psd_path = _out_dir() / f"psd_{timestamp}.png"` + `plot_psd` block before `# 6.`, the line `blanks, of = int(torch.isnan(G_at_omegas).sum()), G_at_omegas.numel()` followed by the all-blank refusal, and a TWO-path closing block.

Call `writer.payload()` / `writer.figure_path()` at the moment of writing, never at the top of the run (P49): `__exit__` keeps a record once either has been called.

**A1 — Step 6, imports (the quoted anchor is not found; T12 inserted `import warnings`).** Find:
```python
import logging
import math
import warnings
from datetime import datetime

import torch

from core import config
from core.config import FDTConfig
```
Replace with:
```python
import logging
import math
import random
import warnings

import torch

from core import config
from core.config import FDTConfig
from core.diagnostics.rng import seeded
from core.runs import public_entry
```
`config` STAYS: T12's `thin_notices` reads `config.FDT_THIN_N_FREQS` / `config.FDT_THIN_ENSEMBLE_M` live, so dropping it is a `NameError` on every run. Only `datetime` goes. Delete the parenthetical under the block. It is false: importing `core.diagnostics.rng` runs `core/diagnostics/__init__.py`, which imports `core.orchestrator`. No cycle results. Say this in the report.

**A2 — Step 6, `_settings_block` (P70).** Use this signature and tail instead of the one given:
```python
def _settings_block(cfg, *, skip_sanity=None, confirm_production=None) -> dict:
    """(docstring as given, plus:) ``skip_sanity`` and ``confirm_production`` are run_fdt's ARGUMENTS,
    not FDTConfig fields, so the caller hands them in; a sweep passes neither and records both null."""
    return {"n_freqs": int(cfg.n_freqs), "ensemble_M": int(cfg.ensemble_M),
            "freqs_per_batch": int(cfg.freqs_per_batch), "F0": float(cfg.F0),
            "freq_bounds": [float(v) for v in cfg.freq_bounds],
            "burn_in_nd": float(cfg.burn_in_nd), "T_obs_periods": int(cfg.T_obs_periods),
            "dt_nd": float(cfg.dt_nd), "psd_T_obs_nd": float(cfg.psd_T_obs_nd),
            "skip_sanity": None if skip_sanity is None else bool(skip_sanity),
            "confirm_production": None if confirm_production is None else bool(confirm_production)}
```
Interfaces, Produces: read `_settings_block(cfg, *, skip_sanity=None, confirm_production=None) -> dict`.

**A3 — Step 7, the body of the new `run_fdt` (P15, P12, P51, T12's pin).** Keep the decorator, the signature and the docstring as given. Replace everything after the docstring, down to `return writer.store.load_fdt(writer.id)`, with:
```python
    seed = _resolve_seed(seed, cfg)
    cfg.seed = seed                                  # on the PRIVATE copy; recorded in the body below
    # 0. The settings too thin to trust (T12, E5): warned now and KEPT in body.notices (P51). HERE, in
    #    the decorated function itself: T12's pin reads inspect.getsource(fdt_pipeline.run_fdt), which
    #    is this function's source (public_entry uses functools.wraps), not _measure's.
    notices = warn_thin_settings(cfg)
    # The first body is UPDATED IN PLACE, never replaced (P15): a front end may already have set the
    # study and the notices (T25's panel does). The stage owns `settings` and the RESOLVED seed.
    body = writer.body
    body.setdefault("study", "single")
    body["settings"] = _settings_block(cfg, skip_sanity=skip_sanity,
                                       confirm_production=confirm_production)
    body["seed"] = seed
    body["notices"] = [*(body.get("notices") or []), *notices]
    for key in ("grid", "points", "offgrid", "compared", "results"):
        body.setdefault(key, None)
    body["complete"] = False
    with writer:
        with seeded(seed, cfg.hw.device):
            _measure(cfg, skip_sanity=skip_sanity, confirm_production=confirm_production,
                     writer=writer)
        writer.body["complete"] = True
    return writer.store.load_fdt(writer.id)
```
Then, in `_measure` (the moved body), find T12's block and DELETE it:
```python
    # 0. The settings too thin to trust: not a refusal (E5 keeps the quick look possible), a
    #    judgement the operator sees now and the record keeps afterwards (Task 17 stores it).
    notices = warn_thin_settings(cfg)

```
`_measure` therefore starts with T13's prefactor comment and call. On the `confirm_production=False` branch `_measure` returns after the sanity verdict, and `run_fdt` returns the record FINISHED with `grid`, `offgrid` and `results` null (P55).

**A4 — Step 8 (P74).** The timestamp, passive-plot and trajectory find/replaces stand. The three-line closing-block find/replace is replaced by these two.

Find (T16's early site, just before `# 6.`):
```python
    psd_path = _out_dir() / f"psd_{timestamp}.png"
```
Replace with:
```python
    psd_path = writer.figure_path("Spontaneous PSD")
```
Find:
```python
    # 9. Plot + save (timestamp set at the top of run_fdt; the PSD went to disk before Campaign 2)
    ratio_path = _out_dir() / f"fdt_ratio_{timestamp}.png"
    chi_path = _out_dir() / f"chi_components_{timestamp}.png"
```
Replace with:
```python
    # 9. Plot + save (the PSD went to disk before Campaign 2)
    ratio_path = writer.figure_path("Effective temperature ratio")
    chi_path = writer.figure_path("Chi components")
```
Also delete the comment line `    # (timestamp set at the top of run_fdt.)` above the trajectory plot. Afterwards `grep -n "_out_dir\|timestamp" core/FDT/fdt_pipeline.py` must print nothing.

**A5 — Step 9.** The `grid` insertion after the peak line stands. Do NOT add a second blank count (drop the `blanks = int(...)` / `offgrid` insertion after `G_at_omegas = ...`). Instead find T16's line:
```python
    blanks, of = int(torch.isnan(G_at_omegas).sum()), G_at_omegas.numel()
```
and add immediately after it, before `if blanks == of:`:
```python
    writer.body["offgrid"] = {"blanks": blanks, "of": int(of)}   # the Campaign-2 probe grid (P57)
```
The results anchor is the TWO-path line (P74). Find:
```python
    log.info(f"Saved plots to:\n  {ratio_path}\n  {chi_path}")
```
Replace with:
```python
    writer.body["results"] = _results_block(omegas, ratio, omega_natural, blanks)
    writer.refresh()
    log.info(f"Saved plots to:\n  {ratio_path}\n  {chi_path}")
```
T18 anchors on the first two lines verbatim. `_results_block` stands as given.

**A6 — Step 10.**

Tool (P79): `--seed` only lands in T28, so read it defensively. The replacement becomes:
```python
    writer = store.create("fdt", cfg)
    rec = fdt_pipeline.run_fdt(cfg, skip_sanity=args.skip_sanity,
                               confirm_production=not args.no_production, writer=writer,
                               seed=getattr(args, "seed", None))
    print(f"[prism fdt] record {rec.id} at {rec.path}")
```
Panel: do NOT replace `from core import cli, config, registry`. T21's and T25's anchors quote that line verbatim, and `config` may stay unused. Instead find:
```python
from core.FDT.fdt_pipeline import run_fdt
```
Replace with:
```python
from core.FDT.fdt_pipeline import run_fdt
from core.artifacts import default_store
```
Keep the guard signature, the `run_fdt(...)` call and the dispatch block exactly as the step writes them: T25 anchors on `writer = default_store().create("fdt", cfg)` and on that dispatch (P80).

**A7 — Step 11, `_LogWriter` at MODULE scope.** T19 uses `_LogWriter` in the SWEEP half of the same logging test, and that half runs before the `_Hopf` half. A class defined inside the function below that point raises UnboundLocalError there. So do not define it inside the test. In `tests/test_fdt_user.py`, add `from types import SimpleNamespace` beside `from pathlib import Path` (the module imports), and add after `class _FakeCfg`:
```python
class _LogWriter:
    """Enough writer for a run whose RECORDS are the subject, not its manifest: a real directory, a
    body, payload and figure paths inside it, and a store that hands back what it was given. A real
    ArtifactWriter would drag a manifest (and its validation) into tests about the pipeline's logs."""
    def __init__(self, d):
        self.id, self.dir, self.body = "x", Path(d), {}
        self.store = SimpleNamespace(load_fdt=lambda ref: ref)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def refresh(self):
        pass

    def payload(self, filename):
        return self.dir / filename

    def figure_path(self, title):
        return self.dir / f"{title}.png"
```
The rest of Step 11's logging-test edit stands: remove the `_out_dir` patch, pass `writer=_LogWriter(tmp_path), seed=1`, widen `_Hopf`, and add `from core import config` to the test's local imports. Your `_Hopf` replacement matches the prefix of T13's version. T13's comment and `params_dict` stay below it and must not be deleted.

**A8 — Step 11, the three science-fix tests (no step updated them; each goes red at T17).** `test_the_prefactor_is_refused_before_anything_is_simulated` (T13), `test_a_band_below_the_spectrums_resolution_refuses_before_the_driven_campaign` (T15) and `test_a_run_that_can_measure_nothing_refuses_and_leaves_the_spectrum_behind` (T16) each patch the deleted `_out_dir` (`monkeypatch.setattr` raises AttributeError) and call `run_fdt` without `writer`. In each:
- delete `monkeypatch.setattr(fdt_pipeline, "_out_dir", lambda: tmp_path)`;
- add `writer=_LogWriter(tmp_path), seed=1` to the `fdt_pipeline.run_fdt(...)` call;
- give the stub config class (`_NoN`, `_Cfg`, `_Cfg`) whichever of these attributes it lacks, never overwriting one it already sets: `n_freqs, ensemble_M, freqs_per_batch, F0 = 2, 8, 1, 0.05`; `freq_bounds, burn_in_nd, T_obs_periods = (0.1, 30.0), 100.0, 30`; `dt_nd, psd_T_obs_nd, seed = 0.01, 8000.0, None`; `hw = config.cpu_device()` with `from core import config` in the test's imports. `_settings_block`, `warn_thin_settings` and `seeded` read these.

In the T16 test, find:
```python
    written = sorted(p.name.rsplit("_", 2)[0] for p in tmp_path.glob("*.png"))
    assert written == ["psd", "spontaneous_trajectory"], written
```
Replace with:
```python
    written = sorted(p.name for p in tmp_path.glob("*.png"))
    assert written == ["Spontaneous PSD.png", "Spontaneous trajectory.png"], written
```
Leave every refusal, field and message assertion exactly as T13, T15 and T16 left it.

**A9 — Step 11, `tests/test_tool.py`.** The recorders return the sentinel `"CFG"`, and the handler now calls `store.create("fdt", "CFG")`, which runs `mf.config_from_cfg("CFG")`. T7's FDT branch then raises AttributeError on `cfg.model`. In `test_fdt_and_crossval_flags_reach_their_builders`, after its four `monkeypatch.setattr(...)` lines, add:
```python
    # The builders are recorders returning the sentinel "CFG", and the handlers now open an fdt record
    # on it; the record's config block is not this test's subject (T34's real run covers it).
    from core.artifacts import manifest as mf
    monkeypatch.setattr(mf, "config_from_cfg", lambda cfg: {})
```
In `test_fdt_ctrl_c_gets_its_own_interrupt_note`, change `def _boom(cfg, *, skip_sanity, confirm_production):` to `def _boom(cfg, *, skip_sanity, confirm_production, writer, seed=None):`, and add the same two lines after `monkeypatch.setattr(fdt_pipeline, "run_fdt", _boom)`. T29 rewrites only that test's assertions.

**A10 — P73, `tests/test_nav_and_gating.py`.** In `test_fdt_panel_guard_translates_model_error_and_gate_admits_builtins`:
- change `def boom(cfg, *, skip_sanity, confirm_production):` and `def missing(cfg, *, skip_sanity, confirm_production):` to take `(cfg, *, skip_sanity, confirm_production, writer, seed=None)`;
- change both `fdt_panel._run_fdt_guarded(Cfg(), skip_sanity=True, confirm_production=False)` calls to `fdt_panel._run_fdt_guarded(Cfg(), skip_sanity=True, confirm_production=False, writer=None)`.

**A11 — Step 4, the new tests.**

(a) `_stub_campaigns`: the spectrum must resolve the band around its peak, or T15's band refusal fires. With `linspace(0, 4, 16)` the peak is 1.067, the lowest probe 0.107 and the first real bin 0.267. Change `def _stub_campaigns(monkeypatch, n_psd=16):` to `def _stub_campaigns(monkeypatch, n_psd=1601):`, and `omegas_psd = torch.linspace(0.0, 4.0, n_psd, dtype=torch.float64)` to `omegas_psd = torch.linspace(0.0, 40.0, n_psd, dtype=torch.float64)`. That puts the peak at 1.0 with bins every 0.025 up to 40, so the 0.1..30 probe grid is fully resolved.

(b) Figure names: `figure_path` uses `store.slug`, which gives underscores. Replace the expected list with `["figures/chi_components.png", "figures/effective_temperature_ratio.png", "figures/spontaneous_psd.png", "figures/spontaneous_trajectory.png"]`. In the failed-run test, use `w.dir / "figures" / "spontaneous_trajectory.png"`.

(c) In the round-trip test, replace `assert rec.body["notices"] == [], "no notices is an empty list, never null (spec §2.3)"` with:
```python
    assert rec.body["notices"] == fdt_pipeline.thin_notices(cfg) and len(rec.body["notices"]) == 1, \
        "ensemble_M=2 is below FDT_THIN_ENSEMBLE_M: T12's sentence is KEPT in the record (P51)"
    assert rec.body["offgrid"]["blanks"] == 0
    assert rec.body["settings"]["confirm_production"] is True, "P70"
```

(d) P82: add after the seed test:
```python
def test_the_seed_determines_the_numbers(store, monkeypatch):
    """P82 / E7. Recording a seed is worth nothing unless the seed DETERMINES what the run draws --
    and compare repeats (E8) reads the spread across repeats as the measurement error on exactly that
    premise. Campaign 1 is stubbed to draw from the ambient torch generator, which is what the solver
    draws its noise from: one seed twice draws the same, another seed draws differently."""
    import torch
    from core import cli, config
    from core.FDT import fdt_pipeline

    _stub_campaigns(monkeypatch)
    stub_c1 = fdt_pipeline.run_campaign1_psd
    drawn = []

    def _c1(cfg, return_trajectory=False):
        drawn.append(float(torch.rand(())))
        return stub_c1(cfg, return_trajectory=return_trajectory)

    monkeypatch.setattr(fdt_pipeline, "run_campaign1_psd", _c1)
    cfg = cli.make_fdt_config("HOPF", False, str(config.CELL_PATH / "hopf" / "cell.txt"),
                              n_freqs=3, ensemble_M=8)
    for seed in (5, 5, 6):
        fdt_pipeline.run_fdt(cfg, skip_sanity=True, confirm_production=True,
                             writer=store.create("fdt", cfg), seed=seed)
    assert drawn[0] == drawn[1], "one seed, one draw: the seed determines the numbers"
    assert drawn[0] != drawn[2], "a different seed draws differently"
```
Step 5's `-k` becomes `"record or seed or unfinished or determines"`.

**A12 — Step 12 commands.** `-k` applies to every path on the line, so the first command as written deselects all of `tests/test_fdt_user.py`. Run instead:
- `pytest tests/test_artifact_store.py -k "public_entry or callers_config" -v`
- `pytest tests/test_fdt_user.py tests/test_conditioning_repair.py::test_the_prompt_cli_is_retired "tests/test_nav_and_gating.py::test_fdt_panel_guard_translates_model_error_and_gate_admits_builtins" -v`
- `pytest tests/test_tool.py -k "fdt or crossval" -v`

**A13 — Files block and Step 13.** Add `tests/test_nav_and_gating.py` (P73) to the Files block. Step 13's `git add` line becomes:
```bash
git add core/FDT/fdt_pipeline.py core/tool/fdt.py core/gui/panels/fdt_panel.py tests/test_artifact_store.py tests/test_fdt_user.py tests/test_tool.py tests/test_nav_and_gating.py
```

#### Controller rulings (pre-flight, binding — these supersede the amendments above and the task text)

- **F38** — import `seeded` LOCALLY inside the function that uses it, not at module scope. Importing `core.diagnostics.rng` runs `core/diagnostics/__init__.py`, which imports the diagnostics and through them `core.orchestrator`; Task 12 already imports orchestrator locally for the same reason.
- **F27** — run `pytest tests/test_fdt_user.py -v` UNFILTERED as its own command. pytest keeps only the last `-k`, and a single `-k` applies to every file on the command line, so a combined command silently deselects exactly Tasks 13, 15 and 16's tests, which this task must convert.
- **F13** — confirmed: the handler passes `seed=getattr(args, "seed", None)`; `--seed` does not exist until Task 28.
- **F11** — confirmed: `warn_thin_settings(cfg)` is called in `run_fdt` before the first body is built, so the first manifest already carries the notices and Task 12's source pin on `run_fdt` holds.

---

### Task 18: the single-cell numbers into `data.h5`

**Why:** E1 and spec §2.3, §3.8 — the frequency grid, the spontaneous spectrum with its own frequency
axis, the complex susceptibility, the ratio and the normalisation prefactor are computed, plotted and
today discarded. Written into the record they make §7's comparison possible at all, and a re-plot no
longer needs a re-run. §2.3 requires a `study` attribute at the file's root so a reader can tell which
of the three layouts it holds.

**Files:**
- Modify: `core/FDT/fdt_pipeline.py` — `_measure`'s step 8, and one new module-private writer
- Test: `tests/test_fdt_user.py`

**Interfaces:**
- Consumes, from T17: `_measure(cfg, *, skip_sanity, confirm_production, writer)`, `writer.payload`,
  `writer.refresh()`.
- Produces, for T36: `data.h5` for `study == "single"` — root attrs `study`, `model`, `omega_0`,
  `omega_0_source`, `prefactor`, `freq_bounds`, `n_freqs`; datasets `omegas` (float64),
  `T_eff_over_T` (float64), `chi` (complex128), `PSD_omegas` (float64), `PSD_G` (float64). The
  spontaneous datasets keep the names the sweep's layout already uses, so one reader understands both.

- [ ] **Step 1: Write the failing test**

In `tests/test_fdt_user.py`, add:

```python
def test_the_single_cell_record_holds_the_numbers_not_only_the_pictures(store, monkeypatch):
    """E1 / spec §3.8. Every number the four figures draw is recoverable from the record: before piece
    5 the grid, the spectrum, the susceptibility and the ratio existed only inside the run, so a
    re-plot -- and any comparison of two cells -- meant re-running hours of simulation. The ``study``
    attribute at the root is what lets a reader check the layout before reading it: three layouts
    share this filename (spec §2.3)."""
    import h5py
    import numpy as np
    from core import cli, config
    from core.FDT import fdt_pipeline

    _stub_campaigns(monkeypatch)
    cfg = cli.make_fdt_config("HOPF", False, str(config.CELL_PATH / "hopf" / "cell.txt"),
                              n_freqs=5, ensemble_M=2)
    rec = fdt_pipeline.run_fdt(cfg, skip_sanity=True, confirm_production=True,
                               writer=store.create("fdt", cfg, name="numbers"), seed=2)

    assert rec.data_path == rec.path / "data.h5" and rec.data_path.exists()
    assert rec.manifest.payloads["data.h5"], "a committed payload is hashed (spec §2.2)"
    with h5py.File(rec.data_path, "r") as h5:
        assert h5.attrs["study"] == "single"
        assert h5.attrs["model"] == "HOPF"
        assert float(h5.attrs["prefactor"]) == 2.0 / cfg.params_dict["sigma_x"][0] ** 2
        assert h5["omegas"].shape == (5,) and h5["T_eff_over_T"].shape == (5,)
        assert h5["chi"].dtype == np.complex128 and h5["chi"].shape == (5,)
        assert h5["PSD_omegas"].shape == h5["PSD_G"].shape, \
            "the spontaneous spectrum carries its OWN frequency axis -- it is not on the chi grid"
        assert h5["PSD_omegas"].shape != h5["omegas"].shape
```

- [ ] **Step 2: Run it and watch it fail**

Run: `pytest tests/test_fdt_user.py -k numbers -v`

Expected: FAIL with `AssertionError: assert None == WindowsPath('.../data.h5')` on the `rec.data_path`
line — `load_fdt` reports `None` because the run never wrote a payload.

- [ ] **Step 3: Write the file**

In `core/FDT/fdt_pipeline.py`, find:

```python
import logging
import math
import random

import torch
```

Replace with:

```python
import logging
import math
import random

import h5py
import numpy as np
import torch
```

Add beside `_results_block`:

```python
def _write_single_h5(path, cfg, omegas, ratio, chis, freqs_psd, G, omega_natural, prefactor) -> None:
    """The single-cell layout of ``data.h5`` (spec §2.3, §3.8).

    ``study`` sits at the ROOT so a reader can check the layout before reading a dataset: a sweep's
    file and a comparison's file carry the same name inside their own records. The spontaneous
    spectrum keeps its OWN frequency axis -- it is a Welch grid, not the log-spaced chi grid, and
    interpolating one onto the other is exactly the step §3.5's off-grid fix made honest.
    """
    def _f64(t):
        return t.detach().cpu().numpy().astype(np.float64)

    with h5py.File(path, "w") as h5:
        h5.attrs["study"] = "single"
        h5.attrs["model"] = cfg.model
        h5.attrs["omega_0"] = float(omega_natural)
        h5.attrs["omega_0_source"] = "spectrum peak"
        h5.attrs["prefactor"] = float(prefactor)
        h5.attrs["n_freqs"] = int(cfg.n_freqs)
        h5.attrs["freq_bounds"] = np.asarray(cfg.freq_bounds, dtype=np.float64)
        h5.create_dataset("omegas", data=_f64(omegas), compression="gzip")
        h5.create_dataset("T_eff_over_T", data=_f64(ratio), compression="gzip")
        h5.create_dataset("chi", data=chis.detach().cpu().numpy().astype(np.complex128),
                          compression="gzip")
        h5.create_dataset("PSD_omegas", data=_f64(freqs_psd), compression="gzip")
        h5.create_dataset("PSD_G", data=_f64(G), compression="gzip")
```

In `_measure`, find:

```python
    writer.body["results"] = _results_block(omegas, ratio, omega_natural, blanks)
    writer.refresh()
```

Replace with:

```python
    _write_single_h5(writer.payload("data.h5"), cfg, omegas, ratio, chis, freqs_psd, G,
                     omega_natural, prefactor)
    writer.body["results"] = _results_block(omegas, ratio, omega_natural, blanks)
    writer.refresh()
```

(`prefactor` is the name T13's moved `observable_noise_prefactor(cfg)` call binds at the top of the
body. If T13 named it otherwise, use that name.)

- [ ] **Step 4: Run the tests and watch them pass**

Run: `pytest tests/test_fdt_user.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add core/FDT/fdt_pipeline.py tests/test_fdt_user.py
git commit -m "fdt: the single-cell numbers go into the record's data.h5"
```

#### Amendments (binding — these supersede the text above)

**A1 — Step 3, the import anchor (not found after T17).** T17 keeps T12's `import warnings` between `import random` and `import torch`. Find:
```python
import random
import warnings

import torch
```
Replace with:
```python
import random
import warnings

import h5py
import numpy as np
import torch
```

**A2 — Step 3, the dataset names (P71, P5).** The single-cell layout is the Interface contract's. In `_write_single_h5`, replace the two lines that create `"omegas"` and `"chi"`:
```python
        h5.create_dataset("omegas", data=_f64(omegas), compression="gzip")
```
```python
        h5.create_dataset("chi", data=chis.detach().cpu().numpy().astype(np.complex128),
                          compression="gzip")
```
with:
```python
        h5.create_dataset("omega_grid", data=_f64(omegas), compression="gzip")
```
```python
        h5.create_dataset("chi_prime", data=_f64(chis.real), compression="gzip")
        h5.create_dataset("chi_double_prime", data=_f64(chis.imag), compression="gzip")
```
The root attributes `study`, `omega_0` and `prefactor` stay; `model`, `omega_0_source`, `n_freqs` and `freq_bounds` are permitted extras. In the docstring, say the names are `cross_validation._fdt_measure`'s and the sweep file's own vocabulary (`omega_grid`, `T_eff_over_T`, `chi_prime`, `chi_double_prime`, `PSD_omegas`, `PSD_G`), which T36 reads. Interfaces, Produces for T36: read "root attrs `study`, `omega_0`, `prefactor` (+ `model`, `omega_0_source`, `n_freqs`, `freq_bounds`); datasets `omega_grid`, `T_eff_over_T`, `chi_prime`, `chi_double_prime`, `PSD_omegas`, `PSD_G`, all float64".

**A3 — Step 1, the test.** Find:
```python
        assert h5["omegas"].shape == (5,) and h5["T_eff_over_T"].shape == (5,)
        assert h5["chi"].dtype == np.complex128 and h5["chi"].shape == (5,)
```
Replace with:
```python
        assert h5["omega_grid"].shape == (5,) and h5["T_eff_over_T"].shape == (5,)
        assert h5["chi_prime"].dtype == np.float64 and h5["chi_prime"].shape == (5,)
        assert h5["chi_double_prime"].shape == (5,)
        assert np.allclose(h5["chi_double_prime"][...], 1.0), "the stub's chi is 1+1j"
        assert float(h5.attrs["omega_0"]) == rec.body["grid"]["omega_0"]
```
and change `assert h5["PSD_omegas"].shape != h5["omegas"].shape` to `assert h5["PSD_omegas"].shape != h5["omega_grid"].shape`. With T17's amended `_stub_campaigns` (1601 Welch bins, 5 probes), both shape assertions hold.

The rest of the task stands: the file is written at the moment of writing, after both campaigns (P49), and `rec.manifest.payloads["data.h5"]` is the hash the commit writes (P47).

---

### Task 19: the sweep writes two records, under one seed, as a public entry

**Why:** Spec §4.1 and §4.2 — a two-parameter study writes TWO records, one per swept parameter (E4:
each sweep is answerable on its own, and an all-failed activity sweep no longer costs the temperature
sweep). `run_fdt_param_sweep` takes the open writer instead of an `output_path` and enters it, as
§3.1's run does. One seed is drawn once and recorded on BOTH records (E7), with each operating point
deriving its own stream from it. The midpoint-plot special case disappears: each sweep plots into its
own record when it finishes.

**Files:**
- Modify: `core/FDT/cross_validation.py:44-46` (`_out_dir`), `:158-301` (`run_fdt_param_sweep`),
  `:304-339` (`run_param_study_cli`)
- Modify: `core/FDT/cross_validation_plots.py:20-22` (`_plot_dir`), `:71-80` and `:131-141`
  (`plot_fdt_3d_vs_param`'s destination)
- Modify: `core/tool/fdt.py:147-161` (`run_crossval`), `core/gui/panels/crossval_panel.py:140-165`
- Test: `tests/test_artifact_store.py` (the scan and its two companion sets, again)
- Test: `tests/test_fdt_user.py` (the sweep's logging test, which calls `run_fdt_param_sweep` directly)
- Test: `tests/test_tool.py:1188-1190,1238` (the `_study` recorder)

**Interfaces:**
- Consumes, from T17: `_resolve_seed(seed, cfg)`, `_settings_block(cfg)`, and the writer contract
  (create outside, enter inside).
- Produces, for T20/T26/T28/T36:
  - `run_param_study_cli(cfg, *, s_grid, t_grid, writers: dict, seed=None) -> list[LoadedFdt]`,
    `writers == {"s": ArtifactWriter, "temp": ArtifactWriter}`
  - `run_fdt_param_sweep(cfg, sweep_param, sweep_grid, fixed_overrides=None, *, writer) -> LoadedFdt`
    — reads its seed from `cfg.seed`, writes nothing on `cfg`
  - `plot_fdt_3d_vs_param(records, param_symbol, title, save_path, ...)` — `filename_tag` and `save`
    are gone; the caller names the destination.

- [ ] **Step 1: Add `run_param_study_cli` to the scan and its two companion sets**

In `tests/test_artifact_store.py`, find:

```python
    want |= {("core/FDT/fdt_pipeline.py", "run_fdt")}
```

Replace with:

```python
    want |= {("core/FDT/fdt_pipeline.py", "run_fdt"),
             ("core/FDT/cross_validation.py", "run_param_study_cli")}
```

Find:

```python
    "run_fdt": _leg_run_fdt,
}
```

Replace with:

```python
    "run_fdt": _leg_run_fdt,
    "run_param_study_cli": _leg_run_param_study_cli,
}
```

Find:

```python
    "run_fdt": "cell",                               # a cell with no FDT normalisation constant (§3.4)
}
```

Replace with:

```python
    "run_fdt": "cell",                               # a cell with no FDT normalisation constant (§3.4)
    "run_param_study_cli": "cell",                   # the same check, before the first phase's spend
}
```

- [ ] **Step 2: Write the study's leg**

In `tests/test_artifact_store.py`, after `_leg_run_fdt`, add:

```python
def _leg_run_param_study_cli(case, monkeypatch, tmp_path):
    """The study's three endings. Its own write on its working config is ``cfg.seed = ...`` -- the one
    integer both records must agree on -- made before either sweep runs, so the seam that raises
    _Injected is the first sweep. Its pre-spend refusal is §3.4's normalisation check, applied to the
    sweep for the same reason it is applied to the single run: a cell missing n or beta would
    otherwise cost the whole first phase before anything noticed."""
    import numpy as np
    from core.FDT import cross_validation as cv
    cfg = _fdt_cfg_for_leg(case)
    cfg.model = "HOPF"

    def _sweep(c, sweep_param, sweep_grid, fixed_overrides=None, *, writer):
        if case == "boom":
            _raise_injected()
        return f"REC-{sweep_param}"

    monkeypatch.setattr(cv, "run_fdt_param_sweep", _sweep)
    writers = {"s": _FdtWriter(tmp_path / "s"), "temp": _FdtWriter(tmp_path / "t")}
    return cfg, lambda: cv.run_param_study_cli(cfg, s_grid=np.array([0.0, 0.1]),
                                               t_grid=np.array([1.0, 1.1]),
                                               writers=writers, seed=5)
```

- [ ] **Step 3: Run the two pins and watch them fail**

Run: `pytest tests/test_artifact_store.py -k "public_entry or callers_config" -v`

Expected: FAIL — the scan reports
`missing [('core/FDT/cross_validation.py', 'run_param_study_cli')]; unexpected []`, and the three
`[run_param_study_cli-*]` cases fail with
`TypeError: run_param_study_cli() got an unexpected keyword argument 'writers'`.

- [ ] **Step 4: Write the two-record test**

In `tests/test_fdt_user.py`, add:

```python
def test_a_study_writes_one_record_per_swept_parameter_under_one_seed(store, monkeypatch, tmp_path):
    """Spec §4.1 (E4, E7). Today the study runs the S sweep, plots it, then runs the T sweep, and an
    all-failed S sweep raises before the T sweep even starts -- one listing entry for two
    measurements, and the second measurement hostage to the first. Two records make each sweep
    answerable on its own.

    ONE seed is drawn once and recorded on BOTH, because the two sweeps are one study: a reader who
    wants to repeat the study repeats it, not half of it."""
    import numpy as np
    import torch
    from core import cli, config
    from core.FDT import cross_validation as cv

    monkeypatch.setattr(cv, "run_campaign1_psd",
                        lambda c: (torch.linspace(0.1, 3.0, 8, dtype=torch.float64),
                                   torch.ones(8, dtype=torch.float64)))
    monkeypatch.setattr(cv, "_detect_resonance", lambda omegas, G, w0: (1.0, True))
    monkeypatch.setattr(cv, "_campaign2_ratio",
                        lambda c, om, f, g: (torch.ones(om.shape, dtype=torch.complex128),
                                             torch.full(om.shape, 2.0, dtype=torch.float64)))
    monkeypatch.setattr(cv, "plot_fdt_3d_vs_param", lambda *a, save_path=None, **k:
                        Path(save_path).write_bytes(b"\x89PNG"))

    cfg, s_grid, t_grid = cli.make_param_sweep_config(
        str(config.CELL_PATH / "nadrowski" / "master_spont.txt"),
        preset=dict(cli.SWEEP_PRESETS["exploratory"]), preset_name="exploratory",
        s_spec=(0.0, 0.1, 2), t_spec=(1.0, 1.1, 2), n_freqs=3, ensemble_M=2)
    writers = {"s": store.create("fdt", cfg, name="sweep_s"),
               "temp": store.create("fdt", cfg, name="sweep_t")}
    recs = cv.run_param_study_cli(cfg, s_grid=s_grid, t_grid=t_grid, writers=writers, seed=77)

    assert [r.name for r in recs] == ["sweep_s", "sweep_t"]
    assert [r.body["points"]["param"] for r in recs] == ["s", "temp"]
    assert {r.body["seed"] for r in recs} == {77}, "one study, one seed (spec §4.1)"
    for r in recs:
        assert r.body["study"] == "sweep" and r.body["complete"] is True
        assert r.body["grid"] is None, "each point's grid is in data.h5, not in the body (§2.3)"
        assert r.body["settings"]["preset"] == "exploratory"
        assert r.data_path.exists() and r.manifest.figures, \
            "each sweep plots into its OWN record when it finishes -- no midpoint special case"
```

- [ ] **Step 5: Run it and watch it fail**

Run: `pytest tests/test_fdt_user.py -k study -v`

Expected: FAIL with `TypeError: run_param_study_cli() got an unexpected keyword argument 'writers'`.

- [ ] **Step 6: Let the plotter be told where to save**

In `core/FDT/cross_validation_plots.py`, find and DELETE:

```python
def _plot_dir():
    """Where the crossval 3D plots are saved: <artifacts root>/crossval."""
    return config.artifacts_root() / "crossval"

```

Find:

```python
from __future__ import annotations
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from .. import config
```

Replace with:

```python
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
```

Find:

```python
    filename_tag: str,
    omega_norm_max: float = _OMEGA_NORM_MAX,
    z_clip: tuple[float, float] = (0.0, 2.0),
    save: bool = True,
    show: bool = False,
) -> Path | None:
```

Replace with:

```python
    save_path,
    omega_norm_max: float = _OMEGA_NORM_MAX,
    z_clip: tuple[float, float] = (0.0, 2.0),
    show: bool = False,
) -> Path | None:
```

Find:

```python
    :param filename_tag: prefix for the saved PNG.
```

Replace with:

```python
    :param save_path: where to write the PNG -- since piece 5 a path inside the sweep's own record
                      (``writer.figure_path(...)``), which is why this function no longer builds one.
                      None draws without saving.
```

Find:

```python
    if save:
        plot_dir = _plot_dir()
        plot_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = plot_dir / f"{filename_tag}_{stamp}.png"
        fig.savefig(out, dpi=160, bbox_inches="tight")
        if not show:
            plt.close(fig)
        return out
```

Replace with:

```python
    if save_path is not None:
        out = Path(save_path)
        fig.savefig(out, dpi=160, bbox_inches="tight")
        if not show:
            plt.close(fig)
        return out
```

(`Path` is already imported by that module's return annotation; if it is not, add
`from pathlib import Path`.)

- [ ] **Step 7: `run_fdt_param_sweep` takes the writer**

In `core/FDT/cross_validation.py`, find and DELETE:

```python
def _out_dir() -> Path:
    """Where the FDT parameter-sweep study saves its HDF5 output: <artifacts root>/crossval."""
    return config.artifacts_root() / "crossval"

```

Find:

```python
from .. import config
from ..config import FDTConfig
from .campaigns import run_campaign1_psd, run_campaign2_chi, observable_noise_prefactor
from .spectral import gen_freqs_log, eff_temp_ratio
from .sanity import _interp_log
from .fdt_pipeline import _estimate_omega_0
```

Replace with:

```python
from ..config import FDTConfig
from ..runs import public_entry
from .campaigns import run_campaign1_psd, run_campaign2_chi, observable_noise_prefactor
from .spectral import eff_temp_ratio
from .sanity import _interp_log
from .fdt_pipeline import _estimate_omega_0, _resolve_seed, _settings_block
from .cross_validation_plots import plot_fdt_3d_vs_param
```

(`gen_freqs_log` and `config` lose their last users in this file at Step 9/Task 20;
`plot_fdt_3d_vs_param` moves to the top because each sweep now plots itself.)

Find:

```python
    fixed_overrides: dict | None = None,
    *,
    output_path: Optional[Path] = None,
) -> Path:
```

Replace with:

```python
    fixed_overrides: dict | None = None,
    *,
    writer,
) -> "object":
```

Find:

```python
    :param output_path: target .h5. Defaults to <artifacts root>/crossval/sweep_<param>_<stamp>.h5.
    :returns: the HDF5 output path.
    """
    fixed_overrides = fixed_overrides or {}
    if output_path is None:
        out_dir = _out_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = out_dir / f"sweep_{sweep_param}_{stamp}.h5"

    fixed_str = ", ".join(f"{k}={v}" for k, v in fixed_overrides.items())

    with h5py.File(output_path, "w") as h5:
```

Replace with:

```python
    :param writer: the OPEN-BUT-NOT-ENTERED ArtifactWriter for this sweep's own record. THIS function
                   enters it, so __enter__, every refresh() and __exit__ run on the thread whose run
                   log becomes log.txt (spec §1.2, §4.1). Its seed comes from ``cfg.seed``; nothing
                   here writes on ``cfg``.
    :returns: the LoadedFdt for the record just written.
    """
    fixed_overrides = fixed_overrides or {}
    seed = _resolve_seed(None, cfg)
    fixed_str = ", ".join(f"{k}={v}" for k, v in fixed_overrides.items())
    n_points = int(len(sweep_grid))
    writer.body = {
        "study": "sweep", "settings": _settings_block(cfg), "seed": seed, "grid": None,
        "points": {"param": sweep_param, "planned": n_points, "done": 0, "failed": 0},
        "offgrid": None, "notices": [], "compared": None, "complete": False, "results": None}
    # NESTED, not combined: the HDF5 file must be CLOSED before the sweep's own plot reopens it with
    # load_param_sweep at the end of Step 8, and that plot must still be drawn inside the writer so
    # its PNG lands in the record's figures/.
    with writer:
      with h5py.File(writer.payload("data.h5"), "w") as h5:
```

**Indentation:** the whole former body of the `with h5py.File(...)` block keeps its current indent and
sits under the inner `with`; re-indent the inner `with` and its body to the normal four spaces once
the outer `with writer:` is in place (the two-space form above is only to show the nesting).

Also rewrite the paragraph of the docstring that begins `DELIBERATELY NOT an atomic write` — find its
last sentence:

```python
    it up front -- that, not a torn write, is the way this function can lose data.
```

Replace with:

```python
    it up front -- that, not a torn write, is the way this function can lose data. SINCE PIECE 5 the
    file lives inside the record (``writer.payload("data.h5")``) and E2 is what makes the incremental
    write safe: an interrupted sweep keeps its folder, so the partial file is listed and deletable
    rather than invisible.
```

- [ ] **Step 8: Seed each operating point, and finish into the record**

In `core/FDT/cross_validation.py`, find (Phase A's per-point body):

```python
            cfg_op = cfg.with_overrides(**{sweep_param: float(value), **fixed_overrides})
            omega_0_lin, _ = _estimate_omega_0(cfg_op)
            freqs_psd, G = run_campaign1_psd(cfg_op)
            w0, res = _detect_resonance(freqs_psd, G, omega_0_lin)
```

Replace with:

```python
            cfg_op = cfg.with_overrides(**{sweep_param: float(value), **fixed_overrides})
            omega_0_lin, _ = _estimate_omega_0(cfg_op)
            # One stream per operating point, derived from the study's single seed (spec §4.1): a
            # point is reproducible from the seed and its index, and seeding once for the whole sweep
            # would make point k's draw depend on how many points preceded it.
            with seeded(seed + idx, cfg.hw.device):
                freqs_psd, G = run_campaign1_psd(cfg_op)
            w0, res = _detect_resonance(freqs_psd, G, omega_0_lin)
```

Find (Phase B's call):

```python
            try:
                chis, ratio = _campaign2_ratio(cfg_op, omegas_common, freqs_psd, G)
```

Replace with:

```python
            try:
                with seeded(seed + n_points + idx, cfg.hw.device):
                    chis, ratio = _campaign2_ratio(cfg_op, omegas_common, freqs_psd, G)
```

and add `from ..diagnostics.rng import seeded` to the import block edited in Step 7.

Find the tail of the function:

```python
    n_points = len(sweep_grid)
    if n_failed == n_points:
        raise RuntimeError(
            f"{sweep_param} sweep produced NO usable points: all {n_points} operating points failed "
            f"in Campaign 2 (per-point reasons are in the HDF5 'error' attrs). This is almost always "
            f"one systematic cause -- a CUDA OOM repeating identically at every point -- not "
            f"{n_points} independent failures. {output_path} contains the PSDs but no response data.")
    if n_failed:
        warnings.warn(
            f"{sweep_param} sweep: {n_failed}/{n_points} operating points failed in Campaign 2 and "
            f"carry no response data. See their 'error' attrs in {output_path}.", stacklevel=2)
    log.info(f"{sweep_param} sweep complete ({n_points - n_failed}/{n_points} points). "
             f"Saved to: {output_path}")
    return output_path
```

Replace with (still inside the `with`, at the same indentation as the Phase B loop — Task 20 rewrites
the all-failed branch; this step only moves the tail into the record):

```python
        if n_failed:
            warnings.warn(
                f"{sweep_param} sweep: {n_failed}/{n_points} operating points failed in Campaign 2 "
                f"and carry no response data. See their 'error' attrs in {writer.dir}.", stacklevel=2)
        if n_failed == n_points:
            raise RuntimeError(
                f"{sweep_param} sweep produced NO usable points: all {n_points} operating points "
                f"failed in Campaign 2 (per-point reasons are in the HDF5 'error' attrs). This is "
                f"almost always one systematic cause -- a CUDA OOM repeating identically at every "
                f"point -- not {n_points} independent failures. {writer.dir} contains the PSDs but "
                f"no response data.")
        log.info(f"{sweep_param} sweep complete ({n_points - n_failed}/{n_points} points). "
                 f"Saved to: {writer.dir}")
    plot_fdt_3d_vs_param(load_param_sweep(writer.payload("data.h5")),
                         param_symbol=_PARAM_SYMBOL[sweep_param], title=_PARAM_TITLE[sweep_param],
                         save_path=writer.figure_path(f"FDT ratio vs {sweep_param}"))
    writer.body["complete"] = True
    writer.refresh()
    return writer.store.load_fdt(writer.id)
```

**Placement:** the `plot_fdt_3d_vs_param(...)`, `writer.body["complete"] = True` and
`writer.refresh()` lines sit inside `with writer:` but OUTSIDE the inner `with h5py.File(...)` that
Step 7 nested — the file must be closed before `load_param_sweep` reopens it, and the figure must be
drawn while the writer is still entered so the PNG lands in the record's `figures/`. Only
`return writer.store.load_fdt(writer.id)` is outside both.

Add beside the module's other constants:

```python
# The two canonical sweeps' labels, which used to live inline in run_param_study_cli's two plot calls.
_PARAM_SYMBOL = {"s": r"$S$", "temp": r"$T_a/T$"}
_PARAM_TITLE = {"s": r"FDT ratio vs $(\tilde\omega/\Omega_0,\ S)$  ($T_a/T=1$)",
                "temp": r"FDT ratio vs $(\tilde\omega/\Omega_0,\ T_a/T)$  ($S=0$)"}
```

- [ ] **Step 9: `run_param_study_cli` becomes the public entry over two writers**

In `core/FDT/cross_validation.py`, find:

```python
def run_param_study_cli(cfg: FDTConfig, s_grid: np.ndarray, temp_grid: np.ndarray) -> tuple[Path, Path]:
    """
    CLI entry: run the S sweep (T_a/T=1), plot it, then run the T sweep (S=0) and
    plot it. Each sweep's 3D plot is saved as soon as that sweep finishes -- so a
    long study gives you the S-sweep plot at its midpoint rather than only at the
    very end. Returns (s_h5_path, temp_h5_path).
    """
    from .cross_validation_plots import plot_fdt_3d_vs_param

    # --- S sweep, then plot immediately ---
    log.info("#" * 64)
    log.info("# S sweep:  vary S, hold T_a/T = 1   (FDT restored as S -> 0)")
    log.info("#" * 64)
    s_path = run_fdt_param_sweep(cfg, sweep_param="s", sweep_grid=s_grid,
                                 fixed_overrides={"temp": 1.0})
    log.info("Plotting S sweep...")
    p1 = plot_fdt_3d_vs_param(load_param_sweep(s_path), param_symbol=r"$S$",
                              title=r"FDT ratio vs $(\tilde\omega/\Omega_0,\ S)$  ($T_a/T=1$)",
                              filename_tag="fdt3d_vs_S")
    if p1:
        log.info(f"  Saved S-sweep plot: {p1}")

    # --- T sweep, then plot immediately ---
    log.info("#" * 64)
    log.info("# T sweep:  vary T_a/T, hold S = 0   (FDT restored as T_a/T -> 1)")
    log.info("#" * 64)
    temp_path = run_fdt_param_sweep(cfg, sweep_param="temp", sweep_grid=temp_grid,
                                    fixed_overrides={"s": 0.0})
    log.info("Plotting T sweep...")
    p2 = plot_fdt_3d_vs_param(load_param_sweep(temp_path), param_symbol=r"$T_a/T$",
                              title=r"FDT ratio vs $(\tilde\omega/\Omega_0,\ T_a/T)$  ($S=0$)",
                              filename_tag="fdt3d_vs_T")
    if p2:
        log.info(f"  Saved T-sweep plot: {p2}")

    return s_path, temp_path
```

Replace with:

```python
@public_entry
def run_param_study_cli(cfg: FDTConfig, *, s_grid: np.ndarray, t_grid: np.ndarray,
                        writers: dict, seed=None) -> list:
    """The two canonical sweeps, as TWO records (spec §4.1).

    One record per swept parameter, each complete on its own, because an all-failed S sweep used to
    raise before the T sweep had started -- two measurements held hostage to one. One SEED, drawn once
    when none is given and recorded on both, because the study is one experiment: repeating half of it
    at a fresh seed answers a different question.

    :param writers: {"s": ArtifactWriter, "temp": ArtifactWriter} -- open, not entered. Each sweep
                    enters its own.
    :returns: [LoadedFdt for the S sweep, LoadedFdt for the T sweep].
    """
    # §3.4's check, applied to the sweep for the same reason: a cell that cannot supply the
    # normalisation constant would otherwise cost the whole first phase before anything noticed.
    observable_noise_prefactor(cfg)
    cfg.seed = _resolve_seed(seed, cfg)              # on the PRIVATE copy; both sweeps read it

    log.info("#" * 64)
    log.info("# S sweep:  vary S, hold T_a/T = 1   (FDT restored as S -> 0)")
    log.info("#" * 64)
    s_rec = run_fdt_param_sweep(cfg, sweep_param="s", sweep_grid=s_grid,
                                fixed_overrides={"temp": 1.0}, writer=writers["s"])

    log.info("#" * 64)
    log.info("# T sweep:  vary T_a/T, hold S = 0   (FDT restored as T_a/T -> 1)")
    log.info("#" * 64)
    temp_rec = run_fdt_param_sweep(cfg, sweep_param="temp", sweep_grid=t_grid,
                                   fixed_overrides={"s": 0.0}, writer=writers["temp"])
    return [s_rec, temp_rec]
```

Add `"preset"` to each record's settings: in `run_fdt_param_sweep`, immediately after
`"settings": _settings_block(cfg)` is built, the body line becomes

```python
        "study": "sweep", "settings": {**_settings_block(cfg), "preset": getattr(cfg, "preset_name", None),
                                       "sweep_grid": [float(v) for v in sweep_grid]},
```

(T11 puts `preset_name` on the config; if it named the attribute differently, use that name.)

- [ ] **Step 10: Update the three call sites and the two tests that would go red**

In `core/tool/fdt.py`, find:

```python
    s_path, t_path = cross_validation.run_param_study_cli(cfg, s_grid, temp_grid)
    print(f"[prism crossval] S sweep: {s_path}")
    print(f"[prism crossval] T sweep: {t_path}")
    print(f"[prism crossval] plots under {config.artifacts_root() / 'crossval'}")
```

Replace with:

```python
    writers = {"s": store.create("fdt", cfg), "temp": store.create("fdt", cfg)}
    s_rec, t_rec = cross_validation.run_param_study_cli(cfg, s_grid=s_grid, t_grid=temp_grid,
                                                        writers=writers)
    print(f"[prism crossval] S sweep record {s_rec.id} at {s_rec.path}")
    print(f"[prism crossval] T sweep record {t_rec.id} at {t_rec.path}")
```

and drop `config` from that handler's local import line if nothing else in it uses it.

In `core/gui/panels/crossval_panel.py`, find:

```python
        watch = config.artifacts_root() / "crossval"
        self.dispatch(run_param_study_cli, cfg, s_grid, temp_grid, watch_dir=watch,
                      on_result=self._on_result)
```

Replace with:

```python
        # Two records, created here so the watcher knows both folders before the run is dispatched;
        # each sweep enters its own on the worker thread (spec §1.2, §4.1). The watcher takes ONE
        # directory and does not recurse, so it follows the S sweep's figures and the T sweep's arrive
        # with the result line. T26 gives this panel its own picker and Seed box.
        writers = {"s": default_store().create("fdt", cfg), "temp": default_store().create("fdt", cfg)}
        self.dispatch(run_param_study_cli, cfg, s_grid=s_grid, t_grid=temp_grid, writers=writers,
                      watch_dir=writers["s"].dir / "figures", on_result=self._on_result)
```

and:

```python
    def _on_result(self, paths):
        if not paths:
            return
        for path in paths:
            self.log_pane.append_line(f"Sweep data: {path}")
```

Replace with:

```python
    def _on_result(self, records):
        if not records:
            return
        for rec in records:
            self.log_pane.append_line(f"Sweep record: {rec.name or rec.id} at {rec.path}")
```

with `from core.artifacts import default_store` added to the panel's imports.

In `tests/test_tool.py`, find:

```python
    def _study(cfg, s_grid, temp_grid):
        seen["run_param_study_cli"] = (cfg, s_grid, temp_grid)
        return Path("s.h5"), Path("t.h5")
```

Replace with:

```python
    def _study(cfg, *, s_grid, t_grid, writers, seed=None):
        seen["run_param_study_cli"] = (cfg, s_grid, t_grid)
        seen["crossval_writer_kinds"] = sorted(w.kind for w in writers.values())
        return [SimpleNamespace(id="s.h5", path=writers["s"].dir),
                SimpleNamespace(id="t.h5", path=writers["temp"].dir)]
```

Find:

```python
    assert seen["run_param_study_cli"] == ("CFG", "S", "T")
    assert "s.h5" in capsys.readouterr().out
```

Replace with:

```python
    assert seen["run_param_study_cli"] == ("CFG", "S", "T")
    assert seen["crossval_writer_kinds"] == ["fdt", "fdt"], "one record per swept parameter (§4.1)"
    assert "s.h5" in capsys.readouterr().out
```

In `tests/test_fdt_user.py`'s logging test, find:

```python
    class _SweepCfg:
        model, n_freqs, ensemble_M, F0, psd_T_obs_nd = "STUB", 4, 2, 0.1, 1.0

        def with_overrides(self, **kw):
            return self
```

Replace with:

```python
    class _SweepCfg:
        model, n_freqs, ensemble_M, F0, psd_T_obs_nd = "STUB", 4, 2, 0.1, 1.0
        freqs_per_batch, freq_bounds, burn_in_nd = 1, (0.1, 30.0), 100.0
        T_obs_periods, dt_nd, seed = 30, 0.01, 5
        hw = config.cpu_device()

        def with_overrides(self, **kw):
            return self
```

Find:

```python
    out_h5 = tmp_path / "s.h5"
    caplog.clear()
    with pytest.warns(UserWarning, match="1/2 operating points failed"):
        cv.run_fdt_param_sweep(_SweepCfg(), "s", np.array([0.0, 0.1]), {"temp": 1.0}, output_path=out_h5)
```

Replace with:

```python
    monkeypatch.setattr(cv, "plot_fdt_3d_vs_param", lambda *a, save_path=None, **k: None)
    w = _LogWriter(tmp_path)
    caplog.clear()
    with pytest.warns(UserWarning, match="1/2 operating points failed"):
        cv.run_fdt_param_sweep(_SweepCfg(), "s", np.array([0.0, 0.1]), {"temp": 1.0}, writer=w)
```

Find:

```python
    assert got[-1] == (name, "INFO", f"s sweep complete (1/2 points). Saved to: {out_h5}"), got
```

Replace with:

```python
    assert got[-1] == (name, "INFO", f"s sweep complete (1/2 points). Saved to: {w.dir}"), got
```

(`_LogWriter` is the stub Task 17 added to this module; give it a `payload(name)` returning
`self.dir / name` if Task 17's version does not already have one.)

- [ ] **Step 11: Run the tests and watch them pass**

Run: `pytest tests/test_fdt_user.py tests/test_artifact_store.py -k "public_entry or callers_config or study or sweep or record or numbers or seed or unfinished" -v`
then `pytest tests/test_tool.py -k "fdt or crossval" -v`

Expected: PASS.

- [ ] **Step 12: Commit**

```bash
git add core/FDT/cross_validation.py core/FDT/cross_validation_plots.py core/tool/fdt.py core/gui/panels/crossval_panel.py tests/test_artifact_store.py tests/test_fdt_user.py tests/test_tool.py
git commit -m "crossval: the study writes one record per swept parameter"
```

#### Amendments (binding — these supersede the text above)

**A1 — the id collision (BLOCKING; the owner may rule otherwise, and a ruling beats this).** Two writers created back to back, before either is entered, get the SAME id, because `_new_id` only excludes ids on disk. Unnamed, the T sweep's `__enter__` then hits `FileExistsError`. Named, `load_fdt(writer.id)` returns the S record for the T sweep, and Step 4's test fails. Add `core/artifacts/store.py` to the Files block and the `git add` line.

In `core/artifacts/store.py`, find:
```python
class ArtifactStore:
    def __init__(self, root, *, clock=None):
        self.root = Path(root)
        self._clock = clock or _utc_now
```
Replace with:
```python
class ArtifactStore:
    def __init__(self, root, *, clock=None):
        self.root = Path(root)
        self._clock = clock or _utc_now
        # Ids this store has MINTED, per kind, entered or not. create() puts nothing on disk (the
        # writer's __enter__ does), so two writers created in one second -- a sweep study's two
        # records, created by the front end before dispatch -- would otherwise share an id.
        self._minted: dict = {}
```
Find:
```python
        cand, n = stamp, 2
        while cand in taken:
            cand, n = f"{stamp}-{n}", n + 1
        return cand
```
Replace with:
```python
        taken |= self._minted.setdefault(kind, set())
        cand, n = stamp, 2
        while cand in taken:
            cand, n = f"{stamp}-{n}", n + 1
        self._minted[kind].add(cand)
        return cand
```
In `tests/test_artifact_store.py`, after `test_same_second_ids_get_a_suffix`, add:
```python
def test_two_writers_created_before_either_is_entered_get_different_ids(tmp_path):
    """Piece 5, spec §4.1: the sweep study's front end creates BOTH records before it dispatches, and
    create() puts nothing on disk -- so an id checked only against the disk would be minted twice in
    one second, and the second record would load as the first."""
    fixed = datetime(2026, 9, 22, 12, 0, 0, tzinfo=timezone.utc)
    s = st.ArtifactStore(tmp_path, clock=lambda: fixed)
    a, b = s.create("fdt", None), s.create("fdt", None)
    assert a.id != b.id and a.dir != b.dir, (a.id, b.id)
```

**A2 — Step 5's prediction.** The first failure is `AttributeError: <module 'core.FDT.cross_validation'> has no attribute 'plot_fdt_3d_vs_param'`, raised by the test's own `monkeypatch.setattr`: the name only reaches module scope in Step 7. Record that failure instead of the TypeError.

**A3 — Step 6, the `Path` import.** `Path` is NOT imported in `cross_validation_plots.py`. The return annotation is a string under `from __future__ import annotations`. In the import replacement, write:
```python
from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
```

**A4 — Step 7, the opening of `run_fdt_param_sweep` (the quoted anchor is not found: T12 inserted two lines; P15, P72, P6, T12's notices).**

Imports: replace Step 7's import block with the one below. It keeps `gen_freqs_log` because `_fdt_measure` still calls it until T20 deletes it:
```python
from ..config import FDTConfig
from ..diagnostics.rng import seeded
from ..runs import public_entry
from .campaigns import run_campaign1_psd, run_campaign2_chi, observable_noise_prefactor
from .spectral import gen_freqs_log, eff_temp_ratio
from .sanity import _interp_log
from .fdt_pipeline import _estimate_omega_0, _resolve_seed, _settings_block
from .cross_validation_plots import plot_fdt_3d_vs_param
```
Step 8's "add `from ..diagnostics.rng import seeded`" is thereby done.

The docstring tail and body head: find the block from `    :param output_path: target .h5. Defaults to <artifacts root>/crossval/sweep_<param>_<stamp>.h5.` through `    with h5py.File(output_path, "w") as h5:`. After T12 it contains `notices = warn_thin_settings(cfg)` and, unless T12 moved it to module scope, `from .fdt_pipeline import warn_thin_settings`. Replace the whole block with:
```python
    :param writer: the OPEN-BUT-NOT-ENTERED ArtifactWriter for this sweep's own record. THIS function
                   enters it, so __enter__, every refresh() and __exit__ run on the thread whose run
                   log becomes log.txt (spec §1.2, §4.1). Its seed comes from ``cfg.seed``; nothing
                   here writes on ``cfg``.
    :returns: the LoadedFdt for the record just written.
    """
    from .fdt_pipeline import warn_thin_settings
    notices = warn_thin_settings(cfg)          # E5: kept in this sweep's body.notices below
    fixed_overrides = fixed_overrides or {}
    seed = _resolve_seed(None, cfg)
    fixed_str = ", ".join(f"{k}={v}" for k, v in fixed_overrides.items())
    n_points = int(len(sweep_grid))
    # The first body is UPDATED IN PLACE, never replaced (P15): the front end may already have set the
    # study, a seed and notices (T26's panel does). The stage owns settings, points and the RESOLVED
    # seed; the swept parameter is recorded once, in points.param (spec §2.3).
    body = writer.body
    body.setdefault("study", "sweep")
    body["settings"] = {**_settings_block(cfg), "preset": cfg.preset_name,
                        "sweep_grid": [float(v) for v in sweep_grid]}
    body["seed"] = seed
    body["points"] = {"param": sweep_param, "planned": n_points, "done": 0, "failed": 0}
    body["notices"] = [*(body.get("notices") or []), *notices]
    for key in ("grid", "offgrid", "compared", "results"):
        body.setdefault(key, None)
    body["complete"] = False
    # NESTED, not combined: the HDF5 file must be CLOSED before the sweep's own plot reopens it with
    # load_param_sweep at the end, and that plot must still be drawn inside the writer so its PNG
    # lands in the record's figures/.
    with writer:
        with h5py.File(writer.payload("data.h5"), "w") as h5:
            # The contract's root attributes (P6), on every data.h5 so a reader can check the layout
            # before reading a dataset. omega_0 is the common grid's reference, set once it exists.
            h5.attrs["study"] = "sweep"
            h5.attrs["prefactor"] = float(observable_noise_prefactor(cfg))
            h5.attrs["omega_0"] = math.nan
```
The former body of the `with h5py.File(...)` block (starting `h5.attrs["timestamp"] = ...`) follows, indented one level deeper. The docstring edit for `DELIBERATELY NOT an atomic write` stands. Also rewrite its two stale `output_path` sentences ("Here ``output_path`` defaults to a TIMESTAMPED name ..." and "If you ever pass an explicit ``output_path`` ...") so they speak of `writer.payload("data.h5")`, a fresh file inside a fresh record.

Step 9's closing paragraph ("Add `\"preset\"` to each record's settings ... `getattr(cfg, \"preset_name\", None)` ...") is superseded by the body above: P72 reads `cfg.preset_name` directly.

**A5 — the sweep's `omega_0` root attribute (P6).** In the common-grid block, find:
```python
        h5.attrs["omega_0_ref"] = omega_0_ref
```
Replace with:
```python
        h5.attrs["omega_0_ref"] = omega_0_ref
        h5.attrs["omega_0"] = omega_0_ref
```

**A6 — Step 4's test.** After `assert r.data_path.exists() and r.manifest.figures, ...`, add inside the loop:
```python
        import h5py
        with h5py.File(r.data_path, "r") as h5:
            assert h5.attrs["study"] == "sweep" and "prefactor" in h5.attrs, "the contract's root attributes (P6)"
        assert r.body["settings"]["skip_sanity"] is None, "P70: a sweep has no sanity branch"
```
The test depends on A1 (without it, the second record loads as the first).

**A7 — Step 10, call sites (P79, P77, T21's anchor).**

Tool: the replacement becomes the block below, which prints whatever comes back. T20 makes the study return only the sweeps that finished (P77).
```python
    writers = {"s": store.create("fdt", cfg), "temp": store.create("fdt", cfg)}
    recs = cross_validation.run_param_study_cli(cfg, s_grid=s_grid, t_grid=temp_grid,
                                                writers=writers, seed=getattr(args, "seed", None))
    for rec in recs:
        print(f"[prism crossval] sweep record {rec.id} at {rec.path}")
```
`getattr` because `--seed` lands in T28. The test's `"s.h5" in ...out` still holds, since the recorder's first id is "s.h5".

Panel: do NOT put the new import between `from core import cli, config` and `from core.config import CELL_PATH`; T21's anchor quotes those two lines together. Find:
```python
from core.FDT.cross_validation import run_param_study_cli
```
Replace with:
```python
from core.FDT.cross_validation import run_param_study_cli
from core.artifacts import default_store
```
The dispatch and `_on_result` replacements stand. T26 (P80) anchors on them.

**A8 — Step 10, the logging test.** In the `_SweepCfg` replacement, add a line `        preset_name = None` (P72 reads `cfg.preset_name` directly). After `monkeypatch.setattr(cv, "_campaign2_ratio", _campaign2)`, add:
```python
    monkeypatch.setattr(cv, "observable_noise_prefactor", lambda c: 1.0)   # the "STUB" model has none
```
(A4's root attribute calls it.) `_LogWriter` is T17's MODULE-level class and already has `payload`; do not redefine it. `from core import config` is in this test's imports since T17.

**A9 — Step 11 commands.** Run:
- `pytest tests/test_artifact_store.py -k "public_entry or callers_config or same_second or before_either_is_entered" -v`
- `pytest tests/test_fdt_user.py -v`
- `pytest tests/test_tool.py -k "fdt or crossval" -v`

**A10 — Step 12.**
```bash
git add core/artifacts/store.py core/FDT/cross_validation.py core/FDT/cross_validation_plots.py core/tool/fdt.py core/gui/panels/crossval_panel.py tests/test_artifact_store.py tests/test_fdt_user.py tests/test_tool.py
```

#### Controller rulings (pre-flight, binding — these supersede the amendments above and the task text)

- **F2/F4** — confirmed as the amendment applied it: `ArtifactStore._new_id` also treats as taken every id this store object has already MINTED for the kind, so two writers created back to back before either is entered get different ids. Its pin test belongs to this task.
- **F40** — `body.settings["sweep_grid"]` is `[min, max, N]` as spec §2.3 says — `[float(sweep_grid[0]), float(sweep_grid[-1]), int(len(sweep_grid))]` — not the whole linspace array.
- **F6** — confirmed: this task owns the crossval leg of `test_fdt_and_crossval_flags_reach_their_builders` (the `_sweep_cfg` recorder returns a stub carrying what `config_from_cfg` reads, not the string "CFG").

---

### Task 20: the sweep's failure counts, the all-failed refusal, and two defects on the same path

**Why:** Spec §4.3 (E4). The second phase already catches, logs and counts per-point failures; the
FIRST phase catches nothing, so one bad operating point in Phase A takes the whole study down with a
traceback. Both phases now count, and the counts become `body.points.done` / `.failed`, visible in both
listings. All points failed becomes a calm refusal naming the grid and the cell, raised **after** the
record's final refresh so the spectra the message points at are on disk. Two smaller defects go with
it: the per-point peak log line sits OUTSIDE the try that guards Phase B, so an all-NaN ratio raises
`ValueError` from `np.nanmax` after the first phase's whole cost; and the dead helper `_fdt_measure`
— whose `cfg.omega_0 = omega_0_emp` is the module's only write on a caller's settings object — has no
caller anywhere.

**Files:**
- Modify: `core/FDT/cross_validation.py:97-126` (`_fdt_measure`), `:219-245` (Phase A),
  `:258-286` (Phase B), and the tail Task 19 moved into the `with`
- Test: `tests/test_fdt_user.py`

**Interfaces:**
- Consumes, from T19: `run_fdt_param_sweep(cfg, sweep_param, sweep_grid, fixed_overrides=None, *,
  writer)`, `writer.body["points"]`, `writer.refresh()`.
- Consumes, from T8: the field keys `s_grid` and `t_grid`.
- Produces, for T26/T34: `body.points == {"param", "planned", "done", "failed"}` kept current as the
  sweep runs; an all-failed sweep raises `Refusal(field="s_grid" | "t_grid")` from inside the entered
  writer, so E2 keeps the folder.

- [ ] **Step 1: Write the failing tests**

In `tests/test_fdt_user.py`, add:

```python
def _sweep_stubs(monkeypatch, *, phase_a_fail=(), phase_b_fail=()):
    """Campaign 1 and Campaign 2 stubbed per operating point, with chosen indices made to fail."""
    import torch
    from core.FDT import cross_validation as cv

    seen_a = {"n": 0}

    def _c1(cfg_op):
        idx = seen_a["n"]
        seen_a["n"] += 1
        if idx in phase_a_fail:
            raise RuntimeError(f"stub phase-A failure at {idx}")
        return (torch.linspace(0.1, 3.0, 8, dtype=torch.float64),
                torch.ones(8, dtype=torch.float64))

    seen_b = {"n": 0}

    def _c2(cfg_op, omegas, freqs_psd, G):
        idx = seen_b["n"]
        seen_b["n"] += 1
        if idx in phase_b_fail:
            raise RuntimeError(f"stub phase-B failure at {idx}")
        return (torch.ones(omegas.shape, dtype=torch.complex128),
                torch.full(omegas.shape, 2.0, dtype=torch.float64))

    monkeypatch.setattr(cv, "run_campaign1_psd", _c1)
    monkeypatch.setattr(cv, "_detect_resonance", lambda omegas, G, w0: (1.0, True))
    monkeypatch.setattr(cv, "_campaign2_ratio", _c2)
    monkeypatch.setattr(cv, "plot_fdt_3d_vs_param", lambda *a, save_path=None, **k: None)


def test_a_sweep_counts_failures_in_both_phases(store, monkeypatch):
    """E4: some points failed is a COMPLETED record carrying the count, and both phases count.

    Phase A caught nothing before piece 5 (core/FDT/cross_validation.py's Phase A loop has no try at
    all), so one operating point whose spontaneous campaign raised -- a transient OOM, a solver blow-up
    at the grid's far end -- ended the whole study with a traceback and left no record of the points
    that had already worked. The counts are what let a reader judge the answer."""
    import pytest
    from core import cli, config
    from core.FDT import cross_validation as cv

    _sweep_stubs(monkeypatch, phase_a_fail=(0,), phase_b_fail=(1,))
    cfg, s_grid, _t = cli.make_param_sweep_config(
        str(config.CELL_PATH / "nadrowski" / "master_spont.txt"),
        preset=dict(cli.SWEEP_PRESETS["exploratory"]), preset_name="exploratory",
        s_spec=(0.0, 0.3, 4), t_spec=(1.0, 1.1, 2), n_freqs=3, ensemble_M=2)
    cfg.seed = 1
    w = store.create("fdt", cfg, name="partly")
    with pytest.warns(UserWarning, match="operating points failed"):
        rec = cv.run_fdt_param_sweep(cfg, "s", s_grid, {"temp": 1.0}, writer=w)

    assert rec.body["complete"] is True, "some points failed is a completed record (E4)"
    assert rec.body["points"] == {"param": "s", "planned": 4, "done": 2, "failed": 2}
    (summary,) = store.list("fdt")
    assert (summary.points_done, summary.points_failed, summary.points_planned) == (2, 2, 4)


def test_a_sweep_with_every_point_failed_refuses_after_its_record_is_written(store, monkeypatch):
    """E4 and §4.3. A run that measured NOTHING refuses, naming the setting to change -- today it is a
    RuntimeError the command line reports as a crash -- and the refusal is raised AFTER the final
    refresh, so the spectra the message tells the reader to look at are already on disk. The folder
    stays (E2): that is the whole reason the first phase's PSDs are worth keeping."""
    import pytest
    from core import cli, config
    from core.FDT import cross_validation as cv
    from core.refusals import Refusal

    _sweep_stubs(monkeypatch, phase_b_fail=(0, 1))
    cfg, s_grid, _t = cli.make_param_sweep_config(
        str(config.CELL_PATH / "nadrowski" / "master_spont.txt"),
        preset=dict(cli.SWEEP_PRESETS["exploratory"]), preset_name="exploratory",
        s_spec=(0.0, 0.1, 2), t_spec=(1.0, 1.1, 2), n_freqs=3, ensemble_M=2)
    cfg.seed = 1
    w = store.create("fdt", cfg, name="nothing")
    with pytest.raises(Refusal) as e:
        cv.run_fdt_param_sweep(cfg, "s", s_grid, {"temp": 1.0}, writer=w)
    assert e.value.field == "s_grid"
    assert "all 2" in str(e.value)

    (summary,) = store.list("fdt")
    assert summary.complete and not summary.finished, "unfinished, listed, and deletable"
    assert summary.points_failed == 2
    body = store.get("fdt", summary.id).body
    assert body["points"]["failed"] == 2, "the final refresh ran BEFORE the refusal"
    assert (w.dir / "data.h5").exists(), "the message points at the PSDs; they must be there"


def test_an_empty_ratio_is_a_counted_failure_not_a_numpy_crash(store, monkeypatch, caplog):
    """The nanmax defect (§4.3). ``log.info(f"... {np.nanmax(ratio.cpu().numpy()):.3g}")`` sits
    OUTSIDE the try that guards Campaign 2, so a point whose ratio comes back EMPTY raises
    ``ValueError: zero-size array to reduction operation fmax which has no identity`` from numpy --
    after the first phase's whole cost has been paid, and with a traceback rather than a count. Inside
    the try it is one logged, counted, recorded failure like any other."""
    import logging
    import pytest
    import torch
    from core import cli, config
    from core.FDT import cross_validation as cv
    from core.refusals import Refusal

    _sweep_stubs(monkeypatch)
    monkeypatch.setattr(cv, "_campaign2_ratio",
                        lambda c, om, f, g: (torch.zeros(0, dtype=torch.complex128),
                                             torch.zeros(0, dtype=torch.float64)))
    cfg, s_grid, _t = cli.make_param_sweep_config(
        str(config.CELL_PATH / "nadrowski" / "master_spont.txt"),
        preset=dict(cli.SWEEP_PRESETS["exploratory"]), preset_name="exploratory",
        s_spec=(0.0, 0.1, 2), t_spec=(1.0, 1.1, 2), n_freqs=3, ensemble_M=2)
    cfg.seed = 1
    with caplog.at_level(logging.INFO, logger="core"):
        with pytest.raises(Refusal) as e:
            cv.run_fdt_param_sweep(cfg, "s", s_grid, {"temp": 1.0},
                                   writer=store.create("fdt", cfg, name="empty"))
    assert e.value.field == "s_grid", "every point failed, so the ending is the all-failed refusal"
    errors = [r.getMessage() for r in caplog.records if r.levelname == "ERROR"]
    assert len(errors) == 2 and all("Campaign 2 FAILED" in m for m in errors), errors


def test_the_dead_single_point_helper_is_gone():
    """``_fdt_measure`` has no caller anywhere in the tree, and its ``cfg.omega_0 = omega_0_emp`` is
    the only write on a caller's settings object left in this module -- exactly the V1 defect the rest
    of the piece removes, sitting in code nothing runs. Dead code that models the wrong thing is worse
    than dead code."""
    from core.FDT import cross_validation as cv
    assert not hasattr(cv, "_fdt_measure")
```

- [ ] **Step 2: Run them and watch them fail**

Run: `pytest tests/test_fdt_user.py -k "counts_failures or every_point_failed or empty_ratio or dead_single" -v`

Expected: FAIL —
`test_a_sweep_counts_failures_in_both_phases` errors with `RuntimeError: stub phase-A failure at 0`
(Phase A catches nothing); `test_a_sweep_with_every_point_failed_refuses_after_its_record_is_written`
fails with `DID NOT RAISE <class 'core.refusals.Refusal'>` after a `RuntimeError`;
`test_an_empty_ratio_is_a_counted_failure_not_a_numpy_crash` fails with
`ValueError: zero-size array to reduction operation fmax which has no identity`, raised from the
`T_eff/T peak` log line outside the try; and
`test_the_dead_single_point_helper_is_gone` fails on `assert not hasattr(cv, "_fdt_measure")`.

- [ ] **Step 3: Delete the dead helper**

In `core/FDT/cross_validation.py`, find and DELETE the whole function:

```python
def _fdt_measure(cfg: FDTConfig) -> dict:
    """
    Single-operating-point FDT: Campaign 1 -> robust omega_0 -> Campaign 2 on a grid
    centered on that omega_0 -> ratio. Used for one-off measurements; the parameter
    sweep uses the two-phase path in run_fdt_param_sweep (shared grid across rows).

    :returns: dict with omega_grid, omega_norm, T_eff_over_T, chi_prime,
              chi_double_prime, PSD_omegas, PSD_G, omega_0_empirical, omega_0_linearized.
    """
```

through its closing brace:

```python
        "omega_0_empirical": omega_0_emp,
        "omega_0_linearized": float(omega_0_lin),
    }
```

- [ ] **Step 4: Phase A counts its failures**

In `core/FDT/cross_validation.py`, find:

```python
        log.info(f"--- Phase A ({sweep_param} sweep): spontaneous PSD + omega_0 detection ---")
        cfg_ops, psds, omega0s, is_res, omega0_lins = [], [], [], [], []
        for idx, value in enumerate(sweep_grid):
            log.info(f"  [A {idx+1}/{len(sweep_grid)}] {sweep_param}={value:.6g} ({fixed_str})")
            cfg_op = cfg.with_overrides(**{sweep_param: float(value), **fixed_overrides})
            omega_0_lin, _ = _estimate_omega_0(cfg_op)
            with seeded(seed + idx, cfg.hw.device):
                freqs_psd, G = run_campaign1_psd(cfg_op)
            w0, res = _detect_resonance(freqs_psd, G, omega_0_lin)

            cfg_ops.append(cfg_op); psds.append((freqs_psd, G))
            omega0s.append(w0); is_res.append(res); omega0_lins.append(float(omega_0_lin))

            grp = ops.create_group(f"{idx:03d}")
```

Replace with:

```python
        log.info(f"--- Phase A ({sweep_param} sweep): spontaneous PSD + omega_0 detection ---")
        cfg_ops, psds, omega0s, is_res, omega0_lins, indices = [], [], [], [], [], []
        n_failed = 0
        for idx, value in enumerate(sweep_grid):
            log.info(f"  [A {idx+1}/{len(sweep_grid)}] {sweep_param}={value:.6g} ({fixed_str})")
            cfg_op = cfg.with_overrides(**{sweep_param: float(value), **fixed_overrides})
            grp = ops.create_group(f"{idx:03d}")
            grp.attrs["param_value"] = float(value)
            grp.attrs["sweep_param"] = sweep_param
            grp.attrs["failed"] = True     # flipped to False once Campaign 2 lands in Phase B
            try:
                omega_0_lin, _ = _estimate_omega_0(cfg_op)
                with seeded(seed + idx, cfg.hw.device):
                    freqs_psd, G = run_campaign1_psd(cfg_op)
                w0, res = _detect_resonance(freqs_psd, G, omega_0_lin)
            except Exception as e:         # noqa: BLE001 -- counted, recorded, and the sweep goes on
                # Phase A caught NOTHING before piece 5: one bad operating point ended the study with
                # a traceback and lost every point that had already worked. E4 makes it a count.
                log.error(f"      Campaign 1 FAILED: {e}")
                grp.attrs["error"] = str(e)
                n_failed += 1
                _refresh_points(writer, n_points, n_failed)
                h5.flush()
                continue

            cfg_ops.append(cfg_op); psds.append((freqs_psd, G)); indices.append(idx)
            omega0s.append(w0); is_res.append(res); omega0_lins.append(float(omega_0_lin))
```

Find, immediately below, the attrs the old code wrote on the group it had just created:

```python
            grp.attrs["param_value"] = float(value)
            grp.attrs["sweep_param"] = sweep_param
            for k, v in fixed_overrides.items():
                grp.attrs[f"fixed_{k}"] = float(v)
            grp.attrs["omega_0_resonance"] = float(w0)
            grp.attrs["omega_0_linearized"] = float(omega_0_lin)
            grp.attrs["is_resonant"] = bool(res)
            grp.attrs["failed"] = True   # flipped to False once Campaign 2 lands in Phase B
```

Replace with:

```python
            for k, v in fixed_overrides.items():
                grp.attrs[f"fixed_{k}"] = float(v)
            grp.attrs["omega_0_resonance"] = float(w0)
            grp.attrs["omega_0_linearized"] = float(omega_0_lin)
            grp.attrs["is_resonant"] = bool(res)
```

Add beside the module's other helpers:

```python
def _refresh_points(writer, planned: int, failed: int) -> None:
    """Keep ``body.points`` current as the sweep runs, and rewrite the manifest and the log.

    A sweep is hours long, and a reader watching the browser wants the counts NOW -- which is what the
    progressive record is for (E2, spec §2.2). ``done`` is derived, never accumulated separately: two
    counters for one fact is how a listing ends up saying 3 done and 2 failed of 4.
    """
    writer.body["points"] = {**writer.body["points"], "planned": planned,
                             "failed": failed, "done": planned - failed}
    writer.refresh()
```

- [ ] **Step 5: Phase B counts against the surviving points, and the peak line moves inside the try**

In `core/FDT/cross_validation.py`, find:

```python
        log.info(f"--- Phase B ({sweep_param} sweep): forced response on common grid ---")
        n_failed = 0
        for idx, (cfg_op, (freqs_psd, G)) in enumerate(zip(cfg_ops, psds)):
            log.info(f"  [B {idx+1}/{len(sweep_grid)}] {sweep_param}={sweep_grid[idx]:.6g}")
            grp = ops[f"{idx:03d}"]
```

Replace with:

```python
        log.info(f"--- Phase B ({sweep_param} sweep): forced response on common grid ---")
        for cfg_op, (freqs_psd, G), idx in zip(cfg_ops, psds, indices):
            log.info(f"  [B {idx+1}/{n_points}] {sweep_param}={sweep_grid[idx]:.6g}")
            grp = ops[f"{idx:03d}"]
```

(`indices` is what keeps Phase B's group keys aligned with Phase A's after a Phase-A point dropped
out; `enumerate` over the survivors would silently write point 2's response into point 1's group.)

Find:

```python
            except Exception as e:
                # Recorded per point, and COUNTED. A systematic failure (a CUDA OOM, say) fails every
                # point identically, so an overnight sweep could "complete" with nothing in it -- the
                # per-point note scrolled past hours ago and the summary said nothing.
                log.error(f"      Campaign 2 FAILED: {e}")
                grp.attrs["error"] = str(e)
                n_failed += 1
                continue
```

Replace with:

```python
            except Exception as e:
                # Recorded per point, and COUNTED. A systematic failure (a CUDA OOM, say) fails every
                # point identically, so an overnight sweep could "complete" with nothing in it -- the
                # per-point note scrolled past hours ago and the summary said nothing.
                log.error(f"      Campaign 2 FAILED: {e}")
                grp.attrs["error"] = str(e)
                n_failed += 1
                _refresh_points(writer, n_points, n_failed)
                h5.flush()
                continue
```

Find:

```python
            grp.attrs["failed"] = False
            log.info(f"      T_eff/T peak = {np.nanmax(ratio.cpu().numpy()):.3g}")
            h5.flush()
```

Replace with:

```python
            _refresh_points(writer, n_points, n_failed)
            h5.flush()
```

and move the block that currently sits between the `except ... continue` and that flush —

```python
            grp.attrs["omega_0_ref"] = omega_0_ref
            grp.create_dataset("omega_grid", data=omega_grid_np, compression="gzip")
            grp.create_dataset("omega_norm", data=omega_norm_np, compression="gzip")
            grp.create_dataset("T_eff_over_T",
                               data=ratio.cpu().numpy().astype(np.float64), compression="gzip")
            grp.create_dataset("chi_prime",
                               data=chis.real.cpu().numpy().astype(np.float64), compression="gzip")
            grp.create_dataset("chi_double_prime",
                               data=chis.imag.cpu().numpy().astype(np.float64), compression="gzip")
            grp.attrs["failed"] = False
            log.info(f"      T_eff/T peak = {np.nanmax(ratio.cpu().numpy()):.3g}")
```

— up INTO the `try`, directly after the `_campaign2_ratio` call, in this order:

```python
                with seeded(seed + n_points + idx, cfg.hw.device):
                    chis, ratio = _campaign2_ratio(cfg_op, omegas_common, freqs_psd, G)
                # The peak line FIRST, before anything about this point is called a success: an empty
                # or all-blank ratio raises here, and a group already flipped to failed=False would
                # then be counted a failure and stored as a success at the same time.
                log.info(f"      T_eff/T peak = {np.nanmax(ratio.cpu().numpy()):.3g}")
                grp.attrs["omega_0_ref"] = omega_0_ref
                grp.create_dataset("omega_grid", data=omega_grid_np, compression="gzip")
                grp.create_dataset("omega_norm", data=omega_norm_np, compression="gzip")
                grp.create_dataset("T_eff_over_T",
                                   data=ratio.cpu().numpy().astype(np.float64), compression="gzip")
                grp.create_dataset("chi_prime",
                                   data=chis.real.cpu().numpy().astype(np.float64), compression="gzip")
                grp.create_dataset("chi_double_prime",
                                   data=chis.imag.cpu().numpy().astype(np.float64), compression="gzip")
                grp.attrs["failed"] = False
```

so an empty ratio is one counted, logged failure rather than a `ValueError` from `np.nanmax` after the
first phase's whole cost.

- [ ] **Step 6: All points failed becomes a refusal, after the final refresh**

In `core/FDT/cross_validation.py`, find the guard Task 19 left just before the common grid is built —
if Task 19 left none, add one here, because `_build_common_grid` calls `min(basis)` on an empty list
when Phase A lost every point:

```python
        # --- Common grid covering every row's resonance band ---
        omegas_common, omega_0_ref = _build_common_grid(cfg, omega0s, is_res)
```

Replace with:

```python
        # --- Common grid covering every row's resonance band ---
        if not omega0s:
            # Every point failed in Phase A: _build_common_grid would raise ValueError from min() on
            # an empty sequence. Skip Phase B and fall through to the refusal below, which names the
            # setting to change and leaves the record behind.
            n_failed = n_points
            omegas_common = None
        else:
            omegas_common, omega_0_ref = _build_common_grid(cfg, omega0s, is_res)
```

and guard the three lines that follow (`omega_grid_np = ...` through the `Common grid:` log line) and
the whole Phase B loop with `if omegas_common is not None:`.

Find:

```python
        if n_failed == n_points:
            raise RuntimeError(
                f"{sweep_param} sweep produced NO usable points: all {n_points} operating points "
                f"failed in Campaign 2 (per-point reasons are in the HDF5 'error' attrs). This is "
                f"almost always one systematic cause -- a CUDA OOM repeating identically at every "
                f"point -- not {n_points} independent failures. {writer.dir} contains the PSDs but "
                f"no response data.")
```

Replace with:

```python
        if n_failed == n_points:
            # AFTER the final refresh below, so the spectra this message points at are on disk before
            # the refusal unwinds (spec §4.3). A Refusal, not a RuntimeError: the command line reports
            # one as an operator line naming the flag, the window as the yellow box naming the grid.
            _refresh_points(writer, n_points, n_failed)
            raise Refusal(
                f"The {sweep_param} sweep measured nothing: all {n_points} operating points failed "
                f"(each point's reason is in its 'error' attribute in the record's data.h5). This is "
                f"almost always one systematic cause repeating identically at every point, not "
                f"{n_points} independent failures. The record holds the spontaneous spectra; widen "
                f"or move the grid, or choose a cell whose operating points are reachable.",
                field=_GRID_FIELD[sweep_param])
```

Add beside `_PARAM_SYMBOL`:

```python
# Which front-end setting an all-failed sweep names (spec §4.3, E4). The swept parameter IS the grid
# the operator would change, and each grid has its own control and its own flag.
_GRID_FIELD = {"s": "s_grid", "temp": "t_grid"}
```

and add `from ..refusals import Refusal` to the module's imports.

- [ ] **Step 7: Run the tests and watch them pass**

Run: `pytest tests/test_fdt_user.py -v`

Expected: PASS.

- [ ] **Step 8: Run the store and tool suites that touch this path**

Run: `pytest tests/test_artifact_store.py -k "public_entry or callers_config" tests/test_tool.py -k "fdt or crossval" -v`

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add core/FDT/cross_validation.py tests/test_fdt_user.py
git commit -m "crossval: both phases count failures; an empty sweep refuses"
```
## THE FIVE SURFACES (T21–T24)

> These four tasks land after T9 (`core/gui/fields.py` widened to screens) and T8 (`require_below`,
> the eight sweep/FDT field keys, the tool table). T24 additionally lands after T2 (`Summary` grows).
> Every line number below was read off the tree at `a0d85da`; the QUOTED text is the anchor.

#### Amendments (binding — these supersede the text above)

**A1 — imports.** In `core/FDT/cross_validation.py`:
- find `from .spectral import gen_freqs_log, eff_temp_ratio` and replace it with `from .spectral import eff_temp_ratio` (`_fdt_measure`, its last user, is deleted in Step 3);
- find `from .fdt_pipeline import _estimate_omega_0, _resolve_seed, _settings_block` and replace it with `from .fdt_pipeline import _estimate_omega_0, _resolve_seed, _results_block, _settings_block` (P78);
- Step 6's `from ..refusals import Refusal` stands.

**A2 — Step 4, Phase A (the quoted anchor is not found: T19 put a 3-line comment before `with seeded(...)`; indentation is one level deeper after T19's nesting, so match whitespace aside).** Find the block from `log.info(f"--- Phase A ({sweep_param} sweep): spontaneous PSD + omega_0 detection ---")` through `grp = ops.create_group(f"{idx:03d}")`. It contains T19's lines:
```python
            omega_0_lin, _ = _estimate_omega_0(cfg_op)
            # One stream per operating point, derived from the study's single seed (spec §4.1): a
            # point is reproducible from the seed and its index, and seeding once for the whole sweep
            # would make point k's draw depend on how many points preceded it.
            with seeded(seed + idx, cfg.hw.device):
                freqs_psd, G = run_campaign1_psd(cfg_op)
            w0, res = _detect_resonance(freqs_psd, G, omega_0_lin)
```
Replace it with Step 4's replacement, changed in three places:
1. `n_failed = 0` becomes `n_failed, n_done = 0, 0`.
2. T19's three comment lines are kept, inside the `try`, directly above `with seeded(seed + idx, cfg.hw.device):`.
3. The except branch's `_refresh_points(writer, n_points, n_failed)` becomes `_refresh_points(writer, planned=n_points, done=n_done, failed=n_failed)`.

The second Step-4 find/replace (the attrs block) stands.

**A3 — `_refresh_points` counts `done`, never derives it.** Use this instead of the definition Step 4 gives:
```python
def _refresh_points(writer, *, planned: int, done: int, failed: int) -> None:
    """Keep ``body.points`` current as the sweep runs, and rewrite the manifest and the log.

    ``done`` counts the operating points whose Campaign 2 has LANDED and ``failed`` those that failed
    in either phase; ``planned - done - failed`` are still to run. Deriving ``done`` as
    ``planned - failed`` would report every not-yet-run point as done for the whole run -- the
    browser row this refresh exists for would read "4 done of 4" after the first point.
    """
    writer.body["points"] = {**writer.body["points"], "planned": planned, "done": done,
                             "failed": failed}
    writer.refresh()
```
Every call site passes `planned=n_points, done=n_done, failed=n_failed`.

**A4 — Step 5, net result.** Phase B's header replacement stands. It removes `n_failed = 0`, which Phase A now initialises. Directly above that header's `for` loop (inside the guard Step 6 adds), put `ok_ratios, blanks_total = [], 0` — or immediately before the guard, so both names exist on every path.

The try/except becomes:
```python
            try:
                with seeded(seed + n_points + idx, cfg.hw.device):
                    chis, ratio = _campaign2_ratio(cfg_op, omegas_common, freqs_psd, G)
                # The peak line FIRST ... (Step 5's comment)
                log.info(f"      T_eff/T peak = {np.nanmax(ratio.cpu().numpy()):.3g}")
                grp.attrs["omega_0_ref"] = omega_0_ref
                ... (the five create_dataset lines, as Step 5 gives them) ...
                grp.attrs["failed"] = False
            except Exception as e:
                # (Step 5's comment)
                log.error(f"      Campaign 2 FAILED: {e}")
                grp.attrs["error"] = str(e)
                n_failed += 1
                _refresh_points(writer, planned=n_points, done=n_done, failed=n_failed)
                h5.flush()
                continue
            n_done += 1
            ok_ratios.append(ratio.detach().cpu().to(torch.float64).reshape(-1))
            blanks_total += int(torch.isnan(ratio).sum())      # this point's blanks on the common grid (P78)
            _refresh_points(writer, planned=n_points, done=n_done, failed=n_failed)
            h5.flush()
```

**A5 — Step 6, the tail.** In the refusal, the pre-raise refresh is `_refresh_points(writer, planned=n_points, done=n_done, failed=n_failed)`. The warning T19 left above it now covers both phases. Find:
```python
                f"{sweep_param} sweep: {n_failed}/{n_points} operating points failed in Campaign 2 "
                f"and carry no response data. See their 'error' attrs in {writer.dir}.", stacklevel=2)
```
Replace with:
```python
                f"{sweep_param} sweep: {n_failed}/{n_points} operating points failed (in either "
                f"campaign) and carry no response data. See their 'error' attrs in {writer.dir}.",
                stacklevel=2)
```

**A6 — P78, a finished sweep fills `results` and `offgrid`.** Immediately after the `if n_failed == n_points: ... raise Refusal(...)` block, and before `log.info(f"{sweep_param} sweep complete ...")`, add:
```python
        # A FINISHED sweep fills results and offgrid (P78; spec §2.3 "null only until the run
        # finishes"). results: T17's _results_block over every usable point's ratio on the common grid;
        # offgrid: the blanks those points left on it, of every planned point's slot.
        writer.body["results"] = _results_block(
            omegas_common.detach().cpu().to(torch.float64).repeat(len(ok_ratios)),
            torch.cat(ok_ratios), omega_0_ref, blanks_total)
        writer.body["offgrid"] = {"blanks": int(blanks_total),
                                  "of": int(n_points * omegas_common.numel())}
```
This is reached only when at least one point landed, so `omegas_common` is not None and `ok_ratios` is not empty.

**A7 — P77, an all-failed first sweep must not cost the second.** In `run_param_study_cli` (T19's version), find:
```python
    log.info("#" * 64)
    log.info("# S sweep:  vary S, hold T_a/T = 1   (FDT restored as S -> 0)")
    log.info("#" * 64)
    s_rec = run_fdt_param_sweep(cfg, sweep_param="s", sweep_grid=s_grid,
                                fixed_overrides={"temp": 1.0}, writer=writers["s"])

    log.info("#" * 64)
    log.info("# T sweep:  vary T_a/T, hold S = 0   (FDT restored as T_a/T -> 1)")
    log.info("#" * 64)
    temp_rec = run_fdt_param_sweep(cfg, sweep_param="temp", sweep_grid=t_grid,
                                   fixed_overrides={"s": 0.0}, writer=writers["temp"])
    return [s_rec, temp_rec]
```
Replace with:
```python
    recs, refused = [], []
    for key, grid, fixed, banner in (
            ("s", s_grid, {"temp": 1.0}, "# S sweep:  vary S, hold T_a/T = 1   (FDT restored as S -> 0)"),
            ("temp", t_grid, {"s": 0.0},
             "# T sweep:  vary T_a/T, hold S = 0   (FDT restored as T_a/T -> 1)")):
        log.info("#" * 64)
        log.info(banner)
        log.info("#" * 64)
        try:
            recs.append(run_fdt_param_sweep(cfg, sweep_param=key, sweep_grid=grid,
                                            fixed_overrides=fixed, writer=writers[key]))
        except Refusal as e:
            # P77, spec §4.3: a sweep that measured nothing costs ITSELF, never the other one. Its
            # record stays on disk, unfinished (E2), and this line is in both records' log.txt.
            log.error(f"The {key} sweep measured nothing; its unfinished record is kept. {e}")
            refused.append(e)
    if len(refused) == 2:
        raise refused[0]            # the study measured nothing at all: the first grid's refusal
    return recs                     # the sweeps that FINISHED, in study order
```
A crash or a cancel (not a `Refusal`) still propagates at once. Update the docstring's `:returns:` to read "the LoadedFdt of every sweep that finished, S first; a sweep that measured nothing is logged and left on disk unfinished, and the study refuses only when both did".

**A8 — tests (P78, P77, P82).**

In `test_a_sweep_counts_failures_in_both_phases`, add after the `points` assertion:
```python
    assert rec.body["results"] is not None and rec.body["results"]["peak_ratio"] == 2.0, "P78"
    assert rec.body["offgrid"] == {"blanks": 0, "of": 4 * 3}, \
        "P78: 4 planned points x the 3-point common grid (every stubbed omega_0 is 1.0)"
```
Then add:
```python
def test_an_all_failed_first_sweep_does_not_cost_the_second(store, monkeypatch):
    """P77 and spec §4.3/§8.2. Before piece 5 an all-failed S sweep raised out of the study before the
    T sweep had started. Now the S record stays on disk, unfinished (E2), and the T sweep runs and
    finishes. _sweep_stubs' Campaign-2 counter is shared by both sweeps: calls 0-1 are the S grid's."""
    from core import cli, config
    from core.FDT import cross_validation as cv

    _sweep_stubs(monkeypatch, phase_b_fail=(0, 1))
    cfg, s_grid, t_grid = cli.make_param_sweep_config(
        str(config.CELL_PATH / "nadrowski" / "master_spont.txt"),
        preset=dict(cli.SWEEP_PRESETS["exploratory"]), preset_name="exploratory",
        s_spec=(0.0, 0.1, 2), t_spec=(1.0, 1.1, 2), n_freqs=3, ensemble_M=2)
    writers = {"s": store.create("fdt", cfg, name="s_half"),
               "temp": store.create("fdt", cfg, name="t_half")}
    recs = cv.run_param_study_cli(cfg, s_grid=s_grid, t_grid=t_grid, writers=writers, seed=3)

    assert [r.name for r in recs] == ["t_half"], "only the sweep that measured something returns"
    rows = {s.name: s for s in store.list("fdt")}
    assert rows["t_half"].finished, "the temperature sweep ran and finished"
    assert rows["s_half"].complete and not rows["s_half"].finished, "the S record stays, unfinished"
    assert rows["s_half"].points_failed == 2


def test_a_study_whose_two_sweeps_both_measured_nothing_refuses(store, monkeypatch):
    """P77's other half: only when BOTH sweeps measured nothing does the study refuse, and both
    unfinished records stay on disk."""
    import pytest
    from core import cli, config
    from core.FDT import cross_validation as cv
    from core.refusals import Refusal

    _sweep_stubs(monkeypatch, phase_b_fail=(0, 1, 2, 3))
    cfg, s_grid, t_grid = cli.make_param_sweep_config(
        str(config.CELL_PATH / "nadrowski" / "master_spont.txt"),
        preset=dict(cli.SWEEP_PRESETS["exploratory"]), preset_name="exploratory",
        s_spec=(0.0, 0.1, 2), t_spec=(1.0, 1.1, 2), n_freqs=3, ensemble_M=2)
    writers = {"s": store.create("fdt", cfg, name="s_none"),
               "temp": store.create("fdt", cfg, name="t_none")}
    with pytest.raises(Refusal) as e:
        cv.run_param_study_cli(cfg, s_grid=s_grid, t_grid=t_grid, writers=writers, seed=3)
    assert e.value.field == "s_grid"
    rows = store.list("fdt")
    assert len(rows) == 2 and all(r.complete and not r.finished for r in rows)


def test_a_sweep_point_is_reproducible_from_the_seed_and_its_index(store, monkeypatch):
    """P82 / spec §4.1: point k of a sweep under `seed` draws exactly what a lone point under seed+k
    draws (Phase A) and seed+n_points+k (Phase B) -- so a point is reproducible from the seed and its
    index, and does not depend on how many points preceded it."""
    import torch
    from core import cli, config
    from core.diagnostics.rng import seeded
    from core.FDT import cross_validation as cv

    _sweep_stubs(monkeypatch)
    c1, c2 = cv.run_campaign1_psd, cv._campaign2_ratio
    drawn_a, drawn_b = [], []

    def _c1(cfg_op):
        drawn_a.append(float(torch.rand(())))
        return c1(cfg_op)

    def _c2(cfg_op, omegas, freqs_psd, G):
        drawn_b.append(float(torch.rand(())))
        return c2(cfg_op, omegas, freqs_psd, G)

    monkeypatch.setattr(cv, "run_campaign1_psd", _c1)
    monkeypatch.setattr(cv, "_campaign2_ratio", _c2)
    cfg, s_grid, _t = cli.make_param_sweep_config(
        str(config.CELL_PATH / "nadrowski" / "master_spont.txt"),
        preset=dict(cli.SWEEP_PRESETS["exploratory"]), preset_name="exploratory",
        s_spec=(0.0, 0.2, 3), t_spec=(1.0, 1.1, 2), n_freqs=3, ensemble_M=2)
    cfg.seed = 40
    cv.run_fdt_param_sweep(cfg, "s", s_grid, {"temp": 1.0}, writer=store.create("fdt", cfg))

    def alone(seed):
        with seeded(seed, torch.device("cpu")):
            return float(torch.rand(()))

    n = len(s_grid)
    assert drawn_a == [alone(40 + k) for k in range(n)]
    assert drawn_b == [alone(40 + n + k) for k in range(n)]
    assert len(set(drawn_a + drawn_b)) == 2 * n
```

**A9 — Steps 2, 7 and 8.** Step 2: `test_a_sweep_with_every_point_failed_refuses_after_its_record_is_written` ERRORS with the propagating `RuntimeError: s sweep produced NO usable points ...`; it does not report "DID NOT RAISE". The three new tests fail as follows:
- the P77 test with the same RuntimeError propagating out of the study;
- the both-refuse test with that RuntimeError instead of a Refusal;
- the P82 test with `RuntimeError: stub phase-A ...`? No: it fails on `drawn_b` only if the move breaks the seeding. Expect it to PASS already, since T19 landed the per-point seeding. Record that it passes and that it pins T19's P14 behaviour.

Step 7: `pytest tests/test_fdt_user.py -v`.

Step 8 (one `-k` per command):
- `pytest tests/test_artifact_store.py -k "public_entry or callers_config" -v`
- `pytest tests/test_tool.py -k "fdt or crossval" -v`

#### Controller rulings (pre-flight, binding — these supersede the amendments above and the task text)

- **F14/F45** — confirmed: `run_param_study_cli` returns only the FINISHED records (a list of `LoadedFdt`, never `None`); when both sweeps refuse it re-raises the activity sweep's `Refusal` (field `s_grid`) after logging both at error. State this in its docstring.
- **F41** — the all-points-failed refusal names the cell's FILE NAME as well as the grid, as spec §4.3 says ("a calm operator line naming the grid and the cell").

---

### Task 21: the four panels route a refusal apart from a bug; `_config_error` goes

**Why:** Spec §5.1 — the FDT, CrossVal, Reduction and Simulate panels stop wrapping their builder in
a broad `except` that ends at `BasePanel._config_error` ("The configuration could not be built."); a
`Refusal` goes to `BasePanel._refusal` and anything else to the red box with its traceback, and
`_config_error` is then uncalled and removed, which is what piece 3's spec said piece 5 would do
(`2026-09-15-validation-and-logging-design.md:322-323`). §5.1's second half — the Reduction panel's
two most plausible cell mistakes must stay in the yellow box — is already satisfied by piece 3's
commit `388d628` (see the objections); what it was there to buy, §6.2's one message builder for the
cell rule, is delivered here so Task 29 has it.

**Files:**
- Modify: `core/gui/panels/fdt_panel.py:130-137` (`except Exception as e:  # noqa: BLE001 -- see BasePanel._config_error`)
- Modify: `core/gui/panels/crossval_panel.py:146-153` (`except Exception as e:  # noqa: BLE001 -- see _config_error`)
- Modify: `core/gui/panels/reduction_panel.py:67-71` (`except Exception as e:  # noqa: BLE001 -- see BasePanel._config_error`)
- Modify: `core/gui/panels/simulate_panel.py:105-109` (same comment)
- Modify: `core/gui/panels/simulate_runner.py:49-72` (the two `raise ValueError(` in `build_stream_config`)
- Modify: `core/gui/panels/base_panel.py:398-416` (`def _config_error(self, exc: Exception):`)
- Modify: `core/refusals.py` (add `missing_values_phrase`)
- Modify: `core/cli.py:68` and `core/sim_config.py:247-250` (the two other wordings of one rule)
- Test: `tests/test_nav_and_gating.py:1688-1733` (rewritten), `:1615-1617` (a stale sentence),
  `tests/test_refusals.py` (the new phrase)

**Interfaces:**
- Consumes: `core.gui.fields.fix_sentence(key)` widened by T9 so that `cell`, `model` and `units`
  name every place they appear; `core.refusals.Refusal(message, *, field=None)`;
  `BasePanel._refusal(exc)` and `BasePanel._on_error(exc, tb, cancel_noted=False)` as they are today.
- Produces: `BasePanel` no longer has `_config_error` (T22, T23, T25, T26 must not call it);
  `core.refusals.missing_values_phrase(label: str, missing) -> str`, the one wording Task 29's
  §6.2 item needs; `simulate_runner.build_stream_config` raises `Refusal(field="cell")` and
  `Refusal(field="model")` instead of bare `ValueError`s (both remain `ValueError` subclasses, so
  `tests/test_user_models.py:639`'s `except ValueError` still catches).

- [ ] **Step 1: Rewrite the test that pins today's unconverted behaviour**

In `tests/test_nav_and_gating.py`, find:

```python
def test_the_secondary_panels_still_show_a_bad_cell_as_check_your_inputs(monkeypatch, tmp_path):
    """The Simulate, Reduction, CrossVal and FDT panels are piece 5's. Until then their builder
    failures stay on BasePanel._config_error, whose box is titled "Check your inputs" and reads "The
    configuration could not be built." over the builder's own sentence. Pinned for BOTH shapes the
    builders raise across piece 3 -- the bare ValueError they raise today for a cell missing a
    parameter, and the Refusal(field="cell") they raise once cli is converted -- so the four panels
    keep the same box whichever lands first, and nothing is dispatched. The routing change on the
    inference tabs must not leak here."""
```

and replace the whole function (it ends at `assert box.detailedText() == "", name`) with:

```python
def test_the_four_secondary_panels_route_a_refusal_apart_from_a_bug(monkeypatch, tmp_path):
    """Spec §5.1. The FDT, CrossVal, Reduction and Simulate panels used to wrap their builder in a
    broad ``except`` ending at ``BasePanel._config_error``, whose box read "The configuration could
    not be built." over whatever sentence it had caught -- one box for a blank number and for a bug
    in the parser alike, and no way to tell which you were looking at. They now do what the six
    inference tabs have done since piece 3: a ``Refusal`` opens the yellow "Check your inputs" box
    with the CORE's own sentence as its text and this front end's "where to fix it" under it, and
    anything else is a bug and keeps the red box with its traceback behind Details. Nothing is
    dispatched either way.

    This REPLACES test_the_secondary_panels_still_show_a_bad_cell_as_check_your_inputs, which pinned
    the unconverted behaviour across both exception shapes on purpose: a half-converted state cannot
    be expressed in it, so it is rewritten rather than extended (spec §5.1).

    The source pin at the end is the point of the section: ``_config_error`` is gone from BasePanel
    and no module under core/gui names it, so no later panel can quietly route a failure back into a
    box that says nothing about what was wrong.
    """
    import ast
    from pathlib import Path
    from PySide6.QtWidgets import QMessageBox
    from core import cli
    from core.gui import fields as gui_fields
    from core.gui.panels import simulate_panel as sim_mod
    from core.gui.panels.base_panel import BasePanel
    from core.gui.panels.crossval_panel import CrossValPanel
    from core.gui.panels.fdt_panel import FdtPanel
    from core.gui.panels.reduction_panel import ReductionPanel
    from core.gui.panels.simulate_panel import SimulatePanel
    from core.refusals import Refusal
    from tests._fixtures import SHOWN, qt_app

    qt_app()
    cell = tmp_path / "bad_cell.txt"
    cell.write_text("# a cell that will not build\n", encoding="utf-8")
    panels = {"fdt": FdtPanel(), "reduction": ReductionPanel(), "crossval": CrossValPanel(),
              "simulate": SimulatePanel()}
    for p in panels.values():
        p.cell_picker.selected_path = lambda: str(cell)
        p.dispatch = lambda *a, **k: pytest.fail("a panel dispatched with a config that did not build")
    clicks = {"fdt": panels["fdt"]._run, "reduction": panels["reduction"]._run,
              "crossval": panels["crossval"]._run, "simulate": panels["simulate"]._start}
    msg = f"Cell file {cell.name!r} is missing value(s) the bounds file requires: k_gs."

    def _patch(raiser):
        monkeypatch.setattr(cli, "make_fdt_config", raiser)
        monkeypatch.setattr(cli, "make_reduction_config", raiser)
        monkeypatch.setattr(cli, "make_param_sweep_config", raiser)
        monkeypatch.setattr(sim_mod, "build_stream_config", raiser)

    # (a) a Refusal: the yellow box, the core's sentence, this front end's fix line, no traceback
    def _refusing(*a, **k):
        raise Refusal(msg, field="cell")

    _patch(_refusing)
    for name, click in clicks.items():
        SHOWN.clear()
        click()
        box = SHOWN[-1]
        assert box.windowTitle() == "Check your inputs" and box.icon() == QMessageBox.Warning, name
        assert box.text() == msg, name
        assert box.informativeText() == gui_fields.fix_sentence("cell"), name
        assert box.detailedText() == "", name

    # (b) anything else is a bug: the red box, the message, and the traceback behind Details
    def _booming(*a, **k):
        raise RuntimeError("the parser fell over")

    _patch(_booming)
    for name, click in clicks.items():
        SHOWN.clear()
        click()
        box = SHOWN[-1]
        assert box.windowTitle() == "Error" and box.icon() == QMessageBox.Critical, name
        assert box.text() == "the parser fell over", name
        assert "RuntimeError" in box.detailedText(), name

    # (c) the generic box is gone, and nothing under core/gui reaches for it
    assert not hasattr(BasePanel, "_config_error")
    gui_root = Path(sim_mod.__file__).resolve().parents[1]
    for path in sorted(gui_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        named = [n for n in ast.walk(tree)
                 if isinstance(n, ast.Attribute) and n.attr == "_config_error"]
        assert not named, f"{path.name} still routes a failure through _config_error"
```

- [ ] **Step 2: Run it and watch it fail**

Run: `pytest tests/test_nav_and_gating.py::test_the_four_secondary_panels_route_a_refusal_apart_from_a_bug -v`
Expected: FAIL at (a)'s first box text —
`AssertionError: fdt` with `assert 'The configuration could not be built.' == "Cell file 'bad_cell.txt' is missing value(s) the bounds file requires: k_gs."`

- [ ] **Step 3: Route the FDT panel's builder failure by kind**

In `core/gui/panels/fdt_panel.py`, find:

```python
from PySide6.QtWidgets import QCheckBox, QComboBox, QFormLayout, QGroupBox, QPushButton

from core import cli, config, registry
```

Replace with:

```python
import traceback

from PySide6.QtWidgets import QCheckBox, QComboBox, QFormLayout, QGroupBox, QPushButton

from core import cli, config, registry
from core.refusals import Refusal
```

Then find:

```python
        except Exception as e:                       # noqa: BLE001 -- see BasePanel._config_error
            self._config_error(e)
            return
```

Replace with:

```python
        except Refusal as e:                         # a setting the user can change: the yellow box
            self._refusal(e)
            return
        except Exception as e:                       # noqa: BLE001 -- a bug in the builder: the red box
            self._on_error(e, traceback.format_exc())
            return
```

- [ ] **Step 4: The same for the CrossVal panel**

In `core/gui/panels/crossval_panel.py`, find:

```python
from core import cli, config
from core.config import CELL_PATH
```

Replace with:

```python
import traceback

from core import cli, config
from core.config import CELL_PATH
from core.refusals import Refusal
```

(the `import traceback` goes above the `from PySide6...` block at the top of the file, beside the
module docstring). Then find:

```python
        except Exception as e:                       # noqa: BLE001 -- see _config_error
            self._config_error(e)
            return
```

Replace with:

```python
        except Refusal as e:                         # a setting the user can change: the yellow box
            self._refusal(e)
            return
        except Exception as e:                       # noqa: BLE001 -- a bug in the builder: the red box
            self._on_error(e, traceback.format_exc())
            return
```

Leave `_on_cell_changed`'s broad `except` exactly as it is: it guards `__init__`, and its comment
says why any exception there must degrade one label rather than brick the window at launch.

- [ ] **Step 5: The same for the Reduction panel**

In `core/gui/panels/reduction_panel.py`, find:

```python
from core import cli, config
from core.config import CELL_PATH
from core.Reduction.sweep import run_reduction_map
```

Replace with:

```python
import traceback

from core import cli, config
from core.config import CELL_PATH
from core.refusals import Refusal
from core.Reduction.sweep import run_reduction_map
```

Then find:

```python
        except Exception as e:                       # noqa: BLE001 -- see BasePanel._config_error
            self._config_error(e)
            return
```

Replace with:

```python
        except Refusal as e:                         # cli.parse_cell's cell refusals: the yellow box
            self._refusal(e)
            return
        except Exception as e:                       # noqa: BLE001 -- a bug in the builder: the red box
            self._on_error(e, traceback.format_exc())
            return
```

- [ ] **Step 6: The same for the Simulate panel**

In `core/gui/panels/simulate_panel.py`, find:

```python
from pathlib import Path

import numpy as np
```

Replace with:

```python
import traceback
from pathlib import Path

import numpy as np
```

Then find:

```python
        except Exception as e:                       # noqa: BLE001 -- see BasePanel._config_error
            self._config_error(e)
            return
```

Replace with:

```python
        except Refusal as e:                         # a cell or model problem: the yellow box
            self._refusal(e)
            return
        except Exception as e:                       # noqa: BLE001 -- a bug in the builder: the red box
            self._on_error(e, traceback.format_exc())
            return
```

- [ ] **Step 7: `build_stream_config`'s two bare ValueErrors become Refusals**

Removing `_config_error` would otherwise move the Simulate panel's two most plausible mistakes into
the red crash box — the same hazard spec §5.1 names for the Reduction panel. Both keys already exist
in `core.refusals.FIELDS`, so no registry change is needed.

In `core/gui/panels/simulate_runner.py`, find:

```python
from core import cli, forcing, registry
from core.SBI import pipeline
from core.Solvers import sdeint
from core.config import BOUNDS_PATH, DT_EXP_S, VALID_LABELS, VALID_MODELS, cpu_device
```

Replace with:

```python
from core import cli, forcing, registry
from core.refusals import Refusal
from core.SBI import pipeline
from core.Solvers import sdeint
from core.config import BOUNDS_PATH, DT_EXP_S, VALID_LABELS, VALID_MODELS, cpu_device
```

Then find:

```python
    if bounds_file is None:
        raise ValueError(
            f"No bounds file governs '{Path(cell_path).name}': tried the sibling "
            f"{BOUNDS_PATH / model.lower() / Path(cell_path).name} and the shared "
            f"{BOUNDS_PATH / model.lower() / cli.MASTER_BOUNDS_NAME}. Bounds declare which "
            f"parameters are inferred and in what order, so one is required.")
```

Replace with:

```python
    if bounds_file is None:
        raise Refusal(
            f"No bounds file governs '{Path(cell_path).name}': tried the sibling "
            f"{BOUNDS_PATH / model.lower() / Path(cell_path).name} and the shared "
            f"{BOUNDS_PATH / model.lower() / cli.MASTER_BOUNDS_NAME}. Bounds declare which "
            f"parameters are inferred and in what order, so one is required.", field="cell")
```

Then find:

```python
        if actual != expected:
            raise ValueError(
                f"Model '{model}' is out of sync with its bounds file: the definition uses "
                f"parameters {expected} but the bounds file lists {actual}. Re-save the model "
                "from the Settings model builder to regenerate its files.")
```

Replace with:

```python
        if actual != expected:
            raise Refusal(
                f"Model '{model}' is out of sync with its bounds file: the definition uses "
                f"parameters {expected} but the bounds file lists {actual}. Re-saving the model "
                "regenerates its files.", field="model")
```

(the sentence stops naming the Settings screen and the builder: a core message names no surface, and
`core/gui/fields.py` is what says where to go.)

- [ ] **Step 8: Remove `_config_error`**

In `core/gui/panels/base_panel.py`, find and DELETE the whole method:

```python
    def _config_error(self, exc: Exception):
        """Report a failed config build as user-input trouble, not a crash.

        For the Simulate, Reduction, CrossVal and FDT panels until piece 5; the inference tabs route
        through _refusal/_on_error. Deliberately catches broadly at the call sites: cli's builders
        raise a bare ValueError (NOT UnitParseError) for the two most plausible user mistakes -- a
        cell with no sibling bounds file (cli.parse_cell) and a cell missing a param the bounds file
        requires (cli.load_and_validate_gt -> SimConfig.inject_ground_truth). A narrow
        `except cli.UnitParseError` lets those escape the clicked slot and surface as a raw traceback
        in app.py's last-resort excepthook, with nothing in the panel's own log.
        """
        msg = str(exc)
        self.log_pane.append_line(f"Could not build the config: {msg}", "error")
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)                  # user input, not a crash
        box.setWindowTitle("Check your inputs")
        box.setText("The configuration could not be built.")
        box.setInformativeText(msg)
        box.exec()
```

`QMessageBox` stays imported: `_on_error` still builds one.

- [ ] **Step 9: Correct the sentence that promised the method would survive**

In `tests/test_nav_and_gating.py`, find:

```python
    And a source pin, because the routing is a rule for all five inference tabs: none of
    core/gui/panels/inference/*.py calls _config_error any more. That method stays on BasePanel for
    the Simulate, Reduction, CrossVal and FDT panels until piece 5 retires it."""
```

Replace with:

```python
    And a source pin, because the routing is a rule for all five inference tabs: none of
    core/gui/panels/inference/*.py calls _config_error any more. Piece 5 retired that method
    altogether once the four section panels were converted; the pin stays, because the scan is what
    stops a new tab reintroducing the call."""
```

- [ ] **Step 10: Run the panel tests and watch them pass**

Run: `pytest tests/test_nav_and_gating.py -v -k "secondary or config_error or rename_failure" tests/test_simulate.py tests/test_user_models.py::test_stale_bounds_file_is_detected`
Expected: PASS

- [ ] **Step 11: Write the failing test for the one cell wording**

§6.2 asks that `load_and_validate_gt`'s refusal and the dry run's problem strings stop being two
wordings for one rule. Three sites say it three ways today: `core/cli.py:68`
(`"missing {label}(s) the bounds file requires: ..."`), `core/cli.py:114`
(`"Cell file '...' is missing value(s) for {label} required by the bounds file: [...]"`) and
`core/sim_config.py:250` (`"Cell file is missing {label} required by the bounds file: [...]"`).

In `tests/test_refusals.py`, add after `test_the_registry_holds_exactly_the_initial_keys_with_neutral_descriptions`:

```python
def test_one_phrase_says_the_cell_is_missing_what_the_bounds_file_declares():
    """Spec §6.2. "the cell does not supply something the bounds file declares" is ONE rule, and it
    was worded three ways: the dry run's problem string (cli.validate_gt_file), the decoupled
    parser's refusal (cli._merge_vals_bounds) and the injection's (SimConfig._fill_checked). The
    command line prints one of them and the window's cell picker prints another for the same cell,
    which is how "the tool and the window disagree" reports start. One builder now produces the
    phrase and all three splice it, so the operator reads the same words wherever the mistake is
    caught.

    It returns a FRAGMENT -- lower case, no trailing period -- because the dry run joins its problems
    with "; " into a sentence of its own (infer_tab._on_cell_changed), while the two refusals put a
    capital in front of it and a period after it. A list is rendered as plain comma-separated names,
    never as a repr'd Python list, which is what two of the three did.
    """
    from core.refusals import missing_values_phrase
    assert missing_values_phrase("ND parameter", ["k_gs", "gamma"]) == \
        "missing ND parameter(s) the bounds file requires: k_gs, gamma"
    assert missing_values_phrase("rescale parameter", ("x_scale",)) == \
        "missing rescale parameter(s) the bounds file requires: x_scale"
    assert "[" not in missing_values_phrase("forcing parameter", ["amp"])
    phrase = missing_values_phrase("ND parameter", ["k_gs"])
    assert phrase[0].islower() and not phrase.endswith(".")


def test_the_three_cell_sites_all_splice_the_one_phrase():
    """The other half of §6.2: the phrase exists AND is the only wording each of the three sites
    produces. Asserted on the executable source rather than by provoking three failures, because two
    of the three need a parsed bounds file and a SimConfig to reach -- and what would regress is
    somebody re-typing the sentence beside the call, which source is the honest witness for."""
    from core import cli, sim_config
    from tests._fixtures import code_only
    for obj in (cli.validate_gt_file, cli._merge_vals_bounds, sim_config.SimConfig._fill_checked):
        src = code_only(obj)
        assert "missing_values_phrase(" in src, obj
        assert "the bounds file requires" not in src.replace("missing_values_phrase(", ""), obj
```

- [ ] **Step 12: Run them and watch them fail**

Run: `pytest tests/test_refusals.py -v -k "one_phrase or three_cell_sites"`
Expected: FAIL with `ImportError: cannot import name 'missing_values_phrase' from 'core.refusals'`

- [ ] **Step 13: Add the phrase builder**

In `core/refusals.py`, find:

```python
def describe(key: str) -> str:
```

and insert ABOVE it:

```python
def missing_values_phrase(label: str, missing) -> str:
    """"missing ND parameter(s) the bounds file requires: k_gs, gamma" -- the ONE wording for a cell
    that does not supply something the bounds file declares.

    A fragment, deliberately: ``cli.validate_gt_file`` joins its problems with "; " into a sentence
    of its own, while ``cli._merge_vals_bounds`` and ``SimConfig._fill_checked`` put a capital in
    front and a period after. Three sites said this three ways before piece 5 (spec §6.2), so the
    tool and the window could describe one mistake differently. Here rather than in ``core/cli.py``
    because ``core/sim_config.py`` is imported by ``core/config.py``, which ``cli`` imports: this
    module is the one both may already reach, and it costs nothing to import.
    """
    return f"missing {label}(s) the bounds file requires: {', '.join(str(n) for n in missing)}"
```

- [ ] **Step 14: Splice it at the three sites**

In `core/cli.py`, find:

```python
            problems.append(f"missing {label}(s) the bounds file requires: {', '.join(missing)}")
```

Replace with:

```python
            problems.append(missing_values_phrase(label, missing))
```

In the same file find:

```python
    missing = [name for name in bounds if name not in vals]
    if missing:
        raise Refusal(
            f"Cell file '{cell_file}' is missing value(s) for {label} required by the bounds file: {missing}.",
            field="cell")
```

Replace with:

```python
    missing = [name for name in bounds if name not in vals]
    if missing:
        raise Refusal(f"Cell file '{cell_file}' is "
                      f"{missing_values_phrase(label, missing)}.", field="cell")
```

and change the import line

```python
from .refusals import Refusal, require_file
```

to

```python
from .refusals import Refusal, missing_values_phrase, require_file
```

`_merge_vals_bounds`'s callers pass `label` as `"ND parameters"`, `"rescale parameters"` and
`"forcing parameters"`; make them singular so the phrase reads "(s)" once. In `parse_cell`, find:

```python
        params_dict = _merge_vals_bounds(v_params, b_params, "ND parameters", cell_file)
        rescale_params = _merge_vals_bounds(v_rescale, b_rescale, "rescale parameters", cell_file)
        force_params_dict = _merge_vals_bounds(v_forcing, b_forcing, "forcing parameters", cell_file)
```

Replace with:

```python
        params_dict = _merge_vals_bounds(v_params, b_params, "ND parameter", cell_file)
        rescale_params = _merge_vals_bounds(v_rescale, b_rescale, "rescale parameter", cell_file)
        force_params_dict = _merge_vals_bounds(v_forcing, b_forcing, "forcing parameter", cell_file)
```

In `core/sim_config.py`, find:

```python
        missing = sorted(set(cfg_dict) - set(cell_vals))
        if missing:
            raise Refusal(
                f"Cell file is missing {label} required by the bounds file: {missing}.", field="cell")
```

Replace with:

```python
        missing = sorted(set(cfg_dict) - set(cell_vals))
        if missing:
            raise Refusal(f"Cell file is {missing_values_phrase(label, missing)}.", field="cell")
```

and add `missing_values_phrase` to that file's import of the refusals module (find the existing
`from .refusals import` line and append the name to it; if `Refusal` is imported alone, the line
becomes `from .refusals import Refusal, missing_values_phrase`).

`_fill_checked`'s `label` is also spliced into the out-of-bounds message
(`f"Cell file {label} outside the bounds file's bounds: "`), so keep `inject_ground_truth`'s calls
passing `"ND parameters"` / `"rescale parameters"` / `"forcing parameters"` unchanged and pass the
singular only to the phrase: replace the line above with

```python
            raise Refusal(f"Cell file is "
                          f"{missing_values_phrase(label.rstrip('s'), missing)}.", field="cell")
```

- [ ] **Step 15: Run the tests and watch them pass**

Run: `pytest tests/test_refusals.py tests/test_nav_and_gating.py tests/test_simulate.py tests/test_artifact_consistency.py -v`
Expected: PASS

- [ ] **Step 16: Commit**

```bash
git add core/gui/panels/base_panel.py core/gui/panels/fdt_panel.py core/gui/panels/crossval_panel.py core/gui/panels/reduction_panel.py core/gui/panels/simulate_panel.py core/gui/panels/simulate_runner.py core/refusals.py core/cli.py core/sim_config.py tests/test_nav_and_gating.py tests/test_refusals.py
git commit -m "gui: the four section panels route refusals apart from bugs; _config_error is gone"
```

#### Amendments (binding — these supersede the text above)

Rulings that change this task: **P38, P53, P76**, plus anchor corrections against the files T10, T17 and T19 leave behind. Rulings that confirm what is written here: P4, P8, P29. P32: `units` is NOT widened by T9, so ignore "and `units`" in the Interfaces line; no step changes.

**A. P53 — `cli._merge_vals_bounds` and `parse_cell` are NOT touched.**
- In Step 14, DELETE the second find/replace: the block starting `missing = [name for name in bounds if name not in vals]` and its `raise Refusal(f"Cell file '{cell_file}' is " ...` replacement.
- Also DELETE the whole paragraph "`_merge_vals_bounds`'s callers pass `label` as ... make them singular" and its `parse_cell` find/replace. `parse_cell` keeps `"ND parameters"`, `"rescale parameters"` and `"forcing parameters"`.
- The phrase is spliced at exactly TWO sites: `cli.validate_gt_file` and `SimConfig._fill_checked`.
- `_merge_vals_bounds` keeps its own sentence, which carries the cell path. It is a known third wording, handed on (spec §6.2).
- The Files block ("`core/cli.py:68` and `core/sim_config.py:247-250`") is now right as written.

**B. P38 — the phrase uses its label AS GIVEN and never pluralises it.**

Step 13: insert this in place of the function shown there, still immediately ABOVE `def describe(key: str) -> str:`:
```python
def missing_values_phrase(label: str, missing) -> str:
    """"missing ND parameters the bounds file requires: k_gs, gamma" -- the ONE wording for a cell
    that does not supply something the bounds file declares (spec §6.2).

    A fragment, deliberately: ``cli.validate_gt_file`` returns it as one of its problem strings,
    which the Infer tab joins with "; ", while ``SimConfig._fill_checked`` puts "Cell file is " in
    front and a period after -- so the dry run and the refusal say the same words. ``label`` is used
    AS GIVEN (the callers pass the plural, "ND parameters") and is never pluralised here. A third
    wording of the same rule, ``cli._merge_vals_bounds``, is knowingly left alone: it carries the
    cell path. Here rather than in ``core/cli.py`` because ``core/sim_config.py`` cannot import
    ``core/cli.py``.
    """
    return f"missing {label} the bounds file requires: {', '.join(str(n) for n in missing)}"
```

Step 14, `cli.validate_gt_file`: replace `problems.append(f"missing {label}(s) the bounds file requires: {', '.join(missing)}")` with:
```python
            # the phrase takes its label as given; this loop's labels are singular because the
            # out-of-bounds line below reads "ND parameter k = ...", so the plural is spelled here
            problems.append(missing_values_phrase(f"{label}s", missing))
```

Step 14, `SimConfig._fill_checked`: replace the `missing` block with the following, and DELETE the later `label.rstrip('s')` paragraph and its code:
```python
        missing = sorted(set(cfg_dict) - set(cell_vals))
        if missing:
            raise Refusal(f"Cell file is {missing_values_phrase(label, missing)}.", field="cell")
```
`inject_ground_truth`'s plural labels stay as they are, and so does the out-of-bounds message.

**C. Import anchors, rewritten against the files as earlier tasks leave them.**
- Step 14, core/cli.py: T10 has already replaced `from .refusals import Refusal, require_file` with a parenthesised list. Add `missing_values_phrase` to THAT list (after `describe`). Do not add a second import line. Expected result:
  `from .refusals import (Refusal, describe, missing_values_phrase, refuse, require_at_least,`
  `                       require_below, require_choice, require_file, require_finite, require_positive)`
  If T10/T11 left a different list, add the one name to it.
- Step 14, core/sim_config.py: the line is `from core.refusals import Refusal` (absolute, not `from .refusals`). Replace it with `from core.refusals import Refusal, missing_values_phrase`.
- Step 3, core/gui/panels/fdt_panel.py: T17 has already changed `from core import cli, config, registry` to `from core import cli, registry` + `from core.artifacts import default_store`, so the quoted three-line block is gone. Instead:
  (i) find `from PySide6.QtWidgets import QCheckBox, QComboBox, QFormLayout, QGroupBox, QPushButton` and insert `import traceback` plus one blank line immediately ABOVE it;
  (ii) find `from core.config import CELL_PATH, VALID_MODELS` and insert `from core.refusals import Refusal` immediately BELOW it.
  The except-arm find/replace is unchanged.
- Step 4, core/gui/panels/crossval_panel.py: do not rely on the two-line anchor, because T19 added `from core.artifacts import default_store` somewhere in that block. Instead:
  (i) find `from PySide6.QtWidgets import (QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QPushButton,` and insert `import traceback` plus one blank line immediately ABOVE it;
  (ii) find the line `from core.config import CELL_PATH` and insert `from core.refusals import Refusal` immediately BELOW it.
  Leave `from core import cli, config` exactly as T19 left it. The except-arm find/replace is unchanged.

**D. Step 11: the tests, replaced (P38, P53, P76).**

In `tests/test_refusals.py`, after `test_the_registry_holds_exactly_the_initial_keys_with_neutral_descriptions`, add these three tests INSTEAD of the two in Step 11. `pytest` and `Refusal` are already module-level imports there.
```python
def test_one_phrase_says_the_cell_is_missing_what_the_bounds_file_declares():
    """Spec §6.2. "the cell does not supply something the bounds file declares" is ONE rule, and the
    dry run (cli.validate_gt_file) and the injection (SimConfig._fill_checked) worded it two ways. One
    builder now produces the phrase. It is a FRAGMENT -- lower case, no trailing period -- because
    the dry run's problems are joined with "; " (infer_tab._on_cell_changed) while the refusal puts
    "Cell file is " in front and a period after. The label is used as given, never pluralised
    (P38), and a list is plain comma-separated names, never a repr'd Python list."""
    from core.refusals import missing_values_phrase
    assert missing_values_phrase("ND parameters", ["k_gs", "gamma"]) == \
        "missing ND parameters the bounds file requires: k_gs, gamma"
    assert missing_values_phrase("rescale parameters", ("x_scale",)) == \
        "missing rescale parameters the bounds file requires: x_scale"
    assert "[" not in missing_values_phrase("forcing parameters", ["amp"])
    phrase = missing_values_phrase("ND parameters", ["k_gs"])
    assert phrase[0].islower() and not phrase.endswith(".")


def test_the_two_cell_sites_both_splice_the_one_phrase():
    """The other half of §6.2: each of the two sites builds its wording through the phrase and does
    not re-type the sentence beside the call. Asserted on executable source, because what would
    regress is somebody re-typing the sentence. cli._merge_vals_bounds is a known third wording,
    left alone on purpose (P53: it carries the cell path), so it is not in this list."""
    from core import cli, sim_config
    from tests._fixtures import code_only
    for obj in (cli.validate_gt_file, sim_config.SimConfig._fill_checked):
        src = code_only(obj)
        assert "missing_values_phrase(" in src, obj
        assert "the bounds file requires" not in src, obj


def test_the_dry_run_and_the_injection_word_a_missing_value_identically(monkeypatch):
    """Spec §6.2 and §8.2 ("the cell refusal's wording is identical from load_and_validate_gt and
    from the dry run"), P76. The refusal is exactly "Cell file is " + the dry run's problem + ".".
    Both are reached without a bounds file: the dry run through a stand-in config carrying the three
    dicts it reads and a stubbed values parser; the injection through the static
    SimConfig._fill_checked, called with the label inject_ground_truth passes."""
    import types
    from collections import OrderedDict
    from core import cli
    from core.sim_config import SimConfig

    monkeypatch.setattr(cli.file_manager, "parse_values_file",
                        lambda path: ({"x": 0.0}, {}, {}, {}))
    declared = OrderedDict(k_gs=(None, (0.0, 1.0)))
    cfg = types.SimpleNamespace(params_dict=OrderedDict(declared), rescale_params=OrderedDict(),
                                force_params_dict=OrderedDict())
    problems = cli.validate_gt_file(cfg, "unused_cell.txt")
    assert problems == ["missing ND parameters the bounds file requires: k_gs"], problems
    with pytest.raises(Refusal) as exc:
        SimConfig._fill_checked("ND parameters", {}, OrderedDict(declared), check_bounds=True)
    assert exc.value.field == "cell"
    assert exc.value.message == f"Cell file is {problems[0]}."
```

**Step 12 becomes:**
- Run: `pytest tests/test_refusals.py -v -k "one_phrase or two_cell_sites or dry_run_and_the_injection"`
- Expected, first test: FAIL with `ImportError: cannot import name 'missing_values_phrase' from 'core.refusals'`.
- Expected, second test: FAIL on `assert "missing_values_phrase(" in src`.
- Expected, third test: FAIL on `assert problems == [...]`. Today the problem reads `missing ND parameter(s) the bounds file requires: k_gs`.

**E. Step 10: the run command, corrected.** `-k` applies to every path given, so the command as written deselects `test_stale_bounds_file_is_detected`. Run it as two commands:
```
pytest tests/test_nav_and_gating.py -v -k "secondary or config_error or rename_failure"
pytest tests/test_simulate.py tests/test_user_models.py::test_stale_bounds_file_is_detected -v
```

**F. Step 7 docstring correction.** In `build_stream_config`'s docstring, replace `the panel catches it as a config error.` with `the panel shows it as a refusal (the yellow box).`: `_config_error` no longer exists.

**G. P76 / P4.** T29 no longer consumes `missing_values_phrase` and does not re-splice these sites. T21 is the phrase's only producer and the only task that splices it. Ignore the Why's "so Task 29 has it" and the Interfaces' "the one wording Task 29's §6.2 item needs".

Steps 1-9 (apart from C), 15 and 16 are unchanged. The `git add` line is unchanged: `core/cli.py` and `core/sim_config.py` are still modified.

#### Controller rulings (pre-flight, binding — these supersede the amendments above and the task text)

- **F15** — the Reduction panel's F0 is checked at the click, panel-side, as Task 23 does for Simulate: inside the try in `ReductionPanel._run`, `F0 = require_positive("f0", self.f0.value_or_none())`, then `cli.make_reduction_config(cell, F0=F0)`. `make_reduction_config` itself stays untouched (P19). Add a blank-box case to this task's test.
- **F42** — keep the stubbed Reduction test, and add one docstring sentence naming the real builder's two refusals (`core/cli.py`'s missing-value and time-unit refusals) and why the stub suffices.
- **F43** — keep the new cell-site tests in `tests/test_refusals.py` and correct that module's docstring, which claims it is torch-free apart from one test; it no longer is.

---

### Task 22: the model-builder screen shows its refusals in the yellow box

**Why:** Spec §1.2 puts the model builder inside §5's set of five surfaces, and §5.1 says its refusal
goes through `core/gui/widgets/refusal_box.show_refusal` because it is a `QWidget`, not a
`BasePanel`. Spec §5.3 gives its own numeric fields field keys — the parameter row's value, minimum
and maximum, the initial condition, `x_scale`, `t_scale`, and each forcing field.

**Files:**
- Modify: `core/refusals.py` (seven keys in `FIELDS`)
- Modify: `core/gui/fields.py` (`CONTROL` — seven entries on the Model Builder screen)
- Modify: `core/tool/fields.py` (`FLAG` — seven `None` entries)
- Modify: `core/gui/screens/model_builder_screen.py:107-116` (`_VarRow.values`),
  `:366-437` (`_validate`), `:439-443` (`_validate_clicked`), `:445-466` (`_save`),
  `:258-263` (the Display scales rows)
- Test: `tests/test_user_models.py` (three existing tests updated, one new)
- Test: `tests/test_refusals.py:29-39` (`BASE_KEYS`), `:91` (the count), `:805` (the None-flag set)

**Interfaces:**
- Consumes: T8's `core.refusals.refuse(key, message)` / `require_positive` (both already exist) and
  T9's `core.gui.fields.SCREENS`, which makes a `("Model Builder", …)` tuple render "on the Model
  Builder screen"; `core.gui.widgets.refusal_box.show_refusal(parent, exc)`.
- Produces: `FIELDS` keys `param_value`, `param_min`, `param_max`, `init`, `x_scale`, `t_scale`,
  `forcing_param`; `ModelBuilderScreen._refusal(exc)`; `ModelBuilderScreen._validate()` now RAISES
  `Refusal` for a field problem instead of returning `None` (it still returns `None` for the
  non-field refusals — a task running, no variables set, a parse or integration failure);
  `_VarRow.forcing_fields() -> dict`.

- [ ] **Step 1: Write the failing test**

In `tests/test_user_models.py`, add after `test_builder_refuses_a_log_box_with_a_non_positive_minimum`:

```python
def test_the_builder_shows_a_field_refusal_in_the_yellow_box(monkeypatch):
    """Spec §1.2, §5.1, §5.3. The model builder is the fifth surface of piece 5's set, and the only
    one that is a plain QWidget rather than a BasePanel -- it has no ``_refusal`` and no log pane --
    so its refusals go through the shared ``refusal_box.show_refusal`` and are recorded on its own
    status line instead. Before this, every one of its input problems was a status-line sentence
    only: a form a full screen tall could refuse to save with a message at the bottom of it, and the
    wording was this screen's alone, so the same mistake read differently here and everywhere else.

    Each of its numeric fields now carries a registry key, so the box's informative line names the
    box on THIS screen -- "on the Model Builder screen", not "on the Infer tab", which is the collision
    E6 widened the table for. Four are checked here, one per shape: a blank bound (FloatField.value()
    reads a blank as 0.0, the hazard _ParamRow's own docstring warns about), an inverted pair, a
    non-positive display scale, and a blank forcing parameter -- which reached model_store as a real
    0.0 nobody typed.
    """
    from PySide6.QtWidgets import QMessageBox
    from core.gui import fields as gui_fields
    from core.gui.screens.model_builder_screen import ModelBuilderScreen
    from core.refusals import Refusal
    from tests._fixtures import SHOWN, qt_app

    qt_app()
    mb = ModelBuilderScreen()
    mb.vars_edit.setText("x")
    mb._set_variables()
    mb._var_rows[0].drift.setText("-k*x")
    mb._var_rows[0].noise.setText("d0")
    mb.name_edit.setText("UMTESTYELLOW")
    mb._detect_params()
    mb._param_fields["k"].set_spec(1.0, 0.5, 1.5)
    mb._param_fields["d0"].set_spec(0.01, 0.001, 0.1)

    def _refused(field):
        SHOWN.clear()
        mb._validate_clicked()
        box = SHOWN[-1]
        assert box.windowTitle() == "Check your inputs" and box.icon() == QMessageBox.Warning
        assert box.detailedText() == "", box.detailedText()
        assert box.informativeText() == gui_fields.fix_sentence(field), box.informativeText()
        assert box.text() in mb.status.text(), mb.status.text()
        return box.text()

    # (a) a blank minimum: refused as blank, never read as a bound of 0.0
    mb._param_fields["k"].lo.setText("")
    assert mb._param_fields["k"].spec()[1] is None
    assert "'k'" in _refused("param_min")
    with pytest.raises(Refusal) as ei:
        mb._validate()
    assert ei.value.field == "param_min"

    # (b) an inverted pair
    mb._param_fields["k"].set_spec(1.0, 0.5, 1.5)
    mb._param_fields["k"].lo.setText("2.0")
    assert "minimum must be below the maximum" in _refused("param_min")

    # (c) a display scale at zero
    mb._param_fields["k"].set_spec(1.0, 0.5, 1.5)
    mb.x_scale.setText("0")
    assert "must be greater than 0" in _refused("x_scale")
    mb.x_scale.setText("10.0")

    # (d) a blank forcing parameter, which used to reach model_store as a 0.0 nobody typed
    row = mb._var_rows[0]
    row.force_kind.setCurrentIndex(1)                      # "Sinusoidal"
    assert set(row.forcing_fields()) == set(row._force_fields["sin"])
    row.forcing_fields()["freq"].setText("")
    assert "'freq'" in _refused("forcing_param")

    # the screen names every one of its own boxes, and on the right surface
    for key in ("param_value", "param_min", "param_max", "init", "x_scale", "t_scale",
                "forcing_param"):
        assert gui_fields.fix_sentence(key).endswith("on the Model Builder screen."), key
```

- [ ] **Step 2: Run it and watch it fail**

Run: `pytest tests/test_user_models.py::test_the_builder_shows_a_field_refusal_in_the_yellow_box -v`
Expected: FAIL with `IndexError: list index out of range` at `box = SHOWN[-1]` — `_validate_clicked`
shows no box at all today, it only writes the status line.

- [ ] **Step 3: Register the seven keys**

In `core/refusals.py`, find:

```python
    # the artifact browser (piece 4): the artifact a browse action acts on, and its note
    Field("artifact", "the artifact", None),
    Field("note", "the note", None),
```

Replace with:

```python
    # the artifact browser (piece 4): the artifact a browse action acts on, and its note
    Field("artifact", "the artifact", None),
    Field("note", "the note", None),
    # the model builder (piece 5, §5.3). Window-only settings: core/tool/fields.py maps each to None.
    Field("param_value", "the parameter's value", None),
    Field("param_min", "the parameter's lower bound", None),
    Field("param_max", "the parameter's upper bound", None),
    Field("init", "the initial condition", None),
    Field("x_scale", "the length scale, in nm per non-dimensional unit", None),
    Field("t_scale", "the time scale, in seconds per non-dimensional time unit", None),
    Field("forcing_param", "the forcing parameter", None),
```

- [ ] **Step 4: Name the control for each, and the flag that does not exist**

In `core/gui/fields.py`, find:

```python
    "artifact": "Select an artifact in the list on the Artifacts screen.",
    "note": "Edit it in the Note box on the Artifacts screen.",
```

Replace with:

```python
    "artifact": "Select an artifact in the list on the Artifacts screen.",
    "note": "Edit it in the Note box on the Artifacts screen.",
    # the model builder (piece 5, §5.3). A SCREEN, not an inference tab -- fix_sentence renders
    # "on the Model Builder screen" for a place in SCREENS. The parameter rows repeat per parameter,
    # so the label is the row's own ("value" / "min" / "max") and the message names which parameter.
    "param_value": ("Model Builder", "value"),
    "param_min": ("Model Builder", "min"),
    "param_max": ("Model Builder", "max"),
    "init": ("Model Builder", "init"),
    "x_scale": ("Model Builder", "x_scale (nm)"),
    "t_scale": ("Model Builder", "t_scale (s)"),
    # a sentence, not a tuple: the forcing rows' labels are the forcing parameter NAMES (amp, freq,
    # tau, ...), so there is no one label(key) to build a row from.
    "forcing_param": "Set it in the forcing parameter's own box on the Model Builder screen.",
```

In `core/tool/fields.py`, find:

```python
    "artifact": None,
    "note": "--note",                                   # add_name_flags defines it beside --name
```

Replace with:

```python
    "artifact": None,
    "note": "--note",                                   # add_name_flags defines it beside --name
    # the model builder (piece 5): a window-only screen, so no option string answers any of these
    "param_value": None, "param_min": None, "param_max": None, "init": None,
    "x_scale": None, "t_scale": None, "forcing_param": None,
```

- [ ] **Step 5: Expose a variable row's live forcing fields**

In `core/gui/screens/model_builder_screen.py`, find:

```python
    def values(self) -> dict:
        kind = _FORCE_KINDS[self.force_kind.currentIndex()][0]
```

and insert ABOVE it:

```python
    def forcing_fields(self) -> dict:
        """``{param name: FloatField}`` for the forcing kind currently selected, ``{}`` for "None".

        ``values()`` reads these through ``FloatField.value()``, which returns 0.0 for a blank box --
        so a cleared "freq" used to be saved as a real frequency of zero nobody typed. The screen
        checks them through this accessor before assembling the document (spec §5.3).
        """
        kind = _FORCE_KINDS[self.force_kind.currentIndex()][0]
        return dict(self._force_fields[kind]) if kind else {}
```

- [ ] **Step 6: Build the two display-scale rows from the table**

In the same file, find:

```python
        add_help_row(sform, "x_scale (nm)", self.x_scale, HELP["x_scale"])
        add_help_row(sform, "t_scale (s)", self.t_scale, HELP["t_scale"])
```

Replace with:

```python
        add_help_row(sform, gui_fields.label("x_scale"), self.x_scale, HELP["x_scale"])
        add_help_row(sform, gui_fields.label("t_scale"), self.t_scale, HELP["t_scale"])
```

and find:

```python
from ..design import SPACE
from ..panels.base_panel import BasePanel
```

Replace with:

```python
from .. import fields as gui_fields
from ..design import SPACE
from ..panels.base_panel import BasePanel
from ..widgets.refusal_box import show_refusal
```

and find:

```python
import torch
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
```

Replace with:

```python
import math

import torch
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
```

and find:

```python
from core.Models.user_model import ModelParseError, UserModel, parse_user_model
```

Replace with:

```python
from core.Models.user_model import ModelParseError, UserModel, parse_user_model
from core.refusals import Refusal, refuse, require_positive
```

- [ ] **Step 7: Give the screen its own `_refusal`**

In the same file, find:

```python
    def _set_status(self, text: str, error: bool = False) -> None:
        self.status.setText(("⚠ " if error else "") + text)
```

Replace with:

```python
    def _refusal(self, exc: Refusal) -> None:
        """Show a refusal the way every other surface does: the yellow "Check your inputs" box with
        the core's neutral sentence and this front end's "where to fix it" under it.

        Through ``refusal_box.show_refusal`` rather than ``BasePanel._refusal`` because this screen
        is a plain QWidget (spec §1.2): a BasePanel enrols in ``BasePanel._instances``, and a run in
        any panel would then grey out the builder's own controls. It has no log pane either, so the
        status line is this screen's record of the sentence -- the same role the pane plays for a
        panel and the status line plays for the Artifacts screen.
        """
        self._set_status(f"{exc.message} {gui_fields.fix_sentence(exc.field)}".rstrip(), error=True)
        show_refusal(self, exc)

    def _set_status(self, text: str, error: bool = False) -> None:
        self.status.setText(("⚠ " if error else "") + text)
```

- [ ] **Step 8: Refuse the display scales, the initial conditions and the forcing fields**

In `_validate`, find:

```python
        doc = self._assemble_doc()
        doc["name"] = name
```

Replace with:

```python
        # Read the numeric boxes through value_or_none BEFORE assembling the document: value()
        # returns 0.0 for a blank, so a check written against the assembled doc would judge a blank
        # box as a typed zero (Review Focus 3).
        require_positive("x_scale", self.x_scale.value_or_none())
        require_positive("t_scale", self.t_scale.value_or_none())
        for row in self._var_rows:
            v = row.init.value_or_none()
            if v is None or not math.isfinite(v):
                refuse("init", f"The initial condition for '{row.var_name}' is blank or not a "
                               f"finite number.")
            for pname, fld in row.forcing_fields().items():
                fv = fld.value_or_none()
                if fv is None or not math.isfinite(fv):
                    refuse("forcing_param", f"The forcing parameter '{pname}' of '{row.var_name}' "
                                            f"is blank or not a finite number.")
        doc = self._assemble_doc()
        doc["name"] = name
```

Then find and DELETE:

```python
        t_scale = doc["rescale"]["t_scale"]
        if doc["rescale"]["x_scale"] <= 0 or t_scale <= 0:
            self._set_status("x_scale and t_scale must be > 0.", error=True)
            return None
```

replacing it with:

```python
        t_scale = doc["rescale"]["t_scale"]
```

- [ ] **Step 9: Refuse the parameter rows**

In `_validate`, find:

```python
        for p, e in doc["params"].items():         # UX pre-check; model_store._check_schema re-enforces
            # None means the field did not parse -- a blank box, or "-" mid-typing. Caught here so it
            # is a message about THAT box rather than model_store rejecting a 0.0 the user never typed.
            if e["lo"] is None or e["hi"] is None:
                which = "min" if e["lo"] is None else "max"
                self._set_status(f"Parameter '{p}': {which} is blank or not a number.", error=True)
                return None
            if not e["lo"] < e["hi"]:
                self._set_status(f"Parameter '{p}': min must be < max.", error=True)
                return None
            if not e["lo"] <= e["value"] <= e["hi"]:
                self._set_status(f"Parameter '{p}': value {e['value']} is outside its bounds "
                                 f"[{e['lo']}, {e['hi']}].", error=True)
                return None
            # A log box needs a positive lower bound. reparam._log_mask would otherwise downgrade it
            # to linear with a warnings.warn that never reaches the GUI, so the model would train in a
            # coordinate the user did not pick, silently. model_store._check_schema re-enforces this.
            if e["box"] == "log" and e["lo"] <= 0:
                self._set_status(f"Parameter '{p}': a log box needs min > 0 (got {e['lo']}). "
                                 f"Raise min, or set its box to 'linear'.", error=True)
                return None
```

Replace with:

```python
        for p, e in doc["params"].items():         # UX pre-check; model_store._check_schema re-enforces
            # None means the field did not parse -- a blank box, or "-" mid-typing. Refused here so
            # it is a message about THAT parameter rather than model_store rejecting a 0.0 the user
            # never typed. The parameter's NAME is in every sentence because these rows repeat: one
            # key names the kind of field, the sentence names which row (spec §5.3).
            for key, which, v in (("param_value", "value", e["value"]),
                                  ("param_min", "minimum", e["lo"]),
                                  ("param_max", "maximum", e["hi"])):
                if v is None or not math.isfinite(v):
                    refuse(key, f"Parameter '{p}': the {which} is blank or not a finite number.")
            if not e["lo"] < e["hi"]:
                refuse("param_min", f"Parameter '{p}': the minimum must be below the maximum "
                                    f"(got {e['lo']:g} and {e['hi']:g}).")
            if not e["lo"] <= e["value"] <= e["hi"]:
                refuse("param_value", f"Parameter '{p}': the value {e['value']:g} is outside its "
                                      f"bounds ({e['lo']:g}, {e['hi']:g}).")
            # A log coordinate needs a positive lower bound. reparam._log_mask would otherwise
            # downgrade it to linear with a warnings.warn that never reaches the GUI, so the model
            # would train in a coordinate the user did not pick, silently. model_store._check_schema
            # re-enforces this.
            if e["box"] == "log" and e["lo"] <= 0:
                refuse("param_min", f"Parameter '{p}': a log coordinate needs a minimum above 0 "
                                    f"(got {e['lo']:g}); raise the minimum, or set this parameter "
                                    f"back to 'linear'.")
```

- [ ] **Step 10: Catch the refusal at the two click entries**

In the same file, find:

```python
    def _validate_clicked(self):
        doc = self._validate()
        if doc is not None:
```

Replace with:

```python
    def _validate_clicked(self):
        try:
            doc = self._validate()
        except Refusal as e:
            self._refusal(e)
            return
        if doc is not None:
```

Then find:

```python
        doc = self._validate()
        if doc is None:
            return
        json_path = config.MODELS_PATH / f"{doc['name']}.json"
```

Replace with:

```python
        try:
            doc = self._validate()
        except Refusal as e:
            self._refusal(e)
            return
        if doc is None:
            return
        json_path = config.MODELS_PATH / f"{doc['name']}.json"
```

- [ ] **Step 11: Update the three tests that read the old status sentences**

In `tests/test_user_models.py`, find:

```python
    mb._param_fields["k"].set_spec(5.0, 0.0, 1.0)                  # value outside its box
    assert mb._validate() is None and "outside its bounds" in mb.status.text()
```

Replace with:

```python
    mb._param_fields["k"].set_spec(5.0, 0.0, 1.0)                  # value outside its box
    # A field problem is a Refusal since piece 5 (§5.1); the click handler is what shows it.
    with pytest.raises(Refusal) as ei:
        mb._validate()
    assert ei.value.field == "param_value" and "outside its bounds" in ei.value.message
```

and add `from core.refusals import Refusal` to that test's imports (find
`from core.gui.screens.model_builder_screen import ModelBuilderScreen, _ParamRow` and add the line
beneath it).

Find:

```python
    assert row.spec()[1] is None, row.spec()                       # not 0.0
    assert mb._validate() is None
    assert "min is blank" in mb.status.text(), mb.status.text()
```

Replace with:

```python
    assert row.spec()[1] is None, row.spec()                       # not 0.0
    with pytest.raises(Refusal) as ei:
        mb._validate()
    assert ei.value.field == "param_min", ei.value.field
    assert "the minimum is blank" in ei.value.message, ei.value.message
```

and add `from core.refusals import Refusal` beneath that test's
`from core.gui.screens.model_builder_screen import ModelBuilderScreen`.

Find:

```python
    mb._param_fields["k"].set_spec(1.0, -1.0, 2.0, "log")
    assert mb._validate() is None
    assert "log box needs min > 0" in mb.status.text(), mb.status.text()
```

Replace with:

```python
    mb._param_fields["k"].set_spec(1.0, -1.0, 2.0, "log")
    with pytest.raises(Refusal) as ei:
        mb._validate()
    assert ei.value.field == "param_min", ei.value.field
    assert "log coordinate needs a minimum above 0" in ei.value.message, ei.value.message
```

and add `from core.refusals import Refusal` beneath that test's
`from core.gui.screens.model_builder_screen import ModelBuilderScreen`.

- [ ] **Step 12: Widen the two closed key sets in the refusals suite**

In `tests/test_refusals.py`, find:

```python
    "artifact", "note",
)
```

Replace with:

```python
    "artifact", "note",
    # piece 5, the model builder (spec §5.3)
    "param_value", "param_min", "param_max", "init", "x_scale", "t_scale", "forcing_param",
)
```

Then find:

```python
    assert len(FIELDS) == len(BASE_KEYS) + len(TOOL_ONLY_KEYS) == 63, "a key is listed twice above"
```

and raise the literal by seven plus whatever T8 and T9 already added — re-derive it by running
`pytest tests/test_refusals.py -k registry_holds -v` once and reading the count off the failure,
then write that number in place of `63`.

Then find:

```python
    assert {k for k, f in tool_fields.FLAG.items() if f is None} == {
        "units", "chi_k_pad", "chi_max_cycles", "chi_f0", "chi_freq_bounds", "artifact"}
```

Replace with:

```python
    assert {"units", "chi_k_pad", "chi_max_cycles", "chi_f0", "chi_freq_bounds", "artifact"} <= \
        {k for k, f in tool_fields.FLAG.items() if f is None}, \
        "a key the tool never exposes lost its None entry"
```

(if T8 has already replaced this equality with a subset assertion, the quoted text will not be
found — say so in the task report and leave it alone.)

- [ ] **Step 13: Run the tests and watch them pass**

Run: `pytest tests/test_user_models.py tests/test_refusals.py tests/test_nav_and_gating.py -v`
Expected: PASS

- [ ] **Step 14: Commit**

```bash
git add core/refusals.py core/gui/fields.py core/tool/fields.py core/gui/screens/model_builder_screen.py tests/test_user_models.py tests/test_refusals.py
git commit -m "model builder: field refusals in the yellow box, with keys for its own boxes"
```

#### Amendments (binding — these supersede the text above)

Rulings that change this task: **P36**, **P20** (with **P75**), and the Interface Contract's FIELDS order. Confirmed as written: T9's `SCREENS`/`_where` (consumed exactly as Step 4 and the test expect), and the instance-dialog rule (`show_refusal`).

**A. P36 — the forcing key is `forcing_value`, and its window entry names the forcing rows.** Replace the key `forcing_param` with `forcing_value` everywhere in this task:
- **Step 1 test:** `assert "'freq'" in _refused("forcing_param")` becomes `assert "'freq'" in _refused("forcing_value")`. In the final loop's tuple, `"forcing_param"` becomes `"forcing_value"`.
- **Step 3:** the last Field line is `Field("forcing_value", "the forcing parameter", None),`.
- **Step 4, core/gui/fields.py:** the last entry becomes:
```python
    # ONE key for every forcing field (P36): the forcing rows' labels are the parameter NAMES (amp,
    # freq, tau, ...), built per kind, so there is no one label(key) to build a row from.
    "forcing_value": ("Set it in the forcing parameter's own box, beneath the variable's 'forcing' "
                      "choice, on the Model Builder screen."),
```
- **Step 4, core/tool/fields.py:** the line is `"x_scale": None, "t_scale": None, "forcing_value": None,`.
- **Step 8:** `refuse("forcing_param", ...)` becomes `refuse("forcing_value", ...)`, with the same sentence.
- **Step 12:** the BASE_KEYS line is `"param_value", "param_min", "param_max", "init", "x_scale", "t_scale", "forcing_value",`.
- **Interfaces:** Produces `FIELDS` keys `param_value`, `param_min`, `param_max`, `init`, `x_scale`, `t_scale`, `forcing_value`.

**B. Contract order — Step 3's insertion point.** The contract fixes the FIELDS order as T8's keys first, then the Simulate/Reduction/model-builder keys. Do NOT insert after `Field("note", "the note", None),`: T8 has put its block there. Instead, find
```python
    # tool-only (the diagnostics); no window control, CONTROL[key] is None in core/gui/fields.py
```
and insert the whole block immediately ABOVE it. The block is the comment line `    # the model builder (piece 5, §5.3). Window-only settings: core/tool/fields.py maps each to None.` plus the seven Field lines, with `forcing_value` as in A.

**C. P20 / P75 — Step 12's three anchors, as T8 left them.**
- **BASE_KEYS:** `"artifact", "note",` is no longer followed by `)`. T8 appended its keys (and, under P75, the five non-knob keys) after it. Find the closing `)` of the `BASE_KEYS = (` tuple (the line immediately before `TOOL_ONLY_KEYS = (`) and insert immediately ABOVE it:
```python
    # piece 5, the model builder (spec §5.3)
    "param_value", "param_min", "param_max", "init", "x_scale", "t_scale", "forcing_value",
```
- **The count:** find the line beginning `    assert len(FIELDS) == len(BASE_KEYS) + len(TOOL_ONLY_KEYS) == ` (T8 changed the number). Run `pytest tests/test_refusals.py -k registry_holds -v` once and write the number the failure reports.
- **The None-flag equality:** T8 Step 8 has already replaced it with a subset assertion. The quoted text will not be found. Leave it alone and say so in the report.

Everything else in Steps 1-14 stands as written. Also read the finding on `param_value`: if the owner rules option (a), Step 9's tuple reads `("param_value", "value", self._param_fields[p].value.value_or_none())` and Step 1 gains the blank-value case.

#### Controller rulings (pre-flight, binding — these supersede the amendments above and the task text)

- **F16** — `param_value` is read through `self._param_fields[p].value.value_or_none()`, not `e["value"]`, so a cleared box is refused as blank instead of read as 0.0. Add the cleared-box case to the test.

---

### Task 23: the Simulate panel's two silent clamps and its unchecked observation length

**Why:** Spec §5.6 — the Simulate panel dispatches `max(1, frame_steps.value())` and
`float(max(1, fps.value()))`, silent clamps that turn a blank box (which reads as `0`) into a value
nobody typed, and passes its observation length unchecked. The clamps become refusals naming the
box, and the observation length is checked. Review Focus 3: each floor is asserted against a BLANK
box, not only a typed zero.

**Files:**
- Modify: `core/refusals.py` (two keys: `frame_steps`, `fps`)
- Modify: `core/gui/fields.py` (two entries, and `t_obs` widened to both places it appears)
- Modify: `core/tool/fields.py` (two `None` entries)
- Modify: `core/gui/panels/simulate_panel.py:86-90` (the three `add_help_row` calls),
  `:99-118` (`_start`), `:143-145` (`fps = float(max(1, self.fps.value()))`)
- Test: `tests/test_simulate.py` (new), `tests/test_nav_and_gating.py:1509` (the `t_obs` sentence),
  `tests/test_refusals.py:29-39,91,805`

**Interfaces:**
- Consumes: T21's `except Refusal` / `except Exception` pair in `SimulatePanel._start`;
  `core.refusals.require_at_least` / `require_positive`; T9's widened place shape, which lets a
  tuple entry name a tab title outside Parameter Inference ("Live simulation").
- Produces: `FIELDS` keys `frame_steps` and `fps`; `CONTROL["t_obs"] == (("Infer", "Live
  simulation"), "T_obs (s)")` — T25/T26/T27 must not narrow it back;
  `run_simulation_stream` is dispatched with checked values, never clamped ones.

- [ ] **Step 1: Write the failing test**

In `tests/test_simulate.py`, add after
`test_saving_mp4_without_ffmpeg_is_a_refusal_not_a_config_error`:

```python
def test_the_simulate_panel_refuses_its_blank_boxes_instead_of_clamping_them(monkeypatch, tmp_path):
    """Spec §5.6. The panel used to dispatch ``max(1, frame_steps.value())`` and
    ``float(max(1, fps.value()))`` and hand ``tobs.value()`` straight through. Every numeric box in
    this window returns 0 for a BLANK (labeled_inputs.FloatField.value), so those clamps quietly
    turned "I cleared this box" into 1 step per frame and 1 frame per second -- a run that looks
    wedged rather than refused -- and a blank T_obs reached plan_stream, where ``n_obs`` came out 0
    and ``run_simulation_stream`` returned at once, reporting success with nothing streamed.

    Each is now a refusal naming the setting, raised at the CLICK and before the config is built, so
    nothing is dispatched. Asserted against a blank box and not only a typed zero, because a rule
    written as "reject below zero" would accept every blank box in the application (Review Focus 3).

    ``t_obs`` is the shared key the Infer tab already owns, so its fix sentence now names both places
    it appears -- E6's widening. Without that a blank length on this screen sent the operator to a
    tab that has nothing to do with it.
    """
    from PySide6.QtWidgets import QMessageBox
    from core.gui import fields as gui_fields
    from core.gui.panels import simulate_panel as sim_mod
    from core.gui.panels.simulate_panel import SimulatePanel
    from tests._fixtures import SHOWN, qt_app

    qt_app()
    cell = tmp_path / "cell.txt"
    cell.write_text("# values\n", encoding="utf-8")
    p = SimulatePanel()
    p.cell_picker.selected_path = lambda: str(cell)
    sent = []
    p.dispatch = lambda *a, **k: sent.append((a, k))
    built = []
    monkeypatch.setattr(sim_mod, "build_stream_config",
                        lambda *a, **k: built.append(a) or pytest.fail(
                            "the config was built before the boxes were checked"))

    for box_widget, key, blank_needle in ((p.tobs, "t_obs", "is blank"),
                                          (p.frame_steps, "frame_steps", "is blank"),
                                          (p.fps, "fps", "is blank")):
        original = box_widget.text()
        box_widget.setText("")
        SHOWN.clear()
        p._start()
        shown = SHOWN[-1]
        assert shown.windowTitle() == "Check your inputs" and shown.icon() == QMessageBox.Warning, key
        assert blank_needle in shown.text(), shown.text()
        assert shown.informativeText() == gui_fields.fix_sentence(key), key
        assert sent == [] and built == [], key
        box_widget.setText(original)

    # a typed zero is refused too, and by the rule the setting deserves
    for box_widget, key, needle in ((p.tobs, "t_obs", "must be greater than 0"),
                                    (p.frame_steps, "frame_steps", "must be at least 1"),
                                    (p.fps, "fps", "must be greater than 0")):
        original = box_widget.text()
        box_widget.setText("0")
        SHOWN.clear()
        p._start()
        assert needle in SHOWN[-1].text(), (key, SHOWN[-1].text())
        assert sent == [] and built == [], key
        box_widget.setText(original)

    # the export path reads the same box and must refuse it the same way
    import numpy as np
    from PySide6.QtWidgets import QFileDialog
    p._record = [np.array([[0.0, 0.1], [1e-3, 0.2]])]
    monkeypatch.setattr(sim_mod, "ffmpeg_available", lambda: True)
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: ("clip.gif", ""))
    p.fps.setText("")
    SHOWN.clear()
    p._save_video()
    assert SHOWN[-1].windowTitle() == "Check your inputs"
    assert sent == [], "an export was dispatched with a blank frame rate"

    # the two new boxes are named on this screen, and T_obs names BOTH places it appears
    assert gui_fields.fix_sentence("frame_steps") == \
        "Set it in the 'Steps / frame' box on the Live simulation tab."
    assert gui_fields.fix_sentence("fps") == \
        "Set it in the 'Max FPS' box on the Live simulation tab."
    assert gui_fields.fix_sentence("t_obs") == \
        "Set it in the 'T_obs (s)' box on the Infer or Live simulation tab."
```

- [ ] **Step 2: Run it and watch it fail**

Run: `pytest tests/test_simulate.py::test_the_simulate_panel_refuses_its_blank_boxes_instead_of_clamping_them -v`
Expected: FAIL at the first loop with `Failed: the config was built before the boxes were checked`
— today `_start` calls `build_stream_config` before looking at any number.

- [ ] **Step 3: Register the two keys**

In `core/refusals.py`, find:

```python
    Field("forcing_param", "the forcing parameter", None),
```

Replace with:

```python
    Field("forcing_param", "the forcing parameter", None),
    # the live simulation (piece 5, §5.6). Both were SILENT CLAMPS -- max(1, ...) -- before.
    Field("frame_steps", "the number of simulation steps per displayed frame", "2000"),
    Field("fps", "the maximum render frame rate, in frames per second", "30"),
```

(the two defaults are the panel's own construction defaults, `IntField(2000)` and `IntField(30)`.)

If Task 22 has not landed, insert the two `Field(...)` lines after `Field("note", "the note", None),`
instead.

- [ ] **Step 4: Name the boxes, and widen `t_obs` to both places**

In `core/gui/fields.py`, find:

```python
    "t_obs": ("Infer", "T_obs (s)"),
```

Replace with:

```python
    # The observation length is asked for on two surfaces with the same label: the Infer tab, and the
    # Live simulation panel (piece 5, §5.6). E6 -- an input that appears in several places lists them
    # all, or a refusal raised on one sends the operator to the other.
    "t_obs": (("Infer", "Live simulation"), "T_obs (s)"),
```

Then find:

```python
    "forcing_param": "Set it in the forcing parameter's own box on the Model Builder screen.",
```

Replace with:

```python
    "forcing_param": "Set it in the forcing parameter's own box on the Model Builder screen.",
    # the live simulation (piece 5, §5.6)
    "frame_steps": ("Live simulation", "Steps / frame"),
    "fps": ("Live simulation", "Max FPS"),
```

In `core/tool/fields.py`, find:

```python
    "x_scale": None, "t_scale": None, "forcing_param": None,
```

Replace with:

```python
    "x_scale": None, "t_scale": None, "forcing_param": None,
    # the live simulation (piece 5): a window-only panel; the tool has no streaming subcommand
    "frame_steps": None, "fps": None,
```

- [ ] **Step 5: Check the three boxes at the click, before the builder**

In `core/gui/panels/simulate_panel.py`, find:

```python
from core.config import CELL_PATH, DT_EXP_S, T_MIN_EXP_S, VALID_MODELS
from core.refusals import Refusal
```

Replace with:

```python
from core.config import CELL_PATH, DT_EXP_S, T_MIN_EXP_S, VALID_MODELS
from core.refusals import Refusal, require_at_least, require_positive
```

Then find:

```python
        model = self.model_combo.currentText()
        try:
            cfg = build_stream_config(model, cell)
        except Refusal as e:                         # a cell or model problem: the yellow box
            self._refusal(e)
            return
        except Exception as e:                       # noqa: BLE001 -- a bug in the builder: the red box
            self._on_error(e, traceback.format_exc())
            return
```

Replace with:

```python
        model = self.model_combo.currentText()
        try:
            # The boxes FIRST, and through value_or_none: every numeric box here returns 0 for a
            # blank, so a blank T_obs used to reach plan_stream (n_obs 0, an instant "complete") and
            # the two below were clamped with max(1, ...) to a value nobody typed (spec §5.6).
            t_obs = require_positive("t_obs", self.tobs.value_or_none())
            frame_steps = require_at_least("frame_steps", self.frame_steps.value_or_none(), 1)
            fps = require_positive("fps", self.fps.value_or_none())
            cfg = build_stream_config(model, cell)
        except Refusal as e:                         # a box, a cell or a model problem: the yellow box
            self._refusal(e)
            return
        except Exception as e:                       # noqa: BLE001 -- a bug in the builder: the red box
            self._on_error(e, traceback.format_exc())
            return
```

Then find:

```python
        self.log_pane.append_line(
            f"Streaming {model} — {Path(cell).name} for {self.tobs.value():g} s of observation…")
        self.dispatch(run_simulation_stream, cfg, self.tobs.value(),
                      max(1, self.frame_steps.value()), float(max(1, self.fps.value())),
                      provide_stream=True, on_chunk=self._on_chunk,
                      on_result=lambda _r: self.log_pane.append_line("Simulation complete."))
```

Replace with:

```python
        self.log_pane.append_line(
            f"Streaming {model} — {Path(cell).name} for {t_obs:g} s of observation…")
        self.dispatch(run_simulation_stream, cfg, t_obs, frame_steps, float(fps),
                      provide_stream=True, on_chunk=self._on_chunk,
                      on_result=lambda _r: self.log_pane.append_line("Simulation complete."))
```

- [ ] **Step 6: The export path reads the same box the same way**

In the same file, find:

```python
        series = np.concatenate(self._record, axis=0)
        fps = float(max(1, self.fps.value()))
```

Replace with:

```python
        series = np.concatenate(self._record, axis=0)
        try:
            fps = require_positive("fps", self.fps.value_or_none())      # the same box, the same rule
        except Refusal as e:
            self._refusal(e)
            return
```

(it sits AFTER the ffmpeg refusal, so the missing-codec message still comes first for an .mp4 —
which `test_saving_mp4_without_ffmpeg_is_a_refusal_not_a_config_error` pins.)

- [ ] **Step 7: Build the three rows from the table**

In the same file, find:

```python
        add_help_row(form, "T_obs (s)", self.tobs, HELP["tobs"])
        add_help_row(form, "Steps / frame", self.frame_steps, HELP["frame"])
        add_help_row(form, "Max FPS", self.fps, HELP["fps"])
```

Replace with:

```python
        add_help_row(form, gui_fields.label("t_obs"), self.tobs, HELP["tobs"])
        add_help_row(form, gui_fields.label("frame_steps"), self.frame_steps, HELP["frame"])
        add_help_row(form, gui_fields.label("fps"), self.fps, HELP["fps"])
```

and find:

```python
from .base_panel import BasePanel
from .simulate_export import (estimate_frame_count, export_animation, export_stride, ffmpeg_available)
```

Replace with:

```python
from .base_panel import BasePanel
from .simulate_export import (estimate_frame_count, export_animation, export_stride, ffmpeg_available)
from .. import fields as gui_fields
```

- [ ] **Step 8: Update the verbatim `t_obs` sentence in the table's own test**

In `tests/test_nav_and_gating.py`, find:

```python
    assert gui_fields.fix_sentence("t_obs") == "Set it in the 'T_obs (s)' box on the Infer tab."
```

Replace with:

```python
    # two surfaces ask for the observation length under the same label (piece 5, §5.6)
    assert gui_fields.fix_sentence("t_obs") == \
        "Set it in the 'T_obs (s)' box on the Infer or Live simulation tab."
```

- [ ] **Step 9: Widen the two closed key sets in the refusals suite**

In `tests/test_refusals.py`, find:

```python
    "param_value", "param_min", "param_max", "init", "x_scale", "t_scale", "forcing_param",
)
```

Replace with:

```python
    "param_value", "param_min", "param_max", "init", "x_scale", "t_scale", "forcing_param",
    # piece 5, the live simulation (spec §5.6)
    "frame_steps", "fps",
)
```

Then raise the `len(FIELDS) == ... == <N>` literal by two (run
`pytest tests/test_refusals.py -k registry_holds -v` and read the real count off the failure), and
confirm the None-flag assertion Task 22 turned into a subset still passes without further edits.

- [ ] **Step 10: Run the tests and watch them pass**

Run: `pytest tests/test_simulate.py tests/test_refusals.py tests/test_nav_and_gating.py tests/test_settings_persistence.py -v`
Expected: PASS

- [ ] **Step 11: Commit**

```bash
git add core/refusals.py core/gui/fields.py core/tool/fields.py core/gui/panels/simulate_panel.py tests/test_simulate.py tests/test_refusals.py tests/test_nav_and_gating.py
git commit -m "simulate: the two frame clamps and the observation length become refusals"
```

#### Amendments (binding — these supersede the text above)

Rulings that change this task: **P33**, **P36**, **P20**, and the Interface Contract's FIELDS order. Confirmed as written: T21's except pair (Step 5 quotes it verbatim), `IntField.value_or_none()` (it already exists), and Review Focus 3 (the blank-box loop).

**A. P33 — `t_obs` is widened by T9, not here.** Drop Step 4's first find/replace (the `"t_obs": ("Infer", "T_obs (s)"),` edit) and all of Step 8. Instead VERIFY:
- `core/gui/fields.py` already reads `"t_obs": (("Infer", "Live simulation"), "T_obs (s)"),`;
- `tests/test_nav_and_gating.py`'s verbatim pin already reads `"Set it in the 'T_obs (s)' box on the Infer or Live simulation tab."`.

If both hold, change nothing there and say so in the report. If T9 left the old line (a T9 defect against P33), apply Step 4's first edit and Step 8 exactly as written above, and report it. The Interfaces line "Produces: `CONTROL["t_obs"] == ...`" is T9's product. T23 still builds its row from `gui_fields.label("t_obs")` (Step 7).

**B. P36 — re-anchor on `forcing_value`** (T22 produces it).
- **Step 4, core/gui/fields.py:** find T22's two-line entry
```python
    "forcing_value": ("Set it in the forcing parameter's own box, beneath the variable's 'forcing' "
                      "choice, on the Model Builder screen."),
```
  and insert immediately BELOW it:
```python
    # the live simulation (piece 5, §5.6)
    "frame_steps": ("Live simulation", "Steps / frame"),
    "fps": ("Live simulation", "Max FPS"),
```
- **Step 4, core/tool/fields.py:** find `    "x_scale": None, "t_scale": None, "forcing_value": None,` and insert the two lines (comment + `"frame_steps": None, "fps": None,`) below it.
- **Step 9:** find
```python
    "param_value", "param_min", "param_max", "init", "x_scale", "t_scale", "forcing_value",
)
```
  and replace it with the same line followed by `    # piece 5, the live simulation (spec §5.6)` / `    "frame_steps", "fps",` / `)`.
- **Count:** read it off the failing `-k registry_holds` run, as written.
- **None-flag subset:** T8 (not T22) made it; confirm it passes and do not edit it.

**C. Contract order — Step 3's insertion point.** The Simulate keys come BEFORE the model-builder keys. Find T22's line
```python
    # the model builder (piece 5, §5.3). Window-only settings: core/tool/fields.py maps each to None.
```
and insert immediately ABOVE it:
```python
    # the live simulation (piece 5, §5.6). Both were SILENT CLAMPS -- max(1, ...) -- before.
    Field("frame_steps", "the number of simulation steps per displayed frame", "2000"),
    Field("fps", "the maximum render frame rate, in frames per second", "30"),
```
Delete the "If Task 22 has not landed" fallback: T22 precedes this task.

**D. Correction: the test needs `pytest`.** `tests/test_simulate.py` has no `import pytest`. In Step 1's test, add `import pytest` as the first line of its local imports, above `from PySide6.QtWidgets import QMessageBox`. Without it, the `build_stream_config` stub raises NameError. T21's `except Exception` routes that to the red "Error" box, and Step 2 fails on the window title instead of the predicted `Failed: the config was built before the boxes were checked`. With the import, Step 2's prediction holds.

**E. The registry-default closure test (see the BLOCKING finding).** `tests/test_refusals.py::test_every_registry_default_is_the_trees_own_default` fails on the two new defaults (`rest` must be None except `t_obs`). Apply the owner's ruling on that finding before committing. Add `tests/test_refusals.py::test_every_registry_default_is_the_trees_own_default` to Step 10's run. The gate must not be left red.

Steps 5, 6, 7, 10 and 11 otherwise stand. `tests/test_nav_and_gating.py` stays in the `git add` line: harmless if A leaves it unmodified.

#### Controller rulings (pre-flight, binding — these supersede the amendments above and the task text)

- **F3** — `frame_steps` and `fps` carry defaults, so `test_every_registry_default_is_the_trees_own_default` in `tests/test_refusals.py` must pin them against the object that owns them: just above its `looked_at = ...` line, build a `SimulatePanel` (after `qt_app()`), assert each key's `FIELDS[key].default` equals the panel's box text, and add those keys to `looked_at`. Add the test to your run command and the file to your Files block.

---

### Task 24: `StorePicker` takes a row predicate and filters on `finished`

**Why:** Spec §5.4 and §2.4 item 19. The FDT and CrossVal screens each gain a `StorePicker` over the
`fdt` kind filtered to their own study, and `refresh` has no filter hook. More seriously, the picker
filters on `Summary.complete`, which means "has a valid manifest" and NOT "the run finished"
(`store.py:83-88`); an `fdt` record is progressive and carries a manifest from its first moment, so
without this change the pickers would silently offer half-written records as results.

**Files:**
- Modify: `core/gui/widgets/artifact_picker.py:139-200` (`class StorePicker`, `__init__`, `refresh`)
- Test: `tests/test_nav_and_gating.py` (new, beside the picker's own test), and the two row stubs at
  `tests/test_nav_and_gating.py:3167-3169` and `tests/test_settings_persistence.py:1066-1068`

**Interfaces:**
- Consumes: `Summary.finished` (piece 4, B3; T2 leaves it a plain `bool` field) and `Summary.study`,
  which T2 adds and fills.
- Produces: `StorePicker(kind, allow_new=False, store=None, parent=None, row_filter=None)`, where
  `row_filter` is `Callable[[Summary], bool] | None` — T25 and T26 pass
  `row_filter=lambda s: s.study == "single"` and `... == "sweep"`; `StorePicker` offers FINISHED rows
  only, for every kind.

- [ ] **Step 1: Write the failing test**

In `tests/test_nav_and_gating.py`, add after the picker test that ends with
`assert pp.post_line.text() == "chi · width 18 · amortized · 2026-09-14T10:22:31", \`:

```python
def test_the_store_picker_offers_finished_rows_only_and_honours_a_row_filter(monkeypatch):
    """Spec §5.4, checklist item 19. ``StorePicker`` skipped rows whose ``complete`` was false, and
    ``complete`` means "has a valid manifest" -- NOT "the run finished" (store.Summary). For the six
    ordinary kinds the two coincide, because ``ArtifactWriter._commit`` writes the manifest last. For
    the two kinds written PROGRESSIVELY -- the training cache, and piece 5's ``fdt`` -- they do not:
    a record carries a manifest from its first moment, so the picker would have offered a run that
    is still going, or one a cancel left half written, as if it were a result. It filters on
    ``finished`` instead, which is the same answer for every kind that is not progressive.

    And ``refresh`` listed every row of the kind with no hook, so the FDT and CrossVal screens could
    not show only their own study out of the one ``fdt`` kind (E3 puts both analyses and their
    comparisons in it). An optional row predicate is the whole addition -- no new widget, because
    every fact the picker shows is already a ``Summary`` field.

    The store is stubbed at the picker's one seam, ``_resolved_store``, as the picker tests already
    stub it; each row carries exactly the ``Summary`` fields ``refresh()`` reads and no more, so the
    test cannot pass on a field the real listing does not fill.
    """
    import types
    from core.gui.widgets.artifact_picker import StorePicker
    from tests._fixtures import qt_app

    qt_app()

    def row(label, id_, *, complete=True, finished=True, study=None):
        return types.SimpleNamespace(complete=complete, finished=finished, label=label, id=id_,
                                     created="2026-09-22T09:00:00", mode=None, width=None,
                                     amortized=None, study=study)

    rows = [row("done_single", "a", study="single"),
            row("running_single", "b", finished=False, study="single"),
            row("done_sweep", "c", study="sweep"),
            row("no_manifest", "d", complete=False, finished=False)]
    store = types.SimpleNamespace(list=lambda kind: list(rows) if kind == "fdt" else [])
    monkeypatch.setattr(StorePicker, "_resolved_store", lambda self: store)

    # (a) unfiltered: the unfinished run and the manifest-less leftover are both absent
    plain = StorePicker("fdt")
    assert [plain.combo.itemText(i) for i in range(plain.combo.count())] == \
        ["done_single", "done_sweep"]

    # (b) the predicate narrows it to one study, and is applied on top of the finished rule
    single = StorePicker("fdt", row_filter=lambda s: s.study == "single")
    assert [single.combo.itemText(i) for i in range(single.combo.count())] == ["done_single"]
    sweep = StorePicker("fdt", row_filter=lambda s: s.study == "sweep")
    assert [sweep.combo.itemText(i) for i in range(sweep.combo.count())] == ["done_sweep"]

    # (c) the predicate survives a refresh, which is how a screen sees a run that just finished
    rows.append(row("second_single", "e", study="single"))
    single.refresh()
    assert [single.combo.itemText(i) for i in range(single.combo.count())] == \
        ["done_single", "second_single"]

    # (d) the default is no predicate at all, so every existing caller is unchanged
    assert StorePicker("fdt", allow_new=True).combo.count() == 3      # the sentinel + the two finished
```

- [ ] **Step 2: Run it and watch it fail**

Run: `pytest tests/test_nav_and_gating.py::test_the_store_picker_offers_finished_rows_only_and_honours_a_row_filter -v`
Expected: FAIL with
`TypeError: StorePicker.__init__() got an unexpected keyword argument 'row_filter'`

- [ ] **Step 3: Take the predicate**

In `core/gui/widgets/artifact_picker.py`, find:

```python
    def __init__(self, kind: str, allow_new: bool = False, store=None, parent=None):
        super().__init__(parent)
        self.kind, self._allow_new, self._store = kind, allow_new, store
```

Replace with:

```python
    def __init__(self, kind: str, allow_new: bool = False, store=None, parent=None,
                 row_filter=None):
        super().__init__(parent)
        self.kind, self._allow_new, self._store = kind, allow_new, store
        # ``Summary -> bool``, or None for "every row of the kind". One kind can hold several things
        # a screen wants to offer separately: the ``fdt`` kind holds single-cell measurements, sweeps
        # and comparisons, told apart by ``body["study"]`` (spec §2.1), and the FDT and CrossVal
        # screens each offer one of them. A predicate here rather than a second widget, because every
        # fact this picker shows is already a Summary field.
        self._row_filter = row_filter
```

- [ ] **Step 4: Filter on `finished`, then on the predicate**

In the same file, find:

```python
        for s in rows:
            if not s.complete:
                continue
```

Replace with:

```python
        for s in rows:
            # FINISHED, not ``complete``. ``complete`` means "has a valid manifest"; for the two
            # kinds written progressively (the training cache, and piece 5's ``fdt``) a record
            # carries one from its first moment, so ``complete`` would offer a run still going, or
            # one a cancel left half written, as a result (spec §5.4). For every other kind the two
            # are the same fact, because _commit writes the manifest last.
            if not s.finished:
                continue
            if self._row_filter is not None and not self._row_filter(s):
                continue
```

- [ ] **Step 5: Say so in the class docstring**

In the same file, find:

```python
    """A combo over one KIND of the artifact store -- the generated-kind twin of ArtifactPicker,
    which stays for the input pickers (cells, bounds). Items are complete artifacts only, labelled by
    name (or ``(unnamed <id>)``), with the id, creation time, mode, width and amortization in the
    tooltip; ``userData`` is the id, which is what ``key()`` persists and ``selected()`` returns."""
```

Replace with:

```python
    """A combo over one KIND of the artifact store -- the generated-kind twin of ArtifactPicker,
    which stays for the input pickers (cells, bounds). Items are FINISHED artifacts only, labelled by
    name (or ``(unnamed <id>)``), with the id, creation time, mode, width and amortization in the
    tooltip; ``userData`` is the id, which is what ``key()`` persists and ``selected()`` returns.

    ``row_filter`` narrows the listing further -- one kind can hold several things a screen offers
    separately (spec §5.4).
    """
```

- [ ] **Step 6: Give the two row stubs the field the picker now reads**

In `tests/test_nav_and_gating.py`, find:

```python
    def row(label, id_, created, *, mode=None, width=None, amortized=None, complete=True):
        return types.SimpleNamespace(complete=complete, label=label, id=id_, created=created,
                                     mode=mode, width=width, amortized=amortized)
```

Replace with:

```python
    def row(label, id_, created, *, mode=None, width=None, amortized=None, complete=True,
            finished=None):
        # For these six kinds ``complete`` and ``finished`` are the same fact, so the stub derives
        # one from the other rather than making every call site repeat it (store.Summary).
        return types.SimpleNamespace(complete=complete,
                                     finished=complete if finished is None else finished,
                                     label=label, id=id_, created=created,
                                     mode=mode, width=width, amortized=amortized)
```

In `tests/test_settings_persistence.py`, find:

```python
    rows = [types.SimpleNamespace(complete=True, label=label, id=id_, created="2026-09-16T12:00:00",
                                  mode="spontaneous", width=50, amortized=None)
            for label, id_ in (("first", "20260916T120000"), ("second", "20260916T130000"))]
```

Replace with:

```python
    rows = [types.SimpleNamespace(complete=True, finished=True, label=label, id=id_,
                                  created="2026-09-16T12:00:00",
                                  mode="spontaneous", width=50, amortized=None)
            for label, id_ in (("first", "20260916T120000"), ("second", "20260916T130000"))]
```

- [ ] **Step 7: Run the tests and watch them pass**

Run: `pytest tests/test_nav_and_gating.py tests/test_settings_persistence.py tests/test_artifact_browser.py -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add core/gui/widgets/artifact_picker.py tests/test_nav_and_gating.py tests/test_settings_persistence.py
git commit -m "picker: offer finished rows only, and take an optional row predicate"
```
# Tasks 25–27: the two panels

> Part of `docs/superpowers/plans/2026-09-22-secondary-analyses.md`. The Global Constraints, the
> Review Focus and the Interface Contract of that document apply to every task here. Line numbers
> are hints; the QUOTED text is the anchor.

**Reading for all three tasks:** spec §5.4 (the pickers, and showing an earlier run), §5.5 (what each
screen remembers), §4.1 (the sweep's two records), §1.2 ("The FRONT END creates the writer; the STAGE
enters it"), §2.2 (the progressive writer), decisions **E1**, **E2**, **E7**, **E8**.

**Standing facts these three tasks rest on**

- `store.create(kind, cfg, *, name, note)` mints the id, runs `assert_name_free` and builds
  `writer.dir`, and does **no** filesystem write — `ArtifactWriter.__enter__` does the `mkdir`
  (`core/artifacts/store.py`: `self.dir = store.kind_dir(kind) / f"{name or mf.UNNAMED_DIR}__{id}"`
  in `__init__`, `self.dir.mkdir(parents=True, exist_ok=False)` in `__enter__`). A writer the panel
  creates and then discards therefore leaves nothing behind.
- `StoreError` is a `Refusal` subclass, so `except Refusal` catches a taken name and
  `BasePanel._refusal` renders it.
- `NewPngWatcher._pngs` returns `set()` for a directory that does not exist
  (`core/gui/plot_watcher.py`: `if not self._dir.is_dir(): return set()`), so pointing a watcher at
  a `figures/` the run has not created yet is safe and needs no `mkdir` from the panel.
- `BasePanel.dispatch` forwards its `**kwargs` to the worker verbatim (`Worker(fn, *args,
  cancel=self._cancel, **kwargs)`), and `cancel` is a named parameter of `Worker.__init__`, not a
  forwarded keyword — so `writer=` and `seed=` reach the stage untouched.
- The panels' controls column is disabled for the whole run (`BasePanel.set_controls_enabled`), so
  no picker signal can fire from the user while a run is live.

#### Amendments (binding — these supersede the text above)

Rulings that change this task: **P9** (the Interface Contract's signature). Confirmed as written: P37 (Step 6), and the P61 keyword use by T25/T26.

**A. P9 — `row_filter` goes BEFORE `parent`.** Step 3's replacement becomes:
```python
    def __init__(self, kind: str, allow_new: bool = False, store=None, row_filter=None,
                 parent=None):
        super().__init__(parent)
        self.kind, self._allow_new, self._store = kind, allow_new, store
        # ``Summary -> bool``, or None for "every row of the kind". One kind can hold several things
        # a screen wants to offer separately: the ``fdt`` kind holds single-cell measurements, sweeps
        # and comparisons, told apart by ``body["study"]`` (spec §2.1), and the FDT and CrossVal
        # screens each offer one of them. Applied AFTER the finished rule. Before ``parent`` (spec
        # §5.4, P9); no caller passes ``parent`` positionally.
        self._row_filter = row_filter
```
The Interfaces "Produces" line reads `StorePicker(kind, allow_new=False, store=None, row_filter=None, parent=None)`. Keep `self._row_filter = row_filter` BEFORE the widget's `self.refresh()` call at the end of `__init__`, as the quoted anchor already places it. Otherwise the first refresh reads an attribute that does not exist yet.

**B. Correction: Step 2's expected failure.** The test builds `StorePicker("fdt")` in (a) first, without `row_filter`. Today that lists the unfinished `running_single` row, which has `complete=True`. Expected:
```
FAIL with AssertionError at (a): ['done_single', 'running_single', 'done_sweep'] == ['done_single', 'done_sweep']
```
not `TypeError ... 'row_filter'`. That TypeError would only come at (b), which is never reached.

Steps 1 and 4-8 stand as written.

---

### Task 25: the FDT panel — the record picker, the Seed box, and the writer

**Why:** Spec §5.4 and **E1**: the FDT screen gains a `StorePicker` over the `fdt` kind filtered to
study `"single"`, and **the panel creates the record before the run is dispatched** — `store.create`
mints the id and runs `assert_name_free` before anything is spent, and `writer.dir` is what the
figure watcher is pointed at. Spec §5.5 and **E7**: the panel keeps remembering its own numeric
fields and remembers the new picker's selection, and the Seed box is **not** remembered — a
remembered seed would silently turn every run into a repeat of the last one.

**Files:**
- Modify: `core/gui/panels/fdt_panel.py:37-52` (`_run_fdt_guarded`), `:74-109` (`_build_controls`),
  `:120-143` (`_run`), `:145-168` (`save_settings` / `restore_settings`), and the class docstring at
  `:54-68`
- Test: `tests/test_settings_persistence.py`, `tests/test_nav_and_gating.py`

**Interfaces:**
- Consumes (T24): `StorePicker(kind, allow_new=False, store=None, row_filter=None, parent=None)`,
  where `row_filter` is `Callable[[Summary], bool] | None` applied to each row of `store.list(kind)`,
  and `refresh` keeps a row only when `s.finished` is true.
- Consumes (T2): `Summary.study: "str | None"`.
- Consumes (T10): `cli.make_fdt_config(model, state_dep_drift, cell_file, *, n_freqs=60,
  ensemble_M=256, freqs_per_batch=1, F0=0.05, seed=None) -> FDTConfig`.
- Consumes (T17): `run_fdt(cfg, *, skip_sanity, confirm_production, writer, seed=None) -> LoadedFdt`.
- Consumes (T1/T3): `store.create("fdt", cfg, name=…, note=…) -> ArtifactWriter` with
  `PROGRESSIVE_KINDS = frozenset({"fdt"})`.
- Produces for T17: the writer reaches `run_fdt` **not entered**, with `writer.body` already carrying
  `study`, `seed`, `notices`, `complete` and the five not-yet-known keys. The stage must **update**
  that dict (`writer.body["settings"] = …`) and never replace it, or the panel's `study` and `seed`
  are lost before the first manifest is written.
- Produces for T27: `FdtPanel.record_picker` (a `StorePicker` over `"fdt"`, study `"single"`),
  persisted under `fdt/record`.

- [ ] **Step 1: Write the failing persistence test**

In `tests/test_settings_persistence.py`, beside `test_settings_round_trip_reduction_and_fdt`:

```python
def test_the_fdt_panel_remembers_its_saved_run_pick_and_never_its_seed(monkeypatch):
    """E7 and spec §5.5. The record picker is a SELECTION and is restored at construction, like the
    cell picker beside it; the Seed box is not, and neither are the record's name and note.

    A remembered seed is the defect this pins: the box is how a run is made a deliberate repeat of an
    earlier one, so a value carried over from the last session would silently turn every later run
    into that repeat, and the spread across repeats -- which E8 calls the measurement error -- would
    collapse to zero without anyone touching the box. A remembered NAME is the same defect wearing a
    different hat: a progressive record occupies its name from its first moment (spec §2.2), so the
    next launch's first click would be refused by ``assert_name_free`` for a name nobody typed.
    """
    import types
    from core.gui import settings as st
    from core.gui.panels.fdt_panel import FdtPanel
    from core.gui.widgets.artifact_picker import StorePicker
    from tests._fixtures import qt_app

    qt_app()
    rows = [types.SimpleNamespace(complete=True, finished=True, study="single", label=label, id=id_,
                                 created="2026-09-22T12:00:00", mode=None, width=None, amortized=None)
            for label, id_ in (("first", "20260922T120000"), ("second", "20260922T130000"))]
    store = types.SimpleNamespace(list=lambda kind: list(rows) if kind == "fdt" else [])
    monkeypatch.setattr(StorePicker, "_resolved_store", lambda self: store)
    fdt = FdtPanel()
    fdt.record_picker.combo.setCurrentIndex(1)
    fdt.seed.setText("4242")
    fdt.record_name.setText("keep_me")
    fdt.record_note.setText("a note about this run")
    fdt.n_freqs.setText("77")

    qs = st.settings()
    fdt.save_settings(qs)
    qs.sync()
    qs.beginGroup("fdt")
    written = set(qs.childKeys())
    qs.endGroup()
    assert not (written & {"seed", "record_name", "record_note"}), \
        f"[fdt] writes a per-run value: {sorted(written)}"

    again = FdtPanel()
    assert again.record_picker.key() == "20260922T130000", "the saved run pick is a selection"
    assert again.n_freqs.text() == "77", "the campaign knobs are still remembered (V5)"
    assert again.seed.text() == "", "a remembered seed repeats the last run in silence"
    assert again.record_name.text() == "" and again.record_note.text() == ""
```

- [ ] **Step 2: Run it and watch it fail**

Run: `pytest tests/test_settings_persistence.py::test_the_fdt_panel_remembers_its_saved_run_pick_and_never_its_seed -v`
Expected: FAIL with `AttributeError: 'FdtPanel' object has no attribute 'record_picker'`.

- [ ] **Step 3: Add the four controls and their persistence**

In `core/gui/panels/fdt_panel.py`, find:

```python
        self.cell_picker = ArtifactPicker(CELL_PATH / "nadrowski")
        self.n_freqs = IntField(60)
        self.ensemble_m = IntField(256)
        self.freqs_per_batch = IntField(1)
        self.f0 = FloatField(0.05)
```

Replace with:

```python
        self.cell_picker = ArtifactPicker(CELL_PATH / "nadrowski")
        self.n_freqs = IntField(60)
        self.ensemble_m = IntField(256)
        self.freqs_per_batch = IntField(1)
        self.f0 = FloatField(0.05)
        # Blank on purpose (IntField(None) is not a thing, so the text is cleared): blank means "draw
        # one and record it" (E7). Never restored -- see the class docstring.
        self.seed = IntField(0)
        self.seed.clear()
        self.record_name = QLineEdit()
        self.record_name.setPlaceholderText("name for this run's record (optional)…")
        self.record_note = QLineEdit()
        self.record_note.setPlaceholderText("a note to keep with it (optional)…")
        # Earlier runs of THIS analysis. Filtered to its own study in _build_controls' sibling step,
        # and to finished records by StorePicker itself (T24).
        self.record_picker = StorePicker("fdt")
```

Find:

```python
        add_help_row(form, "F0 (ND forcing amplitude)", self.f0, HELP["f0"])
        form.addRow(with_badge(self.skip_sanity, HELP["skip_sanity"]))
```

Replace with:

```python
        add_help_row(form, "F0 (ND forcing amplitude)", self.f0, HELP["f0"])
        add_help_row(form, "Seed", self.seed, HELP["seed"])
        add_help_row(form, "Record name", self.record_name, HELP["record_name"])
        add_help_row(form, "Note", self.record_note, HELP["record_note"])
        add_help_row(form, "Saved run", self.record_picker, HELP["record"])
        form.addRow(with_badge(self.skip_sanity, HELP["skip_sanity"]))
```

In the same file, find:

```python
    "confirm_production": "After the sanity checks, proceed to the (long) production sweep automatically.",
}
```

Replace with:

```python
    "confirm_production": "After the sanity checks, proceed to the (long) production sweep automatically.",
    "seed": "The seed this run uses. Leave it blank to draw one — whichever is used is recorded, so a "
            "run can be repeated by typing its seed back here. Repeats of one cell with DIFFERENT "
            "seeds are what measure the spread.",
    "record_name": "A name for the record this run writes. The name is claimed before anything is "
                   "computed, so a name already taken is refused at the click rather than hours in. "
                   "Leave it blank for an unnamed record.",
    "record_note": "Kept with the record and shown in the Artifacts browser.",
    "record": "An earlier single-cell run from the artifact store. Selecting one shows its cell, its "
              "settings, its seed and its notices, and re-opens its figures.",
}
```

Add the imports the block needs. Find:

```python
from PySide6.QtWidgets import QCheckBox, QComboBox, QFormLayout, QGroupBox, QPushButton
```

Replace with:

```python
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFormLayout, QGroupBox, QLineEdit,
                               QPushButton)
```

Find:

```python
from ..widgets.artifact_picker import ArtifactPicker
```

Replace with:

```python
from ..widgets.artifact_picker import ArtifactPicker, StorePicker
```

Then the persistence. Find:

```python
        qs.setValue("cell", self.cell_picker.key())
        for name, fld in (("n_freqs", self.n_freqs), ("ensemble_m", self.ensemble_m),
                          ("freqs_per_batch", self.freqs_per_batch), ("f0", self.f0)):
            settings.save_field(qs, name, fld)
        # The two checkboxes are not written: they are consents (see the class docstring).
        qs.endGroup()
```

Replace with:

```python
        qs.setValue("cell", self.cell_picker.key())
        qs.setValue("record", self.record_picker.key())
        for name, fld in (("n_freqs", self.n_freqs), ("ensemble_m", self.ensemble_m),
                          ("freqs_per_batch", self.freqs_per_batch), ("f0", self.f0)):
            settings.save_field(qs, name, fld)
        # The two checkboxes are not written: they are consents (see the class docstring). Neither
        # are the seed, the record name and the note -- all three belong to ONE run (E7, §5.5).
        qs.endGroup()
```

Find:

```python
        self.cell_picker.restore_key(settings.get_str(qs, "cell"))
        for name, fld in (("n_freqs", self.n_freqs), ("ensemble_m", self.ensemble_m),
                          ("freqs_per_batch", self.freqs_per_batch), ("f0", self.f0)):
            settings.restore_field(qs, name, fld)
```

Replace with:

```python
        self.cell_picker.restore_key(settings.get_str(qs, "cell"))
        self.record_picker.restore_key(settings.get_str(qs, "record"))
        for name, fld in (("n_freqs", self.n_freqs), ("ensemble_m", self.ensemble_m),
                          ("freqs_per_batch", self.freqs_per_batch), ("f0", self.f0)):
            settings.restore_field(qs, name, fld)
```

And the class docstring. Find:

```python
    Persists (group "fdt"): model, cell picker, and the campaign knobs. Restore order matters --
    model FIRST, then the pickers, or the model's refresh() wipes the restored picker.
```

Replace with:

```python
    Persists (group "fdt"): model, cell picker, saved-run picker, and the campaign knobs. Restore
    order matters -- model FIRST, then the pickers, or the model's refresh() wipes the restored
    picker.

    The Seed box, the record name and the note are NOT persisted (E7, spec §5.5). The seed is how a
    run is made a deliberate repeat of an earlier one, so a remembered value would turn every later
    run into that repeat in silence and collapse the spread E8 measures; a remembered NAME would be
    refused by assert_name_free at the next launch's first click, for a name nobody typed; and a note
    describes one run.
```

- [ ] **Step 4: Run it and watch it pass**

Run: `pytest tests/test_settings_persistence.py::test_the_fdt_panel_remembers_its_saved_run_pick_and_never_its_seed -v`
Expected: PASS

- [ ] **Step 5: Write the failing study-filter test**

In `tests/test_nav_and_gating.py`, beside the other FDT panel tests:

```python
def test_the_fdt_panel_offers_only_single_cell_records(monkeypatch):
    """Spec §5.4. One kind holds three studies (E3), so the FDT screen's picker must filter to its
    own: a sweep record has no single-cell ratio curve to draw and a comparison is a picture of other
    records, and offering either would hand T27's viewer a body whose ``grid`` is null by design.

    The filter is the picker's row predicate (T24), applied to the ``Summary`` rows, so nothing about
    the kind's other studies has to be known here."""
    import types
    from core.gui.panels.fdt_panel import FdtPanel
    from core.gui.widgets.artifact_picker import StorePicker
    from tests._fixtures import qt_app

    qt_app()
    rows = [types.SimpleNamespace(complete=True, finished=True, study=study, label=study, id=study,
                                  created="2026-09-22T12:00:00", mode=None, width=None,
                                  amortized=None)
            for study in ("single", "sweep", "comparison")]
    store = types.SimpleNamespace(list=lambda kind: list(rows) if kind == "fdt" else [])
    monkeypatch.setattr(StorePicker, "_resolved_store", lambda self: store)
    panel = FdtPanel()
    combo = panel.record_picker.combo
    assert [combo.itemData(i) for i in range(combo.count())] == ["single"], \
        [combo.itemText(i) for i in range(combo.count())]
```

- [ ] **Step 6: Run it and watch it fail**

Run: `pytest tests/test_nav_and_gating.py::test_the_fdt_panel_offers_only_single_cell_records -v`
Expected: FAIL with `AssertionError: ['single', 'sweep', 'comparison']` — the picker lists every row
of the kind.

- [ ] **Step 7: Give the picker its row predicate**

In `core/gui/panels/fdt_panel.py`, find:

```python
        self.record_picker = StorePicker("fdt")
```

Replace with:

```python
        self.record_picker = StorePicker("fdt", row_filter=lambda s: s.study == "single")
```

- [ ] **Step 8: Run it and watch it pass**

Run: `pytest tests/test_nav_and_gating.py::test_the_fdt_panel_offers_only_single_cell_records -v`
Expected: PASS

- [ ] **Step 9: Write the failing writer-before-dispatch test**

In `tests/test_nav_and_gating.py`:

```python
def test_the_fdt_panel_creates_the_record_before_it_dispatches(tmp_path):
    """Spec §1.2 and §5.4, forced by V4. The panel must know the record's directory BEFORE the run
    starts, because that directory is what the figure watcher is pointed at -- and it must not enter
    the writer itself, because ``log.txt`` is written from ``runs.current_run_log()``, which is
    thread-local and is only populated inside ``capture_run()`` on the WORKER thread. A writer entered
    on the window's thread would write no log at all.

    So: the front end CREATES (the id is minted and the name claimed, and nothing is on disk yet) and
    the stage ENTERS. This pins all four halves of that -- the writer travels as a keyword, the first
    body carries the facts the panel knows, the watch directory is the record's own ``figures/``, and
    ``create`` has written nothing, since ``__enter__`` is what does the mkdir."""
    from core.artifacts import ArtifactStore, use_store
    from core.gui.panels.fdt_panel import FdtPanel, _run_fdt_guarded
    from tests._fixtures import qt_app

    qt_app()
    store = ArtifactStore(tmp_path)
    cap = {}
    with use_store(store):
        panel = FdtPanel()
        panel.dispatch = lambda fn, *a, **k: cap.update(fn=fn, args=a, kwargs=k)
        panel.seed.setText("4242")
        panel.record_name.setText("cell_a_first")
        panel.record_note.setText("the first look")
        panel._run()

    assert cap, "nothing was dispatched"
    assert cap["fn"] is _run_fdt_guarded, cap["fn"]
    writer = cap["kwargs"]["writer"]
    assert writer.kind == "fdt" and writer.name == "cell_a_first" and writer.note == "the first look"
    assert cap["kwargs"]["seed"] == 4242
    assert cap["kwargs"]["watch_dir"] == writer.dir / "figures", cap["kwargs"]["watch_dir"]
    assert not writer.dir.exists(), "create() must not touch the disk; __enter__ does the mkdir"
    assert writer.body["study"] == "single" and writer.body["seed"] == 4242
    assert writer.body["notices"] == [] and writer.body["complete"] is False
    assert all(writer.body[k] is None
               for k in ("grid", "points", "offgrid", "compared", "results")), writer.body
```

`_run_fdt_guarded`, not `run_fdt`, is what the panel dispatches: it translates a malformed cell's
bare `KeyError` into a readable sentence and passes a `Refusal` through untouched. Import it in the
test's import block, beside `FdtPanel`:

```python
    from core.gui.panels.fdt_panel import FdtPanel, _run_fdt_guarded
```

- [ ] **Step 10: Run it and watch it fail**

Run: `pytest tests/test_nav_and_gating.py::test_the_fdt_panel_creates_the_record_before_it_dispatches -v`
Expected: FAIL with `KeyError: 'writer'` — today's `_run` dispatches
`watch_dir=config.artifacts_root() / "fdt"` and no writer at all.

- [ ] **Step 11: Create the writer in `_run` and dispatch it**

In `core/gui/panels/fdt_panel.py`, find (the body of `_run` after the model gate; T21 has already
replaced the old `self._config_error(e)` call with the `Refusal` / bare-exception pair, so keep
whatever it left in the `except` arms and change only what is shown):

```python
        try:
            cfg = cli.make_fdt_config(
                model, registry.state_dep_drift(model), cell,
                n_freqs=self.n_freqs.value(), ensemble_M=self.ensemble_m.value(),
                freqs_per_batch=self.freqs_per_batch.value(), F0=self.f0.value())
```

Replace with:

```python
        # value(), not value_or_none(): a blank box reads as 0 and the BUILDER refuses 0 by name
        # (T10, spec §3.3), which is the one wording both front ends inherit. The seed is the
        # exception -- 0 is a legal seed, so blank must stay blank and mean "draw one" (E7).
        seed = self.seed.value_or_none()
        try:
            cfg = cli.make_fdt_config(
                model, registry.state_dep_drift(model), cell,
                n_freqs=self.n_freqs.value(), ensemble_M=self.ensemble_m.value(),
                freqs_per_batch=self.freqs_per_batch.value(), F0=self.f0.value(), seed=seed)
```

Then find:

```python
        # Explicit bools, never None -- see the module docstring.
        self.dispatch(_run_fdt_guarded, cfg, watch_dir=config.artifacts_root() / "fdt",
                      skip_sanity=self.skip_sanity.isChecked(),
                      confirm_production=self.confirm_production.isChecked(),
                      on_finished=lambda: self.log_pane.append_line("FDT run finished."))
```

Replace with:

```python
        # THE FRONT END CREATES, THE STAGE ENTERS (spec §1.2). create() mints the id and claims the
        # name -- assert_name_free runs here, before a single trajectory is integrated -- and fills
        # writer.dir WITHOUT creating it; run_fdt does `with writer:` on the worker thread, where
        # runs.current_run_log() is populated and log.txt can therefore be written.
        writer = default_store().create("fdt", cfg, name=self.record_name.text().strip(),
                                        note=self.record_note.text().strip())
        # The facts the panel knows. The STAGE fills `settings` from its own private copy of cfg
        # before __enter__ writes the first manifest, and updates this dict in place -- replacing it
        # would drop the study and the seed (spec §2.2, §2.3).
        writer.body = {"study": "single", "settings": None, "seed": seed, "grid": None,
                       "points": None, "offgrid": None, "notices": [], "compared": None,
                       "complete": False, "results": None}
        # Explicit bools, never None -- see the module docstring. The watcher is pointed at the
        # record's own figures/ (spec §5.4); it globs ONE directory and does not recurse, which is
        # exactly that shape, and a directory that does not exist yet lists nothing.
        self.dispatch(_run_fdt_guarded, cfg, writer=writer, seed=seed,
                      watch_dir=writer.dir / "figures",
                      skip_sanity=self.skip_sanity.isChecked(),
                      confirm_production=self.confirm_production.isChecked(),
                      on_result=self._on_record,
                      on_finished=lambda: self.log_pane.append_line("FDT run finished."))

    def _on_record(self, record):
        """The ``LoadedFdt`` run_fdt returned (E1: the run now says what it wrote, where it used to
        return None). The picker is re-listed and moved onto it, so the run that just finished is the
        panel's current selection -- and T27's viewer therefore describes the record whose figures
        are already in the stack, rather than clearing them for whatever was selected before."""
        if record is None:
            return
        self.log_pane.append_line(
            f"FDT record written: {record.name or '(unnamed)'} [{record.id}].")
        self.record_picker.refresh()
        self.record_picker.restore_key(record.id)
```

Widen the guarded call to carry the two new keywords. Find:

```python
def _run_fdt_guarded(cfg, *, skip_sanity, confirm_production):
```

Replace with:

```python
def _run_fdt_guarded(cfg, *, skip_sanity, confirm_production, writer, seed=None):
```

Find:

```python
        return run_fdt(cfg, skip_sanity=skip_sanity, confirm_production=confirm_production)
```

Replace with:

```python
        return run_fdt(cfg, skip_sanity=skip_sanity, confirm_production=confirm_production,
                       writer=writer, seed=seed)
```

Add the store import. Find:

```python
from core import cli, config, registry
```

Replace with:

```python
from core import cli, config, registry
from core.artifacts import default_store
```

- [ ] **Step 12: Run it and watch it pass**

Run: `pytest tests/test_nav_and_gating.py::test_the_fdt_panel_creates_the_record_before_it_dispatches -v`
Expected: PASS

- [ ] **Step 13: Write the failing taken-name test**

In `tests/test_nav_and_gating.py`:

```python
def test_a_taken_fdt_record_name_is_refused_at_the_click(tmp_path):
    """Review Focus 2. A progressive record occupies its name from its first moment (spec §2.2), and
    ``create`` runs ``assert_name_free``, so a second run started while an unfinished record of the
    same name sits on disk is refused BEFORE it spends anything -- rather than colliding hours later
    at a commit, which is how a finished run gets thrown away.

    The refusal is a StoreError, a Refusal subclass carrying ``field="name"``, so it belongs in the
    yellow "Check your inputs" box like every other input refusal on this screen. Raised out of the
    clicked slot instead, it would reach app.py's last-resort excepthook as a raw traceback with
    nothing in the panel's own log -- the defect BasePanel._config_error's docstring describes."""
    from PySide6.QtWidgets import QMessageBox
    from core.artifacts import ArtifactStore, use_store
    from core.gui.panels.fdt_panel import FdtPanel
    from tests._fixtures import SHOWN, qt_app

    qt_app()
    store = ArtifactStore(tmp_path)
    with use_store(store):
        taken = store.create("fdt", None, name="cell_a_first")
        taken.body = {"study": "single", "settings": {}, "seed": 1, "grid": None, "points": None,
                      "offgrid": None, "notices": [], "compared": None, "complete": False,
                      "results": None}
        taken.__enter__()                   # the record now exists, unfinished, holding its name

        panel = FdtPanel()
        panel.dispatch = lambda *a, **k: pytest.fail("a refused click dispatched a run anyway")
        panel.record_name.setText("cell_a_first")
        SHOWN.clear()
        panel._run()

    box = SHOWN[-1]
    assert box.windowTitle() == "Check your inputs" and box.icon() == QMessageBox.Warning
    assert "cell_a_first" in box.text(), box.text()
    assert box.detailedText() == "", "a refusal is not a crash and carries no traceback"
    assert len(list((tmp_path / "fdt").iterdir())) == 1, "the refused click left a second directory"
```

- [ ] **Step 14: Run it and watch it fail**

Run: `pytest tests/test_nav_and_gating.py::test_a_taken_fdt_record_name_is_refused_at_the_click -v`
Expected: FAIL with `core.artifacts.store.StoreError: a fdt named 'cell_a_first' already exists;
rename or delete it first` raised out of `panel._run()` — the `try` wraps only
`cli.make_fdt_config`, and `store.create` was written after it in Step 11.

- [ ] **Step 15: Bring `store.create` inside the guarded block**

In `core/gui/panels/fdt_panel.py`, move the `writer = default_store().create(...)` call inside the
`try` that already wraps the builder, so both refusals take one route. Find:

```python
                freqs_per_batch=self.freqs_per_batch.value(), F0=self.f0.value(), seed=seed)
```

Replace with (keeping the `except` arms T21 wrote, immediately below, untouched):

```python
                freqs_per_batch=self.freqs_per_batch.value(), F0=self.f0.value(), seed=seed)
            # THE FRONT END CREATES, THE STAGE ENTERS (spec §1.2). create() mints the id and claims
            # the name -- assert_name_free runs here, before a single trajectory is integrated -- and
            # fills writer.dir WITHOUT creating it; run_fdt does `with writer:` on the worker thread,
            # where runs.current_run_log() is populated and log.txt can therefore be written. It is
            # inside this try because a taken name is a Refusal about an input on this screen, and
            # belongs in the same yellow box as a bad n_freqs.
            writer = default_store().create("fdt", cfg, name=self.record_name.text().strip(),
                                            note=self.record_note.text().strip())
```

and delete the now-duplicated `writer = default_store().create(...)` line and its comment block from
below the `except` arms, leaving the `writer.body = {...}` assignment where it is.

- [ ] **Step 16: Run the focused tests and watch them pass**

Run: `pytest tests/test_nav_and_gating.py -k "fdt_panel or taken_fdt_record or single_cell_records or secondary_panels" tests/test_settings_persistence.py -k "fdt" -v`
Expected: PASS

- [ ] **Step 17: Commit**

```bash
git add core/gui/panels/fdt_panel.py tests/test_nav_and_gating.py tests/test_settings_persistence.py
git commit -m "fdt panel: a saved-run picker, a Seed box, and the record created before dispatch"
```

#### Amendments (binding — these supersede the text above)

**Before you start (P3, P80).** Tasks 17 and 21 have already edited `core/gui/panels/fdt_panel.py`. **The writer creation and the guard's `writer`/`seed` keywords already exist (P3); if they do not, that is a T17 defect and goes in the report.** When you arrive the file already has:
- `import traceback` and `from core.refusals import Refusal` (T21);
- `from core import cli, registry` and `from core.artifacts import default_store` (T17; `config` is gone);
- `def _run_fdt_guarded(cfg, *, skip_sanity, confirm_production, writer, seed=None):`, whose `return run_fdt(...)` passes `writer=writer, seed=seed` (T17);
- the builder's `except Refusal as e:` / `except Exception as e:` pair (T21);
- below those arms, T17's `writer = default_store().create("fdt", cfg)` and a dispatch that already passes `writer=writer` and `watch_dir=writer.dir / "figures"`.

Add only what is new: `name=`/`note=`, the Seed read, `seed=seed`, `on_result=self._on_record`, the first body, and the record picker.

**A1 — Step 3, the row labels (P34).** Do Step 3 as written. Then, in `_build_controls`, find the rows as Step 3 leaves them:

```python
        add_help_row(form, "Model", self.model_combo, HELP["model"])
        add_help_row(form, "Cell", self.cell_picker, HELP["cell"])
        add_help_row(form, "n_freqs", self.n_freqs, HELP["n_freqs"])
        add_help_row(form, "ensemble_M", self.ensemble_m, HELP["ensemble_M"])
        add_help_row(form, "freqs_per_batch", self.freqs_per_batch, HELP["freqs_per_batch"])
        add_help_row(form, "F0 (ND forcing amplitude)", self.f0, HELP["f0"])
        add_help_row(form, "Seed", self.seed, HELP["seed"])
```

Replace with:

```python
        # Row labels come from core/gui/fields.py (P34), so a row and the fix sentence a refusal prints
        # for it cannot drift apart. "Record name", "Note" and "Saved run" stay literal: `name` and
        # `note` are sentence entries (label() raises KeyError for them) and the saved-run picker is
        # not a refusal field.
        add_help_row(form, gui_fields.label("model"), self.model_combo, HELP["model"])
        add_help_row(form, gui_fields.label("cell"), self.cell_picker, HELP["cell"])
        add_help_row(form, gui_fields.label("n_freqs"), self.n_freqs, HELP["n_freqs"])
        add_help_row(form, gui_fields.label("ensemble_m"), self.ensemble_m, HELP["ensemble_M"])
        add_help_row(form, gui_fields.label("freqs_per_batch"), self.freqs_per_batch,
                     HELP["freqs_per_batch"])
        add_help_row(form, gui_fields.label("f0"), self.f0, HELP["f0"])
        add_help_row(form, gui_fields.label("seed"), self.seed, HELP["seed"])
```

Then add the import. Find `from ..widgets.forms import make_form` and replace it with:

```python
from ..widgets.forms import make_form
from .. import fields as gui_fields
```

Put the import there, NOT beside `from .. import settings`. T27 and T40 anchor on `from .base_panel import BasePanel` / `from .. import settings` / `from ..widgets.artifact_picker import …` staying adjacent.

Each of the seven labels is the literal it replaces (T9's `(place, label)` entries). If `label()` raises `KeyError` when the panel is built, that key is not a tuple entry. Report it; do not fall back to the literal.

Pin the conversion by appending to Step 5's test, after its assertion:

```python
    from tests._fixtures import code_only
    src = code_only(FdtPanel._build_controls)
    for key in ("model", "cell", "n_freqs", "ensemble_m", "freqs_per_batch", "f0", "seed"):
        assert f"label({key!r})" in src, f"the FDT panel must build its {key} row from label({key!r}) (P34)"
```

**A2 — Step 10, the predicted failure (P80).** After T17 the writer is already dispatched, so the test does NOT fail with `KeyError: 'writer'`. Expected: FAIL at `assert writer.kind == "fdt" and writer.name == "cell_a_first" and writer.note == "the first look"`, because T17's `default_store().create("fdt", cfg)` passes no name and no note, so `writer.name == ""`.

**A3 — Step 11 (P80).**

(a) The first find/replace (the `try:` / `cfg = cli.make_fdt_config(` block, gaining `seed = self.seed.value_or_none()` and `seed=seed`) stands. Its anchor is unchanged after T21.

(b) The dispatch anchor Step 11 quotes (`self.dispatch(_run_fdt_guarded, cfg, watch_dir=config.artifacts_root() / "fdt",`) is NOT in the file. Find T17's block instead:

```python
        # The RECORD is created here, on the GUI thread: store.create mints the id and refuses a taken
        # name before anything is spent, and it fills writer.dir WITHOUT creating the directory, so
        # the watcher can be pointed at the record's figures/ before the run is dispatched. The stage
        # enters the writer on the worker thread, where the run log lives (spec §1.2). T25 adds the
        # Seed box, the name/note controls and the record picker on top of this.
        writer = default_store().create("fdt", cfg)
        self.dispatch(_run_fdt_guarded, cfg, watch_dir=writer.dir / "figures",
                      writer=writer,
                      skip_sanity=self.skip_sanity.isChecked(),
                      confirm_production=self.confirm_production.isChecked(),
                      on_finished=lambda: self.log_pane.append_line("FDT run finished."))
```

Replace it with Step 11's second replacement block, verbatim: from `# THE FRONT END CREATES, THE STAGE ENTERS (spec §1.2).` through the end of `def _on_record`. That block still creates the writer BELOW the `except` arms, now with `name=`/`note=`, so Step 14's predicted failure holds; Step 15 then moves the creation.

(c) DELETE Step 11's `_run_fdt_guarded` signature edit and its `return run_fdt(...)` edit. T17 made both. Confirm the signature reads `def _run_fdt_guarded(cfg, *, skip_sanity, confirm_production, writer, seed=None):`.

(d) DELETE Step 11's import edit. `from core import cli, config, registry` is not in the file. Do not add a second `from core.artifacts import default_store`, and do not re-add `config`.

**A4 — Step 15, the shape `_run` must end with.** After Step 15, `_run` from the Seed read down must read as follows. The comments are the ones Steps 11 and 15 give; the `except` arms are T21's, unchanged.

```python
        seed = self.seed.value_or_none()
        try:
            cfg = cli.make_fdt_config(
                model, registry.state_dep_drift(model), cell,
                n_freqs=self.n_freqs.value(), ensemble_M=self.ensemble_m.value(),
                freqs_per_batch=self.freqs_per_batch.value(), F0=self.f0.value(), seed=seed)
            writer = default_store().create("fdt", cfg, name=self.record_name.text().strip(),
                                            note=self.record_note.text().strip())
        except Refusal as e:                         # a setting the user can change: the yellow box
            self._refusal(e)
            return
        except Exception as e:                       # noqa: BLE001 -- a bug in the builder: the red box
            self._on_error(e, traceback.format_exc())
            return

        writer.body = {"study": "single", "settings": None, "seed": seed, "grid": None,
                       "points": None, "offgrid": None, "notices": [], "compared": None,
                       "complete": False, "results": None}
        self.dispatch(_run_fdt_guarded, cfg, writer=writer, seed=seed,
                      watch_dir=writer.dir / "figures",
                      skip_sanity=self.skip_sanity.isChecked(),
                      confirm_production=self.confirm_production.isChecked(),
                      on_result=self._on_record,
                      on_finished=lambda: self.log_pane.append_line("FDT run finished."))
```

Exactly ONE `default_store().create(` must remain in the file.

**A5 — Step 16, the run command.** Two `-k` options keep only the last one, which silently drops `test_the_four_secondary_panels_route_a_refusal_apart_from_a_bug`. Run instead:

`pytest tests/test_nav_and_gating.py tests/test_settings_persistence.py -k "fdt or secondary_panels" -v`

Expected: PASS. The selection includes `test_fdt_panel_guard_translates_model_error_and_gate_admits_builtins`, whose `boom`/`missing` stubs T17 widened to take `writer` and `seed` (P73). If it fails with a `TypeError` about `writer`, that is a T17 defect and goes in the report.

**A6 — unchanged by the rulings, stated so nothing is re-derived.** These are already in the body:
- The Seed box is read once, through `value_or_none()`, and the same integer goes to the builder and to the run (P12, P13).
- The first body is complete, with `settings` left `None` for the stage (P15).
- The record name and the note are not persisted (P30).
- The picker is `self.record_picker`, built with `row_filter=lambda s: s.study == "single"` passed by keyword (P9, P61).

#### Controller rulings (pre-flight, binding — these supersede the amendments above and the task text)

- **F9** — Task 9 left a `_NOT_BUILT_YET` exemption in the label read-back pin for the FDT Seed row, which asserts the row is ABSENT. You build that row, so delete the FDT line from `_NOT_BUILT_YET` in the same commit; the pin then reads your row back.
- **F24** — this is the first task that puts a `StorePicker` where `MainWindow._refresh_store_pickers` walks, so convert `tests/test_artifact_browser.py`'s picker-kinds equality `assert sorted(seen) == ["observation", "posterior", "prior"], seen` to `assert {"observation", "posterior", "prior"} <= set(seen), seen` and add `assert "fdt" in seen`. Add the file to your Files block.
- **F44** — add the assertion spec §8.2 asks for: after a run, the panel names the record it wrote (drive `_on_record` with a record and assert its name or id reaches the log pane).

---

### Task 26: the CrossVal panel — the same, with two writers

**Why:** Spec §4.1 and **E4**: a two-parameter study writes **two** records, one per swept parameter,
so each sweep is answerable on its own and an all-failed activity sweep no longer costs the
temperature sweep. Spec §5.4: the screen's picker is filtered to study `"sweep"`; **E7**: the Seed box
is the study's one seed, recorded on both records, and is not remembered.

**Files:**
- Modify: `core/gui/panels/base_panel.py:205-228` (`dispatch`'s `watch_dir`) and `:259-262`
  (`_finished`'s `watcher.stop()`)
- Modify: `core/gui/panels/crossval_panel.py:50-66` (the class docstring), `:67-106`
  (`_build_controls`), `:140-165` (`_run` and `_on_result`), `:167-190` (persistence)
- Test: `tests/test_nav_and_gating.py`, `tests/test_settings_persistence.py`

**Interfaces:**
- Consumes (T24): `StorePicker(kind, allow_new=False, store=None, row_filter=None, parent=None)`.
- Consumes (T11): `cli.make_param_sweep_config(cell_file, *, preset, preset_name, s_spec, t_spec,
  n_freqs=None, ensemble_M=None, freqs_per_batch=None, F0=None, seed=None)` returning
  `(cfg, s_grid, temp_grid)` as today.
- Consumes (T19): `run_param_study_cli(cfg, *, s_grid, t_grid, writers: dict, seed=None) ->
  list[LoadedFdt]`, where `writers` is `{"s": ArtifactWriter, "temp": ArtifactWriter}` — the same two
  keys `run_fdt_param_sweep` already uses as `sweep_param` (`core/FDT/cross_validation.py`:
  `sweep_param="s"` and `sweep_param="temp"`).
- Produces: `BasePanel.dispatch(..., watch_dir=…)` now accepts a single path **or a sequence of
  paths**, one watcher per path, all feeding the one `FigureStack`. Every existing caller passes a
  single `Path` and is unaffected.
- Produces for T27: `CrossValPanel.record_picker` (a `StorePicker` over `"fdt"`, study `"sweep"`),
  persisted under `crossval/record`.

**How the watcher behaves across two records.** The study writes into two separate record
directories, and `NewPngWatcher` globs one directory and does not recurse. So the panel hands
`dispatch` **both** `figures/` directories and `dispatch` starts one watcher on each; both are
connected to the same `figure_stack.add_png`, so figures appear in the stack in the order they land
on disk — the S sweep's at the study's midpoint, the T sweep's at the end, which is the incremental
behaviour `core/gui/plot_watcher.py`'s docstring says the crossval panel was given a watcher for.
The two sweeps' figure titles already name their parameter (`fdt3d_vs_S`, `fdt3d_vs_T`), so the tabs
stay distinguishable, and `FigureStack` tolerates two tabs with the same title in any case. A record
whose directory never comes into existence — the T sweep never ran because the S sweep refused — is
simply an empty listing: `_pngs()` returns `set()` for a missing directory. `_finished` stops and
deletes **both** watchers, each doing its own final scan.

- [ ] **Step 1: Write the failing multi-directory dispatch test**

In `tests/test_nav_and_gating.py`, beside `test_plot_watcher_only_reports_pngs_written_after_start`:

```python
def test_dispatch_watches_every_directory_it_is_given(tmp_path):
    """Spec §4.1 makes the sweep study write TWO records, so its figures land in two separate
    ``figures/`` directories -- and NewPngWatcher globs ONE directory and does not recurse
    (core/gui/plot_watcher.py: ``self._dir.glob("*.png")``). One watcher would therefore show the S
    sweep's plot and silently lose the T sweep's, which is the half of a long study you waited
    longest for.

    A single path still means one watcher, because every other caller passes one."""
    from pathlib import Path
    from core.gui.panels.base_panel import BasePanel
    from tests._fixtures import qt_app

    app = qt_app()
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(); b.mkdir()
    panel = BasePanel()
    seen = []
    panel.figure_stack.add_png = lambda title, path: seen.append((title, Path(path).name))
    panel.dispatch(lambda: None, watch_dir=[a, b])
    (a / "fdt3d_vs_S_20260922_120000.png").write_bytes(b"s")
    (b / "fdt3d_vs_T_20260922_130000.png").write_bytes(b"t")
    for _ in range(50):
        app.processEvents()
        if len(seen) == 2:
            break
    assert sorted(n for _t, n in seen) == ["fdt3d_vs_S_20260922_120000.png",
                                           "fdt3d_vs_T_20260922_130000.png"], seen
```

- [ ] **Step 2: Run it and watch it fail**

Run: `pytest tests/test_nav_and_gating.py::test_dispatch_watches_every_directory_it_is_given -v`
Expected: FAIL — `NewPngWatcher(watch_dir, self)` is handed the list, `Path([a, b])` raises
`TypeError: argument should be a str or an os.PathLike object where __fspath__ returns a str, not
<class 'list'>`.

- [ ] **Step 3: Let `dispatch` take several watch directories**

In `core/gui/panels/base_panel.py`, find:

```python
        watcher = None
        if watch_dir is not None:
            watcher = NewPngWatcher(watch_dir, self)
            watcher.png_ready.connect(self.figure_stack.add_png)
            watcher.start()
```

Replace with:

```python
        # ONE watcher per directory. NewPngWatcher globs a single directory and does not recurse, and
        # the sweep study writes into TWO records (spec §4.1), so a sequence is a real shape here --
        # a lone path stays a lone path, which is what every other caller passes.
        watchers = []
        for one_dir in ([] if watch_dir is None else
                        [watch_dir] if isinstance(watch_dir, (str, Path)) else list(watch_dir)):
            watcher = NewPngWatcher(one_dir, self)
            watcher.png_ready.connect(self.figure_stack.add_png)
            watcher.start()
            watchers.append(watcher)
```

Find:

```python
            if watcher is not None:
                watcher.stop()          # one last scan: the final figure often lands right at the end
                watcher.deleteLater()
```

Replace with:

```python
            for w in watchers:
                w.stop()                # one last scan: the final figure often lands right at the end
                w.deleteLater()
```

Find:

```python
        ``watch_dir`` is for the FDT / Reduction / CrossVal runners, which save their figures to disk
        instead of handing them back: any PNG appearing there during the run is picked up and shown
        (see core/gui/plot_watcher.py).
```

Replace with:

```python
        ``watch_dir`` is for the FDT / Reduction / CrossVal runners, which save their figures to disk
        instead of handing them back: any PNG appearing there during the run is picked up and shown
        (see core/gui/plot_watcher.py). It is one path, or a SEQUENCE of them -- the sweep study
        writes two records and its figures land in two ``figures/`` directories (spec §4.1) -- and
        every watcher feeds the same figure stack, in the order the files land.
```

Add the import. Find:

```python
import weakref
```

Replace with:

```python
import weakref
from pathlib import Path
```

- [ ] **Step 4: Run it and watch it pass**

Run: `pytest tests/test_nav_and_gating.py::test_dispatch_watches_every_directory_it_is_given -v`
Expected: PASS

- [ ] **Step 5: Write the failing persistence test**

In `tests/test_settings_persistence.py`, beside `test_crossval_does_not_persist_cell_derived_bounds`:

```python
def test_the_crossval_panel_remembers_its_saved_sweep_pick_and_never_its_seed(monkeypatch):
    """The CrossVal half of E7 and spec §5.5, with the same reasoning as the FDT panel's: the picker
    is a selection and is restored, the study's one seed is not. Here the cost of a remembered seed is
    larger -- one seed is recorded on BOTH of the study's records (spec §4.1), so a carried-over value
    would make every later study a bitwise repeat of the last one at every operating point, and the
    two sweeps would agree for a reason that has nothing to do with the physics."""
    import types
    from core.gui import settings as st
    from core.gui.panels.crossval_panel import CrossValPanel
    from core.gui.widgets.artifact_picker import StorePicker
    from tests._fixtures import qt_app

    qt_app()
    rows = [types.SimpleNamespace(complete=True, finished=True, study=study, label=label, id=label,
                                  created="2026-09-22T12:00:00", mode=None, width=None,
                                  amortized=None)
            for study, label in (("sweep", "s_run"), ("sweep", "t_run"), ("single", "one_cell"))]
    store = types.SimpleNamespace(list=lambda kind: list(rows) if kind == "fdt" else [])
    monkeypatch.setattr(StorePicker, "_resolved_store", lambda self: store)
    xv = CrossValPanel()
    combo = xv.record_picker.combo
    assert [combo.itemData(i) for i in range(combo.count())] == ["s_run", "t_run"], \
        "the CrossVal picker must show sweeps only"
    combo.setCurrentIndex(1)
    xv.seed.setText("99")
    xv.record_name.setText("study_one")
    xv.f0.setText("0.077")

    qs = st.settings()
    xv.save_settings(qs)
    qs.sync()
    qs.beginGroup("crossval")
    written = set(qs.childKeys())
    qs.endGroup()
    assert not (written & {"seed", "record_name", "record_note"}), sorted(written)

    again = CrossValPanel()
    assert again.record_picker.key() == "t_run"
    assert again.f0.value() == 0.077, "the free knobs are still remembered (V5)"
    assert again.seed.text() == "" and again.record_name.text() == ""
```

- [ ] **Step 6: Run it and watch it fail**

Run: `pytest tests/test_settings_persistence.py::test_the_crossval_panel_remembers_its_saved_sweep_pick_and_never_its_seed -v`
Expected: FAIL with `AttributeError: 'CrossValPanel' object has no attribute 'record_picker'`.

- [ ] **Step 7: Add the controls and their persistence**

In `core/gui/panels/crossval_panel.py`, find:

```python
        self.n_freqs = IntField(30)
        self.ensemble_m = IntField(256)
        self.freqs_per_batch = IntField(1)
        self.f0 = FloatField(0.05)
```

Replace with:

```python
        self.n_freqs = IntField(30)
        self.ensemble_m = IntField(256)
        self.freqs_per_batch = IntField(1)
        self.f0 = FloatField(0.05)
        # ONE seed for the study, recorded on BOTH records (spec §4.1); blank draws one. Never
        # restored -- see the class docstring.
        self.seed = IntField(0)
        self.seed.clear()
        self.record_name = QLineEdit()
        self.record_name.setPlaceholderText("base name for this study's two records (optional)…")
        self.record_note = QLineEdit()
        self.record_note.setPlaceholderText("a note to keep with both (optional)…")
        self.record_picker = StorePicker("fdt", row_filter=lambda s: s.study == "sweep")
```

Find:

```python
        add_help_row(form, "F0 (ND forcing amplitude)", self.f0, HELP["f0"])
        form.addRow(self.btn_run)
```

Replace with:

```python
        add_help_row(form, "F0 (ND forcing amplitude)", self.f0, HELP["f0"])
        add_help_row(form, "Seed", self.seed, HELP["seed"])
        add_help_row(form, "Record name", self.record_name, HELP["record_name"])
        add_help_row(form, "Note", self.record_note, HELP["record_note"])
        add_help_row(form, "Saved sweep", self.record_picker, HELP["record"])
        form.addRow(self.btn_run)
```

Find:

```python
    "f0": "Non-dimensional forcing amplitude used to probe χ(ω) at each sweep point.",
}
```

Replace with:

```python
    "f0": "Non-dimensional forcing amplitude used to probe χ(ω) at each sweep point.",
    "seed": "The seed this study uses. Leave it blank to draw one — whichever is used is recorded on "
            "BOTH of the study's records, and each operating point derives its own stream from it, so "
            "a point is reproducible from the seed and its index.",
    "record_name": "A base name for the two records this study writes: '_s' and '_temp' are appended, "
                   "one per swept parameter. Both names are claimed before anything is computed. "
                   "Leave it blank for two unnamed records.",
    "record_note": "Kept with both records and shown in the Artifacts browser.",
    "record": "An earlier sweep from the artifact store. Selecting one shows its cell, its settings, "
              "its seed and its notices, and re-opens its figures.",
}
```

Find:

```python
from PySide6.QtWidgets import (QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QPushButton,
                               QWidget)

from core import cli, config
```

Replace with:

```python
from PySide6.QtWidgets import (QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                               QPushButton, QWidget)

from core import cli, config
from core.artifacts import default_store
```

Find:

```python
from ..widgets.artifact_picker import ArtifactPicker
```

Replace with:

```python
from ..widgets.artifact_picker import ArtifactPicker, StorePicker
```

Then persistence. Find:

```python
        qs.setValue("cell", self.cell_picker.key())
        settings.save_field(qs, "f0", self.f0)
```

Replace with:

```python
        qs.setValue("cell", self.cell_picker.key())
        qs.setValue("record", self.record_picker.key())
        # The seed, the record name and the note belong to ONE study and are never written (E7).
        settings.save_field(qs, "f0", self.f0)
```

Find:

```python
        self.cell_picker.restore_key(settings.get_str(qs, "cell"))
```

Replace with:

```python
        self.cell_picker.restore_key(settings.get_str(qs, "cell"))
        self.record_picker.restore_key(settings.get_str(qs, "record"))
```

Find:

```python
    Persists (group "crossval"): the cell picker, the preset, the free knobs (n_freqs, ensemble_M,
    freqs_per_batch, F0) and each grid's point COUNT. Deliberately NOT the grids' lo/hi: those are
    re-derived from the cell, so a value saved against a different cell would be a stale bound.
    """
```

Replace with:

```python
    Persists (group "crossval"): the cell picker, the saved-sweep picker, the preset, the free knobs
    (n_freqs, ensemble_M, freqs_per_batch, F0) and each grid's point COUNT. Deliberately NOT the
    grids' lo/hi: those are re-derived from the cell, so a value saved against a different cell would
    be a stale bound. Deliberately NOT the seed, the record name or the note either (E7, spec §5.5):
    one seed is recorded on both of the study's records, so a remembered one would make every later
    study a repeat of the last at every operating point, and a remembered name would be refused by
    assert_name_free at the next launch's first click.
    """
```

- [ ] **Step 8: Run it and watch it pass**

Run: `pytest tests/test_settings_persistence.py::test_the_crossval_panel_remembers_its_saved_sweep_pick_and_never_its_seed -v`
Expected: PASS

- [ ] **Step 9: Write the failing two-writer test**

In `tests/test_nav_and_gating.py`:

```python
def test_the_crossval_panel_creates_one_writer_per_swept_parameter(tmp_path):
    """Spec §4.1 and E4. The study sweeps two parameters and now writes a record for each, so the
    panel creates TWO writers and hands them over keyed by the same names ``run_fdt_param_sweep``
    already uses for ``sweep_param`` -- "s" and "temp". One record would put an all-failed activity
    sweep and a good temperature sweep in one folder with one ``points`` block, which is precisely the
    coupling E4 removes: today an all-failed S sweep raises before the T sweep even starts.

    The base name is suffixed per parameter, because two records cannot hold one name: a name is
    claimed at create() and a progressive record holds it from its first moment (spec §2.2). Both
    ``figures/`` directories are watched (one watcher each), and neither directory exists yet --
    __enter__ on the worker thread is what creates them."""
    from core.artifacts import ArtifactStore, use_store
    from core.gui.panels.crossval_panel import CrossValPanel
    from tests._fixtures import qt_app

    qt_app()
    store = ArtifactStore(tmp_path)
    cap = {}
    with use_store(store):
        panel = CrossValPanel()
        panel.dispatch = lambda fn, *a, **k: cap.update(fn=fn, args=a, kwargs=k)
        panel.seed.setText("99")
        panel.record_name.setText("study_one")
        panel.record_note.setText("both halves")
        panel._run()

    assert cap, "nothing was dispatched"
    writers = cap["kwargs"]["writers"]
    assert sorted(writers) == ["s", "temp"], sorted(writers)
    assert [writers[k].name for k in ("s", "temp")] == ["study_one_s", "study_one_temp"]
    assert all(w.kind == "fdt" and w.note == "both halves" for w in writers.values())
    assert cap["kwargs"]["seed"] == 99
    assert all(w.body["study"] == "sweep" and w.body["seed"] == 99 for w in writers.values())
    assert all(w.body["complete"] is False and w.body["notices"] == [] for w in writers.values())
    assert cap["kwargs"]["watch_dir"] == [writers["s"].dir / "figures",
                                          writers["temp"].dir / "figures"]
    assert not any(w.dir.exists() for w in writers.values()), \
        "create() must not touch the disk; the stage's __enter__ does the mkdir"
    assert "s_grid" in cap["kwargs"] and "t_grid" in cap["kwargs"], \
        f"the grids travel as keywords now (T19's signature): {sorted(cap['kwargs'])}"
```

- [ ] **Step 10: Run it and watch it fail**

Run: `pytest tests/test_nav_and_gating.py::test_the_crossval_panel_creates_one_writer_per_swept_parameter -v`
Expected: FAIL with `KeyError: 'writers'` — today's `_run` dispatches
`run_param_study_cli, cfg, s_grid, temp_grid` positionally, with
`watch_dir = config.artifacts_root() / "crossval"`.

- [ ] **Step 11: Create both writers in `_run` and dispatch them**

In `core/gui/panels/crossval_panel.py`, find (T21 has already replaced `self._config_error(e)` with
the `Refusal` / bare-exception pair; keep its `except` arms and change only what is shown):

```python
        preset = dict(cli.SWEEP_PRESETS[self.preset_combo.currentText()])
        try:
            cfg, s_grid, temp_grid = cli.make_param_sweep_config(
                cell, preset=preset, s_spec=self.s_grid.spec(), t_spec=self.t_grid.spec(),
                n_freqs=self.n_freqs.value(), ensemble_M=self.ensemble_m.value(),
                freqs_per_batch=self.freqs_per_batch.value(), F0=self.f0.value())
```

Replace with:

```python
        preset_name = self.preset_combo.currentText()
        preset = dict(cli.SWEEP_PRESETS[preset_name])
        # value(), not value_or_none(): a blank box reads as 0 and the BUILDER refuses 0 by name
        # (T11). The seed is the exception -- 0 is a legal seed, so blank stays blank (E7).
        seed = self.seed.value_or_none()
        base = self.record_name.text().strip()
        note = self.record_note.text().strip()
        try:
            cfg, s_grid, temp_grid = cli.make_param_sweep_config(
                cell, preset=preset, preset_name=preset_name,
                s_spec=self.s_grid.spec(), t_spec=self.t_grid.spec(),
                n_freqs=self.n_freqs.value(), ensemble_M=self.ensemble_m.value(),
                freqs_per_batch=self.freqs_per_batch.value(), F0=self.f0.value(), seed=seed)
            # ONE RECORD PER SWEPT PARAMETER (spec §4.1), keyed by the names run_fdt_param_sweep
            # already uses as sweep_param. Two records cannot share one name, so the base is
            # suffixed; both names are claimed here, before anything is spent, and neither directory
            # exists until the stage's __enter__ on the worker thread (spec §1.2).
            store = default_store()
            writers = {key: store.create("fdt", cfg, name=f"{base}_{key}" if base else "", note=note)
                       for key in ("s", "temp")}
```

Find:

```python
        # run_param_study_cli returns the two HDF5 DATA paths, not the figures -- the plots are saved
        # to disk (the S-sweep one at the study's midpoint, deliberately) and arrive via the watcher.
        watch = config.artifacts_root() / "crossval"
        self.dispatch(run_param_study_cli, cfg, s_grid, temp_grid, watch_dir=watch,
                      on_result=self._on_result)

    def _on_result(self, paths):
        if not paths:
            return
        for path in paths:
            self.log_pane.append_line(f"Sweep data: {path}")
```

Replace with:

```python
        for key, w in writers.items():
            # The facts the panel knows. The STAGE fills `settings`, `points`, `offgrid` and
            # `results` and updates this dict in place -- replacing it would drop the study and the
            # seed before the first manifest is written (spec §2.2, §2.3). `points.param` is the
            # stage's too: the swept parameter is recorded once, there.
            w.body = {"study": "sweep", "settings": None, "seed": seed, "grid": None,
                      "points": None, "offgrid": None, "notices": [], "compared": None,
                      "complete": False, "results": None}
        # run_param_study_cli returns the two records, not the figures -- the plots are saved into
        # each record's figures/ (the S-sweep one at the study's midpoint, deliberately) and arrive
        # via one watcher per directory. See the module note on dispatch's watch_dir.
        self.dispatch(run_param_study_cli, cfg, s_grid=s_grid, t_grid=temp_grid,
                      writers=writers, seed=seed,
                      watch_dir=[writers["s"].dir / "figures", writers["temp"].dir / "figures"],
                      on_result=self._on_result)

    def _on_result(self, records):
        """The two ``LoadedFdt`` records the study wrote (E1: it used to return two loose HDF5
        paths). The picker is re-listed and moved onto the LAST of them, so the panel's selection is
        the record whose figures finished the run -- see FdtPanel._on_record for why that matters to
        T27's viewer."""
        if not records:
            return
        for record in records:
            self.log_pane.append_line(
                f"Sweep record written: {record.name or '(unnamed)'} [{record.id}].")
        self.record_picker.refresh()
        self.record_picker.restore_key(records[-1].id)
```

- [ ] **Step 12: Run it and watch it pass**

Run: `pytest tests/test_nav_and_gating.py::test_the_crossval_panel_creates_one_writer_per_swept_parameter -v`
Expected: PASS

- [ ] **Step 13: Update the panel that still reads `config.artifacts_root()`**

`config` is now unused in `core/gui/panels/crossval_panel.py` if nothing else in the file reads it.
Check with `grep -n "config\." core/gui/panels/crossval_panel.py`; if there is no hit, find:

```python
from core import cli, config
from core.artifacts import default_store
```

and replace with:

```python
from core import cli
from core.artifacts import default_store
```

- [ ] **Step 14: Run the focused tests and watch them pass**

Run: `pytest tests/test_nav_and_gating.py -k "crossval or dispatch_watches or secondary_panels or plot_watcher" tests/test_settings_persistence.py -k "crossval" tests/test_figures.py -v`
Expected: PASS

- [ ] **Step 15: Commit**

```bash
git add core/gui/panels/base_panel.py core/gui/panels/crossval_panel.py tests/test_nav_and_gating.py tests/test_settings_persistence.py
git commit -m "crossval panel: two writers, a Seed box, and a watcher per record"
```

#### Amendments (binding — these supersede the text above)

**Before you start (P3, P80).** Tasks 11, 19 and 21 have already edited `core/gui/panels/crossval_panel.py`. **The writer creation and the study's `writers`/`seed` keywords already exist (P3); if they do not, that is a T19 defect and goes in the report.** When you arrive:
- the builder call already passes `preset_name=self.preset_combo.currentText()` (T11);
- the `except Refusal` / `except Exception` arms and the `traceback`/`Refusal` imports exist (T21);
- `from core.artifacts import default_store` is already imported (T19);
- below the `except` arms, T19 creates two unnamed writers and dispatches `run_param_study_cli` by keyword;
- `_on_result` already takes `records` (T19).

**B1 — Step 1, the test must wait for the worker.** This is a correction found against the real code, not a ruling. The watcher polls every 1.2 s and reports a file only once it is 1 s old, except in the final scan that `_finished` runs when the worker's queued `finished` signal arrives. The worker leaves `redirect_streams` only after stopping a pump thread. Fifty bare `processEvents()` calls therefore end before any of that happens, and a miss leaves the class-level `BasePanel._running` True for every later test.

In the test, find:

```python
    for _ in range(50):
        app.processEvents()
        if len(seen) == 2:
            break
```

Replace with:

```python
    # `finished` is QUEUED to this thread; `_finished` stops BOTH watchers and each stop() does the
    # final scan that ignores the settle delay. Wait for it as the module's other dispatch tests do,
    # so BasePanel._running is False again before the next test dispatches anything.
    _wait_for_run(app, panel, limit=30.0)
```

`_wait_for_run` is the module-level helper already in `tests/test_nav_and_gating.py`. Step 2's predicted failure is unchanged: the `TypeError` is raised inside `dispatch` before any waiting; Python 3.12 words it `... not 'list'`.

**B2 — Step 3's prose.** After T19 the sweep figures are `writer.figure_path(f"FDT ratio vs {sweep_param}")`, so the files are `fdt_ratio_vs_s.png` and `fdt_ratio_vs_temp.png` and the tab titles are `fdt ratio vs s` and `fdt ratio vs temp`, not `fdt3d_vs_S` and `fdt3d_vs_T`. No code changes.

**B3 — Step 7's imports.** Do NOT use the four-line anchor: its adjacency to `from core import cli, config` depends on where T21 put `import traceback`. Find only:

```python
from PySide6.QtWidgets import (QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QPushButton,
                               QWidget)
```

Replace with:

```python
from PySide6.QtWidgets import (QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                               QPushButton, QWidget)
```

Do NOT add `from core.artifacts import default_store`; T19 already did. Confirm it with `grep -n "default_store" core/gui/panels/crossval_panel.py`, and if it is absent add it after `from core.config import CELL_PATH` and say so in the report. The `from ..widgets.artifact_picker import ArtifactPicker` → `ArtifactPicker, StorePicker` edit stands.

**B4 — the record names are `<stem>-s` and `<stem>-temp` (P31), with a hyphen.**
- Step 7, HELP: the `"record_name"` text reads `"A base name for the two records this study writes: '-s' and '-temp' are appended, one per swept parameter. Both names are claimed before anything is computed. Leave it blank for two unnamed records."`
- Step 9, the test: `assert [writers[k].name for k in ("s", "temp")] == ["study_one-s", "study_one-temp"]`.
- Step 11: `name=f"{base}-{key}" if base else ""` (see B6).

**B5 — Step 7, the row labels (P34).** After Step 7's row edit, find:

```python
        add_help_row(form, "Cell", self.cell_picker, HELP["cell"])
```

and the rows below it. Replace each keyed literal with `gui_fields.label(key)`:
- `"Cell"` → `gui_fields.label("cell")`
- `"Preset"` → `gui_fields.label("preset")`
- `"S grid  (T_a/T = 1)"` → `gui_fields.label("s_grid")`
- `"T_a/T grid  (S = 0)"` → `gui_fields.label("t_grid")`
- `"n_freqs"` → `gui_fields.label("n_freqs")`
- `"ensemble_M"` → `gui_fields.label("ensemble_m")`
- `"freqs_per_batch"` → `gui_fields.label("freqs_per_batch")`
- `"F0 (ND forcing amplitude)"` → `gui_fields.label("f0")`
- `"Seed"` → `gui_fields.label("seed")`

The `HELP[...]` arguments are unchanged. `"Cell values"`, `"Record name"`, `"Note"` and `"Saved sweep"` stay literal: `Cell values` is an output, `name` and `note` are sentence entries, and `record` is not a field key.

Add `from .. import fields as gui_fields` on the line after `from ..widgets.forms import make_form`, NOT beside `from .. import settings`, so that T27's and T40's import anchors stay adjacent. If `label()` raises `KeyError` when the panel is built, T9 left that key a sentence: report it rather than restore the literal.

Pin the conversion by appending to Step 5's test:

```python
    from tests._fixtures import code_only
    src = code_only(CrossValPanel._build_controls)
    for key in ("cell", "preset", "s_grid", "t_grid", "n_freqs", "ensemble_m", "freqs_per_batch",
                "f0", "seed"):
        assert f"label({key!r})" in src, f"the CrossVal panel must build its {key} row from label({key!r}) (P34)"
```

**B6 — Step 11, the first anchor (P80, after T11).** The quoted builder block is NOT in the file. Find T11's text instead:

```python
        preset = dict(cli.SWEEP_PRESETS[self.preset_combo.currentText()])
        try:
            cfg, s_grid, temp_grid = cli.make_param_sweep_config(
                cell, preset=preset, preset_name=self.preset_combo.currentText(),
                s_spec=self.s_grid.spec(), t_spec=self.t_grid.spec(),
                n_freqs=self.n_freqs.value(), ensemble_M=self.ensemble_m.value(),
                freqs_per_batch=self.freqs_per_batch.value(), F0=self.f0.value())
```

Replace it with Step 11's first replacement block, with one change: the comprehension reads

```python
            writers = {key: store.create("fdt", cfg, name=f"{base}-{key}" if base else "", note=note)
                       for key in ("s", "temp")}
```

The `except` arms directly below are T21's; keep them untouched. The creations are inside the `try`, so a taken name reaches the yellow box.

**B7 — Step 11, the second anchor (P80, after T19).** The quoted `watch = config.artifacts_root() / "crossval"` block is NOT in the file. Find the post-T19 text, from the two surviving comment lines through the end of `_on_result`:

```python
        # run_param_study_cli returns the two HDF5 DATA paths, not the figures -- the plots are saved
        # to disk (the S-sweep one at the study's midpoint, deliberately) and arrive via the watcher.
        # Two records, created here so the watcher knows both folders before the run is dispatched;
        # each sweep enters its own on the worker thread (spec §1.2, §4.1). The watcher takes ONE
        # directory and does not recurse, so it follows the S sweep's figures and the T sweep's arrive
        # with the result line. T26 gives this panel its own picker and Seed box.
        writers = {"s": default_store().create("fdt", cfg), "temp": default_store().create("fdt", cfg)}
        self.dispatch(run_param_study_cli, cfg, s_grid=s_grid, t_grid=temp_grid, writers=writers,
                      watch_dir=writers["s"].dir / "figures", on_result=self._on_result)

    def _on_result(self, records):
        if not records:
            return
        for rec in records:
            self.log_pane.append_line(f"Sweep record: {rec.name or rec.id} at {rec.path}")
```

Replace it with Step 11's second replacement block, from `for key, w in writers.items():` through the end of `_on_result`, with one change for P77: the first line of `_on_result`'s body becomes

```python
        records = [r for r in (records or []) if r is not None]   # P77: one sweep may have refused
```

followed by the existing `if not records: return`. After this step exactly one `store.create(` loop, and no `default_store().create(`, remains in `_run`.

**B8 — Step 10, the predicted failure (P80).** After T19 the writers are already dispatched. Expected: FAIL at `assert [writers[k].name for k in ("s", "temp")] == ["study_one-s", "study_one-temp"]` with `['', '']`, not `KeyError: 'writers'`.

**B9 — Step 13.** The anchor is the single line `from core import cli, config`, since B3 adds nothing under it. After T19, `grep -n "config\." core/gui/panels/crossval_panel.py` finds nothing, so replace that line with `from core import cli`. `from core.config import CELL_PATH` stays.

**B10 — Step 14, the run commands.** Two `-k` options keep only the last. Run:
- `pytest tests/test_nav_and_gating.py tests/test_settings_persistence.py -k "crossval or dispatch_watches or secondary_panels or plot_watcher" -v`
- then `pytest tests/test_figures.py -v`

Expected: PASS.

**B11 — unchanged by the rulings, stated so nothing is re-derived.** These are already in the body:
- One Seed read, through `value_or_none()`, is passed to both the builder and the run (P12, P13, P14).
- Both first bodies are complete, with `settings` left `None` (P15).
- The name and the note are not persisted (P30).
- `self.record_picker` is built with `row_filter=lambda s: s.study == "sweep"` by keyword (P9, P61).

#### Controller rulings (pre-flight, binding — these supersede the amendments above and the task text)

- **F9** — delete the CrossVal Seed line from `_NOT_BUILT_YET` in the label read-back pin, in the same commit that builds the row.
- **F4** — the two-writer test asserts `writers["s"].id != writers["temp"].id`, and a blank-name case asserts `writers["s"].dir != writers["temp"].dir` (Task 19's `_new_id` fix is what makes both true).
- **F18** — one watcher per record, as the draft builds it, is the reading of P35 this piece takes: both records' figures reach the stack, and no worker-to-window "record opened" signal is needed. Say so in the report.
- **F46** — add the assertion that the panel names the records it wrote (drive `_on_result`).

---

### Task 27: showing an earlier run on both screens

**Why:** Spec §5.4, **E1**: "Selecting a record shows its cell, settings, seed and notices, and
re-opens its figures from the record's `figures/`." Without it the pickers T25 and T26 added are
listings nobody can read, and the numbers E1 made the run save stay as unreadable as the flat
`<artifacts root>/fdt` they replaced.

**Files:**
- Create: `core/gui/panels/record_view.py`
- Modify: `core/gui/panels/fdt_panel.py` (`_build_controls`, a new `_show_record`)
- Modify: `core/gui/panels/crossval_panel.py` (`_build_controls`, a new `_show_record`)
- Test: `tests/test_nav_and_gating.py`

**Interfaces:**
- Consumes: `ArtifactStore.get(kind, ref) -> Manifest` and `ArtifactStore.path(kind, ref) -> Path`
  (both already in `core/artifacts/store.py`), `Manifest.inputs` / `.created` / `.body`,
  `plot_watcher._title(name) -> str`, `FigureStack.add_png(title, path)` and `.clear_all()`.
- Consumes (T25/T26): `FdtPanel.record_picker`, `CrossValPanel.record_picker`.
- Produces: `core.gui.panels.record_view.record_summary(m) -> str`,
  `record_figures(path) -> list[tuple[str, str]]`, `details_label() -> QLabel`.

**Why `get`+`path` and not `load_fdt`.** `load_fdt` verifies the payload hash (checklist item 6), and
a sweep's `data.h5` is the whole study's numbers: hashing it on every `currentIndexChanged` would turn
a combo into a whole-file read. The viewer needs only the manifest body and the figures directory,
and both are one cheap manifest read each.

- [ ] **Step 1: Write the failing summary test**

In `tests/test_nav_and_gating.py`:

```python
def test_record_summary_names_the_cell_the_settings_the_seed_and_the_notices():
    """Spec §5.4. A saved run is only usable if you can see what produced it, and §1 measured that the
    old flat output "carries no record of which cell or which settings produced it". The four facts
    the spec names are the four this renders, in that order, off the manifest alone.

    A record with no cell file -- the legacy inline-bounds branch records ``None``
    (core/artifacts/provenance.py) -- must say so rather than raise: this feeds a read-only label, and
    a formatter that raises inside a currentIndexChanged slot takes the panel down with it. The
    notices are E5's "too thin to trust" sentences, and they are shown because a record that is a
    quick look must SAY it is a quick look wherever it is read."""
    from core.gui.panels.record_view import record_summary
    from core.artifacts.manifest import Manifest

    def _m(**over):
        d = dict(schema=1, kind="fdt", id="20260922T120000", name="cell_a", created="2026-09-22T12:00:00",
                 note="", prism={}, env={}, inputs={"cell": {"path": "Resources/Cells/nadrowski/x.txt",
                                                             "sha256": "ab"},
                                                    "bounds": None, "units": None, "model": "NADROWSKI"},
                 config={}, parents={}, fingerprints={}, payloads={}, figures=[],
                 body={"study": "single", "settings": {"n_freqs": 8, "F0": 0.05}, "seed": 4242,
                       "grid": None, "points": None, "offgrid": {"blanks": 1, "of": 8},
                       "notices": ["The frequency grid has 2 points, which is a quick look."],
                       "compared": None, "complete": True, "results": None})
        d.update(over)
        return Manifest(**d)

    text = record_summary(_m())
    assert "2026-09-22T12:00:00" in text and "seed 4242" in text, text
    assert "Resources/Cells/nadrowski/x.txt" in text, text
    assert "F0=0.05" in text and "n_freqs=8" in text, text
    assert "quick look" in text, "a notice must survive to the screen (E5)"
    assert "did not finish" not in text

    body = dict(_m().body, complete=False, seed=None, settings=None, notices=[])
    bare = record_summary(_m(inputs={"cell": None, "bounds": None, "units": None, "model": None},
                             body=body))
    assert "no cell file" in bare, bare
    assert "did not finish" in bare, "an unfinished record must say so wherever it is read (E2)"
```

- [ ] **Step 2: Run it and watch it fail**

Run: `pytest tests/test_nav_and_gating.py::test_record_summary_names_the_cell_the_settings_the_seed_and_the_notices -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'core.gui.panels.record_view'`.

- [ ] **Step 3: Write `core/gui/panels/record_view.py` with its label and its summary**

Create `core/gui/panels/record_view.py` (`record_figures` is added in Step 7, driven by Step 5's
test; write everything else now, imports included — `Path` and `png_title` are that function's and
are unused for the two steps in between):

```python
"""Reading an ``fdt`` record back onto a panel: what it says, and where its pictures are.

TWO SCREENS, ONE WORDING. The FDT and CrossVal panels both show an earlier run (spec §5.4) and both
must show it the same way, so the rendering lives here rather than twice. Nothing here touches the
store's load path: a selection change is a combo event, and ``load_fdt`` verifies the payload hash --
on a sweep's ``data.h5`` that is the whole study re-read for one keystroke. The callers pass the
manifest and the directory, which are two cheap reads.

Every function here feeds a READ-ONLY label or an image tab, so none of them may raise: a formatter
that throws inside a ``currentIndexChanged`` slot brings the panel down with it, and a panel that
fails in ``__init__`` takes ``MainWindow`` and the whole launch with it (the hazard
``CrossValPanel._on_cell_changed`` documents).
"""
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel

# The ONE title formatter. The live watcher names a figure this way as it lands
# (core/gui/plot_watcher.py), so a re-opened figure must be named by the same function or the same
# picture carries two names depending on whether you watched it arrive.
from ..plot_watcher import _title as png_title


def details_label() -> QLabel:
    """A read-only, word-wrapped, plain-text block under a record picker.

    Word-wrapped and PlainText for the reason the inference tabs' derived labels are
    (core/gui/panels/inference/base.py): these carry generated strings that can be long -- a cell
    path and a settings line -- and an unwrapped label widens the whole controls column, which is the
    defect that once put a permanent horizontal scrollbar on the crossval panel.
    """
    lab = QLabel()
    lab.setWordWrap(True)
    lab.setTextFormat(Qt.PlainText)
    return lab


def record_summary(m) -> str:
    """The four facts spec §5.4 names, off an ``fdt`` manifest: when it was written and under which
    seed, its cell, its settings, and its notices -- plus a line when the run did not finish.

    Written to be unraisable on a manifest that is missing anything: ``manifest.validate`` checks
    body KEY SETS and top-level types only, so every value below can legitimately be null and a
    hand-edited one can be anything at all.
    """
    body = getattr(m, "body", None) or {}
    seed = body.get("seed")
    lines = [f"Written {getattr(m, 'created', '?')}"
             + (f" · seed {seed}" if seed is not None else " · seed not recorded")]
    cell = (getattr(m, "inputs", None) or {}).get("cell")
    lines.append(f"Cell: {cell['path']}" if isinstance(cell, dict) and cell.get("path")
                 else "Cell: (no cell file recorded)")
    settings = body.get("settings")
    if isinstance(settings, dict) and settings:
        lines.append(", ".join(f"{k}={settings[k]}" for k in sorted(settings)))
    offgrid = body.get("offgrid")
    if isinstance(offgrid, dict):
        lines.append(f"{offgrid.get('blanks')} of {offgrid.get('of')} probe frequencies came back "
                     f"blank.")
    points = body.get("points")
    if isinstance(points, dict):
        lines.append(f"{points.get('done')} of {points.get('planned')} operating points measured "
                     f"({points.get('failed')} failed), sweeping {points.get('param')}.")
    for notice in (body.get("notices") or []):
        lines.append(f"Notice: {notice}")
    if not body.get("complete", True):
        lines.append("This run did not finish.")
    return "\n".join(lines)
```

- [ ] **Step 4: Run it and watch it pass**

Run: `pytest tests/test_nav_and_gating.py::test_record_summary_names_the_cell_the_settings_the_seed_and_the_notices -v`
Expected: PASS

- [ ] **Step 5: Write the failing figure-listing test**

In `tests/test_nav_and_gating.py`:

```python
def test_record_figures_lists_a_records_pictures_by_the_watchers_own_title(tmp_path):
    """The live watcher and the re-opened view must name the same picture the same way, or a figure
    read off a saved record is a different thing from the one you watched land. Both go through
    plot_watcher._title, which is why record_view imports it rather than restating it.

    A record with no figures/ -- a run cancelled before it drew one, which E2 says keeps its folder --
    is an empty list. It is not an error, and it must not be one: the folder surviving is the point."""
    from core.gui.panels.record_view import record_figures

    rec = tmp_path / "cell_a__20260922T120000"
    (rec / "figures").mkdir(parents=True)
    (rec / "figures" / "t_eff_ratio.png").write_bytes(b"png")
    (rec / "figures" / "spontaneous_psd.png").write_bytes(b"png")
    (rec / "data.h5").write_bytes(b"not a figure")

    assert record_figures(rec) == [
        ("spontaneous psd", str(rec / "figures" / "spontaneous_psd.png")),
        ("t eff ratio", str(rec / "figures" / "t_eff_ratio.png"))]
    assert record_figures(tmp_path / "cell_b__20260922T130000") == []
```

- [ ] **Step 6: Run it and watch it fail**

Run: `pytest tests/test_nav_and_gating.py::test_record_figures_lists_a_records_pictures_by_the_watchers_own_title -v`
Expected: FAIL with `ImportError: cannot import name 'record_figures' from
'core.gui.panels.record_view'`.

- [ ] **Step 7: Add `record_figures`**

In `core/gui/panels/record_view.py`, append to the end of the module:

```python
def record_figures(record_dir) -> list:
    """``[(title, absolute path)]`` for every PNG in ``<record>/figures``, in name order.

    ``figures/`` is the artifact directory's one subdirectory and holds only PNGs
    (``core/artifacts/store.py``), so a glob is the whole listing -- ``data.h5`` and the rest are
    payloads and sit directly in the record directory, not here. A record with no figures -- a run
    that was cancelled before it drew one, whose folder E2 keeps -- is an empty list, never an error.
    """
    figs = Path(record_dir) / "figures"
    if not figs.is_dir():
        return []
    return [(png_title(p.name), str(p)) for p in sorted(figs.glob("*.png"))]
```

- [ ] **Step 8: Run it and watch it pass**

Run: `pytest tests/test_nav_and_gating.py::test_record_figures_lists_a_records_pictures_by_the_watchers_own_title -v`
Expected: PASS

- [ ] **Step 9: Write the failing panel test**

In `tests/test_nav_and_gating.py`:

```python
def test_selecting_a_saved_run_describes_it_and_re_opens_its_figures(tmp_path):
    """Spec §5.4 and E1, on both screens that carry a picker. Selecting a record shows its cell,
    settings, seed and notices, and re-opens its figures from the record's own figures/ -- which is
    what makes a saved run readable at all, and what §1 measured the old flat output could not do.

    Two guards are pinned with it, and both are defects rather than niceties. (1) The slot must not
    touch the figure stack while a run is live: the stack then holds the figures the watcher is
    landing, and clearing it would delete the live run's output to show an older run's. (2) A record
    the store cannot read must degrade to a line in the label, never raise -- this slot fires from
    restore_settings inside __init__, and an exception there escapes CrossValPanel() -> MainWindow()
    -> build_app() before app.py has installed its excepthook, so a single unreadable record would
    leave the application unable to launch at all."""
    from core.artifacts import ArtifactStore, use_store
    from core.gui.panels.crossval_panel import CrossValPanel
    from core.gui.panels.fdt_panel import FdtPanel
    from tests._fixtures import qt_app

    qt_app()
    store = ArtifactStore(tmp_path)

    def _record(study, name):
        w = store.create("fdt", None, name=name)
        w.body = {"study": study, "settings": {"n_freqs": 8}, "seed": 4242, "grid": None,
                  "points": None, "offgrid": None,
                  "notices": ["The frequency grid has 2 points, which is a quick look."],
                  "compared": None, "complete": False, "results": None}
        with w:
            w.figure_path("ratio").write_bytes(b"png")
        return w

    with use_store(store):
        _record("single", "one_cell")
        _record("sweep", "a_sweep")
        for panel, key in ((FdtPanel(), "one_cell"), (CrossValPanel(), "a_sweep")):
            panel.record_picker.restore_key(store.get("fdt", key).id)
            panel._show_record()
            text = panel.record_line.text()
            assert "seed 4242" in text and "quick look" in text, (key, text)
            assert panel.figure_stack.count() == 1, key
            assert panel.figure_stack.tabText(0) == "ratio", panel.figure_stack.tabText(0)

            # (1) a live run owns the figure stack
            panel.figure_stack.clear_all()
            panel._busy = True
            try:
                panel._show_record()
            finally:
                panel._busy = False
            assert panel.figure_stack.count() == 0, f"{key}: a live run's figures were cleared"
            assert "seed 4242" in panel.record_line.text(), "the LINE still follows the selection"

            # (2) an unreadable record is a line, not a crash
            panel.record_picker.combo.setItemData(panel.record_picker.combo.currentIndex(),
                                                  "no_such_record")
            panel._show_record()
            assert "could not read" in panel.record_line.text(), panel.record_line.text()
```

- [ ] **Step 10: Run it and watch it fail**

Run: `pytest tests/test_nav_and_gating.py::test_selecting_a_saved_run_describes_it_and_re_opens_its_figures -v`
Expected: FAIL with `AttributeError: 'FdtPanel' object has no attribute 'record_line'`.

- [ ] **Step 11: Add the viewer to the FDT panel**

In `core/gui/panels/fdt_panel.py`, find:

```python
        add_help_row(form, "Saved run", self.record_picker, HELP["record"])
```

Replace with:

```python
        add_help_row(form, "Saved run", self.record_picker, HELP["record"])
        # Created BEFORE the connect below, because restore_settings at the end of __init__
        # re-selects a saved id and that fires currentIndexChanged straight into the slot.
        self.record_line = record_view.details_label()
        form.addRow("", self.record_line)
        self.record_picker.combo.currentIndexChanged.connect(lambda _i: self._show_record())
```

Find:

```python
    def _on_model_changed(self, model: str):
```

Replace with:

```python
    def _show_record(self) -> None:
        """Spec §5.4: the selected record's cell, settings, seed and notices, and its figures
        re-opened from its own ``figures/``.

        Deliberately broad on the read. This slot runs from restore_settings inside __init__, so ANY
        exception here escapes FdtPanel() -> MainWindow() -> build_app() -- before app.py installs its
        excepthook, so nothing would even show it. One unreadable record must degrade this one label,
        never brick the application.

        The figure stack is left alone while a run is live: it then holds what the watcher is landing,
        and a selection change arriving from _on_record's refresh() would otherwise clear the figures
        of the run that just produced them.
        """
        ref, _is_new = self.record_picker.selected()
        if not ref:
            self.record_line.setText("")
            return
        try:
            store = default_store()
            manifest, path = store.get("fdt", ref), store.path("fdt", ref)
        except Exception as e:                       # noqa: BLE001 -- see the docstring
            self.record_line.setText(f"(could not read the record: {e})")
            return
        self.record_line.setText(record_view.record_summary(manifest))
        if self._busy:
            return
        self.figure_stack.clear_all()
        for title, png in record_view.record_figures(path):
            self.figure_stack.add_png(title, png)

    def _on_model_changed(self, model: str):
```

Find:

```python
from .base_panel import BasePanel
from .. import settings
```

Replace with:

```python
from . import record_view
from .base_panel import BasePanel
from .. import settings
```

- [ ] **Step 12: Add the same viewer to the CrossVal panel**

In `core/gui/panels/crossval_panel.py`, find:

```python
        add_help_row(form, "Saved sweep", self.record_picker, HELP["record"])
```

Replace with:

```python
        add_help_row(form, "Saved sweep", self.record_picker, HELP["record"])
        # Created BEFORE the connect below: restore_settings re-selects a saved id at the end of
        # __init__ and that fires currentIndexChanged straight into the slot.
        self.record_line = record_view.details_label()
        form.addRow("", self.record_line)
        self.record_picker.combo.currentIndexChanged.connect(lambda _i: self._show_record())
```

Find:

```python
    # ── prefill from the cell file: the values cli.make_param_sweep_config then consumes ─────────
```

Replace with:

```python
    def _show_record(self) -> None:
        """Spec §5.4, the CrossVal half: the selected sweep's cell, settings, seed, point counts and
        notices, and its figures re-opened from its own ``figures/``. Identical in shape and wording
        to FdtPanel._show_record -- the rendering is shared in ``record_view`` so the two screens
        cannot word one record two ways -- and identically broad on the read, for the reason this
        panel's _on_cell_changed already states: an exception in a slot __init__ reaches escapes into
        build_app() and the whole GUI fails to launch.
        """
        ref, _is_new = self.record_picker.selected()
        if not ref:
            self.record_line.setText("")
            return
        try:
            store = default_store()
            manifest, path = store.get("fdt", ref), store.path("fdt", ref)
        except Exception as e:                       # noqa: BLE001 -- see the docstring
            self.record_line.setText(f"(could not read the record: {e})")
            return
        self.record_line.setText(record_view.record_summary(manifest))
        if self._busy:
            return
        self.figure_stack.clear_all()
        for title, png in record_view.record_figures(path):
            self.figure_stack.add_png(title, png)

    # ── prefill from the cell file: the values cli.make_param_sweep_config then consumes ─────────
```

Find:

```python
from .base_panel import BasePanel
from .. import settings
```

Replace with:

```python
from . import record_view
from .base_panel import BasePanel
from .. import settings
```

- [ ] **Step 13: Run it and watch it pass**

Run: `pytest tests/test_nav_and_gating.py::test_selecting_a_saved_run_describes_it_and_re_opens_its_figures -v`
Expected: PASS

- [ ] **Step 14: Run the focused suites and watch them pass**

Run: `pytest tests/test_nav_and_gating.py tests/test_settings_persistence.py tests/test_figures.py -v`
Expected: PASS

- [ ] **Step 15: Commit**

```bash
git add core/gui/panels/record_view.py core/gui/panels/fdt_panel.py core/gui/panels/crossval_panel.py tests/test_nav_and_gating.py
git commit -m "fdt and crossval panels: show an earlier run and re-open its figures"
```
## THE COMMAND LINE (T28–T31)

> Read the plan's preamble first: the Global Constraints, the Review Focus and the INTERFACE
> CONTRACT bind every task below. Line numbers are hints taken from the tree at `a0d85da`; the
> QUOTED text is the anchor. Earlier tasks in this piece have already moved lines in
> `core/tool/fdt.py` (T17, T19), `core/tool/browse.py` (T1) and
> `core/gui/screens/artifact_screen.py` (T2, T6) — re-derive every number from the file as you read
> it, and if a quoted anchor is not there, say so in the task report rather than guessing.

#### Amendments (binding — these supersede the text above)

No ruling overrides this task's body. Two things change:

**C1 — Step 10, the predicted failure.** The test calls `panel._show_record()` before it reads `panel.record_line`. Expected: FAIL with `AttributeError: 'FdtPanel' object has no attribute '_show_record'`, not `... 'record_line'`.

**C2 — the anchors, as the earlier tasks leave them.**
- The rows `add_help_row(form, "Saved run", self.record_picker, HELP["record"])` (T25) and `add_help_row(form, "Saved sweep", self.record_picker, HELP["record"])` (T26) are literal. P34's label conversion in T25/T26 does not touch them, because `record` is not a field key. They are your anchors.
- `from core.artifacts import default_store`, which `_show_record` uses, is already imported in `fdt_panel.py` (T17) and in `crossval_panel.py` (T19). Do not add it again.
- If T25/T26 placed `from .. import fields as gui_fields` after `from ..widgets.forms import make_form`, your import anchor `from .base_panel import BasePanel\nfrom .. import settings` is still adjacent in both files. If it is not, re-read the file and say so in the report.

#### Controller rulings (pre-flight, binding — these supersede the amendments above and the task text)

- **F47** — at launch, fill the saved-run LINE for the restored selection but do not re-open its figures (a flag on `_show_record` that skips the figure stack). Re-opening old figures on every launch is behaviour nobody asked for.
- **F48** — `_show_record` removes only the figure tabs IT added (tracked in a list), never `clear_all()`, so a comparison figure drawn by Task 40 survives choosing the next record to append.

---

### Task 28: `fdt` and `crossval` declare `--store-root`, and the three behaviours it keyed are reworked

**Why:** Spec §6.1 and decision **E11**. `core/tool/__init__.py` decides three separate things from
whether a subcommand *declares* `--store-root` — a throwaway `mkdtemp` root, the auto-root cleanup,
and which Ctrl-C advice is printed. Both analyses now write artifact records (T17, T19), so they
need the flag; but all three behaviours are wrong for them, and none may flip by accident. E11 is
the decision that they are "reworked deliberately rather than flipping by accident": no throwaway
root (they follow `artifacts_root()`), no `auto_root` (or a failed run calls `rmdir` on the
operator's real `Artifacts/` root), and their own interrupt advice (which already bypasses the flag
branch because they set `interrupt_note` through `set_defaults`). The dispatch becomes an explicit
per-subcommand property.

**Files:**
- Modify: `core/tool/fdt.py:93-119` (`register`: the two subparsers)
- Modify: `core/tool/smoke.py:120` — `p.set_defaults(handler=run_smoke)`
- Modify: `core/tool/__init__.py:112-119` (the root dispatch) and `:146-147` (the Ctrl-C advice)
- Test: `tests/test_tool.py` — beside the existing fdt/crossval tests (`:1158-1345`)

**Interfaces:**
- Consumes: `main(argv) -> int` and its handler contract `handler(args, store) -> int | None`
  (`core/tool/__init__.py:136`); `core.config.artifacts_root()`; `core/tool/fdt.py`'s
  `register(subparsers) -> dict` and its two handlers `run_fdt_cmd(args, store)` /
  `run_crossval(args, store)` as T17 and T19 left them.
- Produces: `args.store_root` on the `fdt` and `crossval` subcommands (dest `store_root`, default
  `None`), and a new argparse default **`args.temp_store_root: bool`**, set through
  `set_defaults(temp_store_root=True)` on `smoke`'s parser and on no other. `main` reads it with
  `getattr(args, "temp_store_root", False)`. T29 relies on `args.store_root` existing on both
  subcommands and on the interrupt-note branch still winning over the advice branch.

- [ ] **Step 1: Write the failing test — the flag is declared and honoured**

In `tests/test_tool.py`, after `test_fdt_and_crossval_usage_errors`:

```python
def test_fdt_and_crossval_take_store_root_and_otherwise_follow_the_environment(
        tool_env, tmp_path, monkeypatch):
    """E11, first half: both analyses write records now, so both need the flag that says WHERE --
    and without it they must follow PRISM_ARTIFACTS like every subcommand but `smoke`, never a
    throwaway temp root nobody would think to look in.

    The handler is replaced by a recorder, so what is asserted is the root ``main`` OPENED THE STORE
    ON rather than anything a real campaign would write: the dispatch is the subject, and a real
    fdt run is minutes of simulation (it is `slow`-marked, in test_fdt_and_crossval_run_at_tiny_size).
    """
    from core import config
    from core.tool import fdt as fdt_mod
    from core.tool import main

    _bounds, cell, root = tool_env
    seen = {}

    def _rec(args, store):
        seen["root"] = Path(store.root).resolve()
        seen["flag"] = args.store_root
        return 0

    monkeypatch.setattr(fdt_mod, "run_fdt_cmd", _rec)
    monkeypatch.setattr(fdt_mod, "run_crossval", _rec)

    assert main(["fdt", "--cell", cell]) == 0
    assert seen["flag"] is None
    assert seen["root"] == Path(config.artifacts_root()).resolve(), \
        "with no --store-root, fdt must follow PRISM_ARTIFACTS, not a temp directory"
    assert seen["root"] == Path(root).resolve()

    named = tmp_path / "somewhere_else"
    assert main(["fdt", "--cell", cell, "--store-root", str(named)]) == 0
    assert seen["root"] == named.resolve() and seen["flag"] == str(named)

    nad = str(config.CELL_PATH / "nadrowski" / "master_spont.txt")
    assert main(["crossval", "--cell", nad, "--s-grid", "0", "0.1", "2",
                 "--t-grid", "1", "1.1", "2", "--store-root", str(named)]) == 0
    assert seen["root"] == named.resolve(), "crossval takes the same flag, with the same meaning"
```

- [ ] **Step 2: Run it and watch it fail**

Run: `pytest tests/test_tool.py::test_fdt_and_crossval_take_store_root_and_otherwise_follow_the_environment -v`

Expected: FAIL at the first `main([...])` — `assert main(["fdt", "--cell", cell]) == 0` returns `2`,
because argparse has already printed `python -m core fdt: error: unrecognized arguments:
--store-root` for the later call… in fact the first failure is the third assertion,
`seen["root"] == Path(config.artifacts_root()).resolve()`, only if the flag existed. It does not,
so the first `main` call succeeds and the run fails at
`assert main(["fdt", "--cell", cell, "--store-root", str(named)]) == 0` with
`assert 2 == 0` — argparse's `unrecognized arguments: --store-root` exit code.

- [ ] **Step 3: Declare the flag on both subcommands**

In `core/tool/fdt.py`, find:

```python
def _add_fdt_knobs(p) -> None:
```

and insert above it:

```python
def _add_store_root(p) -> None:
    """``--store-root`` for an analysis that writes an ``fdt`` record (E11, spec §6.1).

    DECLARING it changes nothing else by itself. ``main`` used to key three behaviours on whether a
    subcommand declared this flag -- a throwaway ``mkdtemp`` root, the removal of an empty
    auto-created root, and which Ctrl-C advice is printed -- which was ``smoke`` alone. All three are
    wrong here: an unnamed root must be the operator's own PRISM_ARTIFACTS, a failed run must never
    ``rmdir`` that root, and ``_smoke_interrupt_advice`` reads ``args.prior``/``args.save``, which
    these two do not have. So ``main`` now keys them on ``args.temp_store_root``, which only
    ``smoke`` sets.
    """
    p.add_argument("--store-root", dest="store_root", default=None,
                   help="the artifact store this run writes its record into (default: the "
                        "PRISM_ARTIFACTS root, like every subcommand but `smoke`)")
```

Then, in `register`, find:

```python
    fdt.add_argument("--model", default=None,
                     help="model name (default: the cell's parent folder)")
    _add_fdt_knobs(fdt)
```

Replace with:

```python
    fdt.add_argument("--model", default=None,
                     help="model name (default: the cell's parent folder)")
    _add_store_root(fdt)
    _add_fdt_knobs(fdt)
```

and find:

```python
    cv.add_argument("--t-grid", dest="t_grid", nargs=3, type=float, required=True,
                    metavar=("MIN", "MAX", "N"), help="the T_a/T sweep grid")
    _add_fdt_knobs(cv)
```

Replace with:

```python
    cv.add_argument("--t-grid", dest="t_grid", nargs=3, type=float, required=True,
                    metavar=("MIN", "MAX", "N"), help="the T_a/T sweep grid")
    _add_store_root(cv)
    _add_fdt_knobs(cv)
```

- [ ] **Step 4: Run it and watch it fail differently**

Run: `pytest tests/test_tool.py::test_fdt_and_crossval_take_store_root_and_otherwise_follow_the_environment -v`

Expected: FAIL at
`assert seen["root"] == Path(config.artifacts_root()).resolve(), "with no --store-root, fdt must
follow PRISM_ARTIFACTS, not a temp directory"` — the recorded root is a fresh
`%TEMP%\prism_smoke_*` directory, because `has_store_root = hasattr(args, "store_root")` is now
true for `fdt` and `main` takes the `mkdtemp` branch.

- [ ] **Step 5: Make the temp-root dispatch an explicit per-subcommand property**

In `core/tool/smoke.py`, find:

```python
    p.set_defaults(handler=run_smoke)
```

Replace with:

```python
    # THE ONE subcommand that writes to a root of its own rather than to PRISM_ARTIFACTS: `main`
    # reads this property (never `hasattr(args, "store_root")`, which `fdt` and `crossval` now
    # satisfy too) to decide whether an unnamed --store-root means a fresh mkdtemp root, whether an
    # empty auto-created root is removed after a failure, and which Ctrl-C advice is printed.
    p.set_defaults(handler=run_smoke, temp_store_root=True)
```

In `core/tool/__init__.py`, find:

```python
    # smoke is the one subcommand with its own root: a fresh store per run unless one is named, so
    # two runs never share a cache by accident and a named one can be resumed. Keyed on the FLAG
    # (only smoke defines --store-root), never on the subcommand name.
    has_store_root = hasattr(args, "store_root")
    auto_root = has_store_root and not args.store_root         # True only for smoke's own mkdtemp
    if has_store_root:
        root = Path(args.store_root or tempfile.mkdtemp(prefix="prism_smoke_"))
    else:
        root = config.artifacts_root()
```

Replace with:

```python
    # smoke is the one subcommand with its own root: a fresh store per run unless one is named, so
    # two runs never share a cache by accident and a named one can be resumed. Keyed on smoke's OWN
    # PROPERTY (piece 5, E11), never on the FLAG's presence and never on the subcommand name: `fdt`
    # and `crossval` declare --store-root too since piece 5, and all three of the behaviours below
    # are wrong for them. An unnamed root must be the operator's PRISM_ARTIFACTS; `auto_root` must
    # stay false, or a failed run would call rmdir on that real root; and _smoke_interrupt_advice
    # reads args.prior/args.save, which neither of them has.
    temp_store_root = bool(getattr(args, "temp_store_root", False))   # set by smoke's parser alone
    named_root = getattr(args, "store_root", None)
    auto_root = temp_store_root and not named_root             # True only for smoke's own mkdtemp
    if named_root:
        root = Path(named_root)
    elif temp_store_root:
        root = Path(tempfile.mkdtemp(prefix="prism_smoke_"))
    else:
        root = config.artifacts_root()
```

Then find:

```python
            advice = _smoke_interrupt_advice(args, root) if has_store_root else \
                "the same command with --resume require continues them."
```

Replace with:

```python
            advice = _smoke_interrupt_advice(args, root) if temp_store_root else \
                "the same command with --resume require continues them."
```

- [ ] **Step 6: Run the test and watch it pass**

Run: `pytest tests/test_tool.py::test_fdt_and_crossval_take_store_root_and_otherwise_follow_the_environment -v`

Expected: PASS

- [ ] **Step 7: Write the failing test for the auto-root cleanup**

In `tests/test_tool.py`, directly after the test above:

```python
def test_an_fdt_run_that_fails_leaves_the_artifacts_root_where_it_found_it(tmp_path, monkeypatch,
                                                                          capsys):
    """E11's second half, and the accidental flip it exists to prevent. ``main`` removes an
    auto-created store root when the run did not succeed (``_remove_if_still_empty``) -- a safety
    net written for ``smoke``'s own ``mkdtemp`` directory. Keyed on the FLAG's presence, declaring
    ``--store-root`` on ``fdt`` would point that ``rmdir`` at the operator's real ``Artifacts/``
    root. ``rmdir`` refuses a non-empty directory, so nothing would be lost TODAY -- which is
    exactly why this needs a test rather than a reader: the hazard is invisible on any machine whose
    store already holds an artifact.

    PRISM_ARTIFACTS points at a directory that does not exist yet, so ``main`` creates it and the
    root is empty at the moment the cleanup would run: the one state in which the wrong dispatch
    actually deletes something.
    """
    from core import config
    from core.refusals import Refusal
    from core.tool import fdt as fdt_mod
    from core.tool import main

    root = tmp_path / "Artifacts"
    monkeypatch.setenv("PRISM_ARTIFACTS", str(root))
    assert not root.exists()

    def _refuse(args, store):
        raise Refusal("nothing here is real", field="cell")

    monkeypatch.setattr(fdt_mod, "run_fdt_cmd", _refuse)
    capsys.readouterr()
    assert main(["fdt", "--cell", str(tmp_path / "nadrowski" / "cell.txt")]) == 1
    assert root.is_dir(), \
        "a failed fdt run removed the operator's artifacts root (E11: auto_root is smoke's alone)"
    assert Path(config.artifacts_root()).resolve() == root.resolve()
```

- [ ] **Step 8: Run it and watch it fail**

Run: `pytest tests/test_tool.py::test_an_fdt_run_that_fails_leaves_the_artifacts_root_where_it_found_it -v`

Expected: PASS already, because Step 5 has landed. Run it now anyway and record the result: it is
the regression pin for the flip, and running it here proves it exercises the real path rather than a
branch Step 5 removed. **If it fails**, `auto_root` is still keyed on the flag — re-check Step 5's
replacement landed in `core/tool/__init__.py` and not only in the comment.

- [ ] **Step 9: Pin that smoke's own behaviour did not move**

`tests/test_tool.py` already holds `test_smoke_leaves_no_leaked_temp_root_on_a_bad_bounds_file`
(the auto-root cleanup) and `test_smoke_ctrl_c_advice_depends_on_store_root` (the advice). Run both,
plus the whole fdt/crossval group, and confirm they are green.

Run: `pytest tests/test_tool.py -v -k "smoke or fdt or crossval"`
Expected: PASS

- [ ] **Step 10: Commit**

```bash
git add core/tool/__init__.py core/tool/fdt.py core/tool/smoke.py tests/test_tool.py
git commit -m "tool: fdt and crossval take --store-root; the temp root is smoke's own property"
```

#### Amendments (binding — these supersede the text above)

**A1 — Step 2's failure prediction is corrected.** Superseded: "the first `main` call succeeds and the run fails at `assert main(["fdt", "--cell", cell, "--store-root", str(named)]) == 0` with `assert 2 == 0`". Before Step 3 the `fdt` namespace has no `store_root`. The recorder's `seen["flag"] = args.store_root` therefore raises `AttributeError`, and `main`'s last rung prints the traceback and `prism fdt: *** FAILED ***` and returns 1. **Expected: FAIL at the FIRST assertion, `assert main(["fdt", "--cell", cell]) == 0`, as `assert 1 == 0`.** Step 4's prediction stands. The Step-4 run leaves one empty `%TEMP%\prism_smoke_*` directory behind (the old dispatch's `mkdtemp`, with rc 0, so it is not removed); delete it by hand after Step 6.

**A2 — P79: `fdt` and `crossval` gain `--seed`, and this task pins it.** No other task adds the argument. T8 only registered `FLAG["seed"] = "--seed"`, whose pin is green because `smoke` and the diagnostics already define that option string. Insert Steps 6a–6e between Step 6 and Step 7.

- [ ] **Step 6a: Write the failing test.** In `tests/test_tool.py`, directly after `test_fdt_and_crossval_take_store_root_and_otherwise_follow_the_environment`:

```python
def test_fdt_and_crossval_take_seed_and_hand_it_to_the_builder_and_the_run(tool_env, tmp_path,
                                                                           monkeypatch):
    """E7's command-line half (P79). A record carries the seed its run used; without the flag that
    supplies one, that seed can never be supplied back -- the defect E7 names. The ONE integer
    reaches the builder (through ``knobs``, so an unset flag forwards nothing and the builder's own
    default stands) and the run (explicitly, so None there means "draw one") -- the rule both panels
    follow (P12).

    ``ArtifactStore.create`` is replaced because the builder recorders return a placeholder, not an
    FDTConfig, and ``store.create("fdt", cfg)`` reads the run's settings off its cfg: the dispatch is
    the subject here, not the record."""
    from core import cli, config
    from core.artifacts import ArtifactStore
    from core.FDT import cross_validation, fdt_pipeline

    seen = {}

    def _create(self, kind, cfg=None, *, name="", note=""):
        return SimpleNamespace(kind=kind, id="rec", name=name, note=note, dir=tmp_path / "rec",
                               body={})

    def _fdt_cfg(model, state_dep_drift, cell_file, **kw):
        seen["fdt_builder"] = kw
        return "CFG"

    def _run_fdt(cfg, *, skip_sanity, confirm_production, writer, seed=None):
        seen["fdt_run"] = seed
        return SimpleNamespace(id="rec", name="", path=writer.dir)

    def _sweep_cfg(cell_file, **kw):
        seen["sweep_builder"] = kw
        return "CFG", "S", "T"

    def _study(cfg, *, s_grid, t_grid, writers, seed=None):
        seen["study"] = seed
        return [SimpleNamespace(id="s", name="", path=writers["s"].dir),
                SimpleNamespace(id="t", name="", path=writers["temp"].dir)]

    monkeypatch.setattr(ArtifactStore, "create", _create)
    monkeypatch.setattr(cli, "make_fdt_config", _fdt_cfg)
    monkeypatch.setattr(fdt_pipeline, "run_fdt", _run_fdt)
    monkeypatch.setattr(cli, "make_param_sweep_config", _sweep_cfg)
    monkeypatch.setattr(cross_validation, "run_param_study_cli", _study)

    cell = str(config.CELL_PATH / "hopf" / "cell.txt")
    nad = str(config.CELL_PATH / "nadrowski" / "master_spont.txt")
    assert main(["fdt", "--cell", cell, "--seed", "7"]) == 0
    assert seen["fdt_builder"].get("seed") == 7 and seen["fdt_run"] == 7, seen
    assert main(["crossval", "--cell", nad, "--s-grid", "0", "0.1", "2",
                 "--t-grid", "1", "1.1", "2", "--seed", "7"]) == 0
    assert seen["sweep_builder"].get("seed") == 7 and seen["study"] == 7, seen

    seen.clear()
    assert main(["fdt", "--cell", cell]) == 0
    assert "seed" not in seen["fdt_builder"], "an unset --seed forwards nothing to the builder"
    assert seen["fdt_run"] is None, "no --seed: the run draws one and records it (E7, P12)"
```

- [ ] **Step 6b: Run it and watch it fail.** Run: `pytest tests/test_tool.py::test_fdt_and_crossval_take_seed_and_hand_it_to_the_builder_and_the_run -v`. Expected: FAIL at `assert main(["fdt", "--cell", cell, "--seed", "7"]) == 0` as `assert 2 == 0` (argparse: `unrecognized arguments: --seed 7`).

- [ ] **Step 6c: Declare the flag, once, for both subcommands.** In `core/tool/fdt.py`, find:

```python
def _add_fdt_knobs(p) -> None:
    """The four resolution knobs both subcommands share. Each defaults to None and travels only when
    set; the dest is the builder's keyword, capital M and F0 included."""
```

Replace with:

```python
def _add_fdt_knobs(p) -> None:
    """The four resolution knobs both subcommands share, and the seed. Each defaults to None and
    travels only when set; the dest is the builder's keyword, capital M and F0 included. ``--seed``
    is E7's command-line half (P79): a seed a record carries must be one the operator can supply
    back. Unset, the run draws one from [0, 2**31) and records it (P12, P13)."""
```

Then find:

```python
    p.add_argument("--f0", dest="F0", type=float, default=None,
                   help="ND forcing amplitude (keep it inside the linear regime)")
```

Replace with:

```python
    p.add_argument("--f0", dest="F0", type=float, default=None,
                   help="ND forcing amplitude (keep it inside the linear regime)")
    p.add_argument("--seed", type=int, default=None,
                   help="the run's random seed, a whole number from 0 (default: draw one; either "
                        "way the record carries it, so the run can be repeated)")
```

- [ ] **Step 6d: Hand the seed to the builder and to the run, in both handlers.** Under P12 the same integer goes to both. The builder gets it through `knobs`, so an unset flag forwards nothing and the EXACT keyword-set pins in `test_fdt_and_crossval_flags_reach_their_builders` stay green. The run gets it explicitly, so `None` there means "draw one".

In `run_fdt_cmd`, find:

```python
                              **knobs(args, "n_freqs", "ensemble_M", "freqs_per_batch", "F0"))
```

Replace with:

```python
                              **knobs(args, "n_freqs", "ensemble_M", "freqs_per_batch", "F0", "seed"))
```

Then find (T17's call):

```python
    rec = fdt_pipeline.run_fdt(cfg, skip_sanity=args.skip_sanity,
                               confirm_production=not args.no_production, writer=writer)
```

Replace with:

```python
    rec = fdt_pipeline.run_fdt(cfg, skip_sanity=args.skip_sanity,
                               confirm_production=not args.no_production, writer=writer,
                               seed=args.seed)
```

In `run_crossval`, find:

```python
        **knobs(args, "freqs_per_batch", "F0"))
```

Replace with:

```python
        **knobs(args, "freqs_per_batch", "F0", "seed"))
```

Then find (T19's call):

```python
    s_rec, t_rec = cross_validation.run_param_study_cli(cfg, s_grid=s_grid, t_grid=temp_grid,
                                                        writers=writers)
```

Replace with:

```python
    s_rec, t_rec = cross_validation.run_param_study_cli(cfg, s_grid=s_grid, t_grid=temp_grid,
                                                        writers=writers, seed=args.seed)
```

**If an earlier task already passes a seed** (T11, T17 or T19 applying P79 ahead of this flag, e.g. `seed=args.seed` or `seed=getattr(args, "seed", None)`), leave exactly one `seed=args.seed` on each RUN call and NO explicit `seed=` keyword on either BUILDER call. An always-forwarded `seed` adds a key that the flags test's exact-set pins do not list. If an anchor reads differently because an earlier task wrote the call another way, edit the call as it is and say so in the report.

- [ ] **Step 6e: Run it and watch it pass.** Run: `pytest tests/test_tool.py::test_fdt_and_crossval_take_seed_and_hand_it_to_the_builder_and_the_run -v`. Expected: PASS.

**A3 — Interfaces, Produces, gains:** `args.seed: int | None` (dest `seed`, `type=int`, default `None`) on both `fdt` and `crossval`. It is forwarded to `cli.make_fdt_config` / `cli.make_param_sweep_config` through `knobs(...)`, and to `fdt_pipeline.run_fdt` / `cross_validation.run_param_study_cli` as `seed=args.seed`. Its range is the builders' check (T10/T11), not the flag's.

**A4 — Step 9's command is replaced** by `pytest tests/test_tool.py -m "not slow" -v -k "smoke or fdt or crossval"`. Without `-m "not slow"`, the `-k` selects `test_fdt_and_crossval_run_at_tiny_size`, which is slow-marked (about ten minutes) and red by design from T17 until T34 rewrites it. `test_fdt_and_crossval_flags_reach_their_builders` or `test_fdt_ctrl_c_gets_its_own_interrupt_note` may fail with an `AttributeError` raised from `config_from_cfg` on the placeholder `"CFG"`, or a `TypeError` about `_boom`'s keywords. That is the T17/T19 bridging seam (`store.create("fdt", cfg)` reads the settings off the recorder's placeholder): report it and do not fix it here. T29 repairs the ctrl-c test's stubs.

**A5 — Not this task's (P39, P40).** T41 corrects `core/tool/browse.py`'s `NO ``--store-root``, EITHER.` paragraph and the docstring of `test_the_artifacts_family_takes_no_configuration_flags_and_its_help_costs_no_torch`. Leave both alone.

**A6 — Step 10's commit subject becomes** `tool: fdt and crossval take --store-root and --seed; the temp root is smoke's own property`. The `git add` line is unchanged.

Unchanged and confirmed: Step 5 is P10 verbatim, and it keeps `main`'s `interrupt_note` branch ahead of the advice branch, which T29 relies on.

#### Controller rulings (pre-flight, binding — these supersede the amendments above and the task text)

- **F49** — correct, in this commit, the two texts this task makes false, text only, no assertion change: `core/tool/__init__.py`'s epilog sentence "every subcommand but `smoke` writes here, and `smoke` takes --store-root instead" (fdt and crossval now take it too), and the docstring in `tests/test_tool.py` that says the behaviour is "keyed on the --store-root FLAG (hasattr), not the subcommand name" (it is now keyed on `args.temp_store_root`). `core/tool/browse.py`'s own `--store-root` docstring stays Task 41's (P39).
- **F19** — Tasks 17 and 19 passed `seed=getattr(args, "seed", None)`; now that `--seed` exists, normalise both to `seed=args.seed`.

---

### Task 29: the two subcommands' refusals, interrupt notes and corrected wording

**Why:** Spec §6.2. Both subcommands' refusals become `Refusal`s with field keys, so they print as
`main`'s one operator line naming the flag instead of a bare `ValueError` reported with its class
name and raise site (`core/tool/fdt.py:132-139` today). The interrupt notes are rewritten for what
now survives an interrupt — a named, listed, deletable unfinished record (E2), not loose files under
a directory. The two "no bounds file" sentences are corrected (spec §3.2). The CELL-FOLDER branch of
the unsupported-model hint gets the test it has never had — one of the two gaps `docs/STATE.md`
names. And the cell refusal's wording is asserted identical from `load_and_validate_gt` and from the
dry run, closing the item piece 3 handed on
(`2026-09-15-validation-and-logging-design.md:732`).

**Files:**
- Modify: `core/tool/fdt.py:1-13` (module docstring), `:19-51` (the epilogs and the two interrupt
  notes), `:54-59` (`model_for_cell`'s docstring), `:132-139` (the unsupported-model raise)
- Modify: `core/refusals.py:236` (a new message builder after `require_file`)
- Modify: `core/sim_config.py:247-250` (`_fill_checked`'s missing message)
- Modify: `core/cli.py:66-68` (`validate_gt_file`'s missing problem string)
- Test: `tests/test_tool.py` — beside `test_fdt_and_crossval_usage_errors` and
  `test_fdt_ctrl_c_gets_its_own_interrupt_note`

**Interfaces:**
- Consumes: `main`'s refusal rung — `fix_sentence(e.field)` from `core/tool/fields.py`, which maps
  `"model"` to `--model` and `"cell"` to `--cell`; `registry.fdt_support(name) -> (ok, reason)`;
  `args.store_root` from Task 28.
- Produces: **`core.refusals.cell_missing_message(label: str, names) -> str`** — the one sentence for
  "the cell does not supply a parameter the bounds file requires", used by
  `SimConfig._fill_checked` (which raises it as `Refusal(field="cell")`) and by
  `cli.validate_gt_file` (which returns it as a problem string). `FDT_INTERRUPT_NOTE` and
  `CROSSVAL_INTERRUPT_NOTE` keep their names and their `set_defaults(interrupt_note=…)` wiring.

- [ ] **Step 1: Write the failing test for the cell-folder hint branch**

In `tests/test_tool.py`, after `test_fdt_and_crossval_usage_errors`:

```python
def test_an_unsupported_model_named_by_the_cells_folder_is_one_refusal_line(tool_env, tmp_path,
                                                                            capsys):
    """The CELL-FOLDER branch of the unsupported-model hint -- the first of the two gaps
    docs/STATE.md names. Only the ``--model`` branch has ever been tested
    (test_fdt_and_crossval_usage_errors), and the two say different things on purpose: passing the
    wrong ``--model`` and standing a cell in the wrong folder are different mistakes with different
    fixes, and the operator cannot tell which one happened from the reason alone.

    V3 as well (spec §6.2): this used to be a bare ``ValueError``, so ``main``'s unconverted-refusal
    rung printed ``refused: ValueError: ... [raised at fdt.py:139]`` -- the class name and the raise
    site are a hedge for a bug disguised as a refusal, and a fielded ``Refusal`` no longer needs
    either. The cell file is never opened: the model gate runs before the builder, which is the
    point -- refuse before the spend.
    """
    from core.tool import main

    cell = tmp_path / "nosuchmodel" / "cell.txt"
    cell.parent.mkdir(parents=True)
    cell.write_text("", encoding="utf-8")

    capsys.readouterr()
    assert main(["fdt", "--cell", str(cell)]) == 1
    err = capsys.readouterr().err
    lines = [ln for ln in err.splitlines() if ln.startswith("prism fdt: refused:")]
    assert len(lines) == 1, err
    assert "Unknown model 'NOSUCHMODEL'" in lines[0], lines[0]
    assert "the cell's parent folder named it" in lines[0], lines[0]
    assert lines[0].endswith("(--model)"), lines[0]
    assert "ValueError" not in err and "raised at" not in err, err
```

- [ ] **Step 2: Run it and watch it fail**

Run: `pytest tests/test_tool.py::test_an_unsupported_model_named_by_the_cells_folder_is_one_refusal_line -v`

Expected: FAIL at `assert "the cell's parent folder named it" in lines[0]` — the line reads
`prism fdt: refused: ValueError: Unknown model 'NOSUCHMODEL'. (the cell's parent folder set the
model name; pass --model to override it.) [raised at fdt.py:139]`.

- [ ] **Step 3: Make the unsupported-model gate a fielded Refusal**

In `core/tool/fdt.py`, find:

```python
from .config_args import UsageError, knobs, model_from_path
```

Replace with:

```python
from core.refusals import Refusal

from .config_args import UsageError, knobs, model_from_path
```

Then find:

```python
    model = model_for_cell(args)
    ok, reason = registry.fdt_support(model)
    if not ok:
        # M1, fix round 1: say where the model name came from and how to change it, matching
        # config_args.build_cfg's own "Pass --model, or point --bounds at Bounds/<model>/." tail.
        hint = ("pass a different --model" if args.model else
                "the cell's parent folder set the model name; pass --model to override it")
        raise ValueError(f"{reason} ({hint}.)")
```

Replace with:

```python
    model = model_for_cell(args)
    ok, reason = registry.fdt_support(model)
    if not ok:
        # M1, fix round 1, now as V3's one line (spec §6.2): a Refusal with a field key, so the
        # ladder prints `refused: <reason> (<where the name came from>.) (--model)` instead of
        # `refused: ValueError: ... [raised at fdt.py:139]`. The FLAG is fix_sentence's to add, from
        # core/tool/fields.py; what the hint still carries is what the flag cannot -- WHERE the name
        # came from, because --model and a cell's parent folder are two different mistakes.
        hint = ("--model named it" if args.model else
                "the cell's parent folder named it; pass --model to override that")
        raise Refusal(f"{reason} ({hint}.)", field="model")
```

- [ ] **Step 4: Update the `--model`-branch assertion that now goes red**

In `tests/test_tool.py::test_fdt_and_crossval_usage_errors`, find:

```python
    assert main(["fdt", "--cell", cell, "--model", "NOPE"]) == 1
    err = capsys.readouterr().err
    assert "Unknown model" in err
    assert "pass a different --model" in err, "M1: the refusal says where the name came from"
```

Replace with:

```python
    assert main(["fdt", "--cell", cell, "--model", "NOPE"]) == 1
    err = capsys.readouterr().err
    assert "Unknown model" in err
    # M1, as V3's one line since piece 5: the hint says where the NAME came from and the ladder's
    # fix sentence says which flag answers it, so neither has to do the other's job.
    assert "--model named it" in err, "M1: the refusal says where the name came from"
    assert err.rstrip().endswith("(--model)"), err
    assert "ValueError" not in err and "raised at" not in err, err
```

- [ ] **Step 5: Run both tests and watch them pass**

Run: `pytest tests/test_tool.py -v -k "unsupported_model_named_by_the_cells_folder or fdt_and_crossval_usage_errors"`

Expected: PASS

- [ ] **Step 6: Write the failing test for the interrupt notes**

In `tests/test_tool.py`, replace the body of `test_fdt_ctrl_c_gets_its_own_interrupt_note` — find:

```python
    assert main(["fdt", "--cell", cell]) == 130
    err = capsys.readouterr().err
    assert "interrupted" in err
    assert "<artifacts root>/fdt" in err and "stay on disk" in err and "from scratch" in err
    assert "--resume" not in err
```

Replace with:

```python
    assert main(["fdt", "--cell", cell]) == 130
    err = capsys.readouterr().err
    assert "interrupted" in err
    # E2, through the note: what survives an interrupt is a NAMED RECORD, kept and marked
    # unfinished -- not "the plots already written under <artifacts root>/fdt", which is where these
    # outputs stopped going when they became artifacts (spec §6.1).
    assert "unfinished" in err and "artifacts list fdt" in err, err
    assert "artifacts rm fdt" in err, "the note must say how to clear the record it just named"
    assert "from scratch" in err, err
    assert "<artifacts root>/fdt" not in err, "the flat output folder is gone"
    assert "--resume" not in err
```

Also update its docstring's closing sentence — find:

```python
    for them -- there is no cache to resume. The partial plots already on disk are the whole
    recovery story; re-running starts over. Every OTHER subcommand's Ctrl-C message is untouched
```

Replace with:

```python
    for them -- there is no cache to resume. Since piece 5 the recovery story is the RECORD the run
    was writing: kept, marked unfinished, listed and deletable (E2); re-running still starts over,
    because nothing resumes. Every OTHER subcommand's Ctrl-C message is untouched
```

- [ ] **Step 7: Run it and watch it fail**

Run: `pytest tests/test_tool.py::test_fdt_ctrl_c_gets_its_own_interrupt_note -v`

Expected: FAIL at `assert "unfinished" in err and "artifacts list fdt" in err, err` — the note still
reads "the plots already written under <artifacts root>/fdt stay on disk and can be inspected; fdt
keeps no cache, …".

- [ ] **Step 8: Rewrite the two interrupt notes**

In `core/tool/fdt.py`, find:

```python
# I2, fix round 1: fdt/crossval keep no cache and take no --resume (piece 5 wraps their outputs in
# the store; until then there is nothing to resume), so main's generic Ctrl-C advice -- "if a
# [checkpoint] line above says batches were saved, ... --resume require" -- is simply wrong for
# them. Each note names its own output folder and says plainly that the only recovery is what is
# already on disk, and that re-running starts over.
FDT_INTERRUPT_NOTE = (
    "the plots already written under <artifacts root>/fdt stay on disk and can be inspected; fdt "
    "keeps no cache, so re-running the same command starts the analysis from scratch.")
CROSSVAL_INTERRUPT_NOTE = (
    "the outputs already written under <artifacts root>/crossval -- any sweep that finished its "
    "own .h5 and plot -- stay on disk and can be inspected; crossval keeps no cache, so re-running "
    "the same command starts the study from scratch.")
```

Replace with:

```python
# I2, fix round 1, rewritten for piece 5 (E2): fdt/crossval keep no cache and take no --resume, so
# main's generic Ctrl-C advice -- "if a [checkpoint] line above says batches were saved, ...
# --resume require" -- is simply wrong for them. What changed is WHAT SURVIVES. These runs write
# their record PROGRESSIVELY, so an interrupt leaves a real artifact behind -- its directory, its
# manifest marked unfinished, its data file and its figures, and the run's own log.txt -- listed
# like any other and removable by name. Each note therefore says where to find that record and how
# to clear it, and repeats that re-running starts over: an unfinished record is evidence, not a
# resume point (spec §1.3, "Resuming an interrupted sweep": not asked for, and none is added).
FDT_INTERRUPT_NOTE = (
    "the record this run was writing is KEPT and marked unfinished, holding everything measured so "
    "far; `python -m core artifacts list fdt` names it and `python -m core artifacts rm fdt <ref>` "
    "removes it. fdt keeps no cache and nothing resumes, so re-running the same command starts the "
    "analysis from scratch.")
CROSSVAL_INTERRUPT_NOTE = (
    "each sweep writes a record of its own, and the one this run was writing is KEPT and marked "
    "unfinished, holding its data file and whatever spectra it had measured; `python -m core "
    "artifacts list fdt` names both records and `python -m core artifacts rm fdt <ref>` removes "
    "one. crossval keeps no cache and nothing resumes, so re-running the same command starts the "
    "study from scratch.")
```

- [ ] **Step 9: Run it and watch it pass**

Run: `pytest tests/test_tool.py::test_fdt_ctrl_c_gets_its_own_interrupt_note -v`

Expected: PASS

- [ ] **Step 10: Write the failing test for the "no bounds file" sentences**

In `tests/test_tool.py`, after the interrupt-note test:

```python
def test_the_fdt_subcommands_no_longer_say_they_have_no_bounds_file():
    """Spec §3.2. Two sentences in this module claimed these analyses have no bounds file. They are
    false, and the record makes the falsehood expensive: ``cli.parse_cell`` DOES resolve one
    (``resolve_bounds_for_cell`` -- the same-named sibling, else the folder's master), and on the
    decoupled path that file "defines the param set + order" (core/cli.py:156-161). A record that
    did not name which bounds file resolved would not say which parameter set its numbers were
    measured under.

    A source scan, because these are DOCSTRINGS -- nothing executes them, so nothing else can catch
    them going stale. What is true and must stay said is that neither subcommand takes a ``--bounds``
    flag or an observation mode.
    """
    import inspect
    from core.tool import fdt

    text = inspect.getsource(fdt)
    assert "no bounds file" not in text, \
        "a bounds file DOES resolve for the cell; it is recorded by path and hash"
    assert "resolve_bounds_for_cell" in fdt.__doc__, fdt.__doc__
    assert "resolve" in inspect.getdoc(fdt.model_for_cell), inspect.getdoc(fdt.model_for_cell)
```

- [ ] **Step 11: Run it and watch it fail**

Run: `pytest tests/test_tool.py::test_the_fdt_subcommands_no_longer_say_they_have_no_bounds_file -v`

Expected: FAIL at `assert "no bounds file" not in text` — the string occurs twice, in the module
docstring and in `model_for_cell`'s.

- [ ] **Step 12: Correct the two sentences (and the two output claims beside them)**

In `core/tool/fdt.py`, find:

```python
"""``python -m core fdt`` and ``python -m core crossval``: the prompt CLI's two FDT modes as flags.

No new science, and no hardening: each subcommand builds the config ``core/cli.py`` already builds
prompt-free (``make_fdt_config`` / ``make_param_sweep_config``, the same functions the GUI panels
call) and hands it to the same pipeline. FDT/CrossVal hardening, and wrapping their outputs in the
artifact store, is piece 5's work -- until then FDT saves under ``<artifacts root>/fdt`` and the
sweep study under ``<artifacts root>/crossval``, both moved by PRISM_ARTIFACTS and neither touching
the store the tool opens.

Neither takes the SBI config flags: no prior is built and no training runs, so there is no bounds
file and no observation mode. There is no ``--device`` either -- both config builders force the CPU,
where the sequential SDE loop at M ~ 256 is about 3.4x faster than on the card.
"""
```

Replace with:

```python
"""``python -m core fdt`` and ``python -m core crossval``: the two FDT analyses as flags.

Each subcommand builds the config ``core/cli.py`` builds prompt-free (``make_fdt_config`` /
``make_param_sweep_config``, the same functions the GUI panels call) and hands it to the same
pipeline. Since piece 5 both write an ``fdt`` artifact RECORD -- progressively, so an interrupted
run keeps its folder marked unfinished -- into ``--store-root``, or into the PRISM_ARTIFACTS root
when that flag is not given.

Neither takes the SBI config flags: no prior is built and no training runs, so there is no
observation mode and no ``--bounds``. A bounds file is not ABSENT, though: ``cli.parse_cell``
resolves one for the cell (``cli.resolve_bounds_for_cell`` -- the same-named sibling, else the
model folder's master), it defines the parameter set and its order on the decoupled path, and the
record names it by path and SHA-256 like every other kind. There is no ``--device`` either -- both
config builders force the CPU, where the sequential SDE loop at M ~ 256 is about 3.4x faster than
on the card.
"""
```

Then find:

```python
def model_for_cell(args) -> str:
    """``--model``, else the cell's parent folder upper-cased (the ``Cells/<model>/`` layout). The
    SBI subcommands take the model from the BOUNDS folder; these two have no bounds file. A thin
    wrapper over ``config_args.model_from_path`` -- kept as its own function because the interface
    and the test suite name it ``fdt.model_for_cell``."""
```

Replace with:

```python
def model_for_cell(args) -> str:
    """``--model``, else the cell's parent folder upper-cased (the ``Cells/<model>/`` layout). The
    SBI subcommands take the model from the BOUNDS folder, which they are given; these two are given
    a CELL and resolve its bounds file from it (``cli.resolve_bounds_for_cell``), so the folder is
    what names the model here. A thin wrapper over ``config_args.model_from_path`` -- kept as its
    own function because the interface and the test suite name it ``fdt.model_for_cell``."""
```

Then find, in `FDT_EPILOG`:

```python
Plots (PSD, chi components, T_eff/T, the spontaneous trajectory) are written to
<artifacts root>/fdt, timestamped. Sanity checks run first unless --skip-sanity; --no-production
stops after them.
```

Replace with:

```python
The run writes one `fdt` record: its figures (PSD, chi components, T_eff/T, the spontaneous
trajectory), its numbers in data.h5, its settings and seed, and its log. Sanity checks run first
unless --skip-sanity; --no-production stops after them.
```

and, in `CROSSVAL_EPILOG`, find:

```python
--preset drives the resolution levers the flags do not expose (freq_bounds, T_obs_periods,
psd_T_obs_nd) and supplies the defaults for --n-freqs and --ensemble-m. One HDF5 per sweep plus a
3-D plot each go to <artifacts root>/crossval; the S-sweep plot is saved at the study's midpoint,
so a long run gives you half its answer early.
```

Replace with:

```python
--preset drives the resolution levers the flags do not expose (freq_bounds, T_obs_periods,
psd_T_obs_nd) and supplies the defaults for --n-freqs and --ensemble-m. Each sweep writes its OWN
`fdt` record -- its data.h5, its 3-D plot and its point counts -- so the S sweep is a finished,
readable answer before the T sweep starts.
```

- [ ] **Step 13: Run it and watch it pass**

Run: `pytest tests/test_tool.py::test_the_fdt_subcommands_no_longer_say_they_have_no_bounds_file -v`

Expected: PASS

- [ ] **Step 14: Write the failing test for the one cell wording**

In `tests/test_tool.py`, after the previous test:

```python
def test_a_cell_missing_a_bounds_parameter_reads_the_same_from_the_check_and_the_dry_run(tool_env):
    """The item piece 3 handed on (2026-09-15-validation-and-logging-design.md:732), closed here
    (spec §6.2). ONE rule -- "the cell does not supply a parameter the bounds file declares" -- had
    two wordings: ``SimConfig._fill_checked`` raised "Cell file is missing ND parameters required by
    the bounds file: [...]" while ``cli.validate_gt_file``, the non-mutating dry run the Infer tab
    runs the moment a cell is PICKED, returned "missing ND parameter(s) the bounds file requires:
    ...". The operator meets the dry run's sentence on the screen and the check's in the log of the
    run that then failed, and has to work out that they are the same complaint.

    Driven off ONE config with ONE parameter the cell cannot supply, so the two calls answer about
    the same missing name and the comparison is of WORDING, not of contents.
    """
    import pytest
    from core import cli, config
    from core.refusals import Refusal
    from tests._fixtures import _nad_cfg

    cfg = _nad_cfg()
    cfg.params_dict["not_in_any_cell"] = (1.0, (0.0, 2.0))
    cell = str(config.CELL_PATH / "nadrowski" / "master_spont.txt")

    problems = cli.validate_gt_file(cfg, cell)
    with pytest.raises(Refusal) as exc:
        cli.load_and_validate_gt(cfg, cell)
    assert exc.value.field == "cell"
    assert problems == [exc.value.message], (problems, exc.value.message)
    assert "not_in_any_cell" in exc.value.message, exc.value.message
```

- [ ] **Step 15: Run it and watch it fail**

Run: `pytest tests/test_tool.py::test_a_cell_missing_a_bounds_parameter_reads_the_same_from_the_check_and_the_dry_run -v`

Expected: FAIL at `assert problems == [exc.value.message]` — the left side is
`["missing ND parameter(s) the bounds file requires: not_in_any_cell"]`, the right
`["Cell file is missing ND parameters required by the bounds file: ['not_in_any_cell']."]`.

- [ ] **Step 16: Put both sites on one message builder**

In `core/refusals.py`, find:

```python
# The note limit, HERE and not in core/config.py: this module imports only the standard library on
# purpose (see the module docstring), the registry's own defaults are already literals pinned against
# config.py by tests/test_refusals.py, and a one-line description's length is not a science constant.
NOTE_MAX_CHARS = 200
```

and insert above it:

```python
def cell_missing_message(label: str, names) -> str:
    """The ONE sentence for "the cell does not supply a parameter the bounds file declares".

    Not a ``require_*`` rule: it raises nothing and returns nothing to use. It exists because this
    complaint is made from two places that cannot share a call -- ``SimConfig._fill_checked``, which
    RAISES it as a ``Refusal(field="cell")`` when a run injects the cell, and
    ``cli.validate_gt_file``, the non-mutating dry run a GUI runs the moment a cell is picked, which
    RETURNS it as a problem string. Two wordings for one rule made an operator match the sentence on
    the screen against the sentence in the log by hand; piece 3 recorded the gap and piece 5 closes
    it (spec §6.2).

    ``label`` is the section as the bounds file speaks of it ("ND parameters", "rescale parameters",
    "forcing parameters"); ``names`` is whatever is missing, sorted here so the sentence is stable.
    """
    return (f"Cell file is missing {label} required by the bounds file: "
            f"{sorted(str(n) for n in names)}.")
```

In `core/sim_config.py`, find:

```python
        missing = sorted(set(cfg_dict) - set(cell_vals))
        if missing:
            raise Refusal(
                f"Cell file is missing {label} required by the bounds file: {missing}.", field="cell")
```

Replace with:

```python
        missing = sorted(set(cfg_dict) - set(cell_vals))
        if missing:
            # ONE wording, shared with cli.validate_gt_file's dry run (spec §6.2): the sentence the
            # operator reads when a cell is picked and the sentence the run refuses with are the
            # same string, so they cannot drift.
            raise Refusal(cell_missing_message(label, missing), field="cell")
```

and make the name available — find, in `core/sim_config.py`'s imports:

```python
from core.refusals import Refusal
```

Replace with:

```python
from core.refusals import Refusal, cell_missing_message
```

In `core/cli.py`, find:

```python
from .refusals import Refusal, require_file
```

Replace with:

```python
from .refusals import Refusal, cell_missing_message, require_file
```

and find:

```python
    problems = []
    for label, vals, cfg_dict, check_bounds in (
            ("ND parameter", param_vals, cfg.params_dict, True),
            ("rescale parameter", rescale_vals, cfg.rescale_params, True),
            ("forcing parameter", forcing_vals, cfg.force_params_dict, False)):
        missing = sorted(set(cfg_dict) - set(vals))
        if missing:
            problems.append(f"missing {label}(s) the bounds file requires: {', '.join(missing)}")
            continue
```

Replace with:

```python
    problems = []
    # Two labels per section: the SINGULAR one reads correctly in the out-of-bounds sentence below
    # ("ND parameter k = ... is outside its bounds"), the PLURAL one is what the bounds file's own
    # rule speaks of and is what cell_missing_message takes -- the same string SimConfig._fill_checked
    # refuses with, so the dry run and the check cannot word one rule two ways (spec §6.2).
    for label, section, vals, cfg_dict, check_bounds in (
            ("ND parameter", "ND parameters", param_vals, cfg.params_dict, True),
            ("rescale parameter", "rescale parameters", rescale_vals, cfg.rescale_params, True),
            ("forcing parameter", "forcing parameters", forcing_vals, cfg.force_params_dict, False)):
        missing = sorted(set(cfg_dict) - set(vals))
        if missing:
            problems.append(cell_missing_message(section, missing))
            continue
```

- [ ] **Step 17: Run the test and watch it pass**

Run: `pytest tests/test_tool.py::test_a_cell_missing_a_bounds_parameter_reads_the_same_from_the_check_and_the_dry_run -v`

Expected: PASS

- [ ] **Step 18: Run the suites that read those two sentences**

Run: `pytest tests/test_tool.py tests/test_refusals.py tests/test_artifact_consistency.py -v -q`
Expected: PASS. `tests/test_refusals.py` scans this module's registry and its rules; the new helper
is not a `require_*` rule and registers no field key, so neither `len(FIELDS)` nor the flag table
moves.

- [ ] **Step 19: Commit**

```bash
git add core/tool/fdt.py core/refusals.py core/sim_config.py core/cli.py tests/test_tool.py
git commit -m "tool: fdt/crossval refuse with field keys, and one wording for a missing cell value"
```

#### Amendments (binding — these supersede the text above)

**A1 — P76: this task produces no message builder, and its cell-wording half is deleted.** T21 already added `core.refusals.missing_values_phrase` (P4, P8), converted both sites to it, and pins the cross-site wording itself with `assert exc.value.message == f"Cell file is {problems[0]}."`. Therefore:
- **Steps 14, 15, 16 and 17 are deleted in full.** Do not write `test_a_cell_missing_a_bounds_parameter_reads_the_same_from_the_check_and_the_dry_run`, do not create `cell_missing_message`, and do not open `core/refusals.py`, `core/sim_config.py` or `core/cli.py`. Step 16's anchors are already gone because T21 replaced them. That is expected, not a defect to report.
- **Files:** delete `Modify: core/refusals.py:236 …`, `Modify: core/sim_config.py:247-250 …` and `Modify: core/cli.py:66-68 …`. This task modifies `core/tool/fdt.py` and `tests/test_tool.py` only.
- **Interfaces → Produces:** delete the sentence beginning "**`core.refusals.cell_missing_message(label: str, names) -> str`**". Keep "`FDT_INTERRUPT_NOTE` and `CROSSVAL_INTERRUPT_NOTE` keep their names and their `set_defaults(interrupt_note=…)` wiring."
- **Why:** delete its last sentence, "And the cell refusal's wording is asserted identical from `load_and_validate_gt` and from the dry run, closing the item piece 3 handed on (…:732)." T21 closes that item.

**A2 — Step 18 is replaced.** Run: `pytest tests/test_tool.py tests/test_refusals.py -m "not slow" -q`. Expected: PASS. `-m "not slow"` keeps out `test_fdt_and_crossval_run_at_tiny_size`, which takes about ten minutes and is red by design until T34. This task adds no flag and no field key, so neither `len(FIELDS)` nor the FLAG-against-parser pins move. `tests/test_artifact_consistency.py` is dropped from the command because nothing it reads is touched now.

**A3 — Step 19 is replaced:**

```bash
git add core/tool/fdt.py tests/test_tool.py
git commit -m "tool: fdt/crossval refuse with field keys; the interrupt notes name the kept record"
```

**A4 — Steps 10–13 STAY.** P39 said T41 owns the two "no bounds file" sentences. P76, later and explicit, says T29 keeps them, and T41's body carries neither. Do them here as written.

**A5 — Step 6: the test's stubs must survive T17's handler.** Run `pytest tests/test_tool.py::test_fdt_ctrl_c_gets_its_own_interrupt_note -v` BEFORE editing. If it passes, keep its stubs. It fails if `main` returns 1 instead of 130, from either of two causes. One is a `TypeError`: `_boom` does not take `writer`/`seed`. The other is an `AttributeError` from `config_from_cfg` on `"CFG"`: T17's `writer = store.create("fdt", cfg)` reads the run's settings off the placeholder. In that case replace the stub lines, in whatever form they have now. At HEAD they read:

```python
    def _boom(cfg, *, skip_sanity, confirm_production):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "make_fdt_config", lambda *a, **k: "CFG")
    monkeypatch.setattr(fdt_pipeline, "run_fdt", _boom)
```

Replace them with:

```python
    def _boom(*a, **k):
        raise KeyboardInterrupt

    # The interrupt lands in the builder: main's interrupt_note branch is the subject, and the note is
    # fixed text. A placeholder config cannot survive store.create("fdt", cfg), which reads the run's
    # settings off it (T7, T17).
    monkeypatch.setattr(cli, "make_fdt_config", _boom)
```

Then drop `fdt_pipeline` from that test's import line (`from core.FDT import fdt_pipeline`) if nothing else in it uses the name. Then do Step 6 as written; Step 7's prediction holds. In the report, say that this was the T17 seam.

**A6 — P26, read correctly here.** "The core model refusal carries `fdt_support`'s reason and nothing else" refers to T13's `FDTModelError` in `core/FDT`. This handler lives in `core/tool`, which may name a flag, and spec §6.2 keeps both hint branches, so Step 3 stands as written. `core/tool/fields.py`'s `FLAG["model"]` stays `"--model"`: its values are pinned to be real option strings, so no sentence may be put there.

**A7 — P46:** `crossval` has no bare refusal of its own to convert. Its pre-spend raises (`_grid`) are `UsageError`s, reported at exit 2 by design. Convert nothing there, and say so in the task report.

**A8 — P79:** `--seed` belongs to T28, which declared, forwarded and pinned it. Add nothing about it here.

#### Controller rulings (pre-flight, binding — these supersede the amendments above and the task text)

- **F20** — each handler prints `[prism fdt] writing record <id> at <dir>` (and the crossval twin, once per record) right after `store.create`, so the record is on screen when Ctrl-C lands; and both interrupt notes add: if the run was given `--store-root`, set `PRISM_ARTIFACTS` to that root for the `artifacts` commands, which read only the environment. The framing lines are prints, which `core/tool` is allowed.
- **F50** — THIS task corrects `core/tool/fdt.py`'s two "no bounds file" sentences (P76 over P39's earlier clause). Task 41 does not touch them.

---

### Task 30: `artifacts sweep` learns loose files and the legacy directory

**Why:** Spec §6.3 and decision **E10** — "the tidy-up command learns to see the legacy loose files
so the owner can clear them; nothing is deleted on their behalf". Two clearly separated categories
over Task 4's store calls: loose **files** inside a kind directory (the owner's machine has two,
`Artifacts/fdt/*.png` from a run stamped `20260915_153042`, which no command in either front end can
list or delete — `_entries` iterates directories only, `store.py:381`), and a **legacy directory**
beside the kind directories, offered by the all-kinds form alone because `sweep` takes the kind
positionally.

**Files:**
- Modify: `core/tool/browse.py:80-104` (`EPILOG`'s sweep block), `:333-397` (`_sweep`),
  `:465-466` (the sweep mode's help)
- Test: `tests/test_tool.py` — beside
  `test_artifacts_sweep_is_a_dry_run_until_yes_and_removes_only_manifest_less_directories` (`:2128`)

**Interfaces:**
- Consumes, from Task 4: `store.loose_files(kind) -> list[LooseFile]` where
  `LooseFile(name: str, size: int, mtime: float)`; `store.remove_loose(kind, filename) -> None`;
  `store.legacy_dirs() -> list[str]`; `store.remove_legacy(name) -> None`; `store.LEGACY_DIRS`.
  Each remover raises `StoreError` (a `Refusal` with `field="artifact"`) for a recently written
  target and for anything it may not reach. Also `store.sweep_incomplete(entries)` and
  `NO_MANIFEST_REASON`, unchanged.
- Produces: the tool's three preview verbs — `would remove <kind> leftover <dir>`,
  `would remove <kind> loose file <name>`, `would remove legacy directory <name>` — and the matching
  `removed …` lines. Task 31 mirrors the categories in the window but keeps its own wording.

- [ ] **Step 1: Write the failing test for loose files**

In `tests/test_tool.py`, after
`test_artifacts_sweep_is_a_dry_run_until_yes_and_removes_only_manifest_less_directories`:

```python
def test_artifacts_sweep_offers_a_loose_file_and_never_a_records_payload(browse_store, capsys):
    """E10, first category. ``_entries`` iterates DIRECTORIES only (store.py:381), so a file sitting
    directly inside a kind directory is invisible to every listing in both front ends -- and the
    owner's machine has two of them, the PNGs a pre-piece-5 fdt run left in ``Artifacts/fdt``. They
    carry no record of which cell or which settings produced them and no command could reach them.

    The complement matters as much as the category: ``loose_files`` reads the kind directory's own
    files and never descends, so a real artifact's payload -- which lives one level down, inside the
    record's folder -- is not offerable from here even in principle. The payload written below is
    what asserts that, and it must still be on disk after ``--yes``.
    """
    from tests._fixtures import backdate_tree
    root, ids = browse_store
    stray = root / "priors" / "fdt_ratio_20260915_153042.png"
    stray.write_bytes(b"a picture an older build left beside the records")
    record = next(p for p in (root / "priors").iterdir() if p.is_dir() and ids["prior"] in p.name)
    payload = record / "prior.pt"
    payload.write_bytes(b"a real artifact's payload")
    backdate_tree(root / "priors")           # the recency guard is tested on its own, below

    capsys.readouterr()
    assert main(["artifacts", "sweep", "prior"]) == 0
    out = capsys.readouterr().out
    assert f"would remove prior loose file {stray.name}" in out, out
    assert "prior.pt" not in out, "a valid record's payload was offered for removal"
    assert stray.is_file(), "a dry run removed something"

    capsys.readouterr()
    assert main(["artifacts", "sweep", "prior", "--yes"]) == 0
    out = capsys.readouterr().out
    assert f"removed prior loose file {stray.name}" in out, out
    assert not stray.exists()
    assert payload.is_file(), "the sweep reached inside a valid record's own folder"
    assert (record / "manifest.json").is_file(), "the record itself survived its neighbour's removal"
```

- [ ] **Step 2: Run it and watch it fail**

Run: `pytest tests/test_tool.py::test_artifacts_sweep_offers_a_loose_file_and_never_a_records_payload -v`

Expected: FAIL at `assert f"would remove prior loose file {stray.name}" in out` — today's `_sweep`
reads `store.list(kind)` only, which reports directories, so the stray PNG appears nowhere in the
output.

- [ ] **Step 3: Read the two new categories in `_sweep`**

In `core/tool/browse.py`, find:

```python
    from core.artifacts.store import NO_MANIFEST_REASON
    cands, kept, problems = [], [], []
    for kind in ((args.kind,) if args.kind else KINDS):
        try:
            rows = store.list(kind)                # a bad kind is a Refusal: the ladder's exit 1
        except OSError as e:                       # a kind that cannot be read is not an empty one
            problems.append(f"prism artifacts: the {kind} directory could not be read "
                            f"({type(e).__name__}: {e}), so no {kind} leftover was swept")
            continue
        for row in rows:
            if row.complete:
                continue
            (cands if row.reason == NO_MANIFEST_REASON else kept).append(
                (kind, row.dir_name, row.reason))
    reasons = {(k, d): why for k, d, why in cands}
```

Replace with:

```python
    from core.artifacts.store import NO_MANIFEST_REASON
    cands, kept, problems, loose, legacy = [], [], [], [], []
    kinds = (args.kind,) if args.kind else KINDS
    for kind in kinds:
        try:
            rows = store.list(kind)                # a bad kind is a Refusal: the ladder's exit 1
        except OSError as e:                       # a kind that cannot be read is not an empty one
            problems.append(f"prism artifacts: the {kind} directory could not be read "
                            f"({type(e).__name__}: {e}), so no {kind} leftover was swept")
            continue
        for row in rows:
            if row.complete:
                continue
            (cands if row.reason == NO_MANIFEST_REASON else kept).append(
                (kind, row.dir_name, row.reason))
    for kind in kinds:
        # E10's first category. `list` reports DIRECTORIES (store._entries iterates directories
        # only), so a FILE sitting directly inside a kind directory is invisible to every listing in
        # both front ends -- and the owner has two, left in Artifacts/fdt by a pre-piece-5 run.
        # loose_files never descends, so an artifact's own payload cannot be reached from here.
        try:
            loose += [(kind, f) for f in store.loose_files(kind)]
        except OSError as e:
            problems.append(f"prism artifacts: the {kind} directory's files could not be read "
                            f"({type(e).__name__}: {e}), so no {kind} loose file was swept")
    if args.kind is None:
        # E10's second category, and the ALL-KINDS form ALONE: `sweep` takes the kind positionally
        # and a legacy directory sits BESIDE the kind directories, under no kind at all, so there is
        # no per-kind form that could name one.
        try:
            legacy = list(store.legacy_dirs())
        except OSError as e:
            problems.append(f"prism artifacts: the store root could not be read "
                            f"({type(e).__name__}: {e}), so no legacy directory was swept")
    reasons = {(k, d): why for k, d, why in cands}
```

- [ ] **Step 4: Preview and remove them**

In `core/tool/browse.py`, find:

```python
    if not cands:
        print("[prism] nothing to sweep: every directory here carries a manifest.json.")
        return 1 if problems else 0
    if not args.yes:
        for kind, dir_name, why in cands:
            print(f"[prism] would remove {kind} leftover {dir_name} -- {why}")
        print(f"[prism] dry run: nothing was removed. Re-run with --yes to remove "
              f"{len(cands)} director{'y' if len(cands) == 1 else 'ies'}.")
```

Replace with:

```python
    if not cands and not loose and not legacy:
        print("[prism] nothing to sweep: every directory here carries a manifest.json, no loose "
              "file sits inside a kind directory, and no legacy directory sits beside them.")
        return 1 if problems else 0
    if not args.yes:
        for kind, dir_name, why in cands:
            print(f"[prism] would remove {kind} leftover {dir_name} -- {why}")
        for kind, f in loose:
            print(f"[prism] would remove {kind} loose file {f.name} -- {f.size} bytes inside the "
                  f"{kind} directory, part of no artifact")
        for name in legacy:
            print(f"[prism] would remove legacy directory {name} -- written beside the kind "
                  f"directories by an older build; nothing here reads it")
        total = len(cands) + len(loose) + len(legacy)
        print(f"[prism] dry run: nothing was removed. Re-run with --yes to remove "
              f"{total} item{'' if total == 1 else 's'}.")
```

Then find:

```python
    removed, failed = store.sweep_incomplete([(k, d) for k, d, _ in cands])
    for kind, dir_name in removed:
        why = reasons.get((kind, dir_name))
        print(f"[prism] removed {kind} leftover {dir_name}" + (f" -- {why}" if why else ""))
    for kind, dir_name, reason in failed:
        print(f"prism artifacts: could not remove {kind} leftover {dir_name}: {reason}",
              file=sys.stderr)
    return 1 if failed or problems else 0
```

Replace with:

```python
    removed, failed = store.sweep_incomplete([(k, d) for k, d, _ in cands])
    for kind, dir_name in removed:
        why = reasons.get((kind, dir_name))
        print(f"[prism] removed {kind} leftover {dir_name}" + (f" -- {why}" if why else ""))
    for kind, dir_name, reason in failed:
        print(f"prism artifacts: could not remove {kind} leftover {dir_name}: {reason}",
              file=sys.stderr)
    lost = list(failed)
    # R2 for the two new categories as well: `loose` and `legacy` were read BEFORE anything was
    # removed and THOSE LISTS are what is walked, so a file that appeared since is simply not in
    # them. Each remover carries its own refusal (the recency guard among them), which is reported
    # per item and never stops the rest -- sweep_incomplete's rule, applied by hand because these
    # two are one-at-a-time calls.
    for kind, f in loose:
        try:
            store.remove_loose(kind, f.name)
        except (Refusal, OSError) as e:            # StoreError is a Refusal, and fielded
            why = getattr(e, "message", f"{type(e).__name__}: {e}")
            lost.append((kind, f.name, why))
            print(f"prism artifacts: could not remove {kind} loose file {f.name}: {why}",
                  file=sys.stderr)
        else:
            print(f"[prism] removed {kind} loose file {f.name}")
    for name in legacy:
        try:
            store.remove_legacy(name)
        except (Refusal, OSError) as e:
            why = getattr(e, "message", f"{type(e).__name__}: {e}")
            lost.append(("", name, why))
            print(f"prism artifacts: could not remove legacy directory {name}: {why}",
                  file=sys.stderr)
        else:
            print(f"[prism] removed legacy directory {name}")
    return 1 if lost or problems else 0
```

Then make `Refusal` available at module scope — find:

```python
from core.refusals import NOTE_MAX_CHARS
```

Replace with:

```python
from core.refusals import NOTE_MAX_CHARS, Refusal
```

- [ ] **Step 5: Run it and watch it pass**

Run: `pytest tests/test_tool.py::test_artifacts_sweep_offers_a_loose_file_and_never_a_records_payload -v`

Expected: PASS

- [ ] **Step 6: Write the failing test for the legacy directory**

In `tests/test_tool.py`, after the previous test:

```python
def test_a_legacy_directory_is_offered_by_the_all_kinds_sweep_only(browse_store, capsys):
    """E10's second category, and why it has a form of its own. A legacy directory -- a `crossval/`
    an older build wrote -- sits BESIDE the kind directories, under no kind, and the store never
    walks its own root, so it is invisible to every listing. ``sweep`` takes the kind POSITIONALLY
    (core/tool/browse.py's `sweep [<kind>]`), so only the form with no kind can offer something that
    belongs to none: a per-kind sweep that removed it would be answering a question nobody asked.
    """
    from core.artifacts.store import LEGACY_DIRS
    from tests._fixtures import backdate_tree
    root, ids = browse_store
    legacy = root / LEGACY_DIRS[0]
    legacy.mkdir()
    (legacy / "fdt3d_vs_S_20260915_153042.h5").write_bytes(b"an older build's sweep output")
    backdate_tree(root)

    capsys.readouterr()
    assert main(["artifacts", "sweep", "prior"]) == 0
    assert legacy.name not in capsys.readouterr().out, \
        "a per-kind sweep offered a directory that belongs to no kind"
    assert legacy.is_dir()

    capsys.readouterr()
    assert main(["artifacts", "sweep"]) == 0
    out = capsys.readouterr().out
    assert f"would remove legacy directory {legacy.name}" in out, out
    assert legacy.is_dir(), "a dry run removed something"

    capsys.readouterr()
    assert main(["artifacts", "sweep", "--yes"]) == 0
    assert f"removed legacy directory {legacy.name}" in capsys.readouterr().out
    assert not legacy.exists()
```

- [ ] **Step 7: Run it and watch it fail**

Run: `pytest tests/test_tool.py::test_a_legacy_directory_is_offered_by_the_all_kinds_sweep_only -v`

Expected: PASS if Step 4 landed whole. Run it now: it is the pin for the all-kinds-only rule, which
one misplaced `if` would break silently. **If it fails** at
`assert f"would remove legacy directory {legacy.name}" in out`, the `if args.kind is None:` guard
from Step 3 is missing; if it fails at `assert legacy.name not in capsys.readouterr().out`, that
guard was written as a truthiness test and a per-kind sweep is offering it.

- [ ] **Step 8: Write the failing test for the recency guard**

In `tests/test_tool.py`, after the previous test:

```python
def test_a_loose_file_written_seconds_ago_is_refused_rather_than_swept(browse_store, capsys):
    """R3's guard, extended to the new categories (spec §6.3: "the recency guard applies"). A file
    inside a kind directory can be a run in flight writing its own figure as easily as it can be a
    leftover -- and a sweep from a second process has no ``BasePanel._running`` to consult. The
    removal refuses it, the tool reports it, and the exit code says the sweep did not do everything
    it offered.
    """
    from tests._fixtures import backdate_tree
    root, ids = browse_store
    backdate_tree(root / "priors")            # every leftover directory is old; only the file is new
    fresh = root / "priors" / "being_written_right_now.png"
    fresh.write_bytes(b"a run may still be writing this")

    capsys.readouterr()
    assert main(["artifacts", "sweep", "prior", "--yes"]) == 1
    cap = capsys.readouterr()
    assert fresh.is_file(), "a file written seconds ago was removed"
    assert f"could not remove prior loose file {fresh.name}" in cap.err, cap.err
```

- [ ] **Step 9: Run it and watch it pass**

Run: `pytest tests/test_tool.py::test_a_loose_file_written_seconds_ago_is_refused_rather_than_swept -v`

Expected: PASS — Task 4's `remove_loose` shares `RECENT_WRITE_SECONDS` with `remove_incomplete`, and
Step 4 routes its refusal to stderr and into `lost`. **If it fails with exit 0 and the file gone**,
`remove_loose` is not applying the guard and that is Task 4's defect, not this one's: report it
rather than adding a guard here, or the rule would live in two places.

- [ ] **Step 10: Say so in `--help`**

In `core/tool/browse.py`, find, in `EPILOG`:

```python
  sweep [<kind>] [--yes]
```

and the three lines under it ending `-- is reported and never removed`. Replace that whole block
with:

```python
  sweep [<kind>] [--yes]
                        remove what no artifact accounts for: every directory with NO manifest at
                        all, and every loose FILE sitting inside a kind directory. With no kind, all
                        eight -- and then also a LEGACY DIRECTORY beside the kind directories, which
                        is under no kind and so has no per-kind form. A DRY RUN without --yes: it
                        prints exactly what it would remove and removes nothing. A directory that
                        carries a manifest.json -- even one this build cannot read -- is reported
                        and never removed, and nothing inside a record's own folder is ever offered
```

> **Preserve Task 1's count.** Task 1 rewrote the kind counts in this epilog ("all seven" → "all
> eight", checklist item 18). Read the block as it is now and keep whatever count it says; the text
> above assumes Task 1 has landed.

Then find:

```python
    sweep = modes.add_parser("sweep", help="remove every directory with no manifest at all "
                                           "(a dry run until --yes)")
```

Replace with:

```python
    sweep = modes.add_parser("sweep", help="remove every directory with no manifest at all, and "
                                           "every loose file beside the records (a dry run until "
                                           "--yes)")
```

- [ ] **Step 11: Run the whole tool suite's artifacts group**

Run: `pytest tests/test_tool.py -v -k artifacts`
Expected: PASS — including
`test_artifacts_sweep_is_a_dry_run_until_yes_and_removes_only_manifest_less_directories`, whose
`assert "nothing to sweep" in ...` still matches the widened sentence, and
`test_a_sweep_that_could_not_remove_something_exits_1_naming_it`, whose injected
`sweep_incomplete` is untouched by the two new loops.

- [ ] **Step 12: Commit**

```bash
git add core/tool/browse.py tests/test_tool.py
git commit -m "artifacts sweep: offer loose files, and a legacy directory in the all-kinds form"
```

#### Amendments (binding — these supersede the text above)

**A1 — Step 10's EPILOG anchor, re-derived (P45).** Superseded: "find … `  sweep [<kind>] [--yes]` and the three lines under it ending `-- is reported and never removed`. Replace that whole block with:" and the block that follows. There are FOUR lines under the header. After T1 the block reads:

```
  sweep [<kind>] [--yes]
                        remove every directory with NO manifest at all; with no kind, all eight. A
                        DRY RUN without --yes: it prints what it would remove and removes nothing.
                        A directory that carries a manifest.json -- even one this build cannot read
                        -- is reported and never removed
```

Replace those five lines with:

```
  sweep [<kind>] [--yes]
                        remove what no artifact accounts for: every directory with NO manifest at
                        all, and every loose FILE sitting inside a kind directory. With no kind,
                        all eight -- and then also a LEGACY DIRECTORY beside the kind directories,
                        which is under no kind and so has no per-kind form. A DRY RUN without
                        --yes: it prints exactly what it would remove and removes nothing. A
                        directory that carries a manifest.json -- even one this build cannot read
                        -- is reported and never removed, and nothing inside a record's own folder
                        is ever offered
```

Why this reflow: T1's `test_the_rendered_help_counts_the_kinds_correctly` scans the rendered help with `re.findall(r"\ball (\w+)", text)`, which needs a literal space after "all". The drafted text broke "all" and "eight" across a line, so that pin would silently stop checking this block's count. Keep "all eight" on one line. If the count reads differently when you arrive, keep the file's count and still keep it on one line. Run `pytest tests/test_tool.py::test_the_rendered_help_counts_the_kinds_correctly -v` in Step 11 as well; expect PASS.

**A2 — Interfaces:** `LEGACY_DIRS` is a MODULE constant, `core.artifacts.store.LEGACY_DIRS` (re-exported as `core.artifacts.LEGACY_DIRS`), not an attribute of the store object. The tests already import it that way. `core/tool/browse.py` does not need it: it calls `store.legacy_dirs()`.

**A3 — Not this task's (P39):** T41 corrects the module docstring's `NO ``--store-root``, EITHER.` paragraph (browse.py lines 18–24). Leave it alone.

No other ruling (P1–P82) overrides this task.

#### Controller rulings (pre-flight, binding — these supersede the amendments above and the task text)

- **F7** — the tool half: add to this task's test that `python -m core artifacts list fdt` shows an unfinished record as NOT finished (its `finished` cell reads `no`) and that `artifacts sweep fdt` never offers it.

---

### Task 31: the Artifacts screen's sweep learns the same two categories

**Why:** Spec §6.3 — "The Artifacts screen's sweep gains the same category, so the two front ends
stay in step, which is the property piece 4 established." The window must be able to clear what the
tool can clear, and by the same rules: the preview names each item, the removal is bound to the list
the confirmation showed (R2), the recency guard applies (R3), the removal is guarded so nothing
escapes a click (R5), and a valid record's payload files are never offered.

**Files:**
- Modify: `core/gui/screens/artifact_screen.py:657-688` (`_incomplete`, and a sibling beside it),
  `:690-755` (`_sweep`)
- Test: `tests/test_artifact_browser.py` — beside `test_sweep_removes_only_the_leftovers_and_reports_what_it_could_not` (`:612`)

**Interfaces:**
- Consumes, from Task 4: `store.loose_files(kind) -> list[LooseFile]` (`LooseFile.name`,
  `.size`, `.mtime`), `store.remove_loose(kind, filename)`, `store.legacy_dirs() -> list[str]`,
  `store.remove_legacy(name)` — each remover raising `StoreError` (a `Refusal`). From Task 30:
  the shape of the two categories and the all-kinds-only rule for the legacy directory.
- Produces: `ArtifactScreen._loose_and_legacy(store, kind) -> (loose, legacy, problems)` where
  `loose` is `[(kind, LooseFile)]`, `legacy` is `[str]` and `problems` is `[str]`. `_sweep`'s
  signature, its `all_kinds` keyword and its status-line prefixes (`Removed N of M leftover
  director…`, `Nothing to remove: …`) are unchanged, so the existing sweep tests keep their anchors.

- [ ] **Step 1: Write the failing test for loose files in the window**

In `tests/test_artifact_browser.py`, after
`test_sweep_removes_only_the_leftovers_and_reports_what_it_could_not`:

```python
def test_the_screens_sweep_offers_a_loose_file_and_never_a_records_payload(store, monkeypatch):
    """E10 through the window (spec §6.3): the two front ends stay in step, which is the property
    piece 4 established -- the owner must not have to open a terminal to clear something the tool
    can clear. The preview NAMES the file, as it names every leftover directory, because "Remove 3
    items?" with no list is not a confirmation.

    The payload written inside the real artifact's own folder is the complement: ``loose_files``
    reads the kind directory's files and never descends, so nothing a record owns can be offered
    here even in principle -- the same shape ``remove_incomplete`` has against ``delete``.
    """
    from PySide6.QtWidgets import QMessageBox
    from tests._fixtures import SHOWN, artifact_screen, qt_app
    qt_app()
    cfg = _nad_cfg()
    p = _prior_artifact(store, cfg, name="keeper")
    payload = store.path("prior", p.id) / "prior.pt"
    payload.write_bytes(b"a real artifact's payload")
    stray = store.kind_dir("prior") / "fdt_ratio_20260915_153042.png"
    stray.write_bytes(b"a picture an older build left beside the records")
    backdate_tree(store.kind_dir("prior"))

    scr = artifact_screen(store)
    _show_kind(scr, "prior")
    scr._sweep(all_kinds=False)                      # the session guard answers No
    box = SHOWN[-1]
    assert stray.name in box.informativeText(), box.informativeText()
    assert "prior.pt" not in box.informativeText(), "a record's payload was offered"
    assert stray.is_file() and "Nothing was removed." in scr.status.text(), scr.status.text()

    _answer(monkeypatch, QMessageBox.Yes)
    scr._sweep(all_kinds=False)
    assert not stray.exists(), "the file the operator confirmed is still there"
    assert payload.is_file(), "the sweep reached inside a valid record's own folder"
    assert store.get("prior", p.id).name == "keeper"
    assert "Removed 1 of 1 loose file" in scr.status.text(), scr.status.text()
```

- [ ] **Step 2: Run it and watch it fail**

Run: `pytest tests/test_artifact_browser.py::test_the_screens_sweep_offers_a_loose_file_and_never_a_records_payload -v`

Expected: FAIL — with no leftover directory present, `_incomplete` returns no candidates, so `_sweep`
takes its `if not cands:` branch, shows no dialog at all, and `SHOWN[-1]` raises `IndexError: list
index out of range` (or, if an earlier box is in `SHOWN`, fails at
`assert stray.name in box.informativeText()`).

- [ ] **Step 3: Read the two new categories**

In `core/gui/screens/artifact_screen.py`, find the end of `_incomplete`:

```python
            for row in rows:
                if row.complete:
                    continue
                target = out if row.reason == NO_MANIFEST_REASON else kept
                target.append((k, row.dir_name, row.reason))
        return out, kept, problems
```

Replace with:

```python
            for row in rows:
                if row.complete:
                    continue
                target = out if row.reason == NO_MANIFEST_REASON else kept
                target.append((k, row.dir_name, row.reason))
        return out, kept, problems

    def _loose_and_legacy(self, store, kind) -> tuple:
        """``(loose, legacy, problems)``: E10's two categories, beside ``_incomplete``'s directories.

        ``loose`` is ``[(kind, LooseFile)]`` -- a FILE sitting directly inside a kind directory,
        which no artifact accounts for. ``_entries`` iterates directories only (store.py's
        ``for sub in ... if p.is_dir()``), so such a file is invisible to every listing and no front
        end could see or clear one before piece 5; the owner's machine has two, left in
        ``Artifacts/fdt`` by a run stamped 20260915_153042. ``loose_files`` reads the kind
        directory's own files and NEVER DESCENDS, so nothing inside a record's folder is reachable
        from here -- the same complement rule ``remove_incomplete`` has against ``delete``.

        ``legacy`` is ``[name]`` and is filled for the ALL-KINDS sweep alone: a legacy directory
        sits BESIDE the kind directories, under no kind, so the per-kind button has nothing to say
        about one. ``kind is None`` means all kinds, EXPLICITLY, exactly as ``_incomplete`` reads it.

        Both reads are guarded per kind: an unreadable directory is reported and does not stop the
        others (§3.2 -- an unreadable kind is not an empty one).
        """
        loose, legacy, problems = [], [], []
        for k in (list(KIND_DIRS) if kind is None else [kind]):
            try:
                loose += [(k, f) for f in store.loose_files(k)]
            except Exception as e:      # noqa: BLE001 -- an unreadable kind is reported, not fatal
                problems.append(f"The files in the {k} directory could not be read "
                                f"({type(e).__name__}: {e}), so no {k} loose file can be swept; "
                                f"check that folder's permissions on disk and sweep again.")
        if kind is None:
            try:
                legacy = list(store.legacy_dirs())
            except Exception as e:      # noqa: BLE001 -- reported, never swallowed
                problems.append(f"The store root could not be read ({type(e).__name__}: {e}), so "
                                f"no legacy directory can be swept; check its permissions on disk "
                                f"and sweep again.")
        return loose, legacy, problems
```

- [ ] **Step 4: Offer them in the confirmation**

In `core/gui/screens/artifact_screen.py`, find:

```python
        kind = None if all_kinds else self.kind()
        cands, kept, problems = self._incomplete(store, kind)
        tail = (" " + " ".join(problems)) if problems else ""
```

Replace with:

```python
        kind = None if all_kinds else self.kind()
        cands, kept, problems = self._incomplete(store, kind)
        loose, legacy, more = self._loose_and_legacy(store, kind)
        problems = problems + more
        tail = (" " + " ".join(problems)) if problems else ""
```

Then find:

```python
        if not cands:
            where = "any kind's" if kind is None else f"{kind}"
            self._set_status(f"Nothing to remove: every {where} directory carries a manifest.json."
                             + tail, error=bool(problems))
            return
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("Remove leftover directories")
        box.setText(f"Remove {len(cands)} director{'y' if len(cands) == 1 else 'ies'} with no "
                    f"manifest at all?")
        box.setInformativeText("\n".join(f"{k}/{d} — {why}" for k, d, why in cands)
                               + "\n\nA directory that carries a manifest.json of any kind is never "
                                 "touched by this. Nor is one that is still being written: an "
                                 "artifact's manifest is written last, so a run in flight — in "
                                 "another window or at a terminal — looks exactly like a leftover "
                                 "until it commits, and a recently written directory is refused "
                                 "rather than removed.")
```

Replace with:

```python
        if not cands and not loose and not legacy:
            where = "any kind's" if kind is None else f"{kind}"
            self._set_status(f"Nothing to remove: every {where} directory carries a manifest.json, "
                             f"and no loose file or legacy directory is here either." + tail,
                             error=bool(problems))
            return
        total = len(cands) + len(loose) + len(legacy)
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("Remove what no artifact accounts for")
        box.setText(f"Remove {total} item{'' if total == 1 else 's'} that no artifact accounts for?")
        box.setInformativeText(
            "\n".join([*(f"{k}/{d} — {why}" for k, d, why in cands),
                       *(f"{k}/{f.name} — a loose file inside the {k} directory, {f.size} bytes, "
                         f"part of no artifact" for k, f in loose),
                       *(f"{name}/ — a directory an older build wrote beside the kind directories"
                         for name in legacy)])
            + "\n\nA directory that carries a manifest.json of any kind is never touched by this. "
              "Nothing inside a record's own folder is offered either. Nor is anything that is "
              "still being written: an artifact's manifest is written last, so a run in flight — "
              "in another window or at a terminal — looks exactly like a leftover until it "
              "commits, and a recently written directory or file is refused rather than removed.")
```

- [ ] **Step 5: Remove them under the same guard**

In `core/gui/screens/artifact_screen.py`, find:

```python
        try:
            removed, failed = store.sweep_incomplete([(k, d) for k, d, _ in cands])
        except Exception as e:                  # noqa: BLE001 -- reported, never raised out of a click
            self._after_change(f"The sweep stopped part-way: {type(e).__name__}: {e}. Refresh to see "
                               f"what is left." + tail, error=True)
            return
        said = f"Removed {len(removed)} of {len(cands)} leftover director" \
               f"{'y' if len(cands) == 1 else 'ies'}."
        for k, d, why in failed:
            said += f" {k}/{d} could not be removed: {why}."
        self._after_change(said + tail, error=bool(failed or problems))
```

Replace with:

```python
        files_gone, files_failed, legacy_gone, legacy_failed = [], [], [], []
        try:
            removed, failed = store.sweep_incomplete([(k, d) for k, d, _ in cands])
            # R2 for the two new categories too: `loose` and `legacy` were read BEFORE the dialog
            # and THOSE LISTS are walked, so anything that appeared while it sat open is simply not
            # in them. One call per item, because neither remover has a batch form -- each carries
            # its own refusal (the recency guard among them), which is reported and never stops the
            # rest, exactly as sweep_incomplete does for directories.
            for k, f in loose:
                try:
                    store.remove_loose(k, f.name)
                except (Refusal, OSError) as e:        # StoreError is a Refusal, and fielded
                    files_failed.append((f"{k}/{f.name}",
                                         getattr(e, "message", f"{type(e).__name__}: {e}")))
                else:
                    files_gone.append(f.name)
            for name in legacy:
                try:
                    store.remove_legacy(name)
                except (Refusal, OSError) as e:
                    legacy_failed.append((name, getattr(e, "message", f"{type(e).__name__}: {e}")))
                else:
                    legacy_gone.append(name)
        except Exception as e:                  # noqa: BLE001 -- reported, never raised out of a click
            self._after_change(f"The sweep stopped part-way: {type(e).__name__}: {e}. Refresh to see "
                               f"what is left." + tail, error=True)
            return
        said = f"Removed {len(removed)} of {len(cands)} leftover director" \
               f"{'y' if len(cands) == 1 else 'ies'}."
        for k, d, why in failed:
            said += f" {k}/{d} could not be removed: {why}."
        if loose:
            said += (f" Removed {len(files_gone)} of {len(loose)} loose "
                     f"file{'' if len(loose) == 1 else 's'}.")
            for what, why in files_failed:
                said += f" {what} could not be removed: {why}."
        if legacy:
            said += (f" Removed {len(legacy_gone)} of {len(legacy)} legacy "
                     f"director{'y' if len(legacy) == 1 else 'ies'}.")
            for what, why in legacy_failed:
                said += f" {what} could not be removed: {why}."
        self._after_change(said + tail,
                           error=bool(failed or files_failed or legacy_failed or problems))
```

- [ ] **Step 6: Run the new test and watch it pass**

Run: `pytest tests/test_artifact_browser.py::test_the_screens_sweep_offers_a_loose_file_and_never_a_records_payload -v`

Expected: PASS

- [ ] **Step 7: Write the failing test for the legacy directory in the window**

In `tests/test_artifact_browser.py`, after the previous test:

```python
def test_the_screens_legacy_directory_is_offered_by_sweep_all_kinds_only(store, monkeypatch):
    """The window and the tool answer the same question the same way (spec §6.3). A legacy directory
    belongs to no kind, so "Sweep this kind…" has nothing to say about one -- offering it there
    would let a sweep of Priors remove something that has nothing to do with priors. "Sweep all
    kinds…" is the form that covers the whole root, and it is the only one that offers it.
    """
    from PySide6.QtWidgets import QMessageBox
    from core.artifacts.store import LEGACY_DIRS
    from tests._fixtures import SHOWN, artifact_screen, qt_app
    qt_app()
    legacy = store.root / LEGACY_DIRS[0]
    legacy.mkdir(parents=True, exist_ok=True)
    (legacy / "fdt3d_vs_S_20260915_153042.h5").write_bytes(b"an older build's sweep output")
    backdate_tree(store.root)

    scr = artifact_screen(store)
    _show_kind(scr, "prior")
    _answer(monkeypatch, QMessageBox.Yes)
    scr._sweep(all_kinds=False)
    assert legacy.is_dir(), "a per-kind sweep removed a directory that belongs to no kind"
    assert "Nothing to remove" in scr.status.text(), scr.status.text()

    SHOWN.clear()
    scr._sweep(all_kinds=True)
    assert legacy.name in SHOWN[-1].informativeText(), SHOWN[-1].informativeText()
    assert not legacy.exists(), "the directory the operator confirmed is still there"
    assert "Removed 1 of 1 legacy director" in scr.status.text(), scr.status.text()
```

- [ ] **Step 8: Run it and watch it fail**

Run: `pytest tests/test_artifact_browser.py::test_the_screens_legacy_directory_is_offered_by_sweep_all_kinds_only -v`

Expected: PASS if Step 3's `if kind is None:` guard landed as written. Run it now: it is the pin for
the all-kinds-only rule, which one misplaced line would break silently. **If it fails** at
`assert legacy.is_dir()`, the guard was written as a truthiness test on `kind` and a per-kind sweep
is offering a directory under no kind.

- [ ] **Step 9: Run the whole browser suite**

Run: `pytest tests/test_artifact_browser.py -v`

Expected: PASS. The four existing sweep tests keep their anchors on purpose: the informative text
still contains `"A directory that carries a manifest.json of any kind is never touched by this."`
word for word and `"still being written"`; the status line still begins
`"Removed N of M leftover director…"` and `"Nothing to remove: "`; and
`test_a_sweep_that_throws_is_reported_on_the_status_line_and_the_pickers_are_told`'s injected
`sweep_incomplete` still raises inside the same `try`.

- [ ] **Step 10: Commit**

```bash
git add core/gui/screens/artifact_screen.py tests/test_artifact_browser.py
git commit -m "Artifacts screen: sweep offers loose files, and legacy directories in all-kinds"
```
# Tasks 32–35: the gaps and the round trips

These four close coverage gaps rather than build capability. T32 and T33 add tests to paths that
work today but that no test reaches; T34 rewrites the one slow test that piece 5 breaks by design;
T35 adds two round trips with a deliberately bounded repair mandate.

**Read before starting any of them:** the Global Constraints and the Interface Contract at the head
of `docs/superpowers/plans/2026-09-22-secondary-analyses.md`. Two of its rules bite hardest here:
a task's implementer runs **focused tests in the foreground only** and never starts a background
gate run, and **no source file is edited while a suite is running**.

#### Amendments (binding — these supersede the text above)

**A1 — One existing test pins the old dialog title; update it in this task (new Step 8a, before Step 9).** Step 4 retitles the confirmation from "Remove leftover directories" to "Remove what no artifact accounts for". Step 9's claim that the existing sweep tests keep their anchors is therefore false for `test_a_sweep_that_throws_is_reported_on_the_status_line_and_the_pickers_are_told`. In `tests/test_artifact_browser.py`, find:

```python
    assert SHOWN[-1].windowTitle() == "Remove leftover directories", \
        "the confirmation is the only box: a disk problem is not a refusal"
```

Replace with:

```python
    assert SHOWN[-1].windowTitle() == "Remove what no artifact accounts for", \
        "the confirmation is the only box: a disk problem is not a refusal"
```

That assertion's subject is that the confirmation is the only box shown; the title is simply the one Step 4 now sets. After this, Step 9's expected PASS holds. The file is already in the Files block and on the `git add` line.

No ruling (P1–P82) overrides anything else in this task. The interface contract's `loose_files` / `remove_loose` / `legacy_dirs` / `remove_legacy` / `LEGACY_DIRS` are used exactly as T4 defines them.

#### Controller rulings (pre-flight, binding — these supersede the amendments above and the task text)

- **F51** — emit the "Removed N of M leftover director(y/ies)" line only when there were directory candidates; a sweep that removed only loose files or a legacy directory reports those alone.

---

### Task 32: the other three `core/FDT/plots.py` drawing functions get `plot_psd`'s test

**Why:** Spec §8.2, "The drawing functions": the three functions that lack it gain what `plot_psd`
has — the figure is closed, the interactive path is not taken, and no "non-interactive" warning is
emitted. `docs/STATE.md:235-236` hands this on as a piece-5 gap in the words "only `plot_psd` of the
four `core/FDT/plots.py` functions is unit-tested for closing its figure".

**This task is TESTS ONLY, and its implementer must know that before step 1.** Commit `bb22ac4`
("FDT plots: close a saved figure instead of `plt.show()` under Agg") gave **all four** functions the
close-when-saved branch; `git log -S "plt.close(fig)" -- core/FDT/plots.py` returns that one commit.
What is missing is three tests, not three code changes. The three new tests therefore pass the
moment they are written, and step 4 is what gives them teeth: it breaks one function on purpose,
watches the new test go red, and restores the file. Do not "fix" `plots.py` — there is nothing
there to fix, and a task report that claims otherwise is wrong.

**Files:**
- Modify (append three tests + one helper): `tests/test_tool.py:1381-1405` — after
  `def test_fdt_plot_functions_close_a_saved_figure_instead_of_show(tmp_path):`, whose last line is
  `    assert len(plt.get_fignums()) == before`
- Read only, never edited by this task: `core/FDT/plots.py:86-90`, `:128-132`, `:188-192`, `:244-248`

**Interfaces:**
- Consumes: nothing from any earlier task. `core/FDT/plots.plot_eff_temp_ratio(omegas, ratio,
  save_path=None, title=…, omega_natural=None, linthresh=1.0)`,
  `plot_spontaneous_trajectory(t, x_mean, save_path=None, title=…, burn_in=None)` and
  `plot_chi_components(omegas, chis, save_path=None, title=…, omega_natural=None)` are all unchanged
  by this piece.
- Produces: nothing any later task consumes.

- [ ] **Step 1: Write the helper and the first of the three tests**

In `tests/test_tool.py`, find the end of the existing sibling:

```python
    assert out.exists()
    assert not any("non-interactive" in str(w.message) for w in rec), \
        [str(w.message) for w in rec]
    assert len(plt.get_fignums()) == before
```

Append immediately after it:

```python
def _figure_is_saved_closed_and_silent(draw, out, monkeypatch):
    """The three properties the test above asserts for ``plot_psd``, as one helper the other three
    drawing functions reuse: the file is written, ``plt.show`` is never reached, no "non-interactive"
    warning is emitted, and the figure the call drew is closed again.

    ``plt.show`` is spied on rather than inferred from the absence of the warning: under a backend
    that IS interactive (a display-marked run, or a future matplotlib) the warning would simply not
    appear and the interactive half of the assertion would silently stop testing anything.
    """
    import warnings

    from matplotlib import pyplot as plt

    shown = []
    monkeypatch.setattr(plt, "show", lambda *a, **k: shown.append(True))
    before = len(plt.get_fignums())
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        draw(out)
    assert out.exists(), "save_path was given and nothing was written"
    assert shown == [], "plt.show() was called although save_path was given"
    assert not any("non-interactive" in str(w.message) for w in rec), \
        [str(w.message) for w in rec]
    assert len(plt.get_fignums()) == before, "the saved figure was left open"


def test_plot_eff_temp_ratio_closes_a_saved_figure_instead_of_show(tmp_path, monkeypatch):
    """The headline T_eff/T figure. ``bb22ac4`` gave all four of core/FDT/plots.py's drawing
    functions the close-when-saved branch and only ``plot_psd`` got a test -- the gap
    ``docs/STATE.md``'s piece-5 row hands on and spec section 8.2 closes.

    This one matters most of the four: a real ``fdt`` run draws it once at the end AND the passive
    sanity check draws it again (core/FDT/sanity.py, ``plot_eff_temp_ratio(... save_path=
    save_plot_path ...)``), so a regression leaks two live figures per run for the life of the
    process and prints "FigureCanvasAgg is non-interactive" twice at the operator.
    """
    import numpy as np

    from core.FDT.plots import plot_eff_temp_ratio

    _figure_is_saved_closed_and_silent(
        lambda out: plot_eff_temp_ratio(np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.2, 0.9]),
                                        save_path=out, omega_natural=2.0),
        tmp_path / "ratio.png", monkeypatch)
```

The ratio array is deliberately all-finite: a non-finite entry would make the function emit its own
`log.warning` about dropped points, which `test_the_fdt_messages_are_records_with_their_own_levels`
in `tests/test_fdt_user.py` already owns.

- [ ] **Step 2: Run it and watch it pass**

Run: `pytest tests/test_tool.py::test_plot_eff_temp_ratio_closes_a_saved_figure_instead_of_show -v`
Expected: **PASS**. This is a coverage gap, not a defect — see the note at the head of this task.
If it FAILS, stop and report: `plots.py` would then differ from what this plan read at `a0d85da`.

- [ ] **Step 3: Write the other two tests**

Append after the test from step 1:

```python
def test_plot_spontaneous_trajectory_closes_a_saved_figure_instead_of_show(tmp_path, monkeypatch):
    """The Campaign-1 diagnostic trace. The only one of the four with no non-finite filter of its
    own, so it is the one whose save branch is reached on every input -- and the only figure a run
    that is cancelled between the two campaigns will have drawn.
    """
    import numpy as np

    from core.FDT.plots import plot_spontaneous_trajectory

    t = np.linspace(0.0, 10.0, 64)
    _figure_is_saved_closed_and_silent(
        lambda out: plot_spontaneous_trajectory(t, np.sin(t) * 0.1, save_path=out, burn_in=2.0),
        tmp_path / "traj.png", monkeypatch)


def test_plot_chi_components_closes_a_saved_figure_instead_of_show(tmp_path, monkeypatch):
    """The two-panel susceptibility figure. It is the one function that builds a MULTI-axes figure
    (``plt.subplots(2, 1, ...)``), so it is the one where "close the figure" and "close the axes"
    could plausibly come apart: ``plt.get_fignums()`` counts figures, and a two-panel figure left
    open counts once just like a one-panel one.
    """
    import numpy as np

    from core.FDT.plots import plot_chi_components

    chis = np.array([1.0 + 1.0j, 2.0 + 0.5j, 1.5 - 0.2j])
    _figure_is_saved_closed_and_silent(
        lambda out: plot_chi_components(np.array([1.0, 2.0, 3.0]), chis,
                                        save_path=out, omega_natural=2.0),
        tmp_path / "chi.png", monkeypatch)
```

- [ ] **Step 4: Prove the three tests have teeth, then restore the file**

Spec §8.2 asks for checks "shown failing before the change with the behaviour it prevents". There is
no change here, so show the failure against a deliberate break instead.

In `core/FDT/plots.py`, find (inside `plot_chi_components`):

```python
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()
```

Temporarily replace with:

```python
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()
```

Run: `pytest tests/test_tool.py::test_plot_chi_components_closes_a_saved_figure_instead_of_show -v`
Expected: FAIL with `AssertionError: the saved figure was left open`.

Then restore the file exactly — `git checkout -- core/FDT/plots.py` — and confirm with
`git status --short core/FDT/plots.py`, which must print nothing. **`core/FDT/plots.py` is not part
of this task's commit.**

- [ ] **Step 5: Run all four figure tests together and watch them pass**

Run: `pytest tests/test_tool.py -v -k "closes_a_saved_figure or plot_functions_close"`
Expected: PASS, 4 tests.

- [ ] **Step 6: Commit**

```bash
git add tests/test_tool.py
git commit -m "test: the other three FDT plot functions close their saved figure too"
```

#### Amendments (binding — these supersede the text above)

**Rulings that bear on this task.** P17 says this is a tests-only task: step 2 predicts PASS, step 4 proves the tests have teeth and restores `core/FDT/plots.py`, and `plots.py` is never committed. The body already follows it, so no step changes. P48 confirms that no drawing function is rewritten in this piece. P22 (landed by T16) moves `plot_psd` ahead of Campaign 2, which falsifies one docstring sentence below. No other ruling touches this task. Every anchor quoted above was checked against HEAD `673868d` and found verbatim.

**A1. Step 1: the `plot_eff_temp_ratio` test's docstring.** Its second paragraph overstates when the figure is drawn twice: `run_all_sanity` keeps the passive-baseline check only for NADROWSKI. Replace this paragraph:

```python
    This one matters most of the four: a real ``fdt`` run draws it once at the end AND the passive
    sanity check draws it again (core/FDT/sanity.py, ``plot_eff_temp_ratio(... save_path=
    save_plot_path ...)``), so a regression leaks two live figures per run for the life of the
    process and prints "FigureCanvasAgg is non-interactive" twice at the operator.
```

with:

```python
    This one matters most of the four: a real ``fdt`` run draws it once at the end, and on a
    NADROWSKI cell with the sanity checks on the passive-baseline check draws it again
    (core/FDT/sanity.py, ``plot_eff_temp_ratio(... save_path=save_plot_path ...)``) -- so a
    regression here leaks two live figures per such run for the life of the process.
```

**A2. Step 3: the `plot_spontaneous_trajectory` test's docstring.** Two claims in it are false. `plot_chi_components` has no non-finite filter either. And after P22, a run cancelled between the campaigns has also drawn the spontaneous PSD. Replace:

```python
    """The Campaign-1 diagnostic trace. The only one of the four with no non-finite filter of its
    own, so it is the one whose save branch is reached on every input -- and the only figure a run
    that is cancelled between the two campaigns will have drawn.
    """
```

with:

```python
    """The Campaign-1 diagnostic trace. Like ``plot_chi_components`` it has no non-finite filter of
    its own, so its save branch is reached on every input; and it is one of the two figures (with the
    spontaneous PSD, which piece 5 moved ahead of Campaign 2) that a run cancelled between the two
    campaigns has already drawn into its record.
    """
```

Nothing else changes: the helper, the three test bodies, the step-4 break-and-restore and the commit stay as written.

---

### Task 33: the Nadrowski sanity path, end to end

**Why:** Spec §1's last bullet — "The Nadrowski sanity path is reached by no test" — and §1.2's
ruling that "the main-cell-type gap closed here is the Nadrowski SANITY path", not the normalisation
branch. Spec §8.2, "The Nadrowski sanity path", asks for a `slow`-marked end-to-end run on a
Nadrowski cell **with** the sanity checks, covering the Nadrowski-only checks and the
passive-baseline plot they draw.

**What is already covered, and must not be claimed here.** The Nadrowski *normalisation* branch is
covered: `tests/test_fdt_user.py:53` asserts `observable_noise_prefactor` directly
(`assert observable_noise_prefactor(_FakeCfg("NADROWSKI", {"n": 50.0, "beta": 14.1})) == 50.0 * 14.1`),
and the sweep leg of the slow tool test runs a real Nadrowski cell. What no test reaches is
`core/FDT/sanity.py`'s `check_passive_baseline` and `check_high_freq_fdt`, and the passive-baseline
figure the first of them saves — because `tests/test_tool.py`'s single-cell leg runs
`Resources/Cells/hopf/cell.txt` with `--skip-sanity`.

**Files:**
- Modify (append two tests): `tests/test_tool.py:1344-1378` — immediately after the slow test
  `def test_fdt_and_crossval_run_at_tiny_size(tool_env, capsys):`
- Read only: `core/FDT/sanity.py:292-307` (the `checks` list and `_NADROWSKI_ONLY`),
  `core/FDT/fdt_pipeline.py` (the sanity branch)

**Interfaces:**
- Consumes (from T17): `run_fdt(cfg, *, skip_sanity, confirm_production, writer, seed=None) ->
  LoadedFdt`; the `fdt` kind in `store.KIND_DIRS`; `ArtifactStore.load_fdt(ref) -> LoadedFdt` with
  `.body`, `.manifest`, `.path`, `.data_path`; `Summary.study` and `Summary.finished`.
- Consumes (from T28): `--store-root` on the `fdt` subcommand, honoured as the store's root.
- Produces: nothing any later task consumes.

- [ ] **Step 1: Write the fast selection test**

This one runs in the fast gate and costs nothing: it stubs all five check bodies and asserts only
which of them `run_all_sanity` chooses.

In `tests/test_tool.py`, find the last line of `test_fdt_and_crossval_run_at_tiny_size`:

```python
    assert "[prism crossval] S sweep:" in capsys.readouterr().out
```

(Task 34 rewrites that test; if it has already run, find its new last line instead and append after
the whole function.) Append:

```python
def test_the_nadrowski_only_sanity_checks_are_selected_for_a_nadrowski_cell(monkeypatch):
    """``run_all_sanity`` keeps ``passive_baseline`` and ``high_freq_fdt`` for NADROWSKI and drops
    them for every other model (core/FDT/sanity.py, ``_NADROWSKI_ONLY``), because both reason about
    parameters -- the s-feedback and the motor thermostat -- that only Nadrowski has. Until piece 5
    the only end-to-end test of the ``fdt`` subcommand ran a HOPF cell with ``--skip-sanity``, so the
    branch that KEEPS the two was exercised nowhere and neither was the note that announces dropping
    them. Spec section 1's last bullet.

    The five check bodies are stubbed, so this asserts the SELECTION and the note, not the physics;
    the slow test below runs them for real.
    """
    from core.FDT import sanity

    seen = []

    def _stub(name):
        def _fn(cfg, **kw):
            seen.append(name)
            return True, {"stub": name}
        return _fn

    for fn_name in ("check_passive_baseline", "check_high_freq_fdt", "check_linearity",
                    "check_ensemble_convergence", "check_psd_window"):
        monkeypatch.setattr(sanity, fn_name, _stub(fn_name))

    class _Cfg:
        def __init__(self, model):
            self.model = model

    nadrowski = sanity.run_all_sanity(_Cfg("NADROWSKI"))
    assert list(nadrowski) == ["passive_baseline", "high_freq_fdt", "linearity",
                               "ensemble_convergence", "psd_window"], list(nadrowski)
    assert seen[:2] == ["check_passive_baseline", "check_high_freq_fdt"], seen

    seen.clear()
    hopf = sanity.run_all_sanity(_Cfg("HOPF"))
    assert list(hopf) == ["linearity", "ensemble_convergence", "psd_window"], list(hopf)
    assert "check_passive_baseline" not in seen, seen
```

- [ ] **Step 2: Run it and watch it pass**

Run: `pytest tests/test_tool.py::test_the_nadrowski_only_sanity_checks_are_selected_for_a_nadrowski_cell -v`
Expected: **PASS** — the selection branch exists today; what was missing is this test.

If it FAILS with `KeyError` or a different key list, the check list has moved since `a0d85da`;
re-read `core/FDT/sanity.py:292-307` and say so in the task report.

- [ ] **Step 3: Prove step 1's test has teeth, then restore the file**

In `core/FDT/sanity.py`, find:

```python
    _NADROWSKI_ONLY = {"passive_baseline", "high_freq_fdt"}
    if cfg.model.lower() != "nadrowski":
```

Temporarily replace the second line with `if True:`. Run the step-2 command again.
Expected: FAIL with `AssertionError: ['linearity', 'ensemble_convergence', 'psd_window']`.

Restore with `git checkout -- core/FDT/sanity.py` and confirm `git status --short core/FDT/sanity.py`
prints nothing.

- [ ] **Step 4: Write the slow end-to-end test**

Append after step 1's test:

```python
@pytest.mark.slow
def test_fdt_runs_the_nadrowski_sanity_checks_end_to_end(tool_env, tmp_path, capsys):
    """The real sanity checks on a real Nadrowski cell, and the passive-baseline figure they draw.

    This is the gap spec section 1's last bullet names: the single-cell leg above runs a HOPF cell
    with --skip-sanity, so ``check_passive_baseline`` and ``check_high_freq_fdt`` -- the two checks
    that decide whether the whole PSD / lock-in / noise-prefactor convention is right, and the only
    ones that draw a figure of their own -- had never executed under test. Everything is real: the
    campaigns, the checks, the production sweep and the record.

    Production is NOT skipped. ``--no-production`` would be cheaper, but the value ``run_fdt``
    returns and the state of its record on the "Aborted by user" branch are not fixed by the spec
    (fdt_pipeline.py's early ``return`` after the sanity checks), so this test stays on the branch
    whose contract IS fixed: a finished record.

    Its own --store-root, so the record cannot be confused with the one the tiny-size test above
    writes into the module's shared artifacts root. ``tool_env`` is still requested: it pins
    PRISM_ARTIFACTS at a temp root, so any write that escapes the store still misses the real
    ``Artifacts/``.

    Measured <DATE>: <N> s on the CPU -- fill this in from step 6's --durations output, and do not
    copy a number from anywhere else.
    """
    from core import config
    from core.artifacts import ArtifactStore
    from core.tool import main

    root = tmp_path / "store"
    cell = str(config.CELL_PATH / "nadrowski" / "master_spont.txt")
    capsys.readouterr()
    assert main(["fdt", "--cell", cell, "--n-freqs", "2", "--ensemble-m", "8",
                 "--store-root", str(root)]) == 0
    out = capsys.readouterr().out

    # The two Nadrowski-only checks ran, and the note that announces dropping them did not appear.
    assert "[passive_baseline] true equilibrium (s=0): T_eff/T ~ 1" in out, out[-2000:]
    assert "[high_freq_fdt] high omega (s != 0): T_eff/T ~ 1" in out, out[-2000:]
    assert "Nadrowski-specific and are skipped" not in out
    assert "[PASS] passive_baseline" in out or "[FAIL] passive_baseline" in out, out[-2000:]

    rows = [s for s in ArtifactStore(root).list("fdt") if s.study == "single"]
    assert len(rows) == 1 and rows[0].finished, rows
    rec = ArtifactStore(root).load_fdt(rows[0].id)
    assert rec.body["settings"]["skip_sanity"] is False, rec.body["settings"]
    assert rec.body["complete"] is True
    # FIVE figures, not four: the passive-baseline plot is the one --skip-sanity never draws
    # (core/FDT/sanity.py's check_passive_baseline, ``save_plot_path``).
    assert len(rec.manifest.figures) == 5, rec.manifest.figures
    for fig in rec.manifest.figures:
        assert (rec.path / fig).exists(), fig
```

The verdict assertion allows either `[PASS]` or `[FAIL]`: whether a real cell at
`--ensemble-m 8` passes the passive-baseline thresholds is a question about the physics at a size
chosen for speed, and this test is about the path, not the verdict. A failed check is a
`log.warning` from `fdt_pipeline`, not a non-zero exit.

- [ ] **Step 5: Run the slow test and watch it pass**

Run: `pytest tests/test_tool.py::test_fdt_runs_the_nadrowski_sanity_checks_end_to_end -v --durations=1`

Expected: PASS. Budget several minutes — the five checks add a passive campaign pair, a high-frequency
campaign pair, a `psd_window` campaign and one fixed `M_max = 256` convergence integration
(`core/FDT/sanity.py`'s `check_ensemble_convergence`, whose `M_max` is **not** scaled by
`--ensemble-m`) on top of the production run.

If it FAILS, this is the first time the path has ever run under test and the failure may be real.
Report the failure verbatim and do not weaken the assertions to get to green.

- [ ] **Step 6: Write the measured time into the docstring**

Replace `Measured <DATE>: <N> s on the CPU` with today's date and the number `--durations=1` printed.
Nothing else in the docstring changes.

- [ ] **Step 7: Run both of this task's tests together**

Run: `pytest tests/test_tool.py -v -k "nadrowski_only_sanity or nadrowski_sanity_checks_end_to_end"`
Expected: PASS, 2 tests.

- [ ] **Step 8: Commit**

```bash
git add tests/test_tool.py
git commit -m "test: the Nadrowski sanity checks, selected and run end to end"
```

#### Amendments (binding — these supersede the text above)

**Rulings that bear on this task:** P70, P55, P23, P56, P22/P74. P56 is already satisfied: records are addressed by `Summary.study`, and the test takes its own `--store-root`. P22/P74 leave the figure count at five. Every anchor quoted above was checked against HEAD `673868d`, or against the text the earlier tasks produce, and was found.

**A0. Preconditions.** This task needs T28 (`--store-root` on `fdt`) and T17 with P70 applied (`_settings_block(cfg, *, skip_sanity=None, confirm_production=None)`, called from `run_fdt` with both arguments). The task-order line names T17 only.
- If step 5's `main([...])` returns `2` with `unrecognized arguments: --store-root`, T28 has not landed. Stop and report.
- If `rec.body["settings"]["skip_sanity"]` raises `KeyError`, T17 did not apply P70. Report it as a T17 defect. Do not delete the assertion.

**A1. Step 4: the slow test's docstring (P55).** Replace this paragraph:

```python
    Production is NOT skipped. ``--no-production`` would be cheaper, but the value ``run_fdt``
    returns and the state of its record on the "Aborted by user" branch are not fixed by the spec
    (fdt_pipeline.py's early ``return`` after the sanity checks), so this test stays on the branch
    whose contract IS fixed: a finished record.
```

with:

```python
    Production is NOT skipped. ``--no-production`` would be cheaper, and its contract is fixed too
    (planning ruling P55: a finished record with ``grid`` and ``offgrid`` null) -- but it stops
    before Campaign 1, so on a Nadrowski cell with the checks on, the two campaigns, the three
    production figures and ``data.h5`` would go unexercised. This test runs the whole path and
    records what that costs.
```

**A2. Step 5: the one failure that is converted here (P23).** Do this only if the slow test fails with the bare `ValueError` whose message begins `check_passive_baseline: every probe frequency lies outside the PSD grid`. For any other failure, follow Step 5 as written.

In `core/FDT/sanity.py`, find:

```python
    if covered.size == 0:
        raise ValueError(
            "check_passive_baseline: every probe frequency lies outside the PSD grid, so the FDT "
            "ratio is unmeasurable here. Narrow cfg.freq_bounds or lengthen the passive run.")
```

Replace with:

```python
    if covered.size == 0:
        # A refusal, not a bug (planning ruling P23): reachable from the low end since the off-grid
        # fix (E9). No field key: which setting to change is not one control's question (spec §12
        # row 9). Refusal subclasses ValueError, so every existing `except ValueError` still catches.
        raise Refusal(
            "check_passive_baseline: every probe frequency lies outside the PSD grid, so the FDT "
            "ratio is unmeasurable here. Narrow cfg.freq_bounds or lengthen the passive run.",
            field=None)
```

Then add `from core.refusals import Refusal` beside `sanity.py`'s other `from core...` imports; `core.refusals` is torch-free. Next, add `core/FDT/sanity.py` to this task's Files block (Modify) and to Step 8's `git add`. Re-run Step 5.

The test will still fail. The run now ends as the tool's one-line refusal with exit 1, and the record stays on disk unfinished. Do NOT weaken any assertion. Report the refusal line verbatim, and do not commit until the plan owner rules on the test. Spec §12 row 9 already records this deviation, so add no §12 row.

**A3.** Steps 1–3 and 6–8 are unchanged.

#### Controller rulings (pre-flight, binding — these supersede the amendments above and the task text)

- **F25** — in the same commit that adds this third slow test, update `pytest.ini`'s `slow` marker description, which names exactly two slow tests. Describe the set without counting it, or name all three.
- **F52** — IF this test reds on `check_passive_baseline`'s bare `ValueError` (P23): convert it to `Refusal(..., field=None)` as P23 says; then, if a configuration within the tool's flags lets the check measure (do not weaken any assertion), use it; if none does, mark the test `pytest.mark.xfail(strict=True, reason=...)` naming the refusal, commit, and put it in the report as a SCIENCE question for the owner — whether a real Nadrowski cell refusing its own passive baseline is a finding or a test-size artefact. Do not stop the piece for it. This is conditional and not expected: every probe must fall below the first positive bin.

---

### Task 34: `test_fdt_and_crossval_run_at_tiny_size` rewritten onto records

**Why:** Spec §8.3, first bullet: the test's assertions glob `artifacts_root()/fdt` and `/crossval`
for PNGs and `.h5` files, "which is exactly what this piece moves", so it is rewritten to assert the
records, their payloads and their figures. Its docstring's 598 s figure is stale against every gate
since and is corrected **from measurement**. E1 is what moved the outputs; §4.1 is what makes the
study two records.

**This test is red when the task starts.** It is `slow`-marked, so it is not in the fast gate T17,
T19 and T28 were measured against — their landing broke it silently. Step 1 is where that is seen.

**Files:**
- Modify: `tests/test_tool.py:1344-1378` — the whole of
  `def test_fdt_and_crossval_run_at_tiny_size(tool_env, capsys):`, from its `@pytest.mark.slow`
  decorator to `    assert "[prism crossval] S sweep:" in capsys.readouterr().out`

**Interfaces:**
- Consumes (T17): `run_fdt` writing one `fdt` record per single-cell run, with `data.h5` as a payload
  and its four figures under `figures/`.
- Consumes (T19): `run_param_study_cli(cfg, *, s_grid, t_grid, writers, seed=None) ->
  list[LoadedFdt]` — two records, `body["points"]["param"]` of `"s"` and `"temp"`, one shared seed.
- Consumes (T20): `body["points"]` as `{param, planned, done, failed}` and `Summary.points_done /
  points_planned / points_failed`.
- Consumes (T28): with no `--store-root`, both subcommands follow `config.artifacts_root()` and
  **not** a `tempfile.mkdtemp` (spec §6.1).
- Consumes (T1/T2): `ArtifactStore.load_fdt`, `LoadedFdt.body` / `.data_path`, `Summary.study` /
  `.finished`.
- Produces: nothing any later task consumes.

- [ ] **Step 1: Run the existing test and watch it fail**

Run: `pytest tests/test_tool.py::test_fdt_and_crossval_run_at_tiny_size -v`

Expected: FAIL with `AssertionError: fdt_ratio_` at

```python
    out = config.artifacts_root() / "fdt"
    for tag in ("fdt_ratio_", "chi_components_", "psd_", "spontaneous_trajectory_"):
        assert list(out.glob(f"{tag}*.png")), tag
```

— the run now writes its figures inside `<artifacts root>/fdt/<name>__<id>/figures/`, so the glob
one level up finds nothing. If instead it fails earlier with a non-zero exit code from `main`, stop:
that is a defect in T17 or T28, not in this test, and it goes in the task report.

- [ ] **Step 2: Replace the docstring and the single-cell leg**

In `tests/test_tool.py`, find:

```python
    Measured 2026-09-15: 598.55 s on the CPU. fdt's own Campaign 1 (810k Euler steps) is one term,
    but crossval's four operating points (2 S-sweep + 2 T-sweep, ~410k steps each under the
    exploratory preset) are likely the bigger share of the total -- neither psd_T_obs_nd nor the
    sweep step count is a flag -- so it is slow-marked; the recorder test above keeps the fast-gate
    coverage. (Fix round 1, M7: corrected from "Campaign 1 alone", which undercounted crossval's
    share; not re-measured, since the PNG assertions added below are cheap globs.)
    """
```

and the paragraph above it. Replace the whole docstring with:

```python
    """The real pipelines, at the smallest sizes the flags allow, writing real ``fdt`` records.

    No new science: this asks only whether the two subcommands drive the campaigns end to end and
    leave behind what piece 5 promises -- one named record per analysis, its numbers in ``data.h5``,
    its pictures under ``figures/``, and a body that says what ran. Until piece 5 it globbed
    ``artifacts_root()/fdt`` and ``/crossval`` for loose PNGs and .h5 files, which is exactly what E1
    moved (spec section 8.3).

    Neither subcommand is given --store-root, deliberately: with the flag unset these two follow
    ``config.artifacts_root()`` and NOT a throwaway temp root (spec section 6.1), and this test is
    what keeps that true end to end. ``tool_env`` is what makes "the temp artifacts root" true --
    it sets PRISM_ARTIFACTS, which ``artifacts_root()`` reads at every call; without it the records
    land in the real ``Artifacts/`` and the session teardown fails.

    The study writes TWO records, one per swept parameter, carrying ONE seed (spec section 4.1): an
    activity sweep that failed entirely no longer costs the temperature sweep.

    Measured <DATE>: <N> s on the CPU. (The previous figure, 598.55 s recorded 2026-09-15, was stale
    against every gate since -- 243, 208 and 229 s -- so it is replaced by a measurement, not by a
    copied number.)
    """
```

Then find the single-cell leg:

```python
    from core import config
    from core.tool import main

    cell = str(config.CELL_PATH / "hopf" / "cell.txt")
    assert main(["fdt", "--cell", cell, "--n-freqs", "2", "--ensemble-m", "8",
                 "--skip-sanity"]) == 0
    out = config.artifacts_root() / "fdt"
    for tag in ("fdt_ratio_", "chi_components_", "psd_", "spontaneous_trajectory_"):
        assert list(out.glob(f"{tag}*.png")), tag
```

Replace with:

```python
    from core import config
    from core.artifacts import ArtifactStore
    from core.tool import main

    cell = str(config.CELL_PATH / "hopf" / "cell.txt")
    assert main(["fdt", "--cell", cell, "--n-freqs", "2", "--ensemble-m", "8",
                 "--skip-sanity"]) == 0

    store = ArtifactStore(config.artifacts_root())
    # ``list`` returns complete rows first, newest first, so [0] is this run even if a sibling test
    # in this module has left an older fdt record in the shared root.
    single = [s for s in store.list("fdt") if s.study == "single"]
    assert single and single[0].finished, single
    rec = store.load_fdt(single[0].id)
    assert rec.body["study"] == "single" and rec.body["complete"] is True
    assert isinstance(rec.body["seed"], int), rec.body["seed"]
    assert rec.body["settings"]["n_freqs"] == 2 and rec.body["settings"]["ensemble_M"] == 8
    assert rec.body["settings"]["skip_sanity"] is True
    # 2 frequencies and 8 trajectories are the thin-setting THRESHOLDS themselves, not below them
    # (spec section 3.3: "fewer than two grid frequencies, fewer than eight trajectories"), so this
    # run is not marked as a quick look.
    assert rec.body["notices"] == [], rec.body["notices"]
    assert rec.body["offgrid"]["of"] == 2, rec.body["offgrid"]
    assert rec.body["grid"]["n_freqs"] == 2, rec.body["grid"]
    # The numbers, not only the pictures (E1).
    assert "data.h5" in rec.manifest.payloads and rec.data_path.exists()
    assert len(rec.manifest.figures) == 4, rec.manifest.figures
    for fig in rec.manifest.figures:
        assert (rec.path / fig).exists(), fig
    # Provenance: the cell by path and hash, relative to Resources/ (provenance.file_ref).
    assert rec.manifest.inputs["cell"]["path"] == "Cells/hopf/cell.txt", rec.manifest.inputs
    assert rec.manifest.inputs["cell"]["sha256"], rec.manifest.inputs
    assert rec.manifest.inputs["bounds"] is not None, rec.manifest.inputs
```

- [ ] **Step 3: Replace the sweep leg**

Find:

```python
    nad = str(config.CELL_PATH / "nadrowski" / "master_spont.txt")
    assert main(["crossval", "--cell", nad, "--s-grid", "0", "0.1", "2",
                 "--t-grid", "1", "1.1", "2", "--n-freqs", "2", "--ensemble-m", "8"]) == 0
    cv = config.artifacts_root() / "crossval"
    assert list(cv.glob("sweep_s_*.h5")) and list(cv.glob("sweep_temp_*.h5"))
    for tag in ("fdt3d_vs_S_", "fdt3d_vs_T_"):
        assert list(cv.glob(f"{tag}*.png")), tag
    assert "[prism crossval] S sweep:" in capsys.readouterr().out
```

Replace with:

```python
    nad = str(config.CELL_PATH / "nadrowski" / "master_spont.txt")
    assert main(["crossval", "--cell", nad, "--s-grid", "0", "0.1", "2",
                 "--t-grid", "1", "1.1", "2", "--n-freqs", "2", "--ensemble-m", "8"]) == 0

    sweeps = [s for s in store.list("fdt") if s.study == "sweep"]
    assert len(sweeps) == 2, sweeps
    assert all(s.finished for s in sweeps), sweeps
    assert {s.points_planned for s in sweeps} == {2}, sweeps
    assert {s.points_done for s in sweeps} == {2}, sweeps
    assert {s.points_failed for s in sweeps} == {0}, sweeps

    recs = [store.load_fdt(s.id) for s in sweeps]
    assert {r.body["points"]["param"] for r in recs} == {"s", "temp"}, \
        [r.body["points"] for r in recs]
    assert len({r.body["seed"] for r in recs}) == 1, \
        "the study draws ONE seed and records it on both records (spec section 4.1)"
    for r in recs:
        assert r.body["complete"] is True
        # A sweep's per-point grids live in data.h5, so the body's single grid block is null
        # (spec section 2.3).
        assert r.body["grid"] is None, r.body["grid"]
        assert r.body["settings"]["preset"] == "exploratory", r.body["settings"]
        assert "data.h5" in r.manifest.payloads and r.data_path.exists()
        assert r.manifest.figures, r.manifest.figures
        for fig in r.manifest.figures:
            assert (r.path / fig).exists(), fig
    capsys.readouterr()                     # the tool's own framing prints; T29 owns their wording
```

The old `assert "[prism crossval] S sweep:" in ...` is dropped rather than updated: T28 and T29 rewrite
those lines, and spec §8.3 asks this test to become assertions about the records. The
`capsys.readouterr()` call stays so the captured output is drained as before.

- [ ] **Step 4: Run it, watch it pass, and write the measured time into the docstring**

Run: `pytest tests/test_tool.py::test_fdt_and_crossval_run_at_tiny_size -v --durations=1`
Expected: PASS.

Then replace `Measured <DATE>: <N> s on the CPU` with today's date and the number `--durations=1`
printed. Do not re-use 598.55, 243, 208 or 229 — those are other runs of other code.

- [ ] **Step 5: Commit**

```bash
git add tests/test_tool.py
git commit -m "test: the tiny-size fdt/crossval run asserts records, not loose files"
```

#### Amendments (binding — these supersede the text above)

**Rulings that bear on this task:** P70, P71, P72, P77, P56, P57, P10/E11, P12–P14, P31. P56, P57, P10, P12–P14 and P31 are already satisfied by the body. P78 needs no assertion here. Every anchor quoted above was found verbatim at HEAD `673868d`; no earlier task edits this test.

**A0. What this task also consumes.** The task-order line lists T17, T19 and T28. This test also consumes:
- T18 (`data.h5`, with the P71 names);
- T20 (the `points` counts);
- T11 with P72 (`preset_name` on the config);
- T7 (`sources`, and so `manifest.inputs`).

In numeric order all of these have landed. If an assertion fails on one of them, report it as that task's defect and do not edit the assertion. In particular:
- `KeyError: 'skip_sanity'` means T17 did not apply P70.
- `settings["preset"] is None` means T11/T19 did not apply P72.

**A1. Step 2: the single-cell record's numbers, against the contract layout (P71).** In Step 2's replacement block, immediately after:

```python
    assert "data.h5" in rec.manifest.payloads and rec.data_path.exists()
```

insert:

```python
    # The contract's single-cell layout (P5, P6, P71): the names core.FDT.compare reads back. The
    # only other writer of this layout is a test fixture, so this is the one place a REAL run's file
    # is checked against the names its reader expects.
    import math

    import h5py
    with h5py.File(rec.data_path, "r") as h5:
        assert h5.attrs["study"] == "single", dict(h5.attrs)
        assert float(h5.attrs["omega_0"]) > 0.0, dict(h5.attrs)
        assert math.isfinite(float(h5.attrs["prefactor"])), dict(h5.attrs)
        for key in ("omega_grid", "T_eff_over_T", "chi_prime", "chi_double_prime"):
            assert key in h5 and h5[key].shape == (2,), (key, list(h5))
        assert h5["PSD_omegas"].shape == h5["PSD_G"].shape, list(h5)
```

Then, **only if `core/FDT/compare.py` already exists when you run this task** (it is created by T36, which normally comes later), add directly below the `with` block:

```python
    # P71: the reader itself, not only the names.
    from core.FDT.compare import curve_of
    curve = curve_of(rec)
    assert curve.omegas.shape == (2,) and curve.ratio.shape == (2,), curve
```

If `compare.py` does not exist, do not add those lines. Say in the task report that the compare read is owed by T36 (plan-owner finding).

**A2. New Step 3a: confirm the docstring's sweep sentence is true (P77).** The new docstring says `an activity sweep that failed entirely no longer costs the temperature sweep`. Before Step 4, read `run_param_study_cli` in `core/FDT/cross_validation.py` and confirm T20's P77 step landed:
- each `run_fdt_param_sweep` call is wrapped;
- a `Refusal` is logged at error and recorded;
- the temperature sweep runs regardless;
- the study refuses only when BOTH sweeps measured nothing.

If it did not land, delete that one sentence from the docstring and report it as a T20 defect. Do not add a test for it here: T20 owns §8.2's test.

**A3. Step 4: if the sweep leg fails only on the counts.** If it fails on `points_failed == {0}` or `points_done == {2}` with the records otherwise finished, do not relax the assertion. Report each record's `body["points"]` verbatim; this is a plan-owner finding about the tiny size.

**A4.** Everything else in Steps 1–5 stands as written.

#### Controller rulings (pre-flight, binding — these supersede the amendments above and the task text)

- **F21/F22** — assert the contract's single-cell layout BY NAME on the record the real run wrote: datasets `omega_grid`, `T_eff_over_T`, `chi_prime`, `chi_double_prime`, `PSD_omegas`, `PSD_G`; root attributes `study`, `omega_0`, `prefactor`. The `compare` read of a real record is added by Task 36, which is the first task where `compare.py` exists; do not add a conditional one here.
- **F53** — assert every real operating point succeeded at the tiny size. If the assertion fails ONLY on `points_failed`, do not relax it: report the counts verbatim and I will rule with evidence.

---

### Task 35: the two round trips, with a bounded repair mandate

**Why:** Spec §5.6 — "Round-trip tests that drive a model through the builder and back, and an export
through Simulate and back" — and §8.2's "Round trips". **The mandate is bounded by the same
sentence:** "A defect a round trip exposes is fixed here only if it is a REFUSAL defect or a
DATA-LOSS defect; anything else is recorded in §12 and handed on."

**The two round trips, read off the code:**
- *The model builder.* `ModelBuilderScreen._assemble_doc()` builds the document from the form;
  `_save()` validates it and calls `model_store.save_user_model(doc)`, which writes the
  Bounds/Cells/Units triple and then the JSON as the commit point; `load_existing(name)` reads that
  JSON back through `model_store.load_user_model` and repopulates every widget. The round trip is
  therefore **form → JSON → form**, and its assertion is that `_assemble_doc()` comes back equal.
- *Simulate.* The panel's only export is the video: `_save_video` concatenates `self._record`, tells
  the operator how many frames it will write (`estimate_frame_count(len(series),
  export_stride(1.0 / DT_EXP_S, fps))`, `simulate_panel.py`), and dispatches `export_animation`,
  which strides the series itself. The round trip is **recording → file → frames read back**, and
  its assertion is that the file holds exactly the number of frames the panel promised.

**Files:**
- Modify (append one test): `tests/test_user_models.py:573-589` — after
  `def test_builder_refuses_a_log_box_with_a_non_positive_minimum():`, whose last line is
  `    assert "log box needs min > 0" in mb.status.text(), mb.status.text()`
- Modify (append one test): `tests/test_simulate.py:231-243` — after
  `def test_export_animation_writes_a_readable_gif():`, whose last line is
  `    assert frames[0].shape[0] % 2 == 0 and frames[0].shape[1] % 2 == 0, "exported frame dims must be even"`
- Modify **only on a confirmed refusal or data-loss finding**:
  `core/gui/screens/model_builder_screen.py` or `core/gui/panels/simulate_export.py`
- Modify for every other finding: `docs/superpowers/specs/2026-09-22-secondary-analyses-design.md`
  §12's table, whose header row is `| # | where | deviation | why |`

**Interfaces:**
- Consumes (T22): the model builder's refusals routed through
  `core/gui/widgets/refusal_box.show_refusal`.
- Consumes (T23): Simulate's `frame_steps` / `fps` clamps replaced by refusals naming the box.
- Produces: nothing any later task consumes.

- [ ] **Step 1: Write the model-builder round trip**

In `tests/test_user_models.py`, append after the anchor named above:

```python
def test_a_model_round_trips_through_the_builder_and_back():
    """Build a model in the screen, save it, load it back, and the form assembles the SAME document.

    Spec section 5.6. The builder is the one surface where a user types a model from nothing, and
    every field it drops on the way back is silent: a forcing amplitude, a box coordinate or an
    initial condition that quietly reverts to its default would be discovered only by a later run
    that behaved differently from the form the operator was looking at. The pieces are individually
    tested -- ``_ParamRow`` preserves a custom box across a re-detect, ``model_store`` emits a
    parseable triple -- but nothing has ever asserted that the WHOLE path is lossless.

    Exercises the fields most likely to be lost: a forcing block with four parameters, a log box
    with a positive lower bound, a non-default initial condition, and both display scales. Writes a
    throwaway UMTEST* model into the real Resources tree and removes it in a finally -- the
    convention every other round trip in this file follows.
    """
    from core.gui.screens.model_builder_screen import _FORCE_KINDS, ModelBuilderScreen

    qt_app()
    name = "UMTESTRT"
    sin_index = [k for k, _ in _FORCE_KINDS].index("sin")
    try:
        mb = ModelBuilderScreen()
        mb.name_edit.setText(name)
        mb.vars_edit.setText("x, y")
        mb._set_variables()
        mb._var_rows[0].drift.setText("-k1*x")
        mb._var_rows[0].noise.setText("d0")
        mb._var_rows[0].init.setText("0.1")
        mb._var_rows[0].force_kind.setCurrentIndex(sin_index)
        for pname, val in (("amp", 0.5), ("freq", 10.0), ("phase", 0.25), ("offset", 0.1)):
            mb._var_rows[0]._force_fields["sin"][pname].setText(repr(val))
        mb._var_rows[1].drift.setText("-y + x")
        mb._var_rows[1].noise.setText("0")
        mb._var_rows[1].init.setText("0.0")
        mb.x_scale.setText("10.0")
        mb.t_scale.setText("0.01")
        mb._detect_params()
        assert list(mb._param_fields) == ["k1", "d0"], list(mb._param_fields)
        mb._param_fields["k1"].set_spec(1.0, 0.5, 1.5, "linear")
        mb._param_fields["d0"].set_spec(0.05, 0.01, 0.1, "log")   # a log box needs min > 0

        before = mb._assemble_doc()
        mb._save()
        assert mb.status.text().startswith(f"Saved '{name}'"), mb.status.text()

        mb.reset()
        mb.load_existing(name)
        after = mb._assemble_doc()
        assert after == before, \
            [k for k in set(before) | set(after) if before.get(k) != after.get(k)]
    finally:
        _remove_user_model(name)
```

- [ ] **Step 2: Run it**

Run: `pytest tests/test_user_models.py::test_a_model_round_trips_through_the_builder_and_back -v`

Expected: **PASS**, or a `AssertionError` whose message is the list of document keys that differ
(`['variables']`, `['params']`, `['rescale']`, …). Both outcomes are useful: a pass closes spec
§5.6's first round trip, a failure is a finding that step 4 triages.

If it fails with `AssertionError` on `mb.status.text()` instead, read the status text: that is
`_validate` refusing the document, and it names which field it refused.

- [ ] **Step 3: Write the Simulate export round trip**

In `tests/test_simulate.py`, append after `test_export_animation_writes_a_readable_gif`:

```python
def test_a_simulate_recording_round_trips_through_the_video_export():
    """Every frame the panel PROMISED the operator is in the file it wrote. Spec section 5.6.

    ``_save_video`` prints "Exporting {n} frames" from ``estimate_frame_count(len(series),
    export_stride(1.0 / DT_EXP_S, fps))`` and then hands the concatenated recording to
    ``export_animation``, which strides it a second time with its own default ``sample_rate_hz``. The
    two arithmetics are written out separately, in two modules, so if they ever drift the video
    silently holds fewer frames than the run it claims to be -- and nothing would say so. The
    existing gif test asks only for "at least 2".

    Also asserts the frames ADVANCED: a writer that appended the same buffer every time would pass a
    frame count on its own.
    """
    import os
    import tempfile

    import numpy as np

    from core.config import DT_EXP_S
    from core.gui.panels.simulate_export import (estimate_frame_count, export_animation,
                                                 export_stride)

    series = _tiny_series()
    kw = _export_kwargs()
    promised = estimate_frame_count(len(series), export_stride(1.0 / DT_EXP_S, kw["video_fps"]))
    assert promised > 2, promised                    # the fixture must exercise more than the edges

    path = os.path.join(tempfile.mkdtemp(), "roundtrip.gif")
    export_animation(series, path, **kw)

    import imageio
    frames = imageio.mimread(path)
    assert len(frames) == promised, (len(frames), promised)
    assert not np.array_equal(frames[0], frames[-1]), "the exported frames never advanced"
```

- [ ] **Step 4: Run it**

Run: `pytest tests/test_simulate.py::test_a_simulate_recording_round_trips_through_the_video_export -v`

Expected: PASS with `promised == 18` (`_tiny_series()` is 600 samples at 1 ms; `export_stride(1000,
30) == 33`; `len(range(32, 600, 33)) == 18`).

One failure mode here is **not** a PRISM defect: if `len(frames)` differs from `promised` by a
constant and the two arithmetics in `simulate_export.py` agree with each other, that is the GIF
reader coalescing or padding, not data loss. Confirm it by reading the writer's own count —
`sum(1 for _ in range(export_stride(1000.0, 30.0) - 1, len(series), export_stride(1000.0, 30.0)))`
— and if the writer wrote `promised` frames, record it as a §12 row (step 5) and assert
`len(frames) >= promised` with that reason in a comment instead.

- [ ] **Step 5: Triage every finding against the bounded mandate**

For each failure either test produced, classify it with this rule and act:

- **DATA-LOSS — fix here.** A value the form or the recording held before the write differs after the
  read back, or is absent. Example shape: a forcing parameter that comes back at
  `_FORCE_DEFAULTS[pname]` rather than the typed value, or a video that holds fewer frames than the
  writer's own stride produced. Fix the smallest thing in
  `core/gui/screens/model_builder_screen.py` or `core/gui/panels/simulate_export.py` that makes the
  round trip equal, add the failing assertion as its own named test beside the round trip, and
  commit it as a **separate** commit before step 6's.
- **REFUSAL — fix here.** The screen accepted something that the layer below then rejected with a
  bare exception, or refused with a message that names no box. The fix is a `Refusal` carrying the
  field key, routed through `core/gui/widgets/refusal_box.show_refusal` (T22) — not a new
  `_set_status` string.
- **ANYTHING ELSE — record and hand on.** Cosmetic status wording, dict ordering, a default that is
  re-derived to the same value, a reader artefact, an ergonomic wish. Add a row to
  `docs/superpowers/specs/2026-09-22-secondary-analyses-design.md` §12:

```
| 1 | §5.6, T35 | <one sentence: what the round trip showed> | Not a refusal or data-loss defect, so §5.6's bounded mandate hands it on. Cost if wrong: <what breaks if it is left> |
```

If both tests passed, write nothing into §12 and say so in the task report.

- [ ] **Step 6: Run both suites' relevant files**

Run: `pytest tests/test_user_models.py tests/test_simulate.py -v`
Expected: PASS, no test that was green before this task is red now. In particular
`test_builder_param_row_preserves_and_defaults`, `test_builder_refuses_a_blank_bound_instead_of_reading_it_as_zero`
and `test_export_animation_writes_a_readable_gif` must all still pass — a round trip that needed one
of them relaxed is a finding, not a fix.

- [ ] **Step 7: Commit**

```bash
git add tests/test_user_models.py tests/test_simulate.py
git commit -m "test: a model round-trips through the builder and an export through Simulate"
```

If step 5 added a §12 row, include the spec file in this commit:

```bash
git add docs/superpowers/specs/2026-09-22-secondary-analyses-design.md
```
# Tasks 36-40 — COMPARING (spec §7, decision E8)

> The last part of the piece (E12); nothing before it depends on it. Every task here assumes the
> `fdt` kind exists (T1), the writer has its progressive mode and `refresh()` (T3), `store.load_fdt`
> returns a `LoadedFdt` carrying `.body` and `.data_path` (T1), a single-cell run has written its
> numbers to `data.h5` (T18), and the sweep writes one record per swept parameter (T19-T20).
>
> **One public entry for the whole facility.** `core/FDT/compare.compare(mode, refs, ...)` is the
> only `@public_entry` these five tasks add, so the `public_entry` source scan and its two companion
> sets are touched **once**, in Task 36, and Tasks 37-39 add plain drawing functions that register
> themselves in `compare._DRAWERS`. Neither T37, T38 nor T39 may add a second decorator.

#### Amendments (binding — these supersede the text above)

**Rulings and earlier tasks that bear on this task:** P58, P59, T22 (with P36 and P20 for a key), T23. P58 and T22 change steps. T23 changes nothing: the arithmetic Step 3 mirrors survives it.

**A1. Step 1: the insertion point (T22 moved it).** T22 has already replaced the old last line of `test_builder_refuses_a_log_box_with_a_non_positive_minimum` (it now reads `assert "log coordinate needs a minimum above 0" in ei.value.message, ei.value.message`). T22 also inserted `test_the_builder_shows_a_field_refusal_in_the_yellow_box` right after that test. So the text Step 1 anchors on is gone. Insert the round-trip test immediately ABOVE:

```python
def test_model_store_rejects_unusable_values_and_names():
```

That places it after T22's test. The test body is unchanged.

**A2. Step 2: reading a refusal.** Since T22, a field problem in `_save()` raises `Refusal` inside `_validate`, and `_save` shows it through `ModelBuilderScreen._refusal`. So if the `startswith(f"Saved '{name}'")` assertion fails, read two things:
- `mb.status.text()`, which carries the message and the fix sentence;
- `tests._fixtures.SHOWN[-1].text()`.

That is `_validate` refusing the form, and the refusal carries a field key.

**A3. Step 3: say which half is new (P59).** In `test_a_simulate_recording_round_trips_through_the_video_export`, replace:

```python
    silently holds fewer frames than the run it claims to be -- and nothing would say so. The
    existing gif test asks only for "at least 2".
```

with:

```python
    silently holds fewer frames than the run it claims to be -- and nothing would say so.

    Half of this round trip is NOT new: test_export_animation_writes_a_readable_gif already renders a
    series to a GIF and reads it back, asking only for "at least 2" frames. What this test adds is
    precision: the exact count the panel promises, and that the frames advance (planning ruling P59).
```

The task report must also say that the model-builder round trip is new ground, and that the Simulate one adds precision to an existing test.

**A4. Step 4: the fallback must count what the WRITER received.** The body's 'writer's own count' formula is the same arithmetic as `promised`, so it cannot separate a reader artefact from data loss. Its `>=` also fails for a reader that coalesces frames. Replace the fallback paragraph with the following.

If `len(frames) != promised`, give the test a `monkeypatch` parameter and count the frames handed to the writer:

```python
    from core.gui.panels import simulate_export
    appended = []
    real_open = simulate_export._open_writer

    class _Counting:
        def __init__(self, w):
            self._w = w

        def append_data(self, im, *a, **k):
            appended.append(1)
            return self._w.append_data(im, *a, **k)

        def close(self):
            return self._w.close()

    monkeypatch.setattr(simulate_export, "_open_writer", lambda p, fps: _Counting(real_open(p, fps)))
```

Install the spy before calling `export_animation`, then:

```python
    assert len(appended) == promised, (len(appended), promised)     # the data-loss question
```

How to act on the result:
- If `len(appended) == promised`, the writer lost nothing and the difference is the GIF reader's. Keep the `len(appended)` equality. Replace the `len(frames) == promised` line with a commented bound in the observed direction (`<=` if the reader coalesced, `>=` if it padded). Record the reader behaviour per A5.
- If `len(appended) != promised`, it is a DATA-LOSS finding. Fix it under Step 5.

**A5. Steps 4, 5 and 7, and the Files block: where a handed-on finding goes (P58).** Every mention of §12 in this task is superseded: the Why quote 'anything else is recorded in §12 and handed on', the Files line naming '§12's table, whose header row is `| # | where | deviation | why |`', Step 4's 'record it as a §12 row', Step 5's 'Add a row to ... §12' and its `| 1 | §5.6, T35 | ...` template, and Step 7's reason for adding the spec file.

A finding that is neither a refusal nor data loss goes into §1.3 of `docs/superpowers/specs/2026-09-22-secondary-analyses-design.md`. That table's header is `| thing | why | owner |`. Add the new row as its last row, directly after the row beginning `| Deleting the owner's legacy files |`, in this shape:

```
| <one sentence: what the round trip showed> (found by T35's round trip, §5.6) | Not a refusal or data-loss defect, so §5.6's bounded mandate hands it on (P58). Cost if it is left: <what breaks if nobody takes it> | nobody yet |
```

Name each such item in the task report too; T41 carries it into `docs/STATE.md`. If both tests passed, write nothing in the spec and say so. Step 7's conditional `git add` of the spec file stands, for a §1.3 row.

**A6. Step 5: the REFUSAL bullet after T22.** A refusal fix in the model builder is a `refuse(<key>, <sentence>)` raised inside `ModelBuilderScreen._validate`, which `_save`/`_validate_clicked` already route to `self._refusal(exc)` and so to `show_refusal`. Use a key T22 registered: `param_value`, `param_min`, `param_max`, `init`, `x_scale`, `t_scale`, or T22's forcing key (P36 names it `forcing_value`; T22's draft said `forcing_param`; use the one in `core/refusals.py`).

If the fix needs a key that does not exist, add it in full, as P20 requires:
- the `Field` in `core/refusals.py` `FIELDS`;
- a `core/gui/fields.py` `CONTROL` entry on the Model Builder screen;
- a `None` entry in `core/tool/fields.py` `FLAG`;
- the key in `tests/test_refusals.py`'s `BASE_KEYS`, and the `len(FIELDS)` literal read off the failing pin.

Never write that count blind. Add those files to the fix's own separate commit.

**A7.** Steps 6 and 7 are otherwise unchanged. Step 6's three named tests are T22's updated versions, and they must still pass.

#### Controller rulings (pre-flight, binding — these supersede the amendments above and the task text)

- **F54** — a round-trip defect handed on (P58) goes into the spec's §1.3 table, "Out of scope, and who owns it", as a row with its owner. Re-read the table first and append after its last row.

---

### Task 36: the comparison record, the common grid, and what a comparison refuses

**Why:** Spec §7.2 fixes the common grid ("log-spaced over the intersection of their spans, with N
the smallest `n_freqs` among the records"; blanks are never interpolated across) and §7.3 fixes the
record ("an `fdt` record with `study = "comparison"`, whose `body.compared` names the mode and the
ids and names of the records it drew, whose figures are its output, and whose `data.h5` holds the
common grid and the interpolated curves"). This task owns **Review Focus item 5**: a comparison
asked for one record, or for records of different studies, is refused naming what it got, and an
unfinished record (E2 leaves those on disk) is refused naming it.

**Files:**
- Create: `core/FDT/compare.py`
- Modify: `core/refusals.py` — the `FIELDS` tuple, after `Field("note", "the note", None),`
- Modify: `core/gui/fields.py:86-87` — `CONTROL`, beside `"artifact"` and `"note"`
- Modify: `core/tool/fields.py:63-64` — `FLAG`, beside `"artifact"` and `"note"`
- Modify: `core/tool/fdt.py:119` — `register`'s `return {"fdt": fdt, "crossval": cv}`
- Modify: `tests/_fixtures.py` — `build_fdt_record` (see Step 1's note)
- Modify: `tests/test_refusals.py:29-41,91` — `BASE_KEYS` and the registry count
- Modify: `tests/test_artifact_store.py:2497-2566` — `_UNTOUCHED_LEGS`, `_REFUSAL_FIELDS`, the scan
- Test: Create `tests/test_fdt_compare.py`

**Interfaces:**
- Consumes: `store.create("fdt", None, name=…, note=…) -> ArtifactWriter` with the progressive mode
  (T3: `__enter__` writes a first manifest carrying every `BODY_KEYS["fdt"]` key, `refresh()`
  rewrites it, `__exit__` sets `complete` True on the clean path and KEEPS the directory on an
  exception); `store.load_fdt(ref) -> LoadedFdt` with `.body` and `.data_path` (T1);
  `manifest.BODY_KEYS["fdt"] == ("study", "settings", "seed", "grid", "points", "offgrid",
  "notices", "compared", "complete", "results")`.
- Produces, for Tasks 37-40:
  - `MODE_RULES: dict[str, tuple[str, int]]` — `mode -> (study the records must carry, fewest records)`
  - `MODE_OPTIONS: dict[str, tuple[str, ...]]` — the keyword settings each mode accepts
  - `_DRAWERS: dict[str, callable]` — `mode -> drawer(w, records, *, sink, **options) -> (results, notices)`
  - `Curve` (frozen dataclass: `id, name, label, omegas, ratio, omega_0, prefactor`)
  - `curve_of(rec) -> Curve`, `common_grid(curves) -> np.ndarray`,
    `interpolate_onto(grid, omegas, values) -> np.ndarray`, `blank_notice(values) -> (int, list)`,
    `write_curves(w, grid, values, labels) -> None`, `finish_record(w, *, results, notices) -> None`
  - `compare(mode, refs, *, name="", note="", fig_sink=None, store=None, **options) -> LoadedFdt`
  - `python -m core compare {cells,repeats,renormalise,sweeps} --record REF [--record REF] …`
  - the refusal field key `compare_records`, flag `--record`

- [ ] **Step 1: Add the record fixture**

`tests/_fixtures.py` is promised a `build_fdt_record` by spec §8.1. **If an earlier task already
added one, extend it with the keyword arguments and the `data.h5` body below rather than adding a
second helper** — §7's readers require exactly these dataset names. Append after `build_browse_store`:

```python
class _FdtStop(Exception):
    """Ends a record's write block where a cancel or a crash would, so ``build_fdt_record`` can leave
    an UNFINISHED record on disk (E2) without pretending to fail inside the store."""


def build_fdt_record(store, *, study="single", name="", note="", finished=True,
                     omegas=(0.5, 1.0, 2.0, 4.0), ratio=(1.0, 3.0, 1.2, 1.05),
                     omega_0=1.0, prefactor=2.0, settings=None):
    """One ``fdt`` record with a real manifest and a real ``data.h5``, written in seconds.

    ``cfg=None``, like ``build_browse_store``'s rows: nothing here reads a bounds file, builds a
    SimConfig or simulates. The dataset names are ``cross_validation._fdt_measure``'s own vocabulary
    (``omega_grid``, ``T_eff_over_T``), which is what the single-cell run writes (spec §2.3) and what
    ``core.FDT.compare`` reads back. ``finished=False`` leaves the record unfinished on disk with its
    numbers already written -- the state E2 exists to preserve, and the one a comparison refuses.

    :returns: the record's id.
    """
    import h5py
    import numpy as np
    w = store.create("fdt", None, name=name, note=note)
    w.body = {"study": study, "settings": dict(settings or {"n_freqs": len(omegas), "F0": 0.05}),
              "seed": 7, "grid": None, "points": None, "offgrid": None, "notices": [],
              "compared": None, "complete": False, "results": None}
    try:
        with w:
            with h5py.File(w.payload("data.h5"), "w") as h5:
                h5.attrs["study"] = study
                h5.attrs["omega_0"] = float(omega_0)
                h5.attrs["prefactor"] = float(prefactor)
                h5.create_dataset("omega_grid", data=np.asarray(omegas, dtype=np.float64))
                h5.create_dataset("T_eff_over_T", data=np.asarray(ratio, dtype=np.float64))
            if not finished:
                raise _FdtStop("interrupted after the numbers were written")
    except _FdtStop:
        pass
    return w.id
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_fdt_compare.py`:

```python
"""Comparing saved FDT records (piece 5, spec §7; decision E8).

The comparison facility is the last part of the piece and nothing before it depends on it (E12).
This suite covers the shared machinery -- the common grid, the interpolation that never blends a
blank away, and what a comparison refuses before it draws anything -- and then one test per mode.

Nothing here simulates. Every record is written straight into a store by ``build_fdt_record``, the
way ``build_browse_store`` writes its rows (``cfg=None``, the smallest body the kind allows), so the
whole file costs the writer's git calls and no solver at all.

Run:  pytest tests/test_fdt_compare.py
"""
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib                                                  # noqa: E402
matplotlib.use("Agg")                                              # match the app; no interactive path

import h5py                                                        # noqa: E402
import numpy as np                                                 # noqa: E402
import pytest                                                      # noqa: E402

from core.artifacts import ArtifactStore                           # noqa: E402
from core.FDT import compare as cmp                                # noqa: E402
from core.refusals import Refusal                                  # noqa: E402
from tests._fixtures import build_fdt_record                       # noqa: E402


def _store(tmp_path):
    return ArtifactStore(tmp_path / "artifacts")


def _closing():
    """A sink that records the titles it is handed and closes each figure, like the store writer's
    own sink does when nothing is forwarded."""
    seen = []

    def _sink(title, fig):
        from matplotlib import pyplot as plt
        seen.append(title)
        plt.close(fig)
    return seen, _sink


def test_the_common_grid_is_the_log_spaced_intersection_at_the_smallest_point_count():
    """Spec §7.2. Every run detects its own resonance and builds its grid around it, so two runs of
    the same cell land on different frequencies; drawing them on one axis without a common grid would
    compare a value at 1.9 with a value at 2.1 and call the difference a result. The intersection is
    what both runs actually measured, and the smallest point count is the only one neither has to be
    invented for."""
    a = cmp.Curve("a", "a", "a", np.array([1.0, 2.0, 4.0, 8.0]), np.ones(4), 1.0, 1.0)
    b = cmp.Curve("b", "b", "b", np.array([2.0, 4.0, 8.0]), np.ones(3), 1.0, 1.0)
    grid = cmp.common_grid([a, b])
    assert grid.shape == (3,), grid
    assert grid[0] == pytest.approx(2.0) and grid[-1] == pytest.approx(8.0)
    # log-spaced: the ratio between neighbours is constant
    assert grid[1] / grid[0] == pytest.approx(grid[2] / grid[1])
    # no overlap at all is a refusal, not an empty picture
    far = cmp.Curve("c", "c", "c", np.array([100.0, 200.0]), np.ones(2), 1.0, 1.0)
    with pytest.raises(Refusal) as e:
        cmp.common_grid([a, far])
    assert e.value.field == "compare_records" and "overlap" in str(e.value), e.value


def test_no_target_point_is_interpolated_across_a_blank():
    """Spec §7.2, and the whole point of the off-grid fix (E9): a frequency the run could not measure
    comes back blank, and a comparison that quietly interpolated over it would put the fabricated tail
    back -- the exact defect §1 of the spec measured in ``_interp_log``. A target between two good
    samples is a blend; a target whose bracket includes a blank, or that lies outside the source's
    span, is blank. Asserted through an interpolated INDICATOR rather than by trusting NaN to survive
    ``np.interp``, which retries a non-finite result from the other bracket."""
    om = np.array([1.0, 2.0, 4.0, 8.0])
    vals = np.array([1.0, np.nan, 3.0, 4.0])
    out = cmp.interpolate_onto(np.array([1.0, 1.5, 2.0, 3.0, 4.0, 16.0]), om, vals)
    assert out[0] == pytest.approx(1.0), "a target ON a good sample keeps that sample"
    assert math.isnan(out[1]), "bracketed by the blank at 2.0"
    assert math.isnan(out[2]), "the blank itself"
    assert math.isnan(out[3]), "bracketed by the blank at 2.0 on the other side"
    assert out[4] == pytest.approx(3.0)
    assert math.isnan(out[5]), "outside the source's span"
    # and a curve with no blanks is interpolated normally, in LOG omega
    clean = cmp.interpolate_onto(np.array([2.0]), np.array([1.0, 4.0]), np.array([0.0, 2.0]))
    assert clean[0] == pytest.approx(1.0), "linear in log-omega: log2 is halfway between log1 and log4"


def test_a_comparison_refuses_one_record_another_study_and_an_unfinished_one(tmp_path):
    """Review Focus item 5, and spec §7.1's "a comparison draws only records whose complete is true".
    Every mode's arity is a real input class -- one cell is not a comparison, and a sweep drawn as a
    single-cell run reads a layout that is not there -- and E2 means half-written records sit on disk
    carrying a valid manifest, so the picker and the flag can both hand one over. Each refusal names
    what it got and carries the field both front ends map to their own control."""
    store = _store(tmp_path)
    one = build_fdt_record(store, name="a")
    sweep = build_fdt_record(store, name="sw", study="sweep")
    part = build_fdt_record(store, name="half", finished=False)

    with pytest.raises(Refusal) as e:
        cmp.compare("cells", [one], store=store)
    assert e.value.field == "compare_records" and "at least 2" in str(e.value), e.value
    assert "got 1" in str(e.value), e.value

    with pytest.raises(Refusal) as e:
        cmp.compare("cells", [one, sweep], store=store)
    assert "'sw'" in str(e.value) and "sweep" in str(e.value), e.value

    with pytest.raises(Refusal) as e:
        cmp.compare("cells", [one, part], store=store)
    assert "'half'" in str(e.value) and "did not" in str(e.value), e.value

    # an unfinished record is listed and loadable -- it is the comparison that refuses it, not the store
    assert store.load_fdt(part).body["complete"] is False
    # a setting the mode has no use for is refused rather than recorded
    with pytest.raises(Refusal) as e:
        cmp.compare("cells", [one, one], prefactor=3.0, store=store)
    assert "prefactor" in str(e.value), e.value


def test_a_comparison_is_a_record_that_names_the_mode_and_the_runs_it_drew(tmp_path, monkeypatch):
    """Spec §7.3: a comparison is an ``fdt`` record of its own, with ``study = "comparison"``, its
    sources in ``body.compared`` and NOT in ``parents`` (the parents block is a flat {key: id} map and
    cannot carry an arbitrary number of ids -- spec §1.2), the common grid and the interpolated curves
    in ``data.h5``, and its figures as its output. Driven through a stub drawer, so this pins the
    record and not any one mode's picture."""
    store = _store(tmp_path)
    a = build_fdt_record(store, name="cell_a")
    b = build_fdt_record(store, name="cell_b", omegas=(1.0, 2.0, 4.0), ratio=(1.0, 2.0, 1.1))

    def _stub(w, records, *, sink):
        curves = [cmp.curve_of(r) for r in records]
        grid = cmp.common_grid(curves)
        values = [cmp.interpolate_onto(grid, c.omegas, c.ratio) for c in curves]
        cmp.write_curves(w, grid, values, [c.label for c in curves])
        from matplotlib import pyplot as plt
        sink("Stub comparison", plt.figure())
        return {"n_records": len(curves)}, ["a notice the record keeps"]

    monkeypatch.setitem(cmp._DRAWERS, "cells", _stub)
    rec = cmp.compare("cells", [a, b], name="ab", note="two cells", store=store)

    assert rec.body["study"] == "comparison" and rec.body["complete"] is True
    assert rec.body["compared"]["mode"] == "cells"
    assert [r["id"] for r in rec.body["compared"]["records"]] == [a, b]
    assert [r["name"] for r in rec.body["compared"]["records"]] == ["cell_a", "cell_b"]
    assert rec.manifest.parents == {}, "a comparison names its sources in the body, never in parents"
    assert rec.body["results"] == {"n_records": 2} and rec.body["notices"] == ["a notice the record keeps"]
    assert rec.body["seed"] is None and rec.body["points"] is None and rec.body["grid"] is None
    assert rec.manifest.figures == ["figures/stub_comparison.png"]
    assert (rec.path / "figures" / "stub_comparison.png").is_file()
    assert rec.manifest.payloads["data.h5"], "the payload is hashed at the commit, like every other kind"
    assert (rec.path / "log.txt").is_file(), "public_entry: the run's records land beside the manifest"
    with h5py.File(rec.data_path, "r") as h5:
        assert h5.attrs["study"] == "comparison" and h5.attrs["mode"] == "cells"
        assert h5["omega_common"].shape == (3,)
        assert sorted(h5["curves"]) == ["000", "001"]
        assert h5["curves"]["000"].attrs["label"] == "cell_a"
```

- [ ] **Step 3: Run them and watch them fail**

Run: `pytest tests/test_fdt_compare.py -x -q`
Expected: collection FAILS with
`ModuleNotFoundError: No module named 'core.FDT.compare'` (and, before that is fixed,
`ImportError: cannot import name 'build_fdt_record' from 'tests._fixtures'` if Step 1 was skipped).

- [ ] **Step 4: Register the field key in all three tables**

In `core/refusals.py`, find:

```python
    # the artifact browser (piece 4): the artifact a browse action acts on, and its note
    Field("artifact", "the artifact", None),
    Field("note", "the note", None),
```

Replace with:

```python
    # the artifact browser (piece 4): the artifact a browse action acts on, and its note
    Field("artifact", "the artifact", None),
    Field("note", "the note", None),
    # comparing saved FDT records (piece 5, E8): which records a comparison was asked to draw
    Field("compare_records", "the saved runs to compare", None),
```

In `core/gui/fields.py`, find:

```python
    "artifact": "Select an artifact in the list on the Artifacts screen.",
    "note": "Edit it in the Note box on the Artifacts screen.",
```

Replace with:

```python
    "artifact": "Select an artifact in the list on the Artifacts screen.",
    "note": "Edit it in the Note box on the Artifacts screen.",
    # A sentence and not a (place, label) pair: the comparison list is one control that appears on
    # two screens of the FDT section, and the box it sits in is added by the comparison controls.
    "compare_records": ("Add the runs to compare to the comparison list on the FDT analysis or the "
                        "Sweep study cross-validation tab."),
```

In `core/tool/fields.py`, find:

```python
    "artifact": None,
    "note": "--note",                                   # add_name_flags defines it beside --name
```

Replace with:

```python
    "artifact": None,
    "note": "--note",                                   # add_name_flags defines it beside --name
    # comparing saved records: `compare <mode> --record REF --record REF`, one flag per record
    "compare_records": "--record",
```

- [ ] **Step 5: Move the two registry pins**

In `tests/test_refusals.py`, find (line numbers are hints; the anchor is the text):

```python
    "chi_max_cycles", "chi_f0", "chi_freq_bounds", "device", "model", "observation", "posterior", "prior",
    "artifact", "note",
)
```

Replace with:

```python
    "chi_max_cycles", "chi_f0", "chi_freq_bounds", "device", "model", "observation", "posterior", "prior",
    "artifact", "note", "compare_records",
)
```

Then find `== 63, "a key is listed twice above"` and raise the number by one — **re-read it first**:
Tasks 8, 12, 13, 23, 38 and 39 each add keys, so the number in the file is no longer 63. Add one to
whatever is there.

- [ ] **Step 6: Run the registry pins**

Run: `pytest tests/test_refusals.py -q -k "registry_holds or the_tool_table"`
Expected: PASS. (`--record` does not exist yet, so if the tool-table pin fails on
`FLAG names an option no subcommand defines: {'compare_records': '--record'}`, that is Step 8's
work — run this again after it.)

- [ ] **Step 7: Write `core/FDT/compare.py`**

```python
"""Comparing saved FDT records (piece 5, spec §7; decision E8).

A comparison is itself an ``fdt`` record, with ``body.study = "comparison"`` (§7.3): ``body.compared``
names the mode and the records it drew, its figures are its output, and its ``data.h5`` holds the
common grid and the interpolated curves. It names its sources in the BODY and not in ``parents``,
because the parents block is a flat ``{key: id}`` map read as ``m.parents.get(pk) == id_``, so it
cannot carry an arbitrary number of ids without widening the store's contract for every kind (spec
§1.2). Deleting a record a comparison drew is therefore not refused; ``report.render_lineage``
resolves the body's ids and prints ``MISSING`` for one the store no longer holds.

ONE PUBLIC ENTRY, four modes. Every run detects its own resonance and builds its grid around it, so
two runs of the same cell land on different frequencies; each mode therefore goes through the same
three steps -- load and check the records, interpolate onto a common grid, write the record -- and
differs only in what it draws. The modes register themselves in ``_DRAWERS``.

BLANKS ARE NEVER INTERPOLATED ACROSS. A frequency a run could not measure comes back blank (E9), and
a comparison that blended over one would put back exactly the fabricated tail the off-grid fix
removed. A target point whose bracketing samples include a blank is blank, and the record says how
many there were, in its notices and on the axis.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from core.artifacts import resolve_store
from core.refusals import refuse
from core.runs import public_entry

# The comparison's own report -- which records it drew, how much of the common grid was usable -- is
# information (piece 3, V4); the record's log.txt keeps it beside the manifest.
log = logging.getLogger(__name__)

#: ``mode -> (the study its records must carry, the fewest records it can draw)``. The arity is a
#: real input class and not a nicety: one cell is not a comparison, and a sweep drawn as a
#: single-cell run reads a layout that is not there.
MODE_RULES: dict[str, tuple[str, int]] = {
    "cells": ("single", 2),
    "repeats": ("single", 2),
    "renormalise": ("single", 1),
    "sweeps": ("sweep", 2),
}

#: ``mode -> the keyword settings it accepts``. A setting a mode has no use for is refused rather
#: than recorded: ``compare`` writes the options it was given into ``body.settings``, and a typo
#: recorded there would describe a comparison that never happened.
MODE_OPTIONS: dict[str, tuple[str, ...]] = {
    "cells": (), "repeats": (), "renormalise": ("prefactor",), "sweeps": ("at",),
}

#: ``mode -> drawer(w, records, *, sink, **options) -> (results, notices)``. Filled by the mode
#: modules below as each lands; a mode with no drawer is refused, which also covers a mode string
#: that reached here without passing through the tool's own choices.
_DRAWERS: dict = {}

#: What a single-cell record's ``data.h5`` calls its two curves (spec §2.3). The names are
#: ``cross_validation._fdt_measure``'s own vocabulary, which the sweep's file already uses, so one
#: reader's words describe both files.
OMEGA_GRID, RATIO = "omega_grid", "T_eff_over_T"

#: Appended to the frequency axis whenever a drawn curve has blanks, so the picture says what the
#: record's notices say.
BLANK_AXIS_NOTE = "gaps are frequencies a run did not measure; they are never interpolated across"


@dataclass(frozen=True)
class Curve:
    """One record's ratio curve, on its OWN frequency grid."""
    id: str
    name: str
    label: str              # the legend entry: the cell file's stem, else the record's own label
    omegas: np.ndarray
    ratio: np.ndarray
    omega_0: float
    prefactor: float


def _num(x):
    """A finite float, or None -- a manifest refuses a non-finite number in the body
    (``manifest._check_finite``) and this ratio legitimately carries NaN (spec §2.3). The same rule
    as ``orchestrator._num``, restated here so this module needs no orchestrator import."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _label_of(rec) -> str:
    """A record's legend entry: its cell file's stem, else its name, else its id. ``inputs.cell`` is
    None for a record written without a configuration, so the fallbacks are not decoration."""
    cell = (rec.manifest.inputs or {}).get("cell")
    if isinstance(cell, dict) and cell.get("path"):
        return Path(str(cell["path"])).stem
    return rec.name or rec.id


def load_records(store, refs, mode: str) -> list:
    """The records ``refs`` names, checked against ``mode``'s own rules before anything is drawn.

    Refuses: too few records for the mode, a record of the wrong study, a record whose run did not
    finish (E2 leaves those on disk with a valid manifest, so both front ends can offer one), and a
    record with no numbers beside its manifest.
    """
    store = resolve_store(store)
    want_study, need = MODE_RULES[mode]
    refs = [str(r) for r in refs]
    if len(refs) < need:
        refuse("compare_records",
               f"The saved runs to compare must be at least {need} for a {mode} comparison; "
               f"got {len(refs)}.")
    out = []
    for ref in refs:
        rec = store.load_fdt(ref)
        study = rec.body.get("study")
        if study != want_study:
            refuse("compare_records",
                   f"The saved runs to compare must all be {want_study} runs for a {mode} "
                   f"comparison; {ref!r} is a {study} run.")
        if not rec.body.get("complete"):
            refuse("compare_records",
                   f"The saved runs to compare must have finished; {ref!r} did not (it was "
                   f"interrupted, or it is still running), so the numbers it holds are partial.")
        if rec.data_path is None or not Path(rec.data_path).is_file():
            refuse("compare_records",
                   f"The saved runs to compare must hold their numbers; {ref!r} has no data file "
                   f"beside its manifest, so there is nothing to draw.")
        out.append(rec)
    return out


def curve_of(rec) -> Curve:
    """A single-cell record's ratio curve, read back from its ``data.h5``."""
    with h5py.File(rec.data_path, "r") as h5:
        for key in (OMEGA_GRID, RATIO):
            if key not in h5:
                refuse("compare_records",
                       f"The saved runs to compare must hold their numbers; the data file of "
                       f"{(rec.name or rec.id)!r} has no {key!r} in it.")
        omegas = np.asarray(h5[OMEGA_GRID][...], dtype=np.float64)
        ratio = np.asarray(h5[RATIO][...], dtype=np.float64)
        omega_0 = float(h5.attrs.get("omega_0", math.nan))
        prefactor = float(h5.attrs.get("prefactor", math.nan))
    return Curve(id=rec.id, name=rec.name, label=_label_of(rec), omegas=omegas, ratio=ratio,
                 omega_0=omega_0, prefactor=prefactor)


def common_grid(curves: list) -> np.ndarray:
    """Log-spaced over the INTERSECTION of the curves' spans, with N the smallest point count among
    them (spec §7.2). Curves that share no band at all are refused: an empty picture is not an
    answer."""
    lo = max(float(np.min(c.omegas)) for c in curves)
    hi = min(float(np.max(c.omegas)) for c in curves)
    n = min(int(np.size(c.omegas)) for c in curves)
    if not (lo > 0.0) or not (hi > lo) or n < 2:
        refuse("compare_records",
               f"The saved runs to compare must overlap in frequency; the intersection of their "
               f"grids is [{lo:g}, {hi:g}] over {n} point(s).")
    return np.exp(np.linspace(math.log(lo), math.log(hi), n))


def interpolate_onto(grid, omegas, values) -> np.ndarray:
    """``values`` on ``grid``, linear in log-omega. A target point whose bracketing samples include a
    blank -- or that lies outside the source's span -- is blank.

    The blank mask is carried by an INDICATOR interpolated the same way, never by trusting NaN to
    propagate through ``np.interp``: numpy retries a non-finite result from the other bracket, so
    "the NaN comes out anyway" is not a property to lean on for the guarantee §7.2 states.
    """
    x = np.log(np.asarray(omegas, dtype=np.float64))
    y = np.asarray(values, dtype=np.float64)
    order = np.argsort(x)
    x, y = x[order], y[order]
    t = np.log(np.asarray(grid, dtype=np.float64))
    finite = np.isfinite(y)
    keep = np.interp(t, x, finite.astype(np.float64), left=0.0, right=0.0)
    out = np.interp(t, x, np.where(finite, y, 0.0), left=np.nan, right=np.nan)
    out[keep < 1.0] = np.nan
    return out


def blank_notice(values: list) -> tuple:
    """``(blanks, notices)`` -- how many common-grid points are blank in at least one curve, and the
    sentence the record keeps when there are any (spec §7.2: "the drawing says so, in the axis label
    and in the record's notices")."""
    stack = np.stack([np.asarray(v, dtype=np.float64) for v in values], axis=0)
    blanks = int((~np.isfinite(stack)).any(axis=0).sum())
    if blanks == 0:
        return 0, []
    return blanks, [f"{blanks} of {stack.shape[1]} points on the common grid are blank in at least "
                    f"one run: a point whose bracketing samples include a blank is left blank, "
                    f"never interpolated across."]


def open_record(store, mode: str, records: list, *, name: str, note: str, settings: dict):
    """The comparison's own record, CREATED and not entered.

    ``create`` mints the id and runs ``assert_name_free`` before anything is spent, and the body is
    set between ``create()`` and the ``with``: the progressive mode's first manifest carries every
    body key, so the facts already known have to be there before ``__enter__`` writes it (spec §2.2).
    """
    w = store.create("fdt", None, name=name, note=note)
    w.body = {
        "study": "comparison",
        "settings": {"mode": mode, "records": len(records), **settings},
        "seed": None,          # a comparison measures nothing; it draws what was measured
        "grid": None, "points": None, "offgrid": None,
        "notices": [],
        "compared": {"mode": mode,
                     "records": [{"kind": "fdt", "id": r.id, "name": r.name} for r in records]},
        "complete": False,
        "results": None,
    }
    return w


def write_curves(w, grid, values: list, labels: list) -> None:
    """The comparison's ``data.h5``: the common grid and one interpolated curve per record, each
    carrying the label the picture used (spec §7.3)."""
    with h5py.File(w.payload("data.h5"), "w") as h5:
        h5.attrs["study"] = "comparison"
        h5.attrs["mode"] = w.body["compared"]["mode"]
        h5.create_dataset("omega_common", data=np.asarray(grid, dtype=np.float64))
        group = h5.create_group("curves")
        for i, (label, vals) in enumerate(zip(labels, values)):
            d = group.create_dataset(f"{i:03d}", data=np.asarray(vals, dtype=np.float64))
            d.attrs["label"] = str(label)


def finish_record(w, *, results: dict, notices) -> None:
    """Fill in what the drawing found and re-write the manifest. ``complete`` is the writer's to set,
    on the clean exit (spec §2.2 step 4)."""
    w.body["results"] = results
    w.body["notices"] = list(notices)
    w.refresh()


@public_entry
def compare(mode: str, refs, *, name: str = "", note: str = "", fig_sink=None, store=None,
            **options):
    """Draw one comparison of saved ``fdt`` records and write it as a record of its own.

    :param mode: one of ``MODE_RULES`` -- cells, repeats, renormalise, sweeps.
    :param refs: the records to draw, by name or id. The arity and the study each mode needs are
                 ``MODE_RULES``', and are refused before anything is created.
    :param options: the mode's own settings (``MODE_OPTIONS``): renormalise's ``prefactor``, sweeps'
                 ``at``. A setting the mode has no use for is refused.
    :returns: the comparison record.
    """
    store = resolve_store(store)
    if mode not in MODE_RULES:
        refuse("compare_records",
               f"The saved runs to compare cannot be drawn as a {mode!r} comparison; this build "
               f"draws {', '.join(sorted(MODE_RULES))}.")
    unknown = sorted(set(options) - set(MODE_OPTIONS[mode]))
    if unknown:
        refuse("compare_records",
               f"The saved runs to compare were given settings a {mode} comparison has no use for: "
               f"{', '.join(unknown)}.")
    drawer = _DRAWERS.get(mode)
    if drawer is None:
        refuse("compare_records",
               f"The saved runs to compare cannot be drawn: this build has no {mode} comparison.")
    records = load_records(store, refs, mode)
    log.info(f"Comparing {len(records)} record(s) as a {mode} comparison: "
             f"{', '.join(r.name or r.id for r in records)}")
    w = open_record(store, mode, records, name=name, note=note, settings=dict(options))
    with w:
        results, notices = drawer(w, records, sink=w.fig_sink(fig_sink), **options)
        for sentence in notices:
            log.warning(sentence)
        finish_record(w, results=results, notices=notices)
    return store.load_fdt(w.id)
```

- [ ] **Step 8: Register the subcommand**

In `core/tool/fdt.py`, find:

```python
from .config_args import UsageError, knobs, model_from_path
```

Replace with:

```python
from .config_args import UsageError, add_name_flags, knobs, model_from_path
```

Then find (the last three lines of `register`; T28 has already edited around them):

```python
    _add_fdt_knobs(cv)
    cv.set_defaults(handler=run_crossval, interrupt_note=CROSSVAL_INTERRUPT_NOTE)
    return {"fdt": fdt, "crossval": cv}
```

Replace with:

```python
    _add_fdt_knobs(cv)
    cv.set_defaults(handler=run_crossval, interrupt_note=CROSSVAL_INTERRUPT_NOTE)
    return {"fdt": fdt, "crossval": cv, **_register_compare(subparsers)}
```

And append to the end of the module:

```python
COMPARE_EPILOG = """\
Every mode draws SAVED records from the artifact store and writes a comparison record of its own
(kind fdt, study "comparison"): its figures are its output and its data.h5 holds the common grid and
the interpolated curves. Nothing is simulated, and no record it draws is modified.

  cells        two or more single-cell runs' ratio curves on one axis, labelled by cell
  repeats      several runs of one cell, with the spread across them as a band
  renormalise  one run's ratio recomputed with --prefactor, drawn against the original
  sweeps       two sweep records together, and a slice of both at one operating point

--record is repeatable and names a record by name or id; an unfinished record is refused, naming it.
Runs land on different frequencies (each detects its own resonance), so curves are interpolated onto
a grid log-spaced over the intersection of their spans -- and a point whose bracketing samples
include a blank stays blank rather than being drawn through.
"""

_COMPARE_MODES = (
    ("cells", "two or more single-cell runs' ratio curves on one axis, labelled by cell"),
    ("repeats", "several runs of one cell, with the spread across them as a band"),
    ("renormalise", "one run's ratio recomputed with a supplied normalisation constant"),
    ("sweeps", "two sweep records together, and a slice of both at one operating point"),
)


def _register_compare(sub) -> dict:
    """``python -m core compare <mode>``: one parser per mode, the
    ``identifiability {rotation,laplace,jacobian}`` shape (core/tool/diagnostics.py), so a flag that
    means nothing to a mode is an argparse error rather than a setting silently ignored. No
    --store-root and no configuration flags, for core/tool/browse.py's reasons: the root is
    PRISM_ARTIFACTS, and --help here costs no torch import."""
    p = sub.add_parser("compare", help="compare saved FDT records (cells, repeats, renormalise, sweeps)",
                       epilog=COMPARE_EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    modes = p.add_subparsers(dest="variant", required=True,
                             metavar="{cells,repeats,renormalise,sweeps}")
    built = {}
    for name, helptext in _COMPARE_MODES:
        m = modes.add_parser(name, help=helptext)
        m.add_argument("--record", action="append", required=True, metavar="REF",
                       help="a saved fdt record, by name or id; repeat the flag once per record")
        add_name_flags(m)
        m.set_defaults(handler=run_compare)
        built[name] = m
    built["renormalise"].add_argument(
        "--prefactor", type=float, required=True, metavar="VALUE",
        help="the normalisation constant to recompute T_eff/T with")
    built["sweeps"].add_argument(
        "--at", type=float, default=None, metavar="VALUE",
        help="the operating point to slice both sweeps at (default: the middle of the range they "
             "share)")
    return {"compare": p}


def run_compare(args, store):
    """One comparison, into a record of its own. Heavy imports inside the handler, per this package's
    rule, so building the parser costs no torch import."""
    from core.FDT import compare as comparisons
    from .config_args import close_sink, report
    options = {}
    if args.variant == "renormalise":
        options["prefactor"] = args.prefactor
    elif args.variant == "sweeps" and args.at is not None:
        options["at"] = args.at
    return report(comparisons.compare(args.variant, args.record, name=args.name, note=args.note,
                                      fig_sink=close_sink, store=store, **options))
```

- [ ] **Step 9: Join the `public_entry` source scan**

`tests/test_artifact_store.py`'s scan asserts `found == want` by equality, with `_UNTOUCHED_LEGS` and
`_REFUSAL_FIELDS` closed the same way, so this cannot be deferred: the decorator in Step 7 reds the
fast gate until it lands. Tasks 17 and 19 already added `run_fdt` and `run_param_study_cli`; add the
third. Find:

```python
    want |= {("core/diagnostics/sbc.py", "sbc_repeats"), ("core/diagnostics/ablation.py", "channel_ablation")}
```

Replace with:

```python
    want |= {("core/diagnostics/sbc.py", "sbc_repeats"), ("core/diagnostics/ablation.py", "channel_ablation")}
    want |= {("core/FDT/compare.py", "compare")}
```

Add the leg beside the others (after `_leg_channel_ablation`):

```python
def _leg_compare(case, monkeypatch, tmp_path):
    """The comparison facility's leg. A comparison takes NO configuration -- it draws saved records --
    so the watched config is one the entry never receives; the leg is here because the set is closed
    against the scan, and what it still pins is real: the three endings, and that the refusal is the
    entry's own (an arity it will not draw), carrying its field. A real store in tmp_path rather than
    _EntryStore, because the writer's body is set BEFORE the `with`, which _EntryWriter answers by
    raising _BodyDone outside any block that absorbs it."""
    from core.artifacts import ArtifactStore
    from core.FDT import compare as comparisons
    from tests._fixtures import build_fdt_record
    cfg = _nad_cfg()
    store = ArtifactStore(tmp_path / "compare_leg")
    ids = [build_fdt_record(store, name=f"leg_{i}") for i in range(2)]

    def _stub(w, records, *, sink):
        if case == "boom":
            _raise_injected()
        return {"n_records": len(records)}, []

    monkeypatch.setitem(comparisons._DRAWERS, "cells", _stub)
    refs = ids[:1] if case == "refusal" else ids
    return cfg, lambda: comparisons.compare("cells", refs, store=store)
```

Then add `"compare": _leg_compare,` to `_UNTOUCHED_LEGS` and `"compare": "compare_records",` to
`_REFUSAL_FIELDS`, and correct the count word in the scan test's name and docstring (it reads
"fifteen" at `a0d85da`; Tasks 17 and 19 have raised it twice already — set it to the number the set
now holds, and leave the name alone if an earlier task already made it countless).

- [ ] **Step 10: Run the tests and watch them pass**

Run: `pytest tests/test_fdt_compare.py tests/test_refusals.py -q`
then: `pytest tests/test_artifact_store.py -q -k "public_entry or untouched"`
Expected: PASS.

- [ ] **Step 11: Commit**

```bash
git add core/FDT/compare.py core/refusals.py core/gui/fields.py core/tool/fields.py core/tool/fdt.py tests/_fixtures.py tests/test_fdt_compare.py tests/test_refusals.py tests/test_artifact_store.py
git commit -m "compare: the comparison record, the common grid and its refusals"
```

#### Amendments (binding — these supersede the text above)

Land this task only after T1, T3, T5, T8, T9, T17, T18, T19, T22, T23, T28 and T29. It edits the three field tables and the registry pins that T8/T22/T23 already moved, and the `public_entry` scan that T17/T19 already extended. Rulings applied: P11, P20, P30, P49, P65, P71. (SC) marks a fix for a contradiction inside this task.

**A1 — Step 7, `load_records` names the RECORD, not the ref (SC; Review Focus 5; spec §7.1 "refused, naming it").** The three per-record refusals quote `{ref!r}`, which is whatever the caller passed. The tests pass the ids `build_fdt_record` returns, so `"'sw'" in str(e.value)` and `"'half'" in str(e.value)` can never hold. Replace the whole `for ref in refs:` loop with:
```python
    for ref in refs:
        rec = store.load_fdt(ref)
        who = rec.name or rec.id
        study = rec.body.get("study")
        if study != want_study:
            refuse("compare_records",
                   f"The saved runs to compare must all be {want_study} runs for a {mode} "
                   f"comparison; {who!r} is a {study} run.")
        if not rec.body.get("complete"):
            refuse("compare_records",
                   f"The saved runs to compare must have finished; {who!r} did not (it was "
                   f"interrupted, or it is still running), so the numbers it holds are partial.")
        if rec.data_path is None or not Path(rec.data_path).is_file():
            refuse("compare_records",
                   f"The saved runs to compare must hold their numbers; {who!r} has no data file "
                   f"beside its manifest, so there is nothing to draw.")
        out.append(rec)
```

**A2 — Step 7, `compare`: P11 ("a mode with no drawer is refused, naming what this build draws").** Replace
```python
               f"The saved runs to compare cannot be drawn: this build has no {mode} comparison.")
```
with
```python
               f"The saved runs to compare cannot be drawn: this build has no {mode} comparison "
               f"(it draws {', '.join(sorted(_DRAWERS)) or 'none yet'}).")
```

**A3 — Step 2, `test_a_comparison_refuses_one_record_another_study_and_an_unfinished_one` (SC).** At this task `_DRAWERS` is empty, so `compare` refuses at the no-drawer guard before `load_records` runs, and the "at least 2" / "'sw'" / "'half'" assertions fail. Change the signature to `(tmp_path, monkeypatch)` and insert immediately after `store = _store(tmp_path)`:
```python
    # these refusals are load_records'; a stub drawer gets past the no-drawer guard, which answers
    # first while _DRAWERS is still empty (Tasks 37-39 fill it)
    monkeypatch.setitem(cmp._DRAWERS, "cells", lambda w, records, *, sink: ({}, []))
```

**A4 — Step 3's expected failure (SC).** The from-import form reports the missing module as `ImportError: cannot import name 'compare' from 'core.FDT'`, not `ModuleNotFoundError`.

**A5 — Step 4, `core/gui/fields.py` anchor (P30).** P30 has T9 widen `CONTROL["note"]`, so the line `"note": "Edit it in the Note box on the Artifacts screen.",` may no longer be verbatim. Find the line `    "artifact": "Select an artifact in the list on the Artifacts screen.",`. The `"note"` entry follows it, possibly now spanning several lines. Insert the `compare_records` comment and entry immediately AFTER the complete `"note"` entry. The `core/refusals.py` and `core/tool/fields.py` anchors are found as written: T8 and T22 insert below them and keep them.

**A6 — Step 5, the `BASE_KEYS` anchor is NOT found, and the count (P20).** After T8, the line `"artifact", "note",` is followed by T8's comment and keys, not by `)`. Find instead:
```python
)
TOOL_ONLY_KEYS = ("repeats",
```
That first `)` closes `BASE_KEYS`. Insert above it:
```python
    # piece 5: comparing saved FDT records (E8)
    "compare_records",
```
Do NOT add one to the count blind. Run `pytest tests/test_refusals.py -q -k registry_holds`, read the real `len(FIELDS)` off the failure, and write it in place of the literal on the line ending `"a key is listed twice above"`. Expect 86 if T8 registered thirteen keys (P75), T22 seven and T23 two. Strike the sentence naming "Tasks 8, 12, 13, 23, 38 and 39": the keys added before this task are T8's, T22's and T23's; T12 and T13 add none; T38 and T39 come after this task.

**A7 — Step 6's run command (SC).** No test is named `the_tool_table`; the flag pin is `test_every_field_key_has_a_flag_and_every_key_literal_under_core_is_registered`. Run `pytest tests/test_refusals.py -q -k "registry_holds or every_field_key_has_a_flag"`.

**A8 — Step 9, against the post-T17/T19 file.** The `want |= {("core/diagnostics/sbc.py", ...` anchor is found as written. Put `_leg_compare` immediately ABOVE `_UNTOUCHED_LEGS = {`: T17's `_FdtWriter`/`_leg_run_fdt` and T19's `_leg_run_param_study_cli` already sit after `_leg_channel_ablation`. Add `    "compare": _leg_compare,` immediately after `    "run_param_study_cli": _leg_run_param_study_cli,`. Add `    "compare": "compare_records",` immediately after T19's `    "run_param_study_cli": "cell", ...` line. T17 renamed the test to `test_the_public_entries_carry_public_entry_and_nothing_else_does`, with no count. Leave the name, and extend its docstring's list of decorated functions with "and the comparison facility's one entry, core/FDT/compare.py". Step 10's `-k "public_entry or untouched"` still selects both tests.

**A9 — already correct; leave these unchanged.** `curve_of` reads the contract's single-cell names (`omega_grid`, `T_eff_over_T`, root attrs `omega_0`, `prefactor`; P5, P71). If T18 still writes `omegas`/`chi`, that is a T18 defect; report it, and do not rename `OMEGA_GRID`/`RATIO`. The comparison layout (`omega_common`, `curves/<NNN>` with a `label` attr) is P71's. `compare` refuses everything it can before `store.create` (P49). The leg uses a real `ArtifactStore`, not `_EntryWriter` (P65).

#### Controller rulings (pre-flight, binding — these supersede the amendments above and the task text)

- **F21/F22** — add to `tests/test_tool.py`'s tiny-size test (slow-marked) the read of the REAL single-cell record: `cmp.curve_of(store.load_fdt(<the single-cell record's id>))` succeeds. That is the only check that a real record, not the fixture, is readable by `compare` (P71).
- **F55** — a comparison's `data.h5` writes root `omega_0` and `prefactor` as NaN (not applicable), and each `curves/<NNN>` dataset carries its source record's `omega_0` and `prefactor` as attributes.
- **F56** — add a `COMPARE_INTERRUPT_NOTE` through `set_defaults(interrupt_note=...)`: the unfinished comparison record stays on disk and re-running writes a new one; there is no resume.
- **F57** — `compare()` checks each mode's options before `open_record` (renormalise: `require_positive("prefactor", ...)`; sweeps: `require_finite("slice_at", at)` when given). Refuse before the spend; the drawers keep theirs as a backstop.
- **F58** — `load_records` catches the store's missing-record `StoreError` and re-raises `refuse("compare_records", f"The saved runs to compare must exist; {ref!r} names no fdt record.")`, so the control that answers it is named.
- **F59** — `MODE_RULES` carries `(study, fewest, most)`: renormalise `(1, 1)`, sweeps `(2, 2)`; more than `most` is refused with `compare_records`, never silently dropped.
- **F60** — `build_fdt_record` writes one figure through `w.figure_path`, as spec §8.1 says.
- **F62** — correct the docstring that says the window can offer an unfinished record: only the tool can.

---

### Task 37: compare cells, and compare repeats

**Why:** Spec §7.1's first two modes — "the ratio curves of two or more single-cell records on one
axis, labelled by cell" and "several records of the same cell, with the spread across them shown as
a band". E7 makes the spread meaningful: repeats of a cell use different seeds, and their spread is
the measurement error.

**Files:**
- Modify: `core/FDT/compare.py` — append the two drawers and the shared axes helper
- Test: `tests/test_fdt_compare.py`, `tests/test_tool.py`

**Interfaces:**
- Consumes: `curve_of`, `common_grid`, `interpolate_onto`, `blank_notice`, `write_curves`,
  `_DRAWERS`, `BLANK_AXIS_NOTE`, `_num` (Task 36).
- Produces: `_DRAWERS["cells"]`, `_DRAWERS["repeats"]`, and `peak_of(curve, grid, values) -> dict`
  and `ratio_axes(title, blanks) -> (fig, ax)`, which Tasks 38 and 39 reuse.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_fdt_compare.py`:

```python
def test_compare_cells_draws_every_record_on_one_axis_and_records_its_peak(tmp_path):
    """Spec §7.1: several cells' ratio curves on ONE axis, labelled by cell. The labels matter as much
    as the curves -- a picture of four unlabelled traces answers nothing -- and the peak of each is in
    the record, so the comparison can be read back without re-opening the figure. The two records here
    measure different bands on purpose: the drawn grid is their intersection (§7.2), not either one's
    own."""
    store = _store(tmp_path)
    a = build_fdt_record(store, name="master_spont", omegas=(0.5, 1.0, 2.0, 4.0),
                         ratio=(1.0, 5.0, 1.3, 1.1))
    b = build_fdt_record(store, name="master_weak", omegas=(1.0, 2.0, 4.0, 8.0),
                         ratio=(1.0, 2.0, 1.2, 1.05))
    seen, sink = _closing()
    rec = cmp.compare("cells", [a, b], name="two_cells", fig_sink=sink, store=store)

    assert seen == ["FDT ratio by cell"], seen
    assert rec.manifest.figures == ["figures/fdt_ratio_by_cell.png"]
    res = rec.body["results"]
    assert res["n_records"] == 2 and res["n_grid"] == 4 and res["blanks"] == 0
    assert [p["label"] for p in res["per_record"]] == ["master_spont", "master_weak"]
    # the common grid is [1, 4]: master_spont's peak at omega=1 is on it, master_weak's at omega=2
    assert res["per_record"][0]["peak_omega"] == pytest.approx(1.0)
    assert res["per_record"][1]["peak_omega"] == pytest.approx(2.0)
    assert res["per_record"][1]["peak_ratio"] == pytest.approx(2.0)
    with h5py.File(rec.data_path, "r") as h5:
        assert [h5["curves"][k].attrs["label"] for k in sorted(h5["curves"])] == \
            ["master_spont", "master_weak"]


def test_compare_cells_keeps_a_blank_blank_and_says_so_in_the_record(tmp_path):
    """E9 carried through a comparison. A run that could not measure a frequency reports a blank, and
    the comparison must not blend it away: the blank stays blank, the count is in the record's
    notices, and the notices are what the walkthrough row reads. A silently interpolated gap is the
    fabricated tail the off-grid fix removed, put back one layer up."""
    store = _store(tmp_path)
    a = build_fdt_record(store, name="a", omegas=(1.0, 2.0, 4.0, 8.0),
                         ratio=(1.0, float("nan"), 1.3, 1.1))
    b = build_fdt_record(store, name="b", omegas=(1.0, 2.0, 4.0, 8.0), ratio=(1.0, 2.0, 1.2, 1.05))
    _seen, sink = _closing()
    rec = cmp.compare("cells", [a, b], fig_sink=sink, store=store)
    assert rec.body["results"]["blanks"] == 1, rec.body["results"]
    assert rec.body["notices"] and "never interpolated across" in rec.body["notices"][0]
    with h5py.File(rec.data_path, "r") as h5:
        drawn = h5["curves"]["000"][...]
    assert math.isnan(drawn[1]) and not np.isnan(drawn[[0, 2, 3]]).any(), drawn


def test_compare_repeats_draws_the_spread_across_the_runs_as_a_band(tmp_path):
    """Spec §7.1 and E7: repeats of one cell run at different seeds, and the spread across them IS the
    measurement error -- the number that says whether a difference between two cells means anything.
    The band is the envelope across the repeats at each frequency, so a point blank in ANY repeat is
    blank in the band: an envelope silently narrowed by a missing run would understate exactly the
    error it exists to show."""
    store = _store(tmp_path)
    ids = [build_fdt_record(store, name=f"rep{i}", omegas=(1.0, 2.0, 4.0), ratio=r)
           for i, r in enumerate([(1.0, 3.0, 1.0), (1.2, 5.0, 1.1), (0.8, 4.0, 0.9)])]
    seen, sink = _closing()
    rec = cmp.compare("repeats", ids, fig_sink=sink, store=store)

    assert seen == ["FDT ratio across repeats"], seen
    res = rec.body["results"]
    assert res["n_records"] == 3
    # at omega = 2 the three repeats give 3, 5 and 4: the band spans 3 to 5 and the mean is 4
    assert res["band"]["lo"][1] == pytest.approx(3.0)
    assert res["band"]["hi"][1] == pytest.approx(5.0)
    assert res["band"]["mean"][1] == pytest.approx(4.0)
    assert res["widest"] == pytest.approx(2.0), "the widest spread over the grid"
    with h5py.File(rec.data_path, "r") as h5:
        assert sorted(h5["curves"]) == ["000", "001", "002"]
        assert list(h5["band"]) == ["hi", "lo", "mean"]
```

And append to `tests/test_tool.py`, beside the other `fdt`/`crossval` subcommand tests:

```python
def test_compare_cells_from_the_command_line_writes_a_record_and_names_it(tool_env, capsys):
    """``python -m core compare cells --record A --record B``: the tool's half of E8. The subcommand
    mirrors identifiability's shape -- a parser per mode -- reads PRISM_ARTIFACTS like every
    subcommand but smoke, and prints the one [prism] line per artifact written that every other
    write path prints. One record is a refusal, not a traceback: it exits 1 and names --record."""
    from core.artifacts import ArtifactStore
    from core.tool import main
    from tests._fixtures import build_fdt_record
    store = ArtifactStore(tool_env)
    a = build_fdt_record(store, name="cell_a")
    b = build_fdt_record(store, name="cell_b")

    assert main(["compare", "cells", "--record", a, "--record", b, "--name", "ab"]) == 0
    out = capsys.readouterr().out
    assert "[prism] fdt ab__" in out, out
    rec = store.load_fdt("ab")
    assert rec.body["study"] == "comparison" and rec.body["compared"]["mode"] == "cells"

    assert main(["compare", "cells", "--record", a]) == 1
    err = capsys.readouterr().err
    assert "refused:" in err and "at least 2" in err and "(--record)" in err, err
```

- [ ] **Step 2: Run them and watch them fail**

Run: `pytest tests/test_fdt_compare.py -q -k "cells or repeats"`
Expected: FAIL — `core.refusals.Refusal: The saved runs to compare cannot be drawn: this build has
no cells comparison.` (Task 36's guard; `_DRAWERS` is still empty.)

- [ ] **Step 3: Add the shared axes helper and the peak summary**

Append to `core/FDT/compare.py`:

```python
def peak_of(curve: Curve, grid, values) -> dict:
    """One record's line in ``body.results.per_record``: where its ratio peaked on the common grid,
    and how high. Every float goes through ``_num``, so an all-blank curve records nulls rather than
    reaching the manifest writer, which refuses a non-finite number outright."""
    vals = np.asarray(values, dtype=np.float64)
    if not np.isfinite(vals).any():
        return {"id": curve.id, "label": curve.label, "peak_ratio": None, "peak_omega": None}
    i = int(np.nanargmax(vals))
    return {"id": curve.id, "label": curve.label,
            "peak_ratio": _num(vals[i]), "peak_omega": _num(np.asarray(grid)[i])}


def ratio_axes(title: str, blanks: int):
    """A figure and axes for T_eff/T against frequency, drawn the way ``plots.plot_eff_temp_ratio``
    draws its own: log x, symlog y so the chi''=0 crossing is visible on both sides of zero, and the
    two reference lines in ``axes.edgecolor`` -- figure CHROME follows the theme, which is why it is
    not a hardcoded gray (core/FDT/plots.py says why at length)."""
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.axhline(1.0, color=plt.rcParams["axes.edgecolor"], linestyle="--", linewidth=0.8,
               label=r"$T_{\rm eff}/T = 1$ (equilibrium)")
    ax.axhline(0.0, color=plt.rcParams["axes.edgecolor"], linestyle=":", linewidth=0.6)
    xlabel = r"$\tilde\omega$ (ND), log scale"
    if blanks:
        xlabel += f"  --  {BLANK_AXIS_NOTE}"
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$T_{\rm eff}(\tilde\omega) / T$, symlog scale")
    ax.set_title(title)
    ax.grid(False)
    return fig, ax
```

- [ ] **Step 4: Add the two drawers**

Append to `core/FDT/compare.py`:

```python
def draw_cells(w, records, *, sink):
    """Several single-cell records' ratio curves on one axis, labelled by cell (spec §7.1)."""
    curves = [curve_of(r) for r in records]
    grid = common_grid(curves)
    values = [interpolate_onto(grid, c.omegas, c.ratio) for c in curves]
    blanks, notices = blank_notice(values)
    write_curves(w, grid, values, [c.label for c in curves])
    fig, ax = ratio_axes("FDT ratio by cell", blanks)
    for curve, vals in zip(curves, values):
        ax.plot(grid, vals, marker="o", markersize=4, linewidth=1.0, label=curve.label)
    ax.legend()
    fig.tight_layout()
    sink("FDT ratio by cell", fig)
    results = {"n_records": len(curves), "n_grid": int(grid.size), "blanks": blanks,
               "per_record": [peak_of(c, grid, v) for c, v in zip(curves, values)]}
    log.info(f"Common grid: {grid.size} pts spanning [{grid[0]:.4f}, {grid[-1]:.4f}]; "
             f"{blanks} blank point(s)")
    return results, notices


def draw_repeats(w, records, *, sink):
    """Repeats of one cell, with the spread ACROSS them as a band (spec §7.1, E7).

    The band is the envelope -- the lowest and the highest repeat at each frequency -- and not a
    standard deviation: with the two or three repeats this is written for, an SD is a number computed
    from too little to mean what its name promises, while the envelope is exactly "what the runs
    disagreed by". Plain ``min``/``max`` over the stack, so a point blank in ANY repeat is blank in
    the band: an envelope narrowed by a missing run would understate the error it exists to show.
    """
    curves = [curve_of(r) for r in records]
    grid = common_grid(curves)
    values = [interpolate_onto(grid, c.omegas, c.ratio) for c in curves]
    blanks, notices = blank_notice(values)
    stack = np.stack(values, axis=0)
    lo, hi, mean = stack.min(axis=0), stack.max(axis=0), stack.mean(axis=0)
    write_curves(w, grid, values, [c.label for c in curves])
    with h5py.File(w.payload("data.h5"), "a") as h5:
        band = h5.create_group("band")
        for key, arr in (("lo", lo), ("hi", hi), ("mean", mean)):
            band.create_dataset(key, data=np.asarray(arr, dtype=np.float64))
    fig, ax = ratio_axes("FDT ratio across repeats", blanks)
    ax.fill_between(grid, lo, hi, alpha=0.25, color="steelblue", label="spread across the repeats")
    ax.plot(grid, mean, color="steelblue", linewidth=1.4, label="mean of the repeats")
    for curve, vals in zip(curves, values):
        ax.plot(grid, vals, linewidth=0.6, alpha=0.7, label=curve.label)
    ax.legend()
    fig.tight_layout()
    sink("FDT ratio across repeats", fig)
    widest = hi - lo
    results = {"n_records": len(curves), "n_grid": int(grid.size), "blanks": blanks,
               "widest": _num(np.nanmax(widest)) if np.isfinite(widest).any() else None,
               "band": {"lo": [_num(v) for v in lo], "hi": [_num(v) for v in hi],
                        "mean": [_num(v) for v in mean]},
               "per_record": [peak_of(c, grid, v) for c, v in zip(curves, values)]}
    return results, notices


_DRAWERS["cells"] = draw_cells
_DRAWERS["repeats"] = draw_repeats
```

- [ ] **Step 5: Run the tests and watch them pass**

Run: `pytest tests/test_fdt_compare.py -q` then
`pytest tests/test_tool.py -q -k compare`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add core/FDT/compare.py tests/test_fdt_compare.py tests/test_tool.py
git commit -m "compare: cells on one axis, and repeats with their spread"
```

#### Amendments (binding — these supersede the text above)

Land this task after T36 (with T36's amendments). Rulings applied: P60, and P11 through T36-A2. (SC) marks a fix for a contradiction inside this task.

**A1 — Step 4, `draw_repeats` reports the cells it drew (P60).** P60 says the mode "reports the cells it drew in the drawing and in `body.notices`, and refuses only on arity". The legend already labels each curve by its cell; the notices do not name the cells yet. In `draw_repeats`, immediately after `blanks, notices = blank_notice(values)`, add:
```python
    # P60: "the same cell" is not enforced -- a record's cell lives in its manifest's inputs and
    # nothing stops a caller passing two -- so the mode REPORTS which cells it drew: in the legend
    # (each curve is labelled by its cell) and here, in the record's notices.
    cells = sorted({c.label for c in curves})
    notices = list(notices) + [
        f"The repeats drawn come from {len(cells)} cell(s): {', '.join(cells)}. The band is the "
        f"spread across these runs, and it is one cell's measurement error only when every run is "
        f"of the same cell."]
```
Also add `"cells": cells,` to `draw_repeats`' `results` dict. Then append to `test_compare_repeats_draws_the_spread_across_the_runs_as_a_band`:
```python
    assert res["cells"] == ["rep0", "rep1", "rep2"], "no cell input on the fixture: the label is the name"
    assert any("rep0, rep1, rep2" in n for n in rec.body["notices"]), rec.body["notices"]
```

**A2 — Step 1, `test_compare_cells_draws_every_record_on_one_axis_and_records_its_peak` (SC).** With `b`'s grid `(1.0, 2.0, 4.0, 8.0)`, the common grid is four log-spaced points on [1, 4]: [1.0, 1.587, 2.520, 4.0]. omega = 2 is not on it, and `master_weak` interpolates to [1.0, 1.667, 1.733, 1.2], so the `peak_omega`/`peak_ratio == 2.0` assertions fail. Replace `b` with:
```python
    b = build_fdt_record(store, name="master_weak", omegas=(1.0, 2.0, 4.0),
                         ratio=(1.0, 2.0, 1.2))
```
Replace `res["n_grid"] == 4` with `res["n_grid"] == 3`. Change the comment to: "the common grid is [1, 2, 4] -- the intersection [1, 4] at the smaller point count, 3: master_spont's peak at omega=1 is on it, master_weak's at omega=2". The computed curves are master_spont [5.0, 1.3, 1.1] and master_weak [1.0, 2.0, 1.2].

**A3 — Step 1, the tool test (SC).** `tool_env` yields `(bounds, cell, root)`, not a path, so `ArtifactStore(tool_env)` raises `TypeError`. Replace `store = ArtifactStore(tool_env)` with:
```python
    _bounds, _cell, root = tool_env
    store = ArtifactStore(root)
```
`main` opens `config.artifacts_root()`, and `tool_env` points that at `root`.

**A4 — Step 2's expected failure (P11, through T36-A2).** The failure now reads `core.refusals.Refusal: The saved runs to compare cannot be drawn: this build has no cells comparison (it draws none yet).`

#### Controller rulings (pre-flight, binding — these supersede the amendments above and the task text)

- **F61** — keep addressing the comparison record by the test's own `--name`: P56 concerns names a run chooses for itself, and this name is the thing the test pins.

---

### Task 38: compare renormalise

**Why:** Spec §7.1's third mode — "one record's ratio recomputed with a normalisation constant
supplied on the command line, drawn against the original". The normalisation constant is the one
input a measurement cannot re-derive for itself (`observable_noise_prefactor` reads it from the cell,
spec §3.4), so being able to ask "what would this run have said with a different one" without
re-simulating is what makes an hours-long run answerable.

**Files:**
- Modify: `core/FDT/compare.py` — append the drawer
- Modify: `core/refusals.py` — the `FIELDS` tuple, after `Field("compare_records", …)`
- Modify: `core/gui/fields.py` — `CONTROL`, after `"compare_records"`
- Modify: `core/tool/fields.py` — `FLAG`, after `"compare_records"`
- Modify: `tests/test_refusals.py` — `BASE_KEYS` and the registry count
- Test: `tests/test_fdt_compare.py`

**Interfaces:**
- Consumes: `curve_of`, `write_curves`, `ratio_axes`, `peak_of`, `_num`, `_DRAWERS`,
  `MODE_OPTIONS["renormalise"] == ("prefactor",)`, `--prefactor` (Task 36),
  `core.refusals.require_positive`.
- Produces: `_DRAWERS["renormalise"]`, the field key `prefactor` with the flag `--prefactor`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_fdt_compare.py`:

```python
def test_compare_renormalise_rescales_one_run_and_draws_it_against_the_original(tmp_path):
    """Spec §7.1. T_eff/T is LINEAR in the normalisation constant -- spectral.eff_temp_ratio is
    ``prefactor * omega * G / (4 chi'')`` -- so recomputing it with another constant is an exact
    rescaling of what the record already holds, and nothing is re-simulated. That is why the run
    records the prefactor it used (spec §2.3): without it the stored ratio cannot be undone, and this
    mode would have to guess. A record that carries no usable constant is refused, naming it, rather
    than silently rescaling from a NaN."""
    store = _store(tmp_path)
    one = build_fdt_record(store, name="cellA", omegas=(1.0, 2.0, 4.0), ratio=(1.0, 4.0, 1.5),
                           prefactor=2.0)
    seen, sink = _closing()
    rec = cmp.compare("renormalise", [one], prefactor=3.0, fig_sink=sink, store=store)

    assert seen == ["FDT ratio renormalised"], seen
    assert rec.body["settings"]["prefactor"] == 3.0
    res = rec.body["results"]
    assert res["prefactor"] == 3.0 and res["prefactor_recorded"] == 2.0
    assert res["scale"] == pytest.approx(1.5)
    with h5py.File(rec.data_path, "r") as h5:
        labels = [h5["curves"][k].attrs["label"] for k in sorted(h5["curves"])]
        renormalised = h5["curves"]["001"][...]
    assert labels == ["cellA (recorded, 2)", "cellA (renormalised, 3)"], labels
    assert renormalised == pytest.approx([1.5, 6.0, 2.25])

    # a blank box and a non-positive constant are both refused, by name, before anything is created
    for bad in (None, 0.0, -1.0):
        with pytest.raises(Refusal) as e:
            cmp.compare("renormalise", [one], prefactor=bad, store=store)
        assert e.value.field == "prefactor", (bad, e.value.field)
    assert len(store.list("fdt")) == 2, "a refused comparison leaves no record behind"

    no_pref = build_fdt_record(store, name="cellB", prefactor=float("nan"))
    with pytest.raises(Refusal) as e:
        cmp.compare("renormalise", [no_pref], prefactor=3.0, store=store)
    assert e.value.field == "compare_records" and "'cellB'" in str(e.value), e.value
```

- [ ] **Step 2: Run it and watch it fail**

Run: `pytest tests/test_fdt_compare.py -q -k renormalise`
Expected: FAIL — `core.refusals.Refusal: The saved runs to compare cannot be drawn: this build has
no renormalise comparison.`

- [ ] **Step 3: Register the field key**

In `core/refusals.py`, find:

```python
    # comparing saved FDT records (piece 5, E8): which records a comparison was asked to draw
    Field("compare_records", "the saved runs to compare", None),
```

Replace with:

```python
    # comparing saved FDT records (piece 5, E8): which records a comparison was asked to draw, and
    # the normalisation constant the renormalise mode recomputes the ratio with
    Field("compare_records", "the saved runs to compare", None),
    Field("prefactor", "the normalisation constant", None),
```

In `core/gui/fields.py`, after the `"compare_records"` entry, add:

```python
    "prefactor": "Set it in the 'Normalisation constant' box on the FDT analysis tab.",
```

In `core/tool/fields.py`, after the `"compare_records"` entry, add:

```python
    "prefactor": "--prefactor",
```

In `tests/test_refusals.py`, add `"prefactor"` to `BASE_KEYS` beside `"compare_records"` and raise
the registry count by one — **re-read the number in the file first**; it has moved.

- [ ] **Step 4: Add the drawer**

Append to `core/FDT/compare.py`:

```python
def draw_renormalise(w, records, *, sink, prefactor):
    """One record's ratio recomputed with a supplied normalisation constant, drawn against the
    original (spec §7.1).

    An exact rescaling, not a re-measurement: ``spectral.eff_temp_ratio`` is
    ``prefactor * omega * G / (4 chi'')``, linear in the constant, so the new curve is the recorded
    one times the ratio of the two constants and nothing is simulated. The record's OWN grid is the
    common grid here -- there is one record, so there is nothing to interpolate onto, and
    interpolating a curve onto a regenerated copy of its own grid would only add float error.
    """
    want = require_positive("prefactor", prefactor)
    curve = curve_of(records[0])
    if not (math.isfinite(curve.prefactor) and curve.prefactor > 0):
        refuse("compare_records",
               f"The saved runs to compare must record the normalisation constant they used; "
               f"{(records[0].name or records[0].id)!r} does not, so its ratio cannot be "
               f"recomputed with another one.")
    scale = want / curve.prefactor
    grid = np.asarray(curve.omegas, dtype=np.float64)
    original = np.asarray(curve.ratio, dtype=np.float64)
    renormalised = original * scale
    labels = [f"{curve.label} (recorded, {curve.prefactor:g})",
              f"{curve.label} (renormalised, {want:g})"]
    values = [original, renormalised]
    blanks, notices = blank_notice(values)
    write_curves(w, grid, values, labels)
    fig, ax = ratio_axes("FDT ratio renormalised", blanks)
    for label, vals, style in zip(labels, values, ("--", "-")):
        ax.plot(grid, vals, style, marker="o", markersize=3, linewidth=1.0, label=label)
    ax.legend()
    fig.tight_layout()
    sink("FDT ratio renormalised", fig)
    log.info(f"Renormalised {curve.label} from {curve.prefactor:g} to {want:g} "
             f"(x{scale:g}); nothing was re-simulated")
    results = {"n_records": 1, "n_grid": int(grid.size), "blanks": blanks,
               "prefactor": _num(want), "prefactor_recorded": _num(curve.prefactor),
               "scale": _num(scale),
               "per_record": [peak_of(curve, grid, renormalised)]}
    return results, notices


_DRAWERS["renormalise"] = draw_renormalise
```

and extend the module's import of the rules, finding:

```python
from core.refusals import refuse
```

replacing with:

```python
from core.refusals import refuse, require_positive
```

- [ ] **Step 5: Run the tests and watch them pass**

Run: `pytest tests/test_fdt_compare.py tests/test_refusals.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add core/FDT/compare.py core/refusals.py core/gui/fields.py core/tool/fields.py tests/test_fdt_compare.py tests/test_refusals.py
git commit -m "compare: one run renormalised against itself"
```

#### Amendments (binding — these supersede the text above)

**A1 — dependency.** The drawer calls `ratio_axes` and `peak_of`, which T37 adds. The task-order line `T38 compare renormalise ── T36` is incomplete: land this task after T36 AND T37.

**A2 — Step 2's expected failure (P11, through T36-A2).** The failure now reads `core.refusals.Refusal: The saved runs to compare cannot be drawn: this build has no renormalise comparison (it draws cells, repeats).`

**A3 — Step 3, `tests/test_refusals.py` (P20).** Add `"prefactor",` immediately after the `"compare_records",` line that T36 put last in `BASE_KEYS`. Do not raise the count blind. Run `pytest tests/test_refusals.py -q -k registry_holds` and write the number the failure reports on the line ending `"a key is listed twice above"` (87 if T36 left 86).

**A4 — Step 3, `core/gui/fields.py` (P34).** Keep the sentence entry exactly as written. T40 builds the 'Normalisation constant' box and turns this entry into `("FDT analysis", "Normalisation constant")`. The rendered sentence is byte-identical, so nothing here changes.

**A5 — Step 1's test comment (SC).** A blank or non-positive constant is refused INSIDE the writer block: the record is created, then removed by the writer (T3 removes a progressive record whose `Refusal` came before any payload or figure). That is why `len(store.list("fdt")) == 2` holds. Change the comment `# a blank box and a non-positive constant are both refused, by name, before anything is created` to `# a blank box and a non-positive constant are both refused, by name, and leave no record behind`.

---

### Task 39: compare sweeps

**Why:** Spec §7.1's fourth mode — "two sweep records together, and a slice of both at a common
operating point". A sweep's own record already holds the layout `load_param_sweep` reads (spec §2.3,
E1), and `core/Reduction/plots.plot_cross_validation_3d` is the only function in the tree written to
consume that shape, so the read-back it defines is what this mode draws from.

**Files:**
- Modify: `core/FDT/compare.py` — append the drawer
- Modify: `core/refusals.py`, `core/gui/fields.py`, `core/tool/fields.py` — the `slice_at` key
- Modify: `tests/_fixtures.py` — `build_fdt_record`'s sweep branch
- Modify: `tests/test_refusals.py` — `BASE_KEYS` and the registry count
- Test: `tests/test_fdt_compare.py`

**Interfaces:**
- Consumes: `cross_validation.load_param_sweep(path) -> list[dict]` with `param_value`,
  `sweep_param`, `failed`, `omega_norm`, `T_eff_over_T` per row; `cross_validation_plots._stack`;
  `common_grid`, `interpolate_onto`, `Curve`, `ratio_axes`, `write_curves`, `peak_of` (Tasks 36-37);
  `MODE_OPTIONS["sweeps"] == ("at",)` and `--at` (Task 36).
- Produces: `_DRAWERS["sweeps"]`, the field key `slice_at` with the flag `--at`.

- [ ] **Step 1: Give the fixture a sweep branch**

In `tests/_fixtures.py`, find, inside `build_fdt_record`:

```python
    try:
        with w:
            with h5py.File(w.payload("data.h5"), "w") as h5:
                h5.attrs["study"] = study
                h5.attrs["omega_0"] = float(omega_0)
                h5.attrs["prefactor"] = float(prefactor)
                h5.create_dataset("omega_grid", data=np.asarray(omegas, dtype=np.float64))
                h5.create_dataset("T_eff_over_T", data=np.asarray(ratio, dtype=np.float64))
```

Replace with:

```python
    try:
        with w:
            with h5py.File(w.payload("data.h5"), "w") as h5:
                h5.attrs["study"] = study
                h5.attrs["omega_0"] = float(omega_0)
                h5.attrs["prefactor"] = float(prefactor)
                if study == "sweep":
                    # The layout load_param_sweep reads, which is what the sweep record holds (spec
                    # §2.3): one group per operating point, each with the shared omega/omega_0 axis
                    # and its own ratio row. `points` is [(param_value, ratio-row), ...].
                    h5.attrs["sweep_param"] = str(sweep_param)
                    h5.attrs["omega_0_ref"] = float(omega_0)
                    ops = h5.create_group("operating_points")
                    for idx, (value, row) in enumerate(points or []):
                        grp = ops.create_group(f"{idx:03d}")
                        grp.attrs["param_value"] = float(value)
                        grp.attrs["sweep_param"] = str(sweep_param)
                        grp.attrs["omega_0_resonance"] = float(omega_0)
                        grp.attrs["omega_0_ref"] = float(omega_0)
                        grp.attrs["is_resonant"] = True
                        grp.attrs["failed"] = False
                        grp.create_dataset("omega_norm", data=np.asarray(omegas, dtype=np.float64))
                        grp.create_dataset("omega_grid", data=np.asarray(omegas, dtype=np.float64))
                        grp.create_dataset("T_eff_over_T", data=np.asarray(row, dtype=np.float64))
                else:
                    h5.create_dataset("omega_grid", data=np.asarray(omegas, dtype=np.float64))
                    h5.create_dataset("T_eff_over_T", data=np.asarray(ratio, dtype=np.float64))
```

and widen the signature, finding:

```python
                     omega_0=1.0, prefactor=2.0, settings=None):
```

replacing with:

```python
                     omega_0=1.0, prefactor=2.0, settings=None, sweep_param="s", points=None):
```

- [ ] **Step 2: Write the failing test**

Append to `tests/test_fdt_compare.py`:

```python
def test_compare_sweeps_draws_both_surfaces_and_one_slice_through_them(tmp_path):
    """Spec §7.1's fourth mode. Two sweeps answer "does FDT come back" along one parameter each, and
    the question this mode exists for is whether they agree -- which is read at ONE operating point,
    not off two surfaces side by side. The slice point defaults to the middle of the range the two
    sweeps share, is recorded, and each sweep contributes the row nearest it (the two grids are set
    independently, so an exact match is not something to require). Two sweeps of DIFFERENT parameters
    are refused: "the same operating point" means nothing across an S sweep and a temperature one."""
    store = _store(tmp_path)
    om = (0.5, 1.0, 2.0)
    a = build_fdt_record(store, name="s_low", study="sweep", omegas=om, sweep_param="s",
                         points=[(0.0, (1.0, 1.0, 1.0)), (0.5, (1.0, 3.0, 1.1))])
    b = build_fdt_record(store, name="s_high", study="sweep", omegas=om, sweep_param="s",
                         points=[(0.25, (1.0, 2.0, 1.0)), (0.75, (1.0, 6.0, 1.2))])
    seen, sink = _closing()
    rec = cmp.compare("sweeps", [a, b], fig_sink=sink, store=store)

    assert seen == ["Sweep surfaces", "Sweep slice"], seen
    res = rec.body["results"]
    assert res["param"] == "s"
    # the shared range is [0.25, 0.5]; its middle is 0.375, and the nearest rows are 0.5 and 0.25
    assert res["at"] == pytest.approx(0.375)
    assert [p["param_value"] for p in res["per_record"]] == [0.5, 0.25]
    assert res["per_record"][0]["peak_ratio"] == pytest.approx(3.0)
    with h5py.File(rec.data_path, "r") as h5:
        assert sorted(h5["curves"]) == ["000", "001"]
        assert h5["curves"]["000"].attrs["label"].startswith("s_low")

    # an operating point outside the range they share is refused, naming the setting
    with pytest.raises(Refusal) as e:
        cmp.compare("sweeps", [a, b], at=9.0, store=store)
    assert e.value.field == "slice_at" and "0.25" in str(e.value), e.value

    # and two different swept parameters are refused, naming both
    t = build_fdt_record(store, name="t_sweep", study="sweep", omegas=om, sweep_param="temp",
                         points=[(1.0, (1.0, 2.0, 1.0)), (1.5, (1.0, 4.0, 1.1))])
    with pytest.raises(Refusal) as e:
        cmp.compare("sweeps", [a, t], store=store)
    assert e.value.field == "compare_records" and "temp" in str(e.value), e.value
```

- [ ] **Step 3: Run it and watch it fail**

Run: `pytest tests/test_fdt_compare.py -q -k sweeps`
Expected: FAIL — `core.refusals.Refusal: The saved runs to compare cannot be drawn: this build has
no sweeps comparison.`

- [ ] **Step 4: Register the field key**

In `core/refusals.py`, after `Field("prefactor", "the normalisation constant", None),` add:

```python
    # No default string: `refuse` appends " (default X)" AFTER the caller's own period, and this
    # field's default is a sentence ("the middle of the range the two sweeps share"), not a value an
    # operator types. The flag's help says it instead.
    Field("slice_at", "the operating point to slice at", None),
```

In `core/gui/fields.py`, after the `"prefactor"` entry, add:

```python
    "slice_at": "Set it in the 'Slice at' box on the Sweep study cross-validation tab.",
```

In `core/tool/fields.py`, after the `"prefactor"` entry, add:

```python
    "slice_at": "--at",
```

In `tests/test_refusals.py`, add `"slice_at"` to `BASE_KEYS` and raise the registry count by one.

- [ ] **Step 5: Add the drawer**

Append to `core/FDT/compare.py`:

```python
def _sweep_rows(rec):
    """One sweep record's usable operating points, in ``load_param_sweep``'s own shape -- the read-back
    ``Reduction.plots.plot_cross_validation_3d`` was written against, and the only one in the tree."""
    from .cross_validation import load_param_sweep
    rows = [r for r in load_param_sweep(rec.data_path) if not r["failed"] and "T_eff_over_T" in r]
    if not rows:
        refuse("compare_records",
               f"The saved runs to compare must hold at least one measured operating point; "
               f"{(rec.name or rec.id)!r} has none that finished.")
    return rows


def draw_sweeps(w, records, *, sink, at=None):
    """Two sweep records together, and a slice of both at one operating point (spec §7.1).

    The two grids are chosen independently, so an exact shared operating point is not something to
    require: each sweep contributes the row NEAREST the slice point, and the record says which row
    that was. ``at`` defaults to the middle of the range the sweeps share; outside that range it is
    refused, because a slice each sweep answers from its own end point is not one measurement.
    """
    from .cross_validation_plots import _stack, _OMEGA_NORM_MAX
    from matplotlib import pyplot as plt

    rows = [_sweep_rows(r) for r in records]
    params = sorted({str(row["sweep_param"]) for rr in rows for row in rr})
    if len(params) != 1:
        refuse("compare_records",
               f"The saved runs to compare must sweep the SAME parameter; these sweep "
               f"{', '.join(params)}, and one operating point does not name a state of both.")
    param = params[0]
    lo = max(min(row["param_value"] for row in rr) for rr in rows)
    hi = min(max(row["param_value"] for row in rr) for rr in rows)
    if hi < lo:
        refuse("compare_records",
               f"The saved runs to compare must overlap in {param}; their ranges do not meet.")
    if at is None:
        at = 0.5 * (lo + hi)
    at = float(at)
    if not (lo <= at <= hi):
        refuse("slice_at",
               f"The operating point to slice at must be inside the range both sweeps cover "
               f"([{lo:g}, {hi:g}] in {param}); got {at:g}.")

    # the surfaces, side by side, each on its own axes
    fig = plt.figure(figsize=(12, 5))
    for i, (rec, rr) in enumerate(zip(records, rows)):
        omega_norm, values, matrix = _stack(rr, _OMEGA_NORM_MAX)
        ax = fig.add_subplot(1, len(records), i + 1)
        mesh = ax.pcolormesh(*np.meshgrid(omega_norm, values), matrix, cmap="viridis",
                             shading="auto", vmin=0.0, vmax=2.0)
        ax.axvline(1.0, color="darkorange", ls=":", lw=1.2)
        ax.axhline(at, color=plt.rcParams["axes.edgecolor"], ls="--", lw=0.8)
        ax.set_xlabel(r"$\tilde\omega / \Omega_0$")
        ax.set_ylabel(param)
        ax.set_title(_label_of(rec))
        fig.colorbar(mesh, ax=ax, label=r"$T_{\rm eff}/T$")
    fig.tight_layout()
    sink("Sweep surfaces", fig)

    # the slice: the row of each sweep nearest `at`, on one common grid
    curves, chosen = [], []
    for rec, rr in zip(records, rows):
        row = min(rr, key=lambda r: abs(float(r["param_value"]) - at))
        chosen.append(float(row["param_value"]))
        curves.append(Curve(id=rec.id, name=rec.name,
                            label=f"{_label_of(rec)}  ({param} = {row['param_value']:g})",
                            omegas=np.asarray(row["omega_norm"], dtype=np.float64),
                            ratio=np.asarray(row["T_eff_over_T"], dtype=np.float64),
                            omega_0=1.0, prefactor=math.nan))
    grid = common_grid(curves)
    values = [interpolate_onto(grid, c.omegas, c.ratio) for c in curves]
    blanks, notices = blank_notice(values)
    write_curves(w, grid, values, [c.label for c in curves])
    fig, ax = ratio_axes(f"Sweep slice at {param} = {at:g}", blanks)
    ax.set_xlabel(ax.get_xlabel().replace(r"$\tilde\omega$ (ND)", r"$\tilde\omega / \Omega_0$"))
    for curve, vals in zip(curves, values):
        ax.plot(grid, vals, marker="o", markersize=4, linewidth=1.0, label=curve.label)
    ax.legend()
    fig.tight_layout()
    sink("Sweep slice", fig)

    log.info(f"Sliced both sweeps at {param} = {at:g} (nearest rows: "
             f"{', '.join(f'{v:g}' for v in chosen)})")
    per_record = [dict(peak_of(c, grid, v), param_value=_num(p))
                  for c, v, p in zip(curves, values, chosen)]
    results = {"n_records": len(records), "n_grid": int(grid.size), "blanks": blanks,
               "param": param, "at": _num(at), "shared_range": [_num(lo), _num(hi)],
               "per_record": per_record}
    return results, notices


_DRAWERS["sweeps"] = draw_sweeps
```

- [ ] **Step 6: Run the tests and watch them pass**

Run: `pytest tests/test_fdt_compare.py tests/test_refusals.py -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add core/FDT/compare.py core/refusals.py core/gui/fields.py core/tool/fields.py tests/_fixtures.py tests/test_fdt_compare.py tests/test_refusals.py
git commit -m "compare: two sweeps and a slice through both"
```

#### Amendments (binding — these supersede the text above)

**A1 — dependency.** Step 4 anchors on T38's `Field("prefactor", ...)` and on its `"prefactor"` entries in both front-end tables. Step 5 uses T37's `ratio_axes` and `peak_of`. The order line `── T36` is incomplete: land this task after T36, T37 AND T38.

**A2 — Step 3's expected failure (P11, through T36-A2).** The failure now reads `core.refusals.Refusal: The saved runs to compare cannot be drawn: this build has no sweeps comparison (it draws cells, renormalise, repeats).`

**A3 — Step 4, `tests/test_refusals.py` (P20).** Put `"slice_at",` immediately after the `"prefactor",` line in `BASE_KEYS`. Read the count off the failing `pytest tests/test_refusals.py -q -k registry_holds` run (88 if T38 left 87); do not add one blind.

**A4 — Step 4, `core/gui/fields.py` (P34).** Keep the sentence entry. T40 builds the 'Slice at' box and converts this entry to `("Sweep study cross-validation", "Slice at")`, which renders the same sentence.

**A5 — Step 5, `draw_sweeps`: every refusal before the first write (P49).** As drafted, the surfaces figure is saved first, and `sink` calls `figure_path`, which counts as a write. The slice's `common_grid` can still refuse after that, and such a refusal leaves an unfinished record on disk that nothing can clear. Settle the slice rows and the common grid FIRST. Replace everything from the line `    # the surfaces, side by side, each on its own axes` down to and including `    sink("Sweep slice", fig)` with the block below. Leave the lines before it (the rows, parameter, range and `at` checks) and after it (`log.info`, `per_record`, `results`, `return`) unchanged. `_stack`'s middle return is renamed `param_values`, so it no longer overwrites the slice's `values`.
```python
    # P49: every refusal before the first write. The slice's rows and its common grid are settled
    # FIRST -- common_grid refuses curves that share no band -- so a refused slice leaves no record
    # behind (the writer removes a progressive record refused before any figure or payload).
    curves, chosen = [], []
    for rec, rr in zip(records, rows):
        row = min(rr, key=lambda r: abs(float(r["param_value"]) - at))
        chosen.append(float(row["param_value"]))
        curves.append(Curve(id=rec.id, name=rec.name,
                            label=f"{_label_of(rec)}  ({param} = {row['param_value']:g})",
                            omegas=np.asarray(row["omega_norm"], dtype=np.float64),
                            ratio=np.asarray(row["T_eff_over_T"], dtype=np.float64),
                            omega_0=1.0, prefactor=math.nan))
    grid = common_grid(curves)
    values = [interpolate_onto(grid, c.omegas, c.ratio) for c in curves]
    blanks, notices = blank_notice(values)

    # the surfaces, side by side, each on its own axes
    fig = plt.figure(figsize=(12, 5))
    for i, (rec, rr) in enumerate(zip(records, rows)):
        omega_norm, param_values, matrix = _stack(rr, _OMEGA_NORM_MAX)
        ax = fig.add_subplot(1, len(records), i + 1)
        mesh = ax.pcolormesh(*np.meshgrid(omega_norm, param_values), matrix, cmap="viridis",
                             shading="auto", vmin=0.0, vmax=2.0)
        ax.axvline(1.0, color="darkorange", ls=":", lw=1.2)
        ax.axhline(at, color=plt.rcParams["axes.edgecolor"], ls="--", lw=0.8)
        ax.set_xlabel(r"$\tilde\omega / \Omega_0$")
        ax.set_ylabel(param)
        ax.set_title(_label_of(rec))
        fig.colorbar(mesh, ax=ax, label=r"$T_{\rm eff}/T$")
    fig.tight_layout()
    sink("Sweep surfaces", fig)

    # the slice: the row of each sweep nearest `at`, on the common grid settled above
    write_curves(w, grid, values, [c.label for c in curves])
    fig, ax = ratio_axes(f"Sweep slice at {param} = {at:g}", blanks)
    ax.set_xlabel(ax.get_xlabel().replace(r"$\tilde\omega$ (ND)", r"$\tilde\omega / \Omega_0$"))
    for curve, vals in zip(curves, values):
        ax.plot(grid, vals, marker="o", markersize=4, linewidth=1.0, label=curve.label)
    ax.legend()
    fig.tight_layout()
    sink("Sweep slice", fig)
```
The test's `seen == ["Sweep surfaces", "Sweep slice"]` still holds.

---

### Task 40: the two screens' comparison controls

**Why:** Spec §7.1: "The FDT screen carries the first three over its picker; the CrossVal screen
carries the fourth. Because `StorePicker` is single-selection, **the comparison controls hold a list
the picker appends to** — no multi-select widget is built." E1's reading side is what makes a saved
run worth keeping; this is where the owner uses it without a terminal.

**Files:**
- Create: `core/gui/widgets/compare_list.py`
- Modify: `core/gui/panels/fdt_panel.py` — `_build_controls`'s `self.controls_layout.addWidget(box)`
- Modify: `core/gui/panels/crossval_panel.py` — `_build_controls`'s `self.controls_layout.addWidget(box)`
- Test: `tests/test_nav_and_gating.py`

**Interfaces:**
- Consumes: `FdtPanel.record_picker` and `CrossValPanel.record_picker` — the `StorePicker` over the
  `fdt` kind that Tasks 25 and 26 add, filtered to the panel's own study (T24's row predicate).
  **Re-read both panels before editing: if T25/T26 named that attribute something else, pass what
  they named.** `BasePanel.dispatch(fn, *args, provide_fig_sink=True, **kwargs)`;
  `compare.compare(mode, refs, ...)` (Task 36); `FloatField.value_or_none()`.
- Produces: `CompareList(picker, parent=None)` with `ids() -> list[str]`, `add_selected() -> bool`,
  `remove_selected()`; `FdtPanel.compare_list`, `FdtPanel.compare_mode`, `FdtPanel.renorm_prefactor`,
  `FdtPanel.btn_compare`; `CrossValPanel.compare_list`, `CrossValPanel.slice_at`,
  `CrossValPanel.btn_compare`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_nav_and_gating.py`:

```python
def test_the_comparison_list_appends_from_the_single_selection_picker():
    """Spec §7.1: StorePicker is single-selection, so the comparison controls hold a LIST the picker
    appends to and no multi-select widget is built. Adding the same record twice is not an error and
    is not a second entry -- a curve drawn twice is a curve drawn once with a fatter line -- and the
    order the list keeps is the order the legend will read."""
    from core.gui.widgets.compare_list import CompareList
    from tests._fixtures import qt_app

    qt_app()

    class _Picker:
        def __init__(self):
            self.current = ("id_a", "run A")

        def selected(self):
            return (self.current[0], self.current[0] is None)

        def selection_text(self):
            return self.current[1]

    picker = _Picker()
    lst = CompareList(picker)
    assert lst.ids() == []
    assert lst.add_selected() is True and lst.ids() == ["id_a"]
    assert lst.add_selected() is False, "the same record twice is one curve, not two"
    picker.current = ("id_b", "run B")
    assert lst.add_selected() is True and lst.ids() == ["id_a", "id_b"]
    lst.list.setCurrentRow(0)
    lst.remove_selected()
    assert lst.ids() == ["id_b"]
    picker.current = (None, "")
    assert lst.add_selected() is False, "nothing selected adds nothing"


def test_the_fdt_and_crossval_screens_dispatch_their_comparison_modes():
    """Spec §7.1: the FDT screen carries cells, repeats and renormalise over its own picker, the
    CrossVal screen carries sweeps. Each button dispatches the ONE public entry with the mode, the
    ids the list holds and the mode's own setting -- the panel reimplements nothing, so the arity,
    the study and the unfinished-record refusals are the stage's and reach the yellow box through
    _on_error. The renormalise box is read with value_or_none(), so a BLANK box is refused as blank
    rather than read as the zero every numeric box returns for one."""
    from core.FDT.compare import compare
    from core.gui.panels.crossval_panel import CrossValPanel
    from core.gui.panels.fdt_panel import FdtPanel
    from tests._fixtures import qt_app

    qt_app()
    sent = {}
    panel = FdtPanel()
    panel.dispatch = lambda fn, *a, **k: sent.update(fn=fn, args=a, kwargs=k)
    panel.compare_list.list.addItem("run A")
    panel.compare_list.list.item(0).setData(_USER_ROLE, "id_a")
    panel.compare_list.list.addItem("run B")
    panel.compare_list.list.item(1).setData(_USER_ROLE, "id_b")

    panel.compare_mode.setCurrentText("cells")
    panel.btn_compare.click()
    assert sent["fn"] is compare and sent["args"] == ("cells", ["id_a", "id_b"]), sent
    assert sent["kwargs"]["provide_fig_sink"] is True and "prefactor" not in sent["kwargs"]

    sent.clear()
    panel.compare_mode.setCurrentText("renormalise")
    panel.renorm_prefactor.setText("3.5")
    panel.btn_compare.click()
    assert sent["args"][0] == "renormalise" and sent["kwargs"]["prefactor"] == 3.5, sent

    sent.clear()
    panel.renorm_prefactor.setText("")
    panel.btn_compare.click()
    assert sent["kwargs"]["prefactor"] is None, "a blank box travels as blank; the stage refuses it"

    sent.clear()
    xv = CrossValPanel()
    xv.dispatch = lambda fn, *a, **k: sent.update(fn=fn, args=a, kwargs=k)
    xv.compare_list.list.addItem("S sweep")
    xv.compare_list.list.item(0).setData(_USER_ROLE, "id_s")
    xv.compare_list.list.addItem("T sweep")
    xv.compare_list.list.item(1).setData(_USER_ROLE, "id_t")
    xv.slice_at.setText("0.4")
    xv.btn_compare.click()
    assert sent["fn"] is compare and sent["args"] == ("sweeps", ["id_s", "id_t"]), sent
    assert sent["kwargs"]["at"] == 0.4
    sent.clear()
    xv.slice_at.setText("")
    xv.btn_compare.click()
    assert "at" not in sent["kwargs"], "a blank slice point means 'the middle of the shared range'"
```

and add, beside the module's other imports at the top of `tests/test_nav_and_gating.py`:

```python
from PySide6.QtCore import Qt as _Qt                              # noqa: E402

_USER_ROLE = _Qt.UserRole
```

- [ ] **Step 2: Run them and watch them fail**

Run: `pytest tests/test_nav_and_gating.py -q -k "comparison_list or comparison_modes"`
Expected: FAIL — `ModuleNotFoundError: No module named 'core.gui.widgets.compare_list'`.

- [ ] **Step 3: Write the list widget**

Create `core/gui/widgets/compare_list.py`:

```python
"""The comparison list: what the comparison controls hold instead of a multi-select picker.

``StorePicker`` is single-selection (one combo, one id), and spec §7.1 settles what to do about it:
the comparison controls hold a LIST the picker APPENDS to, and no multi-select widget is built. So a
record is chosen the way every other record in the application is chosen -- in the picker, with its
tooltip and its summary line -- and Add puts that choice on the list.

The list holds ids in ``Qt.UserRole`` and shows whatever the picker showed, which is the same
``<name>`` or ``(unnamed <id>)`` label the rest of the window uses. It is deliberately NOT persisted:
a remembered list would name records a later session may have deleted, and the comparison would
refuse on a choice nobody made this time.
"""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QListWidget, QListWidgetItem, QPushButton, QVBoxLayout, QWidget

_ID_ROLE = Qt.UserRole


class CompareList(QWidget):
    """A list of artifact ids, filled from a single-selection picker."""

    def __init__(self, picker, parent=None):
        super().__init__(parent)
        self._picker = picker
        self.list = QListWidget()
        self.list.setMaximumHeight(110)
        self.btn_add = QPushButton("Add selected")
        self.btn_remove = QPushButton("Remove")
        self.btn_clear = QPushButton("Clear")
        self.btn_add.clicked.connect(self.add_selected)
        self.btn_remove.clicked.connect(self.remove_selected)
        self.btn_clear.clicked.connect(self.list.clear)
        buttons = QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        for b in (self.btn_add, self.btn_remove, self.btn_clear):
            buttons.addWidget(b)
        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.addWidget(self.list)
        column.addLayout(buttons)

    def add_selected(self) -> bool:
        """Append the picker's current selection. False when there is nothing selected, or when the
        list already holds it: the same record twice is one curve drawn twice, not two records."""
        id_, _is_new = self._picker.selected()
        if id_ is None or str(id_) in self.ids():
            return False
        item = QListWidgetItem(self._picker.selection_text())
        item.setData(_ID_ROLE, str(id_))
        self.list.addItem(item)
        return True

    def remove_selected(self) -> None:
        for item in self.list.selectedItems():
            self.list.takeItem(self.list.row(item))

    def ids(self) -> list:
        """The ids, in the order the legend will read them."""
        return [str(self.list.item(i).data(_ID_ROLE)) for i in range(self.list.count())]
```

`selection_text()` is the picker's current item text. If `StorePicker` has no such method after
T24-T26, add it there beside `selection_summary()`:

```python
    def selection_text(self) -> str:
        """The current item's own text -- what a list built from this picker should show."""
        return self.combo.currentText() if self.combo.currentData() is not None else ""
```

- [ ] **Step 4: Add the FDT screen's controls**

In `core/gui/panels/fdt_panel.py`, find (the last line of `_build_controls`):

```python
        form.addRow(self.btn_run)

        self.controls_layout.addWidget(box)
```

Replace with:

```python
        form.addRow(self.btn_run)

        self.controls_layout.addWidget(box)
        self.controls_layout.addWidget(self._build_compare())

    def _build_compare(self):
        """The comparison controls (spec §7.1, E8): the first three modes, over this screen's own
        record picker. The list is what the single-selection picker appends to; the normalisation
        constant is read only by the renormalise mode, so its box is enabled only there."""
        box = QGroupBox("Compare saved runs")
        form = make_form(box)
        self.compare_list = CompareList(self.record_picker)
        self.compare_mode = QComboBox()
        self.compare_mode.addItems(["cells", "repeats", "renormalise"])
        self.compare_mode.currentTextChanged.connect(
            lambda mode: self.renorm_prefactor.setEnabled(mode == "renormalise"))
        self.renorm_prefactor = FloatField(None)
        self.renorm_prefactor.setEnabled(False)
        self.btn_compare = QPushButton("Compare saved runs")
        self.btn_compare.clicked.connect(self._compare)
        add_help_row(form, "Runs to compare", self.compare_list, HELP["compare_list"])
        add_help_row(form, "Comparison", self.compare_mode, HELP["compare_mode"])
        add_help_row(form, "Normalisation constant", self.renorm_prefactor, HELP["prefactor"])
        form.addRow(self.btn_compare)
        return box

    def _compare(self):
        """Dispatch one comparison. The panel checks nothing itself: the arity, the study and the
        unfinished-record refusals belong to the stage (one wording for both front ends), and a
        Refusal from the worker reaches the yellow box through BasePanel._on_error."""
        mode = self.compare_mode.currentText()
        options = {}
        if mode == "renormalise":
            # value_or_none, never value(): a blank box returns 0.0 from value(), and 0 is refused
            # with a sentence about a value nobody typed instead of "it is blank".
            options["prefactor"] = self.renorm_prefactor.value_or_none()
        self.dispatch(compare, mode, self.compare_list.ids(), provide_fig_sink=True,
                      on_result=lambda rec: self.log_pane.append_line(
                          f"Comparison written: {rec.path}"),
                      **options)
```

Add the three help strings to `HELP`, finding:

```python
    "confirm_production": "After the sanity checks, proceed to the (long) production sweep automatically.",
}
```

and replacing with:

```python
    "confirm_production": "After the sanity checks, proceed to the (long) production sweep automatically.",
    "compare_list": "The saved runs a comparison draws. Pick one in the record picker above and press "
                    "Add selected; the picker holds one at a time, so the list is how several are chosen.",
    "compare_mode": "cells: each run's ratio curve on one axis, labelled by cell. repeats: several "
                    "runs of one cell, with the spread across them as a band. renormalise: one run's "
                    "ratio recomputed with the constant below, drawn against the original.",
    "prefactor": "The normalisation constant to recompute T_eff/T with. The ratio is linear in it, so "
                 "nothing is re-simulated.",
}
```

and extend the imports, finding:

```python
from .base_panel import BasePanel
from .. import settings
from ..widgets.artifact_picker import ArtifactPicker
```

replacing with:

```python
from core.FDT.compare import compare

from .base_panel import BasePanel
from .. import settings
from ..widgets.artifact_picker import ArtifactPicker
from ..widgets.compare_list import CompareList
```

- [ ] **Step 5: Add the CrossVal screen's control**

In `core/gui/panels/crossval_panel.py`, find:

```python
        form.addRow(self.btn_run)

        self.controls_layout.addWidget(box)
```

Replace with:

```python
        form.addRow(self.btn_run)

        self.controls_layout.addWidget(box)
        self.controls_layout.addWidget(self._build_compare())

    def _build_compare(self):
        """The fourth comparison mode (spec §7.1): two sweep records together, and a slice of both at
        one operating point. A blank slice point means the middle of the range the two sweeps share,
        which is what the stage does with no `at` at all."""
        box = QGroupBox("Compare saved sweeps")
        form = make_form(box)
        self.compare_list = CompareList(self.record_picker)
        self.slice_at = FloatField(None)
        self.btn_compare = QPushButton("Compare saved sweeps")
        self.btn_compare.clicked.connect(self._compare)
        add_help_row(form, "Sweeps to compare", self.compare_list, HELP["compare_list"])
        add_help_row(form, "Slice at", self.slice_at, HELP["slice_at"])
        form.addRow(self.btn_compare)
        return box

    def _compare(self):
        """Dispatch the sweeps comparison. Nothing is checked here: a slice point outside the range
        the sweeps share, two sweeps of different parameters and an unfinished record are all the
        stage's refusals, and reach the yellow box through BasePanel._on_error."""
        options = {}
        at = self.slice_at.value_or_none()
        if at is not None:
            options["at"] = at
        self.dispatch(compare, "sweeps", self.compare_list.ids(), provide_fig_sink=True,
                      on_result=lambda rec: self.log_pane.append_line(
                          f"Comparison written: {rec.path}"),
                      **options)
```

Add the two help strings to `HELP`, finding:

```python
    "f0": "Non-dimensional forcing amplitude used to probe χ(ω) at each sweep point.",
}
```

and replacing with:

```python
    "f0": "Non-dimensional forcing amplitude used to probe χ(ω) at each sweep point.",
    "compare_list": "The saved sweep records a comparison draws. Pick one in the record picker above "
                    "and press Add selected; the picker holds one at a time.",
    "slice_at": "The operating point to slice both sweeps at. Blank means the middle of the range the "
                "two of them share.",
}
```

and extend the imports, finding:

```python
from .base_panel import BasePanel
from .. import settings
from ..widgets.artifact_picker import ArtifactPicker
```

replacing with:

```python
from core.FDT.compare import compare

from .base_panel import BasePanel
from .. import settings
from ..widgets.artifact_picker import ArtifactPicker
from ..widgets.compare_list import CompareList
```

- [ ] **Step 6: Run the tests and watch them pass**

Run: `pytest tests/test_nav_and_gating.py -q -k "comparison_list or comparison_modes"`
then the two panels' existing suites, which construct both panels:
`pytest tests/test_settings_persistence.py tests/test_figures.py -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add core/gui/widgets/compare_list.py core/gui/panels/fdt_panel.py core/gui/panels/crossval_panel.py core/gui/widgets/artifact_picker.py tests/test_nav_and_gating.py
git commit -m "compare: the two screens' comparison controls"
```

#### Amendments (binding — these supersede the text above)

Land this task after T24, T25, T26 and T27 (the panels as they now read) and after T37, T38 and T39. Rulings applied: P34, P61, P80. The stale anchors are re-derived against the panels as T25, T26 and T27 left them.

**A1 — the pickers (P61, P80).** `FdtPanel.record_picker` and `CrossValPanel.record_picker` exist from T25/T26. If either is missing, that is a T25/T26 defect: say so in the report and do not build a second picker.

**A2 — Step 3, `selection_text` is not optional.** `StorePicker` has no `selection_text` at HEAD, and no task T24–T27 adds one. Add it to `core/gui/widgets/artifact_picker.py`, immediately after the body of `selection_summary`, exactly as Step 3 gives it. Add `core/gui/widgets/artifact_picker.py` to this task's Files block (the git add line already names it).

**A3 — Steps 4/5, the two rows for registered keys are labelled from the table (P34).** P34 says the FDT and CrossVal panels build their row labels from `core.gui.fields.label(key)`. T38 and T39 registered `prefactor` and `slice_at` as sentences because the boxes did not exist yet. This task creates the boxes, so convert both entries. The rendered fix sentence is byte-identical, so this is a pure refactor. In `core/gui/fields.py`, find
```python
    "prefactor": "Set it in the 'Normalisation constant' box on the FDT analysis tab.",
```
and replace it with `    "prefactor": ("FDT analysis", "Normalisation constant"),`. Then find
```python
    "slice_at": "Set it in the 'Slice at' box on the Sweep study cross-validation tab.",
```
and replace it with `    "slice_at": ("Sweep study cross-validation", "Slice at"),`. In `FdtPanel._build_compare`, replace `add_help_row(form, "Normalisation constant", self.renorm_prefactor, HELP["prefactor"])` with `add_help_row(form, label("prefactor"), self.renorm_prefactor, HELP["prefactor"])`. In `CrossValPanel._build_compare`, replace `add_help_row(form, "Slice at", self.slice_at, HELP["slice_at"])` with `add_help_row(form, label("slice_at"), self.slice_at, HELP["slice_at"])`. If a panel does not already import `label` (T25/T26 may have, under P34), add `from ..fields import label` beside its other `..` imports. The 'Runs to compare', 'Sweeps to compare' and 'Comparison' rows answer no registered key and keep their literals. Add `core/gui/fields.py` to the Files block and the git add line. In Step 6 also run `pytest tests/test_nav_and_gating.py -q -k fix_sentences_name_it`.

**A4 — Step 4, the FDT panel import anchor is NOT found.** After T25 the third line reads `from ..widgets.artifact_picker import ArtifactPicker, StorePicker`, and T27 added `from . import record_view` above `from .base_panel import BasePanel`. Do NOT substring-match `from ..widgets.artifact_picker import ArtifactPicker`: the replacement would land inside T25's import line. Instead:
- insert `from core.FDT.compare import compare` on the line immediately after `from core.FDT.fdt_pipeline import run_fdt`;
- insert `from ..widgets.compare_list import CompareList` on the line immediately after the `from ..widgets.artifact_picker import ...` line.

**A5 — Step 4, the FDT panel `HELP` anchor is NOT found.** T25 appended `"seed"`, `"record_name"`, `"record_note"` and `"record"` after `"confirm_production"`. Insert the three new entries (`"compare_list"`, `"compare_mode"`, `"prefactor"`, text exactly as in Step 4) immediately before the closing `}` of the module-level `HELP` dict, i.e. after its last entry, which is `"record"` after T25.

**A6 — Step 5, the CrossVal panel import and `HELP` anchors are NOT found**, for the same two reasons (T26, T27). Instead:
- insert `from core.FDT.compare import compare` immediately after `from core.FDT.cross_validation import run_param_study_cli`;
- insert `from ..widgets.compare_list import CompareList` immediately after the `from ..widgets.artifact_picker import ...` line;
- insert the two `HELP` entries (`"compare_list"`, `"slice_at"`, text as in Step 5) immediately before the dict's closing `}`, after T26's `"record"`.

**A7 — anchors found as written.** Both panels' `        form.addRow(self.btn_run)` + blank line + `        self.controls_layout.addWidget(box)` anchors are still present: T25–T27 insert rows above `btn_run`, never below it.

**A8 — Step 7.** `git add core/gui/widgets/compare_list.py core/gui/panels/fdt_panel.py core/gui/panels/crossval_panel.py core/gui/widgets/artifact_picker.py core/gui/fields.py tests/test_nav_and_gating.py`

---

### Task 41: the documents, the walkthrough rows, and the gates of record

**Why:** Four documents state things this piece makes false, and the piece's own record — the
deviations table, the decisions log entry, the gates — is written from measurement at the end.
Spec §2.4 checklist item 28 names `CLAUDE.md`'s kind list and its "Every stage writes its artifact at
completion" sentence, `core/tool/browse.py`'s `--store-root` docstring (which E11 changes, and whose
own line numbers are already stale) and `docs/STATE.md`'s piece-5 row; spec §11's "Documents edited
in place at the end" adds `docs/superpowers/specs/2026-09-11-one-flow-design.md:425`, whose stated
reason for having no store flag on `fdt`/`crossval` — that their outputs are not store artifacts —
stops being true under E1 and E11; spec §10 marks walkthrough rows 15, 16 and B8 superseded (E1, E4)
and writes a new E-row section; spec §9 makes the final fast gate, the slow set and the GPU
smoke-gate judgement gates of record; spec §12 is the deviations table, empty at approval.

**Files:**
- Modify: `CLAUDE.md:135-138` ("The kinds are priors, simulations (the training cache, keyed by
  identity digest), posteriors,")
- Modify: `CLAUDE.md:125-127` ("The Simulate, FDT and CrossVal panels still remember their")
- Modify: `CLAUDE.md:169-174` ("`core/tool/` — the command-line tool")
- Modify: `core/tool/browse.py:18-24` ("NO ``--store-root``, EITHER.")
- Modify: `tests/test_tool.py:1971-1978` (the docstring of
  `test_the_artifacts_family_takes_no_configuration_flags_and_its_help_costs_no_torch`)
- Modify: `docs/superpowers/specs/2026-09-11-one-flow-design.md:425` ("There is no global
  `--artifacts` flag.") and `:60` ("FDT/CrossVal hardening, and wrapping them in the store.")
- Modify: `docs/checklists/display-walkthrough.md:23,24,72` (rows 15, 16, B8) and its end (the new
  E-row section)
- Modify: `docs/superpowers/specs/2026-09-22-secondary-analyses-design.md:993-1001` (§12)
- Modify: `docs/STATE.md` — the header paragraph, "Owed" item 7's piece-5 bullet, the "Last gate"
  table, the decisions log
- Test: none new. Two source files are touched and both edits are docstrings; the focused runs in
  steps 6 and 7 prove they broke nothing, and the gates of record in steps 14–17 are this task's
  verification.

**Interfaces:**
- Consumes: every task of the piece. From T1: the kind `fdt` and `store.KIND_DIRS`. From T3:
  `PROGRESSIVE_KINDS`, `ArtifactWriter.refresh()`. From T4: `LEGACY_DIRS`, `loose_files`,
  `legacy_dirs`. From T28: `--store-root` on `fdt` and `crossval`, and the per-subcommand choice
  that replaced `hasattr(args, "store_root")` in `core/tool/__init__.py`. From T30/T31: the tidy-up's
  two new categories. From T37–T40: the `compare` subcommand and its four modes. From the ledger:
  every ruling made during execution.
- Produces: nothing any task consumes — this is the last task. It produces the durable record:
  `docs/STATE.md`'s piece-5 row, gate rows and decisions-log entry; the spec's §12; the E-rows the
  OWNER will run.

**Order.** Steps 2–12 edit documents and two docstrings and end in one commit; steps 13–17 are the
gates of record, run by the CONTROLLER on that commit's tree with no source edited while a suite
runs; steps 18–22 write `docs/STATE.md` from the measured numbers and commit it. A `docs/STATE.md`-
only commit needs no second gate — it contains no code — which is the precedent piece 4 set
(`44ed3ed`/`80be144`).

**NO TASK MAY FILL THE E-ROWS' DATE AND RESULT COLUMNS.** They are the owner's, on a real screen.

---

- [ ] **Step 1: Collect the deviations and the rulings from the ledger**

Read, and take notes from, in this order:

```
.superpowers/sdd/2026-09-22-secondary-analyses/progress.md        (rulings R1-R10, and every
                                                                   execution ruling added since)
.superpowers/sdd/2026-09-22-secondary-analyses/                   (the task reports: every
                                                                   "OBJECTION" and "what I did")
```

Write nothing yet. What you are looking for is every ruling made during execution that **differs
from what the spec says**, with what it costs if it is wrong — that is exactly what §12's four
columns hold (`| # | where | deviation | why |`). A task report's objection that was answered by
doing what the spec said is NOT a deviation; one that was answered by doing something else IS.

- [ ] **Step 2: `CLAUDE.md` — the kind list and the completion sentence**

In `CLAUDE.md`, find:

```markdown
  The kinds are priors, simulations (the training cache, keyed by identity digest), posteriors,
  observations, calibrations, inferences and diagnostics. Every stage writes its artifact at
  completion and returns a `Loaded*` wrapper; Save is a rename; loading refuses any verifiable
  mismatch, and `Accept(truncated, other_observation)` are the only escape hatches (each use is
  recorded downstream).
```

Replace with:

```markdown
  The kinds are priors, simulations (the training cache, keyed by identity digest), posteriors,
  observations, calibrations, inferences, diagnostics and fdt (the effective-temperature
  measurement, the parameter sweep and their comparisons — piece 5). Every stage writes its
  artifact at completion and returns a `Loaded*` wrapper, with two exceptions: the training cache,
  which commits batch by batch and has no writer at all, and an `fdt` record, which is
  PROGRESSIVE — its directory and a first manifest exist from the moment the run starts,
  `refresh()` rewrites them as it goes, and a cancel or a crash keeps the folder, marked unfinished
  (piece-5 spec §2.2, E2). Save is a rename; loading refuses any verifiable
  mismatch, and `Accept(truncated, other_observation)` are the only escape hatches (each use is
  recorded downstream).
```

- [ ] **Step 3: `CLAUDE.md` — what the two panels remember, and the tool's subcommands**

In `CLAUDE.md`, find:

```markdown
  `PRISM.ini` is ignored, not restored. The Simulate, FDT and CrossVal panels still remember their
  own numeric fields — piece 5's (piece-3 spec §1.3).
```

Replace with:

```markdown
  `PRISM.ini` is ignored, not restored. The Simulate, FDT and CrossVal panels remember their own
  numeric fields and their pickers' selections; the Seed box is never remembered, because a
  remembered seed would silently turn every run into a repeat of the last one (piece-5 spec §5.5,
  E7).
```

Then find, in the same file's "Where things are" section:

```markdown
- `core/tool/` — the command-line tool: `python -m core --help` lists every subcommand (the stages
  `prior train tsnpe validate infer`, the diagnostics `sbc identifiability ablation`, plus `smoke`,
  `fdt`, `crossval` and `artifacts` (list/show/note/rm/sweep/summary)).
```

Replace with:

```markdown
- `core/tool/` — the command-line tool: `python -m core --help` lists every subcommand (the stages
  `prior train tsnpe validate infer`, the diagnostics `sbc identifiability ablation`, plus `smoke`,
  `fdt`, `crossval`, `compare` (cells/repeats/renormalise/sweeps) and `artifacts`
  (list/show/note/rm/sweep/summary)).
```

Check the replacement against reality before you write it: run
`& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m core --help` and name the subcommands the
help actually lists, in the order it lists them. If `compare`'s modes are spelled differently, the
help wins.

- [ ] **Step 4: `CLAUDE.md` — the suite count, from measurement**

`CLAUDE.md` says "`tests/` — the eighteen suites". Count them:

```bash
ls tests/test_*.py | wc -l
```

If the number is still eighteen, change nothing. If this piece added a suite, write the new number
in words and add the new file to the parenthetical list beside `test_artifact_browser.py`, in the
same voice ("`test_x.py` the …'s"). Record what you measured; do not guess.

- [ ] **Step 5: `core/tool/browse.py` — the `--store-root` docstring**

Its stated reason stops being true with Task 28, and its four line references into
`core/tool/__init__.py` (`:95`, `:96-98`, `:156-157`, `:121-122`) were already wrong before this
piece: at `a0d85da` those lines were `:114`, `:117`, `:183-184`, `:146`. The correction drops the
numbers rather than re-deriving numbers that will go stale again.

In `core/tool/browse.py`, find:

```python
NO ``--store-root``, EITHER. The root is ``config.artifacts_root()`` (PRISM_ARTIFACTS), like every
subcommand but ``smoke``, and ``main`` has already opened the store by the time a handler runs. The
name is not reused because ``main`` keys three of smoke's behaviours on the FLAG's presence:
``has_store_root = hasattr(args, "store_root")`` (:95) decides whether a fresh ``mkdtemp`` root is
created (:96-98), whether an empty auto-created root is removed afterwards (:156-157), and which
Ctrl-C advice is printed (:121-122). Declaring ``--store-root`` here would change all three for this
family, for a root it does not want.
```

Replace with:

```python
NO ``--store-root``, EITHER. The root is ``config.artifacts_root()`` (PRISM_ARTIFACTS) and ``main``
has already opened the store by the time a handler runs, so this family has no second root to point
at: it creates nothing, spends nothing a Ctrl-C could abandon, and reads the one root the
environment names.

The flag is no longer a proxy for anything else, either. It used to be: ``main`` keyed three of
smoke's behaviours on ``hasattr(args, "store_root")`` -- the throwaway ``mkdtemp`` root, the removal
of an auto-created root left empty by a failure, and the Ctrl-C advice -- so declaring the name here
would have changed all three for a root this family does not want. Piece 5 gave ``fdt`` and
``crossval`` a ``--store-root`` of their own and reworked those three deliberately rather than
letting them flip (piece-5 spec §6.1, E11): the throwaway root and its cleanup are ``smoke``'s
alone, chosen per subcommand instead of by the flag's presence, and a subcommand's Ctrl-C advice
follows its own ``interrupt_note``. The absence here is now what it says on the face of it -- this
family wants one root, the environment's.

(No line numbers into ``core/tool/__init__.py`` on purpose: the four this paragraph used to carry
had gone stale before piece 5 moved them again.)
```

Then read `core/tool/__init__.py`'s dispatch as Task 28 left it. If the second paragraph's last
sentence is not literally true of that code — for instance if `smoke` still reaches
`_smoke_interrupt_advice` through a branch that is not an `interrupt_note` — rewrite the clause so
it states what the code in front of you does. A docstring that gives a reason must give the reason
the code has.

- [ ] **Step 6: the same stale reason in `tests/test_tool.py`'s docstring**

`tests/test_tool.py`'s `test_the_artifacts_family_takes_no_configuration_flags_and_its_help_costs_no_torch`
repeats browse.py's old reason, with the same stale numbers. Its assertions do not change — the
family still declares no `--store-root` — only the sentence that says why.

Find:

```python
    """B9's two halves, pinned. (1) No configuration flags anywhere in the family: a listing must not
    be able to fail on a bounds file it does not need, and --store-root in particular is absent
    because ``main`` keys smoke's temp-root behaviour, its empty-root cleanup and its Ctrl-C advice on
    ``hasattr(args, "store_root")`` (core/tool/__init__.py:95-98, 121-122, 156-157). (2) ``--help``
```

Replace with:

```python
    """B9's two halves, pinned. (1) No configuration flags anywhere in the family: a listing must not
    be able to fail on a bounds file it does not need, and --store-root in particular is absent
    because this family creates no root and reads the one PRISM_ARTIFACTS names. It used to be
    absent for a second reason as well -- ``main`` keyed smoke's temp root, its empty-root cleanup
    and its Ctrl-C advice on ``hasattr(args, "store_root")`` -- which piece 5 retired when it gave
    fdt/crossval the flag and made those three an explicit per-subcommand choice (piece-5 spec
    §6.1). (2) ``--help``
```

- [ ] **Step 7: Run the focused tests that read these two files**

Run:
`& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_tool.py -k "artifacts or help" -q`

Expected: PASS, with no collection error. Both edits are docstrings, so a failure here means the
quoted anchor matched something it should not have — re-read the file before going on.

- [ ] **Step 8: `docs/superpowers/specs/2026-09-11-one-flow-design.md:425`**

Find:

```markdown
   - There is no global `--artifacts` flag. The `fdt` and `crossval` outputs follow the environment, not the store (`fdt_pipeline.py:28-33`, `cross_validation.py:40-41`), so a flag for the store alone would split one run across two roots.
```

Replace with:

```markdown
   - There is no global `--artifacts` flag. When this was written the `fdt` and `crossval` outputs followed the environment rather than the store (`fdt_pipeline.py:28-33`, `cross_validation.py:40-41`), so a flag for the store alone would have split one run across two roots. **Corrected in place by piece 5 (E1, E11):** both analyses now write store records of the `fdt` kind and both subcommands declare their own `--store-root`, so the reason given here no longer holds — see `docs/superpowers/specs/2026-09-22-secondary-analyses-design.md` §6.1. There is still no global `--artifacts` flag.
```

- [ ] **Step 9: the same fact in that spec's out-of-scope table (`:60`)**

The row that hands FDT/CrossVal to piece 5 states the same thing, and would be left false. Find:

```markdown
| FDT/CrossVal hardening, and wrapping them in the store. Their outputs stay in `artifacts_root()/fdt` and `/crossval`. | piece 5 |
```

Replace with:

```markdown
| FDT/CrossVal hardening, and wrapping them in the store. Their outputs stay in `artifacts_root()/fdt` and `/crossval`. **Piece 5's answer: one store kind `fdt`, whose records live at `artifacts_root()/fdt/<name>__<id>/`; a sweep is a record of that kind too, so nothing is written to `/crossval` any more (piece-5 spec §2.1).** | piece 5 |
```

- [ ] **Step 10: `docs/checklists/display-walkthrough.md` — mark rows 15, 16 and B8**

The three rows keep their date and their result: they passed, on the code as it then was. What
changes is a marker in the surface column saying they no longer describe the application. The
reasons go in the new section's preamble (step 11), which is where the B-, C- and D-sections put
theirs.

Find:

```markdown
| 15 | FDT full run | small settings | figures land; a run where every row failed is reported, not "complete" | 2026-09-11 | pass (the user's run on the real screen) |
| 16 | CrossVal | S-sweep and T-sweep at small sizes | figures land | 2026-09-11 | pass (the user's run on the real screen) |
```

Replace with:

```markdown
| 15 | FDT full run — **SUPERSEDED by E8 and E9 (piece 5)** | small settings | figures land; a run where every row failed is reported, not "complete" | 2026-09-11 | pass (the user's run on the real screen) |
| 16 | CrossVal — **SUPERSEDED by E7, E8 and E9 (piece 5)** | S-sweep and T-sweep at small sizes | figures land | 2026-09-11 | pass (the user's run on the real screen) |
```

Then find:

```markdown
| B8 | FDT full run, after the saved figures started closing (`bb22ac4`; re-checks row 15) | Run the FDT panel at small settings. | The figures still land in the panel as before; the log pane shows no "FigureCanvasAgg is non-interactive" line. | 2026-09-15 | pass (the user's run on the real screen) |
```

Replace with:

```markdown
| B8 | FDT full run, after the saved figures started closing (`bb22ac4`; re-checks row 15) — **SUPERSEDED by E9 and E16 (piece 5)** | Run the FDT panel at small settings. | The figures still land in the panel as before; the log pane shows no "FigureCanvasAgg is non-interactive" line. | 2026-09-15 | pass (the user's run on the real screen) |
```

- [ ] **Step 11: `docs/checklists/display-walkthrough.md` — the E-row section**

Append at the end of the file (after row D17), exactly:

```markdown

## Piece-5 GUI checks (the secondary analyses; design spec §10)

The rows piece 5 creates; the user runs them once on a real display, the way the A-, B-, C- and
D-rows were run. Rows 1–20, A1–A9, B1–B8, C1–C11 and D1–D17 all stand except the three marked
SUPERSEDED above, and piece 5 re-checks only what it changes. The row letters here are independent
of the piece's decisions E1–E12; they are simply the next letter after D.

What the three superseded rows asserted, and why they no longer describe the application:

- **Row 15 ("a run where every row failed is reported, not 'complete'")** — under E4 a run that
  measured nothing is a REFUSAL naming the setting to change, not a completed run reported as
  failed, and its unfinished record stays on disk with the spectra that diagnose it. E8 is its
  replacement.
- **Rows 15 and 16 ("figures land")** — under E1 the figures land inside the record's own
  `figures/` folder, not in a flat directory with a wall-clock stamp, and the panel names the record
  it wrote. E9 is their replacement.
- **Row B8** — its first half is row 15's "figures land" (E9 again); its second half, that no
  "FigureCanvasAgg is non-interactive" line appears in the log pane, is carried unchanged into E16.

| # | surface | do | expect | date | result |
|---|---|---|---|---|---|
| E1 | The record appears in the browser WHILE the run is going | Start an FDT analysis at small settings (FDT Analysis → "FDT analysis": n_freqs 4, ensemble_M 8, freqs_per_batch 1, "Skip sanity checks" ticked). While it runs, open Home → Artifacts and pick the new kind in the Kind picker. | The run's record is already listed, by the name it was given, marked unfinished, with its study reading "single"; the detail pane shows its manifest and the records written so far. Nothing has to be refreshed by hand. | | |
| E2 | A cancelled run keeps its record, marked unfinished | Start the same run again under a new name and press Cancel once the spontaneous campaign has started. | "Cancelling…", a clean stop, controls unlock, no error dialog; the record is still on disk and still listed, unfinished; its folder holds `log.txt` and whatever figures had been written; the Artifacts screen's Sweep does NOT offer it (it carries a manifest). | | |
| E3 | A bad box on the FDT screen: the yellow box names the box AND the screen | On "FDT analysis", clear the "n_freqs" box (leave it blank, do not type 0) and press "Run FDT analysis". Repeat with "ensemble_M" blank, then with "F0 (ND forcing amplitude)" blank. | Each time: the yellow "Check your inputs" box, the message naming the setting, its rule and its default, and the line underneath naming the box and the FDT analysis tab — not a traceback, not a clamp, and nothing simulated. | | |
| E4 | A bad grid on the sweep screen | On "Sweep study cross-validation", set the S grid's number of points to 1 and press "Run sweep study"; then set the S grid's min above its max and run again. | The yellow box both times, naming the S grid and where it is; nothing runs; no record is created. | | |
| E5 | The normalisation constant is refused before anything is simulated | Point the FDT screen's Cell picker at a cell whose file lacks the normalisation constant (`n` or `beta`) and run. | The refusal arrives within a second or two, in the yellow box, naming the cell; no progress bar ever starts; no figures. | | |
| E6 | The band is refused after the grid is built and before the driven campaign | Run a single-cell analysis whose spontaneous recording is too short to resolve the bottom of the band (piece-5 spec §3.4's second refusal; the FDT panel's smallest settings with the shortest spontaneous window). | The spontaneous campaign runs and its figure appears; then the run refuses in the yellow box, naming the band and the spontaneous duration; the driven campaign never starts; the unfinished record holds the spontaneous spectrum. | | |
| E7 | A sweep with SOME points failed completes, and the counts are on the row | Run an S sweep at 3 points on a cell where one point fails (the smallest preset; if none fails naturally, it is enough to read the row of a sweep that completed — the Points column must show done/planned and the failed count). | The run completes; the log pane says how many points were done and how many failed; the browser's row for that record shows the same counts, and the record is NOT marked unfinished. | | |
| E8 | A sweep where EVERY point failed refuses, and the record is still there | Run a sweep whose every operating point fails. | A calm one-line refusal in the yellow box naming the grid and the cell — no red box, no traceback; the record is on disk, listed, unfinished, and holds the spectra the message points at; if this was the S sweep, the T sweep is unaffected (it is a record of its own). | | |
| E9 | Where the figures land now, and what the panel says it wrote | Complete one FDT analysis and one sweep study at small settings, then open each record's folder under `Artifacts/` and its row in the browser. | The figures are inside `<record>/figures/*.png` and nowhere else — no loose PNGs beside the kind directory, no wall-clock-stamped names; the figures still appear in the panel as they always did; the log pane names the record that was written (kind, name and id). | | |
| E10 | The picker shows an earlier run | On the FDT screen, choose an earlier record in the run picker; then do the same on the sweep screen. | The panel shows that record's cell, its settings, its seed and any notices it recorded, and re-opens its figures from the record's `figures/`; nothing is re-run. Quit and relaunch: the picker's selection is restored, and the Seed box is EMPTY. | | |
| E11 | compare cells | On the FDT screen, add two or more single-cell records of different cells to the comparison list and run the cells comparison. | One axis with one ratio curve per record, labelled by cell; a new record of study "comparison" appears in the browser holding the figure and its data file. | | |
| E12 | compare repeats | Add two or more records of the SAME cell (different seeds) and run the repeats comparison. | The curves with the spread across them drawn as a band; a comparison record is written. An unfinished record added to the list is refused, naming it. | | |
| E13 | compare renormalise | Pick one single-cell record and run the renormalise comparison with a normalisation constant of your own. | The recomputed ratio drawn against the original, both labelled; a comparison record is written. | | |
| E14 | compare sweeps | On the sweep screen, pick two sweep records and run the sweeps comparison. | The two sweeps drawn together and a slice of both at a common operating point; a comparison record is written. | | |
| E15 | The tidy-up offers the legacy loose files | With the owner's legacy pictures still in `Artifacts/fdt/` (the two PNGs stamped `20260915_153042`), press Sweep on the Artifacts screen; then run `python -m core artifacts sweep` and `python -m core artifacts sweep --yes`. | Both front ends offer the loose FILES as their own clearly separated category, by name, alongside the manifest-less directories; a legacy directory beside the kind directories is offered by the all-kinds sweep only; a dry run removes nothing; nothing valid is ever offered; after a confirmed removal the files are gone and every real record is untouched. | | |
| E16 | A full run says nothing unprefixed (B8's carried half) | Run a complete FDT analysis and watch the log pane throughout. | No "FigureCanvasAgg is non-interactive" line; no library line without its `library: <logger name>: ` prefix; no `core` line twice. | | |
```

Leave the date and result columns empty. They are the owner's.

- [ ] **Step 12: the piece-5 spec's §12 deviations table**

In `docs/superpowers/specs/2026-09-22-secondary-analyses-design.md`, find:

```markdown
| # | where | deviation | why |
|---|---|---|---|
| | | | |
```

Replace the empty row with one row per deviation collected in step 1, keeping the four columns:
`#` a number from 1 up; `where` the spec section and the decision or task it departs from
(e.g. "§2.2, T3"); `deviation` what was done instead, stated so a reader who has not seen the code
can check it; `why` the reason AND what it costs if the ruling is wrong — that last clause is the
practice pieces 2, 3 and 4 followed and is what makes the table worth keeping.

If a ruling was made at planning time rather than during execution, say so in a sentence above the
table, the way piece 4's §12 does ("Rows 1–12 were ruled at **planning** time…"), and point at the
ledger for the full text. If an outright factual error in this spec was corrected inline rather than
recorded as a deviation, list those corrections in that same sentence rather than as rows.

If the ledger holds no deviation at all, replace the empty row with a single line under the table
saying so in one sentence, with the date — an empty table and an unfilled one look identical, and
the next piece's reader must be able to tell them apart.

- [ ] **Step 13: Commit the documents and the two docstrings**

```bash
git add CLAUDE.md core/tool/browse.py tests/test_tool.py \
        docs/superpowers/specs/2026-09-11-one-flow-design.md \
        docs/superpowers/specs/2026-09-22-secondary-analyses-design.md \
        docs/checklists/display-walkthrough.md
git commit -m "docs: piece 5's document corrections, the E-rows and the deviations"
```

- [ ] **Step 14: The collected count**

Check first that no `python` process is already running a suite
(`Get-Process python -ErrorAction SilentlyContinue`). Then run, from the repository root:

```powershell
& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest --collect-only -q *> .superpowers/sdd/2026-09-22-secondary-analyses/gate-collect.log
```

Record what you measured: the total collected, the difference against 732 at `80be144`, and how that
sits against spec §8.4's stated band ("about 170 ± 40 added, for roughly 900 collected"). If it is
outside the band, say so — piece 3's count went over its budget unrecorded and that is a thing the
gate table now names.

- [ ] **Step 15: The final fast gate (ONE process, background, nothing edited while it runs)**

```powershell
& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest -m "not slow" -q --durations=15 *> .superpowers/sdd/2026-09-22-secondary-analyses/gate-final-fast.log
```

Start it in the background and touch no source file until it exits. Record what you measured:
passed / skipped / deselected, the warning count against piece 4's baseline of 181, the wall time
against spec §9's 16-minute target, the exit code, the three slowest tests from `--durations`, and
that `git status` is clean, the real `Artifacts/` gained nothing and no `<repo>/sbi-logs` appeared.
If a new warning class appeared, name it and say which test emits it: a leaked warning is a finding.

- [ ] **Step 16: The slow set**

```powershell
& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest -m slow -q --durations=5 *> .superpowers/sdd/2026-09-22-secondary-analyses/gate-slow.log
```

Budget: about 45 minutes (spec §9, up from ~31 with the new Nadrowski end-to-end run and the
rewritten tiny-size test). Record the count, the wall time, the exit code, and each slow test's own
duration — in particular `test_tool.py::test_fdt_and_crossval_run_at_tiny_size` as Task 34 rewrote
it, whose docstring's stale 598 s figure that task corrected from measurement, and
`test_user_sbi.py::test_chi_mode_full_sbi_pipeline` against the 1610 s piece 4 recorded and is
watching.

- [ ] **Step 17: The GPU smoke-gate judgement, and the gate if it is needed**

Spec §9: the smoke gate is run if any line that creates or moves a tensor changed. This path is
pinned to the CPU (`cli.make_fdt_config` forces `cpu_device()`), so the expectation is that only the
seed work qualifies. **Make the judgement against the diff and record it either way** — that is
what pieces 3 and 4 did.

```bash
git diff --stat <the piece's first commit>^..HEAD
git diff <the piece's first commit>^..HEAD -- core/ | grep -nE "torch\.|\.to\(|\.cuda|device=|manual_seed|from_numpy|tensor\("
```

If nothing in that output creates or moves a tensor on a path `smoke` reaches, record the judgement
with its evidence and do not run the card. If something does, run `CLAUDE.md`'s four command lines
(run 1, run 2, run 2b, run 3) alone on the card, with `$S` an empty scratch directory, and read the
pass criteria off `$LASTEXITCODE` and the printed lines: run 1 and run 3 exit 0 with stage timings
near piece 4's (92.3 / 198.5 / 22.3 / 85.9 s and 100.4 / 63.9 / 15.0 / 19.3 s) and masked-probe
counts within ±12 pp of 37 %; run 2 prints the Fisher-reuse line and `[checkpoint] resuming at batch
4/4`; run 2b exits 1 naming `n_runs` with no `[fisher]` line and no new `simulations/` directory.
Delete the scratch stores afterwards. The diagnostic card is needed only if a line under
`core/diagnostics` that moves tensors changed — this piece touches `core/artifacts/report.py`, not
the diagnostics, so say so explicitly if that is what the diff shows.

- [ ] **Step 18: `docs/STATE.md` — the piece-5 header paragraph**

Find the opening of the file:

```markdown
**Last updated:** 2026-09-22. **Piece 5, "the secondary analyses", is IN PROGRESS**: brainstormed
with the owner 2026-09-22 (decisions E1–E12), design committed `a0d85da`, implementation plan being
written. **Piece 4, "GUI usability and the artifact browser", is DONE**:
```

Replace its first sentence with a piece-5 DONE paragraph in the voice of the piece-4 one below it:
the date, the commit range (`git log --oneline <first>^..HEAD` — the first is the commit that
committed this piece's plan), that it is on the local `main` branch and NOT YET PUSHED, how far
`main` is ahead of `origin/main`, the design and plan paths with their commits and the count of
tasks, then **what landed**, in one paragraph: the eighth kind `fdt` and its progressive writer;
both analyses as public entries writing named records with provenance, a seed and their numbers in
`data.h5`; the sweep split into one record per swept parameter with its failure counts and its
all-failed refusal; the two pre-spend refusals and the off-grid fix; five screens' builder refusals
in the yellow box with `_config_error` gone and the fix-hint table widened to screens; the pickers
and the figure watcher pointed at the record's `figures/`; `--store-root` on both subcommands with
the three behaviours reworked; the tidy-up's loose files and legacy directory; and the comparison
facility's four modes. Then the gates: point at the table below for the final fast gate, the slow
set and the GPU smoke-gate judgement. Then, in bold, that rows **E1–E16 of
`docs/checklists/display-walkthrough.md` are OWED by the owner on a real screen** — do not claim
them. Keep the existing "Piece 4 … is DONE" text that follows, and rewrite the later sentence that
reads "**Piece 5 is under way** … The owner is swapping models before implementation: the session
that designed it stops after the plan." so it no longer describes a piece that has finished.

Record what you measured: every commit hash and count in that paragraph comes from `git log`, not
from memory.

- [ ] **Step 19: `docs/STATE.md` — the gate table's new rows**

Add to the "Last gate" table, after the piece-4 rows and in the same order the earlier pieces use
(collect-only, fast gate, slow set, GPU gate), one row per measurement from steps 14–17, each in the
voice of its piece-4 sibling:

```markdown
| `pytest --collect-only -q` after piece 5 | <date> at `<commit>`: **<N>** collected (the fast gate's <p> passed and <s> skipped, plus the <k> slow tests), <d> more than the 732 at `80be144`, against design spec §8.4's stated band of 170 ± 40 added. `tests/test_*.py` holds <n> suites |
| **final fast gate, piece 5**, ONE process, `pytest -m "not slow" -q --durations=15` | <date> at `<commit>`: **<p> passed, <s> skipped**, <d> deselected, **<w> warnings**, **<mm> min <ss> s** (against design spec §9's 16-minute target), exit 0. Tree clean; the real `Artifacts/` gained nothing; no `sbi-logs/`. <the warning baseline, moved or not, and by what>. Slowest: <the three from --durations>. Every task had its own one-process gate, recorded in the execution ledger |
| slow set of record, `pytest -m slow -q --durations=5` (piece 5) | <date> at `<commit>`: **<n> passed**, <d> deselected, <w> warnings, **<mm> min <ss> s**, exit 0, against design spec §9's ~45-minute budget. <each slow test and its duration, with the new ones named> |
| **GPU smoke gate, piece 5** | <date>: <either the four runs with their timings, exit codes and masked-probe counts in the piece-4 row's shape, or the JUDGEMENT that it was not required, with the evidence: what the diff touches and why no line on a path `smoke` reaches creates or moves a tensor> |
```

Do **not** add a display-walkthrough row for E1–E16. That row is written by the session that records
the owner's result, exactly as the B-, C- and D-walkthrough rows were.

- [ ] **Step 20: `docs/STATE.md` — the decisions log entry for E1–E12**

Append to the decisions log, after the piece-3 entry and in its voice:

```markdown
- **2026-09-22** — **piece 5 (the secondary analyses): decisions E1–E12** (spec
  `docs/superpowers/specs/2026-09-22-secondary-analyses-design.md` §1.1; plan
  `docs/superpowers/plans/2026-09-22-secondary-analyses.md`), each the owner's answer to a question
  asked in plain language. E1 both analyses move into the artifact store, fully, with a reading
  side: named records, provenance with file fingerprints, listing, deletion, the outputs inside the
  record's folder, the numbers saved and not just the pictures, and a way to pick an earlier run
  from the screens. E2 an interrupted run KEEPS its folder, marked unfinished — the training cache's
  rule, adopted because these runs take hours. E3 one new kind, `fdt`, covers both analyses, with a
  `study` field saying which. E4 nothing measurable is a REFUSAL naming the setting to change, and
  some points failed is a completed record carrying the count. E5 checks refuse only what breaks;
  a setting too thin to trust warns, and the warning is recorded in the record. E6 the fix-hint
  table is widened to understand SCREENS, not only the six inference tabs, and an input that
  appears in several places lists them all. E7 every run records the seed it used, drawing one when
  none is supplied, with reseeding confined the way the diagnostics confine theirs. E8 comparing
  covers all four modes: several cells, repeats of one cell with their spread, one run
  re-normalised, two sweeps together. E9 the off-grid range test is fixed and the band is refused
  before the driven campaign. E10 the tidy-up command learns to see the legacy loose files so the
  owner can clear them; nothing is deleted on their behalf. E11 both subcommands gain the
  destination flag, and the three behaviours that keyed off its presence are reworked deliberately
  rather than flipping by accident. E12 one piece, ordered so every part of the hardening lands
  before the first line of the comparison facility.
  - The design rulings that carry weight (spec §1.2): the progressive record is a MODE on the
    existing `ArtifactWriter`, never a second writer and never a default — the six ordinary kinds
    still lose their directory on any exception, pinned by a test written before the mode existed.
    The FRONT END creates the writer and the STAGE enters it: `log.txt` comes from a thread-local
    run log populated only inside `capture_run()` on the worker thread, while the figure watcher
    needs the directory name before dispatch, so a writer entered on the window's thread would
    write no log at all. `FDTConfig` gains `copy_for_run()` and not just fields, because
    `public_entry` is duck-typed on that method: without it the decorator gives the run log and NO
    private copy, and V1 is claimed and silently absent. A two-parameter study writes TWO records,
    one per swept parameter, so one sweep's failure no longer costs the other. A comparison names
    its sources in the BODY, not in `parents` — the parents block is a flat `{key: id}` map — so
    deleting a compared run is not refused and `render_lineage` prints `MISSING <kind> [<id>]`.
    `fdt` and `crossval` gain `--seed`, widening piece 2's recorded reading that only `smoke` and
    the diagnostics take one, because a recorded seed you cannot supply back does not make a run
    reproducible; the Seed box is never remembered. Five checked settings (`freq_bounds`,
    `burn_in_nd`, `T_obs_periods`, `dt_nd`, `psd_T_obs_nd`) are exposed by neither front end, so
    they are checked with `field=None` and get no table entry — a fix sentence would promise a
    control that does not exist. The kind directory is the legacy `fdt/` on purpose: it makes the
    owner's stray pictures loose files INSIDE a kind directory, which is the only place the tidy-up
    can reach them.
  - D11 and D12 stay standing refusals; this piece re-opened neither.
  - The execution rulings are in the spec's §12 and in the gitignored ledger
    `.superpowers/sdd/2026-09-22-secondary-analyses/progress.md`.
```

Then add, as the last bullet of that entry, the rulings your step 1 collected that belong in a
durable record rather than only in §12 — the ones a later piece would otherwise re-litigate. Write
each with what it costs if it is wrong.

- [ ] **Step 21: `docs/STATE.md` — "Owed" item 7's piece-5 bullet**

Find, in "Owed, in order" item 7:

```markdown
   - **Piece 5.** FDT/CrossVal hardening and wrapping them in the store (their outputs still go to
     `artifacts_root()/fdt` and `/crossval`). Two test gaps: the cell-folder branch of the `fdt`
```

Rewrite the whole bullet in the shape the piece-3 and piece-4 bullets above it use: a header line
`- **Piece 5** — DONE <date> (`<first>`..`<last>`). Every item carried into it is closed:` followed
by one struck-through sub-bullet per carried item, each naming what landed and the task that closed
it. The items to strike, all of them named in the bullet as it stands today:

- ~~FDT/CrossVal hardening and wrapping them in the store~~ — the `fdt` kind, both runs as public
  entries writing records (the task numbers from the ledger).
- ~~the cell-folder branch of the `fdt` unsupported-model hint is untested~~ — T29.
- ~~only `plot_psd` of the four `core/FDT/plots.py` functions is unit-tested for closing its
  figure~~ — T32.
- ~~the five panels adopt `core/refusals.py`'s rules and the two field tables~~ — T21, T22, T23;
  `_config_error` is gone.
- ~~`load_and_validate_gt`'s `Refusal(field="cell")` and the FDT dry run's problem strings are two
  wordings for one rule~~ — one message builder, T21 and T29.
- ~~the artifact table's numeric columns sort as text~~ — T2.
- ~~`tests/test_artifact_browser.py:864` asserts the window's picker kinds by EQUALITY~~ — replaced
  by a subset assertion when the two new pickers landed.

Check each against the ledger before you strike it: strike only what the piece actually closed, and
leave anything it did not as a live item with its reason. Then add, as the piece's own owed item:

```markdown
     - **Rows E1–E16 of `docs/checklists/display-walkthrough.md` on the real screen** — OWED by the
       USER. No task may fill their date and result columns; the gate table gains its walkthrough
       row when the owner has run them.
```

Finally, add under "Open, with no piece owning them yet" whatever this piece hands on: every item a
task report or the whole-piece review left open, each in one line with its reason, in the voice of
the "*From piece 4.*" entries above it.

- [ ] **Step 22: Commit the state**

```bash
git add docs/STATE.md
git commit -m "docs: piece 5 is done — STATE, the gates of record, the decisions"
```

This commit is documents only and needs no further gate; the gates of record were run on the tree of
step 13's commit, which holds every line of code this piece wrote.

- [ ] **Step 23: Hand the walkthrough to the owner**

Tell the owner, in the session's closing message: rows E1–E16 are written and unfilled, what each
needs (a real display, `run.bat`, an `Artifacts/` window beside the app, and for E15 the two legacy
PNGs still in `Artifacts/fdt/`), and that the piece's gate table gains its walkthrough row once they
report. Do not fill a single result cell.

#### Amendments (binding — these supersede the text above)

These come from rulings P2/P75, P10, P17, P21, P39/P76, P40, P42, P43, P44, P53, P58, P62, P72 and P77 in the plan's preamble, and from checking every anchor against HEAD `673868d`. Where this block and the body disagree, this block wins. Say so in your report.

**A1. Files block.** Delete the line `Modify: docs/checklists/display-walkthrough.md:23,24,72 (rows 15, 16, B8) and its end`. Replace it with `Modify: docs/checklists/display-walkthrough.md — its end only (the new E-row section); rows 15, 16 and B8 are NOT edited (P42)`. Add `Modify: CLAUDE.md:163-167 (the suite list, Step 4)`. For `docs/superpowers/specs/2026-09-22-secondary-analyses-design.md`, add the lines §1.3 (one table row), §3.3 (one sentence) and §5.6 (one sentence) beside §12. **`core/tool/fdt.py` is NOT modified by this task.** P39 gave it to T41, but P76 supersedes that ('T29 keeps ... the two "no bounds file" sentences'). Instead, before Step 13, run `grep -n "no bounds file" core/tool/fdt.py`. Expect no output. If it prints anything, do not fix it: report it as a T29 defect.

**A2. Order paragraph.** Read it as: 'steps 2–12 edit documents and two docstrings; step 13 commits them; steps 14–17 are the gates of record, run by the CONTROLLER (never by the implementer, and never with a background run the implementer started) on step 13's tree; steps 18–22 write docs/STATE.md from the measured numbers.' An implementer subagent stops after Step 13 and hands Steps 14–17 to the controller.

**A3. Step 2.** The anchor ends mid-line. CLAUDE.md line 139 reads `  recorded downstream). No code outside `core/config.py` builds a literal `Resources/` or`. Replace only up to `recorded downstream).` and keep ` No code outside ...` and everything after it unchanged.

**A4. Step 3, first replacement (P30: name and note are not persisted either).** Use this text instead:

```markdown
  `PRISM.ini` is ignored, not restored. The Simulate, FDT and CrossVal panels remember their own
  numeric fields and their pickers' selections; the Seed box, the record name and the note are
  never remembered — a remembered seed would silently turn every run into a repeat of the last one,
  and a remembered name would be refused as taken at the next launch's first run (piece-5 spec
  §5.5, E7; plan ruling P30).
```

The second anchor (the `core/tool/` bullet) ends mid-line at `(list/show/note/rm/sweep/summary)).`. Line 171 continues ` `scripts/` is gone: six of`; keep that remainder.

**A5. Step 4 (P62).** The expected count is **nineteen**: T36 created `tests/test_fdt_compare.py`. Still measure it with `ls tests/test_*.py | wc -l`. If the measurement is 19, find:

```markdown
- `tests/` — the eighteen suites (`test_artifact_store.py` is the store's, `test_tool.py` the
  command-line tool's, `test_diagnostics.py` the five diagnostics', `test_refusals.py` the
  torch-free rules', tables' and run-buffer's, `test_artifact_browser.py` the artifact browser's;
```

and replace it with:

```markdown
- `tests/` — the nineteen suites (`test_artifact_store.py` is the store's, `test_tool.py` the
  command-line tool's, `test_diagnostics.py` the five diagnostics', `test_refusals.py` the
  torch-free rules', tables' and run-buffer's, `test_artifact_browser.py` the artifact browser's,
  `test_fdt_compare.py` the comparison facility's;
```

If the measurement is anything other than 19, write what you measured and name each new file.

**A6. Step 5: the second paragraph must state what T28's code does (P10).** After T28, `main` checks `getattr(args, "interrupt_note", None)` first. Otherwise it prints `_smoke_interrupt_advice(args, root) if temp_store_root else` the generic advice. So smoke's Ctrl-C advice is not an `interrupt_note`. In the replacement docstring, find the sentence:

```
letting them flip (piece-5 spec §6.1, E11): the throwaway root and its cleanup are ``smoke``'s
alone, chosen per subcommand instead of by the flag's presence, and a subcommand's Ctrl-C advice
follows its own ``interrupt_note``. The absence here is now what it says on the face of it -- this
family wants one root, the environment's.
```

and write instead:

```
letting them flip (piece-5 spec §6.1, E11): the throwaway root, its cleanup and smoke's Ctrl-C
advice are now keyed on ``temp_store_root``, a property only ``smoke``'s parser sets
(``set_defaults(temp_store_root=True)``), never on the flag's presence; ``fdt`` and ``crossval``
print their own ``interrupt_note``, which ``main`` checks first. The absence here is now what it
says on the face of it -- this family wants one root, the environment's.
```

Then re-read `core/tool/__init__.py`, as the body says, and adjust the text if T28 landed differently.

**A7. Steps 6–7 (P40: prove that only docstrings changed).** Step 7's command becomes `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m pytest tests/test_tool.py -m "not slow" -k "artifacts or help" -q`. Then run `git diff -U0 core/tool/browse.py tests/test_tool.py`. Every changed line must lie inside the browse.py module docstring or inside the docstring of `test_the_artifacts_family_takes_no_configuration_flags_and_its_help_costs_no_torch`. State in the report that no assertion line changed.

**A8. Step 10 is DELETED (P42).** Older walkthrough rows are never edited: rows 15, 16 and B8 keep their text exactly. The supersession is stated only in the new section's preamble.

**A9. Step 11 (P42, P44).** In the appended section, replace the first paragraph ('The rows piece 5 creates; ... simply the next letter after D.') and the line 'What the three superseded rows asserted, and why they no longer describe the application:' with:

```markdown
The rows piece 5 creates; the user runs them once on a real display, the way the A-, B-, C- and
D-rows were run. Rows 1–20, A1–A9, B1–B8, C1–C11 and D1–D17 are NOT edited — each is a dated
record of what was seen on the code as it then was — and piece 5 re-checks only what it changes.
Three of them no longer describe the application and are superseded HERE rather than marked in
place (design spec §10): rows 15, 16 and B8. The row letters here are independent of the piece's
decisions E1–E12; they are simply the next letter after D.

What the three superseded rows asserted, and why they no longer describe the application:
```

Keep the three bullets that follow. Replace the E15 row with:

```markdown
| E15 | The tidy-up offers the legacy loose files and the legacy directory | Leave the owner's legacy pictures in `Artifacts/fdt/` (the two PNGs stamped `20260915_153042`). No build writes `Artifacts/crossval/` any more, so for the legacy-directory half first create an EMPTY folder `Artifacts/crossval/` by hand. Then press Sweep on the Artifacts screen, once with all kinds and once with one kind chosen; then run `python -m core artifacts sweep` and `python -m core artifacts sweep --yes`. | Both front ends offer the loose FILES as their own clearly separated category, by name, alongside the manifest-less directories; the `crossval` legacy directory is offered by the all-kinds sweep only, never by a one-kind sweep; a dry run removes nothing; nothing valid is ever offered; after a confirmed removal the files and the empty `crossval/` are gone and every real record is untouched. | | |
```

The other rows stand as drafted, and the date and result columns stay empty.

**A10. Step 12: the anchor does not exist.** Spec §12 already holds a preamble and rows 1–9. There is no empty `| | | | |` row. Make four edits.

(a) **P72 is a planning-time deviation** (spec §1.2 says '`FDTConfig` gains two fields and one method'). Find:

```markdown
the practice pieces 2, 3 and 4 followed. Rows 1–9 were ruled at PLANNING time, before any code was
written, when ten drafters read the code for their tasks and raised 69 objections; the full texts are
in the ledger and the rulings are **P1–P69** in the plan. Rows from 10 on are filled during execution.
```

Replace it with:

```markdown
the practice pieces 2, 3 and 4 followed. Rows 1–10 were ruled at PLANNING time, before any code was
written: rows 1–9 when ten drafters read the code for their tasks and raised 69 objections (rulings
**P1–P69** in the plan), row 10 when three lenses and a judge read the assembled plan (**P72**, one
of P70–P82). The full texts are in the ledger. Rows from 11 on are filled during execution.
```

Then append after row 9:

```markdown
| 10 | §1.2 ("`FDTConfig` gains two fields and one method"), §4.4 (P72, planning time) | `FDTConfig` gains a third defaulted field, `preset_name: str \| None = None`; `make_param_sweep_config` passes it into the construction and the sweep reads `cfg.preset_name` for `body.settings["preset"]` | §4.4 requires the preset NAME in the record, and the builder's keyword alone carried it nowhere, so `settings["preset"]` would always have been null. A defaulted field, so §1.3's reduction-map constraint holds. *Cost if wrong: one more field on a shared dataclass.* |
```

Execution deviations from the ledger follow as rows 11, 12 and so on, in the four columns the body describes. If the ledger holds none, add one line under the table: `No execution-time deviation was ruled (<date>); rows 1–10 are all planning-time.`

(b) **P58.** A defect a round trip exposed and did not fix is NOT a §12 row. It goes in §1.3's table ('Out of scope, and who owns it') as a row `| <the defect> | <why not fixed here> | nobody yet |`. In §5.6, find `anything else is recorded in §12 and handed on, so the mandate is bounded.` and replace it with `anything else is recorded in §1.3 and handed on, so the mandate is bounded (**P58**).`

(c) **P53.** §6.2 says the `_merge_vals_bounds` wording 'is handed on in §1.3', but §1.3 has no such row. Append to §1.3's table:

```markdown
| A third wording of "the cell does not supply a parameter the bounds file declares", in `cli._merge_vals_bounds` (§6.2, **P53**) | It carries the cell PATH, which the FDT builders need and the other two sites do not; folding it into `missing_values_phrase` would drop that or change the other two | nobody yet |
```

(d) **P2 residue in §3.3.** Find:

```markdown
front-end knobs (§1.2), so they are checked with `field=None` and appear in neither front-end table.
```

and replace it with:

```markdown
front-end knobs (§1.2), so they are registered in `FIELDS` with `None` in both front-end tables and a
refusal about one names the setting and offers no fix (**P2**).
```

If execution rulings corrected the spec inline, append their P-numbers or R-numbers to the existing 'Ten further rulings corrected this document' sentence.

**A11. Step 18: the anchor is stale.** Find the real text:

```markdown
**Last updated:** 2026-09-22. **Piece 5, "the secondary analyses", is DESIGNED AND PLANNED, not yet
started**: brainstormed with the owner 2026-09-22 (decisions E1–E12), design `a0d85da`, plan
`e902256`+`5354e6b` (41 tasks, 433 steps, rulings P1–P82). The owner swaps models before
implementation. **Piece 4, "GUI usability and the artifact browser", is DONE**:
```

Replace everything before `**Piece 4, ...` with the piece-5 DONE paragraph. In 'what landed', after 'the sweep split into one record per swept parameter with its failure counts and its all-failed refusal', add: ', and an all-failed first sweep no longer costs the second — the temperature sweep runs regardless and the study refuses only if both measured nothing (P77)'. Also add '`--seed` on both subcommands (P79)'. The 'Piece 5 is under way ... the session that designed it stops after the plan.' anchor is at STATE.md:47-53. Keep the sentence that follows it (` Piece 3 (`3db271e`..`9e2f7ef`) is DONE and pushed ...`). If `git rev-list --count origin/main..main` shows piece 4 is still unpushed, the piece-4 'NOT YET PUSHED' clause stays; otherwise correct it.

**A12. Step 19 (P43).** The instruction 'Do **not** add a display-walkthrough row for E1–E16' is overruled. After the four measurement rows, add:

```markdown
| display walkthrough, piece-5 rows E1–E16 (`docs/checklists/display-walkthrough.md`) | **Not yet run.** Owed by the owner, on a real screen |
```

This matches piece 4's row at the same moment (commit `44ed3ed`). The session that records the owner's result REPLACES this row. It does not add a second.

**A13. Step 20: edit the existing entry, do not append a new one.** STATE's decisions log already holds `- **2026-09-22** — **piece 5 (the secondary analyses): decisions E1–E12**` directly after the piece-3 entry, and appending would duplicate it. The drafted text also contradicts P2/P75. Instead:

(a) Change that entry's date to `**2026-09-22/<the date of this step>**`, the piece-3 entry's `2026-09-15/17` style.

(b) In its design-rulings bullet, find:

```markdown
    deliberately widening piece 2's recorded reading; five checked settings that no front end exposes
    carry `field=None` and get no table entry.
```

and replace it with:

```markdown
    deliberately widening piece 2's recorded reading; five checked settings that no front end exposes
    (`freq_bounds`, `burn_in_nd`, `t_obs_periods`, `dt_nd`, `psd_t_obs_nd`) are registered in `FIELDS`
    like any other key, with `None` in BOTH front-end tables, so a refusal names the setting and
    `fix_sentence` offers no fix — not unregistered, because every `require_*` rule builds its
    sentence through `describe(key)`, which raises a bare `KeyError` for a key `FIELDS` does not hold
    (plan rulings P2, P75). The progressive mode is opt-in per kind, and the six ordinary kinds
    still lose their directory on any exception, pinned by a test written before the mode existed;
    the kind directory is the legacy `fdt/` on purpose, which makes the owner's stray pictures loose
    files INSIDE a kind directory, the only place the tidy-up can reach them.
```

Use the lower-case key spellings exactly as above: they are the registered keys.

(c) After the entry's last bullet (the one ending `rulings R1–R10, each carrying what it costs if it is wrong).`), append:

```markdown
  - The plan (`e902256`+`5354e6b`) carries 82 planning rulings, P1–P82; the spec's §12 rows 1–10 are
    the ones that depart from the spec. D11 and D12 stay standing refusals; this piece re-opened
    neither.
  - The execution rulings are in the spec's §12 (rows from 11) and in the gitignored ledger
    `.superpowers/sdd/2026-09-22-secondary-analyses/progress.md`.
```

Then add the bullet of durable execution rulings from Step 1, as the body says, each with its cost.

**A14. Step 21.**

- Replace the strike text '— one message builder, T21 and T29.' with '— one message builder, `core.refusals.missing_values_phrase`, delivered by T21 (P4, P76); T29 produces none.'
- To the plots strike add: '(tests only: all four functions already closed their figures since `bb22ac4`; P17)'.
- Strike the `tests/test_artifact_browser.py:864` item only if the ledger names the task that converted `assert sorted(seen) == ["observation", "posterior", "prior"], seen` to a subset assertion, and attribute it to that task. No task in the plan as written does this conversion. If none did, leave the item live with that reason.
- In the owed E-row item, change 'the gate table gains its walkthrough row when the owner has run them' to 'the gate table's walkthrough row, now reading "Not yet run", is filled in when the owner has run them'.
- Check each of these hand-on candidates against the ledger. Add those still open under 'Open, with no piece owning them yet', one line each in the '*From piece 5.*' voice:
  - the `_merge_vals_bounds` third wording (P53; spec §1.3);
  - the tool's now-redundant call-site preset fallback in `core/tool/fdt.py` (P50: 'A later piece may delete the now-redundant one');
  - `check_passive_baseline`'s bare `ValueError`, if T33 did not need to convert it (P23);
  - every defect T35's round trips recorded in spec §1.3 (P58).

**A15. Step 17.** In the recorded judgement, name T28's change to `core/tool/__init__.py` and `core/tool/smoke.py` explicitly. It is on `smoke`'s path, choosing the store root through `temp_store_root`, but it creates and moves no tensor, and T28's tests pin the root choice. The judgement must say this rather than leave the reader to find the change in the diff.

**A16. Step 23 (P44).** Tell the owner that E15 also needs an empty `Artifacts/crossval/` created by hand before it is run. The two legacy PNGs are already in `Artifacts/fdt/` (verified at `673868d`: `fdt_ratio_passive_20260915_153042.png` and `spontaneous_trajectory_20260915_153042.png`).

#### Controller rulings (pre-flight, binding — these supersede the amendments above and the task text)

- **F50** — do NOT edit `core/tool/fdt.py`'s "no bounds file" sentences; Task 29 owns them.
- **F26** — re-read the spec's §1.3 table before adding the P53 row (Task 35 may have appended rows) and append after whatever the last row is at that moment.
- **F62/F63** — walkthrough row E12 exercises the unfinished-record refusal from the command line (`python -m core compare repeats --record <unfinished> --record <other>`), because the window's pickers offer finished records only.
- **F64** — update `CLAUDE.md`'s Tests bullet, which names the slow set as two tests with piece-3 timings, in the second (documents) commit, from the slow set's measured numbers.
- **F65** — the piece's first commit is `a0d85da` (the design), as piece 4's paragraph counts from its design commit; `e902256` and `5354e6b` are the plan, and the pre-flight amendments commit is the one after `673868d` whose subject names the pre-flight — take its hash from `git log`.
- **F66** — reword the E-section preamble: "E8 replaces it for a sweep; the single-cell refusal (spec §3.6) has no row of its own and is pinned by Task 16's tests."
- **F24** — strike the `:864` item in STATE.md only if the ledger shows a task converted it (Task 25 does), and attribute it to Task 25.

---
