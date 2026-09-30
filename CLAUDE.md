# PRISM — instructions for Claude Code

Read `docs/STATE.md` first: it is the moving state (what is done, what is owed and in what order,
what is on disk, the last gate). Update it at the end of every session.

## Environment

- Interpreter: the conda env `biophys-env`, `C:\Users\J\anaconda3\envs\biophys-env\python.exe`
  (Python 3.12, torch 2.9.0+cu130, PySide6 6.9.3, sbi 0.25.0 installed while 0.26.1 is pinned).
  The `python` on PATH is NOT this env and has none of the dependencies.
- Anything that imports torch needs `KMP_DUPLICATE_LIB_OK=TRUE` (torch + MKL ship two OpenMP
  runtimes; without it the first simulation aborts with OMP Error #15 and no traceback), and any
  HEADLESS check that imports PySide6 also needs `QT_QPA_PLATFORM=offscreen`. The root
  `conftest.py` defaults both; `run.bat` and `run.sh` set `KMP_DUPLICATE_LIB_OK` only, because the
  GUI needs a real platform plugin. A bare `python -c` or script sets neither.
- CUDA is available (RTX 5070 Ti, 16 GB shared with the desktop). Read free VRAM with
  `nvidia-smi --query-gpu=memory.used --format=csv`, never `torch.cuda.mem_get_info()` (it
  overstates free memory by the desktop's share).
- Launch the GUI with `run.bat` (from a shell where `biophys-env` is on PATH; it sets
  `KMP_DUPLICATE_LIB_OK` for you). A direct launch must set it by hand, because
  `core/gui/__main__.py` sets only the Agg backend:
  `$env:KMP_DUPLICATE_LIB_OK="TRUE"; & "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m core.gui`.
  The command-line tool is `python -m core <subcommand>` (`core/tool/`; `--help` lists them): its
  entry sets `KMP_DUPLICATE_LIB_OK` and the Agg backend itself, before any torch or core import, so
  it needs no environment set up around it. Every tool flag maps 1:1 onto a stage argument;
  the tool reads no environment but the two roots and rebinds no module constant. Paths do not
  depend on the working directory: `core/config.py` resolves `RESOURCES_ROOT` (inputs;
  `PRISM_RESOURCES` overrides) and `artifacts_root()` (generated artifacts; `PRISM_ARTIFACTS`
  overrides) from its own location. Point `PRISM_ARTIFACTS` at a scratch directory, or use
  `core.artifacts.use_store`, to keep a check away from the real `Artifacts/`.
- Two core-level environment settings stay environment settings, deliberately (piece 2's D13), and
  are named in the tool's `--help` epilog: `PRISM_VRAM_CEILING_GIB` is read live on each batch plan;
  `PRISM_MEM_LOG_EVERY` is read ONCE, when `core.SBI.pipeline` is imported, so setting it inside a
  running GUI changes nothing.

## Tests

- pytest. Fast gate: `pytest -m "not slow"` (the recorded one-process target is 16 minutes, piece-3
  spec §8.4). Full: `pytest` (never timed in one process; the two halves at the piece-6 gates were
  15 min 24 s + 1 h 15 min 27 s, so budget about 95 minutes. The slow set is five tests: the chi
  full-pipeline test `test_chi_mode_full_sbi_pipeline` in `tests/test_user_sbi.py` (2618 s; 1397 s
  at piece 5 — the open list), in `tests/test_tool.py` the Nadrowski sanity-path run
  `test_fdt_runs_the_nadrowski_sanity_checks_end_to_end` (619 s) and the FDT/crossval tiny-size run
  `test_fdt_and_crossval_run_at_tiny_size` (452 s), the tier-1 tiny `smoke`
  `test_smoke_runs_every_stage_on_the_tier1_box` in `tests/test_tier1.py` (828 s) and the probes
  twin-cell check `test_probes_band_gives_the_same_criteria_on_the_master_cell_and_its_tier1_twin`
  in `tests/test_diagnostics.py` (7 s): 1 h 15 min 27 s at `067eddc`; the slow set alone is
  `pytest -m slow`). Count: `pytest --collect-only -q`.
- Markers: `slow`; `gpu` (skipped when CUDA is absent); `display` (skipped offscreen). The
  display-marked tests run on the real screen with `QT_QPA_PLATFORM=windows pytest -m display`
  (the root conftest only DEFAULTS the variable); they create hidden native windows, nothing shows.
- Do not edit a source file while a suite is running. Several tests assert on
  `inspect.getsource`, which reads the file as it is now with the line numbers the function was
  loaded with; an edit mid-run produces a failure that is not real. Never run two pytest
  processes at once.
- The suites run against a temp artifact root (`tests/conftest.py` installs it once per process
  and asserts at teardown that the real `Artifacts/` gained nothing). Only a SINGLE-process run
  can catch code that swaps the default store and never restores it; splitting the gate into
  several processes hides that, so the recorded gate is one `pytest -m "not slow"` invocation.
  A tool call cannot hold a run longer than ten minutes: start long runs in the background,
  logging to a file, and touch no source until they exit.
- `tests/conftest.py` turns training checkpointing OFF for the session
  (`_checkpointing_off_unless_asked`, which rebinds `orchestrator.TRAINING_CHECKPOINT_EVERY`); it
  is the only session-wide KNOB default the suites install, and a test that wants a simulation
  cache passes `checkpoint_every` explicitly. Four more session-wide autouse guards stand beside
  it: a temp artifact root (`_sandbox_default_store`), a temp settings file so no panel reads the
  real `PRISM.ini` and no test can write it (`_settings_home`, plus a fresh `.ini` per test), no
  modal dialog can stall the run (`_no_modal_dialogs`), and `<repo>/sbi-logs` must not exist
  (`_no_sbi_logs`) — asserted at SETUP too, so a leftover tree makes every pytest run error before
  the first test.
- A green suite does not certify the GPU path. The gpu-marked tests do run on the card inside every
  fast gate when CUDA is present (`tests/test_gpu_paths.py`'s CUDA inference run, six in
  `tests/test_user_sbi.py` and one in `tests/test_diagnostics.py`), but they are not a substitute: no
  checkpoint resume, no real bounds or
  cell files. After touching code that moves tensors, run the smoke gate on the card — five command
  lines, from the repo root — and check `$LASTEXITCODE` after each one:

  ```powershell
  $py = "C:\Users\J\anaconda3\envs\biophys-env\python.exe"
  $B  = "--bounds","Resources/Bounds/nadrowski/master.txt"
  $C  = "--cell","Resources/Cells/nadrowski/master_spont.txt"
  $S  = "<scratch>"
  # run 1: chi; builds and names smoke_prior/smoke_posterior; writes the simulation cache
  & $py -m core smoke --chi --t-obs 4.5 @B @C --checkpoint --save --store-root "$S/smoke"
  # run 2: same store, same --num-runs; must resume
  & $py -m core smoke --chi --t-obs 4.5 @B @C --checkpoint --store-root "$S/smoke" --prior smoke_prior --stages prior,posterior --resume require
  # run 2b: one setting away; must exit 1 naming n_runs
  & $py -m core smoke --chi --t-obs 4.5 @B @C --checkpoint --store-root "$S/smoke" --prior smoke_prior --stages prior,posterior --num-runs 2
  # run 3: forced mode, its own store
  & $py -m core smoke --no-chi --t-obs 4.5 @B --cell Resources/Cells/nadrowski/master_weak.txt --checkpoint --save --store-root "$S/smoke_chi0"
  # run 4: the tier-1 box (temperature inferred, the force scale derived) at the larger network, its own store
  & $py -m core smoke --chi --t-obs 4.5 --bounds Resources/Bounds/nadrowski/master_tier1.txt --cell Resources/Cells/nadrowski/master_spont_tier1.txt --hidden-features 256 --num-transforms 10 --checkpoint --save --store-root "$S/smoke_t1"
  ```

  Pass criteria, read off `$LASTEXITCODE` and the printed lines after each command:
  - runs 1, 3 and 4: `$LASTEXITCODE` 0, no OOM lines, stage timings near the last recorded card
    run; on the two chi runs (1 and 4) the training `[chi] masked probes` run-total line (the one
    scoped to every committed batch of the simulation cache; the calibration stage logs a second
    one, scoped to this process only, which is not the criterion) within ±12 pp of 37 %;
  - run 2: prints `Reusing the Fisher rotation stored with the training checkpoint` and
    `[checkpoint] resuming at batch 4/4`, `$LASTEXITCODE` 0;
  - run 2b: `$LASTEXITCODE` 1, the refusal names `n_runs`, no `[fisher]` line, no new directory
    under `$S/smoke/simulations/`;
  - run 4, also: both `[tier1]` lines, `rescale order: ['x_scale', 't_scale', 'T']` in the banner,
    and `smoke_posterior`'s `manifest.json` lists `T` in `body.transform.param_keys` and carries a
    `config.tier1` block.

  `--bounds` is required: the same-named sibling rule would otherwise resolve the 12-dim
  spontaneous box. After a change under `core/diagnostics` that moves tensors, also run the
  diagnostic card, each command with `$env:PRISM_ARTIFACTS` set to the store named. Against
  `$S/smoke`: `sbc --chi @B --posterior smoke_posterior --repeats 1 --n-cal 20`,
  `identifiability jacobian --chi @B @C --t-obs 4.5` and `ablation --chi @B --posterior
  smoke_posterior`. Against `$S/smoke_chi0`: `identifiability laplace --no-chi @B --posterior
  smoke_posterior --cell Resources/Cells/nadrowski/master_weak.txt --t-obs 4.5`. Against
  `$S/smoke_t1`: `identifiability jacobian --chi --bounds Resources/Bounds/nadrowski/master_tier1.txt
  --cell Resources/Cells/nadrowski/master_spont_tier1.txt --t-obs 4.5` (its temperature column is a
  measurement, not a failure). Against an empty scratch store: `probes band @B @C` and `probes drive
  @B @C`. Against `$S/smoke`: `probes mask @B --prior smoke_prior`. Each exits 0 and writes one
  record under `diagnostics/`. Delete `$S` afterwards.
  The last result, with its stage timings, is in `docs/STATE.md`'s gate table; that table and
  `docs/guide/testing.md` carry this block verbatim.
- A foreground `python` check that imports torch and touches the prior or checkpoint machinery
  can hang the tool call for good. Write such checks to a script and run them with a timeout.
- Do not pipe large Python or Markdown through a bash heredoc (`cat <<EOF`); it dies on
  multi-line content here. Write the file with the Write tool and run it.

## Rules that are load-bearing

- Every knob is an ARGUMENT, never a config write. `core/orchestrator.py` binds config constants
  at import (`from .config import X`), so assigning `config.X` at runtime changes nothing and the
  run silently uses the default. Each tunable travels as an argument (some positionally, e.g.
  `build_prior(cfg, None, True, …)`); tests pin this.
- No public stage, composition or diagnostic mutates the configuration it is handed:
  `core.runs.public_entry` copies it on entry (piece 3, V1). A caller that wants what a stage
  wrote reads the artifact.
- Every pre-spend refusal is a `core.refusals.Refusal` with a field key; the front ends name
  the control or flag from their own tables (`core/gui/fields.py`, `core/tool/fields.py`). In
  the converted modules no message names a box, tab, flag or button. Stage messages are
  `logging` records at info/warning/error; never print or log between steps 1 and 3 of a
  checkpoint save.
- Nothing shippable cites the working record (piece 6, H4): no file under `core/` or `tests/`, nor
  `conftest.py`, `README.md`, `requirements.txt`, `pytest.ini`, `run.bat`, `run.sh` or
  `docs/guide/`, may cite the handoff, `docs/STATE.md`, this file, the specs or the plans, or their
  labels (piece numbers, "§N", decision, ruling and review ids, "Task N", trap ids, walkthrough row
  ids). A provenance tag is deleted; a reason given by reference is written out in words.
  `tests/test_source_hygiene.py` enforces it; a legitimate new token shaped like a label (a paper's
  section, a new feature id) gets an allowlist entry there in the commit that introduces it. The
  working record may cite code and each other; no test or code reads it.
- The inference tabs remember SELECTIONS — the pickers, the mode, and the boxes that describe the
  recording (the three observation lengths, the physical drive amplitude `Drive F₀ (N)`, the
  recording paths) — plus the training budget (piece 3, V5). Every science knob opens at
  `config.py` on every launch (the chi probe count, slots and lock-in ceiling, the sweep, network,
  rotation and calibration knobs, the HPD level and the direction count), the Config tab's
  non-dimensional χ drive amplitude and band are read-only displays of it, and a consent is never
  persisted — it is answered by the session that runs. A stale key an older build left in
  `PRISM.ini` is ignored, not restored. The Simulate, FDT and CrossVal panels remember their own
  numeric fields — except the ones the CrossVal panel re-derives instead, n_freqs and M_ensemble
  from its preset and the grid ends from its cell — and their pickers' selections; the Seed box,
  the record name and the note are never remembered — a remembered seed would silently turn every
  run into a repeat of the last one, and a remembered name would be refused as taken at the next
  launch's first run (piece-5 spec §5.5, E7; plan ruling P30) — and neither are the "Compare
  saved …" controls, whose remembered list would name records a later session may have deleted.
- `Resources/` holds the hand-edited inputs (`Bounds/`, `Cells/`, `Units/`, `Models/`). The
  bounds file declares WHICH parameters are inferred and in what order, and therefore the
  observation mode; a cell's model comes from its parent folder. Everything generated lives under
  `Artifacts/<kind>/<name>__<id>/` with a `manifest.json` and, since piece 3, a `log.txt` of that
  run's records — every `core` record and Python warning since the outermost public entry began,
  stamped `HH:MM:SS info/warning/error` (`core/runs.py`) — for every kind but the simulation cache,
  which has no writer. The store's code is `core/artifacts/`; the `Artifacts/` tree is gitignored.
  The kinds are priors, simulations (the training cache, keyed by identity digest), posteriors,
  observations, calibrations, inferences, diagnostics and fdt (the effective-temperature
  measurement, the parameter sweep and their comparisons — piece 5). Every stage writes its
  artifact at completion and returns a `Loaded*` wrapper, with two exceptions: the training cache,
  which commits batch by batch and has no writer at all, and an `fdt` record, which is
  PROGRESSIVE — its directory and a first manifest exist from the moment the run starts,
  `refresh()` rewrites them as it goes, and a cancel or a crash keeps the folder, marked unfinished
  (piece-5 spec §2.2, E2). Save is a rename; loading refuses a mismatch in
  what it compares (`docs/guide/architecture.md`'s "The artifact store" lists the checks and the keys no
  loader compares yet — the open list's load-check audit), and `Accept(truncated, other_observation)` are
  the only escape hatches (each use is
  recorded downstream). No code outside `core/config.py` builds a literal `Resources/` or
  `Artifacts/` path (a test pins it). The old `Resources/{Priors,Posteriors,Checkpoints,Observations,
  Plots,CrossValidation,ReductionMap}` trees were deleted by the clean-break runbook on 2026-09-11
  (design spec §9); `Resources/` holds only the four input folders and `Artifacts/` started empty.
- The narrowing-round safety rules (where each is enforced and the test that pins it) and the traps
  that still bite are in `docs/guide/rules-and-traps.md`; the reasoning behind the science settings
  is in `docs/guide/science.md`. The handoff they were written from is archived (see `docs/STATE.md`).
- Git: work directly on the local `main` branch — no feature branches, no worktrees (decided
  2026-09-11 after piece 1's merge). Claude makes the local commits, one per logical step, never
  amended, ending with the Co-Authored-By line the harness provides. Commit messages are SHORT:
  a one-line subject, a brief body only when the subject cannot carry the why. The user pushes
  and handles every other remote operation.
  Undo an edit with `git stash`, never `git checkout -- <file>`, and read `git status` before
  touching a file you did not create: `checkout --` once discarded about seventy lines of
  uncommitted test work.
- Archiving a file means MOVING it on disk into the gitignored `archive/` and then `git rm
  --cached` — never `git mv`, which would re-create a tracked-but-ignored file. A script that was
  folded into a subcommand is `git rm`'d instead: git history is its archive.

## Where things are

- `docs/STATE.md` — the moving state. Read first, update last.
- `docs/superpowers/specs/` — approved designs. `docs/superpowers/plans/` — implementation plans.
- `.superpowers/sdd/<date>-<slug>/` — a piece's gitignored execution workspace: the ledger
  (`progress.md`), the task briefs and reports, the review reports and the per-task gate logs.
  Piece 2 is `2026-09-12-one-flow`, piece 3 `2026-09-16-validation-and-logging`, piece 5
  `2026-09-22-secondary-analyses`, piece 6 `2026-09-25-documentation-and-retrain-readiness`.
  Untracked, so no git command finds them.
- `docs/checklists/display-walkthrough.md` — GUI features never exercised on a real screen.
- `docs/guide/` — the reader pages, starting at its `README.md`: getting started, the window, the
  command line, bringing recordings, the science, the architecture, the rules and traps, testing,
  and the retrain runbook. They cite no working document (the source scan walks them), and each
  opens with the commit it was checked against.
- `tests/` — the twenty-two suites (`test_artifact_store.py` is the store's, `test_tool.py` the
  command-line tool's, `test_diagnostics.py` the diagnostics' (the five and the three `probes`
  modes), `test_refusals.py` the torch-free rules', tables' and run-buffer's,
  `test_artifact_browser.py` the artifact browser's, `test_fdt_compare.py` the comparison
  facility's, `test_source_hygiene.py` the reference scan over the shippable files, `test_tier1.py`
  the tier-1 box's path, and `test_docs.py` the guide's links and the command-line page against the
  parser; `_fixtures.py` holds the shared stand-ins, the
  tiny real prior+posterior, `build_fdt_record` (a single-cell or sweep `fdt` record, finished or
  not, written in seconds) and `compare_preflight_refusals` (every comparison refusal made after
  the records are read, shared by the API's and the tool's tests), and `CODE_ROOTS` plus
  `CODE_FILES`, the directories and top-level files the source scans walk); `core/Reduction/tests/`
  — the reduction map's (out of scope).
- `core/tool/` — the command-line tool: `python -m core --help` lists every subcommand (the stages
  `prior train validate infer tsnpe`, the diagnostics `sbc identifiability ablation probes`
  (band/mask/drive), plus `smoke`,
  `fdt`, `crossval`, `compare` (cells/repeats/renormalise/sweeps) and `artifacts`
  (list/show/note/rm/sweep/summary)); `fdt` and `crossval` take `--store-root`, `--seed`, `--name`
  and `--note`, and a `crossval` study's two records are named `<name>-s` and `<name>-temp`.
  `scripts/` is gone: six of its scripts became `smoke` and the diagnostics (git history keeps
  them), and six more joined the gitignored `archive/scripts/`. The Artifacts screen (piece 4) is
  the store's GUI front end, over the same `core/artifacts/` engine the tool's `artifacts` family
  reads.
