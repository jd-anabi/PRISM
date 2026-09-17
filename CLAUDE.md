# PRISM — instructions for Claude Code

Read `docs/STATE.md` first: it is the moving state (what is done, what is owed and in what order,
what is on disk, the last gate). Update it at the end of every session.

## Environment

- Interpreter: the conda env `biophys-env`, `C:\Users\J\anaconda3\envs\biophys-env\python.exe`
  (Python 3.12, torch 2.9.0+cu130, PySide6 6.9.3, sbi 0.25.0 installed while 0.26.1 is pinned).
  The `python` on PATH is NOT this env and has none of the dependencies.
- Anything that imports torch or PySide6 needs `QT_QPA_PLATFORM=offscreen` (headless Qt) and
  `KMP_DUPLICATE_LIB_OK=TRUE` (torch + MKL ship two OpenMP runtimes; without it the first
  simulation aborts with OMP Error #15 and no traceback). `conftest.py`, `run.bat` and `run.sh`
  set them; a bare `python -c` or script does not.
- CUDA is available (RTX 5070 Ti, 16 GB shared with the desktop). Read free VRAM with
  `nvidia-smi --query-gpu=memory.used --format=csv`, never `torch.cuda.mem_get_info()` (it
  overstates free memory by the desktop's share).
- Launch the GUI with `run.bat` (or `python -m core.gui`). The command-line tool is `python -m core
  <subcommand>` (`core/tool/`; `--help` lists them): its entry sets `KMP_DUPLICATE_LIB_OK` and the
  Agg backend itself, before any torch or core import, so it needs no environment set up around it.
  Every tool flag maps 1:1 onto a stage keyword argument; the tool reads no environment but the two
  roots and rebinds no module constant. Paths do not depend on the working directory:
  `core/config.py` resolves `RESOURCES_ROOT` (inputs; `PRISM_RESOURCES` overrides) and
  `artifacts_root()` (generated artifacts; `PRISM_ARTIFACTS` overrides) from its own location.
  Point `PRISM_ARTIFACTS` at a scratch directory, or use `core.artifacts.use_store`, to keep a
  check away from the real `Artifacts/`.
- Two core-level environment settings stay environment settings, deliberately (piece 2's D13), and
  are named in the tool's `--help` epilog: `PRISM_VRAM_CEILING_GIB` is read live on each batch plan;
  `PRISM_MEM_LOG_EVERY` is read ONCE, when `core.SBI.pipeline` is imported, so setting it inside a
  running GUI changes nothing.

## Tests

- pytest. Fast gate: `pytest -m "not slow"` (the recorded one-process target is 16 minutes, piece-3
  spec §8.4). Full: `pytest` (about an hour; the slow set is the chi full-pipeline test in
  `tests/test_user_sbi.py` and the FDT/crossval tiny-size run in `tests/test_tool.py`, ~10 min; the
  slow set alone is `pytest -m slow`). Count: `pytest --collect-only -q`.
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
  (`_checkpointing_off_unless_asked`); it is the only session-wide knob default the suites
  install, and a test that wants a simulation cache passes `checkpoint_every` explicitly.
- A green suite does not certify the GPU path (every suite runs on the CPU). After touching code
  that moves tensors, run the smoke gate on the card — four command lines, from the repo root —
  and check `$LASTEXITCODE` after each one:

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
  ```

  Pass criteria, read off `$LASTEXITCODE` and the printed lines after each command:
  - run 1 and run 3: `$LASTEXITCODE` 0, no OOM lines, stage timings near the last recorded gate in
    `docs/STATE.md`, masked-probe counts within ±12 pp of 37 %;
  - run 2: prints `Reusing the Fisher rotation stored with the training checkpoint` and
    `[checkpoint] resuming at batch 4/4`, `$LASTEXITCODE` 0;
  - run 2b: `$LASTEXITCODE` 1, the refusal names `n_runs`, no `[fisher]` line, no new directory
    under `$S/smoke/simulations/`.

  `--bounds` is required: the same-named sibling rule would otherwise resolve the 12-dim
  spontaneous box. After a change under `core/diagnostics` that moves tensors, also run the
  diagnostic card (`sbc`, `identifiability jacobian`, `ablation` against `$S/smoke` with
  `$env:PRISM_ARTIFACTS` set; `identifiability laplace` against `$S/smoke_chi0`), as recorded in
  `docs/STATE.md`'s piece-2 GPU gate row. Delete `$S` afterwards; the last result is in
  `docs/STATE.md`.
- A foreground `python` check that imports torch and touches the prior or checkpoint machinery
  can hang the tool call for good. Write such checks to a script and run them with a timeout.
- Do not pipe large Python or Markdown through a bash heredoc (`cat <<EOF`); it dies on
  multi-line content here. Write the file with the Write tool and run it.

## Rules that are load-bearing

- Every knob is an ARGUMENT, never a config write. `core/orchestrator.py` binds config constants
  at import (`from .config import X`), so assigning `config.X` at runtime changes nothing and the
  run silently uses the default. Each tunable travels as a keyword argument; tests pin this.
- No public stage, composition or diagnostic mutates the configuration it is handed:
  `core.runs.public_entry` copies it on entry (piece 3, V1). A caller that wants what a stage
  wrote reads the artifact.
- Every pre-spend refusal is a `core.refusals.Refusal` with a field key; the front ends name
  the control or flag from their own tables (`core/gui/fields.py`, `core/tool/fields.py`). In
  the converted modules no message names a box, tab, flag or button. Stage messages are
  `logging` records at info/warning/error; never print or log between steps 1 and 3 of a
  checkpoint save.
- `Resources/` holds the hand-edited inputs (`Bounds/`, `Cells/`, `Units/`, `Models/`). The
  bounds file declares WHICH parameters are inferred and in what order, and therefore the
  observation mode; a cell's model comes from its parent folder. Everything generated lives under
  `Artifacts/<kind>/<name>__<id>/` with a `manifest.json` (`core/artifacts/`, gitignored): priors,
  simulations (the training cache, keyed by identity digest), posteriors, observations,
  calibrations, inferences. Every stage writes its artifact at completion and returns a `Loaded*`
  wrapper; Save is a rename; loading refuses any verifiable mismatch, and `Accept(truncated,
  other_observation)` are the only escape hatches (each use is recorded downstream). No code
  outside `core/config.py` builds a literal `Resources/` or `Artifacts/` path (a test pins it).
  The old `Resources/{Priors,Posteriors,Checkpoints,Observations,Plots,CrossValidation,
  ReductionMap}` trees were deleted by the clean-break runbook on 2026-09-11 (design spec §9);
  `Resources/` holds only the four input folders and `Artifacts/` started empty.
- The science guardrails are in `PRISM_HANDOFF.md` §11.6 (TSNPE) and the traps in §5; the
  handoff is being split into `docs/` by piece 6 but is still the reference until then.
- Git: work directly on the local `main` branch — no feature branches, no worktrees (decided
  2026-09-11 after piece 1's merge). Claude makes the local commits, one per logical step, never
  amended, ending with the Co-Authored-By line the harness provides. Commit messages are SHORT:
  a one-line subject, a brief body only when the subject cannot carry the why. The user pushes
  and handles every other remote operation.
- Archiving a file means MOVING it on disk into the gitignored `archive/` and then `git rm
  --cached` — never `git mv`, which would re-create a tracked-but-ignored file. A script that was
  folded into a subcommand is `git rm`'d instead: git history is its archive.

## Where things are

- `docs/STATE.md` — the moving state. Read first, update last.
- `docs/superpowers/specs/` — approved designs. `docs/superpowers/plans/` — implementation plans.
- `docs/checklists/display-walkthrough.md` — GUI features never exercised on a real screen.
- `tests/` — the seventeen suites (`test_artifact_store.py` is the store's, `test_tool.py` the
  command-line tool's, `test_diagnostics.py` the five diagnostics', `test_refusals.py` the
  torch-free rules', tables' and run-buffer's; `_fixtures.py` holds the shared stand-ins, the tiny
  real prior+posterior, and `CODE_ROOTS` plus `CODE_FILES`, the directories and top-level files the
  source scans walk); `core/Reduction/tests/` — the reduction map's (out of scope).
- `core/tool/` — the command-line tool: `python -m core --help` lists every subcommand (the stages
  `prior train tsnpe validate infer`, the diagnostics `sbc identifiability ablation`, plus `smoke`,
  `fdt` and `crossval`). `scripts/` is gone: six of its scripts became `smoke` and the diagnostics
  (git history keeps them), and six more joined the gitignored `archive/scripts/`.
