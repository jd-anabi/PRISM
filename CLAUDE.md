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
- Launch the GUI with `run.bat` (or `python -m core.gui`). Paths do not depend on the working
  directory: `core/config.py` resolves `RESOURCES_ROOT` (inputs; `PRISM_RESOURCES` overrides) and
  `artifacts_root()` (generated artifacts; `PRISM_ARTIFACTS` overrides) from its own location.
  Point `PRISM_ARTIFACTS` at a scratch directory, or use `core.artifacts.use_store`, to keep a
  check away from the real `Artifacts/`.

## Tests

- pytest. Fast gate: `pytest -m "not slow"` (minutes). Full: `pytest` (about an hour; the chi
  full-pipeline test in `tests/test_user_sbi.py`). Count: `pytest --collect-only -q`.
- Markers: `slow`; `gpu` (skipped when CUDA is absent); `display` (skipped offscreen).
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
- A green suite does not certify the GPU path (every suite runs on the CPU). After touching code
  that moves tensors, run `scripts/smoke_train.py` on the card with an explicit
  `BOUNDS=Resources/Bounds/nadrowski/master.txt`; the last result is in `docs/STATE.md`.
- A foreground `python` check that imports torch and touches the prior or checkpoint machinery
  can hang the tool call for good. Write such checks to a script and run them with a timeout.
- Do not pipe large Python or Markdown through a bash heredoc (`cat <<EOF`); it dies on
  multi-line content here. Write the file with the Write tool and run it.

## Rules that are load-bearing

- Every knob is an ARGUMENT, never a config write. `core/orchestrator.py` binds config constants
  at import (`from .config import X`), so assigning `config.X` at runtime changes nothing and the
  run silently uses the default. Each tunable travels as a keyword argument; tests pin this.
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

## Where things are

- `docs/STATE.md` — the moving state. Read first, update last.
- `docs/superpowers/specs/` — approved designs. `docs/superpowers/plans/` — implementation plans.
- `docs/checklists/display-walkthrough.md` — GUI features never exercised on a real screen.
- `tests/` — the thirteen suites (`test_artifact_store.py` is the store's; `_fixtures.py` holds the
  shared stand-ins and the tiny real prior+posterior); `core/Reduction/tests/` — the reduction
  map's (out of scope).
- `scripts/` — diagnostics configured by environment variables (each file's docstring lists them).
