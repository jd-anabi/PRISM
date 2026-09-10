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
- Launch the GUI from the repo root: `run.bat` (or `python -m core.gui`). Scripts run from the
  repo root too: `core/config.py` builds the `Resources/` paths from the working directory.

## Tests

- pytest. Fast gate: `pytest -m "not slow"` (minutes). Full: `pytest` (about an hour; the chi
  full-pipeline test in `tests/test_user_sbi.py`). Count: `pytest --collect-only -q`.
- Markers: `slow`; `gpu` (skipped when CUDA is absent); `display` (skipped offscreen).
- Do not edit a source file while a suite is running. Several tests assert on
  `inspect.getsource`, which reads the file as it is now with the line numbers the function was
  loaded with; an edit mid-run produces a failure that is not real.
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
  observation mode; a cell's model comes from its parent folder. Generated artifacts live in
  `Resources/{Priors,Posteriors,Checkpoints,Observations,Plots}` today and in `Artifacts/` once
  piece 1 of the hardening programme lands (`docs/superpowers/specs/`).
- The science guardrails are in `PRISM_HANDOFF.md` §11.6 (TSNPE) and the traps in §5; the
  handoff is being split into `docs/` by piece 6 but is still the reference until then.
- Git: make local commits only; the user handles remote operations. Do not amend; one commit
  per logical step; end messages with the Co-Authored-By line the harness provides.

## Where things are

- `docs/STATE.md` — the moving state. Read first, update last.
- `docs/superpowers/specs/` — approved designs. `docs/superpowers/plans/` — implementation plans.
- `docs/checklists/display-walkthrough.md` — GUI features never exercised on a real screen.
- `tests/` — the twelve suites; `core/Reduction/tests/` — the reduction map's (out of scope).
- `scripts/` — diagnostics configured by environment variables (each file's docstring lists them).
