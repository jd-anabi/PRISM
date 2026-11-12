# PRISM — instructions for Claude Code

Read `docs/STATE.md` first: it is the moving state (what is done, what is owed and in what order,
what is on disk, the last gate). Update it at the end of every session.

## Environment

- Interpreter: the conda env `biophys-env`, `C:\Users\J\anaconda3\envs\biophys-env\python.exe`
  (Python 3.12, torch 2.9.0+cu130, PySide6 6.9.3, sbi 0.26.1 as pinned since 2026-10-01, with skorch
  and tabulate; 0.25.0 before). The `python` on PATH is NOT this env and has none of the dependencies.
- Anything that imports torch needs `KMP_DUPLICATE_LIB_OK=TRUE` (torch + MKL ship two OpenMP
  runtimes; without it the first simulation aborts with OMP Error #15 and no traceback), and any
  HEADLESS check that imports PySide6 also needs `QT_QPA_PLATFORM=offscreen`. The root
  `conftest.py` defaults both; `run.bat` and `run.sh` set `KMP_DUPLICATE_LIB_OK` only, because the
  GUI needs a real platform plugin; the two entry modules (`python -m core.gui`, `python -m core`)
  set `KMP_DUPLICATE_LIB_OK` and the Agg backend themselves, before any torch import. A bare
  `python -c` or script sets neither.
- CUDA is available (RTX 5070 Ti, 16 GB shared with the desktop). Read free VRAM with
  `nvidia-smi --query-gpu=memory.used --format=csv`, never `torch.cuda.mem_get_info()` (it
  overstates free memory by the desktop's share).
- The GUI: `run.bat` (double-click it, or type it in any shell) starts PRISM with no console and
  returns at once; a second start brings the open window to the front. `docs/guide/getting-started.md`,
  "Launching PRISM", says which Python it finds, the desktop shortcut `PRISM.lnk` it makes once (with
  the mark `shortcut-made` beside the settings file), `run.bat --where` and `run.bat --make-shortcut`
  (neither starts anything), and where the output goes: a run's full text in the dated file its page
  opens, in the `runs` folder beside the settings file (`%APPDATA%\PRISM`), everything else the window
  prints, and a crash, in `window-log.txt` there, which the plain Windows message of a start that
  failed names. `PRISM_DESKTOP` (the shortcut's folder) is for the
  launcher's tests alone. The shortcut starts PRISM with Explorer's environment, so a
  `PRISM_ARTIFACTS` or `PRISM_VRAM_CEILING_GIB` set in a shell reaches the window only through
  `run.bat` started from that shell. A direct launch keeps its console:
  `& "C:\Users\J\anaconda3\envs\biophys-env\python.exe" -m core.gui`.
- The command-line tool is `python -m core <subcommand>` (`core/tool/`; `--help` lists them). Every
  tool flag maps 1:1 onto a stage argument; the tool reads no environment but the two roots and
  rebinds no module constant. Paths do not depend on the working directory: `core/config.py`
  resolves `RESOURCES_ROOT` (inputs; `PRISM_RESOURCES` overrides) and `artifacts_root()` (generated
  artifacts; `PRISM_ARTIFACTS` overrides) from its own location. Point `PRISM_ARTIFACTS` at a scratch
  directory, or use `core.artifacts.use_store`, to keep a check away from the real `Artifacts/`.
- Two core-level environment settings stay environment settings, deliberately, and are named in the
  tool's `--help` epilog: `PRISM_VRAM_CEILING_GIB` is read live on each batch plan;
  `PRISM_MEM_LOG_EVERY` is read ONCE, when `core.SBI.pipeline` is imported, so setting it inside a
  running GUI changes nothing.

## Tests

`docs/guide/testing.md` is the reference: the gates, their times of record, the slow set's tests,
the markers, every session guard and per-test check in `tests/conftest.py`, and the card gates. The
rules:

- The fast gate is ONE `pytest -m "not slow"` process (its target of 16 minutes has not been met
  since stage 2; the times of record are in `docs/STATE.md`'s gate table). The slow set is
  `pytest -m slow` (three tests); the full run is `pytest`, never timed in one process: budget about
  an hour and a half. Count with `pytest --collect-only -q`.
- Markers: `slow`; `gpu` (skipped when CUDA is absent); `display` (skipped offscreen). The
  display-marked tests run on the real screen with `QT_QPA_PLATFORM=windows pytest -m display`
  (the root conftest only DEFAULTS the variable); they create hidden native windows, nothing shows;
  four of them make PRISM's real Windows requests for a run (keeping the computer awake, the reason
  against shutting down, the taskbar's flash, the answer to Windows' end of session) on those hidden
  windows, and release each at once.
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
  cache passes `checkpoint_every` explicitly. Two of the other guards bite from outside a test: the
  real settings folder must hold at teardown exactly what it held, so a PRISM opened while a suite
  runs fails the run with its `PRISM.lock`; and `<repo>/sbi-logs` must not exist, checked at setup
  too, so a leftover tree makes every pytest run error before the first test.
- A green suite does not certify the GPU path. The gpu-marked tests do run on the card inside every
  fast gate when CUDA is present (`tests/test_gpu_paths.py`'s CUDA inference run, six in
  `tests/test_user_sbi.py` and one in `tests/test_diagnostics.py`), but they are not a substitute: no
  checkpoint resume, no real bounds or cell files. After touching code that moves tensors, run the
  card smoke gate — five command lines, then run 5's resume drill (`tests/_fit_drill.py`, which ends
  a training like a power cut right after a chosen pass is saved), from the repo root — and check
  `$LASTEXITCODE` after each one; after a change under `core/diagnostics` that moves tensors, also
  run the diagnostic card. The block (the command lines, the drill and every pass criterion) is in
  `docs/guide/testing.md`, "The card smoke gate", and `docs/STATE.md`'s gate table carries it word
  for word: change both together. The last result, with its stage timings, is in that table.
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
  `core.runs.public_entry` copies it on entry. A caller that wants what a stage wrote reads the
  artifact.
- Every pre-spend refusal is a `core.refusals.Refusal` with a field key; the front ends name
  the control or flag from their own tables (`core/gui/fields.py`, `core/tool/fields.py`). In
  the converted modules no message names a box, tab, flag or button. Stage messages are
  `logging` records at info/warning/error; never print or log between steps 1 and 3 of a
  checkpoint save.
- Nothing shippable cites the working record: no file under `core/` or `tests/`, nor
  `conftest.py`, `README.md`, `requirements.txt`, `pytest.ini`, `run.bat`, `run.sh` or
  `docs/guide/`, may cite the handoff, `docs/STATE.md`, this file, the specs or the plans, or their
  labels (piece numbers, "§N", decision, ruling and review ids, "Task N", trap ids, walkthrough row
  ids). A provenance tag is deleted; a reason given by reference is written out in words.
  `tests/test_source_hygiene.py` enforces it; a legitimate new token shaped like a label (a paper's
  section, a new feature id) gets an allowlist entry there in the commit that introduces it. The
  working record may cite code and each other; no test or code reads it.
- The inference tabs remember SELECTIONS — the pickers, the mode, and the boxes that describe the
  recording (the three observation lengths, the physical drive amplitude `Drive F₀ (N)`, the
  recording paths) — plus the training budget. Every science knob opens at `config.py` on every
  launch (the chi probe count, slots and lock-in ceiling, the sweep, network, rotation and
  calibration knobs, the HPD level and the direction count), the Config tab's non-dimensional χ
  drive amplitude and band are read-only displays of it, and a consent is never persisted — it is
  answered by the session that runs. A stale key an older build left in `PRISM.ini` is ignored, not
  restored. The plan picker is the one picker that is not remembered: it opens on "No plan" at every
  launch, and only the name of the plan last used with each model, and when, is kept, for the
  picker's hint.
- `Resources/` holds the hand-edited inputs (`Bounds/`, `Cells/`, `Units/`, `Models/`, `Plans/`). The
  bounds file declares WHICH parameters are inferred and in what order, and therefore the
  observation mode; a cell's model comes from its parent folder; a run plan is
  `Plans/<model>/<name>.txt`, read by `core/plans.py`, and like the bounds and cell files it is tracked
  in git and committed before a long run, so that the run's records are not stamped as made from a
  changed tree (no plan is shipped; the first is saved from the window). Everything generated lives
  under `Artifacts/<kind>/<name>__<id>/` with a `manifest.json` and a `log.txt` of that run's records
  — every `core` record and Python warning since the outermost public entry began, stamped
  `HH:MM:SS info/warning/error` (`core/runs.py`) — for every kind but the simulation cache, which has
  no writer. The store's code is `core/artifacts/`; the `Artifacts/` tree is gitignored. The kinds
  are priors, simulations (the training cache, keyed by identity digest), posteriors, observations,
  calibrations, inferences, diagnostics and fits (a training kept partway; removed when its posterior
  is written); an `fdt/` folder an older PRISM left is no kind, and the store never lists or touches
  it. Every stage writes its artifact at completion and returns a `Loaded*` wrapper,
  except the training cache, which commits batch by batch and has no writer at all. Save is a rename;
  loading refuses a mismatch in what it compares (`docs/guide/architecture.md`'s "The artifact
  store" lists the checks and the keys no loader compares yet — the open list's load-check audit),
  and `Accept(truncated, other_observation)` are the only escape hatches (each use is recorded
  downstream). No code outside `core/config.py` builds a literal `Resources/` or `Artifacts/` path
  (a test pins it).
- A run plan fills the forms of steps 1 to 3 only when a press asks it to: the picker's "Fill the forms
  from this plan", Home's "Continue with these" when a plan was in use last time, or "Continue from batch
  N", which fills the settings the run was started with and leaves the name and the note blank; a plan is
  never applied at launch, the picker opens on "No plan", and only the name of the plan last used with
  each model, and when, is remembered, for the picker's hint.
- The narrowing-round safety rules (where each is enforced and the test that pins it) and the traps
  that still bite are in `docs/guide/rules-and-traps.md`; the reasoning behind the science settings
  is in `docs/guide/science.md`. The handoff they were written from is archived (see `docs/STATE.md`).
- Long PRISM runs — a training, a narrowing round, a full-size calibration: hours or days on the
  card — are started from the window, by the owner (the owner, 2026-10-05, who stopped a
  console-run retrain about 63 hours into its fit rather than finish it there). This is not about
  pytest runs or the card gates, which stay command lines. Before any long run, assume the window
  and confirm it with the owner; give a run's steps as on-screen controls (tab, field, button);
  use the command line only for what the window cannot do, and say so when that is the case.
  `docs/guide/retrain.md` is still a command-line runbook until it is rewritten as the window
  route.
  - A long run has the card to itself: while one is live, no pytest (the fast gate runs its
    gpu-marked tests on the card), no card gate and no second window (opening the window touches
    the card).
  - A full-size generation on this card needs the memory limit set (with none the 1 October run died
    at batch 237; at 6 GiB 9,500 batches went through): it is set once in Settings, "This computer",
    is remembered for this computer, and is shown in the Training step's "Before you start";
    `PRISM_VRAM_CEILING_GIB`, set in the environment the window starts in, takes precedence over it
    when it holds a number. A run started from the command line reads the variable only.
  - While a long run is live, edit no file under `core/` (whether an import late in a run can read
    an edited file is not established), and note `git rev-parse HEAD` when it starts: a posterior
    stamps the checkout's git revision and dirty flag when it is written at the end, so a commit of
    any file during the run, documents included, moves that stamp.
  - A fit is saved after every pass when its rows are in a kept simulation set, which the window and
    `train` keep by default (a commit every 50 batches): any stop loses at most the pass in progress
    (about an hour at full size), and the stopped training is kept as a record of its own (kind
    `fit`, under `Artifacts/fits/`). A fit trained with checkpointing off is not saved and cannot be
    resumed.
  - A kept training is resumed by hand, from the command line with `train --resume-fit <id>`, given
    the same `--bounds`, `--prior`, `--chi` and `--device` it was trained with (the window offers
    Resume on the run's page of a training that stopped or failed in its fit, until PRISM is closed,
    and after a reopening on the Training step's list of kept trainings and on Home; a kept narrowing
    round after a reopening is still resumed from the command line, until the round's page is
    rebuilt), only while its settings, its network's shape, the code that builds or trains the
    network, the libraries and its simulation set are what it was trained with: an edit inside that
    code (`fit_state.FIT_CODE`), even a comment, makes every training kept before it unresumable, so
    finish a kept training you want before such an edit. A resume reads the simulation set back
    first (about four minutes at full size).
- Git: work directly on the local `main` branch — no feature branches, no worktrees (decided
  2026-09-11). Claude makes the local commits, one per logical step, never amended, ending with the
  Co-Authored-By line the harness provides. Commit messages are SHORT: a one-line subject, a brief
  body only when the subject cannot carry the why. The user pushes and handles every other remote
  operation. Undo an edit with `git stash`, never `git checkout -- <file>`, and read `git status`
  before touching a file you did not create: `checkout --` once discarded about seventy lines of
  uncommitted test work.
- Archiving a file means MOVING it on disk into the gitignored `archive/` and then `git rm
  --cached` — never `git mv`, which would re-create a tracked-but-ignored file. A script that was
  folded into a subcommand is `git rm`'d instead: git history is its archive.

## Where things are

- `docs/STATE.md` — the moving state. Read first, update last.
- `docs/superpowers/specs/` — approved designs; `plans/` — implementation plans; `notes/` — findings a
  later chat starts from (among them the code audit's list); `prompts/` — the hand-off prompts for the
  chats that follow.
- `.superpowers/sdd/<date>-<slug>/` — a piece's gitignored execution workspace: the ledger
  (`progress.md`), the task briefs and reports, the review reports and the per-task gate logs; the
  window redesign's is `2026-10-06-window-redesign`, with one folder per stage (`stage-N/`).
  Untracked, so no git command finds them.
- `docs/checklists/display-walkthrough.md` — GUI features never exercised on a real screen.
- `docs/guide/` — the reader pages, starting at its `README.md`: getting started, the window, the
  command line, bringing recordings, the science, the architecture, the rules and traps, testing,
  and the retrain runbook. They cite no working document (the source scan walks them), and each
  opens with the commit it was checked against.
- `tests/` — the suites, one file per area, each named for what it tests (`docs/guide/testing.md`
  describes the window's). `_fixtures.py` holds the
  shared stand-ins, the builders every window test goes through (`build_window`, `build_part`,
  `build_inference`, `build_run_host`), the tiny real prior+posterior, and `CODE_ROOTS` plus
  `CODE_FILES`, the directories and top-level files the source scans walk.
- `core/tool/` — the command-line tool: `python -m core --help` lists every subcommand (the stages
  `prior train validate infer tsnpe`, the diagnostics `sbc identifiability ablation probes`
  (band/mask/drive), plus `smoke` and `artifacts` (list/show/note/rm/sweep/summary)). `scripts/` is
  gone: git history and the gitignored `archive/scripts/` keep its scripts. The Records screen is the
  store's front end in the window, over the same `core/artifacts/` engine the tool's `artifacts`
  family reads.
