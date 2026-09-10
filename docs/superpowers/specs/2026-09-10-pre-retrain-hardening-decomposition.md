# PRISM pre-retrain hardening — decomposition

**Date:** 2026-09-10 · **Status:** approved (brainstorming session, plan approved) · **Owner:** J

## 1. Goal

Every trained prior and posterior will be deleted and the model retrained from scratch. Before the
first prior is built, the application must be rigorous, complete, easy to use, and proof against
misuse. A whole-application audit found the gaps fall into several independent projects, so the
work is decomposed into seven pieces, each with its own spec → plan → implementation cycle. The
retrain starts only after all seven have landed.

## 2. Decisions

| decision | choice | consequence |
|---|---|---|
| Existing artifacts | **Clean break.** Priors, posteriors, checkpoints, observations and the quarantined directory are deleted. | Artifact formats, naming, the sidecar and the training identity are free to change. Run A's cached simulation and the forensics reproduction are given up. |
| Audience | **All four:** the user doing the science; lab members bringing recordings; a successor maintaining the code; reviewers reproducing results. | Rigor, a self-explanatory inference path, maintainability and provenance all count. |
| Product surfaces | **GUI + scripted reproduction.** The GUI is the product. The fifteen scripts become one tested command-line tool over the same stage functions. The prompt-driven CLI (`python -m core`) is retired. | The three copies of the inference flow (GUI runners, `orchestrator.run`, `scripts/_common.py`) collapse into one. |
| Scope | **In:** parameter inference on simulated cells and on experimental recordings (passive, driven, chi); FDT analysis and cross-validation; Simulate and the model builder. **Out:** the reduction map (ongoing theoretical work; will change later). | |
| Sequencing | **Everything lands before the first retrain.** | Order is by dependency, not by the retrain clock. |
| Test runner | **pytest** (piece 0). | `tmp_path` sandboxing, markers, one command, exact counts. |
| Artifact store shape | **One directory per artifact with a `manifest.json`** (piece 1). | See `2026-09-10-artifact-store-design.md`. |
| Ids | **UTC timestamps** `YYYYMMDDTHHMMSS`, unique within a kind, `-2`/`-3` on same-second collision. | Directory names sort by creation. |
| Persistence | **Auto-persist.** Every completed stage writes its artifact at once; Save becomes "name and annotate". | Nothing is lost by a forgotten click; every object has an id for lineage. |

## 3. What the audit found

- **Silent-wrong paths.** A blank observation length reads as 0 and nothing checks it. Stale file
  paths restored from settings fail only inside the worker. The driven experimental branch checks
  no paths. Runners mutate the session's `SimConfig` in place, so a failed inference leaves the
  previous cell's ground truth behind.
- **Provenance.** Calibration writes nothing to disk. No run manifest exists. Truncated and
  amortized posteriors share a picker. Nothing checks conditioning width on load. GUI-saved
  sidecars carry no Fisher eigenvalues. Inputs and outputs interleave in `Resources/`.
- **Never seen on a display.** The handoff keeps a standing list of GUI features no human has
  exercised: the Save buttons, Validate, Infer, cancel during training, the model-builder round
  trip, the navigation click-through.
- **Untested surfaces.** Scripts have zero tests and no flags. The prompt CLI has no end-to-end
  test. `core/FDT` has five helper tests. `core/Reduction/tests` import pytest and cannot run.
  Nothing certifies the GPU path.
- **Documentation.** `PRISM_HANDOFF.md` (530 KB) is spec, changelog and trap list at once, and
  says of itself that it is being deleted.

### Verified defects

| # | where | defect | fixed in |
|---|---|---|---|
| 1 | `core/SBI/run_guards.py:159` | `warnings.warn` with no `import warnings` → `NameError` on a legacy prior lacking `model`/`param_keys` | piece 1 (the branch is rewritten into the store's `load_prior`) |
| 2 | `core/gui/panels/inference/tsnpe_tab.py` | defines `restore_settings` but `__init__` never calls it; every other inference tab does | piece 3 |
| 3 | `core/SBI/observations.py:68` | bare `N_ND_MAX`; the module imports only `config` and `SimConfig` | piece 1 (blocks the stage-contract test) |
| 4 | `core/SBI/observations.py:82` | bare `FORCING_SI_UNITS`; with 3, `build_experiment_obs` (forced recording) raises `NameError` today | piece 1 |

## 4. The seven pieces

| # | piece | scope | why this position |
|---|---|---|---|
| 0 | **Verification baseline** | pytest adopted; GPU `smoke_train` run as a recorded baseline; `KMP_DUPLICATE_LIB_OK` in the launchers; Reduction tests run under pytest and recorded; display-walkthrough checklist; `CLAUDE.md` and `docs/STATE.md` | Small; every later piece adds tests against it |
| 1 | **Run contract, artifact store, provenance** | stage inputs/outputs as a contract; `Resources/` (inputs) vs `Artifacts/` (generated); a manifest per artifact (git rev, versions, device, inputs by hash, every knob, parent ids); the sidecar folded into the manifest; calibration and inference results persisted; refuse-by-default loading with named accept flags; the clean-break runbook | The clean break makes this free now and expensive after the retrain |
| 2 | **One flow underneath** | unify GUI `runners.py`, `orchestrator.run` and `scripts/_common.py` into one stage API; retire the prompt CLI; one command-line tool (`prior`, `train`, `tsnpe`, `validate`, `infer`, `sbc`, `identifiability`, `ablation`, `smoke`, …) over the same stages; scripts folded in with tests | Hardening three copies is wasted work; the tool is the reviewer's reproduction path |
| 3 | **Validation and misuse-proofing** | boundary validation for every field (`T_obs > 0`, paths exist, budgets, device); science constants never restored from QSettings; copy-on-run and reset-after-success for session state; `logging` with severity replacing `print`; error dialogs that name the fix; defect 2 | Depends on the stage API from 2 |
| 4 | **GUI usability and artifact browser** | list/inspect/delete artifacts with their manifests; picker shows mode, width, amortized/truncated; the two-entry-point trap (M1b) resolved; one-task-at-a-time made visible; the never-seen list walked on a display with the checklist; a saveable run summary | Depends on the store from 1 |
| 5 | **Secondary panels** | FDT/CrossVal tests and validation (the all-rows-failed completion, the atomicity decision, the prompt booleans); Simulate and the model builder round trip verified | Independent of 4; after 3 |
| 6 | **Documentation and the retrain runbook** | user guide per audience; per-subsystem design docs; the runbook as a document and a scripted pipeline; the handoff archived | Documents the finished app |

**Sequence:** 0 → 1 → 2 → 3 → (4 ∥ 5) → 6. Each piece returns to brainstorming for its own spec.

## 5. Claude Code context files

The handoff serves three readers at once. Claude Code gets its own two files:

- **`CLAUDE.md`** (repo root, loaded every session; created in piece 0; stable): interpreter path;
  required env vars; how to run the fast and the slow suites; "every knob is an argument, never a
  config write"; never edit a source file while a suite runs (`inspect.getsource` tests); git
  remote operations are user-owned; where `docs/STATE.md` and `docs/superpowers/` live; the
  Resources-vs-Artifacts split once piece 1 lands.
- **`docs/STATE.md`** (the moving state; updated at the end of every session; short enough to read
  in full): last updated · where things stand · owed, in order · artifacts on disk and which must
  be kept · last gate result · dated decisions log. It replaces the handoff's stacked PICK UP HERE
  banners and Appendix "Closing state" from now on. Piece 6 moves the handoff's remaining content
  into the user guide and the design docs and archives it.

## 6. Piece 0 — verification baseline (bounded)

- **pytest.** `pytest` added to `requirements.txt` and installed into `biophys-env`.
  `pytest.ini`: `testpaths = tests core/Reduction/tests`, `pythonpath = .`, `addopts = -ra`,
  markers `slow` (the chi full-pipeline test in `test_user_sbi.py`), `gpu` (needs CUDA; skipped
  when absent), `display` (needs a real screen; skipped offscreen). `tests/conftest.py`:
  `os.environ.setdefault` for `QT_QPA_PLATFORM=offscreen` and `KMP_DUPLICATE_LIB_OK=TRUE` before any
  PySide6/torch import, `matplotlib.use("Agg")`, a session-scoped `qapp` fixture. Gate commands:
  fast `pytest -m "not slow"`, full `pytest`. Acceptance: `pytest --collect-only -q` reports the 322
  existing tests (plus whatever `core/Reduction/tests` collects); then the twelve
  `if __name__ == "__main__"` runner blocks are removed in one mechanical commit.
- **Launchers.** `run.bat` / `run.sh` set `KMP_DUPLICATE_LIB_OK=TRUE`.
- **GPU smoke baseline.** `scripts/smoke_train.py` on the card once (CHI=1 and CHI=0,
  `BOUNDS=Resources/Bounds/nadrowski/master.txt`), result recorded in `docs/STATE.md`. Re-run after
  pieces 1, 2 and 3.
- **Reduction tests** run once under pytest and recorded (Reduction is otherwise out of scope).
- **Display walkthrough checklist** at `docs/checklists/display-walkthrough.md`, seeded from the
  handoff's "never run for real on a display" list; executed in piece 4.

**Verification:** all suites green under pytest; `run.bat` launches the GUI with the env set; a
fresh session sees `CLAUDE.md`; `docs/STATE.md` carries the smoke result.

## 7. Done means

The retrain can start: every piece's verification has passed, the display walkthrough is complete,
the GPU smoke train is green on the final code, and `docs/STATE.md` says so.
