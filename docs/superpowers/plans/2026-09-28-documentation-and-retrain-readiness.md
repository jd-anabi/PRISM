# PRISM piece 6 — the reference cleanup, retrain readiness, the probe checks and the documents: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Before the retrain, make the shippable code cite no AI working document, make the retrain's tier-1 path correct, tested and self-reporting, give labs a `probes` diagnostic to re-measure the chi settings, and replace `PRISM_HANDOFF.md` with reader documents under `docs/guide/`.

**Architecture:** Five parts in order (spec H10): (1) a comment/docstring sweep of `core/` and `tests/` behind a new source-scan suite with a shrinking not-yet-clean list; (2) a text-only tidy of the command-line help; (3) readiness fixes on the training/calibration/inference path; (4) a new three-mode diagnostic family `python -m core probes` built on the existing diagnostic framework; (5) the reader pages, the README summary, the tool reference and the retrain runbook, then the handoff is archived. Every behaviour change is test-first; every task has its own review and one-process fast gate.

**Tech Stack:** Python 3.12, torch 2.9.0+cu130, sbi 0.25.0 (installed; 0.26.1 pinned), PySide6 6.9.3, pytest; the conda env `biophys-env`.

**Spec:** `docs/superpowers/specs/2026-09-25-documentation-and-retrain-readiness-design.md` — read it with this plan; every task cites the spec sections it implements, and those sections are binding.

## Global Constraints

- Interpreter `C:\Users\J\anaconda3\envs\biophys-env\python.exe`; anything importing torch needs `KMP_DUPLICATE_LIB_OK=TRUE`; headless PySide6 needs `QT_QPA_PLATFORM=offscreen` (the root `conftest.py` defaults both for pytest).
- Git: work on the local `main` branch; one commit per logical step, never amended; SHORT subject lines; every commit ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`; `git add` every changed file explicitly (the Write tool does not stage). The owner pushes.
- Every knob is a keyword argument, never a config write (`core/orchestrator.py` binds config constants at import).
- Every public stage/composition/diagnostic is `@public_entry` (it gets a private config copy) and mutates nothing it is handed; the exact public-entry set is pinned by `tests/test_artifact_store.py:3410`.
- Every pre-spend refusal is a `core.refusals.Refusal` with a registered field key (FIELDS + `core/tool/fields.py` FLAG + `core/gui/fields.py` CONTROL); core messages name no box, tab, flag or button. A programming error is a plain exception (e.g. `RuntimeError`), never a Refusal.
- Messages are `logging` records (info / warning / error), never `print`, in `core/diagnostics` and every converted module.
- No code outside `core/config.py` builds a literal `Resources/` or `Artifacts/` path.
- **H4, the reference rule (spec §2):** nothing new or edited in `core/`, `tests/`, `conftest.py`, `README.md`, `requirements.txt`, `pytest.ini`, `run.bat`, `run.sh` or `docs/guide/` cites the handoff, STATE, CLAUDE.md, the specs, the plans or their labels (piece numbers, "§N", decision ids such as D7/V5/B12/E2/H4, rulings, review ids, "fix round", "Task N", trap ids, "guardrail N", "C-11", "Appendix A", walkthrough row ids). State the reason in words. From Task 1 on, `tests/test_source_hygiene.py` enforces it.
- No `.py` file under `.superpowers/` (it trips the code-directory guard); controller scripts live outside the repository.
- Never edit a source file while a pytest run is in progress; never run two pytest processes at once. Implementers run only their focused tests (in the background with a log when a run can exceed ten minutes, as Tasks 4, 5, 6 and 12 say); the controller runs each task's one-process fast gate, `pytest -m "not slow" -q`, in the background with output to a log file, starting it at the same moment as the task's review (the owner's rule). Target: 16 minutes; the recorded gate was 14 min 42 s at `7e51275`, so new fast tests must be cheap (stand-ins, not real simulations).
- The GPU smoke gate is run by the controller only (Task 26).
- The inference tabs remember selections and the budget only; every science knob opens at `config.py`; nothing new is remembered by this piece (the Validate tab gains no Seed box).
- Standing refusal D12 is unchanged. D11 is unchanged for everything that trains or infers; only the `probes` family may take its own frequency and drive grids (spec H8), which never reach a `SimConfig` field or `gen_training_data`'s `chi_f0`/`chi_freq_bounds` keywords.
- Archiving a file = move it on disk into the gitignored `archive/`, then `git rm --cached` — never `git mv`.
- The working record (CLAUDE.md, docs/STATE.md, docs/superpowers/, docs/checklists/, the gitignored ledger) may cite code and each other; tests and code never read them.

## Review Focus

1. **A probe the cell's own peak pushes past the sampling limit** (a fast cell, a high multiplier): `probes band` must report that point as masked, never crash or clamp — test in Task 22.
2. **A resume from a checkpoint written before this piece** (no eigenvalue key, no masked-count list in `state.pt`): the run must resume, record the eigenvalues as unknown and say the masked total covers only this process — tests in Tasks 16 and 18.
3. **The same `validate --seed` twice**: identical calibration sets and identical verdicts; and `validate` with no seed inside `smoke` leaves smoke's seeded stream untouched — tests in Task 15.
4. **A flag added later without a help-defaults entry**: the help-walking test must fail naming the subcommand and the flag, not pass silently — test in Task 11.
5. **A legitimate new token shaped like a label** (a future feature id, a paper section): the scan's failure message must name the file, line, family and the allowlist to extend — test in Task 1.

## File Map

New files:
- `tests/test_source_hygiene.py` — the reference scan (Task 1; extended to `docs/guide/` by Task 27).
- `tests/test_tier1.py` — the tier-1 path's tests (Task 12, extended by 13–19 where they touch tier 1).
- `tests/test_docs.py` — link checker and command-line reference test (Tasks 27, 29).
- `core/tool/help_defaults.py` — the help default-class table (Task 11).
- `core/diagnostics/probe_math.py` — pure helpers for the probes family (Task 20).
- `core/diagnostics/probes.py` — the three `probes` stages (Tasks 22–24).
- `core/tool/probes.py` — the `probes` family parser and handlers (Tasks 22–24).
- `docs/guide/README.md`, `getting-started.md`, `window.md`, `command-line.md`, `recordings.md`, `science.md`, `architecture.md`, `rules-and-traps.md`, `testing.md`, `retrain.md` (Tasks 27–36).
- Controller tool (outside the repo): `C:\Users\J\AppData\Local\Temp\prism-piece6\comment_only_check.py` (Task 1).

Modified (main ones; each task lists its own): about 95 files under `core/` and 20 under `tests/` in the cleanup (Tasks 2–9); `core/tool/{stages,diagnostics,smoke,fdt,browse,config_args,__init__,fields}.py`, `core/refusals.py` (Tasks 10–11); `core/forcing.py`, `core/SBI/{chi_probes,derived,pipeline,truncate,analysis,training_checkpoint}.py`, `core/orchestrator.py`, `core/sim_config.py`, `core/artifacts/{manifest,identity,store,report}.py`, `core/rng.py`, `core/diagnostics/{identifiability,sbc,__init__}.py`, `core/gui/panels/simulate_runner.py`, `core/gui/panels/inference/{validate_tab,help_text,base}.py`, `core/gui/fields.py` (Tasks 12–19); `README.md`, `CLAUDE.md`, `docs/STATE.md`, `docs/checklists/display-walkthrough.md`, `pytest.ini`, `requirements.txt`; `PRISM_HANDOFF.md` archived (Task 38).

## Cross-task interfaces (the registry)

- **T1** `tests/test_source_hygiene.py`: `Hit = namedtuple("Hit", "path line family match")`; `PROSE_PATTERNS: dict[str, re.Pattern]` (families of spec §2.2, each pattern assembled from pieces so the module does not match itself); `TEST_NAME_PATTERN`; `ALLOW_TOKENS`, `ALLOW_PER_FILE: dict[str, frozenset[str]]`, `ALLOW_SHAPES: tuple[re.Pattern, ...]`, `ALLOW_TEST_NAME_TOKENS = frozenset({"f0", "d0"})`, `ALLOW_TEST_NAMES: frozenset[str]`; `PROSE_FILES = ("README.md", "requirements.txt", "pytest.ini", "run.bat", "run.sh")`; `READER_DOCS_DIR = "docs/guide"` (scanned when present: `*.md`, every line); `NOT_YET_CLEAN: frozenset[str]` (repo-relative posix paths); `scan_python(path: Path, *, used: set | None = None) -> list[Hit]`; `scan_prose(path: Path, *, used: set | None = None) -> list[Hit]`; `scanned_files() -> list[Path]`. Tests: `test_no_file_outside_the_pending_list_cites_a_working_document`, `test_every_pending_file_still_has_a_hit`, `test_the_matchers_catch_each_family_and_spare_the_allowed_tokens`, `test_every_allowlist_entry_still_matches_something`, `test_no_test_name_carries_a_process_label`, `test_a_failure_names_the_file_line_family_and_the_allowlist`.
- **T1** controller tool `comment_only_check.py`: `python comment_only_check.py --base <git-ref> --allow <allow-file> <path> [<path> ...]`; the allow file has one permitted non-docstring difference per line, `<repo-relative-path>|<kind>|<quoted text fragment>` with kind ∈ {`message`, `string`, `rename`, `assert`}; exit 0 when every difference is allowed, 1 otherwise, printing `path:line: <old> -> <new>` per difference.
- **T11** `core/refusals.py`: `default_clause(key: str) -> str` (the renamed `_default_clause`, behaviour unchanged; every caller updated) and `default_text(value: str) -> str` returning `" (default {value})"`; `FIELDS["max_num_epochs"].default == "no ceiling"`; `FIELDS["min_valid"].default == "0.5"`. `core/tool/help_defaults.py`: `DEFAULT_CLASS: dict[tuple[str, str], tuple[str, str | None]]` keyed by `(leaf_path, option_string)` — leaf path as typed after `python -m core`, e.g. `"sbc"`, `"identifiability laplace"`, `"probes band"` — to `(cls, text)` with `cls ∈ {"value", "behaviour", "none"}` and `text` the pinned literal (value), the phrase (behaviour) or None; `default_for(leaf_path: str, action) -> str | None` returns the full `(default X)` clause the help must end with, or None, by the precedence of spec §3.5; `LEAVES` — every leaf path.
- **T12** `tests/test_tier1.py` helpers (module-level): `tier1_cfg(*, chi: bool = True, hw=None) -> SimConfig` (built from `Resources/Bounds/nadrowski/master_tier1.txt` via the tool's `config_args.build_cfg` path or `cli.make_sim_config`, CPU), `force_scale_spy(monkeypatch) -> list` (records the force scale every simulation receives).
- **T13** `core/SBI/derived.py`: `for_simulation(cfg, params_nd: Tensor, rescale: Tensor) -> tuple[Tensor, dict[str, int]]` — returns `(to_sim_rescale(params_nd, rescale, cfg.rescale_idx, *cfg.tier1_args), cfg.sim_rescale_idx)`; the one call every simulating diagnostic and the live runner use. The two force builders (`core/forcing.py` sinusoidal builder, `core/SBI/chi_probes.gen_chi_raw`) raise `RuntimeError` when the index has `"T"` and no `"f_scale"`.
- **T14** `SimConfig.assumed_params -> tuple[str, ...]` (`("T",)` when tier 1 is on, else `()`); `SimConfig.report_labels -> list[str]` (the inferred labels with `" (assumed input, K)"` appended to an assumed parameter's label); `posterior_summary` entries gain `"assumed": bool`; `analysis.describe_informativeness(info, *, assumed: Sequence[str] = ())`.
- **T15** `core/rng.py`: `CALIBRATION_TAG: int`, `calibration_seed(seed: int) -> int` (from `numpy.random.SeedSequence([seed, CALIBRATION_TAG])`). `core/orchestrator.py`: `calibration_verdict(ks_pvals: Sequence[float], names: Sequence[str], assumed: Sequence[str], tarp_ks_p: float, *, alpha: float = 0.05) -> dict` → `{"passed": bool, "alpha": 0.05, "threshold": alpha / len(names), "parameters": [{"name", "ks_p", "passed", "assumed"}], "tarp": {"ks_p", "passed"}, "caveat": str}`; `validate_calibration(..., seed: int | None = None)` writes `body.results["verdict"]` and `body.results["seed"]`; the verdict record is one info line starting `"[verdict] "`. `sbc` writes `results["rank_verdict"] = {"per_repeat": [bool], "fraction_passed": float, "scope": "rank-uniformity half only"}`. Tool: `validate --seed N` (key `seed`).
- **T16** checkpoint header key `"fisher_eigenvalues"` (list[float] or None; readers use `.get`); `TruncationRegion.containment(theta_latent: Tensor) -> list[dict]` → `[{"direction": int, "lo": float, "hi": float, "value": float | None, "inside": bool}]` (`value` None when not finite); the child posterior's `body.training["truth_containment"]` holds that list; info lines start `"[tsnpe] truth "`.
- **T17** `manifest.config_from_cfg(cfg)` adds `"tier1": {"relation": "f_scale = N * beta * k_B * T / x_scale", "k_b_cell": float, "T_range": [lo, hi]}` and `"assumed_params": list[str]` when tier 1 is on (absent otherwise); a simulated observation's config adds `"simulated_f_scale": float`; `core/artifacts/identity.py` `FORMAT = "training-rows/3"` and the identity values gain `"units_sha256"` (None when the config has no units source).
- **T18** `core/SBI/pipeline.py`: class `_BatchProbeLedger(committed=(), first_batch=0)` (Task 21 adds `observer=None`) with `add(lo: int, hi: int, masked: int, total: int, record=None) -> None`, `discard() -> None`, `commit(batch_k: int) -> None`, `committed: list[tuple[int, int]]` (masked, total per committed batch, in batch order), `summary_line(scope: str) -> str`; module global `_ACTIVE_LEDGER` set for the life of one `gen_training_data` call (like `_BATCH_TAG`); `chi_probes.gen_chi_block` calls `pipeline._ACTIVE_LEDGER.add(...)` when it is set. Checkpoint `state.pt` gains `"chi_masked": list[list[int]]` (readers `.get("chi_masked", [])`). The run-total record is one info line starting `"[chi] masked probes"`. `smoke` gains `--hidden-features N`, `--num-transforms N`.
- **T19** `body.results["accepted"]` on a calibration; `body.training["accepted"]` on a narrowing child; an inference's `accepted` is the union of its posterior's load acceptances and its own `other_observation` use; `core/artifacts/report.py::_accepted` reads `body.training.accepted` for a posterior.
- **T20** `core/diagnostics/probe_math.py`: `Geometry = NamedTuple("Geometry", n_obs=int, n_fine=int, subsample=int)`; `recording_geometry(cfg, t_obs_s: float, t_scale: float) -> Geometry`; `training_ceiling_s(cfg, t_scale: float) -> float` (the ceiling training enforces: `n_fine <= min(N_ND_MAX, len(cfg.t))`); `default_lengths(cfg, t_scale: float, n: int = 5) -> list[float]`; `default_multipliers(n: int = 4) -> list[float]` (log-spaced across `config.CHI_FREQ_BOUNDS`, rounded to 4 decimals, plus one control at twice the top edge); `own_peak_ratio(forced: Tensor, reference: Tensor, omega0: float, window_frac: float, dt: float) -> float`; `circular_spread(z: Tensor) -> float`; `harmonic_flags(multipliers: Sequence[float], window_frac: float, max_order: int = 5) -> list[bool]`.
- **T21** `core/SBI/chi_probes.py`: `@dataclass(frozen=True) class ProbeRecord: batch_tag: str; lo: int; hi: int; f_peak: Tensor; duration_frac: Tensor | None; k: int; u: Tensor; logcyc: Tensor; valid: Tensor; packed_mask: Tensor; dt_exp: float; n_points: int`; `pipeline.gen_training_data(..., probe_observer: Callable[[ProbeRecord], None] | None = None)` (keyword-only; default None changes nothing); the ledger buffers records per row range and hands them to the observer only in `commit`.
- **T22** `core.diagnostics.probe_band(cfg, *, lengths=None, multipliers=None, drives=None, repeats=24, cycle_caps=None, cv_max=0.20, phase_max=None, snr_min=3.0, sup_min=0.50, peak_window=0.10, seed=0, name="", note="", fig_sink=None, store=None) -> LoadedDiagnostic`; body `{"diagnostic": "probes", "variant": "band", "settings", "results"}`; `core/tool/probes.py`: `register(sub) -> dict[str, argparse.ArgumentParser]` (the family parent `probes` with `dest="variant"`; the tool loop in `core/tool/__init__.py` gains it); field keys `probe_seed`, `probe_lengths`, `probe_multipliers`, `probe_drives`, `band_repeats`, `probe_cycle_caps`, `cv_max`, `phase_max`, `snr_min`, `sup_min`, `band_peak_window`.
- **T23** `core.diagnostics.probe_mask(cfg, prior: LoadedPrior, *, num_runs=12, run_size=32, chi_k_fixed=None, seed=0, name="", note="", fig_sink=None, store=None) -> LoadedDiagnostic`; field keys `mask_num_runs`, `mask_run_size` (reuses `prior`, `chi_k_fixed`, `probe_seed`).
- **T24** `core.diagnostics.probe_drive(cfg, *, t_obs_s=5.0, repeats=16, detune=1.4, strengths=None, free_min=0.70, captured_max=0.10, peak_window=0.02, clarity_min=3.0, seed=0, name="", note="", fig_sink=None, store=None) -> LoadedDiagnostic`; field keys `drive_t_obs`, `drive_repeats`, `drive_detune`, `drive_strengths`, `free_min`, `captured_max`, `drive_peak_window`, `clarity_min`.
- **Docs** (T27–T36): `docs/guide/` page names as in the file map; every page's first line after its title is `Checked against commit <hash>.`; headings in `command-line.md` are `## <subcommand>` and `### <subcommand> <mode>` exactly as typed (Task 29's test reads them).

## How to run this plan

- **The plan is committed at planning time.** `<P>`, which Tasks 25 and 39 read, is the commit that added this file: `git log --diff-filter=A --format=%h -- docs/superpowers/plans/2026-09-28-documentation-and-retrain-readiness.md`. The empty `git status --porcelain` checks of Tasks 25, 26, 38 and 39 assume the tree is clean before Task 1; any later plan correction by the controller is its own commit.
- **Order is execution order**, and it follows the spec's parts: Tasks 1–9 (part 1, the reference cleanup), 10–11
  (part 2, the help tidy), 12–19 (part 3, readiness), 20–24 (part 4, the probe checks), 25–26 (the whole-code
  review and the card run, controller), 27–36 (part 5, the documents), 37–39 (the documents review, retiring the
  handoff, the final gates, controller).
- **Each task's text is self-contained**: its Binding block restates every rule it must keep, because a task's
  implementer sees only that task. The spec sections a task cites are binding; read them.
- **Controller tasks** (25, 26, 37, 38, 39 and the last step of 24) are run by the controller, not an implementer.
- **The per-task loop (the owner's rule).** On an implementer's DONE: check that no python process is alive, start the task's one-process fast gate in the background, and dispatch the task's reviewer at the same moment. Every reviewer and re-reviewer (per task, Task 25 Step 7's, Task 37's) is read-only and runs no pytest of any kind: torch-free scripts, `--help` and the comment-only checker only. A task is complete when its review is clean AND its gate is green. If the review sends fixes, wait for the gate to exit before any source edit, then resume the implementer, re-review the fix commits and start a fresh gate on the fixed head. Dispatch the next task only after the current gate has exited: an implementer's edits and focused runs never overlap a gate.
- **Owner questions the plan leaves open, each with a default** (settle them with the owner when the task is
  reached; a changed answer is recorded in the spec's §11): the runbook's epoch-timing stop point (default: stop when
  the projected fit exceeds twice the measured reference, Task 36); the retrain's prior (default: a fresh prior built on
  `master_tier1.txt`, Task 36); stratified SBC in the runbook as a characterisation, not a gate (Task 36); whether
  `store.load_posterior` should record only the acceptances a load actually used (today it records every flag passed;
  left as is, Task 19); where the git stash-versus-checkout process note lives (Task 37's checklist).
- **Departures from the spec already ruled while planning.** They are rows 1–14 of the spec's §11 (recorded at planning time); Task 25's spec lens reads them as recorded, and execution deviations are added from row 15. (1) Task 1 widens §2.2's `section` and `review` patterns to catch `SECTION n.n` and sentence-initial `Fix round`, and cleans `core/SBI/derived.py`'s one upper-case hit, which the recon missed. (2) Task 1 exempts `ALLOW_TEST_NAME_TOKENS` (`f0`, `d0`) from the matches-nothing rule, because no test name uses them yet. (3) Task 9 renames the scan test once the pending list is gone (§2.7 (c) names only Task 1's two renames). (4) Task 11's artifacts epilog states no note length at all (§3.6 said it reads `NOTE_MAX_CHARS`), and fdt's `--n-freqs`/`--ensemble-m` get pinned literals. (5) Task 14 leaves `core/Helpers/labels.py` alone and puts the unit and the mark in `SimConfig.report_labels` (§5.3 named labels.py), and it logs the posterior summary so row F2 can be checked on the Infer tab. (6) Task 15 seeds the whole calibration battery, not only the set's draw, and the Validate tab's closing line names the verdict. (7) Task 16's containment `value` is None when not finite (a manifest refuses non-finite numbers), and its resume record uses the `[checkpoint] ` prefix. (8) Task 17 records `simulated_f_scale` on any box whose simulator index has `f_scale`, and hashes the units file with CRLF normalised. (9) Task 21's `ProbeRecord` gains `n_points` and a tensor `duration_frac`. (10) Task 22 judges `drive_holds` on the configured drive only, and always adds the configured ceiling to `--cycle-caps`. (11) Task 24 has its own `--strengths` default phrase, a single-bin clarity, a two-bin floor on the own-peak window, and the 0.02 default committed before the parity run that accepts it (§4.5 puts the parity first).
- **Scratch conventions**: the controller tool and every allow file, hit list and throwaway script live in
  `C:\Users\J\AppData\Local\Temp\prism-piece6\` (allow files `allow-task-NN.txt`), never under `.superpowers/`.

---

### Task 1: The scan suite, the pending list, and the comment-only checker

**Files:**
- Create: `tests/test_source_hygiene.py`
- Create (outside the repository, never under `.superpowers/`): `C:\Users\J\AppData\Local\Temp\prism-piece6\comment_only_check.py`, `C:\Users\J\AppData\Local\Temp\prism-piece6\list_pending.py`, `C:\Users\J\AppData\Local\Temp\prism-piece6\allow-task-01.txt`
- Modify: `tests/test_nav_and_gating.py`, the definition `def test_the_d7_and_d8_dialogs_default_to_cancel(monkeypatch):` (~:1280)
- Modify: `tests/test_user_sbi.py`, the definition `def test_the_cuda_graph_preserves_the_rng_contract_c11_depends_on():` (~:3184)
- Modify: `tests/conftest.py`, the `_no_modal_dialogs` docstring, "as test_the_d7_and_d8_dialogs_default_to_cancel does" (~:176)
- Modify: `tests/test_artifact_browser.py`, the `_answer` docstring, "tests/test_nav_and_gating.py::test_the_d7_and_d8_dialogs_default_to_cancel uses" (~:62)
- Modify: `core/SBI/chi.py`, docstring "the network also sees A3 (log f_peak)" (~:335)
- Modify: `core/SBI/embedded_network.py`, docstring "-- ~1e29-magnitude traces for A1, exactly-constant ones for D3" (~:65)
- Modify: `tests/test_conditioning_repair.py`, comment "# exactly constant -> D3 = 1/_EPS" (~:241)
- Modify: `core/SBI/derived.py`, the `describe_derived_f_scale` docstring, "HAZARD 1 OF SECTION 11.5, answered" (~:127)
- Modify: `tests/test_worker_dispatch.py`, the docstring "the way test_the_d7_and_d8_dialogs_default_to_cancel does" (~:344)

**Interfaces:**
- Consumes: `tests._fixtures.CODE_ROOTS`, `tests._fixtures.CODE_FILES`, `core.SBI.statistics.FEATURE_LABELS` (all existing).
- Produces (the T1 registry, exactly these names): in `tests/test_source_hygiene.py`: `Hit = namedtuple("Hit", "path line family match")`; `PROSE_PATTERNS: dict[str, re.Pattern]`; `TEST_NAME_PATTERN`; `ALLOW_TOKENS`; `ALLOW_PER_FILE: dict[str, frozenset[str]]`; `ALLOW_SHAPES: tuple[re.Pattern, ...]`; `ALLOW_TEST_NAME_TOKENS = frozenset({"f0", "d0"})`; `ALLOW_TEST_NAMES: frozenset[str]`; `PROSE_FILES = ("README.md", "requirements.txt", "pytest.ini", "run.bat", "run.sh")`; `READER_DOCS_DIR = "docs/guide"`; `NOT_YET_CLEAN: frozenset[str]`; `scan_python(path: Path, *, used: set | None = None) -> list[Hit]`; `scan_prose(path: Path, *, used: set | None = None) -> list[Hit]`; `scanned_files() -> list[Path]`; tests `test_no_file_outside_the_pending_list_cites_a_working_document`, `test_every_pending_file_still_has_a_hit`, `test_the_matchers_catch_each_family_and_spare_the_allowed_tokens`, `test_every_allowlist_entry_still_matches_something`, `test_no_test_name_carries_a_process_label`, `test_a_failure_names_the_file_line_family_and_the_allowlist`. Module-private helpers other tasks may call from scripts: `_scan_file(path)`, `_scan_text(text, rel, line=1, used=None)`, `_failure_message(hits)`.
- Produces (controller tool): `python comment_only_check.py --base <git-ref> --allow <allow-file> <path> [<path> ...]`; allow-file lines `<repo-relative-path>|<kind>|"<quoted text fragment>"`, kind in {`message`, `string`, `rename`, `assert`}; exit 0 when every difference is allowed, 1 otherwise (2 on a usage error); prints `path:line: <old> -> <new>` per difference.

**Binding (read before starting):**
- `python` below means `C:\Users\J\anaconda3\envs\biophys-env\python.exe` (in the Bash tool, `/c/Users/J/anaconda3/envs/biophys-env/python.exe`); the `python` on PATH is a different interpreter without torch or pytest. Commands written with backslash paths are PowerShell commands; in Bash use forward slashes.
- Spec §2.1 (what is walked, excluded and read), §2.2 (patterns and allowlist exactly), §2.3 (the three feature ids, the two renames), §2.7 (the checker), §2.8 (the shrinking list). Review Focus 5: a legitimate new token shaped like a label must fail with a message naming the file, line, family and the allowlist to extend.
- Allowlist exactly: tokens `F0`, `D0`, `L2` everywhere (NO `L1`: the only one in the tree, `tests/test_settings_persistence.py:409`, is a process label and stays a hit); summary-feature ids only in `core/SBI/statistics.py` and `core/SBI/summaries.py`; `S2` only in `core/FDT/spectral.py`; shapes `\bPhase [A-Z]\d\b`, `U\+[0-9A-F]{4}`, and README's Apple-chip line; test-name tokens `f0`, `d0`; exact names `test_the_B7_pair_shares_one_flag`, `test_v1_params_migrate_to_placeholder_boxes`, `test_v2_params_migrate_to_a_linear_box`. NO published-paper section shape.
- The module stays under its own scan (no `Path(__file__)` exclusion). Every literal in it must be clean on its own: patterns are built from adjacent literals (each a separate token), the section sign is `chr(0xA7)`, the family names "ruling" and "walkthrough" are the constants `_RULING`/`_WALKTHROUGH`, and each positive sample carries U+00B7 (MIDDLE DOT, `_GAP = chr(0xB7)`) inside its label, removed at run time. Never type a backslash-u escape through the Write/Edit tools: they decode it into the character itself, and a literal section sign in this module is a hit on itself.
- H4 in everything written into `core/` and `tests/`: no piece numbers, section signs, decision ids, rulings, review ids, "Task N", trap ids, "guardrail N" in code, comments, docstrings, messages, test names or test docstrings. State reasons in words.
- Renames (§2.7 (c), the only non-docstring change this task may make): `test_the_d7_and_d8_dialogs_default_to_cancel` → `test_the_fresh_cache_and_narrowed_posterior_dialogs_default_to_cancel`; `test_the_cuda_graph_preserves_the_rng_contract_c11_depends_on` → `test_the_cuda_graph_preserves_the_rng_contract_a_cache_resume_depends_on`. Their docstrings are left for the test-cleanup tasks (both files stay on the list).
- Feature ids by full label (`core/SBI/statistics.py` `FEATURE_LABELS`): A3 → `A3_log_fpeak`, A1 → `A1_mean`, D3 → `D3_bimodality`.
- `NOT_YET_CLEAN` is exactly the set of files with hits after this task's edits, computed by running the scan (Step 8). Expected: 107 entries (85 under `core/`, 20 under `tests/`, `conftest.py`, `requirements.txt`) = the 108 files of `recon/design-scanrule-hits.tsv` minus `core/SBI/embedded_network.py`. `core/SBI/derived.py`, which the recon missed (its one hit is an upper-case working-document section number), is cleaned in Step 7 and is absent too. `tests/test_source_hygiene.py` and `README.md` are never on it.
- No `.py` under `.superpowers/`; the controller tool and helper script live in `C:\Users\J\AppData\Local\Temp\prism-piece6\`.
- Never edit a source file while a pytest run is in progress. Run only the focused tests below; the controller runs the fast gate.

- [ ] **Step 1: Write the controller tool.** Create the directory `C:\Users\J\AppData\Local\Temp\prism-piece6\` and write `comment_only_check.py` there, exactly:

```python
"""Prove that an edit to Python files changed only comments, docstrings and an allowed list of other things.

    python comment_only_check.py --base <git-ref> --allow <allow-file> <path> [<path> ...]

For each path (repository-relative, run from inside the repository), the version at <base>
(``git show <base>:<path>``) and the working-tree version are each tokenized, their COMMENT and NL
tokens dropped, the rest untokenized and parsed, and every docstring (the first statement of a
module, class or function, when it is a string constant) removed. The two trees are then walked node
by node, ignoring positions. Every difference prints as

    path:line: <old> -> <new>    [allowed: <kind> "<fragment>"]   or   [NOT ALLOWED]

The allow file holds one permitted difference per line, ``<path>|<kind>|"<fragment>"``, kind one of
message, string, rename, assert; blank lines and lines starting with # are ignored. A difference is
allowed when an entry for its path has a kind that fits the difference and a fragment contained in
the difference's old or new text:

    string difference (a str constant, or an f-string whose {...} fields are unchanged)   message, string, assert
    rename (a def or class name changed)                        rename
    anything else (statements added, removed or changed)        assert

Exit 0 when every difference is allowed, 1 when any is not, 2 on a usage error.
"""
import argparse
import ast
import difflib
import io
import subprocess
import sys
import tokenize
from pathlib import Path

KINDS = {"message", "string", "rename", "assert"}
FITS = {"string": {"message", "string", "assert"}, "rename": {"rename"}, "structure": {"assert"}}
WIDTH = 160


class UsageError(Exception):
    pass


def comment_free_tree(source: str, label: str) -> ast.Module:
    """Tokenize, drop COMMENT and NL tokens, untokenize and parse; then strip the docstrings.

    The comment-free reconstruction must parse to exactly the tree of the original source (a comment
    never reaches the AST); the original's tree is the one returned, because untokenize does not keep
    line numbers and every difference is reported at its real line."""
    try:
        toks = [t for t in tokenize.generate_tokens(io.StringIO(source).readline)
                if t.type not in (tokenize.COMMENT, tokenize.NL)]
        stripped = ast.parse(tokenize.untokenize(toks))
        tree = ast.parse(source)
    except (SyntaxError, tokenize.TokenError, ValueError) as exc:
        raise UsageError(f"{label}: does not tokenize and parse: {exc}") from exc
    if ast.dump(stripped) != ast.dump(tree):
        raise UsageError(f"{label}: dropping the comments changed the parse")
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                and isinstance(body, list) and body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str)):
            body.pop(0)
            if not body:
                body.append(ast.Pass())
    return tree


def _text(node) -> str:
    if node is None:
        return ""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.AST):
        try:
            return ast.unparse(node)
        except Exception:  # noqa: BLE001 -- a fragment of a tree may not unparse on its own
            return ast.dump(node)
    return repr(node)


def _stringish(node) -> bool:
    return isinstance(node, ast.JoinedStr) or (isinstance(node, ast.Constant) and isinstance(node.value, str))


def _key(item) -> str:
    return ast.dump(item) if isinstance(item, ast.AST) else repr(item)


def tree_diff(old, new, line: int, out: list) -> None:
    """Append (line, cls, old_text, new_text) for every difference between two parsed trees."""
    here = getattr(new, "lineno", None) or getattr(old, "lineno", None) or line
    if _stringish(old) and _stringish(new):
        if _key(old) != _key(new):
            fields = [[ast.dump(v) for v in getattr(n, "values", []) if not isinstance(v, ast.Constant)]
                      for n in (old, new)]
            out.append((here, "string" if fields[0] == fields[1] else "structure", _text(old), _text(new)))
        return
    if isinstance(old, list) and isinstance(new, list):
        _list_diff(old, new, line, out)
        return
    if isinstance(old, ast.AST) and isinstance(new, ast.AST) and type(old) is type(new):
        for field in old._fields:
            a, b = getattr(old, field, None), getattr(new, field, None)
            if isinstance(a, (ast.AST, list)) or isinstance(b, (ast.AST, list)):
                tree_diff(a, b, here, out)
            elif a != b:
                if field == "name" and isinstance(old, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    out.append((here, "rename", str(a), str(b)))
                else:
                    out.append((here, "structure", _text(old), _text(new)))
        return
    if _key(old) != _key(new):
        out.append((here, "structure", _text(old), _text(new)))


def _list_diff(old: list, new: list, line: int, out: list) -> None:
    if len(old) == len(new):
        for a, b in zip(old, new):
            tree_diff(a, b, line, out)
        return
    matcher = difflib.SequenceMatcher(a=[_key(x) for x in old], b=[_key(x) for x in new], autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if tag == "replace" and i2 - i1 == j2 - j1:
            for a, b in zip(old[i1:i2], new[j1:j2]):
                tree_diff(a, b, line, out)
            continue
        for a in old[i1:i2]:
            out.append((f'{getattr(a, "lineno", line)} (base)', "structure", _text(a), ""))
        for b in new[j1:j2]:
            out.append((getattr(b, "lineno", line), "structure", "", _text(b)))


def read_allow(path: Path) -> list:
    entries = []
    for n, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("|", 2)
        if len(parts) != 3 or parts[1].strip() not in KINDS:
            raise UsageError(f"{path}:{n}: expected <path>|<kind>|\"<fragment>\" with kind in {sorted(KINDS)}")
        frag = parts[2].strip()
        if len(frag) >= 2 and frag[0] == frag[-1] == '"':
            frag = frag[1:-1]
        if not frag:
            raise UsageError(f"{path}:{n}: empty fragment")
        entries.append({"path": parts[0].strip().replace("\\", "/"), "kind": parts[1].strip(),
                        "frag": frag, "where": f"{path.name}:{n}", "used": False})
    return entries


def _short(s: str) -> str:
    s = repr(s)
    return s if len(s) <= WIDTH else s[:WIDTH - 3] + "..."


def check(repo: Path, base: str, rel: str, allow: list) -> tuple[int, int]:
    """Print every difference in one file; return (differences, not allowed)."""
    shown = subprocess.run(["git", "show", f"{base}:{rel}"], cwd=repo, capture_output=True)
    if shown.returncode != 0:
        raise UsageError(f"{rel}: not readable at {base}: {shown.stderr.decode(errors='replace').strip()}")
    new_path = repo / rel
    if not new_path.is_file():
        raise UsageError(f"{rel}: not in the working tree")
    old = comment_free_tree(shown.stdout.decode("utf-8"), f"{rel}@{base}")
    new = comment_free_tree(new_path.read_text(encoding="utf-8"), rel)
    diffs: list = []
    tree_diff(old, new, 1, diffs)
    bad = 0
    for line, cls, a, b in diffs:
        verdict = "[NOT ALLOWED]"
        for e in allow:
            if e["path"] == rel and e["kind"] in FITS[cls] and (e["frag"] in a or e["frag"] in b):
                e["used"] = True
                verdict = f'[allowed: {e["kind"]} "{e["frag"]}"]'
                break
        else:
            bad += 1
        print(f"{rel}:{line}: {_short(a)} -> {_short(b)}    {verdict}")
    return len(diffs), bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Prove an edit changed only comments, docstrings and allowed items.")
    ap.add_argument("--base", required=True, help="the git ref to compare against, e.g. HEAD or a commit")
    ap.add_argument("--allow", required=True, type=Path, help="the allow file")
    ap.add_argument("paths", nargs="+", help="repository-relative Python files")
    args = ap.parse_args(argv)
    try:
        top = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True)
        repo = Path(top.stdout.strip())
        allow = read_allow(args.allow)
        total = bad = 0
        for p in args.paths:
            rel = Path(p).as_posix() if not Path(p).is_absolute() else Path(p).resolve().relative_to(repo).as_posix()
            n, b = check(repo, args.base, rel, allow)
            total, bad = total + n, bad + b
    except (UsageError, subprocess.CalledProcessError, OSError, ValueError) as exc:
        print(f"usage error: {exc}", file=sys.stderr)
        return 2
    for e in allow:
        if not e["used"] and e["path"] in {Path(p).as_posix() for p in args.paths}:
            print(f'note: {e["where"]} allowed nothing: {e["path"]}|{e["kind"]}|"{e["frag"]}"')
    print(f"{len(args.paths)} file(s), {total} difference(s), {bad} not allowed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Self-check the tool on the clean tree.** Write `C:\Users\J\AppData\Local\Temp\prism-piece6\allow-empty.txt` containing the single line `# nothing is permitted`. From the repo root run
  `python C:\Users\J\AppData\Local\Temp\prism-piece6\comment_only_check.py --base HEAD --allow C:\Users\J\AppData\Local\Temp\prism-piece6\allow-empty.txt core/orchestrator.py tests/test_tool.py`
  Expected: the single line `2 file(s), 0 difference(s), 0 not allowed`, exit code 0.

- [ ] **Step 3: Write the permitted-change list before any edit.** `C:\Users\J\AppData\Local\Temp\prism-piece6\allow-task-01.txt`, one permitted difference per line (the format every cleanup task uses):

```
# Task 1: the only non-docstring changes are the two test renames
tests/test_nav_and_gating.py|rename|"test_the_d7_and_d8_dialogs_default_to_cancel"
tests/test_user_sbi.py|rename|"test_the_cuda_graph_preserves_the_rng_contract_c11_depends_on"
```

  (A `message` or `string` line allows a changed string constant or f-string whose old or new text contains the fragment; `rename` a changed def/class name; `assert` a changed, added or removed statement. Quote a distinctive piece of the old or the new text.)

- [ ] **Step 4: Write `tests/test_source_hygiene.py`**, with `NOT_YET_CLEAN` empty for now. The patterns are spec §2.2's, each split into adjacent literals so that no single literal matches any family:

```python
"""No shippable file points at a working document.

The program, the tests, the launchers, the README, the requirements file and the reader pages under
docs/guide must explain themselves in words. A comment that says why by naming a design document, a
decision id, a numbered rule or a review finding stops making sense the day that document is
archived, and the reader it was written for never had it. This suite reads the prose of every such
file -- comments, docstrings and string literals in Python (never code identifiers, where short math
names would be false positives), every line of the prose files and reader pages, and the names of
test functions and classes -- and fails on anything shaped like such a reference.

Every pattern below is assembled from adjacent literals, each of which is clean on its own, so this
module stays under its own scan.
"""
import ast
import functools
import io
import re
import tokenize
from collections import namedtuple
from pathlib import Path

from core.SBI.statistics import FEATURE_LABELS
from tests._fixtures import CODE_FILES, CODE_ROOTS

_REPO = Path(__file__).resolve().parents[1]
_SELF = Path(__file__).resolve()

Hit = namedtuple("Hit", "path line family match")

_SECTION_SIGN = chr(0xA7)  # the section sign, never written literally here
_RULING, _WALKTHROUGH = "rul" "ing", "walk" "through"  # two family names that are labels themselves

PROSE_PATTERNS: dict[str, re.Pattern] = {name: re.compile(source) for name, source in {
    "doc": (r"(?i)PRISM_" r"HAND" r"OFF|\bhand" r"off\b|\bSTATE\.md\b|\bCLAUDE\.md\b|super" r"powers"
            r"|\bsdd\b|\bprogress\.md\b|\bled" r"ger\b"),
    "piece": (r"(?i)\bpieces?[ -]?\d+\b|\bwhole-" r"piece\b"
              r"|\b(?:this|that|each|same|earlier|later|next|previous) " r"piece\b"),
    "section": _SECTION_SIGN + r"|\b(?:[Ss]ec|SEC)(?:tion|TION|\.)? ?\d+(?:\.\d+)+\b",
    "spec": (r"\bspec(?:'s)? (?:" + _SECTION_SIGN + r"|[Ss]ec|section|\d)|\b(?:design|piece-\d+) " r"spec\b"
             r"|\bthe spec" r"'s\b|\bspec " r"NOTES\b|\bplan(?:ning)? " r"rul" r"ing"),
    _RULING: r"\brul" r"ings?\b|\bowner " r"ruled\b",
    "review": (r"\breview" r"'s\b|\b[Ff]ix " r"round\b|\bReview " r"Focus\b|\bfinding \d|\breviewers?(?:'s)? " r"probe"
               r"|\breviewers? " r"probed|\bprobed by (?:\w+ )?" r"reviewers\b|\bthe review " r"of\b"
               r"|\bper the " r"review\b|\b[Cc]hecklist(?: item)? \d|\bBack" r"log\b"
               r"|\b(?:CRITICAL|IMPORTANT|Critical|Important) \d\b|\bdefect [A-Z]-?\d+\b"),
    "task": r"\bTasks? \d+\b|\btasks? T\d+",
    "trap": r"\b(?:trap|TRAP)s? [A-Z]+\d+[a-z]?\b|\bCHI\d+\b",
    "guardrail": r"(?i)\bguardrails?[ -]\d+\b",
    "series": r"(?<![\w-])[CS]-\d+\b",
    "appendix": r"\bAppendix " r"[A-Z]\b",
    _WALKTHROUGH: r"\bwalk" r"through\b|\brows? [A-Z]\d+\b",
    "programme": r"\bhardening " r"programme\b|\bone-flow " r"design\b|\bclean[- ]break\b",
    "id": r"(?<![\w./])(?<![a-z]-)(?:R-F\d+|FE\d+|[A-Z]\d{1,3}[a-z]?)(?!\w)(?!-[a-z])",
}.items()}

TEST_NAME_PATTERN = re.compile(
    r"(?i)(?:^|_)(?:piece\d+|[a-z]\d{1,3}[a-z]?|rf\d+|fe\d+|chi\d+|task\d+|guardrail\d*|round\d+"
    r"|whole_piece|fix_round|walk" r"through|hand" r"off|appendix|rul" r"ing)(?:_|$)")

# The summary features' short ids, allowed only in the two files that define and compute them.
FEATURE_IDS = frozenset(label.split("_", 1)[0] for label in FEATURE_LABELS)

ALLOW_TOKENS = frozenset({"F0", "D0", "L2"})  # a force amplitude, a diffusion constant, a norm
ALLOW_PER_FILE: dict[str, frozenset[str]] = {
    "core/SBI/statistics.py": FEATURE_IDS,
    "core/SBI/summaries.py": FEATURE_IDS,
    "core/FDT/spectral.py": frozenset({"S" "2"}),  # scipy's name for the window's sum of squares
}
ALLOW_SHAPES: tuple[re.Pattern, ...] = (
    re.compile(r"\bPhase [A-Z]\d\b"),               # the reduction map's phases
    re.compile(r"U\+[0-9A-F]{4}"),                  # Unicode code points
    re.compile(r"\bM[1-4](?: / M[1-4]){3}\b"),      # the Apple chips the README names
)
ALLOW_TEST_NAME_TOKENS = frozenset({"f0", "d0"})
ALLOW_TEST_NAMES: frozenset[str] = frozenset({
    "test_the_B7_pair_shares_one_flag",                 # names a summary feature by its id
    "test_v1_params_migrate_to_placeholder_boxes",      # model-file schema versions
    "test_v2_params_migrate_to_a_linear_box",
})

PROSE_FILES = ("README.md", "requirements.txt", "pytest.ini", "run.bat", "run.sh")
READER_DOCS_DIR = "docs/guide"

# Files not yet cleaned. It may only shrink: every file on it must still have a hit, so a cleaned
# file has to leave it, and the last cleaning step deletes it together with the test that reads it.
NOT_YET_CLEAN: frozenset[str] = frozenset({
})

_DIRECTIVE = re.compile(r"noqa(?::[ \t]*[A-Z]+[0-9]+(?:[ \t]*,[ \t]*[A-Z]+[0-9]+)*)?"
                        r"|type:[ \t]*ignore(?:\[[^\]]*\])?")
_PROSE_TOKENS = {tokenize.COMMENT, tokenize.STRING, tokenize.FSTRING_MIDDLE}


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(_REPO).as_posix()
    except ValueError:
        return path.as_posix()


def _scan_text(text: str, rel: str, line: int = 1, used: set | None = None) -> list[Hit]:
    """Every match in one line of prose. Allowed shapes are blanked first; allowed tokens and a file's
    own tokens are spared from the id family. ``used`` collects which allowlist entries spared
    something."""
    for i, shape in enumerate(ALLOW_SHAPES):
        text, n = shape.subn(lambda m: " " * len(m.group()), text)
        if n and used is not None:
            used.add(("shape", i))
    hits = []
    for family, pattern in PROSE_PATTERNS.items():
        for m in pattern.finditer(text):
            token = m.group()
            if family == "id" and token in ALLOW_TOKENS:
                if used is not None:
                    used.add(("token", token))
                continue
            if family == "id" and token in ALLOW_PER_FILE.get(rel, ()):
                if used is not None:
                    used.add(("file", rel))
                continue
            hits.append(Hit(rel, line, family, token))
    return hits


def scan_python(path: Path, *, used: set | None = None) -> list[Hit]:
    """The hits in a Python file's comments, docstrings and string literals (never its code). A lint
    directive (``noqa: CODE`` or ``type: ignore[...]``) is stripped; an explanation after it is read."""
    rel = _rel(path)
    hits = []
    source = path.read_text(encoding="utf-8")
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type not in _PROSE_TOKENS:
            continue
        text = _DIRECTIVE.sub("", tok.string) if tok.type == tokenize.COMMENT else tok.string
        for i, segment in enumerate(text.split("\n")):
            hits += _scan_text(segment, rel, tok.start[0] + i, used)
    return hits


def scan_prose(path: Path, *, used: set | None = None) -> list[Hit]:
    """The hits in every line of a prose file."""
    rel = _rel(path)
    hits = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        hits += _scan_text(line, rel, n, used)
    return hits


def _scan_file(path: Path, *, used: set | None = None) -> list[Hit]:
    return scan_python(path, used=used) if path.suffix == ".py" else scan_prose(path, used=used)


def scanned_files() -> list[Path]:
    """Every file the rule covers: the Python under CODE_ROOTS, CODE_FILES and tests/ (this module
    included), the prose files, and every page under docs/guide once it exists."""
    python = {p for root in CODE_ROOTS for p in (_REPO / root).rglob("*.py")}
    python |= {_REPO / name for name in CODE_FILES}
    python |= set((_REPO / "tests").rglob("*.py"))
    files = sorted(python) + [_REPO / name for name in PROSE_FILES]
    guide = _REPO / READER_DOCS_DIR
    if guide.is_dir():
        files += sorted(guide.glob("*.md"))
    return files


@functools.lru_cache(maxsize=None)
def _scan_everything() -> tuple[dict[str, list[Hit]], frozenset]:
    """Hits per file over the whole rule, and the allowlist entries that spared something outside
    this module (whose own literals must not keep an entry alive)."""
    by_file, used = {}, set()
    for path in scanned_files():
        by_file[_rel(path)] = _scan_file(path, used=None if path.resolve() == _SELF else used)
    return by_file, frozenset(used)


def _test_name_labels(name: str) -> list[str]:
    labels, pos = [], 0
    while (m := TEST_NAME_PATTERN.search(name, pos)):
        token = m.group().strip("_").lower()
        if token not in ALLOW_TEST_NAME_TOKENS:
            labels.append(token)
        pos = m.start() + 1
    return labels


@functools.lru_cache(maxsize=None)
def _test_definitions() -> tuple[tuple[str, int, str], ...]:
    out = []
    for path in scanned_files():
        if path.suffix != ".py":
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                    and node.name.lower().startswith("test")):
                out.append((_rel(path), node.lineno, node.name))
    return tuple(out)


def _failure_message(hits: list[Hit]) -> str:
    listing = "\n  ".join(f"{h.path}:{h.line} {h.family} {h.match!r}" for h in hits)
    return (f"{len(hits)} reference(s) to a working document or one of its labels:\n  {listing}\n"
            "State the reason in words instead of pointing at a document. If a match is a legitimate "
            "token shaped like a label (a physics name, a new feature id, a published paper's section "
            "number), add the narrowest entry that spares it to tests/test_source_hygiene.py -- "
            "ALLOW_TOKENS (one token, everywhere), ALLOW_PER_FILE (one token, one file) or ALLOW_SHAPES "
            "(a pattern) -- in the commit that introduces it.")


def test_no_file_outside_the_pending_list_cites_a_working_document():
    """The rule itself, over every file it covers. The walk is checked first: it must reach the
    top-level code files, the five prose files and this module, and at least 150 Python files, so a
    walk that silently stopped cannot pass."""
    files = scanned_files()
    missing = [name for name in (*CODE_FILES, *PROSE_FILES) if not (_REPO / name).is_file()]
    assert not missing, f"a file the rule names does not exist: {missing}"
    assert _SELF in {p.resolve() for p in files}, "this module must stay under its own scan"
    walked = sum(p.suffix == ".py" for p in files)
    assert walked >= 150, f"the scan walked only {walked} Python files"
    by_file, _used = _scan_everything()
    hits = [h for rel, found in sorted(by_file.items()) if rel not in NOT_YET_CLEAN for h in found]
    assert not hits, _failure_message(hits)


def test_every_pending_file_still_has_a_hit():
    """The pending list is always the true remainder: a file cleaned, or deleted, must leave it."""
    by_file, _used = _scan_everything()
    done = sorted(rel for rel in NOT_YET_CLEAN if not by_file.get(rel))
    assert not done, ("on NOT_YET_CLEAN with no hit left, or no longer scanned -- remove from the "
                      "list:\n  " + "\n  ".join(done))


_GAP = chr(0xB7)  # written inside each label below and removed at run time, so no literal matches


def _u(text: str) -> str:
    return text.replace(_GAP, "")


def test_the_matchers_catch_each_family_and_spare_the_allowed_tokens(tmp_path):
    """The matchers are checked on samples first: with the tree clean, the tree walk alone would go on
    passing if a pattern stopped matching anything."""
    catch = {
        "doc": ["see PRISM_HAND·OFF.md", "noted in STATE·.md", "the super·powers workspace"],
        "piece": ["added in pie·ce 3", "this pie·ce keeps it"],
        "section": ["as " + _SECTION_SIGN + "3.4 says", "Sec·tion 11.6 covers it"],
        "spec": ["spec " + _SECTION_SIGN + "4.1 fixes it", "the design ·spec says"],
        _RULING: ["the controller rul·ing", "the owner rul·ed so"],
        "review": ["the whole-code review·'s finding", "found in a fix ·round", "defect D·3 again"],
        "task": ["done in Ta·sk 7", "tasks T·13 to T·18"],
        "trap": ["see trap X·5", "see CHI·10"],
        "guardrail": ["guard·rail 2 holds", "GUARD·RAIL 5 again"],
        "series": ["the C·-11 cache", "an S·-1 note"],
        "appendix": ["Appen·dix A tells it"],
        _WALKTHROUGH: ["the walk·through checklist", "row D·15 checks it"],
        "programme": ["the hardening program·me", "since the clean ·break"],
        "id": ["refused (D·7)", "the rows R-F·1 and FE·3", "ids T·13-T·18"],
    }
    for family, samples in catch.items():
        for sample in samples:
            found = {h.family for h in _scan_text(_u(sample), "core/example.py")}
            assert family in found, (family, _u(sample), found)
    spare = {
        "doc": ["the state of the bundle", "a ledgerless cache"],
        "piece": ["the piece removes", "a two-piece model"],
        "section": ["Fig. 3", "the section headings"],
        "spec": ["carried on the spec", "the batch plan"],
        _RULING: ["an owner guard", "the ruler's edge"],
        "review": ["a reviewer reads it", "round 2 of a narrowing"],
        "task": ["the batch task", "tasks run in turn"],
        "trap": ["a trap for the unwary", "trapped in a well"],
        "guardrail": ["a guardrail against drift", "the guardrails hold"],
        "series": ["a C-contiguous array", "an S-shaped curve"],
        "appendix": ["an appendix to the paper", "Appendix of results"],
        _WALKTHROUGH: ["rows A to D", "walk through the steps"],
        "programme": ["a clean breakpoint", "the training programme"],
        "id": ["the F0 amplitude, the L2 norm and D0", "A1_mean and D3_bimodality", "UTF-8, SHA-256, HDF5",
               "V's columns, 0..K-1, B=64/T=20000", "Phase B1", "U+E000..U+E004", "M1 / M2 / M3 / M4"],
    }
    assert set(spare) == set(catch) == set(PROSE_PATTERNS)
    for family, samples in spare.items():
        for sample in samples:
            assert _scan_text(sample, "core/example.py") == [], (family, sample, _scan_text(sample, "core/example.py"))
    feature = _u("the A·3 feature")
    assert _scan_text(feature, "core/SBI/statistics.py") == []
    assert [h.family for h in _scan_text(feature, "core/SBI/chi.py")] == ["id"]
    assert _scan_text(_u("scipy's S·2"), "core/FDT/spectral.py") == []
    assert [h.family for h in _scan_text(_u("scipy's S·2"), "core/FDT/plots.py")] == ["id"]
    # a lint directive is stripped; an explanation after it is still read
    lint = tmp_path / "lint.py"
    lint.write_text(_u("import os  # noqa: E·402, F·401 -- kept for Ta·sk 7\nimport re  # noqa: F·401\n"),
                    encoding="utf-8")
    assert [(h.line, h.family) for h in scan_python(lint)] == [(1, "task")]
    # test names: every label shape is caught, the physics tokens and plain words are spared, and an
    # allowed token does not hide a label right beside it
    for name in ("test_the_d7_and_d8_dialogs_default_to_cancel", "test_the_contract_c11_depends_on",
                 "test_piece5_records", "test_task12_gate", "test_guardrail_holds", "test_after_the_fix_round",
                 "test_round2_findings", "test_rf1_fix", "test_fe3_fix", "test_chi10_channel",
                 "test_the_handoff_is_gone", "test_the_walkthrough_rows", "test_the_appendix_incident",
                 "test_the_ruling_holds", "test_the_whole_piece_gate"):
        assert _test_name_labels(name), name
    for name in ("test_the_f0_grid_is_log_spaced", "test_the_d0_constant", "test_the_sha256_digest",
                 "test_log_t_obs_widens_the_row"):
        assert _test_name_labels(name) == [], name
    assert _test_name_labels("test_the_f0_d7_case") == ["d7"]


def test_every_allowlist_entry_still_matches_something():
    """An entry that spares nothing is stale, and would hide the next real reference of its shape.
    Only uses outside this module count: its own literals would otherwise keep every entry alive.
    ALLOW_TEST_NAME_TOKENS is part of the test-name matcher rather than an entry: it keeps a physics
    name such as the drive amplitude's usable in a test name, whether or not one uses it today."""
    _by_file, used = _scan_everything()
    stale = [f"ALLOW_TOKENS {t!r}" for t in sorted(ALLOW_TOKENS) if ("token", t) not in used]
    stale += [f"ALLOW_PER_FILE {rel!r}" for rel in sorted(ALLOW_PER_FILE) if ("file", rel) not in used]
    stale += [f"ALLOW_SHAPES {s.pattern!r}" for i, s in enumerate(ALLOW_SHAPES) if ("shape", i) not in used]
    names = {name for _path, _line, name in _test_definitions()}
    stale += [f"ALLOW_TEST_NAMES {n!r}" for n in sorted(ALLOW_TEST_NAMES) if n not in names]
    assert not stale, "allowlist entries that spare nothing -- delete them:\n  " + "\n  ".join(stale)


def test_no_test_name_carries_a_process_label():
    """A test's name is what a failing run prints, so it states the behaviour, never a label."""
    offenders = [f"{path}:{line} {name} {_test_name_labels(name)}" for path, line, name in _test_definitions()
                 if name not in ALLOW_TEST_NAMES and _test_name_labels(name)]
    assert not offenders, ("test names carrying a label -- rename each so the name states the behaviour; "
                           "a legitimate name goes in ALLOW_TEST_NAMES:\n  " + "\n  ".join(offenders))


def test_a_failure_names_the_file_line_family_and_the_allowlist(tmp_path):
    """A legitimate new token shaped like a label -- a future feature id, a published paper's section
    number -- fails with everything needed to act on it: where, which family, what matched, and the
    allowlist to extend."""
    module = tmp_path / "new_feature.py"
    new_id = "Q" + "7"
    module.write_text('"""A new module."""\n\nDRIFT = 1  # the ' + new_id + " channel\n"
                      "GAIN = 2  # as in Sec" + "tion 4.2 of Gardiner\n", encoding="utf-8")
    hits = scan_python(module)
    where = module.as_posix()
    assert hits == [Hit(where, 3, "id", new_id), Hit(where, 4, "section", "Sec" + "tion 4.2")]
    message = _failure_message(hits)
    assert f"{where}:3 id '{new_id}'" in message
    assert f"{where}:4 section 'Sec" + "tion 4.2'" in message
    for entry in ("ALLOW_TOKENS", "ALLOW_PER_FILE", "ALLOW_SHAPES", "tests/test_source_hygiene.py"):
        assert entry in message, entry
```

- [ ] **Step 5: Run the new suite and see it fail.** `python -m pytest tests/test_source_hygiene.py -q`
  Expected: `2 failed, 4 passed`. `test_no_file_outside_the_pending_list_cites_a_working_document` fails listing about 3,300 hits across 109 files (the recon TSV's 1,976 lines, several hits to a line, plus `tests/test_settings_persistence.py:409`'s `L1`, plus the lines the case-insensitive "Fix round" and "SECTION" forms add — all in the TSV's files except `core/SBI/derived.py:127`, "SECTION 11.5", which Step 7 cleans) (none in `tests/test_source_hygiene.py`, none in `README.md`); `test_no_test_name_carries_a_process_label` fails naming exactly `tests/test_nav_and_gating.py:1280 test_the_d7_and_d8_dialogs_default_to_cancel ['d7', 'd8']` and `tests/test_user_sbi.py:3184 test_the_cuda_graph_preserves_the_rng_contract_c11_depends_on ['c11']`. If the module itself appears in a listing, or any of the other four fails, fix the module (never add it to a list).

- [ ] **Step 6: Rename the two tests.** `tests/test_nav_and_gating.py`: `def test_the_d7_and_d8_dialogs_default_to_cancel(monkeypatch):` → `def test_the_fresh_cache_and_narrowed_posterior_dialogs_default_to_cancel(monkeypatch):`. `tests/test_user_sbi.py`: `def test_the_cuda_graph_preserves_the_rng_contract_c11_depends_on():` → `def test_the_cuda_graph_preserves_the_rng_contract_a_cache_resume_depends_on():` (keep its `@pytest.mark.gpu`). Update the three docstrings that name the old nav test (`tests/conftest.py` `_no_modal_dialogs`, `tests/test_artifact_browser.py` `_answer`, and the `tests/test_worker_dispatch.py` docstring near :344, "the way test_the_d7_and_d8_dialogs_default_to_cancel does") to the new name. Check no other reference remains: `git grep -n "d7_and_d8\|c11_depends_on" -- core tests conftest.py` prints nothing.
  Run `python -m pytest tests/test_source_hygiene.py::test_no_test_name_carries_a_process_label -q` → `1 passed`.

- [ ] **Step 7: Name the three features by full label.** `core/SBI/chi.py` docstring: "the network also sees A3 (log f_peak)" → "the network also sees A3_log_fpeak (log f_peak)". `core/SBI/embedded_network.py` docstring: "~1e29-magnitude traces for A1, exactly-constant ones for D3 (which make" → "~1e29-magnitude traces for ``A1_mean``, exactly-constant ones for ``D3_bimodality`` (which make". `tests/test_conditioning_repair.py` comment: "# exactly constant -> D3 = 1/_EPS" → "# exactly constant -> D3_bimodality = 1/_EPS". `core/SBI/derived.py`, the `describe_derived_f_scale` docstring (~:127): "HAZARD 1 OF SECTION 11.5, answered" → "The first of the two hazards in this module's docstring, answered". It is a working-document section number in upper case, which the recon missed; it is that file's only hit, so cleaning it here keeps `NOT_YET_CLEAN` at 107.

- [ ] **Step 8: Compute `NOT_YET_CLEAN` by running the scan.** Write `C:\Users\J\AppData\Local\Temp\prism-piece6\list_pending.py`:

```python
"""Print every scanned file that still has a hit, as NOT_YET_CLEAN entries (run from the repo root)."""
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.getcwd())
from tests import test_source_hygiene as hygiene  # noqa: E402

pending = sorted({hit.path for path in hygiene.scanned_files() for hit in hygiene._scan_file(path)})
print(f"# {len(pending)} files")
for rel in pending:
    print(f'    "{rel}",')
```

  From the repo root run it (Bash tool: `timeout 300 /c/Users/J/anaconda3/envs/biophys-env/python.exe C:/Users/J/AppData/Local/Temp/prism-piece6/list_pending.py`; a bare `python` in Git Bash is a system Python 3.13 without torch or pytest). Expected first line `# 107 files`: 85 under `core/`, 20 under `tests/`, plus `conftest.py` and `requirements.txt`; `core/SBI/embedded_network.py`, `core/SBI/derived.py`, `README.md` and `tests/test_source_hygiene.py` are absent. Paste the entry lines, sorted, one per line, between `frozenset({` and `})` of `NOT_YET_CLEAN`. If the count is not 107, stop and report the difference against `.superpowers/sdd/2026-09-25-documentation-and-retrain-readiness/recon/design-scanrule-hits.tsv` (its 108 files, plus `tests/test_settings_persistence.py:409`, which the TSV lacks because it allowed `L1`).

- [ ] **Step 9: Run the suite green.** `python -m pytest tests/test_source_hygiene.py -q` → `6 passed`.

- [ ] **Step 10: Run the comment-only check on every edited file** (not the new module, which has no base version). From the repo root:
  `python C:\Users\J\AppData\Local\Temp\prism-piece6\comment_only_check.py --base HEAD --allow C:\Users\J\AppData\Local\Temp\prism-piece6\allow-task-01.txt tests/test_nav_and_gating.py tests/test_user_sbi.py tests/conftest.py tests/test_artifact_browser.py core/SBI/chi.py core/SBI/embedded_network.py tests/test_conditioning_repair.py core/SBI/derived.py tests/test_worker_dispatch.py`
  Expected: two lines, each ending `[allowed: rename "..."]`, then `9 file(s), 2 difference(s), 0 not allowed`; exit 0. Paste the allow file and this output into the task report.

- [ ] **Step 11: Run the focused tests for the touched files.**
  `python -m pytest tests/test_source_hygiene.py -q` → `6 passed`
  `python -m pytest tests/test_nav_and_gating.py::test_the_fresh_cache_and_narrowed_posterior_dialogs_default_to_cancel -q` → `1 passed`
  `python -m pytest tests/test_user_sbi.py::test_the_cuda_graph_preserves_the_rng_contract_a_cache_resume_depends_on -q` → `1 passed` (skipped only on a machine without CUDA)
  `python -m pytest tests/test_conditioning_repair.py::test_pathological_counter_separates_the_three_populations -q` → `1 passed`
  `python -m pytest tests/test_artifact_store.py::test_the_source_scans_cover_every_code_directory -q` → `1 passed`

- [ ] **Step 12: Commit.**
  `git add tests/test_source_hygiene.py tests/test_nav_and_gating.py tests/test_user_sbi.py tests/conftest.py tests/test_artifact_browser.py tests/test_conditioning_repair.py core/SBI/chi.py core/SBI/embedded_network.py core/SBI/derived.py tests/test_worker_dispatch.py`
  `git commit -m "tests: scan shippable files for working-document references" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"`
  The controller starts the one-process fast gate (`python -m pytest -m "not slow" -q`, in the background, logged) at the same moment as this task's review. Do not run it yourself.

---

### Task 2: Cleanup: the inference core

**Files:**
- Modify (the entries `NOT_YET_CLEAN` lists under `core/SBI/` and `core/orchestrator.py` at the start of this task; expected 14, 126 hit lines): `core/orchestrator.py`, `core/SBI/analysis.py`, `core/SBI/chi.py`, `core/SBI/chi_probes.py`, `core/SBI/observations.py`, `core/SBI/pipeline.py`, `core/SBI/prior_screen.py`, `core/SBI/reparam.py`, `core/SBI/run_guards.py`, `core/SBI/statistics.py`, `core/SBI/summaries.py`, `core/SBI/train.py`, `core/SBI/training_checkpoint.py`, `core/SBI/truncate.py`. `core/Solvers/`, `core/Simulator/` and `core/Models/` carry no hits and are not edited.
- Modify: `core/SBI/embedded_network.py` (a wrong fact only: "including ``posterior_08232026``, which is the baseline", ~:84; already off the list)
- Modify: `tests/test_source_hygiene.py` (`NOT_YET_CLEAN`: remove the 14 entries)
- Test: `tests/test_conditioning_repair.py`, in `test_build_truncation_region_records_the_parents_basis`: `assert "rotation" in str(e) and "D6" in str(e), e` (~:680)
- Outside the repository: `C:\Users\J\AppData\Local\Temp\prism-piece6\allow-task-02.txt`

**Interfaces:**
- Consumes: `tests/test_source_hygiene.py::NOT_YET_CLEAN`, `test_no_file_outside_the_pending_list_cites_a_working_document`, `test_every_pending_file_still_has_a_hit` (Task 1); the controller tool `C:\Users\J\AppData\Local\Temp\prism-piece6\comment_only_check.py --base <git-ref> --allow <allow-file> <path> ...` (Task 1).
- Produces: no new names. The 14 files leave `NOT_YET_CLEAN`.

**Binding (read before starting):**
- `python` below means `C:\Users\J\anaconda3\envs\biophys-env\python.exe` (in the Bash tool, `/c/Users/J/anaconda3/envs/biophys-env/python.exe`); the `python` on PATH is a different interpreter without torch or pytest. Commands written with backslash paths are PowerShell commands; in Bash use forward slashes.
- Spec §2.3 (treatment, vocabulary, id collisions), §2.4 messages 1–6 and 8, §2.5, §2.6, §2.7. H4: nothing you write cites the handoff, STATE, CLAUDE.md, a spec, a plan or any label (piece numbers, section signs, D/V/B/E/P/R/N/F ids, "Task N", "fix round", trap ids, "guardrail N", C-11, "Appendix A"); state the reason in words.
- **Treatment.** A provenance tag ("(piece 3, V4)", "(spec §3.4)", "(Task 7 finding)", "(D7)") is deleted. A reason given by reference is rewritten to state the reason; where a label is a sentence's subject, name the rule in words. Vocabulary (use these words, so the code and the future `rules-and-traps.md` agree):
  - the one thing a narrowing round must not get wrong → the truncated-prior rule: a narrowing round draws from the prior restricted to the region, never from the posterior
  - guardrail 1 → the observation-digest rule: a round refuses unless the stored observation matches
  - guardrail 2 → the narrowed-model rule: a narrowed posterior is marked as such and never loads or infers as a broad one
  - guardrail 3 → the eigenbasis rule: the region is cut in the rotation's leading directions, flat ones left full width
  - guardrail 4 → the unweighted-draws rule: the region comes from unweighted posterior draws, not best fits
  - guardrail 5 → the generous-region rule: a 99.9 % region, with the truth-outside rate watched
  - guardrail 6 → the cost-on-screen rule
  - guardrail 7 → the region-carries-its-basis rule: the child reuses the parent's rotation, never recomputes it, skips directions loaded on t_scale, and names its base prior
  - guardrail 8 → the calibrate-on-the-region rule
  - C-11 → the training checkpoint (the simulation cache); C-1…C-10 → the specific measure in words (e.g. C-8 "the per-row lock-in duration")
  - trap X5 → the calibration's operating-point count is t_scale's effective sample size; trap X10 → the rotation is not reproducible across processes, so a resume reuses the stored one; trap CHI10 → a near-constant channel inflates a standardised Jacobian
  - handoff defect D1 → a region drawn in one rotation and enforced in another; D3 → the region belongs to the simulation cache's identity; D4 → a direction loaded on t_scale would turn the restriction into a reweighting; D6 → a rotation saved transposed
  - "Appendix A <date>" → the incident told in one clause, or nothing; decision ids → the behaviour they decided, in words.
- **Ids collide; map by the nearby words** ("defect", "trap", "Appendix A", "the whole-piece review's", "row", "piece N"), never by the id alone. In these files: D1/D3/D4/D6 next to "defect", a region, V or t_scale are the handoff defects above; D3 is also the bimodality feature; D7 is the near-miss cache refusal; D9 is "every driven recording states the frequency it was driven at, in Hz"; D11 is "there is no chi override: a non-default band or drive amplitude means editing config.py deliberately"; D12 is "a narrowing round whose parent is itself narrowed and whose observation is not that parent's is refused, with no escape hatch"; V1 "every public stage works on a private copy of its config"; V2 "a bad value is refused, never clamped or silently defaulted"; V4 "messages are logging records at info, warning and error"; V6 "one training preview behind the budget lines"; V7 "the Fisher settings are recorded only when the rotation ran in this process"; V8 "a stage given no figure sink closes what it draws"; V9 "sbi's summary writer is off"; B15 "a deferred-cancel section"; CHI2 "the Fisher builds its probes without the resolution filter"; CHI9 "a longer lock-in is not a better one"; X6/X7 are out-of-memory traps (delete the tag). R-, N-, M-, P-, F-ids are review findings or rulings: delete the tag, or state the behaviour when it is the subject. The reviewer reads every rewritten sentence at these ids against its context.
- **Known misses (spec §2.2):** read every file on your list in full, not only the listed lines, and rewrite unlabelled references ("the spec names", "the plan's table", "the piece removes").
- **Messages (spec §2.4), new text fixed here** (the rest of each message unchanged):
  1. `core/orchestrator.py`, `build_posterior`, `if x_obs_digest is None:` Refusal: "…(x_obs_digest is None): guardrail 2 could never fire for the posterior it would produce, and the store…" → "…(x_obs_digest is None): the posterior it would produce could never be checked against its own observation, and the store…". Keep "must name the observation" verbatim (pinned by `tests/test_artifact_store.py::test_a_round_whose_region_names_no_observation_is_refused_before_the_spend`).
  2. `core/orchestrator.py`, `build_truncation_region`: "…must not be truncated (D4) -- and " → "…must not be truncated, since cutting it would turn the restriction into a reweighting -- and ".
  3. `core/orchestrator.py`, `build_truncation_region`, the rotation-mismatch Refusal: `f"-- a transposed, rotation-less or foreign sidecar (D6). Reload the posterior through "` `f"build_posterior: it reconciles a transposed or missing sidecar rotation against the prior, "` `f"and refuses one that is neither."` → `f"-- a transposed, missing or foreign rotation. Load the posterior from the store: loading "` `f"checks its recorded rotation against the one inside its training prior and refuses any "` `f"disagreement."`. (The old tail is also a wrong fact: the sidecar reconciler is retired, and `store.load_posterior` now refuses any disagreement through `reparam.assert_rotation_consistent`.) Keep "has no rotation" (pinned at ~:685).
  4. `core/SBI/truncate.py`, `TruncationRegion.check_basis`: keep the first sentence verbatim ("The truncation region was measured WITH/WITHOUT a Fisher rotation, but the training bijection has none/one." — pinned in `test_a_region_measured_in_one_basis_is_refused_in_a_sign_flipped_one`); "…deletes support the parent never excluded (defect D1, Appendix A 2026-09-09)." → "…deletes support the parent never excluded -- a region drawn in one rotation and enforced in another."
  5. `core/SBI/truncate.py`, same method: keep "DIFFERENT V" verbatim (pinned); "…a slab the parent never occupied -- the 2026-09-02 round kept 0.01% of the parent posterior that way (defect D1). A truncated round…" → "…a slab the parent never occupied -- one round run that way kept 0.01% of the parent posterior. A truncated round…" (spec §2.6: the incident is told as a fact, without its date label).
  6. `core/SBI/truncate.py`, the `[tsnpe] direction {j} NOT truncated` warning: "…turning the restriction into a reweighting (D4)." → "…turning the restriction into a reweighting."
  8. `core/SBI/run_guards.py`, the chi-config Refusal: "This has cost a ~5-day run once already (Appendix A, 2026-08-19).\n" → "This has cost a ~5-day run once already.\n".
  Every message stays a Refusal or a logging record as it is; no message names a box, tab, flag or button.
- **Wrong facts (spec §2.5), what each becomes** (all comments/docstrings):
  - `truncate.py` module docstring, first line: delete "Section 11.6." (it is a working-document section, not the paper's).
  - `truncate.py` "WHY THE REGION LIVES IN THE FISHER EIGENBASIS (guardrail 3)…": delete the sentence naming `posterior_08232026` (deleted) and the claim that ``k`` is FLAT with "99.9% of its weight on one direction loading -1.00*k" (withdrawn: it was read off a transposed table); say instead that some parameters are barely constrained by the data; keep the rest of the reasoning (an HPD box in 13-D physical space would cut those axes on noise; deleted support is permanent; truncate directions 0..K-1 of V, leave the flat ones full width); the heading's tag becomes "(the eigenbasis rule)".
  - `statistics.py:509/:514`: the table row "V_B1_Q 30.3% substituted" → "69.7% substituted"; the self-correcting parenthetical ":514" becomes "(V_B1_Q is the largest by a wide margin.)"; ":506" "(pre-piece-1 layout, deleted by the 2026-09-11 clean break; the figures stand as history)" → "(an older layout, since deleted; the figures stand as history)".
  - `train.py:38` "``chi_k_fixed`` is script-only" → "``chi_k_fixed`` belongs to the ``sbc`` diagnostic".
  - `embedded_network.py:84-85`: delete ", including ``posterior_08232026``, which is the baseline every conditioning-repair gate is measured against" (the artifact was deleted).
  - `chi_probes.py:3-7`: drop "the mask audit script" from the consumer list (it is archived and cannot run); say "observation and predictive-check paths" for "observation/PPC paths" if you touch the line.
  - `chi.py:271-274` ("the duration ceiling is keyed on the batch's FASTEST row… lock_in_batched takes a scalar T_obs today"): rewrite to the present fact — `gen_chi_raw` applies the duration ceiling per row (`lock_in_batched` takes a per-row sample count), so a slow row the placement rescued keeps the prefix its own frequency needs; when the ceiling was one scalar keyed on the fastest row it truncated rescued slow rows back under the floor and held the rescue to ~47 % live.
  - `pipeline.py:1849` "orchestrator, the scripts, the test suites" → "orchestrator, the diagnostics, the window and the test suites".
- **Permitted non-docstring changes (spec §2.7)**, written before any edit into `allow-task-02.txt`, exactly:
  ```
  core/orchestrator.py|message|"guardrail 2 could never fire"
  core/orchestrator.py|message|"must not be truncated (D4)"
  core/orchestrator.py|message|"foreign sidecar (D6)"
  core/SBI/truncate.py|message|"(defect D1, Appendix A 2026-09-09)"
  core/SBI/truncate.py|message|"the 2026-09-02 round kept 0.01%"
  core/SBI/truncate.py|message|"restriction into a reweighting (D4)"
  core/SBI/run_guards.py|message|"(Appendix A, 2026-08-19)"
  tests/test_conditioning_repair.py|assert|"a transposed, missing or foreign rotation. Load"
  tests/test_source_hygiene.py|assert|"core/SBI/"
  tests/test_source_hygiene.py|assert|"core/orchestrator.py"
  ```
  Anything else the checker prints is a defect of this task (undo it; a comment or docstring change never prints).
- **Raw-source hazards** (tests that read these files' text, comments included, or their string constants): `tests/test_user_sbi.py::test_the_vram_ceiling_bounds_the_plan_and_stays_out_of_the_identity` — `inspect.getsource(orchestrator.training_identity)` must not contain "SIM_VRAM_CEILING"; `::test_build_posterior_takes_the_budget_as_arguments_because_the_constants_are_snapshotted` — `build_posterior`'s source after its docstring must contain "TRAINING_NUM_RUNS if num_runs is None" and "TRAINING_RUN_SIZE if run_size_cap is None"; `::test_the_drive_is_charged_at_its_build_peak` and `::test_the_batch_retry_waits_releases_and_restores_the_rng` — `gen_training_data`'s raw source keeps their code needles; `::test_every_sbi_message_is_a_record_at_its_level_and_nothing_prints` — "Config tab" absent from `pipeline`'s code and strings; `tests/test_nav_and_gating.py::test_tsnpe_tab_is_gated_and_never_proposes_from_the_posterior` — no string in `orchestrator.tsnpe_round` may contain "proposal" or "set_default_x"; `tests/test_artifact_store.py::test_the_orchestrator_says_everything_through_its_logger` — no `print` and no "setLevel" string in `orchestrator`; `tests/test_user_sbi.py::test_the_graph_cache_is_not_hung_off_the_solver_class` reads `core/Solvers/sdeint.py` (not on this list: do not edit it; it must keep "_GRAPH_CACHE" and never gain "_SOLVER_SINGLETON" or "_SOLVER = Solver()"); `:3551` "SIM_VRAM_CEILING" as above.
- Global: never edit a source file while pytest runs; no `.py` under `.superpowers/`; messages are logging records, never `print`; one commit, never amended.

- [ ] **Step 1: Confirm the list.** `git grep -n "\"core/SBI/\|\"core/orchestrator" -- tests/test_source_hygiene.py` prints the 14 `NOT_YET_CLEAN` entries above (no `core/Solvers`, `core/Simulator`, `core/Models`), plus four lines that are not entries and stay: the two `ALLOW_PER_FILE` keys (`core/SBI/statistics.py`, `core/SBI/summaries.py`) and the two sample paths in `test_the_matchers_catch_each_family_and_spare_the_allowed_tokens` (`core/SBI/statistics.py`, `core/SBI/chi.py`).

- [ ] **Step 2: Write the permitted-change list** `C:\Users\J\AppData\Local\Temp\prism-piece6\allow-task-02.txt` with the ten lines above, before any other edit.

- [ ] **Step 3: Take the files off the list and see the scan fail.** Remove the 14 entries from `NOT_YET_CLEAN`. Run `python -m pytest tests/test_source_hygiene.py::test_no_file_outside_the_pending_list_cites_a_working_document -q` → FAIL; the message lists 126 hit lines, every one in these 14 files.

- [ ] **Step 4: Message 3, test-first.** In `tests/test_conditioning_repair.py::test_build_truncation_region_records_the_parents_basis` change the assertion to:
  ```python
            assert "rotation" in str(e) and "a transposed, missing or foreign rotation. Load" in str(e), e
  ```
  Run `python -m pytest tests/test_conditioning_repair.py::test_build_truncation_region_records_the_parents_basis -q` → FAIL (AssertionError: the message still says "sidecar (D6)"). Rewrite message 3 as fixed above. Run again → `1 passed`.

- [ ] **Step 5: Messages 1, 2, 4, 5, 6 and 8.** Rewrite each as fixed above. Run
  `python -m pytest tests/test_conditioning_repair.py::test_a_region_measured_in_one_basis_is_refused_in_a_sign_flipped_one tests/test_artifact_store.py::test_a_round_whose_region_names_no_observation_is_refused_before_the_spend -q` → `2 passed`.

- [ ] **Step 6: Clean every comment and docstring hit** in the 14 files by the treatment and vocabulary above, reading each file in full for unlabelled references. Re-run the Step 3 test until it passes.

- [ ] **Step 7: Fix the wrong facts** listed above (including `core/SBI/embedded_network.py`).

- [ ] **Step 8: The scan.** `python -m pytest tests/test_source_hygiene.py -q` → `6 passed`.

- [ ] **Step 9: The comment-only check.** From the repo root:
  `python C:\Users\J\AppData\Local\Temp\prism-piece6\comment_only_check.py --base HEAD --allow C:\Users\J\AppData\Local\Temp\prism-piece6\allow-task-02.txt core/orchestrator.py core/SBI/analysis.py core/SBI/chi.py core/SBI/chi_probes.py core/SBI/observations.py core/SBI/pipeline.py core/SBI/prior_screen.py core/SBI/reparam.py core/SBI/run_guards.py core/SBI/statistics.py core/SBI/summaries.py core/SBI/train.py core/SBI/training_checkpoint.py core/SBI/truncate.py core/SBI/embedded_network.py tests/test_conditioning_repair.py tests/test_source_hygiene.py`
  (drop from the command any listed file you did not edit). Expected: 8 string differences (the seven messages and the one assertion) plus 14 removed `NOT_YET_CLEAN` entries, every line `[allowed: …]`, and the last line `17 file(s), 22 difference(s), 0 not allowed`; exit 0. Paste the allow file and the output into the report.

- [ ] **Step 10: Focused tests.**
  `python -m pytest tests/test_source_hygiene.py tests/test_conditioning_repair.py -q` → all passed
  `python -m pytest tests/test_artifact_store.py::test_a_round_whose_region_names_no_observation_is_refused_before_the_spend tests/test_artifact_store.py::test_the_orchestrator_says_everything_through_its_logger tests/test_nav_and_gating.py::test_tsnpe_tab_is_gated_and_never_proposes_from_the_posterior tests/test_diagnostics.py::test_the_calibration_draw_is_three_helpers_with_the_stratification_seam -q` → `4 passed`
  `python -m pytest tests/test_user_sbi.py::test_the_vram_ceiling_bounds_the_plan_and_stays_out_of_the_identity tests/test_user_sbi.py::test_build_posterior_takes_the_budget_as_arguments_because_the_constants_are_snapshotted tests/test_user_sbi.py::test_the_drive_is_charged_at_its_build_peak tests/test_user_sbi.py::test_the_batch_retry_waits_releases_and_restores_the_rng tests/test_user_sbi.py::test_every_sbi_message_is_a_record_at_its_level_and_nothing_prints tests/test_user_sbi.py::test_cufft_plan_cache_is_cleared_between_training_batches -q` → `6 passed`

- [ ] **Step 11: Commit.**
  `git add core/orchestrator.py core/SBI/analysis.py core/SBI/chi.py core/SBI/chi_probes.py core/SBI/observations.py core/SBI/pipeline.py core/SBI/prior_screen.py core/SBI/reparam.py core/SBI/run_guards.py core/SBI/statistics.py core/SBI/summaries.py core/SBI/train.py core/SBI/training_checkpoint.py core/SBI/truncate.py core/SBI/embedded_network.py tests/test_conditioning_repair.py tests/test_source_hygiene.py`
  `git commit -m "Clean working-document references out of the inference core" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"`
  The controller starts the one-process fast gate (`python -m pytest -m "not slow" -q`, background, logged) at the same moment as this task's review. Do not run it yourself.

---

### Task 3: Cleanup: the store and its neighbours

**Files:**
- Modify (the entries `NOT_YET_CLEAN` lists for these paths at the start of this task; expected 14, 142 hit lines): `core/artifacts/identity.py`, `core/artifacts/manifest.py`, `core/artifacts/report.py`, `core/artifacts/store.py`, `core/cli.py`, `core/config.py`, `core/logging_root.py`, `core/refusals.py`, `core/rng.py`, `core/runs.py`, `core/sim_config.py`, `core/Helpers/file_manager.py`, `core/Helpers/model_store.py`, `core/Helpers/visualizers.py`. `core/registry.py`, `core/forcing.py` and `core/progress.py` carry no hits and are not edited.
- Modify: `tests/test_source_hygiene.py` (`NOT_YET_CLEAN`: remove the 14 entries)
- Test: `tests/test_artifact_store.py`, in `test_a_posterior_manifests_amortized_flag_must_agree_with_its_region`: `with pytest.raises(mf.ManifestError, match="D3"):` (~:71)
- Outside the repository: `C:\Users\J\AppData\Local\Temp\prism-piece6\allow-task-03.txt`

**Interfaces:**
- Consumes: `tests/test_source_hygiene.py::NOT_YET_CLEAN`, `test_no_file_outside_the_pending_list_cites_a_working_document`, `test_every_pending_file_still_has_a_hit` (Task 1); the controller tool `comment_only_check.py` (Task 1).
- Produces: no new names. The 14 files leave `NOT_YET_CLEAN`. (Tasks 15 and 17 later edit `core/rng.py` and `core/artifacts/identity.py` behaviour; this task touches only their prose.)

**Binding (read before starting):**
- `python` below means `C:\Users\J\anaconda3\envs\biophys-env\python.exe` (in the Bash tool, `/c/Users/J/anaconda3/envs/biophys-env/python.exe`); the `python` on PATH is a different interpreter without torch or pytest. Commands written with backslash paths are PowerShell commands; in Bash use forward slashes.
- Spec §2.3–§2.7. H4: nothing you write cites the handoff, STATE, CLAUDE.md, a spec, a plan or any label (piece numbers, section signs, decision/review/ruling ids, "Task N", "fix round", trap ids, "guardrail N", C-11, walkthrough rows, "hardening programme", "clean break"); state the reason in words.
- **Treatment and vocabulary (spec §2.3):** delete a provenance tag; rewrite a reason given by reference. The words to use: guardrail 2 → the narrowed-model rule: a narrowed posterior is marked as such and never loads or infers as a broad one; guardrail 1 → the observation-digest rule; guardrail 7 → the region-carries-its-basis rule; C-11 → the training checkpoint (the simulation cache); trap X5 → the calibration's operating-point count is t_scale's effective sample size; trap X10 → the rotation is not reproducible across processes, so a resume reuses the stored one; handoff defect D3 → the region belongs to the simulation cache's identity; D6 → a rotation saved transposed; "Appendix A <date>" → the incident in one clause, or nothing; decision ids → the behaviour they decided, in words.
- **Ids collide; map by the nearby words**, never the id alone. In these files: `core/cli.py:4` "(piece 2, D1)" is the retired prompt CLI (delete the tag); D5 at `store.py` `load_diagnostic`/`load_fdt` is "a diagnostic (or fdt record) is a measurement, so its loader verifies nothing it does not hand out"; `logging_root.py:75` "walkthrough row D15 expects" → say what the window shows; `store.py:1397` "GUARDRAIL 2 (the G2 refusal in infer_and_visualize)" → "the narrowed-model rule (the refusal in infer_and_visualize)"; `manifest.py:159` "DEFECT D3's signature" → "the amortization flag and the region disagreeing". Decision words: V1 private config copy per public stage; V2 a bad value refused, never clamped or silently defaulted; V3 one Refusal kind with a field key, core messages neutral; V4 logging records plus a `log.txt` per artifact; V5 the inference tabs remember selections and the budget only; B2 every kind listed in a sortable table; B3 `complete` means a valid manifest, `finished` means the run finished; B4 the detail view is text only; B6 delete one artifact at a time, never forced; B7 sweep removes only directories with no usable manifest; B10 the saved summary and lineage report are files, never a store kind; B14 a root-logger handler at start-up; B15 a deferred-cancel section; D7 a cache one setting away is refused before any simulation; D8 `--accept-*` maps onto `Accept`; D11 no chi override; E2 an interrupted fdt run keeps its folder, marked unfinished; E3 one `fdt` kind with a `study` field; E5 a setting too thin to trust warns and the warning is recorded; E7 every run records its seed; E10 the tidy-up sees legacy loose files and deletes nothing on its own. R-, N-, M-, P-, F-ids are review findings or rulings: delete the tag, or state the behaviour when it is the subject (e.g. "R3's recency guard" → "the recency guard: a directory touched within RECENT_WRITE_SECONDS may be a run in flight"). The reviewer reads every rewritten sentence at these ids against its context.
- **Known misses:** read every file on the list in full and rewrite unlabelled references by hand.
- **Message 7 (spec §2.4)**, `core/artifacts/manifest.py`, `validate`: `f"a truncated posterior carries its region and an amortized one carries none (defect D3)")` → `f"a truncated posterior carries its region and an amortized one carries none")`. It stays a `ManifestError`.
- **Wrong facts (spec §2.5), what each becomes** (all comments/docstrings):
  - `config.py:455-467` (the `CHI_FREQ_BOUNDS` comment's "…So that variability is SYSTEMATIC, not statistical… Neither a stronger drive nor a longer recording can recover those probes."): keep the measurements, replace the conclusion with the corrected account — that reading was later overturned: re-locking the same traces over a shorter prefix recovers every one of those probes, so the in-band failures were a drive-cycle limit of the lock-in (a fixed 5 s slice reaches the cycle wall soonest at the high multipliers), not a frequency limit; `CHI_MAX_CYCLES` below is the fix; the high edge 0.3 still stands (capped it stays marginal on phase coherence, and above it the drive entrains the bundle). The "old (0.1, 10.0)" paragraph after it keeps its facts but no longer says those frequencies are unusable.
  - "sidecar" → the manifest (`core/artifacts/store.py` `load_posterior` reads everything from the manifest; the posterior manifest's `transform` block holds the box, log mask and V): `config.py:407` "persisted beside each posterior (<name>.rot.pt)" → "recorded in each posterior manifest's transform block"; `config.py:493-494` "which is why the sidecar records it" → "which is why the manifest records it"; `config.py:497` "written to the sidecar" → "recorded in the manifest"; `config.py:538` "The sidecar carries it" → "The manifest carries it"; `sim_config.py:77` "(the <name>.rot.pt sidecar stores the resulting V)" → "(the posterior's manifest records the resulting V)"; `Helpers/model_store.py:38` "persisted into the posterior's ``.rot.pt`` sidecar" → "recorded in the posterior manifest's transform block"; `Helpers/file_manager.py:312` "a posterior picker that hides ``.rot.pt`` sidecars / ``.loss.npz`` curves" → a present-day example (e.g. "a picker that hides auxiliary files").
  - `config.py:499` "conditioning width 42 + 72 = 114" → "conditioning width 49 + 1 + 72 = 122" (49 summary columns: 41 features + 8 valid flags, `SUMMARY_WIDTH` in `core/SBI/statistics.py`; 1 log T_obs; the 72-wide chi block). `config.py:291` "The cost model at 5000 x 2048 rows, chi width 114 (+13 latent targets), i.e. the retrain's shape:" → "The cost model at 2048 rows per batch, chi width 122 (+13 latent targets):" and its next line "50 x 2048 x 127 x 4 B ~= 52 MB" → "50 x 2048 x 135 x 4 B ~= 55 MB". (`:430`'s "all 114" is history and stays.)
  - `config.py:410` "(the keeper posterior_07012026's coordinate)" and `:419-420` "(prior_forcing_no_forcing.pt) + posterior_07012026 already match this box" → no artifact named as present ("a linear ND prior already matches this box").
  - `config.py:48` "so the CLI, the scripts and the tests get it too" → "so the command-line tool, the window and the tests all get it".
  - `artifacts/identity.py:2-3` "Successor of ``orchestrator.training_identity`` (training-rows/1), which the clean break retires along with every directory it named." → "Successor of the training-rows/1 identity, whose directories were deleted; ``orchestrator.training_identity`` survives as a delegate to this class."
  - Met on the way: `rng.py:12` "exactly as ``scripts/smoke_train.py`` did" → "exactly as ``smoke`` does" (scripts/ is gone).
- **Permitted non-docstring changes (spec §2.7)**, written before any edit into `allow-task-03.txt`, exactly:
  ```
  core/artifacts/manifest.py|message|"(defect D3)"
  tests/test_artifact_store.py|assert|"an amortized one carries none"
  tests/test_source_hygiene.py|assert|"core/artifacts/"
  tests/test_source_hygiene.py|assert|"core/Helpers/"
  tests/test_source_hygiene.py|assert|"core/cli.py"
  tests/test_source_hygiene.py|assert|"core/config.py"
  tests/test_source_hygiene.py|assert|"core/logging_root.py"
  tests/test_source_hygiene.py|assert|"core/refusals.py"
  tests/test_source_hygiene.py|assert|"core/rng.py"
  tests/test_source_hygiene.py|assert|"core/runs.py"
  tests/test_source_hygiene.py|assert|"core/sim_config.py"
  ```
- **Hazards:** `tests/test_nav_and_gating.py::test_the_chi_drive_and_band_are_read_only_and_the_draft_carries_config` reads `core/config.py`'s raw text: it must not contain "TUNABLE per config in the Config tab". `tests/test_refusals.py::test_the_two_cell_sites_both_splice_the_one_phrase` reads the code and strings of `cli.validate_gt_file` and `SimConfig._fill_checked`. `tests/test_refusals.py::test_every_domain_error_is_a_refusal_and_carries_a_field`: `StoreError`, `ManifestError` and `UnitParseError` keep non-empty docstrings. `tests/test_refusals.py::test_the_module_is_torch_free_and_imports_only_the_standard_library` and `::test_logging_root_imports_only_the_standard_librarys_logging` pin the two modules' imports (unchanged by prose edits). `tests/test_refusals.py::test_the_registry_holds_exactly_the_initial_keys_with_neutral_descriptions`: the `FIELDS` descriptions in `core/refusals.py` stay lower-case noun phrases with no control words (they carry no hits; do not edit them).
- Global: never edit a source file while pytest runs; no `.py` under `.superpowers/`; no code outside `core/config.py` builds a literal `Resources/` or `Artifacts/` path; one commit, never amended.

- [ ] **Step 1: Confirm the list.** `git grep -n "\"core/artifacts/\|\"core/Helpers/\|\"core/cli.py\|\"core/config.py\|\"core/logging_root.py\|\"core/refusals.py\|\"core/rng.py\|\"core/runs.py\|\"core/sim_config.py" -- tests/test_source_hygiene.py` prints exactly the 14 paths above.

- [ ] **Step 2: Write the permitted-change list** `C:\Users\J\AppData\Local\Temp\prism-piece6\allow-task-03.txt` with the eleven lines above, before any other edit.

- [ ] **Step 3: Take the files off the list and see the scan fail.** Remove the 14 entries from `NOT_YET_CLEAN`. `python -m pytest tests/test_source_hygiene.py::test_no_file_outside_the_pending_list_cites_a_working_document -q` → FAIL; the message lists 142 hit lines (the recon's 141 plus `core/artifacts/store.py:965`, "Fix round 1"), every one in these 14 files.

- [ ] **Step 4: Message 7, test-first.** In `tests/test_artifact_store.py::test_a_posterior_manifests_amortized_flag_must_agree_with_its_region` change the pin to:
  ```python
        with pytest.raises(mf.ManifestError, match="an amortized one carries none$"):
  ```
  (The `$` pins that nothing follows the plain sentence.) Run `python -m pytest tests/test_artifact_store.py::test_a_posterior_manifests_amortized_flag_must_agree_with_its_region -q` → FAIL (the message still ends "(defect D3)"). Rewrite message 7 as fixed above. Run again → `1 passed`.

- [ ] **Step 5: Clean every comment and docstring hit** in the 14 files by the treatment above, reading each file in full. Re-run the Step 3 test until it passes.

- [ ] **Step 6: Fix the wrong facts** listed above.

- [ ] **Step 7: The scan.** `python -m pytest tests/test_source_hygiene.py -q` → `6 passed`.

- [ ] **Step 8: The comment-only check.** From the repo root:
  `python C:\Users\J\AppData\Local\Temp\prism-piece6\comment_only_check.py --base HEAD --allow C:\Users\J\AppData\Local\Temp\prism-piece6\allow-task-03.txt core/artifacts/identity.py core/artifacts/manifest.py core/artifacts/report.py core/artifacts/store.py core/cli.py core/config.py core/logging_root.py core/refusals.py core/rng.py core/runs.py core/sim_config.py core/Helpers/file_manager.py core/Helpers/model_store.py core/Helpers/visualizers.py tests/test_artifact_store.py tests/test_source_hygiene.py`
  Expected: 2 string differences (message 7, the `match=` pin) plus 14 removed `NOT_YET_CLEAN` entries, every line `[allowed: …]`, last line `16 file(s), 16 difference(s), 0 not allowed`; exit 0. Paste the allow file and the output into the report.

- [ ] **Step 9: Focused tests.**
  `python -m pytest tests/test_source_hygiene.py tests/test_refusals.py -q` → all passed
  `python -m pytest tests/test_artifact_store.py::test_a_posterior_manifests_amortized_flag_must_agree_with_its_region tests/test_artifact_store.py::test_the_source_scans_cover_every_code_directory tests/test_nav_and_gating.py::test_the_chi_drive_and_band_are_read_only_and_the_draft_carries_config -q` → `3 passed`
  `python -m pytest tests/test_artifact_store.py -q -k "manifest or literal or public_entry or store"` → all selected passed (the keyword `store` also matches the module name, so this runs the whole file, tiny real run included: start it in the background with output to `C:\Users\J\AppData\Local\Temp\prism-piece6\task3-store.log` and edit nothing until it exits)

- [ ] **Step 10: Commit.**
  `git add core/artifacts/identity.py core/artifacts/manifest.py core/artifacts/report.py core/artifacts/store.py core/cli.py core/config.py core/logging_root.py core/refusals.py core/rng.py core/runs.py core/sim_config.py core/Helpers/file_manager.py core/Helpers/model_store.py core/Helpers/visualizers.py tests/test_artifact_store.py tests/test_source_hygiene.py`
  `git commit -m "Clean working-document references out of the store and config" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"`
  The controller starts the one-process fast gate (`python -m pytest -m "not slow" -q`, background, logged) at the same moment as this task's review. Do not run it yourself.

---

### Task 4: Cleanup: `core/FDT/`

**Files:**
- Modify: every `core/FDT/` entry of `NOT_YET_CLEAN` at the start of the task. Trust the list in the file. The reconnaissance expects 7 files with 171 hit lines: `core/FDT/campaigns.py` (2), `core/FDT/compare.py` (44), `core/FDT/cross_validation.py` (64), `core/FDT/cross_validation_plots.py` (1), `core/FDT/fdt_pipeline.py` (53), `core/FDT/plots.py` (3), `core/FDT/sanity.py` (4)
- Modify: `tests/test_source_hygiene.py` (the `NOT_YET_CLEAN` literal only: its `core/FDT/` entries are removed)
- Test: `tests/test_source_hygiene.py`, `tests/test_fdt_user.py`, `tests/test_fdt_compare.py`, `tests/test_refusals.py`

**Interfaces:**
- Consumes (Task 1): in `tests/test_source_hygiene.py`, `NOT_YET_CLEAN: frozenset[str]` (repo-relative posix paths), `test_no_file_outside_the_pending_list_cites_a_working_document`, `test_every_pending_file_still_has_a_hit`, `test_every_allowlist_entry_still_matches_something` (a failure prints `path:line family 'match'`). Also the controller tool `C:\Users\J\AppData\Local\Temp\prism-piece6\comment_only_check.py --base <git-ref> --allow <allow-file> <path> [<path> ...]`. Its allow file holds one permitted non-docstring difference per line, `<repo-relative-path>|<kind>|<quoted text fragment>`, with kind ∈ {`message`, `string`, `rename`, `assert`}. It exits 0 when every difference is allowed; otherwise it exits 1 and prints `path:line: <old> -> <new>` for each difference.
- Produces: nothing new. `NOT_YET_CLEAN` loses every `core/FDT/` entry.

**Binding (read before starting):** Read spec §2.1–§2.7 (`docs/superpowers/specs/2026-09-25-documentation-and-retrain-readiness-design.md`). `python` below means `C:\Users\J\anaconda3\envs\biophys-env\python.exe`. The root conftest sets the environment variables for pytest.
- **The rule.** No comment, docstring or string in these files may cite a working document or one of its labels. That covers `PRISM_HANDOFF.md`/"the handoff", `docs/STATE.md`, `CLAUDE.md`, the specs, the plans and the ledgers. The labels include: piece numbers ("piece 5", "PIECE 5", "pre-piece-5", "whole-piece"); "§N", "spec 4.1", "Sec. 2.3"; decision ids (E1, E2, E4, E5, E7, E8, E9, V1, V4, D1); ruling ids (P49, P78, F10, R-F6, "ruling", "the owner ruled"); review ids ("the whole-piece review's N11", M1, M2, "fix round", "the review of Task 16"); task ids ("Task 17", T12, T25, T36); "Appendix A". New text keeps the same rule, including the unlabelled forms the scan cannot see ("the spec names", "the piece removes", "the plan's table", "the design"). This is the manual pass of spec §2.2: read each file in full, not only its matched lines.
- **Treatment (spec §2.3).**
  - A provenance tag is deleted, e.g. "(piece 5, spec §7; decision E8)", "(the whole-piece review's N11)", "(Task 17, fix round 1)", "(P49)".
  - A reason given by reference is rewritten to state the reason.
  - To learn what a label meant, look it up in the working record, then write the behaviour, never the label. E-ids are in `docs/superpowers/specs/2026-09-22-secondary-analyses-design.md` §1.1, V-ids in `docs/superpowers/specs/2026-09-15-validation-and-logging-design.md`, D-ids in `docs/superpowers/specs/2026-09-11-one-flow-design.md`, and P/F/N/M/R-F ids in `.superpowers/sdd/2026-09-22-secondary-analyses/progress.md`.
  - Words for the labels common here:
    - **E2**: an interrupted run keeps its record's folder, marked unfinished ("E2 keeps the folder" becomes "a cancel or a crash keeps the folder, marked unfinished").
    - **E4**: nothing measurable is a refusal; failed operating points are counted in a completed record.
    - **E5**: a setting too thin to trust warns, and the warning is kept in the record's notices.
    - **E7**: every run records the seed it used, drawing one when none is given; repeats use different seeds.
    - **E9**: a frequency the spontaneous spectrum does not resolve comes back blank and counted, never interpolated.
    - **V1**: a public stage works on a private copy of the configuration it is handed.
    - **V4**: messages are logging records at info, warning or error.
- **Ids collide (spec §2.3).** Map a label by its nearby words ("defect", "trap", "the whole-piece review's", "row", "piece N"), never by the id alone.
  - `fdt_pipeline.py:252` "(D1 retired the CLI)" means the retirement of the old prompt-driven command line. It is not a defect, a walkthrough row or the kurtosis feature.
  - `compare.py:579` "(A1)" is a label, not the summary feature. Delete it.
  - Short feature ids (A1…G7) are allowed only in `core/SBI/statistics.py` and `core/SBI/summaries.py`. Never write one here; use the full label, e.g. `A1_mean`.
- **Keep (spec §2.2 allowlist).** An allowlist entry that stops matching fails `test_every_allowlist_entry_still_matches_something`, so these stay:
  - the physics and norm names allowed everywhere: `F0`, `D0` (`campaigns.py:167-173`, "D0 <= 0", is the noise constant) and `L2`;
  - scipy's `S2` in `core/FDT/spectral.py`, which is not on the list: do not touch it;
  - "Phase A" and "Phase B", which have no digit.
- **Dates (spec §2.6).** A date that dates a measurement stays. The scan does not enforce dates.
- **Facts (spec §2.5).** No §2.5 item lives in `core/FDT/`. Fix a wrong fact only where a rewrite meets it.
  - Do NOT touch `core/FDT/sanity.py`'s `x_steady = sol[0, 0, :, burn_idx:]` (about `:291`, `:328`). It is an open engineering item recorded elsewhere, and this task changes no code.
- **Only comments and docstrings change (spec §2.7).**
  - No §2.4 message lives here, and the reconnaissance found no plain-string hit in `core/FDT/`, so the permitted non-docstring change list starts EMPTY.
  - If the manual pass finds a user-visible string that carries a label, add it to the list as kind `string` before editing it, and name it in the report.
- **Raw-source tests over these files (spec §2.7).** A rewrite must not add or remove the words they look for:
  - `tests/test_fdt_user.py::test_the_band_refusal_offers_a_longer_recording_only_where_it_lowers_the_resolution`: `inspect.getsource(campaigns.run_campaign1_psd)` must contain "WELCH_NPERSEG_CAP" and must NOT contain "2 ** 14".
  - `tests/test_fdt_user.py::test_a_thin_setting_warns_and_hands_back_the_sentence_for_the_record`: the raw source of `fdt_pipeline.run_fdt` and of `cross_validation.run_param_study_cli` must contain "warn_thin_settings".
  - The same test requires the raw source of `cross_validation.run_fdt_param_sweep` to NOT contain "warn_thin_settings" anywhere, comments and docstring included. That function runs from `def run_fdt_param_sweep(`, about line 286, to about line 594, and most of this file's hits are inside it. Write "the study warns once, at its top" there, never the function name.
  - `tests/test_refusals.py::test_every_domain_error_is_a_refusal_and_carries_a_field`: `FDTModelError` (`class FDTModelError(Refusal):`, `campaigns.py:31`) keeps a non-empty docstring.
  - The rotation-reader and `input()` scans in `tests/test_conditioning_repair.py` read code tokens and AST calls only, so they are unaffected.
- Never edit a source file while a pytest run is in progress, and never run two pytest processes at once.
- **Git.** Work on local `main`. Make one commit and never amend it. `git add` every changed file explicitly. The commit body is `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

- [ ] **Step 1: List the task's files.** Read the `NOT_YET_CLEAN` literal in `tests/test_source_hygiene.py`. Copy every entry that starts with `core/FDT/` into the task report. Expected: the seven files above.

- [ ] **Step 2: Shrink the pending list first, and see the scan fail.** Delete exactly those entries from `NOT_YET_CLEAN`, then run:
  `python -m pytest tests/test_source_hygiene.py::test_no_file_outside_the_pending_list_cites_a_working_document -q > C:\Users\J\AppData\Local\Temp\prism-piece6\task4-hits.txt`
  Expected: FAIL. The message lists `core/FDT/<file>:<line> <family> '<match>'` lines, about 171 of them, and names no file outside `core/FDT/`. This file is the work list.

- [ ] **Step 3: Write the permitted-change list before editing.** Create `C:\Users\J\AppData\Local\Temp\prism-piece6\allow-task-04.txt`. It is empty, per the Binding. Record "permitted non-docstring changes: none" in the task report.

- [ ] **Step 4: Clean the files one at a time, reading each in full.** For every hit, delete the tag or rewrite the sentence in the Binding's words. Examples:
  - `compare.py:1` `"""Comparing saved FDT records (piece 5, spec §7; decision E8).` becomes `"""Comparing saved FDT records.`
  - `fdt_pipeline.py:252-254` becomes `REQUIRED: there is no prompt to fall back to (the interactive prompts were retired), and the old None default meant an input() that a GUI worker thread could never answer.`
  - `fdt_pipeline.py:284-286` becomes `# 0. The settings too thin to trust: warned now and KEPT in body.notices. HERE, in the decorated function itself: the thin-settings test reads inspect.getsource(fdt_pipeline.run_fdt), which is this function's own source (public_entry uses functools.wraps), not _measure's.`
  - `cross_validation.py:357-359` "E5: the sentences for body.notices, and NO warning. ... (Task 12's review)." becomes "The too-thin sentences for body.notices, and NO warning: the study warns once, at its top, for the whole spend." This sentence is inside `run_fdt_param_sweep`, so it must not name the function (see the raw-source rule).

  Result: no line of `task4-hits.txt` survives in the edited file.

- [ ] **Step 5: Prove only comments and docstrings changed.** Run (PowerShell):
  ```powershell
  $files = git diff --name-only -- 'core/FDT/*.py'
  python C:\Users\J\AppData\Local\Temp\prism-piece6\comment_only_check.py --base HEAD --allow C:\Users\J\AppData\Local\Temp\prism-piece6\allow-task-04.txt @files
  ```
  Expected: `$LASTEXITCODE` is 0 and no difference is printed. Undo any printed difference. The only exception is a string added to the list under the Binding's rule; then re-run.

- [ ] **Step 6: Run the scan suite.** `python -m pytest tests/test_source_hygiene.py -q` Expected: all pass, including `test_no_file_outside_the_pending_list_cites_a_working_document`, `test_every_pending_file_still_has_a_hit` and `test_every_allowlist_entry_still_matches_something`.

- [ ] **Step 7: Run the raw-source pins.**
  `python -m pytest "tests/test_fdt_user.py::test_the_band_refusal_offers_a_longer_recording_only_where_it_lowers_the_resolution" "tests/test_fdt_user.py::test_a_thin_setting_warns_and_hands_back_the_sentence_for_the_record" "tests/test_refusals.py::test_every_domain_error_is_a_refusal_and_carries_a_field" -q`
  Expected: `3 passed`.

- [ ] **Step 8: Run the focused suites for the touched files.** `python -m pytest tests/test_fdt_user.py tests/test_fdt_compare.py -m "not slow" -q` Expected: 0 failed. If the run could exceed the tool's ten-minute limit, start it in the background with output to `C:\Users\J\AppData\Local\Temp\prism-piece6\task4-focused.log`, and edit nothing until it exits.

- [ ] **Step 9: Commit.** Add exactly the files `git status` shows changed:
  ```powershell
  git add tests/test_source_hygiene.py core/FDT/campaigns.py core/FDT/compare.py core/FDT/cross_validation.py core/FDT/cross_validation_plots.py core/FDT/fdt_pipeline.py core/FDT/plots.py core/FDT/sanity.py
  git commit -m "cleanup: core/FDT states its reasons in words" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
  ```
  The controller then starts the task's review and, at the same moment, the one-process fast gate `python -m pytest -m "not slow" -q` in the background with output to a log. The implementer does not run the gate. The review re-runs Step 5's checker and reads every rewritten sentence at a collision id (D1, A1, E-ids) against its context.

---

### Task 5: Cleanup: the window (`core/gui/`)

**Files:**
- Modify: every `core/gui/` entry of `NOT_YET_CLEAN` at the start of the task. Trust the list in the file. The reconnaissance expects 35 files with 302 hit lines:
  - `core/gui/`: `app.py`, `app_icon.py`, `fields.py`, `main_window.py`, `plot_watcher.py`, `settings.py`, `streams.py`, `worker.py`
  - `core/gui/panels/`: `base_panel.py`, `crossval_panel.py`, `fdt_panel.py`, `inference_tabs.py`, `record_view.py`, `reduction_panel.py`, `simulate_panel.py`, `simulate_runner.py`
  - `core/gui/panels/inference/`: `base.py`, `config_tab.py`, `infer_tab.py`, `posterior_tab.py`, `prior_tab.py`, `rows.py`, `tsnpe_tab.py`, `validate_tab.py`
  - `core/gui/screens/`: `artifact_screen.py` (67 lines), `home_screen.py`, `inference_screen.py`, `model_builder_screen.py`, `nav_shell.py`, `section_screen.py`
  - `core/gui/widgets/`: `artifact_picker.py`, `artifact_table.py`, `compare_list.py`, `labeled_inputs.py`, `refusal_box.py`
- Modify: `core/gui/panels/inference/help_text.py` (`"flow_patience":`, about `:85-86`). It is not on the list: the date is not a scan hit. This is the §2.5/§2.6 rewording.
- Modify: `tests/test_source_hygiene.py` (the `NOT_YET_CLEAN` literal only: its `core/gui/` entries are removed)
- Test: `tests/test_source_hygiene.py`, `tests/test_nav_and_gating.py`, `tests/test_settings_persistence.py`, `tests/test_artifact_browser.py`

**Interfaces:**
- Consumes (Task 1): `NOT_YET_CLEAN`, `test_no_file_outside_the_pending_list_cites_a_working_document`, `test_every_pending_file_still_has_a_hit`, `test_every_allowlist_entry_still_matches_something` (failure lines `path:line family 'match'`). Also `C:\Users\J\AppData\Local\Temp\prism-piece6\comment_only_check.py --base <git-ref> --allow <allow-file> <path> [<path> ...]`, whose allow file has one line per permitted difference, `<repo-relative-path>|<kind>|<quoted text fragment>`, kind ∈ {`message`, `string`, `rename`, `assert`}. It exits 0 when every difference is allowed.
- Produces: nothing new. `NOT_YET_CLEAN` loses every `core/gui/` entry. The module docstring of `core/gui/fields.py` names "the refusal-naming rule", the same phrase Task 6 gives `core/tool/fields.py`.

**Binding (read before starting):** Read spec §2.1–§2.7. `python` means `C:\Users\J\anaconda3\envs\biophys-env\python.exe`.
- **The rule (H4).** No comment, docstring or string here may cite the handoff, `docs/STATE.md`, `CLAUDE.md`, the specs, the plans, the ledgers, the walkthrough checklist, or their labels. The labels include:
  - "piece 4", "whole-piece", "this piece";
  - "§N", "spec §5.4", "design §3.1", "plan ruling";
  - B-ids, V-ids, E-ids and D-ids;
  - R/N/M/P/F/T ids, "fix round", "Review Focus 3", "Backlog", "Task 40", "the whole-piece review's";
  - "guardrail N", "trap X5", "M1b", "C-3";
  - "walkthrough", "row 1".

  New text keeps the same rule, including unlabelled forms such as "the spec names", "the design" and "the piece". This is the manual pass: read every file in full.
- **Treatment and words (spec §2.3).** Delete a provenance tag. Rewrite a reason given by reference in these words:
  - "guardrail 2": the narrowed-model rule. A narrowed posterior is marked as such and never loads or infers as a broad one.
  - "guardrail 8": the calibrate-on-the-region rule. A narrowed posterior calibrates on the prior restricted to its region.
  - "trap X5": the calibration's operating-point count is t_scale's effective sample size.
  - "the handoff's trap M1b": replacing the session's configuration versus installing into it.
  - "Backlog C-3:": delete the prefix; the sentence already says what it means.
  - D7: the near-miss consent. A cache one setting away is refused before any simulation unless the operator asks for a new run.
  - D8: the tool refuses a non-amortized load by default; the Infer tab's box is the window's way to accept another observation.
  - D11: there is no chi override. `build_prior` refuses any band or amplitude but config.py's.
  - "part of the identity (D3)" at `tsnpe_tab.py:182`: the region belongs to the simulation cache's identity (delete the tag; the sentence says it).
  - V2: a bad input is refused at the click, and a blank box is a refusal, never a zero.
  - V3: the refusal-naming rule. A refusal names its field by a key, and each front end names the control or flag that answers it.
  - V4: messages are logging records.
  - V5: selections and the training budget are remembered; science knobs open at config.py on every launch.
  - B1: the Artifacts screen is a plain QWidget, not a panel, so a run elsewhere cannot grey it out.
  - B11: the live run is shown app-wide.
  - B12: Apply confirms before discarding a session.
  - B13: a narrowed posterior is marked in the picker text.
  - B15: the rescue save runs inside a deferred-cancel section.
  - B16: a new probe row's frequency box starts blank.
  - E2: an interrupted run keeps its folder, marked unfinished.
  - E6: an input that appears in several places lists them all.
  - E7: every run records its seed.

  For other ids, look them up (B-ids in `docs/superpowers/specs/2026-09-17-gui-usability-and-artifact-browser-design.md` §1.1, V-ids in `2026-09-15-validation-and-logging-design.md`, E-ids in `2026-09-22-secondary-analyses-design.md`, D-ids in `2026-09-11-one-flow-design.md`) and write the behaviour. Map a collision id by its nearby words, never the id alone. `fdt_panel.py:7`'s "D1 deleted the prompts" is the retirement of the prompt-driven command line.
- **The §2.5/§2.6 facts in this task:**
  - `help_text.py:85-86` `HELP["flow_patience"]` becomes exactly `"Early-stopping patience in epochs. One measured run stopped at epoch 130 on a patience of 20, with its best validation loss at epoch 110."` It keeps the measurement and drops the incident date. No test pins this text.
  - `tsnpe_tab.py:32-33` "(``default_x`` is None on posterior_08232026)" becomes "(its ``default_x`` is None)". That artifact was deleted.
  - `config_tab.py:136` "NOT PERSISTED, ON PURPOSE, and it is the only field on this tab that is not." is false, because the chi probe count, slots and lock-in ceiling are not persisted either. It becomes "NOT PERSISTED, ON PURPOSE, like every science knob on this tab: the chi probe count, slots and lock-in ceiling open at config.py on every launch too."
  - `rows.py:34` "See orchestrator.build_experiment_obs_chi" becomes "See core.SBI.observations.build_experiment_obs_chi".
  - `inference_screen.py:153-156` and `:172-176` ("the handoff's trap M1b") are rewritten with "replacing the session's configuration versus installing into it".
  - Dates that date a measurement stay, e.g. `help_text.py:48` "measured 2026-08-27" and `app_icon.py:57` "measured on Windows 11 26200, 2026-09-11". Drop only "the display walkthrough's row 1".
- **Only comments and docstrings change, plus one string (spec §2.7).** The permitted non-docstring change list is exactly one line:

  `core/gui/panels/inference/help_text.py|string|The 2026-08-25 run stopped at 130`

  The reconnaissance found no other plain-string hit in `core/gui/`. If the manual pass finds one, add it as kind `string` before editing it and name it in the report.
- **Raw-source and docstring tests over these files.** A rewrite must not add or remove the words they look for.
  - `tests/test_nav_and_gating.py::test_the_panel_docstrings_no_longer_count_nine_panels_or_five_tabs` reads the whitespace-normalised full source of `core/gui/panels/base_panel.py` and `core/gui/panels/inference/base.py`, comments and docstrings included. In `base_panel.py`:
    - these must stay ABSENT: "Nine of these", "8 of the 9", "all nine differ", "nine independent splitters", "the five inference tabs";
    - these must stay PRESENT: "Ten of these exist" (`:91`), "9 of the 10" (`:100`, `:357`), "all ten differ" (`:364`).
  - The same test requires `inference/base.py` to keep "the five inference tabs" ABSENT and "the six inference tabs" (`:15`) PRESENT.
  - `tests/test_nav_and_gating.py::test_main_window_stops_claiming_it_owns_the_only_settings_write` reads the `MainWindow` class source. It must NOT contain "the only QSettings WRITE site" and MUST contain "_persist_layout" and "1500". The `:80` hit sits in the same docstring as `:81-82`; keep that sentence.
  - `tests/test_nav_and_gating.py::test_the_tsnpe_new_run_box_is_not_persisted` unparses `TSNPEPanel.save_settings` and `restore_settings`. The AST keeps docstrings, and `save_settings`' docstring at `tsnpe_tab.py:213` is a hit. Neither may contain "new_run".
  - `tests/test_nav_and_gating.py::test_the_infer_tab_other_observation_box` requires `InferPanel.save_settings` and `restore_settings` to not contain "other_obs".
  - The Help screen (`core/gui/screens/settings_screen.py` `_first_paragraph`) shows the first docstring paragraph of `reduction_panel`, `fdt_panel`, `crossval_panel`, `inference_tabs` and `simulate_panel`, and those modules' `HELP` dicts. None of those first paragraphs carries a hit: leave them as they are.
  - `tests/test_nav_and_gating.py::test_the_chi_drive_and_band_are_read_only_and_the_draft_carries_config` needs `HELP["chi_f0"]` and `HELP["chi_range"]` to contain "config.py" and "measurement". Do not touch them.
  - Keep `nav_shell.py:18` "U+E000..U+E004". It is an allowed shape, and its allowlist entry must keep matching.
  - The `code_only`/AST scans over GUI code (e.g. `InferPanel._plan_chi_probes`, `ArtifactPicker.refresh`) read code only and are unaffected.
- Never edit a source file while a pytest run is in progress, and never run two pytest processes at once.
- **Git.** One commit on `main`, never amended. `git add` each file. Body `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

- [ ] **Step 1: List the task's files.** Copy every `core/gui/` entry of `NOT_YET_CLEAN` into the task report. Expected: the 35 files above.

- [ ] **Step 2: Shrink the pending list and see the scan fail.** Delete exactly those entries, then run:
  `python -m pytest tests/test_source_hygiene.py::test_no_file_outside_the_pending_list_cites_a_working_document -q > C:\Users\J\AppData\Local\Temp\prism-piece6\task5-hits.txt`
  Expected: FAIL, listing about 302 `core/gui/...:<line> <family> '<match>'` lines and nothing outside `core/gui/`.

- [ ] **Step 3: Write the permitted-change list before editing.** Write the one allow line from the Binding into `C:\Users\J\AppData\Local\Temp\prism-piece6\allow-task-05.txt`, and copy it into the task report.

- [ ] **Step 4: Make the §2.5/§2.6 edits.** Apply the six Binding edits exactly: `help_text.py` flow_patience, `tsnpe_tab.py:32-33`, `config_tab.py:136`, `rows.py:34`, and the two `inference_screen.py` sentences. For `inference_screen.py:175-176`, write e.g. "That asymmetry is the difference between replacing the session's configuration and installing into it: its destructive twin ``new_draft`` is the one the Config tab confirms, and this one is never confirmed at all."

- [ ] **Step 5: Clean the rest, file by file, reading each in full.** Examples:
  - `fields.py:1` becomes `"""The window's half of the refusal-naming rule: which control answers each refusal field.`
  - `infer_tab.py:284` "Backlog C-3: say what is in band ..." becomes "Say what is in band ...".
  - `infer_tab.py:488` "must never silence guardrail 2 for the new one" becomes "must never silence the narrowed-model rule for the new one: a narrowed posterior never infers on another observation unasked".
  - `validate_tab.py:66` "-- guardrail 8)" becomes "-- the calibrate-on-the-region rule)".
  - `fields.py:203` "which is what keeps the sentences the walkthrough quotes byte-identical" becomes "which keeps the sentences byte-identical wherever they are quoted".
  - `streams.py:24` keeps its `noqa` directive and loses only "(spec §1.2)".

  Result: `task5-hits.txt` has no surviving line.

- [ ] **Step 6: Prove only comments, docstrings and the one string changed.**
  ```powershell
  $files = git diff --name-only -- 'core/gui/*.py'
  python C:\Users\J\AppData\Local\Temp\prism-piece6\comment_only_check.py --base HEAD --allow C:\Users\J\AppData\Local\Temp\prism-piece6\allow-task-05.txt @files
  ```
  Expected: `$LASTEXITCODE` 0. The only difference printed is the `flow_patience` string, which is allowed.

- [ ] **Step 7: Run the scan suite.** `python -m pytest tests/test_source_hygiene.py -q` Expected: all pass, including `test_every_allowlist_entry_still_matches_something`.

- [ ] **Step 8: Run the raw-source pins.**
  `python -m pytest "tests/test_nav_and_gating.py::test_the_panel_docstrings_no_longer_count_nine_panels_or_five_tabs" "tests/test_nav_and_gating.py::test_main_window_stops_claiming_it_owns_the_only_settings_write" "tests/test_nav_and_gating.py::test_the_tsnpe_new_run_box_is_not_persisted" "tests/test_nav_and_gating.py::test_the_infer_tab_other_observation_box" "tests/test_nav_and_gating.py::test_the_chi_drive_and_band_are_read_only_and_the_draft_carries_config" -q`
  Expected: `5 passed`.

- [ ] **Step 9: Run the focused suites for the touched files.** These take several minutes. Start the run in the background with output to `C:\Users\J\AppData\Local\Temp\prism-piece6\task5-focused.log`, and edit nothing until it exits:
  `python -m pytest tests/test_nav_and_gating.py tests/test_settings_persistence.py tests/test_artifact_browser.py -m "not slow" -q`
  Expected: 0 failed.

- [ ] **Step 10: Commit.** Add every changed file explicitly: `tests/test_source_hygiene.py`, `core/gui/panels/inference/help_text.py`, and each cleaned `core/gui/` file from Step 1 that `git status` shows changed.
  ```powershell
  git add tests/test_source_hygiene.py core/gui/panels/inference/help_text.py <each cleaned core/gui file>
  git commit -m "cleanup: core/gui states its reasons in words" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
  ```
  The controller then starts the review and, at the same moment, the one-process fast gate `python -m pytest -m "not slow" -q` in the background with output to a log. The implementer does not run the gate. The review re-runs Step 6's checker and reads every rewritten sentence at a collision id against its context.

---

### Task 6: Cleanup: the tool, the diagnostics, the reduction map, the root conftest and the prose files

**Files:**
- Modify: every entry of `NOT_YET_CLEAN` under `core/tool/`, `core/diagnostics/` or `core/Reduction/`, plus `conftest.py`, `README.md`, `requirements.txt`, `pytest.ini`, `run.bat` and `run.sh` when listed. Trust the list. The reconnaissance expects:
  - `core/tool/`: `__init__.py`, `browse.py` (47 lines), `config_args.py`, `diagnostics.py`, `fdt.py`, `fields.py`, `logging_console.py`, `smoke.py`, `stages.py`, 101 lines in all;
  - `core/diagnostics/`: `__init__.py`, `ablation.py`, `feature_sets.py`, `identifiability.py`, `rng.py`, `sbc.py`, 29 lines;
  - `conftest.py` (`:1`) and `requirements.txt` (`:9`, `:44`).

  Nothing is expected under `core/Reduction/`: its only label-shaped tokens are "Phase B1/B2", allowed by shape. `README.md` has only the allowed "M1 / M2 / M3 / M4" line. `pytest.ini`, `run.bat` and `run.sh` are clean. Do not modify a file that is not listed.
- Modify: `core/tool/config_args.py` (message 10, `f"driven at -- write --forced PATH@HZ. Missing for: {bare}. (D9)"`, `:248`), `core/diagnostics/identifiability.py` (message 9, `f"chi.fisher_features and the gen_chi_raw unpack (trap CHI10)."`, `:597`)
- Modify: `tests/test_tool.py` (`assert "(D9)" in str(exc.value), "spec 3.6 requires the guardrail id in the message"`, `:376`, in `test_parse_forced_and_recording_set_rules`). Only this assertion changes; `test_tool.py` stays on `NOT_YET_CLEAN` for the tests cleanup.
- Modify: `tests/test_source_hygiene.py` (the `NOT_YET_CLEAN` literal only)
- Test: `tests/test_source_hygiene.py`, `tests/test_tool.py`, `tests/test_diagnostics.py`, `tests/test_refusals.py`

**Interfaces:**
- Consumes (Task 1): `NOT_YET_CLEAN`, `test_no_file_outside_the_pending_list_cites_a_working_document`, `test_every_pending_file_still_has_a_hit`, `test_every_allowlist_entry_still_matches_something`. Also `C:\Users\J\AppData\Local\Temp\prism-piece6\comment_only_check.py --base <git-ref> --allow <allow-file> <path> [<path> ...]`, whose allow lines are `<repo-relative-path>|<kind>|<quoted text fragment>`, kind ∈ {`message`, `string`, `rename`, `assert`}.
- Produces: message 10's plain sentence, pinned by `test_parse_forced_and_recording_set_rules`, and message 9's plain sentence. `core/tool/fields.py`'s docstring names "the refusal-naming rule", the same phrase Task 5 gives `core/gui/fields.py`. `NOT_YET_CLEAN` loses this task's entries.

**Binding (read before starting):** Read spec §2.1–§2.7 and §2.9 item (6). `python` means `C:\Users\J\anaconda3\envs\biophys-env\python.exe`.
- **The rule (H4).** Nothing here may cite the handoff (`PRISM_HANDOFF.md`), `docs/STATE.md`, `CLAUDE.md` (`config_args.py:8` does), the specs, the plans, the ledgers or their labels:
  - "piece 3", "Piece 5", "piece-5 spec", "whole-piece";
  - "§N", "spec Sec 4.5", "design §4";
  - D/V/B/E ids, "one-flow design";
  - R/N/M/P/F/T/K/I/S ids, "fix round 1", "IMPORTANT 3", "probed by three reviewers", "Task 29";
  - "trap X5", "trap CHI10", "guardrail 8".

  New text keeps the same rule, including unlabelled forms. This is the manual pass: read each file in full. The `requirements.txt` and `conftest.py` lines are read whole, like every prose line.
- **Words (spec §2.3).** Delete a provenance tag, e.g. "(piece 2)", "(fix round 1, K9)", "(spec §1.2)", "(E11, spec §6.1)". Rewrite a reason given by reference:
  - "the D6 trap" (`identifiability.py:70`) and "the silently-ignored flag D6 forbids" (`stages.py:186`) are the one-flow decision D6, not the transposed rotation. The words are: a setting is refused, never silently clamped or ignored.
  - "D7's consent pair" (`config_args.py:61`): the near-miss consent, where a cache one setting away is refused before any simulation unless `--new-run` is given.
  - "D8:" (`config_args.py:196`): the tool refuses a non-amortized load by default. Keep the rest of that docstring as it is; its claim about calibrations is true until a later task changes the behaviour.
  - "guardrail 8" (`sbc.py:6`, `:93`, `:141`): the calibrate-on-the-region rule, where a narrowed posterior calibrates on the prior restricted to its region.
  - "(trap X5)" (`smoke.py:39`): delete it; the sentence already states the reason.
  - V3: the refusal-naming rule. `core/tool/fields.py:1` becomes `"""The command-line tool's half of the refusal-naming rule: which flag answers each refusal field.`
  - V4: messages are logging records.
  - E2: an interrupted run keeps its folder, marked unfinished.
  - E11: `fdt` and `crossval` take `--store-root` too.
  - B6: no `--force`; dependents are read before anything is deleted.
  - R3 (`browse.py:386`): the store's recency guard, where a directory touched within `store.RECENT_WRITE_SECONDS` may be a run in flight.

  Look other ids up in the one-flow, validation-and-logging, gui-usability and secondary-analyses specs under `docs/superpowers/specs/` or the ledgers, and write the behaviour.
- **Messages (spec §2.4).**
  - Message 10 (`UsageError`) loses " (D9)" and keeps the rest verbatim: `f"chi mode: every driven recording must state the frequency (Hz) it was driven at -- write --forced PATH@HZ. Missing for: {bare}."`.
  - Message 9 (`ValueError`) loses " (trap CHI10)" and keeps the rest verbatim, ending `"... Check what is being passed to chi.fisher_features and the gen_chi_raw unpack."`. `tests/test_diagnostics.py::test_the_probe_budget_refuses_a_miswired_cos_sin_pair` pins `match=r"cos\^2 \+ sin\^2"`, so "cos^2 + sin^2" stays byte-identical.
- **The reversed pin (spec §2.4).** The test pins a distinctive phrase of the plain sentence, and an absence regex. It must never contain a literal like "(D9)", which the scan's id family would flag, and its failure message must not cite a spec.
- **Facts (spec §2.5).**
  - `config_args.py:94` "the environment-free ``_common.script_cfg``" names a deleted helper. It becomes "``(cfg, ignored)`` from the shared flags and nothing else: no environment variable reaches the config."
  - `smoke.py:6` "sidecar round-trip" is a wrong fact in the same docstring as the `:39` hit: the rotation travels in the manifest now. It becomes "the rotation's round trip through the manifest".
  - `requirements.txt:8-9` becomes "... (2) the calibration numbers (0.26's TARP changes make them non-comparable to numbers recorded before the upgrade)." and drops the handoff sentence.
  - `requirements.txt:44` becomes `# Test runner. Fast gate: pytest -m "not slow"; full: pytest.`
  - `conftest.py:1` becomes `"""Root pytest configuration.`
  - `logging_console.py:13` keeps `# noqa: F401 -- sets the ``core`` logger to INFO at import` and loses only "(spec §1.2)". Only the explanation after a `noqa` directive changes.
- **Permitted non-docstring changes (spec §2.7).** Write these to the allow file before editing:
  ```
  core/tool/config_args.py|message|(D9)
  core/diagnostics/identifiability.py|message|(trap CHI10)
  tests/test_tool.py|assert|spec 3.6 requires the guardrail id in the message
  tests/test_tool.py|assert|import re
  tests/test_tool.py|assert|msg = str(exc.value)
  tests/test_tool.py|assert|must state the frequency (Hz) it was driven at
  tests/test_tool.py|assert|re.search
  ```
  Nothing else may differ. Help strings, epilogs and parser descriptions carry no hits and are the help-text tasks' job. Do not touch them.
- **Raw-source and docstring tests over these files.** A rewrite must not add or remove the words they look for:
  - `tests/test_diagnostics.py::test_the_diagnostic_guards_refuse_naming_no_flag_or_file`: the raw source of `feature_sets.assert_not_chi` (`:72`), `assert_nadrowski` (`:92`) and `assert_forced` (`:106`) must not contain "SystemExit" or "raise ValueError". The module docstring's "from ``SystemExit`` to a ``Refusal``" (`:5`) lies outside them. Rewrite it in place, and never move that history into a guard.
  - `tests/test_tool.py::test_the_fdt_subcommands_no_longer_say_they_have_no_bounds_file`: `inspect.getsource(core.tool.fdt)` (the whole module) must not contain "no bounds file"; `core.tool.fdt.__doc__` must keep "resolve_bounds_for_cell" (`:11`); `inspect.getdoc(fdt.model_for_cell)` must keep "resolve".
  - `tests/test_refusals.py::test_every_domain_error_is_a_refusal_and_carries_a_field`: `UsageError` (`config_args.py`) keeps a non-empty docstring.
  - The `code_only`/AST scans over `tool.main`, `logging_console`, `sbc`, `identifiability` and `ablation` read code only and are unaffected.
- Never edit a source file while a pytest run is in progress, and never run two pytest processes at once.
- **Git.** One commit on `main`, never amended. `git add` each file. Body `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

- [ ] **Step 1: List the task's files.** Copy into the task report every `NOT_YET_CLEAN` entry under `core/tool/`, `core/diagnostics/` or `core/Reduction/`, and any of `conftest.py`, `README.md`, `requirements.txt`, `pytest.ini`, `run.bat`, `run.sh`. Expected: the 15 core files, `conftest.py` and `requirements.txt`. Report it if `README.md` or a `core/Reduction/` file appears; clean it under the same rules.

- [ ] **Step 2: Shrink the pending list and see the scan fail.** Delete exactly those entries, keeping `tests/test_tool.py`, then run:
  `python -m pytest tests/test_source_hygiene.py::test_no_file_outside_the_pending_list_cites_a_working_document -q > C:\Users\J\AppData\Local\Temp\prism-piece6\task6-hits.txt`
  Expected: FAIL, listing about 133 lines across those files and nothing else.

- [ ] **Step 3: Write the permitted-change list before editing.** Save the Binding's seven lines to `C:\Users\J\AppData\Local\Temp\prism-piece6\allow-task-06.txt`, and copy them into the task report.

- [ ] **Step 4: Reverse the pin first, and see it fail.** In `tests/test_tool.py::test_parse_forced_and_recording_set_rules`, add `import re` beside the function's existing local import `from core.tool.config_args import UsageError, parse_forced, recording_set`. Replace the `:376` assertion with:
  ```python
      with pytest.raises(UsageError, match="frequency") as exc:
          recording_set(chi, _args(forced=["a.npy"], f0_si=1e-12))
      msg = str(exc.value)
      assert "must state the frequency (Hz) it was driven at" in msg, msg
      assert not re.search(r"\([A-Z]\d+\)", msg), \
          f"the refusal carries an internal label in place of its reason: {msg}"
  ```
  Run `python -m pytest tests/test_tool.py::test_parse_forced_and_recording_set_rules -q`. Expected: FAIL on the `re.search` assertion, with the message ending "Missing for: ['a.npy']. (D9)".

- [ ] **Step 5: Change message 10 and see the pin pass.** Delete " (D9)" from the f-string at `config_args.py:248`, as the Binding gives it. Re-run the Step 4 test. Expected: `1 passed`.

- [ ] **Step 6: Change message 9.** Delete " (trap CHI10)" at `identifiability.py:597`. Run `python -m pytest tests/test_diagnostics.py::test_the_probe_budget_refuses_a_miswired_cos_sin_pair -q`. Expected: `1 passed`.

- [ ] **Step 7: Clean the rest, file by file, reading each in full.** Apply the §2.5 edits from the Binding (`config_args.py:94`, `smoke.py:6`, `requirements.txt:8-9` and `:44`, `conftest.py:1`, `logging_console.py:13`) and the collision rewrites. Examples:
  - `identifiability.py:70` becomes `# Refused, not clamped: a clamp turned --n-worst 0 into 1 without a word, and a setting is never silently clamped or ignored.`
  - `stages.py:184-186` becomes `# --forced/--drive/--f0-si describe a MEASURED recording. --cell re-simulates the cell's own drive, so on that branch they name nothing the command reads -- a flag that would be silently ignored, which the tool refuses instead.`
  - `rng.py:3` becomes "``seeded`` lives in ``core/rng.py`` because the FDT measurement needs it too, and ...".

  Result: `task6-hits.txt` has no surviving line.

- [ ] **Step 8: Prove only the permitted changes happened.** Run (PowerShell):
  ```powershell
  $files = git diff --name-only -- '*.py' | Where-Object { $_ -ne 'tests/test_source_hygiene.py' }
  python C:\Users\J\AppData\Local\Temp\prism-piece6\comment_only_check.py --base HEAD --allow C:\Users\J\AppData\Local\Temp\prism-piece6\allow-task-06.txt @files
  ```
  Expected: `$LASTEXITCODE` 0. The printed differences are only the two messages and the reversed pin. `requirements.txt` has no Python AST; review it with `git diff requirements.txt`, where only lines 8-9 and 44 may change.

- [ ] **Step 9: Run the scan suite.** `python -m pytest tests/test_source_hygiene.py -q` Expected: all pass. `tests/test_tool.py` is still listed and still has hits.

- [ ] **Step 10: Run the raw-source pins.**
  `python -m pytest "tests/test_diagnostics.py::test_the_diagnostic_guards_refuse_naming_no_flag_or_file" "tests/test_tool.py::test_the_fdt_subcommands_no_longer_say_they_have_no_bounds_file" "tests/test_refusals.py::test_every_domain_error_is_a_refusal_and_carries_a_field" -q`
  Expected: `3 passed`.

- [ ] **Step 11: Run the focused suites for the touched files.** Start the run in the background with output to `C:\Users\J\AppData\Local\Temp\prism-piece6\task6-focused.log`, and edit nothing until it exits:
  `python -m pytest tests/test_tool.py tests/test_diagnostics.py tests/test_refusals.py -m "not slow" -q`
  Expected: 0 failed.

- [ ] **Step 12: Commit.** Add every changed file explicitly:
  ```powershell
  git add tests/test_source_hygiene.py tests/test_tool.py conftest.py requirements.txt <each cleaned core/tool and core/diagnostics file>
  git commit -m "cleanup: the tool, the diagnostics and the prose files state their reasons" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
  ```
  The controller then starts the review and, at the same moment, the one-process fast gate `python -m pytest -m "not slow" -q` in the background with output to a log. The implementer does not run the gate. The review re-runs Step 8's checker and reads every rewritten sentence at a collision id (D6, D7, D8, guardrail 8) against its context.

---

### Task 7: Cleanup: tests, batch A (the tool and FDT suites)

**Files:**
- Modify: `tests/test_tool.py`. 206 hit lines in `recon/design-scanrule-hits.tsv`; 205 remain once Task 6 has rewritten the `"(D9)"` pin near `:376`.
- Modify: `tests/test_fdt_user.py`. 149 hit lines.
- Modify: `tests/test_fdt_compare.py`. 38 hit lines.
- Modify: `tests/test_source_hygiene.py`. Only `NOT_YET_CLEAN` changes: it loses the three paths above.
- Create, outside the repository: `C:\Users\J\AppData\Local\Temp\prism-piece6\allow-task-07.txt`, the list of permitted changes.
- Test: `tests/test_source_hygiene.py`

**Interfaces:**
- Consumes (Task 1):
  - `NOT_YET_CLEAN: frozenset[str]`, holding repo-relative posix paths.
  - The tests `test_no_file_outside_the_pending_list_cites_a_working_document`, `test_every_pending_file_still_has_a_hit` and `test_every_allowlist_entry_still_matches_something`.
  - The controller tool `python comment_only_check.py --base <git-ref> --allow <allow-file> <path> [<path> ...]`. The allow file holds one allowed difference per line, written `<repo-relative-path>|<kind>|<quoted text fragment>`, with kind one of `message`, `string`, `rename` or `assert`. The tool exits 0 when every difference is allowed and 1 otherwise. It prints `path:line: <old> -> <new>` for each difference.
- Produces: nothing new. After this task, `NOT_YET_CLEAN` no longer holds `tests/test_tool.py`, `tests/test_fdt_user.py` or `tests/test_fdt_compare.py`.

**Binding (read before starting):** spec §2.1–§2.7 are binding, so read them. These are the points that carry weight:
- `python` below means `C:\Users\J\anaconda3\envs\biophys-env\python.exe` (in the Bash tool, `/c/Users/J/anaconda3/envs/biophys-env/python.exe`); the `python` on PATH is a different interpreter without torch or pytest. Commands written with backslash paths are PowerShell commands; in Bash use forward slashes.
- **What may change.** Comments, docstrings, and the 34 message strings listed in Step 2. Nothing else may change:
  - no code;
  - no test name (the two renames belonged to Task 1);
  - no test data;
  - no string a test compares against;
  - no `match=` text.

  Task 6's rewritten pin near `tests/test_tool.py:376`, the absence check using `\([A-Z]\d+\)`, belongs to Task 6. Leave it alone.
- **The rule (spec §2.1, H4).** When you finish, no comment, docstring or string in these files cites any of these working documents: `PRISM_HANDOFF` (the handoff), `docs/STATE.md`, `CLAUDE.md`, the specs, the plans, a ledger or a review. None cites their labels either:
  - piece numbers: "piece 5", "whole-piece";
  - section and spec references: "§N", "section N.N", "spec Sec. N.N", "spec N";
  - process words: "Task N", "fix round", "finding N", "Review Focus", "ruling";
  - decision and review ids: `D7`, `V3`, `E2`, `B6`, `P72`, `N20`, `M13`, `K1`, `R-F1`, `FE6`, `S2`, `T12` and the like;
  - review line numbers such as `L483`;
  - "walkthrough" and row ids such as "row D15";
  - "hardening programme", "clean break".

  Replacement text follows the same rule and states the reason in words. It may name code, other tests, and files under `tests/` or `core/`.
- **How to treat a hit (spec §2.3).** Delete a provenance tag, e.g. "(E7, P12)", "(the whole-piece review's N6)", "(spec §1.2)", "K8, fix round 1:". When a label is the subject of a sentence, replace it with the behaviour it stands for, read from the test's own assertions. The ids in these files mean the following:
  - `D6`: every setting is a flag that reaches its stage as a keyword, and a value is refused, never silently clamped or ignored. It never means the transposed rotation. At `tests/test_tool.py:178` ("D6: every setting is a flag that travels to its stage as a keyword."), delete "D6:" and keep the sentence.
  - `D7`: a committed simulation cache that is one setting away is refused before the Fisher step and before any simulation.
  - `D8`: a narrowed or truncated load is refused unless an `--accept-*` flag answers it.
  - `D9`: a driven chi recording states its frequency in Hz.
  - `D11`: there is no chi override; the band and drive are `config.py`'s.
  - `D13`: the two roots and the two core-level environment settings are named in the help epilog and read nowhere else in the tool.
  - "Walkthrough row D15" at `:3715`: say that the module name is what a user reads on the console.
  - `E1`: the analyses write named records.
  - `E2`: an interrupted `fdt` run keeps its folder, marked unfinished.
  - `E4`: a run where some points failed is a completed record with the count in it.
  - `E5`: a check refuses what breaks; a setting too thin to trust warns, and the warning is recorded.
  - `E7`: every run records its seed, drawing one when none is given.
  - `E8`: comparing covers four modes.
  - `E9`: a frequency the run could not measure comes back blank, and the band is refused before the driven campaign.
  - `E10`: the tidy-up sees the legacy loose files and deletes nothing unasked.
  - `E11`: only `smoke` makes and removes its own artifacts root.
  - `E12`: an ordering note. Delete it.
  - Every other letter-plus-number (`B…`, `F…`, `I…`, `K…`, `M…`, `N…`, `P…`, `R…`, `R-F…`, `S…`, `T…`, `V…`, `FE…`, `L<number>`) is a decision, ruling, review finding, task or line number. Delete it, and keep or write the behaviour sentence.
- **Facts that came from a working document stay, without the source:**
  - "(CLAUDE.md)" at `:115`: delete the citation.
  - "the trap CLAUDE.md spells out" at `:181`: name the trap itself. Orchestrator binds config constants at import, so assigning `config.X` changes nothing.
  - "the GPU smoke gate in CLAUDE.md and docs/STATE.md" at `:2681` and `:2725`, and "the GPU recipe in CLAUDE.md" at `:3708`: write "the GPU smoke gate".
  - "the first of the two gaps docs/STATE.md names" at `:1392`, and "docs/STATE.md's piece-5 row hands on and spec section 8.2 closes" at `:2425`: say which untested branch the test covers.
  - `tests/test_fdt_user.py:227`, "PRISM_HANDOFF names the consequence of not having it: '…'": state the consequence yourself, without quotation marks and without a source. Widening `freq_bounds` past the spectrum's resolution gives T_eff/T a smooth, plausible, fabricated tail.
- **Keep the physics names `F0` and `D0`.** They are on the allowlist, not hits. The four `D0`s in `tests/test_fdt_user.py` (near `:143-156`) are four of only six in the tree. If `D0` stops matching anything, `test_every_allowlist_entry_still_matches_something` fails.
- **Dates (spec §2.6).** A date that dates a measurement may stay. An incident used as a name is told in words instead: for "the 2026-09-11 incident" at `:440`, say what happened. The `created="2026-01-01 00:00:00"` literals near `:2912`, `:2930` and `:2940` are test data. Do not touch them.
- **Misses the scan cannot report (spec §2.2).** Find these by reading every line of each file, not only the reported ones:
  - sentence-initial "Fix round N": the scan reports these too (Task 1's pattern takes either case), so this item is a cross-check. There are two in `tests/test_tool.py` (`:3219`, `:3766`) and four in `tests/test_fdt_user.py` (`:1933`, `:2103`, `:2992`, `:3046`);
  - "(FEATURE 1 v3 + B-d)" at `tests/test_fdt_user.py:2`;
  - "an owner item: list 4, item 11" at `:1378`;
  - "the V1 defect the rest of the piece removes" at `:2701`;
  - "the last part of the piece" at `tests/test_fdt_compare.py:3`;
  - "§1 of the spec measured" at `:148`: the § is reported, but "of the spec" is not.
- **Tests that read raw source.** Only two tests read these files' text: the scan, and `tests/test_settings_persistence.py::test_no_suite_points_the_settings_back_at_the_real_ini`. The second fails if any `tests/test_*.py` contains `use_ini_file(` immediately followed by `None)`, so never write that joined text. (`tests/test_tool.py:2094` reads `core/tool/fdt.py`, not this file.)
- **Line numbers are hints.** They were taken at `b24973e`, and Task 6's edit near `:376` can shift later lines. The quoted text is the anchor.
- **Process.** Never edit while a pytest run is in progress. Never run two pytest processes at once. Do not run the fast gate.

- [ ] **Step 1: Record the base commit and the collected tests**

```bash
git rev-parse --short HEAD        # note it as BASE for Step 9
mkdir -p C:/Users/J/AppData/Local/Temp/prism-piece6
python -m pytest tests/test_tool.py tests/test_fdt_user.py tests/test_fdt_compare.py --collect-only -q | grep "::" > C:/Users/J/AppData/Local/Temp/prism-piece6/t7-collect-before.txt
```
Confirm that `NOT_YET_CLEAN` holds all three paths. If any is missing, stop and report.

- [ ] **Step 2: Write the list of permitted changes, before any edit (spec §2.7 (b))**

Read the module docstring of `comment_only_check.py` for the exact line format. Then write `allow-task-07.txt` with one `string` line per message, taking each fragment from the text as it is NOW. This is the known set of 34 messages. If Step 3's report shows another message, add it.
```
tests/test_tool.py|string|ENTERED by the stage (spec §1.2)
tests/test_tool.py|string|body.settings must hold the name (§4.4)
tests/test_tool.py|string|one record per swept parameter (§4.1)
tests/test_tool.py|string|must enter on its own (spec §1.2)
tests/test_tool.py|string|M1: the refusal says where the name came from
tests/test_tool.py|string|draws one and records it (E7, P12)
tests/test_tool.py|string|(E11: auto_root is smoke's alone)
tests/test_tool.py|string|its ONE seed on both records (spec section 4.1)
tests/test_tool.py|string|B6: no force in either front end
tests/test_tool.py|string|a run in flight from a leftover (R3)
tests/test_fdt_user.py|string|every probe is then out of range (P7)
tests/test_fdt_user.py|string|so no fix is offered (P2, P75)
tests/test_fdt_user.py|string|it (P22):
tests/test_fdt_user.py|string|names the setting and offers no fix (P2)
tests/test_fdt_user.py|string|E5: a zero burn-in is legal
tests/test_fdt_user.py|string|is read off the config (P72)
tests/test_fdt_user.py|string|the ruling sets no floor on S
tests/test_fdt_user.py|string|is KEPT in the record (P51)
tests/test_fdt_user.py|string|never consulted (the whole-piece review's N6)
tests/test_fdt_user.py|string|a committed payload is hashed (spec §2.2)
tests/test_fdt_user.py|string|one study, one seed (spec §4.1)
tests/test_fdt_user.py|string|F40: [min, max, N] as spec §2.3 says
tests/test_fdt_user.py|string|(spec §2.3; the whole-piece review's N7)
tests/test_fdt_user.py|string|not in the body (§2.3)
tests/test_fdt_user.py|string|the contract's root attributes (P6)
tests/test_fdt_user.py|string|P70: a sweep has no sanity branch
tests/test_fdt_user.py|string|comes before the thin notice (F10)
tests/test_fdt_user.py|string|is a completed record (E4)
tests/test_fdt_user.py|string|P78
tests/test_fdt_user.py|string|the whole-piece review's M2, departing from P78
tests/test_fdt_user.py|string|review's N2), one per point as it lands or fails
tests/test_fdt_user.py|string|F41: the refusal names the cell
tests/test_fdt_user.py|string|null until the run finishes (§2.3)
tests/test_fdt_compare.py|string|(the whole-piece review's N13)
```

- [ ] **Step 3: Take the three files off the pending list and watch the scan fail**

Delete the three paths from `NOT_YET_CLEAN`, then run:

`python -m pytest tests/test_source_hygiene.py::test_no_file_outside_the_pending_list_cites_a_working_document -q`

Expected: FAIL. Every reported `path:line family 'match'` names one of the three files, about 390 lines in all. Keep the report: it is your work list.

- [ ] **Step 4: Clean `tests/test_tool.py`**

Besides the routine tag deletions, handle these sites:
- the module docstring at `:1`: delete "piece 2 of the 2026-09-10 hardening programme";
- "spec Sec. 8.4" at `:52` and `:932`;
- the facts at `:115`, `:181`, `:1392`, `:2425`, `:2681`, `:2725` and `:3708`, as set out in Binding;
- `D13` at `:126`, `D6` at `:178`, and `D7` plus the incident at `:440`;
- `D8`, "(Tasks 12/16)" and "T12 and T16's own tool tests" at `:717`;
- "(… D11)" at `:2519`;
- the `M13:` comments near `:734-788`, and "K8/K2/K3/K5/M4/M3/M2/M5, fix round 1:" at `:906`, `:1013`, `:1045`, `:1092`, `:1272`, `:1354`, `:1360` and `:1767`;
- "M1, as V3's one line since piece 5:" at `:1341`;
- "(ruling R-F1)" at `:1647`, "(FE1 + S2)" at `:1670`, "(L483)" at `:1757` and "(L728)" at `:3408`;
- "Walkthrough row D15" at `:3715`;
- the 10 messages;
- section-divider comments such as `# ── the artifacts family (piece 4, B9; design §4) ──…`: keep the divider and its topic, and drop the parenthesis.

- [ ] **Step 5: Clean `tests/test_fdt_user.py`**

Handle these sites:
- the module docstring at `:1-20`: "(FEATURE 1 v3 + B-d)", "since piece 5", "added by piece 5";
- openings such as "Spec §3.4, first bullet.": delete them and keep the behaviour;
- "(ruling F10)" at `:176`;
- the handoff at `:227`;
- "Before T14 … after T14" at `:320`: write "before the band was checked … after";
- "ruled after Task 14's" at `:329`;
- "the ruling above" and "the same ruling" at `:396`, `:466` and `:651`: name the rule the earlier docstring states;
- "(the review of Task 16)" at `:543`, and "The review of Task 16 (Critical), at the figure." at `:742`;
- `:1378` and `:2701`;
- "Task 20's departure from A7, pinned (fix round 1)." at `:3080`;
- the four "Fix round 1" openings;
- the 23 messages (24 source lines).

Keep every `D0`.

- [ ] **Step 6: Clean `tests/test_fdt_compare.py`**

Handle these sites:
- "(piece 5, spec §7; decision E8)" at `:1`;
- `:3`: drop "the last part of the piece … (E12)";
- "the exact defect §1 of the spec measured in ``_interp_log``" at `:148`: write "the defect ``_interp_log`` had";
- `:170`: state the rule without attribution. A comparison draws only records whose `complete` is true;
- "(P49, F57-F59)" at `:206`;
- "what the walkthrough row reads" at `:523`: write "what a user reads";
- "as the owner ruled it (R-F6: a notice, never a refusal)" at `:710`: write "a notice, never a refusal";
- the message at `:951`.

- [ ] **Step 7: Do the manual pass**

Read every line of the three files once more for the known misses. Then run:

`grep -nE "[Ff]ix round|\bspec\b|\bpieces?\b|owner item|FEATURE [0-9]|\bB-[a-z]\b" tests/test_tool.py tests/test_fdt_user.py tests/test_fdt_compare.py`

Expected: nothing left that cites a document. Code names such as `s_spec` do not match `\bspec\b`.

- [ ] **Step 8: Run the scan suite**

Run: `python -m pytest tests/test_source_hygiene.py -q`

Expected: every test passes, including `test_every_pending_file_still_has_a_hit` and `test_every_allowlist_entry_still_matches_something`.

- [ ] **Step 9: Run the comment-only checker**

```bash
python C:/Users/J/AppData/Local/Temp/prism-piece6/comment_only_check.py --base BASE --allow C:/Users/J/AppData/Local/Temp/prism-piece6/allow-task-07.txt tests/test_tool.py tests/test_fdt_user.py tests/test_fdt_compare.py
```
Expected:
- exit code 0;
- every printed difference is a line of the allow file;
- every edited f-string message has the same `{…}` fields before and after.

Do not pass `tests/test_source_hygiene.py` to the checker: its list edit is a code change and is reviewed as such.

- [ ] **Step 10: Check that the collection is unchanged and the needle is clean**

```bash
python -m pytest tests/test_tool.py tests/test_fdt_user.py tests/test_fdt_compare.py --collect-only -q | grep "::" > C:/Users/J/AppData/Local/Temp/prism-piece6/t7-collect-after.txt
diff C:/Users/J/AppData/Local/Temp/prism-piece6/t7-collect-before.txt C:/Users/J/AppData/Local/Temp/prism-piece6/t7-collect-after.txt
python -m pytest tests/test_settings_persistence.py::test_no_suite_points_the_settings_back_at_the_real_ini -q
```
Expected: `diff` prints nothing, and the needle test reports `1 passed`.

These checks, together with the scan suite and the checker, are this task's focused tests. The checker proves that nothing executable changed except the listed messages, and a message runs only when its assertion fails. So the three files are run whole by the controller's gate, not here.

- [ ] **Step 11: Commit**

```bash
git add tests/test_tool.py tests/test_fdt_user.py tests/test_fdt_compare.py tests/test_source_hygiene.py
git commit -m "tests: the tool and FDT suites cite no working document" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
The report gives:
- BASE;
- the contents of the allow file;
- the checker's output;
- the hit count from Step 3;
- the `diff` result;
- every rewritten sentence at a colliding id (`D1`, `D3`, `D6`, `C1`/`C2`, `M1`–`M4`, `X1`–`X5`) with its file:line, so the reviewer can read it against its context (spec §2.3).

The controller starts the one-process fast gate (`python -m pytest -m "not slow" -q`, in the background, logged) at the same moment as this task's review. Do not run it yourself.

---

### Task 8: Cleanup: tests, batch B (the window suites)

**Files:**
- Modify: `tests/test_nav_and_gating.py`. 168 hit lines in `recon/design-scanrule-hits.tsv`. Task 1 has already renamed the test at `:1280`.
- Modify: `tests/test_artifact_browser.py`. 65.
- Modify: `tests/test_settings_persistence.py`. The TSV has 34. Add `:409`'s `L1`, which the reconnaissance allowed and the final rule does not, for 35.
- Modify: `tests/test_worker_dispatch.py`. 28.
- Modify: `tests/test_simulate.py`. 10.
- Modify: `tests/test_artifact_consistency.py`. 7.
- Modify: `tests/test_vt_progress.py`, `tests/test_chi_set_encoder.py` and `tests/test_gpu_paths.py`. 2 each.
- Modify: `tests/test_figures.py`. 0 scan hits, so it is not on the pending list. It carries the stale run line (spec §2.5) and two labels the scan misses.
- Modify: `tests/test_source_hygiene.py`. Only `NOT_YET_CLEAN` changes: it loses the nine paths with hits listed above.
- Create, outside the repository: `C:\Users\J\AppData\Local\Temp\prism-piece6\allow-task-08.txt`
- Test: `tests/test_source_hygiene.py`

**Interfaces:**
- Consumes (Task 1):
  - `NOT_YET_CLEAN: frozenset[str]`;
  - the tests `test_no_file_outside_the_pending_list_cites_a_working_document`, `test_every_pending_file_still_has_a_hit` and `test_every_allowlist_entry_still_matches_something`;
  - the controller tool `python comment_only_check.py --base <git-ref> --allow <allow-file> <path> [<path> ...]`. The allow file holds lines written `<repo-relative-path>|<kind>|<quoted text fragment>`, with kind one of `message`, `string`, `rename` or `assert`. The tool exits 0 when every difference is allowed and 1 otherwise, and prints `path:line: <old> -> <new>` for each difference.
- Produces: nothing new. After this task `NOT_YET_CLEAN` no longer holds these paths: `tests/test_nav_and_gating.py`, `tests/test_artifact_browser.py`, `tests/test_settings_persistence.py`, `tests/test_worker_dispatch.py`, `tests/test_simulate.py`, `tests/test_artifact_consistency.py`, `tests/test_vt_progress.py`, `tests/test_chi_set_encoder.py`, `tests/test_gpu_paths.py`.

**Binding (read before starting):** spec §2.1–§2.7 are binding, so read them. These are the points that carry weight:
- `python` below means `C:\Users\J\anaconda3\envs\biophys-env\python.exe` (in the Bash tool, `/c/Users/J/anaconda3/envs/biophys-env/python.exe`); the `python` on PATH is a different interpreter without torch or pytest. Commands written with backslash paths are PowerShell commands; in Bash use forward slashes.
- **What may change.** Comments, docstrings, the 17 messages and the six header lines listed in Step 2. Nothing else: no code, no test name, no test data, no string a test compares against.
- **The rule (H4).** When you finish, no comment, docstring or string in these files cites the handoff, `docs/STATE.md`, `CLAUDE.md`, `docs/checklists/display-walkthrough.md`, the specs, plans, a ledger or a review. None cites their labels either:
  - piece numbers;
  - "§N", "section N.N", "spec N";
  - "Task N", "fix round", "finding N", "Review Focus", "Backlog", "ruling";
  - decision, review and ruling ids;
  - "guardrail N", trap ids, "C-N";
  - "walkthrough", "row C3".

  Replacement text follows the same rule and states the reason in words.
- **How to treat a hit (spec §2.3).** Delete a tag. Where a label is the subject of a sentence, write the rule it stands for. Use the spec's vocabulary wherever these files name one of these rules:
  - guardrail 2 (at `tests/test_nav_and_gating.py:375` and `:1441`) is the narrowed-model rule: a narrowed posterior is marked as such and never loads or infers as a broad one;
  - guardrail 8 (at `:426`) is the calibrate-on-the-region rule;
  - trap X5 (at `:3437`): the calibration's operating-point count is t_scale's effective sample size;
  - trap CHI10 (at `tests/test_chi_set_encoder.py:432`): a near-constant channel inflates a standardised Jacobian;
  - the chi series C-2, C-3 and C-5 (at `:628` and `:667`, `tests/test_chi_set_encoder.py:384`, and `tests/test_artifact_consistency.py:273`): name the specific measure in words, read from the test, e.g. "the probe table's variable rows" or "the probe planner";
  - "Backlog": delete it.
- **What the other ids mean:**
  - `D7`: the near-miss dialog. A committed cache one setting away is refused before the Fisher step and before any simulation.
  - `D8`: the question the Posterior tab asks before it loads a stored narrowed posterior.
  - `D11`: there is no chi override.
  - `D12`: a narrowing round whose parent is itself narrowed, and whose observation is not that parent's, is refused with no escape hatch.
  - `D3` at `tests/test_settings_persistence.py:253` ("D3's user-facing face"): the region belongs to the simulation cache's identity.
  - "defect D7" at `:868`: the defect the message itself describes. The window derived the cache identity on its own and resolved run_size as `cap or hw` instead of `min(hw, cap)`.
  - `L1` at `:409`: "the remembered splitter layout".
  - `V1`: a public stage works on a private copy of the configuration and changes nothing it was handed.
  - `V2`: a bad box is refused at the click, naming the box, the range and the default, with no clamp.
  - `V3`: one Refusal kind with a field key. Core messages stay neutral, and each front end names its own control or flag.
  - `V4`: standard logging, and a `log.txt` in every committed record.
  - `V5`: the inference tabs remember selections and the training budget only, and every science knob opens at `config.py`.
  - `V6`: one `orchestrator.training_preview` behind the budget lines.
  - `E2`: an interrupted run keeps its folder, marked unfinished.
  - `E5`: a notice for a setting too thin to trust is recorded.
  - `E6`: the fix hints understand screens as well as tabs.
  - `E7`: the Seed box, the record name and the note are never remembered, and a run with no seed draws one and records it.
  - "walkthrough row C2/C3/C9/D15/D16": write "what a user reads on the real screen", or delete it.
  - Every other letter-plus-number (`B…`, `F…`, `M…`, `N…`, `P…`, `Q…`, `R…`, `R-F…`, `S…`, `T…`, `L<number>`): delete it and keep the behaviour.
- **Six stale run lines.** spec §2.5 names four, but six exist. Line 15, `      (or just: pytest tests/test_gui_progress.py)`, appears in `tests/test_figures.py`, `tests/test_nav_and_gating.py`, `tests/test_settings_persistence.py`, `tests/test_simulate.py`, `tests/test_vt_progress.py` and `tests/test_worker_dispatch.py`.
  - That file no longer exists, so delete the line in all six.
  - The line sits in the SECOND string statement of each file (lines 2–16). That string is not a docstring, because only a module's first statement is one. So the checker reports this change, and it is on the allow list as `string`.
  - Line 1's "Split from test_gui_progress.py" is history and stays. The rest of that second string stays as it is.
- **Stale references to the test Task 1 renamed.** `tests/test_artifact_browser.py:62` and `tests/test_worker_dispatch.py:344` still name `test_the_d7_and_d8_dialogs_default_to_cancel`. They are lower-case, so the scan does not report them. Replace each with the name the def at `tests/test_nav_and_gating.py:1280` now has, unless Task 1 already did. Verify with `git grep -n test_the_d7_and_d8_dialogs_default_to_cancel -- tests/test_artifact_browser.py tests/test_worker_dispatch.py`, which should print nothing. The same test's own docstring at `:1281` names D7 and D8. Rewrite it as "the near-miss dialog" and "the load question".
- **Tests that read raw source.**
  - `tests/test_nav_and_gating.py:4641-4690` holds phrases that are test DATA asserting on core files' text: "Nine of these", "8 of the 9", "all nine differ", "nine independent splitters", "the five inference tabs", "Ten of these exist", "9 of the 10", "all ten differ", "the six inference tabs", "the only QSettings WRITE site", "_persist_layout", "1500". They are code, not hits: leave them untouched. In that area, change only the docstrings' "(piece 4, B1)", "(ledger P37)" and "(Q16)", and the `# ── piece 6: …` divider.
  - `tests/test_settings_persistence.py::test_no_suite_points_the_settings_back_at_the_real_ini` (`:939-948`) reads every `tests/test_*.py` for `use_ini_file(` followed by `None)`, and assembles its own needle. Never write the joined text anywhere.
  - Every `code_only` and `inspect.getsource` call in these files targets core modules.
- **Misses the scan cannot report (spec §2.2).**
  - "SECTION 11.6" at `tests/test_nav_and_gating.py:375`: the scan reports it (the section pattern accepts upper case); clean it with the rest.
  - Sentence-initial "Fix round 1" (the scan reports these too; a cross-check): at `tests/test_nav_and_gating.py:2122`, `:2822`, `:2867`, `:2906`, `:2967` and `:3005`, `tests/test_artifact_browser.py:1529`, and `tests/test_worker_dispatch.py:1102`.
  - "the spec's test row" at `tests/test_nav_and_gating.py:1117`, and "the spec names" at `:2530`.
  - "(round 4)" at `tests/test_figures.py:248`, and "the B-c hardcoded-white regression" at `:438`.

  Read every line of every file.
- **Keep `F0`.** It is on the allowlist; `tests/test_nav_and_gating.py` holds 11 of its uses.
- **Line numbers are hints** at `b24973e`. The quoted text is the anchor.
- **Process.** Never edit while a pytest run is in progress. Never run two pytest processes at once. Do not run the fast gate.

- [ ] **Step 1: Record the base commit and the collected tests**

```bash
git rev-parse --short HEAD        # BASE
python -m pytest tests/test_nav_and_gating.py tests/test_artifact_browser.py tests/test_settings_persistence.py tests/test_worker_dispatch.py tests/test_simulate.py tests/test_artifact_consistency.py tests/test_vt_progress.py tests/test_chi_set_encoder.py tests/test_gpu_paths.py tests/test_figures.py --collect-only -q | grep "::" > C:/Users/J/AppData/Local/Temp/prism-piece6/t8-collect-before.txt
```
Confirm that `NOT_YET_CLEAN` holds the nine paths and does not hold `tests/test_figures.py`. If not, stop and report.

- [ ] **Step 2: Write the list of permitted changes, before any edit**

Use the format from the checker's docstring and take each fragment from the current text:
```
tests/test_nav_and_gating.py|string|(or just: pytest tests/test_gui_progress.py)
tests/test_nav_and_gating.py|string|cannot drift apart (§5.2)
tests/test_nav_and_gating.py|string|from label({key!r}) (P34)
tests/test_nav_and_gating.py|string|two records minted one id (F4)
tests/test_nav_and_gating.py|string|reach the record's settings (T11, P72)
tests/test_nav_and_gating.py|string|the stage fills the rest (P15)
tests/test_nav_and_gating.py|string|(T19's signature)
tests/test_nav_and_gating.py|string|two unnamed records share one folder (F4)
tests/test_nav_and_gating.py|string|a notice must survive to the screen (E5)
tests/test_nav_and_gating.py|string|wherever it is read (E2)
tests/test_artifact_browser.py|string|B3: complete is 'has a manifest'
tests/test_artifact_browser.py|string|F51: no directory was a candidate
tests/test_settings_persistence.py|string|(or just: pytest tests/test_gui_progress.py)
tests/test_settings_persistence.py|string|the campaign knobs are still remembered (V5)
tests/test_settings_persistence.py|string|the free knobs are still remembered (V5)
tests/test_settings_persistence.py|string|from label({key!r}) (P34)
tests/test_settings_persistence.py|string|defect D7 removed
tests/test_settings_persistence.py|string|the TSNPE budget is remembered (spec §5.3)
tests/test_settings_persistence.py|string|the Validate tab restores nothing (V5)
tests/test_figures.py|string|(or just: pytest tests/test_gui_progress.py)
tests/test_simulate.py|string|(or just: pytest tests/test_gui_progress.py)
tests/test_vt_progress.py|string|(or just: pytest tests/test_gui_progress.py)
tests/test_worker_dispatch.py|string|(or just: pytest tests/test_gui_progress.py)
```

- [ ] **Step 3: Take the nine files off the pending list and watch the scan fail**

Delete the nine paths from `NOT_YET_CLEAN`, then run:

`python -m pytest tests/test_source_hygiene.py::test_no_file_outside_the_pending_list_cites_a_working_document -q`

Expected: FAIL. Every reported line names one of the nine files, about 320 lines in all.

- [ ] **Step 4: Clean `tests/test_nav_and_gating.py`**

Handle these sites:
- `:15`;
- `:375`: SECTION 11.6 and GUARDRAIL 2;
- `:426`: guardrail 8 and D8's question;
- `:628` and `:667`: Backlog and C-2;
- `:1281`;
- `:1441`;
- `:1490` and `:1553`: walkthrough;
- `:3437`: trap X5 and "(spec 3.3)";
- `:3487`: V2 and walkthrough row C3;
- `:4161`: walkthrough row D16;
- `:4640`: the `# ── piece 6` divider;
- `:4649`: ledger P37;
- the six "Fix round 1 of T27 / (T25)" openings;
- the 9 messages.

- [ ] **Step 5: Clean `tests/test_artifact_browser.py` and `tests/test_settings_persistence.py`**

- In the browser file: `:62` (the renamed test), "piece 4 established" at `:800`, `:1529`, and the 2 messages.
- In the settings file:
  - `:15`;
  - `:66` ("(piece 1, Task 4)");
  - `:196`, `:253` and `:409` (`L1`);
  - the V5, E7 and P34 docstrings;
  - `:871` (the `# ── piece 3:` divider);
  - `:1248-1249` (D7, D8);
  - the 6 messages.

  Do not touch the file's needle test.

- [ ] **Step 6: Clean the remaining files**

- `tests/test_worker_dispatch.py`:
  - `:15` and `:344`;
  - `:740`: state the fact, which is that a text sort puts 10 before 9, without "docs/STATE.md" or "piece";
  - `:1099`: walkthrough row D15;
  - `:1102`: Fix round 1;
  - `:1220`: "The rule in CLAUDE.md and in training_checkpoint's module docstring" becomes "The rule in training_checkpoint's module docstring";
  - the `# ── piece 4, B14/B15 …` dividers at `:1079`, `:1163` and `:1478`.
- `tests/test_simulate.py`: `:15`, `:136`, `:141-142` (Task 41, §12 row), `:266`, `:276`, `:293`, `:362`, `:432`, `:441` and `:444`.
- `tests/test_artifact_consistency.py`: `:124`, `:273`, `:297`, `:315`, `:335`, `:339` and `:356`.
- `tests/test_vt_progress.py`: `:15`, `:497` and `:603`.
- `tests/test_chi_set_encoder.py`: `:384` and `:432`.
- `tests/test_gpu_paths.py`: `:1`, and `:10`. At `:10`, the smoke gate on the card is named by what it does, with no document path.
- `tests/test_figures.py`: `:15`, `:248` and `:438`.

- [ ] **Step 7: Do the manual pass**

Read every line of the ten files. Then run:

`grep -nE "[Ff]ix round|SECTION [0-9]|\bspec\b|\bpieces?\b|\(round [0-9]|\bB-[a-z]\b|test_gui_progress\.py\)|dialogs_default_to_cancel" <the ten files>`

Expected: the only matches are the renamed test's new name, which ends `_dialogs_default_to_cancel` (its def at `tests/test_nav_and_gating.py:1280` and the two docstrings that name it, `_answer` in `tests/test_artifact_browser.py` and `tests/test_worker_dispatch.py:344`), and line 1's "Split from test_gui_progress.py" history. `\)` excludes that line from the `test_gui_progress` match.

- [ ] **Step 8: Run the scan suite**

Run: `python -m pytest tests/test_source_hygiene.py -q`

Expected: every test passes.

- [ ] **Step 9: Run the comment-only checker**

```bash
python C:/Users/J/AppData/Local/Temp/prism-piece6/comment_only_check.py --base BASE --allow C:/Users/J/AppData/Local/Temp/prism-piece6/allow-task-08.txt tests/test_nav_and_gating.py tests/test_artifact_browser.py tests/test_settings_persistence.py tests/test_worker_dispatch.py tests/test_simulate.py tests/test_artifact_consistency.py tests/test_vt_progress.py tests/test_chi_set_encoder.py tests/test_gpu_paths.py tests/test_figures.py
```
Expected: exit code 0, every difference is on the list, and the f-string `{…}` fields are unchanged. `tests/test_source_hygiene.py` is not passed to the checker.

- [ ] **Step 10: Check that the collection is unchanged and the needle is clean**

Collect the same ten files again into `t8-collect-after.txt` and `diff` the two lists. Expected: no output.

Then run `python -m pytest tests/test_settings_persistence.py::test_no_suite_points_the_settings_back_at_the_real_ini -q`. Expected: `1 passed`.

These checks, together with the scan suite and the checker, are the focused tests. The files themselves are run whole by the controller's gate.

- [ ] **Step 11: Commit**

```bash
git add tests/test_nav_and_gating.py tests/test_artifact_browser.py tests/test_settings_persistence.py tests/test_worker_dispatch.py tests/test_simulate.py tests/test_artifact_consistency.py tests/test_vt_progress.py tests/test_chi_set_encoder.py tests/test_gpu_paths.py tests/test_figures.py tests/test_source_hygiene.py
git commit -m "tests: the window suites cite no working document" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
The report gives:
- BASE;
- the contents of the allow file;
- the checker's output;
- the hit count from Step 3;
- the `diff` result;
- every rewritten sentence at a colliding id (`D3`, `D7`, `D8`, `C1`/`C2`, `M1`–`M4`, `X5`) with its file:line, for the reviewer (spec §2.3).

The controller starts the one-process fast gate (`python -m pytest -m "not slow" -q`, in the background, logged) at the same moment as this task's review. Do not run it yourself.

---

### Task 9: Cleanup: tests, batch C, and the end of the pending list

**Files:**
- Modify: `tests/test_artifact_store.py`. 165 hit lines in `recon/design-scanrule-hits.tsv`. Task 3 has already rewritten the `match="D3"` pin at `:71`.
- Modify: `tests/test_diagnostics.py`. 63.
- Modify: `tests/test_refusals.py`. 49.
- Modify: `tests/test_user_sbi.py`. 44. Task 1 has already renamed the test at `:3184`.
- Modify: `tests/test_conditioning_repair.py`. 28. Tasks 1 and 2 have already edited `:241` and `:680`.
- Modify: `tests/test_user_models.py`. 17.
- Modify: `tests/_fixtures.py`. 17.
- Modify: `tests/conftest.py`. 7, including the message at `:51`.
- `core/Reduction/tests/`: no hits under the final rule. A re-scan at `b24973e` agrees with the recon, so there is nothing to edit.
- Modify: `tests/test_source_hygiene.py`. First `NOT_YET_CLEAN` empties. Then the pending mechanism is deleted.
- Create, outside the repository: `C:\Users\J\AppData\Local\Temp\prism-piece6\allow-task-09.txt`
- Test: `tests/test_source_hygiene.py`

**Interfaces:**
- Consumes (Task 1):
  - `NOT_YET_CLEAN: frozenset[str]`;
  - `scanned_files() -> list[Path]`;
  - `scan_python(path: Path) -> list[Hit]` and `scan_prose(path: Path) -> list[Hit]`;
  - the tests `test_no_file_outside_the_pending_list_cites_a_working_document`, `test_every_pending_file_still_has_a_hit`, `test_every_allowlist_entry_still_matches_something`, `test_no_test_name_carries_a_process_label` and `test_a_failure_names_the_file_line_family_and_the_allowlist`;
  - the controller tool `python comment_only_check.py --base <git-ref> --allow <allow-file> <path> [<path> ...]`. The allow file holds lines written `<repo-relative-path>|<kind>|<quoted text fragment>`. The tool exits 0 when every difference is allowed and 1 otherwise.
- Produces: the scan's end state, which Task 27 extends.
  - `NOT_YET_CLEAN` and `test_every_pending_file_still_has_a_hit` no longer exist.
  - `test_no_file_outside_the_pending_list_cites_a_working_document` is renamed `test_no_scanned_file_cites_a_working_document` (Step 12, in the commit that ends the list), and checks every path `scanned_files()` returns with no exemption.

**Binding (read before starting):** spec §2.1–§2.8 are binding, so read them. These are the points that carry weight:
- `python` below means `C:\Users\J\anaconda3\envs\biophys-env\python.exe` (in the Bash tool, `/c/Users/J/anaconda3/envs/biophys-env/python.exe`); the `python` on PATH is a different interpreter without torch or pytest. Commands written with backslash paths are PowerShell commands; in Bash use forward slashes.
- **What may change.** In the eight test files: comments, docstrings and the 16 messages listed in Step 2. No code, test name, test data or compared string may change. The rewritten pins at `tests/test_artifact_store.py:71` (Task 3) and `tests/test_conditioning_repair.py:680` (Task 2) are not yours. In `tests/test_source_hygiene.py`: the list edit and, in its own commit, the deletion described in Step 12.
- **The rule (H4).** No comment, docstring or string may cite the handoff, `docs/STATE.md`, `CLAUDE.md`, `docs/checklists/display-walkthrough.md`, the specs, plans, a ledger or a review, or their labels. That covers:
  - piece numbers;
  - "§N", "section N.N";
  - "design §N", "spec N";
  - "Task N", "tasks T13-T18";
  - "fix round", "finding N", "Review finding";
  - "ruling";
  - decision and review ids;
  - "guardrail N", "GUARDRAIL N";
  - trap ids;
  - "C-N";
  - "Appendix A";
  - "walkthrough", "row C9";
  - "hardening programme", "one-flow design", "clean break".

  Replacement text follows the same rule. It states the reason in words, and may name code and files under `tests/` and `core/`.
- **The narrowing-round rules, in the spec's vocabulary (spec §2.3):**
  - the truncated-prior rule: a narrowing round draws from the prior restricted to the region, never from the posterior;
  - guardrail 1 is the observation-digest rule: a round refuses unless the stored observation matches;
  - guardrail 2 is the narrowed-model rule: a narrowed posterior is marked as such and never loads or infers as a broad one;
  - guardrail 3 is the eigenbasis rule: the region is cut in the rotation's leading directions, and flat ones are left full width;
  - guardrail 5 is the generous-region rule: a 99.9 % region, with the truth-outside rate watched;
  - guardrail 6 is the cost-on-screen rule;
  - guardrail 7 is the region-carries-its-basis rule: the child reuses the parent's rotation and never recomputes it, skips directions loaded on t_scale, and names its base prior;
  - guardrail 8 is the calibrate-on-the-region rule;
  - C-11 is the training checkpoint, i.e. the simulation cache;
  - C-9 and C-10 are the specific measure named in words;
  - trap X5: the calibration's operating-point count is t_scale's effective sample size;
  - trap CHI10: a near-constant channel inflates a standardised Jacobian;
  - "Appendix A <date>": tell the incident in one clause, or not at all.
- **Colliding ids: map each one by its nearby words, never by the id alone.**
  - Handoff defect `D1`: a region drawn in one rotation and enforced in another (`tests/test_user_sbi.py:2294`, the message at `:2328`).
  - `D1` at `tests/test_conditioning_repair.py:938` means the prompt CLI is retired. Delete "D1:".
  - Handoff defect `D3`: the region belongs to the simulation cache's identity (`tests/test_user_sbi.py:2233`). `D3` is also the bimodality feature, `D3_bimodality`.
  - Handoff defect `D4`: a direction loaded on t_scale would turn the restriction into a reweighting.
  - `D5`: the diagnostic store kind.
  - Handoff defect `D6`: a rotation saved transposed (`tests/test_diagnostics.py:552`, `tests/test_conditioning_repair.py:665`, `:910`, `:925`, `tests/test_artifact_store.py:1388` "The D6 spec test, moved").
  - The "D6 trap" at `tests/test_diagnostics.py:883`: a setting is refused, never silently clamped.
  - `D7`: a cache one setting away is refused before any simulation (`tests/conftest.py:113`).
  - `D9`: a driven chi recording states its frequency in Hz.
  - `D12` (the message at `tests/test_conditioning_repair.py:1160`): a narrowing round whose parent is itself narrowed, and whose observation is not that parent's, is refused with no escape hatch and names no single control.
  - `V1`: a public stage works on a private copy and changes nothing it was handed.
  - `V3`: one Refusal kind with a field key.
  - `V4`: standard logging and a `log.txt` per record.
  - `V6`: one training preview behind the budget lines.
  - `V7`: the Fisher settings are recorded only when the rotation ran.
  - `V8`: a loaded prior closes its figure.
  - `V9`: sbi's summary writer is off, and nothing writes `sbi-logs`.
  - `E2`: an interrupted record keeps its folder.
  - Every other letter-plus-number (`B…`, `F…`, `H…`, `I…`, `M…`, `N…`, `N1a`, `P…`, `R…`, `S…`, `S-1`, `T…`, `X…`, `L<number>`) is a decision, ruling, review finding, task or line number. Delete it and keep the behaviour.
- **Wrong facts to fix (spec §2.5):**
  - `tests/test_user_sbi.py:1023`: "See test_gui_progress.test_worker_cancelled_passes_through_except_exception" becomes `tests/test_worker_dispatch.py::test_worker_cancelled_passes_through_except_exception` (it lives at `tests/test_worker_dispatch.py:139`).
  - `:2062`: "scripts/chi_mask_audit.py at 7433ced^ and every pre-C-11 call site" becomes "every caller that wants no cache (`analysis.gen_cal_data` among them)".
  - `:2233` says "OMITTED, never None, for an amortized one". The truth, per `core/artifacts/identity.py`'s module docstring, is that `truncation` is always present: None for an amortized run, and the region's fields for a truncated one. So a round at its parent's budget keys a directory of its own. The paragraph's "orphan all five complete ones on disk" reason is stale too. State the current constraint instead: the golden digest pins that the amortized identity does not move.
  - `:2253`: drop "(scripts/migrate_checkpoint_flags.py at e37df41^ is the precedent)". Keep the rest of that comment ("Update it only deliberately, with a migration…", "Belongs to training-rows/2 -- … "463e81d156cd""), which Task 17 edits later. The digest assertions at `:2256-2263` are code, so leave them untouched.
  - `tests/test_conditioning_repair.py:1041`: "see test_gui_progress" becomes "see the docstring of `tests/_fixtures.code_only`".
  - "sidecar" (`:910`, `tests/test_artifact_store.py:1388`) was where a rotation used to be stored; it is the manifest now. Write "stored rotation".
- **Other named sites:**
  - `tests/test_artifact_store.py:1880` names the directory `.superpowers/`. The scan matches the word itself, so describe it without the word, e.g. "the gitignored planning workspace".
  - `:4069`: keep the quoted core phrase "Running anyway (accepted)." and its reason, which is that a user reads it in the log pane. Drop "walkthrough row B3".
  - `tests/conftest.py:176` still names `test_the_d7_and_d8_dialogs_default_to_cancel`. Replace it with the name of the def at `tests/test_nav_and_gating.py:1280`, unless Task 1 already did.
- **Test names keep their names.** `test_the_B7_pair_shares_one_flag` (`tests/test_conditioning_repair.py:164`), `test_v1_params_migrate_to_placeholder_boxes` (`tests/test_user_models.py:391`) and `test_v2_params_migrate_to_a_linear_box` (`:404`) are allowlisted by exact name. Renaming one would leave its allowlist entry matching nothing, and `test_every_allowlist_entry_still_matches_something` would fail.
- **Misses the scan cannot report (spec §2.2).**
  - Sentence-initial "Fix round N" (the scan reports these too; a cross-check): `tests/test_artifact_store.py` `:655`, `:697`, `:836`, `:4669`, `:4714`, `:4734`, `:4752`, `:4779`; `tests/test_refusals.py:1135`.
  - "Review finding (task 3, …)" at `tests/test_artifact_store.py:4318`, and "the spec's ruling" at `:5098`.
  - The "Phase 2/Phase 4" dividers at `tests/test_conditioning_repair.py:281`, `:346` and `:462`: keep the topic.
  - "the design allows" at `tests/test_refusals.py:442`.
  - "icon set (B-e)" at `tests/test_user_models.py:1077`.

  Read every line of every file.
- **Tests that read raw source.**
  - `tests/test_diagnostics.py:154` and `:314`, and `tests/test_user_sbi.py:3241` and `:3551`, read core modules' source. Their string data ("SystemExit", "raise ValueError", "_GRAPH_CACHE", "SIM_VRAM_CEILING", …) is code: leave it untouched.
  - The snippet strings in `tests/test_refusals.py`'s matcher self-checks (near `:880-887`) are test data.
  - Never write `use_ini_file(` followed by `None)` in any test file.
- **Line numbers are hints** at `b24973e`. The quoted text is the anchor.
- **Process.** Never edit while a pytest run is in progress. Never run two pytest processes at once. Do not run the fast gate.

- [ ] **Step 1: Record the base commit, the whole collection and the list**

```bash
git rev-parse --short HEAD        # BASE
python -m pytest --collect-only -q | grep "::" > C:/Users/J/AppData/Local/Temp/prism-piece6/t9-collect-before.txt
python -m pytest tests/test_source_hygiene.py --collect-only -q | tail -1
```
Record the scan suite's test count. Confirm that `NOT_YET_CLEAN` is exactly this set: `{"tests/test_artifact_store.py", "tests/test_diagnostics.py", "tests/test_refusals.py", "tests/test_user_sbi.py", "tests/test_conditioning_repair.py", "tests/test_user_models.py", "tests/_fixtures.py", "tests/conftest.py"}`. If it holds anything else, stop and report. The whole suite is collected here because `tests/conftest.py` and `tests/_fixtures.py` serve every test.

- [ ] **Step 2: Write the list of permitted changes, before any edit**

```
tests/test_artifact_store.py|string|beside the kind directories (spec §2.1)
tests/test_artifact_store.py|string|creates NOTHING (spec §1.2)
tests/test_artifact_store.py|string|E2: an interrupted record keeps its folder
tests/test_artifact_store.py|string|a single-cell run has no preset (P72)
tests/test_artifact_store.py|string|this is what V1 buys run_fdt
tests/test_artifact_store.py|string|make_reduction_config is left alone (P19)
tests/test_diagnostics.py|string|exactly the R1 regression
tests/test_user_sbi.py|string|is back (trap CHI10)
tests/test_user_sbi.py|string|a fresh Fisher rotation (D1)
tests/test_user_sbi.py|string|the duplicate V4 removed
tests/test_user_sbi.py|string|C-11's resume would break
tests/test_user_sbi.py|string|snapshots it at import (X12)
tests/test_conditioning_repair.py|string|D12 has no hatch and no single control
tests/test_user_models.py|string|re-check the walkthrough
tests/test_user_models.py|string|(piece 4, B14)
tests/conftest.py|string|since piece 3
```
The new `tests/conftest.py:51` message keeps its `{logs}` field. For example: "… delete it (nothing in PRISM writes it any more), …".

- [ ] **Step 3: Empty the pending list and watch the scan fail**

Set `NOT_YET_CLEAN = frozenset()`, then run:

`python -m pytest tests/test_source_hygiene.py::test_no_file_outside_the_pending_list_cites_a_working_document -q`

Expected: FAIL. Every reported line names one of the eight files, about 385 lines in all.

- [ ] **Step 4: Clean `tests/test_artifact_store.py`**

Handle these sites:
- `:1` ("piece 1 of the 2026-09-10 hardening programme");
- `:1388`;
- `:1439`: spec §4.1 and walkthrough row C9;
- `:1458`: guardrail 2;
- `:1880`;
- `:2048`: the cost-on-screen rule, i.e. what the run will simulate is said once;
- `:3573`: guardrail 7;
- `:3665`: trap X5;
- `:4069`;
- the eight "Fix round" openings;
- `:4318` and `:5098`;
- the dense `V1` references (14), stated as the private-copy behaviour where they are the subject;
- the 6 messages.

- [ ] **Step 5: Clean `tests/test_diagnostics.py` and `tests/test_refusals.py`**

- `tests/test_diagnostics.py`:
  - `:2` ("piece 2 of the 2026-09-11 one-flow design, tasks T13-T18");
  - `:162`: trap X5;
  - `:193` ("Task 17, fix round 1, finding 2 (spec §3.7)");
  - `:226`: guardrail 8;
  - `:552`;
  - `:883`;
  - the review ids `I1`–`I3`, `M3`–`M12`, `N1a`/`N1b`, `S1`, `R1`, `R3`, `R4`;
  - the message at `:968`.
- `tests/test_refusals.py`:
  - `:1-2` (piece 3, hardening programme, design §3.1);
  - `:442`;
  - `:568` ("(CLAUDE.md)");
  - `:857` ("(whole-piece review, R8)");
  - `:866` ("Piece 4 converted…"): state the fact that the window's last three static boxes were converted;
  - `:1135`.

- [ ] **Step 6: Clean `tests/test_user_sbi.py`**

Handle these sites:
- `:371`: C-9/C-10. Say that `logcyc` left the Fisher channels because it duplicates `A3_log_fpeak`.
- the message at `:375`;
- `:1023`;
- the `# ── C-11: …` dividers at `:1774` and `:2071`: write "the training checkpoint (the simulation cache)";
- `:1873`, `:1921` and `:3185`;
- `:2062`, `:2233` and `:2253`;
- `:2291`: GUARDRAIL 7;
- `:2294`: Appendix A and D1;
- `:2564`: GUARDRAIL 8;
- `:3692` and `:4235`: walkthrough C9. Write "the [mem] lines reach the pane plain".
- `:4133` (CLAUDE.md): keep "training_checkpoint's module docstring" as the source of the rule;
- the messages at `:2328`, `:2547`, `:3224` and `:3371`.

- [ ] **Step 7: Clean `tests/test_conditioning_repair.py` and `tests/test_user_models.py`**

- `tests/test_conditioning_repair.py`:
  - `:281`, `:346` and `:462`;
  - `:395`: the eigenbasis rule;
  - `:623`: the observation-digest rule;
  - `:631`, `:703` and `:1177`: the region-carries-its-basis rule;
  - `:1093`: the eigenbasis rule;
  - `:1224`: the generous-region rule;
  - `:665` and `:908-925`: D6, "sidecar", "since piece 2". The needle line `needle = "parts[0]" + ".M"` is code.
  - `:938`;
  - `:1041`;
  - `:1273` and `:1292`: walkthrough rows;
  - the message at `:1160`.
- `tests/test_user_models.py`:
  - `:349` and `:487` (S-1);
  - `:1077`;
  - `:1173` ("the 2026-09-11 walkthrough, row 1");
  - `:1204`: the checklist path. Write "checked on the real screen".
  - the messages at `:1221` and `:1258`.

- [ ] **Step 8: Clean `tests/_fixtures.py` and `tests/conftest.py`**

- `tests/_fixtures.py`:
  - the module docstring at `:1-17` (piece 1, hardening programme, Task 7, Piece 3 (Task 9));
  - the `CODE_ROOTS` comment at `:29-36`: state that the tool's code lives under `core/tool`, so one root covers every scan;
  - `:48` and `:488`: delete the review ids and keep the reasons.
- `tests/conftest.py`:
  - `:36`: "(V9, spec §6.4)";
  - `:51`;
  - `:85`;
  - `:113`: "Under D7" becomes "Under the rule that refuses a cache one setting away before any simulation";
  - `:133`, `:176`, `:179` and `:181`.

- [ ] **Step 9: Do the manual pass**

Read every line of the eight files. Then run:

`grep -nE "[Ff]ix round|[Rr]eview finding|\bspec\b|\bpieces?\b|[Pp]hase [0-9]|\bB-[a-z]\b|test_gui_progress|sidecar|dialogs_default_to_cancel" <the eight files>`

Expected: nothing that cites a document or a deleted file.

- [ ] **Step 10: Run the scan suite, the checker and the collection**

```bash
python -m pytest tests/test_source_hygiene.py -q
python C:/Users/J/AppData/Local/Temp/prism-piece6/comment_only_check.py --base BASE --allow C:/Users/J/AppData/Local/Temp/prism-piece6/allow-task-09.txt tests/test_artifact_store.py tests/test_diagnostics.py tests/test_refusals.py tests/test_user_sbi.py tests/test_conditioning_repair.py tests/test_user_models.py tests/_fixtures.py tests/conftest.py
python -m pytest --collect-only -q | grep "::" > C:/Users/J/AppData/Local/Temp/prism-piece6/t9-collect-after.txt
diff C:/Users/J/AppData/Local/Temp/prism-piece6/t9-collect-before.txt C:/Users/J/AppData/Local/Temp/prism-piece6/t9-collect-after.txt
python -m pytest tests/test_settings_persistence.py::test_no_suite_points_the_settings_back_at_the_real_ini -q
```
Expected:
- every scan test passes, with `NOT_YET_CLEAN` empty and every scanned file clean;
- the checker exits 0 with only listed differences, and the f-string fields are unchanged;
- `diff` prints nothing;
- the needle test reports `1 passed`.

(The scan test's rename to `test_no_scanned_file_cites_a_working_document` happens in Step 12, in the commit that removes the list; Step 10's collection keeps the old name.)

- [ ] **Step 11: Commit the cleanup**

```bash
git add tests/test_artifact_store.py tests/test_diagnostics.py tests/test_refusals.py tests/test_user_sbi.py tests/test_conditioning_repair.py tests/test_user_models.py tests/_fixtures.py tests/conftest.py tests/test_source_hygiene.py
git commit -m "tests: the store, SBI and shared suites cite no working document" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 12: End the pending mechanism in `tests/test_source_hygiene.py`**

This is a code change with its own commit. The checker is not run on it.
- Delete the `NOT_YET_CLEAN` constant, which is now `frozenset()`, and the comment that explains it.
- Delete `test_every_pending_file_still_has_a_hit` whole.
- In `test_no_file_outside_the_pending_list_cites_a_working_document`, delete only the filter that skipped a path listed in `NOT_YET_CLEAN`. Everything else in the test stays as Task 1 wrote it: it walks `scanned_files()`, scans each file with `scan_python`/`scan_prose`, and fails with the report that names file, line, family and the allowlist to extend. That report is what `test_a_failure_names_the_file_line_family_and_the_allowlist` pins.
- Rename the test to `test_no_scanned_file_cites_a_working_document`, because its old name speaks of a pending list that no longer exists, and rewrite its docstring to say that every scanned file must be clean, with no mention of a list.
- Remove every other mention of the pending list: the module docstring, the failure report's hint and any comments. Use plain words and no labels.

The removed shape, whatever Task 1's exact spelling:
```python
NOT_YET_CLEAN: frozenset[str] = frozenset()
...
        if rel in NOT_YET_CLEAN:          # the skip inside the scan loop
            continue
...
def test_every_pending_file_still_has_a_hit(): ...
```

- [ ] **Step 13: Run the final full-scan check**

```bash
python -m pytest tests/test_source_hygiene.py -q
git grep -n -e NOT_YET_CLEAN -e test_every_pending_file_still_has_a_hit -- core tests conftest.py
python -m pytest --collect-only -q | grep "::" > C:/Users/J/AppData/Local/Temp/prism-piece6/t9-collect-final.txt
diff C:/Users/J/AppData/Local/Temp/prism-piece6/t9-collect-after.txt C:/Users/J/AppData/Local/Temp/prism-piece6/t9-collect-final.txt
```
Expected:
- every test passes, one test fewer than Step 1's count;
- `git grep` prints nothing;
- `diff` shows three lines: removed `tests/test_source_hygiene.py::test_every_pending_file_still_has_a_hit` and `tests/test_source_hygiene.py::test_no_file_outside_the_pending_list_cites_a_working_document`, added `tests/test_source_hygiene.py::test_no_scanned_file_cites_a_working_document`.

The scan now covers every file with no exemption. That is the state spec §2.8 ends in.

- [ ] **Step 14: Commit the end of the list**

```bash
git add tests/test_source_hygiene.py
git commit -m "tests: the reference scan has no pending list" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
The report gives:
- BASE;
- the contents of the allow file;
- the checker's output;
- the hit count from Step 3;
- both `diff` results;
- the Step 12 diff of `tests/test_source_hygiene.py`;
- every rewritten sentence at a colliding id (`D1`, `D3`, `D4`, `D5`, `D6`, `C1`/`C2`, `M1`–`M4`, `X1`–`X5`) with its file:line, for the reviewer (spec §2.3).

The controller starts the one-process fast gate (`python -m pytest -m "not slow" -q`, in the background, logged) at the same moment as this task's review. Do not run it yourself.

---

### Task 10: Help text: descriptions, placeholders, wordings, shared training flags

**Files:**
- Modify: `core/tool/config_args.py`: new `add_training_flags` beside `def add_resume_flags(p) -> None:` (~:60); the `--device` help in `add_config_flags` (`"auto detects CUDA; cpu forces config.cpu_device(); ..."`, ~:45-47); the `--note` help in `add_name_flags` (`help="free text recorded in the artifact's manifest"`, ~:56)
- Modify: `core/tool/stages.py`: train's flag block from `p.add_argument("--num-runs", type=int, default=None, metavar="N",` (~:53) through `--checkpoint-every` (~:72); tsnpe's copy (~:130-138); infer's `p.add_argument("--posterior", required=True, metavar="REF")` (~:95); `--f0-si ... metavar="N"` (~:109); `--level ... metavar="Q"` (~:127); every `subparsers.add_parser(...)`
- Modify: `core/tool/diagnostics.py`: `_add_probe_flags` (~:50-57), the rotation/laplace/jacobian blocks (~:66-101), ablation (~:118-129), sbc (~:135-151)
- Modify: `core/tool/smoke.py`: `register` (~:81-125). The `EPILOG` is not touched here
- Modify: `core/tool/fdt.py`: `register` (~:162-192), `_add_store_root` (~:139), `_add_fdt_knobs` (~:149-159), `_register_compare` (`metavar="VALUE"` twice, ~:319 and :322)
- Modify: `core/tool/browse.py`: `register` (~:555-606), descriptions only
- Test: `tests/test_tool.py`

The line hints are at `b24973e`. The cleanup tasks before this one rewrite comments in these files, so find each site by its quoted code.

**Interfaces:**
- Consumes: nothing from earlier tasks by name. `tests/test_source_hygiene.py` (the reference scan) must stay green over every string edited here.
- Produces: `core.tool.config_args.add_training_flags(p, *, fisher: bool) -> None`. In `tests/test_tool.py` it adds the module-level helpers `_parsers(parser=None, path=())` and `_leaves(parser=None)` and the constant `HOUSE_METAVARS`. Task 11 reuses all three.

**Binding (read before starting):**
- `python` below means `C:\Users\J\anaconda3\envs\biophys-env\python.exe` (in the Bash tool, `/c/Users/J/anaconda3/envs/biophys-env/python.exe`); the `python` on PATH is a different interpreter without torch or pytest. Commands written with backslash paths are PowerShell commands; in Bash use forward slashes.
- Spec §3.1: nothing changes behaviour. Flag names, `dest`, `default`, `type`, `choices`, `nargs`, `required` and `action` stay exactly as they are. Only these change: `help=`, `metavar=`, `description=`, and the order of `add_argument` calls. The knob-forwarding tests stay green without being edited.
- **Scope boundary with Task 11.** This task does not add, remove or reword any default clause, whatever its form: "(stage default 32)", "(default: config.CHI_MODE)", "Default: …", "(default 4)". It touches no epilog and no `_COMPARE_MODES` text. An existing clause stays word for word at the END of its help; move it there if your new words would otherwise follow it. A flag that has no clause today gets none. Task 11 rewrites every clause and every epilog.
- These sections bind the wording: spec §3.2 (the 15 help texts), §3.3 (placeholders), §3.4 (one wording per shared flag) and §3.7 (descriptions, plain words, where `--name`/`--note` go). So do `recon/design-helptidy.md` §1–§3 (under `.superpowers/sdd/2026-09-25-documentation-and-retrain-readiness/`). The exact texts are in Step 5.
- `--help` must stay torch-free. Nothing in `core/tool/` may import `core.config`, `core.cli`, torch, or anything that imports them, at module level. `core.refusals` is torch-free and may be imported, e.g. for `NOTE_MAX_CHARS`. The fresh-interpreter probes in `test_compare_is_a_nested_subcommand_whose_help_costs_no_torch` and `test_the_artifacts_family_takes_no_configuration_flags_and_its_help_costs_no_torch` stay as they are and must pass.
- argparse applies `%`-formatting to every help string, so a help may not contain a literal `%`.
- A parser built with `RawDescriptionHelpFormatter` prints its description exactly as typed: smoke, fdt, crossval, compare and artifacts. Keep each of their one-liners within 79 characters per line, and break a longer one with `\n` (the listing above it collapses the break).
- H4, the reference rule: nothing written into `core/` or `tests/` names a process document or label. That covers piece numbers, "§", decision ids, "Task N" and review ids. A docstring or test docstring gives its reason in words. `tests/test_source_hygiene.py` enforces this.
- Git: commit on `main`, `git add` each file explicitly, never amend. Never edit a source file while a pytest run is in progress. Run only focused tests; the controller runs the fast gate.

- [ ] **Step 1: Write the two failing tests and the walk helpers** in `tests/test_tool.py`. Add `import argparse` to the module imports, then:

```python
HOUSE_METAVARS = frozenset({"N", "X", "S", "K", "REF", "PATH", "NAME", "TEXT", "STAGE[,STAGE...]",
                            "PATH[@HZ]", "NAME=VALUE", ("MIN", "MAX", "N")})


def _parsers(parser=None, path=()):
    """Every parser under ``python -m core`` with its path as typed after it: the top level (path
    ""), each subcommand, and each mode of a subcommand that has modes."""
    parser = build_parser() if parser is None else parser
    yield " ".join(path), parser
    for a in parser._actions:
        if isinstance(a, argparse._SubParsersAction):
            for name, sub in a.choices.items():
                yield from _parsers(sub, (*path, name))


def _leaves(parser=None):
    """The parsers that run something: each subcommand without modes, and each mode."""
    for path, p in _parsers(parser):
        if path and not any(isinstance(a, argparse._SubParsersAction) for a in p._actions):
            yield path, p


def test_every_leaf_parser_has_a_description():
    """``<subcommand> --help`` and ``<subcommand> <mode> --help`` open with what the command does:
    every subcommand and mode parser carries a description, and it is the one-liner the listing
    above it shows, so the two never disagree."""
    seen = []
    for path, parser in _parsers():
        for a in parser._actions:
            if isinstance(a, argparse._SubParsersAction):
                listed = {ca.dest: ca.help for ca in a._choices_actions}
                for name, sub in a.choices.items():
                    where = f"{path} {name}".strip()
                    assert sub.description, f"`{where} --help` says nothing about what it does"
                    assert sub.description == listed[name], (where, sub.description, listed[name])
                    seen.append(where)
    assert {"prior", "smoke", "identifiability laplace", "compare cells", "artifacts sweep"} <= set(seen)


def test_every_value_flag_has_a_house_placeholder():
    """A flag that takes a value says what it is and shows what to type from the house set -- N a
    count, X a real number, S seconds, K the probe count, REF an artifact's name or id, PATH, NAME,
    TEXT -- or lists its choices; never the internal destination name argparse prints otherwise."""
    bad, no_help = [], []
    for path, leaf in _leaves():
        for a in leaf._actions:
            if not a.option_strings or a.nargs == 0:
                continue
            if not a.help:
                no_help.append(f"{path} {a.option_strings[0]}")
            if a.choices is None and a.metavar not in HOUSE_METAVARS:
                bad.append(f"{path} {a.option_strings[0]}: {a.metavar!r}")
    assert not no_help, "value flags that say nothing:\n" + "\n".join(no_help)
    assert not bad, "value flags without a house placeholder:\n" + "\n".join(bad)
    shown = {(path, a.option_strings[0]): a.metavar
             for path, leaf in _leaves() for a in leaf._actions if a.option_strings}
    assert shown[("infer", "--f0-si")] == "X", "a force in newtons, not a count"
    assert shown[("tsnpe", "--level")] == "X"
    assert shown[("compare renormalise", "--prefactor")] == shown[("compare sweeps", "--at")] == "X"
    assert shown[("sbc", "--chi-k-fixed")] == "K", "the chi probe count, as --chi-k shows it"
    assert shown[("smoke", "--stages")] == "STAGE[,STAGE...]"
```

- [ ] **Step 2: Run both tests and watch them fail.** `python -m pytest tests/test_tool.py::test_every_leaf_parser_has_a_description tests/test_tool.py::test_every_value_flag_has_a_house_placeholder -q`. Expected: 2 failed. The first fails with "`prior --help` says nothing about what it does". The second lists 15 value flags that say nothing: infer `--posterior`, tsnpe's seven training flags, `--posterior` on rotation, laplace and ablation, and `--cell`/`--t-obs` on laplace and jacobian.

- [ ] **Step 3: One definition of the training flags.** In `core/tool/config_args.py`, add `def add_training_flags(p, *, fisher: bool) -> None`. It adds, in this order, with today's `type`, `default=None`, `dest` and `metavar`: `--num-runs`, `--run-size` (dest `run_size_cap`), `--hidden-features`, `--num-transforms`, `--learning-rate`, `--stop-after-epochs`, `--max-epochs` (dest `max_num_epochs`); then `--fisher-m`, `--fisher-dz` and `--fisher-points` only when `fisher`; then `--checkpoint-every`. The help words are in Step 5. The docstring gives three reasons in words: train and tsnpe used to define these flags twice, and tsnpe's copy had no help; tsnpe passes `fisher=False` because a narrowing round reuses its parent's rotation, so a Fisher flag would be silently ignored; smoke does not use this helper, because its defaults are its own literals and its `--checkpoint` is an on/off switch. In `stages.py`, train calls `add_training_flags(p, fisher=True)` right after `--prior`, and tsnpe calls `add_training_flags(p, fisher=False)` right after `--level`. Delete both hand-written copies. `TRAIN_KNOBS` and `TSNPE_KNOBS` do not change. Run `python -m pytest tests/test_tool.py::test_every_train_flag_reaches_build_posterior_as_a_keyword tests/test_tool.py::test_every_validate_infer_and_tsnpe_flag_reaches_its_stage_as_a_keyword -q`. Expected: 2 passed.

- [ ] **Step 4: House placeholders** (spec §3.3; 38 sites, 49 rendered actions). Set `metavar=` as follows:

| file | site | flag → metavar |
|---|---|---|
| diagnostics.py | `_add_probe_flags` | `--m` N, `--m-noise` N, `--rel` X, `--min-valid` X, `--seed` N |
| diagnostics.py | rotation / laplace / jacobian | `--n-worst` N, `--top-n` N / `--n-points` N, `--sd-identified` X / `--zero-tol` X, `--noise-eps` X |
| diagnostics.py | ablation / sbc | `--rows` N, `--n-sweep` N / `--repeats` N, `--n-cal` N, `--posterior-samples` N, `--cal-n-scales` N, `--chi-k-fixed` K, `--seed` N |
| smoke.py | `register` | `--cell` PATH, `--t-obs` S, `--seed` N, `--stages` `STAGE[,STAGE...]`, `--num-runs` N, `--run-size` N, `--n-cal` N, `--max-epochs` N, `--store-root` PATH, `--prior` REF |
| fdt.py | `register`, `_add_store_root`, `_add_fdt_knobs` | fdt `--cell` PATH, fdt `--model` NAME, `--store-root` PATH, `--n-freqs` N, `--ensemble-m` N, `--freqs-per-batch` N, `--f0` X, `--seed` N, crossval `--cell` PATH |
| explicit changes | stages.py, fdt.py | `--f0-si` N→X; `--level` Q→X; `--prefactor` VALUE→X; `--at` VALUE→X |

- [ ] **Step 5: The words** (spec §3.2, §3.4, §3.7). Replace the words of each help below and leave every existing default clause at its end (see Binding).
  - **The 15 flags with no help today:**
    - infer `--posterior`: "posterior artifact to run, by name or id; its own training prior is loaded with it"
    - tsnpe's seven training flags: come from `add_training_flags`
    - rotation `--posterior`: "posterior artifact whose Fisher eigenbasis is decomposed, by name or id"
    - laplace `--posterior`: "posterior artifact whose training prior supplies the evaluation points beyond the ground truth, by name or id"
    - laplace `--cell`: "the cell file whose ground truth is the first evaluation point"
    - laplace `--t-obs`: "observation length, in seconds, the metric is measured at"
    - jacobian `--cell`: "the cell file whose ground truth the map is measured at"
    - jacobian `--t-obs`: "observation length, in seconds, the map is measured at; compare only maps made at the same length"
    - ablation `--posterior`: "posterior artifact whose flow is probed, by name or id; the first rows of its own simulation cache set each channel's range"
  - **`add_training_flags` (train and tsnpe):**
    - `--num-runs`: "training batches to simulate"
    - `--run-size`: "ceiling on simulations per training batch; 0 = the hardware batch"
    - `--hidden-features`: "flow width per transform"
    - `--num-transforms`: "flow depth"
    - `--learning-rate`: "Adam learning rate"
    - `--stop-after-epochs`: "early-stopping patience, in epochs"
    - `--max-epochs`: "hard ceiling on training epochs"
    - `--fisher-m`: "ensemble per latent perturbation for the Fisher rotation"
    - `--fisher-dz`: "latent central-difference step"
    - `--fisher-points`: "operating points the Fisher rotation is averaged over"
    - `--checkpoint-every`: "batches between checkpoint commits; 0 = no cache, nothing resumable"
  - **config_args:**
    - `--device`: "auto picks the card when CUDA is present with compute capability 8.0 or above, else Apple's MPS when present, else the CPU; cpu forces the CPU; cuda requires the card and is refused when it is absent or below 8.0"
    - `--note`: `f"free text recorded in the artifact's manifest: one line, at most {NOTE_MAX_CHARS} characters ('' = no note)"`, with `NOTE_MAX_CHARS` imported from `core.refusals`
  - **stages:**
    - prior `--sweep-batch`: "candidate parameter sets per sweep iteration; 0 = automatic"
    - prior `--min-samples`: "HDBSCAN's min_samples: neighbours a point needs to count as a cluster core; larger declares more points noise"
    - validate `--cal-n-scales`: "(t_scale, T_obs) operating points the calibration set is spread over"
    - infer `--t-obs`: "observation length, in seconds: the recordings' length, or the length the cell is re-simulated at"
    - infer `--f0-si`: "chi mode: the physical drive amplitude every probe was driven at, in newtons"
    - infer `--n-samples`: "posterior draws for the corner plot, the posterior predictive check and the summary"
    - tsnpe `--posterior`: "parent posterior artifact the region is measured in, by name or id"
  - **diagnostics:**
    - `_add_probe_flags` `--seed`: "the random seed for the whole run"
    - jacobian `--zero-tol`: "a parameter whose sensitivity, in units of the feature noise, has a norm below this carries no local information"
    - jacobian `--noise-eps`: "a feature channel whose spread across the ensemble is below this fraction of its size is treated as dead and zeroed in the map"
    - sbc's own help: "SBC repeated N times (--repeats) on one posterior (the run-to-run KS spread)"
    - sbc `--posterior`: "posterior artifact to calibrate repeatedly, by name or id"
    - sbc `--n-cal`: "calibration datasets to simulate per repeat"
    - sbc `--posterior-samples`: "posterior draws per calibration dataset"
    - sbc `--cal-n-scales`: "(t_scale, T_obs) operating points the calibration set is spread over"
    - sbc `--chi-k-fixed`: "hold the chi probe count at K instead of pooling over the training mixture; chi mode only. Run once per count and once pooled, then compare"
    - sbc `--seed`: "the random seed for the whole run; repeat r runs at seed + r"
  - **smoke:**
    - smoke's own help: `"every stage end to end at tiny sizes; run it on the card after changing code\nthat moves tensors"`
    - `--cell`: "the cell file whose ground truth the infer stage simulates"
    - `--t-obs`: "observation length, in seconds, of the observation the infer stage simulates"
    - `--seed`: "the random seed for the whole run"
    - `--stages`: `f"the stages to run, a comma-separated subset of {','.join(STAGES)}"`
    - `--num-runs`: "training batches to simulate"
    - `--run-size`: "ceiling on simulations per training batch; the prior sweep keeps the hardware batch"
    - `--n-cal`: "calibration datasets to simulate"
    - `--max-epochs`: "hard ceiling on training epochs"
    - `--store-root`: "the artifact store this run writes (prior, cache, posterior, observation, calibration, inference); reuse one, with --prior and --checkpoint, to resume", followed by the kept sentence "Default: a fresh temp directory, left on disk."
    - `--prior`: "prior artifact in --store-root to load instead of building, by name or id; a resume needs it, because the cache is keyed on the prior's fit and two fits of one box differ"
  - **fdt:**
    - fdt `--cell`: "the cell file whose ground truth the analysis runs at"
    - `--n-freqs`: "drive frequencies in the driven-response sweep"
    - `--f0`: "non-dimensional drive amplitude; keep it inside the linear regime"
    - `--seed`, whole text: "the random seed for the whole run, a whole number from 0; the record carries it, so the run can be repeated (default: draw one)"
    - crossval `--cell`: "a Nadrowski cell file whose ground truth the sweeps start from"
  - **Unchanged:** every other help, validate's `--posterior`, and infer's `--cell`, `--spont`, `--forced` and `--drive`.

- [ ] **Step 6: Descriptions and placement** (spec §3.7).
  - Every `add_parser` call gets `description=` with the same text as its `help=`. That means the 13 subcommands, the three parents (`identifiability`, `compare`, `artifacts`) and all 13 modes. Hold each one-liner in one variable per parser and pass it to both keywords, e.g. `text = "..."; p = subparsers.add_parser("prior", help=text, description=text)`. The compare modes pass `helptext` twice inside their existing loop.
  - Diagnostics: move each `add_name_flags(x)` to the line directly after `config_args.add_config_flags(x)`, in sbc, rotation, laplace, jacobian and ablation.
  - fdt: call `add_name_flags(fdt)` directly after `_add_store_root(fdt)`.
  - crossval: call `add_name_flags(cv, name_help=CROSSVAL_NAME_HELP)` directly after `_add_store_root(cv)`.
  - The stage commands and the compare modes already follow this rule.

- [ ] **Step 7: Run the two new tests and watch them pass.** Same command as Step 2. Expected: 2 passed.

- [ ] **Step 8: Focused tests for every file touched.** Run: `python -m pytest tests/test_tool.py::test_every_leaf_parser_has_a_description tests/test_tool.py::test_every_value_flag_has_a_house_placeholder tests/test_tool.py::test_the_help_epilog_names_the_core_environment_settings tests/test_tool.py::test_the_tool_reads_no_environment_and_writes_no_knob tests/test_tool.py::test_usage_errors_exit_2 tests/test_tool.py::test_every_prior_flag_reaches_build_prior_as_a_keyword tests/test_tool.py::test_every_train_flag_reaches_build_posterior_as_a_keyword tests/test_tool.py::test_every_validate_infer_and_tsnpe_flag_reaches_its_stage_as_a_keyword tests/test_tool.py::test_the_sbc_subcommand_forwards_every_knob_as_a_keyword tests/test_tool.py::test_identifiability_is_a_nested_subcommand_whose_modes_do_not_share_flags tests/test_tool.py::test_the_ablation_subcommand_forwards_its_knobs tests/test_tool.py::test_every_smoke_flag_reaches_its_stage_as_a_keyword tests/test_tool.py::test_fdt_and_crossval_declare_seed_and_store_root_by_name tests/test_tool.py::test_fdt_and_crossval_name_their_records_and_refuse_a_taken_name_before_any_folder tests/test_tool.py::test_the_crossval_help_states_every_rule_its_grids_are_held_to tests/test_tool.py::test_crossval_preset_choices_match_sweep_presets tests/test_tool.py::test_compare_is_a_nested_subcommand_whose_help_costs_no_torch tests/test_tool.py::test_the_rendered_help_counts_the_kinds_correctly tests/test_tool.py::test_the_artifacts_family_takes_no_configuration_flags_and_its_help_costs_no_torch tests/test_tool.py::test_a_sweep_claims_nothing_it_could_not_read_and_its_help_names_every_category tests/test_tool.py::test_artifacts_rm_offers_no_force_and_deletes_one_artifact tests/test_tool.py::test_the_artifacts_family_has_all_six_modes_and_still_no_configuration_flags tests/test_refusals.py::test_every_field_key_has_a_flag_and_every_key_literal_under_core_is_registered tests/test_refusals.py::test_every_registry_default_is_the_trees_own_default -q`. Then run `python -m pytest tests/test_source_hygiene.py -q`. Expected: all pass. If a help-text pin fails, change the new words so they carry the fact it pins (spec §3.8); never weaken the pin.

- [ ] **Step 9: Commit.** `git add core/tool/config_args.py core/tool/stages.py core/tool/diagnostics.py core/tool/smoke.py core/tool/fdt.py core/tool/browse.py tests/test_tool.py`, then `git commit -m "tool: help descriptions, placeholders and shared wordings" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"`. Report the commit hash. The controller starts the one-process fast gate (`python -m pytest -m "not slow" -q`, in the background, logged) at the same moment as this task's review. Do not run it yourself.

---

### Task 11: Help text: one convention for defaults, and the epilogs

**Files:**
- Modify: `core/refusals.py`
  - `def _default_clause(key: str) -> str:` (~:214) and its 12 call sites (~:224, :228, :245, :256, :268, :282, :299, :311, :321, :324, :347, :350)
  - `Field("max_num_epochs", ..., "2147483647")` (~:100)
  - `Field("min_valid", "the minimum valid fraction", None)` (~:180)
- Create: `core/tool/help_defaults.py`
- Modify: `core/tool/__init__.py`: `EPILOG` (~:25-36)
- Modify: the help strings (every default clause) in `core/tool/config_args.py`, `core/tool/stages.py`, `core/tool/diagnostics.py` and `core/tool/smoke.py`
- Modify: `core/tool/smoke.py`: also `EPILOG` (~:47-64)
- Modify: `core/tool/fdt.py`: `FDT_EPILOG`, `CROSSVAL_EPILOG`, `COMPARE_EPILOG`, `_COMPARE_MODES`, `_add_store_root`, `_add_fdt_knobs`, `--at`
- Modify: `core/tool/browse.py`: `EPILOG` (~:93-121) and the mode helps in `register`
- Test: `tests/test_tool.py`
- Test: `tests/test_refusals.py`:
  - `test_every_registry_default_is_the_trees_own_default`: the loop at ~:493-494 and `owned_by_a_signature` at ~:499
  - `test_refuse_appends_the_default_clause_to_the_callers_sentence_and_binds_the_field` (~:420)

The line hints are at `b24973e`; find each site by its quoted code.

**Interfaces:**
- Consumes:
  - `core.tool.config_args.add_training_flags(p, *, fisher: bool)` (Task 10)
  - `_parsers(parser=None, path=())` and `_leaves(parser=None)` in `tests/test_tool.py` (Task 10)
  - `core.tool.fields.FLAG` (existing)
- Produces:
  - In `core/refusals.py`: `default_clause(key: str) -> str`, the renamed `_default_clause` with behaviour unchanged and every caller updated; `default_text(value: str) -> str`, returning `" (default {value})"`; `FIELDS["max_num_epochs"].default == "no ceiling"`; `FIELDS["min_valid"].default == "0.5"`.
  - In `core/tool/help_defaults.py`: `DEFAULT_CLASS: dict[tuple[str, str], tuple[str, str | None]]`, keyed by `(leaf_path, option_string)` and mapping to `(cls, text)` with `cls ∈ {"value", "behaviour", "none"}`; `default_for(leaf_path: str, action) -> str | None`; `LEAVES`, a tuple of every leaf path.

**Binding (read before starting):**
- `python` below means `C:\Users\J\anaconda3\envs\biophys-env\python.exe` (in the Bash tool, `/c/Users/J/anaconda3/envs/biophys-env/python.exe`); the `python` on PATH is a different interpreter without torch or pytest. Commands written with backslash paths are PowerShell commands; in Bash use forward slashes.
- **Spec §3.5, three classes.**
  - (i) **value**: the help ends with `(default X)`, printed by `default_clause`/`default_text`, so the help and the refusal line print the same string. X comes from, in this order: (1) the parser's own default, when it is neither `None` nor `""`; (2) the literal the table pins for that (leaf, flag); (3) `FIELDS[key].default`, where the key is found by reading `core/tool/fields.py` `FLAG` in reverse.
  - (ii) **behaviour**: the help ends with `(default <phrase>)`.
  - (iii) **none**: no default clause. This covers required flags; flags that only name an input (`--bounds`, `--prior`, `--posterior`, `--observation`, `--cell`, `--spont`, `--forced`, `--drive`, `--f0-si`, `--record`, `--s-grid`, `--t-grid`, `--at`, `--prefactor`, `--out`, and `artifacts note --note`); and `--name`/`--note`, whose help says "unnamed" / "no note".
  - The table outranks FIELDS because `FIELDS["seed"]` carries fdt's wording ("none: one is drawn and recorded").
- `--help` never imports `core.config`. `help_defaults.py` imports only `core.refusals` and `core.tool.fields`, both torch-free. No tool module imports `core.config`/`core.cli` at module level. The fresh-interpreter probes (`test_compare_is_a_nested_subcommand_whose_help_costs_no_torch`, `test_the_artifacts_family_takes_no_configuration_flags_and_its_help_costs_no_torch`, and `test_the_module_is_torch_free_and_imports_only_the_standard_library` in test_refusals) stay as they are and must pass. `core/refusals.py` keeps importing only the standard library.
- Spec §3.5 "Also":
  - `max_num_epochs` prints "no ceiling" in both the help and the refusal line, with a named exception in `tests/test_refusals.py` at the ~:493-494 loop.
  - `FIELDS["min_valid"]` gains 0.5 (`identifiability_laplace(..., min_valid: float = 0.5, ...)`) and joins `owned_by_a_signature`.
  - Every default in these forms moves to the one form: `config.CHI_MODE`, `config.CHI_N_FREQS`, `config.cpu_device()`, `config.T_MIN_EXP_S`, "(stage default N)" (15 sites in diagnostics.py) and smoke's "Default: …".
- Spec §3.6, the epilogs, with the exact texts in Step 7:
  - The environment variables are stated once, at the top level.
  - Smoke's copy and its `--run-size` paragraph are dropped, and its what-to-watch paragraph becomes one line.
  - "like every subcommand but smoke" goes from fdt/crossval `--store-root` and from the artifacts epilog.
  - The compare epilog stops repeating the modes, and renormalise's mode help gains "drawn against the original".
  - The artifacts epilog keeps only the pinned facts ("all eight", "<kind> is one of …" in KINDS order), the no-configuration / no-`--store-root` consequence, the `<kind>`/`<ref>` definitions and the exit-0 rule. It types no number by hand.
  - The fdt epilog drops the clauses that restate its flags. The crossval epilog keeps its three grid rules (pinned by `test_the_crossval_help_states_every_rule_its_grids_are_held_to`) and moves its preset sentence into the flags' help.
  - Every epilog is wrapped at 79 columns.
- Spec §3.8 and Review Focus 4:
  - The new test computes each flag's source itself and compares exactly. It never accepts any of the three sources in place of the one the precedence selects.
  - A value flag missing from `DEFAULT_CLASS` fails the test, naming the leaf and the flag.
  - The help pins listed in §3.8 keep pinning the same facts: `tests/test_tool.py` :125, :324, :1519, :1667, :1756, :1766, :1882, :2833, :3101, :3406, :3192 and :3592. `test_the_rendered_help_counts_the_kinds_correctly` fails on any "all <word>" other than "all eight" in the artifacts help, so write no "all the …" there.
- Spec §3.1: nothing changes behaviour. Only help strings, epilogs and one private helper signature (`_add_fdt_knobs`) change. argparse applies `%`-formatting to help strings, so no literal `%`; the placeholder `%(default)s` is allowed.
- H4, the reference rule: new docstrings, comments and strings cite no process document or label. `tests/test_source_hygiene.py` enforces it.
- Git: commit on `main`, `git add` each file explicitly, never amend. Never edit a source file while a pytest run is in progress. Run only focused tests.

- [ ] **Step 1: Failing registry pins** in `tests/test_refusals.py`.
  - In `test_every_registry_default_is_the_trees_own_default`, replace the `owned_by_config` loop with the code below. Keep `"max_num_epochs"` inside `owned_by_config`, so `looked_at` still covers it:

```python
    for key, const in owned_by_config.items():
        if key == "max_num_epochs":
            continue                                    # the one named exception, just below
        assert FIELDS[key].default == str(getattr(config, const)), (key, const, FIELDS[key].default)
    # The epoch cap is the largest 32-bit integer, which the trainer reads as no cap at all; the
    # help and the refusal line say so in words rather than print the number.
    assert config.TRAINING_MAX_NUM_EPOCHS == 2**31 - 1
    assert FIELDS["max_num_epochs"].default == "no ceiling"
```

  - Add `"min_valid": str(_default(identifiability.identifiability_laplace, "min_valid")),` to `owned_by_a_signature`.
  - At the end of `test_refuse_appends_the_default_clause_to_the_callers_sentence_and_binds_the_field`, add:

```python
    from core.refusals import default_clause, default_text
    assert default_clause("num_runs") == default_text("5000") == " (default 5000)"
    assert default_clause("max_num_epochs") == " (default no ceiling)"
    assert default_clause("min_valid") == " (default 0.5)"
    assert default_clause("new_run") == ""
```

  Run `python -m pytest tests/test_refusals.py::test_every_registry_default_is_the_trees_own_default tests/test_refusals.py::test_refuse_appends_the_default_clause_to_the_callers_sentence_and_binds_the_field -q`. Expected: 2 failed, on the "no ceiling" assert and on `ImportError: cannot import name 'default_clause'`.

- [ ] **Step 2: The registry changes** in `core/refusals.py`.
  - Rename `_default_clause` to `default_clause(key: str) -> str`, keep its behaviour, and update all 12 callers in the module. It has no callers outside the module.
  - Add `default_text(value: str) -> str`, returning `f" (default {value})"`. `default_clause` returns `""` for a `None` default and `default_text(default)` otherwise.
  - Set `Field("max_num_epochs", "the maximum number of epochs", "no ceiling")`. Its comment says in words that `TRAINING_MAX_NUM_EPOCHS` is the largest 32-bit integer, so no ceiling is in effect.
  - Set `Field("min_valid", "the minimum valid fraction", "0.5")` with the comment `# identifiability_laplace`.
  - Run the Step 1 command, then all of `python -m pytest tests/test_refusals.py -q`. Expected: all pass.

- [ ] **Step 3: Commit the registry step.** `git add core/refusals.py tests/test_refusals.py`, then `git commit -m "refusals: public default clause; epoch cap reads no ceiling" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"`.

- [ ] **Step 4: Write the failing help-default tests** in `tests/test_tool.py`, below Task 10's helpers:

```python
def _value_actions(parser):
    """The options that take a value, plus --chi/--no-chi, whose default is a value too; the on/off
    switches and --help state no default."""
    for a in parser._actions:
        if a.option_strings and (a.nargs != 0 or isinstance(a, argparse.BooleanOptionalAction)):
            yield a


def _owned_defaults() -> dict:
    """``{(leaf, flag): value}`` for every default the help table pins, read from what owns it."""
    import dataclasses
    import inspect
    from core import cli
    from core.config import FDTConfig
    from core.diagnostics import identifiability, sbc

    def sig(fn, name):
        return str(inspect.signature(fn).parameters[name].default)

    def preset(field):
        values = {name: p[field] for name, p in cli.SWEEP_PRESETS.items()}
        if len(set(values.values())) == 1:
            return str(next(iter(values.values())))
        return "the preset's: " + ", ".join(f"{name} {v}" for name, v in values.items())

    fdt = {f.name: f.default for f in dataclasses.fields(FDTConfig)}
    lap, jac = identifiability.identifiability_laplace, identifiability.identifiability_jacobian
    return {
        ("sbc", "--seed"): sig(sbc.sbc_repeats, "seed"),
        ("sbc", "--n-cal"): sig(sbc.sbc_repeats, "n_cal"),
        ("sbc", "--posterior-samples"): sig(sbc.sbc_repeats, "num_posterior_samples"),
        ("identifiability laplace", "--seed"): sig(lap, "seed"),
        ("identifiability laplace", "--min-valid"): sig(lap, "min_valid"),
        ("identifiability laplace", "--sd-identified"): sig(lap, "sd_identified"),
        ("identifiability jacobian", "--seed"): sig(jac, "seed"),
        ("identifiability jacobian", "--min-valid"): sig(jac, "min_valid"),
        ("identifiability jacobian", "--zero-tol"): sig(jac, "zero_tol"),
        ("identifiability jacobian", "--noise-eps"): sig(jac, "noise_eps"),
        ("smoke", "--t-obs"): str(config.T_MIN_EXP_S),
        ("fdt", "--n-freqs"): str(fdt["n_freqs"]),
        ("fdt", "--ensemble-m"): str(fdt["ensemble_M"]),
        ("crossval", "--n-freqs"): preset("n_freqs"),
        ("crossval", "--ensemble-m"): preset("ensemble_M"),
    }


def _default_problems(parser) -> list:
    """One line per value flag whose help does not state its default the way its class requires."""
    from core.refusals import FIELDS, default_text
    from core.tool import fields as tool_fields
    from core.tool.help_defaults import DEFAULT_CLASS, default_for

    owned, chi_word = _owned_defaults(), ("--chi" if config.CHI_MODE else "--no-chi")
    problems = []
    for path, leaf in _leaves(parser):
        for a in _value_actions(leaf):
            opt = a.option_strings[0]
            where = f"{path} {opt}"
            if (path, opt) not in DEFAULT_CLASS:
                problems.append(f"{where}: no entry in core/tool/help_defaults.py DEFAULT_CLASS")
                continue
            cls, text = DEFAULT_CLASS[(path, opt)]
            shown = " ".join(leaf._get_formatter()._expand_help(a).split()) if a.help else ""
            if not shown:
                problems.append(f"{where}: no help")
                continue
            if cls == "none":
                want = None if text is None else problems.append(f"{where}: class none carries {text!r}")
            elif cls == "behaviour":
                if not text:
                    problems.append(f"{where}: a behaviour default needs its phrase")
                    continue
                want = default_text(text)
            elif cls == "value":
                if a.default not in (None, ""):
                    source, pinned = str(a.default), None
                elif opt == "--chi":
                    source = pinned = chi_word
                elif (path, opt) in owned:
                    source = pinned = owned[(path, opt)]
                else:
                    keys = [k for k, f in tool_fields.FLAG.items() if f == opt]
                    if len(keys) > 1:
                        keys = [k for k in keys if k == a.dest.lower()]
                    source, pinned = (FIELDS[keys[0]].default if len(keys) == 1 else None), None
                if source is None:
                    problems.append(f"{where}: a value default with no source; pin it in DEFAULT_CLASS")
                    continue
                if text != pinned:
                    problems.append(f"{where}: DEFAULT_CLASS says {text!r}, the code's default is {source!r}")
                want = default_text(source)
            else:
                problems.append(f"{where}: unknown class {cls!r}")
                continue
            if default_for(path, a) != want:
                problems.append(f"{where}: default_for gives {default_for(path, a)!r}, not {want!r}")
            count = shown.count("(default")
            if want is None and count:
                problems.append(f"{where}: states a default, but its class is none: {shown!r}")
            if want is not None and (count != 1 or not shown.endswith(want.strip())):
                problems.append(f"{where}: must end with {want.strip()!r}, stated once: {shown!r}")
            if a.dest == "name" and a.default == "" and "unnamed" not in shown:
                problems.append(f"{where}: must say what '' means (unnamed)")
            if a.dest == "note" and a.default == "" and "no note" not in shown:
                problems.append(f"{where}: must say what '' means (no note)")
    return problems


def test_every_value_flag_states_its_default_by_its_class():
    """Every flag that takes a value states its default one way: "(default X)" with the value the
    code really uses, "(default <phrase>)" where the default is a behaviour, or nothing for a
    required flag, an input and --name/--note. The value is computed here from what owns it -- the
    parser, the signature or constant the table pins, or the refusal registry through the flag's
    field key -- and compared exactly. A flag added without an entry in the table fails, naming the
    subcommand and the flag."""
    import inspect
    from core.diagnostics import identifiability
    from core.refusals import FIELDS
    from core.tool.help_defaults import DEFAULT_CLASS, LEAVES

    parser = build_parser()
    problems = _default_problems(parser)
    assert not problems, "\n".join(problems)
    walked = [path for path, _ in _leaves(parser)]
    assert sorted(LEAVES) == sorted(walked), sorted(set(walked) ^ set(LEAVES))
    present = {(path, a.option_strings[0]) for path, leaf in _leaves(parser) for a in _value_actions(leaf)}
    assert not set(DEFAULT_CLASS) - present, sorted(set(DEFAULT_CLASS) - present)
    # the registry pins these three against laplace only; the jacobian's help relies on them too
    jac = inspect.signature(identifiability.identifiability_jacobian).parameters
    for key in ("m", "m_noise", "rel"):
        assert str(jac[key].default) == FIELDS[key].default, key

    parser.subcommands["smoke"].add_argument("--brand-new", type=int, metavar="N", help="a new knob")
    named = [p for p in _default_problems(parser) if p.startswith("smoke --brand-new:")]
    assert named and "DEFAULT_CLASS" in named[0], _default_problems(parser)


def test_every_epilog_and_raw_description_fits_in_79_columns():
    """An epilog, and the description of a parser that prints it as typed, is wrapped by hand; a line
    wider than 79 columns wraps raggedly on an 80-column console."""
    wide = []
    for path, p in _parsers():
        texts = [p.epilog or ""]
        if p.formatter_class is argparse.RawDescriptionHelpFormatter:
            texts.append(p.description or "")
        wide += [f"{path or 'python -m core'}: {ln!r}" for t in texts for ln in t.splitlines() if len(ln) > 79]
    assert not wide, "\n".join(wide)
```

  Run `python -m pytest tests/test_tool.py::test_every_value_flag_states_its_default_by_its_class tests/test_tool.py::test_every_epilog_and_raw_description_fits_in_79_columns -q`. Expected: 2 failed. The first fails with `ModuleNotFoundError: core.tool.help_defaults`. The second lists the over-wide epilog lines (56 of today's 82).

- [ ] **Step 5: Create `core/tool/help_defaults.py`.**
  - The module docstring states the three classes, the precedence and why the table outranks the registry, in words. It also says the module imports nothing that costs torch, because `--help` must not import `core.config`.
  - Imports: `from core.refusals import default_clause, default_text` and `from .fields import FLAG`.
  - Constants: `VALUE, BEHAVIOUR, NONE = "value", "behaviour", "none"`.
  - Build the table from small shared groups, then per-leaf dicts:

```python
_CONFIG = {"--bounds": (NONE, None),                       # add_config_flags
           "--model": (BEHAVIOUR, "the bounds file's parent folder, upper-cased"),
           "--chi": (VALUE, "--no-chi"), "--chi-k": (VALUE, None), "--device": (VALUE, None)}
_NAMES = {"--name": (NONE, None), "--note": (NONE, None)}
_TRAINING = {f: (VALUE, None) for f in ("--num-runs", "--run-size", "--hidden-features",
             "--num-transforms", "--learning-rate", "--stop-after-epochs", "--max-epochs",
             "--checkpoint-every")}
_FISHER = {f: (VALUE, None) for f in ("--fisher-m", "--fisher-dz", "--fisher-points")}
_RESUME = {"--resume": (VALUE, None)}
_PROBE = {"--m": (VALUE, None), "--m-noise": (VALUE, None), "--rel": (VALUE, None),
          "--min-valid": (VALUE, "0.5"), "--seed": (VALUE, "0")}
_FDT_KNOBS = {"--store-root": (BEHAVIOUR, "the artifacts root"), "--freqs-per-batch": (VALUE, None),
              "--f0": (VALUE, None), "--seed": (VALUE, None)}
_COMPARE = {"--record": (NONE, None), **_NAMES}
# _PER_LEAF = {"prior": {**_CONFIG, **_NAMES, <7 prior knobs>: (VALUE, None)}, ...}  (table below)
LEAVES = tuple(_PER_LEAF)
DEFAULT_CLASS = {(leaf, flag): entry for leaf, flags in _PER_LEAF.items() for flag, entry in flags.items()}
```

  The per-leaf entries, in build order. Every flag not listed under a leaf's own entries comes from its groups. `(V)` means `(VALUE, None)`: the parser default or FIELDS supplies the value.

| leaf | groups | own entries |
|---|---|---|
| prior | _CONFIG, _NAMES | `--num-iterations`, `--sweep-batch`, `--max-sets`, `--walk-step`, `--stability-units`, `--min-cluster-size`, `--min-samples` (V) |
| train | _CONFIG, _NAMES, _TRAINING, _FISHER, _RESUME | `--prior` NONE |
| validate | _CONFIG, _NAMES | `--posterior` NONE; `--n-cal`, `--cal-n-scales`, `--posterior-samples` (V) |
| infer | _CONFIG, _NAMES | `--posterior`, `--cell`, `--spont`, `--t-obs`, `--forced`, `--drive`, `--f0-si` NONE; `--n-samples` (V) |
| tsnpe | _CONFIG, _NAMES, _TRAINING, _RESUME | `--posterior`, `--observation` NONE; `--directions`, `--level` (V) |
| sbc | _CONFIG, _NAMES | `--posterior` NONE; `--repeats`, `--cal-n-scales` (V); `--n-cal` (VALUE, "2000"); `--posterior-samples` (VALUE, "1000"); `--chi-k-fixed` (BEHAVIOUR, "pooled over the training mixture"); `--seed` (VALUE, "0") |
| identifiability rotation | _CONFIG, _NAMES | `--posterior` NONE; `--n-worst`, `--top-n` (V) |
| identifiability laplace | _CONFIG, _NAMES, _PROBE | `--posterior`, `--cell`, `--t-obs` NONE; `--n-points` (V); `--sd-identified` (VALUE, "0.3") |
| identifiability jacobian | _CONFIG, _NAMES, _PROBE | `--cell`, `--t-obs` NONE; `--zero-tol` (VALUE, "0.05"); `--noise-eps` (VALUE, "1e-06") |
| ablation | _CONFIG, _NAMES | `--posterior` NONE; `--rows`, `--n-sweep` (V) |
| smoke | _CONFIG, _RESUME | `--cell`, `--prior` NONE; `--t-obs` (VALUE, "1.0"); `--seed`, `--stages`, `--num-runs`, `--run-size`, `--n-cal`, `--max-epochs` (V); `--store-root` (BEHAVIOUR, "a fresh temporary directory, left on disk") |
| fdt | _NAMES, _FDT_KNOBS | `--cell` NONE; `--model` (BEHAVIOUR, "the cell file's parent folder, upper-cased"); `--n-freqs` (VALUE, "60"); `--ensemble-m` (VALUE, "256") |
| crossval | _NAMES, _FDT_KNOBS | `--cell`, `--s-grid`, `--t-grid` NONE; `--preset` (V); `--n-freqs` (VALUE, "the preset's: exploratory 30, production 60"); `--ensemble-m` (VALUE, "256") |
| compare cells / compare repeats | _COMPARE | none |
| compare renormalise / compare sweeps | _COMPARE | `--prefactor` NONE / `--at` NONE |
| artifacts list / show / rm / sweep | (empty dict: positionals only) | none |
| artifacts note / artifacts summary | none | `--note` NONE / `--out` NONE |

  The rule for completing the table: every value action of every leaf (`nargs != 0`, plus `BooleanOptionalAction`) has exactly one entry. The total is 201 today. A required flag or an input is NONE. A flag whose parser default is neither None nor `""`, or whose FIELDS key holds the true value, is `(VALUE, None)`. Only a value FIELDS would state wrongly gets a pinned literal, and that literal must also appear in the test's `_owned_defaults`.

  `default_for(leaf_path: str, action) -> str | None` applies spec §3.5's precedence and returns the whole clause, leading space included (`" (default X)"`), or None:
  - Look up `DEFAULT_CLASS[(leaf_path, action.option_strings[0])]`. A missing entry raises `KeyError` naming the leaf and the flag.
  - `none` returns None. `behaviour` returns `default_text(text)`.
  - `value` returns `default_text(str(action.default))` when that default is neither None nor `""`; else `default_text(text)` when the table pins a literal; else `default_clause(key)`.
  - To find `key`: take the keys whose `FLAG` entry is the option string. When there are several, keep the one equal to `action.dest.lower()`. Anything other than exactly one key, or a key with no default, raises `LookupError` naming the leaf and the flag.

  Run the Step 4 command. Expected: the first test still fails, now listing help-clause mismatches such as `prior --num-iterations: must end with '(default 50)'`.

- [ ] **Step 6: Every help's default clause.** Replace each existing clause with the expression below, placed at the end of the help. Import `default_clause`/`default_text` from `core.refusals`.
  - config_args:
    - `--model`: `"model name" + default_text("the bounds file's parent folder, upper-cased")`
    - `--chi`: `default_text("--no-chi")`
    - `--chi-k`: `default_clause("chi_n_freqs")`
    - `--device`: `default_text("%(default)s")`
    - `--resume`: `default_clause("resume")`
    - every `add_training_flags` flag: `default_clause(<its key>)` (`run_size_cap`, `max_num_epochs`, …)
  - stages:
    - the prior knobs, validate `--n-cal`/`--cal-n-scales`/`--posterior-samples`, infer `--n-samples` and tsnpe `--directions`/`--level`: `default_clause(<key>)` (`hpd_level` for `--level`, `n_directions` for `--directions`)
  - diagnostics:
    - `--m`, `--m-noise`, `--rel`, `--n-worst`, `--top-n`, `--n-points`, `--rows`, `--n-sweep`, `--repeats` and sbc `--cal-n-scales`: `default_clause(<key>)`
    - `--min-valid`: `default_text("0.5")`; both `--seed`: `default_text("0")`
    - `--sd-identified`: `default_text("0.3")`; `--zero-tol`: `default_text("0.05")`; `--noise-eps`: `default_text("1e-06")`
    - sbc `--n-cal`: `default_text("2000")`; `--posterior-samples`: `default_text("1000")`; `--chi-k-fixed`: `default_text("pooled over the training mixture")`
  - smoke:
    - `--t-obs`: `default_text("1.0")`
    - `--seed`, `--stages`, `--num-runs`, `--run-size`, `--n-cal`, `--max-epochs`: `default_text("%(default)s")`
    - `--store-root`: drop "Default: a fresh temp directory, left on disk." and append `default_text("a fresh temporary directory, left on disk")`
  - fdt:
    - fdt `--model`: `"model name" + default_text("the cell file's parent folder, upper-cased")`
    - `_add_store_root`: `"the artifact store this run writes its record into" + default_text("the artifacts root")`
    - `_add_fdt_knobs` becomes `_add_fdt_knobs(p, *, sweep: bool) -> None`. `--n-freqs` gets `default_text("the preset's: exploratory 30, production 60" if sweep else "60")` and `--ensemble-m` gets `default_text("256")`. `--freqs-per-batch` and `--f0` get `default_clause(<key>)`. `--seed` gets `default_clause("seed")`, replacing "(default: draw one)". fdt passes `sweep=False` and crossval passes `sweep=True`.
    - crossval `--preset`: `"resolution preset" + default_text("%(default)s")`
    - `--at`: "the operating point to slice both sweeps at; without it, the middle of the range they share" (class none, no clause)

- [ ] **Step 7: The epilogs and mode helps.** Replace each constant with exactly this text. Build browse's kinds line so it reads in `KINDS` order.

`core/tool/__init__.py` `EPILOG`:
```
environment -- the only variables PRISM reads; none stands in for a flag:
  PRISM_RESOURCES         the inputs root: Bounds/ Cells/ Units/ Models/
                          (default <repo>/Resources)
  PRISM_ARTIFACTS         the artifacts root (default <repo>/Artifacts). Every
                          subcommand writes here except smoke, which makes a
                          fresh temporary root for each run; smoke, fdt and
                          crossval also take --store-root, naming another root
  PRISM_VRAM_CEILING_GIB  GiB one simulation batch may plan to occupy
                          (0 = automatic); read afresh for every batch plan
  PRISM_MEM_LOG_EVERY     batches between memory log lines; read once, when the
                          simulation pipeline is first imported
The last two are read by the simulation pipeline, never by this tool, and are
deliberately not flags: they change the memory plan for a batch, not the rows
it produces.
```
`smoke.py` `EPILOG`:
```
What to watch in its output, and why it shows that the chain runs but not that
it is calibrated: docs/guide/command-line.md#smoke
A seeded run is not bitwise-reproducible on CUDA or across devices.
```
`FDT_EPILOG`:
```
A model FDT cannot run is refused with the reason: FDT drives the observable
itself, so a user model needs additive, non-zero observable noise and no
intrinsic forcing.

The run writes one fdt record: its figures (PSD, chi components, T_eff/T, the
spontaneous trajectory), its numbers in data.h5, its settings and seed, and
its log.
```
`CROSSVAL_EPILOG`:
```
Two sweeps probe FDT restoration on the Nadrowski model (the model is fixed):
the S sweep holds T_a/T = 1 and varies S (FDT restored as S -> 0), the T sweep
holds S = 0 and varies T_a/T (restored as T_a/T -> 1). Each grid is MIN MAX N:
N must be a whole number of at least 2, MIN must be below MAX, and the T_a/T
grid's MIN may not be below 0 (a negative temperature ratio is unphysical).

--preset sets the resolution the flags do not expose: the drive frequency band,
the drive window and the spontaneous recording length. Each sweep writes its
own fdt record -- its data.h5, its 3-D plot and its point counts -- so the S
sweep is a finished, readable answer before the T sweep starts.
```
`COMPARE_EPILOG`:
```
Every mode draws saved records from the artifact store and writes a comparison
record of its own (kind fdt, study "comparison"): its figures are its output
and its data.h5 holds the common grid and the interpolated curves. Nothing is
simulated, no record it draws is modified, and an unfinished record is
refused, naming it.

Runs land on different frequencies (each detects its own resonance), so curves
are interpolated onto a grid log-spaced over the intersection of their spans,
and a point whose bracketing samples include a blank stays blank rather than
being drawn through.
```
`browse.py` `EPILOG`:
```
Reads the artifacts root PRISM_ARTIFACTS names. There is no --store-root and
there are no configuration flags: a listing must not be able to fail on a
bounds file it does not need, and --help here costs no torch import. So this
family cannot tell you whether a posterior matches your bounds file; loading
it does (python -m core validate --posterior ...).

<kind> is one of prior, simulation, posterior, observation, calibration,
inference, diagnostic, fdt; list and sweep with no kind cover all eight.
<ref> is an artifact's name or its id. An empty listing exits 0: a script must
be able to tell "nothing on disk" from "you asked for something wrong".
```
  The mode helps change too; each description follows its help, because they share one variable.
  - `_COMPARE_MODES` renormalise: "one run's ratio recomputed with a supplied normalisation constant, drawn against the original".
  - artifacts `list`: "one line per artifact of a kind, or, with no kind, of all eight under headings".
  - artifacts `rm`: "delete one artifact; there is no --force, so one that anything depends on is refused until its children are deleted".
  - artifacts `summary`: "the lineage report: this artifact, then its parents, oldest last".
  - `show`, `note` and `sweep` keep their help.

- [ ] **Step 8: Run the new tests and watch them pass.** Same command as Step 4. Expected: 2 passed.

- [ ] **Step 9: Focused tests for every file touched.** Run Task 10's Step 8 node list with these two added: `tests/test_tool.py::test_every_value_flag_states_its_default_by_its_class tests/test_tool.py::test_every_epilog_and_raw_description_fits_in_79_columns`. Then run `python -m pytest tests/test_refusals.py -q` and `python -m pytest tests/test_source_hygiene.py -q`. Expected: all pass. If a §3.8 help pin fails, reword so the same fact is stated; do not loosen the pin.

- [ ] **Step 10: Commit.** `git add core/tool/help_defaults.py core/tool/__init__.py core/tool/config_args.py core/tool/stages.py core/tool/diagnostics.py core/tool/smoke.py core/tool/fdt.py core/tool/browse.py tests/test_tool.py`, then `git commit -m "tool: one default convention in the help; shorter epilogs" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"`. Report both commit hashes. The controller starts the one-process fast gate (`python -m pytest -m "not slow" -q`, in the background, logged) at the same moment as this task's review. Do not run it yourself.

---

### Task 12: The tier-1 path, tested

**Files:**
- Create: `tests/test_tier1.py`
- Modify: `tests/_fixtures.py`. It receives `_tiny_nadrowski_gen_prior`, moved byte for byte from `tests/test_user_sbi.py`. Put it directly after `def _tiny_gen_prior(` (near :283).
- Modify: `tests/test_user_sbi.py`. Delete `def _tiny_nadrowski_gen_prior(` (near :157). Add the name to the existing line `from tests._fixtures import _tiny_gen_prior, code_only` (near :38). The name is used 5 times in that file.
- Modify: `pytest.ini` (the `slow:` marker line)

**Interfaces:**
- Consumes: nothing from earlier tasks. Task 1's `tests/test_source_hygiene.py` scans the new file.
- Produces, as module-level names in `tests/test_tier1.py` that Tasks 13–19 extend:
  - `tier1_cfg(*, chi: bool = True, hw=None) -> SimConfig`
  - `force_scale_spy(monkeypatch) -> list`
  - Additions to the skeleton's registry: `with_truth(cfg, *, amp: float = 3.0, freq: float = 0.05) -> SimConfig`, `derived_force_scale(cfg) -> float`, `stand_in_gen_obs(...)` (the signature of `pipeline.gen_obs`), `_narrow_prior(cfg, rel: float = 1e-4)`, `_assert_every_drive_used_the_derived_scale(spy, f, rtol=2e-3)`, `_close(title, fig)`
  - `tests._fixtures._tiny_nadrowski_gen_prior`

**Binding (read before starting):**
- Spec §5.1 is binding; read it. These tests pin wiring that exists today, so they are expected to pass.
  - If a test fails because of **production** code, that is a readiness defect. Report it, fix it in this task (the failing test is its test-first proof), and name it in the task report.
  - If a test fails because of the **stand-in** (non-finite stand-in features, a shape the stand-in did not provide), fix the stand-in, never the production code.
  - `identifiability jacobian`/`laplace` are known to be wrong on this box. That is Task 13's work: do not test them here.
- The tier-1 box is `config.BOUNDS_PATH / "nadrowski" / "master_tier1.txt"`: rescale `x_scale, t_scale, T`, with T in (280, 310). Its cell is `config.CELL_PATH / "nadrowski" / "master_spont_tier1.txt"`: n=50, beta=14.1, x_scale=62.14, t_scale=3.73, T=300, and an all-zero drive.
- The relation is `f_scale = n * beta * k_B * T / x_scale`, with `k_B = 1.380649e-2` pN·nm/K in the nm/ms/pN/kHz units. It gives **46.99** at the cell.
  - The silent fallback `x_scale / t_scale` gives 16.66. A test that sees 16.66 has found the fallback.
  - The targets (theta) keep **T**. Only simulations see the derived scale.
- Build every path through `config.BOUNDS_PATH` / `config.CELL_PATH`. No literal `Resources/` path: a test pins it.
- Cheap by design. The fast gate has about 78 s of headroom for the whole piece.
  - Every fast test replaces `pipeline.gen_obs` with the stand-in, keeps recordings at 1–2 s, and keeps batches at 2 × 4 rows.
  - Target under ~3 s per test. Record the focused run's `--durations=0` in the task report.
- H4 applies to everything written into `tests/` and `pytest.ini`:
  - no piece numbers, §N, decision/review/row ids, "Task N", trap ids, or the words walkthrough/ruling/handoff in names, docstrings, comments or messages;
  - no capital-letter-plus-digits token (the scan's id family) in any comment or string;
  - no `t1`/`x2`-shaped token between underscores in a test name. `tier1` is fine.
- Interpreter: `C:\Users\J\anaconda3\envs\biophys-env\python.exe`, written `python` below. The root conftest sets `KMP_DUPLICATE_LIB_OK` and `QT_QPA_PLATFORM`.
- Never edit a source file while a pytest run is in progress. Never run two pytest processes at once.

- [ ] **Step 1: Move the tiny Nadrowski prior stub.**
  - Cut `_tiny_nadrowski_gen_prior` (the whole def, `**_kw` included) from `tests/test_user_sbi.py` and paste it unchanged into `tests/_fixtures.py`.
  - Import it in `test_user_sbi.py` beside `_tiny_gen_prior`.
  - Check: `python -m pytest tests/test_user_sbi.py --collect-only -q` collects with no error.

- [ ] **Step 2: Write the helpers at the top of `tests/test_tier1.py`.** Imports: `warnings`, `from types import SimpleNamespace`, `pytest`, `torch`, `from matplotlib import pyplot as plt`, `from core import cli, config, orchestrator, registry`, `from core.SBI import pipeline, reparam, truncate`, `from core.SBI import training_checkpoint as tc`, `from core.SBI.run_guards import _log_params_for`, and `from tests._fixtures import _FakeDP, _nad_cfg, _prior_artifact`. Define `_close(title, fig)` as `plt.close(fig)`.
  - `tier1_cfg`: `cli.make_sim_config("NADROWSKI", config.VALID_LABELS[config.VALID_MODELS.index("NADROWSKI")], registry.state_dep_drift("NADROWSKI"), str(config.BOUNDS_PATH / "nadrowski" / "master_tier1.txt"), chi_mode=chi, hw=hw or config.cpu_device())`.
  - `with_truth`: `cli.load_and_validate_gt(cfg, <the tier-1 cell>)`, then set `force_params_dict["amp"]` and `["freq"]` to `(amp, bounds)` and `(freq, bounds)`. The cell ships a zero drive, and a forced run needs a real one for the force scale to reach the trace. Return `cfg`.
  - `derived_force_scale`: `n * beta * cfg.k_b_cell * T / x_scale`, read from `params_dict` and `rescale_params` by name.
  - `_narrow_prior`: `sbi.utils.BoxUniform(v * (1 - rel), v * (1 + rel))`, with `v = torch.tensor(cfg.ground_truth)`. Every truth value of the cell is positive.
  - The spy and the stand-in fix behaviour, so write them exactly:

  ```python
  def force_scale_spy(monkeypatch) -> list:
      """Record the force scale every drive is divided by. Every sinusoidal and user drive goes
      through forcing.build_nondim_force_tensor. One record per call:
      {"declared": the index names f_scale, "f_scale": (B,) float64 on the CPU, "index": the index}.
      When nothing is declared, the record holds x_scale / t_scale, the builder's fallback."""
      from core import forcing
      real, seen = forcing.build_nondim_force_tensor, []

      def spy(forcing_params, t_nd, rescale_params, forcing_idx, rescale_idx, *a, **k):
          if "f_scale" in rescale_idx:
              f = rescale_params[:, rescale_idx["f_scale"]]
          else:
              f = rescale_params[:, rescale_idx["x_scale"]] / rescale_params[:, rescale_idx["t_scale"]]
          seen.append({"declared": "f_scale" in rescale_idx, "index": dict(rescale_idx),
                       "f_scale": f.detach().double().cpu().clone()})
          return real(forcing_params, t_nd, rescale_params, forcing_idx, rescale_idx, *a, **k)

      monkeypatch.setattr(forcing, "build_nondim_force_tensor", spy)
      return seen


  def stand_in_gen_obs(model=None, params=None, t=None, inits=None, force=None, n_segs=None,
                       steady_idx=0, fixed_dict=None, state_dep_drift=False, batch_size=1,
                       var_idx=None, dtype=torch.float32, device=torch.device("cpu")):
      """A cheap stand-in for pipeline.gen_obs with its shape: the drive's first channel as the
      response, a unit oscillation at angular frequency 1 in ND time (so the passive trace has a
      spectral peak), and a little noise from torch's global stream."""
      n = t.shape[0] - steady_idx
      x = (force[:, 0, steady_idx:].to(dtype).expand(batch_size, n) + torch.sin(t[steady_idx:]).to(dtype)
           + 0.01 * torch.randn(batch_size, n, dtype=dtype, device=device))
      out = torch.zeros((1 if var_idx is not None else inits.shape[-1], batch_size, n), dtype=dtype,
                        device=device)
      out[0] = x
      return out


  def _assert_every_drive_used_the_derived_scale(spy, f, rtol=2e-3):
      assert spy, "no drive was built"
      for e in spy:
          assert e["declared"] and "T" not in e["index"], e["index"]
          assert torch.allclose(e["f_scale"], torch.full_like(e["f_scale"], f), rtol=rtol), e["f_scale"]
  ```

- [ ] **Step 3: Write the parsing test.**

  ```python
  def test_the_tier1_box_parses_to_temperature_in_place_of_the_force_scale():
      from core.SBI import derived
      cfg = tier1_cfg()
      assert list(cfg.rescale_params) == ["x_scale", "t_scale", "T"]
      assert derived.uses_derived_f_scale(cfg.rescale_idx)
      assert cfg.sim_rescale_idx == {"x_scale": 0, "t_scale": 1, "f_scale": 2}
      nd_idx, k_b = cfg.tier1_args
      assert nd_idx == cfg.nd_idx and {"n", "beta"} <= set(nd_idx)
      assert k_b == pytest.approx(1.380649e-2, rel=1e-9)
      assert cfg.observation_mode == "chi" and tier1_cfg(chi=False).observation_mode == "forced"
      assert derived_force_scale(with_truth(tier1_cfg())) == pytest.approx(46.99, abs=0.01)
      master = _nad_cfg()
      assert master.tier1_args == (None, None) and master.sim_rescale_idx == master.rescale_idx
  ```

- [ ] **Step 4: Write the training-rows and calibration-set tests.**

  ```python
  @pytest.mark.parametrize("chi", [True, False])
  def test_training_rows_keep_temperature_while_every_simulation_gets_the_derived_force_scale(monkeypatch, chi):
      cfg = with_truth(tier1_cfg(chi=chi))
      monkeypatch.setattr(pipeline, "gen_obs", stand_in_gen_obs)
      spy = force_scale_spy(monkeypatch)
      x, theta = pipeline.gen_training_data(
          cfg.model, _narrow_prior(cfg), orchestrator.build_forcing_prior(cfg), cfg.t, 4, 2,
          cfg.steady_idx, cfg.dt_nd_min, len(cfg.params_dict), cfg.forcing_idx, cfg.rescale_idx,
          dt_exp=cfg.dt_exp, t_min_exp=cfg.t_min_exp, t_max_exp=2 * cfg.t_min_exp,
          t_scale_bounds=cfg.t_scale_bounds, chi_mode=chi, chi_f0=cfg.chi_f0,
          chi_freq_bounds=cfg.chi_freq_bounds, chi_k_pad=cfg.chi_k_pad, chi_max_cycles=cfg.chi_max_cycles,
          n_vars=3, nd_idx=cfg.tier1_args[0], k_b_cell=cfg.tier1_args[1],
          dtype=cfg.hw.dtype, device=cfg.hw.device)
      i_T = len(cfg.params_dict) + cfg.rescale_idx["T"]
      assert theta.shape[1] == 13 and theta.shape[0] > 0
      assert torch.allclose(theta[:, i_T], torch.full_like(theta[:, i_T], 300.0), rtol=1e-3)
      _assert_every_drive_used_the_derived_scale(spy, derived_force_scale(cfg))


  def test_the_calibration_set_simulates_at_the_derived_force_scale(monkeypatch):
      cfg = with_truth(tier1_cfg())
      cfg.t_max_exp = 2 * cfg.t_min_exp                      # a test config: keep the recordings short
      monkeypatch.setattr(pipeline, "gen_obs", stand_in_gen_obs)
      spy = force_scale_spy(monkeypatch)
      x_cal, theta = orchestrator._draw_calibration_set(
          cfg, _narrow_prior(cfg), None, orchestrator.build_forcing_prior(cfg), n_cal=8, cal_n_scales=2)
      i_T = len(cfg.params_dict) + cfg.rescale_idx["T"]
      assert torch.allclose(theta[:, i_T], torch.full_like(theta[:, i_T], 300.0), rtol=1e-3)
      _assert_every_drive_used_the_derived_scale(spy, derived_force_scale(cfg))
  ```

- [ ] **Step 5: Write the Fisher test and the observation-plus-predictive-check test.**

  ```python
  def test_the_fisher_operating_points_simulate_at_the_derived_force_scale(monkeypatch):
      from core.SBI import decorrelate
      cfg = with_truth(tier1_cfg(chi=False))
      cfg.T_obs = 1000.0                                      # cell units: 1 s at dt_exp = 1 ms
      monkeypatch.setattr(pipeline, "gen_obs", stand_in_gen_obs)
      spy = force_scale_spy(monkeypatch)
      V = decorrelate.build_latent_fisher_rotation(cfg, m=2, n_points=1)
      f = derived_force_scale(cfg)
      assert tuple(V.shape) == (13, 13)
      assert all(e["declared"] and "T" not in e["index"] for e in spy)
      assert torch.allclose(spy[0]["f_scale"], torch.full_like(spy[0]["f_scale"], f), rtol=1e-4)  # the anchor
      every = torch.cat([e["f_scale"] for e in spy])       # the +-dz arms move n, beta, x_scale, T a little
      assert bool(((every > 0.5 * f) & (every < 2.0 * f)).all())
  ```

  `test_a_simulated_observation_and_its_predictive_checks_use_the_derived_force_scale(store, monkeypatch)`:
  - Setup:
    - `cfg = with_truth(tier1_cfg(chi=False)); cfg.T_obs = 1000.0`
    - install the stand-in and the spy
    - `obs = orchestrator.generate_observations(cfg, fig_sink=_close, name="tier1_obs")`
    - assert `_assert_every_drive_used_the_derived_scale(spy, f, rtol=1e-5)`, then `spy.clear()`
  - Stand-ins for the inference:
    - `post = SimpleNamespace(posterior=SimpleNamespace(x_obs_digest=None, truncation=None, T=None, sample=lambda shape, x=None, **k: gt.expand(shape[0], -1).clone()), manifest=SimpleNamespace(body={"mode": obs.mode, "conditioning": {"width": obs.width}}), id="post", name="", accepted=[])`, with `gt = cfg.ground_truth_tensor`
    - monkeypatch `orchestrator.pairplot` to `lambda *a, **k: (plt.figure(), None)`
    - monkeypatch `orchestrator._emit_overlay_figures` to `lambda *a, **k: None`
  - Run `orchestrator.infer_and_visualize(cfg, post, obs, n_samples=4, fig_sink=_close)`.
  - Assert `_assert_every_drive_used_the_derived_scale(spy, f, rtol=1e-5)`. That covers the PPC bins and the eye test's central trajectories.

- [ ] **Step 6: Write the narrowing-round test and the prior-loading test.**
  - `test_a_narrowing_round_on_a_tier1_parent_judges_the_truth_in_inferred_coordinates(store, monkeypatch, caplog)` uses the stubbed-training pattern of `tests/test_artifact_store.py::test_fisher_settings_are_recorded_only_when_the_rotation_ran`.
  - Setup:
    - `cfg = with_truth(tier1_cfg(chi=False)); cfg.reparam_rotate = False; cfg.hw.batch_size = 4`
    - `lp = store.load_prior(cfg, _prior_artifact(store, cfg, name="p").id)`
    - `T = reparam.build_inferred_bijection(cfg, log_params=_log_params_for(cfg))`
    - `i_T = 12`, `c = float(T.inv(cfg.ground_truth_tensor.reshape(1, -1))[0, i_T])`
    - `region(lo, hi) = truncate.TruncationRegion([i_T], [lo], [hi], n_latent=13, V=None, probe=tc.bijection_probe(T, 13, device=cfg.hw.device), x_obs_digest="d" * 16)`
  - Stub the training:
    - monkeypatch `orchestrator.pipeline.train_nn` with a fake that appends its `plan` to `plans` and returns `(_FakeDP with .prior = kw["prior"], {"training_loss": [1.0], "validation_loss": [1.0], "best_validation_loss": 1.0, "epochs_trained": 1, "stop_after_epochs": 1})`
    - monkeypatch `orchestrator.pipeline.gen_training_data` to `lambda plan, **kw: (torch.zeros(8, 50), torch.zeros(8, 13))`
    - `budget = dict(num_runs=2, run_size_cap=4, hidden_features=8, num_transforms=1, stop_after_epochs=1, checkpoint_every=0, fig_sink=_close)`
  - Assertions:

  ```python
  caplog.set_level("INFO", logger="core")
  with warnings.catch_warnings(record=True) as caught:
      warnings.simplefilter("always")
      orchestrator.build_posterior(cfg, lp, None, True, truncation=region(c - 0.5, c + 0.5), **budget)
  assert not any("GROUND TRUTH" in str(w.message) for w in caught)
  assert plans[0].nd_idx == cfg.nd_idx and plans[0].k_b_cell == pytest.approx(cfg.k_b_cell)
  assert any(r.getMessage().startswith("[tier1] f_scale is DERIVED") for r in caplog.records)
  with warnings.catch_warnings(record=True) as caught:
      warnings.simplefilter("always")
      orchestrator.build_posterior(cfg, lp, None, True, truncation=region(c + 1.0, c + 2.0), **budget)
  truth = [w for w in caught if "GROUND TRUTH" in str(w.message)]
  assert len(truth) == 1 and "direction 12" in str(truth[0].message)
  # The region is judged in inferred coordinates: the same truth with its temperature replaced by
  # the force scale it implies is not a point of the box at all.
  sim_truth = cfg.ground_truth_tensor.clone()
  sim_truth[i_T] = derived_force_scale(cfg)
  assert orchestrator._truth_outside_region(T, region(c - 0.5, c + 0.5), sim_truth) != []
  ```

  ```python
  def test_a_prior_built_on_the_master_box_loads_under_the_tier1_box(store):
      cfg = tier1_cfg(chi=False)
      lp = store.load_prior(cfg, _prior_artifact(store, _nad_cfg(), name="master_prior").id)
      draws = lp.prior.sample((64,))
      t_col = draws[:, len(cfg.params_dict) + cfg.rescale_idx["T"]]
      assert tuple(draws.shape) == (64, 13) and bool(((t_col >= 280.0) & (t_col <= 310.0)).all())
  ```

- [ ] **Step 7: Write the slow real run.**
  - Signature: `@pytest.mark.slow`, `test_smoke_runs_every_stage_on_the_tier1_box(tmp_path, monkeypatch, capsys)`.
  - Setup:
    - `monkeypatch.setattr(orchestrator.pipeline, "gen_prior", _tiny_nadrowski_gen_prior)`. This bounds the prior sweep; the rest is real.
    - Run `main(["smoke", "--bounds", <tier-1 box>, "--device", "cpu", "--chi", "--cell", <tier-1 cell>, "--t-obs", "1.0", "--num-runs", "2", "--run-size", "8", "--n-cal", "20", "--max-epochs", "1", "--save", "--store-root", str(tmp_path / "tier1")])`.
  - Assertions:
    - `== 0`
    - stdout contains `"[tier1] f_scale is DERIVED"` and `"[tier1] chi drive amplitude"`
    - `ArtifactStore(tmp_path / "tier1").get("posterior", "smoke_posterior").body["transform"]["param_keys"]` ends with `"T"` and holds no `"f_scale"`.

- [ ] **Step 8: Reword the slow marker.**
  - In `pytest.ini`, the `slow:` line becomes exactly: `slow: the long real-simulation tests; excluded by the fast gate (pytest -m "not slow")`

- [ ] **Step 9: Run the fast focused tests.**
  - `python -m pytest tests/test_tier1.py -m "not slow" -q --durations=0`. Expected: all pass.
  - Fix any failure as the Binding says: production defect → report and fix; stand-in → fix the stand-in.
  - Then `python -m pytest tests/test_user_sbi.py --collect-only -q`. Expected: no collection error.

- [ ] **Step 10: Run the slow test once, alone, in the background.**
  - `python -m pytest "tests/test_tier1.py::test_smoke_runs_every_stage_on_the_tier1_box" -q > C:\Users\J\AppData\Local\Temp\prism-piece6\task12-slow.log 2>&1`, run in the background.
  - Wait for it to exit and touch no source while it runs. Expected: `1 passed`.
  - Put its duration in the task report: the controller adds it to CLAUDE.md's slow-set description in the final gate.

- [ ] **Step 11: Commit.**
  - `git add tests/test_tier1.py tests/_fixtures.py tests/test_user_sbi.py pytest.ini`
  - `git commit -m "test: the tier-1 path, end to end" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"`
  - The controller starts the one-process fast gate (`python -m pytest -m "not slow" -q`, in the background, logged) at the same moment as this task's review. Do not run it yourself.

---

### Task 13: The silent fallback closed

**Files:**
- Modify: `core/SBI/derived.py`. Add `for_simulation` after `def to_sim_rescale(` (near :71).
- Modify: `core/forcing.py`:
  - add `require_simulator_index`;
  - call it first in `def build_nondim_force_tensor(` (near :89). The sinusoidal builder and the user-force builder both delegate there.
  - rewrite the `rescale_idx` docstring lines near :116-117 and :230-235, "If "f_scale" is absent … Hopf-style". The fallback now applies only when the index names neither `f_scale` nor `T`.
- Modify: `core/SBI/chi_probes.py`. The guard is the first statement of `def gen_chi_raw(` (near :43), before `chi.peak_freq`. Also update the fallback comment near :112: `# Hopf-style: build_nondim uses f_scale = x_scale / t_scale`.
- Modify: `core/diagnostics/identifiability.py`:
  - `_laplace_raw`: the builder call `forcef = pipeline.build_nondim_sin_force_tensor(fv.expand(m, -1), t_fine, rv,` (near :248);
  - `_jacobian_features`: the `pipeline.gen_chi_raw(` call (near :525) and `force = pipeline.build_nondim_sin_force_tensor(ctx.forcing_gt.expand(m, -1), t_fine, rv,` (near :543).
- Modify: `core/gui/panels/simulate_runner.py`. In `plan_stream` (near :106), the `StreamPlan(` return (near :140-146) gets the simulator's rescale block and index.
- Test: `tests/test_tier1.py`

**Interfaces:**
- Consumes (Task 12, `tests/test_tier1.py`): `tier1_cfg`, `with_truth`, `derived_force_scale`, `force_scale_spy`, `stand_in_gen_obs`, `_assert_every_drive_used_the_derived_scale`; `tests._fixtures._nad_cfg`.
- Produces:
  - `core.SBI.derived.for_simulation(cfg, params_nd: Tensor, rescale: Tensor) -> tuple[Tensor, dict[str, int]]`. It returns `(to_sim_rescale(params_nd, rescale, cfg.rescale_idx, *cfg.tier1_args), cfg.sim_rescale_idx)`. Every simulating diagnostic, the live runner and the probe checks (Task 22) call it.
  - Addition to the registry: `core.forcing.require_simulator_index(rescale_idx: dict) -> None`, raising `RuntimeError`.
  - Test helpers in `tests/test_tier1.py`: `twin_cfg(cfg) -> SimConfig` and `_jac_ctx(cfg, n_obs)`.

**Binding (read before starting):**
- Spec §5.2 is binding; read it. A tier-1 index handed to a force builder is a **programming error**:
  - `RuntimeError`, never a `core.refusals.Refusal` (never the yellow box);
  - both front ends show it as a bug with a traceback;
  - it raises before any simulation is spent.
- The message is exact copy (Step 3). It names the index and the relation. It must not contain "out of memory" (`pipeline._is_oom` would retry it).
- Detection is `core.SBI.derived.uses_derived_f_scale(rescale_idx)`. `derived.py` imports only torch, and `core/SBI/__init__.py` is empty, so `forcing.py` stays free of the sbi library.
- A Hopf-style index (neither `T` nor `f_scale`) keeps the `x_scale / t_scale` form, unchanged.
- **Every caller of both builders, verified at `b24973e`.** Re-grep (`build_nondim_sin_force_tensor|build_nondim_force_tensor|build_user_force_tensor|gen_chi_raw|gen_chi_block`) and paste the list, with each caller's verdict, into the task report.
  - Already pass the simulator's index: `core/SBI/pipeline.py:984` (`gen_chi_block(..., sim_ridx, ...)`), `:1054`; `core/SBI/decorrelate.py:236`, `:257`; `core/orchestrator.py:201`, `:232`, `:2111`; `core/SBI/ppc.py:85`, `:118`; `core/SBI/chi_probes.py:208`, `:260` (internal).
  - Passed the inferred index and fixed here: `core/diagnostics/identifiability.py:248`, `:525`, `:543`; `core/gui/panels/simulate_runner.py:220`, `:233` (via `plan.rescale_idx`).
  - Internal: `core/forcing.py:197`, `:241`.
  - The runner can meet a tier-1 box only through a user-added sibling bounds file with a Forcing section. It derives the scale anyway.
- Inside identifiability, `x_scale`, `x_offset` and `t_scale` keep being read from the inferred vector (the substitution does not touch them). The finite-difference arms stay in inferred coordinates, so d/dT is taken through the derived force scale.
- Tests are cheap: the stand-in simulator and feature-level calls, no public diagnostic run.
- H4 applies to every line you write in `core/` and `tests/`.
- Messages stay `logging` records. Never edit a source file while pytest runs. Interpreter as in Task 12.

- [ ] **Step 1: Write the failing tests** in `tests/test_tier1.py`. First the helpers:

  ```python
  def twin_cfg(cfg):
      """The master-box twin of a tier-1 config with its truth loaded: the same cell, with the force
      scale the tier-1 relation derives declared as f_scale."""
      twin = _nad_cfg(chi_mode=cfg.chi_mode)
      twin.inject_ground_truth(
          dict(cfg.inits_dict), {k: v for k, (v, _) in cfg.params_dict.items()},
          {"x_scale": cfg.rescale_params["x_scale"][0], "t_scale": cfg.rescale_params["t_scale"][0],
           "f_scale": derived_force_scale(cfg)},
          {k: v for k, (v, _) in cfg.force_params_dict.items()})
      return twin
  ```

  `_jac_ctx(cfg, n_obs)` builds `identifiability._JacCtx` exactly as `identifiability_jacobian` does:
  - `mults=chi.chi_multipliers_for(cfg)` in chi mode;
  - `forcing_gt=torch.tensor([[v for v, _ in cfg.force_params_dict.values()]], dtype=cfg.hw.dtype)` otherwise;
  - `keep_idx=feature_sets.summary_keep_idx()`, `feat_labels=feature_sets.feature_labels(cfg)`;
  - `n_force_ch=forcing.n_force_channels(cfg.model, cfg.forcing_idx, cfg.inits_tensor.shape[-1])`.

  ```python
  def _res(c):
      return torch.tensor([v for v, _ in c.rescale_params.values()], dtype=c.hw.dtype)


  def test_for_simulation_derives_the_force_scale_on_a_tier1_box_and_passes_every_other_box_through():
      from core.SBI import derived
      cfg = with_truth(tier1_cfg(chi=False))
      res = _res(cfg).unsqueeze(0)
      sim, idx = derived.for_simulation(cfg, cfg.params_tensor, res)
      assert idx == {"x_scale": 0, "t_scale": 1, "f_scale": 2}
      assert float(sim[0, 2]) == pytest.approx(derived_force_scale(cfg), rel=1e-5)
      assert torch.equal(sim[:, :2], res[:, :2]) and float(res[0, 2]) == 300.0   # the input is not written
      master = _nad_cfg()
      cli.load_and_validate_gt(master, str(config.CELL_PATH / "nadrowski" / "master_weak.txt"))
      res_m = _res(master).unsqueeze(0)
      sim_m, idx_m = derived.for_simulation(master, master.params_tensor, res_m)
      assert sim_m is res_m and idx_m == master.rescale_idx


  def test_a_force_builder_handed_a_temperature_index_raises_before_anything_is_simulated(monkeypatch):
      from core import forcing
      from core.refusals import Refusal
      from core.SBI import chi_probes
      idx = {"x_scale": 0, "t_scale": 1, "T": 2}
      fp, fidx = torch.tensor([[1.0, 0.05, 0.0, 0.0]]), {"amp": 0, "freq": 1, "phase": 2, "offset": 3}
      t, res = torch.linspace(0.0, 1.0, 16), torch.tensor([[62.14, 3.73, 300.0]])
      relation = r"f_scale = N \* beta \* k_B \* T / x_scale"
      with pytest.raises(RuntimeError, match=relation) as e:
          forcing.build_nondim_sin_force_tensor(fp, t, res, fidx, idx)
      assert "'T'" in str(e.value) and not isinstance(e.value, Refusal)
      monkeypatch.setattr(pipeline, "gen_obs", lambda *a, **k: pytest.fail("a simulation ran"))
      with pytest.raises(RuntimeError, match=relation):
          chi_probes.gen_chi_raw("NADROWSKI", torch.zeros(1, 10), res, torch.zeros(1, 64), t,
                                 torch.zeros(1, 3), idx, 1, 0, 1, 64, 1.0, torch.tensor([0.1]), 0.15)
      hopf = forcing.build_nondim_sin_force_tensor(fp, t, res[:, :2], fidx, {"x_scale": 0, "t_scale": 1})
      assert tuple(hopf.shape) == (1, 1, 16)                  # a Hopf-style index keeps its fallback


  def test_laplace_simulates_a_tier1_box_at_its_derived_force_scale(monkeypatch):
      import numpy as np
      from core.diagnostics import identifiability
      cfg = with_truth(tier1_cfg(chi=False))
      twin = twin_cfg(cfg)
      monkeypatch.setattr(pipeline, "gen_obs", stand_in_gen_obs)
      spy = force_scale_spy(monkeypatch)
      force = torch.tensor([v for v, _ in cfg.force_params_dict.values()], dtype=cfg.hw.dtype)
      got, _, _ = identifiability._laplace_raw(cfg, cfg.params_tensor[0], _res(cfg), force, 4, True, 1000)
      want, _, _ = identifiability._laplace_raw(twin, twin.params_tensor[0], _res(twin), force, 4, True, 1000)
      assert np.allclose(got, want, rtol=1e-4, atol=1e-8, equal_nan=True)
      _assert_every_drive_used_the_derived_scale(spy[:len(spy) // 2], derived_force_scale(cfg), rtol=1e-5)


  @pytest.mark.parametrize("chi", [True, False])
  def test_jacobian_simulates_a_tier1_box_at_its_derived_force_scale(monkeypatch, chi):
      import numpy as np
      from core.diagnostics import identifiability
      cfg = with_truth(tier1_cfg(chi=chi))
      twin = twin_cfg(cfg)
      monkeypatch.setattr(pipeline, "gen_obs", stand_in_gen_obs)
      spy = force_scale_spy(monkeypatch)
      got, _, _ = identifiability._jacobian_features(_jac_ctx(cfg, 1000), cfg.params_tensor[0], _res(cfg), 4, True)
      n_tier1 = len(spy)
      want, _, _ = identifiability._jacobian_features(_jac_ctx(twin, 1000), twin.params_tensor[0], _res(twin), 4, True)
      assert np.allclose(got, want, rtol=1e-4, atol=1e-8, equal_nan=True)
      _assert_every_drive_used_the_derived_scale(spy[:n_tier1], derived_force_scale(cfg), rtol=1e-5)


  def test_the_live_simulation_derives_the_force_scale_on_a_tier1_box(monkeypatch):
      from core.gui.panels import simulate_runner
      cfg = with_truth(tier1_cfg(chi=False))
      f = derived_force_scale(cfg)
      plan = simulate_runner.plan_stream(cfg, 0.005)
      assert plan.rescale_idx == {"x_scale": 0, "t_scale": 1, "f_scale": 2}
      assert float(plan.rescale_gt[0, 2]) == pytest.approx(f, rel=1e-5)
      assert plan.x_scale == pytest.approx(62.14)
      spy, chunks = force_scale_spy(monkeypatch), []
      simulate_runner.run_simulation_stream(cfg, 0.005, frame_steps=500, fps=0.0, emit_chunk=chunks.append)
      assert chunks
      _assert_every_drive_used_the_derived_scale(spy, f, rtol=1e-5)
  ```

  In the laplace test, use the spy count taken after the first call, as the jacobian test does. The slice shown is illustrative; the tier-1 call's records are the ones asserted.

- [ ] **Step 2: Run them and see them fail.**
  - Run `python -m pytest tests/test_tier1.py -q -k "for_simulation or temperature_index or laplace or jacobian or live_simulation"`.
  - Expected failures:
    - `for_simulation`: AttributeError, no such function;
    - builder: no RuntimeError raised;
    - laplace/jacobian: the `np.allclose` assertion. They silently drive at 16.66, not 46.99;
    - runner: `plan.rescale_idx` holds `T`.

- [ ] **Step 3: Implement `for_simulation` and the guard.**
  - `for_simulation(cfg, params_nd, rescale)` in `core/SBI/derived.py`, with the body the Interfaces line gives. Its docstring says every caller that splits a parameter vector for simulation calls it once, and gets back the rescale block and the index the builders must be handed together.
  - `require_simulator_index(rescale_idx: dict) -> None` in `core/forcing.py`. It raises exactly this:

  ```python
  raise RuntimeError(
      f"This rescale index names 'T' and no 'f_scale' ({dict(rescale_idx)}): temperature stands in "
      f"the force scale's column, and the force scale must be derived as "
      f"f_scale = N * beta * k_B * T / x_scale before a drive is built. Pass the simulator's rescale "
      f"block and index (core.SBI.derived.for_simulation), not the inferred ones.")
  ```

  - Call it first in `build_nondim_force_tensor` and first in `chi_probes.gen_chi_raw` (`_forcing.require_simulator_index(rescale_idx)`).

- [ ] **Step 4: Route the three identifiability call sites and the runner through `for_simulation`.**
  - `_laplace_raw`: after `rv = ...`, add `rv_sim, sim_idx = derived.for_simulation(cfg, p, rv)` and hand `rv_sim, sim_idx` to the builder.
  - `_jacobian_features`: the same, with `rescale=rv_sim, rescale_idx=sim_idx` to `gen_chi_raw` and `rv_sim, sim_idx` to the forced builder.
  - `plan_stream`: `rescale_sim, idx_sim = derived.for_simulation(cfg, cfg.params_tensor, rescale_gt)`, then store `rescale_gt=rescale_sim, rescale_idx=idx_sim` in the plan. Keep reading `t_scale`/`x_scale`/`x_offset` from the inferred block with `cfg.rescale_idx`.
  - Update the two `StreamPlan` field comments ("the simulator's rescale block: a tier-1 box carries its derived force scale in T's column").

- [ ] **Step 5: Run the new tests and see them pass.**
  - Same command as Step 2. Expected: all pass.

- [ ] **Step 6: Run the focused tests for every file touched.**
  - `python -m pytest tests/test_tier1.py -m "not slow" -q`
  - `python -m pytest tests/test_diagnostics.py -q -k "laplace or jacobian or identifiability"`
  - `python -m pytest tests/test_user_models.py -q -k "force or golden or sin"` (the builder's golden pins)
  - `python -m pytest tests/test_simulate.py -q`
  - Expected: all pass.

- [ ] **Step 7: Commit.**
  - `git add core/SBI/derived.py core/forcing.py core/SBI/chi_probes.py core/diagnostics/identifiability.py core/gui/panels/simulate_runner.py tests/test_tier1.py`
  - `git commit -m "tier-1: derive the force scale in every simulating caller" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"`
  - Put the caller list in the report. The controller starts the fast gate with the review; do not run it.

---

### Task 14: Temperature as an assumed input

**Files:**
- Modify: `core/sim_config.py`. Add `assumed_params` and `report_labels` after `def inferred_labels(` (near :319).
- Modify: `core/orchestrator.py`:
  - the SBC table `for j, label in enumerate(cfg.inferred_labels):` (near :1812);
  - both `parameter_labels=cfg.inferred_labels` (near :1837, :1841);
  - `log.info(analysis.describe_informativeness(info))` (near :1869) and `"description": analysis.describe_informativeness(info)` (near :1892);
  - the corner `labels=cfg.inferred_labels,` (near :2009);
  - `"posterior_summary": [` (near :2161), moved to a new private `_posterior_summary`.
- Modify: `core/SBI/analysis.py`: `def describe_informativeness(info: dict)` (near :367).
- Modify: `core/diagnostics/sbc.py`: the `per_param.append(` record (near :190), the KS table rows and header (near :186, :193-195), and `parameter_labels=labels` (near :201).
- Modify, wording only:
  - `core/SBI/pipeline.py`, the `ValueError` `f"No (t_scale, T) pair in the declared bounds fits the fine-grid ceiling of "` (near :1183);
  - `core/gui/fields.py` `"cal_n_scales": ("Validate", "(t_scale, T) operating points"),` (near :70);
  - `core/gui/panels/inference/help_text.py` `"cal_scales": "(t_scale, T) operating points` (near :88) and `"num_runs": "... one Sobol (t_scale, T) "` (near :94);
  - `core/gui/panels/inference/base.py` `f"\nBatches is also the (t_scale, T) diversity count` (near :155).
- Modify: `tests/_fixtures.py`, adding `stub_calibration_battery`.
- Test: `tests/test_tier1.py`, `tests/test_nav_and_gating.py`.
- Test, pins updated:
  - `tests/test_artifact_store.py:1735` (`{"name", "q05", "median", "q95"}`);
  - `tests/test_diagnostics.py:362` (the sbc `per_param` key set);
  - `tests/test_settings_persistence.py:110` (message) and `:340` (the budget-line pin).

**Interfaces:**
- Consumes (Task 12): `tier1_cfg`, `_close`; `tests._fixtures._nad_cfg`.
- Produces:
  - `SimConfig.assumed_params -> tuple[str, ...]`: `("T",)` when tier 1 is on, else `()`.
  - `SimConfig.report_labels -> list[str]`: `inferred_labels` with `" (assumed input, K)"` appended at an assumed parameter's index.
  - `posterior_summary` entries gain `"assumed": bool`.
  - `analysis.describe_informativeness(info, *, assumed: Sequence[str] = ())`.
  - Additions to the registry:
    - `orchestrator._posterior_summary(samples: Tensor, keys: list[str], assumed: Sequence[str]) -> list[dict]`;
    - `tests._fixtures.stub_calibration_battery(monkeypatch, *, ks_pvals=None, tarp_ks_p=0.5) -> dict`, used by Tasks 15 and 19;
    - sbc `per_param` records gain `"assumed": bool`.

**Binding (read before starting):**
- `python` below means `C:\Users\J\anaconda3\envs\biophys-env\python.exe` (in the Bash tool, `/c/Users/J/anaconda3/envs/biophys-env/python.exe`); the `python` on PATH is a different interpreter without torch or pytest. Commands written with backslash paths are PowerShell commands; in Bash use forward slashes.
- Spec §5.3 and §1.2 ("Temperature on the tier-1 box") are binding; read them.
- `assumed_params` holds parameter **keys**, never the LaTeX `inferred_labels`. It is derived from the box through `core.SBI.derived.uses_derived_f_scale` and `derived.TEMPERATURE_PARAM`, never typed.
  - Consumers match by index through `list(cfg.params_dict) + list(cfg.rescale_params)`.
  - Both new members are plain `@property`, never `cached_property`: `core/sim_config.py`'s `_CACHED` tuple is pinned to name every cached property.
- Temperature **stays** in every test and every total:
  - in the calibration's rank table;
  - in the joint coverage test, which runs exactly as today;
  - in the informativeness total, which stays joint.
  - It is only **marked**:
    - `"$T$ (assumed input, K)"` via `report_labels` in the corner plot, the calibration's SBC log table and both SBC rank figures;
    - `"T (assumed input)"` in `sbc`'s table and pooled figure (which use keys);
    - `"  (assumed input)"` after the informativeness per-parameter line;
    - `"assumed": true` in `posterior_summary` and sbc's `per_param`.
- Wording: where "T" means the recording length, the window text and the one pipeline message say `T_obs` ("(t_scale, T_obs) operating points", "(t_scale, T_obs) diversity count", and "T_obs in [...]" in the pipeline message).
  - Labels already written into records stay: the batch tag `f"[t_scale=..., T=..., ...]"` at `core/SBI/pipeline.py:1491-1493` (pinned at `tests/test_user_sbi.py:3490`, read by the probe observer) and the ablation channel `logT`.
  - The spec's `pipeline.py:1491-1493` anchor IS the batch tag. Leave it.
- `labels.py` is unchanged: `inferred_labels` keeps the bare `$T$`, and `report_labels` adds the unit and the mark.
- H4 applies to every line written. Knobs stay arguments. Messages stay `logging` records. Never edit a source file while pytest runs.

- [ ] **Step 1: Write the calibration stand-ins in `tests/_fixtures.py`.**
  - They are test scaffolding that later tests rely on, so write them this way:

  ```python
  def stub_calibration_battery(monkeypatch, *, ks_pvals=None, tarp_ks_p=0.5) -> dict:
      """Stand in for everything validate_calibration and sbc_repeats simulate or sample. The
      stand-ins draw from torch's and numpy's global streams as the real calls do, so a seed is
      observable and nothing is simulated.

      ks_pvals: None draws each rank test's KS p-values from torch's stream; a flat list is returned
      by every rank test; a list of lists is consumed one list per rank test, in order. Returns a dict
      with the posterior and prior stand-ins and what the stand-ins saw: "x_cal" (each calibration set
      drawn), "numpy_at_draw" (numpy's state vector at each draw), "rank_plot_labels" (each rank
      figure's parameter labels)."""
      import numpy as np
      from types import SimpleNamespace
      from matplotlib import pyplot as plt
      from core import orchestrator as orch
      seen = {"x_cal": [], "numpy_at_draw": [], "rank_plot_labels": [],
              "posterior": SimpleNamespace(posterior=SimpleNamespace(truncation=None, x_obs_digest=None),
                                           id="post", name="", accepted=[]),
              "prior": SimpleNamespace(prior=None, force_prior=None, id="prior", name="", fingerprint=None)}
      queue = list(ks_pvals) if ks_pvals and isinstance(ks_pvals[0], (list, tuple)) else None

      def draw(cfg, vlp, T, force_prior, *, n_cal, cal_n_scales, chi_k_fixed=None):
          P = len(cfg.params_dict) + len(cfg.rescale_params)
          seen["numpy_at_draw"].append(np.random.get_state()[1].copy())
          x = torch.randn(n_cal, 5)
          seen["x_cal"].append(x.clone())
          theta = torch.rand(n_cal, P) + torch.as_tensor(np.random.standard_normal((n_cal, P)),
                                                         dtype=torch.float32)
          return x, theta

      def check_sbc(ranks, prior_samples, dap_samples, num_posterior_samples):
          P = prior_samples.shape[1]
          ks = (queue.pop(0) if queue is not None
                else list(ks_pvals) if ks_pvals is not None else torch.rand(P).tolist())
          return {"ks_pvals": ks, "c2st_ranks": [0.5] * P, "c2st_dap": [0.5] * P}

      def rank_plot(**k):
          seen["rank_plot_labels"].append(list(k.get("parameter_labels") or []))
          return plt.figure(), None

      def informativeness(post, theta_star, x_cal, prior, *, param_names=None, **k):
          n, P = theta_star.shape
          return {"total_nats": 1.0, "sem_nats": 0.1, "n_used": n, "n_dropped": 0,
                  "param_names": param_names, "per_param": [0.1] * P, "per_direction": None,
                  "n_decompose": n}

      for name, fn in {
          "_calibration_prior": lambda cfg, posterior, prior: (None, None, None),
          "_assert_prior_used_matches_posterior": lambda *a, **k: None,
          "_draw_calibration_set": draw,
          "run_sbc": lambda thetas, xs, posterior, num_posterior_samples, **k: (
              torch.randint(0, num_posterior_samples + 1, tuple(thetas.shape)), torch.zeros(tuple(thetas.shape))),
          "_sbc_reference_sample": lambda cfg, vlp, T, truncation, prior, theta_star: torch.zeros_like(theta_star),
          "check_sbc": check_sbc, "sbc_rank_plot": rank_plot,
          "run_tarp": lambda thetas, xs, posterior, **k: (torch.linspace(0, 1, 5), torch.linspace(0, 1, 5)),
          "check_tarp": lambda ecp, alpha: (0.0, tarp_ks_p),
          "plot_tarp": lambda *a, **k: plt.figure(),
      }.items():
          monkeypatch.setattr(orch, name, fn)
      monkeypatch.setattr(orch.analysis, "informativeness", informativeness)
      return seen
  ```

- [ ] **Step 2: Write the failing tests** in `tests/test_tier1.py`:

  ```python
  def test_temperature_is_the_one_assumed_parameter_and_reports_carry_its_unit_and_mark():
      cfg = tier1_cfg()
      keys = list(cfg.params_dict) + list(cfg.rescale_params)
      i_T = keys.index("T")
      assert cfg.assumed_params == ("T",) and len(cfg.report_labels) == len(keys) == 13
      assert cfg.report_labels[i_T] == cfg.inferred_labels[i_T] + " (assumed input, K)" == "$T$ (assumed input, K)"
      assert [l for i, l in enumerate(cfg.report_labels) if i != i_T] == \
             [l for i, l in enumerate(cfg.inferred_labels) if i != i_T]
      master = _nad_cfg()
      assert master.assumed_params == () and master.report_labels == master.inferred_labels


  def test_the_posterior_summary_marks_the_assumed_parameter_and_the_corner_uses_the_report_labels():
      import ast, inspect, textwrap
      keys = list(tier1_cfg().params_dict) + ["x_scale", "t_scale", "T"]
      samples = torch.arange(3 * 13, dtype=torch.float32).reshape(3, 13)
      out = orchestrator._posterior_summary(samples, keys, ("T",))
      assert [e["name"] for e in out] == keys and [e["assumed"] for e in out] == [False] * 12 + [True]
      assert set(out[0]) == {"name", "q05", "median", "q95", "assumed"}
      assert out[0]["median"] == 13.0 and out[0]["q05"] == pytest.approx(1.3)
      tree = ast.parse(textwrap.dedent(inspect.getsource(orchestrator.infer_and_visualize)))
      calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
      pair = [c for c in calls if c.func.id == "pairplot"]
      assert len(pair) == 1
      assert ast.unparse({k.arg: k.value for k in pair[0].keywords}["labels"]) == "cfg.report_labels"
      assert [ast.unparse(c.args[2]) for c in calls if c.func.id == "_posterior_summary"] == ["cfg.assumed_params"]


  def test_describe_informativeness_marks_only_what_it_is_told_is_assumed():
      from core.SBI import analysis
      info = {"total_nats": 1.0, "sem_nats": 0.1, "n_used": 8, "n_dropped": 0, "param_names": ["k", "T"],
              "per_param": [0.2, 0.01], "per_direction": None, "n_decompose": 8}
      marked, plain = analysis.describe_informativeness(info, assumed=("T",)), analysis.describe_informativeness(info)
      assert "T +0.0100  (assumed input)" in marked and marked.count("(assumed input)") == 1
      assert "(assumed input)" not in plain and marked.splitlines()[0] == plain.splitlines()[0]
  ```

  `test_the_calibration_table_plots_and_informativeness_mark_the_assumed_parameter(store, monkeypatch, caplog)`:
  - `seen = stub_calibration_battery(monkeypatch, ks_pvals=[0.5] * 13)`
  - `caplog.set_level("INFO", logger="core")`
  - `cal = orchestrator.validate_calibration(tier1_cfg(), seen["posterior"], seen["prior"], n_cal=8, cal_n_scales=2, fig_sink=_close, store=store)`
  - Assertions:
    - the `core.orchestrator` records include exactly `"  $T$ (assumed input, K): KS p=0.500  c2st_ranks=0.500  c2st_dap=0.500"`
    - `seen["rank_plot_labels"] == [cfg.report_labels, cfg.report_labels]`
    - the one record starting `"Informativeness"` has exactly one line containing `"(assumed input)"`, and it is the `T` line
    - `"(assumed input)" in cal.results["informativeness"]["description"]`

  `test_sbc_marks_the_assumed_parameter_in_its_table_and_its_records(store, monkeypatch, caplog)`:
  - Same stand-ins, `sbc_repeats(tier1_cfg(), seen["posterior"], seen["prior"], repeats=2, n_cal=8, num_posterior_samples=10, fig_sink=_close, store=store)`.
  - Assertions:
    - `[p["assumed"] for p in d.results["per_param"]] == [False] * 12 + [True]`
    - exactly one `core.diagnostics.sbc` record starts with `"T (assumed input)"`
    - `seen["rank_plot_labels"][-1][-1] == "T (assumed input)"`

  In `tests/test_nav_and_gating.py`:

  ```python
  def test_the_window_says_t_obs_where_it_means_the_recording_length():
      from core.gui import fields as gui_fields
      from core.gui.panels.inference.help_text import HELP
      assert gui_fields.label("cal_n_scales") == "(t_scale, T_obs) operating points"
      assert HELP["cal_scales"].startswith("(t_scale, T_obs) operating points")
      assert "(t_scale, T_obs)" in HELP["num_runs"]
      assert "(t_scale, T)" not in HELP["cal_scales"] + HELP["num_runs"]
  ```

- [ ] **Step 3: Run them and see them fail.**
  - Run `python -m pytest tests/test_tier1.py tests/test_nav_and_gating.py -q -k "assumed or informativeness or t_obs_where"`.
  - Expected: AttributeError (`assumed_params`, `report_labels`, `_posterior_summary`), TypeError (`describe_informativeness` has no `assumed`), and assertion failures on the table and window wording.

- [ ] **Step 4: Implement.**
  - The two `SimConfig` properties.
  - `describe_informativeness(info: dict, *, assumed: Sequence[str] = ()) -> str`: the per-parameter line gains `"  (assumed input)"` when `names[j] in assumed`. Both call sites in `validate_calibration` pass `assumed=cfg.assumed_params`.
  - `_posterior_summary`: the quantile code moved out of `infer_and_visualize` unchanged (`torch.quantile` over float64 CPU samples at 0.05/0.5/0.95, `_num` each), plus `"assumed": k in assumed`. Call it as `_posterior_summary(samples, keys, cfg.assumed_params)`. Then log the summary once, as ONE info record on `core.orchestrator`, so the Infer tab's log pane (which shows the stage's records) carries the summary the owner checks: build its text in a pure helper `_describe_posterior_summary(entries: list[dict]) -> str`, a head line `[infer] posterior summary (5 % / median / 95 %):` and one row per entry named by its key, with `(assumed input, K)` after an assumed one's name. Test-first: before writing the helper, add `test_the_posterior_summary_text_marks_the_assumed_parameter` to `tests/test_tier1.py` (feed it `orchestrator._posterior_summary(samples, keys, ("T",))` as in the test above; assert the head line, and that exactly one line contains `(assumed input` and it is the `T` row), and extend `test_the_posterior_summary_marks_the_assumed_parameter_and_the_corner_uses_the_report_labels` with an AST check that `infer_and_visualize` calls `_describe_posterior_summary` exactly once, inside a `log.info(...)`; run both and see them fail, then implement.
  - Corner, SBC table and both SBC rank figures use `cfg.report_labels` (the `n_sbc_rows` count may use either list).
  - `sbc.py`:
    - `assumed = set(cfg.assumed_params)`;
    - each `per_param` record gains `"assumed": key in assumed`;
    - the table rows print `name + " (assumed input)"` for an assumed one (widen the name column and its header from 16 to 18);
    - the pooled figure's labels get the same mark.
  - The four window strings and the pipeline message take the `T_obs` wording.

- [ ] **Step 5: Update the pins.**
  - `tests/test_artifact_store.py:1735` gains `"assumed"`.
  - `tests/test_diagnostics.py:362` gains `"assumed"`.
  - `tests/test_settings_persistence.py:340` reads `"(t_scale, T_obs) diversity count: every row in a batch shares one operating point, so "`, and the message at `:110` says `(t_scale, T_obs)`.

- [ ] **Step 6: Run the new tests and see them pass.** Same command as Step 3. Expected: all pass.

- [ ] **Step 7: Run the focused tests for every file touched.**
  - `python -m pytest tests/test_tier1.py -m "not slow" -q`
  - `python -m pytest tests/test_nav_and_gating.py -q -k "t_obs_where or validate_tab or budget"`
  - `python -m pytest tests/test_settings_persistence.py -q`
  - `python -m pytest tests/test_diagnostics.py -q -k sbc`
  - `python -m pytest tests/test_artifact_store.py -q -k "inference_records or calibration_writes"`
  - `python -m pytest tests/test_user_sbi.py -q -k "budget_credits_once_per_training_batch"` (the batch-tag pin is untouched)
  - Expected: all pass.

- [ ] **Step 8: Commit.**
  - `git add core/sim_config.py core/orchestrator.py core/SBI/analysis.py core/diagnostics/sbc.py core/SBI/pipeline.py core/gui/fields.py core/gui/panels/inference/help_text.py core/gui/panels/inference/base.py tests/_fixtures.py tests/test_tier1.py tests/test_nav_and_gating.py tests/test_artifact_store.py tests/test_diagnostics.py tests/test_settings_persistence.py`
  - `git commit -m "report temperature as an assumed input" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"`
  - The controller starts the fast gate with the review; do not run it.

---

### Task 15: The calibration verdict and a repeatable calibration set

**Files:**
- Modify: `core/rng.py`. Add `CALIBRATION_TAG` and `calibration_seed` after `SEED_MAX` (near :34). The module docstring's paragraph ending "no ``seed`` argument on any stage" (near :12-18) names the one exception.
- Modify: `core/orchestrator.py`:
  - import `calibration_seed, require_seed, seeded` from `.rng` at the top;
  - new `calibration_verdict` and private `_verdict_message` just above `def validate_calibration(` (near :1723);
  - `validate_calibration` gains `seed` and the seeded block (from `val_latent_prior, T, truncation = _calibration_prior(` near :1788 through the informativeness `try/except`), and writes `results["verdict"]`, `results["seed"]` and the verdict record.
- Modify: `core/diagnostics/sbc.py`. After the repeat loop, before `results = {` (near :218), add `results["rank_verdict"]` and its info line.
- Modify: `core/tool/stages.py`:
  - `validate`'s parser block (`p = subparsers.add_parser("validate"`, near :77) gains `--seed`;
  - `def _validate(args, store)` (near :166) draws and passes the seed;
  - the module docstring names the exception;
  - `import random`.
- Modify: `core/tool/help_defaults.py`: add the entry `("validate", "--seed")`.
- Modify: `core/gui/panels/inference/validate_tab.py`: `_validate` (near :57) passes a drawn seed; `_on_calibration` (near :73) names the verdict and the seed; the class docstring's "Persists: nothing." paragraph.
- Test: `tests/test_artifact_store.py`, `tests/test_tier1.py`, `tests/test_diagnostics.py`, `tests/test_tool.py`, `tests/test_nav_and_gating.py`

**Interfaces:**
- Consumes:
  - Task 14: `SimConfig.assumed_params`, `tests._fixtures.stub_calibration_battery`;
  - Task 11: `core.tool.help_defaults.DEFAULT_CLASS`, `default_for(leaf_path, action)`;
  - Task 12: `tier1_cfg`, `_close`.
- Produces:
  - `core.rng.CALIBRATION_TAG: int` and `calibration_seed(seed: int) -> int`.
  - `core.orchestrator.calibration_verdict(ks_pvals: Sequence[float], names: Sequence[str], assumed: Sequence[str], tarp_ks_p: float, *, alpha: float = 0.05) -> dict`. It returns `{"passed": bool, "alpha": 0.05, "threshold": alpha / len(names), "parameters": [{"name", "ks_p", "passed", "assumed"}], "tarp": {"ks_p", "passed"}, "caveat": str}`.
  - `validate_calibration(..., seed: int | None = None)`, writing `body.results["verdict"]` and `body.results["seed"]`. The verdict record is one info record starting `"[verdict] "`.
  - `sbc` writes `results["rank_verdict"] = {"per_repeat": [bool], "fraction_passed": float, "scope": "rank-uniformity half only"}`.
  - Tool: `validate --seed N` (field key `seed`).

**Binding (read before starting):**
- `python` below means `C:\Users\J\anaconda3\envs\biophys-env\python.exe` (in the Bash tool, `/c/Users/J/anaconda3/envs/biophys-env/python.exe`); the `python` on PATH is a different interpreter without torch or pytest. Commands written with backslash paths are PowerShell commands; in Bash use forward slashes.
- Spec §5.4, H11 in §1.1, and §1.2 "Seeds" are binding; read them.
- **The rule.** PASS when both hold:
  - **every** inferred parameter's rank-test KS p ≥ `alpha / n`, with `n = len(names)` and the assumed temperature **included** (0.05/13 = 0.003846… on the tier-1 box);
  - the joint coverage (TARP) KS p ≥ `alpha` (0.05).
  - Comparisons are `>=`, so a p exactly at the threshold passes.
  - A None or non-finite p **fails**, and is recorded as `None` (the manifest refuses NaN).
  - A length mismatch between `ks_pvals` and `names` is a programming error: `ValueError`.
- `calibration_verdict` is a pure function: no `@public_entry`. The public-entry set is pinned by `tests/test_artifact_store.py:3410`.
- **A FAIL is a result**:
  - the stage completes, writes its record and returns;
  - the tool exits 0;
  - the verdict record is the stage's **last** `core.orchestrator` record, so the Validate tab's log shows it.
- **The caveat**: `"t_scale's rank test rests on the calibration's (t_scale, T_obs) operating points, not on its datasets, so it has less power than the other parameters' tests."` The record's caveat line adds `f" This calibration: {cal_n_scales} operating points for {n_cal} datasets."`.
- **The verdict record**: exact copy in Step 4.
- **Seeds.**
  - `seed=None` (the default) enters **no** seeded block and draws nothing extra. The calibration runs on in the caller's streams, so `smoke`'s one seed covers its calibration.
    - `core/tool/smoke.py` stays unchanged and passes no seed. `tests/test_tool.py::test_every_smoke_flag_reaches_its_stage_as_a_keyword` already pins `set(kw3) == {"fig_sink", "store", "n_cal"}`.
  - A given seed:
    - is validated with `require_seed` **before** `store.create`: a Refusal with field key `seed`, nothing written;
    - runs everything that draws inside `seeded(calibration_seed(seed), cfg.hw.device)`: `_calibration_prior`, the draw, SBC and its reference sample, TARP and informativeness. The verdict depends on the posterior's own sampling, so "same seed, same verdict" needs the SBC and TARP draws inside the block too, not only the set's draw.
  - Open the block with `contextlib.nullcontext()` when `seed` is None. Keep exactly one call each to `_calibration_prior`, `_draw_calibration_set` (with `chi_k_fixed=None`) and `_sbc_reference_sample`: `tests/test_diagnostics.py:314` parses the source and pins those calls.
  - `calibration_seed` imports numpy inside the function: `core/rng.py` stays free of torch and numpy at import. Its body is `int(np.random.SeedSequence([int(seed), CALIBRATION_TAG]).generate_state(1)[0])`, with `CALIBRATION_TAG = 0x43414C` (the ASCII bytes of "CAL").
- **Front ends.**
  - The tool always passes a seed: `args.seed`, else `random.randrange(2 ** 31)` (the house draw of `core/FDT/fdt_pipeline.py::_resolve_seed`).
  - The Validate tab draws one the same way on every run. There is no Seed box and nothing is remembered: `CONTROL["seed"]` stays the FDT/CrossVal one.
  - Help:
    - `--seed` has metavar `N` and `type=int, default=None`;
    - its `DEFAULT_CLASS[("validate", "--seed")]` entry copies exactly the class and text of `DEFAULT_CLASS[("fdt", "--seed")]`;
    - the help string ends with the clause `default_for("validate", action)` returns;
    - `--help` stays torch-free (`random` is stdlib).
- H4 applies to every line you write. The rng docstring and the test docstrings state the reason in words, with no labels.
- Messages stay `logging` records. Knobs stay arguments. Never edit a source file while pytest runs.

- [ ] **Step 1: Write the failing verdict tests** in `tests/test_artifact_store.py`:

  ```python
  NAMES = ["k", "lam", "f_max", "tau", "tau_c", "s", "delta_E", "beta", "n", "temp", "x_scale", "t_scale", "T"]


  def test_the_calibration_verdict_judges_every_parameter_at_a_family_wise_threshold():
      from core.orchestrator import calibration_verdict
      ok = [0.5] * 13
      v = calibration_verdict(ok, NAMES, ("T",), 0.3)
      assert v["passed"] is True and v["alpha"] == 0.05 and v["threshold"] == pytest.approx(0.05 / 13)
      assert [p["name"] for p in v["parameters"]] == NAMES
      assert [p["assumed"] for p in v["parameters"]] == [False] * 12 + [True]
      assert v["tarp"] == {"ks_p": 0.3, "passed": True} and "operating points" in v["caveat"]
      assumed_fails = ok[:12] + [0.0038]                      # under 0.05/13: the assumed one counts
      assert calibration_verdict(assumed_fails, NAMES, ("T",), 0.3)["passed"] is False
      assert calibration_verdict([0.05 / 13] + ok[1:], NAMES, ("T",), 0.3)["passed"] is True   # >= passes
      assert calibration_verdict([0.004] * 13, NAMES, (), 0.3)["passed"] is True               # corrected, not 0.05
      tarp_fails = calibration_verdict(ok, NAMES, ("T",), 0.049)
      assert tarp_fails["passed"] is False and all(p["passed"] for p in tarp_fails["parameters"])
      assert calibration_verdict(ok, NAMES, ("T",), 0.05)["passed"] is True
      nan = calibration_verdict(ok[:3] + [float("nan")] + ok[4:], NAMES, (), 0.3)
      assert nan["passed"] is False
      assert nan["parameters"][3] == {"name": "tau", "ks_p": None, "passed": False, "assumed": False}
      assert calibration_verdict(ok, NAMES, (), float("nan"))["tarp"] == {"ks_p": None, "passed": False}
      with pytest.raises(ValueError):
          calibration_verdict(ok[:12], NAMES, (), 0.3)


  def test_validate_ends_on_one_verdict_record_and_a_failing_verdict_still_writes_the_calibration(store, monkeypatch, caplog):
      from core import orchestrator
      from tests._fixtures import stub_calibration_battery
      seen = stub_calibration_battery(monkeypatch, ks_pvals=[0.5] * 12 + [0.003], tarp_ks_p=0.2)
      caplog.set_level("INFO", logger="core")
      cal = orchestrator.validate_calibration(_nad_cfg(), seen["posterior"], seen["prior"], n_cal=8,
                                              cal_n_scales=2, seed=3, fig_sink=_close, store=store,
                                              name="failing")
      v = cal.results["verdict"]
      assert v["passed"] is False and [p["passed"] for p in v["parameters"]] == [True] * 12 + [False]
      assert v["tarp"] == {"ks_p": 0.2, "passed": True} and cal.results["seed"] == 3
      assert store.get("calibration", "failing").body["results"]["verdict"] == v
      msgs = [r.getMessage() for r in caplog.records if r.name == "core.orchestrator"]
      verdicts = [m for m in msgs if m.startswith("[verdict] ")]
      assert len(verdicts) == 1 and msgs[-1] == verdicts[0]
      head, *rows = verdicts[0].splitlines()
      assert head == ("[verdict] FAIL: every parameter's rank test at KS p >= 0.05/13 = 0.003846 and "
                      "the joint coverage test at KS p >= 0.05 (calibration seed 3)")
      assert [r for r in rows if r.split()[:1] == ["f_scale"]][0].rstrip().endswith("FAIL")
      assert v["caveat"] in verdicts[0] and "2 operating points for 8 datasets" in verdicts[0]
      with pytest.raises(Refusal) as e:
          orchestrator.validate_calibration(_nad_cfg(), seen["posterior"], seen["prior"], seed=-1, store=store)
      assert e.value.field == "seed" and len(store.list("calibration")) == 1
  ```

  The Review Focus 3 tests, in the same file:

  ```python
  def test_the_same_validate_seed_draws_the_same_set_and_reaches_the_same_verdict(store, monkeypatch):
      from core import orchestrator
      from tests._fixtures import stub_calibration_battery
      seen = stub_calibration_battery(monkeypatch)            # KS p-values drawn from torch's stream
      cfg = _nad_cfg()
      run = lambda s: orchestrator.validate_calibration(cfg, seen["posterior"], seen["prior"], n_cal=8,
                                                        cal_n_scales=2, seed=s, fig_sink=_close, store=store)
      a, b = run(7), run(7)
      assert torch.equal(seen["x_cal"][0], seen["x_cal"][1])
      assert a.results["verdict"] == b.results["verdict"] and a.results["sbc"] == b.results["sbc"]
      assert a.results["seed"] == b.results["seed"] == 7
      run(8)
      assert not torch.equal(seen["x_cal"][2], seen["x_cal"][0])


  def test_a_seeded_calibration_never_replays_the_stream_a_training_run_with_that_seed_starts(store, monkeypatch):
      from core import orchestrator
      from core.rng import SEED_MAX, calibration_seed, seeded
      from tests._fixtures import stub_calibration_battery
      assert calibration_seed(7) == calibration_seed(7) != 7 and calibration_seed(7) != calibration_seed(8)
      assert all(isinstance(calibration_seed(s), int) and 0 <= calibration_seed(s) <= SEED_MAX
                 for s in (0, 1, SEED_MAX))
      seen = stub_calibration_battery(monkeypatch)
      cfg = _nad_cfg()
      orchestrator.validate_calibration(cfg, seen["posterior"], seen["prior"], n_cal=8, cal_n_scales=2,
                                        seed=7, fig_sink=_close, store=store)
      drawn = seen["x_cal"][0]
      with seeded(7, cfg.hw.device):
          assert not torch.equal(torch.randn(drawn.shape), drawn), "the set replayed a training run's stream"
      with seeded(calibration_seed(7), cfg.hw.device):
          assert torch.equal(torch.randn(drawn.shape), drawn)


  def test_validate_without_a_seed_runs_on_in_the_callers_stream(store, monkeypatch):
      import numpy as np
      from core import orchestrator
      from core.rng import seeded
      from tests._fixtures import stub_calibration_battery
      seen = stub_calibration_battery(monkeypatch)
      monkeypatch.setattr(orchestrator, "seeded",
                          lambda *a, **k: pytest.fail("a calibration with no seed entered a seeded block"))
      cfg = _nad_cfg()
      with seeded(5, cfg.hw.device):                          # smoke's one stream, and its next draws
          numpy_next = np.random.get_state()[1].copy()
          torch_next = torch.randn(8, 5)
      with seeded(5, cfg.hw.device):
          fresh = torch.randn(3)
      with seeded(5, cfg.hw.device):
          cal = orchestrator.validate_calibration(cfg, seen["posterior"], seen["prior"], n_cal=8,
                                                  cal_n_scales=2, fig_sink=_close, store=store)
          after = torch.randn(3)
      assert torch.equal(seen["x_cal"][0], torch_next) and np.array_equal(seen["numpy_at_draw"][0], numpy_next)
      assert not torch.equal(after, fresh), "the caller's stream was handed back rather than run on"
      assert cal.results["seed"] is None
  ```

- [ ] **Step 2: Write the other failing tests.**
  - **Real run** (the real-sbi pin): in `tests/test_artifact_store.py::test_calibration_writes_results_ranks_figures_and_refuses_a_foreign_prior`, after `assert r.store.load_calibration(cal.id).results == res`, add `assert res["seed"] is None and res["verdict"]["threshold"] == pytest.approx(0.05 / len(keys)) and [p["name"] for p in res["verdict"]["parameters"]] == keys`.
  - **`tests/test_tier1.py`, `test_the_calibration_verdict_judges_temperature_and_marks_it_assumed(store, monkeypatch, caplog)`:**
    - Run `stub_calibration_battery(monkeypatch, ks_pvals=[0.5] * 12 + [0.001])`, then validate `tier1_cfg()` with `seed=1`.
    - Assert `v["passed"] is False` and `v["parameters"][-1] == {"name": "T", "ks_p": 0.001, "passed": False, "assumed": True}`.
    - Assert the verdict record has exactly one row starting (stripped) with `"T (assumed input)"`, and it ends with `"FAIL"`.
  - **`tests/test_diagnostics.py`:**

  ```python
  def test_sbc_records_the_rank_uniformity_half_of_the_verdict_per_repeat(store, monkeypatch, caplog):
      from matplotlib import pyplot as plt
      from core.diagnostics import sbc_repeats
      from tests._fixtures import stub_calibration_battery
      good, bad, edge = [0.5] * 13, [0.5] * 12 + [0.003], [0.05 / 13] * 13
      seen = stub_calibration_battery(monkeypatch, ks_pvals=[good, bad, good, edge])
      caplog.set_level("INFO", logger="core")
      d = sbc_repeats(_nad_cfg(), seen["posterior"], seen["prior"], repeats=4, n_cal=8,
                      num_posterior_samples=10, fig_sink=lambda t, f: plt.close(f), store=store)
      assert d.results["rank_verdict"] == {"per_repeat": [True, False, True, True], "fraction_passed": 0.75,
                                           "scope": "rank-uniformity half only"}
      lines = [r.getMessage() for r in caplog.records if r.name == "core.diagnostics.sbc"]
      assert any("rank-uniformity half" in m and "3/4 repeats pass" in m for m in lines), lines
  ```

  - **`tests/test_tool.py`, validate keyword pin** (`test_every_validate_infer_and_tsnpe_flag_reaches_its_stage_as_a_keyword`, the "validate: every VALIDATE_KNOBS flag" leg near :583):
    - Add `"--seed", "5"` to the argv.
    - The key set becomes `{"name", "note", "fig_sink", "store", "n_cal", "cal_n_scales", "num_posterior_samples", "seed"}`, with `kw["seed"] == 5`.
    - Then `assert main(["validate", *_cfg(bounds), "--posterior", "tpost"]) == 0; drawn = v.calls[1][1]["seed"]; assert type(drawn) is int and 0 <= drawn < 2 ** 31`.
  - **`tests/test_tool.py`, new test `test_validate_takes_a_seed_and_a_failing_verdict_still_exits_zero(tool_run, monkeypatch, capsys)`:**
    - Run `stub_calibration_battery(monkeypatch, ks_pvals=[0.5, 0.5, 0.5, 0.001])` (SBITEST infers 4 parameters).
    - `main(["validate", *_cfg(bounds), "--posterior", "tpost", "--seed", "3", "--n-cal", "8", "--name", "failing_cal"]) == 0`.
    - stdout contains `"[verdict] FAIL"` and `"(calibration seed 3)"`.
    - `ArtifactStore(root).get("calibration", "failing_cal").body["results"]` has `seed == 3` and `verdict["passed"] is False`.
  - **`tests/test_nav_and_gating.py`:**
    - In `test_the_validate_tab_refuses_bad_boxes_at_the_click_and_dispatches_nothing`, after the last assert, add `seed = sent["kwargs"]["seed"]; assert type(seed) is int and 0 <= seed < 2 ** 31`.
    - New `test_the_validate_tab_summary_names_the_verdict_and_the_seed()`:
      - build the screen as that test does, and set `vp.log_pane.append_line = lines.append`;
      - call `vp._on_calibration(SimpleNamespace(name="c1", id="x", results={"tarp": {"atc": 0.01, "ks_p": 0.4}, "informativeness": None, "verdict": {"passed": False}, "seed": 11}))`;
      - assert `lines[-1].startswith("Calibration verdict FAIL (seed 11), recorded as c1: TARP ATC=0.01, KS p=0.4")`.

- [ ] **Step 3: Run the new tests and see them fail.**
  - `python -m pytest tests/test_artifact_store.py -q -k "verdict or validate_seed or seeded_calibration or without_a_seed"`
  - `python -m pytest tests/test_tier1.py -q -k verdict`
  - `python -m pytest tests/test_diagnostics.py -q -k rank_uniformity`
  - Expected: ImportError/AttributeError on `calibration_verdict`, `calibration_seed` and `orchestrator.seeded`; TypeError on the `seed` keyword; KeyError on `rank_verdict`.

- [ ] **Step 4: Implement `core/rng.py` and the orchestrator side.**
  - `CALIBRATION_TAG` and `calibration_seed`, as in the Binding.
  - The docstring paragraph gains: `validate` with a seed is the one stage that seeds itself. It draws from a stream derived from the seed and a fixed calibration tag, so it never starts where a training run seeded with the same number started. With no seed it takes none, and runs on in the caller's streams.
  - `calibration_verdict`, per the Binding, using `_num` for every p.
  - `_verdict_message(verdict: dict, seed: "int | None", n_scales: int, n_cal: int) -> str`, one multi-line message:

  ```python
  head = (f"[verdict] {'PASS' if verdict['passed'] else 'FAIL'}: every parameter's rank test at KS p >= "
          f"{verdict['alpha']:g}/{len(verdict['parameters'])} = {verdict['threshold']:.4g} and the joint "
          f"coverage test at KS p >= {verdict['alpha']:g}"
          + ("" if seed is None else f" (calibration seed {seed})"))
  row = lambda label, p, ok: (f"  {label:<20s} KS p {'n/a' if p is None else f'{p:.4g}':<8s} "
                              f"{'pass' if ok else 'FAIL'}")
  lines = [head] + [row(p["name"] + (" (assumed input)" if p["assumed"] else ""), p["ks_p"], p["passed"])
                    for p in verdict["parameters"]]
  lines += [row("joint coverage", verdict["tarp"]["ks_p"], verdict["tarp"]["passed"]),
            f"  {verdict['caveat']} This calibration: {n_scales} operating points for {n_cal} datasets."]
  return "\n".join(lines)
  ```

  - In `validate_calibration`, add `seed: int | None = None`:
    - `seed_used = None if seed is None else require_seed(seed)` beside the other refusals, before `store.create`;
    - the seeded block per the Binding;
    - after the informativeness `try/except`: `verdict = calibration_verdict(sbc_stats["ks_pvals"], keys, cfg.assumed_params, tarp_kspval)`;
    - `results` gains `"verdict": verdict, "seed": seed_used`;
    - `log.info(_verdict_message(verdict, seed_used, n_scales_used, n_cal_used))` as the last record, before `_write_results_json`;
    - document `:param seed:` in words.

- [ ] **Step 5: Implement `sbc` and the front ends.**
  - **`sbc.py`:**
    - `per_repeat = [all(p["passed"] for p in orch.calibration_verdict(ks[r].tolist(), labels, cfg.assumed_params, float("nan"))["parameters"]) for r in range(repeats)]`. This is the rule's rank half, written once. `n_pass = sum(per_repeat)`.
    - `log.info(f"[sbc] rank-uniformity half of the calibration verdict (every parameter's KS p >= 0.05/{len(labels)}): {n_pass}/{repeats} repeats pass; the joint coverage half is the calibration stage's")`.
    - `results["rank_verdict"] = {"per_repeat": per_repeat, "fraction_passed": n_pass / repeats, "scope": "rank-uniformity half only"}`.
  - **`stages.py`:**
    - the `--seed` flag per the Binding. Help text: `"the calibration set's random seed: the same seed draws the same set and reaches the same verdict, and never replays the stream a training run with that seed used"`, then the default clause.
    - In `_validate`: `seed = args.seed if args.seed is not None else random.randrange(2 ** 31)`, passed as `seed=seed`.
    - The module docstring's "every value flag … forwarded only when it was given" sentence names this one exception and why: the record must name the seed it ran with.
    - The `DEFAULT_CLASS` entry.
  - **`validate_tab.py`:**
    - `import random`; the dispatch adds `seed=random.randrange(2 ** 31)`;
    - `_on_calibration` builds `f"Calibration verdict {'PASS' if (res.get('verdict') or {}).get('passed') else 'FAIL'} (seed {res.get('seed')}), recorded as {payload.name or '(unnamed, id ' + payload.id + ')'}: TARP ATC={res['tarp']['atc']}, KS p={res['tarp']['ks_p']}"` plus the existing informativeness tail;
    - the docstring says each run draws its seed afresh and the record keeps it; no Seed box, nothing remembered.

- [ ] **Step 6: Run the new tests and see them pass.** The commands of Step 3 plus:
  - `python -m pytest tests/test_tool.py -q -k "validate_infer_and_tsnpe or failing_verdict or smoke_flag"`
  - `python -m pytest tests/test_nav_and_gating.py -q -k validate_tab`
  - Expected: all pass.

- [ ] **Step 7: Run the focused tests for every file touched.**
  - `python -m pytest tests/test_artifact_store.py -q -k "verdict or seed or calibration or public_entr"`
  - `python -m pytest tests/test_diagnostics.py -q -k "sbc or seeded or validate_calibration"`
  - `python -m pytest tests/test_tier1.py -m "not slow" -q`
  - `python -m pytest tests/test_tool.py -q -k "validate or help or default or placeholder or smoke_flag"` (Task 11's help-walking test must pass with the new entry)
  - `python -m pytest tests/test_refusals.py -q`
  - `python -m pytest tests/test_nav_and_gating.py -q -k validate_tab`
  - `python -m pytest tests/test_settings_persistence.py -q -k "restores_its_observation_and_budget_only or consents_are_never_persisted"`
  - Expected: all pass.

- [ ] **Step 8: Commit.**
  - `git add core/rng.py core/orchestrator.py core/diagnostics/sbc.py core/tool/stages.py core/tool/help_defaults.py core/gui/panels/inference/validate_tab.py tests/test_artifact_store.py tests/test_tier1.py tests/test_diagnostics.py tests/test_tool.py tests/test_nav_and_gating.py`
  - `git commit -m "calibration verdict and a repeatable calibration seed" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"`
  - The controller starts the one-process fast gate (background, logged) at the same moment as this task's review. Do not run it yourself.

---

### Task 16: Eigenvalues across a resume; truth containment both ways

**Files:**
- Modify: `core/SBI/training_checkpoint.py` (`def create(` ~:165, the header dict beside `"V": None if V is None else V.detach().cpu(),` ~:182; the module docstring's LAYOUT line for `header.pt`)
- Modify: `core/SBI/pipeline.py` (the preflight `_tc.create(_ck_dir, checkpoint["identity"],` ~:1427; the `:param checkpoint:` key list in `gen_training_data`'s docstring ~:1252-1264)
- Modify: `core/SBI/train.py` (`TrainingPlan` docstring, "its key set (dir/identity/probe/V/every/resume)", text only)
- Modify: `core/SBI/truncate.py` (`class TruncationRegion`, new method after `def contains(` ~:162)
- Modify: `core/orchestrator.py` (`build_posterior`: the comment and `fisher_evals = None` ~:1083-1085; the truncation branch ~:1086-1163; the resume branch `elif ckpt_resumed is not None and rotate:` ~:1164-1182; the fresh branch ~:1183-1196; the `if cfg.has_ground_truth:` block ~:1215-1232; `training_params.checkpoint = {` ~:1275; the comment above `_fisher_ran = fisher_evals is not None` ~:1364-1370; `"fisher_eigenvalues": tensor_to_json(fisher_evals)` ~:1387; the `"training": {` dict ~:1390-1399)
- Modify: `tests/_fixtures.py` (new helper `stubbed_training`, placed after `class _FakeDP`)
- Test: `tests/test_artifact_store.py`, `tests/test_conditioning_repair.py`, `tests/test_user_sbi.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: checkpoint header key `"fisher_eigenvalues"` (list[float] or None; readers use `.get`); `TruncationRegion.containment(theta_latent: Tensor) -> list[dict]` → `[{"direction": int, "lo": float, "hi": float, "value": float, "inside": bool}]` (`value` is None when the coordinate is not finite, because a manifest refuses a non-finite number); the child posterior's `body.training["truth_containment"]` holds that list (None when no known cell is loaded); info lines start `"[tsnpe] truth "`. Test helper `tests/_fixtures.py::stubbed_training(store, monkeypatch, *, rotate=False)` (Task 19 uses it).

**Binding (read before starting):**
- `python` below means `C:\Users\J\anaconda3\envs\biophys-env\python.exe` (in the Bash tool, `/c/Users/J/anaconda3/envs/biophys-env/python.exe`); the `python` on PATH is a different interpreter without torch or pytest. Commands written with backslash paths are PowerShell commands; in Bash use forward slashes.
- Spec §5.5 and §5.6 (read both). The header stores the rotation's eigenvalues beside V. A resumed run records the header's eigenvalues in `body.transform.fisher_eigenvalues`; a narrowing child records its parent posterior's (`parent_posterior.manifest.body["transform"]["fisher_eigenvalues"]`) because it reuses that rotation; a region with no rotation, or no parent posterior, gives None. A header written before this task has no key: `.get` reads it as unknown (None), the run still resumes, and one info record says the eigenvalues are unknown.
- **The witness that the Fisher ran stays `fisher_evals`, set only by the freshly computed branch.** Keep `_fisher_ran = fisher_evals is not None` exactly; `fisher_m`/`fisher_dz`/`fisher_points` stay None on a resume and on a truncated round. The eigenvalues to record travel in a separate local. `body.training["fisher_spread"]` stays the fresh branch's value (None on a resume or a child).
- Body keys are a closed set per kind (`core/artifacts/manifest.py` `BODY_KEYS`): new facts go inside the existing `transform` and `training` dicts, never as a new top-level body key.
- Resume line prefix is `[checkpoint] `, never `[fisher]`: the card recipe reads "[fisher] eigenvalue spread" as the sign that a Fisher was computed, and `tests/test_tool.py`'s `_GATE_LINES` pins that text to `log.info` calls only.
- The new `[tsnpe] truth ` info lines must NOT contain the substring `GROUND TRUTH` (`tests/test_user_sbi.py::test_a_tsnpe_round_reuses_the_parents_basis_and_refuses_every_mismatch` asserts no record carries it). The existing warning (`[tsnpe] WARNING: the loaded cell's GROUND TRUTH lies OUTSIDE ...`) stays verbatim, said once, as the `PreflightWarning`. `_truth_outside_region` stays unchanged: `tests/test_conditioning_repair.py::test_the_round_reads_this_observations_truth_and_no_other` pins its wording.
- `core/SBI/truncate.py` has exactly three `log.*` calls, and they are pinned by `tests/test_user_sbi.py::test_every_sbi_message_is_a_record_at_its_level_and_nothing_prints`. `containment` logs nothing. The info lines are logged from `core/orchestrator.py`.
- Messages are `logging` records, never `print`. No refusal is added. H4: no piece numbers, §N, decision ids (V7, D1, …), "guardrail N", "Task N", or trap or review ids go into any comment, docstring, message, test name or test docstring you write or edit. State the reason in words.
- Never edit a source file while a pytest run is in progress. Run only focused tests.

- [ ] **Step 1: Write the failing tests.**

In `tests/_fixtures.py` add `stubbed_training(store, monkeypatch, *, rotate: bool = False) -> SimpleNamespace`. It builds `_nad_cfg()` with `hw.batch_size = 4` and `reparam_rotate = rotate`. It loads a real prior with `store.load_prior(cfg, _prior_artifact(store, cfg, name="p").id)`. It monkeypatches `core.orchestrator.pipeline.train_nn` with a stand-in that sets `seen["plan"] = plan` and returns `(_FakeDP with .prior = kw["prior"], {"training_loss": [1.0], "validation_loss": [1.0], "best_validation_loss": 1.0, "epochs_trained": 1, "stop_after_epochs": 1})`. It returns `SimpleNamespace(cfg, prior, P=len(params_dict)+len(rescale_params), seen, budget=dict(run_size_cap=4, hidden_features=8, num_transforms=1, stop_after_epochs=1, fig_sink=<closes the figure>))`. The caller passes `num_runs` and `checkpoint_every` itself. Give it a docstring that says what it builds.

In `tests/test_artifact_store.py`:
```python
def test_the_training_cache_header_keeps_the_rotation_eigenvalues_beside_the_rotation(tmp_path):
    """The header written before the first simulation stores the rotation's eigenvalues next to V, so a
    resumed run can record how strongly each direction is constrained without re-running the Fisher."""
    from core.SBI import training_checkpoint as tc
    Q, _ = torch.linalg.qr(torch.randn(3, 3))
    for ident, V, evals, want in (({"model": "a"}, Q, torch.tensor([3.0, 2.0, 1.0]), [3.0, 2.0, 1.0]),
                                  ({"model": "b"}, None, None, None)):
        d = tc.resolve_dir(ident, tmp_path)
        tc.create(d, ident, schedule_t_scales=torch.ones(2), schedule_Ts=torch.ones(2), inits=torch.zeros(1, 2),
                  V=V, fisher_eigenvalues=evals, probe=torch.zeros(0), run_size=1, n_runs=2)
        assert tc.read_header(d)["fisher_eigenvalues"] == want
```
In `test_fisher_settings_are_recorded_only_when_the_rotation_ran`, leg (c) (the `tc.create(d, ident, ...)` with `V=Q`, ~:3640): pass `fisher_eigenvalues=[float(v) for v in range(P, 0, -1)]` and replace the leg's final assertion with:
```python
    assert got == (None, None, None), (got, m.config)
    assert m.body["transform"]["fisher_eigenvalues"] == [float(v) for v in range(P, 0, -1)], m.body["transform"]
```
Leg (d) keeps `fisher_eigenvalues is None`. Add `assert m.body["training"]["truth_containment"] is None` there (no cell is loaded). Then add these two tests:
```python
def test_a_fresh_rotated_run_hands_its_eigenvalues_to_the_cache_and_a_narrowing_child_inherits_them(store, monkeypatch):
    """The Fisher's eigenvalues travel into the simulation cache's plan beside the rotation; a narrowing
    round reuses its parent's rotation and never runs the Fisher, so it records the parent's
    eigenvalues and no Fisher settings of its own."""
    from types import SimpleNamespace
    from core import orchestrator
    from core.SBI import reparam, truncate, training_checkpoint as tc
    from core.SBI.run_guards import _log_params_for
    from tests._fixtures import stubbed_training
    h = stubbed_training(store, monkeypatch, rotate=True)
    Q, _ = torch.linalg.qr(torch.randn(h.P, h.P))
    evals = [float(v) for v in range(h.P, 0, -1)]
    monkeypatch.setattr(orchestrator.decorrelate, "build_latent_fisher_rotation", lambda *a, **k: (Q, torch.tensor(evals)))
    fresh = orchestrator.build_posterior(h.cfg, h.prior, None, True, num_runs=3, checkpoint_every=1, **h.budget)
    assert h.seen["plan"].checkpoint["fisher_eigenvalues"] == evals
    parent_m = store.get("posterior", fresh.id)
    assert parent_m.body["transform"]["fisher_eigenvalues"] == evals

    def fisher_must_not_run(*a, **k):
        raise AssertionError("a narrowing round computed a Fisher")
    monkeypatch.setattr(orchestrator.decorrelate, "build_latent_fisher_rotation", fisher_must_not_run)
    T = reparam.build_inferred_bijection(h.cfg, log_params=_log_params_for(h.cfg))
    region = truncate.TruncationRegion(
        [0], [-50.0], [50.0], n_latent=h.P, V=Q, x_obs_digest="d" * 16,
        probe=tc.bijection_probe(reparam.build_rotated_bijection(T, Q), h.P, device=h.cfg.hw.device))
    parent = SimpleNamespace(id=fresh.id, accepted=[], manifest=parent_m)
    child = store.get("posterior", orchestrator.build_posterior(
        h.cfg, h.prior, None, True, num_runs=2, checkpoint_every=0, truncation=region,
        parent_posterior=parent, **h.budget).id)
    assert child.body["transform"]["fisher_eigenvalues"] == evals
    assert (child.config["fisher_m"], child.config["fisher_dz"], child.config["fisher_points"]) == (None, None, None)


def test_a_resume_from_a_checkpoint_that_stores_no_eigenvalues_records_them_as_unknown(store, monkeypatch, caplog):
    """A cache written before the eigenvalues were kept beside the rotation still resumes: the posterior
    records them as unknown, says so in an information record, and records no Fisher settings."""
    from core import orchestrator
    from core.artifacts.identity import SimulationIdentity
    from core.SBI import training_checkpoint as tc
    from tests._fixtures import stubbed_training
    h = stubbed_training(store, monkeypatch, rotate=True)

    def fisher_must_not_run(*a, **k):
        raise AssertionError("a resumed run recomputed the Fisher")
    monkeypatch.setattr(orchestrator.decorrelate, "build_latent_fisher_rotation", fisher_must_not_run)
    Q, _ = torch.linalg.qr(torch.randn(h.P, h.P))
    ident = SimulationIdentity.from_cfg(h.cfg, h.prior, 4, 2).to_dict()
    d = tc.resolve_dir(ident, store.kind_dir("simulation"))
    tc.create(d, ident, schedule_t_scales=torch.zeros(2), schedule_Ts=torch.zeros(2), inits=torch.zeros(4, 3),
              V=Q, probe=torch.zeros(0), run_size=4, n_runs=2)
    header = torch.load(d / "header.pt", weights_only=False)
    header.pop("fisher_eigenvalues", None)
    torch.save(header, d / "header.pt")
    torch.save({"batches_done": 1, "complete": False, "rng": None}, d / "state.pt")
    caplog.clear()
    m = store.get("posterior", orchestrator.build_posterior(
        h.cfg, h.prior, None, True, num_runs=2, checkpoint_every=1, **h.budget).id)
    assert m.body["training"]["resumed_from_batch"] == 1
    assert m.body["transform"]["fisher_eigenvalues"] is None
    assert (m.config["fisher_m"], m.config["fisher_dz"], m.config["fisher_points"]) == (None, None, None)
    said = [r.getMessage() for r in caplog.records if r.name == "core.orchestrator" and r.levelname == "INFO"]
    assert any(s.startswith("[checkpoint] ") and "unknown" in s for s in said), said
```
In `test_a_round_records_parent_posterior_and_observation_as_parents` (tiny_run, ~:1782), which runs a real round with a truth loaded, add:
```python
    cont = m.body["training"]["truth_containment"]
    assert [set(e) for e in cont] == [{"direction", "lo", "hi", "value", "inside"}], cont
```
In `tests/test_user_sbi.py`, near `_ck`:
```python
def test_the_training_generator_passes_the_eigenvalues_to_the_cache_header(monkeypatch, tmp_path):
    """gen_training_data hands the checkpoint's eigenvalues to the header it creates before the first
    simulation."""
    from core.SBI import training_checkpoint as tc
    handed = {}

    class _Stop(Exception):
        pass

    def _create(path, identity, **kw):
        handed.update(kw)
        raise _Stop

    monkeypatch.setattr(tc, "create", _create)
    with pytest.raises(_Stop):
        _gen_td("chi", seed=3, n_runs=2, run_size=4, checkpoint=_ck(tmp_path / "c", fisher_eigenvalues=[3.0, 2.0, 1.0]))
    assert handed["fisher_eigenvalues"] == [3.0, 2.0, 1.0]
```
In `tests/test_conditioning_repair.py`:
```python
def test_a_region_reports_containment_per_truncated_direction():
    """For one latent point: each truncated direction's bounds, the coordinate, and whether it lies in the
    closed interval. A coordinate that is not a finite number is recorded as None and never inside."""
    r = truncate.TruncationRegion([0, 2], [-1.0, 0.0], [1.0, 2.0], n_latent=4)
    assert r.containment(torch.tensor([0.5, 9.0, 3.0, -9.0])) == [
        {"direction": 0, "lo": -1.0, "hi": 1.0, "value": 0.5, "inside": True},
        {"direction": 2, "lo": 0.0, "hi": 2.0, "value": 3.0, "inside": False}]
    got = r.containment(torch.tensor([[float("nan"), 0.0, 2.0, 0.0]]))
    assert got[0]["value"] is None and got[0]["inside"] is False and got[1]["inside"] is True
```
In `tests/test_tier1.py::test_a_narrowing_round_on_a_tier1_parent_judges_the_truth_in_inferred_coordinates` (it exists since the tier-1 task), keep both `build_posterior` results and read each child's `store.get("posterior", out.id).body["training"]["truth_containment"]`: the inside region gives `[(e["direction"], e["inside"]) for e in cont] == [(12, True)]` with `cont[0]["value"] == pytest.approx(c, abs=1e-6)`, and the outside one gives `[(12, False)]`. This pins containment on a tier-1 parent in inferred coordinates. Also add `test_a_narrowing_round_reports_and_records_whether_the_truth_lies_inside_each_direction(store, monkeypatch, caplog)`. It uses `stubbed_training(store, monkeypatch)` (unrotated) and loads `master_weak.txt` with `cli.load_and_validate_gt`. Compute `z0 = T.inv(truth)[0, 0]` under `build_inferred_bijection(cfg, log_params=_log_params_for(cfg))`, and `w0` = dim 0 of `orchestrator._build_latent_prior_for_validation(cfg, h.prior.prior).sample((4000,))`. Run two unrotated rounds, each `build_posterior(..., num_runs=2, checkpoint_every=0, truncation=TruncationRegion([0], [lo], [hi], n_latent=P, probe=bijection_probe(T, P), x_obs_digest="d"*16))`, inside `warnings.catch_warnings(record=True)` with `simplefilter("always")`:
  - Inside, `[min(z0, q0.02) - 0.5, max(z0, q0.98) + 0.5]`: `[(e["direction"], e["inside"]) for e in truth_containment] == [(0, True)]`; `value == pytest.approx(z0, abs=1e-6)`; exactly one INFO record on `core.orchestrator` that starts with `"[tsnpe] truth direction 0:"` and contains `" inside ["`; no Python warning that contains `"GROUND TRUTH"`.
  - Outside, `[q0.6, q0.9]` if `z0 <= median` else `[q0.1, q0.4]`: `[(0, False)]`; one such record, and it contains `" outside ["`; exactly one warning that contains `"GROUND TRUTH"` and `"direction 0"`.

Also add `test_a_round_on_a_simulated_observation_reports_truth_containment_without_a_cell(store, monkeypatch, caplog)` in `tests/test_conditioning_repair.py`. The tool's `tsnpe` takes no `--cell`: the truth reaches `build_posterior` only because `tsnpe_round` calls `observation.install(cfg)` on a SIMULATED observation, and this test pins that path. Harness:
  - `h = stubbed_training(store, monkeypatch)`; `assert not h.cfg.has_ground_truth`.
  - A stored parent: `fresh = orchestrator.build_posterior(h.cfg, h.prior, None, True, num_runs=2, checkpoint_every=0, **h.budget)`, then `parent = SimpleNamespace(posterior=SimpleNamespace(x_obs_digest=None), id=fresh.id, name="", accepted=[], manifest=store.get("posterior", fresh.id))`.
  - A simulated observation, built the way `test_the_round_reads_this_observations_truth_and_no_other` builds its experimental one: `LoadedObservation(kind="observation", id="20300101T000000", name="", path=None, manifest=SimpleNamespace(body=body), digest="d" * 16)`, where `body` has `T_obs_cell` 200.0, `n_obs` 200, `chi_obs_freqs` None, `forcing_vals` {} and a `source` block of kind `"simulated"` holding the truth of a deep copy of `h.cfg` loaded with `master_weak.txt` (`cli.load_and_validate_gt`, the path through `config.CELL_PATH`), in exactly the form `orchestrator._write_observation` writes and `LoadedObservation.install` reads (`inits`, `params`, `rescale`, `forcing`).
  - `T = reparam.build_inferred_bijection(h.cfg, log_params=_log_params_for(h.cfg))`; monkeypatch `orchestrator.build_truncation_region` with `lambda *a, **k: truncate.TruncationRegion([0], [-1e6], [1e6], n_latent=h.P, probe=tc.bijection_probe(T, h.P, device=h.cfg.hw.device), x_obs_digest=obs.digest)`. `build_posterior` stays real.
  - `child = orchestrator.tsnpe_round(h.cfg, parent, h.prior, obs, num_runs=2, checkpoint_every=0, store=store, **h.budget)`.
  - Assert `[(e["direction"], e["inside"]) for e in store.get("posterior", child.id).body["training"]["truth_containment"]] == [(0, True)]`, and that exactly one INFO record on `core.orchestrator` starts with `"[tsnpe] truth direction 0:"`.

- [ ] **Step 2: Run them and see them fail.**

Run `python -m pytest tests/test_artifact_store.py::test_the_training_cache_header_keeps_the_rotation_eigenvalues_beside_the_rotation tests/test_artifact_store.py::test_fisher_settings_are_recorded_only_when_the_rotation_ran tests/test_artifact_store.py::test_a_fresh_rotated_run_hands_its_eigenvalues_to_the_cache_and_a_narrowing_child_inherits_them tests/test_artifact_store.py::test_a_resume_from_a_checkpoint_that_stores_no_eigenvalues_records_them_as_unknown tests/test_user_sbi.py::test_the_training_generator_passes_the_eigenvalues_to_the_cache_header tests/test_conditioning_repair.py::test_a_region_reports_containment_per_truncated_direction tests/test_conditioning_repair.py::test_a_narrowing_round_reports_and_records_whether_the_truth_lies_inside_each_direction tests/test_conditioning_repair.py::test_a_round_on_a_simulated_observation_reports_truth_containment_without_a_cell tests/test_tier1.py::test_a_narrowing_round_on_a_tier1_parent_judges_the_truth_in_inferred_coordinates -q`.

Expect these failures:
- `TypeError: create() got an unexpected keyword argument 'fisher_eigenvalues'`.
- A `KeyError` for `fisher_eigenvalues` on the plan's checkpoint dict.
- The missing `[checkpoint] ... unknown` record.
- `AttributeError: 'TruncationRegion' object has no attribute 'containment'`.
- A `KeyError` for `truth_containment`.

- [ ] **Step 3: The header.**

`training_checkpoint.create(..., fisher_eigenvalues=None)` is a new keyword-only argument. It writes `"fisher_eigenvalues": None if fisher_eigenvalues is None else [float(v) for v in fisher_eigenvalues]` into the header dict, beside `"V"` (it accepts a tensor or a list). No `CHECKPOINT_FORMAT` bump, because readers use `.get`. Add the eigenvalues to the LAYOUT line for `header.pt` in the module docstring.

- [ ] **Step 4: The generator forwards them.**

In `gen_training_data`'s preflight `_tc.create(...)` call, add `fisher_eigenvalues=checkpoint.get("fisher_eigenvalues")`. Add `fisher_eigenvalues` to the checkpoint key list in the `:param checkpoint:` docstring and to `TrainingPlan`'s docstring. No other line of the batch loop changes.

- [ ] **Step 5: `TruncationRegion.containment(self, theta_latent: torch.Tensor) -> list[dict]`.**

It takes one latent point, shape `(P,)` or `(1, P)`, and raises `ValueError` for more rows. It converts to CPU float64 itself. It returns one dict per entry of `self.dims`, in order: `lo`/`hi` from the exact bounds as floats; `value` the coordinate as a float, or None when it is not finite; and `inside` = finite and `lo <= value <= hi`, the same closed interval `contains` uses. No logging.

- [ ] **Step 6: The orchestrator.**

In `build_posterior`:
- **One local, `evals_rec: list[float] | None`, is what the record and the cache carry.**
  - Truncation branch: `_parent_fisher_eigenvalues(parent_posterior)` when `truncation.V is not None`, else None. Add a new private module-level helper `_parent_fisher_eigenvalues(parent_posterior) -> list[float] | None`. It reads `parent_posterior.manifest.body["transform"].get("fisher_eigenvalues")` and returns None for a missing parent, manifest or key.
  - Resume branch: `ckpt_resumed.get("fisher_eigenvalues")`.
  - Fresh branch: `tensor_to_json(fisher_evals)`.
  - Unrotated: None.
- **Rewrite the comment above `fisher_evals = None`.** `fisher_evals` remains the only witness that the Fisher ran in this process.
- **The resume branch logs one info record after the "Reusing the Fisher rotation…" record, whose text stays unchanged.**
  - When eigenvalues are stored: `f"[checkpoint] the stored rotation's eigenvalues come with it (spread best/worst {s:.3g}); the Fisher settings stay unrecorded, since the Fisher did not run in this process."`, where `s = e[0]/e[-1]`, or `inf` when `e[-1] <= 0`.
  - When none are stored: `"[checkpoint] this checkpoint stores no rotation eigenvalues (it was written before they were kept beside the rotation); the posterior records them as unknown."`
- **The cache and the record.** `training_params.checkpoint` gains `"fisher_eigenvalues": evals_rec`. `body["transform"]["fisher_eigenvalues"]` becomes `evals_rec`.
- **Truth containment.** Set `_containment = None` before `if truncation is not None:` (~:1209). Inside `if cfg.has_ground_truth:`, and before the existing `_bad = _truth_outside_region(...)`: take `z = T_train.inv(cfg.ground_truth_tensor.reshape(1, -1))` under `torch.no_grad()`, then `_containment = truncation.containment(z)`. For each entry, log one info record:
  - finite value: `f"[tsnpe] truth direction {d}: {v:+.3g} {'inside' if inside else 'outside'} [{lo:.3g}, {hi:.3g}]"`;
  - None: `f"[tsnpe] truth direction {d}: not a finite number, so not inside [{lo:.3g}, {hi:.3g}]"`.

  Keep the warning block unchanged. Add `"truth_containment": _containment` to the `"training"` dict.
- **Comment above `_fisher_ran`.** Update it so it says, in words, that the eigenvalues may come from the cache or the parent while the three settings are written only when the Fisher ran here.

- [ ] **Step 7: Run the Step 2 command again; all nine tests pass.**

- [ ] **Step 8: Run the neighbouring tests for every file touched.**

Run `python -m pytest tests/test_artifact_store.py::test_a_round_records_parent_posterior_and_observation_as_parents tests/test_user_sbi.py::test_a_tsnpe_round_reuses_the_parents_basis_and_refuses_every_mismatch tests/test_user_sbi.py::test_a_resumed_training_run_is_bit_identical_to_an_uninterrupted_one tests/test_user_sbi.py::test_a_resume_keeps_the_stratification_schedule_identical tests/test_user_sbi.py::test_every_sbi_message_is_a_record_at_its_level_and_nothing_prints tests/test_user_sbi.py::test_nothing_prints_or_logs_inside_a_checkpoint_commit tests/test_conditioning_repair.py::test_the_round_reads_this_observations_truth_and_no_other tests/test_tool.py::test_the_gate_lines_keep_their_text_and_stream tests/test_source_hygiene.py -q`.

Pass means every test passes. The first test builds the module's tiny real run, which takes a few minutes.

(Step 6a's test, `test_a_round_on_a_simulated_observation_reports_truth_containment_without_a_cell`, is written in Step 1 and runs in Steps 2 and 7.)

- [ ] **Step 9: Commit.**

`git add core/SBI/training_checkpoint.py core/SBI/pipeline.py core/SBI/train.py core/SBI/truncate.py core/orchestrator.py tests/_fixtures.py tests/test_artifact_store.py tests/test_conditioning_repair.py tests/test_user_sbi.py tests/test_tier1.py`, then `git commit -m "Keep Fisher eigenvalues across a resume; report truth containment" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"`.

The controller starts the one-process fast gate (`python -m pytest -m "not slow" -q`, in the background, logged) at the same moment as this task's review. Do not run it yourself.

---

### Task 17: The tier-1 constraint recorded; the identity format

**Files:**
- Modify: `core/artifacts/manifest.py` (`def config_from_cfg(cfg)` ~:247, after its returned dict for a SimConfig)
- Modify: `core/artifacts/identity.py` (`FORMAT = "training-rows/2"` ~:21; the `vals = {` dict in `SimulationIdentity.from_cfg` ~:41-75; the module docstring)
- Modify: `core/orchestrator.py` (`generate_observations`, the `rescale_gt = derived.to_sim_rescale(` line ~:136 and the `return _write_observation(` call ~:269; `def _write_observation(` ~:341)
- Test: `tests/test_tier1.py`, `tests/test_artifact_store.py` (`_IDENTITY_KEYS` ~:1041), `tests/test_user_sbi.py` (`test_a_truncated_round_routes_to_its_own_checkpoint_and_the_amortized_digest_is_untouched`, the golden `assert tc.identity_digest(ia) == "1912d2139359"` ~:2256 and the `assert set(ia) == {` ~:2258)

**Interfaces:**
- Consumes: `SimConfig.assumed_params -> tuple[str, ...]` (`("T",)` when tier 1 is on, else `()`); `tests/test_tier1.py` helpers `tier1_cfg(*, chi: bool = True, hw=None) -> SimConfig` and `force_scale_spy(monkeypatch) -> list`.
- Produces: `manifest.config_from_cfg(cfg)` adds `"tier1": {"relation": "f_scale = N * beta * k_B * T / x_scale", "k_b_cell": float, "T_range": [lo, hi]}` and `"assumed_params": list[str]` when tier 1 is on, and neither key otherwise. A simulated observation's config adds `"simulated_f_scale": float`. `core/artifacts/identity.py` has `FORMAT = "training-rows/3"`, and the identity values gain `"units_sha256"` (None when the config has no units source).

**Binding (read before starting):**
- `python` below means `C:\Users\J\anaconda3\envs\biophys-env\python.exe` (in the Bash tool, `/c/Users/J/anaconda3/envs/biophys-env/python.exe`); the `python` on PATH is a different interpreter without torch or pytest. Commands written with backslash paths are PowerShell commands; in Bash use forward slashes.
- Spec §5.7 (read it). Body keys are a closed set per kind (`manifest.py` `BODY_KEYS`, checked in `validate`), so every new fact goes in `config`, which is free-form but finite. Do not add a body key.
- "Tier 1 is on" means `derived.uses_derived_f_scale(cfg.rescale_idx)`. `k_b_cell` is `float(cfg.k_b_cell)`, read only when tier 1 is on, because it raises for a units file with no force unit. `T_range` is the `T` bounds from `cfg.rescale_params[derived.TEMPERATURE_PARAM]` as two floats. `assumed_params` is `list(cfg.assumed_params)`.
- `simulated_f_scale` is the force scale the observation was really simulated at: `float(rescale_gt[0, cfg.sim_rescale_idx["f_scale"]])` after `derived.to_sim_rescale`. It is derived on the tier-1 box and declared otherwise. It is recorded whenever the simulator's index has an `f_scale` column, and absent on a box with none (the spontaneous box). `build_experiment_observation` records nothing new.
- **The identity.** `FORMAT` becomes `"training-rows/3"`. The new value `"units_sha256"` is the sha256 hex of the units file named by `cfg.sources["units"]`, taken over its bytes with `b"\r\n"` replaced by `b"\n"`. The repository runs with `core.autocrlf=true`; `git ls-files --eol` shows `units.txt` checked out LF while `master.txt` is CRLF. Hashing raw bytes would therefore re-key a cache, and break the golden digest, after a re-checkout. Fail open: None when the config has no `sources`, no `"units"` entry, or the file is gone. Read with `getattr(cfg, "sources", None) or {}`, because the window computes identities from stand-ins before Train is pressed. Hash the file, never `cfg.units_dict`, whose order comes from a set and differs between processes.
- The golden digest and the key set in `tests/test_user_sbi.py` are recomputed **in this same commit**. So is `_IDENTITY_KEYS` in `tests/test_artifact_store.py`, which pins the same key set and which the spec did not list. The hand-built `{"format": "training-rows/2", ...}` identity dicts elsewhere in the tests are self-consistent and stay. No cache exists, so none is stranded.
- H4 in everything you write, including the golden test's comment: no process labels. State the reason in words ("the units file's fingerprint joined the identity, because the derived force scale depends on Boltzmann's constant in the cell's units").
- Never edit a source file while a pytest run is in progress. Run only focused tests.

- [ ] **Step 1: Write the failing tests.**

In `tests/test_tier1.py`:
```python
def test_a_tier1_record_carries_the_constraint_and_the_assumed_parameters(store):
    """Every record written from a tier-1 config states the relation that derives the force scale, k_B in
    the cell's units, the temperature range and the assumed parameters; a record without tier 1 has none."""
    from core.artifacts import manifest as mf
    from tests._fixtures import _nad_cfg, _prior_artifact
    cfg = tier1_cfg()
    block = mf.config_from_cfg(cfg)
    assert block["tier1"] == {"relation": "f_scale = N * beta * k_B * T / x_scale",
                              "k_b_cell": cfg.k_b_cell, "T_range": [280.0, 310.0]}
    assert block["assumed_params"] == ["T"]
    m = store.get("prior", _prior_artifact(store, cfg, name="t1").id)
    assert m.config["tier1"] == block["tier1"] and m.config["assumed_params"] == ["T"]
    plain = mf.config_from_cfg(_nad_cfg())
    assert "tier1" not in plain and "assumed_params" not in plain
```
Add `test_a_simulated_tier1_observation_records_the_force_scale_it_ran_at(store, monkeypatch)`. Set it up exactly as this file's simulated-observation test does: `cfg = with_truth(tier1_cfg(chi=False))` (it loads the tier-1 cell and gives its zero drive a real one), `cfg.T_obs = 1000.0`, `monkeypatch.setattr(pipeline, "gen_obs", stand_in_gen_obs)` so nothing real is simulated (every fast test in this file uses the stand-in), and `force_scale_spy(monkeypatch)` installed. Pass `fig_sink=_close`. `with_truth`, `stand_in_gen_obs` and `_close` are module-level helpers of `tests/test_tier1.py`. Call `orchestrator.generate_observations(cfg, fig_sink=<closes the figure>, store=store)`. Compute `expected` as `derived.to_sim_rescale(cfg.params_tensor, <the cell's rescale row>, cfg.rescale_idx, *cfg.tier1_args)[0, cfg.sim_rescale_idx["f_scale"]]`. Assert:
```python
    got = obs.manifest.config["simulated_f_scale"]
    assert got == pytest.approx(float(expected), rel=1e-6) and 46.5 < got < 47.5   # n*beta*k_B*T/x_scale = 47.0 pN
```
In `tests/test_artifact_store.py`, add `"units_sha256"` to `_IDENTITY_KEYS` and add:
```python
def test_the_simulation_identity_fingerprints_the_units_file_it_was_built_with(tmp_path):
    """The derived force scale depends on Boltzmann's constant in the cell's units, so the cache identity
    carries the units file's fingerprint: line endings do not change it, an edit does, and a config with
    no units source fails open to None."""
    import hashlib
    from core.artifacts.identity import FORMAT, SimulationIdentity
    cfg = _nad_cfg()
    lf = Path(cfg.sources["units"]).read_bytes().replace(b"\r\n", b"\n")
    ident = SimulationIdentity.from_cfg(cfg, None, 4, 2).to_dict()
    assert FORMAT == "training-rows/3" and ident["format"] == FORMAT
    assert ident["units_sha256"] == hashlib.sha256(lf).hexdigest()
    crlf, edited = tmp_path / "crlf.txt", tmp_path / "edited.txt"
    crlf.write_bytes(lf.replace(b"\n", b"\r\n"))
    edited.write_bytes(lf + b"# a declared change\n")
    assert SimulationIdentity.from_cfg(_nad_cfg(units_override=str(crlf)), None, 4, 2).to_dict()["units_sha256"] == ident["units_sha256"]
    assert SimulationIdentity.from_cfg(_nad_cfg(units_override=str(edited)), None, 4, 2).to_dict()["units_sha256"] != ident["units_sha256"]
    bare = _nad_cfg()
    bare.sources = {}
    assert SimulationIdentity.from_cfg(bare, None, 4, 2).to_dict()["units_sha256"] is None
```
In `tests/test_user_sbi.py`'s golden test, add `"units_sha256"` to the `set(ia) == {...}` literal. Leave the digest literal as it is for now.

- [ ] **Step 2: Run them and see them fail.**

Run `python -m pytest tests/test_tier1.py::test_a_tier1_record_carries_the_constraint_and_the_assumed_parameters tests/test_tier1.py::test_a_simulated_tier1_observation_records_the_force_scale_it_ran_at tests/test_artifact_store.py::test_the_simulation_identity_fingerprints_the_units_file_it_was_built_with tests/test_artifact_store.py::test_identity_carries_truncation_always_and_feature_set_version_rekeys "tests/test_user_sbi.py::test_a_truncated_round_routes_to_its_own_checkpoint_and_the_amortized_digest_is_untouched" -q`.

Expect `KeyError: 'tier1'`, `KeyError: 'simulated_f_scale'`, the `FORMAT` assertion, and key-set mismatches.

- [ ] **Step 3: `config_from_cfg`.**

For a SimConfig, build the dict as today. When tier 1 is on, add the `"tier1"` and `"assumed_params"` entries with exactly the values in Binding. The FDT branch is untouched.

- [ ] **Step 4: The identity.**

Set `FORMAT = "training-rows/3"`. Add a private `_units_sha256(cfg) -> str | None` in `identity.py` (hashlib, the normalised bytes, fail open) and put `"units_sha256": _units_sha256(cfg)` in `vals`. Update the module docstring to say why the units fingerprint is part of the identity.

- [ ] **Step 5: The simulated observation.**

`_write_observation(..., simulated_f_scale: float | None = None)` is a new keyword-only argument: when it is not None, set `w.config["simulated_f_scale"] = float(simulated_f_scale)` inside the writer block. `generate_observations` computes it from `rescale_gt` after `derived.to_sim_rescale`, when `"f_scale" in cfg.sim_rescale_idx`, and passes it.

- [ ] **Step 6: Recompute the golden digest.**

Run the golden test alone. It fails with `the amortized identity moved to <12 hex>`. Pin that value in place of `"1912d2139359"`, and rewrite the comment's "Belongs to …" sentence to: "Belongs to training-rows/3 (the units file's fingerprint joined the identity, because the derived force scale depends on k_B in the cell's units); the training-rows/2 digest this superseded was "1912d2139359"." Run it again; it passes.

- [ ] **Step 7: Run the Step 2 command again; all pass.**

- [ ] **Step 8: Run the neighbouring tests for every file touched.**

Run `python -m pytest tests/test_artifact_store.py -q -k "identity or config or observation or simulation or checkpoint"`, `python -m pytest tests/test_user_sbi.py -q -k "checkpoint or identity or routes"`, `python -m pytest tests/test_tier1.py -q -m "not slow"` and `python -m pytest tests/test_source_hygiene.py -q`. All pass.

- [ ] **Step 9: Commit.**

`git add core/artifacts/manifest.py core/artifacts/identity.py core/orchestrator.py tests/test_tier1.py tests/test_artifact_store.py tests/test_user_sbi.py`, then `git commit -m "Record the tier-1 constraint; units in the cache identity" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"`.

The controller starts the one-process fast gate (`python -m pytest -m "not slow" -q`, in the background, logged) at the same moment as this task's review. Do not run it yourself.

---

### Task 18: Network flags on smoke; the masked-probe run total; the disk estimate

**Files:**
- Modify: `core/tool/smoke.py`
  - the flags after `p.add_argument("--max-epochs", dest="max_num_epochs", ...` ~:102;
  - the `build_posterior(` call's `**knobs(args, "resume")` ~:212;
  - the module docstring's first "WHAT TO WATCH" bullet.
- Modify: `core/tool/help_defaults.py` (`DEFAULT_CLASS`: two entries).
- Modify: `core/SBI/pipeline.py`
  - beside `_BATCH_TAG = ""` ~:151: new `_ACTIVE_LEDGER`, `class _BatchProbeLedger`, `_chi_counts`, `_cache_size_bytes`;
  - `_chi_rows` ~:954 and its `gen_chi_block(` call ~:984;
  - the `_rows` closure's `return _chi_rows(geom, ...)` ~:1592;
  - the disk preflight `_w = statistics.SUMMARY_WIDTH + 1 + (` / `_need = ...` ~:1437-1439;
  - `global _BATCH_TAG` ~:1448;
  - the batch-retry `for _attempt in range(_attempts + 1):` block ~:1645-1701;
  - `x_buf[_lo:_hi] = _rows_out` ~:1718;
  - the three `_tc.save(` calls ~:1768, ~:1804, ~:1841;
  - the `finally:` ~:1811;
  - after the `[patho] run total` block ~:1817-1822.
- Modify: `core/SBI/chi_probes.py` (`def gen_chi_block(` ~:247-270).
- Modify: `core/SBI/training_checkpoint.py` (`create`'s zeroed state ~:187, `def save(` ~:330/:356, `def mark_complete(` ~:360/:369, and the module docstring's `state.pt` line).
- Test: `tests/test_user_sbi.py` (new tests beside `_gen_td`/`_ck`/`_kill_at`; the log-site table in `test_every_sbi_message_is_a_record_at_its_level_and_nothing_prints` ~:4046-4091).
- Test: `tests/test_tool.py` (`test_every_smoke_flag_reaches_its_stage_as_a_keyword` ~:931).

**Interfaces:**
- Consumes: `core/tool/help_defaults.py` `DEFAULT_CLASS` and `default_for(leaf_path, action)`; `config_args.add_training_flags(p, *, fisher: bool)`, which holds the canonical help text for `--hidden-features` / `--num-transforms`.
- Produces: class `_BatchProbeLedger` in `core/SBI/pipeline.py` with:
  - `add(lo: int, hi: int, masked: int, total: int, record=None) -> None`, `discard() -> None` and `commit(batch_k: int) -> None`;
  - `committed: list[tuple[int, int]]`, the (masked, total) of each committed batch, in batch order;
  - `summary_line(scope: str) -> str`;
  - additively, a constructor `_BatchProbeLedger(committed=(), first_batch: int = 0)` and an attribute `first_batch`.

  Also produced:
  - the module global `_ACTIVE_LEDGER`, set for the life of one `gen_training_data` call, like `_BATCH_TAG`;
  - `chi_probes.gen_chi_block` calls `pipeline._ACTIVE_LEDGER.add(...)` when the ledger is set and the call carries `row_range`;
  - checkpoint `state.pt` gains `"chi_masked": list[list[int]]` (readers `.get("chi_masked", [])`);
  - the run-total record is one info line starting `"[chi] masked probes"`;
  - `smoke` gains `--hidden-features N` and `--num-transforms N`.

**Binding (read before starting):**
- `python` below means `C:\Users\J\anaconda3\envs\biophys-env\python.exe` (in the Bash tool, `/c/Users/J/anaconda3/envs/biophys-env/python.exe`); the `python` on PATH is a different interpreter without torch or pytest. Commands written with backslash paths are PowerShell commands; in Bash use forward slashes.
- **The word "ledger" is a scan hit** (the `doc` family matches it as a word). The class name `_BatchProbeLedger` and the global `_ACTIVE_LEDGER` are identifiers and are never scanned, but no comment, docstring or log message may say "ledger": write "the batch probe tally".
- **Spec §5.8, §5.9 and §5.10**, plus §4.4's paragraph "The observer", because this task builds its commit seam. Read them.
- **Commit and discard.** A batch's counts are committed once, immediately after its rows are stored (right after `x_buf[_lo:_hi] = _rows_out` / `th_buf[_lo:_hi] = _th_out`). An attempt that the batch-level retry loop abandons is discarded before the re-run.
- **Row halving.** `_rows_with_oom_retry` runs one batch as several row ranges under one batch tag, and an out-of-memory error raised after `gen_chi_block` returned leaves a report for the failed full range. So `add` first drops every pending entry whose `[lo, hi)` overlaps the new one. A halved batch is then counted exactly once. A later task hands the buffered `record`s to an observer at `commit`, under the same rule.
- **Resume.** The counts ride in `state.pt` beside `batches_done`. A stored list is trusted only when its length equals `batches_done`. Otherwise (a cache committed before this task, or a mismatch) the ledger starts at `first_batch = batches_done`: the line then says it covers only the batches this process generated, and later saves omit the key, so it never claims batches it did not count.
- **The line.** One info record at the end of generation, including when a resume finds the cache complete and nothing is simulated. It is logged only in chi mode; forced and spontaneous runs log none. Exact format, from `summary_line(scope)`:
  - `f"[chi] masked probes: {M:,} of {N:,} ({100*M/N:.1f}%) over {n} batches, {scope}; per batch {lo:.1f}-{hi:.1f}%, median {med:.1f}%"`. Here the per-batch fractions are over committed batches with total > 0, and `statistics.median` gives the median.
  - With N == 0: `f"[chi] masked probes: none counted, {scope}"`.

  Scope phrases, exact:
  - no checkpoint: `"this process only (no simulation cache)"`;
  - full coverage: `"every committed batch of the simulation cache"`;
  - partial: `f"only the batches this process generated; the cache's first {first_batch} batches were committed without per-batch counts"`.

  The calibration draw also goes through `gen_training_data`, so it logs its own line, with the no-cache phrase.
- **Logging.** Log it as `masked_line = _ledger.summary_line(_scope)` then `log.info(masked_line)`. A variable named `line` would collide with the `"{line}"` row of the pipeline log-site table. No print: smoke shows it through its console handler. Nothing logs between steps 1 and 3 of a checkpoint save: `save` and `mark_complete` only store the list.
- **No production number may change.** The ledger draws no random numbers and touches no tensor. Seeded output must be byte-identical before and after this task (Step 11 checks it outside the suite).
- **Raw-source hazards.** `tests/test_user_sbi.py::test_the_batch_retry_waits_releases_and_restores_the_rng` reads `inspect.getsource(gen_training_data)` and requires `src.index("Waiting") < src.index("_release_device_memory(device)")`. Put neither the word `Waiting` nor the text `_release_device_memory(device)` in any new code or comment above the retry block. The log-site table's count assertion rises by exactly one.
- **The disk line** counts the targets as `nd_dim + len(rescale_idx)`, i.e. `len(param_keys)`: 13 on the tier-1 box. The retrain's line then reads ~10.3 GiB, not ~9.9. The 8-column bound for the forcing block, whose width is unknown before the first batch, stays.
- **Smoke.** The two flags keep the destinations `hidden_features` / `num_transforms`, metavar `N` and parser default None. Forward them with `knobs(args, "resume", "hidden_features", "num_transforms")`, so an absent flag forwards nothing and the stage keeps its own default. Every knob is a keyword argument. Their help text is the one `add_training_flags` gives train's flags. `DEFAULT_CLASS` needs an entry for `("smoke", "--hidden-features")` and `("smoke", "--num-transforms")`, in the same form as train's two entries: the help walker fails, naming the leaf and flag, if either is missing.
- **H4** in all code, comments, docstrings, messages, test names and test docstrings. Never edit a source file while pytest runs. Run only focused tests. No `.py` under `.superpowers/`.

- [ ] **Step 1: Record the seeded digests before any edit.**

Write the scratch script `C:\Users\J\AppData\Local\Temp\prism-piece6\t18\seeded_digest.py` (outside the repository, beside the plan's other scratch files; `%TEMP%` does not expand in the Bash tool). It sets `os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")` and inserts the repo root at `sys.path[0]`. It imports `_gen_td` from `tests.test_user_sbi`. For `mode in ("chi", "forced", "spontaneous")` it prints `mode, hashlib.sha256(x.numpy().tobytes() + th.numpy().tobytes()).hexdigest()[:16]` for `x, th = _gen_td(mode, seed=7)`.

Run it with the biophys-env interpreter and a timeout, and keep the three lines.

- [ ] **Step 2: Smoke flags, test first.**

In `test_every_smoke_flag_reaches_its_stage_as_a_keyword`:
- add `"--hidden-features", "11", "--num-transforms", "3"` to the `main([...])` argv;
- extend the `set(kw2) == {...}` literal with `"hidden_features", "num_transforms"`;
- add `assert kw2["hidden_features"] == 11 and kw2["num_transforms"] == 3`;
- after the existing assertions, run `main(["smoke", *_cfg(bounds), "--cell", cell, "--stages", "prior,posterior", "--store-root", str(tmp_path / "smoke2")]) == 0` and assert `"hidden_features" not in post_rec.calls[-1][1] and "num_transforms" not in post_rec.calls[-1][1]`.

Run `python -m pytest tests/test_tool.py::test_every_smoke_flag_reaches_its_stage_as_a_keyword -q` and see it fail (`unrecognized arguments: --hidden-features`).

- [ ] **Step 3: Implement the two flags and the forwarding.**

Add the two `DEFAULT_CLASS` entries. Update smoke's docstring bullet so it says the masked-probe number to read is the `[chi] masked probes` run total (the per-call warnings remain).

Run the Step 2 test plus `tests/test_tool.py::test_every_value_flag_states_its_default_by_its_class tests/test_tool.py::test_every_value_flag_has_a_house_placeholder tests/test_tool.py::test_every_leaf_parser_has_a_description`. All pass.

Commit: `git add core/tool/smoke.py core/tool/help_defaults.py tests/test_tool.py`, then `git commit -m "smoke takes --hidden-features and --num-transforms" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"`.

- [ ] **Step 4: Disk estimate, test first.**

In `tests/test_user_sbi.py`:
```python
def test_the_cache_size_estimate_counts_every_target_column(monkeypatch, tmp_path):
    """The pre-flight's cache size counts one float32 column per inferred parameter, not a fixed eight:
    10,000 x 2,048 chi rows of 122 conditioning columns and 13 targets are 10.3 GiB."""
    got = pipeline_mod._cache_size_bytes(10_000, 2048, chi_mode=True, chi_k_pad=12, n_targets=13)
    assert got == 10_000 * 2048 * (SUMMARY_WIDTH + 1 + config.CHI_ELEM_W * 12 + 13) * 4
    assert f"{got / 2 ** 30:.1f}" == "10.3"
    assert pipeline_mod._cache_size_bytes(1, 1, chi_mode=False, chi_k_pad=None, n_targets=13) == (SUMMARY_WIDTH + 1 + 8 + 13) * 4
    seen = {}

    class _Stop(Exception):
        pass

    def _spy(n_runs, run_size, **k):
        seen.update(k, n_runs=n_runs, run_size=run_size)
        raise _Stop
    monkeypatch.setattr(pipeline_mod, "_cache_size_bytes", _spy)
    with pytest.raises(_Stop):
        _gen_td("chi", seed=3, n_runs=2, run_size=4, checkpoint=_ck(tmp_path / "d"))
    assert seen == {"n_runs": 2, "run_size": 4, "chi_mode": True, "chi_k_pad": 4, "n_targets": 13}, seen
```
Run it; it fails (`AttributeError: ... '_cache_size_bytes'`).

- [ ] **Step 5: Implement the helper and call it.**

Add `_cache_size_bytes(n_runs: int, run_size: int, *, chi_mode: bool, chi_k_pad: int | None, n_targets: int) -> int`, which returns `n_runs * run_size * (SUMMARY_WIDTH + 1 + (CHI_ELEM_W * chi_k_pad if chi_mode else 8) + n_targets) * 4`. Its docstring says the forcing width is an upper bound because it is not known before the first batch. Replace `_w`/`_need` with `_need = _cache_size_bytes(n_runs, run_size, chi_mode=chi_mode, chi_k_pad=chi_k_pad, n_targets=nd_dim + len(rescale_idx))`. Remove the "+8 covers the latent targets" comment; the log text is unchanged.

Run the test; it passes. Commit: `git add core/SBI/pipeline.py tests/test_user_sbi.py`, then `git commit -m "Disk estimate counts every target column" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"`.

- [ ] **Step 6: The ledger and the run total, tests first** (all in `tests/test_user_sbi.py`).
```python
def test_the_probe_ledger_commits_each_batch_once_and_drops_abandoned_attempts():
    """Counts are committed only after a batch's rows are stored: a range re-run in halves replaces the
    report it overlaps, an attempt the retry loop abandons is discarded, and batches commit in order."""
    L = pipeline_mod._BatchProbeLedger()
    L.add(0, 8, 5, 24)                   # the whole batch, reported before it ran out of memory
    L.add(0, 4, 2, 12)                   # re-run in halves
    L.add(4, 8, 1, 12)
    L.commit(0)
    assert L.committed == [(3, 24)]
    L.add(0, 8, 7, 24)
    L.discard()
    L.add(0, 8, 4, 24)
    L.commit(1)
    L.commit(2)
    assert L.committed == [(3, 24), (4, 24), (0, 0)]
    with pytest.raises(RuntimeError):
        L.commit(7)
    resumed = pipeline_mod._BatchProbeLedger(committed=[[1, 10], [2, 10]])
    resumed.add(0, 4, 3, 10)
    resumed.commit(2)
    assert resumed.committed == [(1, 10), (2, 10), (3, 10)] and resumed.first_batch == 0
    partial = pipeline_mod._BatchProbeLedger(first_batch=5)
    partial.add(0, 4, 1, 8)
    partial.commit(5)
    assert partial.committed == [(1, 8)] and partial.first_batch == 5


def test_the_probe_ledger_summary_names_its_scope_and_spread():
    L = pipeline_mod._BatchProbeLedger(committed=[(1, 10), (3, 10), (2, 20)])
    assert L.summary_line("every committed batch of the simulation cache") == (
        "[chi] masked probes: 6 of 40 (15.0%) over 3 batches, every committed batch of the simulation "
        "cache; per batch 10.0-30.0%, median 10.0%")
    assert pipeline_mod._BatchProbeLedger().summary_line("this process only (no simulation cache)") == (
        "[chi] masked probes: none counted, this process only (no simulation cache)")
```
Also write these tests (helpers: `_gen_td`, `_ck`, `_kill_at`, `_KillRun`). A "line" means the records on `core.SBI.pipeline` whose message starts with `"[chi] masked probes"`.
- `test_the_masked_probe_total_is_logged_once_for_a_run_without_a_cache(caplog, monkeypatch)`:
  - wrap `pipeline_mod.gen_chi_raw` to record `out[0].shape` (B, K);
  - run `_gen_td("chi", seed=7, n_runs=3, run_size=4)` under `warnings.catch_warnings(record=True)` with `simplefilter("always")`;
  - exactly one INFO line; it contains `f"{masked:,} of {total:,}"`, where `masked` is the sum of `chi: (\d+)/\d+ probes masked` over the warnings and `total` = Σ B·K; it contains `"over 3 batches"` and `"this process only (no simulation cache)"`;
  - then `_gen_td("forced", seed=5, n_runs=1, run_size=2)` and `"spontaneous"` log no line.
- `test_the_masked_probe_total_covers_every_committed_batch_across_a_resume(caplog)`, in a `tempfile.mkdtemp()` root:
  - (ref) `_gen_td("chi", seed=11, n_runs=4, run_size=4, checkpoint=_ck(tmp/"ref", every=1))`: one line containing `"every committed batch of the simulation cache"`; `tc.peek(tmp/"ref")["chi_masked"]` has 4 entries, each total > 0;
  - (kill) with `pipeline_mod.gen_stats` swapped for `_kill_at(2)`'s spy, `resume="never"` raises `_KillRun`; `tc.peek(tmp/"a")["chi_masked"] == ref[:2]`;
  - (resume) `resume="require"`: the line equals ref's line; `tc.peek(tmp/"a")["chi_masked"] == ref`;
  - (complete) the same call again, which simulates nothing: the line equals ref's line.
- `test_a_resume_from_a_cache_without_per_batch_counts_says_the_total_covers_only_this_process(caplog, monkeypatch)`:
  - kill at batch 2 as above (`seed=11`, `n_runs=4`, `every=1`);
  - load `state.pt`, `pop("chi_masked")`, save it back;
  - wrap `gen_chi_raw` as above; resume with `resume="require"` under a warnings capture;
  - a `"[checkpoint] resuming at batch 2/4"` record appears; exactly one line containing `"over 2 batches"`, `"only the batches this process generated"`, `"first 2 batches"` and `f"{masked:,} of {total:,}"` (counted during the resumed call only);
  - the returned rows have 16 rows, and `"chi_masked" not in tc.peek(tmp/"old")`.
- `test_a_batch_re_run_in_halves_is_counted_once(monkeypatch)`:
  - monkeypatch `pipeline_mod._BUDGET_CAP_ELEMENTS` and `_budget_clean_runs` to their current values so they are restored, and `_MIN_SIM_CHUNK` to 1;
  - wrap `_BatchProbeLedger.add`/`.commit` to record calls;
  - replace `pipeline_mod._subset_probe_rows` with one that raises `SimulationError(... ) from torch.AcceleratorError("CUDA error: out of memory")` when `block.shape[0] > 2`, and otherwise calls the real one;
  - `_gen_td("chi", seed=5, n_runs=1, run_size=4)`: the add ranges are `[(0, 4), (0, 2), (2, 4)]`, and the one commit equals the two halves' summed (masked, total).
- `test_an_abandoned_batch_attempt_is_discarded_before_the_re_run(monkeypatch)`:
  - same budget restore; `_MIN_SIM_CHUNK` = 64, so no halving; `config.TRAINING_BATCH_RETRY_DELAYS_S` = `(0.0,)`;
  - `_subset_probe_rows` raises the out-of-memory error on its first call only;
  - count `discard` calls; `_gen_td("chi", seed=5, n_runs=1, run_size=4)`: exactly one discard, two adds of `(0, 4)`, and the commit equals the second add's counts.

In `test_every_sbi_message_is_a_record_at_its_level_and_nothing_prints`, add the row `("{masked_line}", "INFO"),` to `pipeline_mod`'s tuple and raise the `sum(...) ==` total by one.

- [ ] **Step 7: Run the new tests and see them fail.**

Run `python -m pytest tests/test_user_sbi.py -q -k "probe_ledger or masked_probe_total or halves_is_counted or abandoned_batch or every_sbi_message"`. Expect `AttributeError: ... '_BatchProbeLedger'` and a table count mismatch.

- [ ] **Step 8: Build the ledger in `core/SBI/pipeline.py`.**
- **Module level, beside `_BATCH_TAG`:**
  - `_ACTIVE_LEDGER = None`, with a comment saying why it is a global, following `_BATCH_TAG`'s;
  - `class _BatchProbeLedger`, with the semantics pinned above: `add` drops overlapping pending entries and then appends `(lo, hi, masked, total, record)`; `commit` raises `RuntimeError` unless `batch_k == first_batch + len(committed)`, appends the summed pending counts (`(0, 0)` when none) and clears pending; `summary_line` uses the exact format; the constructor normalises stored `[m, t]` lists to int tuples;
  - `_chi_counts(ledger, k) -> list[list[int]] | None`, which returns the first `k` committed entries as lists when `ledger is not None and ledger.first_batch == 0`, else None.
- **Threading the row range.** `_chi_rows(..., _patho, row_range=None)` forwards `row_range=row_range` to `gen_chi_block`. `_rows(lo, hi)` passes `row_range=(lo, hi)`.
- **In `gen_training_data`**, after the resume block and only when `chi_mode`: `_stored = _state.get("chi_masked", []) if _ck_resumed is not None else []`. Then `_ledger = _BatchProbeLedger(committed=_stored)` if `len(_stored) == _start_k`, else `_BatchProbeLedger(first_batch=_start_k)`. Otherwise `_ledger = None`.
  - Make the global statement `global _BATCH_TAG, _ACTIVE_LEDGER`. Set `_ACTIVE_LEDGER = _ledger` just before the loop's `try:`; the `finally:` resets it to None beside `_BATCH_TAG = ""`.
  - In the retry loop, on the path where an attempt failed and will be re-run (after the re-raise check, before `_delay = ...`), call `_ledger.discard()` when `_ledger` is set.
  - `_ledger.commit(batch_k)` goes right after the two row-storing lines.
  - The three saves pass `chi_masked=_chi_counts(_ledger, <the k being saved>)`: `batch_k + 1`, `batch_k`, `n_runs`.
  - After the `[patho] run total` block: when `_ledger` is set, choose the scope phrase (`_ck_dir is None` → no-cache; `first_batch == 0` → full; else partial), then `masked_line = _ledger.summary_line(_scope)` and `log.info(masked_line)`.

- [ ] **Step 9: `gen_chi_block` and the checkpoint.**
- `gen_chi_block(*args, k_pad=None, bounds=None, row_range: tuple[int, int] | None = None, **kwargs)` takes `row_range` as an explicit keyword, so it never reaches `gen_chi_raw`. After `dropped` is computed, and before the warning: when `row_range is not None and _pipeline._ACTIVE_LEDGER is not None`, call `_pipeline._ACTIVE_LEDGER.add(int(row_range[0]), int(row_range[1]), dropped, B * K)`. The warning is unchanged.
- In `training_checkpoint`: `create`'s zeroed state gains `"chi_masked": []`. `save(..., chi_masked=None)` is a new keyword-only argument: it writes `"chi_masked": [[int(m), int(t)] for m, t in chi_masked]` into the state dict only when it is not None, inside the existing `cancel_deferred()` block, with no logging. `mark_complete` carries `st["chi_masked"]` forward when present. Update the module docstring's `state.pt` line.

- [ ] **Step 10: Run the Step 7 command again; all pass.**

- [ ] **Step 11: Seeded output is unchanged.**

Re-run `C:\Users\J\AppData\Local\Temp\prism-piece6\t18\seeded_digest.py`. Pass means the three lines are identical to Step 1's; put both outputs in the task report. Delete the scratch directory.

- [ ] **Step 12: Run the neighbouring tests for every file touched.**

Run `python -m pytest tests/test_user_sbi.py -q -k "reproducible_from_a_seed or bit_identical or stratification_schedule or complete_checkpoint or checkpointing_off or recovers_from_an_oom or chi_k_fixed or batch_retry or nothing_prints_or_logs or every_sbi_message or probe_ledger or masked_probe_total or halves_is_counted or abandoned_batch or cache_size_estimate"`. Then run `python -m pytest tests/test_tool.py::test_the_gate_lines_keep_their_text_and_stream tests/test_tool.py::test_every_smoke_flag_reaches_its_stage_as_a_keyword tests/test_source_hygiene.py -q`. All pass.

- [ ] **Step 13: Commit.**

`git add core/SBI/pipeline.py core/SBI/chi_probes.py core/SBI/training_checkpoint.py tests/test_user_sbi.py`, then `git commit -m "One masked-probe run total, kept across resumes" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"`.

The controller starts the one-process fast gate (`python -m pytest -m "not slow" -q`, in the background, logged) at the same moment as this task's review. Do not run it yourself.

---

### Task 19: The acceptances calibrations and narrowing rounds ran under

**Files:**
- Modify: `core/orchestrator.py`
  - `validate_calibration`'s `results = {` dict ~:1882-1898;
  - `build_posterior`'s `"training": {` dict ~:1390-1399;
  - `infer_and_visualize`, from `_want, accepted = post.x_obs_digest, []` ~:1969 through `accepted = accept.used()` ~:1985, and its `:param accept:` docstring.
- Modify: `core/artifacts/store.py` (the `class Accept` docstring ~:96-101).
- Modify: `core/artifacts/report.py` (`def _accepted(m)` ~:177-189).
- Modify: `core/tool/config_args.py` (the `add_accept_flags` docstring ~:195-200, text only).
- Test: `tests/test_artifact_store.py`
  - `test_calibration_writes_results_ranks_figures_and_refuses_a_foreign_prior` ~:1704;
  - `test_inference_refuses_a_foreign_observation_for_a_truncated_posterior_unless_accepted` ~:1744;
  - `test_a_round_records_parent_posterior_and_observation_as_parents` ~:1782;
  - new tests.

**Interfaces:**
- Consumes: `LoadedPosterior.accepted` (set by `store.load_posterior`); `tests/_fixtures.py::stubbed_training(store, monkeypatch, *, rotate=False)`; `validate_calibration(..., seed=None)`, whose `body.results` already carries `"verdict"` and `"seed"`.
- Produces:
  - `body.results["accepted"]` on a calibration;
  - `body.training["accepted"]` on a narrowing child (and `[]` on every trained posterior without a parent);
  - an inference's `accepted` is the union of its posterior's load acceptances and its own `other_observation` use;
  - `core/artifacts/report.py::_accepted` reads `body.training.accepted` for a posterior.

**Binding (read before starting):**
- `python` below means `C:\Users\J\anaconda3\envs\biophys-env\python.exe` (in the Bash tool, `/c/Users/J/anaconda3/envs/biophys-env/python.exe`); the `python` on PATH is a different interpreter without torch or pytest. Commands written with backslash paths are PowerShell commands; in Bash use forward slashes.
- **Spec §5.11 and decision H13** (read both). A loaded posterior carries the acceptances its load used (`LoadedPosterior.accepted`). Records take them like this, following the precedent the diagnostics already use: `list(getattr(posterior, "accepted", None) or [])`.
  - `validate_calibration` writes the posterior's acceptances to `results["accepted"]`. That also lands in `results.json`, because it is written from `results`.
  - `build_posterior` writes `parent_posterior`'s acceptances to `training["accepted"]`, with `[]` when there is no parent. This is how `tsnpe_round`'s child gets its parent's: `tsnpe_round` passes `parent_posterior=posterior` and needs no change.
  - `infer_and_visualize` records `[a for a in ("truncated", "other_observation") if a in load or (a == "other_observation" and used)]`, where `load` is the posterior's list and `used` is True only when the foreign-observation branch was taken and accepted. So an inference on a narrowed posterior's own observation, loaded with `Accept(truncated=True)`, records `["truncated"]` instead of `[]`. The refusal, the warning text and its order are unchanged.
- Each of the three records carries `"truncated"` when its posterior was loaded with `Accept(truncated=True)`, and nothing otherwise.
- **`build_posterior`'s own return keeps `loaded.accepted = []`**: a round just trained is not an accepted load.
- Body keys are a closed set per kind. Everything new goes inside the existing `results` / `training` dicts.
- **Rewrite the docstrings that say calibrations and narrowing rounds record no acceptance**, in plain words: `Accept`, `add_accept_flags`, `_accepted`. Keep "An INFERENCE writes the flags used into its own manifest's results.accepted".
- `docs/STATE.md`'s parked item about this is closed by the controller's STATE update, not in this task.
- H4 in every comment, docstring, test name and test docstring. Never edit a source file while pytest runs. Run only focused tests. The tests that use `tiny_run` build the module's tiny real run once, which takes minutes.

- [ ] **Step 1: Write the failing tests** (all in `tests/test_artifact_store.py`).

In `test_calibration_writes_results_ranks_figures_and_refuses_a_foreign_prior`, add `assert res["accepted"] == []`. In `test_a_round_records_parent_posterior_and_observation_as_parents`, add `assert m.body["training"]["accepted"] == []` (its parent is amortized). In `test_inference_refuses_a_foreign_observation_for_a_truncated_posterior_unless_accepted`, replace the final line with:
```python
    same.accepted = ["truncated"]                 # loaded with Accept(truncated=True), run on its own observation
    assert orchestrator.infer_and_visualize(r.cfg, same, obs, fig_sink=r.sink, n_samples=20).results["accepted"] == ["truncated"]
```
New tests:
```python
def test_a_calibration_records_the_acceptance_its_posterior_was_loaded_under(tiny_run):
    """A calibration of a posterior loaded with Accept(truncated=True) records that acceptance in its
    results, as an inference does."""
    from dataclasses import replace
    from core import orchestrator
    r = tiny_run
    cal = orchestrator.validate_calibration(r.cfg, replace(r.posterior, accepted=["truncated"]), r.prior,
                                            fig_sink=r.sink, n_cal=4, cal_n_scales=1, num_posterior_samples=20)
    assert cal.results["accepted"] == ["truncated"]
    assert json.loads((cal.path / "results.json").read_text(encoding="utf-8"))["accepted"] == ["truncated"]


def test_a_narrowing_round_records_the_acceptance_its_parent_was_loaded_under(store, monkeypatch):
    """A round drawn from a parent loaded with Accept(truncated=True) records that acceptance in its
    training block; a round from a parent loaded without it, and an amortized run, record none."""
    from types import SimpleNamespace
    from core import orchestrator
    from core.SBI import reparam, truncate, training_checkpoint as tc
    from core.SBI.run_guards import _log_params_for
    from tests._fixtures import stubbed_training
    h = stubbed_training(store, monkeypatch)
    T = reparam.build_inferred_bijection(h.cfg, log_params=_log_params_for(h.cfg))
    region = truncate.TruncationRegion([0], [-50.0], [50.0], n_latent=h.P, x_obs_digest="d" * 16,
                                       probe=tc.bijection_probe(T, h.P, device=h.cfg.hw.device))

    def child_of(accepted):
        parent = SimpleNamespace(id="20300101T000000", accepted=accepted,
                                 manifest=SimpleNamespace(body={"transform": {"fisher_eigenvalues": None}}))
        out = orchestrator.build_posterior(h.cfg, h.prior, None, True, num_runs=2, checkpoint_every=0,
                                           truncation=region, parent_posterior=parent, **h.budget)
        return store.get("posterior", out.id).body["training"]["accepted"]

    assert child_of(["truncated"]) == ["truncated"]
    assert child_of([]) == []
    plain = orchestrator.build_posterior(h.cfg, h.prior, None, True, num_runs=2, checkpoint_every=0, **h.budget)
    assert store.get("posterior", plain.id).body["training"]["accepted"] == []


def test_the_lineage_report_prints_the_acceptance_a_calibration_and_a_narrowing_round_ran_under(store):
    """The report reads a calibration's acceptance from its results and a posterior's from its training
    block, and prints each on its own step."""
    from core.artifacts import render_lineage
    post = _make(store, "posterior", name="round",
                 body={"mode": "chi", "conditioning": {"width": 50}, "transform": {}, "amortized": True,
                       "truncation": None, "training": {"accepted": ["truncated"]}})
    cal = _make(store, "calibration", name="cal", body={"results": {"n_cal": 4, "accepted": ["truncated"]}},
                parents={"posterior": post.id})
    text = render_lineage(store, "calibration", cal.id)
    assert text.count("\n  accepted: truncated\n") == 2, text
```

- [ ] **Step 2: Run them and see them fail.**

Run `python -m pytest tests/test_artifact_store.py::test_a_narrowing_round_records_the_acceptance_its_parent_was_loaded_under tests/test_artifact_store.py::test_the_lineage_report_prints_the_acceptance_a_calibration_and_a_narrowing_round_ran_under tests/test_artifact_store.py::test_a_calibration_records_the_acceptance_its_posterior_was_loaded_under tests/test_artifact_store.py::test_calibration_writes_results_ranks_figures_and_refuses_a_foreign_prior tests/test_artifact_store.py::test_inference_refuses_a_foreign_observation_for_a_truncated_posterior_unless_accepted tests/test_artifact_store.py::test_a_round_records_parent_posterior_and_observation_as_parents -q`.

Expect `KeyError: 'accepted'`; the report printing `accepted` once instead of twice; the inference recording `[]`.

- [ ] **Step 3: The three writers in `core/orchestrator.py`.**

Make the three changes described in Binding. In `infer_and_visualize`, replace the single `accepted = accept.used()` assignment with a boolean `used` set in the accepted branch, and compute the union once, before `observation.install(cfg)`. Update the `:param accept:` docstring to say that the record holds the posterior's load acceptances plus this inference's own `other_observation` use.

- [ ] **Step 4: The reader and the docstrings.**

`_accepted(m)` reads `m.body.get("training")` when `m.kind == "posterior"`, and `m.body.get("results")` otherwise. It returns `[str(a) for a in (block.get("accepted") or [])]`, or `[]` when the block is not a dict. That covers older records without the key. Rewrite its docstring: an inference, a calibration and every diagnostic record under `results.accepted`; a trained posterior under `training.accepted`. Rewrite `Accept`'s and `add_accept_flags`' docstrings to match: a calibration, a narrowing round and an inference each record the acceptances they ran under.

- [ ] **Step 5: Run the Step 2 command again; all pass.**

- [ ] **Step 6: Run the neighbouring tests for every file touched.**

Run `python -m pytest tests/test_artifact_store.py -q -k "lineage or render_manifest or accept or calibration or inference or round or fisher_settings"`, then `python -m pytest tests/test_tool.py::test_validate_refuses_a_non_amortized_posterior_without_accept_truncated tests/test_diagnostics.py -q -k "accepted or accept"`, then `python -m pytest tests/test_source_hygiene.py -q`. All pass.

- [ ] **Step 7: Commit.**

`git add core/orchestrator.py core/artifacts/store.py core/artifacts/report.py core/tool/config_args.py tests/test_artifact_store.py`, then `git commit -m "Record the acceptances calibrations and rounds ran under" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"`.

The controller starts the one-process fast gate (`python -m pytest -m "not slow" -q`, in the background, logged) at the same moment as this task's review. Do not run it yourself.

---

### Task 20: The probes' pure helpers

**Files:**
- Create: `core/diagnostics/probe_math.py`
- Test: `tests/test_diagnostics.py` (six new tests appended at the end; `_nad_cfg` is already imported at the top)

**Interfaces:**
- Consumes: nothing from earlier tasks. Reads `SimConfig.dt_exp`, `.dt_nd_min`, `.steady_idx`, `.t`, `.get_unit_conversion_factor("s")`, and `config.N_ND_MAX`, `T_MIN_EXP_S`, `T_MAX_EXP_S`, `CHI_FREQ_BOUNDS`.
- Produces (registry T20): `Geometry` (a `typing.NamedTuple` with fields `n_obs: int`, `n_fine: int`, `subsample: int`; the class syntax is fine, it is the same type as the registry's form); `recording_geometry(cfg, t_obs_s: float, t_scale: float) -> Geometry`; `training_ceiling_s(cfg, t_scale: float) -> float`; `default_lengths(cfg, t_scale: float, n: int = 5) -> list[float]`; `default_multipliers(n: int = 4) -> list[float]`; `own_peak_ratio(forced: Tensor, reference: Tensor, omega0: float, window_frac: float, dt: float) -> float`; `circular_spread(z: Tensor) -> float`; `harmonic_flags(multipliers: Sequence[float], window_frac: float, max_order: int = 5) -> list[bool]`. Also one package-private helper that Tasks 22 and 24 import: `_ensemble_power(x: Tensor, dt: float) -> tuple[Tensor, Tensor]` (the frequency axis and the float64 ensemble-mean power).

**Binding (read before starting):**
- `python` below means `C:\Users\J\anaconda3\envs\biophys-env\python.exe` (in the Bash tool, `/c/Users/J/anaconda3/envs/biophys-env/python.exe`); the `python` on PATH is a different interpreter without torch or pytest. Commands written with backslash paths are PowerShell commands; in Bash use forward slashes.
- Spec §4.6: **no production number may change.** This task creates one module and edits no production file. The helpers restate two production formulas for the probe checks; they do not replace, move or refactor the production copies (`core/orchestrator.py:141-172` and the Sobol pre-filter differ in float32 rounding, so they are not merged).
- `recording_geometry` uses the training batch loop's own arithmetic (`core/SBI/pipeline.py:1478-1484`, anchor `N_points_k = int(T_nd_k / dt_nd_k)`): `T_cell = t_obs_s * cfg.get_unit_conversion_factor("s")`; `T_nd = T_cell / t_scale`; `dt_nd = cfg.dt_exp / t_scale`; `subsample = max(1, round(dt_nd / cfg.dt_nd_min))`; `n_obs = int(T_nd / dt_nd)`; `n_fine = cfg.steady_idx + n_obs * subsample`. No clipping, and no "fix" of the float truncation: at t_scale 3.73 one second gives 999 samples in training too.
- `training_ceiling_s` is the Sobol pre-filter's bound (`pipeline.py:1159`, anchor `n_fine_max = min(N_ND_MAX, t.shape[0])`): `n_fine_max = min(config.N_ND_MAX, cfg.t.shape[0])`; `n_obs_cap = max(1, (n_fine_max - cfg.steady_idx) // subsample)` with the same `subsample` rule; return `n_obs_cap * cfg.dt_exp / hz`. The master cell at t_scale 3.73 gives **26.909 s**.
- `default_lengths` (spec §4.3): `max(2, n)` points log-spaced from `config.T_MIN_EXP_S` to `min(config.T_MAX_EXP_S, training_ceiling_s(cfg, t_scale))`, each rounded DOWN to 0.01 s (`math.floor(v * 100) / 100`; rounding up would put the top length a few fine steps past the ceiling); `[T_MIN_EXP_S]` alone when the ceiling is not above it. Master at 3.73 gives `[1.0, 2.27, 5.18, 11.81, 26.9]`.
- `default_multipliers` (spec §4.3): `max(2, n)` points log-spaced across `config.CHI_FREQ_BOUNDS`, each `round(v, 4)`, plus one control `round(2 * hi, 4)`. Today that is `[0.03, 0.0646, 0.1392, 0.3, 0.6]`.
- `own_peak_ratio` (spec §4.5, one function used by band and drive with their own thresholds). `omega0` is the undriven peak FREQUENCY in cell frequency units (cycles per cell time unit, what `chi.peak_freq` returns), not an angular frequency; say so in the docstring. Spectra are float64. `_ensemble_power(x, dt)`: demean each row, `torch.fft.rfft` in float64, take `|.|**2`, mean over rows; `freqs = torch.fft.rfftfreq(n, d=dt)`. The window sums power over bins `[max(1, i - hw), min(len, i + hw + 1))`, where `i` is the bin nearest `omega0` and `hw = max(2, round(window_frac * omega0 / df))`. That is at least two bins either side, as the earlier band measurement used. At 5 s on the master cell it makes 0.018 and 0.02 the same ±2-bin window, which Task 24's parity run relies on. Ratio = forced window power ÷ max(reference window power, 1e-30). Return NaN when `omega0` is not finite and positive. Raise `ValueError` (a programming error, never a Refusal) when the two ensembles differ in length.
- `circular_spread`: `sqrt(max(0, -2 ln max(R, 1e-30)))` with `R = |mean(z / max(|z|, 1e-30))|`. It is circular, so two phases straddling ±π read as small.
- `harmonic_flags` (spec §4.3, second caveat): True when `k * m` lies in `[1 - window_frac, 1 + window_frac]` for some `k` in `1..max_order` (`k = 1` is the probe itself). Compare with a 1e-9 tolerance so that `3 * 0.3 = 0.8999999999999999` counts as on the edge.
- Importing torch and `core.config` here is fine: only the stage module imports this one, and `--help` never does.
- H4 (spec §2): nothing in this module or its tests cites the spec, a plan, STATE, the handoff or any process label (piece numbers, §N, decision or ruling ids, "Task N"). Docstrings give the reasons in words, e.g. "the ceiling the training pre-filter enforces".

- [ ] **Step 1: Write the failing tests** (append to `tests/test_diagnostics.py`)

```python
def test_the_recording_geometry_agrees_with_the_training_batch_formula():
    """A probe length is sized with the training loop's own arithmetic, float truncation included, so
    it describes the trace a training batch of that length holds."""
    from core.diagnostics import probe_math
    cfg = _nad_cfg()
    hz = cfg.get_unit_conversion_factor("s")
    for t_obs_s, t_scale in ((1.0, 3.73), (1.0, 3.7300000190734863), (5.0, 3.73),
                             (26.909, 3.7300000190734863), (2.5, 1.0), (7.3, 40.0)):
        dt_nd_k = cfg.dt_exp / t_scale
        subsample = max(1, round(dt_nd_k / cfg.dt_nd_min))
        n_points = int((t_obs_s * hz / t_scale) / dt_nd_k)
        want = (n_points, cfg.steady_idx + n_points * subsample, subsample)
        assert tuple(probe_math.recording_geometry(cfg, t_obs_s, t_scale)) == want, (t_obs_s, t_scale)
    assert probe_math.recording_geometry(cfg, 5.0, 3.73) == probe_math.Geometry(
        n_obs=5000, n_fine=59000, subsample=11)


def test_the_training_ceiling_is_the_bound_the_sobol_prefilter_enforces():
    """The longest recording training can draw at a t_scale: the pre-filter admits it and refuses two
    samples more."""
    from core.diagnostics import probe_math
    from core.SBI import pipeline
    cfg = _nad_cfg()
    assert probe_math.training_ceiling_s(cfg, 3.73) == pytest.approx(26.909, abs=1e-9)
    top = probe_math.training_ceiling_s(cfg, 3.73) * cfg.get_unit_conversion_factor("s")
    args = (cfg.dt_exp, cfg.dt_nd_min, cfg.steady_idx)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        pipeline._batch_schedule(4, cfg.t, (3.73, 3.73), top, top, *args)
        beyond = top + 2 * cfg.dt_exp
        with pytest.raises(ValueError, match="fine-grid ceiling"):
            pipeline._batch_schedule(4, cfg.t, (3.73, 3.73), beyond, beyond, *args)


def test_the_default_probe_grids_follow_the_configured_band_and_the_cells_ceiling():
    from core import config
    from core.diagnostics import probe_math
    cfg = _nad_cfg()
    assert config.CHI_FREQ_BOUNDS == (0.03, 0.3)
    assert probe_math.default_lengths(cfg, 3.73) == [1.0, 2.27, 5.18, 11.81, 26.9]
    assert probe_math.default_lengths(cfg, 3.73, n=2) == [1.0, 26.9]
    assert probe_math.default_multipliers() == [0.03, 0.0646, 0.1392, 0.3, 0.6]


def test_the_own_peak_ratio_counts_only_the_power_inside_the_window():
    import math
    from core.diagnostics import probe_math
    t = torch.arange(5000, dtype=torch.float64)                 # dt 1, bins of 1/5000
    rows = torch.linspace(0.0, 2.0, 4, dtype=torch.float64).unsqueeze(1)
    ref = torch.sin(2 * math.pi * 0.02 * t + rows)               # bin 100
    assert probe_math.own_peak_ratio(0.5 * ref, ref, 0.02, 0.10, 1.0) == pytest.approx(0.25, rel=1e-9)
    near = torch.sin(2 * math.pi * 0.0216 * t + rows)            # bin 108
    assert probe_math.own_peak_ratio(near, ref, 0.02, 0.10, 1.0) == pytest.approx(1.0, rel=1e-6)
    assert probe_math.own_peak_ratio(near, ref, 0.02, 0.05, 1.0) < 1e-12
    two_off = torch.sin(2 * math.pi * 0.0204 * t + rows)         # bin 102: inside the two-bin floor
    assert probe_math.own_peak_ratio(two_off, ref, 0.02, 1e-6, 1.0) == pytest.approx(1.0, rel=1e-6)
    assert math.isnan(probe_math.own_peak_ratio(ref, ref, float("nan"), 0.10, 1.0))
    with pytest.raises(ValueError):
        probe_math.own_peak_ratio(ref[:, :4000], ref, 0.02, 0.10, 1.0)


def test_the_circular_spread_is_zero_for_one_phase_and_ignores_the_wrap():
    import math
    from core.diagnostics import probe_math
    one = torch.polar(torch.ones(5, dtype=torch.float64), torch.full((5,), 1.3, dtype=torch.float64))
    assert probe_math.circular_spread(one) == pytest.approx(0.0, abs=1e-7)
    across = torch.polar(torch.ones(2, dtype=torch.float64),
                         torch.tensor([math.pi - 0.05, -math.pi + 0.05], dtype=torch.float64))
    assert probe_math.circular_spread(across) == pytest.approx(math.sqrt(-2 * math.log(math.cos(0.05))), rel=1e-9)
    spread = torch.polar(torch.ones(4, dtype=torch.float64),
                         torch.tensor([0.0, math.pi / 2, math.pi, 3 * math.pi / 2], dtype=torch.float64))
    assert probe_math.circular_spread(spread) > 5.0


def test_the_harmonic_flag_marks_a_probe_whose_low_harmonic_lands_in_the_own_peak_window():
    from core.diagnostics import probe_math
    assert probe_math.harmonic_flags([0.03, 0.0646, 0.1392, 0.3, 0.6], 0.10) == [False, False, False, True, False]
    assert probe_math.harmonic_flags([0.189, 0.25, 0.5, 1.0, 1.4], 0.10) == [True, True, True, True, False]
    assert probe_math.harmonic_flags([0.189], 0.10, max_order=4) == [False]
    assert probe_math.harmonic_flags([1.4], 0.02) == [False]
```

- [ ] **Step 2: Run them and see them fail.** `python -m pytest tests/test_diagnostics.py -q -k "recording_geometry or training_ceiling or default_probe_grids or own_peak_ratio or circular_spread or harmonic_flag"`. Expected: 6 failures with `ModuleNotFoundError: No module named 'core.diagnostics.probe_math'`.
- [ ] **Step 3: Create `core/diagnostics/probe_math.py`** with the module docstring (what the helpers are for, and that they restate training's geometry without replacing it) and the signatures listed under Interfaces, following the Binding rules above. No logging and no `print`.
- [ ] **Step 4: Run the six tests and see them pass** (same command as Step 2; expected `6 passed`).
- [ ] **Step 5: Commit.** `git add core/diagnostics/probe_math.py tests/test_diagnostics.py`, then `git commit -m "probes: shared geometry and spectrum helpers" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"`. At the same moment the controller starts this task's review and the one-process fast gate (`python -m pytest -m "not slow" -q`, in the background, logged). The implementer does not run the gate.

---

### Task 21: The training generator's probe observer

**Files:**
- Modify: `core/SBI/chi_probes.py`: add `ProbeRecord` after the imports (about line 19). The hook goes at the ledger `add(...)` call that Task 18 placed in `gen_chi_block` after `block, mask = chi.pack_probe_block(...)` (anchor, about line 261).
- Modify: `core/SBI/pipeline.py`: the `gen_training_data` signature (anchor `def gen_training_data(`, line 1195; its last parameter is `device: torch.device = torch.device('cpu')) -> tuple:`) and its docstring; Task 18's `_BatchProbeLedger`; the place `gen_training_data` constructs the ledger.
- Test: `tests/test_user_sbi.py` (next to `_gen_td`, about line 1794)

**Interfaces:**
- Consumes (T18): `pipeline._BatchProbeLedger` with `add(lo: int, hi: int, masked: int, total: int, record=None) -> None`, `discard() -> None`, `commit(batch_k: int) -> None`, `committed: list[tuple[int, int]]`, `summary_line(scope: str) -> str`; the module global `pipeline._ACTIVE_LEDGER` (set for the life of one `gen_training_data` call); `gen_chi_block`'s call to `pipeline._ACTIVE_LEDGER.add(...)` and the keyword Task 18 uses to bring the row range `(lo, hi)` into `gen_chi_block`, which it pops before `**kwargs` reach `gen_chi_raw`.
- Produces (registry T21): `@dataclass(frozen=True) class ProbeRecord` in `core/SBI/chi_probes.py` with fields in this order: `batch_tag: str; lo: int; hi: int; f_peak: Tensor; duration_frac: Tensor | None; k: int; u: Tensor; logcyc: Tensor; valid: Tensor; packed_mask: Tensor; dt_exp: float; n_points: int`. The last field is an addition to the registry, and `duration_frac` is typed as a tensor rather than a float: it holds the per-probe (K,) fractions `gen_chi_raw` receives, and `n_points` carries the full recording length the cycle-floor split needs. Also `pipeline.gen_training_data(..., *, probe_observer: Callable[[ProbeRecord], None] | None = None)`, and `_BatchProbeLedger(observer=None)` with an `observer` attribute.

**Binding (read before starting):**
- `python` below means `C:\Users\J\anaconda3\envs\biophys-env\python.exe` (in the Bash tool, `/c/Users/J/anaconda3/envs/biophys-env/python.exe`); the `python` on PATH is a different interpreter without torch or pytest. Commands written with backslash paths are PowerShell commands; in Bash use forward slashes.
- **The word "ledger" is a scan hit** (the `doc` family matches it as a word). The class name `_BatchProbeLedger` and the global `_ACTIVE_LEDGER` are identifiers and are never scanned, but no comment, docstring or log message may say "ledger": write "the batch probe tally".
- Spec §4.4, "The observer", as it applies here:
  - The record is built at the hook inside `gen_chi_block`, after `pack_probe_block`. Nothing new may reach `gen_chi_raw`'s keywords: the row-range keyword is popped first.
  - `f_peak` is recomputed with `chi.peak_freq(x_spont_dim, dt_exp)`, which is deterministic and draws no random numbers.
  - `x_spont_dim`, `dt_exp`, `N_points` and `duration_frac` are read from the arguments `gen_chi_raw` received, via `inspect.signature(gen_chi_raw).bind(*args, bounds=bounds, **kwargs)`, because `_chi_rows` passes them positionally (`pipeline.py:976-985`).
  - Every tensor in the record is detached and moved to the CPU. `packed_mask` is the mask before the per-row subsetting.
- **The record is built only when the active ledger has an observer.** With the default `None`, nothing is built and nothing changes. A test pins that seeded `gen_training_data` output is byte-identical with no observer, with a no-op observer and with a collecting one, in chi, forced and spontaneous modes. Forced and spontaneous rows never reach the observer.
- Records reach the observer **only at `commit`**, which is Task 18's seam after the batch's rows are stored (`x_buf[_lo:_hi] = _rows_out`, `pipeline.py:1718`). They are delivered in ascending `lo` order, after the counts are committed. `discard` (a batch the whole-batch retry abandons, `pipeline.py:1645-1701`) drops them. The counts and the records share one rule: **a new `add` replaces every buffered range it overlaps**. That is the row-halving case, where `_rows_with_oom_retry` (`pipeline.py:806-813`) re-runs `[0, n)` as `[0, n/2)` and `[n/2, n)` under one batch tag after the full range may already have added. If Task 18's ledger uses a different rule, change it here and say so in the report.
- An exception raised by the observer propagates. A broken audit must fail loudly, never hide.
- This hook is why no probe check may monkeypatch `gen_chi_raw` or `gen_chi_block` at run time (spec §1.2).
- No production number changes; the golden digests in `tests/test_user_sbi.py` are unaffected. H4 applies to the dataclass docstring, the comments and the test names.

- [ ] **Step 1: Read Task 18's committed code:** `_BatchProbeLedger`, where `gen_training_data` constructs it and sets `_ACTIVE_LEDGER`, and the `add(...)` call in `gen_chi_block` with the popped row-range keyword. Note that keyword's name for Step 5.
- [ ] **Step 2: Write the failing tests** (in `tests/test_user_sbi.py`, after `_gen_td`)

```python
@pytest.mark.parametrize("mode", ["chi", "forced", "spontaneous"])
def test_a_probe_observer_leaves_seeded_training_rows_byte_identical(mode, monkeypatch):
    """Watching the probes changes nothing about the rows produced, and with no observer no record
    is even built. Forced and spontaneous rows never reach the observer."""
    from core.SBI import chi_probes

    def _never(*a, **k):
        raise AssertionError("a probe record was built with no observer attached")

    with monkeypatch.context() as m:
        m.setattr(chi_probes, "ProbeRecord", _never)
        x0, th0 = _gen_td(mode, n_runs=2, run_size=2)
    seen = []
    x1, th1 = _gen_td(mode, n_runs=2, run_size=2, probe_observer=lambda rec: None)
    x2, th2 = _gen_td(mode, n_runs=2, run_size=2, probe_observer=seen.append)
    assert torch.equal(x0, x1) and torch.equal(th0, th1)
    assert torch.equal(x0, x2) and torch.equal(th0, th2)
    assert bool(seen) == (mode == "chi"), len(seen)


def test_the_probe_observer_sees_each_committed_row_range_once_under_a_row_halving(monkeypatch):
    from core.SBI.chi_probes import ProbeRecord
    from core.Simulator.simulator import SimulationError
    run_size, real_subset = 4, pipeline_mod._subset_probe_rows

    def _flaky(block, mask, k_pad, generator):          # fails AFTER the probes were packed and added
        if block.shape[0] > run_size // 2:
            try:
                raise torch.AcceleratorError("CUDA error: out of memory")
            except RuntimeError as e:
                raise SimulationError("subset: AcceleratorError: CUDA error: out of memory") from e
        return real_subset(block, mask, k_pad, generator)

    monkeypatch.setattr(pipeline_mod, "_subset_probe_rows", _flaky)
    monkeypatch.setattr(pipeline_mod, "_MIN_SIM_CHUNK", 1)
    seen = []
    x, _ = _gen_td("chi", n_runs=2, run_size=run_size, probe_observer=seen.append)
    assert x.shape[0] == 2 * run_size
    ranges = {}
    for rec in seen:
        assert isinstance(rec, ProbeRecord)
        rows = rec.hi - rec.lo
        assert rec.f_peak.shape == (rows,) and rec.valid.shape == rec.u.shape == rec.logcyc.shape == (rows, rec.k)
        assert rec.packed_mask.shape[0] == rows and rec.duration_frac.shape == (rec.k,)
        assert rec.dt_exp == pytest.approx(_td_cfg().dt_exp) and rec.n_points > 0
        ranges.setdefault(rec.batch_tag, []).append((rec.lo, rec.hi))
    assert len(ranges) == 2 and all(r == [(0, 2), (2, 4)] for r in ranges.values()), ranges


def test_the_ledger_hands_records_to_the_observer_only_at_commit():
    seen = []
    ledger = pipeline_mod._BatchProbeLedger(observer=seen.append)
    ledger.add(0, 4, 1, 8, record="abandoned attempt")
    ledger.discard()
    ledger.add(0, 4, 2, 8, record="superseded range")
    ledger.add(0, 2, 1, 4, record="first half")
    ledger.add(2, 4, 0, 4, record="second half")
    assert seen == []
    ledger.commit(0)
    assert seen == ["first half", "second half"]
    ledger.add(0, 4, 0, 8, record=None)
    ledger.commit(1)
    assert seen == ["first half", "second half"]
    assert ledger.committed == [(1, 8), (0, 8)]
```

- [ ] **Step 3: Run them and see them fail.** `python -m pytest tests/test_user_sbi.py -q -k "probe_observer or ledger_hands_records"`. Expected: the three parametrized cases fail at `m.setattr(chi_probes, "ProbeRecord", _never)` with `AttributeError: ... has no attribute 'ProbeRecord'`, the ledger test with `TypeError ... unexpected keyword argument 'observer'`, and the row-halving test with an `ImportError` for `ProbeRecord`.
- [ ] **Step 4: Add `ProbeRecord`** to `core/SBI/chi_probes.py`. It is a frozen dataclass with the fields and order under Interfaces. Its docstring says what each field is: the row range of one `gen_chi_block` call; each row's own peak frequency; the per-probe duration fractions `gen_chi_raw` got (None when it got none); the probe count; `gen_chi_raw`'s returned `u`, `logcyc` and `valid`; the packed mask before per-row subsetting; the sampling interval; and the full recording length in samples. It also says that the audit recomputes the non-finite and Nyquist predicates from `f_peak * exp(u)` and `dt_exp`, because `gen_chi_raw` returns only `valid`.
- [ ] **Step 5: Wire the observer.**
  - `_BatchProbeLedger.__init__(self, committed=(), first_batch: int = 0, *, observer=None)` keeps Task 18's two arguments and their behaviour and stores `self.observer`; `gen_training_data` passes `observer=probe_observer` in both of its constructions (the resumed `committed=` one and the `first_batch=` one).
  - `add` keeps `record` with the range's counts and applies the overlap rule.
  - `commit` calls `self.observer(record)` for each non-None record of the committed ranges, in ascending `lo` order, after committing the counts.
  - `discard` drops them.
  - `gen_training_data` gains `*, probe_observer: "Callable[[ProbeRecord], None] | None" = None` after `device`, a `:param probe_observer:` entry ("receives one ProbeRecord per committed chi row range, in batch order; None changes nothing"), and passes it to the ledger's constructor.
  - In `gen_chi_block`, at Task 18's `add(...)` call: when `_pipeline._ACTIVE_LEDGER.observer is not None`, build the record from the bound `gen_chi_raw` arguments, `_pipeline._batch_tag()`, the popped `(lo, hi)`, `K = chi_stack.shape[1]` and the returned tensors, and pass it as `record=`. Otherwise pass `record=None` and build nothing.
- [ ] **Step 6: Run the new tests and see them pass** (Step 3's command; expected `5 passed`).
- [ ] **Step 7: Run the focused tests for the touched files.** `python -m pytest tests/test_user_sbi.py -q -m "not slow" -k "observer or ledger or recovers_from_an_oom or chi_k_fixed or kept_fraction or masked"`. Expected: all pass, including Task 18's ledger tests.
- [ ] **Step 8: Commit.** `git add core/SBI/chi_probes.py core/SBI/pipeline.py tests/test_user_sbi.py`, then `git commit -m "training: optional probe observer on committed batches" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"`. The controller starts the review and the one-process fast gate (`python -m pytest -m "not slow" -q`, background, logged) at the same moment. The implementer does not run the gate.

---

### Task 22: `probes band`, and the family

**Files:**
- Create: `core/diagnostics/probes.py`. It holds the module docstring, the simulation seam `_simulate`, `probe_band` (`@public_entry`), and the private helpers `probe_band` needs.
- Create: `core/tool/probes.py`. It holds `PROBES_INTERRUPT_NOTE`, `register(sub)` and `_band(args, store)`.
- Modify: `core/diagnostics/__init__.py`. Add `from .probes import probe_band  # noqa: F401` after the anchor `from .ablation import channel_ablation` (line 19), and make the docstring's list of functions name the probe checks.
- Modify: `core/tool/__init__.py`. Add `probes` to the anchor `from . import browse, config_args, diagnostics, fdt, smoke, stages` (line 20) and to `for module in (stages, diagnostics, smoke, fdt, browse):` (line 46).
- Modify: `core/tool/config_args.py`. Change `def add_config_flags(p) -> None:` (line 28) to `add_config_flags(p, *, chi: bool = True) -> None`; `chi=False` leaves out `--chi` and `--chi-k` and nothing else.
- Modify: `core/rng.py`. Change `def require_seed(seed) -> int:` (line 37) to `require_seed(seed, *, key: str = "seed") -> int`, with the same rule and sentences, keyed on `key`.
- Modify: `core/refusals.py`. Add 11 keys after `Field("chi_k_fixed", "the fixed chi probe count", None),` (line 183).
- Modify: `core/tool/fields.py` (after `"chi_k_fixed": "--chi-k-fixed",`, line 124) and `core/gui/fields.py` (the tool-only None block, anchor `"rel": None, "min_valid": None, "rows": None, "n_sweep": None, "chi_k_fixed": None,`, line 180).
- Modify: `core/tool/help_defaults.py` (`LEAVES` and `DEFAULT_CLASS`).
- Modify: `core/artifacts/manifest.py:36` (the comment `# A measurement ABOUT other artifacts (...)`) and the `load_diagnostic` docstring at `core/artifacts/store.py:1498` ("A diagnostic is a MEASUREMENT about other artifacts"). Both are widened: a diagnostic measures other artifacts, or a cell (`identifiability jacobian`, `probes band` and `drive`), or a prior (`probes mask`).
- Test: `tests/test_diagnostics.py`, `tests/test_tool.py`, `tests/test_refusals.py` (`TOOL_ONLY_KEYS` at line 56, the count at line 107, `owned_by_a_signature` at line 499, the import `from core.diagnostics import ablation, identifiability, sbc` in that test), `tests/test_nav_and_gating.py:1607-1612`, `tests/test_artifact_store.py` (the `want` set at lines 3431-3446, `_leg_probe_band` beside `_leg_channel_ablation` at line 3235, `_UNTOUCHED_LEGS` at line 3361, `_REFUSAL_FIELDS` at line 3388).

**Interfaces:**
- Consumes: T13 `derived.for_simulation(cfg, params_nd, rescale) -> (Tensor, dict)`. T20 `Geometry`, `recording_geometry`, `training_ceiling_s`, `default_lengths`, `default_multipliers`, `own_peak_ratio`, `circular_spread`, `harmonic_flags`, `_ensemble_power`. T11 `help_defaults.LEAVES`, `DEFAULT_CLASS`, `default_for` and `core.refusals.default_text`. T10's help conventions (house metavars, `description=` on every leaf, the canonical `--cell` and `--seed` wordings, `add_name_flags` placement).
- Produces (registry T22): `core.diagnostics.probe_band(cfg, *, lengths=None, multipliers=None, drives=None, repeats=24, cycle_caps=None, cv_max=0.20, phase_max=None, snr_min=3.0, sup_min=0.50, peak_window=0.10, seed=0, name="", note="", fig_sink=None, store=None) -> LoadedDiagnostic`. The body is `{"diagnostic": "probes", "variant": "band", "settings", "results"}`. `core/tool/probes.py: register(sub) -> dict[str, argparse.ArgumentParser]` returns `{"probes": parent}`, with modes under `dest="variant"`. Field keys: `probe_seed`, `probe_lengths`, `probe_multipliers`, `probe_drives`, `band_repeats`, `probe_cycle_caps`, `cv_max`, `phase_max`, `snr_min`, `sup_min`, `band_peak_window`. Also produced for Tasks 23 and 24: `PROBES_INTERRUPT_NOTE`; the seam `probes._simulate(cfg, geom, nd, res_sim, sim_idx, inits, *, amp_dim=None, freq=None) -> Tensor`; `config_args.add_config_flags(p, *, chi=True)`; `rng.require_seed(seed, *, key="seed")`.

**Binding (read before starting):** spec §4.1, §4.1a (the band rows and constraints 1–3), §4.2, §4.3, §4.7, §4.8 and Review Focus 1 are binding. Restated:
- `python` below means `C:\Users\J\anaconda3\envs\biophys-env\python.exe` (in the Bash tool, `/c/Users/J/anaconda3/envs/biophys-env/python.exe`); the `python` on PATH is a different interpreter without torch or pytest. Commands written with backslash paths are PowerShell commands; in Bash use forward slashes.
- **Stage order** (identifiability's pattern, `core/diagnostics/identifiability.py:375-420`):
  1. `store = resolve_store(store)`, then `store.assert_name_free("diagnostic", name)`.
  2. The guards. A non-chi config is refused with `Refusal(..., field=None)` (no `chi_mode` key exists or is added). `_ = cfg.ground_truth` refuses a config with no cell.
  3. Every rule.
  4. Resolving the defaults.
  5. Only then `with store.create("diagnostic", cfg, name=name, note=note) as w:` and, inside it, `with seeded(seed, cfg.hw.device):`. A refusal after `create` would leave a directory behind.
- **Rules, all before any simulation.** Each is a Refusal with a registered key; core messages name no box, tab, flag or button.
  - `seed = require_seed(seed, key="probe_seed")`
  - `repeats = require_at_least("band_repeats", repeats, 2)`
  - `lengths`, `multipliers`, `drives`, `cycle_caps`, when given: non-empty lists, each value through `require_positive(<key>, v)`. An empty list goes through `refuse(<key>, ...)`.
  - Each length must also give at least one sample (`recording_geometry(...).n_obs >= 1`, else `refuse("probe_lengths", ...)`). A length beyond the training ceiling is flagged and warned, never refused.
  - `require_finite("cv_max", ...)`, `require_finite("snr_min", ...)`, `require_between("sup_min", sup_min, 0, 1, open_lo=True)`.
  - `phase_max` is None or `require_finite("phase_max", ...)`.
  - `require_between("band_peak_window", peak_window, 0, 1, open_lo=True, open_hi=True)`.
- **It measures only** (spec §4.2, H8; standing refusals D11 and D12 are unchanged):
  - The probe grids reach the simulator only as the drive builder's frequency and amplitude inside `core/diagnostics/probes.py`. They never pass through a `SimConfig` field (`cli.make_sim_config`, `dataclasses.replace`), never through `gen_training_data`'s `chi_f0` or `chi_freq_bounds` keywords, and never through an assignment to an upper-case config name or to a `chi_*` attribute.
  - No `--chi`, `--no-chi`, `--chi-k`, `--chi-f0`, `--chi-band` or `--f0` flag on the family. No `chi_f0`, `chi_freq_bounds` or `f0` key is reused (those keep their FLAG/CONTROL entries).
  - The tool builds the config with chi mode ON through `set_defaults(chi_mode=True, chi_n_freqs=None)` (`make_cfg` reads `args.chi_mode` and `args.chi_n_freqs` directly, `core/tool/config_args.py:138-139`).
  - Every record's `settings["configured"]` states `{"chi_f0", "chi_freq_bounds": [lo, hi], "chi_k_pad", "chi_min_cycles", "chi_max_cycles", "probe_count"}`, read from the config (`probe_count` is `cfg.chi_n_freqs`, `chi_min_cycles` is `config.CHI_MIN_CYCLES`). The manifest's `config` nulls the chi constants outside chi mode and never records K.
  - Verdict lines read "the configured … holds / does not hold for this cell". A failing one is followed by one info line naming `core/config.py` as the place a deliberate change is made.
- **Faithful to training today** (spec §4.2):
  - `chi.peak_freq` gives each row's own peak frequency.
  - A probe at or above 0.9 × Nyquist (`0.5 / cfg.dt_exp`), non-finite or non-positive is **masked, never clamped** (`chi_probes.py:169`).
  - The per-row lock-in ceiling rounds DOWN: `N_row = clamp(floor(cap / freq / dt), 1, n_obs)` in float64 (`chi_probes.py:186-192`), with `chi.lock_in_batched(..., n_samples=N_row)`.
  - Spectra are float64. `x_offset` is applied when the box has it.
  - The force uses a literal index `{"amp": 0, "freq": 1, "phase": 2, "offset": 3}`, never `cfg.forcing_idx["amp"]`. The drive phase is π/2, recorded as `settings["drive_phase"]`.
  - The same drive amplitude divides the floor lock-in, so the signal-over-floor measure is a pure ratio of |χ|.
- **Tier-1 aware:** `res_sim, sim_idx = derived.for_simulation(cfg, cfg.params_tensor, res)`. The amplitude is `F0 * f_scale_eff`, where `f_scale_eff` is the simulation vector's `f_scale` column, or `x_scale / t_scale` when the index has no `f_scale` (exactly `gen_chi_raw`'s rule, `chi_probes.py:110-114`).
- **Record** (spec §4.1):
  - Parents `{}`; the `.npz` payload `probe_band.npz`; figures through `w.fig_sink(fig_sink)`.
  - The report goes out as `log.info` records from `logging.getLogger(__name__)`, never `print`. Things to act on are warnings, and a multi-line warning is one record.
  - Every float that reaches `results` goes through `orch._num` (non-finite becomes null). Per-item records are lists of dicts, never dicts keyed by name.
  - The stage returns `store.load_diagnostic(w.id)`.
  - No code builds a literal `Resources/` or `Artifacts/` path.
- **Keys:**
  - Each key goes into `FIELDS` with a lower-case noun-phrase description and no control words (`tab|box|flag|button|click|tick|dialog`), into `FLAG`, into `CONTROL` as None, and into the no-control set at `tests/test_nav_and_gating.py:1607-1612`.
  - Defaults: `probe_seed` "0", `band_repeats` "24", `cv_max` "0.2", `snr_min` "3.0", `sup_min` "0.5", `band_peak_window` "0.1". The four grid keys and `phase_max` default to None, because their defaults are behaviours, not values.
  - Flags: `probe_seed` → `--seed`, `probe_lengths` → `--lengths`, `probe_multipliers` → `--multipliers`, `probe_drives` → `--drives`, `band_repeats` → `--repeats`, `probe_cycle_caps` → `--cycle-caps`, `cv_max` → `--cv-max`, `phase_max` → `--phase-max`, `snr_min` → `--snr-min`, `sup_min` → `--sup-min`, `band_peak_window` → `--peak-window`.
  - Suggested descriptions: "the probe check's random seed", "the recording lengths, in seconds", "the probe frequencies, as multiples of the peak frequency", "the non-dimensional drive amplitudes", "the number of noise repeats per point", "the lock-in ceilings, in drive cycles", "the largest amplitude spread that passes", "the largest phase spread that passes, in radians", "the smallest signal over the lock-in floor that passes", "the smallest own-peak power fraction that passes", "the own-peak window, as a fraction of the peak frequency".
- **Help (spec §3.3–§3.5):**
  - Every parser gets `description=` equal to its `help=`.
  - House metavars: `S` for lengths, `X` for multipliers, drives, thresholds and the window, `N` for repeats, cycle caps and the seed, `PATH`.
  - `--help` imports neither torch nor `core.config`.
  - Each value flag's help ends with the clause `help_defaults.default_for("probes band", action)` returns. Add `"probes band"` to `LEAVES` and one `DEFAULT_CLASS` entry per flag, in the format Task 11 set:
    - `--bounds` and `--cell`: none.
    - `--model`: the same entry as the other leaves ("the bounds file's parent folder, upper-cased").
    - `--device`: the same entry as the other leaves (parser default `auto`).
    - `--lengths`: behaviour, "five log-spaced from the shortest expected recording to this cell's training ceiling".
    - `--multipliers`: behaviour, "four across the configured band plus one control at twice its top".
    - `--drives`: behaviour, "the configured chi drive".
    - `--repeats`: value, "24".
    - `--cycle-caps`: behaviour, "the configured cycle ceiling only".
    - `--cv-max`: value, "0.2".
    - `--phase-max`: behaviour, "not judged".
    - `--snr-min`: value, "3.0".
    - `--sup-min`: value, "0.5".
    - `--peak-window`: value, "0.1".
    - `--seed`: value, "0".
    - `--name` and `--note`: as on every other leaf.
  - Extend `_owned_defaults()` in `tests/test_tool.py` (the help table's owner pin; it is not `owned_by_a_signature` in `tests/test_refusals.py`, which you extend separately) with one entry per pinned literal, keyed `("probes band", "<flag>")` and valued `str(inspect.signature(core.diagnostics.probe_band).parameters[<the flag's dest>].default)`: `--seed`, `--repeats`, `--cv-max`, `--snr-min`, `--sup-min`, `--peak-window`. `_default_problems` accepts a pinned literal only when `_owned_defaults` holds it; through the registry, `--seed` would resolve to the `seed` key.
- **`PROBES_INTERRUPT_NOTE`** (exact): `"a probe check writes its record only when it finishes, so the interrupted one was removed and there is nothing to clear; nothing resumes, so the same command starts the measurement again."` Every mode sets it through `set_defaults(interrupt_note=...)` (the generic `--resume require` advice would be wrong).
- **Test cost:** every fast test uses the stand-in simulator. The one real twin-cell check is `@pytest.mark.slow` (spec §7). A tiny real call of `_simulate` stays fast.
- **Global rules:**
  - Every knob is a keyword argument, with its default in the signature only; the tool forwards only the flags given (`config_args.knobs`).
  - `probe_band` is `@public_entry` and joins the exact public-entry set.
  - Never edit a source file while a pytest run is in progress.
  - H4: test names, docstrings, comments and messages carry no process label. Describe the earlier one-off measurement in words, never by an archive path.

- [ ] **Step 1: Read the model code:** `core/diagnostics/identifiability.py` (the jacobian mode, lines 817-918), `core/tool/diagnostics.py`, `core/tool/fdt.py`'s `interrupt_note` use, the current `core/tool/help_defaults.py` (Task 11) and `derived.for_simulation` (Task 13).
- [ ] **Step 2: Write the failing stage tests** in `tests/test_diagnostics.py`. Add these helpers, which Task 24 reuses:

```python
def _probe_cfg(bounds="master.txt", cell="master_spont.txt", *, chi=True):
    """A CPU Nadrowski config with a cell loaded, built the way the probe checks build one."""
    from core import cli, config, registry
    from core.config import VALID_LABELS, VALID_MODELS
    cfg = cli.make_sim_config("NADROWSKI", VALID_LABELS[VALID_MODELS.index("NADROWSKI")],
                              registry.state_dep_drift("NADROWSKI"),
                              str(config.BOUNDS_PATH / "nadrowski" / bounds), chi_mode=chi)
    cfg.hw = config.cpu_device()
    cli.load_and_validate_gt(cfg, str(config.CELL_PATH / "nadrowski" / cell))
    return cfg


def _probe_stand_in(*, f_own=0.025, own_factor=None, gain=2.0, noise=0.01, seen=None, nan_driven=False):
    """In place of probes._simulate, a cell of known response. Its own oscillation sits at f_own (cell
    frequency units, one value or one per row) with a random phase per row. A drive adds a response of
    |chi| = gain in phase with it, and scales the own oscillation by own_factor(strength in model units)."""
    import math
    own_factor = own_factor or (lambda a: torch.where(a >= 1.0, 0.1, 1.0))

    def _sim(cfg, geom, nd, res_sim, sim_idx, inits, *, amp_dim=None, freq=None):
        B, n = nd.shape[0], geom.n_obs
        t = torch.arange(n, dtype=torch.float64) * cfg.dt_exp
        f = torch.as_tensor(f_own, dtype=torch.float64).reshape(-1, 1)
        own = torch.sin(2 * math.pi * f * t + 2 * math.pi * torch.rand(B, 1, dtype=torch.float64))
        x = own + noise * torch.randn(B, n, dtype=torch.float64)
        if amp_dim is not None:
            amp = amp_dim.double().reshape(-1, 1)
            strength = amp / res_sim[:, sim_idx["f_scale"]].double().reshape(-1, 1)
            if seen is not None:
                seen.append((strength.flatten().tolist(), freq.double().flatten().tolist()))
            x = x + (own_factor(strength) - 1.0) * own + gain * amp * torch.cos(2 * math.pi * freq.double().reshape(-1, 1) * t)
            x = torch.full_like(x, float("nan")) if nan_driven else x
        return x.to(cfg.hw.dtype)
    return _sim
```

The tests, with their assertions:

```python
def test_probes_band_judges_the_configured_band_and_drive_on_a_cell_of_known_response(store, monkeypatch, caplog):
    import math
    import numpy as np
    from core.diagnostics import probes, probe_band
    monkeypatch.setattr(probes, "_simulate", _probe_stand_in())
    d = probe_band(_probe_cfg(), lengths=[1.0, 2.0], multipliers=[0.05, 0.3, 0.6], repeats=4, name="band1")
    s, r = d.settings, d.results
    assert (d.diagnostic, d.variant, d.manifest.parents) == ("probes", "band", {})
    assert s["drives"] == [0.15] and s["cycle_caps"] == [20.0] and s["phase_max"] is None
    assert s["drive_phase"] == pytest.approx(math.pi / 2)
    assert s["configured"] == {"chi_f0": 0.15, "chi_freq_bounds": [0.03, 0.3], "chi_k_pad": 12,
                               "chi_min_cycles": 2.0, "chi_max_cycles": 20.0, "probe_count": 6}
    assert r["band_holds"] is True and r["drive_holds"] is True
    assert [(f["multiplier"], f["in_band"], f["harmonic"], f["passes_capped"]) for f in r["frequencies"]] == [
        (0.05, True, False, True), (0.3, True, True, True), (0.6, False, False, True)]
    assert len(r["points"]) == 6 and all(p["n_valid"] == 4 and p["nyquist_masked"] == 0 for p in r["points"])
    slow = r["points"][0]                      # 0.05 x 25 Hz over 1 s is 1.25 drive cycles
    assert (slow["length_s"], slow["multiplier"], slow["drive"]) == (1.0, 0.05, 0.15)
    assert slow["full"]["floor_masked"] == 1.0 and slow["full"]["verdict"] == "pass"   # reported, not judged
    assert r["lengths"][0]["omega0_hz"] == pytest.approx(25.0) and r["training_ceiling_s"] == pytest.approx(26.909)
    assert len(r["caveats"]) == 2
    z = np.load(d.path / "probe_band.npz")
    assert z["cv_capped"].shape == z["snr_full"].shape == (2, 3, 1) and d.manifest.figures
    said = [m.getMessage() for m in caplog.records if m.name == "core.diagnostics.probes" and m.levelname == "INFO"]
    assert any(m.startswith("[band] the configured band (0.03, 0.3) holds for this cell") for m in said), said
    assert any(m.startswith("[band] the configured drive 0.15 holds for this cell") for m in said), said


def test_probes_band_fails_a_frequency_captured_at_any_drive_and_judges_the_configured_drive_alone(store, monkeypatch, caplog):
    from core.diagnostics import probes, probe_band
    kw = dict(lengths=[1.0], multipliers=[0.12, 0.2], repeats=4)        # drives on exact bins: no leakage
    monkeypatch.setattr(probes, "_simulate", _probe_stand_in())        # captures from strength 1.0
    r = probe_band(_probe_cfg(), drives=[0.15, 2.0], name="band2", **kw).results
    assert r["band_holds"] is False and r["drive_holds"] is True
    assert all(not f["passes_capped"] and any("captured" in why for why in f["reasons"]) for f in r["frequencies"])
    assert [(x["drive"], x["configured"], len(x["captured_in_band"])) for x in r["drives"]] == [(0.15, True, 0), (2.0, False, 2)]
    monkeypatch.setattr(probes, "_simulate", _probe_stand_in(own_factor=lambda a: torch.where(a >= 0.1, 0.1, 1.0)))
    r = probe_band(_probe_cfg(), name="band3", **kw).results
    assert r["band_holds"] is False and r["drive_holds"] is False
    said = [m.getMessage() for m in caplog.records if m.name == "core.diagnostics.probes"]
    assert any(m.startswith("[band] the configured drive 0.15 does not hold for this cell") for m in said), said
    assert any("core/config.py" in m for m in said)


def test_probes_band_reports_a_probe_past_the_sampling_limit_as_masked_and_never_clamps_it(store, monkeypatch):
    from core.diagnostics import probes, probe_band
    seen = []
    monkeypatch.setattr(probes, "_simulate", _probe_stand_in(f_own=[0.2, 0.2, 0.4, 0.4], seen=seen))
    d = probe_band(_probe_cfg(), lengths=[1.0], multipliers=[0.05, 1.5, 3.0], repeats=4, name="fast")
    pts = {p["multiplier"]: p for p in d.results["points"]}
    assert (pts[0.05]["n_valid"], pts[0.05]["nyquist_masked"]) == (4, 0)
    assert (pts[1.5]["n_valid"], pts[1.5]["nyquist_masked"]) == (2, 2) and pts[1.5]["full"]["cv"] is not None
    assert (pts[3.0]["n_valid"], pts[3.0]["nyquist_masked"]) == (0, 4)
    assert pts[3.0]["full"]["verdict"] == pts[3.0]["capped"]["verdict"] == "masked"
    assert pts[3.0]["full"]["cv"] is None and pts[3.0]["sup"] is None
    fast = next(f for f in d.results["frequencies"] if f["multiplier"] == 3.0)
    assert not fast["passes_capped"] and any("masked" in why for why in fast["reasons"])
    assert max(f for _, freqs in seen for f in freqs) == pytest.approx(0.6, rel=1e-6)   # driven where asked
    assert store.load_diagnostic(d.id).results == d.results
```

Also write:
- `test_probes_band_brackets_the_wall_only_when_caps_are_given`: `lengths=[2.0], multipliers=[0.3], repeats=2`. With no caps, `results["wall"] is None`. With `cycle_caps=[8, 12]`: `settings["cycle_caps"] == [8.0, 12.0, 20.0]`; the caps listed in `wall["caps"]` are `[8.0, 12.0, 20.0]` with `n_truncated` `[1, 1, 0]`; `n_fail == 0` for every cap that binds; `"above 12" in wall["reading"]`; `points[0]["full"]["cycles"] == approx(15.0)`.
- `test_probes_band_turns_non_finite_measures_into_null`: `_probe_stand_in(nan_driven=True)`, `lengths=[1.0], multipliers=[0.12], repeats=2`. `full["cv"]`, `full["snr"]` and `sup` are None; the verdict is `"not measured"`; `band_holds is False`; `drive_holds is None`; the reloaded record's results equal `d.results`.
- `test_probes_band_refuses_before_the_writer_opens`, parametrized over `(kw, field)`: `{"repeats": 1}` → band_repeats; `{"lengths": [0.0]}`, `{"lengths": [1e-7]}` and `{"lengths": []}` → probe_lengths; `{"multipliers": [0.0]}` and `{"multipliers": [nan]}` → probe_multipliers; `{"drives": [-1.0]}` → probe_drives; `{"cycle_caps": [0.0]}` → probe_cycle_caps; `{"cv_max": nan}` → cv_max; `{"snr_min": inf}` → snr_min; `{"sup_min": 0.0}` and `{"sup_min": 1.5}` → sup_min; `{"phase_max": nan}` → phase_max; `{"peak_window": 0.0}` and `{"peak_window": 1.0}` → band_peak_window; `{"seed": -1}` → probe_seed; `{"name": "taken"}` → name. Write `_diagnostic(store, cfg, name="taken")` first. Base kwargs are `lengths=[1.0], multipliers=[0.1], repeats=2`. `_simulate` is patched to `pytest.fail`. Assert the Refusal's field, and that the set of directories under `store.kind_dir("diagnostic")` is unchanged. A companion `test_probes_band_refuses_a_config_without_chi_mode` uses `_probe_cfg(chi=False)` and expects a Refusal with `field is None`, with nothing created.
- `test_a_probes_run_leaves_a_training_configuration_and_config_py_untouched`: take `snapshot_cfg` of `_nad_cfg(chi_mode=True)` and `{k: getattr(config, k) for k in dir(config) if k.startswith("CHI_")}`, then run `probe_band(_probe_cfg(), lengths=[1.0], multipliers=[0.05, 0.6], drives=[0.5], repeats=2)` with the stand-in collecting `seen`. Afterwards `assert_cfg_unchanged` holds; the CHI_ constants are equal; a freshly built `_nad_cfg(chi_mode=True)` has `(chi_f0, chi_freq_bounds) == (config.CHI_F0, config.CHI_FREQ_BOUNDS)`; and every strength in `seen` is 0.5 (the grid reached only the simulator).
- `test_the_probe_simulator_drives_a_tier1_cell_at_its_derived_force_scale`, fast and real at 0.2 s. On `_probe_cfg("master_tier1.txt", "master_spont_tier1.txt")`, spy `pipeline.build_nondim_sin_force_tensor` (wrap the real one and record the rescale vector and index). Call `probes._simulate` DRIVEN with 2 rows (an undriven call builds no force, so the spy would record nothing): `amp_dim` a (2,) tensor of `cfg.chi_f0 * f_scale_eff` and `freq` a (2,) tensor of 0.005 cell frequency units, using `derived.for_simulation` and `probe_math.recording_geometry(cfg, 0.2, t_scale)`. Assert: the shape is `(2, geom.n_obs)` and every value is finite; the recorded index has `"f_scale"` and no `"T"`; the recorded `f_scale` is `approx(50 * 14.1 * cfg.k_b_cell * 300 / 62.14, rel=1e-5)` (≈ 46.99).
- `@pytest.mark.slow test_probes_band_gives_the_same_criteria_on_the_master_cell_and_its_tier1_twin`: `lengths=[1.0], multipliers=[0.1392], repeats=8, seed=0` on `_probe_cfg()` and `_probe_cfg("master_tier1.txt", "master_spont_tier1.txt")`, both real on the CPU. For `col` in ("full", "capped") and `m` in ("cv", "phase", "snr"), `a[col][m] == approx(b[col][m], rel=1e-3, abs=1e-6)`. Also `a["sup"] == approx(b["sup"], rel=1e-3)`, and `a["full"]["chi_mag"] / b["full"]["chi_mag"] == approx(50 * 14.1 * 1.380649e-2 * 300 / 62.14 / 10.0, rel=1e-3)`.
- [ ] **Step 3: Write the failing tool, registry and table tests.**
  - In `tests/test_tool.py`:
    - `test_probes_band_builds_a_chi_config_with_the_cell_and_forwards_every_knob`: patch `"core.diagnostics.probe_band"` with a recorder and pass every flag with a distinct value (`--lengths 1 2.5 --multipliers 0.05 0.3 --drives 0.2 --repeats 6 --cycle-caps 8 12 --cv-max 0.25 --phase-max 0.6 --snr-min 2.5 --sup-min 0.4 --peak-window 0.08 --seed 3 --name b1 --note hello`). Assert that `set(kw)` is exactly `{"name", "note", "fig_sink", "store", "lengths", "multipliers", "drives", "repeats", "cycle_caps", "cv_max", "phase_max", "snr_min", "sup_min", "peak_window", "seed"}` and that each value is right (lists as floats). Assert the positional config has `chi_mode is True`, a loaded ground truth, and `chi_f0`/`chi_freq_bounds` equal to config.py's. A bare call forwards exactly `{"name", "note", "fig_sink", "store"}`.
    - `test_the_probes_family_keeps_its_modes_apart_and_its_help_imports_no_torch`. The set of mode names is `{"band"}`. `band` lacks each of `--chi --no-chi --chi-k --chi-f0 --chi-band --f0 --prior --posterior --num-runs --strengths --t-obs`. `band.get_default("chi_mode") is True` and its interrupt note is `tool_probes.PROBES_INTERRUPT_NOTE`. `probes band ... --chi` exits 2. A stage stub raising `KeyboardInterrupt` exits 130, and stderr starts with `"prism probes: interrupted: a probe check writes its record only when it finishes"`. A fresh-interpreter probe (the pattern at `tests/test_tool.py:3126-3134`) runs `main(['probes', '--help'])` and `main(['probes', 'band', '--help'])`, which return 0 with no torch module loaded.
    - `test_the_probe_modules_write_no_setting_and_hand_no_grid_to_training`. For `core/diagnostics/probes.py`, `core/diagnostics/probe_math.py` and `core/tool/probes.py`: `_env_reads_and_knob_writes(tree) == []`; no call whose unparsed function is `replace` or `dataclasses.replace` or ends with `make_sim_config`; every keyword named `chi_f0` or `chi_freq_bounds` sits in a call ending with `gen_training_data`, and its value unparses to `cfg.chi_f0` / `cfg.chi_freq_bounds`; no assignment target contains an attribute starting with `chi_` or `CHI_`.
  - In `tests/test_refusals.py`: append the 11 keys to `TOOL_ONLY_KEYS`; raise the count from 88 (read the current number) by 11; add `probe_seed`, `band_repeats`, `cv_max`, `snr_min`, `sup_min` and `band_peak_window` to `owned_by_a_signature` against `probes.probe_band`'s `seed`, `repeats`, `cv_max`, `snr_min`, `sup_min` and `peak_window`. New test `test_the_seed_rule_carries_the_field_key_it_is_given`: `require_seed(-1, key="probe_seed")` and `require_seed(SEED_MAX + 1, key="probe_seed")` both refuse with `field == "probe_seed"`, and the message, with any trailing period stripped, ends with `"(default 0)"`; `require_seed(-1)` still carries `"seed"`; `require_seed(7, key="probe_seed") == 7`.
  - In `tests/test_nav_and_gating.py:1607-1612`: add the 11 keys to the set.
  - In `tests/test_artifact_store.py`:
    - Add `("core/diagnostics/probes.py", "probe_band")` to `want`.
    - Add `_leg_probe_band`. It uses `cfg = _nad_cfg(chi_mode=True)` with `master_spont.txt` loaded and `probes._simulate` patched to `_body_done`. Its call is `probes.probe_band(cfg, lengths=[1.0], multipliers=[0.1], repeats=1 if case == "refusal" else 2, store=_EntryStore(case))`.
    - Add `"probe_band": _leg_probe_band` to `_UNTOUCHED_LEGS` and `"probe_band": "band_repeats"` to `_REFUSAL_FIELDS`.
- [ ] **Step 4: Run the new tests and see them fail.** `python -m pytest tests/test_diagnostics.py tests/test_tool.py tests/test_refusals.py -q -k "probe or seed_rule"`. Expected: an `ImportError` for `probe_band` / `core.tool.probes`, and a `TypeError` for `require_seed`'s `key=`.
- [ ] **Step 5: Implement the keys, the seed rule and `add_config_flags(..., chi=...)`** as the Binding lists them.
- [ ] **Step 6: Implement `_simulate` and `probe_band`** in `core/diagnostics/probes.py`.
  - `_simulate` returns `(B, geom.n_obs)` traces in the cell's length unit.
    - Undriven when `amp_dim is None` (`forcing.zero_force`).
    - Otherwise the forcing parameters are `[amp_dim, freq, π/2, 0]` against the literal index, built through `pipeline.build_nondim_sin_force_tensor(fp, t_fine, res_sim, fidx, sim_idx)` and padded to `forcing.n_force_channels(cfg.model, fidx, inits.shape[-1])` channels as `gen_chi_raw` pads.
    - Simulate with `pipeline.gen_obs(model=cfg.model, params=nd, t=cfg.t[:geom.n_fine], inits=inits, force=force, n_segs=max(1, ceil(geom.n_fine / CHUNK_LEN)), steady_idx=cfg.steady_idx, state_dep_drift=cfg.state_dep_drift, batch_size=B, var_idx=0, dtype=..., device=...)[0]`, then `[:, ::geom.subsample][:, :geom.n_obs]`, then `helpers.rescale(x, x_scale, x_offset)`.
    - Look up both pipeline functions through the module at call time.
  - `probe_band` follows the Binding's order, then measures:
    - (a) **Resolve.** Compute these before the rules, because the length rule needs `t_scale` (the default grids are still resolved after the rules, before `store.create`): `res` is the cell's rescale values as a `(1, n)` tensor in `cfg.hw.dtype` on `cfg.hw.device`, built the way `generate_observations` builds `rescale_gt`; `res_sim, sim_idx = derived.for_simulation(...)`; `t_scale = float(res_sim[0, sim_idx["t_scale"]])`. Keep that dtype: on the CPU test configs it is float32, so 3.73 reads 3.7300000190734863 and 1 s gives 1000 samples, which the tests' exact bins rely on (`omega0_hz == approx(25.0)`, the 0.6 drive in the sampling-limit test); a float64 3.73 gives 999 samples and fails them. Default `lengths = probe_math.default_lengths(cfg, t_scale)`, `multipliers = probe_math.default_multipliers()`, `drives = [cfg.chi_f0]`. `caps = sorted(set(cycle_caps or []) | {cfg.chi_max_cycles})`, and the configured ceiling is the "capped" column. The ensemble is `repeats` copies of the cell's parameters, simulation vector and initial conditions.
    - (b) **Each length.** Compute the geometry. When `n_fine > cfg.t.shape[0]`, clip `n_obs` to `(len(t) - steady_idx) // subsample` and flag `clipped`. Flag `beyond_ceiling` when `n_fine > min(N_ND_MAX, len(t))`, and warn for each flag. Simulate the undriven ensemble `x0`. Take `f_peak = chi.peak_freq(x0, dt)` per row, and the ensemble peak Ω₀ = the argmax of `probe_math._ensemble_power(x0, dt)` with DC excluded.
    - (c) **Each multiplier.** `freq = m * f_peak`. The rows kept are those with finite freq, `freq > 0` and `freq < 0.9 * nyq`. Fewer than two kept rows makes the point `"masked"` (every measure None, and no driven simulation). Otherwise, for each drive: `amp = F0 * f_scale_eff`. Floor lock-ins of `x0[ok]` at full length and at each cap. Drive the whole ensemble at its own `freq` (never clamped). Driven lock-ins at full length and at each cap, with `n_samples = N_row`.
    - (d) **Measures over the kept rows.**
      - `cv` = the unbiased std of |χ| ÷ its mean.
      - `phase = circular_spread(χ)`.
      - `snr` = mean |χ_driven| ÷ mean |χ_floor|.
      - `chi_mag` = mean |χ_driven|.
      - `cycles` = mean of `freq · T_row`.
      - `floor_masked` = the fraction of kept rows with `freq · T_row < config.CHI_MIN_CYCLES` (reported, never judged).
      - `sup = own_peak_ratio(x_driven[ok], x0, Ω₀, peak_window, dt)`. It is taken over the whole trace, so it has no capped counterpart.
    - (e) **Verdicts.** `"not measured"` if any judged measure is None. Otherwise the failing tags joined by ", " from `noisy` (`cv > cv_max`), `phase` (only when `phase_max` is given), `low signal` (`snr < snr_min`) and `captured` (`sup < sup_min`), or `"pass"`.
    - (f) **Summaries.**
      - A frequency passes only if it passes at every length and every drive (full and capped separately), with `reasons` such as `"captured at 1.00 s, drive 2"` or `"masked at the sampling limit at 1.00 s"`.
      - `in_band` = the multiplier lies inside `cfg.chi_freq_bounds` (1e-9 relative tolerance). `band_holds` = every in-band frequency passes capped (None when no multiplier is in band).
      - `drive_holds` is judged on the configured drive's in-band points only: True when every one has a finite `sup ≥ sup_min`, False when any finite `sup < sup_min`, None otherwise. Each drive's captured in-band points go under `drives`.
      - `cap_rescue` = `{"failed_full", "shortened", "rescued", "reading"}`, where a failure is rescued when the configured cap shortened it and its capped verdict passes.
      - `wall` is present only when `cycle_caps` was given. For each cap it records the truncated points and their worst cv, snr and phase and `n_fail`, then the first failing cap and the largest clean cap below it. `reading` is one of "no cap binds any point", "no cap in the grid fails: the wall lies above X cycles" (X is the largest cap that binds any point, printed with `:g`: caps 8, 12 and 20 over a 15-cycle point read "above 12"), "the smallest cap that binds (X cycles) already fails: the grid does not bracket the wall from below", or "the wall lies between X and Y cycles".
      - `harmonic` comes from `probe_math.harmonic_flags(multipliers, peak_window)`.
      - The two `caveats` sentences: the reproducibility, floor and phase thresholds are conventions and only capture is physical evidence; and a flagged probe's harmonic power inflates the not-captured measure and can hide capture.
    - (g) **Write.**
      - `results` = `{"lengths", "points", "frequencies", "drives", "band_holds", "drive_holds", "cap_rescue", "wall", "training_ceiling_s", "caveats", "summary"}`. The `points` rows carry `length_s, multiplier, drive, n_valid, nyquist_masked, harmonic, sup, full{cv, phase, snr, chi_mag, cycles, floor_masked, verdict}, capped{…same}, caps[{cap, cv, phase, snr, cycles, verdict}]`. The `lengths` rows carry `length_s, achieved_s, n_obs, n_fine, beyond_ceiling, clipped, omega0_cell, omega0_hz, peak_median_hz, peak_spread_hz`.
      - `settings` = `{"lengths", "multipliers", "drives", "repeats", "cycle_caps", "cv_max", "phase_max", "snr_min", "sup_min", "peak_window", "seed", "drive_phase", "configured"}`.
      - The npz holds `(L, M, D)` arrays `cv_full cv_capped phase_full phase_capped snr_full snr_capped chi_mag sup cycles_full cycles_capped n_valid`, plus the axes.
      - One 2×2 heatmap figure per drive (cv and snr, full and capped, one colour scale per measure), titled `f"Probe band at drive {F0:g}"`.
      - `w.config.update(settings)`, then `w.body = {...}`, then return `store.load_diagnostic(w.id)`.
- [ ] **Step 7: Implement the tool family** in `core/tool/probes.py`.
  - `register(sub)`: the parent `probes` gets `help` and `description` ("check the chi probe settings on a cell or a prior"), then `add_subparsers(dest="variant", required=True)`.
  - The `band` mode (`help` and `description`: "do the configured chi band and drive hold for this cell?") calls `config_args.add_config_flags(band, chi=False)`, then `add_name_flags` (directly after the configuration flags, where every other subcommand has `--name`/`--note`), then `--cell` (required; canonical wording plus "the probes are measured on"), then the eleven value flags with their destinations equal to the stage's keywords (`nargs="+"` for the four lists), then `set_defaults(handler=_band, chi_mode=True, chi_n_freqs=None, interrupt_note=PROBES_INTERRUPT_NOTE)`.
  - `_band` imports `core.diagnostics` inside the function, runs `cfg, _ = config_args.build_cfg(args, load_gt=True)` (it returns `(cfg, ignored)`), and returns `report(diag.probe_band(cfg, name=..., note=..., fig_sink=config_args.close_sink, store=store, **knobs(args, ...eleven knob names...)))`.
  - Register it in `core/tool/__init__.py`, and write the help-defaults entries, the `__init__` re-export and the widened comments.
- [ ] **Step 8: Run the new tests and see them pass.** Step 4's command (expected: all pass), then `python -m pytest tests/test_diagnostics.py -q -m slow -k twin` (expected: `1 passed`).
- [ ] **Step 9: Run the focused tests for every touched file:** `python -m pytest tests/test_diagnostics.py -q -m "not slow"`; `python -m pytest tests/test_tool.py -q -k "probes or every_leaf_parser_has_a_description or every_value_flag or torch or reads_no_environment or print_call"`; `python -m pytest tests/test_refusals.py -q`; `python -m pytest tests/test_nav_and_gating.py -q -k window_control`; `python -m pytest tests/test_artifact_store.py -q -k "public_entr"`. Expected: all pass.
- [ ] **Step 10: Commit.** `git add` every file under Files explicitly (`core/diagnostics/probes.py core/tool/probes.py core/diagnostics/__init__.py core/tool/__init__.py core/tool/config_args.py core/rng.py core/refusals.py core/tool/fields.py core/gui/fields.py core/tool/help_defaults.py core/artifacts/manifest.py core/artifacts/store.py tests/test_diagnostics.py tests/test_tool.py tests/test_refusals.py tests/test_nav_and_gating.py tests/test_artifact_store.py`), then `git commit -m "probes band: the band and drive check, and the probes family" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"`. The controller starts the review and the one-process fast gate (`python -m pytest -m "not slow" -q`, background, logged) at the same moment. The implementer does not run the gate.

---

### Task 23: `probes mask`

**Files:**
- Modify: `core/diagnostics/probes.py`: add `probe_mask` (`@public_entry`), `_capture_masked_warnings()` and `_audit(cfg, records, warned) -> dict`.
- Modify: `core/diagnostics/__init__.py`: re-export `probe_mask`.
- Modify: `core/tool/probes.py`: the `mask` mode and `_mask(args, store)`.
- Modify: `core/tool/diagnostics.py:3-6` (the docstring's "There is no ``--prior`` on a diagnostic ..."), `core/tool/help_defaults.py`, `core/refusals.py`, `core/tool/fields.py`, `core/gui/fields.py`.
- Test: `tests/test_diagnostics.py`, `tests/test_tool.py`, `tests/test_refusals.py`, `tests/test_nav_and_gating.py:1607-1612`, `tests/test_artifact_store.py` (`want`, `_leg_probe_mask`, `_UNTOUCHED_LEGS`, `_REFUSAL_FIELDS`).

**Interfaces:**
- Consumes: T21 `ProbeRecord` (all fields, including `n_points`) and `pipeline.gen_training_data(..., probe_observer=...)`. T22 `PROBES_INTERRUPT_NOTE`, `add_config_flags(p, *, chi=False)`, `require_seed(seed, key="probe_seed")`, the family parser and the `probe_seed` key. `store.load_prior(cfg, ref) -> LoadedPrior` (`.prior`, `.force_prior`, `.id`, `.fingerprint`). `orchestrator._observation_inits(cfg)`. `run_guards._assert_chi_config_is_deliberate(cfg)`.
- Produces (registry T23): `core.diagnostics.probe_mask(cfg, prior: LoadedPrior, *, num_runs=12, run_size=32, chi_k_fixed=None, seed=0, name="", note="", fig_sink=None, store=None) -> LoadedDiagnostic`, and the field keys `mask_num_runs` and `mask_run_size`. It reuses `prior`, `chi_k_fixed` and `probe_seed`.

**Binding (read before starting):** spec §4.1a (the mask rows), §4.2 and §4.4 are binding. Restated:
- `python` below means `C:\Users\J\anaconda3\envs\biophys-env\python.exe` (in the Bash tool, `/c/Users/J/anaconda3/envs/biophys-env/python.exe`); the `python` on PATH is a different interpreter without torch or pytest. Commands written with backslash paths are PowerShell commands; in Bash use forward slashes.
- **Inputs.** The tool's handler loads the prior with `store.load_prior(cfg, args.prior)`, which refuses a model, order, box or log-mask mismatch, and names a missing one. A missing prior is **refused, never built**. The prior is the recorded parent: `w.parents = {"prior": prior.id}` and `w.fingerprints["gmm"] = prior.fingerprint`. There is no cell: training draws no truth, and the initial conditions come from `orchestrator._observation_inits(cfg)` (`cfg.inits_tensor` would raise on a cell-free config).
- **Stage order:**
  1. `resolve_store`, then `assert_name_free("diagnostic", name)`.
  2. Refuse a non-chi config (`Refusal(..., field=None)`), then `run_guards._assert_chi_config_is_deliberate(cfg)`, which `store.load_prior` does not run.
  3. The rules:
     - `require_seed(seed, key="probe_seed")`
     - `require_at_least("mask_num_runs", num_runs, 1)`
     - `require_at_least("mask_run_size", run_size, 1)`
     - When `chi_k_fixed` is given: `require_at_least("chi_k_fixed", chi_k_fixed, config.CHI_K_MIN_TRAIN)` (2, the floor of training's per-batch draw), and `refuse("chi_k_fixed", ...)` above `cfg.chi_k_pad`. Both are refused up front, never left to the mid-run `ValueError` at `core/SBI/pipeline.py:1342-1348`.
  4. `store.create`.
- **The real generator, measure-only** (spec §4.2, §4.4):
  - Call `pipeline.gen_training_data(cfg.model, prior.prior, prior.force_prior, cfg.t, run_size=..., n_runs=..., steady_idx=cfg.steady_idx, dt_nd_min=cfg.dt_nd_min, nd_dim=len(cfg.params_dict), forcing_idx=cfg.forcing_idx, rescale_idx=cfg.rescale_idx, dt_exp=..., t_min_exp=..., t_max_exp=..., t_scale_bounds=cfg.t_scale_bounds, state_dep_drift=cfg.state_dep_drift, chi_mode=True, chi_f0=cfg.chi_f0, chi_freq_bounds=cfg.chi_freq_bounds, chi_k_pad=cfg.chi_k_pad, chi_k_fixed=chi_k_fixed, chi_max_cycles=cfg.chi_max_cycles, n_vars=orchestrator._observation_inits(cfg).shape[-1], checkpoint=None, nd_idx=..., k_b_cell=... (from *cfg.tier1_args), dtype=..., device=..., probe_observer=records.append)`.
  - Everything from `run_size` on is passed by keyword. There is no `theta_transform`: the physical prior has the same physical distribution. The values come from the config the deliberate-config guard just checked, so it is config.py's band and drive.
  - `checkpoint=None`, so **no simulation record is written**. Never mutate `cfg.hw`.
  - Look up `pipeline.gen_training_data` through the module at call time.
  - Never monkeypatch production code: the observer is the only tap.
  - Run it inside `seeded(seed, device)` and `_capture_masked_warnings()`.
- **`_capture_masked_warnings()`** is a context manager yielding a list of `(batch_tag, masked, total)`. Inside `warnings.catch_warnings()` it sets `warnings.simplefilter("always")`, because two halves of a batch can warn identical text, which the default once-per-message registry would swallow. It replaces `warnings.showwarning` with a tee that appends every message matching `re.match(r"(?P<tag>.+?): chi: (?P<masked>\d+)/(?P<total>\d+) probes masked", str(message))` and **forwards every warning** to the hook that was in force, so the lines still reach the console and `log.txt`.
- **`_audit(cfg, records, warned) -> dict`** is pure; floats go through `orch._num`. Per record: `freq = f_peak[:, None] * exp(u)` (float64), `nyq = 0.5 / dt_exp`, `T_full = n_points * dt_exp`, `(u_mid, u_half) = chi.band_norm(cfg.chi_freq_bounds)`. Then:
  - `bad = ~isfinite(freq) | (freq <= 0)`
  - `nyqm = (freq >= 0.9 * nyq) & ~bad`
  - `floor = ~valid & ~bad & ~nyqm`
  - `too_slow` = the row's `f_peak * T_full * hi < config.CHI_MIN_CYCLES`, which splits `floor` into `too_slow_at_band_top` and `shortened_by_duration_draw`
  - packer extra per row = `(valid.sum(1) - packed_mask[:, :k].sum(1)).clamp(min=0)`
  - `out_of_band` = valid probes with finite `u` and `|(u - u_mid) / u_half| > config.CHI_UHAT_MAX`
  - `non_finite_lock_in` = packer extra − `out_of_band`
  - `masked = (~packed_mask[:, :k]).sum()` (production's count); `live = packed_mask[:, :k].sum()`.
- **Results keys:**
  - `probes`, `live`, `masked`, `masked_fraction`.
  - `causes` = `{cycle_floor, too_slow_at_band_top, shortened_by_duration_draw, non_finite_lock_in}`, each `{"count", "share"}`, where the share is taken of all probes.
  - `invariants` = `{"non_finite_frequency", "at_or_above_nyquist", "out_of_band", "hold"}`. These causes cannot fire in training (`recon/design-maskaudit.md` §5); a violation is logged as a warning.
  - `per_batch` = `{"fractions", "mean", "sd", "n_batches"}`. Group by `batch_tag`; `sd` uses `ddof=1` and is None for one batch. The report says the effective sample size is the batch count.
  - `omega0` = `{"quantiles": [0.05, 0.25, 0.5, 0.75, 0.95], "cell_units", "hz"}`.
  - `rows` = `{"total", "zero_live", "one_live"}`, counted after packing and before per-row subsetting.
  - `span` = `{"quantiles": [0.25, 0.5, 0.75], "values", "rows", "single_probe_rows"}`. It is each row's max/min of `freq` over its valid probes: the frequencies actually driven, never the drawn multipliers.
  - `driven_multipliers` = `{"quantiles": [0.05, 0.5, 0.95], "values"}` of `exp(u)` over valid probes.
  - `dominant_cause`: the key of the largest of the three leaf causes' counts (`too_slow_at_band_top`, `shortened_by_duration_draw`, `non_finite_lock_in`; a tie goes to the first in that order), or None when nothing is masked. `cycle_floor` is the sum of the first two and is never the answer.
  - `cross_check` = `{"row_ranges", "mismatches", "abandoned_attempt_warnings"}`, compared **per row range**, tag by tag. The expected entries are the committed records with `masked > 0`, in range order, as `(masked, (hi - lo) * k)`. They must equal the **last** that many captured warnings of the same tag. Earlier captured warnings of that tag, and warnings of a tag with no masked record, count as `abandoned_attempt_warnings`: the whole-batch retry and the row halving both re-run work that already warned. Every other difference is a mismatch `{"batch_tag", "lo", "hi", "audit", "production"}` (production None when no warning is left to match). It is recorded and logged as ONE warning record containing "does not match", never silently accepted.
- **Report and record.**
  - Log lines `"[mask] ..."`: the total and the shares by cause; the reading (too slow at the band's top dominating → "the lever is the prior's peak-frequency range"; shortened by the duration draw dominating → "the lever is the duration draw"); Ω₀ quantiles in Hz; the per-batch spread; the zero-live and one-live rows; the invariant checks; the cross-check.
  - `settings` = `{"num_runs", "run_size", "chi_k_fixed", "seed", "configured", "probe_layout"}`. `probe_layout` is exactly: `"the probe count, placement and duration draw come from training's own fixed probe-generator seed, so they do not change with the seed"`.
  - The npz `probe_mask.npz` holds `f_peak`, `batch_masked`, `batch_total` and `span`. No figure is needed.
- **Keys:**
  - `mask_num_runs`: "the number of training batches to audit", default "12", flag `--num-runs`.
  - `mask_run_size`: "the rows per audited batch", default "32", flag `--run-size`.
  - Both are CONTROL None and join the no-control set.
- **Help.** `probes mask` (`help` and `description`: "why are training probes thrown out, over a prior?") has `add_config_flags(chi=False)`, then `add_name_flags` (directly after the configuration flags, as on every other subcommand), then:
  - `--prior REF`: required, "prior artifact to audit, by name or id; it is loaded, never built".
  - `--num-runs N` (value "12").
  - `--run-size N` (value "32").
  - `--chi-k-fixed K`: behaviour, "pooled over the training mixture".
  - `--seed N` (value "0"): "the random seed for the whole run; the probe layout keeps training's own fixed seed (default 0)".
  - `set_defaults(handler=_mask, chi_mode=True, chi_n_freqs=None, interrupt_note=PROBES_INTERRUPT_NOTE)`.
  - Add the `DEFAULT_CLASS` entries for `"probes mask"` and the leaf in `LEAVES`, in Task 11's format. Extend `_owned_defaults()` in `tests/test_tool.py` (not `owned_by_a_signature`) with `("probes mask", "--num-runs")`, `("probes mask", "--run-size")` and `("probes mask", "--seed")`, each `str(inspect.signature(core.diagnostics.probe_mask).parameters[<dest>].default)`; without them the help-default test fails on each pinned literal (`--num-runs` would resolve to train's key).
- **Docstring at `core/tool/diagnostics.py:3-6`.** No diagnostic that reads a posterior takes `--prior`; the one diagnostic with no posterior, `probes mask`, takes the prior it audits as its input and recorded parent.
- **Global rules:** keyword-only knobs with their defaults in the signature; `@public_entry`; logging only; no process labels (H4); fast tests use a stand-in generator, plus one tiny real run of the generator for the cross-check.

- [ ] **Step 1: Write the failing tests** in `tests/test_diagnostics.py`, with these helpers:

```python
def _mask_record(tag, lo, hi, *, n_points=5000, k_pad=12):
    """Rows lo..hi of a designed four-row batch: row 2 is too slow for the band's top, row 1 loses one
    probe to a shortened lock-in, row 3 loses one to the packer -- 5 of 12 probes masked in all."""
    from core.SBI.chi_probes import ProbeRecord
    f_peak = torch.tensor([0.02, 0.02, 0.0001, 0.02], dtype=torch.float64)[lo:hi]
    u = torch.log(torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)).expand(4, 3)[lo:hi]
    valid = torch.tensor([[1, 1, 1], [0, 1, 1], [0, 0, 0], [1, 1, 1]], dtype=torch.bool)[lo:hi]
    live = torch.tensor([3, 2, 0, 2])[lo:hi]
    return ProbeRecord(batch_tag=tag, lo=lo, hi=hi, f_peak=f_peak, duration_frac=torch.ones(3), k=3, u=u,
                       logcyc=torch.log(f_peak.unsqueeze(1) * u.exp() * n_points), valid=valid,
                       packed_mask=torch.arange(k_pad).unsqueeze(0) < live.unsqueeze(1),
                       dt_exp=1.0, n_points=n_points)


def _mask_generator(script, calls):
    """In place of pipeline.gen_training_data: per batch, the warnings an abandoned attempt left, then
    each committed range's own masked-probe warning, then the records, handed over at commit."""
    import warnings

    def _gen(*a, probe_observer=None, **kw):
        calls.append(kw)
        for tag, stale, records in script:
            for text in stale:
                warnings.warn(text)
            for rec in records:
                n = int((~rec.packed_mask[:, :rec.k]).sum())
                if n:
                    warnings.warn(f"{tag}: chi: {n}/{(rec.hi - rec.lo) * rec.k} probes masked (below 2.0 drive cycles).")
            for rec in records:
                probe_observer(rec)
        return torch.zeros(1, 1), torch.zeros(1, 1)
    return _gen


def test_probes_mask_splits_the_thrown_out_probes_by_cause_over_the_committed_row_ranges(store, monkeypatch):
    from core import config
    from core.diagnostics import probe_mask
    from core.SBI import pipeline
    from tests._fixtures import _prior_artifact
    cfg = _nad_cfg(chi_mode=True)
    _prior_artifact(store, cfg, name="mp")
    prior = store.load_prior(cfg, "mp")
    t1, t2 = (f"training batch {k}/2 [t_scale=3.73, T=5000, n_fine=59000, N_points=5000, rows=4]" for k in (1, 2))
    script = [(t1, [], [_mask_record(t1, 0, 4)]),
              (t2, [f"{t2}: chi: 5/12 probes masked (an attempt the row halving abandoned)"],
               [_mask_record(t2, 0, 2), _mask_record(t2, 2, 4)])]
    calls = []
    monkeypatch.setattr(pipeline, "gen_training_data", _mask_generator(script, calls))
    d = probe_mask(cfg, prior, num_runs=2, run_size=4, name="mask1")
    r, kw = d.results, calls[0]
    assert kw["checkpoint"] is None and kw["chi_mode"] is True and kw.get("theta_transform") is None
    assert (kw["chi_f0"], kw["chi_freq_bounds"]) == (config.CHI_F0, config.CHI_FREQ_BOUNDS)
    assert (kw["n_runs"], kw["run_size"]) == (2, 4)
    assert store.list("simulation") == []
    assert (d.variant, d.manifest.parents, d.manifest.fingerprints["gmm"]) == ("mask", {"prior": prior.id}, prior.fingerprint)
    assert (r["probes"], r["live"], r["masked"]) == (24, 14, 10)
    c = r["causes"]
    assert (c["cycle_floor"]["count"], c["too_slow_at_band_top"]["count"],
            c["shortened_by_duration_draw"]["count"], c["non_finite_lock_in"]["count"]) == (8, 6, 2, 2)
    assert r["invariants"] == {"non_finite_frequency": 0, "at_or_above_nyquist": 0, "out_of_band": 0, "hold": True}
    assert r["per_batch"]["fractions"] == [pytest.approx(5 / 12)] * 2 and r["per_batch"]["sd"] == 0.0
    assert r["omega0"]["hz"][2] == pytest.approx(20.0)
    assert r["rows"] == {"total": 8, "zero_live": 2, "one_live": 0}
    assert r["span"]["values"][1] == pytest.approx(3.0) and r["span"]["single_probe_rows"] == 0
    assert r["dominant_cause"] == "too_slow_at_band_top"
    assert r["cross_check"] == {"row_ranges": 3, "mismatches": [], "abandoned_attempt_warnings": 1}
    assert "fixed probe-generator seed" in d.settings["probe_layout"]
```

Also write:
- `test_probes_mask_records_and_warns_a_count_the_production_warning_does_not_confirm`. Use one record `_mask_record(t1, 0, 4)` and a generator that warns `f"{t1}: chi: 4/12 probes masked (...)"` before calling the observer. Assert `mismatches == [{"batch_tag": t1, "lo": 0, "hi": 4, "audit": 5, "production": 4}]` and that a WARNING from `core.diagnostics.probes` contains "does not match".
- `test_the_mask_audit_agrees_with_the_production_warnings_on_real_training_rows`, real and tiny. Take `cfg = _nad_cfg(chi_mode=True)` with `master_weak.txt` loaded, and a fixed prior returning `cfg.ground_truth_tensor` rows. Call `pipeline.gen_training_data` directly on a 12,000-point grid with `steady_idx=500`, `run_size=4`, `n_runs=2`, `chi_k_pad=4`, `chi_f0=config.CHI_F0`, `chi_freq_bounds=config.CHI_FREQ_BOUNDS` and `probe_observer=records.append`, inside `seeded(0, cfg.hw.device)` and `probes._capture_masked_warnings() as warned`. Then run `res = probes._audit(cfg, records, warned)`. Assert that `res["probes"] > 0`, that there are no mismatches, that `res["masked"]` equals the sum of `(~r.packed_mask[:, :r.k]).sum()`, and that `cycle_floor + non_finite_lock_in + the three invariant counts == masked`.
- `test_probes_mask_refuses_before_the_writer_opens`, parametrized: `{"num_runs": 0}` → mask_num_runs; `{"run_size": 0}` → mask_run_size; `{"chi_k_fixed": 1}` and `{"chi_k_fixed": 13}` → chi_k_fixed; `{"seed": -1}` → probe_seed; `{"name": "taken"}` → name. Patch the generator to `pytest.fail`. The diagnostic directory set is unchanged. A companion test covers a non-chi config (field None) and `_nad_cfg(chi_mode=True, chi_freq_bounds=(0.1, 10.0))`: a Refusal with field None whose message contains "does not match config.py", with the generator never called.
- In `tests/test_tool.py`, `test_probes_mask_loads_its_prior_by_reference_and_forwards_every_knob`. Write `_prior_artifact(ArtifactStore(tmp_path / "A"), _nad_cfg(chi_mode=True), name="mp")` with `PRISM_ARTIFACTS` at `tmp_path / "A"`, and patch `"core.diagnostics.probe_mask"` with a recorder. Running with `--prior mp --num-runs 3 --run-size 5 --chi-k-fixed 6 --seed 4 --name m1` exits 0. Assert `set(kw) == {"name", "note", "fig_sink", "store", "num_runs", "run_size", "chi_k_fixed", "seed"}`, the values, `prior.name == "mp"` and `cfg.chi_mode is True`. `--prior nope` exits 1, and nothing is written under `priors/` beyond `mp`. `--cell x` exits 2. Widen the family test to modes `{"band", "mask"}`: `mask` lacks `--chi --no-chi --chi-k --cell --lengths --strengths --posterior`, and `mask --help` stays torch-free.
- The table edits: the two keys go into `TOOL_ONLY_KEYS`, the count (+2) and `owned_by_a_signature` (against `probe_mask`'s `num_runs` and `run_size`), the no-control set, `want`, `_leg_probe_mask` (`cfg = _nad_cfg(chi_mode=True)`, `pipeline.gen_training_data` patched to `_body_done`, and the call `probes.probe_mask(cfg, _prior_stub(), num_runs=0 if case == "refusal" else 1, run_size=1, store=_EntryStore(case))`), `_UNTOUCHED_LEGS["probe_mask"]` and `_REFUSAL_FIELDS["probe_mask"] = "mask_num_runs"`.
- [ ] **Step 2: Run them and see them fail.** `python -m pytest tests/test_diagnostics.py tests/test_tool.py tests/test_refusals.py -q -k "mask or probes_family"`. Expected: an `ImportError` for `probe_mask`, `KeyError`s for the new keys, and exit code 2 for the unknown `mask` mode.
- [ ] **Step 3: Implement the keys, `_capture_masked_warnings`, `_audit` and `probe_mask`** as the Binding specifies. `probe_mask` ends with `w.parents`, `w.fingerprints["gmm"]`, `w.config.update(settings)`, `w.body = {"diagnostic": "probes", "variant": "mask", "settings": ..., "results": ...}` and `return store.load_diagnostic(w.id)`.
- [ ] **Step 4: Implement the `mask` mode and `_mask`.** `_mask` runs `cfg, _ = config_args.build_cfg(args, load_gt=False)`, then `prior = store.load_prior(cfg, args.prior)`, then `report(diag.probe_mask(cfg, prior, ..., **knobs(args, "num_runs", "run_size", "chi_k_fixed", "seed")))`. Also narrow the docstring at `core/tool/diagnostics.py:3-6` and add the help-defaults entries.
- [ ] **Step 5: Run the new tests and see them pass** (Step 2's command; expected: all pass).
- [ ] **Step 6: Run the focused tests for every touched file:** `python -m pytest tests/test_diagnostics.py -q -m "not slow" -k "probe or mask"`; `python -m pytest tests/test_tool.py -q -k "probes or every_leaf_parser_has_a_description or every_value_flag or torch or reads_no_environment or print_call"`; `python -m pytest tests/test_refusals.py -q`; `python -m pytest tests/test_nav_and_gating.py -q -k window_control`; `python -m pytest tests/test_artifact_store.py -q -k "public_entr"`. Expected: all pass.
- [ ] **Step 7: Commit.** `git add core/diagnostics/probes.py core/diagnostics/__init__.py core/tool/probes.py core/tool/diagnostics.py core/tool/help_defaults.py core/refusals.py core/tool/fields.py core/gui/fields.py tests/test_diagnostics.py tests/test_tool.py tests/test_refusals.py tests/test_nav_and_gating.py tests/test_artifact_store.py`, then `git commit -m "probes mask: why training probes are thrown out, over a prior" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"`. The controller starts the review and the one-process fast gate (`python -m pytest -m "not slow" -q`, background, logged) at the same moment. The implementer does not run the gate.

---

### Task 24: `probes drive`

**Files:**
- Modify: `core/diagnostics/probes.py` (`probe_drive`, `@public_entry`, and the module-private `_DRIVE_STRENGTHS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0)`), `core/diagnostics/__init__.py` (re-export), `core/tool/probes.py` (the `drive` mode and `_drive`), `core/tool/help_defaults.py`, `core/refusals.py`, `core/tool/fields.py`, `core/gui/fields.py`
- Test: `tests/test_diagnostics.py` (reuses `_probe_cfg` and `_probe_stand_in` from the band task), `tests/test_tool.py`, `tests/test_refusals.py`, `tests/test_nav_and_gating.py:1607-1612`, `tests/test_artifact_store.py` (`want`, `_leg_probe_drive`, `_UNTOUCHED_LEGS`, `_REFUSAL_FIELDS`)

**Interfaces:**
- Consumes: T13 `derived.for_simulation`. T20 `recording_geometry`, `own_peak_ratio`, `_ensemble_power`. T22 `probes._simulate`, `PROBES_INTERRUPT_NOTE`, `add_config_flags(p, *, chi=False)`, `require_seed(..., key="probe_seed")` and the family parser. `feature_sets.assert_forced`. `SummaryStatistics` and `FEATURE_LABELS` (`core/SBI/statistics.py`).
- Produces (registry T24): `core.diagnostics.probe_drive(cfg, *, t_obs_s=5.0, repeats=16, detune=1.4, strengths=None, free_min=0.70, captured_max=0.10, peak_window=0.02, clarity_min=3.0, seed=0, name="", note="", fig_sink=None, store=None) -> LoadedDiagnostic`, and the field keys `drive_t_obs`, `drive_repeats`, `drive_detune`, `drive_strengths`, `free_min`, `captured_max`, `drive_peak_window`, `clarity_min`.

**Binding (read before starting):** spec §4.1a (the drive rows), §4.2 and §4.5 are binding. Restated:
- `python` below means `C:\Users\J\anaconda3\envs\biophys-env\python.exe` (in the Bash tool, `/c/Users/J/anaconda3/envs/biophys-env/python.exe`); the `python` on PATH is a different interpreter without torch or pytest. Commands written with backslash paths are PowerShell commands; in Bash use forward slashes.
- **Inputs:** `--bounds` with a Forcing section and `--cell`. The config is built with chi mode OFF (`set_defaults(chi_mode=False, chi_n_freqs=None)`) and declares no chi flags. The Forcing section is required through `feature_sets.assert_forced(cfg, "probes drive")`: a box without one also has no force scale (the spontaneous master box drops `f_scale`), so neither the amplitudes nor the suggested Forcing lines could be stated in the cell's force unit.
- **Stage order:**
  1. `resolve_store`, then `assert_name_free`.
  2. `assert_forced`, then `_ = cfg.ground_truth`.
  3. The rules:
     - `require_seed(seed, key="probe_seed")`
     - `require_positive("drive_t_obs", t_obs_s)`, and `recording_geometry(...).n_obs >= 1` (else `refuse("drive_t_obs", ...)`)
     - `require_at_least("drive_repeats", repeats, 2)`
     - `require_between("drive_peak_window", peak_window, 0, 1, open_lo=True, open_hi=True)`
     - `require_positive("drive_detune", detune)`, and `refuse("drive_detune", ...)` unless `abs(detune - 1) > peak_window`, so the window measures the cell's own rhythm and not the drive
     - `strengths`, when given: non-empty, each `require_positive("drive_strengths", s)`, and strictly increasing (else `refuse("drive_strengths", ...)`)
     - `require_between("free_min", free_min, 0, 1, open_lo=True)` and `require_between("captured_max", captured_max, 0, 1, open_lo=True, open_hi=True)`, then `refuse("captured_max", ...)` unless `captured_max < free_min`
     - `require_positive("clarity_min", clarity_min)`
  4. `create`, then `seeded`.
- **Algorithm (spec §4.5):**
  - Default strengths are `sorted(set(_DRIVE_STRENGTHS) | {cfg.chi_f0})`, so the configured chi drive is measured explicitly, never read off a neighbouring grid point.
  - Simulate the undriven ensemble at the cell with `_simulate`. The simulation vector comes from `derived.for_simulation` (tier-1 aware), and `f_scale_eff` is its `f_scale` column.
  - From `freqs, power = probe_math._ensemble_power(x0, dt)`: Ω₀ is the peak bin with DC excluded, in cell units and in Hz (`× cfg.get_unit_conversion_factor("s")`).
  - The per-trace peak median and spread come from `chi.peak_freq(x0, dt)`, in Hz.
  - Clarity = the power at that peak bin ÷ the median power with DC excluded. It is the single peak bin, "peak ÷ median power": a summed window would let pure noise pass 3.
  - Peak-to-peak is the median over rows of max − min. Cycles in the recording = Ω₀ · n_obs · dt.
  - The drive frequency is `detune · Ω₀`.
  - For each strength `a`: `amp = a · f_scale_eff`, and simulate driven at the drive frequency (phase π/2). Measure:
    - the median |χ| from `chi.lock_in_batched(x_f, 2π f_drive, amp, n_obs·dt, dt)`
    - the median phase-locking value from `SummaryStatistics(x0, x_f, dt, amp, f_drive, π/2).compute_statistics()[:, FEATURE_LABELS.index("G6_plv")]`, which is reported and never judged
    - `ratio = own_peak_ratio(x_f, x0, Ω₀, peak_window, dt)`
  - Each strength's verdict is `"free-running"` when ratio ≥ free_min, `"captured"` when ratio ≤ captured_max, and `"in between"` otherwise. A non-finite ratio gives None.
  - **The strongest free-running strength is the largest in the leading unbroken run of free-running strengths, from the weakest up.** This deliberately replaces the earlier "strongest free-running below the first captured", so a strength whose own-peak bin refills at very large drives can never be picked. It is None, with a warning, when the weakest strength is not free-running.
  - **The weakest captured strength is the first captured one.** It is None, with a warning containing "nothing in the grid captured", when there is none. Both are reported in model units and in the cell's own force units (`a · f_scale_eff`, the derived scale on a tier-1 box).
  - The configured chi drive's row is `chi_f0_row` = `{"strength", "own_peak_ratio", "verdict", "note"}`, where the note says its thresholds depend on the detune, so this mode cannot certify the chi drive and `probes band` judges it. It is None when that strength is not in the grid.
  - `notes` states that free-running versus captured is the criterion and that there is no linearity test.
- **Results, not refusals** (known only after the spend). Each writes the record with every verdict null, logs one warning, and skips the driven sweep:
  - clarity below `clarity_min` ("no clear oscillation")
  - a drive at or above 0.9 × Nyquist (the warning names "Nyquist").
  - "Nothing captured within the grid" leaves the sweep in place and only the weakest captured strength null.
- **Suggested Forcing lines and box.** They are logged at info as `"[drive] suggested Forcing section ..."` and recorded as `results["suggested_forcing"]`; **never written** to any file, and no code builds a `Resources/` path.
  - `free_running` = `{"amp": strongest · f_scale_eff, "freq": f_drive, "phase": 0.0, "offset": 0.0}` or None.
  - `captured` has the same shape, with the weakest captured strength.
  - `box` = `{"amp": [0.0, 3 · captured amp], "freq": [Ω₀/10, Ω₀·10], "phase": [0.0, 6.283185307], "offset": [-50.0, 50.0]}`, or None when nothing is captured.
  - `settings["drive_phase"]` records the π/2 phase the drive was measured at.
- **Log lines** (pinned by tests): `f"[drive] strongest free-running strength {a:g} ({a * f_scale_eff:.4g} in the cell's force unit)"` and `f"[drive] weakest captured strength {a:g} (...)"`.
- **Record.**
  - `settings` = `{"t_obs_s", "repeats", "detune", "strengths", "free_min", "captured_max", "peak_window", "clarity_min", "seed", "drive_phase", "configured"}`, where `configured` is the band task's block read from the config.
  - `results` = `{"omega0": {"cell_units", "hz"}, "peak_per_trace_hz": {"median", "spread"}, "clarity", "oscillating", "peak_to_peak", "cycles_in_recording", "n_obs", "drive_frequency": {"cell_units", "hz", "detune"}, "force_scale", "strengths": [{"strength", "cell_force", "chi_median", "plv_median", "own_peak_ratio", "verdict"}], "strongest_free_running": {"strength", "cell_force"} | None, "weakest_captured": {...} | None, "chi_f0_row", "suggested_forcing", "notes"}`, with floats through `orch._num`.
  - The npz `probe_drive.npz` holds `strengths`, `own_peak_ratio`, `chi_median`, `plv_median`, and the undriven `freqs` and `power`.
  - One figure, `"Probe drive own-peak ratio"`: the ratio against strength on a log x axis, with the two thresholds drawn.
  - Parents `{}`.
- **Keys and help:**
  - FIELDS descriptions and defaults: `drive_t_obs` "the driven recording length, in seconds" "5.0"; `drive_repeats` "the number of noise repeats per drive strength" "16"; `drive_detune` "the drive frequency, as a multiple of the peak frequency" "1.4"; `drive_strengths` "the non-dimensional drive strengths" None; `free_min` "the own-peak power fraction that counts as free-running" "0.7"; `captured_max` "the own-peak power fraction that counts as captured" "0.1"; `drive_peak_window` "the own-peak window, as a fraction of the peak frequency" "0.02"; `clarity_min` "the smallest peak clarity that counts as an oscillation" "3.0".
  - FLAG: `--t-obs`, `--repeats`, `--detune`, `--strengths`, `--free-min`, `--captured-max`, `--peak-window`, `--clarity-min`. CONTROL None for every key, and every key joins the no-control set.
  - `probes drive` (`help` and `description`: "how hard can a lab drive this cell?") has `add_config_flags(chi=False)`, then `add_name_flags` (directly after the configuration flags, as on every other subcommand), then:
    - `--cell PATH`: required, "the cell file whose ground truth is driven".
    - `--t-obs S` (value "5.0").
    - `--repeats N` (value "16").
    - `--detune X` (value "1.4").
    - `--strengths X [X ...]`: behaviour, "sixteen from 0.01 to 20, plus the configured chi drive".
    - `--free-min X` (value "0.7").
    - `--captured-max X` (value "0.1").
    - `--peak-window X` (value "0.02").
    - `--clarity-min X` (value "3.0").
    - `--seed N` (value "0").
    - `set_defaults(handler=_drive, chi_mode=False, chi_n_freqs=None, interrupt_note=PROBES_INTERRUPT_NOTE)`.
    - The destination for `--t-obs` is `t_obs_s`.
  - Add the `LEAVES` and `DEFAULT_CLASS` entries in Task 11's format. Extend `_owned_defaults()` in `tests/test_tool.py` (not `owned_by_a_signature`) with one `("probes drive", "<flag>")` entry for each pinned literal: `--t-obs`, `--repeats`, `--detune`, `--free-min`, `--captured-max`, `--peak-window`, `--clarity-min`, `--seed`, each `str(inspect.signature(core.diagnostics.probe_drive).parameters[<dest>].default)` (`--t-obs`'s dest is `t_obs_s`).
- **Parity first (spec §4.5).** The 0.02 default window is accepted only after the controller's parity run on the card (last step) reproduces the earlier readings. The default is committed with the code (Step 7) because the parity run passes `--peak-window 0.018` explicitly; it counts as unaccepted until Step 8 passes, and the controller's ledger deviation row records that order (spec §4.5 says the parity comes before the default is set).
- **Global rules:** keyword-only knobs; `@public_entry`; logging only; no process labels (H4); fast tests use the stand-in.

- [ ] **Step 1: Write the failing tests** in `tests/test_diagnostics.py`:

```python
def test_probes_drive_picks_the_leading_free_running_run_and_the_weakest_capture(store, monkeypatch, caplog):
    import numpy as np
    from core.diagnostics import probes, probe_drive
    refill = lambda a: torch.where(a < 0.1, 1.0, torch.where(a < 0.2, 0.5, torch.where(a < 5.0, 0.1, 1.0)))
    monkeypatch.setattr(probes, "_simulate", _probe_stand_in(own_factor=refill))   # own peak on bin 125 at 5 s
    d = probe_drive(_probe_cfg(chi=False), repeats=2, strengths=[0.01, 0.05, 0.1, 0.2, 1.0, 8.0], name="drive1")
    r = d.results
    assert [s["verdict"] for s in r["strengths"]] == ["free-running", "free-running", "in between",
                                                    "captured", "captured", "free-running"]
    assert r["strongest_free_running"]["strength"] == 0.05                    # never the refilled 8.0
    assert r["strongest_free_running"]["cell_force"] == pytest.approx(0.5)
    assert (r["weakest_captured"]["strength"], r["weakest_captured"]["cell_force"]) == (0.2, pytest.approx(2.0))
    assert r["omega0"]["hz"] == pytest.approx(25.0) and r["drive_frequency"]["hz"] == pytest.approx(35.0)
    assert r["oscillating"] is True and r["clarity"] > 3.0 and r["chi_f0_row"] is None
    sf = r["suggested_forcing"]
    assert sf["free_running"]["amp"] == pytest.approx(0.5) and sf["free_running"]["freq"] == pytest.approx(0.035)
    assert sf["captured"]["amp"] == pytest.approx(2.0)
    assert sf["box"]["amp"] == [0.0, pytest.approx(6.0)] and sf["box"]["freq"] == [pytest.approx(0.0025), pytest.approx(0.25)]
    assert np.load(d.path / "probe_drive.npz")["own_peak_ratio"].shape == (6,)
    said = [m.getMessage() for m in caplog.records if m.name == "core.diagnostics.probes" and m.levelname == "INFO"]
    assert any(m.startswith("[drive] strongest free-running strength 0.05") for m in said), said
    assert any(m.startswith("[drive] weakest captured strength 0.2") for m in said), said
    assert any(m.startswith("[drive] suggested Forcing section") for m in said), said
```

Also write:
- `test_probes_drive_measures_the_configured_chi_drive_explicitly_on_its_default_grid`: `probe_drive(_probe_cfg(chi=False), repeats=2)` with the default stand-in. Assert `settings["strengths"] == [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0]`, `chi_f0_row["strength"] == 0.15` with `"detune" in chi_f0_row["note"]`, some entry of `notes` containing `"no linearity test"`, and `settings["drive_phase"] == approx(math.pi / 2)`.
- `test_probes_drive_reports_what_it_cannot_judge_as_results_not_refusals`:
  - (a) `clarity_min=1e9`: `oscillating is False`, `seen == []` (no driven simulation), every verdict None, and both picks None.
  - (b) `strengths=[0.1, 0.5]`, which the default stand-in never captures: `weakest_captured is None` and `strongest_free_running["strength"] == 0.5`.
  - (c) `_probe_stand_in(f_own=0.4)`, `strengths=[0.1]`: `drive_frequency["cell_units"] == approx(0.56)` and the verdict is None.
  - WARNING records from `core.diagnostics.probes` contain "no clear oscillation", "nothing in the grid captured" and "Nyquist". All three records load back.
- `test_probes_drive_turns_non_finite_measures_into_null`: `_probe_stand_in(nan_driven=True)`, `strengths=[0.1]`. `own_peak_ratio` and the verdict are None, and the reloaded results equal the original.
- `test_probes_drive_refuses_before_the_writer_opens`, parametrized: `{"t_obs_s": 0.0}` and `{"t_obs_s": 1e-7}` → drive_t_obs; `{"repeats": 1}` → drive_repeats; `{"detune": 0.0}` and `{"detune": 1.01}` → drive_detune; `{"strengths": []}`, `{"strengths": [0.1, 0.05]}` and `{"strengths": [0.1, nan]}` → drive_strengths; `{"free_min": 1.2}` → free_min; `{"captured_max": 0.0}` and `{"captured_max": 0.8}` → captured_max; `{"peak_window": 0.0}` → drive_peak_window; `{"clarity_min": 0.0}` → clarity_min; `{"seed": -1}` → probe_seed; `{"name": "taken"}` → name. `_simulate` is patched to `pytest.fail`, and the diagnostic directory set is unchanged. A companion test on `_probe_cfg("master_spont.txt", "master_spont.txt", chi=False)` (no Forcing section) expects a Refusal with field None.
- In `tests/test_tool.py`, `test_probes_drive_builds_a_non_chi_config_with_the_cell_and_forwards_every_knob`. Every flag with a distinct value exits 0. Assert `set(kw) == {"name", "note", "fig_sink", "store", "t_obs_s", "repeats", "detune", "strengths", "free_min", "captured_max", "peak_window", "clarity_min", "seed"}`, the values, `cfg.chi_mode is False` and a loaded ground truth. A bare call forwards only the four framing keywords. Widen the family test to `{"band", "mask", "drive"}`: `drive` lacks `--chi --no-chi --chi-k --lengths --multipliers --prior --num-runs`, the modes share no knob flag they should not (for example `--strengths` only on drive, `--lengths` only on band, `--prior` only on mask), and `drive --help` stays torch-free.
- The table edits: 8 keys to `TOOL_ONLY_KEYS`, the count (+8), `owned_by_a_signature` (every value default against `probe_drive`), the no-control set, `want`, `_leg_probe_drive` (`_probe_cfg`-style `cfg = _nad_cfg()` with `master_spont.txt` loaded, `probes._simulate` patched to `_body_done`, and the call `probes.probe_drive(cfg, repeats=1 if case == "refusal" else 2, strengths=[0.1], store=_EntryStore(case))`), `_UNTOUCHED_LEGS["probe_drive"]` and `_REFUSAL_FIELDS["probe_drive"] = "drive_repeats"`.
- [ ] **Step 2: Run them and see them fail.** `python -m pytest tests/test_diagnostics.py tests/test_tool.py tests/test_refusals.py -q -k "drive or probes_family"`. Expected: an `ImportError` for `probe_drive`, `KeyError`s for the new keys, and exit code 2 for the unknown `drive` mode.
- [ ] **Step 3: Implement the keys and `probe_drive`** as the Binding specifies. It ends with `w.body = {"diagnostic": "probes", "variant": "drive", ...}` and `return store.load_diagnostic(w.id)`.
- [ ] **Step 4: Implement the `drive` mode and `_drive`.** `_drive` runs `cfg, _ = config_args.build_cfg(args, load_gt=True)`, then `report(diag.probe_drive(cfg, ..., **knobs(args, "t_obs_s", "repeats", "detune", "strengths", "free_min", "captured_max", "peak_window", "clarity_min", "seed")))`. Add the help-defaults entries.
- [ ] **Step 5: Run the new tests and see them pass** (Step 2's command; expected: all pass).
- [ ] **Step 6: Run the focused tests for every touched file:** `python -m pytest tests/test_diagnostics.py -q -m "not slow" -k "probe or drive or mask"`; `python -m pytest tests/test_tool.py -q -k "probes or every_leaf_parser_has_a_description or every_value_flag or torch or reads_no_environment or print_call"`; `python -m pytest tests/test_refusals.py -q`; `python -m pytest tests/test_nav_and_gating.py -q -k window_control`; `python -m pytest tests/test_artifact_store.py -q -k "public_entr"`. Expected: all pass.
- [ ] **Step 7: Commit.** `git add core/diagnostics/probes.py core/diagnostics/__init__.py core/tool/probes.py core/tool/help_defaults.py core/refusals.py core/tool/fields.py core/gui/fields.py tests/test_diagnostics.py tests/test_tool.py tests/test_refusals.py tests/test_nav_and_gating.py tests/test_artifact_store.py`, then `git commit -m "probes drive: free-running and captured drive strengths for a cell" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"`. The controller starts the review and the one-process fast gate (`python -m pytest -m "not slow" -q`, background, logged) at the same moment. The implementer does not run the gate.
- [ ] **Step 8 (controller, on the card, alone, after the fast gate is green): the parity run.** It checks that the drive check reproduces the earlier readings at the earlier window before the 0.02 default is accepted. From the repo root in PowerShell:

```powershell
$py = "C:\Users\J\anaconda3\envs\biophys-env\python.exe"
$S  = "C:\Users\J\AppData\Local\Temp\prism-piece6\drive_parity"
$env:PRISM_ARTIFACTS = $S
& $py -m core probes drive --bounds Resources/Bounds/nadrowski/master.txt --cell Resources/Cells/nadrowski/master_spont.txt --peak-window 0.018 --device cuda --name drive_parity
$LASTEXITCODE
Remove-Item Env:PRISM_ARTIFACTS
```

Pass: `$LASTEXITCODE` is 0, and the output shows `[drive] strongest free-running strength 0.02` and `[drive] weakest captured strength 0.2`. These are statistical matches, because the old solver and the old window differ in detail. Record Ω₀ (Hz), the clarity, both picks, the 0.15 row's ratio and verdict, and the wall time in the ledger `.superpowers/sdd/2026-09-25-documentation-and-retrain-readiness/progress.md`, then delete `$S`. If either pick differs, the 0.02 default is not accepted: stop, record the full strengths table in the ledger, and take it to the owner before Task 25.

---

### Task 25: The whole-code review, one fix dispatch, a scoped re-review

**Files:**
- Create (gitignored workspace, Markdown only, never `.py`): `.superpowers/sdd/2026-09-25-documentation-and-retrain-readiness/review/lens-correctness.md`, `lens-spec.md`, `lens-tests.md`, `lens-hazards.md`, `lens-reference.md`, one `verify-<lens>.md` per lens, `synthesis.md`, `rereview.md`; gate logs `.superpowers/sdd/2026-09-25-documentation-and-retrain-readiness/gates/t25-slow.log` and `t25-fast.log`
- Modify: only what the fix dispatch touches. `synthesis.md` names each fix's files and its test. Nothing else is edited in this task.

**Interfaces:**
- Consumes: every registry name T1–T24, checked against the code as it stands: T1 (`tests/test_source_hygiene.py` and `comment_only_check.py`), T11 (`default_clause`, `default_text`, `DEFAULT_CLASS`, `default_for`, `LEAVES`), T12 (`tier1_cfg`, `force_scale_spy`), T13 (`derived.for_simulation(cfg, params_nd, rescale) -> tuple[Tensor, dict[str, int]]`), T14 (`SimConfig.assumed_params`, `SimConfig.report_labels`, `describe_informativeness(info, *, assumed=())`), T15 (`CALIBRATION_TAG`, `calibration_seed(seed)`, `calibration_verdict(...)`, `validate_calibration(..., seed=None)`, `results["rank_verdict"]`), T16 (`"fisher_eigenvalues"`, `TruncationRegion.containment`, `body.training["truth_containment"]`), T17 (`config.tier1`, `config.assumed_params`, `config.simulated_f_scale`, `FORMAT = "training-rows/3"`, `units_sha256`), T18 (`_BatchProbeLedger`, `_ACTIVE_LEDGER`, `"chi_masked"`, the `"[chi] masked probes"` line, `smoke --hidden-features/--num-transforms`), T19 (`accepted` on the three records), T20 (`probe_math` helpers), T21 (`ProbeRecord`, `probe_observer`), T22–T24 (`probe_band`, `probe_mask`, `probe_drive` and their field keys).
- Produces: `<F>`, the fixed head whose fast gate passed. Task 26 runs the card on it. Also the parked list in `synthesis.md` (each finding with its reason and its cost if wrong), which Task 39 carries into STATE's open list, and any deviation the fixes make, which goes to the ledger and then into spec §11 at Task 39.

**Binding (read before starting):**
- Spec §8, "After part 4": independent readers for correctness, match to the spec, tests, the hazards no CPU suite reaches, and the reference rule. Each reader is followed by an adversarial verifier. Then one fix dispatch and a scoped re-review. The house end-of-piece shape is one fix dispatch and one scoped re-review. A second round is allowed only for a Critical found by the re-review, and it is recorded as a deviation.
- Lenses and verifiers are **read-only**. They edit nothing in the repository, run **no pytest** and nothing on the GPU. Five lenses run in parallel, so any pytest by them would break "never two pytest processes at once". They may run torch-free scripts, `python -m core <sub> --help` (torch-free by design), and CPU scripts that import torch. Those scripts live under `C:\Users\J\AppData\Local\Temp\prism-piece6\review\`, run with a timeout (a foreground torch check can hang the tool call), and set `PRISM_ARTIFACTS` to a scratch directory there.
- `python` means `C:\Users\J\anaconda3\envs\biophys-env\python.exe`. Never edit a source file while a pytest run is in progress. Only the controller runs pytest.
- The Review Focus, verbatim. Every item must have its test, and a lens names which one:
  1. A probe the cell's own peak pushes past the sampling limit (a fast cell, a high multiplier): `probes band` must report that point as masked, never crash or clamp.
  2. A resume from a checkpoint written before this piece (no eigenvalue key, no masked-count list in `state.pt`): the run must resume, record the eigenvalues as unknown and say the masked total covers only this process.
  3. The same `validate --seed` twice: identical calibration sets and identical verdicts. `validate` with no seed inside `smoke` leaves smoke's seeded stream untouched.
  4. A flag added later without a help-defaults entry: the help-walking test must fail naming the subcommand and the flag, not pass silently.
  5. A legitimate new token shaped like a label (a future feature id, a paper section): the scan's failure message must name the file, line, family and the allowlist to extend.
- Severity:
  - **Critical**: a wrong science number, lost data, a crash on the retrain's path (tier-1 box, chi, 256 × 10, resume, narrowing round), a GPU-only failure, or a broken standing refusal (D11 for what trains or infers; D12).
  - **Important**: a spec deviation nobody recorded; a behaviour change without a test that failed first; an unpinned Review Focus item; a working-document reference in shippable text; a hazard on the card's path.
  - **Minor**: wording, cosmetics, a cheaper test.
- **Fix-dispatch rule.** One implementer dispatch, working sequentially, takes every Critical, every Important and each cheap Minor (a few lines, no new behaviour).
  - Each behaviour fix is test-first: write the failing test, run it and see it fail, fix, run it and see it pass.
  - One commit per logical fix: `git add` each file explicitly, a short subject, the body line `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`, never amended.
  - The implementer runs only the focused tests of the files it touched.
  - Everything it writes into `core/`, `tests/` or the prose files obeys the reference rule: no piece numbers, "§N", decision/ruling/review ids, "Task N", trap ids, walkthrough row ids, or citations of the handoff, STATE, CLAUDE.md, specs or plans; the reason is stated in words.
  - Every global rule still holds: knobs are keyword arguments; `@public_entry` copies the config; a pre-spend refusal is a `Refusal` with a registered key; messages are `logging` records; no `Resources/`/`Artifacts/` literal outside `core/config.py`; D11/D12 unchanged; nothing new is remembered by the window; no `.py` under `.superpowers/`.
- **Gate (the owner's rule).** The one-process fast gate on the fixed head starts at the same moment as the scoped re-review. The target is 16 minutes; the baseline is 181 warnings at piece 5, plus any movement the ledger accepted since.

- [ ] **Step 1: Fix the range and check the preconditions.**
  - The ledger (`.superpowers/sdd/2026-09-25-documentation-and-retrain-readiness/progress.md`) shows Tasks 1–24 each reviewed, with a green one-process gate. Task 24's parity numbers are recorded: 0.02 free-running and 0.2 captured at window 0.018.
  - `git status --porcelain` prints nothing.
  - Record `<P>`, the plan's commit (the first line of the ledger), and `<H>` = `git rev-parse HEAD`.
  - Save `git log --oneline <P>..<H>` and `git diff --stat <P>..<H>` into `synthesis.md`'s header.

- [ ] **Step 2: Start the slow set as an early reading, in the background.** From the repository root, Bash tool, `run_in_background: true`:

  ```bash
  cd /c/Users/J/PycharmProjects/PRISM && "/c/Users/J/anaconda3/envs/biophys-env/python.exe" -m pytest -m slow -q --durations=5 > .superpowers/sdd/2026-09-25-documentation-and-retrain-readiness/gates/t25-slow.log 2>&1; echo "exit=$?" >> .superpowers/sdd/2026-09-25-documentation-and-retrain-readiness/gates/t25-slow.log
  ```

  It takes about 45 minutes: the three existing slow tests plus the tier-1 tiny `smoke` and the probes twin-cell check. It is not the gate of record (Task 39's is). **The fix dispatch (Step 6) does not start until this exits.** A failure here is a Critical finding.

- [ ] **Step 3: Dispatch the five lenses in parallel.** Each lens gets:
  - the range `<P>..<H>`, the spec path, the plan path and its own charter below;
  - the output format: numbered findings, each `[Critical|Important|Minor] file:line — the claim — failure scenario (concrete inputs → wrong output or crash) — evidence (the quoted code)`, written to `review/lens-<name>.md`.

  The charters:
  - **correctness.** Every new path against its own contract:
    - `for_simulation` and the two builders' `RuntimeError`;
    - `calibration_verdict`: threshold `alpha / len(names)`, TARP `ks_p >= 0.05`, assumed parameters included, and a FAIL exits 0;
    - the `calibration_seed` stream;
    - the eigenvalue header read with `.get`, and a narrowing child taking its parent's eigenvalues while `fisher_m`/`dz`/`points` stay None;
    - `containment` computed in inferred coordinates;
    - the tier-1 block and the identity fail-open for a stub config;
    - `_BatchProbeLedger`: an abandoned attempt discarded; the completed sub-ranges of a halved batch merged under one tag; the commit placed after `x_buf[_lo:_hi] = _rows_out`; `_ACTIVE_LEDGER` cleared on every exit path;
    - the observer handing records only at `commit`;
    - the probes verdict logic: masked, not clamped, at or above 0.9 × Nyquist (Review Focus 1); the leading-run rule; null weakest-captured with a warning; the harmonic flag; `_num` making non-finite values null;
    - the refusal order: store resolved, then `assert_name_free`, then the rules, before `store.create`;
    - the `accepted` union on an inference;
    - `default_for`'s precedence;
    - Review Focus 1, 2, 3.
  - **spec.** Walk spec §2–§5 and §4 section by section against the diff:
    - every registry name, key, default and message the spec pins;
    - the §4.1a table (flags, keys, class-(i) defaults 24/0.20/3.0/0.50/0.10/12/32/5.0/16/1.4/0.70/0.10/0.02/3.0; `probe_seed` 0);
    - §4.2's measure-only rule: grids never reach a `SimConfig` field or `gen_training_data`'s `chi_f0`/`chi_freq_bounds`; the family assigns no upper-case config name; every record states `CHI_F0`, `CHI_FREQ_BOUNDS`, `CHI_K_PAD`, `CHI_MIN_CYCLES`, `CHI_MAX_CYCLES` and K;
    - D11 narrowed only for `probes`; D12 untouched; the Validate tab gains no box and nothing new is remembered;
    - every new key in FIELDS, `core/tool/fields.py` FLAG, `core/gui/fields.py` CONTROL (None) and the pinned no-control set;
    - the public-entry set;
    - `drive`'s 0.02 default window accepted by the parity run Task 24's last step recorded in the ledger before this review started (the default is in the code from Task 24's commit, and the ledger's deviation row says so; a missing or failed parity record is the finding, not the commit order);
    - any departure from the spec that is not in the ledger is a finding.
  - **tests.**
    - Each behaviour change has a test the ledger shows failing first.
    - Assertions pin the spec's exact values: 26.909 s; `[0.03, 0.0646, 0.1392, 0.3, 0.6]`; `"training-rows/3"`; the digest recomputed.
    - For each Review Focus item, name the one-line revert its test would catch.
    - The byte-identical observer test covers chi, forced and spontaneous.
    - New fast tests use stand-ins: read each task's gate delta in the ledger, and flag any fast test that runs a real simulation longer than about a second.
    - The two real runs carry `@pytest.mark.slow`.
    - Test names and docstrings carry no labels; no test reads CLAUDE.md, STATE or the specs; `tests/_fixtures.py`'s scan roots are unchanged.
  - **hazards no CPU suite reaches.**
    - Where every new tensor lives: `ProbeRecord` fields buffered per row range must not hold device memory across OOM retries (detached and on the CPU before buffering). Also check the counts' host syncs inside the batch loop.
    - `chi_masked` saved with `state.pt`, with no print or log between steps 1 and 3 of a checkpoint save.
    - `calibration_seed` inside `core.rng.seeded` on CUDA.
    - The derived force scale's dtype and device on CUDA.
    - The probes' float64 spectra and default grids against 16 GB shared with the desktop.
    - The OOM path of `gen_training_data` with the ledger active.
    - `_ACTIVE_LEDGER` left set after an exception inside a window session's second run.
    - Windows file handles on the npz payloads.
    - Every `smoke` line the card will run.
  - **reference rule.**
    - Read the last gate log for `tests/test_source_hygiene.py`, which must be green with no pending list left.
    - Do the manual pass for unlabelled references the patterns miss, over every added line: `git diff <P>..<H> -U0 -- core tests conftest.py README.md requirements.txt pytest.ini run.bat run.sh` searched for "spec", "plan", "piece", "review", "ruling", "decision", "item", "list <n>", "the design", "handoff", "ledger".
    - Re-read every rewritten sentence at the colliding ids (D1, D3, D5, D6, C1/C2 against C-1…C-11, M1–M4 and M1b, X1–X5, P/S/F ids) against spec §2.3's collision rules.
    - The ten §2.4 messages carry no label, and each pinning test pins a distinctive phrase with an absence regex, not a literal.
    - Each allowlist entry is exact or narrow.
    - `Get-ChildItem -Recurse .superpowers -Filter *.py` is empty.
    - The comment-only checker's outputs recorded for Tasks 1–9 show only permitted differences.
    - Review Focus 5.

- [ ] **Step 4: Dispatch one adversarial verifier per lens report.**
  - Each verifier tries to refute every finding from the code: it quotes the lines, runs the allowed read-only checks, and marks each finding CONFIRMED, PLAUSIBLE or REJECTED with its evidence.
  - It may raise a severity or add one missed finding in its lens's area, with evidence.
  - Output goes to `review/verify-<lens>.md`.

- [ ] **Step 5: Synthesize and rank.**
  - The controller merges the ten reports, removes duplicates, and ranks everything CONFIRMED or PLAUSIBLE, Critical first.
  - It writes `synthesis.md` with the fix list and the parked list.
    - Each fix is self-contained: file and symbol, failure scenario, the test that must fail first and its assertion.
    - Each parked item has its reason and its cost if wrong.
  - A fix that changes what the spec says gets a ledger deviation row: where, what changed, why, cost if wrong.
  - Result: every Critical and Important is on the fix list or parked with a stated reason.

- [ ] **Step 6: One fix dispatch (only after Step 2's run has exited).**
  - Brief one implementer with the fix list and the Binding's fix-dispatch rule, verbatim.
  - It works through the list in order and makes one commit per logical fix.
  - For each behaviour fix it reports the failing run and then the passing run of its focused test.
  - Result: `git log --oneline <H>..HEAD` shows one commit per fix, each with the Co-Authored-By line.

- [ ] **Step 7: Scoped re-review and the fast gate, started at the same moment.**
  - Set `<F>` = `git rev-parse HEAD`.
  - Start the gate in the background (Bash tool, `run_in_background: true`):

    ```bash
    cd /c/Users/J/PycharmProjects/PRISM && "/c/Users/J/anaconda3/envs/biophys-env/python.exe" -m pytest -m "not slow" -q --durations=15 > .superpowers/sdd/2026-09-25-documentation-and-retrain-readiness/gates/t25-fast.log 2>&1; echo "exit=$?" >> .superpowers/sdd/2026-09-25-documentation-and-retrain-readiness/gates/t25-fast.log
    ```

  - Dispatch one re-reviewer on `<H>..<F>` and `synthesis.md` only. It checks four things: each fix resolves its finding; each fix's test would fail without it (named revert); no new defect was introduced; each parked reason is sound. Verdict: approve, or approve with minors parked, into `review/rereview.md`. A Critical from the re-review triggers one targeted fix and re-review, recorded as a deviation.
  - The gate passes when:
    - the log ends `exit=0` with 0 failed;
    - the passed count equals Task 24's gate plus the fix tests;
    - the warning count equals the ledger's last accepted count (any change is explained in the ledger);
    - the wall time is recorded against the 16-minute target;
    - afterwards `git status --porcelain` is empty, the real `Artifacts/` listing is unchanged, and no `sbi-logs/` exists.

- [ ] **Step 8: Close.**
  - The ledger records the counts (Critical, Important, Minor; fixed, parked), `<F>`, the slow-set early reading, and the gate result.
  - The parked list waits for Task 39.
  - No commit of its own: the fixes carry their own commits, and the ledger is gitignored.

---

### Task 26: The card run

**Files:**
- Modify: `CLAUDE.md`.
  - The Tests bullet beginning "A green suite does not certify the GPU path" (line ~67): "four command lines" becomes "five command lines" (line ~70).
  - The recipe block from `  ```powershell` (line ~73) through "the last result is in `docs/STATE.md`." (line ~100-101) is replaced as in Step 13.
- Modify: `docs/STATE.md`.
  - "Last gate": one new row after the row beginning "| display walkthrough, piece-5 rows E1–E17".
  - The paragraph beginning "**The GPU gate, as command lines.**" (line ~595) and the ````markdown fence under it (line ~600-630).
- Create (outside the repository): `C:\Users\J\AppData\Local\Temp\prism-piece6\card_read.py`, `C:\Users\J\AppData\Local\Temp\prism-piece6\recipe_equal.py`, and the scratch root `C:\Users\J\AppData\Local\Temp\prism-piece6\card\`, which is deleted at the end.

**Interfaces:**
- Consumes:
  - T18: `smoke --hidden-features N`, `--num-transforms N`, and the info line starting `"[chi] masked probes"`.
  - T15: the info line starting `"[verdict] "`, `body.results["verdict"]`, `body.results["seed"]`.
  - T16: `body.transform.fisher_eigenvalues` carried across a resume and into a narrowing child; the lines starting `"[tsnpe] truth "`; `body.training["truth_containment"]`.
  - T17: `config.tier1`, `config.assumed_params`, and `config.simulated_f_scale` on a simulated observation.
  - T19: `body.results["accepted"]` on a calibration and an inference, and `body.training["accepted"]` on a narrowing child.
  - T14: the "(assumed input)" mark.
  - T22–T24: `python -m core probes band|mask|drive`.
- Produces:
  - The card recipe block: reference-free, five `smoke` lines, the diagnostic card with `probes`. It runs from the line `` ```powershell `` through the line ending "Delete `$S` afterwards." Task 36 copies it verbatim into `docs/guide/testing.md`, and Task 37 compares the three copies.
  - `recipe_equal.py <file> <file> [<file>]`: exit 0 when the blocks are identical after removing their common indentation.
  - `card_read.py <store root>`.
  - `<K>`, the head the card ran on, which Task 39 reads.

**Binding (read before starting):**
- Spec §8, "The card run", legs 1–6, **exactly**: the four existing lines unchanged; run 4; its resume; the narrowing leg; the diagnostic card with the tier-1 jacobian; the three probe checks against spec §4.8.
- The run is alone on the card: no pytest, no other GPU job, and no source edit until the last leg exits. Read VRAM with `nvidia-smi --query-gpu=memory.used --format=csv`, never `torch.cuda.mem_get_info()`. Check `$LASTEXITCODE` after every command.
- The stage commands and `probes` have no `--store-root`. They follow `PRISM_ARTIFACTS`, which must be set in the same PowerShell call (shell state does not persist between calls) and removed at the end of it. `smoke` never follows `PRISM_ARTIFACTS`.
- Pass criteria, per leg, are in the steps.
  - A crash or an exit code other than the expected one is a defect: stop the card run, diagnose (superpowers:systematic-debugging), fix it test-first in a dispatch, record a ledger deviation if the spec changes, and re-run the affected legs from the top of their store.
  - A science reading that disagrees with spec §4.8, or the tier-1 jacobian's temperature column, is **recorded as a measurement**, not fixed. A §4.8 disagreement is put to the owner in plain words before Part 5 starts. The controller changes no setting.
- **The recipe block must be reference-free.** `docs/guide/testing.md` carries it verbatim and the source scan walks `docs/guide/`. The block today has four scan hits: `docs/STATE.md` three times and "piece-2" once (checked with spec §2.2's patterns). The replacement in Step 13 has none. The pointer to STATE moves to a sentence after the block. Spec §5.9 requires the pass criterion to name the `[chi] masked probes` line; run 3 is forced mode and has no probes. This is the only CLAUDE.md edit before Task 38. It also puts `probes` into the diagnostic card, because that paragraph sits inside the verbatim block (Task 38 only verifies it).
- Commit: `git add CLAUDE.md docs/STATE.md`, a short subject, the body line `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`, never amended.

- [ ] **Step 1: Preconditions and helpers.**
  - HEAD is Task 25's `<F>` and its fast gate passed. Record `<K>` = `git rev-parse HEAD`.
  - No pytest is running. `nvidia-smi --query-gpu=memory.used --format=csv` shows only the desktop's baseline.
  - Create `C:\Users\J\AppData\Local\Temp\prism-piece6\card\logs\`; the `card` root holds nothing else.
  - Write the two torch-free helpers.

  `recipe_equal.py`:

  ```python
  """Exit 0 when the card recipe block is the same text in every file given (after removing the block's
  common indentation); otherwise print a unified diff against the first file and exit 1."""
  import difflib, sys, textwrap
  from pathlib import Path

  START, END = "```powershell", "Delete `$S` afterwards."

  def block(path):
      lines = Path(path).read_text(encoding="utf-8").splitlines()
      i = next(n for n, s in enumerate(lines) if s.strip() == START)
      j = next(n for n in range(i, len(lines)) if lines[n].rstrip().endswith(END))
      return textwrap.dedent("\n".join(lines[i:j + 1])).splitlines()

  ref, bad = block(sys.argv[1]), 0
  for p in sys.argv[2:]:
      other = block(p)
      if other != ref:
          bad = 1
          print("\n".join(difflib.unified_diff(ref, other, sys.argv[1], p, lineterm="")))
  print("recipe blocks DIFFER" if bad else "recipe blocks identical")
  sys.exit(bad)
  ```

  `card_read.py` walks `<root>/{posteriors,observations,calibrations,inferences}/*/manifest.json` by modification time. It prints one line per record with:
  - for a posterior: `body.transform.param_keys`, `body.transform.fisher_eigenvalues`, `config.fisher_m/fisher_dz/fisher_points`, `body.training.resumed_from_batch`, `config.tier1`, `config.assumed_params`, `body.training.truth_containment`, `body.training.accepted`;
  - for an observation: `config.simulated_f_scale`;
  - for a calibration or an inference: `body.results.accepted`, `.seed` and `.verdict.passed`.

  Missing keys print as `None`.

- [ ] **Step 2: The prelude every card call starts with** (PowerShell tool, from the repository root; each smoke or diagnostic call with `run_in_background: true`):

  ```powershell
  $py = "C:\Users\J\anaconda3\envs\biophys-env\python.exe"
  $B  = "--bounds","Resources/Bounds/nadrowski/master.txt"
  $C  = "--cell","Resources/Cells/nadrowski/master_spont.txt"
  $T1 = "--bounds","Resources/Bounds/nadrowski/master_tier1.txt"
  $S  = "C:\Users\J\AppData\Local\Temp\prism-piece6\card"
  $L  = "$S\logs"
  ```

  Each command below is run as written, followed by `*> "$L\<leg>.log"; "exit=$LASTEXITCODE" | Set-Content -Encoding ascii "$L\<leg>.exit"`. Read both files after the call returns.

- [ ] **Step 3: Leg 1, run 1** (`<leg>` = `run1`):

  ```powershell
  & $py -m core smoke --chi --t-obs 4.5 @B @C --checkpoint --save --store-root "$S/smoke"
  ```

  Pass:
  - exit 0; no OOM line; no `Traceback`;
  - stage timings near piece 4's 92.3 / 198.5 / 22.3 / 85.9 s;
  - the training `[chi] masked probes` line, the one scoped to "every committed batch of the simulation cache", within 37 ± 12 % (the validate stage's calibration draw logs a second line scoped "this process only (no simulation cache)", which is not the criterion). Expected exactly 260 of 704 (36.9 %), as pieces 2–4 measured at seed 0. A different count means the generation path changed: explain it in the row, since it is not itself a failure.
  - The `simulations/` digest differs from `4d8022b100db`, because of the identity format bump. List it.

- [ ] **Step 4: Run 2** (`run2`):

  ```powershell
  & $py -m core smoke --chi --t-obs 4.5 @B @C --checkpoint --store-root "$S/smoke" --prior smoke_prior --stages prior,posterior --resume require
  ```

  Pass:
  - exit 0;
  - the log contains `Reusing the Fisher rotation stored with the training checkpoint` and `[checkpoint] resuming at batch 4/4`;
  - its `[chi] masked probes` line (run 2 stops after posterior, so it prints only the training one) equals run 1's training line (spec §5.9: a resume finding the cache complete reports the committed total).

- [ ] **Step 5: Run 2b** (`run2b`). First save `(Get-ChildItem "$S/smoke/simulations" -Directory).Name`.

  ```powershell
  & $py -m core smoke --chi --t-obs 4.5 @B @C --checkpoint --store-root "$S/smoke" --prior smoke_prior --stages prior,posterior --num-runs 2
  ```

  Pass:
  - exit 1;
  - the refusal names `n_runs` and ends `(--new-run)`;
  - no `[fisher]` line;
  - the directory listing is unchanged.

- [ ] **Step 6: Run 3** (`run3`):

  ```powershell
  & $py -m core smoke --no-chi --t-obs 4.5 @B --cell Resources/Cells/nadrowski/master_weak.txt --checkpoint --save --store-root "$S/smoke_chi0"
  ```

  Pass: exit 0; no OOM line; timings near piece 4's 100.4 / 63.9 / 15.0 / 19.3 s.

- [ ] **Step 7: Leg 2, run 4** (`run4`):

  ```powershell
  & $py -m core smoke --chi --t-obs 4.5 --bounds Resources/Bounds/nadrowski/master_tier1.txt --cell Resources/Cells/nadrowski/master_spont_tier1.txt --hidden-features 256 --num-transforms 10 --checkpoint --save --store-root "$S/smoke_t1"
  ```

  Pass:
  - exit 0; no OOM line;
  - both `[tier1]` lines;
  - `[cfg] rescale order: ['x_scale', 't_scale', 'T']`;
  - the training `[chi] masked probes` line (scoped to every committed batch of the simulation cache; the calibration's line is not the criterion) within 37 ± 12 %.

  Then `python C:\Users\J\AppData\Local\Temp\prism-piece6\card_read.py "$S/smoke_t1"` shows:
  - `smoke_posterior`'s `param_keys` ending `x_scale, t_scale, T`;
  - `config.tier1` present with `relation` `"f_scale = N * beta * k_B * T / x_scale"`, a `k_b_cell` and `T_range` `[280.0, 310.0]`;
  - `assumed_params` `["T"]`;
  - `fisher_eigenvalues` a list of 13 floats;
  - the observation's `simulated_f_scale` a positive float.

  The calibration's log (in `run4.log`) marks T "(assumed input)" and ends its stage with the `[verdict] ` line.

- [ ] **Step 8: Leg 3, run 4's resume** (`run4r`):

  ```powershell
  & $py -m core smoke --chi --t-obs 4.5 --bounds Resources/Bounds/nadrowski/master_tier1.txt --cell Resources/Cells/nadrowski/master_spont_tier1.txt --hidden-features 256 --num-transforms 10 --checkpoint --store-root "$S/smoke_t1" --prior smoke_prior --stages prior,posterior --resume require
  ```

  Pass:
  - exit 0; the reuse line and `[checkpoint] resuming at batch 4/4`; no `[fisher] averaged` line;
  - `card_read.py` shows the new `_unnamed__<id>` posterior with `fisher_eigenvalues` equal to `smoke_posterior`'s (not None), `fisher_m/dz/points` all None, and `resumed_from_batch` 4;
  - its `[chi] masked probes` line equals run 4's training line.

- [ ] **Step 9: Leg 4, the narrowing leg** (one PowerShell call per command, each with `run_in_background: true` and its own `<leg>` log: `t1-list`, `t1-tsnpe`, `t1-validate`, `t1-infer`; `validate` runs at the stage's default calibration size on a truncated posterior and may take longer than ten minutes; each call starts with the prelude plus `$env:PRISM_ARTIFACTS = "$S/smoke_t1"` and ends with `Remove-Item Env:PRISM_ARTIFACTS`):

  ```powershell
  & $py -m core artifacts list observation   # the one observation run 4's infer stage wrote: its id
  & $py -m core tsnpe --chi @T1 --posterior smoke_posterior --observation <id> --num-runs 4 --run-size 32 --max-epochs 5 --name t1_round
  & $py -m core validate --chi @T1 --posterior t1_round --accept-truncated
  & $py -m core infer --chi @T1 --posterior t1_round --cell Resources/Cells/nadrowski/master_spont_tier1.txt --t-obs 4.5 --accept-truncated --accept-other-observation
  ```

  Pass:
  - `artifacts list` exits 0 and lists exactly one observation.
  - `tsnpe` exits 0. Its log carries the `[tsnpe] region from observation` line and one `[tsnpe] truth ` line per truncated direction (an "outside" direction also keeps the GROUND TRUTH warning). It prints `[prism] posterior t1_round__<id>`.
  - `validate` exits 0 with one `[verdict] ` line. PASS or FAIL is a result at this size. The line lists all 13 parameters with T marked assumed. The calibration records a `seed`.
  - `infer` exits 0.
  - `card_read.py "$S/smoke_t1"` shows:
    - `t1_round`: `truth_containment` with one entry per truncated direction, `fisher_eigenvalues` equal to `smoke_posterior`'s, and `accepted` `[]` (its parent was loaded with no acceptance);
    - the new calibration: `accepted` `["truncated"]`;
    - the new inference: `accepted` `["truncated", "other_observation"]`.

- [ ] **Step 10: Leg 5, the diagnostic card** (each call with its own `PRISM_ARTIFACTS`, removed at its end):

  ```powershell
  $env:PRISM_ARTIFACTS = "$S/smoke"
  & $py -m core sbc --chi @B --posterior smoke_posterior --repeats 1 --n-cal 20
  & $py -m core identifiability jacobian --chi @B @C --t-obs 4.5
  & $py -m core ablation --chi @B --posterior smoke_posterior
  $env:PRISM_ARTIFACTS = "$S/smoke_chi0"
  & $py -m core identifiability laplace --no-chi @B --posterior smoke_posterior --cell Resources/Cells/nadrowski/master_weak.txt --t-obs 4.5
  $env:PRISM_ARTIFACTS = "$S/smoke_t1"
  & $py -m core identifiability jacobian --chi @T1 --cell Resources/Cells/nadrowski/master_spont_tier1.txt --t-obs 4.5
  ```

  Pass: each exits 0 and adds one `diagnostics/` record. Piece 2's times were 19 / 84 / 6 / 128 s. `sbc` records `rank_verdict`. From `artifacts show diagnostic <id>`, write the tier-1 jacobian's `T` column into the gate row as a measurement.

- [ ] **Step 11: Leg 6, the probe checks:**

  ```powershell
  $env:PRISM_ARTIFACTS = "$S/probes"
  & $py -m core probes band @B @C
  & $py -m core probes drive @B @C
  $env:PRISM_ARTIFACTS = "$S/smoke"
  & $py -m core probes mask @B --prior smoke_prior
  ```

  Pass: each exits 0 and writes one diagnostic record (`artifacts list diagnostic`, then `artifacts show diagnostic <id>`). Compare with spec §4.8, statistically:
  - band: the configured (0.03, 0.3) holds and the 0.6 control is captured;
  - drive: the strongest free-running strength is about 0.02 and the weakest captured about 0.2;
  - mask: the masked share is within 37 ± 12 %, the cycle floor is the only cause, the invariant checks are zero, and the per-row-range cross-check matches.

  Remove `PRISM_ARTIFACTS` afterwards.

- [ ] **Step 12: Write the gate row** into STATE's "Last gate" table, after the piece-5 walkthrough row, filling every `<…>` from the logs:

  `| **GPU card run, piece 6** (`python -m core smoke`, the five command lines of `CLAUDE.md`, run 4's resume, the narrowing leg, the diagnostic card and the probe checks, at `<K>`) | <date>, alone on the card: **run 1** chi `master_spont`: prior <s>, posterior <s>, validate <s>, infer <s>, exit 0 (piece 4: 92.3/198.5/22.3/85.9), `[chi] masked probes` <m> of <n> (<p> %), cache `<digest>`; **run 2** reuse line and `[checkpoint] resuming at batch 4/4`, exit 0, masked line equal to run 1's; **run 2b** exit 1, "<refusal text>", no `[fisher]` line, `simulations/` unchanged; **run 3** forced `master_weak`: <s>/<s>/<s>/<s>, exit 0; **run 4** tier-1, 256 × 10: <s>/<s>/<s>/<s>, exit 0, both `[tier1]` lines, masked <p> %, T in the keys, `config.tier1` and `assumed_params` ["T"], 13 eigenvalues, observation `simulated_f_scale` <v>; **run 4 resume** exit 0, eigenvalues carried, Fisher settings null; **narrowing leg** tsnpe <s> exit 0, containment <inside/outside per direction>, verdict <PASS/FAIL> (a result at this size), accepted [] / ["truncated"] / ["truncated", "other_observation"]; **diagnostic card** sbc <s>, jacobian <s>, ablation <s>, laplace <s>, tier-1 jacobian <s> with T column <values> (a measurement); **probes** band <verdict>, drive free-running <x> captured <y>, mask <p> % (<causes>). No OOM line and no Traceback in any run; <warning: lines seen>. Scratch stores deleted |`

- [ ] **Step 13: The recipe edits.**
  - In `CLAUDE.md`, change "four command lines" to "five command lines".
  - Replace the block from `  ```powershell` through "…the last result is in `docs/STATE.md`." with exactly the block below.
  - Add one new line after it, still indented two spaces: "  The last result, with its stage timings, is in `docs/STATE.md`'s gate table, which carries this block verbatim."
  - In STATE, replace the ````markdown fence's contents with the same block.
  - Change the paragraph above it: "keep the two copies identical" becomes "keep the copies identical (the documents review compares them)", and "the four `smoke` runs" becomes "the five `smoke` runs".

  The block:

  ````markdown
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
  ````

  Run `python C:\Users\J\AppData\Local\Temp\prism-piece6\recipe_equal.py CLAUDE.md docs/STATE.md`. Expected: `recipe blocks identical`, exit 0.

- [ ] **Step 14: Clean up and commit.**
  - `Remove-Item -Recurse -Force C:\Users\J\AppData\Local\Temp\prism-piece6\card`. Keep the two helpers: Task 37 uses `recipe_equal.py`.
  - The ledger records every leg's numbers and `<K>`.
  - `git add CLAUDE.md docs/STATE.md`, then `git commit -m "docs: card run of record; recipe gains the tier-1 line" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"`.
  - Only documents changed, so no fast gate is owed for this commit.
  - A code fix made during the card run (Step 11's defect rule) gets its own review and one-process fast gate, started at the same moment.

---

### Task 27: The guide's skeleton, the start page, getting started, and the link test

**Files:**
- Create: `tests/test_docs.py`
- Create: `docs/guide/README.md` and `docs/guide/getting-started.md` (both filled in this task)
- Create: `docs/guide/window.md`, `command-line.md`, `recordings.md`, `science.md`, `architecture.md`, `rules-and-traps.md`, `testing.md`, `retrain.md` (title, stamp and headings only; Tasks 28–36 fill them)
- Modify: `README.md` (after line 3, "The application (**PRISM**) is a PySide6 desktop GUI.": one sentence that links to `docs/guide/`)
- Modify: `tests/test_source_hygiene.py` (add `test_every_reader_page_is_scanned`)

**Interfaces:**
- Consumes: from Task 1, `READER_DOCS_DIR = "docs/guide"` and `scanned_files() -> list[Path]` in `tests/test_source_hygiene.py`.
- Produces:
  - In `tests/test_docs.py`: `REPO`, `GUIDE`, `PAGES`, `_prose_lines(text)`, `_slug(title)`, `_anchors(text)` and `_link_problems(md, root) -> list[str]`, plus three tests: `test_every_relative_link_in_the_guide_and_readme_resolves`, `test_every_guide_page_opens_with_its_commit_stamp` and `test_the_link_checker_catches_a_missing_file_and_a_missing_anchor`.
  - The stamp line `Checked against commit <hash>.` from the Docs registry.
  - The ten pages with the heading sets below. These are the anchors Tasks 28–36 keep and link to.
  - The command headings in `command-line.md`: `## <subcommand>` and `### <subcommand> <mode>`, written exactly as typed.

**Binding (read before starting):**
- `python` below means `C:\Users\J\anaconda3\envs\biophys-env\python.exe` (in the Bash tool, `/c/Users/J/anaconda3/envs/biophys-env/python.exe`); the `python` on PATH is a different interpreter without torch or pytest. Commands written with backslash paths are PowerShell commands; in Bash use forward slashes.
- Spec §6.1 and §6.2 rows 1–2 are binding; read them. The pages live in `docs/guide/`, and `docs/guide/README.md` is the start page (GitHub shows it when the folder is opened). The root README links to it. Write every page fresh from today's code. `PRISM_HANDOFF.md` (§0, §1.1 and §1.2 for this task) and `.superpowers/sdd/2026-09-25-documentation-and-retrain-readiness/recon/handoff-map.md` (Chunk A, the first three entries) are sources only. Every one of the handoff sections you use is stale; the map lists the corrections.
- **The page rules.** These bind every line under `docs/guide/`.
  - Cite code by module and function name in backticks (`core.artifacts.store.ArtifactStore`). Never cite a line number, and never use a line anchor.
  - Name summary features by their full label from `core.SBI.statistics.FEATURE_LABELS`.
  - Stamp every page. Line 1 is `# <Title>`. The next non-blank line is `Checked against commit <hash>.`, where `<hash>` is `git rev-parse --short HEAD`, taken before you check any facts. Part 5 changes no code, so the code at that hash is the code the page describes.
- **Never carry these:**
  - The three front ends and the prompt command-line interface.
  - "Launch from the repo root because paths follow the working directory."
  - "There is no pytest."
  - Generated outputs under `Resources/`.
  - A `scripts/` directory, or environment-variable recipes for scripts.
  - Test totals or suite sizes.
  - Deleted artifacts presented as present.
- **The reference scan.** `tests/test_source_hygiene.py` reads every line of every `docs/guide/*.md`, code blocks included.
  - Never write these words: handoff, STATE.md, CLAUDE.md, superpowers, sdd, progress.md, ledger, walkthrough, "clean break", "hardening programme", "one-flow design", "piece N", "this piece", "Task N", "Appendix A", ruling, "review's", "the review of", "fix round", or a capitalised "Backlog".
  - Never write these shapes: "§", a "section 2.3"-shaped number, "trap X5", "guardrail 3", "C-11", "row F1".
  - Never write a stand-alone capital letter followed by one to three digits (A1, B7, H6, M1, T1, X5, a PowerShell `$T1`). F0, D0 and L2 are the only exceptions.
  - When the scan fails, rewrite the line. Do not extend the allowlist.
- **The root README.** The scan reads every line of the root `README.md`. Its allowlist keeps the Apple-chip line "M1 / M2 / M3 / M4", so do not touch that line. An allowlist entry that stops matching fails the scan.
- **Links.**
  - Write inline relative links only; the checker refuses reference-style definitions.
  - Link targets must exist with exactly the file name's case. GitHub is case-sensitive, even though Windows is not.
  - An anchor must be the GitHub slug of a heading that exists: lower case, punctuation dropped, spaces turned into hyphens.
- **H4 applies to `tests/test_docs.py` too.** Test names, docstrings and strings carry no process label. Write "#nowhere" as the bad-anchor example, never a line-number anchor such as "#L" followed by a digit.
- **Heading sets.** Create exactly these headings, in this order; later tasks add `###`/`####` under them, never rename them:
  - `README.md` — `# The PRISM guide`; `## What PRISM is`; `## Two front ends`; `## Reading paths` with `### For the owner-scientist`, `### For a lab member bringing recordings`, `### For a successor`, `### For a reviewer`; `## The pages`.
  - `getting-started.md` — `# Getting started`; `## Launching PRISM`; `## Inputs and records`; `## A first run in the window`; `## A first run on the command line`; `## The artifact store`; `## Browsing the store`.
  - `window.md` — `# The window`; `## Screens`; `## The inference settings`; `## Settings that are not speed dials`; `## The Fisher settings on a resumed run`; `## What the window remembers`; `## Supported models`.
  - `command-line.md` — `# The command-line tool`; `## Conventions`; `## Shared flags` with `### Configuration flags`, `### Naming flags`, `### Training-cache flags`, `### Acceptance flags`; `## Environment variables`; `## Exit codes`; then one `## <sub>` per subcommand and one `### <sub> <mode>` per mode, taken from the parser (`python -m core --help` and each family's `--help`): `## prior`, `## train`, `## validate`, `## infer`, `## tsnpe`, `## sbc`, `## identifiability` (+ `rotation`, `laplace`, `jacobian`), `## ablation`, `## probes` (+ `band`, `mask`, `drive`), `## smoke`, `## fdt`, `## crossval`, `## compare` (+ `cells`, `repeats`, `renormalise`, `sweeps`), `## artifacts` (+ `list`, `show`, `note`, `rm`, `sweep`, `summary`); `## Worked examples`; `## Where the old scripts went`. No non-command heading may begin with a lower-case subcommand word.
  - `recordings.md` — `# Bringing recordings`; `## Input files`; `## How a cell finds its bounds file`; `## Observation modes`; `## Recording files and drive frequencies`; `## Recording length`; `## The chi assumptions, and checking them`; `## Defining a model of your own`.
  - `science.md` — `# The science behind the settings`; `## Nondimensionalisation`; `## Conditioning features`; `## Chi probe design`; `## What chi buys`; `## The prior`; `## The tier-1 constraint and temperature`; `## Reading calibration honestly`; `## Narrowing rounds`; `## Identifiability limits`; `## The August 2026 retrain`; `## The solver's physics check`; `## Open questions`.
  - `architecture.md` — `# Architecture`; `## Module map`; `## Stages and compositions`; `## The artifact store`; `## The command-line tool`; `## The window`; `## The solver, CUDA graphs and reproducibility`; `## Memory planning and out-of-memory recovery`; `## The FDT pipeline`.
  - `rules-and-traps.md` — `# Rules and traps`; `## Rules the code keeps`; `## Narrowing-round safety rules`; `## Traps`.
  - `testing.md` — `# Testing`; `## Environment and interpreter`; `## Environment variables`; `## Test gates and markers`; `## The card smoke gate`; `## The diagnostic card`; `## What a green suite does not certify`; `## Writing tests`.
  - `retrain.md` — `# The retrain runbook`; `## Decisions`; `## Pre-flight`; `## The run`; `## Resuming`; `## What to watch`; `## Gates`; `## The certification round`; `## Afterwards`.
  - Under each heading of a skeleton page, put the single line `To be written.` until its task fills it.

- [ ] **Step 1: Write the failing link and stamp tests.** Create `tests/test_docs.py` with this module (the shared helpers are used by Task 29 too):

```python
"""The reader pages under docs/guide, and the root README: every relative link resolves, every page
carries the commit it was checked against, and the command-line page matches the tool's parser.

Nothing here imports torch: the parser and its default table are torch-free by design.
"""
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GUIDE = REPO / "docs" / "guide"
PAGES = ("README.md", "getting-started.md", "window.md", "command-line.md", "recordings.md",
         "science.md", "architecture.md", "rules-and-traps.md", "testing.md", "retrain.md")

_FENCE = re.compile(r"^\s*(```|~~~)")
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_CODE_SPAN = re.compile(r"`[^`\n]*`")
_LINK = re.compile(r"\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
_REF_DEF = re.compile(r"^\s{0,3}\[[^\]]+\]:\s*\S")
_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
_STAMP = re.compile(r"^Checked against commit [0-9a-f]{7,40}\.$")


def _prose_lines(text):
    """(line number, line) for every line outside a fenced code block."""
    in_fence = False
    for n, line in enumerate(text.splitlines(), 1):
        if _FENCE.match(line):
            in_fence = not in_fence
            continue
        if not in_fence:
            yield n, line


def _slug(title):
    """The anchor GitHub gives a heading: code and link markup reduced to their text, lower case,
    everything but letters, digits, spaces, hyphens and underscores dropped, spaces to hyphens."""
    text = _CODE_SPAN.sub(lambda m: m.group(0)[1:-1], title)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"[^\w\- ]", "", text.strip().lower())
    return text.replace(" ", "-")


def _anchors(text):
    """Every heading anchor of a page; a repeated heading is numbered -1, -2, ... as GitHub does."""
    seen, out = {}, set()
    for _n, line in _prose_lines(text):
        m = _HEADING.match(line)
        if m:
            base = _slug(m.group(2))
            k = seen.get(base, 0)
            out.add(base if k == 0 else f"{base}-{k}")
            seen[base] = k + 1
    return out


def _rel(path, root):
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def _link_problems(md, root):
    """One line per link in ``md`` that does not resolve: an absolute path, a target outside ``root``,
    no such file (exact case), a folder with no README.md, an anchor no heading of the target carries,
    and a reference-style definition, which the pages do not use. Links to web addresses are skipped."""
    problems = []
    for n, line in _prose_lines(md.read_text(encoding="utf-8")):
        where = f"{_rel(md, root)}:{n}"
        if _REF_DEF.match(line):
            problems.append(f"{where}: reference-style link definition; write the link inline")
            continue
        for target in _LINK.findall(_CODE_SPAN.sub("", line)):
            if _SCHEME.match(target):
                continue
            path_part, _, anchor = target.partition("#")
            if path_part.startswith("/"):
                problems.append(f"{where}: {target}: an absolute path; write it relative to the page")
                continue
            dest = (md.parent / path_part).resolve() if path_part else md.resolve()
            if dest != root.resolve() and root.resolve() not in dest.parents:
                problems.append(f"{where}: {target}: points outside the repository")
                continue
            if not dest.exists() or dest.name not in {p.name for p in dest.parent.iterdir()}:
                problems.append(f"{where}: {target}: no such file")
                continue
            if dest.is_dir():
                dest = dest / "README.md"
                if not dest.is_file():
                    problems.append(f"{where}: {target}: a folder with no README.md")
                    continue
            if anchor and (dest.suffix != ".md"
                           or anchor not in _anchors(dest.read_text(encoding="utf-8"))):
                problems.append(f"{where}: {target}: no heading with the anchor #{anchor}")
    return problems


def test_every_relative_link_in_the_guide_and_readme_resolves():
    missing = [p for p in PAGES if not (GUIDE / p).is_file()]
    assert not missing, f"reader pages missing from docs/guide: {missing}"
    pages = [REPO / "README.md", *sorted(GUIDE.glob("*.md"))]
    problems = [p for md in pages for p in _link_problems(md, REPO)]
    assert not problems, "links that do not resolve:\n" + "\n".join(problems)


def test_every_guide_page_opens_with_its_commit_stamp():
    bad = []
    for name in PAGES:
        page = GUIDE / name
        lines = [l for l in page.read_text(encoding="utf-8").splitlines() if l.strip()] \
            if page.is_file() else []
        if len(lines) < 2 or not lines[0].startswith("# ") or not _STAMP.match(lines[1]):
            bad.append(name)
    assert not bad, ("each page opens with its '# Title' line, then 'Checked against commit "
                     f"<hash>.': {bad}")


def test_the_link_checker_catches_a_missing_file_and_a_missing_anchor(tmp_path):
    (tmp_path / "other.md").write_text("# Other\n\n## A section\n", encoding="utf-8")
    page = tmp_path / "page.md"
    page.write_text("# Page\n\n"
                    "[fine](other.md#a-section) [self](#page) [web](https://example.org)\n"
                    "[gone](missing.md) [typo](other.md#nowhere)\n"
                    "[ref]: other.md\n"
                    "`[in code](missing.md)`\n"
                    "```\n[in a fence](missing.md)\n```\n", encoding="utf-8")
    problems = _link_problems(page, tmp_path)
    assert len(problems) == 3, problems
    assert any("missing.md: no such file" in p for p in problems)
    assert any("#nowhere" in p for p in problems)
    assert any("reference-style" in p for p in problems)
```

- [ ] **Step 2: Run the three tests and see two of them fail.**
  - Command: `python -m pytest tests/test_docs.py -q`.
  - Expected result:
    - `test_every_relative_link_in_the_guide_and_readme_resolves` fails with "reader pages missing from docs/guide" and lists all ten pages.
    - The stamp test fails and lists all ten pages.
    - The checker's self-test passes. It proves the checker is not vacuous.

- [ ] **Step 3: Create the ten pages.** Each page gets its title, its stamp line and its headings, exactly as listed in the binding, with `To be written.` under each heading.

- [ ] **Step 4: Fill `docs/guide/README.md` (the start page). Sources: handoff §0 and §1.1, rewritten.**
  - **What PRISM is:** a research tool for the generalised fluctuation-dissipation theorem (GFDT) and for simulation-based inference of the parameters of a simulated inner-ear hair-bundle model.
  - **Two front ends:** the window (`python -m core.gui`, or `run.bat`/`run.sh`) and the flags-only tool (`python -m core <subcommand>`), both over one core. The window has a peer screen for browsing the store. Every setting in either front end reaches the same stage as a keyword argument.
  - **Reading paths:** one ordered list per reader. Each item is a link to a page or a section, plus one line on what the reader gets from it.
    - Owner-scientist: `science.md` in order; `retrain.md`; `window.md#the-inference-settings`; `window.md#settings-that-are-not-speed-dials`; `command-line.md#validate`, `#tsnpe`, `#probes`; `testing.md#the-card-smoke-gate`.
    - Lab member: `getting-started.md`; `recordings.md` in order; `command-line.md#probes`, then `#infer`; `window.md#what-the-window-remembers`, `#supported-models`; `science.md#chi-probe-design`, `#reading-calibration-honestly`.
    - Successor: `getting-started.md`; `architecture.md`; `rules-and-traps.md`; `testing.md`; `command-line.md#conventions`, `#exit-codes`.
    - Reviewer: `science.md#reading-calibration-honestly`, `#identifiability-limits`, `#the-august-2026-retrain`; `architecture.md#the-artifact-store`; `rules-and-traps.md#narrowing-round-safety-rules`; `retrain.md#gates`, `#afterwards`; `command-line.md#validate`, `#artifacts`.
    - The paths must let three readers finish a job (spec §6.7):
      - a lab member checks a preparation with `probes` and runs an inference on a recording;
      - a successor finds where a refusal is raised and why;
      - a reviewer reproduces a calibration from a record: the posterior ref plus the recorded seed, via `validate --seed`.
  - **The pages:** one line per page, with a link.

- [ ] **Step 5: Fill `docs/guide/getting-started.md`.** Verify each fact against the code named here.
  - **Launching PRISM.**
    - `run.bat` (Windows) and `bash run.sh` (macOS/Linux) set `KMP_DUPLICATE_LIB_OK=TRUE`. A direct launch sets it by hand. The reason: torch and MKL ship two OpenMP runtimes, and without it the first simulation aborts with OMP Error #15 and no traceback. Verify in `run.bat` and `run.sh`.
    - `python -m core` sets the variable itself and forces matplotlib's Agg backend before any torch import (`core/__main__.py`).
    - The working directory does not matter.
    - The interpreter is the conda environment from the root README's installation steps.
  - **Inputs and records.**
    - `Resources/` holds the hand-edited inputs: Bounds, Cells, Units and Models.
    - `Artifacts/<kind>/<name>__<id>/` holds everything generated.
    - `core.config` resolves both roots from its own location: `RESOURCES_ROOT`, and `artifacts_root()`.
    - `PRISM_RESOURCES` is read once, at import. `PRISM_ARTIFACTS` is read at every call.
    - `smoke` writes to `--store-root`, or to a fresh temporary root. `fdt` and `crossval` take `--store-root`. The `artifacts` family reads only `PRISM_ARTIFACTS`. Verify each in `core/tool/__init__.py` (`main`) and `core/tool/browse.py`.
    - To keep a first try away from the real store, point `PRISM_ARTIFACTS` at a scratch folder.
  - **A first run in the window.**
    - Home has five sections: `core.gui.screens.home_screen.SECTIONS`.
    - Parameter Inference is six tabs used in order: Config, Prior, Posterior, Validate, Infer, TSNPE.
    - Name the buttons exactly as the tab modules label them: `core/gui/panels/inference/prior_tab.py` has "Build / Load prior", `posterior_tab.py` has "Train / Load posterior", `validate_tab.py` has "Run calibration".
    - Say that a small budget is for a first look only, and link to `window.md` for the detail.
  - **A first run on the command line.**
    - The tool's own tiny end-to-end run: `python -m core smoke --chi --t-obs 4.5 --bounds Resources/Bounds/nadrowski/master.txt --cell Resources/Cells/nadrowski/master_spont.txt`. The `[smoke]` banner prints the temporary store it wrote. Why each flag is there:
      - `--bounds` is required on every inference subcommand.
      - `--chi` is needed because `config.CHI_MODE` is False.
      - `--t-obs 4.5` is needed because smoke's default of 1 s is below the chi recording-length threshold.
    - Then show the shape of a real chain, `prior` → `train` → `validate` → `infer`, and link to `command-line.md#worked-examples`.
    - Check every command line with `python -m core <sub> --help` (torch-free). Run no training in this task.
  - **The artifact store.**
    - The eight kinds are the keys of `core.artifacts.store.KIND_DIRS`, singular, with plural directories.
    - Each record directory holds:
      - `manifest.json`;
      - a `log.txt` of the run's records, stamped `HH:MM:SS info/warning/error`, for every kind but the simulation cache (`core/runs.py`);
      - `figures/` and its payload files.
    - The simulation cache holds the training rows. It is keyed by an identity digest, committed batch by batch, reused on a resume, and has no log.
    - Names and ids:
      - A record directory is `<name>__<id>`; an unnamed one is `_unnamed__<id>`.
      - A reference is a name or a bare id; `_unnamed__<id>` itself does not resolve.
      - A taken name is refused before anything is spent.
    - Saving is a rename. Loading refuses any mismatch it can verify: model, parameter order, box, mode, width.
    - Two acceptances exist, `core.artifacts.Accept(truncated, other_observation)`. Each use is recorded downstream: in an inference's, a calibration's and a narrowing round's records.
    - An `fdt` record is progressive: a cancel or a crash keeps the folder, marked unfinished.
  - **Browsing the store.**
    - The Artifacts screen: list, detail, note, delete (refused when something depends on the record), sweep.
    - `python -m core artifacts list [<kind>]`, `show <kind> <ref>`, `note <kind> <ref> --note TEXT`, `rm <kind> <ref>`, `sweep [<kind>] [--yes]`, `summary <kind> <ref> [--out PATH]`.
    - Verify the modes against `core/tool/browse.py` and `core/gui/screens/artifact_screen.py`.

- [ ] **Step 6: Link the guide from the root README.** In `README.md`, after the intro sentence at line 3, add one sentence with the link `[docs/guide/](docs/guide/)` saying the guide is where to start. Change nothing else. The installation steps and the Apple-chip line are untouched.

- [ ] **Step 7: Pin that the reader pages are scanned.** Add this test to `tests/test_source_hygiene.py`. Use the module's repository-root constant if it has one:

```python
def test_every_reader_page_is_scanned():
    root = Path(__file__).resolve().parents[1]
    pages = sorted((root / READER_DOCS_DIR).glob("*.md"))
    assert len(pages) >= 10, f"expected the ten reader pages under {READER_DOCS_DIR}, found {len(pages)}"
    scanned = {p.resolve() for p in scanned_files()}
    missing = [p.name for p in pages if p.resolve() not in scanned]
    assert not missing, f"reader pages the reference scan does not read: {missing}"
```

- [ ] **Step 8: Run the focused tests.**
  - Command: `python -m pytest tests/test_docs.py tests/test_source_hygiene.py -q`.
  - Expected: everything passes. A scan failure names `docs/guide/<page>.md:<line> <family> '<match>'`; rewrite that line and run again.

- [ ] **Step 9: Commit.**
  - `git add tests/test_docs.py tests/test_source_hygiene.py README.md docs/guide/README.md docs/guide/getting-started.md docs/guide/window.md docs/guide/command-line.md docs/guide/recordings.md docs/guide/science.md docs/guide/architecture.md docs/guide/rules-and-traps.md docs/guide/testing.md docs/guide/retrain.md`
  - `git commit -m "docs: the guide's skeleton, start page and getting started" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"`
  - Do not run the fast gate. The controller runs the one-process fast gate (`python -m pytest -m "not slow" -q`, in the background, logged) at the same moment as this task's review.

---

### Task 28: `window.md`

**Files:**
- Modify: `docs/guide/window.md` (fill the skeleton Task 27 created)

**Interfaces:**
- Consumes:
  - Task 27's headings: `## Screens`, `## The inference settings`, `## Settings that are not speed dials`, `## The Fisher settings on a resumed run`, `## What the window remembers`, `## Supported models`.
  - `core.gui.fields.CONTROL`, `core.tool.fields.FLAG`, `core.refusals.FIELDS` (all torch-free dicts).
  - From Task 14, `SimConfig.report_labels`, which gives the label "T (assumed input, K)".
  - From Task 15, the Validate tab's drawn seed.
  - From Task 16, the checkpoint header key `"fisher_eigenvalues"`.
- Produces: the anchors above, which the start page, `command-line.md` and `retrain.md` link to.

**Binding (read before starting):**
- `python` below means `C:\Users\J\anaconda3\envs\biophys-env\python.exe` (in the Bash tool, `/c/Users/J/anaconda3/envs/biophys-env/python.exe`); the `python` on PATH is a different interpreter without torch or pytest. Commands written with backslash paths are PowerShell commands; in Bash use forward slashes.
- Spec §6.2 row 3 is binding. The page holds:
  - each screen;
  - the inference settings table, with columns field, flag, argument, default, and what it really changes;
  - the five settings that are not speed dials;
  - the Fisher settings doing nothing on a resume;
  - what the window remembers, and why;
  - the supported-models table, with a command-line column, CrossVal and the Artifacts screen included.
- Check every default against `core/config.py` and the stage signature. The `FIELDS` defaults are already pinned against both by `tests/test_refusals.py`.
- Sources:
  - `PRISM_HANDOFF.md` §2.3 (lines 238–286) and §4.5 (lines 1335–1346).
  - The Appendix entries on the knobs: lines 4405–4432 (the prior picker), 4478–4507 (the VRAM ceiling), 4709–4757 (the sweep, clustering and flow knobs) and 5257–5308 (the training budget).
  - `recon/handoff-map.md`: the §2.3 and §4.5 entries, and the Chunk B notes on groups M and Q.
  - The knob table there maps 1:1 onto today's flags, but add the column for them yourself.
- **The page rules.** These bind every line under `docs/guide/`.
  - Cite code by module and function name, never by line number or line anchor.
  - Name summary features by their full label from `core.SBI.statistics.FEATURE_LABELS`.
  - Keep the stamp line (`Checked against commit <hash>.`) and update it to `git rev-parse --short HEAD`, taken before you check facts.
  - Keep every heading Task 27 created. Add `###` headings under them as needed.
- **Never carry:**
  - "five tabs"; there are six.
  - A Config-tab picker; the bounds picker is on the Prior tab.
  - The retired band restored from saved settings, as if it could still happen.
  - `FloatField.value()` advice; the panels read `value_or_none`.
- **The reference scan** reads every line.
  - Never write: handoff, STATE.md, CLAUDE.md, superpowers, sdd, ledger, walkthrough, clean break, "piece N", "Task N", "§", "section 2.3"-shaped numbers, ruling, "review's", "Appendix A", "trap X5", "guardrail 3", "C-11", "row F1".
  - No stand-alone capital letter followed by one to three digits. F0, D0 and L2 are the only exceptions. The handoff's trap names M1, M2b, Q4 and X10 must become words.
- **Links:** inline, relative, to files and heading anchors that exist. `tests/test_docs.py` checks them.

- [ ] **Step 1: Stamp the page and gather the table rows.**
  - Update the stamp.
  - With a throwaway script in the session's scratchpad (never under `.superpowers/`, never in the repo), print one row per key whose `CONTROL` entry names one of the tabs Config, Prior, Posterior, Validate, Infer or TSNPE. Each row carries the tab and label, `FLAG[key]`, `FIELDS[key].default` and `FIELDS[key].description`.
  - Add the keys the same stages take that have no window control: `checkpoint_every`, `resume`, `max_num_epochs`, `num_posterior_samples`, `n_samples`. Mark those rows "command line only".
  - The stage keyword is the key. Confirm each one appears in the owning signature: `core.orchestrator.build_prior`, `build_posterior`, `validate_calibration`, `tsnpe_round`, and the inference compositions `simulated_inference` / `experimental_inference`.

- [ ] **Step 2: Write `## Screens`.**
  - **The five sections** (`core.gui.screens.home_screen.SECTIONS`) and the tabs each one holds (`core.gui.main_window`).
  - **The six inference tabs and their gating** (`core.gui.screens.inference_screen`):
    - Validate needs the inference prior, not the forcing prior. Otherwise a model with no forcing could never validate.
    - TSNPE also needs a recorded observation.
  - **One run at a time, app-wide.** The controls lock while a run is active.
  - **The prior picker does not load a prior.** Train uses whichever prior "Build / Load prior" last loaded. Verify that this is still true in `prior_tab.py`.
  - **The dialogs:**
    - the Posterior tab's near-miss question ("Start a new run anyway");
    - the confirmation before loading a narrowed (TSNPE) posterior;
    - the Infer tab's "Run on a different observation" tick.
  - **What a run shows:**
    - Validate draws a fresh seed for each run and records it. There is no Seed box. The command line's `validate --seed N` repeats a calibration set.
    - Validate's log shows the one `[verdict]` record (a head line, one row per parameter with temperature marked "(assumed input)", the joint coverage row and the caveat), and the tab then closes with its own line naming the verdict and the seed it drew.
    - The Infer tab's summary and corner plot mark temperature "T (assumed input, K)".
    - The TSNPE tab's log reports truth containment for every truncated direction.
  - **The Artifacts, Settings and model-builder screens:** one line each.

- [ ] **Step 3: Write `## The inference settings`.**
  - One table from Step 1, with columns: tab — field — flag — argument — default — what it really changes.
  - For "what it really changes", use the tooltips in `core.gui.panels.inference.help_text.HELP` and the handoff's notes, rewritten.
  - Include the Config tab's rows:
    - the χ toggle and the probe count (`--chi`, `--chi-k`);
    - the read-only χ drive amplitude and band. These are displays of `core/config.py`, have no flag, and cannot be changed per run;
    - the VRAM ceiling. Verify these facts in `core/gui/panels/inference/config_tab.py`:
      - it is the one field that writes a module attribute, because the pipeline reads it live on every batch plan (`core.SBI.pipeline.vram_ceiling_gib`);
      - it is never remembered;
      - its command-line twin is the `PRISM_VRAM_CEILING_GIB` environment variable, not a flag.
  - State once that every field travels to its stage as a keyword argument. Assigning to a `core/config.py` constant would change nothing, because `core.orchestrator` binds the constants at import.

- [ ] **Step 4: Write `## Settings that are not speed dials`.** The five, each with its measured reason:
  - **Candidates per round.** The global sweep is bounded by iterations, so shrinking it makes the prior worse without making it faster: 527 s at 2048 against more than 70 minutes, unfinished, at 32.
  - **Stability duration.** It defines what "stable" means, so it moves the prior's support.
  - **Min cluster size and min samples.** HDBSCAN's label count is the GMM's component count. Measured: 5 gives 15 components, 60 gives 1.
  - **The (t_scale, T_obs) operating points.** They are t_scale's effective sample size in calibration. Lowering them is a different measurement, not a faster one.
  - **Max accepted sets.** It buys coverage of the stable region, not precision.
  - Check each against its tooltip in `help_text.HELP`.

- [ ] **Step 5: Write `## The Fisher settings on a resumed run`.**
  - A resumed run reuses the rotation stored with the training checkpoint and skips the Fisher step, because the rotation is not reproducible across processes.
  - The three Fisher settings are therefore recorded as not run.
  - The rotation's eigenvalues do carry over from the checkpoint header (`"fisher_eigenvalues"`). An older checkpoint reads them as unknown.
  - A narrowing round takes no Fisher settings at all: it reuses its parent's rotation.
  - Verify in `core.orchestrator.build_posterior` and `core.SBI.training_checkpoint`.

- [ ] **Step 6: Write `## What the window remembers`.** Verify against each panel's `save_settings`/`restore_settings` and `core/gui/settings.py`.
  - The inference tabs remember:
    - selections: the pickers, the mode, the three observation lengths, "Drive F₀ (N)" and the recording paths;
    - the training budget.
  - Every science setting opens at its `core/config.py` value on every launch.
  - A consent is never remembered.
  - A stale key left by an older build is ignored.
  - The TSNPE tab keeps only its observation and its two budget fields.
  - The Simulate, FDT and CrossVal panels remember their own number fields and pickers, with these exceptions:
    - CrossVal re-derives n_freqs and M_ensemble from its preset, and the grid ends from its cell;
    - the Seed box, the record name, the note and the "Compare saved …" controls are never remembered.
  - Say why in one line each:
    - A remembered seed would silently make every run a repeat.
    - A remembered name would be refused as taken at the next launch.
    - A remembered science constant once outlived its change in `core/config.py` and cost a multi-day retrain.

- [ ] **Step 7: Write `## Supported models`.**
  - A table with rows Live simulation, Parameter Inference, FDT analysis, Sweep study cross-validation, Reduction map and Artifacts.
  - Columns: built-in models (`core.config.VALID_MODELS`), user models, and the command-line equivalent.
  - Facts to verify:
    - Inference takes a user model only with no forcing and at least one ND parameter: `core.registry.is_sbi_user_model`. The tool refuses in `core.tool.config_args.make_cfg`.
    - FDT support is per model (`core.registry.fdt_support`): FDT drives the observable itself, so a user model needs additive, non-zero observable noise and no intrinsic forcing.
    - CrossVal and Reduction are Nadrowski only: `_MODEL` in `crossval_panel.py` and `reduction_panel.py`; `crossval` on the command line.
    - The Artifacts screen works for every model.
    - `probes` needs a model the inference path can build, and `probes drive` also needs a bounds file with a Forcing section.
    - The inference pipeline is tuned on the Nadrowski model; calibration is not pre-tuned for user models.

- [ ] **Step 8: Run the focused tests.**
  - Command: `python -m pytest tests/test_docs.py tests/test_source_hygiene.py -q`.
  - Expected: everything passes.

- [ ] **Step 9: Commit.**
  - `git add docs/guide/window.md`
  - `git commit -m "docs: the window page" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"`
  - Do not run the fast gate. The controller runs the one-process fast gate (`python -m pytest -m "not slow" -q`, in the background, logged) at the same moment as this task's review.

---

### Task 29: `command-line.md` and its test

**Files:**
- Modify: `docs/guide/command-line.md` (fill the skeleton)
- Modify/Test: `tests/test_docs.py` (add `_tree`, `_flags`, `_headed_sections`, `_fenced_lines`, and the two tests)

**Interfaces:**
- Consumes:
  - From Task 11:
    - `core.tool.help_defaults.default_for(leaf_path: str, action) -> str | None`, which returns the full `(default X)` clause the help ends with, or None. The leaf path is written as typed, e.g. `"identifiability laplace"`.
    - The table `DEFAULT_CLASS`.
  - `core.tool.build_parser()` with its `.subcommands` dict.
  - From Task 27: `GUIDE`, `_prose_lines`, `_HEADING`, `_CODE_SPAN`, `_FENCE`.
  - From Task 15: `validate --seed`.
  - From Task 18: smoke's `--hidden-features` and `--num-transforms`.
  - From Tasks 22–24: the `probes` family and its flags.
- Produces:
  - The tests `test_the_command_line_page_names_every_subcommand_mode_and_flag_and_no_other` and `test_the_command_line_page_states_each_default_as_the_help_does`.
  - The page's anchors: `#<subcommand>`, `#<subcommand>-<mode>`, `#configuration-flags`, `#worked-examples`, `#exit-codes`, …

**Binding (read before starting):**
- `python` below means `C:\Users\J\anaconda3\envs\biophys-env\python.exe` (in the Bash tool, `/c/Users/J/anaconda3/envs/biophys-env/python.exe`); the `python` on PATH is a different interpreter without torch or pytest. Commands written with backslash paths are PowerShell commands; in Bash use forward slashes.
- Spec §6.4 is binding. Write the page **from** the tidied help; do not generate it.
- **Page structure:**
  - The shared flag groups come once.
  - Then a heading per subcommand and mode, written exactly as typed (`## train`, `### probes band`). Under each heading: its flags, its outputs, its refusals and its exit codes.
  - The rules the parser cannot express.
  - Worked examples.
- **The entry format, which the test reads:**
  - Under each command heading, every flag the command takes is one list item on one line, beginning `- ` followed by the flag in backticks with its placeholder: ``- `--num-runs N` — …``.
  - A switch pair is one item: ``- `--chi` / `--no-chi` — …``.
  - An item for a shared flag may be terse (``- `--bounds PATH` — see [Configuration flags](#configuration-flags).``), but it must be there.
  - The item states exactly the `(default X)` clause the help ends with, or no clause when the help has none.
  - No other bullet in a command section may begin with a backticked `--flag`. Write "- With `--resume require`, …".
- **Headings:**
  - Only command headings begin with a lower-case subcommand word.
  - A family (`identifiability`, `probes`, `compare`, `artifacts`) is `##`. Its modes are `###`. A subcommand with no modes is `##`.
- **Flags must be real.** Every `--flag` in a code span or code block, and every `-m core <sub> [<mode>]` invocation, must be one the parser has.
- Keep the sections of Task 27: `## Conventions`, `## Shared flags` with its four `###`, `## Environment variables`, `## Exit codes`, `## Worked examples`, `## Where the old scripts went`.
- Sources:
  - `recon/cli-inventory.md` (the whole file; its per-subcommand sections are pre-tidy, so read the current `--help` instead);
  - `recon/design-runbook.md` §0, for the output conventions;
  - `core/tool/*.py`.
- **The page rules** (spec §6.1):
  - Cite code by module and function name, never by line number.
  - Stamp: `Checked against commit <hash>.`, the short HEAD taken before checking.
  - Keep Task 27's headings.
- **Never carry:**
  - the prompt command-line interface;
  - `scripts/*.py` invocations, apart from the one history paragraph, which names the old scripts only as history and says where each went;
  - environment-variable recipes.
- **The reference scan** reads every line, code blocks included.
  - Never write: handoff, STATE.md, CLAUDE.md, superpowers, sdd, ledger, walkthrough, clean break, "piece N", "Task N", "§", "section 2.3"-shaped numbers, ruling, "review's", "Appendix A", "trap X5", "guardrail 3", "C-11", "row F1".
  - No stand-alone capital letter followed by one to three digits. F0, D0 and L2 are the only exceptions.
  - H4 applies to the test code as well.
- **Behaviour change:** none. The page and its tests are new. Test-first: write the tests, see them fail on the skeleton, then write the page.

- [ ] **Step 1: Write the two failing tests.** Append to `tests/test_docs.py`:

```python
import argparse

from core.tool import build_parser, help_defaults

COMMAND_PAGE = GUIDE / "command-line.md"
_ENTRY = re.compile(r"^[-*] `(--[a-z0-9][a-z0-9-]*)[^`]*`")
_OPTION = re.compile(r"(?<![\w-])--[a-z][a-z0-9-]*")
_INVOCATION = re.compile(r"-m core ([a-z][a-z-]*)(?: ([a-z][a-z-]*))?")
_SWITCHES = (argparse._HelpAction, argparse._StoreTrueAction, argparse._StoreFalseAction,
             argparse._StoreConstAction, argparse._CountAction)


def _tree():
    """(leaves, families): {path: parser} for every command typed after `python -m core` -- a
    subcommand with no modes, or `<subcommand> <mode>` -- and {subcommand: {modes}} for a family."""
    leaves, families = {}, {}

    def walk(path, parser):
        subs = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)]
        if not subs:
            leaves[path] = parser
            return
        families[path] = set(subs[0].choices)
        for name, child in subs[0].choices.items():
            walk(f"{path} {name}", child)

    for name, parser in build_parser().subcommands.items():
        walk(name, parser)
    return leaves, families


def _flags(parser):
    """{long option string: action} for every flag of a command but -h/--help."""
    return {next(o for o in a.option_strings if o.startswith("--")): a for a in parser._actions
            if a.option_strings and not isinstance(a, argparse._HelpAction)}


def _headed_sections(text):
    """[(level, title, body lines)] for every heading of level 1 to 3 outside code fences; a body
    runs to the next such heading."""
    out = []
    for _n, line in _prose_lines(text):
        m = _HEADING.match(line)
        if m and len(m.group(1)) <= 3:
            out.append((len(m.group(1)), m.group(2).strip(), []))
        elif out:
            out[-1][2].append(line)
    return out


def _fenced_lines(text):
    in_fence = False
    for line in text.splitlines():
        if _FENCE.match(line):
            in_fence = not in_fence
        elif in_fence:
            yield line


def test_the_command_line_page_names_every_subcommand_mode_and_flag_and_no_other():
    text = COMMAND_PAGE.read_text(encoding="utf-8")
    leaves, families = _tree()
    tops = {path.split()[0] for path in leaves}
    heads = [(lvl, t, body) for lvl, t, body in _headed_sections(text) if t.split()[0] in tops]
    titles = [t for _l, t, _b in heads]
    problems = [f"heading '{t}' appears {titles.count(t)} times" for t in set(titles)
                if titles.count(t) > 1]
    level = {t: lvl for lvl, t, _b in heads}
    body = {t: b for _l, t, b in heads}
    problems += [f"missing '## {fam}'" for fam in families if level.get(fam) != 2]
    for leaf in leaves:
        want = 3 if " " in leaf else 2
        if level.get(leaf) != want:
            problems.append(f"missing '{'#' * want} {leaf}'")
    problems += [f"heading '{t}' names no subcommand or mode of the tool" for t in level
                 if t not in leaves and t not in families]
    for leaf, parser in leaves.items():
        listed = [m.group(1) for m in map(_ENTRY.match, body.get(leaf, [])) if m]
        flags = _flags(parser)
        problems += [f"{leaf}: {opt} listed {listed.count(opt)} times, want once"
                     for opt in flags if listed.count(opt) != 1]
        problems += [f"{leaf}: lists {opt}, which this command does not take"
                     for opt in sorted(set(listed) - set(flags))]
    known = {o for p in leaves.values() for a in p._actions for o in a.option_strings}
    code = "\n".join(_CODE_SPAN.findall(text)) + "\n" + "\n".join(_fenced_lines(text))
    problems += [f"the page names {opt}, which no command takes"
                 for opt in sorted(set(_OPTION.findall(code)) - known)]
    for m in _INVOCATION.finditer(code):
        sub, mode = m.group(1), m.group(2)
        if sub not in tops:
            problems.append(f"the page runs `{m.group(0)}`: no such subcommand")
        elif mode is not None and mode not in families.get(sub, set()):
            problems.append(f"the page runs `{m.group(0)}`: {sub} has no mode {mode}")
    assert not problems, "\n".join(problems)


def test_the_command_line_page_states_each_default_as_the_help_does():
    text = COMMAND_PAGE.read_text(encoding="utf-8")
    leaves, _families = _tree()
    body = {t: b for _l, t, b in _headed_sections(text)}
    problems = []
    for leaf, parser in leaves.items():
        entries = {m.group(1): line for line in body.get(leaf, []) if (m := _ENTRY.match(line))}
        for opt, action in _flags(parser).items():
            line = entries.get(opt)
            if line is None:
                continue                              # the naming test reports a missing entry
            want = None if isinstance(action, _SWITCHES) else help_defaults.default_for(leaf, action)
            stated = line.count("(default ")
            if want is None and stated:
                problems.append(f"{leaf} {opt}: the help states no default, the page does: {line!r}")
            elif want is not None and (stated != 1 or want.strip() not in line):
                problems.append(f"{leaf} {opt}: the help says {want.strip()!r}; the page: {line!r}")
    assert not problems, "\n".join(problems)
```

- [ ] **Step 2: Run both tests and see them fail.**
  - Command: `python -m pytest tests/test_docs.py -q -k command_line`.
  - Expected: both fail on the skeleton. The naming test lists "`<leaf>: --<flag> listed 0 times, want once`" for every flag. The default test passes vacuously, because it finds no entries. That is expected: it bites once the entries exist.
  - If `default_for` raises for any action the walk hands it, report that in the task report. Do not work around it.

- [ ] **Step 3: Write the shared sections.** Verify every fact against the code named here.
  - **Conventions.**
    - A reference is a name or a bare id; `_unnamed__<id>` does not resolve (`core.artifacts.store`).
    - The tool's framing lines (`[cfg]`, `[prism]`, `[smoke]`) are prints on stdout and are not in `log.txt`.
    - Stage messages are records: information on stdout; warnings and errors on stderr, prefixed `warning: ` / `error: ` (`core/tool/logging_console.py`).
    - A library's records go to stderr whatever their level.
    - `[cfg]` prints the parameter order on every run, because simulators bind parameters by position.
    - Every value flag is forwarded only when given, so the stage's own default applies. The one exception is `validate --seed`: when none is given the tool draws one and passes it, so the calibration record names the seed it ran with.
    - `--help` imports no torch.
  - **The four shared groups**, each flag with its meaning and default:
    - configuration: `--bounds`, `--model`, `--chi`/`--no-chi`, `--chi-k`, `--device`. `--bounds` is required and declares the parameter set, its order and so the mode. `--chi-k` does not reach training, which draws its own probe count per batch.
    - naming: `--name`, `--note`. The note is one line, at most `core.refusals.NOTE_MAX_CHARS` characters.
    - training cache: `--resume auto|require|never`, `--new-run`.
    - acceptance: `--accept-truncated`, `--accept-other-observation`.
  - **Environment variables:** the four in `core.tool.EPILOG`, with when each is read. Add `KMP_DUPLICATE_LIB_OK` (set by the entry itself) and `PYTORCH_CUDA_ALLOC_CONF` (set by default in `core/config.py`; an export wins).
  - **Exit codes** (`core.tool.main`):
    - `0`: success or `--help`.
    - `2`: argparse errors and `UsageError` combinations, printed as `prism <cmd>: usage: …`.
    - `1`: a refusal, printed as `prism <cmd>: refused: <message>`, followed in parentheses by the flag that answers it (for example `(--n-cal)`; never a placeholder such as a literal `--flag`, which the page's flag check rejects); also a bug, with its traceback and `*** FAILED ***`.
    - `130`: Ctrl-C, with each family's advice (`interrupt_note`, `_smoke_interrupt_advice`).

- [ ] **Step 4: Write every command section.**
  - Walk the leaves in parser order, reading `python -m core <leaf> --help` for each. Write the entries, then what the command writes, then its refusals.
  - Rules the parser cannot express, each to be verified in the code named:
    - **infer** (`config_args.recording_set` and `_infer`):
      - chi mode needs `--forced PATH@HZ` on every driven recording, plus `--f0-si`, and no `--drive`;
      - spontaneous mode takes `--spont` only;
      - forced mode takes exactly one `--forced` without `@HZ`, plus `--drive NAME=VALUE` for every forcing parameter;
      - `--cell` excludes `--forced`, `--drive` and `--f0-si`;
      - `--name` names only the inference, so the observation is referred to by the id its `[prism] observation` line prints.
    - **train and tsnpe:**
      - the near-miss refusal: one setting away is refused unless `--new-run`; two or more settings away gives a warning only;
      - `--resume require` with no cache is refused;
      - a round takes no Fisher flags;
      - a round does not inherit its parent's network size or batch count;
      - `--level` below 0.99 warns;
      - a narrowed parent on another observation is refused, with no escape;
      - the round reports truth containment.
    - **validate:**
      - the `[verdict]` line and `results.verdict`;
      - a FAIL still exits 0;
      - `--seed` repeats a calibration set, and one is drawn and recorded when it is not given;
      - `results.accepted`.
    - **sbc:** the rank-uniformity half per repeat (`results.rank_verdict`), and no joint coverage test.
    - **smoke:**
      - its own store root;
      - `--resume require|never` needs `--checkpoint`, and `--resume require` needs `--prior`;
      - a resume must use the same `--num-runs` and `--run-size`;
      - `--checkpoint` commits every `max(1, num_runs // 2)` batches;
      - the `[chi] masked probes` run-total lines: the training one (scoped to every committed batch of the simulation cache under `--checkpoint`) is the one to read; the calibration stage logs its own;
      - calibration at these sizes has no power.
    - **fdt:** `--skip-sanity` with `--no-production` is refused; the record is progressive; it runs on the CPU only; the model comes from the cell's folder unless `--model` is given.
    - **crossval:**
      - the grid rules: N a whole number of at least 2, MIN below MAX, and the T_a/T grid's MIN not below 0;
      - the two records `NAME-s` and `NAME-temp`;
      - the preset's defaults.
    - **compare:** `--record` is repeatable; an unfinished record is refused; runs are interpolated onto a common grid, with gaps left blank.
    - **artifacts:**
      - no configuration flags and no `--store-root`; the root is `PRISM_ARTIFACTS`;
      - `<kind>` is one of the eight;
      - `rm` is refused while dependents exist;
      - `sweep` is a dry run unless `--yes`.
    - **identifiability:**
      - rotation warns when the eigenvalues were not stored;
      - laplace and jacobian simulate at the cell's truth and need `--t-obs`;
      - jacobian prints the probe budget and the recording-length line;
      - on a tier-1 box both derive the force scale as training does.
    - **ablation:** it reads the posterior's own simulation cache, so the posterior must have trained with checkpointing on.
    - **probes:**
      - the grids are measurements, never settings;
      - every record states the configured band and drive it judged;
      - `band` and `mask` build in chi mode and take no chi flags; `drive` builds in non-chi mode;
      - `mask` takes a prior and writes no simulation record;
      - `drive` logs its suggested Forcing lines and never writes them;
      - "no clear oscillation", "nothing captured" and "at Nyquist" are results, not refusals.

- [ ] **Step 5: Write `## Worked examples` and `## Where the old scripts went`.**
  - Three examples, each in a fenced block:
    - a stage chain, `prior` → `train` → `validate` → `infer`, with small sizes and `PRISM_ARTIFACTS` at a scratch folder;
    - a probes check, `probes band` then `probes drive` on one cell;
    - an FDT run twice, then `compare repeats --record <a> --record <b>`.
  - One history paragraph on where the old scripts went:
    - the smoke train became `smoke`;
    - the SBC characterisation became `sbc`;
    - the three identifiability scripts became the `identifiability` modes;
    - channel ablation became `ablation`;
    - the band, mask and master-cell measurement scripts were archived and rebuilt as `probes band`, `mask` and `drive`;
    - the interactive prompt interface was retired.
  - Name the old scripts by file name only. Do not cite any document.

- [ ] **Step 6: Run both tests and see them pass.**
  - Command: `python -m pytest tests/test_docs.py -q -k command_line`.
  - Expected: `2 passed`. Fix the page, never the test, for any line they print.

- [ ] **Step 7: Run the focused tests.**
  - Command: `python -m pytest tests/test_docs.py tests/test_source_hygiene.py -q`.
  - Expected: everything passes.

- [ ] **Step 8: Commit.**
  - `git add docs/guide/command-line.md tests/test_docs.py`
  - `git commit -m "docs: the command-line reference and its parser test" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"`
  - Do not run the fast gate. The controller runs the one-process fast gate (`python -m pytest -m "not slow" -q`, in the background, logged) at the same moment as this task's review.

---

### Task 30: The README's command-line summary

**Files:**
- Modify: `README.md` (delete only the first sentence of line 155, "`python -m core --help` lists the command-line tool's subcommands.", keeping the rest of that line (the macOS/Linux reminder to activate the environment) under `## Running the app`, and add a new section `## The command-line tool` after that section)

**Interfaces:**
- Consumes:
  - From Task 29, `docs/guide/command-line.md`'s anchors: `#<subcommand>`, `#<subcommand>-<mode>`, `#shared-flags`, `#exit-codes`, `#worked-examples`.
  - From Task 27, the link test.
- Produces: none.

**Binding (read before starting):**
- `python` below means `C:\Users\J\anaconda3\envs\biophys-env\python.exe` (in the Bash tool, `/c/Users/J/anaconda3/envs/biophys-env/python.exe`); the `python` on PATH is a different interpreter without torch or pytest. Commands written with backslash paths are PowerShell commands; in Bash use forward slashes.
- Spec §6.3 is binding.
- **What stays untouched:**
  - Installation, every platform section, and the Apple-chip line "M1 / M2 / M3 / M4". The reference scan's allowlist keeps that line, and an entry that stops matching fails the scan.
  - The launcher table.
  - Task 27's link to `docs/guide/`.
- **The new section is about 80 lines**, containing:
  - what the tool is, and how it relates to the window;
  - one line per subcommand and mode;
  - the shared flag groups, once;
  - what each command writes, in store kinds;
  - the environment settings;
  - the exit codes;
  - three worked examples: a stage chain, a probes check, and an FDT run followed by a comparison;
  - a pointer to `docs/guide/command-line.md` and to `python -m core <sub> --help`.
- **Everything must be real.** Every subcommand, mode and flag you name must exist; check each with `--help`. This page is not parsed by Task 29's test, so you check it by hand.
- **The reference scan** reads every line of `README.md`.
  - Never write: handoff, STATE.md, CLAUDE.md, superpowers, walkthrough, clean break, "piece N", "Task N", "§", ruling, "Appendix A", "trap X5", "C-11", "row F1".
  - No new stand-alone capital letter followed by one to three digits. F0, D0 and L2 are the only exceptions.
- **The link test** (`tests/test_docs.py::test_every_relative_link_in_the_guide_and_readme_resolves`) reads the root README: every relative link and anchor must resolve.

- [ ] **Step 1: Write the section.**
  - A short paragraph:
    - two front ends over one core;
    - every flag reaches its stage as a keyword argument;
    - `--bounds` is required on the inference subcommands;
    - `--chi` is off by default.
  - A table with one row per command (all fourteen subcommands, each family's modes listed in its row): command | what it does | what it writes, in store kinds.
  - The shared groups in one line each.
  - The four environment variables.
  - The exit codes 0, 1, 2 and 130.
  - Three fenced examples, taken from `command-line.md#worked-examples` in shortened form.
  - The two pointers.

- [ ] **Step 2: Check every command line.**
  - Run `python -m core <sub> --help` for each subcommand and mode named. These are torch-free.
  - Confirm every flag in the examples appears in that help.

- [ ] **Step 3: Run the focused tests.**
  - Command: `python -m pytest tests/test_docs.py tests/test_source_hygiene.py -q`.
  - Expected: everything passes.

- [ ] **Step 4: Commit.**
  - `git add README.md`
  - `git commit -m "docs: README command-line summary" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"`
  - Do not run the fast gate. The controller runs the one-process fast gate (`python -m pytest -m "not slow" -q`, in the background, logged) at the same moment as this task's review.

---

### Task 31: `recordings.md`

**Files:**
- Modify: `docs/guide/recordings.md` (fill the skeleton)

**Interfaces:**
- Consumes:
  - Task 27's headings.
  - From Task 20: `core.diagnostics.probe_math.training_ceiling_s(cfg, t_scale) -> float`, which gives 26.909 s for the master cell at t_scale 3.73.
  - From Tasks 22–24: the `probes band|mask|drive` flags and their records.
  - From Task 29: `command-line.md#probes-band`, `#probes-mask`, `#probes-drive`, `#infer`.
- Produces: `recordings.md`'s anchors, used by the start page's lab-member path.

**Binding (read before starting):**
- `python` below means `C:\Users\J\anaconda3\envs\biophys-env\python.exe` (in the Bash tool, `/c/Users/J/anaconda3/envs/biophys-env/python.exe`); the `python` on PATH is a different interpreter without torch or pytest. Commands written with backslash paths are PowerShell commands; in Bash use forward slashes.
- Spec §6.2 row 5 is binding. The page is the lab-member guide. It covers:
  - the Bounds, Cells, Units and Models formats;
  - how a cell finds its bounds file;
  - the three observation modes and what each needs;
  - drive frequencies in Hz;
  - recording length against the training range;
  - the band, drive and masking assumptions, and how to check them with `probes`;
  - defining a user model.
- Sources:
  - `PRISM_HANDOFF.md` §3.3 (lines 355–372), §3.5 (lines 379–388) and §4.3 (lines 792–872);
  - the "T_obs ceiling" note (lines 6552–6560) and the 2026-08-05 consolidation (lines 6683–6689);
  - the U traps (lines 1726–1746);
  - `recon/handoff-map.md`: those entries, the Chunk G notes items 2 and 5, and the Chunk B trap list's U1–U6.
  - `recon/design-f0sweep.md`, `design-maskaudit.md` and `design-drive.md` for what each `probes` mode measures.
- **The page rules** (spec §6.1):
  - Cite code by module and function name, never by line number.
  - Name summary features by their full label.
  - Stamp: `Checked against commit <hash>.`, the short HEAD taken before checking.
  - Keep Task 27's headings.
- **Never carry:**
  - "one bounds file per cell";
  - the retired archive scripts as tools a lab can run; say `probes` replaced them;
  - a promise that the tool can change the band, drive or cycle limits per run. It cannot: a deliberate change is an edit of `core/config.py` followed by a retrain.
- **The reference scan** reads every line.
  - Never write: handoff, STATE.md, CLAUDE.md, superpowers, sdd, ledger, walkthrough, clean break, "piece N", "Task N", "§", "section 2.3"-shaped numbers, ruling, "review's", "Appendix A", "trap X5", "guardrail 3", "C-11", "row F1".
  - No stand-alone capital letter followed by one to three digits. F0, D0 and L2 are the only exceptions. The trap names U1–U6 must become words.
- **Links:** they must resolve (`tests/test_docs.py`).

- [ ] **Step 1: Stamp the page. Write `## Input files`.** Read the real files as examples: `Resources/Bounds/nadrowski/master.txt`, `master_tier1.txt`, `Resources/Cells/nadrowski/master_spont.txt`, `Resources/Units/nadrowski/units.txt`, `Resources/Models/SHM.json`.
  - **Bounds files:**
    - the section headers `# Non-dimensional Parameters`, `# Dimensional Parameters`, `# Forcing Parameters`;
    - the line `name in (lo, hi)`;
    - the bounds file declares which parameters are inferred, in what order, and so the observation mode: a Forcing section means forced or chi; none means spontaneous.
  - **Cells:**
    - `name = value`, plus the initial-conditions section;
    - `core.sim_config` / `cli.load_and_validate_gt` refuses a cell missing a declared value, naming it, and ignores extra values, which the tool's `[cfg]` line prints. That lets one cell serve several bounds files.
  - **Units:**
    - `# Units` then tokens such as `nm ms pN kHz`; these declare what the numbers in the bounds and cell files are written in;
    - the frequency token must be the reciprocal of the time token, or `SimConfig.check_unit_consistency` warns.
  - **Models:** a user model's JSON in `Resources/Models/`.
  - A cell's model is its parent folder.

- [ ] **Step 2: Write `## How a cell finds its bounds file`.**
  - `cli.resolve_bounds_for_cell` uses the same-named sibling, else `Bounds/<model>/master.txt`. It is used by the FDT analyses, the parameter sweep and the Live simulation (`cli.parse_cell`, `simulate_runner`).
  - Every inference subcommand takes `--bounds` explicitly, and the window's Prior tab picks the bounds file.
  - The tier-1 cell `master_spont_tier1.txt` is for the inference path with `master_tier1.txt`. The FDT path resolves it to `master.txt` and refuses it for its missing `f_scale`; verify in `cli._merge_vals_bounds`.

- [ ] **Step 3: Write `## Observation modes` and `## Recording files and drive frequencies`.** Verify in `core.tool.config_args.recording_set`, `core.SBI.observations` (`RecordingSet` and the loader: accepted file types, which column holds the trace, the sampling interval `config.DT_EXP_S`) and `help_text.HELP["spont"]`/`["forced"]`.
  - **Spontaneous:** one passive recording.
  - **Forced:** one driven recording, plus the drive it was made at, in SI units, one value per forcing parameter.
  - **Chi:**
    - a passive recording plus at least one single-tone recording, each with its drive frequency in Hz (`PATH@HZ`), plus the physical drive amplitude in newtons (`--f0-si`; "Drive F₀ (N)" in the window);
    - the lock-in locks where it is told, so a wrong frequency decays like a sinc instead of failing;
    - Hz are converted with `SimConfig.freq_si_to_cell`;
    - a sub-cycle probe is masked, while non-finite, aliased, out-of-band, over-capacity and all-masked input is refused (`chi.probe_verdict`).
  - The window equivalents, and links to `command-line.md#infer`.

- [ ] **Step 4: Write `## Recording length`.**
  - Training draws T_obs log-uniformly over `config.T_MIN_EXP_S` to `config.T_MAX_EXP_S` (1–60 s), but the pre-filter keeps a batch only if its fine steps fit, so the reachable length scales with t_scale:
    - about 7.4 s at t_scale 1;
    - 26.9 s at the master cell's 3.73 (`probe_math.training_ceiling_s`);
    - the 60 s draw limit at the top of the box.
  - A simulated observation warns when it is out of distribution. An experimental recording, whose t_scale is unknown, is checked only against 1–60 s. So a lab member checks the length themselves; `probes band` measures up to the cell's ceiling and flags lengths beyond it.
  - The chi recording-length threshold: both band edges hit their cycle limits at one length, 2.93 s on the `master_weak` cell. `identifiability jacobian` prints it for any cell. Record above it; 4.5 s is the house choice.

- [ ] **Step 5: Write `## The chi assumptions, and checking them`.**
  - **The configured values** (`core/config.py`): `CHI_FREQ_BOUNDS` (0.03, 0.3) times each cell's own Ω₀; `CHI_F0` 0.15 in model units; `CHI_MIN_CYCLES` 2 (below it a probe is masked); `CHI_MAX_CYCLES` 20 (the per-row lock-in ceiling).
  - **Each mode as a lab uses it**, with a fenced command whose every flag is real:
    - `probes band --bounds … --cell …` asks whether the band and drive hold for this cell. It reports four criteria and verdicts that read "the configured … holds / does not hold for this cell", and two caveats: the thresholds are conventions and only capture is physical, and a low harmonic can inflate the not-captured measure.
    - `probes drive --bounds <a box with a Forcing section> --cell …` asks how hard a lab can drive this cell. It reports the strongest free-running and the weakest captured strength, in model units and in the cell's force units. It logs and records suggested Forcing lines to copy by hand; it never writes them.
    - `probes mask --bounds … --prior REF` asks why training probes are masked, over a prior. It reports the share by cause, the Ω₀ quantiles, and the fact that the probe layout does not change with `--seed`.
  - Where the record is (`artifacts show diagnostic <ref>`).
  - What to do when a check fails: a deliberate change in `core/config.py`, then a retrain. The store refuses a posterior trained at other values.
  - Links to `command-line.md#probes-band` and the other two modes.

- [ ] **Step 6: Write `## Defining a model of your own`.** Cover the model-builder screen, and these traps rewritten as rules:
  - numeric literals are stripped before parameter names are discovered;
  - `E` is a parameter, and only `pi` is a constant;
  - the parameter order must equal the bounds file's non-dimensional order, because parameters bind by position and a mismatch gives wrong physics with no error;
  - a save writes the JSON last, which is its commit point;
  - inference and FDT eligibility, linked to `window.md#supported-models`.
  - Verify in `core/Models/user_model.py` and `core/Helpers/model_store.py`.

- [ ] **Step 7: Run the focused tests.**
  - Command: `python -m pytest tests/test_docs.py tests/test_source_hygiene.py -q`.
  - Expected: everything passes.

- [ ] **Step 8: Commit.**
  - `git add docs/guide/recordings.md`
  - `git commit -m "docs: the recordings page" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"`
  - Do not run the fast gate. The controller runs the one-process fast gate (`python -m pytest -m "not slow" -q`, in the background, logged) at the same moment as this task's review.

---

### Task 32: `science.md`, part 1

**Files:**
- Modify: `docs/guide/science.md` (fill `## Nondimensionalisation`, `## Conditioning features`, `## Chi probe design`, `## What chi buys`; leave the other headings with `To be written.` for Task 33)

**Interfaces:**
- Consumes:
  - Task 27's headings.
  - `core.SBI.statistics.FEATURE_LABELS`, `VALID_FLAG_LABELS`, `SUMMARY_WIDTH`.
  - `core.SBI.chi.CHI_FISHER_CHANNELS`.
  - The chi constants in `core/config.py`.
  - The card readings of `probes band`, `drive` and `mask` recorded by Task 26 in `docs/STATE.md`'s gate table. You may read that table; the page never cites it.
- Produces: the anchors `#nondimensionalisation`, `#conditioning-features`, `#chi-probe-design`, `#what-chi-buys`.

**Binding (read before starting):**
- `python` below means `C:\Users\J\anaconda3\envs\biophys-env\python.exe` (in the Bash tool, `/c/Users/J/anaconda3/envs/biophys-env/python.exe`); the `python` on PATH is a different interpreter without torch or pytest. Commands written with backslash paths are PowerShell commands; in Bash use forward slashes.
- Spec §6.2 row 6 is binding for these four sections.
- Sources:
  - `recon/handoff-map.md` Chunks A–B, meaning the entries §3.4–§3.6 and §4.3–§4.4.1 and the Chunk B notes item 7;
  - the Chunk E entries on the valid flags (handoff lines 4808–4848);
  - the Chunk G entries on the band (lines 6405–6464, 6499–6560, 6658–6682);
  - the §8 note on `dt_nd_min` (lines 2680–2692).
- Read the handoff lines themselves. Carry only what the map marks as current or load-bearing, rewritten.
- **The page rules** (spec §6.1):
  - Cite code by module and function name, never by line number.
  - **Name every summary feature by its full label** (`A1_mean`, `D3_bimodality`, `B1_log_Q`, `E1_w_fast`), never by the short id.
  - Stamp: `Checked against commit <hash>.`, the short HEAD taken before checking.
  - Keep Task 27's headings.
- **Never carry:**
  - `core/config.py`'s overturned conclusion that "neither a stronger drive nor a longer recording can recover those probes";
  - "the 20-cycle ceiling is a guess"; the wall was bracketed;
  - the condition-number claim as a chi gain; it is retired;
  - a 42-wide summary or a width of 114; today it is 49 + 1 + 72 = 122 in chi mode;
  - the `.rot.pt` sidecar; the manifest records the layout;
  - a SEED environment variable;
  - any direction-table reading;
  - deleted artifacts as present. An earlier posterior may be described only as "an earlier chi posterior".
- **The reference scan** reads every line.
  - Never write: handoff, STATE.md, CLAUDE.md, superpowers, sdd, ledger, walkthrough, clean break, "piece N", "Task N", "§", "section 2.3"-shaped numbers, ruling, "review's", "Appendix A", "trap CHI10"-shaped names, "guardrail 3", or "C-1" to "C-11". Write each chi change as the measure it names, e.g. "the per-row lock-in duration".
  - No stand-alone capital letter followed by one to three digits. F0, D0 and L2 are the only exceptions.
- **Links:** they must resolve.

- [ ] **Step 1: Stamp the page. Write `## Nondimensionalisation`.**
  - Drift and noise are non-dimensional. Physics enters only through the rescale parameters, x_scale, t_scale and f_scale, which are inferred (f_scale is derived on the tier-1 box; link to `#the-tier-1-constraint-and-temperature`).
  - Nothing nondimensionalises automatically.
  - Frequency is cycles per cell time unit (`SimConfig.freq_si_to_cell`).
  - `SimConfig.dt_nd_min` comes from the prior's t_scale upper bound: it is a sampling constraint, so widening the box silently changes cost and accuracy. A per-batch dt would make integration error depend on the inferred t_scale. The defensible fix is a convergence study, which is an open question.

- [ ] **Step 2: Write `## Conditioning features`.**
  - **The layout** (`statistics.conditioning_rows`, the one layout constructor): 41 features in groups A–G by full label, plus the 8 valid flags (`VALID_FLAG_LABELS`), which make `SUMMARY_WIDTH` 49; plus log T_obs; plus the forcing or the chi block (6 channels per slot × `CHI_K_PAD` 12 = 72).
  - **The three modes and their widths**; verify with `orchestrator.expected_forcing_dim`.
  - **Why eight flags:**
    - a sentinel value was substituted in six channels, not one;
    - one flag each for the `B7_*` pair and the `E1_*` pair, because their co-occurrence was checked on every row;
    - the flag for the fast `E1_*` pair (`V_E1_tau_fast`) keys on `E1_w_fast`, and `E1_log_tau_slow` has its own flag (`V_E1_tau_slow`);
    - a float32/float64 comparison once silently reported zero sentinels.
  - **Rank-Gaussianisation** (`config.RANK_GAUSS_KNOTS` 1024): a contaminated standardisation had attenuated `A1_mean` and `D3_bimodality` by about 1e7.
  - **Per-column winsorisation** (`config.WINSOR_PCT`, 0.1 % / 99.9 %; the `[winsor]` line) and the pathological-trajectory count (`summaries.count_pathological`; the `[patho]` lines).
  - **The flags never reach the Fisher** (`decorrelate` uses the statistics without flags): a binary flag stepping between the two arms divides by the 1e-9 noise floor.
  - **Chi trains with no per-column z-score**, because a near-constant mask column would become an amplifier and a per-column affine map breaks permutation invariance.
  - **Two chi-block rules:** a dead slot is exactly 0.0, and a failed probe is masked, never a phantom.
  - `CHI_K_PAD` is frozen into every artifact's manifest. Width alone cannot identify a layout: 6·5 equals 3·10.

- [ ] **Step 3: Write `## Chi probe design`.**
  - **The band (0.03, 0.3) × Ω₀ and its corrected account:**
    - inside the band, failure tracks drive cycles, not frequency;
    - the boundary was about 31 cycles;
    - re-locking the same trace over at most 20 cycles restored every failing point;
    - the wall was bracketed at 32–36 cycles;
    - 12–16 cycles were not chosen, because of the unpriced trade in frequency selectivity;
    - the high edge is set by entrainment, whose knee is at 0.35–0.4×, not by reproducibility or signal-to-noise;
    - phase scatter has no knee and is advisory.
  - **`CHI_F0` 0.15** is bounded on both sides: by reproducibility, and by the onset of entrainment near 0.2.
  - **The cycle floor and ceiling:**
    - below `CHI_MIN_CYCLES` a probe is masked, never moved or dropped;
    - `CHI_MAX_CYCLES` applies per row, inside `chi_probes.gen_chi_raw`.
  - **The Ω₀ physics of masking:**
    - only the cycle floor masks, and it depends on each row's own Ω₀;
    - live probes go from about 0 % below 1 Hz to 98 % above 30 Hz;
    - the slow draws are real oscillators, so the mask is right;
    - about a third of rows keep no live probe, which only a prior change could reach;
    - raising the maximum recording length was rejected, because it would need about 600 s.
  - **Per-row placement** (`chi.resolvable_multipliers`):
    - placement is judged by frequency span, not live count;
    - the rejected variant bounded placement by the cycle ceiling, which collapses the band because 10 = 20/2.
  - **Per-row lock-in durations** (`chi.lock_in_batched` with `n_samples`):
    - the mean is taken over each row's own prefix;
    - the mask is applied after demeaning.
  - **The masked-fraction reference and its precision:**
    - the reference is about 37 %, within ±12 points;
    - the effective sample size is the batch count;
    - per-batch SD was 12.2 points over 12 batches; run SD is about 6.1;
    - the inference-time check masks about 80 %, by design, because it reads the posterior.
  - **The experimental path truncates an over-long probe** instead of refusing it.
  - **Card readings.** If Task 26 recorded them, add the `probes band`, `drive` and `mask` readings from its card run as the current re-measurement, as numbers with their date, citing no document.

- [ ] **Step 4: Write `## What chi buys`.**
  - The forced-versus-chi Jacobian on the master cell: every parameter's unique handle improves.
  - lam~t_scale was never degenerate on this cell.
  - k~x_scale survives chi, 0.98 → 0.95, because both are dominated by `A1_mean`. Say plainly that chi will not break it.
  - Phase channels carry lam and f_scale; magnitude carries t_scale.
  - Compare only runs of equal T_obs, and read a unique-handle score beside ‖g‖.
  - The rotation is on under chi. The Fisher's feature set is 41 + 3K (`CHI_FISHER_CHANNELS`), seeded again before the chi block.
  - Supplying 6 probes into 12 slots is an untested lever.
  - How to run the chi-versus-forced comparison with `identifiability jacobian`:
    - matched T_obs, because the Jacobian is in signal-to-noise units and the noise falls as about 1/√T;
    - an unmasked upper bound;
    - placement adaptation off.

- [ ] **Step 5: Run the focused tests.**
  - Command: `python -m pytest tests/test_docs.py tests/test_source_hygiene.py -q`.
  - Expected: everything passes.

- [ ] **Step 6: Commit.**
  - `git add docs/guide/science.md`
  - `git commit -m "docs: science page, part 1" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"`
  - Do not run the fast gate. The controller runs the one-process fast gate (`python -m pytest -m "not slow" -q`, in the background, logged) at the same moment as this task's review.

---

### Task 33: `science.md`, part 2

**Files:**
- Modify: `docs/guide/science.md` (fill `## The prior`, `## The tier-1 constraint and temperature`, `## Reading calibration honestly`, `## Narrowing rounds`, `## Identifiability limits`, `## The August 2026 retrain`, `## The solver's physics check`, `## Open questions`)

**Interfaces:**
- Consumes:
  - From Task 14: `SimConfig.assumed_params` and `SimConfig.report_labels`.
  - From Task 15: `core.orchestrator.calibration_verdict(...)`, whose result has the keys `passed`, `alpha`, `threshold`, `parameters`, `tarp` and `caveat`; the `[verdict] ` line; and `sbc`'s `results["rank_verdict"]`.
  - From Task 16: `TruncationRegion.containment`, `body.training["truth_containment"]`, the `[tsnpe] truth ` lines, and the checkpoint key `"fisher_eigenvalues"`.
  - From Task 17: `manifest.config_from_cfg`'s `"tier1"` block.
  - From Task 32: the page's first four sections.
- Produces: the anchors `#the-prior`, `#the-tier-1-constraint-and-temperature`, `#reading-calibration-honestly`, `#narrowing-rounds`, `#identifiability-limits`, `#the-august-2026-retrain`, `#the-solvers-physics-check`, `#open-questions`.

**Binding (read before starting):**
- `python` below means `C:\Users\J\anaconda3\envs\biophys-env\python.exe` (in the Bash tool, `/c/Users/J/anaconda3/envs/biophys-env/python.exe`); the `python` on PATH is a different interpreter without torch or pytest. Commands written with backslash paths are PowerShell commands; in Bash use forward slashes.
- Spec §6.2 row 6, §1.2 ("Temperature on the tier-1 box") and H11 are binding.
- Sources:
  - `recon/handoff-map.md` Chunks D–G: §11 to §11.7, the transpose retraction at handoff lines 3723–3760, and the Chunk E–G science entries.
  - The Chunk D notes items 1 and 4, and the Chunk F notes item 6.
  - For open questions, the owner's science items in `docs/STATE.md`'s open list. Read it; never cite it.
- **The page rules** (spec §6.1):
  - Cite code by module and function name, never by line number.
  - Name summary features by their full label.
  - Stamp: `Checked against commit <hash>.`, the short HEAD taken before checking.
  - Keep Task 27's headings.
- **Never carry — this task is where they live:**
  - The transposed per-direction table of the August 2026 retrain.
  - "k is unmeasured".
  - "k is its own null direction" and "direction 11 is k alone".
  - "direction 4 sitting high argues against the strata".
  - "Expect k to stay unmeasured".
  - "Tier 1 is on by default". It is opt-in by the box.
  - "T's posterior will be its prior". It may be weakly informed in chi mode.
  - "report T as a fixed input" as if T were left out of the verdict. It stays in, labelled.
  - Inference on another observation "hard-warns". It refuses unless accepted.
  - The identity's truncation key "omitted, never None". It is always present.
  - Deleted posteriors as present.
- **The reference scan** reads every line.
  - Never write: handoff, STATE.md, CLAUDE.md, superpowers, sdd, ledger, walkthrough, clean break, "piece N", "Task N", "§", "section 2.3"-shaped numbers, ruling, "review's", "the review of", "Appendix A", "trap X13"-shaped names, "guardrail 3", "C-11", or decision ids.
  - No stand-alone capital letter followed by one to three digits: never H6 or H11, and never tiers written as T1. Write "tier 1". F0, D0 and L2 are the only exceptions.
  - For the narrowing rules, use the words of spec §2.3's table.
- **Links:** they must resolve.

- [ ] **Step 1: Write `## The prior`.**
  - The stability screen (`orchestrator.build_prior`, `core.SBI.Priors.prior`):
    - global sweep rounds, then a local flood-fill to the accepted-set limit;
    - HDBSCAN clusters become the GMM's components;
    - a fixed random state makes it reproducible.
  - The rescale block is log-uniform for names containing "scale" (`orchestrator.build_rescale_prior`); T is uniform.
  - The box is linear. A log box for f_scale was tried and came out worse.
  - Link to `window.md#settings-that-are-not-speed-dials`.

- [ ] **Step 2: Write `## The tier-1 constraint and temperature`.**
  - f_scale = N·beta·k_B·T/x_scale (`core.SBI.derived`), with k_B in the cell's units (`SimConfig.k_b_cell`).
  - Derive f_scale, never beta: the old box sampled the gating-spring energy twice, and only about 2.27 % of its training rows sat at a physical 280–310 K.
  - Over the box corners, the derived f_scale spans about 0.2 to 12,400 pN. The `[tier1]` banner reports this and never refuses.
  - The tier-1 box has its own bounds and cell files, instead of an edited `master.txt`.
  - A `master.txt` prior loads under `master_tier1.txt`, because the non-dimensional box is the same.
  - Temperature enters only through the force scale:
    - no effect in spontaneous mode;
    - in chi mode it scales every probe's |χ|, so it may be weakly informed.
  - It is reported as an assumed input (`SimConfig.assumed_params`, the label "T (assumed input, K)").
  - It stays in the calibration verdict and in the joint coverage test.
  - Records carry the constraint in their `config` block (`"tier1"`, `"assumed_params"`).
  - Tiers 2 and 3 are open: instrument calibration of x_scale, and a population band. So is the alternative of fixing T at 300 K for a 12-dimensional box. Link to `#open-questions`.

- [ ] **Step 3: Write `## Reading calibration honestly`.**
  - **The verdict rule** (`orchestrator.calibration_verdict`):
    - every inferred parameter's rank-uniformity KS p ≥ 0.05 ÷ the number of inferred parameters, temperature included;
    - and the joint coverage test's p ≥ 0.05;
    - a FAIL is a result, not an error;
    - the caveat: t_scale's test rests on the calibration's operating points (200), not its 2,000 datasets.
  - **Reading conventions:**
    - read KS p-values over c2st ranks, because about 0.58 is c2st's finite-sample floor;
    - pooled rank histograms show severity that KS cannot;
    - a flat SBC means "not overconfident", never "informative": a posterior that returns the prior is flat;
    - a TARP value of 1.000 is not "perfectly calibrated".
  - **The best-fit table is not a recovery measurement.** `overlay.rank_by_stats` standardises by the spread across posterior draws, not by measurement noise.
  - **Informativeness** (`core.SBI.analysis`): the expected prior-to-posterior KL on the fresh calibration set.
    - Read its sign first. It was −23.1 nats for a 5-epoch smoke train.
    - Never compare it with a figure measured on training rows.
    - It is joint, with temperature's own entry marked.
  - **Pooled SBC over mixed probe counts** can be flat while each count is miscalibrated in compensating directions. `sbc --chi-k-fixed` stratifies. `sbc`'s per-repeat verdict is the rank-uniformity half only.
  - **Repeating a calibration:** `validate --seed`. The calibration stream is derived from the seed, so it never replays a training run's stream (`core.rng.calibration_seed`).

- [ ] **Step 4: Write `## Narrowing rounds`.**
  - **Draw from the truncated prior, never the posterior.** Using the posterior tempers: intervals contract by (L+1)^−1/2 while SBC stays flat, which the proposal test pins.
  - **Link the rules table** at `rules-and-traps.md#narrowing-round-safety-rules`, and give each rule's reasoning in one line.
  - **The t_scale override** turns a box into a smooth reweighting, p(θ)·P(A|θ₋ₜ)/P(A). Directions loaded on t_scale are skipped above 1/√d.
  - **The box-mass bound** is 1 − k(1 − level).
  - **The region is generous**: 99.9 %, with a warning below 0.99.
  - **The round reports** truth inside and outside for every truncated direction (`TruncationRegion.containment`).
  - **The −log P(A) KL inflation** is printed, not corrected.
  - **Deliberately open:** pooling rows across rounds, the batch-by-scale t_scale override, and changes to the Fisher eigenbasis.

- [ ] **Step 5: Write `## Identifiability limits`, `## The August 2026 retrain` and `## The solver's physics check`.**
  - **Identifiability limits:**
    - n, temp and tau_c enter only through noise amplitudes, never the drift, and the observable is state column 0. Verify in `core/Models/nadrowski_model.py`.
    - The Fisher decomposition uses locally measured single-trajectory noise, not the contaminated standardisation. Verify in `core.SBI.decorrelate`.
  - **The August 2026 retrain, correct orientation only:**
    - SBC and TARP were excellent;
    - the posterior-predictive check was over-dispersed: 99.1 % at nominal 90 %;
    - the loss plateaued cleanly, with train and validation tracking, which points to an identifiability limit rather than under-fitting;
    - t_scale alone is the best-constrained direction; k loads on the second and fifth directions, both well constrained;
    - eigenvectors are columns, decoded only through `reparam.rotation_of`;
    - only 6 of the 12 chi slots were filled;
    - its "more capacity will not help" verdict was measured on the broken conditioning, which is why the retrain uses a larger network;
    - its artifacts no longer exist, so it cannot be re-run.
  - **The solver's physics check:**
    - the CUDA-graph solver against eager, with noise live, 3,072 rows per arm: Ω₀ two-sample z = −0.01, KS D = 0.0020 against a critical 0.0347;
    - throughput 123,349 against 12,639 steps/s;
    - the bitwise test covers only a noise-zeroed model.
  - **Batch count against width:** the count sets the diversity of (t_scale, T_obs), the width is replication, so 5,000 × 2,048 ≠ 10,000 × 1,024.

- [ ] **Step 6: Write `## Open questions`.** Each item gets one paragraph: what is unknown, and what would settle it.
  - M-replicate conditioning, first, before any experiment design: widths falling as M^−1/2 mean estimator noise; saturating widths mean real degeneracy.
  - The thermal-tail feature S_X → 2k_BT/(λω²).
  - The strata test: is the effective sample size the batch count?
  - A step (force-clamp) transient for k; two-tone intermodulation for delta_E, S, N and phi.
  - K against the slot capacity.
  - Tiers 2 and 3, and fixing T.
  - The non-dimensional log-sampling decision: `user_prior._global_map` samples uniformly in the linear box.
  - Pooling rows across rounds.
  - The `dt_nd_min` convergence study.
  - The owner's science items from STATE's open list, each reworded without labels:
    - the NaN column in `identifiability jacobian`;
    - the fixed probe-layout seed across repeats;
    - one-probe observations (`--chi-k 1`);
    - the passive-baseline Ω₀;
    - the Hopf band;
    - any other item marked as science or as the owner's decision.

- [ ] **Step 7: Run the focused tests.**
  - Command: `python -m pytest tests/test_docs.py tests/test_source_hygiene.py -q`.
  - Expected: everything passes.

- [ ] **Step 8: Commit.**
  - `git add docs/guide/science.md`
  - `git commit -m "docs: science page, part 2" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"`
  - Do not run the fast gate. The controller runs the one-process fast gate (`python -m pytest -m "not slow" -q`, in the background, logged) at the same moment as this task's review.

---

### Task 34: `architecture.md`

**Files:**
- Modify: `docs/guide/architecture.md` (fill the skeleton)

**Interfaces:**
- Consumes:
  - Task 27's headings.
  - From Task 11: `core.tool.help_defaults`.
  - From Task 13: `core.SBI.derived.for_simulation`.
  - From Task 17: `core.artifacts.identity.FORMAT == "training-rows/3"` and `units_sha256`.
  - From Task 18: `pipeline._BatchProbeLedger`, `_ACTIVE_LEDGER` and `state.pt`'s `"chi_masked"`.
  - From Task 19: the `accepted` records.
  - From Task 20: `core.diagnostics.probe_math`.
  - From Task 21: `chi_probes.ProbeRecord` and `gen_training_data(probe_observer=...)`.
  - From Tasks 22–24: `core.diagnostics.probes`.
- Produces: the anchors `#module-map`, `#stages-and-compositions`, `#the-artifact-store`, `#the-command-line-tool`, `#the-window`, `#the-solver-cuda-graphs-and-reproducibility`, `#memory-planning-and-out-of-memory-recovery`, `#the-fdt-pipeline`.

**Binding (read before starting):**
- `python` below means `C:\Users\J\anaconda3\envs\biophys-env\python.exe` (in the Bash tool, `/c/Users/J/anaconda3/envs/biophys-env/python.exe`); the `python` on PATH is a different interpreter without torch or pytest. Commands written with backslash paths are PowerShell commands; in Bash use forward slashes.
- Spec §6.2 row 7 is binding. **Regenerate the module map from `core/`**; do not copy it.
- Sources:
  - `PRISM_HANDOFF.md` §2, §2.1 and §2.2, used only for the notes that survive: the pipeline façade, and "Reduction is irrelevant to inference";
  - §8.3 (lines 2624–2692) and §10.5 (lines 2946–3062);
  - the P, S and C trap groups (lines 1503–1614), for the design only;
  - the Appendix memory and recovery entries: lines 4142–4226, 4263–4379, 4460–4477, 4528–4634, 6144–6220.
  - `recon/handoff-map.md`: those entries, and the Chunk A, C and E notes.
- **The page rules** (spec §6.1):
  - Cite code by module and function name, never by line number.
  - Stamp: `Checked against commit <hash>.`, the short HEAD taken before checking.
  - Keep Task 27's headings.
- **Never carry:**
  - `save_*_artifacts` or `_emit`;
  - `runners.py`;
  - `cli._parse_cell`;
  - the `.rot.pt` sidecar;
  - `Resources/Checkpoints`;
  - "torch.compile"; it was never used;
  - "nothing is checkpointed";
  - the handoff's old file sizes;
  - `PYTORCH_ALLOC_CONF`; the parsed variable is `PYTORCH_CUDA_ALLOC_CONF`;
  - the handoff's memory table rows as open. `psd_welch` and Campaign 1 are fixed. The FDT sanity checks' solution view is an open engineering item for the owner's list, not this page.
- **The reference scan** reads every line.
  - Never write: handoff, STATE.md, CLAUDE.md, superpowers, sdd, walkthrough, clean break, "piece N", "Task N", "§", "section 2.3"-shaped numbers, ruling, "review's", "Appendix A", "trap X14"-shaped names, "C-11", or decision ids.
  - Never write the lowercase word **ledger**. Write "the per-batch probe tally" or the class name `_BatchProbeLedger`.
  - No stand-alone capital letter followed by one to three digits. F0, D0 and L2 are the only exceptions.
- **Links:** they must resolve.

- [ ] **Step 1: Stamp the page. Generate the module map.**
  - With a throwaway script in the session scratchpad (never in the repo), walk `core/**/*.py`, skipping `__pycache__` and `core/Reduction/tests`. Print each module's path and the first sentence of its docstring (`ast.get_docstring`).
  - Write `## Module map` from that output as a grouped list: entry points, the core stages, `SBI`, `Simulator`/`Solvers`/`Models`, `artifacts`, `diagnostics`, `tool`, `gui`, `FDT`, `Helpers`, `Reduction`. Give one line per module in plain words, not the docstring's history.
  - State two notes:
    - `core.SBI.pipeline` is a façade that re-imports its extracted siblings at its end, because tests rebind names on it. Removing a "redundant" re-import fails silently: tests go slow or quiet, not red.
    - `Reduction` is not used by inference.

- [ ] **Step 2: Write `## Stages and compositions`.**
  - **The data flow:** bounds, cell and units → `cli.make_sim_config` → `build_prior` → `build_posterior` → `validate_calibration` → `infer_and_visualize`.
  - **`build_posterior`'s generation:** `gen_training_data` runs Sobol (t_scale, T_obs) batches, with chi probes and `conditioning_rows`.
  - **The observation builders:** `generate_observations`, `build_experiment_observation`.
  - **The compositions:** `simulated_inference`, `experimental_inference`, `tsnpe_round`, `build_truncation_region`, `training_preview`, `fresh_run_near_misses`.
  - **`core.runs.public_entry`:**
    - it gives each public stage a private copy of the configuration;
    - it writes the run's `log.txt`;
    - no stage mutates what it is handed.
  - **Every knob is a keyword argument**, because `core.orchestrator` binds `core/config.py` constants at import.
  - **Refusals** are `core.refusals.Refusal` with a field key. Each front end names its own control or flag, from `core/gui/fields.py` and `core/tool/fields.py`.
  - **Messages** are `logging` records.
  - **The force scale:** every simulating path goes through `derived.for_simulation`, and the two force builders raise `RuntimeError` on a tier-1 index.
  - **The probe observer and the per-batch probe tally:**
    - `gen_training_data(probe_observer=...)`, `chi_probes.ProbeRecord`, `pipeline._BatchProbeLedger`;
    - the tally commits after a batch's rows are stored and discards an abandoned attempt;
    - it drives the masked-probe run total.

- [ ] **Step 3: Write `## The artifact store` and `## The command-line tool`.**
  - **The store:**
    - `KIND_DIRS` and `manifest.json`; each kind's body keys are a closed set (`core.artifacts.manifest`);
    - saving is a rename; `log.txt`;
    - the simulation cache's identity (`core.artifacts.identity.SimulationIdentity`, `FORMAT`):
      - its fields, including the prior fingerprint, the truncation region (None when amortized) and the units fingerprint;
      - the feature-set version re-keys it;
      - network settings are not in it, so a complete cache can be refit at another capacity;
    - loading refuses any verifiable mismatch;
    - the two acceptances, and where each use is recorded:
      - an inference's `results.accepted`;
      - a calibration's `results.accepted`;
      - a narrowing child's `training.accepted`;
    - `fdt` records are progressive;
    - the near-miss refusal is exactly one field;
    - `core.artifacts.report` produces the lineage summary.
  - **The tool:**
    - one module per family;
    - parsers are built torch-free;
    - knobs are forwarded only when given (`config_args.knobs`), except `validate --seed`, which always forwards a seed (drawn when not given) so the calibration record names it;
    - the refusal ladder and exit codes in `core.tool.main`;
    - `core/tool/fields.py`;
    - the console handlers;
    - `help_defaults`, the one table of default clauses;
    - `smoke`'s own store root, keyed on `temp_store_root`.

- [ ] **Step 4: Write `## The window`.**
  - Screens, and panels on `BasePanel`.
  - **Worker threads:**
    - keep the worker's reference until it finishes;
    - render figures to PNG on the worker, never paint a worker-built figure;
    - force Agg before any core import.
  - **Stream routing:** `core.gui.streams._SignalStream` and `_PumpLogHandler`; `core.gui.vt`, which classifies each atomic chunk of tqdm output.
  - **Progress:**
    - one overall bar;
    - the solver meter reads `core.progress.SOLVER`, never bar text;
    - the solver bar stays enabled because cancel and the stall detector depend on its writes.
  - **Cancellation:**
    - cooperative, with a `CancelToken`;
    - `WorkerCancelled` derives from `BaseException`;
    - about 1 s latency, up to one training epoch;
    - the leaked tqdm lock is reset;
    - the token raises only on its owner thread;
    - a hard process cancel was considered and declined.
  - **Settings persistence:** `core/gui/settings.py` helpers; the restore order owned by `ArtifactPicker`.
  - `core/gui/fields.py`.
  - Qt here has no SVG image plugin, so PNG icons ship.
  - qfluentwidgets was rejected: it is GPLv3.

- [ ] **Step 5: Write `## The solver, CUDA graphs and reproducibility`, `## Memory planning and out-of-memory recovery` and `## The FDT pipeline`.**
  - **The solver:**
    - `core.Solvers.sdeint` is Itô Euler–Maruyama;
    - `euler_compiled` replays a TorchScript step from a CUDA graph: `SOLVER_CUDA_GRAPHS`, a 50-step chunk, a cache of 8;
    - launch overhead was 88 % of solver time; 5,520 ms became 698 ms, a 7.9× gain;
    - the graph-path invariants;
    - a capture failure falls back to eager with a warning;
    - `Solver()` is built per call on purpose, as a test seam;
    - the segment seam duplicates one sample;
    - `core.rng.seeded` seeds torch and numpy;
    - runs are not bitwise-reproducible on CUDA or across devices;
    - TorchScript is not bitwise-reproducible on its first runs, so warm up before comparing.
  - **Memory:**
    - the planner's budget and `vram_ceiling_gib`, which `SIM_VRAM_CEILING_GIB` and `PRISM_VRAM_CEILING_GIB` set;
    - the learned per-row budget, with its per-batch credit;
    - `_FORCE_BUILD_PEAK_MULTIPLE` is 4, measured at 4.10×;
    - the ladders: `_gen_obs_retry`, `_rows_with_oom_retry` (row halving) and `retry_on_oom` (the Fisher and the posterior-predictive check);
    - the batch-level retry, with its attempts and delays in `core/config.py`;
    - restore, then release, then wait;
    - "the rows outrank the restore point";
    - `_we_are_the_holder`;
    - WDDM spills into shared memory: 21.67 GiB completed on a 15.92 GiB card, 9× slower. That is why a slow run is the warning sign and waiting beats shrinking;
    - the allocator policy `PYTORCH_CUDA_ALLOC_CONF` is set by default and recovers about 6 GiB;
    - accepted gaps: the prior sweep has no out-of-memory retry, and the Fisher's RNG restore can replace an out-of-memory traceback on a starved card.
    - Verify each name in `core/SBI/pipeline.py` and `core/config.py`.
  - **FDT:**
    - `campaigns`, `spectral.psd_welch`, `sanity`, `fdt_pipeline.run_fdt`, `cross_validation` (the S and T_a/T sweeps), `compare`, `plots`;
    - it runs on the CPU only, because the sequential SDE loop at M ≈ 256 is about 3.4× faster there;
    - its records are progressive.

- [ ] **Step 6: Run the focused tests.**
  - Command: `python -m pytest tests/test_docs.py tests/test_source_hygiene.py -q`.
  - Expected: everything passes.

- [ ] **Step 7: Commit.**
  - `git add docs/guide/architecture.md`
  - `git commit -m "docs: the architecture page" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"`
  - Do not run the fast gate. The controller runs the one-process fast gate (`python -m pytest -m "not slow" -q`, in the background, logged) at the same moment as this task's review.

---

### Task 35: `rules-and-traps.md`

**Files:**
- Modify: `docs/guide/rules-and-traps.md` (fill `## Rules the code keeps`, `## Narrowing-round safety rules`, `## Traps`)

**Interfaces:**
- Consumes:
  - Task 27's headings.
  - Spec §2.3's replacement vocabulary.
  - From Task 13: the builders' `RuntimeError`.
  - From Task 16: truth containment.
  - From Task 19: the acceptances recorded.
- Produces: the anchors `#rules-the-code-keeps`, `#narrowing-round-safety-rules`, `#traps`, and the `###` groups under `## Traps`.

**Binding (read before starting):**
- `python` below means `C:\Users\J\anaconda3\envs\biophys-env\python.exe` (in the Bash tool, `/c/Users/J/anaconda3/envs/biophys-env/python.exe`); the `python` on PATH is a different interpreter without torch or pytest. Commands written with backslash paths are PowerShell commands; in Bash use forward slashes.
- Spec §6.2 row 8 and §2.3 are binding.
- **The narrowing-round rules table** uses §2.3's words exactly:
  - the truncated-prior rule;
  - the observation-digest rule;
  - the narrowed-model rule;
  - the eigenbasis rule;
  - the unweighted-draws rule;
  - the generous-region rule;
  - the cost-on-screen rule;
  - the region-carries-its-basis rule;
  - the calibrate-on-the-region rule.
- **Each rule** says where it is enforced, as module and function, and names the test that pins it, as `tests/<file>.py::<name>`.
- **The rules still only prose** are named as such:
  - the rate of truths outside the region;
  - a region cut in the box when the parent has no rotation;
  - the −log P(A) inflation, printed but not corrected.
- **Traps:** every trap the map marks still-applies, rewritten against today's code, grouped by subsystem, **without the old labels**.
- **Also cover the traps a later design handed on:**
  - `core/gui/panels/inference_tabs.py` is a re-export shim;
  - the tab gating includes the TSNPE tab;
  - there are six inference tabs;
  - the panels read `value_or_none`, not `FloatField.value()`.
- Sources:
  - `PRISM_HANDOFF.md` §3 (lines 290–457), §5 (lines 1455–2074), §6.1's comment rules (lines 2229–2352) and §11.6 (lines 3337–3427);
  - Appendix traps: TorchScript warm-up (lines 2653–2668); the façade re-import (4217–4226); waiting on memory one holds (4310–4339); the allocator variable (4340–4379); a pre-flight must share the run's configuration source (5655–5677); the flags must not reach the Fisher (4841–4848); order, not adjacency (4445–4459);
  - `recon/handoff-map.md` Chunk B notes: the trap-by-trap status list;
  - `docs/superpowers/specs/2026-09-11-one-flow-design.md` §2.5, the rule-by-rule audit. Read it; never cite it.
- **The page rules** (spec §6.1):
  - Cite code by module and function name, never by line number.
  - Stamp: `Checked against commit <hash>.`, the short HEAD taken before checking.
  - Keep Task 27's headings.
- **Never carry:**
  - obsolete traps: the prompt interface's `_prompt_index`; the save keywords; the `run_fdt` prompts; the "Simulator constructed once" history; the transposed-sidecar repair;
  - the handoff's pinning-test names that no longer exist;
  - `PYTORCH_ALLOC_CONF`.
- **The reference scan** reads every line.
  - Never write: handoff, STATE.md, CLAUDE.md, superpowers, sdd, ledger, walkthrough, clean break, "piece N", "Task N", "§", "section 2.3"-shaped numbers, ruling, "review's", "Appendix A".
  - **Never** write trap labels ("trap X5", "CHI10", "G1", "M1b", "P7", "SIM2", "U3"), "guardrail 3", "C-11", or decision ids (D11, D12, V5).
  - No stand-alone capital letter followed by one to three digits. F0, D0 and L2 are the only exceptions.
- **Links:** they must resolve.

- [ ] **Step 1: Stamp the page. Write `## Rules the code keeps`.** One short paragraph per rule, each verified against the code named:
  - **the model/solver contract:**
    - there is no model base class;
    - `f(x, t)` takes the integer step index, with forcing folded into the drift;
    - `g` is diagonal;
    - a bare `force` attribute;
    - construction is positional, so the bounds order must equal the constructor order;
    - the observable is state column 0.
    - Verify in `core.Simulator.simulator`, `core/Models/`.
  - **the force-channel rule:** `forcing.n_force_channels`; Hopf reads channel 1 always.
  - **the bounds/cell/units triple.**
  - **everything non-dimensional.**
  - **the conditioning layout:** `statistics.conditioning_rows`; a dead chi slot is exactly 0.0.
  - **every knob an argument.**
  - **the private configuration copy:** `core.runs.public_entry`.
  - **refusals with field keys.**
  - **no chi override for anything that trains or infers:**
    - the band and drive are `core/config.py`'s;
    - `run_guards._assert_chi_config_is_deliberate`;
    - the store refuses others;
    - `probes` may take its own grids because it only measures.
  - **the narrowing-round refusal with no escape hatch:** a narrowed parent is refused on another observation.
  - **a persisted value that defines what is measured is compared with `core/config.py` before the spend.**
  - **a pre-flight must read its configuration from the same place as the run it clears.**
  - **the comment policy:**
    - state the reason in words;
    - cite no planning or working document, nor its labels, which `tests/test_source_hygiene.py` enforces;
    - cite functions, not line numbers;
    - never delete a measured number, a "tried and regressed" note, an ordering requirement, or a "looks removable and is not" note.

- [ ] **Step 2: Build the rules table.**
  - For each of the nine rules, find where it is enforced and the pinning test. Candidates to verify: grep `def <name>` under `tests/`, and keep only names that exist.
    - **truncated-prior:** `test_conditioning_repair.py::test_the_proposal_is_the_TRUNCATED_PRIOR_and_not_the_posterior`, `::test_the_truncated_prior_is_the_base_density_inside_and_minus_inf_outside`.
    - **observation-digest:** `test_artifact_store.py::test_a_round_whose_region_names_no_observation_is_refused_before_the_spend`, `test_conditioning_repair.py::test_a_non_amortized_parent_is_refused_on_another_observation`.
    - **narrowed-model:** `test_artifact_store.py::test_a_posterior_manifests_amortized_flag_must_agree_with_its_region`, `::test_a_non_amortized_posterior_needs_accept_and_the_flag_is_recorded`, `test_tool.py::test_validate_refuses_a_non_amortized_posterior_without_accept_truncated`.
    - **eigenbasis:** `test_conditioning_repair.py::test_the_region_is_built_over_the_leading_fisher_directions_only`.
    - **generous-region:** `::test_tsnpe_round_refuses_bad_direction_counts_and_hpd_levels_before_any_spend`, `::test_a_tight_hpd_warns_through_the_preflight_channel_and_still_runs`, plus `test_conditioning_repair.py::test_a_narrowing_round_reports_and_records_whether_the_truth_lies_inside_each_direction` and `::test_a_region_reports_containment_per_truncated_direction`.
    - **region-carries-its-basis:** `::test_a_region_measured_in_one_basis_is_refused_in_a_sign_flipped_one`, `::test_build_truncation_region_records_the_parents_basis`, `::test_a_truncated_round_refuses_a_prior_other_than_the_parents`, `::test_a_t_scale_loaded_direction_is_excluded_from_the_region`, `test_user_sbi.py::test_a_tsnpe_round_reuses_the_parents_basis_and_refuses_every_mismatch`, `::test_a_truncated_round_routes_to_its_own_checkpoint_and_the_amortized_digest_is_untouched`.
    - **calibrate-on-the-region:** `test_user_sbi.py::test_calibration_theta_star_lies_inside_the_region_when_one_is_given`, `::test_the_reported_kept_fraction_is_measured_after_the_t_scale_override`.
    - **unweighted-draws and cost-on-screen:** find the pin, or write "structural, no dedicated test".
  - Columns: rule (in §2.3's words) — what it prevents — where it is enforced — the pinning test.
  - Then the three prose-only rules, with one line each on why they are not enforced.

- [ ] **Step 3: Write `## Traps`, grouped.** Use the `###` groups below. Include every trap the Chunk B notes list marks still-applies, plus the Appendix traps named in the sources. Each trap is one rule stated plainly, plus the silent failure it prevents and where the guard lives (module and function).
  - `### The window: threads, figures and paths`:
    - keep a worker's reference until it finishes;
    - never paint a worker-built figure;
    - force Agg first;
    - one run app-wide;
    - save with `visualizers.save_figure`; note which FDT and Reduction plots still call `savefig`;
    - drop and report non-finite draws before quantiles;
    - detach pop-out figures from pyplot and import the Qt backend lazily.
  - `### Progress, the solver meter and cancellation`: the tqdm chunk-classification rules, the 15 Hz pump, keying by position, the overall-bar election, and the two cancellation rules.
  - `### Settings persistence and the inference screen`:
    - the restore order;
    - CrossVal's derived grid ends are never persisted;
    - picker keys are restored with a missing-item guard;
    - the gating table, TSNPE included;
    - new draft versus install into the session;
    - the six-tab control lock;
    - the re-export shim;
    - `value_or_none`.
  - `### Simulate and video export`:
    - m+1 grid points per frame, emitting all but the first;
    - `no_grad`;
    - cancel between frames;
    - set the SDE's force directly;
    - pin the CPU;
    - export frames contiguous with even dimensions;
    - GIF durations in ms;
    - import imageio after torch, or OMP Error #15 aborts with no traceback.
  - `### Labels and units`.
  - `### User-defined models`.
  - `### The chi probe set and its Fisher`:
    - three feature sets;
    - the Fisher needs `resolution_filter=False` and a seeded chi block;
    - never a per-column z-score under chi;
    - the mask gate on both sides;
    - no max-pool and no BatchNorm;
    - width cannot identify the layout;
    - the slot capacity is frozen;
    - the lock-in ceiling is per row;
    - Hz go through `freq_si_to_cell`;
    - never clip the chi block;
    - a new statistics channel reaches the Fisher;
    - a near-constant channel inflates a standardised Jacobian;
    - binary flags stay out of the Fisher.
  - `### Memory and devices`:
    - `torch.cuda.mem_get_info()` overstates free memory on WDDM, so read `nvidia-smi`;
    - an unwrapped out-of-memory error rules out the solver;
    - recovery paths restore before releasing and never raise;
    - never wait on memory you hold yourself;
    - the allocator variable's real name.
  - `### Randomness and reproducibility`:
    - resolve the solver at call time;
    - `SimConfig.chi_mode` is a plain False;
    - seed immediately before the chi block;
    - numpy drives training's initial states;
    - TorchScript warm-up;
    - the rotation is not reproducible across processes, so a resume reuses the stored one;
    - `torch.save` of a view writes the whole storage.
  - `### Rotations and narrowing rounds`:
    - V holds eigenvectors in columns, decoded only by `reparam.rotation_of`;
    - calibrate on the truncated prior;
    - the region belongs to the cache identity;
    - a direction loaded on t_scale turns the restriction into a reweighting;
    - torch's sigmoid inverse clamps at the box edge;
    - a fitted posterior used as the next prior is tempering.
  - `### Module layout`:
    - the pipeline façade's re-imports;
    - `core/config.py` constants are snapshotted at import.
  - `### Reading diagnostics`:
    - flat SBC is not informativeness;
    - the best-fit table is not recovery;
    - calibration's operating points are t_scale's effective sample size.

- [ ] **Step 4: Run the focused tests.**
  - Command: `python -m pytest tests/test_docs.py tests/test_source_hygiene.py -q`.
  - Expected: everything passes.

- [ ] **Step 5: Commit.**
  - `git add docs/guide/rules-and-traps.md`
  - `git commit -m "docs: the rules and traps page" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"`
  - Do not run the fast gate. The controller runs the one-process fast gate (`python -m pytest -m "not slow" -q`, in the background, logged) at the same moment as this task's review.

---

### Task 36: `testing.md` and `retrain.md`

**Files:**
- Modify: `docs/guide/testing.md` and `docs/guide/retrain.md` (fill both skeletons)

**Interfaces:**
- Consumes:
  - From Task 26: `CLAUDE.md`'s card-recipe block, as updated with run 4, and the card-run row in `docs/STATE.md`'s gate table, which holds the diagnostic-card command lines and readings. Both are read and never cited.
  - From Task 15: `validate --seed`, the `[verdict] ` line, and `results["verdict"]`.
  - From Task 16: `transform.fisher_eigenvalues` carried across a resume, `training["truth_containment"]`, and the `[tsnpe] truth ` lines.
  - From Task 18: the `[chi] masked probes` run-total line, and smoke's `--hidden-features`/`--num-transforms`.
  - From Task 19: `results["accepted"]` and `training["accepted"]`.
  - From Tasks 22–24: the `probes` commands.
  - From Task 29: the `command-line.md` anchors.
- Produces:
  - `testing.md#the-card-smoke-gate` and `#the-diagnostic-card`, which `retrain.md` links to.
  - `retrain.md#gates` and `#afterwards`, which the reviewer path links to.

**Binding (read before starting):**
- `python` below means `C:\Users\J\anaconda3\envs\biophys-env\python.exe` (in the Bash tool, `/c/Users/J/anaconda3/envs/biophys-env/python.exe`); the `python` on PATH is a different interpreter without torch or pytest. Commands written with backslash paths are PowerShell commands; in Bash use forward slashes.
- Spec §6.2 rows 9–10, §6.5, §5.12 and §8 are binding. H6's decisions are binding, **but never write "H6" or "H11"**: the scan reads them as ids.
- **`testing.md` holds:**
  - the environment and interpreter;
  - the environment variables;
  - the gates, markers and their budgets;
  - the card smoke gate;
  - the diagnostic card, now with `probes`;
  - what a green suite does not certify;
  - notes for writing tests.
- **The card recipe.** `CLAUDE.md`'s recipe block, as Task 26 left it — from the line `` ```powershell `` through the line ending "Delete `$S` afterwards.", pass criteria included — is copied **byte for byte** (indentation may differ). Task 26 made that block reference-free, so it passes the scan. That is the one recipe text; `docs/STATE.md` holds a verbatim copy, and the documents review checks all three are equal.
  - The sentence after the block in `CLAUDE.md` that points at STATE's timings is NOT copied.
  - If the block contains a token the scan flags (a `$T1`-shaped variable), stop and report it. Do not alter the copy.
  - The copied block must be the first ```` ```powershell ```` fence on `testing.md`: fence every earlier command example on that page with no language tag (or `text`), because `recipe_equal.py` (and the Step 7 check) starts at the first line reading ```` ```powershell ````.
- **`retrain.md` is the runbook** (spec §6.5). It covers:
  - the decisions, each with its reason;
  - the pre-flight, including the epoch-timing projection and a stop point;
  - the exact commands, with every flag;
  - resuming;
  - the watch-lines;
  - the gates;
  - the certification round;
  - afterwards: which record answers which gate.
  - It names no working document. The controller copies the results into its own records.
- Sources:
  - `recon/design-runbook.md` (the whole file; its §9 gaps are now fixed by Tasks 13–19) and `recon/design-tier1.md` §5–§6;
  - `PRISM_HANDOFF.md` §1.2–§1.3 (live rules only), §4.1 steps 5–7 (lines 624–750), §10.3 (lines 2916–2932), §11.8–§11.9 (lines 3468–3628);
  - Appendix lines 4575–4583, 4748–4757, 5257–5308, 5540–5569 and 6207–6220;
  - `recon/handoff-map.md` Chunk D notes item 5 and Chunk E notes item 1.
- **Never carry:**
  - Run A, Run B or their caches;
  - the environment-variable recipes;
  - `Resources/Checkpoints`;
  - "nothing is checkpointed";
  - "resume is automatic with nothing to remember"; the policy flags and the near-miss refusal exist now;
  - "expect k to stay unmeasured";
  - the old "29 of 42" ablation baseline as a baseline;
  - the implied-temperature gate;
  - old-versus-new KL on one calibration set as a gate;
  - test counts or suite sizes;
  - `PYTORCH_ALLOC_CONF`.
- **The page rules** (spec §6.1):
  - Cite code by module and function name, never by line number.
  - Stamp: `Checked against commit <hash>.`, the short HEAD taken before checking.
  - Keep Task 27's headings.
- **The reference scan** reads every line, code blocks included.
  - Never write: handoff, STATE.md, CLAUDE.md, superpowers, sdd, ledger, walkthrough (say "checked by hand on a real screen"), clean break, "piece N", "Task N", "§", ruling, "review's", "Appendix A", "trap X5", "guardrail 3", "C-11", "row F1", or decision ids.
  - No stand-alone capital letter followed by one to three digits. F0, D0 and L2 are the only exceptions.
  - PowerShell variables need id-free names. Define them once near the top of the runbook: `$py` the interpreter; `$Box = "--bounds","Resources/Bounds/nadrowski/master_tier1.txt"`; `$Cell = "--cell","Resources/Cells/nadrowski/master_spont_tier1.txt"`; `$M = "--bounds","Resources/Bounds/nadrowski/master.txt"`; `$MC = "--cell","Resources/Cells/nadrowski/master_spont.txt"`; `$S` a scratch folder.
- **Links:** they must resolve.

- [ ] **Step 1: Stamp both pages. Write `testing.md`'s first three sections.**
  - **Environment and interpreter:**
    - the `biophys-env` interpreter: Python 3.12, torch 2.9.0+cu130, PySide6 6.9.3;
    - sbi 0.25.0 is installed while `requirements.txt` pins 0.26.1. An sbi upgrade must re-check that sbi's per-epoch print still fires, because the window's cooperative cancel hooks into it.
  - **Environment variables:**
    - `KMP_DUPLICATE_LIB_OK`: defaulted by the root `conftest.py`; set by the tool's entry and the launchers.
    - `QT_QPA_PLATFORM=offscreen`: headless checks only.
    - `PRISM_RESOURCES`, `PRISM_ARTIFACTS`, or `core.artifacts.use_store` to keep a check away from the real store.
    - `PRISM_VRAM_CEILING_GIB`: read live.
    - `PRISM_MEM_LOG_EVERY`: read once, at import.
    - `PYTORCH_CUDA_ALLOC_CONF`: set by default in `core/config.py`.
  - **Test gates and markers** (`pytest.ini`, `conftest.py`, `tests/conftest.py`):
    - the fast gate is `pytest -m "not slow"`, run as one process, with a 16-minute target. Only one process catches code that swaps the default store;
    - the full run is `pytest`, budget about an hour;
    - the slow set: list its tests by running `python -m pytest -m slow --collect-only -q`;
    - the markers `slow`, `gpu` and `display`, and `QT_QPA_PLATFORM=windows pytest -m display`;
    - count with `pytest --collect-only -q`;
    - never two pytest processes at once, and never edit a source file while a suite runs, because some tests read source with `inspect.getsource`;
    - the session guards in `tests/conftest.py`: the temporary artifact root with its teardown check, checkpointing off (pass `checkpoint_every` to get a cache), a temporary settings file, no modal dialogs, and no `sbi-logs` directory;
    - the source scans: `tests/test_source_hygiene.py`, the literal-path scan, and `tests/test_docs.py`.

- [ ] **Step 2: Write `testing.md`'s card sections and the rest.**
  - **`## The card smoke gate`:**
    - when to run it: after touching code that moves tensors;
    - the block copied byte for byte;
    - the pass criteria in words, per run: 1, 2, 2b, 3 and 4 (the block's five lines; run 4's resume is a card-run leg, not part of the recipe);
    - why `--bounds` is required: the sibling rule would otherwise resolve the 12-dimensional spontaneous box;
    - the masked-probe criterion reads the training run's `[chi] masked probes` run-total line (scoped to every committed batch of the simulation cache; the calibration stage logs its own, scoped to this process only), within ±12 points of 37 %. The effective sample size is the batch count: per-batch SD 12.2 points, run SD about 6.1;
    - check `$LASTEXITCODE` after each command;
    - delete the scratch stores afterwards.
  - **`## The diagnostic card`:**
    - the command lines Task 26 ran, each with every flag and with `$env:PRISM_ARTIFACTS` at the right store:
      - `sbc`, `identifiability jacobian` and `ablation` against the smoke store;
      - `identifiability laplace` against the forced store;
      - `identifiability jacobian` on the tier-1 cell, its temperature column recorded as a measurement;
      - `probes band` and `probes drive` on the master cell, in a scratch store;
      - `probes mask --prior smoke_prior` against the smoke store;
    - when to run it: after changing code under `core/diagnostics` that moves tensors;
    - what the probes should show: the configured band holds and the 0.6× control is captured; about 0.02 free-running and 0.2 captured; masking within ±12 points of 37 %, with the cycle floor the only cause.
  - **`## What a green suite does not certify`:**
    - the gpu-marked tests run on the card in every fast gate when CUDA is present, but they cover no checkpoint resume and no real bounds or cell files;
    - TorchScript's first runs are not bitwise-reproducible;
    - seeded runs are not bitwise-reproducible on CUDA or across devices;
    - SBC at smoke sizes has no power;
    - window features are checked by hand on a real screen.
  - **`## Writing tests`:**
    - use stand-ins, not real simulations, because the fast gate has little headroom; the helpers in `tests/_fixtures.py`;
    - offscreen, a widget never shown reports a 640×480 placeholder from `width()`: assert on `sizeHint()`, `minimumSizeHint()` or `maximumWidth()`, or show, resize and pump first;
    - assert order, not adjacency: `src.index(a) < src.index(b)` survives an insertion;
    - a test that reads source must keep seeing the words it looks for;
    - pin that a knob reaches its target, not just the signature;
    - slow-mark a real simulation that takes more than about a minute.

- [ ] **Step 3: Write `retrain.md` `## Decisions`.** Each decision gets its reason:
  - **The tier-1 box** (`master_tier1.txt`), with the prior choice:
    - verify in `orchestrator.build_prior` and `store.load_prior` what a prior records and refuses;
    - recommend building `retrain_prior` on the tier-1 box with `--chi`: the screen is the same as a `master.txt` prior's, and the record then names the retrain's own box;
    - temperature is an assumed input and stays in the verdict.
  - **10,000 × 2,048:** batch count against width.
  - **256 hidden features × 10 transforms from the start:** the old "capacity will not help" was measured on broken conditioning, and network settings are not in the cache identity.
  - **6 probes supplied into 12 slots**, and why the slot capacity stays.
  - **The certification round.**
  - **The verdict rule.**
  - **`core/config.py`'s defaults stay** (128 × 8, 5,000 batches): every knob is an argument. A window retrain would need Batches, Hidden features and Transforms typed into the Posterior tab on every launch.

- [ ] **Step 4: Write `## Pre-flight`, in order.**
  1. **The card smoke gate, run 4 included** (link `testing.md#the-card-smoke-gate`).
  2. **The epoch timing**, in a scratch store:

     ```
     & $py -m core prior @Box --chi --device cuda --name timing_prior
     & $py -m core train @Box --chi --device cuda --prior timing_prior --num-runs 20 --hidden-features 256 --num-transforms 10 --max-epochs 3 --checkpoint-every 0 --name timing
     ```

     - Seconds per epoch come from the fit's span in the posterior's `log.txt`. Find the record that ends generation and the one after fitting in `build_posterior` / `core.SBI.train`, and state which records bracket the fit.
     - The projection = seconds per epoch × (20,480,000 ÷ 40,960 rows) × the expected epoch count. 130 was the last full run at patience 20. Give it in hours.
     - The reference: about 92 h, meaning the 46.1 h of fitting at 128 × 8 and 10.24 M rows, scaled to 20.48 M.
     - The stop point: if the projection is more than twice the reference, the page says **stop for the owner's decision**. The threshold is the draft's; the owner confirms it at the documents review.
  3. **The recording-length check:**
     - `identifiability jacobian --chi @M @MC --t-obs 4.5 --m 32 --m-noise 128 --seed 0 --device cuda`;
     - the same on `@Box @Cell`, its temperature column recorded as a measurement;
     - quote the line to read.
  4. **Build the retrain prior, then the probe checks:**
     - `prior @Box --chi --device cuda --name retrain_prior`, with `PRISM_ARTIFACTS` unset;
     - `probes band @Box @Cell --device cuda`;
     - `probes mask @Box --prior retrain_prior --device cuda`.
  5. **Resources:**
     - read VRAM with `nvidia-smi --query-gpu=memory.used --format=csv`, never `torch.cuda.mem_get_info()`. The margin is about 1 GiB: the worst 2,048-row geometry needs about 11.44 GiB against a budget of about 12.48 GiB on an idle card, so close GPU-using desktop apps;
     - host memory is about 50 of 63 GB at the start of training, because sbi copies the data several times, so close other applications;
     - disk: the cache is about 10.3 GiB, and the `[checkpoint] writing to …` line states it;
     - the prior sweep has no out-of-memory retry.
  6. **One console-capture recipe** that keeps the live display and writes both streams to a file. Verify it on this machine with torch-free commands, `& $py -m core --help` for stdout and `& $py -m core validate` (a usage error, exit 2) for stderr, and put the verified form on the page.

- [ ] **Step 5: Write `## The run`, `## Resuming` and `## What to watch`.**
  - **The run.** Every command on the page appears in full. Each command lists the flags left at their defaults, with values: `--run-size` 0 (the hardware default, 2,048 on this card; the `[budget]` line must read 2,048 rows), `--learning-rate`, `--stop-after-epochs`, `--max-epochs`, the three Fisher flags, `--resume auto` and `--chi-k`, which does not reach training.

    ```
    & $py -m core train @Box --chi --device cuda --prior retrain_prior --num-runs 10000 --hidden-features 256 --num-transforms 10 --checkpoint-every 50 --name retrain --note "<one line>"
    & $py -m core validate @Box --chi --device cuda --posterior retrain --n-cal 2000 --cal-n-scales 200 --posterior-samples 1000 --seed <N> --name retrain_cal
    & $py -m core identifiability rotation @Box --chi --posterior retrain --name retrain_rotation
    & $py -m core ablation @Box --chi --device cuda --posterior retrain --name retrain_ablation
    ```

    Then the stratified `sbc` runs (`--chi-k-fixed 2`, `6`, `12`, and pooled) as a characterisation, not a gate, with the reason: pooled SBC can hide compensating miscalibration.
  - **Resuming:**
    - the same command plus `--resume require`;
    - the two lines it prints: `Reusing the Fisher rotation stored with the training checkpoint (…)` and `[checkpoint] resuming at batch N/10000 …`;
    - the near-miss refusal ("… ONE setting away …", ending `(--new-run)`), and when `--new-run` is right;
    - the Ctrl-C message;
    - capture the console, because a resumed run's first process's lines, including its `[fisher]` lines, exist nowhere else.
  - **What to watch**, quoting each line in the form the code prints it:
    - the `[cfg] chi` banner;
    - the `[tier1]` lines;
    - the batch-count override line;
    - `[budget]`;
    - `[fisher]`, printed only by a fresh process;
    - the per-batch masked warnings and the `[chi] masked probes` run total;
    - `[patho]`;
    - `[winsor]`, a few tenths of a percent;
    - `[mem]`;
    - `[checkpoint] complete`;
    - sbi's convergence print, which is not in `log.txt`;
    - `[prism] posterior`;
    - the loss-curve figure `figures/training_loss.png` and `loss.npz`.

- [ ] **Step 6: Write `## Gates`, `## The certification round` and `## Afterwards`.**
  - **Gates:**
    - the `[verdict]` line PASSES;
    - the previously dead input channels are revived, per `ablation` against the healthy channels of the same run;
    - the training run's masked total (`train`'s `[chi] masked probes` line, scoped to every committed batch of the simulation cache; `validate` logs its own) is within ±12 points of 37 %;
    - the eigenvalues are recorded (`identifiability rotation`);
    - the loss curve and the reading of its plateau;
    - informativeness, total and per parameter, written down as the first baseline.
    - The dropped gates, each with its reason:
      - implied temperature, which is 100 % by construction;
      - a pre-run dry run, which the tool does not have and the timing replaces;
      - old against new on one calibration set, because no old posterior exists.
  - **The certification round:**

    ```
    & $py -m core infer @Box --chi --device cuda --posterior retrain @Cell --t-obs 4.5 --name cert_parent
    & $py -m core tsnpe @Box --chi --device cuda --posterior retrain --observation <id> --directions 5 --level 0.999 --num-runs 10000 --hidden-features 256 --num-transforms 10 --checkpoint-every 50 --name cert_round
    & $py -m core validate @Box --chi --device cuda --posterior cert_round --accept-truncated --n-cal 2000 --cal-n-scales 200 --posterior-samples 1000 --seed <N> --name cert_cal
    ```

    - `<id>` comes from the `[prism] observation _unnamed__<id>` line.
    - An optional child inference uses `--accept-truncated --accept-other-observation`.
    - It passes when:
      - the `[tsnpe] truth` lines say inside for every truncated direction, as recorded in `training.truth_containment`;
      - the child's verdict passes, and its record's `accepted` shows `truncated`;
      - the widths shrink no more than the data supports.
  - **Afterwards:**
    - `artifacts note posterior retrain --note …` and `artifacts summary posterior retrain --out <path>`;
    - a table of which record, key or file answers each gate.

- [ ] **Step 7: Check every command line.**
  - For every subcommand used on both pages, run `python -m core <sub> --help` (torch-free) and confirm each flag.
  - Check the verbatim block against `CLAUDE.md` with a throwaway script in the scratchpad that extracts the block from the line `` ```powershell `` through the line ending "Delete `$S` afterwards." from each file, strips the common indentation, and compares them. That script is never a test, because no test may read `CLAUDE.md`.

- [ ] **Step 8: Run the focused tests.**
  - Command: `python -m pytest tests/test_docs.py tests/test_source_hygiene.py -q`.
  - Expected: everything passes.

- [ ] **Step 9: Commit.**
  - `git add docs/guide/testing.md docs/guide/retrain.md`
  - `git commit -m "docs: the testing page and the retrain runbook" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"`
  - Do not run the fast gate. The controller runs the one-process fast gate (`python -m pytest -m "not slow" -q`, in the background, logged) at the same moment as this task's review.

---

### Task 37: The documents review

**Files:**
- Create (gitignored workspace, Markdown only): `.superpowers/sdd/2026-09-25-documentation-and-retrain-readiness/only-record-checklist.md`, and `.superpowers/sdd/2026-09-25-documentation-and-retrain-readiness/docs-review/lens-{accuracy,completeness,paths}.md`, `verify-<lens>.md`, `synthesis.md`, `recheck.md`; gate log `.superpowers/sdd/2026-09-25-documentation-and-retrain-readiness/gates/t37-fast.log`
- Modify: `docs/guide/*.md` and `README.md` (the fix dispatch); `docs/STATE.md` (the "Open, with no piece owning them yet" list under "Owed" item 7)

**Interfaces:**
- Consumes:
  - the Docs registry: the ten pages `docs/guide/{README,getting-started,window,command-line,recordings,science,architecture,rules-and-traps,testing,retrain}.md`; every page's first line after its title is `Checked against commit <hash>.`; `command-line.md` headings are `## <subcommand>` / `### <subcommand> <mode>`;
  - `tests/test_docs.py` (`test_every_relative_link_in_the_guide_and_readme_resolves`, `test_the_command_line_page_names_every_subcommand_mode_and_flag_and_no_other`, `test_the_command_line_page_states_each_default_as_the_help_does`) and `tests/test_source_hygiene.py`;
  - Task 26's `recipe_equal.py` and its recipe block;
  - `core.tool.help_defaults.default_for`.
- Produces: the closed only-record checklist (ledger); STATE's open list with the handoff's open engineering items; pages ready for Task 38 to retire the handoff.

**Binding (read before starting):**
- Spec §6.7: three lenses, each followed by an adversarial verifier, **before the handoff moves**. Then one fix dispatch and the checklist closed. Spec §6.6 step 1 gives the checklist's minimum contents; spec §6.1 gives the page rules:
  - a page cites code by module and function name, never by line number;
  - no page cites a working document (the handoff, STATE, CLAUDE.md, specs, plans, ledgers, the walkthrough checklist) or their labels;
  - summary features are named by full label (`A1_mean`, not A1);
  - the transposed table, "k is unmeasured", the five false lines and the stale memory rows are never carried.
- **Recipe equality** (spec §6.6): CLAUDE.md's block is canonical, and STATE's and `testing.md`'s are verbatim. No test can read CLAUDE.md or STATE, so this review checks it.
- **Stamps**: a page's stamp `<hash>` is valid when `git diff --quiet <hash> HEAD -- core tests conftest.py pytest.ini requirements.txt run.bat run.sh ':!tests/test_docs.py' ':!tests/test_source_hygiene.py'` exits 0 (the code has not moved since; the two suites that only read the pages are excluded, since Tasks 27 and 29 change them after pages are stamped). Otherwise the accuracy lens re-checks that page against HEAD and the fix restamps it.
- Lenses and verifiers are read-only: no pytest, no GPU. They may run torch-free scripts under `C:\Users\J\AppData\Local\Temp\prism-piece6\docs-review\`, including parse checks through `core.tool.build_parser().parse_args([...])` (torch-free) and `python -m core <sub> --help`. `python` means `C:\Users\J\anaconda3\envs\biophys-env\python.exe`.
- The fix dispatch edits pages only.
  - Everything it writes in `docs/guide/` and `README.md` obeys the reference rule and names features by full label.
  - It runs `python -m pytest tests/test_docs.py tests/test_source_hygiene.py -q` after its edits.
  - It commits: `git add` each file explicitly, a short subject, the body line `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`, never amended.
  - The one-process fast gate starts at the same moment as the scoped re-check. Never two pytest processes at once, and no page edit while the gate runs (`tests/test_docs.py` reads the pages).

- [ ] **Step 1: Preconditions.**
  - The ledger shows Tasks 27–36 reviewed and gated green, including both `tests/test_docs.py` tests and `tests/test_source_hygiene.py` over the ten pages.
  - `git status --porcelain` prints nothing.
  - List each page with its stamp, and record which stamps are still valid by the rule above.

- [ ] **Step 2: Build the only-record checklist.**
  - `only-record-checklist.md` is one table: `| # | item | source (handoff lines / recon chunk note) | home | status | evidence |`. Status is one of: `found` (evidence: `page.md#heading`), `appended` (evidence: the STATE line), or `dropped` (evidence: the reason).
  - Seed it from `recon/handoff-map.md`: every "only record" item and every open item in the chunk notes A–G, and every section marked load-bearing. It must hold at least the items below.
  - **Only-record facts, each to a reader page:**
    - `rules-and-traps.md`:
      - the TorchScript warm-up trap (:2653-2668);
      - the comment policy's never-delete list and "facts live in code, no pointers" (:2332-2333, :2246-2256);
      - the allocator variable's real name, `PYTORCH_CUDA_ALLOC_CONF`, never the ignored `PYTORCH_ALLOC_CONF` (:6193-6194 is the false line);
      - the facade re-import's silent failure mode;
      - the pre-flight must share the run's config source (:5666-5677);
      - the narrowing-round rules, enforced against prose-only (chunk D note 3);
      - every still-applies trap of chunk B's list, rewritten without labels;
      - the rest of the M group (the shim header, the gating table with the TSNPE tab, six tabs, `value_or_none` over `FloatField.value()`);
      - the §3 contracts (the one-bounds-file rule is now sibling, else `master.txt`).
    - `testing.md`:
      - the offscreen-geometry test constraint (:2930-2932);
      - "assert order, not adjacency".
    - `architecture.md`:
      - the production-denominator ranking and the 7.9× graph measurement (:2553-2566, :2537-2539);
      - the prior sweep has no out-of-memory retry (:6217-6219);
      - qfluentwidgets rejected as GPLv3 (:6919-6920);
      - the declined hard process cancel (:6979);
      - the WDDM pressure-test table (:4593-4598).
    - `window.md`:
      - the prior picker loads nothing until "Build / Load prior" is pressed;
      - the §2.3 knob table, with its field, flag, argument and default;
      - `CAL_N_SCALES` as a measurement, not a speed dial;
      - six inference tabs;
      - the capability matrix with CrossVal, the Artifacts screen and a command-line column.
    - `recordings.md`:
      - the training `T_obs` ceiling of ~26.9 s at t_scale 3.73 (:6552-6559);
      - the jacobian recipe (matched `T_obs`, T* ≈ 2.93 s);
      - the two tier-1 input files;
      - `probes` as the tool for re-measuring a cell.
    - `science.md`:
      - C-6's rationale that masked chi rows are genuine oscillators (:2208) and the Ω₀ physics of masking;
      - why `CHI_MAX_CYCLES` is 20 and why the band is (0.03, 0.3);
      - the span-not-count objective and the rejected ceiling-bounded placement;
      - the ±12-point error model, with its why, written once;
      - PPC masking as a posterior diagnostic;
      - chi's payload and "k with x_scale is unbreakable";
      - SBC reading conventions (KS over c2st ranks with the ~0.58 floor; pooled histograms; :6966-6968);
      - best fits are not recovery and flat SBC is not informativeness (:5679-5692);
      - n, temp and tau_c enter only through noise amplitudes (:5648-5653);
      - the graphs on/off physics check (:5533-5536);
      - batch count against width as statistics (:5280-5283);
      - k_B T = 4.1419 pN nm, and the tier-1 derived f_scale range of 0.2–12426 pN;
      - stratified plus pooled SBC;
      - the August findings in the correct orientation only;
      - the `dt_nd_min` note;
      - open questions: the tier ladder and the fix-T 12-dimension alternative, M-replicate, the thermal tail, the strata test, step and intermodulation observables, K against `chi_k_pad`, pooling rows across rounds, the batch-by-scale t_scale override, changes to the latent Fisher rotation, the log-sampling decision (`_global_map` samples the linear box), and the owner's science items from STATE.
    - `retrain.md`:
      - close GPU-touching desktop applications; the 11.44 of 12.48 GiB worst geometry; the ~1 GiB VRAM margin at 2048 rows;
      - the measured costs (generation ~20 % and flow ~80 %; 11.4 h + 46.1 h at 5,000 × 2,048, 130 epochs; 8.1 h to re-simulate 5,000 batches; ~31 h of generation for 10,000);
      - the watch-lines and gates harvested in chunk D note 5;
      - the trained-network comparison the old gate never made (met by the ablation revival gate, or dropped with a reason);
      - the capacity contradiction settled by the 256 × 10 decision;
      - resume semantics rewritten for the near-miss refusal and `--resume`/`--new-run`.
    - `command-line.md`:
      - `smoke --t-obs` defaults to 1.0 s, below T* ≈ 2.93 s;
      - the scripts-to-subcommands name map as one history paragraph.
  - **Open engineering items, appended to STATE's open list:**
    - the FDT sanity checks keep a view of the whole solution (`core/FDT/sanity.py`, `check_ensemble_convergence` and `check_psd_window`: `x_steady = sol[0, 0, :, burn_idx:]` with no `del sol`);
    - `decorrelate`'s per-call re-derivation of subs, `n_fine`, `t_fine`, `n_segs` and `base_inits`;
    - the open-performance table and its blockers (probe concatenation gated on the force tensor's 4× transient, strided output, PPC binning, `gen_stats` dedup);
    - the `vt.py` rename and the directory casing;
    - the orphan `.claude/worktrees/` copies;
    - the FDT and reduction plots saving with `plt`/`fig.savefig` instead of `visualizers.save_figure`;
    - `file:line` citations back in code comments (`core/artifacts/report.py`, `core/cli.py`, `core/gui/screens/artifact_screen.py`);
    - `training_checkpoint.checkpoints_using_prior`, possibly dead code (spec §1.3);
    - the git stash-versus-checkout process note (home: CLAUDE.md or STATE, as the owner prefers).
  - **Never carried (the accuracy lens confirms each is absent):**
    - the transposed direction table (:1404-1416);
    - "k is unmeasured" (:3333-3335, :3621-3625);
    - direction 4 read as t_scale strata evidence (:3450);
    - the false lines :7-9, :49, :51, :53-54, :74-140, :76, :148, :157-158, :185-189, :190-191, :547-553, and the env-var recipes (:584-585, :697-710, :724);
    - the stale memory rows (:2614-2620);
    - `PYTORCH_ALLOC_CONF`;
    - the 1.8 pp binomial error (:5800, :5909);
    - "Nothing in §11 has been implemented" (:5179);
    - Appendix A's reversed decisions;
    - the handoff's test totals;
    - the "29 of 42" ablation baseline as applicable;
    - "more capacity will not help" as settled.

- [ ] **Step 3: Dispatch the three lenses in parallel.** Each writes numbered findings to `docs-review/lens-<name>.md` in this form: `[Critical|Important|Minor] page#heading — the claim — why it is wrong or missing — evidence`.
  - **accuracy.** Every checkable fact on every page against the code at its stamp:
    - modules, functions, flags, defaults (through `default_for` and the stage signatures), numbers and commands;
    - every command line in the pages parses: a torch-free script feeds each one to `core.tool.build_parser().parse_args`;
    - the stamps are valid;
    - the never-carried list is absent;
    - no line numbers, no working-document citations, features named by full label;
    - the recipe equality: `python C:\Users\J\AppData\Local\Temp\prism-piece6\recipe_equal.py CLAUDE.md docs/STATE.md docs/guide/testing.md` must print `recipe blocks identical`.
  - **completeness.**
    - Every section `recon/handoff-map.md` marks load-bearing has a home or a recorded reason to drop it.
    - Every checklist row can be given a status: the lens proposes one, with evidence.
    - The four reading paths on `docs/guide/README.md` each reach every page they need.
  - **reading paths.** A fresh reader given only `docs/guide/` (and allowed `--help`) writes down the exact commands and answers for each job, noting every point where the pages left it stuck:
    - a lab member checks a preparation with `probes band`, `probes drive` and `probes mask`, then runs an inference on a recording (`infer --spont …` with its forced or chi recordings);
    - a successor finds where the narrowed-posterior load refusal and the near-miss refusal are raised (module and function) and why;
    - a reviewer reproduces a calibration from its record: which manifest fields name the posterior, `n_cal`, `cal_n_scales` and the seed, and the `validate --seed N` command;
    - the owner-scientist's path ends at `retrain.md`, and every command it names appears in `command-line.md`.

    The verifier then checks each command parses and each answer against the code.

- [ ] **Step 4: Dispatch one adversarial verifier per lens.** Each marks every finding CONFIRMED, PLAUSIBLE or REJECTED, with the code or page quoted, and writes `docs-review/verify-<lens>.md`.

- [ ] **Step 5: Synthesize.** `docs-review/synthesis.md` holds:
  - the ranked fix list (page, heading, the corrected fact and its source in code);
  - the checklist rows still open, with the page each needs;
  - the parked items, with reasons.

- [ ] **Step 6: One fix dispatch.**
  - One implementer takes the fix list and the page rules of this task's Binding, verbatim.
  - It edits pages and restamps only pages whose stamp was invalid.
  - It runs `python -m pytest tests/test_docs.py tests/test_source_hygiene.py -q`. Expected: all passed.
  - It commits `git add <each page>`, then `git commit -m "docs: guide fixes from the documents review" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"` (or one commit per page group if the fixes are large).

- [ ] **Step 7: STATE's open list.**
  - The controller appends one bullet per open engineering item from Step 2 to "Open, with no piece owning them yet". Each bullet is marked *From the handoff's retirement*, states the item in a sentence and names its file.
  - `git add docs/STATE.md`, then `git commit -m "docs: STATE takes the handoff's open engineering items" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"`.

- [ ] **Step 8: Close the checklist.**
  - Every row has status `found`, `appended` or `dropped`, with its evidence.
  - Every science idea is confirmed under `science.md`'s open-questions heading (quote the heading).
  - Nothing still says "open".

- [ ] **Step 9: Re-check and gate, started at the same moment.**
  - Start in the background (Bash tool):

    ```bash
    cd /c/Users/J/PycharmProjects/PRISM && "/c/Users/J/anaconda3/envs/biophys-env/python.exe" -m pytest -m "not slow" -q --durations=15 > .superpowers/sdd/2026-09-25-documentation-and-retrain-readiness/gates/t37-fast.log 2>&1; echo "exit=$?" >> .superpowers/sdd/2026-09-25-documentation-and-retrain-readiness/gates/t37-fast.log
    ```

  - Dispatch one re-checker on the fix commits and the closed checklist, whose verdict goes to `docs-review/recheck.md`.
  - Re-run `recipe_equal.py` on the three files. Expected: `recipe blocks identical`.
  - The gate passes when the log ends `exit=0`, the count equals the last gate's (documents only), and the warnings are unchanged.
  - The ledger records the result.

---

### Task 38: Retire the handoff

**Files:**
- Move: `PRISM_HANDOFF.md` → `archive/PRISM_HANDOFF.md` (on disk), then `git rm --cached PRISM_HANDOFF.md`
- Modify: `CLAUDE.md`, in five places:
  - "Rules that are load-bearing", the bullet beginning "The science guardrails are in `PRISM_HANDOFF.md` §11.6" (line ~154);
  - the same section's bullet list, which gains one bullet;
  - "Where things are": the `.superpowers/sdd/` bullet (line ~169), the `tests/` bullet beginning "the nineteen suites" (line ~174), and the `core/tool/` bullet (line ~183);
  - the Tests section's first bullet, the slow set's membership (line ~38-44);
  - the line after the recipe block that Task 26 added.
- Modify: `docs/checklists/display-walkthrough.md` (header lines 3-5; a new section at the end)
- Modify: `docs/STATE.md` ("Owed" item 7's piece-6 sub-bullets)

**Interfaces:**
- Consumes:
  - the pages under `docs/guide/` reviewed by Task 37 (`rules-and-traps.md`, `science.md`, `testing.md`);
  - Task 26's recipe block and the sentence after it;
  - T14's "(assumed input)" and "T (assumed input, K)" marks and the `T_obs` wording;
  - T15's `"[verdict] "` line;
  - T16's `"[tsnpe] truth "` lines;
  - message 7 as rewritten in `core/artifacts/manifest.py` (the posterior amortization `ManifestError`).
- Produces: `<A>`, the archiving commit, named in CLAUDE.md, the checklist header and STATE; rows F1–F5, which Task 39 hands to the owner.

**Binding (read before starting):**
- Spec §6.6 steps 2–3 and spec §9.
- Archiving is a move on disk into the gitignored `archive/`, then `git rm --cached`, **never `git mv`**, which would re-create a tracked-but-ignored file. The closed specs and plans under `docs/superpowers/` are not edited: their handoff citations resolve through git history.
- The archiving commit removes the file from the tree, so it is read back with `git show <A>^:PRISM_HANDOFF.md`, the parent of `<A>`. Spec §6.6 writes `<A>:`; Task 39 records this as a deviation.
- Two commits:
  - **A** holds the `git rm --cached` together with the CLAUDE.md repointing, so no commit has CLAUDE.md pointing at a missing file.
  - **B** holds the checklist and STATE, which name `<A>`.
  - Each: `git add` explicitly, a short subject, the body line `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`, never amended. The owner pushes.
- CLAUDE.md, STATE and the checklist are working record and may cite labels. No test or code reads them.
- Row F3 stays. Its route is real: `ArtifactStore._entries` turns a `ManifestError` into `f"unreadable manifest: {e}"`, and the Artifacts screen's detail pane shows `f"reason:  {s.reason}"` for an incomplete row (`core/gui/screens/artifact_screen.py`, `_detail_text`).
- Walkthrough columns: `| # | surface | do | expect | date | result |`. The owner fills date and result, so both start empty.

- [ ] **Step 1: Preconditions.**
  - Task 37 is closed: every checklist row has a status.
  - `git status --porcelain` prints nothing.
  - `Test-Path archive/PRISM_HANDOFF.md` prints `False`. If it prints `True`, stop and ask the owner; overwrite nothing.
  - `git check-ignore -v archive/PRISM_HANDOFF.md` prints `.gitignore:1:/archive/`.

- [ ] **Step 2: No live reference remains.**
  - `git grep -n "PRISM_HANDOFF"` lists only `PRISM_HANDOFF.md` itself (it names itself), `CLAUDE.md` (the line this task repoints), `docs/STATE.md`, `docs/checklists/display-walkthrough.md` and files under `docs/superpowers/`.
  - `git grep -n -i "handoff" -- core tests conftest.py README.md requirements.txt pytest.ini run.bat run.sh docs/guide` prints only `tests/test_source_hygiene.py`'s matcher sample `test_the_handoff_is_gone` (a string literal the scan spares, since an underscore is no word boundary). The source scan pins the rest.

- [ ] **Step 3: Move, then untrack:**

  ```powershell
  Move-Item -LiteralPath PRISM_HANDOFF.md -Destination archive/PRISM_HANDOFF.md
  git rm --cached PRISM_HANDOFF.md
  ```

  Expected:
  - `Test-Path archive/PRISM_HANDOFF.md` is `True`;
  - `git ls-files PRISM_HANDOFF.md` prints nothing;
  - `git status --porcelain` shows `D  PRISM_HANDOFF.md` and nothing under `archive/`.

- [ ] **Step 4: The CLAUDE.md edits** (spec §6.6 step 3):
  1. Replace the bullet "The science guardrails are in `PRISM_HANDOFF.md` §11.6 (TSNPE) and the traps in §5; the handoff is being split into `docs/` by piece 6 but is still the reference until then." with: "- The narrowing-round safety rules (where each is enforced and the test that pins it) and the traps that still bite are in `docs/guide/rules-and-traps.md`; the reasoning behind the science settings is in `docs/guide/science.md`. The handoff they were written from is archived (see `docs/STATE.md`)."
  2. Add after the Refusal bullet: "- Nothing shippable cites the working record (piece 6, H4): no file under `core/` or `tests/`, nor `conftest.py`, `README.md`, `requirements.txt`, `pytest.ini`, `run.bat`, `run.sh` or `docs/guide/`, may cite the handoff, `docs/STATE.md`, this file, the specs or the plans, or their labels (piece numbers, "§N", decision, ruling and review ids, "Task N", trap ids, walkthrough row ids). A provenance tag is deleted; a reason given by reference is written out in words. `tests/test_source_hygiene.py` enforces it; a legitimate new token shaped like a label (a paper's section, a new feature id) gets an allowlist entry there in the commit that introduces it. The working record may cite code and each other; no test or code reads it."
  3. In the `.superpowers/sdd/` bullet, after "piece 5 `2026-09-22-secondary-analyses`", add: ", piece 6 `2026-09-25-documentation-and-retrain-readiness`".
  4. Change "the nineteen suites" to "the twenty-two suites". Add `test_source_hygiene.py` the reference scan over the shippable files, `test_tier1.py` the tier-1 box's path, and `test_docs.py` the guide's links and the command-line page against the parser. Change "`test_diagnostics.py` the five diagnostics'" to "`test_diagnostics.py` the diagnostics' (the five and the three `probes` modes)".
  5. In the `core/tool/` bullet, change "the diagnostics `sbc identifiability ablation`" to "the diagnostics `sbc identifiability ablation probes` (band/mask/drive)".
  6. Add a new bullet after the checklist bullet: "- `docs/guide/` — the reader pages, starting at its `README.md`: getting started, the window, the command line, bringing recordings, the science, the architecture, the rules and traps, testing, and the retrain runbook. They cite no working document (the source scan walks them), and each opens with the commit it was checked against."
  7. Slow-set membership in the Tests section's first bullet: "The slow set is three tests: …" becomes five tests. Name each of the two new ones exactly as `python -m pytest -m slow --collect-only -q` prints it (the tier-1 tiny `smoke` in `tests/test_tier1.py` and the probes twin-cell check in `tests/test_diagnostics.py`), with "(timed at the final gate)" in place of their seconds. Task 39 writes the measured times.
  8. The line after the recipe block becomes: "  The last result, with its stage timings, is in `docs/STATE.md`'s gate table; that table and `docs/guide/testing.md` carry this block verbatim."
  9. Verify, without editing, that the card recipe already has run 4 and that `probes` is in the diagnostic card (Task 26). Run `python C:\Users\J\AppData\Local\Temp\prism-piece6\recipe_equal.py CLAUDE.md docs/STATE.md docs/guide/testing.md`. Expected: `recipe blocks identical`.

- [ ] **Step 5: Commit A.**
  - `git grep -n "PRISM_HANDOFF" -- CLAUDE.md` prints nothing.
  - `git add CLAUDE.md` (the removal is already staged), then `git commit -m "Retire the handoff to archive/; CLAUDE.md repointed" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"`.
  - Record `<A>` = `git rev-parse HEAD`. `git show <A>^:PRISM_HANDOFF.md | Select-Object -First 3` prints the handoff's first lines.

- [ ] **Step 6: The checklist.**
  - Replace header lines 3-5 with: "Features the headless suites cover by wiring only, never by a human looking at the screen. Each row is checked off on a real display, with the date and what was seen. Seeded 2026-09-10 from the handoff document's §6 (the ONGOING row and row 3); that document was retired to the gitignored `archive/` at `<A>` and is read back with `git show <A>^:PRISM_HANDOFF.md`."
  - Append the section below. In F3's expect cell, `<message>` is the posterior amortization message from `core/artifacts/manifest.py` as it stands now, copied exactly (it no longer ends in a label).

  ```markdown
  ## Piece-6 GUI checks (retrain readiness in the window; design spec §9)

  The rows piece 6 creates (H12): only what the piece changes in the window. Rows 1–20, A1–A9, B1–B8,
  C1–C11, D1–D17 and E1–E17 are not edited; each is a dated record of what was seen on the code as it
  then was. The letter F is simply the next after E, independent of the piece's decisions H1–H13.

  How to read the rows below:

  - **Setup, before F1.** Launch `run.bat` and open Home → "Parameter Inference". On the Config tab
    pick the Nadrowski model, tick "Multi-frequency χ(ω) conditioning" and press "Apply model &
    options". On the Prior tab choose `Resources/Bounds/nadrowski/master_tier1.txt` as the bounds file
    (the tier-1 box: temperature `T`, 280–310 K, is inferred and the force scale is derived from it)
    and press "Build / Load prior". On the Posterior tab set 'Batches' to 4 and 'Max rows per batch
    (0 = auto)' to 32 and press "Train / Load posterior". A few minutes; nothing here is a check.
  - **Order.** F1 and F2 use the model the setup builds, F5 uses F2's observation, and F3 edits F5's
    narrowed posterior by hand: run F1, F2, F4, F5, then F3.
  - **F3 edits a record.** Copy its `manifest.json` outside `Artifacts/` first and put it back
    afterwards; the record is unusable while the edit stands.
  - Labels draw the part after an underscore as a subscript ('T_obs' reads T with "obs" below it).
    At this tiny size PASS or FAIL are both results; the rows check what is shown, not the verdict.

  | # | surface | do | expect | date | result |
  |---|---|---|---|---|---|
  | F1 | The Validate tab ends with the calibration verdict | On the Validate tab set 'Calibration datasets' to 40, leave the operating points at their default, and press "Run calibration". | The log pane shows the verdict record, whose head line begins `[verdict] ` and says PASS or FAIL, with no dialog; the run's last line is the tab's own closing line, beginning 'Calibration verdict PASS' or 'Calibration verdict FAIL' and naming the seed the tab drew. It judges every inferred parameter — all 13, temperature included — each KS p against the threshold 0.05 ÷ 13, temperature marked "(assumed input)"; the joint coverage test is judged against 0.05; the caveat that t_scale's test rests on the calibration's operating points is printed with it. | | |
  | F2 | Temperature is marked an assumed input in an inference | On the Infer tab's simulated page pick `master_spont_tier1.txt`, set 'T_obs (s)' to 4.5 and press "Run inference". | The posterior summary in the log pane marks temperature as an assumed input, and the corner plot labels temperature "T (assumed input, K)"; the other twelve parameters read as before. | | |
  | F3 | A reworded store message reaches the window as a plain sentence | In Explorer, copy F5's narrowed posterior's `manifest.json` (`Artifacts/posteriors/<name>__<id>/`) outside `Artifacts/`; in the original change `"amortized": false` to `"amortized": true` and save. On Home → Artifacts choose the posteriors kind and select that row. Afterwards put the copy back. | The row is listed as an incomplete directory, which nothing can load; the detail pane's `reason:` line reads "unreadable manifest: <message>" — a plain sentence with no parenthesised label. With the copy put back, the row reads as a narrowed posterior again. | | |
  | F4 | Tooltips: the patience measurement, and T_obs where T meant the recording length | On the Posterior tab hover the help badge beside 'Early-stop patience' in the "Density estimator" group, then the badge beside 'Batches', and read the "Training budget" group's lines; on the Validate tab read the operating-points label and hover its badge. | The patience tooltip gives the measurement — the run stopped at epoch 130 on a patience of 20, its best validation loss at epoch 110 — and names no incident date. The Batches tooltip, the budget group's diversity line and the Validate tab's operating-points label and tooltip all say "(t_scale, T_obs)", never "(t_scale, T)". | | |
  | F5 | A narrowing round reports truth containment per direction | On the TSNPE tab pick F2's observation in 'Observation', leave 'HPD level' and 'Directions truncated' at their defaults, set 'Batches' 4 and 'Max rows per batch (0 = auto)' 32, and press "Run TSNPE round". | Before training, the log pane reports for every truncated direction whether the loaded cell's truth lies inside the region — one `[tsnpe] truth ` line per direction, reading inside or outside (an outside direction also keeps the "GROUND TRUTH lies OUTSIDE" warning); the round then trains and writes a narrowed posterior. | | |
  ```

- [ ] **Step 7: STATE.**
  - Under "Owed" item 7's piece-6 bullet, strike through the carried sub-bullets that the documents meet:
    - the `docs/` split and its false lines ("met: the reader pages under `docs/guide/`; `PRISM_HANDOFF.md` archived at `<A>`, read back with `git show <A>^:PRISM_HANDOFF.md`");
    - the README reference;
    - the memory figures (the sanity half is now on the open list);
    - the rest of group M (`rules-and-traps.md`).
  - Keep the Provenance sub-bullet's closure from Task 19.
  - Item 7 as a whole closes at Task 39.

- [ ] **Step 8: Focused tests and commit B.**
  - Run `python -m pytest tests/test_source_hygiene.py tests/test_docs.py -q`. Expected: all passed. No test reads these three files, and the run confirms nothing shippable moved.
  - `git add docs/checklists/display-walkthrough.md docs/STATE.md`, then `git commit -m "docs: walkthrough rows F1-F5; STATE names the handoff's archive commit" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"`.
  - The controller starts the one-process fast gate in the background (log `gates/t38-fast.log`) at the same moment as this task's review. Expected: the count and warnings equal Task 37's gate.

---

### Task 39: Final gates and the closing documents

**Files:**
- Modify: `docs/STATE.md`:
  - the top paragraph ("**Last updated:** …");
  - "Where things stand";
  - "Owed" item 7 (the piece-6 bullet, the open list) and item 8;
  - "Last gate": new rows after Task 26's card-run row;
  - "Decisions log": a new entry after the "2026-09-25/28" piece-6 entry.
- Modify: `CLAUDE.md` (the Tests section's first bullet: timings and slow set; the gpu-marked count in the bullet beginning "A green suite does not certify the GPU path")
- Modify: `docs/superpowers/specs/2026-09-25-documentation-and-retrain-readiness-design.md` (§11 "Deviations (filled during execution)", the table at the end)
- Gate logs: `.superpowers/sdd/2026-09-25-documentation-and-retrain-readiness/gates/final-collect.log`, `final-fast.log`, `final-slow.log`

**Interfaces:**
- Consumes:
  - `<P>` (the plan commit);
  - Task 25's parked list and the ledger's deviation rows;
  - Task 26's card row and `<K>`;
  - Task 37's checklist;
  - Task 38's `<A>` and rows F1–F5;
  - T15's `calibration_seed` / `CALIBRATION_TAG` (named in the seed entry);
  - T22–T24's `probes` family (named in the D11 entry).
- Produces: the closed piece. STATE says what landed and what is next; CLAUDE.md's counts match the measured ones; spec §11 is filled; the owner has rows F1–F5.

**Binding (read before starting):**
- Spec §8, "Final": the fast gate on a quiet machine, then the slow set, then STATE and CLAUDE.md from the measured numbers. STATE's decisions log records D11's narrowed reading and the seed widening (spec §1.2).
- The card run stands unless a later change touched code that moves tensors, in which case the affected lines are re-run.
- A quiet machine means no GPU job, no other pytest, no source edits, and nothing heavy on the desktop.
- One pytest process at a time: collect, then fast, then slow, each in the background with its log. A tool call cannot hold a run longer than ten minutes.
- Budgets:
  - fast gate 16 minutes, one process;
  - tests: about 80 ± 25 new on the 968 collected at `7e51275`, so a band of 1,023–1,073;
  - warnings: 181 at piece 5, and any change is explained.
  - An overrun is recorded in STATE's row, not hidden.
- Working-record files may cite labels. Commits: `git add` explicitly, short subjects, the body line `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`, never amended. The owner pushes.

- [ ] **Step 1: Preconditions and the card judgement.**
  - `git status --porcelain` prints nothing; `nvidia-smi --query-gpu=memory.used --format=csv` shows the desktop baseline only.
  - Record `<Z>` = `git rev-parse HEAD`.
  - Run `git diff --name-only <K>..<Z> -- core conftest.py`. Expected: empty (Tasks 27–38 changed documents and `tests/` only). If a `core/` file is listed, judge whether it creates or moves a tensor. If it does, re-run the affected card legs from Task 26 and record them.
  - Save the real `Artifacts/` listing: `Get-ChildItem Artifacts -Recurse -Directory | Select-Object -ExpandProperty FullName > C:\Users\J\AppData\Local\Temp\prism-piece6\artifacts-before.txt`.

- [ ] **Step 2: Count.** Run `"/c/Users/J/anaconda3/envs/biophys-env/python.exe" -m pytest --collect-only -q > .superpowers/sdd/2026-09-25-documentation-and-retrain-readiness/gates/final-collect.log 2>&1` (Bash tool, repository root). Record `N` from the last line ("N tests collected"), N − 968, and whether it is inside 1,023–1,073. Also run `python -m pytest -m gpu --collect-only -q` and record the gpu-marked tests by file.

- [ ] **Step 3: The final fast gate, on a quiet machine** (Bash tool, `run_in_background: true`):

  ```bash
  cd /c/Users/J/PycharmProjects/PRISM && "/c/Users/J/anaconda3/envs/biophys-env/python.exe" -m pytest -m "not slow" -q --durations=15 > .superpowers/sdd/2026-09-25-documentation-and-retrain-readiness/gates/final-fast.log 2>&1; echo "exit=$?" >> .superpowers/sdd/2026-09-25-documentation-and-retrain-readiness/gates/final-fast.log
  ```

  Pass:
  - `exit=0`; 0 failed;
  - passed + skipped + 5 deselected = N; 1 skipped (the display-marked test);
  - the warnings count recorded against 181;
  - the wall time recorded against 16 minutes;
  - afterwards `git status --porcelain` is empty, `Artifacts/`'s listing equals `artifacts-before.txt`, there is no `sbi-logs/`, and `Resources/Models` holds only `SHM.json` and `SHM2.json`.

- [ ] **Step 4: The slow set of record, after Step 3 exits** (Bash tool, `run_in_background: true`):

  ```bash
  cd /c/Users/J/PycharmProjects/PRISM && "/c/Users/J/anaconda3/envs/biophys-env/python.exe" -m pytest -m slow -q --durations=5 > .superpowers/sdd/2026-09-25-documentation-and-retrain-readiness/gates/final-slow.log 2>&1; echo "exit=$?" >> .superpowers/sdd/2026-09-25-documentation-and-retrain-readiness/gates/final-slow.log
  ```

  Pass: `exit=0` with 5 passed. Record each test's seconds and the total (piece 5 was 38 min 6 s for three). Afterwards run the same clean-tree checks as Step 3.

- [ ] **Step 5: STATE's "Last gate" rows** (after Task 26's card row), with every `<…>` filled:
  - `| `pytest --collect-only -q` after piece 6 | <date> at `<Z>`: **N** collected (the fast gate's <p> passed and 1 skipped, plus the 5 slow tests), N − 968 more than the 968 at `7e51275`, against design spec §7's budget of about 80 ± 25 added (<inside / above by X, with the reason>). `tests/test_*.py` holds twenty-two suites (new: `test_source_hygiene.py`, `test_tier1.py`, `test_docs.py`) |`
  - `| **final fast gate, piece 6**, ONE process, `pytest -m "not slow" -q --durations=15` | <date> at `<Z>`, on a quiet machine: **<p> passed, 1 skipped**, 5 deselected, **<w> warnings**, **<m> min <s> s** (<inside/over> the 16-minute target), exit 0. Tree clean; the real `Artifacts/` gained nothing; no `sbi-logs/`. Warnings 181 → <w>: <explanation>. Slowest: <three with seconds>. Every task had its own one-process gate, recorded in the execution ledger |`
  - `| slow set of record, `pytest -m slow -q --durations=5` (piece 6) | <date> at `<Z>`: **5 passed**, <d> deselected, <w> warnings, **<total>**, exit 0. <each test and its seconds>. Tree clean afterwards |`
  - `| GPU card run, piece 6 — stands | `git diff --name-only <K>..<Z> -- core conftest.py` <empty / lists …, judged …>: the card run at `<K>` is the piece's card measurement |`

- [ ] **Step 6: STATE's narrative.**
  - Top paragraph: "**Last updated:** <date>. **Piece 6 is DONE**", naming:
    - the commit range `<P>`..`<Z>` with its count (`git log --oneline <P>^..<Z>`), unpushed;
    - what landed, part by part: the reference scan and the cleanup; the help tidy; the readiness fixes; `probes band|mask|drive`; `docs/guide/` and the README summary; the handoff archived at `<A>`;
    - the whole-code review's counts (Critical/Important/Minor, fixed/parked) and the card run;
    - the documents review;
    - "**Next:** the owner runs rows F1–F5 of `docs/checklists/display-walkthrough.md`; then the retrain ("Owed" item 8) from `docs/guide/retrain.md`."
  - "Where things stand": a "**Piece 6 is DONE**" bullet in the shape of piece 3's, with the spec and plan paths, H1–H13 and the §11 row count.
  - "Owed" item 7: the piece-6 bullet is marked DONE with each carried item struck and its resolution, and the item as a whole is closed. The open list's *Provenance* item is struck ("closed by piece 6, H13"). The Task 25 and Task 37 parked items are added with their reasons and costs. The piece-3 item "a resumed run records its Fisher settings as not run" gains "the eigenvalues now carry over; the settings still do not".
  - Item 8, the retrain, is marked next.

- [ ] **Step 7: The decisions log entry** "**<dates>** — piece 6's execution" contains, word for word:
  - "- **D11, read narrowly (H8), restated at the piece's close.** D11 stays a standing refusal for everything that trains or infers: no flag, box or argument gives a training run, a calibration, an inference or a narrowing round a non-default chi band or drive amplitude; a change is an edit to `core/config.py`. The `probes` family alone takes its own frequency, drive and length grids, because they are measurements: they reach the simulator only as the drive builder's frequency and amplitude inside `core/diagnostics/probes.py`, never through a `SimConfig` field or `gen_training_data`'s `chi_f0`/`chi_freq_bounds`, and every record states the configured band and drive it judged; the measure-only tests in `tests/test_diagnostics.py` and `tests/test_tool.py` pin this. A later piece must not read this as licence for a chi override anywhere else. D12 is unchanged."
  - "- **The seed reading widened again.** Piece 2 recorded that only `smoke` and the diagnostics take `--seed`; piece 5 widened it for `fdt` and `crossval`. Piece 6 adds `validate --seed`, and the Validate tab draws one per run (no Seed box, nothing remembered): the calibration set is drawn inside `core.rng.seeded` from `calibration_seed(seed)`, a `numpy.random.SeedSequence([seed, CALIBRATION_TAG])` stream, so a calibration never starts where a training run with the same seed started — the replay `core/rng.py` exists to prevent, whose docstring names this one exception — and `seed=None` leaves the streams alone, so `smoke`'s calibration still follows `smoke`'s seed. The `probes` modes follow the diagnostics' convention (`--seed`, default 0, recorded)."

  Then, from the ledger:
  - the execution rulings a later piece would otherwise re-litigate, each with what it costs if wrong;
  - the whole-code review's verdict and the fix dispatch;
  - the card run's judgement;
  - a pointer to spec §11 with its row count.

- [ ] **Step 8: CLAUDE.md.**
  - The Tests section's first bullet: "the two halves at the piece-5 gates were 14 min 42 s + 38 min 6 s, so budget about 55 minutes" becomes "the two halves at the piece-6 gates were <fast> + <slow>, so budget about <sum, rounded up to 5> minutes".
  - The five slow tests, each with its measured seconds and "<total> at `<Z>`", replacing Task 38's "(timed at the final gate)".
  - The gpu-marked count ("plus six in `tests/test_user_sbi.py`") is updated from Step 2 if it changed.

- [ ] **Step 9: Spec §11.**
  - Fill the table's rows `| # | where | what changed | why | cost if wrong |` from the ledger's deviation rows. These two are known:
    1. §6.6 step 3 and §8 item 2: the card recipe block was made reference-free and gained `probes` in its diagnostic card at the card run (Task 26), not at the handoff's retirement. Why: `docs/guide/testing.md` carries the block verbatim under the source scan, and the block named `docs/STATE.md` three times and "piece-2" once. Cost: the pointer to STATE's timings moved to a sentence after the block.
    2. §6.6 step 3: the archived handoff is read back with `git show <A>^:PRISM_HANDOFF.md`, because the archiving commit removes it from the tree. Cost: none.
  - Add every other ruled departure in the ledger: the planning-time rows the controller wrote before Task 1 ("How to run this plan") and every one recorded during execution.
  - The spec is this piece's own, so editing §11 is expected. The closed specs stay untouched.

- [ ] **Step 10: Commit.**
  - `git add docs/STATE.md CLAUDE.md docs/superpowers/specs/2026-09-25-documentation-and-retrain-readiness-design.md`, then `git commit -m "docs: piece 6 closed; final gates, STATE and spec deviations" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"`.
  - `git status --porcelain` prints nothing. Do not push: the owner pushes.

- [ ] **Step 11: Hand the owner rows F1–F5.** The closing message to the owner, in plain words:
  - the piece is done, with its commit range and the gate numbers;
  - "Please run rows F1–F5 of `docs/checklists/display-walkthrough.md` on the real screen. The setup paragraph above the table builds the tiny tier-1 model they need, and F3 edits a record by hand, so copy its manifest first. Tell me each row's result and I will record it in the table and in STATE";
  - the retrain is next, from `docs/guide/retrain.md`.
