# PRISM — state

**Last updated:** 2026-09-30. **Piece 6, "the reference cleanup, retrain readiness, the probe checks and the
documents", is DONE**: commits `779c73c`..`067eddc` (115, counted by `git log --oneline 779c73c^..067eddc`) plus this closing
documents commit, on the local `main` branch, UNPUSHED (the owner pushes). Its design is
`docs/superpowers/specs/2026-09-25-documentation-and-retrain-readiness-design.md` (`907098d`; decisions H1–H13 in
§1.1; §11 now 30 deviations, rows 1–14 ruled at planning and 15–30 during execution, each with what it costs if
wrong) and its plan `docs/superpowers/plans/2026-09-28-documentation-and-retrain-readiness.md` (`779c73c`, 39
tasks). **What landed, part by part:** (1) the reference cleanup — nothing shippable (`core/`, `tests/`,
`conftest.py`, the launchers, `README.md`, `requirements.txt`, `pytest.ini`, `docs/guide/`) cites a working
document or its labels, and `tests/test_source_hygiene.py` enforces it; (2) the help tidy — one default convention
in every help line, pinned by `core.tool.help_defaults`; (3) the readiness fixes — the tier-1 path tested end to
end, the silent Hopf force-scale fallback closed, temperature marked an assumed input, the calibration verdict and a
repeatable `validate --seed`, the Fisher eigenvalues carried across a resume, truth containment reported both ways,
the tier-1 constraint and the acceptances recorded; plus two found on the way: loading refuses a chi posterior
trained at another drive (§11 row 24), and a training run sets aside shards a killed save left uncommitted instead
of letting them refuse the whole cache at its final load (row 25); (4) `python -m core probes band|mask|drive`,
measuring the chi band, the masking and how hard a cell can be driven; (5) the reader guide `docs/guide/` (ten
pages, each stamped with the commit it was checked against) and the root README's command-line summary; the handoff
archived at `bf5dd92` (read back with `git show bf5dd92^:PRISM_HANDOFF.md`). **Execution (2026-09-28/30, Opus 5.5,
subagent-driven, controller rulings in the gitignored ledger):** every task had a review and its own one-process
fast gate, started together; the whole-code review (Task 25: five lenses, each with an adversarial verifier) found 0
Critical, 2 Important and 23 Minor (+3 from the verifiers) — both Importants and 12 Minors fixed in 25 commits
`8fef76f`..`04fbe54`, the rest parked with reasons ("Owed" item 7); the card run of record (2026-09-29, at `16e5bae`)
passed every leg, with the tier-1 run, the narrowing leg, the diagnostic card and the probes (the gate table); the
documents review (Task 37: an only-record checklist of 157 rows, three accuracy lenses, completeness and reading
paths, a verifier each) found 2 Critical and 39 Important page errors and 55 false code texts, all fixed, with the
checklist closed (103 found, 19 appended, 35 dropped). The final fast gate, the slow set and the card judgement are
in the gate table below.
**Next:** the owner runs rows F1–F5 of `docs/checklists/display-walkthrough.md` on a real screen; then the retrain
("Owed" item 8) from `docs/guide/retrain.md` — its stop point (about 740 hours) waits for the owner's confirmation
(the open list).
**Piece 5, "the secondary analyses", is DONE**: commits
`a0d85da`..`7e51275` (77, counted by `git log --oneline a0d85da^..7e51275`) plus the documents
commits that close it (this file's `fec2cde`, its review fixes `6c68a22`, and a count correction), on
the local `main` branch, PUSHED — `origin/main` is `8556546` (seen 2026-09-28), so pieces 4 and 5 are
both pushed. The 77: `a0d85da` the
design and `0c974c1` its STATE note; `e902256`+`5354e6b` the plan and its review's thirteen fixes;
`673868d` the hand-over note; `cbc2187` the pre-flight (every ruling folded into the task it
binds); Tasks 1–40 in 58 commits (`f6a722c`..`7b12c34`, their fix rounds and two mid-execution
STATE notes `74a7693`, `407ce0c` included); the whole-piece review's fix dispatch in 11
(`d13ca96`..`6be832c`); and Task 41's documents (`c29320c`, `7e51275`). Its design is
`docs/superpowers/specs/2026-09-22-secondary-analyses-design.md` (`a0d85da`; decisions E1–E12 in
§1.1, the design rulings in §1.2, what is handed on in §1.3, and §12 now 36 deviations with what
each costs if wrong) and its plan `docs/superpowers/plans/2026-09-22-secondary-analyses.md`
(`e902256`+`5354e6b`, 41 tasks, 433 steps, 82 planning rulings P1–P82). **What landed:** an eighth
store kind, `fdt`, whose records are PROGRESSIVE — the folder and a first manifest exist from the
moment a run starts, `refresh()` rewrites them as it goes, and a cancel or a crash keeps the folder
marked unfinished, while the six ordinary kinds still lose theirs on any exception (a mode on
`ArtifactWriter`, never a second writer). Both analyses are public entries writing named records
with provenance (the cell, the units and the resolved bounds file by path and SHA-256), a recorded
seed and their numbers in `data.h5`. The sweep is split into one record per swept parameter,
carrying its done/failed counts, with an all-failed refusal — and an all-failed first sweep no
longer costs the second: the temperature sweep runs regardless and the study refuses only if both
measured nothing (P77). Two pre-spend refusals — the normalisation constant before the first
campaign, the band once the grid is built and before the driven campaign — plus a third for a
diverged spectrum, and the off-grid fix (probes the spectrum cannot supply come back blank and are
counted, not fabricated). The five screens' builder refusals reach the yellow box, `_config_error`
is gone, and the fix-hint table understands screens as well as the six inference tabs. The pickers
offer earlier runs, and the figure watcher points at the record's `figures/`. `fdt` and `crossval`
take `--store-root`, with the three behaviours that keyed off the flag reworked (smoke's throwaway
root is its own `temp_store_root` property), and `--seed` (P79) — and, since the review, `--name`
and `--note`. The tidy-up offers loose files inside a kind directory and a legacy directory beside
them, in both front ends. The comparison facility has four modes — `compare cells`, `repeats`,
`renormalise` and `sweeps` on the command line, and a "Compare saved runs"/"Compare saved sweeps"
group on the two analysis screens — each writing a comparison record. **Execution (2026-09-23/24,
Opus 5.5, subagent-driven):** the pre-flight found that the brief extractor never carries a plan's
preamble and ruled 66 further findings into the tasks; every task had a review that drove the real
API and its own one-process gate, from Task 4 on started with the review (the owner's decision of
2026-09-23). A whole-piece review by five lenses, each followed by an adversarial verifier, found 0
Critical and 5 Important (a diverged sweep point counted as done,
a sweep's summary naming no point's own ratio, a delete prompt false for a run writing in another
process, no names for the tool's records or the window's comparisons, and a walkthrough row that
could not be run as drafted); one fix dispatch took all four code items and 42 cheap fixes, each
behaviour fix test-first, and a scoped re-review approved it with nine minors parked ("Owed" item
7). The final fast gate, the slow set and the GPU smoke-gate judgement of record are in the gate
table below.
**Rows E1–E17 of `docs/checklists/display-walkthrough.md` were run by the OWNER on a real screen,
reported 2026-09-25: all seventeen pass** (the gate table below).
**Piece 4, "GUI usability and the artifact browser", is DONE**:
commits `5902259`..`80be144` (47), on the local `main` branch, since pushed (the owner pushes) —
`main` was 49 commits ahead of `origin/main` (`cfe261b`) when it closed: `691a233` (piece 3's
C1–C11 rows recorded), the 47 of `5902259`..`80be144` (`5902259` the design, `6fc399f` the plan,
`644dcdb` the pre-flight fixes, then the 26 tasks, then the whole-piece review's fix wave `e9318ff`,
`114711b`, `5e4543b`, `ee2dd33`, `80be144`), and that piece's documents commit. Its design is
`docs/superpowers/specs/2026-09-17-gui-usability-and-artifact-browser-design.md` (`5902259`,
decisions B1–B17 in §1.1, §12 the deviations) and its plan
`docs/superpowers/plans/2026-09-17-gui-usability-and-artifact-browser.md` (`6fc399f`, 26 tasks). What
landed: the artifact browser — a fifth Home tile, not a `BasePanel` — lists all seven store kinds in
a real sortable table, a text-only detail pane (the manifest, then the run's `log.txt` records), a
one-line note, one-artifact-at-a-time delete refused by any dependent (with no `force=True` in
either front end), a sweep that removes only a directory carrying no manifest at all — bound to the
list the confirmation showed, going through the single-directory `remove_incomplete` call, and
refusing a directory written in the last few minutes — a Save of what is on screen, and a lineage
report walking the parent chain; any change it makes refreshes the three `StorePicker`s (B1–B10).
Its command-line twin, `python -m core artifacts` (`list/show/note/rm/sweep/summary`), takes no
configuration flags, prints the same facts with a comma where the window prints "·", and sweeps as a
dry run unless `--yes` is passed (B9). Three behaviours: a live run is visible app-wide — a clickable
shell-header line naming what is running, where and for how long, plus a marker on the running
section's Home tile and its tab (B11); Apply shows a permanent line naming what the session holds and
confirms, defaulting to No, before discarding a non-empty one (B12); the three inference pickers
spell out mode, width and amortization on the line beneath them, and a narrowed posterior reads
"narrowed (TSNPE)" in the closed dropdown and its tooltip, replacing "NON-AMORTIZED" there (B13). Four
handed-on defects closed: a front-end root-logger handler stops a library's own module-level warning
from doubling a `core` record or silently dropping one between runs, by a mechanism that differs from
what the design named (B14; spec §12 row 15); the training rescue save runs inside a deferred-cancel
critical section, and — after the whole-piece review replaced a catch-time judgement with a
raise-time one — the worker reports the real crash with the cancel noted for the direct collision
**and** the residual unwind race alike, restoring the original choice in full (B15; spec §12 row 3);
a newly added chi probe row's frequency box starts blank, never a typed-looking `0.0`, with one
wording for "blank" across the tab and the planner (B16); `PaneCapture` wraps both pane channels into
one ordered list so a test sees what a user sees (B17). A whole-piece review by four independent
readers (correctness, spec compliance, tests, the hazards no CPU suite reaches) found 0 Critical, 12
Important and 38 Minor findings; the fix wave landed in `e9318ff`, `114711b`, `5e4543b`, `ee2dd33`,
`80be144` with a scoped re-review — ten of the twelve Importants addressed there, the other two being
that piece's own spec-document corrections, made by that piece's documents task. The final fast
gate, the slow set and the GPU smoke gate of record are in the table below; the diagnostic card was
judged unnecessary (no line under `core/diagnostics` moves a tensor) and that judgement held.
**Rows D1–D17 of `docs/checklists/display-walkthrough.md` were run by the OWNER on a real screen on
2026-09-21: all seventeen pass** (the gate table below).
Piece 3
(`3db271e`..`9e2f7ef`) is DONE and pushed, and its rows C1–C11 all pass; piece 2
(`0016dae`..`d34997c`) is also pushed, and its rows B1–B8 all pass.

## Where things stand

- The pre-retrain hardening programme is approved: the seven-piece decomposition
  (`docs/superpowers/specs/2026-09-10-pre-retrain-hardening-decomposition.md`) and the piece-1
  artifact-store design (`docs/superpowers/specs/2026-09-10-artifact-store-design.md`, whose §11
  lists the ten deviations ruled during execution).
- On `main`: the 2026-09-09/10 TSNPE fixes X1–X5 and the parent-prior refusal (`ed16397`), the
  specs (`aff8f7e`), piece 0 (`4b1834a`..`d05d3cd`, `3913a1b`), the piece-1 plan (`afa484f`).
- **Piece 0 is DONE** (2026-09-10): pytest, `CLAUDE.md`, this file, the launchers, the display
  checklist, the GPU smoke baseline (table below).
- **Piece 1 is MERGED** (2026-09-11, merge commit `461d780`; the branch and its worktree are
  deleted). It was built on branch `piece-1-artifact-store` (forked from `main` at `8ec18aa`,
  28 commits, tip `7bce752`; `main`'s own `70cd11a` sandboxed the old artifact-consistency tests
  and is superseded by the branch's rewrite of that file): the 14 tasks of
  `docs/superpowers/plans/2026-09-10-artifact-store.md`, one commit each with a per-task review
  and fix loop, then a whole-branch review ("ready with fixes"), one fix wave (`830cee1`) and one
  residual fix (`3f1a1b3`). The execution ledger (rulings R1–R7, every parked finding, every
  review verdict) was git-ignored scratch and is deleted with the workspace once the branch is
  finished; its rulings live on in the decisions log below and in the design spec's §11, and
  the items it carried forward are in "Owed" item 5. What landed: `core/artifacts/`
  (store, writer, manifests, provenance, identity); every stage takes and returns `Loaded*`
  wrappers and writes its artifact at completion under `Artifacts/<kind>/<name>__<id>/`; the
  `.rot.pt` sidecar, the save/load helpers and the generated-path constants are gone; loading
  refuses any verifiable mismatch with `Accept(truncated, other_observation)` as the only
  hatches; the simulation identity is `training-rows/2`; pickers list the store; scripts are
  re-pointed (nine marked broken until piece 2). Defects 3–4 of the decomposition are fixed
  (the forced-recording path runs); defect 1 disappeared with `load_prior`; defect 2 was fixed by
  piece 3 (T20, `42d8f5b`).
- **Piece 2 is DONE** (2026-09-15, commits `0016dae`..`d34997c`, on the local `main` branch, pushed 2026-09-15). It
  follows the design `docs/superpowers/specs/2026-09-11-one-flow-design.md` (§1.1 holds decisions
  D1–D13; §11 lists the 58 deviations ruled during execution) and the 25 tasks of
  `docs/superpowers/plans/2026-09-12-one-flow.md`. Each task had a review and fix loop and a
  one-process fast gate. A whole-branch review then found six must-fix defects and 21 cheap fixes;
  they landed in `2010250`..`d34997c` with a scoped re-review. The execution ledger is gitignored
  scratch under `.superpowers/sdd/2026-09-12-one-flow/`. Its rulings live on in the decisions log
  below and in spec §11; its open items that a later piece or the owner must take are in "Owed"
  item 7. What landed: the prompt CLI is gone (D1); `python -m core <subcommand>` (`core/tool/`)
  passes every flag to its stage as a keyword argument (D2, D6); `simulated_inference`,
  `experimental_inference` and `tsnpe_round` hold the checks each front end used to carry alone
  (D3); the diagnostics write a `diagnostic` store kind (D4, D5); training refuses a "near miss",
  meaning a saved simulation cache that differs from the new run in exactly one setting, before
  the Fisher step (the costly rotation computed before training simulates) and before any
  simulation (D7); the tool refuses by default and `--accept-*` maps onto `Accept` (D8); every
  driven chi recording states its drive frequency in Hz (D9).
- **Piece 3 is DONE** (2026-09-16/17, commits `3db271e`..`9e2f7ef`, on the local `main` branch,
  pushed — `origin/main` is `cfe261b`, its last documents commit). It follows the design
  `docs/superpowers/specs/2026-09-15-validation-and-logging-design.md` (§1.1 holds decisions V1–V9;
  §11 lists the deviations ruled during execution) and the 24 tasks of
  `docs/superpowers/plans/2026-09-16-validation-and-logging.md`. Each task had tests written first, a
  review and fix loop, and a one-process fast gate. A whole-piece review then ran as four independent
  readers (correctness, spec compliance, tests, the hazards no CPU suite reaches). They found two
  must-fix defects: the bounds and cell file rules were never raised where the files are read, and a
  failed run in the window kept its frames (and so its host buffers and GPU tensors) alive through
  the exception the worker now carries. Those two and thirteen cheap fixes landed in one dispatch,
  each test-first, with a scoped re-review. The execution ledger and the four review reports are
  gitignored scratch under `.superpowers/sdd/2026-09-16-validation-and-logging/`; their rulings live
  on in the decisions log below and in spec §11, and what they hand on is in "Owed" item 7. What
  landed: every public stage, composition and diagnostic runs on a private deep copy of the config
  (`core.runs.public_entry`, V1); bad boxes are refused at the click with no clamp or silent default
  (V2); one `core.refusals.Refusal` kind with a field key, neutral core messages, and two front-end
  tables naming the control or flag, a yellow "Check your inputs" box for a refusal and the red box
  for a bug (V3); standard `logging` at info, warning and error in place of the in-stage prints, a
  window handler that is also the cancel checkpoint, the tool's info-to-stdout and
  `warning: `-to-stderr handlers, and a `log.txt` in every committed artifact but the simulation cache
  (V4); the inference tabs remember SELECTIONS — the pickers, the mode, and the boxes that describe
  the recording (the three observation lengths, the physical drive amplitude `Drive F₀ (N)`, the
  recording paths) — plus the training budget, every inference science knob opens at `config.py` on
  every launch, the Config tab's non-dimensional χ drive amplitude and band are read-only displays
  of it, and a consent is never persisted (V5; the Simulate, FDT and CrossVal panels still remember
  their own numeric fields — piece 5's, spec §1.3); one `orchestrator.training_preview` behind the
  budget lines (V6); the Fisher settings recorded only when the rotation ran (V7); a loaded prior
  closes its figure (V8); sbi's summary writer is switched off, and the repository root's `sbi-logs/` tree
  (1359 run directories under `NPE_C`) was deleted (V9).
- **Piece 6 is DONE** (2026-09-28/30, commits `779c73c`..`067eddc`, on the local `main` branch, unpushed). It follows the
  design `docs/superpowers/specs/2026-09-25-documentation-and-retrain-readiness-design.md` (§1.1 holds decisions
  H1–H13; §11 lists 30 deviations, 1–14 ruled at planning and 15–30 during execution) and the 39 tasks of
  `docs/superpowers/plans/2026-09-28-documentation-and-retrain-readiness.md`. Each task had its brief, an implementer,
  a review (a workflow of scoped reviewers and adversarial verifiers for every large page and for the whole-code
  review) and a one-process fast gate started with the review; behaviour changes were test-first. The execution
  ledger, the pre-flight scan, the reviews and the gate logs are gitignored scratch under
  `.superpowers/sdd/2026-09-25-documentation-and-retrain-readiness/`; their rulings live on in the decisions log
  below and in spec §11, and what they hand on is in "Owed" item 7's open list. What landed is in the paragraph at
  the top.
- **Decision 2026-09-10: CLEAN BREAK.** Every generated artifact is deleted once piece 1 is
  merged (runbook: design spec §9 = plan Task 14). The old runbook's Run A, Run B and the TSNPE
  round (`PRISM_HANDOFF.md` §11.9) are ABANDONED; the retrain restarts from scratch after piece 6.
  **Executed 2026-09-11** (section "Clean break" below).

## Owed, in order

1. ~~Merge piece 1~~ — done 2026-09-11 (`461d780`); the one-process fast gate on merged `main`
   is recorded in the table below.
2. ~~Clean-break runbook~~ — done 2026-09-11 (design spec §9 / plan Task 14, run by Claude at
   the user's request; the "Clean break" section below records every step, the one deviation
   and the fast gate afterwards).
3. ~~GPU gate after piece 1~~ — done 2026-09-11, GREEN at `bb38f1a` (table below). **The first
   attempt at `19363d2` FAILED twice** and both failures were piece-1 regressions on paths no CPU
   suite can reach: (a) infer's PPC met a CUDA observation row and CPU simulated rows —
   `store.load_observation` handed the row back on `cfg.hw.device` and `infer_and_visualize` bound
   it, while `statistics.conditioning_rows` assembles every row on the CPU by contract; (b) the
   forced-mode prior was refused by its own store the moment `build_prior` read it back
   ("prior.pt holds GMM …, not the one the manifest records") — `file_manager.load_mix_dist`
   rebuilt `Categorical(probs=w)`, whose re-normalisation `w / w.sum()` is not a bitwise fixed
   point in float32 and depends on the device's reduction order (the chi prior round-tripped
   exactly on CUDA and not on the CPU; the forced one on neither). Fixes, each with a failing test
   first: `896e8ff` the loader restores the saved weights into `.probs`/`._param` (exact round
   trip; two CPU tests reproduce the refusal); `bb38f1a` the row stays on the CPU and `sample()`
   moves its own copy, pinned by the new gpu-marked `tests/test_gpu_paths.py` (the tiny SBITEST
   chain on `config.detect_device()`, ~12 s; `build_tiny_run(store, hw=None)`).
   **Recipe correction:** run 2 must use the SAME `NUM_RUNS` as run 1 — the simulation identity
   includes `n_runs`, so `NUM_RUNS=2` after a `NUM_RUNS=4` run 1 keys a NEW cache directory,
   rebuilds the Fisher rotation and never resumes (it did exactly that here, silently, exit 0).
   The recipe is four command lines since T19 (`CLAUDE.md`, spec §3.8): `python -m core smoke --chi
   --t-obs 4.5 --bounds …/master.txt --cell …/master_spont.txt --checkpoint --save --store-root
   <scratch>/smoke` (run 1); run 2 is that line WITHOUT `--save`, with `--prior smoke_prior --stages
   prior,posterior --resume require` added (must resume); run 2b is run 2's line WITHOUT `--resume
   require`, with `--num-runs 2` added instead (must EXIT 1 naming `n_runs` — the incident, now
   loud); then the forced-mode run against its own store. Read literally, keeping `--save` on run 2
   would refuse on the taken name `smoke_posterior` instead of resuming — fixed in the fix-round-1
   pass over this recipe (K4), kept in sync with `CLAUDE.md`. The environment-variable recipe is
   gone with `scripts/smoke_train.py`.
4. ~~Manual GUI check on a display~~ — done 2026-09-11 by the user on the real screen: rows A1–A9
   of `docs/checklists/display-walkthrough.md` all pass (reported "everything passes"). The one
   failure was the walkthrough's row 1, the APP ICON: the window's title bar showed the mark, the
   taskbar button the generic Windows "application" glyph. Root-caused the same day by screenshot
   experiments (the "Taskbar icon" entry in the decisions log) and fixed by installing the mark as
   Qt's Win32 window-CLASS icon from a new `assets/app/prism.ico`
   (`app_icon.set_windows_class_icon`); verified by a screenshot of the real launcher's button
   and then by the user on the next launch (row 1 PASS). **The user also ran rows 2–20 of the
   piece-4 walkthrough table the same day: all pass** — piece 4's display walkthrough is therefore
   already done once, on the code as of `df7491e`; piece 4 re-checks only what it changes.
5. ~~**Piece 2**~~ — DONE 2026-09-15, commits `0016dae`..`d34997c`, on the local `main` branch, pushed
   2026-09-15. Every item piece 1 carried into it is closed: ~~extend the source scan to
   `scripts/`~~ (`scripts/` is gone; the scans walk `CODE_ROOTS` plus `CODE_FILES` in
   `tests/_fixtures.py`, and `test_the_source_scans_cover_every_code_directory` keeps the set
   closed — T21, T22 and the final review); ~~retire `_common.require_mode` and the unconditional
   `Accept(truncated=True)`~~ (`_common.py` is gone; the tool refuses by default, and
   `--accept-truncated` / `--accept-other-observation` map 1:1 onto `Accept` — T12, T21);
   ~~thread the store into the TSNPE runner~~ (`tsnpe_round` takes a `LoadedObservation` and
   `store=` — T5, T7, T12); ~~the stale comments the clean break left~~ (T23, `7c5ff48`); ~~make
   the resume drill loud~~ (a cache one setting away is refused before the Fisher step and before
   any simulation, with `--resume require` and `--new-run`; GPU run 2b exited 1 naming `n_runs` —
   T2, T19).
6. ~~**The display walkthrough's piece-2 rows B1–B8**~~ — done 2026-09-15 by the USER on the real
   screen: **all eight pass**, recorded in that file's last two columns and in the gate table below.
   B8 re-checked the FDT panel's full run, because `bb22ac4` changed the plots it draws.
7. **Pieces 3 → (4 ∥ 5) → 6 — ALL DONE** (piece 6 closed 2026-09-30), each brainstormed → spec → plan →
   implementation. Carried into them
   from pieces 2 and 3 (each design spec's §1.3, the final reviews' "left open" lists, and the
   ledgers):
   - **Piece 3** — DONE 2026-09-16/17 (`3db271e`..`9e2f7ef`). Every bullet piece 2 carried into it is
     closed:
     - ~~Copy-on-run session config~~ (a refused stage leaves nothing on the session; training in a
       session with a ground truth is truth-free; an experimental chi inference no longer leaves its
       probe count on the session — T2, T3, and the window pin on `screen_run` in T15).
     - ~~Logging with severity in place of the in-stage prints~~ (T16–T19).
     - ~~Boundary validation of every field, and dialogs that name the fix~~, including the TSNPE
       directions refusal, now keyed to its control and `--directions` (T4–T15).
     - ~~`tsnpe_tab.restore_settings` is never called~~ (T20).
     - ~~The Config tab's chi drive and band fields are editable and restored~~ (read-only since T11;
       not remembered since T20).
     - ~~The budget status line derives the simulation identity separately~~
       (`orchestrator.training_preview`, T21).
     - ~~The TSNPE manifest records Fisher defaults that did not run~~ (V7, T7).
     - ~~`build_prior`'s load branch calls `plt.show()`~~ and ~~sbi writes `<cwd>/sbi-logs`~~ (T22).
     - ~~Rows C1–C11 of `docs/checklists/display-walkthrough.md` on the real screen~~ — done
       2026-09-17 by the USER: **all eleven pass**, recorded in that file's last two columns and in
       the gate table below.
   - **Piece 4** — DONE 2026-09-17/21 (`5902259`..`80be144`). Every item piece 3's final review
     handed into it is closed:
     - ~~The artifact browser, including a listing of diagnostics and each artifact's `log.txt`~~ —
       landed as the Artifacts screen, a fifth Home tile and not a `BasePanel`: all seven kinds in a
       sortable table, a text-only detail pane (the manifest, then the log) — T9, T10.
     - ~~Annotate (`set_note` has no GUI caller)~~ — landed as the Note box: one line, trimmed,
       refused over 200 characters or with a newline — T2, T10.
     - ~~Cleanup of incomplete directories~~ — landed as Sweep, narrowed at the whole-piece review to
       remove only a directory with no manifest at all, bound to the list the confirmation showed,
       and refusing a directory written in the last five minutes — T5, T11; spec §12 row 16.
     - ~~`Summary.complete` for simulations means only "has a manifest"~~ — a separate `finished`
       field now answers "did the run finish", with `batches_done`/`batches_planned`/`rows` carrying
       a cache's progress — B3, T3.
     - ~~The observation width guard in `load_observation` has no isolated test~~ — both payload
       guards (the digest and the width) now have their own tests, each shown to trip empirically —
       T24.
     - ~~The front-end root handler for a library's module-level warning~~ — landed (B14), by a
       mechanism that differs from piece 3's own proposal: the sink drops a record when a handler
       below the root already emitted it, not by a name-based rule — T20; spec §12 row 15.
     - ~~The rescue-save cancel collision~~ — landed: the rescue save runs inside a deferred-cancel
       critical section, and the worker reports the real crash with the cancel noted, for the direct
       collision and the residual unwind race alike — B15, T21–T22; spec §12 row 3.
     - ~~The blank probe row~~ — landed: a newly added row's frequency box is empty, never a
       typed-looking `0.0` — B16, T23.
     - ~~`PaneCapture` does not see lines a worker sends to the pane~~ — landed: it wraps both pane
       channels into one ordered list — B17, T1.
     - ~~Rows D1–D17 of `docs/checklists/display-walkthrough.md` on the real screen~~ — done
       2026-09-21 by the USER: **all seventeen pass**, recorded in that file's last two columns and in
       the gate table below.
   - **Piece 5** — DONE 2026-09-22/24 (`a0d85da`..`7e51275`). Every item carried into it is closed:
     - ~~FDT/CrossVal hardening and wrapping them in the store~~ — the `fdt` kind, its progressive
       writer and the loose-file and legacy-directory calls (T1–T4); the settings object's sources,
       seed and private copy (T7); the field keys, the fix-hint table widened to screens, the checks
       and the pre-spend refusals (T8–T16); the single-cell run as a public entry writing a record
       with its numbers (T17–T18); the sweep split into two records with its failure counts
       (T19–T20); the screens' pickers and watcher (T24–T27); the command line's `--store-root`,
       `--seed`, refusals and tidy-up (T28–T30), and the Artifacts screen's tidy-up (T31); and the
       comparison facility (T36–T40). Nothing is written to `artifacts_root()/crossval` any more.
     - ~~The cell-folder branch of the `fdt` unsupported-model hint is untested~~ — T29 (`70d23ab`).
     - ~~Only `plot_psd` of the four `core/FDT/plots.py` functions is unit-tested for closing its
       figure~~ — T32 (tests only: all four functions already closed their figures since `bb22ac4`;
       P17).
     - ~~The five panels adopt `core/refusals.py`'s rules and the two field tables~~ — T21, T22,
       T23; `_config_error` is gone since `f19e6dd`, and every builder refusal reaches the yellow
       box.
     - ~~`load_and_validate_gt`'s `Refusal(field="cell")` and the FDT dry run's problem strings are
       two wordings for one rule~~ — one message builder, `core.refusals.missing_values_phrase`,
       delivered by T21 (P4, P76); T29 produces none. A THIRD wording, `cli._merge_vals_bounds`, is
       knowingly left (P53; below).
     - ~~The artifact table's numeric columns sort as text~~ — T2 (`d9afaba`).
     - ~~`tests/test_artifact_browser.py:864` asserts the window's picker kinds by EQUALITY~~ —
       T25 (`da5c392`) made it a subset assertion with `"fdt" in seen` (ruling F24).
     - ~~Rows E1–E17 of `docs/checklists/display-walkthrough.md` on the real screen~~ — done by the
       USER, reported 2026-09-25: **all seventeen pass**, recorded in that file's last two columns and
       in the gate table below.
   - **Piece 6** — DONE 2026-09-30 (`779c73c`..`067eddc`; BRAINSTORMED 2026-09-25, design `907098d`, plan
     `779c73c`, 39 tasks; the paragraph at the top; spec
     `docs/superpowers/specs/2026-09-25-documentation-and-retrain-readiness-design.md`). Everything
     carried into it is closed:
     - ~~The `docs/` split of the handoff, including the `PRISM_HANDOFF.md` lines D1 makes false
       (`:49,53-54,76,190-191`)~~ — met: the reader pages under `docs/guide/`; `PRISM_HANDOFF.md`
       archived at `bf5dd92`, read back with `git show bf5dd92^:PRISM_HANDOFF.md`.
     - ~~A README reference section for the tool~~ — met: the root README's command-line summary and
       `docs/guide/command-line.md`, checked by its parser-walking test.
     - ~~The stale memory figures at `PRISM_HANDOFF.md:2615-2616` against `:6756`~~ — met: the reader
       pages carry the right figures; the still-open half (the FDT sanity checks keep a view of the
       whole solution) is on the open list below.
     - ~~Piece 4's "rest of group M" of the handoff's traps~~ — met by `docs/guide/rules-and-traps.md`.
     - ~~The *Provenance* open item (a calibration and a TSNPE child do not record
       `--accept-truncated`; an inference on its posterior's own observation records `accepted: []`)~~
       — closed by H13: calibrations record `results.accepted` and narrowing rounds `training.accepted`.
   - **Open, with no piece owning them yet** (the owner decides each, or where it goes):
     - *Science.* In `identifiability jacobian`, a NaN in a measurable Jacobian column makes the
       least-squares step raise `LinAlgError` after all the simulations are spent (the retired
       script crashed the same way). The choice is between excluding that column and zeroing that
       row.
     - *Science.* Repeat-SBC in chi mode cannot see probe-design variance: `pipeline.py` reseeds the
       chi probe generator to a fixed seed (20260805) on every `gen_training_data` call, so every
       repeat uses the same probe design whatever `--seed` says.
     - ~~*Provenance.* A calibration and a TSNPE child do not record `--accept-truncated` in
       `results.accepted`, and an inference on a non-amortized posterior's own anchor observation
       records `accepted: []` even when `--accept-truncated` loaded it~~ — closed by piece 6, H13.
     - *Spec.* `tsnpe_round` refuses more directions than the latent width but allows exactly as
       many, which truncates every direction. That matches spec §2.5 word for word, so tightening
       it needs a spec decision.
     - *Dormant.* `seeded()` — since piece 5 it lives in `core/rng.py`, re-exported unchanged from
       `core.diagnostics.rng` — restores only the CUDA device it is given, while `torch.manual_seed`
       reseeds every CUDA device; on a machine with several GPUs the others' streams are not
       restored. On a CPU device it no longer touches CUDA at all: it seeds only the CPU generator
       and numpy (spec §12 row 19). Revisit if a gpu-marked test starts to depend on test order.
     - *Tests.* The code-directory guard skips an enumerated list of directories, so a local
       virtual environment at the repository root would fail it for a non-code reason.
     - *Science (from piece 3).* The tool accepts `--chi-k 1`: `SimConfig.__post_init__` allows one
       probe, while the Config tab and spec §3.3 refuse fewer than two, and training draws at least
       `CHI_K_MIN_TRAIN = 2`. Whether a one-probe SIMULATED observation is in distribution is the
       owner's call. If not, the fix is a floor of 2 in `__post_init__`; the bench path, which
       legitimately takes one probe, never re-runs that check (spec §11).
     - *Provenance (from piece 3, spec §1.3).* A resumed training run records its Fisher settings as
       "not run" (V7): the checkpoint header carries `V` but not the parent's `m`, `dz` and `points`. Since
       piece 6 the eigenvalues carry over; the settings still do not.
     - *From piece 4.* The Posterior tab's D8 load dialog (piece 2's consent that fires on loading a
       round's posterior) still says "NON-AMORTIZED" where the three pickers now say "narrowed
       (TSNPE)" (B13) — one fact, two wordings. Changing D8's text touches walkthrough row B1's
       expected string, so Task 19 left it for a deliberate decision rather than a drive-by fix, and
       the whole-piece review confirmed the item stands, owed rather than fixed.
     - *From piece 4.* A sweep re-reads every candidate's manifest one at a time, so it costs O(n²) at
       a large leftover count — correct and irrelevant at today's scale, the same judgement spec §1.3
       makes about the listing generally; no index is built.
     - *From piece 4.* The new source scan that forbids a static `QMessageBox` call under `core/`
       (design spec §12 row 21) would miss an aliased import spelling (e.g. `QMessageBox as MsgBox`).
     - *From piece 4.* The slow set's chi full-pipeline test ran 1610 s at the piece's final
       measurement against 1181 s recorded for piece 3, on code this piece's diff does not reach on
       that test's path (the root handler and the cancel deferral are consulted only by the two front
       ends, never by a bare `core` pipeline run). It passed both times; worth watching at the
       retrain, where the same test runs for real. (Piece 5's slow set: 1397 s.)
     - *From piece 4, declined by the owner or left unsound (design spec §1.3).* Figures in the
       browser, an open-the-folder button and "use this" jumps into a stage tab — considered and
       declined 2026-09-17. A session summary at quit — declined in favour of Apply's session line
       (B12) and the Save/lineage-report pair (B10). `rename` from the browser — Save-as-rename stays
       on the Prior and Posterior tabs; renaming a simulation cache is additionally unsound today (it
       writes a name into the manifest and moves nothing, because a cache's directory name is always
       its digest), and the browser never calls it. `unnamed()` stays uncalled — every simulation
       cache has `name=""`, so a caller built on it would offer to delete committed training rows.
     - *From piece 5 — the owner's decisions* (the whole-piece review's list 4,
       `final-review-synthesis.md` §4 in the gitignored ledger folder; each is science, an owner
       resource file or an environment choice, so none was decided for the owner):
       - The passive-baseline check's Ω₀: the passive process is reversible, so its spectrum has no
         finite-frequency peak and the check's "resonance" is always its search band's first bin
         (0.268 on `master_spont`). Anchor on `cfg.omega_0`, report "no peak", clip the probes to
         the resolved span, or relabel the line? The two probes it then drops cost about 37 % of
         the check's 466 s.
       - The single-run `ratio_at_resonance` takes the probe nearest ω₀, which is a decade away at 2
         frequencies; and no test pins the lock-in's χ″ SIGN — Task 33's test accepts
         `[FAIL] passive_baseline`, so a sign flip would pass the suite. Pin the sign or not?
       - `find_spectral_peak` takes a NaN bin as the peak, which moves ω₀ for a spectrum that is NaN
         in some bins and would shift two pinned counts. A science pass, piece 6 or later.
       - The shipped Hopf cell is refused at the default band (at the shipped band and settings a
         cell is refused iff its spontaneous peak lies below 0.3835 ND, because the Welch segment's
         2^14-sample cap fixes the first bin at 0.0383 ND): one band for every model, or a per-model
         preset? (Spec §12 row 16.)
       - `render_lineage`'s `MISSING <kind> [<id>]` line could also print the record's recorded
         name (a §12 row if done), and a comparison with a null mode prints `compared ((none)):`.
       - `Resources/Cells/nadrowski/master_spont_tier1.txt` resolves to `master.txt` and is refused
         (it lacks `f_scale`), yet both FDT pickers offer it: fix the cell, add a same-named bounds
         file, or remove it.
       - sbi 0.25.0 is installed while 0.26.1 is pinned; syncing would let provenance read the
         version with `importlib.metadata.version` and settle a warning whitelist (review items
         L573, L576).
       - The fix sentences everywhere else still quote a control's raw label; piece 5 relabelled
         only 'M_ensemble' and 'freqs / batch' (N23). A house-wide pass is the owner's call.
     - *From piece 5 — decided at the end of the piece, by controller rulings the owner may
       reverse* (list 4's items 5, 7, 8, 9, 10): a T_a/T grid reaching below 0 is refused, 0 is
       allowed and S has no floor (R-F1, §12 row 27); the Live-simulation crash on a spontaneous
       built-in cell is fixed, departing from §5.6's bound (R-F4, §12 row 34); a saved model with a
       negative forcing amplitude is refused at load as well as at Validate and Save (R-F5, §12 row
       35); repeats that share a seed get a notice, never a refusal (R-F6); the window names its
       comparisons, with Comparison name and note boxes never remembered (R-F3).
     - *From piece 5.* A third wording of "the cell lacks what the bounds file requires" stays in
       `cli._merge_vals_bounds` (P53, spec §1.3): it carries the cell PATH, which the FDT builders
       need and the other two sites do not, so folding it into `missing_values_phrase` would drop
       that or change the other two.
     - *From piece 5.* `core/tool/fdt.py`'s crossval handler still resolves `--n-freqs` and
       `--ensemble-m` from the preset at the call site (`preset[…] if args.… is None`), though
       `make_param_sweep_config` now resolves a None from the preset itself (P50: "a later piece may
       delete the now-redundant one"). Harmless; two sources for one default.
     - *From piece 5.* `check_passive_baseline` still raises a bare `ValueError` when every probe
       lies outside the PSD grid: the new end-to-end Nadrowski test never reached it, so P23 left it
       unconverted (spec §12 row 9) — converting it needs a field-key decision.
     - *From piece 5.* The round trips' one handed-on defect (the Live-simulation crash, spec §1.3)
       was fixed after all by N17; §1.3's remaining defect row is the re-review's minor 2 below.
     - *From piece 5.* No ceilings on 'M_ensemble' and 'n_freqs': a 10⁷ ensemble fails at allocation
       within seconds (walkthrough E8 relies on exactly that), but a huge value that CAN be
       allocated could exhaust RAM before anything refuses.
     - *From piece 5.* A sweep that measured nothing because of one systematic cause (out of memory,
       as in E8) is still advised to move its grid or change its cell; the sentence does say "one
       systematic cause", but not which.
     - *From piece 5 (pre-existing).* A new cell copied into `Cells/hopf/` or `Cells/bp/` under
       another name falls to the legacy parse branch and is refused with a misleading "Could not
       detect time unit from cell file".
     - *From piece 5, out of scope by spec §1.3.* No cross-process lock on a record being written:
       `artifacts note` from a second shell is overwritten by the running writer's next refresh, and
       deleting a live record from another process now leaves nothing rather than a husk (§12 row
       36). Piece 4's delete half-deletes a record when a file late in its walk is read-only.
     - *From piece 5, out of scope by spec §1.3.* Resuming an interrupted sweep — not asked for; E2
       keeps the folder so its data file can be read, not so a later run can continue it. A later
       piece, if wanted.
     - *From piece 5, for a later refusals-hardening pass.* Builder paths no front end reaches:
       `int()` coercion of counts, NaN or infinite counts, a malformed `freq_bounds`, and agreement
       between `preset` and `preset_name` (review items L473, L483). `_newest_mtime` walks into
       directory junctions (latency only, pre-existing, L396).
     - *From piece 5, cosmetic or breadth* — each judged not worth a change now, listed with its
       reason in `final-review-synthesis.md` §3: e.g. `remove_loose`'s alias handling (L394), the
       both-sweeps-measured-nothing case printing its refusal three times (L616; since N9 each
       sweep's error line carries its own sentence and one refusal names both grids — revisit
       whether that still reads as repetition), an empty surface panel for a one-point sweep (L826),
       stale "Runs to compare" entries after a delete and comparison tabs piling up with one title
       (L839), curve colours and keying (L807), literal captions (L641).
     - *From piece 5, the fix re-review's nine parked minors* (`final-fix-rereview.md` §Minor; the
       ledger's ruling parked them rather than send a second fix loop — none load-bearing; its cost:
       a regression of the negative-grid or zero-knob refusal would HANG a gate instead of failing
       it, visible but slow). Line numbers as at `6be832c`:
       1. `tests/test_tool.py:1646`, `:1732`: if the builder refusal regressed, these two would run
          a REAL study or `fdt` run inside the fast gate — stub `run_param_study_cli` /
          `run_fdt` with `pytest.fail`.
       2. `core/FDT/fdt_pipeline.py`: the passive-baseline figure is listed one refresh late in an
          unfinished record (spec §1.3's new row): one `writer.refresh()` after `run_all_sanity`.
       3. Half-typed text ('-') in the four FDT/CrossVal knob boxes is reported "is blank", while
          Slice at, the constant and Seed say "must be a number": read them through
          `number_or_blank` too.
       4. `core/gui/widgets/labeled_inputs.py:44`: the Seed sentence puts its default clause after
          the period (the house `refuse()` shape, L711).
       5. `core/Helpers/model_store.py:158`: "the amplitude's saved box starts at 0" reads as a GUI
          box in a yellow box; "its saved prior bounds start at 0" says what is meant.
       6. `core/gui/panels/record_view.py:71`: the settings line prints `held={'temp': 1.0}`, a
          Python dict repr.
       7. `core/tool/fdt.py:75-77`: in `FDT_INTERRUPT_NOTE` the "it" of "lists it" follows the hedge
          saying there may be no record; put the hedge last.
       8. `tests/test_tool.py:1615`, `:1646`, `:1667`: three fast tool tests request `tool_env`
          (which installs SBITEST into the real `Resources/`) though they need only
          `PRISM_ARTIFACTS`.
       9. Process: several fix-dispatch commit subjects are longer than the house's short subject
          (no amend is allowed).
     - *From the handoff's retirement.* The FDT sanity checks keep the whole solution alive while they
       measure: `check_ensemble_convergence` and `check_psd_window` take `x_steady = sol[0, 0, :, burn_idx:]`,
       a view, and never delete `sol`, where `campaigns.run_campaign1_psd` copies channel 0 with
       `.contiguous()` and deletes the solution (`core/FDT/sanity.py`).
     - *From the handoff's retirement.* The Fisher's `feats` closure in
       `decorrelate.build_latent_fisher_rotation` re-derives `subs`, `n_fine`, `t_fine` and `n_segs` on
       every call, and its `s()` rebuilds `base_inits.expand(mm, -1).contiguous()` on every simulation;
       hoisting them is free and has not been done (`core/SBI/decorrelate.py`).
     - *From the handoff's retirement.* Four speed-ups stay open, each gated: (1) running `gen_chi_raw`'s
       1+K probe simulations as one wider call (up to 8× on the dominant cost) waits on the drive
       builder's 4× transient — `forcing.build_nondim_force_tensor` materialises a full (batch, T) time
       tensor and three more, which `pipeline.peak_sim_elements` and `_max_sim_batch` still count as
       `n_ch * n_fine`; (2) strided single-variable solver output (about 23 % of time, 6× peak) must
       reproduce the segment-seam duplication, keep every variable's final state and take the
       predictive check's per-row subsample; (3) binning the predictive check by a step budget instead
       of `PPC_BIN_SIZE` (bins are t_scale-sorted and keyed on their smallest t_scale); (4)
       `SummaryStatistics`'s duplicate `_acf` and `_analytic_bandpass` calls and `chi.peak_freq`'s
       repeated peak search, about 6 minutes of a production run. A fifth, stacking the Fisher's arms,
       was dropped: 15–30 minutes of about 38 hours, at a high risk to its common random numbers
       (`core/SBI/pipeline.py`, `core/forcing.py`, `core/SBI/ppc.py`, `core/SBI/statistics.py`).
     - *From the handoff's retirement.* Two renames were deferred and never made: `core/gui/vt.py`,
       whose name does not say that it turns tqdm's output into progress rows and log lines, and the
       mixed casing of `core/`'s packages (`SBI`, `Simulator`, `Solvers`, `Models`, `Helpers`, `FDT`,
       `Reduction` against `gui`, `artifacts`, `diagnostics`, `tool`, with spaces in three folder names
       under `core/SBI/Priors/`). Either touches every import of what it renames.
     - *From the handoff's retirement.* `.claude/worktrees/` holds two orphaned copies of the
       repository, `jolly-jang` and `trusting-einstein`, each with a full `core/` that a search over the
       working tree also hits, and an empty `upbeat-rhodes-c8d30f`; `git worktree list` shows none of
       them. Removing them is the owner's call.
     - *From the handoff's retirement.* The FDT and reduction-map figures still save with
       `plt.savefig`/`fig.savefig` rather than `core.Helpers.visualizers.save_figure`, so an appearance
       change during a long run can save them with unreadable colours: `core/FDT/plots.py`
       (`plot_eff_temp_ratio`, `plot_spontaneous_trajectory`, `plot_psd`, `plot_chi_components`),
       `core/FDT/cross_validation_plots.py` (`plot_fdt_3d_vs_param`) and `core/Reduction/plots.py`
       (`plot_sweep_summary`, `plot_cross_validation_3d`).
     - *From the handoff's retirement.* In-repository `file:line` citations came back into comments
       and docstrings and rot as files move; piece 6's documents review replaced every one it found in
       `core/` with a function name (`e965fa0`), but nothing stops new ones: no source scan looks for
       `<file>.py:<n>` in a comment (such a scan would join `tests/test_source_hygiene.py`). Citations
       of third-party files (tqdm, sbi, torch) are a separate question.
     - *From the handoff's retirement.* `core.SBI.training_checkpoint.checkpoints_using_prior` has had
       no caller in `core` since the prior-save path it served was retired (the store refuses to delete
       a prior a cache depends on); only `tests/test_artifact_store.py` calls it, and its docstring tells
       an overwrite story the store made impossible. Possibly dead code
       (`core/SBI/training_checkpoint.py`).
     - *From piece 6 — the load-check audit, for the owner.* Recorded but never compared when a record
       loads: a posterior's and an observation's `feature_set_version` and `summary_flags` (only a width
       change is caught); an observation's box; an observation's chi drive, band and lock-in ceiling (a
       posterior's are compared); a posterior's and an observation's time grid and units-file hash (only
       the simulation cache's identity carries them). Two constants are recorded in no record a load
       compares (only a `probes` diagnostic record carries them) and are not in
       `SimulationIdentity`: the cycle floor `CHI_MIN_CYCLES` and the smallest probe count training
       draws, `CHI_K_MIN_TRAIN`, so a retrain after changing either reuses a cache simulated under the
       old value. Settle this before any post-retrain change to `core/config.py` or to the features
       (`core/artifacts/store.py`, `core/artifacts/identity.py`).
     - *From piece 6.* A simulated chi observation made before an edit to `CHI_F0` or
       `CHI_FREQ_BOUNDS` loads unchecked beside the new configuration: `ArtifactStore.load_observation`
       compares neither. A check belongs on simulated observations only (an experimental one records
       config.py's drive, not the lab's) (`core/artifacts/store.py`).
     - *From piece 6 — science, for the owner.* The prior's stability screen: the census integrates
       only the first half of the stability duration, and its accepted points seed the flood-fill
       without a second screen; the flood-fill's walk has no box check, so walked points outside the
       box are fitted clamped near ±13.8 in the latent coordinate. How many points each affects is
       unmeasured (`core/SBI/prior_screen.py`, `core/SBI/Priors/prior.py`, `core/SBI/reparam.py`).
     - *From piece 6.* A prior rebuild is not reproducible: `prior` seeds nothing (the Sobol scramble,
       the numpy start points and the flood-fill walk draw from unseeded streams); only the mixture's
       random state is fixed. A seed flag would make rebuilds repeatable — and would then break the
       runbook's new-cache route, which relies on a rebuilt prior's new fingerprint
       (`core/SBI/prior_screen.py`, `docs/guide/retrain.md`).
     - *From piece 6, for the owner.* `probes drive` judges an undriven cell as oscillating when its
       highest spectral bin, over the whole band's median power, reaches the clarity threshold, so a
       low-pass spectrum whose maximum sits above the lowest few bins can pass as an oscillation. A
       local-baseline clarity, or a test on a quiescent cell with a mid-band hump, would settle it
       (`core/diagnostics/probes.py`, `probe_drive`).
     - *From piece 6 — tool code gaps.* A Ctrl-C in an `artifacts` command prints the training cache's
       resume advice, since that family sets no `interrupt_note` (`core/tool/browse.py`); `ablation`'s
       three simulation-cache refusals and `make_cfg`'s unsupported-model refusal are plain `ValueError`s
       (`core/diagnostics/ablation.py`, `core/tool/config_args.py`); the store's error for a reference
       that resolves to no record carries no field (`core/artifacts/store.py`); the `units` refusal
       field maps to no flag (`core/tool/fields.py`); `--chi-k`'s shared help reads as a training
       setting, though training draws its own count (`core/tool/config_args.py`); the `identifiability
       laplace` and `jacobian` help omits their mode and model refusals (`core/tool/diagnostics.py`);
       argparse accepts abbreviated flags on every subcommand but `probes`.
     - *From piece 6.* `core.Helpers.file_manager.parse_units_file` collects the unit tokens in a set,
       so `cfg.units_dict` and the `units` list each manifest's config block records
       (`core/artifacts/manifest.py`) come out in hash order, different in each process.
     - *From piece 6.* Probably dead: `core/Helpers/fdt.py`, `core/Helpers/model_helpers.py` and
       `core/SBI/Priors/Forcing Priors/sin_prior.py` — nothing in `core/` or `tests/` imports them.
       Archive or keep: the owner's call.
     - *From piece 6.* The BP model: `BPModelSteady.g` returns (batch, 4, 4) diagonal matrices, which
       the Euler step cannot use — it fails loudly, and only a 16-parameter BP box reaches it (every
       shipped BP box has 17); and the shipped cells `Resources/Cells/bp/cell_1.txt` and `cell_3.txt`
       set `tau_t = 0`, which the 17-parameter branch refuses (read in the code, not run)
       (`core/Models/bp_model_steady.py`, `core/Simulator/bp_simulator.py`).
     - *From piece 6.* No flag or control says "start a new simulation cache here" under an unchanged
       identity: `--resume never` refuses when the run's own cache holds batches, and `artifacts rm`
       refuses a cache while any record names it. The routes that exist: building a new prior, which
       re-keys the cache (the first run is refused as one setting away and goes ahead with `--new-run`);
       deleting from the leaves up, which the Artifacts screen can also do; a fresh records root; and
       `--checkpoint-every 0`, which cannot resume and leaves the old cache for the next run at the
       default cadence to reuse. It joins the load-check audit, since a change to the cycle floor needs
       exactly this (`core/orchestrator.py`, `core/artifacts/store.py`).
     - *From piece 6.* Smoke run 1's masked-probe count is not reproducible across processes on the
       card: 260 and 254 of 704 have both been read with identical settings, one borderline row
       flipping together with the Fisher's near-zero eigenvalue. The gate reads a ±12-point band, so it
       passes either way (`core/SBI/pipeline.py`, `core/SBI/decorrelate.py`).
     - *From piece 6's documents review.* Two sbi behaviours the guide and the `--max-epochs` help now
       state were read in the installed sbi 0.25.0 only: its loop counts the ceiling inclusively, so
       `--max-epochs N` trains up to N + 1 epochs; and a run the ceiling stops keeps its LAST epoch's
       network, not its best, because sbi restores the best only when the early-stop patience runs
       out. Re-read both in `sbi/inference/trainers/npe/npe_base.py` and `trainers/base.py` when the
       pinned 0.26.1 is installed (the sbi pin item above), and correct the pages and the help if
       either changed (`core/SBI/train.py`, `core/tool/config_args.py`).
     - *From piece 6's documents review, for the owner before the retrain.* The runbook stops the
       retrain when the projected fit exceeds about 740 hours, twice the 370 hours expected for the
       256 × 10 flow; the plan had set twice about 92 hours (the 128 × 8 fit scaled to the retrain's
       rows), which would stop a normally costed run, and asked the owner to confirm. Confirm 740 hours
       or choose another before pre-flight step 2 (`docs/guide/retrain.md`).
     - *From piece 6's documents review — science, for the owner.* The runbook judges both the
       12-batch `probes mask` audit and the 10,000-batch run total against 37 % ± 12 points. That band
       was derived as about two standard deviations of a four-batch smoke run; by the same model, with
       the batch count as the effective sample size, the standard deviation is about 3.5 points at 12
       batches and about 0.12 at 10,000. A systematic shift of the tier-1 prior's masked fraction may
       justify a wide band, but nothing says so. Decide the band for each (`docs/guide/retrain.md`,
       `core/diagnostics/probes.py`).
     - *From piece 6's documents review.* `UserPrior._local_map` returns its accepted set unsorted
       (`list(accepted_params)`) where the built-in models return `sorted(...)`, so a user model's prior
       fit is not reproducible from its accepted points although the mixture's random state is fixed;
       adding `sorted()` is a behaviour change (`core/SBI/Priors/user_prior.py`).
     - *From piece 6's documents review.* A narrowing round from an unrotated parent truncates the
       first eligible parameter axes in box order (`eye(p)`), which has nothing to do with how well each
       is constrained. Refuse unrotated rounds, or order their axes? The owner decides
       (`core/SBI/truncate.py`, `core/orchestrator.py`).
     - *From piece 6's documents review.* When neither state file of a simulation cache reads, or
       `state.pt` fails and `state.prev.pt` reads 0, the next run takes the fresh path: `create()`
       rewrites `header.pt` (a new schedule, initial states and rotation) and every shard is set aside
       into `uncommitted/`. The set-aside rows then belong to a header that no longer exists, so
       `set_aside_uncommitted`'s "moving them back is a plain move" restores nothing in that case.
       Refuse the fresh path over a cache that holds shards, or keep the old header beside them? A
       Windows file lock that refuses the set-aside move fails the run at start (nothing is lost; no
       retry) (`core/SBI/pipeline.py`, `core/SBI/training_checkpoint.py`).
     - *From piece 6's documents review.* A calibration repeated with `validate --seed N` reproduces
       its set only on the device the record names (`config.device`). `ArtifactStore.load_posterior`
       loads onto whatever device the run uses and compares no device, so a repeat on another device
       runs without a word; the reviewer's page now says to pass the recorded device, and a check
       would make it a refusal (`core/artifacts/store.py`).
     - *From piece 6.* A calibration's caveat line and its `results["cal_n_scales"]` record the
       REQUESTED number of operating points (200 by default) while fewer are drawn when `n_cal` is
       small, because a batch is at least 10 datasets wide (at `n_cal` 40, ceil(40/10) = 4 are drawn),
       so the record misstates what ran. Harmless at the retrain's 2000 / 200 (`core/SBI/analysis.py`,
       `core/orchestrator.py`).
   - *From piece 6's whole-code review (Task 25), parked with reasons* (the full list, with each reason and
     cost, is `review/synthesis.md` "Parked" in the gitignored ledger folder):
     - A tier-1 box whose units file declares no force token fails with a bare `ValueError` at the prior's
       `store.create`, after the stability sweep — about 9 minutes lost before what should be a pre-spend
       refusal; not on the card's or the retrain's path, whose units declare a force unit
       (`core/artifacts/store.py`, `core/sim_config.py`).
     - `probes drive`: a drive harmonic past Nyquist aliases (detune 1.4's second harmonic, for a peak above
       about a quarter of Nyquist) and is not flagged — a fast lab cell's check could miss a sub-harmonic
       caveat (`core/diagnostics/probes.py`).
     - Band's `_NYQUIST_SHARE = 0.9` is a literal separate from training's `0.9 * nyq` (and eight
       user-visible "0.9 x Nyquist" strings): after a change to training's share, band masks at the old share
       until the mask audit flags it (`core/diagnostics/probes.py`, `core/SBI/chi.py`).
     - The source scan has no `Phase <digit>` pattern, so a new build-phase banner would pass it
       (`tests/test_source_hygiene.py`).
     - Inherited Fisher eigenvalues are not checked against the parent's V (an API-only risk); a narrowing
       child without them records None silently; a non-finite header eigenvalue fails a resume
       (`core/SBI/training_checkpoint.py`, `core/orchestrator.py`).
     - The rest are test-coverage gaps (an observer on the resume branch, the mask audit's absorption case
       at ≥1024 rows, the real-rows too-slow side), wording drift and stale line numbers in comments the
       documents review later removed.
   - *From piece 6's documents review (Task 37), parked with reasons* (`docs-review/synthesis.md` §(6) and
     `docs-review/recheck.md`, gitignored): "sidecar" survives as history mentions of the retired `.rot.pt`
     file (`core/SBI/reparam.py`, `ArtifactStore.load_posterior`'s docstring), in "so this needs no sidecar"
     (`core/SBI/run_guards.py`), in `reparam.posterior_mode`'s `sidecar` parameter and in two test names —
     history and identifiers, not false; a 135-character docstring line in `core/SBI/training_checkpoint.py`;
     RC3's choice of stand-in values for a preparation whose parameters are unknown is written neutrally on
     `docs/guide/recordings.md` for the owner to settle.
   - *From piece 6's final slow set.* The chi full-pipeline slow test ran 2618 s at the piece's close (quiet machine),
     against 1397 s at piece 5, 1610 at piece 4 and 1181 at piece 3; its body changed only in a comment. The two FDT
     slow tests also ran longer than at piece 5 (619 s against 562, 452 against 323), though all of them ran faster
     than at Task 25 on a loaded machine; the card run's GPU stage timings matched piece 4's. Profile it if it grows again
     (`tests/test_user_sbi.py::test_chi_mode_full_sbi_pipeline`, `core/SBI/pipeline.py`).
   - Piece 3 did not take the optional tidy-up of `decorrelate.py`'s `or` fallbacks and `prior.py`'s
     clamps (spec §1.3 "the plan, if cheap; else none"); it stays unowned.
   - Piece 3's other open minors, each judged not worth a change now: the decorator scan matches only
     the bare `public_entry` spelling; `sbc` records `cal_n_scales` as None where
     `validate_calibration` records the resolved 200; `build_posterior` converts its knobs with
     `int()`/`float()` before the rules, so a non-number raises a bare `ValueError`;
     `build_experiment_observation` hashes recordings one at a time, so a missing second recording is
     refused after the first was hashed; `truncate.check_checkpoint_V` keeps two near-unreachable bare
     `ValueError`s; nothing pins `observations._DRIVE_FIELD` against the field registry; the
     cadence-resolution idiom appears three times in `orchestrator.py`; `RunLog.detach()` without an
     `attach()` would set `warnings.showwarning` to None (only `capture_run` calls them, always
     paired); several test suites keep dead imports. The full list with reasons is in the gitignored
     ledger `.superpowers/sdd/2026-09-16-validation-and-logging/progress.md`.
   - The rest of piece 2's final review's 46 "left open" items are test-coverage gaps and cosmetics,
     each with its reason, in the gitignored `.superpowers/sdd/2026-09-12-one-flow/final-review.md`.
8. **The retrain — NEXT.** From `docs/guide/retrain.md`, after the owner's rows F1–F5 and the
   stop-point confirmation (the open list).

## Clean break — EXECUTED 2026-09-11 (design spec §9 / plan Task 14)

- **Pre-flight.** A read-only four-lens audit (runtime code, tests, scripts and imports,
  Reduction and misc) found NO live dependency on the seven trees, on `sbc_run.log` or on the
  two scripts: every generated kind resolves through the store or `artifacts_root()`, the only
  on-disk assertion in the suites is that `Resources/Bounds` is a directory, the one source scan
  that walks `scripts/` (`tests/test_conditioning_repair.py:923`) has no needle in either script,
  and `core/Reduction` writes only under `artifacts_root()/reduction`. Only stale comments (below).
- **`8dbd93e`** — `git rm --cached` of the 20 tracked-but-ignored files: 7 `CrossValidation/*.h5`,
  5 `Plots/*`, `Priors/shm.pt`, 6 `ReductionMap/*.parquet`, `sbc_run.log` (6.9 MB, still in
  history). `git ls-files -i -c --exclude-standard` is empty since.
- **Deleted from disk** (no commit; all gitignored): `Resources/{Priors,Posteriors,Checkpoints,
  Observations,Plots,CrossValidation,ReductionMap}` — 1351 files, ~32 GB (Checkpoints alone:
  `QUARANTINED_tsnpe_round1_truncated_rows_train_0b471d560271` 5.2 G, `train_3780fd37a16a` 11 G,
  `train_230ae7cb5fc2` 5.2 G, `train_6c80f7d8037d` 4.9 G, `train_98aebd93ed17` 4.9 G,
  `train_85226b80f5d1` 933 M, `train_005ff030d387` 371 M, `train_24922afa61da` 98 M) — and
  `sbc_run.log`. `Resources/` now holds only `Bounds/ Cells/ Units/ Models/` (9 / 10 / 5 / 2
  files). `archive/` untouched. `Artifacts/` does not exist yet; the GUI creates it at launch.
- **`e37df41`** — `scripts/tsnpe_round1_forensics.py` and `scripts/migrate_checkpoint_flags.py`
  moved on disk into the gitignored `archive/scripts/` and `git rm --cached`. **The one deviation
  from the plan's letter:** the plan said `git mv`, but `archive/` is ignored and holds nothing
  tracked, so a `git mv` would have re-created exactly the tracked-but-ignored class step 1
  removed; `0a84fc8` archived the five diagnostics already there the same way. Nothing imported
  either script; `scripts/` keeps 13 files.
- **`19363d2`** — `.gitignore`: the seven `/Resources/*` lines and their comment block are gone;
  `/sbc_run.log` stays ignored (its comment now says untracked and deleted); `/Artifacts/` stays.
- **Docs** — this file; `CLAUDE.md`'s sentence about the old trees is past tense now.
- ~~**Left alone, for piece 2** (stale comments, none a code path)~~ — ALL FIXED by piece 2:
  T23 (`7c5ff48`) made `inference_screen.py` and `test_nav_and_gating.py` name the store's
  `observations/` listing; `pipeline.py` and `training_checkpoint.py` say "a read-only artifact
  root (`Artifacts/`, or `PRISM_ARTIFACTS`)"; `statistics.py` dates its substitution table to the
  deleted pre-piece-1 checkpoint; `test_user_sbi.py` cites `migrate_checkpoint_flags.py` at
  `e37df41^`; `run.sh` no longer claims the working directory decides the roots.
  `smoke_train.py` and `generate_bundle_videos.py` went with their files (`187a096` removed the
  first, `7433ced` archived the second). The original list, for the record: `core/gui/screens/
  inference_screen.py:122` and `tests/test_nav_and_gating.py:273` (say the observation gate reads
  `Resources/Observations`), `core/SBI/pipeline.py:1411` and `core/SBI/training_checkpoint.py:163`
  ("a read-only `Resources/`" for a write that goes under `Artifacts/`), `core/SBI/statistics.py:506`
  (the substitution-rate table's provenance was `Resources/Checkpoints/train_98aebd93ed17`, now
  deleted — the numbers stand as history), `tests/test_user_sbi.py:47` (checkpoints "into
  `Resources/Checkpoints`") and `:2209` (cites the archived `migrate_checkpoint_flags.py` as the
  digest-migration precedent), `scripts/smoke_train.py:6`, and `run.sh:4-5` plus
  `scripts/generate_bundle_videos.py:82-84` (both claim `config.py` resolves `Resources/` from the
  CWD; it resolves from its own location). Also seen and left alone: stale
  `.claude/worktrees/{jolly-jang,trusting-einstein,upbeat-rhodes-c8d30f}` directories (git lists
  no worktrees) and six old `claude/*` branches.

## Last gate

| gate | result |
|---|---|
| `pytest --collect-only -q` | 2026-09-11 at `3f1a1b3`: 356 (355 run by the fast suite + the slow one; Reduction's 5 are collected with it) |
| fast suite, ONE process, `pytest -m "not slow" -q` | 2026-09-11 at `3f1a1b3` (branch tip): 355 passed, 1 deselected, 10 min 36 s, exit 0. **On merged `main` `461d780`: 355 passed, 1 deselected**, 115 warnings, 10 min 56 s, exit 0; the real `Artifacts/` gained nothing and the user-model suite's temporary `Resources/*/sbitest` inputs were cleaned up (the conftest teardown assertion and `git status` both clean afterwards) |
| fast suite AFTER THE CLEAN BREAK, ONE process, `pytest -m "not slow" -q` | 2026-09-11 at `19363d2` (trees deleted, scripts archived, `.gitignore` trimmed; `CLAUDE.md` and this file edited in the working tree): **355 passed, 1 deselected**, 115 warnings, 10 min 54 s, exit 0; the real `Artifacts/` still absent afterwards and the user-model suite's temporary `Resources/*/sbitest` inputs cleaned up (`git status` showed only the two doc edits) |
| fast suite AFTER THE GPU-GATE FIXES, ONE process, `pytest -m "not slow" -q` | 2026-09-11 on the working tree that became `896e8ff`+`bb38f1a` (identical content): **358 passed, 1 deselected** (355 + the three new tests, one of them the gpu-marked CUDA inference test), 125 warnings, 11 min 13 s, exit 0; the real `Artifacts/` still absent |
| fast suite AFTER THE TASKBAR-ICON FIX, ONE process, `pytest -m "not slow" -q` | 2026-09-11 at `df7491e` (the tree was committed before the run finished; identical content): **359 passed, 1 skipped** (the display-marked class-icon test, offscreen), 1 deselected, 125 warnings, 11 min 08 s, exit 0; the real `Artifacts/` — which exists now, created by the user's walkthrough launch — gained nothing |
| display-marked tests, `QT_QPA_PLATFORM=windows pytest -m display` (real screen, hidden native windows only) | 2026-09-11: `test_the_window_class_icon_becomes_ours_on_a_real_windows_display` 1 passed (RED before `df7491e` at the missing function, with the premise assertion — Qt's class icon is the stock IDI_APPLICATION handle — already passing) |
| `tests/test_gpu_paths.py` (gpu-marked; skipped without CUDA) | 2026-09-11: 1 passed on the RTX 5070 Ti, ~10 s fixture + 2 s test; RED before `bb38f1a` with the gate's exact RuntimeError (two devices in `analysis.posterior_predictive_check`) |
| slow test `pytest tests/test_user_sbi.py -m slow -q` | 2026-09-11 at `e4e60eb` (Task 13): 1 passed, 31 min 16 s, exit 0; re-run at `3f1a1b3` after the fix wave: **1 passed**, 93 deselected, 30 min 11 s, exit 0 — the full suite is green on the branch as it stands |
| five-part fast gate (per task; last at `830cee1`) | store suite 42–43 passed; `--ignore=test_user_sbi` 261; `test_user_sbi` non-slow 1 / 16 / 12 / 64 — all green |
| `core/Reduction/tests` under pytest | 2026-09-10: 5 passed (collected with the fast suite since) |
| GPU `scripts/smoke_train.py` baseline (pre-piece-1 code, `BOUNDS=Resources/Bounds/nadrowski/master.txt`, `TOBS_S=4.5`, defaults NUM_RUNS=4 RUN_SIZE=32) | 2026-09-10: CHI=1 (cell `master_spont`): prior 102 s, posterior 747 s, validate 23 s, infer 68 s, exit 0, no OOM lines. CHI=0 (cell `master_weak`): prior 101 s, posterior 226 s, validate 12 s, infer 23 s, exit 0. |
| **GPU gate after piece 1** (`scripts/smoke_train.py` at `bb38f1a`, same BOUNDS/TOBS_S/defaults, `CHECKPOINT=1 SAVE=1 CKPT_DIR=<scratch>/smoke`, `PRISM_ARTIFACTS` pointed at scratch) | 2026-09-11: **run 1** CHI=1 `master_spont`: prior 99 s, posterior 750 s, validate 22 s, infer 59 s, exit 0 — within a few seconds of the baseline on every stage; wrote `smoke_prior`, `smoke_posterior`, `simulations/ff956b932534` (`"complete": true`, 4 batches, rows [128, 122]), an observation, a calibration and an inference. **run 2 as first recipe'd** (`PRIOR=smoke_prior NUM_RUNS=2 STAGES=prior,posterior`): exit 0 but NO resume — prior loaded in 1.4 s, then a new `simulations/294e1c3b81ec` (n_runs 2) and a recomputed Fisher, posterior 544 s (the recipe correction in item 3). **run 2 corrected** (`NUM_RUNS=4`): "Reusing the Fisher rotation stored with the training checkpoint (4/4 batches — COMPLETE, so generation will be skipped)", `[checkpoint] resuming at batch 4/4`, all stages in 3.3 s, exit 0 — the CPU→CUDA rehoming of the stored V is certified. **run 3** CHI=0 `master_weak` (own store `<scratch>/smoke_chi0`): prior 99 s, posterior 219 s, validate 12 s, infer 21 s, exit 0. No OOM lines, no OOD warnings in any leg; run 1's training masked-probe counts 121–159 of 300 (40–53 %), inside the documented ±12 pp band around 37 %. **First attempt at `19363d2`:** run 1 failed in infer (two devices in the PPC) after prior/posterior/validate passed; run 3 failed at the prior's own read-back (fingerprint) — both fixed, see item 3. Scratch stores deleted afterwards. |
| **GPU gate of record, piece 2** (`python -m core smoke`, the four command lines of `CLAUDE.md`, at `0d96d2f`) | 2026-09-14: **run 1** chi `master_spont`: prior 92 s, posterior 198 s, validate 22 s, infer 85 s, exit 0; **run 2** `--resume require`: "Reusing the Fisher rotation stored with the training checkpoint (4/4 batches — COMPLETE, so generation will be skipped)", `[checkpoint] resuming at batch 4/4`, exit 0 in 8 s; **run 2b** `--num-runs 2`: **exit 1** in 6 s, "differs only in n_runs: this run 2, that cache 4", no `[fisher]` line, `simulations/` unchanged (still only `4d8022b100db`); **run 3** forced `master_weak` (own store): prior 92 s, posterior 61 s, validate 14 s, infer 20 s, exit 0. No OOM lines and no Traceback in any run; run 1's training masked probes 79/224, 15/96, 108/192, 58/192 = 260/704 (36.9 %), inside the ±12 pp band around 37 %. Diagnostic card runs, `PRISM_ARTIFACTS` at the same stores, all on the card: `sbc --chi --repeats 1 --n-cal 20` (19 s), `identifiability jacobian --chi` (84 s), `ablation --chi` (6 s) against run 1's store, `identifiability laplace` (128 s) against run 3's — each exit 0, one `diagnostics/` directory each. Scratch stores deleted. **Timings: prior and validate match `bb38f1a` within seconds; posterior and infer do NOT and are not comparable** — `smoke` builds a truth-free config, so the Fisher anchors on the prior median plus seven prior draws (under a numpy-seeded run, different draws) instead of `master_spont`'s truth, and the Fisher's simulation cost is set by those operating points (1341 simulation segments in the posterior stage at `50a9bb6`'s `smoke_train` run vs 608 here). For the same reason V and `fisher_spread` (now `inf`: a numerically zero smallest eigenvalue, where `smoke_train` printed 5.26e16) are not comparable. The interim check at `50a9bb6` (still `smoke_train`, truth-anchored) matched `bb38f1a` on every stage: run 1 93/724/21/59 s, run 3 94/212/12/24 s, masked 37.8 %. **Not repeated after `0d96d2f`:** the only later line that creates a tensor is the final review's fix in `build_experiment_observation` (`2010250`), which builds `cfg.chi_obs_freqs` on `cfg.hw.device` as `generate_observations` does; no `smoke` command reaches that line, because `smoke` simulates its observation; and the gpu-marked `tests/test_gpu_paths.py` ran on the RTX 5070 Ti inside the final fast gate. |
| display walkthrough (`docs/checklists/display-walkthrough.md`) | 2026-09-11, the user on the real screen at `df7491e`: rows 1–20 all pass (row 1 after the taskbar-icon fix) and rows A1–A9 all pass. Piece 4 re-checks only the rows it changes |
| `pytest --collect-only -q` after piece 2 | 2026-09-15 at `d34997c`: **460** collected (the fast gate's 457 passed and 1 skipped, plus the 2 slow tests; Reduction's 5 are collected with them). That is 99 more than the 361 at `df7491e`, where spec §8.5 had budgeted about 42 ± 8 added (spec §11 row 58). Two new suites, `tests/test_tool.py` (the tool, in-process through `main(argv)`) and `tests/test_diagnostics.py`; `tests/test_*.py` holds sixteen suites |
| **final fast gate, piece 2**, ONE process, `pytest -m "not slow" -q --durations=15` | 2026-09-15 at `d34997c`, after the whole-branch review's fixes: **457 passed, 1 skipped** (the display-marked class-icon test, offscreen), 2 deselected (the two slow tests), 174 warnings, **13 min 38 s** (inside spec §8.5's 14-minute target), exit 0. No "More than 20 figures" line. The real `Artifacts/` gained nothing, and `Resources/Models` held only `SHM.json` and `SHM2.json` afterwards. The gpu-marked `tests/test_gpu_paths.py` ran on the RTX 5070 Ti inside it. Slowest test: `test_user_sbi.py::test_train_and_validate_without_a_loaded_cell`, 180 s. Per-suite durations were not measured, so §8.5's ~90 s targets for `test_tool.py` and `test_diagnostics.py` are unchecked. Every task had its own one-process gate; those results are in the execution ledger (spec §11 row 42) |
| slow set, `pytest -m slow -q --durations=5` (piece 2) | 2026-09-15 at `d34997c`: **2 passed**, 458 deselected, 96 warnings, **30 min 53 s**, exit 0. `test_user_sbi.py::test_chi_mode_full_sbi_pipeline` 1606 s (26 min 46 s); `test_tool.py::test_fdt_and_crossval_run_at_tiny_size` 243 s. No "More than 20 figures" line (the slow chi test's sink now closes its figures). The warnings are the known library classes (sklearn KMeans, sbi's SBC-count and `__array__` warnings, nflows `triangular_solve`, the reparam batch cap) plus the expected chi probe-masking notices; `Resources/` was left clean. The set is the chi full-pipeline test in `tests/test_user_sbi.py` (it now passes the drive frequencies it simulated as `(path, Hz)` pairs, D9) and the FDT/crossval tiny-size run in `tests/test_tool.py`. The chi test feeds back exactly the frequencies it simulated, so it cannot tell a stale recorded frequency from a correct one; `tests/test_artifact_store.py::test_an_experimental_observation_records_its_own_length_and_drive_frequencies` pins that |
| display walkthrough, piece-2 rows B1–B8 (`docs/checklists/display-walkthrough.md`) | 2026-09-15, the user on the real screen, piece 2 as pushed: **rows B1–B8 all pass** — the D8 load dialog and its Cancel, the round's own install with no dialog, the other-observation box, the D7 near-miss dialog, the TSNPE stage checks (0 directions, HPD 0.95, the near-miss refusal after reloading the amortized parent), the short-T_obs warning, the D12 refusal, and the FDT panel's full run after `bb22ac4` (B8, added in `d6b1f03`). Rows 1–20 and A1–A9 stand from 2026-09-11 |
| `pytest --collect-only -q` after piece 3 | 2026-09-17 at `9e2f7ef`: **603** collected (the fast gate's 600 passed and 1 skipped, plus the 2 slow tests), 143 more than the 460 at `d34997c`, where spec §8.4 had budgeted about 80 ± 20 added — over the budget and unrecorded in spec §11. `tests/test_*.py` holds seventeen suites (the new one is `tests/test_refusals.py`, torch-free) |
| **final fast gate, piece 3**, ONE process, `pytest -m "not slow" -q --durations=15` | 2026-09-17 at `9e2f7ef`, after the whole-piece review's fixes: **600 passed, 1 skipped** (the display-marked class-icon test, offscreen), 2 deselected, 175 warnings, **12 min 13 s** (inside spec §8.4's 16-minute target), exit 0. `git status` clean afterwards, no `sbi-logs/` at the root, and the real `Artifacts/` gained nothing (it still holds only the `fdt/` folder from the user's walkthrough on 2026-09-15). The conftest teardown also asserts the real `PRISM.ini` is unchanged. The gpu-marked `tests/test_gpu_paths.py` ran on the RTX 5070 Ti inside it. Slowest: `test_user_sbi.py::test_train_and_validate_without_a_loaded_cell` 147 s, then `test_calibration_theta_star_lies_inside_the_region_when_one_is_given` 55 s and `test_no_forcing_user_model_full_sbi_pipeline` 41 s. The warning baseline is 175 since T15 (sklearn's KMeans notice through the `screen_run` fixture). Every task had its own one-process gate, recorded in the execution ledger; the last before the review, at `e78cc8d`, was 594 passed in 12 min 08 s |
| slow set, `pytest -m slow -q --durations=5` (piece 3) | 2026-09-16/17 at `e78cc8d`, run during the read-only whole-piece review: **2 passed**, 595 deselected, 104 warnings, **23 min 13 s**, exit 0. `test_user_sbi.py::test_chi_mode_full_sbi_pipeline` 1181 s; `test_tool.py::test_fdt_and_crossval_run_at_tiny_size` 208 s. No "More than 20 figures" line; `git status` clean and no `sbi-logs/` afterwards. The warnings are piece 2's classes; the count moved from 96 because each chi probe-masking notice carries its masked count in its text, so the de-duplicated total follows the random draws. It stands for the piece: the fix wave `87a0e06`..`9e2f7ef` reaches the two slow tests' paths only through happy-path checks, which the final fast gate covers |
| **GPU gate of record, piece 3** (`python -m core smoke`, the four command lines of `CLAUDE.md` with the same arguments, run from a bash script that kept each run's stdout, stderr and exit code apart, at `9e2f7ef`) | 2026-09-17, alone on the card: **run 1** chi `master_spont`: prior 92 s, posterior 196 s, validate 22 s, infer 84 s, exit 0 (piece 2 at `0d96d2f`: 92/198/22/85); **run 2** `--resume require`: "Reusing the Fisher rotation stored with the training checkpoint (4/4 batches — COMPLETE, so generation will be skipped)", `[checkpoint] resuming at batch 4/4`, exit 0 in 8 s; **run 2b** `--num-runs 2`: **exit 1** in 6 s, the refusal on stderr reads "differs only in n_runs: this run 2, that cache 4" and ends `(--new-run)`, no `[fisher]` line, `simulations/` still only `4d8022b100db`; **run 3** forced `master_weak` (own store): prior 92 s, posterior 60 s, validate 13 s, infer 18 s, exit 0 (piece 2: 92/61/14/20). No OOM line, no Traceback and no `warning: ` record line in any run. Run 1's training masked probes 79/224, 15/96, 108/192, 58/192 = 260/704 (36.9 %), the same as piece 2. **New this piece:** `smoke_posterior/log.txt` holds stamped lines (`01:14:03 info [budget] 4 batches x 32 rows = 128 training rows`, the two `[fisher]` lines, the masking notices at `warning`); every committed artifact in both stores has a `log.txt`, and neither `simulations/<digest>/` has one; the ground-truth `PreflightWarning` is reported at `core/tool/smoke.py:235`, the caller, not at `core/runs.py`. The diagnostic card was not run: piece 3 changed guards, messages and logging under `core/diagnostics`, but no changed line there creates or moves a tensor (checked by grep over the piece's diff), and the tool's new `--device cuda` choice resolves to `config.detect_device()`, the same device configuration `auto` picks on this machine. Scratch stores deleted |
| display walkthrough, piece-3 rows C1–C11 (`docs/checklists/display-walkthrough.md`) | 2026-09-17, the user on the real screen at `9e2f7ef`: **rows C1–C11 all pass** — the Config tab's read-only χ drive and band, the blank `T_obs` refusal in the yellow box on both Infer pages, the 0-directions refusal, what a TSNPE relaunch restores (picker and Batches yes; HPD and Directions back at `config.py`), D12 in the yellow box, the Posterior tab's science knobs opening at `config.py`, the budget group's three lines through train/3/2/blank, the bench-then-simulated probe count (4 recorded, the table's 2 rows kept), the log pane's triangles and the posterior's `log.txt` with no `log.txt` under `simulations/`, the FDT consents not remembered, and the Config tab's K, slots and lock-in ceiling opening at `config.py` while the model, units and χ tick stay. Rows 1–20, A1–A9 and B1–B8 stand |
| `pytest --collect-only -q` after piece 4 | 2026-09-21 at `80be144`: **732** collected (the fast gate's 729 passed and 1 skipped, plus the 2 slow tests), 129 more than the 603 at `691a233` (piece 3 closed), inside design spec §9.4's stated band of 130 ± 30 added. `tests/test_*.py` holds eighteen suites (the new one is `tests/test_artifact_browser.py`) |
| **final fast gate, piece 4**, ONE process, `pytest -m "not slow" -q --durations=15` | 2026-09-21 at `80be144`, after the whole-piece review's fix wave: **729 passed, 1 skipped** (the display-marked class-icon test, offscreen), 2 deselected, **181 warnings**, **13 min 57 s** (inside design spec §10's 16-minute target), exit 0. Tree clean; the real `Artifacts/` gained nothing. The warning baseline moved from piece 3's 175 to 181: all six new warnings come from the two tests Task 22 added that run a real short training and emit its per-batch notices — no pre-existing test started warning and no new warning class appeared. Every task had its own one-process gate, recorded in the execution ledger; the gate briefly touched 16:07 and 16:12 at Tasks 22–23 before the fix wave brought it back under target |
| slow set of record, `pytest -m slow -q --durations=5` (piece 4) | 2026-09-21 at `80be144`: **2 passed**, 730 deselected, 93 warnings, **30 min 43 s**, exit 0. No "More than 20 figures" line, no `sbi-logs/`, tree clean. `test_user_sbi.py::test_chi_mode_full_sbi_pipeline` 1610 s; `test_tool.py::test_fdt_and_crossval_run_at_tiny_size` 229 s (back near piece 3's recorded 208 s, so an earlier 428 s reading was transient). The chi figure is 13 % above piece 3's recorded 1181 s on code this piece's diff does not reach on that test's path; it passed at every measurement and is recorded as a thing to watch at the retrain, not chased further ("Owed" item 7) |
| **GPU smoke gate, piece 4** (`python -m core smoke`, the four command lines of `CLAUDE.md`, at `d1f0b98` — no line on the training path changed after it except the fix wave's sweep/delete/logging work, which that path does not reach) | 2026-09-21, alone on the card: **run 1** chi `master_spont`: prior 92.3 s, posterior 198.5 s, validate 22.3 s, infer 85.9 s, exit 0 (piece 3: 92/196/22/84); **run 2** `--resume require`: "Reusing the Fisher rotation stored with the training checkpoint (4/4 batches — COMPLETE …)", `[checkpoint] resuming at batch 4/4`, exit 0; **run 2b** `--num-runs 2`: **exit 1**, the refusal reads "differs only in n_runs: this run 2, that cache 4" and ends `(--new-run)`, no `[fisher]` line, `simulations/` still only `4d8022b100db`; **run 3** forced `master_weak`: 100.4 / 63.9 / 15.0 / 19.3 s, exit 0 (piece 3: 92/60/13/18). No OOM line, no Traceback and no `warning: ` line in any run. Run 1's masked probes 79/224, 15/96, 108/192, 58/192 = 260/704 = 36.9 %, identical to pieces 2 and 3. The diagnostic card was not required and was not run: the piece's diff touches no file under `core/diagnostics` that creates or moves a tensor (design spec §10); that judgement held. Scratch stores deleted |
| display walkthrough, piece-4 rows D1–D17 (`docs/checklists/display-walkthrough.md`) | 2026-09-21, the user on the real screen at `44ed3ed` (piece 4 as it stands; the code is `80be144`, that commit being documents only): **rows D1–D17 all pass** — the Artifacts tile and its seven kinds, the empty kind's "Nothing here yet.", a cache's batches-against-planned row and its finished form, the detail pane's manifest-then-records and the cache's no-log sentence, the note set/cleared/refused, delete refused by a dependent (the fingerprint-only cache included), the unfinished cache's batch-count confirmation with No then Yes, the sweep of manifest-less directories only, the picker refreshed after a delete, Note/Delete/Sweep refused while a run is live with reading still working, the live-run header line with the tile and tab markers, Apply's session line and its No, "narrowed (TSNPE)" in the closed dropdown, the blank probe row reported as blank, no doubled `core` line and the `library: <logger>: ` prefix, the taskbar mark after the header's new slot (row 1 re-run), and the table's colours, sorting and column widths in Light and Dark. Rows 1–20, A1–A9, B1–B8 and C1–C11 stand |
| `pytest --collect-only -q` after piece 5 | 2026-09-24 at `7e51275`: **968** collected (the fast gate's 964 passed and 1 skipped, plus the 3 slow tests), 236 more than the 732 at `80be144` — **above** design spec §8.4's stated band of 170 ± 40 added, by 26. The 40 tasks added 185 (917 at `7b12c34`, inside the band); the whole-piece review's fix dispatch added 51 more (its fixes' tests and its test-only items N31–N42); the overrun is recorded here, not in the design's §12. `tests/test_*.py` holds nineteen suites (the new one is `tests/test_fdt_compare.py`) |
| **final fast gate, piece 5**, ONE process, `pytest -m "not slow" -q --durations=15` | 2026-09-24 at `7e51275`, after the whole-piece review's fix dispatch and Task 41's documents, on a quiet machine: **964 passed, 1 skipped** (the display-marked test, offscreen), 3 deselected (the three slow tests), **181 warnings**, **14 min 42 s** (882.29 s; wall 14 min 47 s — inside design spec §9's 16-minute target), exit 0. Tree clean; the real `Artifacts/` gained nothing (still `fdt/` with the two legacy PNGs, and `priors/`); no `sbi-logs/`. The warning baseline is piece 4's 181, unchanged across the whole piece: no new warning class appeared. Slowest: `test_user_sbi.py::test_train_and_validate_without_a_loaded_cell` 85.07 s, `::test_calibration_theta_star_lies_inside_the_region_when_one_is_given` 70.89 s, `::test_no_forcing_user_model_full_sbi_pipeline` 42.06 s. Every task had its own one-process gate, recorded in the execution ledger, and so did the fix dispatch (`6be832c`: 964 passed, 181 warnings, 15 min 18 s) |
| slow set of record, `pytest -m slow -q --durations=5` (piece 5) | 2026-09-24 at `c29320c` (the same code as `7e51275`, which changed documents and two docstrings only), beside the read-only Task 41 document review: **3 passed**, 965 deselected, 102 warnings, **38 min 6 s** (2286.51 s; wall 38 min 8 s), exit 0, against design spec §9's ~45-minute budget. `test_user_sbi.py::test_chi_mode_full_sbi_pipeline` 1396.87 s (piece 4: 1610 s); **new**, `test_tool.py::test_fdt_runs_the_nadrowski_sanity_checks_end_to_end` (Task 33) 562.38 s; `test_tool.py::test_fdt_and_crossval_run_at_tiny_size`, rewritten by Task 34 to assert records and on the fixed seed 20260925 since the review's N40, 322.93 s (302 s solo at Task 36). Afterwards no `Resources/*/sbitest` (the two slow tool tests no longer request `tool_env`, which installed SBITEST into the real `Resources/`; they set `PRISM_ARTIFACTS` themselves, N38), no `sbi-logs/`, tree clean, the real `Artifacts/` unchanged |
| **GPU smoke gate, piece 5** | 2026-09-24: **judged NOT REQUIRED; the card was not run.** The judgement was re-made against the whole piece, `a0d85da^..c29320c`. The changed lines that create or move a tensor are in `core/FDT/cross_validation.py`, `fdt_pipeline.py` and `sanity.py` — all on the FDT and sweep path, pinned to `cpu_device()` by `cli.make_fdt_config` and `make_param_sweep_config` — in `core/gui/panels/simulate_runner.py` (N17's zero-forcing tensor, on the Live simulation's CPU-pinned path, which the gate does not reach), and in `core/rng.py`: `seeded`'s CUDA branch (fork the device, `torch.manual_seed`) is the base's byte for byte, and its new CPU branch seeds `torch.default_generator`, the CPU half of `manual_seed`, so no CPU draw changes. `core/diagnostics/rng.py` re-exports the same object, and `core/orchestrator.py` only re-imports `PreflightWarning` from `core/refusals.py`. **T28's change to `core/tool/__init__.py` and `core/tool/smoke.py` IS on `smoke`'s path**: it chooses the store root through `temp_store_root` (set by `set_defaults` on smoke's parser) instead of the flag's presence, creates or moves no tensor, and T28's tests pin the root choice; the gate always passes `--store-root` anyway. The fix range `d13ca96..6be832c` touched nothing under `core/SBI`, `core/diagnostics`, `core/orchestrator.py` or `core/tool/smoke.py`, and `core/rng.py` only in its docstring (the re-review verified); nothing under `core/SBI`, `core/Simulator`, `core/Solvers` or `core/Models` changed in the piece. The diagnostic card is not needed either: no line under `core/diagnostics` that moves a tensor changed (only `rng.py`'s re-export). Piece 4's run (`d1f0b98`) remains the last card measurement |
| display walkthrough, piece-5 rows E1–E17 (`docs/checklists/display-walkthrough.md`) | reported 2026-09-25, the user on the real screen (piece 5 as it stands; the code is `7e51275`, every later commit being documents only): **rows E1–E17 all pass** — the date and result columns of each row record it |
| **GPU card run, piece 6** (`python -m core smoke`, the five command lines of `CLAUDE.md`, run 4's resume, the narrowing leg, the diagnostic card and the probe checks, at `16e5bae` — the code of the whole-code review's fixed head `04fbe54`; the later commits are documents only) | 2026-09-29, alone on the card (after the drive parity run the same evening: `probes drive` on `master_spont` at window 0.018 over the archived sixteen strengths, exit 0 in 13.8 s, Ω₀ 22.600 Hz, clarity 1.41e+04, strongest free-running **0.02**, weakest captured **0.2** — the 0.02 default window accepted): **run 1** chi `master_spont`: prior 92 s, posterior 194 s, validate 22 s, infer 76 s, exit 0 (piece 4: 92.3/198.5/22.3/85.9), training `[chi] masked probes` 254 of 704 (36.1 %: 79/224, 15/96, 102/192, 58/192), cache `5d89e2242bd2` (the identity format moved it from `4d8022b100db`); **run 2** the reuse line and `[checkpoint] resuming at batch 4/4`, exit 0 in 7.7 s, masked line equal to run 1's; **run 2b** exit 1 in 5.8 s, "differs only in n_runs: this run 2, that cache 4 … (--new-run)", no `[fisher]` line, `simulations/` unchanged; **run 3** forced `master_weak`: 91/60/13/20 s, exit 0 (piece 4: 100.4/63.9/15.0/19.3); **run 4** tier-1, 256 × 10: 92/150/20/45 s, exit 0, both `[tier1]` lines, `rescale order: ['x_scale', 't_scale', 'T']`, masked 285 of 704 (40.5 %), T in the posterior's keys, `config.tier1` (`k_b_cell` 0.01380649, `T_range` [280.0, 310.0]) and `assumed_params` ["T"], 13 eigenvalues (7.949e+16 … 0.1652), T marked "(assumed input)" in the calibration table, `[verdict] FAIL` (a result at this size), observation `simulated_f_scale` 46.99; **run 4 resume** exit 0 in 7.8 s, `[checkpoint] the stored rotation's eigenvalues come with it`, eigenvalues carried, Fisher settings null, `resumed_from_batch` 4; **narrowing leg** (`PRISM_ARTIFACTS` at run 4's store) tsnpe 12.6 s exit 0 (direction 0 left full width: t_scale-loaded), truth inside all five truncated directions, `t1_round` carries the parent's eigenvalues; validate 2571 s (the stage's default size on a truncated posterior) exit 0, `[verdict] FAIL` (T's rank test fails at this size, a 5-epoch round), seed 401191828; infer 42 s exit 0; accepted [] (round) / ["truncated"] (calibration) / ["truncated", "other_observation"] (inference); **diagnostic card** sbc 18 s (rank verdict 0/1 repeats), jacobian 85 s, ablation 6 s, laplace 128 s, tier-1 jacobian 85 s with T column (a measurement) ‖g‖ 0.214, unique 0.134, |cos| with n 0.92 (degenerate pair n~T: the force scale is proportional to n·T), with t_scale 0.49, with x_scale 0.44; **probes** band: the configured (0.03, 0.3) and drive 0.15 hold, the 0.6× control captured at every length (own-peak share 0.12–0.18); drive (default grid, the configured 0.15 included): free-running 0.02, captured from 0.15 (0.1 in between); mask over `smoke_prior`: 753 of 2,144 masked (35.1 %), the cycle floor the only cause (too slow at the band's top 31.2 %, shortened by the duration draw 4.0 %), invariants hold, cross-check confirms all 12 row ranges. No OOM line and no Traceback in any run; `warning:` lines only the expected ones (a non-amortized load, three pathological-trajectory notices in the round's calibration). **Run 1's masked count is not reproducible across processes on the card:** pieces 2–4 recorded 260 three times; an export of this piece's first commit gave 260 again, but a bisect over the piece's commits found batch 3 at 108 or 102 (one row's six probes) flipping between TEST-ONLY commits, always together with the Fisher step's eigenvalue spread (≈1.7e+14 with 102; inf or 3.78e+16 with 108) — the Fisher's smallest eigenvalue is near zero and its simulation is not bitwise stable on CUDA, and one borderline row follows it. Either count is inside the ±12 pp band. Scratch stores deleted |
| `pytest --collect-only -q` after piece 6 | 2026-09-30 at `067eddc`: **1178** collected (the fast gate's 1172 passed and 1 skipped, plus the 5 slow tests), 210 more than the 968 at `7e51275`, against design spec §7's budget of about 80 ± 25 added (above by 105: every behaviour fix was test-first, the probes family and the tier-1 path needed their own, and two new suites landed; §11 row 22). `tests/test_*.py` holds twenty-two suites (new: `test_source_hygiene.py`, `test_tier1.py`, `test_docs.py`) |
| **final fast gate, piece 6**, ONE process, `pytest -m "not slow" -q --durations=15` | 2026-09-30 at `067eddc`, on a quiet machine: **1172 passed, 1 skipped**, 5 deselected, **181 warnings**, **15 min 24 s** (inside the 16-minute target), exit 0. Tree clean; the real `Artifacts/` gained nothing; no `sbi-logs/`; `Resources/Models` holds only `SHM.json` and `SHM2.json`. Warnings 181 → 181: unchanged. Slowest: `test_train_and_validate_without_a_loaded_cell` 87.2 s, `test_calibration_theta_star_lies_inside_the_region_when_one_is_given` 73.7 s, `test_no_forcing_user_model_full_sbi_pipeline` 43.7 s. Every task had its own one-process gate, recorded in the execution ledger |
| slow set of record, `pytest -m slow -q --durations=5` (piece 6) | 2026-09-30 at `067eddc`: **5 passed**, 1173 deselected, 153 warnings, **1 h 15 min 27 s**, exit 0. `test_chi_mode_full_sbi_pipeline` 2618 s — OVER piece 5's 1397 s: its body is unchanged but for a comment, the other slow tests ran faster than Task 25's loaded reading and the card run's GPU stages matched piece 4's, and this test has varied 1181–2618 s across pieces, so it is recorded on the open list rather than chased; `test_smoke_runs_every_stage_on_the_tier1_box` 828 s; `test_fdt_runs_the_nadrowski_sanity_checks_end_to_end` 619 s; `test_fdt_and_crossval_run_at_tiny_size` 452 s; `test_probes_band_gives_the_same_criteria_on_the_master_cell_and_its_tier1_twin` 7 s. Tree clean afterwards |
| GPU card run, piece 6 — stands | `git diff --name-only 16e5bae..067eddc -- core conftest.py` lists 41 files, judged: every change is a comment, docstring, help string or message, plus the chi_f0 load comparison (floats read from a manifest), the shard set-aside (file renames) and a return annotation — none creates or moves a tensor, so the card run at `16e5bae` is the piece's card measurement. Insurance, because the set-aside sits in the training resume path: the recipe's runs 1, 2 and 2b re-run on the card on 2026-09-30 at `067eddc` — run 1 exit 0 in 412.8 s (prior 102 s, posterior 196 s, validate 25 s, infer 85 s; training masked probes 260 of 704, 36.9 % — 260 again, where the card run of record read 254: the count moves by one row between processes); run 2 printed the reuse line and `[checkpoint] resuming at batch 4/4`, exit 0 in 8.7 s, nothing set aside; run 2b exit 1 in 6.6 s naming `n_runs`, no `[fisher]` line, no new cache directory |

**The GPU gate, as command lines.** This is `CLAUDE.md`'s recipe of record (its Tests section),
copied verbatim; keep the copies identical (the documents review compares them). Run it from the
repository root, with `$S` set to an empty scratch directory. `PRISM_ARTIFACTS` is not needed for
the five `smoke` runs, which touch
only `--store-root`; the diagnostic card runs need it pointed at the same store.

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

## Decisions log

- **2026-09-10** — brainstorming: clean break; all four audiences; GUI + one tested CLI tool with
  the prompt CLI retired; reduction map out of scope; everything lands before the retrain;
  pytest adopted; artifact store shape B (directory per artifact + manifest); timestamp ids;
  auto-persist.
- **2026-09-10/11** — piece 1 execution rulings (full text in the ledger): R1 keep the old
  `build_posterior` return for one task; R2 three pre-flight plan fixes; R3 worktree; R4 piece 0's
  gate as baseline; R5 `.gitignore` keeps the `/Resources/*` lines until the runbook; R6
  parameter-keyed results are ordered record lists (manifest JSON sorts keys); R7 the region's
  prior fingerprint is the walk over the parent's pickled prior. The final review's "refuse
  before the spend" principle: a taken name, and a region without an observation digest, are
  refused at stage entry. The recorded fast gate is one process (a split gate hid a
  default-store leak once). The smoke-train artifact names are `smoke_prior`/`smoke_posterior`
  (a leading underscore is not a legal name).
- **2026-09-11** — after the merge: **no more feature branches or worktrees.** Every piece from
  here on is built directly on the local `main` branch in the one checkout; the user pushes.
  (The branch cost a merge, a conflict on a file both sides touched, and a stale copy of this file
  on `main` while it was open.)
- **2026-09-11** — clean break executed (section above). Ruling: archiving into the gitignored
  `archive/` means a move on disk plus `git rm --cached`, never `git mv` — the repository holds no
  tracked-but-ignored file again. `/sbc_run.log` stays in `.gitignore`. A file changed with an
  editor is NOT staged: `git commit` without `git add` commits nothing and says so only in its
  output (the runbook's step 4 needed a second attempt because of it; always check the log).
- **2026-09-11** — GPU gate rulings. (1) A prior's save/load round trip is the IDENTITY, bit for
  bit: `load_mix_dist` restores the saved weights rather than letting `Categorical` re-normalise
  them, because every fingerprint that names a prior hashes those bytes and torch's
  re-normalisation is device-dependent at the last ulp. Every other fingerprint comparison (the
  posterior's pickled training prior, the region, the run guards) is unchanged and now agrees
  across devices for free. (2) A loaded observation row lives on the CPU like every conditioning
  row; the flow's `sample()` is the one consumer that moves a copy. (3) The resume drill's two
  runs share `NUM_RUNS` (the identity includes `n_runs`). (4) GPU-only device errors are the class
  the smoke gate exists for: run it after every piece, before any record run; the gpu-marked
  suite covers the inference path cheaply but is not a substitute (no checkpoint resume, no real
  bounds/cell files).
- **2026-09-11** — **Taskbar icon** (walkthrough row 1, found by the user; fixed in `df7491e`).
  What was seen: the title bar showed the mark, the taskbar button Windows' generic "application"
  glyph — not python's icon either. Ruled out by measurement, each with a launch and a screenshot
  of the button: the PNG set (loads, seven sizes), the AppUserModelID (set and read back; removing
  it or using a fresh id changed nothing), the shell's icon cache, and the multi-size QIcon (a
  minimal window using the very same QIcon got the mark). What decided it: a minimal window idle
  right after showing got the mark; the same window blocked 4 s after showing got the glyph; PRISM's
  `show()` keeps the thread busy ~150 ms AFTER the native show (Qt lays out the whole tree —
  profiled: all inside `QWidget.show`), and the shell's icon query at button creation times out
  and falls back to the window CLASS icon, which Qt registers as the stock IDI_APPLICATION glyph
  because python.exe has no icon resource (the class handle read back equals `LoadIcon(NULL,
  IDI_APPLICATION)`). With the class icon set to ours, the blocked window got the mark too.
  Rulings: (1) the mark is installed as Qt's window-CLASS icon before the first show
  (`app_icon.set_windows_class_icon`, from the new `assets/app/prism.ico` that
  `build_app_icon.py` assembles from the PNG set — LoadImage cannot read a PNG); re-asserting
  `setWindowIcon` on the first loop turn also worked but depends on timing piece 4's start-up work
  could break; (2) the AppUserModelID stays for grouping/pinning and its docstring no longer
  claims it decides the icon; (3) a display-marked test pins the class-icon change on the real
  platform (`QT_QPA_PLATFORM=windows pytest -m display`), an offscreen test pins that prism.ico's
  256 frame IS prism-256.png; (4) verified by screenshot of the fixed launcher's button; the user
  re-checks row 1 on the next launch.
- **2026-09-11/15** — **piece 2 (one flow underneath): decisions D1–D13** (spec
  `docs/superpowers/specs/2026-09-11-one-flow-design.md` §1.1; plan
  `docs/superpowers/plans/2026-09-12-one-flow.md`). D1 the prompt CLI is retired WHOLE
  (`orchestrator.run`, `core/app.py`, the `__main__` menu, the prompt half of `core/cli.py`,
  `helpers.clear_screen`'s callers, `run_fdt`'s `input()` calls, whose two booleans are now
  required). D2/D6 the flags-only tool `python -m core <subcommand>`: every flag reaches its stage
  as a keyword argument, `--bounds` is required wherever a prior is built or training runs, and the
  tool reads only `PRISM_RESOURCES` and `PRISM_ARTIFACTS`. D3 three compositions beside the stages
  (`simulated_inference`, `experimental_inference`, `tsnpe_round`), each carrying the checks that
  used to live in one front end only. D4 six scripts became four subcommands (`smoke`, `sbc`,
  `identifiability` with three modes, `ablation`), six were archived, `_common` dissolved. D5 the
  `diagnostic` store kind, whose loader has no refusals. D7 a committed simulation cache ONE
  setting away (a "near miss") is refused before the Fisher step and before any simulation;
  `new_run=True` / `--new-run` overrides it; the resume policy `auto`/`require`/`never` is a stage
  argument and the `--resume` flag; the GUI dialog asks `fresh_run_near_misses`, and the stage
  restates its checks field for field (spec §11 row 2). D8 the tool
  refuses by default and `--accept-*` maps 1:1 onto `Accept`; the Posterior tab asks before it
  loads a stored non-amortized posterior. D9 every driven chi recording states its frequency in Hz;
  the legacy branch is deleted. D10 the entry sets `KMP_DUPLICATE_LIB_OK` and the Agg backend
  before any torch or core import, so `--help` stays torch-free. **D11 there is NO chi override
  anywhere: a non-default band or drive amplitude means editing `config.py` deliberately.** **D12 a
  TSNPE round whose parent is itself non-amortized and whose observation is not that parent's is
  refused, with no escape hatch.** D11 and D12 are STANDING REFUSALS: a later piece must not
  re-open them. D13 `PRISM_VRAM_CEILING_GIB` and `PRISM_MEM_LOG_EVERY` stay core-level environment
  settings, named in the tool's `--help` epilog and in `CLAUDE.md`, not arguments;
  `PRISM_VRAM_CEILING_GIB` is read live on each batch plan, `PRISM_MEM_LOG_EVERY` once, when
  `core.SBI.pipeline` is imported. Spec §1.2's readings stand: `runners.py` is deleted rather than
  wrapped; the one-setting rule still exempts a cache that differs only in its truncation region (a
  TSNPE round at its parent's budget); only `smoke` and the diagnostics take `--seed`; a
  hand-entered truth drops the cell from the recorded sources.
- **2026-09-14/15** — piece 2's working rules. **Folding versus archiving:** a FOLDED script is
  `git rm`'d (git history is its archive, and an archived copy would drift from its replacement);
  an ARCHIVED script is moved on disk into the gitignored `archive/scripts/` and then `git rm
  --cached`, never `git mv` — the same ruling as the clean break's. **Test-side knob defaults:**
  `tests/conftest.py`'s session-scoped autouse `_checkpointing_off_unless_asked` is the only
  session-wide knob default the suites install, and every test that wants a simulation cache
  passes `checkpoint_every` explicitly. **Scan roots:** `CODE_ROOTS` (the top-level directories,
  today `core`) plus `CODE_FILES` (top-level Python files, today `conftest.py`) in
  `tests/_fixtures.py` are what the source scans walk (the literal-path, `input()` and rotation-reader scans); a guard fails if a
  directory holding Python is missing from `CODE_ROOTS`, and a missing `CODE_FILES` entry fails
  with a message naming it. **The Fisher is truth-free on the command line:** `smoke` builds a
  config with no ground truth, so its rotation anchors at the prior median plus seven prior draws
  rather than at the cell's truth as `smoke_train` did. So only the prior and validate timings (and
  the masked-probe share) compare with `bb38f1a`; the posterior and infer timings, `V` and
  `fisher_spread` do not.
- **2026-09-14/15** — piece 2's execution rulings, condensed to the load-bearing ones. The ledger
  (`.superpowers/sdd/2026-09-12-one-flow/progress.md`, compiled in `rulings-and-open-items.md`) is
  gitignored scratch; the durable record is this log plus spec §11.
  - Each task's gate result goes into the ledger, not this file; this table gets only gates of
    record and the final rows (R-C).
  - A plan's predicted counts, error texts and file lists are predictions; what must hold is the
    behaviour: the test fails before the change and passes after (R-A).
  - A leftover-mention search accepts history notes ("Folded from scripts/x.py"); a hit that
    imports, builds a path to, or tells someone to run a retired thing fails it (R-B).
  - Piece 2's commit range starts after `b578f08`; a task whose check changes nothing needs no
    commit (R-H).
  - One GPU recipe text: `CLAUDE.md`'s block, copied verbatim under the gate table (R-I, R-V).
  - Refuse before the spend: `infer` checks its recording flags before loading anything, and
    `infer --cell` with any recording flag exits 2 (R-L); the near-miss refusal sits before the
    truncation refusals and the Fisher step (T2); bad knobs and zero sample counts are refused
    before any simulation (T16–T18, final review).
  - The near-miss refusal message names both GUI controls and `--new-run` (R-O).
  - The diagnostics declare and build their accept flags through the shared `config_args` helpers
    (R-T).
  - Warnings in test output are findings: a leaked warning is captured and asserted; a third-party
    warning from a real run the spec requires is accepted unfiltered and the baseline moves (T4, T6,
    T11, T12).
  - Only inferences and diagnostics record accept flags in `results.accepted`; whether a
    calibration or TSNPE child should is parked for the owner (R-S; "Owed" item 7).
  - `identifiability laplace` isolates only its fixed-seed calls, so its noise-floor draws stay
    independent per point and `--seed` governs every point (T17).
  - The GPU gate of record ran alone on the card at `0d96d2f`, after T19's fix round, and its row
    was committed separately (R-U). It passed although posterior and infer missed "within a few
    seconds of `bb38f1a`": the truth-free Fisher explains the difference, and prior and validate
    match (T19).
  - Walkthrough row B8 was added because `bb22ac4` changed the FDT plots after row 15 last passed
    (T24).
  - End order: whole-branch review, one fix dispatch and a scoped re-review, then the slow set and
    the final fast gate, then these documents from the measured numbers (R-W); the documents
    follow the ledger where the T25 brief was stale (R-X).
  - The final review's six must-fix and 21 fix-now items landed in one dispatch, each behaviour fix
    test-first; 46 items stay open with reasons. Follow-ups: the SBC rank histogram's bin count is
    the largest divisor of `nps + 1` between half the cap and the cap, else the cap; the flow-knob
    range checks run only on a call that trains; the chi probe count left on a GUI session is
    piece 3's.
- **2026-09-15/17** — **piece 3 (validation, logging, the private copy): decisions V1–V9** (spec
  `docs/superpowers/specs/2026-09-15-validation-and-logging-design.md` §1.1; plan
  `docs/superpowers/plans/2026-09-16-validation-and-logging.md`), each chosen by the user in
  plain-language questions. V1 the private copy is a CORE promise: `core.runs.public_entry` deep-copies
  the config on entry to all fifteen public stages, compositions and diagnostics, so window training is
  truth-free like the tool's. V2 a bad box is refused at the click, naming the box, the range and the
  default, with no clamp and no silent default. V3 one `Refusal` kind with a field key; core messages
  are neutral and each front end names its own control (`core/gui/fields.py`) or flag
  (`core/tool/fields.py`); a refusal opens the yellow "Check your inputs" box and a bug the red one. V4
  standard `logging` at info, warning and error, plus a `log.txt` in every committed artifact (none
  for the simulation cache). V5 the inference tabs remember SELECTIONS — the pickers, the mode, and
  the boxes that describe the recording (the three observation lengths, the physical drive amplitude
  `Drive F₀ (N)`, the recording paths) — plus the training budget; every inference science knob
  opens at `config.py` on every launch, the Config tab's non-dimensional χ drive amplitude and band
  are read-only displays of it, and a consent is never persisted. The Simulate, FDT and CrossVal
  panels still remember their own numeric fields — piece 5's (spec §1.3). V6 one
  `orchestrator.training_preview` behind the budget lines. V7 the Fisher settings are recorded
  only when the rotation ran. V8 a loaded prior closes its figure. V9 sbi's summary writer is off.
  The fast-gate target is 16 minutes, one process.
  - The spec's self-review rulings: the diagnostics convert their guards too; the simulation cache
    gets no `log.txt`; the tool's framing prints (`prism <cmd>: refused: …`, the ladder) stay prints;
    the FDT proceed box opens ticked because its consent is never remembered.
  - Execution rulings that carry weight (spec §11 holds all fifteen; rows 14–15 were added by the
    2026-09-17 document audit): `config.unit_registry` IS an
    `lru_cache` singleton, so the deep copy must share it (row 3); the stage-entry order is the name
    first in the four stages and the knobs first only in the two compositions (row 5); the judgements
    said once are `PreflightWarning`s with the `always` filter (row 7); one refusal still names the
    Settings model builder, because nothing else answers it (row 9); the worker drops the exception's
    traceback so a failed run frees its frames (row 10); warnings under `public_entry` skip
    `core/runs.py` when locating their caller (row 11).
  - Working rules: the plan's quoted code was drafted against `ca583f8` and never verified, so each
    implementer adapted a moved quote and said so; implementers ran only their focused tests in the
    foreground and the controller ran each one-process gate (two implementers stalled waiting on a
    background run of a whole suite, T8 and T16); a warning in test output is a finding, and a
    third-party warning from a run the spec requires moves the baseline (175 since T15, sklearn's
    KMeans notice through the `screen_run` fixture); commits never amended or reset (T4's reset of
    its own unpushed commit was accepted once, content unchanged).
  - End order, as in piece 2: a whole-piece review by four independent readers (correctness, spec
    compliance, tests, CPU-invisible hazards), one fix dispatch with each behaviour fix test-first, a
    scoped re-review, then the gates of record and these documents. The slow set ran at `e78cc8d`,
    during the read-only review, and stands for the piece: the fixes reach its tests' paths only
    through happy-path checks, which the final fast gate covers. The slow set's warning count moves
    with the random draws, because each probe-masking notice carries its masked count in its text.

- **2026-09-22/24** — **piece 5 (the secondary analyses): decisions E1–E12** (spec
  `docs/superpowers/specs/2026-09-22-secondary-analyses-design.md` §1.1; plan
  `docs/superpowers/plans/2026-09-22-secondary-analyses.md`), each chosen by the owner in
  plain-language questions. E1 both analyses move into the store FULLY, with a reading side: named
  records, provenance with file fingerprints, listing, deletion, the outputs inside the record's
  folder, the numbers saved and not just the pictures, and a picker on each screen. E2 an interrupted
  run KEEPS its folder, marked unfinished — the training cache's rule, adopted because these runs take
  hours. E3 ONE new kind, `fdt`, covers both, with a `study` field telling them apart. E4 nothing
  measurable is a refusal; some points failed completes with the count in the record and both
  listings. E5 checks refuse what breaks; a setting too thin to trust warns and the warning is
  recorded. E6 the fix-hint table understands SCREENS as well as the six inference tabs, and a shared
  input lists every place it appears. E7 every run records its seed, drawing one when none is given.
  E8 comparing covers all four modes (several cells, repeats of one cell, one run re-normalised, two
  sweeps). E9 the off-grid range test is fixed and the band is refused before the driven campaign.
  E10 the tidy-up command learns to see the legacy loose files; nothing is deleted on the owner's
  behalf. E11 both subcommands gain `--store-root`, and the THREE behaviours that key off its
  presence are reworked deliberately. E12 one piece, ordered so all the hardening lands before the
  first line of the comparison facility.
  - The design's own rulings (spec §1.2), the ones most likely to bite: the progressive record is a
    MODE on `ArtifactWriter`, not a second writer; the FRONT END creates the writer and the STAGE
    enters it (the run log is thread-local, so a writer entered on the window's thread would produce
    no `log.txt`); `FDTConfig` gains `copy_for_run()` and not only fields, because `public_entry` is
    duck-typed on that method and without it V1 is claimed and silently absent; a two-parameter study
    writes TWO records; a comparison names its sources in the BODY, so deleting one is not refused and
    `render_lineage` gains a branch that prints `MISSING`; `fdt` and `crossval` gain `--seed`,
    deliberately widening piece 2's recorded reading; five checked settings that no front end exposes
    (`freq_bounds`, `burn_in_nd`, `t_obs_periods`, `dt_nd`, `psd_t_obs_nd`) are registered in
    `FIELDS` like any other key, with `None` in BOTH front-end tables, so a refusal names the
    setting and `fix_sentence` offers no fix — not unregistered, because every `require_*` rule
    builds its sentence through `describe(key)`, which raises a bare `KeyError` for a key `FIELDS`
    does not hold (plan rulings P2, P75). The progressive mode is opt-in per kind, and the six
    ordinary kinds still lose their directory on any exception, pinned by a test written before the
    mode existed; the kind directory is the legacy `fdt/` on purpose, which makes the owner's stray
    pictures loose files INSIDE a kind directory, the only place the tidy-up can reach them.
  - **The spec was reviewed against the code before the owner read it** — four independent lenses
    (anchors, completeness, consistency, standing rules) and a judge that re-checked every blocking
    and important finding itself. Verdict `needs-rework`: 12 must-fix, 7 rejected with reasons, ~28
    corrections. All are folded in. Three would have stopped an implementer cold: the `copy_for_run`
    duck-typing above; a contradiction over who opens the record; and the destination flag controlling
    THREE behaviours rather than two, the third of which would have called `rmdir` on the real
    `Artifacts/` root after a failed run. The fourteen reports are in the gitignored ledger
    `.superpowers/sdd/2026-09-22-secondary-analyses/` (`recon/`, `spec-review/`, `progress.md` with
    rulings R1–R10, each carrying what it costs if it is wrong).
  - The plan (`e902256`+`5354e6b`) carries 82 planning rulings, P1–P82; the spec's §12 rows 1–10 are
    the planning-time departures recorded at planning time, and rows 11, 12 and 21 are three more
    planning-time departures recorded only at the end of the piece (§12's preamble). D11 and D12
    stay standing refusals; this piece re-opened neither.
  - The execution rulings are the rest of the spec's §12 (rows 13–20 and 22–36: pre-flight F-, task
    T- and the review's M/N/R-F rulings) and the gitignored ledger
    `.superpowers/sdd/2026-09-22-secondary-analyses/progress.md`.
  - Execution rulings a later piece would otherwise re-litigate, each with what it costs if wrong:
    - **The brief extractor drops a plan's preamble**, so every ruling is folded INTO the task body
      it binds (pre-flight, `cbc2187`), with a stated precedence: the latest controller block beats
      the amendments, which beat the task text. *Cost: a task body carries up to three layers an
      implementer must reconcile in order.*
    - **Every task is gated, and from Task 4 on the gate starts at the same moment as the task's
      review** (the owner's decision of 2026-09-23; gate t1 had run solo, after its review). A
      gate's wall time on this shared machine is NOT a regression signal on its own — a count or
      warning change is, or a slowdown concentrated in the tests the task touched; the final fast
      gate of record runs on a quiet machine. *Cost: a global slowdown could hide in the noise until
      the final gate, which is why that gate is taken quiet.*
    - **A sweep's `results` is one entry per finished operating point**, each at its own resonance,
      with the top-level `ratio_at_resonance` null and `offgrid.of` counting only measured probes
      (the review's M2, re-litigating P78, whose single-run summary did not carry over to a
      concatenation). *Cost: a body shape changed before any owner record of the old one existed.*
    - **A T_a/T grid reaching below 0 is refused before anything is spent; 0 is allowed; S has no
      floor** (R-F1), and a diverged operating point is a FAILED point, never a done one (M1).
      *Cost: an owner who wants T_a/T = 0 excluded, or an S floor, adds one rule.*
    - **The window names its comparisons** (R-F3): a Comparison name and note box in both "Compare
      saved …" groups, never remembered, answering the same refusals as the run's Record name.
      *Cost: two rows per screen.*
    - **The store neither reserves names nor re-mints ids**: `_new_id` skips the ids this store
      object has minted (two writers created back to back got the same id), but a name is claimed
      only on disk, so a writer created and never entered does not hold its name. *Cost: a
      programming error creating two same-named writers in one process commits two records with one
      name.*
    - **`PreflightWarning` lives in the torch-free `core/refusals.py`** (re-exported by
      `core.orchestrator`), and `seeded` in the light `core/rng.py` (re-exported by
      `core.diagnostics.rng`), so an FDT run loads neither the orchestrator (and with it sbi's
      inference modules and pytensor) nor the diagnostics — only the bare `sbi` package, whose
      version every manifest's provenance records. *Cost: two names moved module; every old import
      keeps working.*
    - **The GPU smoke gate was judged not required** against the whole piece: every changed line
      that creates or moves a tensor is either off `smoke`'s path (CPU-pinned FDT, sweep and Live
      simulation code) or, in `core/rng.py`'s `seeded`, identical to the base on CUDA (the gate row
      names each site). *Cost: a CUDA-only regression on that path would go unseen until the next
      card run.*
    - **The fix re-review's nine minors were parked**, not sent round a second fix loop: the owner's
      end-of-piece process is one fix dispatch and one scoped re-review, and none is load-bearing
      ("Owed" item 7 lists them). *Cost: a regression of the negative-grid or zero-knob refusal
      would hang a gate instead of failing it — visible, but slow.*
    - **The note rule is run by each front end**, never by the store (a store-level ruling at Task
      25 was superseded, keeping piece 4's store contract): both panels run `require_note` at the
      click, for a run's Note and a comparison's alike, and the tool's `main` judges any `--note`
      once, before any subcommand's handler. *Cost: an over-long or multi-line `--note` that used to
      be stored is refused before any spend; several call sites instead of one.*
    - **`artifacts sweep ""` is refused as an unknown kind** (exit 1), while `artifacts list ""`
      lists every kind (T30's D2). *Cost: the two readings of an empty kind disagree, on purpose —
      the delete command takes the safer one.*
    - **The CrossVal panel fills each grid's ends in ascending order** (min and max of the limit and
      the cell's value), so a cell with T_a/T < 1 is not refused on the panel's own defaults. *Cost:
      a descending sweep is computed ascending; a cell exactly at the limit is refused as an empty
      sweep.*
    - **A sweep record's `settings` names the held parameter** (`held`: `{'temp': 1.0}` on the S
      sweep, `{'s': 0.0}` on the T_a/T sweep; the review's N7). *Cost: none; the value was only in
      `data.h5`.*


- **2026-09-25/28** — **piece 6 (the reference cleanup, retrain readiness, the probe checks and the
  documents): decisions H1–H13**, brainstormed with the owner in plain-language questions (spec
  `docs/superpowers/specs/2026-09-25-documentation-and-retrain-readiness-design.md` §1.1, `907098d`).
  H1 the handoff split, a README tool reference and a new written retrain runbook — no scripted
  runbook. H2 the handoff moves to the gitignored `archive/`; live references are cleaned, closed specs
  and plans stay untouched. H3 topic pages with a reading path per reader, written fresh from the code.
  **H4 nothing shippable cites the working documents or their labels** — core/, tests/, conftest.py,
  README, requirements, pytest.ini, the launchers and the reader pages; reasons in words; a source-scan
  suite enforces it (the owner: "I don't want the code to have any references to handoffs used for LLMs
  in the end when the software is shippable"). H5 an ~80-line README summary plus a full reference page
  checked by a parser-walking test, after a help-text tidy. H6 the retrain's science: the tier-1 box
  (temperature an assumed input), 10,000 × 2,048, a 256 × 10 network from the start, 6 probes into 12
  slots, a certification narrowing round on a simulated cell, hard calibration gates with
  informativeness recorded as the first baseline. H7 `python -m core probes band|mask|drive`, command
  line only, measuring only. **H8 measuring is not overriding: D11's letter is narrowed to what trains
  and infers** — the probe checks may take their own frequency and drive grids (restated at the piece's
  close). H9 every readiness gap fixed in code. H10 one piece, code first, documents last. H11 the
  calibration verdict: every inferred parameter's KS p ≥ 0.05 ÷ their number — temperature included
  (revised 2026-09-28 after the review showed temperature scales |χ| in chi mode) — and the joint
  coverage p ≥ 0.05; a FAIL is a result, exit 0. H12 walkthrough rows F1–F5. H13 (2026-09-28)
  calibrations and narrowing rounds record the acceptances they ran under.
  - The spec was reviewed against the code before the owner read it (four lenses and a judge; report
    `.superpowers/sdd/2026-09-25-documentation-and-retrain-readiness/spec-review.md`): "ready with
    fixes", 19 must-fix and 33 corrections, all folded into `907098d`. The ones a later session would
    otherwise re-litigate: `validate --seed` draws from a stream derived from the seed and a
    calibration tag, so it never replays the training strata `core/rng.py` exists to protect, and
    `seed=None` leaves the streams alone (smoke unchanged) — this widens piece 2's "only smoke and the
    diagnostics take --seed"; the mask audit reads training through an optional observer in
    `gen_chi_block`, committed at the batch loop's row-storing seam (row halving splits one batch into
    several row ranges), never by monkeypatching; the production geometry code is not merged with the
    probes' helper (the two copies differ in float32 rounding); the eigenvalue carry-over on a resume
    keeps the fresh value as the Fisher-ran witness (V7 unchanged); the silent Hopf fallback raises
    `RuntimeError` (a programming error), not a Refusal.
- **2026-09-28/30** — **piece 6's execution** (subagent-driven; the ledger, pre-flight scan and reviews are gitignored
  under `.superpowers/sdd/2026-09-25-documentation-and-retrain-readiness/`).
  - **D11, read narrowly (H8), restated at the piece's close.** D11 stays a standing refusal for everything that trains or infers: no flag, box or argument gives a training run, a calibration, an inference or a narrowing round a non-default chi band or drive amplitude; a change is an edit to `core/config.py`. The `probes` family alone takes its own frequency, drive and length grids, because they are measurements: they reach the simulator only as the drive builder's frequency and amplitude inside `core/diagnostics/probes.py`, never through a `SimConfig` field or `gen_training_data`'s `chi_f0`/`chi_freq_bounds`, and every record states the configured band and drive it judged; the measure-only tests in `tests/test_diagnostics.py` and `tests/test_tool.py` pin this. A later piece must not read this as licence for a chi override anywhere else. D12 is unchanged.
  - **The seed reading widened again.** Piece 2 recorded that only `smoke` and the diagnostics take `--seed`; piece 5 widened it for `fdt` and `crossval`. Piece 6 adds `validate --seed`, and the Validate tab draws one per run (no Seed box, nothing remembered): the calibration set is drawn inside `core.rng.seeded` from `calibration_seed(seed)`, a `numpy.random.SeedSequence([seed, CALIBRATION_TAG])` stream, so a calibration never starts where a training run with the same seed started — the replay `core/rng.py` exists to prevent, whose docstring names this one exception — and `seed=None` leaves the streams alone, so `smoke`'s calibration still follows `smoke`'s seed. The `probes` modes follow the diagnostics' convention (`--seed`, default 0, recorded).
  - Execution rulings a later piece would otherwise re-litigate (each with its cost if wrong):
    - **Two readiness fixes outside the plan**, each test-first, reviewed and gated: loading refuses a chi
      posterior trained at another drive (§11 row 24; cost: none — it fires only after a `CHI_F0` edit); a
      training run sets aside shards a killed save left uncommitted, never after a fallback read of the
      previous state file (row 25; cost: a few set-aside files on disk).
    - **The runbook's stop point** is twice the EXPECTED cost at 256 × 10 (about 740 hours), not twice the
      92-hour 128 × 8 reference; the owner has not confirmed it (row 26; the open list asks).
    - **Part 5 changed code texts, not behaviour**: every false or stale help string, tooltip, message,
      comment and docstring the page reviews found was corrected (rows 23, 27), with the comment-only checker
      on each commit.
    - **The load-check audit is NOT done here**: the documents review found recorded keys no loader compares
      (the feature-set version, the summary flags, an observation's box, drive, band and ceiling, the time
      grid, the units file) and two constants in no compared record and not in the cache identity (the cycle
      floor, the smallest training probe count). The pages state the gaps; the open list holds the audit for
      the owner (cost: a post-retrain change to one of them loads an old record silently until it lands).
    - **Pre-flight rulings** S-01..S-23 (the scan of all 39 tasks before Task 1) were folded into the tasks
      they bind; owner questions 2, 3 and 5 took their recommended answers (a fresh tier-1 prior; stratified SBC
      as a characterisation; the stash note in CLAUDE.md), question 4 its default, question 1 is row 26.
  - **The whole-code review** (Task 25): 0 Critical, 2 Important (the smoke run labels "run 1"/"the drill" that
    reached a usage error; the "Phase 2/3" banners), 23 Minor + 3 from the verifiers; one fix dispatch of 25
    commits (`8fef76f`..`04fbe54`) and a scoped re-review; the rest parked with reasons (the open list).
  - **The card run's judgement**: the card run at `16e5bae` stands for the piece (the gate table); run 1's masked
    count is not reproducible across processes on CUDA (§11 row 30).
  - Spec §11 holds 30 rows (1–14 planning, 15–30 execution).
