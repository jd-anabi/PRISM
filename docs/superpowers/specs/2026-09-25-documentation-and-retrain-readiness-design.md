# PRISM piece 6: the reference cleanup, retrain readiness, the probe checks and the documents (design)

**Status:** draft 2026-09-25, for the owner's review. Brainstormed with the owner the same day: every
decision in §1.1 is the owner's answer to a plain-language question, and §2–§10 were each presented
in chat and approved section by section before this file was written.
**Piece:** 6 of the seven-piece pre-retrain hardening programme
(`docs/superpowers/specs/2026-09-10-pre-retrain-hardening-decomposition.md` §4, row 6), in the scope
the owner re-cut on 2026-09-25 (§1.2 says how it departs from that row).
**Depends on:** pieces 1–5 (the store, the one-flow tool, refusals and logging, the Artifacts screen,
the FDT records). It runs after all five.
**Followed by:** the retrain (`docs/STATE.md` "Owed" item 8), run from the runbook this piece writes.
**Evidence.** The read-only reconnaissance of 2026-09-25 is in the gitignored workspace
`.superpowers/sdd/2026-09-25-documentation-and-retrain-readiness/recon/`, cited below as `recon/…`:
`handoff-map.md` (every handoff section with its status, audience, whether it is load-bearing, its
false claims with evidence, and a suggested home), `inbound-references.md`, `cli-inventory.md`, and
eight design reports, `design-{scanrule,helptidy,f0sweep,maskaudit,drive,diagframe,tier1,runbook}.md`,
plus `design-scanrule-hits.tsv` (every reference line the scan rule of §2.2 matches today, 1,976 rows).

**Anchor rule.** Every `file:line` below was read on 2026-09-25 at `8556546`, but **the quoted text or
the named symbol is the anchor, not the number** — line numbers go stale between the plan and the
task. An implementer who finds the quoted text elsewhere in the named file has found the right place.

---

## 1. Purpose and scope

Before the first prior of the retrain is built, four things must be true.

1. **The shippable code points at no AI working document.** The program, the tests, the launchers,
   the README and the requirements file cite neither the handoff nor the working record (STATE,
   CLAUDE.md, the specs and plans) nor any of their labels. Each comment says its reason in words.
2. **The retrain's own path is correct, tested, and reports what its gates need.** The retrain trains
   on a box that has never been trained, with a larger network, and must end on a calibration verdict.
3. **A lab can check the chi probe settings on its own preparation.** The band, the drive strength and
   the masking rule were measured once, on one simulated cell, by scripts that can no longer run.
4. **The 7,000-line handoff is replaced by reader documents and archived.**

Measured on 2026-09-25:

- **The handoff** (6,979 lines) is, by line count of each section's suggested home
  (`recon/handoff-map.md`): dated session notes ~40 %, science reasoning ~18 %, subsystem internals
  ~13 %, traps ~7 %, contracts ~5 %, runbook material ~3 %, fully superseded ~5 %, the rest
  small. It is false in far more places than the five lines STATE names (`:49,53-54,76,190-191`): the
  three front ends (`:7-9`), the scripts row and tree (`:51`, `:185-189`), the pre-pytest test setup
  (`:74-140`), the generated `Resources/` subtrees (`:190-191`), and every environment-variable recipe
  that runs a deleted script (e.g. `:584-585`, `:3612`). Its memory table contradicts its own later
  entry: `:2615` (the Welch spectrum's float64 promotion) is fixed as `:6756` says, but half of
  `:2616` is still open — `core/FDT/sanity.py:291` and `:328` keep a view of the whole solution
  (`x_steady = sol[0, 0, :, burn_idx:]`, no `del sol`) — and six neighbouring rows carry no status.
  The per-direction table of the August retrain (`:1404-1416`) is the transpose (its own banner at
  `:1355-1380`); the withdrawn "k is unmeasured" conclusion survives in `:3333-3335`, `:3450`,
  `:3621-3625` and in `core/SBI/truncate.py:21-24`.
- **References to the working documents** (`recon/design-scanrule.md` §1): 872 lines in 86 of the
  178 tracked `core/**/*.py` files, 1,101 lines in 20 of the 21 `tests/*.py` files, and three other
  lines (`conftest.py:1`, `requirements.txt:9`, `:44`). About 80 % are provenance tags that can
  simply be deleted; 400–500 lines explain a reason by pointing at a document and must be rewritten.
  Ten messages users see carry labels (§2.4), three tests pin them, and two test names carry them.
- **The retrain's path** (`recon/design-tier1.md`, `design-runbook.md` §9): the tier-1 box
  (`Resources/Bounds/nadrowski/master_tier1.txt`, temperature `T` in 280–310 K in place of `f_scale`,
  which is derived) has zero test coverage beyond three pure-function tests
  (`tests/test_conditioning_repair.py:251-278`); `identifiability jacobian` and `laplace` silently use
  the wrong force scale on it; temperature is presented as an inferred result everywhere; calibration
  prints numbers but no verdict and cannot be repeated on the same set; a resumed training run loses
  the rotation's eigenvalues; the narrowing round is silent when the truth lies inside its region; no
  record names the tier-1 constraint.
- **The tool's help** (`recon/design-helptidy.md`): 835 lines over 27 screens, 15 flags with no
  description, about 50 rendered placeholders that show internal names, shared flags with up to five
  wordings, defaults stated five different ways.
- **The chi settings**: `archive/scripts/chi_f0_sweep.py`, `chi_mask_audit.py` and
  `build_master_cells.py` measured them; none can run today (`import _common` — deleted at `7433ced`
  — and `from core.config import PLOT_PATH` — removed) and each has drifted from the production path
  (`recon/design-f0sweep.md` §4, `design-maskaudit.md` §2, `design-drive.md` §2).

### 1.1 Decisions (binding)

Each is the owner's answer to a plain-language question on 2026-09-25.

| # | decision |
|---|---|
| H1 | **Scope.** The handoff split into reader documents, a README tool reference, and a new written retrain runbook. No scripted runbook. Plus H4, H7 and H9, which the owner added during the questions. |
| H2 | **The handoff moves to the gitignored `archive/`** (move on disk, then `git rm --cached`, never `git mv`). Every LIVE reference is cleaned; the closed specs and plans stay untouched, their handoff citations resolving through git history; STATE records the archiving commit. |
| H3 | **Topic pages with a reading path per reader**, each fact written once: one ordered path each for the owner-scientist, a lab member bringing recordings, a successor, a reviewer. Written fresh from today's code, with the handoff as a source; every checkable fact checked. |
| H4 | **No reference to the working documents anywhere shippable** — nothing under `core/` or `tests/`, nor `conftest.py`, `README.md`, `requirements.txt`, `pytest.ini`, `run.bat`, `run.sh`, nor the new reader pages — cites the handoff, STATE, CLAUDE.md, the specs, the plans, or their labels. A provenance tag is deleted; a reason given by reference is rewritten in words. A source-scan test enforces it. The working record itself (STATE, CLAUDE.md, the specs, plans, ledgers and the walkthrough checklist) may still cite code and each other. |
| H5 | **The tool reference** is an ~80-line README summary plus a full flag-by-flag page in the reader documents, kept in sync by a test that walks the tool's parser; the help text is tidied first. |
| H6 | **The retrain's science**, settled now: the tier-1 box, with temperature reported as an assumed input; 10,000 batches × 2,048 rows; a network of 256 hidden features × 10 transforms from the start; chi mode with 6 probes supplied and the slot ceiling kept at 12; a certification narrowing round on a simulated cell after the broad model passes; hard calibration gates, with informativeness measured and recorded as the first baseline. |
| H7 | **The probe checks**: a new command-line diagnostic family, `python -m core probes`, with three modes — `band` (the band and drive hold for a cell), `mask` (why training probes are thrown out, over a prior), `drive` (the free-running and captured drive strengths for a cell). Command line only; each writes an ordinary diagnostic record; it measures and never changes a setting. Kept after the owner heard that the retrain does not strictly need it. |
| H8 | **Measuring is not overriding.** The standing rule that there is no chi override anywhere (piece 2's D11) is read as governing what trains and infers; the probe checks may take their own frequency and drive grids. Every record states the configured band and drive it judged, and a test pins that nothing in the family can reach a training run. |
| H9 | **Every readiness gap is fixed in code**, with tests, before the retrain (§5), then a card run of the tier-1 box. |
| H10 | **One piece, code first**: parts 1–4 (cleanup, help tidy, readiness, probe checks), then part 5 (the documents), written once the code will not move again. |
| H11 | **The calibration verdict** passes when every inferred, non-assumed parameter's rank-uniformity test has p ≥ 0.05 ÷ (their number) — a 5 % family-wise false-alarm rate — and the joint coverage test, over the same parameters, has p ≥ 0.05. A failing verdict is a result, not an error. |
| H12 | **Walkthrough rows F1–F5** cover only what changes in the window (§9). |

### 1.2 How the design reads the decisions

- **Against the decomposition's row 6.** "A user guide per audience" is realised as topic pages with a
  reading path per reader (H3), because the readers' needs overlap heavily and a fact written four
  times drifts. "Per-subsystem design docs" are `architecture.md`, `rules-and-traps.md` and
  `science.md` (§6.2). "The runbook as … a scripted pipeline" is dropped (H1): the retrain is a
  multi-day run the owner watches and gates by hand. The reference cleanup, the readiness fixes and
  the probe checks are new scope (H4, H7, H9).
- **What the reference rule reads.** Comments, docstrings and string literals in Python, every line of
  the named prose files and reader pages, and test function and class names. Never code identifiers,
  where short math names (`t0`, `p1`, `d0`, `e2_h2`) would be false positives
  (`recon/design-scanrule.md` §3).
- **The walkthrough checklist is working record**, not a reader document: its rows are lettered by
  piece. So no test or code cites its path or its row ids.
- **`.gitignore` is left out of the scan**: it is configuration and has to keep naming
  `/.superpowers/`.
- **Dates.** A date given as evidence ("measured 2026-08-28 on this card") stays. An incident used as a
  label in text a user sees is reworded (§2.6). The scan does not enforce dates.
- **Wrong facts** are fixed where the reconnaissance found them or where a rewrite meets them (§2.5).
  This is not a full audit of every comment.
- **STATE's owed piece-6 bullet** is met without editing the archived file: its five false lines, the
  memory-table contradiction and the transposed table are never carried into a reader page (the
  documents review checks it, §6.7), and the still-open half of `:2616` joins STATE's open list.
- **Piece 4's hand-on** ("the rest of group M" of the handoff's traps: the shim header, the gating
  table missing the TSNPE tab, six tabs not five, `value_or_none` over `FloatField.value()`;
  `docs/superpowers/specs/2026-09-17-gui-usability-and-artifact-browser-design.md:113-116,530`) is
  met by `rules-and-traps.md`, which carries those traps rewritten against today's code.
- **The informativeness total stays joint.** Section 5 as approved in chat said the informativeness
  total would leave temperature out. It cannot cheaply: the total is the mean of the flow's log-density
  minus the prior's at the calibration points (`core/SBI/analysis.py:315-327`), a joint density that
  cannot drop one parameter without marginalising the flow. So the joint coverage test (sample-based)
  leaves assumed parameters out, and the informativeness total stays joint, with each assumed
  parameter's own entropy entry marked and reported apart (§5.3).
- **The mask audit reads training through an observer**, never by patching the program at run time
  (§4.4). The archived script monkeypatched two functions; an in-process tool run must not.
- **Seeds.** The probe checks follow the diagnostics' convention: `--seed` defaults to 0 and is
  recorded. `validate` gains the FDT runs' convention: a given seed makes the calibration set
  repeatable, and when none is given one is drawn and recorded (§5.4).
- **The two engineering items the map found open** — the FDT sanity checks holding a view of the whole
  solution, and `decorrelate`'s per-call re-derivation (`core/SBI/decorrelate.py:149-163`) — go to
  STATE's open list, as do the handoff's other open engineering items (§6.6). Open **science** ideas
  go to `science.md`'s open-questions section.

### 1.3 Out of scope, and who owns it

- **The reduction map's science** (decomposition §2). Its files still fall under the reference rule.
- **A window front end for the probe checks** (H7).
- **The science items STATE lists as the owner's** (the NaN column in `identifiability jacobian`, the
  fixed probe-layout seed across repeats, `--chi-k 1`, the passive-baseline Ω₀, the Hopf band, and the
  rest of STATE's "Open" list). They stay in STATE; the ones that are science are also described in
  `science.md`'s open questions.
- **Making the diagnostic kind progressive.** A crashed probe check loses its record, as every
  diagnostic does today (`core/artifacts/store.py:73`); the default runs are minutes long.
- **`training_checkpoint.checkpoints_using_prior`**, which may be dead code
  (`recon/handoff-map.md`, chunk E notes, item 4): recorded in STATE's open list, not removed.
- **Pushing** — the owner pushes.

---

## 2. Part 1 — the reference cleanup

### 2.1 The rule

**Files walked:** every Python file under `CODE_ROOTS` and `CODE_FILES` (`tests/_fixtures.py:38`,
`:42`; today `core` — which includes `core/Reduction` and its tests — and `conftest.py`), every Python
file under `tests/`, the prose files `README.md`, `requirements.txt`, `pytest.ini`, `run.bat` and
`run.sh`, and, from part 5 on, every `*.md` under `docs/guide/`.

**Excluded:** the scanning test's own module (by `Path(__file__)`), `.gitignore`, the licence files
under `core/gui/assets/`, and the working record (`CLAUDE.md`, `docs/STATE.md`, `docs/superpowers/`,
`docs/checklists/`).

**What is read:** in Python files, only `COMMENT`, `STRING` and `FSTRING_MIDDLE` tokens (the
complement of the skip set in `tests/test_conditioning_repair.py:920-922`), after stripping
`noqa:…` and `type: ignore[…]`; test function and class names in a separate pass; every line of the
prose files and reader pages.

### 2.2 What the scan matches, and what it allows

The patterns (`recon/design-scanrule.md` §5; the plan re-verifies each against
`recon/design-scanrule-hits.tsv`):

```python
PROSE = {
 "doc":        r"(?i)PRISM_HANDOFF|\bhandoff\b|\bSTATE\.md\b|\bCLAUDE\.md\b|superpowers|\bsdd\b|\bprogress\.md\b|\bledger\b",
 "piece":      r"(?i)\bpieces?[ -]?\d+\b|\bwhole-piece\b|\b(?:this|that|each|same|earlier|later|next|previous) piece\b",
 "section":    r"§|\b[Ss]ec(?:tion|\.)? ?\d+(?:\.\d+)+\b",
 "spec":       r"\bspec(?:'s)? (?:§|[Ss]ec|section|\d)|\b(?:design|piece-\d+) spec\b|\bthe spec's\b|\bspec NOTES\b|\bplan(?:ning)? ruling",
 "ruling":     r"\brulings?\b|\bowner ruled\b",
 "review":     r"\breview's\b|\bfix round\b|\bReview Focus\b|\bfinding \d|\breviewers?(?:'s)? probe|\breviewers? probed|"
               r"\bprobed by (?:\w+ )?reviewers\b|\bthe review of\b|\bper the review\b|\b[Cc]hecklist(?: item)? \d|"
               r"\bBacklog\b|\b(?:CRITICAL|IMPORTANT|Critical|Important) \d\b|\bdefect [A-Z]-?\d+\b",
 "task":       r"\bTasks? \d+\b|\btasks? T\d+",
 "trap":       r"\b(?:trap|TRAP)s? [A-Z]+\d+[a-z]?\b|\bCHI\d+\b",
 "guardrail":  r"(?i)\bguardrails?[ -]\d+\b",
 "series":     r"(?<![\w-])[CS]-\d+\b",
 "appendix":   r"\bAppendix [A-Z]\b",
 "walkthrough":r"\bwalkthrough\b|\brows? [A-Z]\d+\b",
 "programme":  r"\bhardening programme\b|\bone-flow design\b|\bclean[- ]break\b",
 "id":         r"(?<![\w./])(?<![a-z]-)(?:R-F\d+|FE\d+|[A-Z]\d{1,3}[a-z]?)(?!\w)(?!-[a-z])",
}
TEST_NAME = r"(?i)(?:^|_)(?:piece\d+|[a-z]\d{1,3}[a-z]?|rf\d+|fe\d+|chi\d+|task\d+|guardrail\d*|round\d+|whole_piece|fix_round|walkthrough|handoff|appendix|ruling)(?:_|$)"
```

**The allowlist**, each entry as narrow as it can be:

- id tokens allowed everywhere: `F0`, `D0`, `L1`, `L2` (physics and norm names);
- per file: the summary-feature ids (`A1`…`G7`, `core/SBI/statistics.py:41-59`) in
  `core/SBI/statistics.py` and `core/SBI/summaries.py` only; scipy's `S2` in `core/FDT/spectral.py`;
- shapes: `\bPhase [A-Z]\d\b` (the reduction map's phases); `README.md`'s Apple-chip line
  ("M1 / M2 / M3 / M4"); a published paper's own section, `(?:§|Section|Sec\.) ?\d+(?:\.\d+)* of [A-Z]\w+`;
- test names, by exact name: `test_the_B7_pair_shares_one_flag` (B7 is a feature) and the two
  `test_v1_…` / `test_v2_…` model-file-schema tests in `tests/test_user_models.py` (`:391`, `:404`).

**Any allowlist entry that matches nothing fails the test**, so the list cannot go stale. A future
science term shaped like an id (B0, H1) needs an entry.

**Known misses** (`recon/design-scanrule.md` §5, under 0.5 % of lines — unlabelled prose such as "the
piece removes", "the spec names", "list 4, item 11") are found by one manual pass per cleanup task,
reading every hit file in full rather than only the matched lines.

### 2.3 How each hit is treated, and the replacement vocabulary

- **A provenance tag is deleted** — "(piece 2)", "(the whole-piece review's N22)", "(spec §7.3)",
  "(controller ruling F29)", "(D4)". About 80 % of the lines.
- **A reason given by reference is rewritten to state the reason.** Where a label is the subject of a
  sentence ("GUARDRAIL 2 at …", "E2 keeps the folder", "since D11"), the sentence names the rule in
  words. One vocabulary is used across the code and the new `rules-and-traps.md`, so a comment and the
  page it would once have pointed at say the same thing:

| label today | the words used instead |
|---|---|
| TSNPE "the one thing" (propose from the truncated prior) | the truncated-prior rule: a narrowing round draws from the prior restricted to the region, never from the posterior |
| guardrail 1 | the observation-digest rule: a round refuses unless the stored observation matches |
| guardrail 2 | the narrowed-model rule: a narrowed posterior is marked as such and never loads as a broad one |
| guardrail 3 | the eigenbasis rule: the region is cut in the rotation's leading directions, flat ones left full width |
| guardrail 4 | the unweighted-draws rule: the region comes from unweighted posterior draws, not best fits |
| guardrail 5 | the generous-region rule: a 99.9 % region, with the truth-outside rate watched |
| guardrail 6 | the cost-on-screen rule |
| guardrail 7 | the region-carries-its-basis rule: the child reuses the parent's rotation, never recomputes it, skips directions loaded on t_scale, and names its base prior |
| guardrail 8 | the calibrate-on-the-region rule |
| C-11 | the training checkpoint (the simulation cache) |
| C-1 … C-10 (chi series) | the specific measure named in words (e.g. "the per-row lock-in duration", "the per-row probe placement") |
| trap X5 | the calibration's operating-point count is t_scale's effective sample size |
| trap X10 | the rotation is not reproducible across processes, so a resume reuses the stored one |
| trap CHI10 | a near-constant channel inflates a standardised Jacobian |
| handoff defect D1 | a region drawn in one rotation and enforced in another |
| handoff defect D3 | the region belongs to the simulation cache's identity |
| handoff defect D4 | a direction loaded on t_scale would turn the restriction into a reweighting |
| handoff defect D6 | a rotation saved transposed |
| trap M1b | replacing the session's configuration versus installing into it |
| "Appendix A <date>" | the incident told in one clause, or nothing |
| piece decisions (D7, V5, B12, E2 …) | the behaviour they decided, in words ("a cache one setting away is refused before any simulation") |

- **Three short feature ids outside the statistics files** become full feature names:
  `core/SBI/chi.py:335` (A3), `core/SBI/embedded_network.py:65` (A1, D3),
  `tests/test_conditioning_repair.py:241` (D3).
- **Two test names** are renamed: `tests/test_nav_and_gating.py:1280`
  `test_the_d7_and_d8_dialogs_default_to_cancel`, `tests/test_user_sbi.py:3184`
  `test_the_cuda_graph_preserves_the_rng_contract_c11_depends_on`.

### 2.4 Messages users see, and the tests that pin them

Each loses its label and keeps a plain sentence (`recon/design-scanrule.md` §4):

| # | site (symbol) | kind | label today |
|---|---|---|---|
| 1 | `core/orchestrator.py:1119` | Refusal | "guardrail 2" |
| 2 | `core/orchestrator.py:1453` | Refusal | "(D4)" |
| 3 | `core/orchestrator.py:1482` | Refusal | "(D6)" |
| 4 | `core/SBI/truncate.py:195` | Refusal | "(defect D1, Appendix A 2026-09-09)" |
| 5 | `core/SBI/truncate.py:206` | Refusal | "(defect D1)" |
| 6 | `core/SBI/truncate.py:371` | warning record | "(D4)" |
| 7 | `core/artifacts/manifest.py:166` | ManifestError | "(defect D3)" |
| 8 | `core/SBI/run_guards.py:158` | Refusal | "(Appendix A, 2026-08-19)" |
| 9 | `core/diagnostics/identifiability.py:597` | ValueError | "(trap CHI10)" |
| 10 | `core/tool/config_args.py:248` | UsageError | "(D9)" |

**The three tests that pin labels change in the same commit as their message:**
`tests/test_artifact_store.py:71` (`match="D3"`), `tests/test_conditioning_repair.py:680`
(`"D6" in str(e)`), and `tests/test_tool.py:376` (`"(D9)" in str(exc.value)`, whose failure message
"spec 3.6 requires the guardrail id in the message" is reversed: the test now pins the plain sentence
and that no label is present). Each new assertion pins a distinctive phrase of the plain sentence.

**Test assertion messages** that carry labels (71 string lines in tests, e.g. `tests/test_tool.py:3201`
"B6: no force in either front end", `tests/conftest.py:51` "since piece 3", which a user sees as a
session error) are rewritten like any other hit.

### 2.5 Wrong facts fixed on the way

Found by the reconnaissance (`recon/handoff-map.md` chunk notes; `recon/design-*.md`); each is
rewritten to what is true, in the task that owns the file:

- `core/config.py:455-466` still states the overturned conclusion ("… SYSTEMATIC, not statistical … Neither a stronger drive nor a longer recording can recover those probes"); it is rewritten to the corrected account (the in-band failures were a lock-in cycle limit, recovered by the cycle ceiling; the high edge 0.3 stands), which `science.md` carries in full.
- ".rot.pt sidecar" / "sidecar" at `core/config.py:407`, `:493-494`, `:497`, `:538`, `core/sim_config.py:77`, `core/Helpers/model_store.py:38`, `core/Helpers/file_manager.py:312`: it is the manifest now (`core/artifacts/store.py:1300-1304`).
- `core/config.py:499` "conditioning width 42 + 72 = 114": it is 50 + 72 = 122 (`SUMMARY_WIDTH = 41 + 8`, `core/SBI/statistics.py:539`).
- `core/config.py:410`, `:419` cite `posterior_07012026` and `prior_forcing_no_forcing.pt` as present; `core/SBI/embedded_network.py:84`, `core/gui/panels/inference/tsnpe_tab.py:32-33` and `core/SBI/truncate.py:22` name the deleted `posterior_08232026` as a live baseline.
- `core/SBI/truncate.py:21-24` "k in particular is FLAT … loading -1.00*k": withdrawn (the transposed reading).
- `core/SBI/statistics.py:507`/`:512`: a substitution-rate line that corrects itself five lines later.
- `core/SBI/train.py:38` "script-only": `chi_k_fixed` is the `sbc` diagnostic's.
- `core/gui/panels/inference/config_tab.py:136` "the only field on this tab" not persisted: since piece 3 the chi boxes are not persisted either.
- `core/config.py:48`, `core/SBI/pipeline.py:1849` "the scripts"; `core/tool/config_args.py:94` names the deleted `_common.script_cfg`.
- `core/SBI/chi_probes.py:3-7` names the mask-audit script as a patching consumer (the observer of §4.4 replaces it); `core/SBI/chi.py:271-274` says the duration ceiling is keyed on the fastest row (it is per row, `chi_probes.py:174-181`).
- `core/artifacts/identity.py:2-3` says the clean break retires `orchestrator.training_identity`, which survives as a delegate (`core/orchestrator.py:368`).
- `core/gui/panels/inference/rows.py:34` cites `orchestrator.build_experiment_obs_chi`, now in `core/SBI/observations.py`.
- `tests/test_figures.py:15`, `test_nav_and_gating.py:15`, `test_settings_persistence.py:15`, `test_simulate.py:15` ("pytest tests/test_gui_progress.py"), `tests/test_user_sbi.py:1023`, `test_conditioning_repair.py:1041`: that file no longer exists.
- `tests/test_user_sbi.py:2233`: the docstring says the truncation key is "OMITTED, never None" while the test asserts None.
- `tests/test_user_sbi.py:2062`, `:2253` cite archived scripts at old commits.
- `core/gui/panels/inference/help_text.py:85` (a tooltip): "The 2026-08-25 run stopped at 130" — kept as a measurement, without the incident label (§2.6).

### 2.6 Dates

- A date that dates a measurement stays (`core/config.py:13` "MEASURED 2026-08-28";
  `help_text.py:48` "measured 2026-08-27: 21.67 GiB completed on a 15.92 GiB card").
- An incident used as a label in text a user sees is reworded to state the fact:
  `core/SBI/truncate.py:195`, `:205` ("the 2026-09-02 round kept 0.01%…"), `core/SBI/run_guards.py:158`,
  `help_text.py:85`.
- Three data literals in `tests/test_tool.py` (`:2913`, `:2930`, `:2940`) are test data, not labels.

### 2.7 Proving that only comments changed

For every Python file a cleanup task edits, the controller runs a check **outside the repository**
(in the session's scratch directory; never under `.superpowers/`): tokenize both versions, drop
`COMMENT` and `NL` tokens, parse, remove docstrings (the first-statement string of a module, class or
function), and compare `ast.dump(..., include_attributes=False)`. The only permitted differences are
the string constants listed in that task's message table (§2.4). A task whose check shows any other
difference is not done.

**Tests that read raw source text** (`inspect.getsource`, `code_only`, the source scans) must not see
a rewrite add or remove the words they look for (`recon/design-scanrule.md` §5 "Hazard"):
`tests/test_diagnostics.py:154` ("SystemExit", "raise ValueError"), `tests/test_nav_and_gating.py:4659-4683`
("the five inference tabs"), `tests/test_tool.py:2094` ("no bounds file"),
`tests/test_user_sbi.py:3241`, `:3551` ("SIM_VRAM_CEILING"), plus every other `getsource` scan, which
each task lists before editing. And no file is edited while a suite runs (CLAUDE.md).

### 2.8 The scan test, and its shrinking list

A new suite, `tests/test_source_hygiene.py`, holds the scan. It reuses `CODE_ROOTS`/`CODE_FILES` and
the directory-closure test (`tests/test_artifact_store.py:1866-1896`), the "scanned ≥ N" floor and
missing-file check (`:1845-1860`), the matcher self-check on snippets (`tests/test_refusals.py:856-872`),
and the "assemble the needle so the file cannot match itself" trick
(`tests/test_conditioning_repair.py:912-916`). A failure prints `path:line family 'match'`.

It lands in the **first** cleanup task with a list of every file not yet cleaned, which may only
shrink:

- every file NOT on the list must be clean;
- every file ON the list must still have at least one hit — so a cleaned file must leave the list, and
  the list is always the true remainder;
- the last cleanup task empties the list and deletes both the list and the second rule.

### 2.9 How the work splits

About nine tasks, each ending with its files off the list, the §2.7 check clean and a one-process fast
gate: (1) the scan test, the full list, the two test renames and the three feature ids; (2) the
inference core — `core/SBI/`, `core/orchestrator.py`, `core/Solvers`, `core/Simulator`, `core/Models`;
(3) the store and its neighbours — `core/artifacts/`, `core/runs.py`, `core/refusals.py`,
`core/rng.py`, `core/logging_root.py`, `core/cli.py`, `core/config.py`, `core/sim_config.py`,
`core/registry.py`, `core/forcing.py`, `core/progress.py`, `core/Helpers/`; (4) `core/FDT/`; (5)
`core/gui/`; (6) `core/tool/`, `core/diagnostics/`, `core/Reduction/`, `conftest.py` and the prose
files; (7)–(9) the tests in three batches, the last removing the list. Each message of §2.4 is changed
in the task that owns its file, with its pinning test.

---

## 3. Part 2 — the help-text tidy

### 3.1 No behaviour changes

Flag names, destinations, defaults, choices, `nargs` and required-ness stay exactly as they are. Only
the words a user reads change: help strings, metavars, descriptions and epilogs. The knob-forwarding
tests pin destinations and flag names, not text, and stay green.

### 3.2 The 15 flags with no description

(`recon/design-helptidy.md` §1.) `infer --posterior` (`core/tool/stages.py:95`); tsnpe's seven
training flags (`stages.py:130-136`); `--posterior` on `identifiability rotation`, `laplace` and
`ablation` (`core/tool/diagnostics.py:69`, `:80`, `:121`); `--cell` and `--t-obs` on `laplace` and
`jacobian` (`diagnostics.py:81-82`, `:94-95`). The seven tsnpe gaps exist because train and tsnpe
define the same training flags twice (`stages.py:53-72` and `:130-138`): one
`config_args.add_training_flags(p, *, fisher: bool)`, modelled on `add_resume_flags`, defines them
once, with tsnpe passing `fisher=False`. Every destination stays the same.

### 3.3 Placeholders

The 38 source sites (50 rendered) that show an internal destination name (`NUM_POSTERIOR_SAMPLES`,
`RUN_SIZE_CAP`, `MAX_NUM_EPOCHS`, `T_OBS_S`, `STORE_ROOT`, `ENSEMBLE_M`, `CHI_K_FIXED`, …;
`recon/design-helptidy.md` §2) switch to the house set the stage commands already use — `N` a count,
`X` a real number, `S` seconds, `REF` a name or id, `PATH`, `NAME`, `K` the probe count,
`STAGE[,STAGE...]`. `--f0-si N` becomes `X` ("in newtons"); `--level Q`, `--prefactor VALUE`,
`--at VALUE` become `X`. New pin: no value action may have `metavar is None` unless it has `choices`.

### 3.4 One wording per shared flag

Where the meaning is the same, one canonical text (`recon/design-helptidy.md` §3 lists each
occurrence and marks it legitimate or accidental): `--cell` ("the cell file whose ground truth …",
with the role appended; crossval's "NWK" becomes "a Nadrowski cell file"); `--posterior` ("posterior
artifact <role>, by name or id"); `--seed` ("the random seed for the whole run (default 0)"; sbc adds
"; repeat r runs at seed + r"; fdt/crossval keep draw-and-record); `--n-cal`, `--num-runs`,
`--run-size`, `--max-epochs`, `--t-obs`, `--cal-n-scales` (one stratum wording: "(t_scale, T_obs)
operating points"), `--posterior-samples` ("per calibration dataset"), `--model` (both say
"upper-cased"), `--prior` (smoke's text no longer names the internal `prior_fingerprint`), the stage
`--note` (appends "one line, at most 200 characters", read from `core.refusals.NOTE_MAX_CHARS`). Where
the meaning or default genuinely differs by subcommand, the text says so explicitly.

### 3.5 One convention for defaults

Every optional value flag ends with `(default X)`, giving the real value, in the exact form of
`core.refusals._default_clause` (`core/refusals.py:214-217`), which becomes public so the help and
the refusal lines print the same string. The value comes from, in order: the parser's own default when
it is not None (`%(default)s`: smoke's literals, `--device`, `--preset`); else the torch-free
`core.refusals.FIELDS` default through the reverse of `core/tool/fields.py` `FLAG` (already pinned
against `config.py` and the stage signatures by `tests/test_refusals.py:462`); else a literal the new
test pins against the owning signature (sbc's seed, `n_cal`, `num_posterior_samples`; laplace/jacobian's
`seed`, `min_valid`, `sd_identified`, `zero_tol`, `noise_eps`; smoke `--t-obs` against
`config.T_MIN_EXP_S`; `--chi` against `config.CHI_MODE`). `--help` never imports `core.config`.

- The epoch cap's `2147483647` reads "no ceiling".
- `FIELDS["min_valid"]` gains its real default, 0.5 (`core/diagnostics/identifiability.py:359`), and
  joins the signature pin.
- `(default: config.CHI_MODE)`, `config.CHI_N_FREQS`, `config.cpu_device()`, `config.T_MIN_EXP_S`, the
  "(stage default N)" literals (15 sites in `diagnostics.py`) and smoke's "Default: …" all move to the
  one form (`recon/design-helptidy.md` §4 lists every site).

### 3.6 Epilogs

(`recon/design-helptidy.md` §5.) The environment variables are stated once, in the top-level epilog
(`core/tool/__init__.py:25-36`, pinned by `tests/test_tool.py:125`); smoke's copy (`smoke.py:48-53`)
is dropped and its "What to watch" paragraph shrinks to one line pointing at the reference page. The
compare epilog stops repeating `_COMPARE_MODES` (`fdt.py:267-270`), whose renormalise line gains
"drawn against the original". The artifacts epilog stops repeating the mode helps and keeps only the
facts pinned by `tests/test_tool.py:2833` ("all eight", "<kind> is one of …"), the no-configuration /
no-`--store-root` consequence, the `<kind>`/`<ref>` definitions and the exit-0 rule; its hand-typed
"200 characters" reads `NOTE_MAX_CHARS`. The fdt epilog drops the clauses that restate its flags; the
crossval epilog keeps its grid rules (pinned by `:1756`) and moves the preset-default sentence into
the two flags' help with the numbers. Every epilog is re-wrapped at 79 columns.

### 3.7 Descriptions, words and order

- Every subcommand and mode parser gets `description=` (today none has one), the same one-liner as
  its `help=`, so `<sub> --help` says what the command does.
- Code words in help become plain: "Campaign 2" → "the driven-response sweep", "NWK", "Adam LR" →
  "Adam learning rate", "HDBSCAN min_samples" explained, the `||g||_std` shorthand, "the GPU gate" →
  "run it on the card after changing code that moves tensors".
- sbc's "repeated K times" → "repeated N times (--repeats)" (K is the probe count elsewhere).
- `--name`/`--note` sit in the same place on every subcommand (after the configuration flags, as on
  the stage commands).

### 3.8 Tests

- **New** (in `tests/test_tool.py`): walk every optional value action of every leaf parser (the
  recursive `_walk` of `tests/test_refusals.py:906` is the model). Each has a non-empty help, a
  house-set metavar or `choices`, and exactly one `(default X)` clause whose value matches the parser
  default, the `FIELDS` default through `FLAG`, or the pinned signature literal. Every leaf has a
  `description`.
- **Updated**: the help-text pins whose wording changes (`recon/design-helptidy.md` §6 lists them:
  `tests/test_tool.py:125`, `:324`, `:1519`, `:1667`, `:1756`, `:1766`, `:1882`, `:2833`, `:3101`,
  `:3406`, `:3192`, `:3592`) keep pinning the same facts in the new words.
- The torch-free help probes (`tests/test_tool.py:1913-1920`, `:3125-3132`) stay as they are.

About two or three tasks, before part 3, so the new family is built to these conventions.

---

## 4. Part 3 — the probe checks (`python -m core probes`)

### 4.1 Shape

A new diagnostic family modelled on the three-mode `identifiability` (`recon/design-diagframe.md` §1):

- **Stage module** `core/diagnostics/probes.py` with three public entries (function names differ
  from the module name, `core/diagnostics/__init__.py:14`; e.g. `probe_band`, `probe_mask`,
  `probe_drive`), each `@public_entry`, re-exported from `core/diagnostics/__init__.py` and added to
  the exact public-entry set (`tests/test_artifact_store.py:3410`) and the config-untouched legs
  (`:3191-3250`).
- **Tool module** `core/tool/probes.py`, registered in the loop at `core/tool/__init__.py:46`, modes
  as `add_subparsers(dest="variant", required=True)`, with its own `interrupt_note` (a diagnostic
  keeps nothing on Ctrl-C; the generic `--resume require` advice would be wrong).
- **Record**: the existing `diagnostic` kind; body `{"diagnostic": "probes", "variant":
  "band"|"mask"|"drive", "settings", "results"}` (`core/artifacts/manifest.py:44`); the numbers in an
  `.npz` payload, figures through `w.fig_sink`, the report as `log.info` records so it lands in
  `log.txt`; every float through `orchestrator._num` (non-finite becomes null). The comment calling a
  diagnostic "a measurement ABOUT other artifacts" (`manifest.py:36`) is widened: `band` and `drive`
  measure a cell and have no parent, as `identifiability jacobian` already does.
- **Common flags**: `--bounds` (required), `--device`, `--seed` (default 0, recorded, run inside
  `core.rng.seeded(seed, device)`), `--name`, `--note`. Every knob is a keyword argument of its stage
  with its default in the signature only; the tool forwards only what is given
  (`core/tool/config_args.py:180-186`).
- **Refusals** before anything is simulated, each a `Refusal` with a registered field key, in the
  stage order the other diagnostics keep: resolve the store, `assert_name_free("diagnostic", name)`,
  the rules, anything else that can refuse, and only then `store.create`. New keys are added to
  `core.refusals.FIELDS` (with the registry count and `TOOL_ONLY_KEYS` in `tests/test_refusals.py`),
  `core/tool/fields.py` `FLAG`, `core/gui/fields.py` `CONTROL` as None, and the pinned no-control set
  (`tests/test_nav_and_gating.py:1607-1612`). Where an existing key's registered default would be
  false for a probes knob (the `m` key reads "32", laplace's), the knob gets its own key.

### 4.2 Rules for all three modes

- **It measures only.** No mode writes a setting, passes `chi_f0` or `chi_freq_bounds` into
  `cli.make_sim_config`, or can feed a training run. Every record's `settings` states the configured
  `CHI_F0`, `CHI_FREQ_BOUNDS`, `CHI_K_PAD`, `CHI_MIN_CYCLES` and `CHI_MAX_CYCLES` it judged (the
  manifest's `config` nulls the chi constants outside chi mode, `manifest.py:279-282`). Verdicts read
  "the configured … holds / does not hold for this cell"; any suggestion names `core/config.py` as the
  place a deliberate change is made. A test pins that the family's modules assign no upper-case config
  name and that a training run's `SimConfig` is unchanged after a probes run in the same process.
- **Faithful to training today.** The modes use the production estimators: `chi.peak_freq` for each
  sample's own Ω₀, `chi.lock_in_batched` with `n_samples` for the per-row cycle ceiling (production
  caps per row and rounds down, `core/SBI/chi_probes.py:176-192`; the archived band script used one
  prefix per batch and rounded up), and a probe at or above 0.9 × Nyquist is **masked, not clamped**
  (`chi_probes.py:164-169`; the archived script clamped). The recording geometry and the training
  recording-length ceiling come from one shared helper (§4.6), not copies.
- **Tier-1 aware.** Each simulation derives the force scale exactly as training does —
  `derived.to_sim_rescale(…, *cfg.tier1_args)` with `cfg.sim_rescale_idx` (`core/orchestrator.py:130-137`).
- **Grids are measurement, not override** (H8): `band` and `drive` take their own frequency and drive
  grids.

### 4.3 `band` — do the configured band and drive hold for this cell?

**Inputs:** `--bounds`, `--cell` (built with `config_args.build_cfg(load_gt=True)`), chi mode.

**Defaults** (`recon/design-f0sweep.md` §3): five recording lengths log-spaced from
`T_MIN_EXP_S` (1 s) to the training ceiling for this cell (26.909 s for the master cell at t_scale
3.73), rounded down to 0.01 s — a length beyond the ceiling may be given and is **flagged**, not
refused; four frequencies log-spaced across `CHI_FREQ_BOUNDS` plus one control at twice its top edge
(`[0.03, 0.0646, 0.1392, 0.3, 0.6]` today); the configured drive `CHI_F0` (a list may be given); 24
noise repeats; the configured cycle ceiling, with an optional list of caps that adds the wall report.
Cost: lengths × (1 + frequencies × drives) ensembles — 30 at the defaults.

**The four criteria** at each (length, frequency, drive), uncapped and capped:

| criterion | measure | default pass |
|---|---|---|
| amplitude reproducibility | spread of \|χ\| over the repeats ÷ its mean | ≤ 0.20 |
| phase scatter | circular spread of arg χ, radians | reported, not judged unless a threshold is given |
| signal over the floor | mean \|χ\| driven ÷ the same lock-in on the undriven ensemble | ≥ 3.0 |
| not captured | the bundle's power within ±10 % of its undriven Ω₀, driven ÷ undriven | ≥ 0.50 |

Each threshold is a flag; `phase_max` is recorded as null when off. The cycle-floor mask
(`CHI_MIN_CYCLES`) is reported per point, not judged.

**Verdict.** A frequency passes only if it passes at every recording length and every drive; the band
holds when every in-band frequency passes capped; the drive holds when no in-band point is captured.
The record also carries the cap-rescue summary (full-length failures that pass under the cap) and,
when caps are given, the wall (the largest clean cap below the first failing one).

**Two caveats, recorded with the verdict.** The reproducibility, floor and phase thresholds are
conventions; only capture is physical evidence. A probe whose low harmonic (up to the fifth) lands
inside the own-peak window is flagged, because harmonic power inflates the not-captured measure and
could hide capture (at 0.3 × Ω₀ the third harmonic sits at 0.9 Ω₀, on the window edge).

**Outputs:** per point the four measures, cycles and flags (npz and results); per frequency the verdict
with its reasons; the overall verdict; heatmaps of reproducibility and signal, capped and uncapped.

### 4.4 `mask` — why are training probes thrown out?

**Inputs:** `--bounds`, chi mode, `--prior REF` (required) — the first diagnostic to take a prior:
loaded with `store.load_prior`, which refuses a model, order, box or log-mask mismatch, and checked by
`run_guards._assert_chi_config_is_deliberate` (which `build_prior`'s load branch runs and
`store.load_prior` does not); recorded as the parent with the prior's fingerprint. A missing prior is
refused, never built. No cell: training is truth-free, and the initial conditions come from
`orchestrator._observation_inits(cfg)`.

**The observer.** `pipeline.gen_training_data` gains one keyword-only argument, an optional probe
observer (default None), carried to the chi row builder and called once per batch attempt with that
batch's tag, each row's Ω₀, the drawn duration fraction, the probe count, and `gen_chi_raw`'s returned
`(u, logcyc, valid)` beside the packed mask. With the default None nothing changes: a test pins that
seeded `gen_training_data` output is byte-identical with no observer and with a no-op observer, in
chi, forced and spontaneous modes. The audit keys records by batch tag and keeps only a batch's last
completed attempt (an out-of-memory retry re-runs a batch; row halving splits one).

**Defaults:** 12 batches × 32 rows through the real generator (the Sobol schedule, the per-batch
probe count, placement, the duration draw and the per-row subsetting are production's); optional
`--chi-k-fixed` audits one probe count; `--seed` changes the noise, and the record says that the probe
layout comes from training's fixed probe-generator seed (`pipeline.py:1353-1354`), so it does not
change with `--seed`.

**Reports:** the share of probes thrown out by each cause, taken from the returned values, not
re-derived — the cycle floor (`~valid & ~bad & ~nyq`), split into "too slow even at the band's top at
full length" (`f_peak · T · hi < CHI_MIN_CYCLES`) and "shortened by the duration draw"; non-finite
lock-ins and the band filter at the packer (`chi.py:548-551`); the per-batch masked fraction with its
spread (the effective sample size is the batch count); Ω₀ quantiles in cell units and Hz; rows left
with zero or one usable probe; the span of the frequencies actually driven (`f_peak · exp(u)`, not the
drawn multipliers — placement lifts them). Causes that cannot fire in training (non-finite frequency,
Nyquist, the band — `recon/design-maskaudit.md` §5) are reported as invariant checks. The audit also
captures `gen_chi_block`'s own per-batch masked warning and records whether each batch's count equals
its own; a mismatch is recorded and warned, never silently accepted.

### 4.5 `drive` — how hard can a lab drive this cell?

**Inputs:** `--bounds` with a Forcing section (`feature_sets.assert_forced`), `--cell`; built as not
chi (the chi flags are not declared on this mode).

**Defaults** (`recon/design-drive.md` §1): recording length 5 s, 16 repeats, drive at 1.4 × the
cell's own Ω₀ (detuned so the window does not see the drive), drive strengths
`[0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0]` in model
units plus the configured `CHI_F0`, measured explicitly; own-peak window ±2 % of Ω₀ (the archived
±2 bins at 5 s, about ±1.8 %); free-running when at least 70 % of the undriven own-peak power remains,
captured at 10 % or less, otherwise in between.

**Reports:** Ω₀ in cell units and Hz, the per-trace peak spread, the peak's clarity (peak ÷ median
power), peak-to-peak and cycles in the recording; per strength the median |χ|, the median phase-locking
value (reported, not judged) and the own-peak ratio with its verdict; **the strongest free-running
strength** — the largest in the leading unbroken run of free-running strengths, so a strength whose
own-peak bin refills at very large drives can never be picked — and **the weakest captured strength**,
each in model units and in the cell's own force units (the derived force scale on the tier-1 box); the
`CHI_F0` row; and suggested Forcing lines and box for a cell file, printed and recorded, **never
written** (nothing outside `core/config.py` builds a `Resources/` path). "Free-running versus
captured" is the criterion; there is no linearity test and the record says so.

**Results, not refusals** (known only after the spend): no clear oscillation (clarity below a
threshold, then every verdict null with a warning), nothing captured within the grid, a drive at or
above 0.9 × Nyquist.

**Parity first.** Before any default changes, the master cell (`master.txt` + `master_spont.txt`) must
reproduce the archived readings: 0.02 free-running and 0.2 captured.

### 4.6 Shared helpers

- **Recording geometry and the training recording-length ceiling**: one function returning
  `n_obs`, `n_fine`, the subsample and the ceiling for a config and a length, extracted from the two
  copies in `orchestrator.generate_observations` (`core/orchestrator.py:141-172`) and the Sobol
  pre-filter (`core/SBI/pipeline.py:1159-1172`); both callers use it, behaviour-neutral, pinned by the
  byte-identical seeded-training test of §4.4 and by a deterministic test (the master cell's ceiling is
  26.909 s).
- **The own-peak ratio**: one function (forced ensemble, undriven reference, Ω₀, window fraction) used
  by `band` and `drive`, each passing its own thresholds.

### 4.7 Tests

- Fast, with a stand-in simulator of known response (the pattern of
  `tests/test_diagnostics.py:703-723`): each mode's verdict logic, the record shape, the refusals
  before the writer opens, non-finite values becoming null, the harmonic flag, the leading-run rule
  for the free-running strength, the observer's by-tag bookkeeping and its cross-check against the
  production warning.
- Deterministic: the master cell's training ceiling (26.909 s) and the default grids.
- The byte-identical seeded-training test (§4.4, §4.6).
- `tests/test_tool.py`: every flag reaches its stage as a keyword (the forwarding pattern of
  `:659`, `:711`, `:826`), `probes` and each mode's `--help` import no torch, modes do not share flags
  they should not.
- The measure-only pin (§4.2).
- One slower real-simulation check: the master cell and its tier-1 twin
  (`master_spont_tier1.txt`) give identical reproducibility, phase, signal and capture at one seed, with
  |χ| scaled by the ratio of their force scales (10 against the derived 47.0 pN).

### 4.8 What the card should show

`probes band` on the master cell (`master.txt` + `master_spont.txt`) says (0.03, 0.3) holds and the
0.6 × control is captured; `probes drive` on the master cell finds about 0.02 free-running and 0.2
captured; `probes mask` against the smoke prior lands within ±12 points of 37 %, with the cycle floor
the only cause. These came from the old solver and the old per-batch lock-in, so they match
statistically, not bit for bit.

About eight tasks.

---

## 5. Part 4 — retrain readiness

Each fix comes with a test that fails before it and passes after.

### 5.1 The tier-1 path, tested

A new suite, `tests/test_tier1.py`, with a stand-in simulator that records the force scale it is
handed:

- `master_tier1.txt` parses to the rescale block `x_scale, t_scale, T` and switches tier 1 on
  (`core/SBI/derived.py:53-55`); `SimConfig`'s `sim_rescale_idx`, `tier1_args` and `k_b_cell`
  (`core/sim_config.py:375-419`) are what the tier-1 relation needs;
- training targets keep `T` while every simulation receives the derived force scale
  (`core/SBI/pipeline.py:1522-1535`); so do the Fisher operating points (`core/SBI/decorrelate.py:158-166`),
  the calibration set (`core/SBI/analysis.py:168-180`), a simulated observation
  (`core/orchestrator.py:131-137`), and the predictive checks (`:2021-2027`, `core/SBI/ppc.py:85-87`);
- a narrowing round with a tier-1 parent runs its region, containment and truth check in inferred
  coordinates;
- a prior built on `master.txt` loads under `master_tier1.txt` (the same ND box; by design);
- the records round-trip (§5.7).

Plus one slower real run: `smoke` at tiny size on `master_tier1.txt` + `master_spont_tier1.txt`.

### 5.2 The silent fallback, closed

When a box declares no `f_scale`, the force builders fall back to the Hopf form
`f_scale = x_scale / t_scale` (`core/forcing.py:136-144`, "Hopf-style nondim …";
`core/SBI/chi_probes.py:112-113`). That is right for a Hopf-style box and silently wrong for a tier-1
box handed its inferred index. Both builders **refuse** when the index names `T` and not `f_scale`
(field None: a programming error, not a user input), which closes the class for every caller, the new
probe checks included. `identifiability laplace` and `jacobian` then derive the force scale as
training does (`core/diagnostics/identifiability.py:248-249`, `:525-531`, `:543-544`, with the rescale
vector built at `:860`). Tests: each diagnostic on a tier-1 box, before the fix returning a wrong
number silently (or, after the builder guard alone, refusing), after it matching the value computed
with the derived force scale.

### 5.3 Temperature as an assumed input

On the tier-1 box temperature's posterior is its prior by construction (`derived.py:37-39`).

- `SimConfig` gains `assumed_labels`: the inferred labels whose posterior is the prior by
  construction — `("T",)` when tier 1 is on, `()` otherwise — derived from the box, never typed.
- Every record whose config has assumed labels records them (`assumed: ["T"]`).
- **Inference:** the corner plot labels temperature "T (assumed, K)" (the bare `$T$` of
  `core/Helpers/labels.py:17-20` gains its unit and the mark); `posterior_summary`
  (`core/orchestrator.py:2161`) marks the entry `assumed: true`.
- **Calibration:** the rank-uniformity table (`core/orchestrator.py:1811-1815`) marks an assumed
  parameter "(assumed: prior by construction)" and leaves it out of the verdict (§5.4); the joint
  coverage test (TARP, `:1849-1855`) is computed over the non-assumed parameters by removing the
  assumed columns from the posterior samples and the ground truths before the coverage computation
  (the plan confirms sbi 0.25's sample-level entry point); the informativeness total stays joint
  (§1.2) and the assumed parameter's own entropy entry is marked and reported apart.
- `sbc` marks assumed parameters the same way.
- Log lines where "T" means the recording length (`(t_scale, T)`, `pipeline.py:1491-1493`,
  `logT`) say `T_obs`, since on the tier-1 box `T` is also temperature.

### 5.4 A calibration verdict, and a repeatable calibration set

- `validate_calibration` ends with one verdict record (and `results["verdict"]`): **PASS** when every
  non-assumed inferred parameter's rank-uniformity KS p ≥ 0.05 ÷ (their number), and the joint
  coverage test over the same parameters has KS p ≥ 0.05 (H11). The record lists each parameter's p
  against its threshold, the assumed parameters left out, and one caveat: t_scale's test rests on the
  calibration's operating points (`CAL_N_SCALES`, 200), not on its 2,000 datasets, so it has less power.
  A FAIL is a result: the stage completes, writes its record and exits 0; the runbook reads the verdict.
- `validate_calibration` gains `seed=None` and the tool `validate --seed`: a given seed makes the
  calibration set repeatable (run inside `core.rng.seeded`); with none, one is drawn and recorded, as
  the FDT runs do.
- `sbc` records, per repeat, whether the same rule passes, and the fraction of repeats that do.
- The Validate tab shows the verdict line in its log pane (it arrives as an info record) — walkthrough
  row F1.

### 5.5 The eigenvalues survive a resume

The training checkpoint's header stores the rotation's eigenvalues beside the rotation
(`core/SBI/training_checkpoint.py:175-186`, beside `"V"`); a resumed run writes them into the
posterior's manifest (`core/orchestrator.py:1370-1387`, today only when the Fisher ran in the same
process); a narrowing child records its parent's, since it reuses that rotation. A header written
without them (none exists after the clean break) records them as unknown. Test: a run resumed from a
complete cache carries the same eigenvalues as the run that wrote it.

### 5.6 The narrowing round reports truth inside as well as outside

With a known cell loaded, the round reports, per truncated direction, whether the truth lies inside the
region — today it warns only when outside (`core/orchestrator.py:1215-1232`) — and records the
per-direction containment in the child's manifest. Test: a truth inside prints and records "inside" for
every direction; one outside keeps today's warning.

### 5.7 The constraint recorded

- Every record trained or simulated on a tier-1 box names the constraint in its `config`: the relation
  `f_scale = N · beta · k_B · T / x_scale`, `k_B` in the cell's units (`k_b_cell`), and the `T` range
  (`core/artifacts/manifest.py:247`, `config_from_cfg`, gains the block; absent otherwise).
- The simulation cache's identity gains the units file's fingerprint, because the derived force scale
  depends on it (`core/artifacts/identity.py`, `FORMAT = "training-rows/2"` becomes `/3`). No cache is
  stranded: none exists.
- A simulated observation records the force scale it was actually simulated at beside `T`.

### 5.8 `smoke` takes the network size

`smoke` gains `--hidden-features` and `--num-transforms`, forwarded to `build_posterior` (today it
cannot set them, `core/tool/smoke.py:207-212`), so the card run exercises the retrain's network end to
end.

### 5.9 One run-total line for masked probes

When generation ends, training logs one record: probes masked over the run (count, total, percentage)
and the per-batch spread; `smoke` prints it too. The ±12-point check around 37 % is then read, not
summed by hand from the per-batch warnings (`core/SBI/chi_probes.py:267`). CLAUDE.md's pass criterion
names the line.

### 5.10 What stays a runbook step

The larger network's training time has never been measured (compute per step rises about fourfold;
`recon/design-tier1.md` §5). The runbook's pre-flight measures seconds per epoch on a short real-width
run and projects the full fit, and stops for the owner's decision if the projection is far longer than
expected (§6.5). Host memory at the start of training is about 50 of 63 GB (sbi 0.25 copies the data
several times; `recon/design-tier1.md` §5); the runbook says to close other applications.

About seven tasks. They touch code that moves tensors, so the card run of §8 is required.

---

## 6. Part 5 — the documents

### 6.1 Where, and the rules every page keeps

- **Location:** `docs/guide/`, with its own `README.md` as the start page (GitHub shows it when the
  folder is opened). The root README links to it. The working record stays where it is and is never
  cited.
- **Self-contained:** a page cites code by module and function name, never by line number, and cites
  no working document (the §2 scan walks `docs/guide/`).
- **Stamped:** each page opens with the commit it was checked against.
- **Links resolve:** a new test (`tests/test_docs.py`) checks every relative link and anchor in
  `docs/guide/` and `README.md`.
- **Written fresh** from today's code, with the handoff and `recon/handoff-map.md` as sources; the
  transposed table, the withdrawn "k is unmeasured" conclusion, the five false lines and the stale
  memory rows are never carried.

### 6.2 The pages, and what each carries

| page | what it holds | from the handoff (by section; `recon/handoff-map.md` has the detail) |
|---|---|---|
| `README.md` | what PRISM is; the window and the tool; one ordered reading path each for the owner-scientist, a lab member, a successor, a reviewer | §0 and §1.1, rewritten |
| `getting-started.md` | launching; `Resources/` inputs versus `Artifacts/` records and the two root overrides; a first run in the window and on the command line; the store's kinds, records, names and ids; browsing it (the Artifacts screen, `artifacts`) | §1.1–§1.2, rewritten |
| `window.md` | each screen; the inference settings table — field, flag, argument, default, what it really changes, the five that are not speed dials, the Fisher settings doing nothing on a resume; what the window remembers and why; which models each screen supports, with a command-line column | §2.3, §4.5 (plus CrossVal and the Artifacts screen), the flow-knob and budget notes of Appendix A |
| `command-line.md` | the full reference (§6.4) | §1.1, and the scripts-to-subcommands table as history in one paragraph |
| `recordings.md` | the lab-member guide: the Bounds / Cells / Units / Models formats, how a cell finds its bounds (sibling, else `master.txt`), the three observation modes and what each needs, drive frequencies in Hz, recording length against the training range (the ceiling of ~27 s at t_scale 3.73; the chi length check at T* ≈ 2.93 s), the band / drive / masking assumptions and checking them with `probes`, defining a user model | §3.3, §3.5, §4.3, the 2026-08-06 "T_obs ceiling" note, U traps |
| `science.md` | the reasoning behind the settings: the nondimensionalisation; the conditioning features, valid flags, rank-Gaussianisation and winsorisation; chi probe design (the band measurement, the cycle floor and ceiling and the wall, the Ω₀ physics of masking, per-row placement and lock-in, the ±12-point precision); what chi buys; the prior (stability screen, component count); the tier-1 constraint and why T is assumed; reading calibration honestly (KS over c2st ranks, pooled histograms, t_scale's effective sample size, best fits are not recovery, flat SBC is not informativeness); narrowing rounds (the truncated prior, tempering, the rules' reasoning); identifiability limits (n, temp and tau_c enter only through noise amplitudes; a clean loss plateau); the August retrain's lasting findings in the correct orientation; the solver's physics check (graphs on/off); **open questions** (M-replicate conditioning, the thermal tail feature, the strata test, step and intermodulation observables, tiers 2 and 3 and the fix-T 12-dimension alternative, the log-sampling decision, pooling rows across rounds, and the owner's science items from STATE) | §4.2–§4.4.1, §4.6 (durable findings only), §8's `dt_nd_min` note, §11.1–§11.7, Appendix A's science items, "Conventions worth keeping" |
| `architecture.md` | the module map; the stages and compositions; the store (kinds, manifests, identity, loading refusals and the two acceptances, progressive records); the tool; the window (panels, worker threads, stream routing, progress, cancellation, settings persistence); the solver, CUDA graphs and reproducibility; memory planning and out-of-memory recovery; the FDT pipeline | §2, §2.1–§2.2 regenerated from `core/`; §8.3, §10.5; the P/S/C trap groups' design; Appendix A's memory and recovery entries |
| `rules-and-traps.md` | the rules the code keeps (the model/solver contract, the force-channel rule, the box/cell/units triple, everything nondimensional, the conditioning layout, every knob an argument, the private configuration copy, refusals with field keys, no chi override, the narrowing-round refusal with no escape hatch, the comment policy); **the narrowing-round safety rules as a table** — each rule in the words of §2.3, where it is enforced, the test that pins it, and the ones still only prose (the rate of truths outside the region; a region cut in the box when the parent has no rotation; the −log P(A) inflation printed, not corrected); the traps that still bite, rewritten against today's code, grouped by subsystem, without the old labels | §3, §5 (every trap the map marks still-applies), §6.1's rules, §11.6, Appendix A's traps (the TorchScript warm-up, the facade re-import, waiting on memory one holds, the allocator variable's real name) |
| `testing.md` | environment and interpreter; the environment variables; the gates and markers and their budgets; the card smoke gate and the diagnostic card (now with `probes`); what a green suite does not certify; test-writing notes (offscreen geometry, assert order not adjacency, never edit during a run) | §1.2–§1.3's live rules, §10.3, Appendix A's test notes |
| `retrain.md` | the runbook (§6.5) | §4.1's live guidance, §11.8–§11.9's gates, costs and watch-lines, rewritten |

### 6.3 The README

Installation stays as it is. The one-line mention of the tool (`README.md:155`) becomes an ~80-line
command-line summary: what the tool is and how it relates to the window; one line per subcommand and
mode; the shared flag groups once; what each writes (the store kinds); the environment settings; the
exit codes; three worked examples (a stage chain, a probes check, an FDT run and a comparison); a
pointer to `docs/guide/command-line.md` and to `<sub> --help`.

### 6.4 The command-line page, and its test

`docs/guide/command-line.md` is **written from** the tidied help (not generated): the shared flag
groups once, then a heading per subcommand and mode with its flags, outputs, refusals and exit codes,
the rules the parser cannot express (infer's recording rules, smoke's resume combinations, fdt's
`--skip-sanity`/`--no-production`, crossval's grid rules, the near-miss and narrowing refusals), and
worked examples. A test in `tests/test_docs.py` walks the parser (the `_walk` of
`tests/test_refusals.py:906`) and checks, in both directions, that every subcommand, mode and flag has
its heading and entry and that the page names no flag or subcommand the parser lacks, and that each
flag's stated `(default X)` equals the help's.

### 6.5 The retrain runbook

- **Decisions**, each with its reason: the tier-1 box; 10,000 × 2,048; 256 × 10; 6 probes supplied into
  12 slots; the certification round; the verdict (H11). The box needs a prior built on it — the ND
  screen is the master box's, so a `master.txt` prior loads under it (§5.1) — and the runbook says
  which to use and why.
- **Pre-flight:** the card gate including the tier-1 and larger-network line (§8); a short real-width
  `train` measuring seconds per epoch at 256 × 10 with the projected fit, **stopping for the owner's
  decision** if the projection is far beyond the measured ~92 h at 128 × 8 scaled to 20 million rows;
  the chi recording-length check (`identifiability jacobian`) on the master cell and, now fixed, on the
  tier-1 cell; `probes band` and `probes mask` on the card; VRAM by `nvidia-smi` (never
  `torch.cuda.mem_get_info()`), host memory (~50 of 63 GB at training start: close other
  applications), disk (the cache is ~10.3 GiB; the pre-flight line prints ~9.9 GiB because it counts
  8 target columns, not 13).
- **The run:** the exact commands with every flag (`--chi` everywhere, since `CHI_MODE` is False by
  default; `--num-runs 10000 --hidden-features 256 --num-transforms 10` — tsnpe does not inherit the
  network size or the batch count from its parent, so the round passes them too); checkpointing every
  50 batches; how to resume (`--resume require`, the two lines it prints, the near-miss refusal and
  `--new-run`); what to watch (the `[cfg] chi` banner, `[tier1]`, `[budget]`, `[fisher]` — printed only
  by a fresh process — the masked-probe run total, `[patho]`, `[winsor]`, the loss curve's figure);
  capturing the console, since a resumed run's first process's lines exist nowhere else.
- **Gates:** the verdict PASSES; the previously dead input channels are revived (`ablation`, against
  the healthy channels of the same run — the old "29 of 42" baseline was on a 42-wide summary and does
  not apply); the masked-probe run total is within ±12 points of 37 %; the eigenvalues are recorded
  (`identifiability rotation`); the loss curve is recorded with the plateau reading; the informativeness
  total and per-parameter decomposition are written down as the first baseline.
- **The certification round:** `infer` on the tier-1 cell to create the observation (its id from the
  `[prism] observation` line); `tsnpe` with the flags above; pass when the truth lies inside the region
  in every truncated direction (§5.6), the child's calibration verdict on the region passes, and the
  widths shrink no more than the data supports (the proposal test of the suite is the arithmetic
  guarantee; the run checks containment and calibration).
- **Afterwards:** the gate results stay with the model — the calibration record holds the verdict and
  the informativeness numbers, the posterior's note names the run, and `artifacts summary` prints its
  lineage report; the page says which record answers which gate. (Being a reader page, it names no
  working document; the controller copies the results into STATE's gate table as usual.)

### 6.6 Retiring the handoff

1. **The only-record checklist.** The documents review confirms that every item whose only record is
   the handoff has a home. The checklist, built in the piece's ledger from `recon/handoff-map.md`,
   includes at least: the facts under "only record" in each chunk's notes (e.g. the TorchScript
   warm-up trap, the offscreen-geometry test constraint, the production-denominator ranking and the
   7.9× graph measurement, the comment policy's never-delete list, the prior picker not loading a prior
   until Build/Load is pressed, the prior sweep having no out-of-memory retry, the allocator variable's
   real name, qfluentwidgets rejected as GPLv3, the declined hard process cancel); the open engineering
   items (the FDT sanity checks' solution view, `decorrelate`'s per-call re-derivation, the §8.3
   open-performance table, the `vt.py` rename and directory casing, the orphan `.claude/worktrees`
   copies), which go to STATE's open list; and the science ideas, which go to `science.md`.
2. **The move:** `PRISM_HANDOFF.md` moves on disk into `archive/`, then `git rm --cached` (CLAUDE.md's
   archiving rule).
3. **Repointing:** `CLAUDE.md` (the "science guardrails are in §11.6, traps in §5" line points at
   `docs/guide/rules-and-traps.md` and `science.md`; the "still the reference until then" clause goes;
   the suite list gains the new suites; `docs/guide/` joins "Where things are"), `requirements.txt`'s
   header (the fact stated, not the section), `docs/checklists/display-walkthrough.md`'s header (seeded
   from the handoff's list, said as history), and `docs/STATE.md` (item 7 closed; the archiving commit
   named so `git show <commit>:PRISM_HANDOFF.md` retrieves it). The closed specs and plans stay
   untouched.

### 6.7 The documents review

Three lenses, each followed by an adversarial verifier, before the handoff moves:

- **accuracy** — every checkable fact on every page (a module, function, flag, default, number,
  command) against the code at the stamped commit;
- **completeness** — every load-bearing section in `recon/handoff-map.md` has a home or a recorded
  reason to drop it, and the only-record checklist is closed;
- **reading paths** — a fresh reader with only `docs/guide/` follows each path and does its job: a lab
  member checks a preparation with `probes` and runs an inference on a recording; a successor finds
  where a refusal is raised and why; a reviewer reproduces a calibration from a record.

About ten to twelve tasks.

---

## 7. Tests

- **Budget:** about 80 ± 25 new tests on the 968 collected at `7e51275` — the readiness fixes and the
  probe checks carry most of them.
- **New suites:** `tests/test_source_hygiene.py` (§2.8), `tests/test_tier1.py` (§5.1),
  `tests/test_docs.py` (§6.1, §6.4). The probe checks' tests join `tests/test_diagnostics.py` and
  `tests/test_tool.py`, as every diagnostic's do. `tests/_fixtures.py`'s scan roots are unchanged
  (`docs/` holds no Python).
- **Changed:** the three label-pinning tests (§2.4), the help-text pins (§3.8), the public-entry and
  field-key tables (§4.1), the two renamed tests.
- **Slow set:** a real-simulation check over about a minute is slow-marked — the tier-1 tiny `smoke`
  (§5.1) and the probe checks' twin-cell check (§4.7). CLAUDE.md's slow-set description is updated.
- **Fast gate target:** 16 minutes, one process.

## 8. Gates

- **Per task:** the task's review and its one-process fast gate start at the same moment (the owner's
  rule since piece 5); a cleanup task also passes the §2.7 check and shrinks the §2.8 list.
- **After part 4:** a whole-code review by independent readers — correctness, match to this spec, tests,
  the hazards no CPU suite reaches, and the reference rule — each followed by an adversarial verifier;
  one fix dispatch; a scoped re-review.
- **The card run**, after the fix dispatch, alone on the card, every result into STATE's gate table:
  1. CLAUDE.md's four existing lines, unchanged, with their pass criteria;
  2. **run 4, the tier-1 line** (added to CLAUDE.md's recipe permanently, since it is the retrain's
     box, and to STATE's verbatim copy of that recipe under its gate table, which must stay identical): `smoke --chi --t-obs 4.5 --bounds Resources/Bounds/nadrowski/master_tier1.txt --cell
     Resources/Cells/nadrowski/master_spont_tier1.txt --hidden-features 256 --num-transforms 10
     --checkpoint --save --store-root "$S/smoke_t1"` — exit 0, no OOM line, the `[tier1]` lines
     present, `T` in the posterior's parameter keys, the constraint block in its manifest, the
     masked-probe run total within ±12 points of 37 %;
  3. its resume (`--prior smoke_prior --stages prior,posterior --resume require`), which must resume and
     carry the eigenvalues (§5.5);
  4. a tiny narrowing leg on run 4's store: `tsnpe` from `smoke_posterior` with a small budget, then
     `validate --accept-truncated` (the verdict line) and `infer --accept-truncated
     --accept-other-observation` (truth containment, §5.6);
  5. the diagnostic card: `sbc`, `identifiability jacobian`, `ablation` against `$S/smoke`,
     `identifiability laplace` against `$S/smoke_chi0`, plus `identifiability jacobian` on the tier-1
     cell (now correct);
  6. `probes band`, `probes drive` on the master cell and `probes mask` against `$S/smoke`'s prior,
     against §4.8's expected readings.
- **After part 5:** the documents review (§6.7), its fixes, the only-record checklist closed, the
  handoff moved.
- **Final:** the fast gate on a quiet machine and the slow set, then STATE and CLAUDE.md from the
  measured numbers. The card run stands for the piece unless a later change touches code that moves
  tensors, in which case the affected lines are re-run.

## 9. The display walkthrough (rows F1–F5)

Added to `docs/checklists/display-walkthrough.md`, run by the owner on the real screen. F1 first walks
through building a tiny tier-1 model in the window (prior on `master_tier1.txt`, a posterior at a tiny
budget, the tier-1 cell loaded) so the rest have something to show.

| row | what the owner checks |
|---|---|
| F1 | the Validate tab's log ends with the verdict line; temperature is listed as assumed and left out |
| F2 | the Infer tab's summary and corner plot mark temperature "T (assumed, K)" |
| F3 | a reworded refusal (loading a narrowed posterior where a broad one is required) shows in the yellow box with a plain sentence and no label |
| F4 | the Posterior tab's budget tooltip gives its measurement with no incident label |
| F5 | a narrowing round's log in the TSNPE tab reports truth containment per direction |

## 10. Risks, and the order of work

**Order:** part 1 (§2.9: nine tasks, the scan first) → part 2 (help tidy) → part 4 (readiness; the
tier-1 tests before the fixes they cover) → part 3 (the probe checks, on top of the shared helpers and
the observer) → the whole-code review and fixes → the card run → part 5 (the pages, the runbook, the
README, the reference page and its test) → the documents review and fixes → retiring the handoff →
final gates → STATE and CLAUDE.md → the owner's walkthrough.

| risk | guard |
|---|---|
| a code change hidden in 2,000 comment edits | the per-file code-unchanged check (§2.7); the known raw-source tests |
| the observer and the shared geometry helper touch the training generator | byte-identical seeded output in three modes; the card run |
| the identity format change | a version bump; no cache exists to strand |
| the verdict read as more than it is | the t_scale power caveat printed with it; informativeness recorded beside it |
| the tier-1 box never trained at scale | the card legs are tiny; scale is the runbook pre-flight's job |
| the larger network's training time is unknown | the runbook's projection and stop point |
| the documents drift | the reference-page test, the link test, the scan over `docs/guide/`, the commit stamp |
| a long plan | the plan gets a pre-flight against the code, and every ruling is folded into the task it binds (a plan's preamble does not reach a task brief) |
| the scan's allowlist grows into a hole | an entry that matches nothing fails; entries are exact names or per-file sets |

## 11. Deviations (filled during execution)

| # | where | what changed | why | cost if wrong |
|---|---|---|---|---|
