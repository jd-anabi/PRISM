# PRISM piece 6: the reference cleanup, retrain readiness, the probe checks and the documents (design)

**Status:** draft for the owner's review. Brainstormed with the owner on 2026-09-25: every decision in
§1.1 is the owner's answer to a plain-language question, and §2–§10 were each presented in chat and
approved section by section before the first draft (`7ad6498`) was written. That draft was then
reviewed against the code by four independent lenses (anchors, completeness, feasibility, standing
rules) and a judge that re-checked every serious finding itself — verdict "ready with fixes": 19
must-fix and 33 corrections, all folded in here (the report is the gitignored
`.superpowers/sdd/2026-09-25-documentation-and-retrain-readiness/spec-review.md`). On 2026-09-28 the
owner took the recommended answer to the review's owner questions: H11 is revised and H13 added
(§1.1); the review's other defaults are recorded in §1.2.
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
plus `design-scanrule-hits.tsv` (every reference line the rule of §2.2 matched on 2026-09-25, 1,976
rows).

**Anchor rule.** Every `file:line` below was read at `8556546`/`7ad6498` (the code is the same), but
**the quoted text or the named symbol is the anchor, not the number** — line numbers go stale between
the plan and the task. An implementer who finds the quoted text elsewhere in the named file has found
the right place.

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
  ~13 %, traps ~7 %, contracts ~5 %, runbook material ~3 %, fully superseded ~5 %, the rest small. It
  is false in far more places than the five lines STATE names (`:49,53-54,76,190-191`): the three front
  ends (`:7-9`), the scripts row and tree (`:51`, `:185-189`), the pre-pytest test setup (`:74-140`),
  and every environment-variable recipe that runs a deleted script (e.g. `:584-585`, `:3612`). Its
  memory table contradicts its own later entry: `:2615` (the Welch spectrum's float64 promotion) is
  fixed as `:6756` says, but half of `:2616` is still open — `core/FDT/sanity.py:291` and `:328` keep
  a view of the whole solution (`x_steady = sol[0, 0, :, burn_idx:]`, no `del sol`) — and six
  neighbouring rows carry no status. The August retrain's per-direction table (`:1404-1416`) is the
  transpose (its own banner, `:1355-1380`); the transposed reading survives in `:3333-3335` and
  `:3621-3625` (k unmeasured), `:3450` (direction 4 read as evidence about t_scale's strata), and in
  `core/SBI/truncate.py:21-24`.
- **References to the working documents** (`recon/design-scanrule.md` §1): 872 lines in 86 of the
  178 tracked `core/**/*.py` files, 1,101 lines in 20 of the 21 `tests/*.py` files, and three other
  lines (`conftest.py:1`, `requirements.txt:9`, `:44`). About 80 % are provenance tags that can simply
  be deleted; 400–500 lines explain a reason by pointing at a document and must be rewritten. Ten
  messages users see carry labels (§2.4), three tests pin them, and two test names carry them.
- **The retrain's path** (`recon/design-tier1.md`, `design-runbook.md` §9): the tier-1 box
  (`Resources/Bounds/nadrowski/master_tier1.txt`: temperature `T` in 280–310 K in place of `f_scale`,
  which is derived) has no test coverage beyond three pure-function tests
  (`tests/test_conditioning_repair.py:251-278`); `identifiability jacobian` and `laplace` silently use
  the wrong force scale on it; temperature is presented everywhere as an ordinary inferred result;
  calibration prints numbers but no verdict and cannot be repeated on the same set; a resumed training
  run loses the rotation's eigenvalues; the narrowing round is silent when the truth lies inside its
  region; no record names the tier-1 constraint; a calibration or narrowing round run under
  `--accept-truncated` does not record it.
- **The tool's help** (`recon/design-helptidy.md`): 835 lines over 27 screens, 15 flags with no
  description, about 50 rendered placeholders that show internal names, shared flags with up to five
  wordings, defaults stated five different ways.
- **The chi settings**: `archive/scripts/chi_f0_sweep.py`, `chi_mask_audit.py` and
  `build_master_cells.py` measured them; none can run today (`import _common` — deleted at `7433ced`
  — and `from core.config import PLOT_PATH` — removed) and each has drifted from the production path
  (`recon/design-f0sweep.md` §4, `design-maskaudit.md` §2, `design-drive.md` §2).

### 1.1 Decisions (binding)

Each is the owner's answer to a plain-language question (2026-09-25 unless marked).

| # | decision |
|---|---|
| H1 | **Scope.** The handoff split into reader documents, a README tool reference, and a new written retrain runbook. No scripted runbook. Plus H4, H7, H9 and H13, which the owner added during the questions. |
| H2 | **The handoff moves to the gitignored `archive/`** (move on disk, then `git rm --cached`, never `git mv`). Every LIVE reference is cleaned; the closed specs and plans stay untouched, their handoff citations resolving through git history; STATE records the archiving commit. |
| H3 | **Topic pages with a reading path per reader**, each fact written once: one ordered path each for the owner-scientist, a lab member bringing recordings, a successor, a reviewer. Written fresh from today's code, with the handoff as a source; every checkable fact checked. |
| H4 | **No reference to the working documents anywhere shippable** — nothing under `core/` or `tests/`, nor `conftest.py`, `README.md`, `requirements.txt`, `pytest.ini`, `run.bat`, `run.sh`, nor the new reader pages — cites the handoff, STATE, CLAUDE.md, the specs, the plans, or their labels. A provenance tag is deleted; a reason given by reference is rewritten in words. A source-scan test enforces it. The working record itself (STATE, CLAUDE.md, the specs, plans, ledgers and the walkthrough checklist) may still cite code and each other. |
| H5 | **The tool reference** is an ~80-line README summary plus a full flag-by-flag page in the reader documents, kept in sync by a test that walks the tool's parser; the help text is tidied first. |
| H6 | **The retrain's science**, settled now: the tier-1 box, with temperature reported as an assumed input; 10,000 batches × 2,048 rows; a network of 256 hidden features × 10 transforms from the start; chi mode with 6 probes supplied and the slot ceiling kept at 12; a certification narrowing round on a simulated cell after the broad model passes; hard calibration gates, with informativeness measured and recorded as the first baseline. |
| H7 | **The probe checks**: a new command-line diagnostic family, `python -m core probes`, with three modes — `band` (the band and drive hold for a cell), `mask` (why training probes are thrown out, over a prior), `drive` (the free-running and captured drive strengths for a cell). Command line only; each writes an ordinary diagnostic record; it measures and never changes a setting. Kept after the owner heard that the retrain does not strictly need it. |
| H8 | **Measuring is not overriding.** The standing rule that there is no chi override anywhere (piece 2's D11) is read as governing what trains and infers: the probe checks may take their own frequency and drive grids, and nothing the family measures with can change what a training run uses. Every record states the configured band and drive it judged. |
| H9 | **Every readiness gap is fixed in code**, with tests, before the retrain (§5), then a card run of the tier-1 box. |
| H10 | **One piece, code first**: part 1 the cleanup, part 2 the help tidy, part 3 the readiness fixes, part 4 the probe checks; then part 5, the documents, written once the code will not move again. |
| H11 | **The calibration verdict** (revised by the owner on 2026-09-28, after the review showed temperature is not uninformed in chi mode) passes when **every** inferred parameter's rank-uniformity test has p ≥ 0.05 ÷ (the number of inferred parameters) — a 5 % family-wise false-alarm rate — **temperature included**, since its test stays valid and a mis-modelled temperature is a real flaw; and the joint coverage test, over all parameters as computed today, has p ≥ 0.05. Temperature is still labelled an assumed input in every report. A failing verdict is a result, not an error. |
| H12 | **Walkthrough rows F1–F5** cover only what changes in the window (§9). |
| H13 | (2026-09-28) **A calibration and a narrowing round record the acceptances they ran under**, as an inference already does — closing STATE's parked provenance item, including an inference on a narrowed posterior's own observation, which records `accepted: []` today even when `--accept-truncated` loaded it (§5.11). |

### 1.2 How the design reads the decisions

- **Against the decomposition's row 6.** "A user guide per audience" is realised as topic pages with a
  reading path per reader (H3), because the readers' needs overlap heavily and a fact written four
  times drifts. "Per-subsystem design docs" are `architecture.md`, `rules-and-traps.md` and
  `science.md` (§6.2). "The runbook as … a scripted pipeline" is dropped (H1): the retrain is a
  multi-day run the owner watches and gates by hand. The cleanup, the readiness fixes, the probe checks
  and the acceptance records are new scope (H4, H7, H9, H13).
- **D11's letter is narrowed, deliberately.** STATE records D11 as a standing refusal "on every
  surface" that a later piece must not re-open. H8 narrows its letter to what trains and infers; the
  probe checks' grids are measurements. The final STATE update records this reading beside D11 in the
  decisions log, as piece 5 recorded its seed widening.
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
- **Temperature on the tier-1 box.** It enters the simulation only through the derived force scale
  `n · beta · k_B · T / x_scale` (`core/SBI/derived.py:102`). In spontaneous mode it has no effect; in
  chi mode it scales every probe's |χ| (`core/SBI/chi_probes.py:61-63`, `:110-114`), so over its narrow
  prior it may be weakly informed. The chat design said its result would equal its prior and that it
  would be left out of the verdict; the review showed the premise false, and the owner revised H11.
  Temperature stays in the verdict and in the joint coverage test as today, and is marked "assumed
  input" wherever it is reported (§5.3). The informativeness total is a joint log-density ratio
  (`core/SBI/analysis.py:315-327`) and stays joint, with temperature's own entry marked. The card run
  records the corrected tier-1 jacobian's temperature column as a measurement, not a failure.
- **The mask audit reads training through an observer**, never by patching the program at run time
  (§4.4). The archived script monkeypatched two functions; an in-process tool run must not.
- **Seeds.** The probe checks follow the diagnostics' convention: `--seed` defaults to 0 and is
  recorded. `validate` gains a seed through a derived calibration stream (§5.4) — it never starts where
  a training run with the same seed started, `core/rng.py`'s docstring names this one exception, and
  with no seed the stage leaves the streams alone (so `smoke`'s calibration still follows `smoke`'s
  seed). This widens piece 2's recorded reading that "only smoke and the diagnostics take --seed"
  (already widened by piece 5 for `fdt` and `crossval`); the final STATE update records it.
- **A diagnostic taking a prior.** `core/tool/diagnostics.py:3-6` says there is no `--prior` on a
  diagnostic. That stays true of every diagnostic that reads a posterior; `probes mask` has no
  posterior, so the prior it audits is its input and its recorded parent. The docstring is narrowed to
  say so.
- **Reader pages name summary features by their full labels** (`A1_mean`, not A1), so the scan's
  feature-id allowance stays limited to the two statistics files.
- **The review's other owner questions, answered with the recommended default on 2026-09-28:** `sbc`
  records per repeat only the rank-uniformity half of the verdict, labelled as such (it computes no
  joint coverage, §5.4); where "T" means the recording length, log lines and the window's text say
  "T_obs", while labels already written into records (the ablation channel `logT`, the batch-tag
  format) stay (§5.3); `core/config.py`'s network and batch defaults stay (128 × 8, 5,000), because
  every knob is an argument, and the retrain passes H6's values as flags (§6.5); and two additions made
  while writing the first draft stand — `sbc` marks the assumed parameter too, and a simulated
  observation records the force scale it was simulated at.
- **Open engineering items** the map found — the FDT sanity checks holding a view of the whole
  solution, and `decorrelate`'s per-call re-derivation (`core/SBI/decorrelate.py:149-163`) — go to
  STATE's open list, with the handoff's other open engineering items (§6.6). Open **science** ideas go
  to `science.md`'s open-questions section.

### 1.3 Out of scope, and who owns it

- **The reduction map's science** (decomposition §2). Its files still fall under the reference rule.
- **A window front end for the probe checks** (H7).
- **The science items STATE lists as the owner's** (the NaN column in `identifiability jacobian`, the
  fixed probe-layout seed across repeats, `--chi-k 1`, the passive-baseline Ω₀, the Hopf band, and the
  rest of STATE's "Open" list other than H13's). They stay in STATE; the ones that are science are also
  described in `science.md`'s open questions.
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
file under `tests/` (the scanning module included: it assembles its own patterns from pieces, the
needle trick of `tests/test_conditioning_repair.py:912-919`, and stays under its own scan), the prose
files `README.md`, `requirements.txt`, `pytest.ini`, `run.bat` and `run.sh`, and, from part 5 on,
every `*.md` under `docs/guide/`.

**Excluded:** `.gitignore`, the licence files under `core/gui/assets/`, and the working record
(`CLAUDE.md`, `docs/STATE.md`, `docs/superpowers/`, `docs/checklists/`).

**What is read:** in Python files, only `COMMENT`, `STRING` and `FSTRING_MIDDLE` tokens — the prose
tokens `_code_only` discards (`tests/test_conditioning_repair.py:920-922`) — after stripping only the
directive of a `noqa: CODE[, CODE…]` or `type: ignore[…]` comment, keeping any explanation after it
(`core/tool/logging_console.py:13`'s "(spec §1.2)" must stay a hit); test function and class names in
a separate pass; every line of the prose files and reader pages.

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

- id tokens allowed everywhere: `F0`, `D0`, `L2` (physics and norm names; the only `L1` in the tree,
  `tests/test_settings_persistence.py:409`, is a layout-defect label and is rewritten as a hit);
- per file: the summary-feature ids (`A1`…`G7`, `core/SBI/statistics.py:41-59`) in
  `core/SBI/statistics.py` and `core/SBI/summaries.py` only; scipy's `S2` in `core/FDT/spectral.py`;
- shapes: `\bPhase [A-Z]\d\b` (the reduction map's phases); `U\+[0-9A-F]{4}` (Unicode code points,
  `core/gui/assets/icons/build_prism_icons.py:6`, `core/gui/screens/nav_shell.py:18`); `README.md`'s
  Apple-chip line ("M1 / M2 / M3 / M4");
- test-name tokens: `f0` and `d0` (a physics name such as `_f0_` in a probes test name would otherwise
  match `[a-z]\d{1,3}`); and by exact name `test_the_B7_pair_shares_one_flag` (B7 is a feature) and the
  two `test_v1_…` / `test_v2_…` model-file-schema tests in `tests/test_user_models.py` (`:391`, `:404`).

**Any allowlist entry that matches nothing fails the test**, so the list cannot go stale. A published
paper's own section number, a future science term shaped like an id (B0, H1), or any other new
legitimate token gets an entry in the commit that introduces it. Reader pages name summary features by
full label (§1.2).

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
| the one thing a narrowing round must not get wrong | the truncated-prior rule: a narrowing round draws from the prior restricted to the region, never from the posterior |
| guardrail 1 | the observation-digest rule: a round refuses unless the stored observation matches |
| guardrail 2 | the narrowed-model rule: a narrowed posterior is marked as such and never loads or infers as a broad one |
| guardrail 3 | the eigenbasis rule: the region is cut in the rotation's leading directions, flat ones left full width |
| guardrail 4 | the unweighted-draws rule: the region comes from unweighted posterior draws, not best fits |
| guardrail 5 | the generous-region rule: a 99.9 % region, with the truth-outside rate watched |
| guardrail 6 | the cost-on-screen rule |
| guardrail 7 | the region-carries-its-basis rule: the child reuses the parent's rotation, never recomputes it, skips directions loaded on t_scale, and names its base prior |
| guardrail 8 | the calibrate-on-the-region rule |
| C-11 | the training checkpoint (the simulation cache) |
| C-1 … C-10 (the chi series) | the specific measure named in words (e.g. "the per-row lock-in duration", "the per-row probe placement") |
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

**Ids collide** (`recon/inbound-references.md`, "Id collisions"). A label is mapped by its nearby words
— "defect", "trap", "Appendix A", "the whole-piece review's", "row", "piece N" — never by the id alone:

- "the D6 trap" / "flag D6 forbids" / "D6:" (`core/diagnostics/identifiability.py:70`,
  `core/tool/stages.py:186`, `tests/test_tool.py:178`, `tests/test_diagnostics.py:883`) are piece 2's
  D6 and mean that a setting is refused, never silently clamped or ignored — not the transposed
  rotation;
- D1 also means the retired prompt CLI (`core/cli.py:4`, `core/FDT/fdt_pipeline.py:252`,
  `core/gui/panels/fdt_panel.py:7`, `tests/test_conditioning_repair.py:938`), a walkthrough row, and
  the kurtosis feature; D3 is also the bimodality feature; D5 is the diagnostic kind;
- C1/C2 (features, walkthrough rows, cancellation traps, review findings) differ from the chi series
  C-1…C-11 only by the hyphen; M1–M4 are review findings, except `inference_screen.py`'s M1b; X1–X5
  are both trap ids and fix ids; P1–P9, S1–S5 and F4…F48 are rulings or review ids.

Each cleanup task's review reads every rewritten sentence at these ids against its context.

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
and that no label is present). Each new assertion pins a distinctive phrase of the plain sentence. An
absence check uses a regex such as `\([A-Z]\d+\)` or an assembled needle, never a literal like "(D9)",
which the scan would flag.

**Phrases other tests already pin, which each rewrite keeps verbatim:**
`tests/test_artifact_store.py:1917` "must name the observation" (message 1),
`tests/test_conditioning_repair.py:579` "DIFFERENT V" (message 5), and `:601` "measured WITHOUT a
Fisher rotation, but the training bijection has one" (message 4).

**Test assertion messages** that carry labels (71 string lines in tests, e.g. `tests/test_tool.py:3201`
"B6: no force in either front end", `tests/conftest.py:51` "since piece 3", which a user sees as a
session error) are rewritten like any other hit.

### 2.5 Wrong facts fixed on the way

Found by the reconnaissance (`recon/handoff-map.md` chunk notes; `recon/design-*.md`); each is
rewritten to what is true, in the task that owns the file:

- `core/config.py:455-467` still states the overturned conclusion ("… SYSTEMATIC, not statistical … Neither a stronger drive nor a longer recording can recover those probes"); it is rewritten to the corrected account (the in-band failures were a lock-in cycle limit, recovered by the cycle ceiling; the high edge 0.3 stands), which `science.md` carries in full.
- ".rot.pt sidecar" / "sidecar" at `core/config.py:407`, `:493-494`, `:497`, `:538`, `core/sim_config.py:77`, `core/Helpers/model_store.py:38`, `core/Helpers/file_manager.py:312`: it is the manifest now (`core/artifacts/store.py:1300-1304`).
- `core/config.py:499` "conditioning width 42 + 72 = 114", and `core/config.py:291` ("chi width 114 (+13 latent targets), i.e. the retrain's shape"): it is 49 summary columns (`SUMMARY_WIDTH` = 41 features + 8 flags, `core/SBI/statistics.py:539`) + 1 log T_obs + 72 chi = 122.
- `core/config.py:410`, `:419` cite `posterior_07012026` and `prior_forcing_no_forcing.pt` as present; `core/SBI/embedded_network.py:84`, `core/gui/panels/inference/tsnpe_tab.py:32-33` and `core/SBI/truncate.py:22` name the deleted `posterior_08232026` as a live baseline.
- `core/SBI/truncate.py:21-24` "k in particular is FLAT … loading -1.00*k": withdrawn (the transposed reading).
- `core/SBI/statistics.py:509`/`:514`: a substitution-rate line that corrects itself five lines later.
- `core/SBI/train.py:38` "script-only": `chi_k_fixed` is the `sbc` diagnostic's.
- `core/gui/panels/inference/config_tab.py:136` "the only field on this tab" not persisted: since piece 3 the chi boxes are not persisted either.
- `core/config.py:48`, `core/SBI/pipeline.py:1849` "the scripts"; `core/tool/config_args.py:94` names the deleted `_common.script_cfg`.
- `core/SBI/chi_probes.py:3-7` names the mask-audit script as a patching consumer (the observer of §4.4 replaces it); `core/SBI/chi.py:271-274` says the duration ceiling is keyed on the fastest row (it is per row, `chi_probes.py:174-181`).
- `core/artifacts/identity.py:2-3` says the clean break retires `orchestrator.training_identity`, which survives as a delegate (`core/orchestrator.py:368`).
- `core/gui/panels/inference/rows.py:34` cites `orchestrator.build_experiment_obs_chi`, now in `core/SBI/observations.py`.
- `tests/test_figures.py:15`, `test_nav_and_gating.py:15`, `test_settings_persistence.py:15`, `test_simulate.py:15` ("pytest tests/test_gui_progress.py"), `tests/test_user_sbi.py:1023`, `test_conditioning_repair.py:1041`: that file no longer exists.
- `tests/test_user_sbi.py:2233`: the docstring says the truncation key is "OMITTED, never None" while the test asserts None.
- `tests/test_user_sbi.py:2062`, `:2253` cite archived scripts at old commits.
- `core/gui/panels/inference/help_text.py:85-86` (the early-stopping patience tooltip, `HELP["flow_patience"]`): "The 2026-08-25 run stopped at 130" — kept as a measurement (stopped at epoch 130 on a patience of 20, best at 110), without the incident label (§2.6).

### 2.6 Dates

- A date that dates a measurement stays (`core/config.py:13` "MEASURED 2026-08-28";
  `help_text.py:48` "measured 2026-08-27: 21.67 GiB completed on a 15.92 GiB card").
- An incident used as a label in text a user sees is reworded to state the fact:
  `core/SBI/truncate.py:195`, `:205` ("the 2026-09-02 round kept 0.01%…"), `core/SBI/run_guards.py:158`,
  `help_text.py:85-86`.
- Three data literals in `tests/test_tool.py` (`:2913`, `:2930`, `:2940`) are test data, not labels.

### 2.7 Proving that only comments changed

Before editing, each cleanup task lists its **permitted non-docstring changes**, by file and quoted
text:

- (a) the §2.4 messages whose files it owns;
- (b) every non-docstring string constant on its own hit list — test assertion and failure messages
  (e.g. `tests/test_tool.py:3201` "B6: no force in either front end", `tests/conftest.py:51` "since
  piece 3") — plus the user-visible strings §2.5 and §2.6 reword (`help_text.py`'s
  `HELP["flow_patience"]`);
- (c) task 1 only: the two test renames of §2.3;
- (d) the assertions of the three pinning tests of §2.4, whose rewrite may add an assertion.

For every Python file it edits, the controller then runs a check **outside the repository** (in the
session's scratch directory; never under `.superpowers/`): tokenize both versions, drop `COMMENT` and
`NL` tokens, parse, remove docstrings (the first-statement string of a module, class or function), and
walk the two trees **node by node**, printing every difference with its `file:line`. A task is done only
when every printed difference is on its list.

**Tests that read raw source text** (`inspect.getsource`, `code_only`, the source scans) must not see
a rewrite add or remove the words they look for. Known today (`recon/design-scanrule.md` §5 "Hazard",
extended by the review): `tests/test_diagnostics.py:154` ("SystemExit", "raise ValueError");
`tests/test_nav_and_gating.py:4659-4683` ("the five inference tabs" must stay absent; "Ten of these
exist", "9 of the 10", "all ten differ" must stay present in `base_panel.py`, "the six inference tabs"
in `inference/base.py`, and "_persist_layout" and "1500" in `main_window.py`, `:4666-4669`,
`:4683-4685`); `tests/test_tool.py:2094` ("no bounds file"); `tests/test_user_sbi.py:3241` (`sdeint.py`
must contain "_GRAPH_CACHE" and must not contain "_SOLVER_SINGLETON" or "_SOLVER = Solver()") and
`:3551` ("SIM_VRAM_CEILING"). Each task lists every other `getsource` scan over its files before
editing. And no file is edited while a suite runs (CLAUDE.md).

### 2.8 The scan test, and its shrinking list

A new suite, `tests/test_source_hygiene.py`, holds the scan. It reuses `CODE_ROOTS`/`CODE_FILES` and
the directory-closure test (`tests/test_artifact_store.py:1866-1896`), the "scanned ≥ N" floor and
missing-file check (`:1845-1860`), the matcher self-check on snippets (`tests/test_refusals.py:856-872`),
and the "assemble the needle so the file cannot match itself" trick
(`tests/test_conditioning_repair.py:912-919`). A failure prints `path:line family 'match'`.

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
files (`requirements.txt:9`, "See PRISM_HANDOFF.md section 1.2", and `:44` are rewritten here to state
their facts); (7)–(9) the tests in three batches, the last removing the list. Each message of §2.4 is
changed in the task that owns its file, with its pinning test.

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

The 38 source sites (49 rendered actions) that show an internal destination name
(`NUM_POSTERIOR_SAMPLES`, `RUN_SIZE_CAP`, `MAX_NUM_EPOCHS`, `T_OBS_S`, `STORE_ROOT`, `ENSEMBLE_M`,
`CHI_K_FIXED`, …; `recon/design-helptidy.md` §2) switch to the house set the stage commands already
use — `N` a count, `X` a real number, `S` seconds, `REF` a name or id, `PATH`, `NAME`, `K` the probe
count, `STAGE[,STAGE...]`. `--f0-si N` becomes `X` ("in newtons"); `--level Q`, `--prefactor VALUE`,
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

Each optional value flag is in one of **three classes**, read from one table in `core/tool` that the
§3.8 and §6.4 tests also read:

- **(i) a value default** — the help ends with `(default X)`, in the exact form of
  `core.refusals._default_clause` (`core/refusals.py:214-217`), which becomes public so the help and
  the refusal lines print the same string. X comes from, **in order**:
  1. the parser's own default when it is neither None nor `''` (`%(default)s`: smoke's literals,
     `--device`, `--preset`);
  2. a **per-(subcommand, flag) table of pinned literals**, each pinned by the new test against its
     owning signature or config constant: sbc's `--seed` (0), `--n-cal` and `--posterior-samples`;
     laplace/jacobian's `--seed` (0), `--min-valid`, `--sd-identified`, `--zero-tol` and
     `--noise-eps`; smoke `--t-obs` (`config.T_MIN_EXP_S`); `--chi` (`config.CHI_MODE`, printed
     "--no-chi"); crossval `--n-freqs`/`--ensemble-m` (the preset's values from `cli.SWEEP_PRESETS`:
     exploratory 30 / production 60); the probe checks' literals (§4.1a);
  3. the `core.refusals.FIELDS` default through the reverse of `core/tool/fields.py` `FLAG` (already
     pinned against `config.py` and the stage signatures by `tests/test_refusals.py:462`).

  The table outranks FIELDS because `FIELDS["seed"]` and `FIELDS["t_obs"]` carry FDT's and a required
  flag's wording ("none: one is drawn and recorded", "none: it must be given"). `--help` never imports
  `core.config`.
- **(ii) a behaviour default that is not a value** — `(default X)` with X a phrase from that table, as
  FIELDS already does: `--model` ("the bounds file's parent folder, upper-cased"), smoke `--store-root`
  ("a fresh temporary directory, left on disk"), fdt/crossval `--store-root` ("the artifacts root"),
  sbc `--chi-k-fixed` ("pooled over the training mixture"), and the probe checks' config-derived grids
  (§4.1a).
- **(iii) no default clause** — required flags; flags that only name an input (`--bounds`, `--prior`,
  `--posterior`, `--observation`, `--cell`, `--spont`, `--forced`, `--drive`, `--f0-si`, `--record`,
  `--s-grid`, `--t-grid`, `--at`, `--prefactor`, `--out`, `artifacts note --note`); and
  `--name`/`--note`, whose help says "unnamed" / "no note" (their parser default is `''`).

Also:

- The epoch cap's `2147483647` reads "no ceiling": `FIELDS["max_num_epochs"].default` becomes
  "no ceiling", with a named exception in `tests/test_refusals.py:493-494`, so the help and the refusal
  line print the same clause.
- `FIELDS["min_valid"]` gains its real default, 0.5 (`core/diagnostics/identifiability.py:359`), and
  joins the signature pin.
- `(default: config.CHI_MODE)`, `config.CHI_N_FREQS`, `config.cpu_device()`, `config.T_MIN_EXP_S`, the
  "(stage default N)" literals (15 sites in `diagnostics.py`) and smoke's "Default: …" all move to the
  one form (`recon/design-helptidy.md` §4 lists every site).

### 3.6 Epilogs

(`recon/design-helptidy.md` §5.) The environment variables are stated once, in the top-level epilog
(`core/tool/__init__.py:25-36`, pinned by `tests/test_tool.py:125`); smoke's copy (`smoke.py:48-53`)
is dropped, with its repetition of the `--run-size` explanation, and its "What to watch" paragraph
shrinks to one line pointing at the reference page. "Like every subcommand but smoke" goes from the
fdt/crossval `--store-root` help (`core/tool/fdt.py:140-141`) and the artifacts epilog (the top-level
epilog says it once). The compare epilog stops repeating `_COMPARE_MODES` (`fdt.py:267-270`), whose
renormalise line gains "drawn against the original". The artifacts epilog stops repeating the mode
helps and keeps only the facts pinned by `tests/test_tool.py:2833` ("all eight", "<kind> is one of
…"), the no-configuration / no-`--store-root` consequence, the `<kind>`/`<ref>` definitions and the
exit-0 rule; its hand-typed "200 characters" reads `NOTE_MAX_CHARS`. The fdt epilog drops the clauses
that restate its flags; the crossval epilog keeps its grid rules (pinned by `:1756`) and moves the
preset-default sentence into the two flags' help, with the numbers (§3.5). Every epilog is re-wrapped
at 79 columns.

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
  recursive `_walk` at `tests/test_refusals.py:936` is the model). Each has a non-empty help, a
  house-set metavar or `choices`, and the clause its class requires (§3.5): for class (i), exactly one
  `(default X)` whose value **equals the source the precedence selects for that flag** — the test
  computes the source; it does not accept any of the three; for class (ii), the table's phrase; for
  class (iii), no clause. Every leaf has a `description`.
- **Updated**: the help-text pins whose wording changes (`recon/design-helptidy.md` §6 lists them:
  `tests/test_tool.py:125`, `:324`, `:1519`, `:1667`, `:1756`, `:1766`, `:1882`, `:2833`, `:3101`,
  `:3406`, `:3192`, `:3592`) keep pinning the same facts in the new words.
- The torch-free help probes (`tests/test_tool.py:1913-1920`, `:3125-3132`) stay as they are.

About two or three tasks, before parts 3 and 4, so the new family is built to these conventions.

---

## 4. Part 4 — the probe checks (`python -m core probes`)

Part 4 in execution order (H10): it runs after part 3 (§5), and its first tasks build the geometry
helper of §4.6 and the observer of §4.4 on the loop-level commit seam of §5.9.

### 4.1 Shape

A new diagnostic family modelled on the three-mode `identifiability` (`recon/design-diagframe.md` §1):

- **Stage module** `core/diagnostics/probes.py` with three public entries (function names differ
  from the module name, `core/diagnostics/__init__.py:14`; e.g. `probe_band`, `probe_mask`,
  `probe_drive`), each `@public_entry`, re-exported from `core/diagnostics/__init__.py`, added to the
  exact public-entry set (`tests/test_artifact_store.py:3410`), with a config-untouched leg each
  (`:3191-3250`) and entries in `_UNTOUCHED_LEGS` (`:3361`, equality at `:3448`) and
  `_REFUSAL_FIELDS` (`:3388`).
- **Tool module** `core/tool/probes.py`, registered in the loop at `core/tool/__init__.py:46`, modes
  as `add_subparsers(dest="variant", required=True)`, with its own `interrupt_note` (a diagnostic
  keeps nothing on Ctrl-C; the generic `--resume require` advice would be wrong).
- **Record**: the existing `diagnostic` kind; body `{"diagnostic": "probes", "variant":
  "band"|"mask"|"drive", "settings", "results"}` (`core/artifacts/manifest.py:44`); the numbers in an
  `.npz` payload, figures through `w.fig_sink`, the report as `log.info` records so it reaches the
  console and `log.txt`; every float through `orchestrator._num` (non-finite becomes null). The
  comment calling a diagnostic "a measurement ABOUT other artifacts" (`manifest.py:36`) is widened:
  `band` and `drive` measure a cell and have no parent, as `identifiability jacobian` already does.
- **Common flags**: `--bounds` (required), `--device`, `--seed` (default 0, recorded, run inside
  `core.rng.seeded(seed, device)`; its own field key with registered default "0", validated by the
  rule of `core.rng.require_seed`), `--name`, `--note`. Every knob is a keyword argument of its stage
  with its default in the signature only; the tool forwards only what is given
  (`core/tool/config_args.py:180-186`).
- **Refusals** before anything is simulated, each a `Refusal` with a registered field key, in the
  stage order the other diagnostics keep: resolve the store, `assert_name_free("diagnostic", name)`,
  the rules, anything else that can refuse, and only then `store.create`. Every new key is added to
  `core.refusals.FIELDS` (with the registry count, `TOOL_ONLY_KEYS` and `owned_by_a_signature` in
  `tests/test_refusals.py`, `:56-57`, `:107`, `:499` with its closing assertion at `:562-563`),
  `core/tool/fields.py` `FLAG`, `core/gui/fields.py` `CONTROL` as None, and the pinned no-control set
  (`tests/test_nav_and_gating.py:1607-1612`).

### 4.1a Flags, keys and refusals per mode

Constraints, from the review (`spec-review.md` MF7):

1. `band` and `mask` build their config with chi mode **on** and declare neither `--chi` nor
   `--no-chi`; `drive` builds it **off** and declares no chi flags. No `chi_mode` key is added.
2. Grid flags get new flag names and new keys — never `--chi-f0`, `--chi-band` or `--f0`, never the
   keys `chi_f0`, `chi_freq_bounds` or `f0`, which keep their `FLAG`/`CONTROL` entries (None, None and
   FDT's `--f0`; `tests/test_refusals.py:953-954`; the one-flow design has "no `--chi-f0` and no
   `--chi-band`").
3. Every knob whose default differs from an existing key's registered default gets its own key
   (FIELDS gives `num_runs` "5000", `run_size_cap` "0", `repeats` "10", `m` "32", `t_obs` "none: it must
   be given", `seed` "none: one is drawn and recorded"). `FLAG` may map several keys to one flag, as it
   does today for `--forced` and `--drive`.

| mode | flag | key (new unless noted) | default (class, §3.5) | pre-spend rule |
|---|---|---|---|---|
| all | `--seed N` | `probe_seed` | 0 (i) | `require_seed`'s rule |
| band | `--cell PATH` | `cell` (existing) | — (iii) | required; `load_and_validate_gt` |
| band | `--lengths S [S ...]` | `probe_lengths` | five log-spaced from the shortest expected recording to this cell's training ceiling (ii) | each > 0 and giving at least one sample; beyond the ceiling is flagged, not refused |
| band | `--multipliers X [X ...]` | `probe_multipliers` | four across the configured band plus one control at twice its top (ii) | each finite and > 0 |
| band | `--drives X [X ...]` | `probe_drives` | the configured chi drive (ii) | each finite and > 0 |
| band | `--repeats N` | `band_repeats` | 24 (i) | ≥ 2 |
| band | `--cycle-caps N [N ...]` | `probe_cycle_caps` | the configured cycle ceiling only (ii) | each > 0 |
| band | `--cv-max X`, `--snr-min X`, `--sup-min X` | `cv_max`, `snr_min`, `sup_min` | 0.20, 3.0, 0.50 (i) | finite; `sup_min` in (0, 1] |
| band | `--phase-max X` | `phase_max` | not judged (ii) | finite when given |
| band | `--peak-window X` | `band_peak_window` | 0.10 (i) | in (0, 1) |
| mask | `--prior REF` | `prior` (existing) | — (iii) | required; `store.load_prior` refusals; never built |
| mask | `--num-runs N` | `mask_num_runs` | 12 (i) | `require_at_least` 1 |
| mask | `--run-size N` | `mask_run_size` | 32 (i) | `require_at_least` 1 |
| mask | `--chi-k-fixed K` | `chi_k_fixed` (existing) | pooled over the training mixture (ii) | 2 ≤ K ≤ the slot ceiling, refused up front (not the mid-run `ValueError` of `core/SBI/pipeline.py:1342-1348`) |
| drive | `--cell PATH` | `cell` (existing) | — (iii) | required; the box must have a Forcing section (`feature_sets.assert_forced`) |
| drive | `--t-obs S` | `drive_t_obs` | 5.0 (i) | > 0 and giving at least one sample |
| drive | `--repeats N` | `drive_repeats` | 16 (i) | ≥ 2 |
| drive | `--detune X` | `drive_detune` | 1.4 (i) | > 0 and \|detune − 1\| greater than the own-peak window |
| drive | `--strengths X [X ...]` | `drive_strengths` | the archived grid plus the configured chi drive (ii) | non-empty, finite, > 0, increasing |
| drive | `--free-min X`, `--captured-max X` | `free_min`, `captured_max` | 0.70, 0.10 (i) | 0 < captured-max < free-min ≤ 1 |
| drive | `--peak-window X` | `drive_peak_window` | 0.02 (i) | in (0, 1) |
| drive | `--clarity-min X` | `clarity_min` | 3.0 (i) | > 0 |

The plan may merge keys whose defaults and meanings coincide; it may not share a key whose registered
default would then be false for one of its flags.

### 4.2 Rules for all three modes

- **It measures only.** Probe grids reach the simulator only as `gen_chi_raw`'s `f0_nd`/`multipliers`
  arguments (or the drive builder's amplitude), inside `core/diagnostics/probes.py`. They never go
  through a `SimConfig` field (`cli.make_sim_config`, `dataclasses.replace`) or through
  `gen_training_data`'s `chi_f0`/`chi_freq_bounds` keywords (`core/SBI/pipeline.py:1203-1204`); `mask`
  runs `gen_training_data` at `config.py`'s band and drive, with
  `run_guards._assert_chi_config_is_deliberate` run on its config. A test pins both, plus that the
  family's modules assign no upper-case config name and that a training run's `SimConfig` is unchanged
  after a probes run in the same process. Every record's `settings` states the configured `CHI_F0`,
  `CHI_FREQ_BOUNDS`, `CHI_K_PAD`, `CHI_MIN_CYCLES`, `CHI_MAX_CYCLES` and probe count K it judged (the
  manifest's `config` nulls the chi constants outside chi mode and never records K,
  `manifest.py:279-282`). Verdicts read "the configured … holds / does not hold for this cell"; a
  suggestion names `core/config.py` as the place a deliberate change is made.
- **Faithful to training today.** The modes use the production estimators: `chi.peak_freq` for each
  sample's own Ω₀, `chi.lock_in_batched` with `n_samples` for the per-row cycle ceiling (production
  caps per row and rounds down, `core/SBI/chi_probes.py:176-192`; the archived band script used one
  prefix per batch and rounded up), and a probe at or above 0.9 × Nyquist is **masked, not clamped**
  (`chi_probes.py:164-169`; the archived script clamped). Spectra are kept in float64. `x_offset` is
  applied when the box has it (`core/orchestrator.py:181-182`). A drive's force is built with a literal
  forcing index, never `cfg.forcing_idx["amp"]`, which raises on a box with no Forcing section
  (`core/sim_config.py:366-368`). The drive phase used (π/2) is recorded.
- **Tier-1 aware.** Each simulation derives the force scale exactly as training does —
  `derived.to_sim_rescale(…, *cfg.tier1_args)` with `cfg.sim_rescale_idx` (`core/orchestrator.py:130-137`).
- **Grids are measurement, not override** (H8, §1.2).

### 4.3 `band` — do the configured band and drive hold for this cell?

**Inputs:** `--bounds`, `--cell` (built with `config_args.build_cfg(load_gt=True)`), chi mode on.

**Defaults** (`recon/design-f0sweep.md` §3; §4.1a): five recording lengths log-spaced from
`T_MIN_EXP_S` (1 s) to the training ceiling for this cell (26.909 s for the master cell at t_scale
3.73), rounded down to 0.01 s; four frequencies log-spaced across `CHI_FREQ_BOUNDS` plus one control at
twice its top edge (`[0.03, 0.0646, 0.1392, 0.3, 0.6]` today); the configured drive `CHI_F0`; 24 noise
repeats; the configured cycle ceiling, with an optional list of caps that adds the wall report. Cost:
lengths × (1 + frequencies × drives) ensembles — 30 at the defaults.

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

**Inputs:** `--bounds`, chi mode on, `--prior REF` (required) — the first diagnostic to take a prior
(§1.2): loaded with `store.load_prior`, which refuses a model, order, box or log-mask mismatch, and
checked by `run_guards._assert_chi_config_is_deliberate` (which `build_prior`'s load branch runs and
`store.load_prior` does not); recorded as the parent with the prior's fingerprint. A missing prior is
refused, never built. No cell: training is truth-free, and the initial conditions come from
`orchestrator._observation_inits(cfg)`. `gen_training_data` is called with `checkpoint=None`, so no
simulation record is written (a test asserts the store's simulation kind is unchanged after a run).

**The observer.** `pipeline.gen_training_data` gains one keyword-only argument, an optional probe
observer (default None), threaded to `chi_probes.gen_chi_block`. The hook sits **inside
`gen_chi_block` after `pack_probe_block`**, and is popped from the keywords before they reach
`gen_chi_raw`. Each call receives the batch tag and the call's row range `(lo, hi)`; each row's Ω₀
(`chi.peak_freq(x_spont_dim, dt_exp)`, deterministic and drawing no random numbers — or threaded out
of `gen_chi_raw`); the duration fraction; K; `gen_chi_raw`'s returned `(u, logcyc, valid)`; and the
packed mask. The non-finite and Nyquist predicates are recomputed from `f_peak · exp(u)` and `dt_exp`,
because `gen_chi_raw` returns only `valid`. The observer **buffers per row range**; the batch loop
commits a batch's buffer only after its rows are stored (after `x_buf[_lo:_hi] = _rows_out`,
`core/SBI/pipeline.py:1718`) and discards an abandoned attempt's buffer — an out-of-memory retry
re-runs a batch, and row halving (`_rows_with_oom_retry`, `pipeline.py:806-813`) splits one batch into
several row ranges under one tag. Forced and spontaneous modes never call it. With the default None
nothing changes: a test pins that seeded `gen_training_data` output is byte-identical with no observer
and with a no-op observer, in chi, forced and spontaneous modes.

**Defaults** (§4.1a): 12 batches × 32 rows through the real generator (the Sobol schedule, the
per-batch probe count, placement, the duration draw and the per-row subsetting are production's);
optional `--chi-k-fixed` audits one probe count; `--seed` changes the noise, and the record says that
the probe layout comes from training's fixed probe-generator seed (`pipeline.py:1353-1354`), so it does
not change with `--seed`.

**Reports:** the share of probes thrown out by each cause — the cycle floor (`~valid & ~bad & ~nyq`),
split into "too slow even at the band's top at full length" (`f_peak · T_obs · hi < CHI_MIN_CYCLES`)
and "shortened by the duration draw"; non-finite lock-ins and the band filter at the packer
(`chi.py:548-551`); the per-batch masked fraction with its spread (the effective sample size is the
batch count); Ω₀ quantiles in cell units and Hz; rows left with zero or one usable probe; the span of
the frequencies actually driven (`f_peak · exp(u)`, not the drawn multipliers — placement lifts them).
Causes that cannot fire in training (non-finite frequency, Nyquist, the band —
`recon/design-maskaudit.md` §5) are reported as invariant checks. The audit also captures
`gen_chi_block`'s own masked warning and compares counts **per row range** (a halved batch warns once
per half); a mismatch is recorded and warned, never silently accepted.

### 4.5 `drive` — how hard can a lab drive this cell?

**Inputs:** `--bounds` with a Forcing section, `--cell`; built as not chi (§4.1a).

**Defaults** (`recon/design-drive.md` §1; §4.1a): recording length 5 s, 16 repeats, drive at 1.4 × the
cell's own Ω₀ (detuned so the window does not see the drive), drive strengths
`[0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0]` in model
units plus the configured `CHI_F0`, measured explicitly; own-peak window ±2 % of Ω₀ (the archived ±2
bins at 5 s, about ±1.8 %); free-running when at least 70 % of the undriven own-peak power remains,
captured at 10 % or less, otherwise in between.

**Reports:** Ω₀ in cell units and Hz, the per-trace peak spread, the peak's clarity (peak ÷ median
power), peak-to-peak and cycles in the recording; per strength the median |χ|, the median phase-locking
value (reported, not judged) and the own-peak ratio with its verdict; **the strongest free-running
strength** — the largest in the leading unbroken run of free-running strengths, which deliberately
replaces the archived "strongest free-running below the first captured", so a strength whose own-peak
bin refills at very large drives can never be picked — and **the weakest captured strength** (null,
with a warning, when nothing in the grid is captured), each in model units and in the cell's own force
units (the derived force scale on the tier-1 box); the `CHI_F0` row, with a note that its thresholds
depend on the detune (so this mode cannot certify the chi drive — that is `band`'s job); and suggested
Forcing lines and box for a cell file, **logged at info and recorded, never written** (nothing outside
`core/config.py` builds a `Resources/` path). "Free-running versus captured" is the criterion; there
is no linearity test and the record says so.

**Results, not refusals** (known only after the spend): no clear oscillation (clarity below
`--clarity-min`, then every verdict null with a warning), nothing captured within the grid, a drive at
or above 0.9 × Nyquist.

**Parity first.** The task that builds `drive` runs it on the card on the master cell (`master.txt` +
`master_spont.txt`) at the archived window (about ±1.8 %) and must reproduce the archived readings —
0.02 free-running and 0.2 captured — before the 0.02 default window is set.

### 4.6 Shared helpers

- **Recording geometry and the training recording-length ceiling.** **No production number may
  change.** The production callers stay as they are; the probe checks use one helper returning
  `n_obs`, `n_fine`, the subsample and the ceiling for a config and a length. A test pins that it
  agrees with the batch loop's formula (`N_points_k = int(T_nd_k / dt_nd_k)`, `core/SBI/pipeline.py:1482-1484`)
  on the master cell, and that the ceiling it returns is the one training enforces — the Sobol
  pre-filter's bound, `n_fine ≤ min(N_ND_MAX, len(t))` (`pipeline.py:1159-1168`); the two production
  copies (`core/orchestrator.py:141-172` and the pre-filter) differ in float32 rounding on a handful of
  candidates, so they are not merged. If a later change does touch a production caller, a golden digest
  of seeded `gen_training_data` output in chi, forced and spontaneous modes, recorded before the
  change, is asserted after it.
- **The own-peak ratio**: one function (forced ensemble, undriven reference, Ω₀, window fraction) used
  by `band` and `drive`, each passing its own thresholds.

### 4.7 Tests

- Fast, with a stand-in simulator of known response (the pattern of
  `tests/test_diagnostics.py:703-723`): each mode's verdict logic, the record shape, the refusals
  before the writer opens, non-finite values becoming null, the harmonic flag, the leading-run rule
  for the free-running strength, the observer's per-row-range bookkeeping, its commit and discard, and
  its cross-check against the production warning.
- Deterministic: the master cell's training ceiling (26.909 s) and the default grids.
- The byte-identical seeded-training test (§4.4).
- `tests/test_tool.py`: every flag reaches its stage as a keyword (the forwarding pattern of
  `:659`, `:711`, `:826`), `probes` and each mode's `--help` import no torch, modes do not share flags
  they should not.
- The measure-only pins (§4.2); the no-simulation-record pin (§4.4).
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

## 5. Part 3 — retrain readiness

Part 3 in execution order (H10): after the help tidy, before the probe checks. Each fix comes with a
test that fails before it and passes after.

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
box handed its inferred index. Both builders **raise `RuntimeError`**, naming the index and the tier-1
relation, when the index names `T` and not `f_scale` — a programming error, so both front ends show it
as a bug with a traceback, never as a yellow-box refusal. Every caller is listed by the task and
checked: `identifiability laplace` and `jacobian` then derive the force scale as training does
(`core/diagnostics/identifiability.py:248-249`, `:525-531`, `:543-544`, with the rescale vector built at
`:860`); the Simulate panel's live runner (`core/gui/panels/simulate_runner.py:114`, `:220-234`), which
passes `cfg.rescale_idx` raw and can meet a tier-1 box only through a user-added sibling bounds file
with a Forcing section, derives it too (`to_sim_rescale` with `sim_rescale_idx`); the new probe checks
follow §4.2. Tests: each diagnostic on a tier-1 box, before the fix returning a wrong number silently,
after it matching the value computed with the derived force scale; a builder handed a `T`-without-
`f_scale` index raises `RuntimeError`.

### 5.3 Temperature as an assumed input

On the tier-1 box temperature enters the simulation only through the derived force scale; in chi mode
it scales every probe's |χ|, so it may be weakly informed; by the owner's decision it is reported as an
assumed input and **stays in the verdict** (H6, H11, §1.2).

- `SimConfig` gains `assumed_params`: parameter **keys** (never the LaTeX `inferred_labels`,
  `core/sim_config.py:318-331`), `("T",)` when tier 1 is on and `()` otherwise, derived from the box,
  never typed; consumers match by index through `list(params_dict) + list(rescale_params)`.
- The assumed parameters are recorded in each record's `config` (§5.7 has the block).
- **Inference:** the corner plot labels temperature "T (assumed input, K)" (the bare `$T$` of
  `core/Helpers/labels.py:17-20` gains its unit and the mark); `posterior_summary`
  (`core/orchestrator.py:2161`) marks the entry `assumed: true`.
- **Calibration:** the rank-uniformity table (`core/orchestrator.py:1811-1815`) marks an assumed
  parameter "(assumed input)"; it stays in the verdict (§5.4) and in the joint coverage test, which runs
  as today (`:1849-1855`); the informativeness total stays joint and the assumed parameter's own entropy
  entry is marked.
- `sbc` marks assumed parameters the same way.
- **Wording.** Where "T" means the recording length, log lines (`(t_scale, T)`,
  `core/SBI/pipeline.py:1491-1493`) and the window's text (`core/gui/fields.py:70`, `help_text.py:88`,
  `:94`, `core/gui/panels/inference/base.py:155`) say `T_obs`, since on the tier-1 box `T` is also
  temperature. Labels already written into records stay: the ablation channel `logT`
  (`core/diagnostics/ablation.py:120`, pinned by `tests/test_diagnostics.py:1226`) and the batch-tag
  format.

### 5.4 A calibration verdict, and a repeatable calibration set

- `validate_calibration` ends with one verdict record, also written to `body.results["verdict"]`:
  **PASS** when every inferred parameter's rank-uniformity KS p ≥ 0.05 ÷ (the number of inferred
  parameters), assumed ones included, and the joint coverage test has KS p ≥ 0.05 (H11). The record
  lists each parameter's p against its threshold, marks the assumed ones, and carries one caveat:
  t_scale's test rests on the calibration's operating points (`CAL_N_SCALES`, 200), not on its 2,000
  datasets, so it has less power. A FAIL is a result: the stage completes, writes its record and exits 0;
  the runbook reads the verdict.
- **A repeatable calibration set.** `validate_calibration` gains `seed=None`. A given seed runs the
  calibration-set draw inside `core.rng.seeded`, from a stream derived from the seed and a fixed
  calibration tag (e.g. numpy `SeedSequence([seed, tag])`), so `validate --seed S` never starts where a
  training run seeded with S started — the replay `core/rng.py:12-18` exists to prevent; its docstring
  is updated to name this one exception and why it is safe. `seed=None` leaves the streams alone, so
  `smoke`'s calibration still follows `smoke`'s single seed. The tool's `validate` draws one with
  `random.randrange` when none is given and passes it; the Validate tab draws one the same way (it gains
  no Seed box and remembers nothing new). The seed is recorded in `body.results`. The exact-keyword pin
  for validate in `tests/test_tool.py` gains a `--seed` leg.
- `sbc` records, per repeat, whether the **rank-uniformity half** of the rule passes (every inferred
  parameter's KS p ≥ 0.05 ÷ their number), labelled as such, and the fraction of repeats that pass. The
  joint coverage half is `validate`'s (`core/diagnostics/sbc.py:156-170` computes no joint coverage).
- The Validate tab shows the verdict line in its log pane (it arrives as an info record) — walkthrough
  row F1.

### 5.5 The eigenvalues survive a resume

The training checkpoint's header stores the rotation's eigenvalues beside the rotation
(`core/SBI/training_checkpoint.py:175-186`, beside `"V"`; readers use `.get`, so an older header reads
as unknown). A resumed run writes them into the posterior's `body.transform.fisher_eigenvalues`; a
narrowing child takes its parent posterior's (`body.transform.fisher_eigenvalues` of the parent) since
it reuses that rotation; a parent with no rotation gives None. **The witness that the Fisher ran stays
the freshly computed value**, a separate variable (`core/orchestrator.py:1364-1376`, "fisher_evals is
its one witness"), so `fisher_m`/`dz`/`points` stay None on a resume and on a truncated round — V7
unchanged. Tests: a run resumed from a complete cache carries the eigenvalues the first run wrote, with
the Fisher settings still None; `tests/test_artifact_store.py`'s V7 legs change accordingly (§7).

### 5.6 The narrowing round reports truth inside as well as outside

With a known cell loaded, the round reports, per truncated direction, whether the truth lies inside the
region — today it warns only when outside (`core/orchestrator.py:1215-1232`) — and records the
per-direction containment in the child's `body.training`. Test: a truth inside prints and records
"inside" for every direction; one outside keeps today's warning.

### 5.7 The constraint recorded

Body keys are a closed set per kind (`core/artifacts/manifest.py:28-35`, checked at `:155-157`), so each
new fact goes into an existing key:

- **`config`** of every record whose config has tier 1 on: the relation
  `f_scale = N · beta · k_B · T / x_scale`, `k_B` in the cell's units (`k_b_cell`), the `T` range, and
  the assumed parameters (`config_from_cfg`, `manifest.py:247`, gains the block; absent otherwise). A
  simulated observation's `config` also records the force scale it was actually simulated at.
- **The simulation cache's identity** gains the units file's fingerprint, because the derived force
  scale depends on it (`core/artifacts/identity.py`, `FORMAT = "training-rows/2"` becomes `/3`). The
  fingerprint is read from the config's sources, and `from_cfg` keeps failing open (None) for a stub
  config with no sources (`identity.py:28-31`). The golden digest at `tests/test_user_sbi.py:2256`
  ("1912d2139359") and the key-set assertion at `:2258-2263` are recomputed in the same commit. No cache
  is stranded: none exists.

### 5.8 `smoke` takes the network size

`smoke` gains `--hidden-features` and `--num-transforms`, forwarded to `build_posterior` (today it
cannot set them, `core/tool/smoke.py:207-212`), so the card run exercises the retrain's network end to
end.

### 5.9 One run-total line for masked probes, across resumes

The batch loop commits each batch's masked and simulated probe counts **at the same seam as the
observer** (§4.4: after the batch's rows are stored), and saves them in the checkpoint's `state.pt`
beside `batches_done` (`core/SBI/training_checkpoint.py:187`, `:356`, `:369`; readers use `.get`, so no
format bump). When generation ends — or when a resume finds the cache complete — training logs one
info record: probes masked over every committed batch (count, total, percentage) and the per-batch
spread; without a checkpoint it covers this process and says so. `smoke` shows it through its console
handler. The ±12-point check around 37 % is then read, not summed by hand from the per-batch warnings
(`core/SBI/chi_probes.py:267`); CLAUDE.md's pass criterion and the runbook's gate name the line.

### 5.10 The disk estimate, corrected

The pre-flight's cache-size line counts eight target columns (`core/SBI/pipeline.py:1434-1437`, "+8
covers the latent targets"); it counts `len(param_keys)` instead, with a test, so the retrain's line
reads ~10.3 GiB, not ~9.9.

### 5.11 The acceptances a calibration and a narrowing round ran under (H13)

A loaded posterior already carries the acceptances its load used (`LoadedPosterior.accepted`,
`core/artifacts/store.py:187`, set at `:1415`); only inferences write them today
(`core/orchestrator.py:1969-1985`, `:2166`), and `Accept`'s docstring says calibrations and narrowing
rounds do not (`store.py:96-100`).

- `validate_calibration` writes the posterior's acceptances to `body.results["accepted"]`.
- `tsnpe_round` writes the parent's to the child's `body.training["accepted"]`.
- `infer_and_visualize` records the union of the posterior's load acceptances and its own
  `other_observation` use — so an inference on a narrowed posterior's own observation, loaded with
  `--accept-truncated`, no longer records `accepted: []`.
- `core/artifacts/report.py`'s `_accepted` (`:177-189`) reads the training location for a posterior;
  `Accept`'s docstring and STATE's open item are updated.

Tests: each of the three records carries `truncated` when loaded with `Accept(truncated=True)` and
nothing otherwise.

### 5.12 What stays a runbook step

The larger network's training time has never been measured (compute per step rises about fourfold;
`recon/design-tier1.md` §5). The runbook's pre-flight measures seconds per epoch on a short real-width
run and projects the full fit, and stops for the owner's decision if the projection is far longer than
expected (§6.5). Host memory at the start of training is about 50 of 63 GB (sbi 0.25 copies the data
several times; `recon/design-tier1.md` §5); the runbook says to close other applications. Two gates of
the old runbook are dropped, with the reason in the runbook: "implied T in 280–310 K" (100 % by
construction on the tier-1 box) and a pre-run dry run of the cost (the tool has none; the pre-flight
measurement replaces it).

About seven or eight tasks. They touch code that moves tensors, so the card run of §8 is required.

---

## 6. Part 5 — the documents

### 6.1 Where, and the rules every page keeps

- **Location:** `docs/guide/`, with its own `README.md` as the start page (GitHub shows it when the
  folder is opened). The root README links to it. The working record stays where it is and is never
  cited.
- **Self-contained:** a page cites code by module and function name, never by line number, and cites
  no working document (the §2 scan walks `docs/guide/`); summary features are named by full label.
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
| `science.md` | the reasoning behind the settings: the nondimensionalisation; the conditioning features, valid flags, rank-Gaussianisation and winsorisation; chi probe design (the band measurement, the cycle floor and ceiling and the wall, the Ω₀ physics of masking, per-row placement and lock-in, the ±12-point precision); what chi buys; the prior (stability screen, component count); the tier-1 constraint and why T is an assumed input (and weakly informed in chi mode); reading calibration honestly (KS over c2st ranks, pooled histograms, t_scale's effective sample size, best fits are not recovery, flat SBC is not informativeness); narrowing rounds (the truncated prior, tempering, the rules' reasoning); identifiability limits (n, temp and tau_c enter only through noise amplitudes; a clean loss plateau); the August retrain's lasting findings in the correct orientation; the solver's physics check (graphs on/off); **open questions** (M-replicate conditioning, the thermal tail feature, the strata test, step and intermodulation observables, tiers 2 and 3 and the fix-T 12-dimension alternative, the log-sampling decision, pooling rows across rounds, and the owner's science items from STATE) | §4.2–§4.4.1, §4.6 (durable findings only), §8's `dt_nd_min` note, §11.1–§11.7, Appendix A's science items, "Conventions worth keeping" |
| `architecture.md` | the module map; the stages and compositions; the store (kinds, manifests, identity, loading refusals and the two acceptances, progressive records); the tool; the window (panels, worker threads, stream routing, progress, cancellation, settings persistence); the solver, CUDA graphs and reproducibility; memory planning and out-of-memory recovery; the FDT pipeline | §2, §2.1–§2.2 regenerated from `core/`; §8.3, §10.5; the P/S/C trap groups' design; Appendix A's memory and recovery entries |
| `rules-and-traps.md` | the rules the code keeps (the model/solver contract, the force-channel rule, the box/cell/units triple, everything nondimensional, the conditioning layout, every knob an argument, the private configuration copy, refusals with field keys, no chi override for what trains and infers, the narrowing-round refusal with no escape hatch, the comment policy); **the narrowing-round safety rules as a table** — each rule in the words of §2.3, where it is enforced, the test that pins it, and the ones still only prose (the rate of truths outside the region; a region cut in the box when the parent has no rotation; the −log P(A) inflation printed, not corrected); the traps that still bite, rewritten against today's code, grouped by subsystem, without the old labels | §3, §5 (every trap the map marks still-applies), §6.1's rules, §11.6, Appendix A's traps (the TorchScript warm-up, the facade re-import, waiting on memory one holds, the allocator variable's real name) |
| `testing.md` | environment and interpreter; the environment variables; the gates and markers and their budgets; the card smoke gate (the recipe verbatim, §6.6) and the diagnostic card (now with `probes`); what a green suite does not certify; test-writing notes (offscreen geometry, assert order not adjacency, never edit during a run) | §1.2–§1.3's live rules, §10.3, Appendix A's test notes |
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
worked examples. A test in `tests/test_docs.py` walks the parser (the `_walk` at
`tests/test_refusals.py:936`) and checks, in both directions, that every subcommand, mode and flag has
its heading and entry and that the page names no flag or subcommand the parser lacks, and that each
flag's stated default clause equals the help's (§3.5's table).

### 6.5 The retrain runbook

- **Decisions**, each with its reason: the tier-1 box; 10,000 × 2,048; 256 × 10; 6 probes supplied into
  12 slots; the certification round; the verdict (H11), temperature included and labelled an assumed
  input. `core/config.py`'s defaults stay (128 × 8, 5,000 batches): the retrain passes H6's values as
  flags, and the page says a retrain from the window would need them typed into the Posterior tab. The
  box needs a prior built on it — the ND screen is the master box's, so a `master.txt` prior loads under
  it (§5.1) — and the runbook says which to use and why.
- **Pre-flight:** the card gate including the tier-1 and larger-network line (§8); a short real-width
  `train` measuring seconds per epoch at 256 × 10 with the projected fit, **stopping for the owner's
  decision** if the projection is far beyond the measured ~92 h at 128 × 8 scaled to 20 million rows;
  the chi recording-length check (`identifiability jacobian`) on the master cell and, now fixed, on the
  tier-1 cell (its temperature column recorded as a measurement); `probes band` and `probes mask` on
  the card; VRAM by `nvidia-smi` (never `torch.cuda.mem_get_info()`), host memory (~50 of 63 GB at
  training start: close other applications), disk (the cache is ~10.3 GiB, §5.10).
- **The run:** the exact commands with every flag (`--chi` everywhere, since `CHI_MODE` is False by
  default; `--num-runs 10000 --hidden-features 256 --num-transforms 10` — tsnpe does not inherit the
  network size or the batch count from its parent, so the round passes them too); checkpointing every
  50 batches; how to resume (`--resume require`, the two lines it prints, the near-miss refusal and
  `--new-run`); what to watch (the `[cfg] chi` banner, `[tier1]`, `[budget]`, `[fisher]` — printed only
  by a fresh process — the masked-probe run total, `[patho]`, `[winsor]`, the loss curve's figure);
  capturing the console, since a resumed run's first process's lines exist nowhere else.
- **Gates:** the verdict PASSES; the previously dead input channels are revived (`ablation`, against
  the healthy channels of the same run — the old "29 of 42" baseline was on a 42-wide summary and does
  not apply); the masked-probe run total (§5.9) is within ±12 points of 37 %; the eigenvalues are
  recorded (`identifiability rotation`); the loss curve is recorded with the plateau reading; the
  informativeness total and per-parameter decomposition are written down as the first baseline. The two
  dropped gates of §5.12 are named, with their reasons.
- **The certification round:** `infer` on the tier-1 cell to create the observation (its id from the
  `[prism] observation` line); `tsnpe` with the flags above; pass when the truth lies inside the region
  in every truncated direction (§5.6), the child's calibration verdict on the region passes (its record
  shows the acceptance it ran under, §5.11), and the widths shrink no more than the data supports (the
  suite's proposal test is the arithmetic guarantee; the run checks containment and calibration).
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
3. **Repointing:**
   - `CLAUDE.md`: the "science guardrails are in §11.6, traps in §5" line points at
     `docs/guide/rules-and-traps.md` and `science.md`; the "still the reference until then" clause
     goes; "Rules that are load-bearing" gains one bullet for H4 (its scope, "state the reason in
     words", and `tests/test_source_hygiene.py` as its enforcement); the suite list gains the three new
     suites; the slow-set description is updated; `probes` joins the diagnostic card; the card recipe
     gains run 4 (§8); `docs/guide/` joins "Where things are".
   - `docs/checklists/display-walkthrough.md`'s header (seeded from the handoff's list, said as
     history).
   - `docs/STATE.md`: item 7 closed; the archiving commit named so `git show
     <commit>:PRISM_HANDOFF.md` retrieves it; its verbatim copy of the card recipe updated.
   - **One recipe text.** CLAUDE.md's block is canonical; STATE's copy and `testing.md`'s are verbatim.
     No test can read CLAUDE.md or STATE (H4), so the documents review's accuracy lens checks the
     equality.
   - The closed specs and plans stay untouched. `requirements.txt` was already rewritten in part 1.

### 6.7 The documents review

Three lenses, each followed by an adversarial verifier, before the handoff moves:

- **accuracy** — every checkable fact on every page (a module, function, flag, default, number,
  command) against the code at the stamped commit, and the recipe equality of §6.6;
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
- **Changed:** the three label-pinning tests (§2.4); the help-text pins (§3.8); the public-entry,
  field-key and untouched-leg tables (§4.1); the two renamed tests; `tests/test_artifact_store.py`'s V7
  legs — (c) RESUMED, whose eigenvalue assertion flips to the stored values while the settings assertion
  stays, and (d) TRUNCATED, which stays None because its region's parent has no rotation in that test
  (`:3646`, `:3657`); the golden simulation-identity digest (§5.7); the validate keyword pin (§5.4).
- **Cheap by design:** the tier-1 laplace/jacobian value match, the resume-with-Fisher test and the
  verdict-through-validate test use stand-ins (the stubbed-Fisher pattern of
  `tests/test_artifact_store.py:3600-3660`), because the recorded fast gate has about 78 s of headroom
  (14 min 42 s at `7e51275` against the 16-minute target).
- **Slow set:** a real-simulation check over about a minute is slow-marked — the tier-1 tiny `smoke`
  (§5.1) and the probe checks' twin-cell check (§4.7). `pytest.ini`'s `slow` marker description is
  reworded generically ("the long real-simulation tests"), and CLAUDE.md's slow-set description is
  updated.
- **Fast gate target:** 16 minutes, one process.

## 8. Gates

- **Per task:** the task's review and its one-process fast gate start at the same moment (the owner's
  rule since piece 5); a cleanup task also passes the §2.7 check and shrinks the §2.8 list.
- **After part 4** (the probe checks, the last code part): a whole-code review by independent readers —
  correctness, match to this spec, tests, the hazards no CPU suite reaches, and the reference rule —
  each followed by an adversarial verifier; one fix dispatch; a scoped re-review.
- **The card run**, after the fix dispatch, alone on the card, every command's `$LASTEXITCODE` checked
  and every result into STATE's gate table:
  1. CLAUDE.md's four existing lines, unchanged, with their pass criteria;
  2. **run 4, the tier-1 line** (added to CLAUDE.md's recipe permanently, since it is the retrain's box,
     and to STATE's verbatim copy): `smoke --chi --t-obs 4.5 --bounds
     Resources/Bounds/nadrowski/master_tier1.txt --cell Resources/Cells/nadrowski/master_spont_tier1.txt
     --hidden-features 256 --num-transforms 10 --checkpoint --save --store-root "$S/smoke_t1"` — exit 0,
     no OOM line, the `[tier1]` lines present, `T` in the posterior's parameter keys, the constraint
     block in its manifest, the masked-probe run total (§5.9) within ±12 points of 37 %;
  3. its resume (the same line with `--prior smoke_prior --stages prior,posterior --resume require`,
     without `--save`), which must resume and carry the eigenvalues (§5.5);
  4. a tiny narrowing leg on run 4's store (the stage commands have no `--store-root`, so
     `PRISM_ARTIFACTS` points at it):

     ```powershell
     $env:PRISM_ARTIFACTS = "$S/smoke_t1"
     $T1 = "--bounds","Resources/Bounds/nadrowski/master_tier1.txt"
     & $py -m core artifacts list observation   # the one observation run 4's infer stage wrote: its id
     & $py -m core tsnpe --chi @T1 --posterior smoke_posterior --observation <id> --num-runs 4 --run-size 32 --max-epochs 5 --name t1_round
     & $py -m core validate --chi @T1 --posterior t1_round --accept-truncated
     & $py -m core infer --chi @T1 --posterior t1_round --cell Resources/Cells/nadrowski/master_spont_tier1.txt --t-obs 4.5 --accept-truncated --accept-other-observation
     ```

     — the verdict line, the truth containment (§5.6), and `accepted` on the calibration, the child and
     the inference (§5.11);
  5. the diagnostic card: `sbc`, `identifiability jacobian`, `ablation` against `$S/smoke`,
     `identifiability laplace` against `$S/smoke_chi0` (each with `$env:PRISM_ARTIFACTS` at that store),
     plus `identifiability jacobian` on the tier-1 cell, its temperature column recorded as a
     measurement;
  6. the probe checks: `probes band` and `probes drive` on the master cell (`--bounds
     Resources/Bounds/nadrowski/master.txt --cell Resources/Cells/nadrowski/master_spont.txt`) with
     `$env:PRISM_ARTIFACTS` at a scratch store, and `probes mask --bounds
     Resources/Bounds/nadrowski/master.txt --prior smoke_prior` with it at `$S/smoke`, against §4.8's
     readings; `PRISM_ARTIFACTS` removed afterwards.
- **After part 5:** the documents review (§6.7), its fixes, the only-record checklist closed, the
  handoff moved.
- **Final:** the fast gate on a quiet machine and the slow set, then STATE and CLAUDE.md from the
  measured numbers (STATE's decisions log also records D11's narrowed reading and the seed widening,
  §1.2). The card run stands for the piece unless a later change touches code that moves tensors, in
  which case the affected lines are re-run.

## 9. The display walkthrough (rows F1–F5)

Added to `docs/checklists/display-walkthrough.md`, run by the owner on the real screen. F1 first walks
through building a tiny tier-1 model in the window (prior on `master_tier1.txt`, a posterior at a tiny
budget, the tier-1 cell loaded) so the rest have something to show.

| row | what the owner checks |
|---|---|
| F1 | the Validate tab's log ends with the verdict line; every parameter is judged, temperature marked "(assumed input)" |
| F2 | the Infer tab's summary and corner plot mark temperature "T (assumed input, K)" |
| F3 | a posterior whose `manifest.json` was hand-edited to say `amortized: true` beside its truncation region is listed on the Artifacts screen as a record with no usable manifest, and its detail pane's reason line is a plain sentence with no label (the one route by which a reworded message reaches the window; if the plan finds the reason is not shown, F3 is dropped and the rows renumbered) |
| F4 | the Posterior tab's early-stopping patience tooltip (the stop-after-epochs field in the Density estimator group) gives its measurement — stopped at epoch 130 on a patience of 20, best at 110 — with no incident label; and the tooltips that name the calibration's operating points say "(t_scale, T_obs)" |
| F5 | a narrowing round's log in the TSNPE tab reports truth containment per direction |

## 10. Risks, and the order of work

**Order:** part 1 (§2.9: nine tasks, the scan first) → part 2 (help tidy) → part 3 (readiness, §5;
the tier-1 tests before the fixes they cover; §5.9 builds the loop-level commit seam of the masked
counts) → part 4 (the probe checks, §4; its first tasks build the geometry helper of §4.6 and the
observer of §4.4 on that seam) → the whole-code review and fixes → the card run → part 5 (the pages,
the runbook, the README, the reference page and its test) → the documents review and fixes → retiring
the handoff → final gates → STATE and CLAUDE.md → the owner's walkthrough.

| risk | guard |
|---|---|
| a code change hidden in 2,000 comment edits | the per-file node-by-node check against each task's permitted list (§2.7); the known raw-source tests |
| the observer and the masked-count seam touch the training generator | byte-identical seeded output in three modes; production geometry untouched (§4.6); the card run |
| the identity format change | a version bump and the recomputed golden digest; no cache exists to strand |
| the verdict read as more than it is | the t_scale power caveat printed with it; informativeness recorded beside it; temperature marked |
| the tier-1 box never trained at scale | the card legs are tiny; scale is the runbook pre-flight's job |
| the larger network's training time is unknown | the runbook's projection and stop point |
| the documents drift | the reference-page test, the link test, the scan over `docs/guide/`, the commit stamp, the recipe-equality check |
| a long plan | the plan gets a pre-flight against the code, and every ruling is folded into the task it binds (a plan's preamble does not reach a task brief) |
| the scan's allowlist grows into a hole | an entry that matches nothing fails; entries are exact names, per-file sets or narrow shapes |
| an id-collision rewrite gives the wrong reason | the §2.3 collision rules; each cleanup review reads every rewritten sentence at those ids |

## 11. Deviations (filled during execution)

| # | where | what changed | why | cost if wrong |
|---|---|---|---|---|
