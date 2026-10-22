# Prompt for the next chat — the window redesign returns: the plan of stage 3 (the run)

Paste everything below the line into a new chat opened on this repository. It is written in the owner's voice by the
session of 22 October 2026 that ran the fit-saving stage's last review and exit and then the Laplace fix. What in it is
not the owner's own words:

- The owner's own words in that session: "Yes, start then" (a quiet machine for the gate of record and the slow set),
  "Yes, straight after" (the card checks), "Yes, as shown" (CLAUDE.md's sentence about the fit), "Yes, add it" (run 5 in
  the card block), "Fix it here, then we can start a new chat" (the Laplace fault) and "Write the hand-off prompt for
  stage 3's plan". "How much to spend", the usage rule and "What must not change" are repeated from the earlier
  prompts, whose words stand.
- Everything else is that session's record and its way of handing over: what stage 3 must carry is gathered from
  `docs/STATE.md` and the ledger, and every count, time and measurement is that session's, not the owner's.

---

PRISM — the redesign of the window returns: the plan of stage 3, the run.

**Where things stand.** Stage 2 of the redesign, the shell, is built and its exit passed on 21 October (code
`c7106c3`; the fast gate of record on a quiet machine 1583 passed in 19 min 21 s; pictures in three sets of 119). The
build then paused, as I decided on 6 October, for the separate design for saving the fit: that design
(`docs/superpowers/specs/2026-10-21-fit-saving-design.md`, `3221fcc`) and the plan of its short stage
(`docs/superpowers/plans/2026-10-21-fit-saving-stage.md`, `e89da79`) were approved on 21 October, and THE FIT-SAVING
STAGE IS BUILT AND ITS EXIT PASSED on 22 October: Tasks 0 to 6 (`3448486`..`cfe80d7`), its last review with one fix
wave (`0158d16`..`bb2857c`), the gate of record 1660 passed (30 min 4 s, not a quiet time), the slow set 5 passed (1 h 33
min), the card gates passed with a new run 5 (a training cut like a power cut after its third pass, then resumed from
the command line). I said yes to CLAUDE.md's new sentence about the fit (`de6f288`) and to run 5 in the card block
(`abe5579`). The card's diagnostic run found an older fault (the Laplace diagnostic could not draw from a posterior
read back from disk); I had it fixed in the same chat (`05c0123`, two tests, the gate 1662 passed, the diagnostic green
on the card again). STATE is at `f227c70`. I pushed up to `1ad02f5`; everything after it is not pushed, and I push when I
choose. Nothing was running when this was written; check that the graphics card is free, no python is alive and no
`PRISM.lock` stands beside my settings file before any work. My real `PRISM.ini` is 3,296 bytes, sha256 79BF94DF…C83E.

**How much to spend is decided: do not ask again.** "In between", as I chose on 12 October: no workflow unless I ask
for one in my own words; every agent given its model by name, Sonnet where reading is enough; the review by three
readers for every task that builds. My words of 16 October stand: "Try to also finish things up at a clean state when
usage limits are near their max, I won't buy anymore credits". So: read the usage card when you start, tell me where
the weekly bar stands, and read it again before every dispatch; do not count on extra usage; when the weekly bar nears
its end (about 90 %), start nothing that cannot be finished below it, and stop at a clean state. The week resets on
25 October at 04:00 UTC; the bar stood at 69 % when this was written. A plan costs less than a build (the fit-saving
design and plan together cost about 1.9 million tokens in background agents), but your own long reading costs too:
read with agents where reading is enough, and keep your own context for the plan's head.

**Read first, in this order.**

1. `docs/STATE.md` — the head; the paragraph "THE FIT-SAVING STAGE'S LAST REVIEW AND EXIT" (its "What waits for the
   window's stages" and the redesign's places the fit-saving design changes); then the paragraph "STAGE 2 OF THE
   WINDOW REDESIGN IS BUILT", for its CARRIED list (search for "**CARRIED**").
2. `CLAUDE.md`.
3. The ledger, `.superpowers/sdd/2026-10-06-window-redesign/progress.md` (gitignored): its first 47 lines; in "## Stage
   2 — the shell (plan)" the lines from "- STAGE 2'S LAST REVIEW, DISPATCHED" to the section's end; in "## The
   fit-saving stage" the lines from "- NEW CHAT 2026-10-22" to the end; "## Deviations from the design" (rows 1 to 82)
   and the last ten lines of "## Gates". Its lines are very long: search it, do not read it through. Never add a line
   above the lines that begin "- HAND-OFF"; append at a section's end (`ledger_append.py` does).
4. The design, `docs/superpowers/specs/2026-10-05-window-redesign-design.md` (section 8, its 82 rows of departures,
   wins over the parts where they differ) and its parts in `docs/superpowers/specs/2026-10-05-window-redesign-design/`:
   `06-run.md` whole (the run is stage 3); `13-tests-stages-docs.md` (13.1 and Table B's stage 3 rows; 13.2's "Stage 3.
   The run."; the card test, which is owed ONCE, at stage 3's exit; the slow set at stage 3's exit; 13.3 and 13.4, the
   walkthrough rows G1, G2, G6 and G7, runnable after stage 3); `11-engine.md` for E1, E3's engine half, E6's two
   refusals, E7 with E10, and E9; `07-plans.md` 7.2.6 (keep-awake's mechanism); and what `03-shell.md`, `04-look.md`
   (4.9, the figures), `05-anatomy.md` and `12-findings.md` say that the run uses; the locked records
   `docs/superpowers/specs/2026-10-05-window-redesign-assets/decisions-as-approved.md` and `rulings-after-the-drafts.md`
   (the later wins). Have readers read the parts against today's code; read yourself what the plan's head needs.
5. The fit-saving design, for the window's share it gives stage 3: its sections 4.3, 5.6, 6.4, 6.6, 8 and 9; with the
   fit-saving last review's notes for stage 3, `.superpowers/sdd/2026-10-06-window-redesign/fit-saving/reviews/
   last-review-two.md` ("For the exit and later stages", "Stage 3 (the run page)") and `last-review-one.md` (section
   11, "Stage 3").
6. The models of a plan: stage 2's plan `docs/superpowers/plans/2026-10-14-window-redesign-stage-2.md` and its head as
   cut for implementers `.superpowers/sdd/2026-10-06-window-redesign/stage-2/briefs/plan-head.md`; the fit-saving plan
   as a model of a short one (its "How to run this plan", the registry, the departures table with costs).
7. The prompts of 14, 20 and 21 October beside this file, for everything this one does not repeat (the preparation and
   review loop, the skills and their limits, the standing rules). Where they differ, the later wins, and this one wins
   over all of them.

**What to do.** Write the plan of stage 3 from the design's parts as they stand, with the `writing-plans` skill, commit
it when it is written, and give it to me to approve. Build nothing before I have. Stage 3 delivers, by part 13.2, for
every run of every part at once (they share one run host today, `BasePanel.dispatch` in
`core/gui/panels/base_panel.py`): U19's list for this stage as part 6 designs it, so that any run is watched, stopped,
closed over and continued in the new way; the guard on the figure hand-over (6.13); the reason against shutting down
(U27) after part 6.10's scratch test; keep-awake's mechanism (7.2.6), always on until its switch arrives with Settings
in stage 4; the launcher with no console, which needs the full text in a file first; and, as a task of its own before
the loss curve is built (ruling D7), the test that renders a figure the window draws itself on the window's thread
while another thread builds and saves one: if they can collide, the loss curve is a painted widget. Engine: E1; E7 (opt
in, and only the window opts in) with E10; E9; E3's engine half; E6's two refusals in the training stage — find their
lines again at today's head, since the fit-saving stage moved `build_posterior`. The forms that start runs stay
today's. Tests rewritten first (Table B, stage 3; the new suites the part names). Its exit: the slow set; the card test
with the diagnostic card and run 5 (tell me first: the commit, the scratch folder, how long the last run took, and
that nothing else may use the card); the display-marked run; pictures.

The fit-saving design gives stage 3 its share of the window, to be planned with it: after "Stop training…" the
question "Keep this training so you can resume it later?" with Keep and Discard (its words are this stage's; the
engine keeps a stopped training at once, "Keep" needs no call and "Discard" is `store.delete("fit", id)`, refused while
a process holds it); the close question's words during a fit; the cancel words (a stop lands after the pass in
progress, up to about an hour at full size; today's "up to ~1 min during training" is wrong); "Before you start" saying
the fit is saved after every pass and that a resume reads its simulation set back first (about four minutes at full
size) — the Training page itself is stage 4's, so say where these lines land; the run history naming a kept training;
the health line when a save fails (the run's log already carries "[fit] the last save failed: …" and the record its
`save_error`). Two things the engine does not give you yet, to be planned (an engine change goes on the code audit's
list): the id of the kept training a stopped run made is not handed back (the stage raises; today it is only in the log
line "[fit] the training stopped after pass N and is kept as fit <id>" and in the listing); and a resumed training that
stops before it saves a pass in its new process is the old record left as it was, so Discard there deletes every
earlier process's passes and the question must say so. When stage 3 lands, the redesign's places the fit-saving design
changes are amended as that stage's departures: U10's "that a fit cannot be resumed", U12's "Stop training…", section
7's first bullet, finding W172, and the parts `06-run.md`, `08-steps.md`, `10-records-and-parts.md` and
`12-findings.md` where they say a stopped fit is lost.

Say honestly how many tasks you expect: stage 1 was sized at eleven and came to about thirty; stage 2's plan had ten
building tasks and grew by fix rounds; the fit-saving stage held to its six. Put the preparation that worked into the
plan: for a task that touches threads, the run host or the engine, one skeptic (opus) first builds it as a stand-in in
a scratch copy, runs its tests red then green, and brings what it finds to the controller, whose rulings go INTO the one
brief; one implementer per task from that one brief; on DONE the one-process fast gate and three readers at the same
moment (seat one sonnet, the brief against what was built; seat two opus, the product before against after; seat three
opus, the tests and the faults they see, its fault runs after the gate); an Important to one skeptic with your
conditional rulings in a file; one fix round, re-read by one reader while a fresh gate runs; a last review over the
whole stage with one fix wave; the exit with me. Ask me, one question at a time, in plain words with a picture where it
helps and your recommendation, what the parts leave to me; decide the small things yourself and write each as a ruling
with its cost.

**What the plan must carry** (STATE's paragraph on stage 2 has them; the ledger's "CARRIED" lines and
`stage-2/reviews/stage-2-final-review-sorting.md`'s table by stage have each):

- To stage 3's FIRST task: one session helper written three times across three suites (`tests/test_shell.py`,
  `tests/test_step_chain.py`, `tests/test_look.py`); the nine lows of stage 2's fix wave's re-read
  (`stage-2/reviews/stage-2-final-fix-reread.md`); the rail's rows shifting 1 to 2 pixels as the open part changes.
- To stage 3's plan: the launcher (a claim before the heavy imports, which then must not be made twice in one process;
  `main()` called or the claim let go; no stream under `pythonw`, so every thread's records go into its log file; the
  junction spellings of the settings path); reopening the running PRISM's own window (my answer Q12: closing the window
  lets go of the lock) and how the run history reads a run now that a lock no longer proves it died; a popped-out
  figure window keeping the process alive; `has_runs` never fed, so the first-time Home ignores a live run; the four
  jobs of the window's run-state handler; Home's read listing the whole store at each refresh; a run view as a part's
  page; the foot's tool tip and Ctrl+R; the tests that see a run only through today's base panel; a knock while a
  question is open on the real screen, and a real second start, as walkthrough rows; the sorting's 19 rows for stage 3;
  and the product seat's section "for stage 3's planner" (`stage-2/reviews/stage-2-final-review-product.md`).
- From the fit-saving stage: the paragraph above; and that a window Cancel during a fit is taken at the first record
  after the pass's save, so a stop loses nothing finished; that during the simulation, before the fit's first save, no
  kept training is made (the run's log says so in one line); and the code audit's list in STATE's paragraph on the
  fit-saving exit, which this stage must not lose.

**Mine to hear, still open:** the replacement wording of `CLAUDE.md`'s launch line (the design's part 13.3.4), approved
only when the launcher with no console lands in this stage, and of its rule on what the window remembers (part 7.3.3),
in stage 4 — each only on my yes; bundling Cascadia Mono so every computer shows it (never asked). The lists of what
was decided without me are in the ledger: `stage-2/exit/rulings-for-the-owner.md` and
`fit-saving/exit/rulings-for-the-owner.md`; I may undo any of it; keep giving me such a list at each stage's exit.

**Facts the plan stands on** (measured; the ledger has each):

- The fast gate holds 1662 tests (1672 collected) and took 25 to 30 minutes this week, on a busy or slow machine (stage
  2's of record: 19 min 21 s for 1583 on a quiet day). The slow set took 1 h 33 min on 22 October.
- The gate tools in `C:\Users\J\AppData\Local\Temp\prism-redesign-scratch\` (`gate.sh`, `ground.ps1`) write into the
  fit-saving workspace's `gates/` now: the plan's Task 0 points them at a workspace of stage 3's own (text copies in the
  ledger's `fit-saving/tools/`); `ledger_append.py` mode `stage1` appends just before "## Engine changes landed", that
  is at the end of the ledger's last section, so stage 3's section heading goes there first.
- The card's free-memory reading moves between about 2.3 and 2.9 GiB with the desktop; a check script's bound should
  allow for that (lowering a guard mid-run was refused by the session's permission check on 22 October).
- `AskUserQuestion` waits for my answer; ask while long runs go, not before them.
- Readers run NO python while a gate runs, and the one-pytest rule covers scratch copies: a seat that wants runs reads
  first, ends its turn with "waiting to run", and is given the machine when the gate has exited. Times in the ledger are
  read from `date`, never guessed. A `cd` inside a Bash call moves the session's working directory.

**What must not change without my say-so:** the science (the models, the simulation, the features, the training and
calibration methods, every setting in `core/config.py`); what a simulated row holds and the simulation set's identity
(the finished set `Artifacts/simulations/eda5c2a1e468` stays usable); what a fit that is never interrupted computes;
the rules in `CLAUDE.md` (edited only on my yes); anything on disk under `Artifacts/` and the console capture
`C:\Users\J\prism-retrain-console.log` (read freely; ask before anything deletes, renames, rewrites or trains on them).

Do not start the retrain, or any training of real size, in this chat; the tiny trainings in the tests and the card
gates are fine. Standing rules: the local `main` branch; one commit per logical step, a short subject, never amended,
ending with the Co-Authored-By line the harness provides; I push. Never two pytest processes at once; never edit a
source file while a suite runs; a run on the card has the card to itself. Scratch files under
`C:\Users\J\AppData\Local\Temp\`; no `.py` file under `.superpowers/` or `docs/`. Update `docs/STATE.md` at the end of
every session, and hand over to a new chat, with STATE, the ledger and a prompt like this one, at two thirds of the room
or when the usage rule says stop, whichever comes first.
