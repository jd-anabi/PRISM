# Prompt for the next chat — the fit-saving stage, from Task 4

Paste everything below the line into a new chat opened on this repository. It is written in the owner's voice by the
session of the afternoon of 21 October 2026 that built the stage's Tasks 0 to 3. What in it is not the owner's own
words:

- The owner said nothing in that session beyond the pasted prompt `2026-10-21-fit-saving-stage-build.md`, whose words
  stand: "How much to spend", the usage rule and "What must not change" are repeated from it.
- Everything else is that session's record: what it learned is under "How a task is run" and "Facts". Every count, time,
  cost and measurement is that session's, not something the owner measured.

---

PRISM — the fit-saving stage: go on building it, from Task 4.

**Where things stand.** The design for saving the fit as it goes (`docs/superpowers/specs/2026-10-21-fit-saving-design.md`,
`3221fcc`) and the plan of its short stage (`docs/superpowers/plans/2026-10-21-fit-saving-stage.md`, `e89da79`) are
approved; I approved both on 21 October. The plan is being built: Tasks 0 to 3 are complete, each building task with its
review by three readers clean and its gate green on the same head. Task 1 (`3448486`, fix `d502dfc`) built the kept
fit's files, lock and identity digests; its fix round also made the identity record the code AS THE PROCESS LOADED IT
(departure F-10, `207b4db`). Task 2 (`a0357b3`, no fix round) split the training call, hooks sbi's per-pass summary
and restores a kept snapshot. Task 3 (`63e9099`, fix `1aecbf8`) made `fit` a record kind of its own, listed in Records
and by the tool. Task 4 is PREPARED: its skeptic built it as a stand-in on `1aecbf8` (40 tests green in its file, the
narrowing round's keyword pin unchanged, six mutants each caught; it measured 1641 collected, 1631 passed expected, the
fast gate about a minute and a half longer) and wrote its one brief
(`.superpowers/sdd/2026-10-06-window-redesign/fit-saving/briefs/task-04-brief.md`): its findings are marked PROPOSED
for you to rule — six where the plan does not hold as written (A1 the plan's way of stopping a test's training cannot
give the stops it asserts; A2 a "no `fits/` folder" assertion depends on the tests' order; A3 a failed first save
leaves one folder per pass; A4 a pending window Cancel between the two `try` blocks can leak the lock; A5 saving
should key on the kept simulation set, not on the checkpoint cadence; A6 nothing tests that the narrowing round
forwards `save_fit`), twelve that hold, twelve choices and four notes for Task 5. Code head `1aecbf8` (the commit that
adds this prompt and STATE is on top of it); the last gate on it: 1619 passed, 5 skipped, 5 deselected in 19 min 49 s.
I pushed up to `1ad02f5`; everything after it is not pushed, and I push when I choose. Nothing was
running when this was written; check that the graphics card is free and no python is alive before any work (a
`python -m core.gui` started from `run.bat` is my own PRISM: leave it alone and wait; while it is open a `PRISM.lock`
stands beside my settings file, so no gate runs then). My real `PRISM.ini` is 3,296 bytes, sha256 79BF94DF…C83E.

**How much to spend is decided: do not ask again.** "In between", as I chose on 12 October: no workflow unless I ask
for one in my own words; every agent given its model by name, Sonnet where reading is enough. I know this stage will
cost more than one week's allowance and that is fine with me; but my words of 16 October stand: "Try to also finish
things up at a clean state when usage limits are near their max, I won't buy anymore credits". So: read the usage card
when you start, tell me where the weekly bar stands, and read it again before every dispatch; do not count on extra
usage; when the weekly bar nears its end (about 90 %), start nothing that cannot be finished below it, and stop at a
clean state. The week resets on 25 October at 04:00 UTC; the bar stood at 51 % when this was written (Tasks 0 to 3
with their preparation, reviews and fix rounds, and Task 4's preparation, cost about 7 points of it: about 4.6 million
tokens in background agents).

**Read first, in this order.**

1. `docs/STATE.md` — from the top down to the line that begins "**THE DESIGN AND THE PLAN FOR SAVING THE FIT", and the
   paragraph "THE FIT-SAVING STAGE, TASKS 0 TO 3" above it, which is the last record.
2. `CLAUDE.md`.
3. The design, whole (about 350 lines); the plan's head down to "## Task 0" (its departures now run F-1 to F-10); Task
   4's section when you rule on its brief, and each later task's when its turn comes.
4. The ledger, `.superpowers/sdd/2026-10-06-window-redesign/progress.md`: its first 47 lines; then the section "## The
   fit-saving stage" whole (its first lines write the loop out once; "CARRIED to Tasks 4 and 5" and "minor (deferred)"
   lines are what later briefs and the stage's last review must carry). Never add a line above the three lines that
   begin "- HAND-OFF". Its lines are very long: search it, do not read it through.
5. In the ledger's `fit-saving/`: `briefs/task-04-brief.md` (the skeptic's findings first: rule each, mark the block
   RULED as Task 2's brief was marked), `reports/task-04-prep-report.md`; `reviews/review-contract.md` (what every
   reader is bound by); `recon-report.md` when you need the evidence the design stands on.
6. The prompts of 6, 10, 11, 12, 13, 14, 15, 16, 20 and 21 October beside this file, for everything this one does not
   repeat. Where they differ, the later wins, and this one wins over all of them.

**Tools.** In `C:\Users\J\AppData\Local\Temp\prism-redesign-scratch\`: `gate.sh` (kinds fast, quiet, slow, display,
collect) and `ground.ps1` (from the PowerShell tool) write into `fit-saving/gates/` (labels `fitNN-<sha>`, after a fix
`fitNN-fix1-<sha>`); `ledger_append.py` (mode `stage1` appends at the end of "## The fit-saving stage", `gates` at the
file's end); in `...\fit\`: `cut_brief.py` (cuts a task's brief from the plan's line ranges under a preface),
`package.sh` (a review package `reviews/<name>.diff`). Text copies of `gate.sh`, `ground.ps1` and `ledger_append.py`
are in the ledger's `fit-saving/tools/`. A scratch script that imports torch and must stay off the card sets
`CUDA_VISIBLE_DEVICES=-1` and asserts CUDA is not available: an EMPTY value does not hide the card here.

**What to do.** Rule on Task 4's brief, then build Tasks 4, 5 and 6, each through the plan's loop (my rule: one fresh
implementer from one brief; the one-process fast gate in the background and the review by three readers started at the
same moment; one fix round; complete when the review is clean and the gate green on the same head; for Tasks 4 and 5
a skeptic first builds the task as a stand-in in a scratch copy — Task 4's is done). Then the stage's last review
(three readers over the stage's whole range, `82424f5` to the last code commit, which reads the ledger's deferred
minors and carried items), then Task 7, the exit, which needs me: a quiet machine for the gate of record and the slow
set, the card gates (tell me first; the card to itself), and my yes or no to the new sentence about the fit in
`CLAUDE.md` and to the new resume run in its card gate block. What the build departs from in the design goes into the
plan's table and the ledger, with its cost; ask me when it touches something I decided. Hand over to a new chat, with
STATE, the ledger and a prompt like this one, after about three tasks, at two thirds of the room, or when the usage
rule says stop, whichever comes first.

**Ask me** what is mine: what I decided or what the design leaves to me, what cannot be undone, and anything under
"What must not change" below; one question at a time, in plain words, with a picture where it helps and your
recommendation. Decide the small things yourself, write each as a ruling with its cost, and give me the list of them at
the exit.

**How a task is run** (what the session of 21 October learned, beside what the earlier prompts list):

- Readers run NO python while a gate runs, and the owner's one-pytest rule covers scratch copies: a skeptic or a seat
  that needs runs does its reading first and ends its turn with "waiting to run"; the controller hands the machine to
  one of them at a time when the gate has exited (seat three's fault runs, then a fix skeptic's, then the next task's
  preparation). A message sent to an agent while it works waits for its next tool round.
- The next task's skeptic can read and draft beside the current task's review; it re-bases its scratch copy when the
  current task's commit (or its fix) lands, and runs only when given its turn.
- An Important finding goes to one skeptic (opus) with the controller's conditional rulings in a file
  (`briefs/task-NN-fix1-rulings.md`); the skeptic confirms by runs and writes `briefs/task-NN-fix1-brief.md` with every
  edit's exact text; the original implementer is resumed to build it; one reader (sonnet) re-reads the fix diff while a
  fresh gate runs.
- A fix round may fold in a Minor that is the product side of an Important's property (Task 3's unreadable manifest),
  with a ruling; other Minors go to the ledger for the stage's last review.
- Times in the ledger are read from `date`, never guessed (two were guessed this session and corrected).
- A `cd` inside a Bash call moves the session's working directory; end such a call at the repository's root.

**Facts the plan now stands on** (measured this session; the ledger has each): the flow's base distribution keeps a
buffer outside its state dict (`_log_z`), so the restore compares only the buffers the state dict keeps; sbi's
in-process resume double-counts a pass only in the stretch that ends the fit, so the resume test's middle point is the
pass right after the best; building the trainer draws from the CPU generator, so the split's state is read just before
`infer.train`; `inspect.getsource` reads a file as it is now (the reason for F-10); the window imports the training
stage at start-up (`main_window.py:26`), so its loaded code is its start-up's; a FitLock's own handle counts as
running, so a session lets its lock go before deleting its own record; the fast gate took 19 to 24 minutes this session
(1596 to 1619 passed).

**What must not change without my say-so:** the science (the models, the simulation, the features, the training and
calibration methods, every setting in `core/config.py`); what a simulated row holds and the simulation set's identity
(the finished set `Artifacts/simulations/eda5c2a1e468` stays usable); what a fit that is never interrupted computes; the
rules in `CLAUDE.md` (edited only on my yes); anything on disk under `Artifacts/` and the console capture
`C:\Users\J\prism-retrain-console.log` (read freely; ask before anything deletes, renames, rewrites or trains on them).

Do not start the retrain, or any training of real size, in this chat; the tiny trainings in the tests and the card gates
are fine. Standing rules: the local `main` branch; one commit per logical step, a short subject, never amended, ending
with the Co-Authored-By line the harness provides; I push. Never two pytest processes at once; never edit a source file
while a suite runs; a run on the card has the card to itself. Scratch files under `C:\Users\J\AppData\Local\Temp\`; no
`.py` file under `.superpowers/` or `docs/`. Update `docs/STATE.md` at the end of every session.
