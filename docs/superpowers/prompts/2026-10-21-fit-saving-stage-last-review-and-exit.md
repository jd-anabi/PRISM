# Prompt for the next chat — the fit-saving stage: its last review, then the exit

Paste everything below the line into a new chat opened on this repository. It is written in the owner's voice by the
session of the evening of 21 October 2026 that built the stage's Tasks 4 to 6. What in it is not the owner's own
words:

- The owner said nothing in that session beyond the pasted prompt `2026-10-21-fit-saving-stage-from-task-4.md`, whose
  words stand: "How much to spend", the usage rule and "What must not change" are repeated from it.
- Everything else is that session's record: what it learned is under "How a task is run" and "Facts". Every count, time,
  cost and measurement is that session's, not something the owner measured.

---

PRISM — the fit-saving stage: its last review, then the exit.

**Where things stand.** The design for saving the fit as it goes (`docs/superpowers/specs/2026-10-21-fit-saving-design.md`,
`3221fcc`) and the plan of its short stage (`docs/superpowers/plans/2026-10-21-fit-saving-stage.md`, `e89da79`) are
approved; I approved both on 21 October. THE STAGE'S CODE IS BUILT: Tasks 0 to 6 are complete, each building task with
its review by three readers clean and its gate green on the same head. Tasks 1 to 3 (`3448486`..`1aecbf8`) built the
kept fit's files, lock and identity, the training call's two halves with the per-pass hook and the restore, and the
`fit` record kind. Task 4 (`bd0b30a`, fix `40d1707`) made the training stage save every pass and keep a stopped
training. Task 5 (`b7f8394`, `f879095`, fix `1f23446`) made a kept training resumable by the stage, the narrowing round
and the tool's `--resume-fit`. Task 6 (`cfe80d7`, no fix round) proved the resume on the graphics card and added the
drill `tests/_fit_drill.py`, which ran once on the card with the exit's proposed run 5 as written and met every
criterion. Code head `cfe80d7` (the commit that adds this prompt and STATE is on top of it); the last gate on it: 1657
passed, 5 skipped, 5 deselected in 30 min 28 s, on a loaded machine. I pushed up to `1ad02f5`; everything after it is
not pushed, and I push when I choose. Nothing was running when this was written; check that the graphics card is free
and no python is alive before any work (a `python -m core.gui` started from `run.bat` is my own PRISM: leave it alone
and wait; while it is open a `PRISM.lock` stands beside my settings file, so no gate runs then). My real `PRISM.ini` is
3,296 bytes, sha256 79BF94DF…C83E.

**How much to spend is decided: do not ask again.** "In between", as I chose on 12 October: no workflow unless I ask
for one in my own words; every agent given its model by name, Sonnet where reading is enough. I know this stage will
cost more than one week's allowance and that is fine with me; but my words of 16 October stand: "Try to also finish
things up at a clean state when usage limits are near their max, I won't buy anymore credits". So: read the usage card
when you start, tell me where the weekly bar stands, and read it again before every dispatch; do not count on extra
usage; when the weekly bar nears its end (about 90 %), start nothing that cannot be finished below it, and stop at a
clean state. The week resets on 25 October at 04:00 UTC; the bar stood at 57 % when this was written (Tasks 4 to 6 with
Task 5's preparation, nine reviews and two fix rounds cost about 6 points of it: about 4.6 million tokens in
background agents).

**Read first, in this order.**

1. `docs/STATE.md` — from the top down to the line that begins "**THE DESIGN AND THE PLAN FOR SAVING THE FIT": the
   paragraphs "THE FIT-SAVING STAGE, TASKS 4 TO 6" (the last record) and "THE FIT-SAVING STAGE, TASKS 0 TO 3".
2. `CLAUDE.md`.
3. The design, whole (about 350 lines); the plan's head down to "## Task 0" (its departures run F-1 to F-10) and Task
   7's section (the exit).
4. The ledger, `.superpowers/sdd/2026-10-06-window-redesign/progress.md`: its first 47 lines; then the section "## The
   fit-saving stage" whole (its first lines write the loop out once; "minor (deferred" and "CARRIED to" lines are what
   the last review must carry). Never add a line above the lines that begin "- HAND-OFF"; append at a section's end
   (`ledger_append.py` does). Its lines are very long: search it, do not read it through.
5. In the ledger's `fit-saving/`: `last-review-inputs.md` (the index for the last review: the range, every review and
   fix report, every deferred Minor and carried item, the facts the readers need); `reviews/review-contract.md` (what
   every reader is bound by); the briefs' RULED paragraphs when a reader's finding touches one.
6. The prompts of 6, 10, 11, 12, 13, 14, 15, 16, 20 and 21 October beside this file, for everything this one does not
   repeat (the prompt of 20 October has how stage 2's last review and exit ran). Where they differ, the later wins, and
   this one wins over all of them.

**Tools.** In `C:\Users\J\AppData\Local\Temp\prism-redesign-scratch\`: `gate.sh` (kinds fast, quiet, slow, display,
collect) and `ground.ps1` (from the PowerShell tool: `& ...\ground.ps1 before|after <label>`; from Bash,
`powershell -File` is refused by the execution policy unless given `-ExecutionPolicy Bypass`) write into `fit-saving/gates/` (labels `fitNN-<sha>`, after a
fix `fitNN-fix1-<sha>`; the last review's `fitLR-<sha>`, the exit's `fitexit-<sha>`); `ledger_append.py` (mode `stage1`
appends at the end of "## The fit-saving stage", `gates` at the file's end; write its input with the Write tool, a long
line through a Bash `printf` breaks on backslashes); in `...\fit\`: `cut_brief.py`, `package.sh` (a review package
`reviews/<name>.diff`). Text copies of `gate.sh`, `ground.ps1` and `ledger_append.py` are in the ledger's
`fit-saving/tools/`. A scratch script that imports torch and must stay off the card sets `CUDA_VISIBLE_DEVICES=-1` and
asserts CUDA is not available: an EMPTY value does not hide the card here.

**What to do.**
1. **The stage's last review**: three readers over the stage's whole range, `82424f5..cfe80d7` (seat one sonnet: the
   design and the plan against what was built, the departures F-1 to F-10 and every ruling; seat two opus: the product
   as a whole — what a person meets across the stage's pieces, the owner's lines; seat three opus: the tests as a whole
   and the faults no test sees), each reading `last-review-inputs.md` and carrying the deferred Minors and the carried
   items; with the one-process fast gate on the head at the same moment. ONE fix wave (an Important to one skeptic with
   your conditional rulings, as in the tasks; the deferred Minors the readers judge worth curing may ride in the same
   wave, each with a ruling and its cost); one reader re-reads the wave while a fresh gate runs.
2. **Task 7, the exit**, which needs me: a quiet machine for the gate of record and the slow set (ask me first; the
   gates of this session took 27 to 31 minutes on a loaded machine — my PyCharm was busy); the card gates exactly as
   `CLAUDE.md` gives them, then the proposed run 5 (tell me first; the card to itself); my yes or no, each alone, to the
   new sentence about the fit in `CLAUDE.md` and to run 5 in its card gate block (and `docs/guide/testing.md`, which
   carries the block verbatim); the documents of Task 7 step 5, with `fit-saving/exit/rulings-for-the-owner.md`: every
   decision made without me in this stage, in plain words, with its cost (the ledger's "Ruling:" lines and the briefs'
   RULED paragraphs; Task 5's R-b — an unreadable file refuses a resume without marking it — is one I should see).
What the build departs from in the design goes into the plan's table and the ledger, with its cost; ask me when it
touches something I decided. Hand over to a new chat, with STATE, the ledger and a prompt like this one, at two thirds
of the room or when the usage rule says stop, whichever comes first.

**Ask me** what is mine: what I decided or what the design leaves to me, what cannot be undone, and anything under
"What must not change" below; one question at a time, in plain words, with a picture where it helps and your
recommendation. Decide the small things yourself, write each as a ruling with its cost, and give me the list of them at
the exit.

**How a task is run** (what the sessions of 21 October learned, beside what the earlier prompts list):

- Readers run NO python while a gate runs, and the owner's one-pytest rule covers scratch copies: a skeptic or a seat
  that needs runs does its reading first and ends its turn with "waiting to run"; the controller hands the machine to
  one of them at a time when the gate has exited (seat three's fault runs, then a fix skeptic's). A seat's runs on the
  card check the card free first (no python, no `PRISM.lock`, the desktop's 2.6 to 2.8 GiB). A message sent to an
  agent while it works waits for its next tool round.
- An Important finding goes to one skeptic (opus) with the controller's conditional rulings in a file; the skeptic
  confirms by runs and writes the fix brief with every edit's exact text; the original implementer is resumed to build
  it; one reader (sonnet) re-reads the fix diff while a fresh gate runs.
- A fix round may fold in a Minor that is the product side of an Important's property, or a missed requirement of the
  plan's own text in the same block (Task 5's cadence), each with a ruling; other Minors go to the ledger.
- A controller's ruling may change a stand-in (Task 5's R-b) when the stand-in's choice would cost a person something
  the plan did not weigh; the implementer then shows the changed part red against the stand-in's choice.
- Times in the ledger are read from `date`, never guessed (this session guessed three and corrected them).
- A `cd` inside a Bash call moves the session's working directory; end such a call at the repository's root.

**Facts the stage now stands on** (measured; the ledger has each): the flow's base distribution keeps a buffer outside
its state dict; sbi's in-process resume double-counts a pass only in the stretch that ends the fit; building the
trainer draws from the CPU generator; `inspect.getsource` reads a file as it is now, so the code is recorded as loaded;
a FitLock's own handle counts as running; a bound given as a numpy float names another simulation-set folder while the
plain identities agree (the click now refuses it); on the training stage's path the kept split state is a second guard
(the cache's resume already restores the processor's generator); after a power cut the kept training's `log.txt`, and
so the resumed posterior's, lacks the last saved pass's line (written inside the save, before that line); the drill
leaves harmless torch and joblib warnings; `fisher_spread` is null on a resumed cache, as before the stage (for the
code audit); the fast gate now holds 1657 tests passed and took 27 to 31 minutes on this evening's loaded machine.

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
