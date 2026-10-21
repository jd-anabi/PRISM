# Prompt for the next chat — the fit-saving stage: build it from Task 0

Paste everything below the line into a new chat opened on this repository. It is written in the owner's voice by the
session of 21 October 2026 that finished stage 2's exit, made the design for saving the fit and wrote its plan. What
in it is not the owner's own words:

- The owner's words in that session: the answers at stage 2's exit (Q15, Q16); "Why don't we do it in this chat";
  the design's answers F1 to F10 (the redesign ledger's section "The design for saving the fit"); "The design is
  approved, write the plan"; on the build's cost, "That's okay with me"; "The plan is approved, write the hand-off
  prompt".
- Everything else is that session's record. Every count, time, cost and measurement is that session's or its agents',
  not something the owner measured.

---

PRISM — the fit-saving stage: build it from Task 0.

Where things stand. Stage 2 of the window's redesign (the shell) is built and its exit passed: STATE's paragraph "STAGE
2 OF THE WINDOW REDESIGN IS BUILT". Before stage 3, as I decided on 6 October, came the separate design for saving the
fit as it goes. It is made and I approved it on 21 October (`docs/superpowers/specs/2026-10-21-fit-saving-design.md`,
`3221fcc`): a fit is saved at the end of every pass when its rows are in a kept simulation set, a stopped fit is kept as
a record of its own, and it is resumed by hand, in a new process, into the same fit, only while nothing it was trained
with has changed (the code that builds or trains the network included: the strict check); only a finished fit gives a
posterior; sbi stays the trainer (PRISM hooks into its own training). I approved its plan the same day
(`docs/superpowers/plans/2026-10-21-fit-saving-stage.md`, `e89da79`): Task 0 (yours), six building tasks, Task 7 the
exit (yours, with my hand). Nothing of it is built. I pushed up to `1ad02f5`; everything after it is not pushed, and I
push when I choose. Nothing was running when this was written; check that the graphics card is free and no python is
alive before any work (a `python -m core.gui` started from `run.bat` is my own PRISM: leave it alone and wait; while it
is open a `PRISM.lock` stands beside my settings file, so no gate runs then). My real `PRISM.ini` is 3,296 bytes, sha256
79BF94DF…C83E.

How much to spend is decided: do not ask again. "In between", as I chose on 12 October: no workflow unless I ask for
one in my own words; every agent given its model by name, Sonnet where reading is enough. I know this stage will cost
more than one week's allowance (its plan says 15 to 25 million tokens) and that is fine with me; but my words of 16
October stand: "Try to also finish things up at a clean state when usage limits are near their max, I won't buy anymore
credits". So: read the usage card when you start, tell me where the weekly bar stands, and read it again before every
dispatch; do not count on extra usage; when the weekly bar nears its end (about 90 %), start nothing that cannot be
finished below it, and stop at a clean state. The week resets on 25 October at 04:00 UTC; the bar stood at 44 % when
this was written.

Read first, in this order.

1. `docs/STATE.md` — from the top down to the line that begins "**STAGE 2 OF THE WINDOW REDESIGN IS BUILT", and that
   paragraph.
2. `CLAUDE.md`.
3. The design, whole (about 350 lines), and the plan's head down to "## Task 0"; each task's section when its turn
   comes (cut it out for the agents, as the redesign's stages did).
4. The ledger, `.superpowers/sdd/2026-10-06-window-redesign/progress.md`: its first 47 lines; then the section that
   begins "## The design for saving the fit" to its end. Never add a line above the three lines that begin "- HAND-OFF".
   Its lines are very long: search it, do not read it through.
5. In the ledger's `fit-saving/`: `recon-report.md` (the reader's evidence: the tested restore, the sizes and times, the
   risks, every file:line), `plan-crosscheck.md` and `plan-drafter-report.md` (its section "After the cross-check": where
   each of the twenty findings went).
6. The prompts of 6, 10, 11, 12, 13, 14, 15, 16, 20 and 21 October beside this file, for everything this one does not
   repeat. Where they differ, the later wins, and this one wins over all of them.

Tools. In `C:\Users\J\AppData\Local\Temp\prism-redesign-scratch\`: `gate.sh` (kinds fast, quiet, slow, display, collect)
and `ground.ps1` write into the redesign's `stage-2/gates/` today: Task 0 makes this stage's copies write into
`fit-saving/gates/`; `ledger_append.py` (mode `stage1` appends before "## Engine changes landed", which is now the end of
the fit-saving section; `gates` at the file's end); the scratch-settings harness for any window script is the ledger's
`stage-2/tools/task-07-fix1-harness.py.txt`; the reader's scripts are in `...\fit\`, the drafter's checks in
`...\fit\plan\`. A scratch script that imports torch and must stay off the card sets `CUDA_VISIBLE_DEVICES=-1` and asserts
CUDA is not available: an EMPTY value does not hide the card here.

What to do. The plan, in its order: Task 0; then Tasks 1 to 6, each through the loop the plan's "How to run this plan"
gives (my rule: one fresh implementer from one brief; the one-process fast gate in the background and the review by three
readers started at the same moment; one fix round; complete when the review is clean and the gate green on the same
head; for Tasks 2, 4 and 5 a skeptic first builds the task as a stand-in in a scratch copy); then Task 7, the exit, which
needs me: a quiet machine for the gate of record and the slow set, the card gates (tell me first; the card to itself),
and my yes or no to the new sentence about the fit in `CLAUDE.md` and to the new resume run in its card gate block. What
the build departs from in the design goes into the plan's table and the ledger, with its cost; ask me when it touches
something I decided. Hand over to a new chat, with STATE, the ledger and a prompt like this one, after about three tasks,
at two thirds of the room, or when the usage rule says stop, whichever comes first.

Ask me what is mine: what I decided or what the design leaves to me, what cannot be undone, and anything under "What must
not change" in the earlier prompts; one question at a time, in plain words, with a picture where it helps and your
recommendation. Decide the small things yourself, write each as a ruling with its cost, and give me the list of them at
the exit.

What must not change without my say-so: the science (the models, the simulation, the features, the training and
calibration methods, every setting in `core/config.py`); what a simulated row holds and the simulation set's identity
(the finished set `Artifacts/simulations/eda5c2a1e468` stays usable); what a fit that is never interrupted computes; the
rules in `CLAUDE.md` (edited only on my yes); anything on disk under `Artifacts/` and the console capture
`C:\Users\J\prism-retrain-console.log` (read freely; ask before anything deletes, renames, rewrites or trains on them).

Do not start the retrain, or any training of real size, in this chat; the tiny trainings in the tests and the card gates
are fine. Standing rules: the local `main` branch; one commit per logical step, a short subject, never amended, ending
with the Co-Authored-By line the harness provides; I push. Never two pytest processes at once; never edit a source file
while a suite runs; a run on the card has the card to itself. Scratch files under `C:\Users\J\AppData\Local\Temp\`; no
`.py` file under `.superpowers/` or `docs/`. Update `docs/STATE.md` at the end of every session.
