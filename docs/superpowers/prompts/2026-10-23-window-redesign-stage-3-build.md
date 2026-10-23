# Prompt for the next chat — the window redesign, stage 3 (the run): build it from Task 0

Paste everything below the line into a new chat opened on this repository. It is written in the owner's voice by the
session of 22 and 23 October 2026 that wrote stage 3's plan. What in it is not the owner's own words:

- The owner's words in that session: the pasted prompt it started from
  (`docs/superpowers/prompts/2026-10-22-window-redesign-stage-3-plan.md`); the four answers to its questions, each the
  option read ("Resume on the page (Recommended)"; "On the page once stopped (Recommended)"; "Both, plus a shortcut";
  "Yes, about 15 min (Recommended)"); a question on the redesign's overall progress; and "I approve it".
- The standing orders below (how much to spend, the 16 October quotation and the usage rule, "Ask me what is mine",
  "What must not change", the standing rules) repeat the owner's words of the earlier prompts, the 22 October one last.
- Everything else is that session's record. Every count, time, cost and measurement is that session's or its agents',
  not something the owner measured.

---

PRISM — the redesign of the window: stage 3, the run. Build it from Task 0.

Where things stand. Stages 1 (the look) and 2 (the shell) of the window's redesign are built, and so is the fit-saving
stage (a training saved after every epoch and resumable); STATE has each. Stage 3's plan,
`docs/superpowers/plans/2026-10-22-window-redesign-stage-3.md` (`f5faaaa`), is written and I approved it on 23 October
("I approve it"), with every departure in its table (83 to 114, also in the design's section 8) and every choice in its
"For the owner" section. It has Task 0 (yours), fourteen building tasks and Task 15, the exit (yours, with my hand).
Nothing of it is built. My answers on it: Resume on the run's own page after a stop that was kept or a failure in the
fit; "Keep this training so you can resume it later?" asked on the run's page once the stop has landed, never a pop-up,
kept when unanswered; the launcher finds PRISM's Python itself, says so in a Windows message when it cannot, and makes a
desktop shortcut (made once and never again, the plan's ruling); one extra real-screen row at the exit, about fifteen minutes, with a Windows sign-out I start and
cancel (no sign-out completed). I pushed up to `1ad02f5`; everything after it is not pushed, and I push when I choose.
Nothing was running when this was written; check that the graphics card is free, no python is alive and no `PRISM.lock`
stands beside my settings file before any work (a `python -m core.gui` started from `run.bat` is my own PRISM: leave it
alone and wait; while it is open no gate runs). My real `PRISM.ini` is 3,296 bytes, sha256 79BF94DF…C83E.

How much to spend is decided: do not ask again. "In between", as I chose on 12 October: no workflow unless I ask for one
in my own words; every agent given its model by name, Sonnet where reading is enough; the review by three readers for
every task that builds. The plan says this stage is likely 45 to 60 million tokens, more than a week's bar; my words of
16 October stand: "Try to also finish things up at a clean state when usage limits are near their max, I won't buy
anymore credits". So: read the usage card when you start, tell me where the weekly bar stands, and read it again before
every dispatch; do not count on extra usage; when the weekly bar nears its end (about 90 %), start nothing that cannot
be finished below it, and stop at a clean state (nothing half-built, the record written). The week resets on 25 October
at 04:00 UTC; the bar stood at 73 % on the morning of 23 October (the writing session's guess: room for Task 0 and
about two building tasks before the reset).

Read first, in this order.

1. `docs/STATE.md` — its head, and the paragraph "THE PLAN OF STAGE 3"; the paragraph "THE FIT-SAVING STAGE'S LAST
   REVIEW AND EXIT" for the code audit's list, which this stage must not lose.
2. `CLAUDE.md`.
3. The plan's head, down to "## Task 0" (the owner's summary, Global Constraints, Review Focus, the File Map, the
   registry, the standing checks, "How to run this plan", the departures), and Task 0's section; each later task's
   section when its turn comes (its skeptic, or for Task 1 its reader, writes the task's one brief from the head and
   the section; the implementer gets that brief alone).
4. The ledger, `.superpowers/sdd/2026-10-06-window-redesign/progress.md` (gitignored): its first 47 lines; then the
   section "## Stage 3 — the run (plan)" whole (fourteen long lines). Never add a line above the lines that begin
   "- HAND-OFF". Its lines are very long: search it, do not read it through.
5. The evidence the plan was written from, in the ledger's `stage-3/plan-drafts/`: `reader-1-run-today.md`,
   `reader-2-engine.md`, `reader-3-tests.md`, `probes-report.md`, `drafter-report.md` (its section 5: the revision after
   the probes; section 6: where each cross-check finding went), `crosscheck-report.md`, `prompt-check.md`. They are the skeptics' starting points; read them yourself only
   as a ruling needs.
6. The design's parts as each task cites them (`docs/superpowers/specs/2026-10-05-window-redesign-design/`, above all
   `06-run.md`), the fit-saving design `docs/superpowers/specs/2026-10-21-fit-saving-design.md`, and the prompts of 14,
   20, 21 and 22 October beside this file, for everything this one does not repeat. Where they differ, the later wins,
   and this one wins over all of them.

Tools. In `C:\Users\J\AppData\Local\Temp\prism-redesign-scratch\`: `gate.sh` (kinds fast, quiet, slow, display,
collect) and `ground.ps1` write into the ledger's `fit-saving/gates/` today: Task 0 makes them write into
`stage-3/gates/` (labels `s3taskNN-<sha>`, after a fix `s3taskNN-fix1-<sha>`) and keeps text copies in `stage-3/tools/`
(today's copies are in `fit-saving/tools/`); `gate.sh` refuses to start while any `python.exe` is alive, so nothing
else runs python at its start; `ledger_append.py` mode `stage1` appends at the end of stage 3's section (just before
"## Engine changes landed"), `deviations` after the last row of the departures table, `gates` at the file's end;
`task01_test_names.py --check carried-over.md` checks the carried-over list (the ledger's `carried-over.md`; its last
line must read "twice []; missing 0: []"); `pictures.py` is stage 1's picture recipe and `s2\t11-pictures\pictures2.py`
stage 2's rail-aware one (copies in `stage-2/tools/task-11-*.txt`), which the exit teaches the run page. The
scratch-settings harness for any window script is the ledger's `stage-2/tools/task-07-fix1-harness.py.txt`. Stage 3's
scratch folder is `s3\`; the plan's probe scripts are in `s3plan\` (copies in `stage-3/tools/`). A scratch script that imports torch and must stay off the card sets `CUDA_VISIBLE_DEVICES=-1` and
asserts CUDA is not available: an EMPTY value does not hide the card here.

What to do. The plan, in its order, one task at a time: Task 0; then Tasks 1 to 14, each through the loop of the plan's
"How to run this plan": for every task that touches threads, the run host or the engine (Tasks 2 to 14), one skeptic
(opus) first builds it as a stand-in in a scratch copy, runs its tests red then green and its one-run faults, and brings
what it finds to you; your rulings go INTO the one brief, walked again; Task 1 is prepared by one reader (sonnet). Then
one implementer per task from that one brief; on DONE the one-process fast gate in the background and three readers at
the same moment (seat one sonnet, the brief against what was built; seat two opus, the product before against after;
seat three opus, the tests and the faults they see, its fault runs after the gate has exited); an Important to one
skeptic with your conditional rulings in a file; one fix round, re-read by one reader while a fresh gate runs; a task is
complete when its review is clean and its gate green on the same head. After every gate, the plan's "After each gate"
checks; after the gates of Tasks 12 and 14, the real-screen run (hidden windows only). Then the last review over the
whole stage (three readers) with one fix wave. Then Task 15, the exit, which needs me: a quiet machine for the gate of record and then the
slow set; the card test with the diagnostic card and run 5 (tell me first: the commit, the scratch folder, how long the
last run took, and that nothing else may use the card); the real-screen test run; the pictures, and my questions on
them; the walkthrough rows, my extra row among them; my yes or no to `CLAUDE.md`'s new launch line and to its
corrected counts of suites and guards; and the list of everything you decided without me. What the build departs from in the design or the plan goes into the plan's table, the design's section 8 and the
ledger, with its cost; ask me when it touches something I decided. Hand over to a new chat, with STATE, the ledger and a
prompt like this one, after about three tasks, at two thirds of the room, or when the usage rule says stop, whichever
comes first.

Two things to check before they bite. Before Task 3 commits its edit of the named training code (the one edit of the
stage inside `core.SBI.fit_state.FIT_CODE`, which makes every training kept before it unresumable), confirm that
`Artifacts/fits/` still does not exist (it did not on 23 October; the plan does not say this, so carry it into Task 3's
brief); if a kept training is there, stop and ask me. And `AskUserQuestion` waits for my
answer: ask while long runs go, not before them.

Facts the build stands on (measured; the ledger has each). The fast gate holds 1662 tests (1672 collected) and took 25
to 30 minutes this week; the plan expects about 1,825 at the exit. The slow set took 1 h 33 min on 22 October. The
desktop's share of the card (`nvidia-smi --query-gpu=memory.used`) moves between about 2.3 and 3.0 GiB; a check
script's bound must allow for that. Readers run NO python while a gate runs, and the one-pytest rule covers scratch copies: a seat that wants runs
reads first, ends its turn with "waiting to run", and is given the machine when the gate has exited. Times in the ledger
are read from `date`, never guessed. A `cd` inside a Bash call moves the session's working directory.

Ask me what is mine: what I decided or what the design leaves to me, what cannot be undone, and anything under "What
must not change" below; one question at a time, in plain words, with a picture where it helps and your recommendation.
Decide the small things yourself, write each as a ruling with its cost, and give me the list at the exit. Still mine to
hear: the replacement wording of `CLAUDE.md`'s launch line (the design's part 13.3.4, adjusted to what the launcher
does), at this stage's exit, only on my yes; its rule on what the window remembers (part 7.3.3), in stage 4; bundling
Cascadia Mono so every computer shows it (never asked).

What must not change without my say-so: the science (the models, the simulation, the features, the training and
calibration methods, every setting in `core/config.py`); what a simulated row holds and the simulation set's identity
(the finished set `Artifacts/simulations/eda5c2a1e468` stays usable); what a fit that is never interrupted computes; the
rules in `CLAUDE.md` (edited only on my yes); anything on disk under `Artifacts/` and the console capture
`C:\Users\J\prism-retrain-console.log` (read freely; ask before anything deletes, renames, rewrites or trains on them).
Nothing signs me out, locks or shuts down the computer, and nothing writes into my real desktop folder: the launcher's
shortcut is tested in a scratch folder until I run it myself.

Do not start the retrain, or any training of real size, in this chat; the tiny trainings in the tests and the card gates
are fine. Standing rules: the local `main` branch; one commit per logical step, a short subject, never amended, ending
with the Co-Authored-By line the harness provides; I push. Never two pytest processes at once; never edit a source file
while a suite runs; a run on the card has the card to itself. Scratch files under `C:\Users\J\AppData\Local\Temp\`; no
`.py` file under `.superpowers/` or `docs/`. Update `docs/STATE.md` at the end of every session.
