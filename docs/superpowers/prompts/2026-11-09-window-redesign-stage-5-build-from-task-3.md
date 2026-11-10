# Prompt for the next chat — the window redesign, stage 5 (Checks, Inference and the Narrowing round): build it from Task 3

Paste everything below the line into a new chat opened on this repository. It is written in the owner's voice by the session of 9 November 2026 that built Tasks 0, 1 and 2 of stage 5, from `docs/superpowers/prompts/2026-11-09-window-redesign-stage-5-build.md`. What in it is not the owner's own words:

- The owner's words in that session: the pasted prompt it started from, and nothing else (no question was put to the owner). The answers on the plan and "I approve it" are from the session before it.
- The standing orders below (how much to spend, the 16 and 26 October quotations and the usage rule, "Ask me what is mine", "What must not change", the standing rules) repeat the owner's words of the earlier prompts, the one it started from last.
- Everything else is that session's record, in the owner's voice as these prompts are, "Before you tell me a fact about my computer, check it" among them. Every count, time, cost and measurement is that session's or its agents', not something the owner measured.

---

PRISM — the redesign of the window: stage 5, Checks, Inference and the Narrowing round. Build it from Task 3.

Where things stand. Stages 1 (the look), 2 (the shell), 3 (the run) and 4 (steps 1 to 3, plans, Settings, applied against pending) of the window's redesign are built, and so is the fit-saving stage; STATE has each. Stage 5's plan, `docs/superpowers/plans/2026-11-09-window-redesign-stage-5.md` (`ce8d57c`), was approved on 9 November ("I approve it"), with every departure in its table (398 to 424 at approval, 425 to 427 since, each also in the design's section 8) and every choice in its "For the owner" section. It has Task 0 (yours), thirteen building tasks and Task 14, the exit (yours, with my hand). TASKS 0, 1 AND 2 ARE COMPLETE, each building task with its review clean and its gate green on the same head:
* Task 0, the workspace and the baseline: the seventeen digests of the named training code equal stage 4's at its exit; `Artifacts/` 428 entries.
* Task 1, the front (tests only): the stand-in run's stages (`stand_in.stage(...)`, told in their turn among its lines), the masked-probes and round lines, the record writers Checks reads (`write_diagnostic`, `write_calibration`, `write_observation`, `write_simulation_set(chi_masked=)`, `check_results`), each held to the engine's own source; `build_step_page(step=)`. `e60c40c`, then its fix round `8d9a8a6`, `c9ddd81`.
* Task 2, the stage's ONE engine task: an inference names its observation (`observation_name`, checked before anything is spent, `FIELDS` 115), the input-use and across-the-prior checks' six refusals name the posterior, the round's refusal of a narrowed parent given another observation names the observation. `9362c71`, `e8e79f3`, `7b5564c`, the departures `b57ce77`, then its fix round (tests only) `18fc0af`, `c5aeca9`. THE ENGINE IS NOW CLOSED FOR THIS STAGE: no later task edits a file under `core/` outside `core/gui/` (the plan's later sections only read `core/orchestrator.py`, `core/refusals.py`, `core/plans.py`, `core/diagnostics/sbc.py` and `core/Helpers/labels.py`).

My answers on it, each in the plan:
* The Checks headline for a posterior trained with χ off reads "n of 3 deciding checks passed; 3 do not apply to this posterior"; with no posterior in use it keeps "of 6" (C15 as designed).
* After a reopening, a narrowing round kept partway is offered on the Narrowing round page only, with Resume and Delete and what a Resume will load; Home's next line and the rail's mark open that page; the Training step lists only ordinary trainings (399).
* A check's settings and "Before you start" stand open on its page, and one press of its blue button runs it (416).

I pushed up to `cdc80d2`. Everything after it is not pushed (read `git status -sb`), and I push when I choose. The walkthrough rows G1, G2, G6, G7, G8, G9, G11, G12, G13 (as far as plans reach), G14 (its frame half), G15 to G25 and G34 to G53 are mine to run on the real screen when it suits me.

Before any work. Nothing is running as this is handed over. Check that:
* the tree is clean;
* the graphics card is free (the desktop's share was 3,952 MiB at the hand-over);
* no `python.exe` or `pythonw.exe` is alive. Two pytest processes, 46012 and 16612, may still be listed: they exited long ago and stale handles hold them. Leave them alone;
* no `PRISM.lock` stands beside my settings file, whose folder holds `PRISM.ini` and `PRISM.ini.bak-preband`;
* my desktop has its four names (hidden ones counted) and no `PRISM.lnk`.

A PRISM started from `run.bat` or its shortcut (a `pythonw.exe -m core.gui`) is my own: leave it alone and wait. While it is open no gate runs. My real `PRISM.ini` is 3,296 bytes, sha256 79BF94DF…C83E.

How much to spend is decided: do not ask again.
* "In between", as I chose on 12 October: no workflow unless I ask for one in my own words; every agent given its model by name, Sonnet where reading is enough; the review by three readers for every task that builds.
* On 26 October: "Let's just keep the same rigor, it doesn't seem worth it to rush things". So do not propose cutting the loop for speed.
* My words of 16 October stand: "Try to also finish things up at a clean state when usage limits are near their max, I won't buy anymore credits". So read the usage card when you start, tell me where the weekly bar stands, and read it again before every dispatch; do not count on extra usage; near about 90 %, start nothing that cannot be finished below it, and stop at a clean state (nothing half-built, the record written).
* The plan says this stage is likely 50 to 65 million tokens, about one weekly bar or more, so it will run into a second week. The bar read 33 % at the hand-over and resets on 15 November at 04:00 UTC. Task 1 cost about 2.3 M tokens and 2 points of the bar; Task 2 between about 1.9 M and 2.8 M tokens (one agent's count is unclear) and 4 points. The pages' tasks (5, 6, 7, 10, 11) will cost more, as stage 4's did (7 to 9 points each).
* Hand over to a new chat, with STATE, the ledger and a prompt like this one, after about two or three tasks, at two thirds of the room, or when the usage rule says stop, whichever comes first.
* Read the clock before you write a time.

Read first, in this order.
1. `docs/STATE.md`: its head.
2. `CLAUDE.md`.
3. The plan's head, down to "## Task 0" (the owner's summary, Global Constraints, Review Focus, the File Map, the registry, the standing checks, "How to run this plan", the departures, THE TASK OUTLINE). Each later task's section when its turn comes: its skeptic writes the task's one brief from the head and the section; the implementer gets that brief alone.
4. The ledger, `.superpowers/sdd/2026-10-06-window-redesign/progress.md` (gitignored). Its lines are very long: search it, do not read it through. Read its first 47 lines, the sections "## Stage 5 — Checks, Inference and the Narrowing round (plan)" and "(build of …)", and "## Engine changes landed". Never add a line above the lines that begin "- HAND-OFF" (the last is "- HAND-OFF 2026-11-09 (21:48"). The build's lines go just above "## Engine changes landed", at the end of the build section.
5. The evidence the plan was written from, in the ledger's `stage-5/plan-drafts/` (`reader-a-checks.md` to `recheck-report.md`, with `probes-report.md` and `probes/`): the skeptics' starting points; read them yourself only as a ruling needs. Tasks 1 and 2's briefs, reports and reviews are in `stage-5/briefs/`, `stage-5/reports/` and `stage-5/reviews/`.
6. The design's parts as each task cites them (`docs/superpowers/specs/2026-10-05-window-redesign-design/`), the fit-saving design `docs/superpowers/specs/2026-10-21-fit-saving-design.md`, stage 4's plan for the names it built, and the prompts of 30 October, 5 November and 9 November beside this file, for everything this one does not repeat. Where they differ the later wins, and this one wins over all.

Tools. The scratch folder is `C:\Users\J\AppData\Local\Temp\prism-redesign-scratch\`; stage 5's is `s5\`, with text copies of its tools in `stage-5/tools/`.
* `gate.sh <fast|quiet|slow|display|collect> <label>`, run from Bash in the background from `s5\` (`cd /c/Users/J/AppData/Local/Temp/prism-redesign-scratch/s5 && ./gate.sh fast s5taskNN-<sha>`; after a fix `s5taskNN-fix1-<sha>`), writes `stage-5/gates/<kind>-<label>.log`; it refuses to start while a `python.exe` is alive (asking Windows whether each has exited, so 46012 and 16612 do not stop it) or while `QT_QPA_PLATFORM`, `QT_SCALE_FACTOR`, `PRISM_ARTIFACTS` or `PRISM_RESOURCES` is set. `ground.ps1 before|after <label>` (`powershell -NoProfile -ExecutionPolicy Bypass -File ground.ps1 …`; copy the last `artifacts-after-*.txt` as the next `artifacts-before-<label>.txt`); `card\card-gate.ps1` (the card test; store `s5\card-exit`); `exit_fence.py` (the engine fence, against `stage-5/gates/digests-at-start.json` and stage 4's at its exit).
* The ledger: insert lines above "## Engine changes landed" with `s5\ledger_insert.py <lines.txt>` (standard library, LF only) ONLY while nothing else runs python; while an agent has the machine or a suite runs, with the Edit tool after reading the place. A time in it is read from `date` first.
* The carried-over list is `.superpowers/sdd/2026-10-06-window-redesign/carried-over.md`: 1662 rows, missing 0, gone 63 (each closed) at the hand-over. Its check, from that folder: `/c/Users/J/anaconda3/envs/biophys-env/python.exe -I /c/Users/J/AppData/Local/Temp/prism-redesign-scratch/s5/task01_test_names.py --check carried-over.md` (stage 5's copy: it reads `test_checks.py` as a whole suite once the file exists). Born rows: `s5\rows_born.py` for a whole suite, `s5\rows_born_nine.py` for the nine first-table suites, and `s5\rows_named.py <suite.py> <task words> <sha> <what> <test names…>` for the named tests of a suite the list keeps only some rows of (`test_artifact_store.py`, `test_diagnostics.py`, `test_tool.py`, …). A row marked for a task ("stage 5, Tasks 2, 6, …") is marked done when that task completes; a test renamed or gone gets its row closed by hand, naming its successor.
* Departures: a skeptic writes its rows in `stage-5/briefs/task-NN-departures.md` as fenced blocks; a short script applies them byte for byte into the plan's table and the design's section 8 after the last row (model: `s5\departures_425.py`), their ledger lines by hand. Commit them where they do not move the base the skeptic walked its trees on: after the implementer's commits and before the gate (Task 2's way), or before the dispatch with the brief's base named as that commit. The next departure is 428.
* The review seats' standing instructions and what every skeptic is told: `stage-5/briefs/review-seats.md` and `stage-5/briefs/skeptic-standing.md`. A seats dispatch is a file (`stage-5/reviews/task-NN-seats-dispatch.md`, Task 2's the newest model) naming BASE, HEAD, the brief, the rulings, the report, the package (`stage-5/reviews/task-NN-package.txt`, from `git log --format="%h %T %s"`, `git diff --stat` and `git diff -U10`) and the output files, with the hardest questions for that task. `SendMessage` to a seat that said "waiting to run" gives it the machine once the gate has exited. A skeptic is resumed by `SendMessage` for its fix round, with the conditional rulings in a file (`stage-5/briefs/task-NN-fix1-rulings.md`). Tell every agent to put its report in its final message if a write under `stage-5/` is refused, and save it yourself.
* The pictures: `stage-4/briefs/exit-pictures-3-dispatch.md` and its report are the newest model; `stage-2/briefs/task-11-pictures-dispatch.md` holds the nine checks. The scratch-settings harness for any window script is the ledger's `stage-2/tools/task-07-fix1-harness.py.txt`.
* Copies for a trial: `git -C <repo> -c core.autocrlf=false archive <sha> | tar -x -C <folder>`, and restore `run.sh`'s 100755 in them. A scratch script that imports torch and must stay off the card sets `CUDA_VISIBLE_DEVICES=-1` and asserts CUDA is not available: an EMPTY value does not hide the card here.

What to do. The plan, in its order, one task at a time: Tasks 3 to 13, each through the loop of the plan's "How to run this plan":
* One skeptic (opus) first builds the task as a stand-in in a scratch copy, runs its tests red then green and its one-run faults, and brings what it finds to you; your rulings go INTO the one brief, walked again.
* One implementer (sonnet) per task, from that one brief. If its tree differs from the brief's by a slip it cannot amend away, a second commit that brings the tree to the brief's is the cure; never a reset or an amend.
* On DONE, at the same moment: the one-process fast gate in the background, and three readers: seat one (sonnet) the brief against what was built; seat two (opus) the product before against after; seat three (opus) the tests and the faults they would miss, its fault runs after the gate has exited.
* An Important goes to the task's skeptic, resumed, with your conditional rulings in a file. One fix round, re-read by one reader (sonnet) while a fresh gate runs. A task is complete when its review is clean and its gate green on the same head.
* After every gate, the plan's "After each gate" checks. After the gate of a task that adds or changes a display-marked test or the start-up path, the real-screen run (hidden windows only).
* Then the last review over the whole stage (three readers) with one fix wave.
* Then Task 14, the exit, which needs me. First, one question to me: the order. Recommended, as I chose for stages 3 and 4: the pictures and my questions on them first (on hidden windows; nothing of mine is touched), any change I ask for as a small task of its own with the same loop, then the gates of record once, on the final head. Then: a quiet machine for the gate of record; THE SLOW SET, owed for stages 4 and 5 together; the card test, five lines and run 5, with the diagnostic card and one card test through the Checks page's own call (tell me first: the commit, the scratch folder, how long the last run took (25 min 30 s at stage 4's exit, without the diagnostics), and that nothing else may use the card); the real-screen run; the walkthrough rows written for me; the guide's pages for Checks, Inference and the round; my yes or no to the `CLAUDE.md` edits, each in exact words; the list of everything you decided without me (`stage-5/exit/rulings-for-the-owner.md`).

What the build departs from in the design or the plan goes into the plan's table, the design's section 8 and the ledger (all three), from 428 on, with its cost; ask me when it touches something I decided.

Carried, each to its task (the ledger's build section has why). Each goes into that task's skeptic dispatch.
* Task 8's skeptic: the stand-in run accepts an order of stage events the engine never makes (Task 1's proposal 1, about 25 lines and a test: build it if the card test's page must refuse such an order); an `on_stage` that raises is untested; the card test's page can be driven with `stand_in.stage(...)`.
* Task 10's skeptic: a forced observation's record written by `write_observation` holds `forcing_vals` `{}` and `source.recordings` `[]`; an observation written before a stop, a failure, a lack of card memory or the inference's own name taken meanwhile is LEFT BEHIND UNDER ITS NAME (`core/orchestrator.py`, the two compositions), so once the page makes the name required, a second Run of the same form is refused for the observation's name: the stopped or failed page says the observation was kept under that name, and a second Run is met as a taken name before it starts; the step page swaps the store's sentence for the field `name` alone (`core/gui/panels/inference/step_page.py`, about `:638`), so it must cover `observation_name` too, or the person reads "a observation named … rename or delete it first".
* The last review: Task 1's lines test checks `epoch_line` against itself (a tautology).
* Task 14: three guide sentences Task 2 made false, `docs/guide/rules-and-traps.md:125-128` (the ablation and Laplace refusals listed as plain `ValueError`s), `docs/guide/command-line.md:244` (`tsnpe`'s narrowed-parent line now ends ` (--observation)`) and `:248-250` (the cache refusals of `ablation`); the plan's diagnostic card at the exit is kept, though Task 2 moved no tensor.

Things to check before they bite.
* THE FIRM REQUIREMENT (departure 125): "at the window's least size, and at every width above it, nothing of the run page is hidden under the form; the form and the run page share the width with neither drawn over the other". The rebuilt Checks list (with a check's page beside it), Inference and the Narrowing round each join its two tests in the task that builds them; the exit's pictures show it.
* The engine is closed: a task that finds it must open a file under `core/` outside `core/gui/` stops, and you bring it to me. The code audit's list, `docs/superpowers/notes/2026-10-29-the-code-audits-list.md`, now has "# Stage 5: what its tasks send to the code audit's list", items 1 to 7; a later task adds to it only what it finds of the engine, and this stage must not lose that list.
* No task edits a function of `core.SBI.fit_state.FIT_CODE`. If one finds it must, it stops and you ask me first: an edit there makes every kept training unresumable. The seventeen digests are held by `test_the_named_training_code_is_as_recorded`; `stage-5/gates/digests-at-start.json` holds them.
* `core/config.py` is not touched by any task.
* `Artifacts/` held 428 entries as the gates list them on 9 November, its kinds `diagnostics`, `fdt`, `priors`, `simulations`, and no `fits/` folder; every gate's listing after equals the one before. Checks only reads records.
* Tasks 12 and 13 were SWAPPED at the cross-check: Task 12 is "Plans fill every step", Task 13 "A kept round after a reopening". An older note may use the old numbers.
* A page built on the step page is added by hand to the six lists that name the rebuilt pages, in the commit that builds it (the plan's Global Constraints, "The builders and their scans"); `test_the_pages_walked_here_are_every_page_built_on_the_step_page` is the net.
* Nothing is read while a run is live: the Checks reading, `peek` of a simulation set, a cell file's check, the round's observation reading, the kept rounds, the history's check entries. The controller's order (not live, then `status_changed`, then `run_ended`) is load-bearing and pinned.
* The card test from the window is held in the suite by stand-in stages only (on the processor its prior stage takes hours). The timing of a short fit is two runs, the second a turn after the first's `run_ended` (411).
* Nothing passes `observation_name` yet: the command line never will; the Inference page passes it once Task 10 rebuilds it (with `if_name_taken="unnamed"`, the window's choice for a name taken during the run).
* `AskUserQuestion` waits for my answer: ask while long runs go, not before them. I often follow over Remote Control, where the next-step box may not show: put a question that waits on me in the closing message too.
* The plan's task sections are unrun until each task's skeptic runs them. In Tasks 1 and 2 the trials corrected the sections (Task 2: three departures; the plan's tests to rewrite partly the wrong ones) and seat three found an Important point the trial had missed each time. Expect the same; a skeptic's trial may overturn a ruling; let it propose the better cure.

Facts the build stands on (measured; the ledger has each).
* The fast gate: 3607 passed, 11 skipped, 5 deselected at `c5aeca9` (3623 collected; 32 min 40 s). Each of this chat's gates took about 32.5 minutes (faster than the 43 min 28 s of record at `850dc23`, on a machine not certified quiet); the plan expects 50 to 55 minutes by the end.
* The last card run took 25 min 30 s (five lines and run 5, no diagnostics) at `850dc23`; with its diagnostics at stage 3's exit, 31 minutes.
* `core.refusals.FIELDS` holds 115 keys (`observation_name` the last, `None` in both front ends' tables; a plan may not hold it: "it belongs to one run").
* The real store's records read in 1 to 3 ms; the store's lookup by id rescans its kind, so Checks reads each manifest once by its folder (414).
* On this computer `.write_text` makes every file CRLF, so a test that needs exact bytes writes bytes. Replacing a file another program holds open fails with "Access is denied" (error 5). On the offscreen platform `QTest.mouseDClick` presses nothing; the four events Windows sends for a double-click do.
* In this checkout `run.bat` and `run.sh` read `i/lf w/crlf`. An edit to `run.bat` is an edit to the pin in `tests/test_shell.py`, in the same commit.

The rules every agent was given, each learned the hard way:
* pytest is started by ONE tool call at a time, never two in one message;
* no python at all beside another python, not even a script of milliseconds that imports nothing (this chat broke it once, with a ledger script run just after a skeptic was given the machine), and readers run NO python while a gate runs; a seat that wants runs reads first, ends its turn with "waiting to run", and is given the machine when the gate has exited;
* a test that reads state the whole application or process shares takes a reading first, or works on a stand-in;
* a `cd` inside a Bash call moves the session's working directory: end it back at the repository root;
* never write a file into the repository while an implementer or a picture agent works there;
* GNU sed reads `\c` in a replacement as a control character: write a path with a backslash through the editor, and check a written document with `file` before committing it;
* do not pipe Markdown or Python through a bash heredoc: write the file with the Write tool;
* times are read from `date`, never guessed.

Ask me what is mine: what I decided or what the design leaves to me, what cannot be undone, and anything under "What must not change" below; one question at a time, in plain words, with a picture where it helps and your recommendation. Decide the small things yourself, write each as a ruling with its cost, and give me the list at the exit. Before you tell me a fact about my computer, check it. Still mine to hear, at the exit: `CLAUDE.md`'s rule on what the inference pages remember and its "Drive F₀ (N)" (404), which these pages change, only on my yes; and the guide's pages for Checks, Inference and the round.

What must not change without my say-so:
* The science: the models, the simulation, the features, the training and calibration methods, and every setting in `core/config.py`.
* What a simulated row holds, and the simulation set's identity: the finished set `Artifacts/simulations/eda5c2a1e468` stays usable.
* What a fit that is never interrupted computes.
* The rules in `CLAUDE.md`, edited only on my yes.
* Anything on disk under `Artifacts/`, and the console capture `C:\Users\J\prism-retrain-console.log`. Read freely; ask before anything deletes, renames, rewrites or trains on them.
* Nothing signs me out, locks or shuts down the computer, and nothing writes into my real desktop folder.

Do not start the retrain, or any training of real size, in this chat; the tiny trainings in the tests and the card gates are fine.

Standing rules:
* the local `main` branch;
* one commit per logical step, with a short subject, never amended, ending with the Co-Authored-By line the harness provides;
* I push;
* never two pytest processes at once, and never edit a source file while a suite runs;
* a run on the card has the card to itself;
* scratch files go under `C:\Users\J\AppData\Local\Temp\`, and no `.py` file goes under `.superpowers/` or `docs/`;
* update `docs/STATE.md` at the end of every session;
* hand over to a new chat, with STATE, the ledger and a prompt like this one, at two thirds of the room or when the usage rule says stop, whichever comes first.
