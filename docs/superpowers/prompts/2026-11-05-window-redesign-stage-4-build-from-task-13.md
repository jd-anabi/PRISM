# Prompt for the next chat — the window redesign, stage 4 (steps 1 to 3, plans, Settings, applied against pending): build it from Task 13

Paste everything below the line into a new chat opened on this repository. It is written in the owner's voice by the session of 3 to 5 November 2026 that built Tasks 10, 11 and 12 of stage 4, from `docs/superpowers/prompts/2026-11-03-window-redesign-stage-4-build-from-task-10.md`. What in it is not the owner's own words:

- The owner's words in that session: the pasted prompt it started from, and two answers, both on 4 November: "Yes, ignore it (Recommended)" on a held Enter (departure 247: a box that opens after a press ignores a held, repeating Enter; only a fresh press answers it), and "Yes, with Records (Recommended)" on Windows' own Save window (departure 256: the wait for Enter to be let go before it opens is built once, in the shared opener, with Records' rebuild in stage 6, and covers Live simulation's too). The answers on the plan and "I approve it" are from the sessions before it.
- The standing orders below (how much to spend, the 16 and 26 October quotations and the usage rule, "Ask me what is mine", "What must not change", the standing rules) repeat the owner's words of the earlier prompts, the one it started from last.
- Everything else is that session's record, in the owner's voice as these prompts are, but for "Before you tell me a fact about my computer, check it", which stands from the session that wrote the plan. Every count, time, cost and measurement is that session's or its agents', not something the owner measured.

---

PRISM — the redesign of the window: stage 4, steps 1 to 3, plans, Settings, applied against pending. Build it from Task 13.

Where things stand. Stages 1 (the look), 2 (the shell) and 3 (the run) of the window's redesign are built, and so is the fit-saving stage; STATE has each. Stage 3's exit passed on 29 October with the code final at `e17f1ec`. Stage 4's plan, `docs/superpowers/plans/2026-10-30-window-redesign-stage-4.md` (`d02073b`), was approved on 30 October ("I approve it"), with every departure in its table (126 to 161, also in the design's section 8) and every choice in its "For the owner" section. It has Task 0 (yours), fourteen building tasks and Task 15, the exit (yours, with my hand). TASKS 0 TO 12 ARE COMPLETE, each building task with its review clean and its gate green on the same head. The last three: Task 10 step 3, "Training", its form and "Before you start"; Task 11 step 3 finished (a box answers only Enter pressed afresh, a saved set's differences in words and the question before simulating from zero, the load view, the choice of a kept training before a new one starts); Task 12 plans in the window (step 1's list of plans, the fill of steps 1 to 3, where each value came from, the plan's fact in each step's "Before you start", the save, the run's history and the plan last used). The code stands at `a911572`: 3226 passed, 10 skipped, 5 deselected, 3241 collected, 33 min 5 s. The real-screen run passed last at `5546069` with 9 tests; the set reads 10 since Task 10, and no task since has added or changed a display-marked test or the start-up path, so none is owed before the exit. Departures 232 to 284 are in the plan, the design's section 8 and the ledger. The engine is closed: no later task opens an engine file. Task 13 is next.

My answers on it, each in the plan:
* Home's one "Next" line names a training I kept on purpose as well as one cut off by accident (departure 136).
* Resume after a reopening for a kept narrowing round waits for stage 5 (137); "Continue from batch N" for a round waits with it (152).
* `CLAUDE.md` follows in three edits at the exit, with the two counts the stage makes false shown at the same yes (143).
* I use a second display at times with text above 100 %, so the window reads its room again on a change of screen (144).
* Cascadia Mono is carried (145, 190), `core/gui/assets/fonts/CascadiaMono-Regular.ttf` with its licence (`4db67b0`).
* When a kept training matches what I am about to start, the question "Start a new training?" lists them with Cancel preselected, and Cancel leaves me at their Resume buttons (153).
* Enter presses the button that shows the ring, window-wide (146).
* On a page with nothing saved to read, the foot leaves out the "read at" line and "Check again"; they come back on Model and options when a plan is in use (196).
* The greyed main button stays bold (the plan's Task 8 Step 4).
* A box that opens after a press ignores a held, repeating Enter; only a fresh press answers it (247, 4 November). Built in Task 11 in the focus ring's follower, `core/gui/style.py`; it reaches the plan's two boxes too (274).
* Windows' own Save window waits for Enter to be let go before it opens, built once in the shared opener with Records' rebuild in stage 6, covering Live simulation's too (256, 4 November).

I pushed up to `5573d6f`. Everything after it is not pushed (read `git status -sb`; `main` was 138 ahead at the hand-over), and I push when I choose. The walkthrough rows G1, G2, G6, G7 and G34 to G36 are mine to run on the real screen when it suits me.

Before any work. Nothing is running as this is handed over. Check that:
* the tree is clean;
* the graphics card is free;
* no `python.exe` or `pythonw.exe` is alive but 46012 and 16612, which Windows still lists: exited pythons (`HasExited` True) held by stale handles; leave them;
* no `PRISM.lock` stands beside my settings file;
* my desktop has its ten names (hidden ones counted) and no `PRISM.lnk`.

A PRISM started from `run.bat` or its shortcut (a `pythonw.exe -m core.gui`) is my own: leave it alone and wait. While it is open no gate runs. My real `PRISM.ini` is 3,296 bytes, sha256 79BF94DF…C83E.

How much to spend is decided: do not ask again.
* "In between", as I chose on 12 October: no workflow unless I ask for one in my own words; every agent given its model by name, Sonnet where reading is enough; the review by three readers for every task that builds.
* On 26 October: "Let's just keep the same rigor, it doesn't seem worth it to rush things". So do not propose cutting the loop for speed.
* My words of 16 October stand: "Try to also finish things up at a clean state when usage limits are near their max, I won't buy anymore credits". So read the usage card when you start, tell me where the weekly bar stands, and read it again before every dispatch; do not count on extra usage; near about 90 %, start nothing that cannot be finished below it, and stop at a clean state (nothing half-built, the record written).
* Measured so far: Task 7 about 16 points of the weekly bar (two fix rounds), Task 8 about 9, Task 9 about 11, Task 10 about 8 (50 % to 58 %), Task 11 about 8 (58 % to 66 %), Task 12 about 7 (66 % to 73 %; a skeptic of 3 h 19 min whose trial ran 81 faults, a review that found 8 Important points, a fix round of nineteen rulings and a second of ten). The bar read 73 % at 00:42 on 5 November and resets on 8 November at 04:00 UTC (20:00 on 7 November, my time). Tasks 13 and 14 at 6 to 9 points each may fit below 90 %; the last review and the exit will likely wait for the reset: stop at a clean state when the bar nears 90 % and wait for it.
* Hand over to a new chat, with STATE, the ledger and a prompt like this one, after about three tasks, at two thirds of the room, or when the usage rule says stop, whichever comes first.
* Read the clock before you write a time.

Read first, in this order.
1. `docs/STATE.md`: its head, with "CARRIED TO STAGE 4".
2. `CLAUDE.md`.
3. The plan's head, down to "## Task 0" (the owner's summary, Global Constraints, Review Focus, the File Map, the registry, the standing checks, "How to run this plan", the departures). Each task's section when its turn comes: its skeptic writes the task's one brief from the head and the section; the implementer gets that brief alone.
4. The ledger, `.superpowers/sdd/2026-10-06-window-redesign/progress.md` (gitignored). Its lines are very long: search it, do not read it through. Read its first 47 lines, the sections "## Stage 4 — steps 1 to 3, plans, Settings, applied against pending (plan)" and "(build of ...)", and "## Engine changes landed". Never add a line above the lines that begin "- HAND-OFF" (the last is "- HAND-OFF 2026-10-29"). The build's lines go just above "## Engine changes landed", in the section Task 0 began.
5. The evidence the plan was written from, in the ledger's `stage-4/plan-drafts/` (`reader-a-pages.md` to `recheck-report.md`, with `probes-report.md` and `probes/`): the skeptics' starting points; read them yourself only as a ruling needs.
6. The design's parts as each task cites them (`docs/superpowers/specs/2026-10-05-window-redesign-design/`), the fit-saving design `docs/superpowers/specs/2026-10-21-fit-saving-design.md`, and the prompts of 22, 23, 28 and 30 October and 3 November beside this file, for everything this one does not repeat. Where they differ the later wins, and this one wins over all.

Tools. The scratch folder is `C:\Users\J\AppData\Local\Temp\prism-redesign-scratch\`.
* `s4\gate.sh` (kinds fast, quiet, slow, display, collect) writes into `stage-4/gates/` (labels `s4taskNN-<sha>`, after a fix `s4taskNN-fix1-<sha>`). Run it from Bash, from the scratch folder (`bash s4/gate.sh fast <label>`), in the background; it refuses to start while a `python.exe` is alive (it asks Windows whether each has exited, so 46012 and 16612 do not stop it) or `QT_QPA_PLATFORM`, `QT_SCALE_FACTOR`, `PRISM_ARTIFACTS` or `PRISM_RESOURCES` is set. Before a gate, copy the last `stage-4/gates/artifacts-after-*.txt` as `artifacts-before-<label>.txt` (the listings live in `stage-4/gates/`, not in the scratch folder). After it, `powershell -NoProfile -ExecutionPolicy Bypass -File "C:\Users\J\AppData\Local\Temp\prism-redesign-scratch\s4\ground.ps1" after <label>` (the machine's policy refuses scripts without the flag), then read the tree, `sbi-logs`, `Resources/` (no `Plans`, `Models` holding `SHM.json` and `SHM2.json`, nothing untracked), the desktop's count and `PRISM.lnk`'s absence. Text copies of the tools stand in `stage-4/tools/`.
* The ledger: insert lines above "## Engine changes landed" with `s4\ledger_fix1.py <lines.txt>` (standard library, LF only), or with the Edit tool after reading the place. A time in it is read from `date` first.
* The carried-over list is `.superpowers/sdd/2026-10-06-window-redesign/carried-over.md`: 1488 rows, twice none, missing 0, gone 59 at `a911572`. Its check, from that folder: `/c/Users/J/anaconda3/envs/biophys-env/python.exe -I /c/Users/J/AppData/Local/Temp/prism-redesign-scratch/task01_test_names.py --check carried-over.md`. `s4\rows_born.py <suite.py> "<task words>" "<sha range>" "<what>"` adds appendix rows ONLY for `test_shell.py`, `test_step_chain.py`, `test_look.py`, `test_plans.py`, `test_pages.py`, `test_applied_and_pending.py`, `test_before_you_start.py`, `test_run_text.py`, `test_run_states.py`, `test_stop_and_close.py`, `test_retry.py`; the nine first-table suites (`test_nav_and_gating`, `test_worker_dispatch`, `test_artifact_browser`, `test_settings_persistence`, `test_vt_progress`, `test_simulate`, `test_figures`, `test_user_models`, `test_docs`) take main-table rows by `s4\rows_born_nine.py` (Task 11 first ran `rows_born.py` on `test_nav_and_gating.py` and had to take 111 wrong rows back). A test renamed or gone gets its row closed by hand, naming its successor.
* Departures: a skeptic writes its rows and its corrections in `stage-4/briefs/task-NN-departures.md` as fenced blocks; a short script modelled on `s4\departures_263.py` (rows after the last row, corrections in the plan) or `s4\departures_277.py` (rows, corrections in both tables, the audit's item) applies them byte for byte; `git write-tree` after `git add` must equal the tree the skeptic walked on, and only then is it committed. The audit's list has several sections numbered from 1: guard on its last section only (`s4\departures_277_audit.py`).
* The review seats' standing instructions: `stage-4/briefs/review-seats.md`; what every skeptic is told, `stage-4/briefs/skeptic-standing.md` (through "What Task 11 taught"); a task's own dispatch, `task-NN-skeptic-dispatch.md`, names only that task's questions. A dispatch names the seat, BASE, HEAD, the brief, the rulings, the report, the package (`stage-4/reviews/task-NN-package.txt`, from `git log`, `git diff --stat` and `git diff -U10`) and the output file, with the hardest questions for that task. `SendMessage` to a seat that said "waiting to run" gives it the machine once the gate has exited. A skeptic is resumed by `SendMessage` for its fix round. The harness refuses a subagent's write under `stage-4/reports/` at times ("Subagents should return findings as text"): tell every agent to put its report in its final message if a write is refused, and save it yourself.
* The pictures: `stage-3/tools/exit3-pictures3.py.txt` is the newest model; `stage-2/briefs/task-11-pictures-dispatch.md` and its ledger lines hold the nine checks. The scratch-settings harness for any window script is the ledger's `stage-2/tools/task-07-fix1-harness.py.txt`.
* Copies for a trial: `git -C <repo> -c core.autocrlf=false archive <sha> | tar -x -C <folder>`, and restore `run.sh`'s 100755 in them. A scratch script that imports torch and must stay off the card sets `CUDA_VISIBLE_DEVICES=-1` and asserts CUDA is not available: an EMPTY value does not hide the card here.

What to do. The plan, in its order, one task at a time: Tasks 13 and 14, each through the loop of the plan's "How to run this plan":
* One skeptic (opus) first builds the task as a stand-in in a scratch copy, runs its tests red then green and its one-run faults, and brings what it finds to you; your rulings go INTO the one brief, walked again, rebuilt from the brief's own blocks byte for byte. `skeptic-standing.md` carries every lesson so far; add what each task teaches. Tasks 10 to 12 taught, above all: a test whose records all match what the session or the forms hold cannot tell where a value was read (Task 11's seat three ran eighteen faults, sixteen green; Task 12's sixteen, twelve green, the eleven combinable ones green at once over six suites), so write records and plans that differ from what is held in each thing the page reads, with another model on the form than the one applied; list every writer of each field a row compares with and drive each through its own page; code that changes the screen asks first whether a run is live; a number is said exactly; and a fault that leaves a file that does not parse proves nothing (Task 12's V11): a deleted body becomes `pass`.
* One implementer per task (sonnet), from that one brief.
* On DONE, at the same moment: the one-process fast gate in the background, and three readers: seat one (sonnet) the brief against what was built; seat two (opus) the product before against after; seat three (opus) the tests and the faults they would miss, its fault runs after the gate has exited.
* An Important goes to the task's skeptic, resumed, with your rulings in a file. One fix round, re-read by one reader (sonnet) while a fresh gate runs. A task is complete when its review is clean and its gate green on the same head.
* After every gate, the checks above. After the gate of a task that adds or changes a display-marked test or the start-up path, the real-screen run (hidden windows only).
* Then the last review over the whole stage (three readers) with one fix wave.
* Then Task 15, the exit, which needs me. First, one question to me: the order. Recommended, as I chose for stage 3: the pictures and my questions on them first (on hidden windows; nothing of mine is touched), any change I ask for as a small task of its own with the same loop, then the gates of record once, on the final head. Then: a quiet machine for the gate of record; the card test, five lines and run 5, no diagnostic card (tell me first: the commit, the scratch folder, how long the last run took (31 minutes with its diagnostics), and that nothing else may use the card); the real-screen run; the walkthrough rows written for me; my yes or no to the `CLAUDE.md` edits, each in exact words; the list of everything you decided without me (`stage-4/exit/rulings-for-the-owner.md`). The slow set waits for stage 5's exit.

What the build departs from in the design or the plan goes into the plan's table, the design's section 8 and the ledger (all three), from 285 on, with its cost, committed before the implementer starts; ask me when it touches something I decided.

Carried, each to its task (the ledger's build section has why). Each goes into that task's dispatch or brief.
* Task 13, first, as a small commit on step 1's plan row: Task 12's second fix reader's seven Minors, each with a test and its fault, as `stage-4/briefs/task-13-carried.md` gives them (`_gone` outlives the usual-values press and the plan is reselected unasked when its file comes back; two names of one plan in different case; with nothing applied a plan of the list's model reads "1 change" from itself; one line no fault sees; a test unguarded off Windows; two docstrings; one name bound twice).
* Task 13: `PriorPanel.load_record(ref)` is built for the chain of loads (219): it reads first and asks nothing, and forgets a keyboard press (227). Task 13 builds on `remembered.put_many` and `load_record`. Any test that builds a narrowing round's spec with `training_spec("Narrowing round")` and reads where its run lives passes `place=("inference", "narrowing")`. From Task 12: `apply_quietly` sets the model before it fills, since a model chosen on step 1's form lets the plan in use go (271), and with it a plan's waiting bounds file (265 as corrected, `PriorPanel.let_the_plans_file_go`); a fill made before the model is set is let go. Home's "Continue with these" names the plan in use (the walkthrough's G13 waits for it).
* Tasks 13 and 14: every test that presses an "Open" button stands in the opener itself (`plans.open_folder`, `QDesktopServices.openUrl`), as today's do; no session guard stands on it yet (Task 12's proposal 1, for me at the exit).
* Task 14: a kept row shows the listing's time, which stops at the last saved pass; the note's runs to the stop; the same UTC and raw words as Task 11. A name typed at Resume is lost by a stop before a pass is saved, and the next Resume asks again. A reader of `settings` checks that it is a dict (`plan` and `settings` are not in `SHAPES`), and what `plain` wrote as text stays text in a fill. A run's `fit.save_error` and `fit.wall_seconds` are read only for a run that did not finish. From Task 11: the choice's "press Resume beside it on this page" is untrue until Task 14 builds Resume after a reopening.
* The last review: Task 11's fix reader's three Minors (`_a_kept_training` leaves patience and the epoch limit at their usual values; "no re-listing while live" pinned for the page's own load, not for a narrowing round, and `_run_ended` has no live guard of its own; `_exact` copies `fields._said`); Task 12 sends none (its second fix reader's seven Minors go to Task 13's first commit).
* Stage 5: "the number given on the Training page" for a round that typed one; the narrowed question's "inference on any other observation asks first" is untrue until stage 5.
* Stage 6: the Save window's wait for Enter (my answer, 256); one wording for a prior that is gone ("Simulation set for a prior that is no longer here" on Records against "a prior no longer in Records" in step 3's line).
* Stage 8's help copy (departure 135): the help marks' old words on step 1 and on Prior, `HELP["bounds_source"]` and `HELP["prior"]`, describe controls that are gone.
* The code audit: items 17 to 21 of `docs/superpowers/notes/2026-10-29-the-code-audits-list.md`'s last section (the units file's set order; the bounds file's reader; the window's exposure to Python's collector of cycles; the engine's reading names the other sets in every state; the plan's facts worked out from widgets in a module with the toolkit).
* The exit, for me to hear, everything the earlier prompt listed (the 3 November prompt's "The exit" bullets: `CLAUDE.md:263-265`, Enter on a probe row's ✕, departures 181, 183, 187, 179, the words from Tasks 7 to 9, the two proposals not built that change what I approved, `CLAUDE.md`'s eighth per-test check (213)), and from Tasks 10 to 12:
  * the new words a person meets, each in its departure's row: 233, 234, 249, 250, 251, 264 (with the undated "Last used: retrain" and the full stop after "8 settings in 4 steps"), 265, 268, 269, 279, 280, 282;
  * pictures: "usual" beside every row a plan fills, from the first launch (266); "Show all" under "Before you start" on each of steps 1 to 3 in the low window, since the plan's fact is always there (268); "Cannot start yet:" on a block that no waiting lifts (Task 11; the design's words);
  * choosing another model on step 1's form and choosing back, with nothing applied, lets the plan in use go though every value stays (7.1.6's choice; the plan line, the Training fact, the history and the hint then read no plan): the cure would let the plan go only when step 1 applies another model;
  * step 3's "Changed, not applied" may show for a few milliseconds at a narrowing round's end (Task 11: not settled by reading);
  * at text 2.0 and above, the save's question grows by one line as a name first breaks the rule (273; about 8 lines and 1 test);
  * a session guard on opening a folder, so that no test can open Explorer on my screen (Task 12's proposal 1; about 25 lines and 2 tests), as a small task on my yes;
  * `CLAUDE.md`, for my yes in exact words: the inference tabs remember their pickers' selections, but the plan picker opens on "No plan" at every launch and is never stored (7.3.1); the guards' list lacks `_no_plans_written`; the line for `_streams_put_back` now says less than the check, which also fails a test that left the plans folder bound to another;
  * walkthrough rows, as built: each brief's "The walkthrough's rows for the controller" (`stage-4/briefs/task-10-brief.md`, `task-11-brief.md`, `task-11-fix1-brief.md`, `task-12-brief.md`, `task-12-fix1-brief.md`), with G13 as far as this stage, a plan with problems and a long name, a held Enter on a plan's question, and a narrowing round's posterior on step 3.
* The exit, documents: everything the 3 November prompt listed, and the plan's Task 12 section as built (departures 263 to 284 hold it).

Things to check before they bite.
* THE FIRM REQUIREMENT (my "Leave it to stage 4", departure 125): "at the window's least size, and at every width above it, nothing of the run page is hidden under the form; the form and the run page share the width with neither drawn over the other". Task 6 holds it on the eight pages that keep the frame and on a video export; Tasks 9 and 10 moved Prior and Training to the step page's own leg; Task 12 measured step 1 with an 86-letter plan name at text 1.0 and 1.5. The exit's pictures show it. Departure 187 says where the sum stops fitting: tell me at the exit, with my screens read first.
* The engine is closed: every task stays under `core/gui/`, `tests/` and the documents the exit lists. If one finds it must open an engine file, it stops and you bring it to me.
* No task edits a function of `core.SBI.fit_state.FIT_CODE`. If one finds it must, it stops and you ask me first. The seventeen digests are held by `test_the_named_training_code_is_as_recorded` in every gate.
* `Artifacts/` holds 428 entries as the gates list them, its kinds `diagnostics`, `fdt`, `priors`, `simulations`, no `fits/`; every gate's listing after equals the one before.
* `AskUserQuestion` waits for my answer: ask while long runs go, not before them.
* No file goes into `Resources/Plans`; the guard of Task 1 fails a run that writes there, and since Task 12 `_no_plans_written` binds the window's folder of plans to the session's and holds it empty.
* The plan's sections for Tasks 13 and 14 were cross-checked and every test in them is unrun: expect the trial to correct each, and each review to find Important points the trial had not.

Facts the build stands on (measured; the ledger has each).
* The fast gate holds 3226 passed, 10 skipped, 5 deselected (3241 collected) at `a911572`, about 33 minutes. Trials have run far ahead of the plan's counts (Task 12: 45 functions against the plan's 27, and 20 more in its fix round).
* The last gate of record is still `e17f1ec`'s, 27 min 16 s. The real-screen run passed last at `5546069` (9); the set reads 10.
* The last card run took 31 minutes with its diagnostics. While a gate runs, `Resources/*/sbitest*` and `Resources/Models/SBITEST.json` stand untracked, a test's own: they are gone at its end.
* On this computer `.write_text` makes every file CRLF, so a test that needs exact bytes writes bytes. Replacing a file another program holds open fails with "Access is denied" (error 5), measured in Task 12's fix round.
* In this checkout `run.bat` and `run.sh` read `i/lf w/crlf`.

The rules every agent was given, each learned the hard way:
* pytest is started by ONE tool call at a time, never two in one message;
* no `python -c` beside another python, and readers run NO python while a gate runs; a seat that wants runs reads first, ends its turn with "waiting to run", and is given the machine when the gate has exited;
* a test that reads state the whole application or process shares takes a reading first, or works on a stand-in;
* a `cd` inside a Bash call moves the session's working directory: end it back at the repository root;
* never write a file into the repository while an implementer works there;
* GNU sed reads `\c` in a replacement as a control character: write a path with a backslash through the editor, and check a written document with `file` before committing it;
* times are read from `date`, never guessed.

Ask me what is mine: what I decided or what the design leaves to me, what cannot be undone, and anything under "What must not change" below; one question at a time, in plain words, with a picture where it helps and your recommendation. Decide the small things yourself, write each as a ruling with its cost, and give me the list at the exit. Before you tell me a fact about my computer, check it. Still mine to hear: the rest of `CLAUDE.md`'s rule on what the window remembers (7.3.3), in the stages that make it true, only on my yes.

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
* update `docs/STATE.md` at the end of every session.
