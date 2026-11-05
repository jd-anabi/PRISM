# Prompt for the next chat — the window redesign, stage 4 (steps 1 to 3, plans, Settings, applied against pending): build it from Task 14

Paste everything below the line into a new chat opened on this repository, after the weekly usage bar has reset (8 November at 04:00 UTC, 20:00 on 7 November PST). It is written in the owner's voice by the session of 5 November 2026 that built Task 13 of stage 4, from `docs/superpowers/prompts/2026-11-05-window-redesign-stage-4-build-from-task-13.md`. What in it is not the owner's own words:

- The owner's words in that session: the pasted prompt it started from, and one question, "Can I use my GPU while you work?" (answered: light use at any time; heavy use waits while a gate or a card test runs). No decision of the owner's was asked or given in it. The answers on the plan and "I approve it" are from the sessions before it.
- The standing orders below (how much to spend, the 16 and 26 October quotations and the usage rule, "Ask me what is mine", "What must not change", the standing rules) repeat the owner's words of the earlier prompts, the one it started from last.
- Everything else is that session's record, in the owner's voice as these prompts are, but for "Before you tell me a fact about my computer, check it", which stands from the session that wrote the plan. Every count, time, cost and measurement is that session's or its agents', not something the owner measured.

---

PRISM — the redesign of the window: stage 4, steps 1 to 3, plans, Settings, applied against pending. Build it from Task 14.

Where things stand. Stages 1 (the look), 2 (the shell) and 3 (the run) of the window's redesign are built, and so is the fit-saving stage; STATE has each. Stage 3's exit passed on 29 October with the code final at `e17f1ec`. Stage 4's plan, `docs/superpowers/plans/2026-10-30-window-redesign-stage-4.md` (`d02073b`), was approved on 30 October ("I approve it"), with every departure in its table (126 to 161, also in the design's section 8) and every choice in its "For the owner" section. It has Task 0 (yours), fourteen building tasks and Task 15, the exit (yours, with my hand). TASKS 0 TO 13 ARE COMPLETE, each building task with its review clean and its gate green on the same head. The last: Task 13, Home's "Continue with these" (a first commit of Task 12's last Minors; a chain of loads that goes on only when each load has ended, `core/gui/load_chain.py`; step 1 applied by code with a plan filled once the model stands; what was in use last time read from the settings file, each pick left out when its load would be refused, named in one sentence and loaded with one press while Home stays in sight; one fix round). The code stands at `b51a27f`: 3317 passed, 10 skipped, 5 deselected, 3332 collected, 34 min 2 s. The real-screen set read 10 passed at `e7508da`, and 9 passed, 1 skipped at `b51a27f`: the keep-awake check skips by its own design while another program keeps the computer awake. Departures 287 to 306 are in the plan, the design's section 8 and the ledger. The engine is closed: no later task opens an engine file. Task 14 is next, the stage's largest: Continue and Resume after a reopening.

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
* A box that opens after a press ignores a held, repeating Enter; only a fresh press answers it (247, 4 November).
* Windows' own Save window waits for Enter to be let go before it opens, built once in the shared opener with Records' rebuild in stage 6, covering Live simulation's too (256, 4 November).
* Home's "Continue with these" asks nothing: the press is the explicit act (7.1.7, approved with the plan; pinned as built by 298). Task 13's proposal 1 would ask first when the press would set values I typed and did not apply: mine to hear at the exit.

I pushed up to `5573d6f`. Everything after it is not pushed (read `git status -sb`; `main` was 146 ahead at the hand-over), and I push when I choose. The walkthrough rows G1, G2, G6, G7 and G34 to G36 are mine to run on the real screen when it suits me.

Before any work. Nothing is running as this is handed over. Check that:
* the tree is clean;
* the graphics card is free;
* no `python.exe` or `pythonw.exe` is alive but 46012 and 16612, which Windows still lists: exited pythons (`HasExited` True) held by stale handles; leave them;
* no `PRISM.lock` stands beside my settings file;
* my desktop has its eleven names (hidden ones counted; the eleventh, a paper I downloaded on 5 November, is mine) and no `PRISM.lnk`.

A PRISM started from `run.bat` or its shortcut (a `pythonw.exe -m core.gui`) is my own: leave it alone and wait. While it is open no gate runs. My real `PRISM.ini` is 3,296 bytes, sha256 79BF94DF…C83E.

How much to spend is decided: do not ask again.
* "In between", as I chose on 12 October: no workflow unless I ask for one in my own words; every agent given its model by name, Sonnet where reading is enough; the review by three readers for every task that builds.
* On 26 October: "Let's just keep the same rigor, it doesn't seem worth it to rush things". So do not propose cutting the loop for speed.
* My words of 16 October stand: "Try to also finish things up at a clean state when usage limits are near their max, I won't buy anymore credits". So read the usage card when you start, tell me where the weekly bar stands, and read it again before every dispatch; do not count on extra usage; near about 90 %, start nothing that cannot be finished below it, and stop at a clean state (nothing half-built, the record written).
* Measured so far: Task 10 about 8 points of the weekly bar, Task 11 about 8, Task 12 about 7, Task 13 about 7 (73 % to 80 %; a skeptic of 2 h 15 min with 62 faults, a review with 6 Important points, one fix round of seven product rulings and nine test rulings). Task 14 is larger (four commits, +21 functions in the plan, a card test): expect about 9 to 11 points. The bar read 80 % at 15:52 on 5 November and resets on 8 November at 04:00 UTC; after the reset Task 14, the last review and the exit may all fit in one week.
* Hand over to a new chat, with STATE, the ledger and a prompt like this one, after about three tasks, at two thirds of the room, or when the usage rule says stop, whichever comes first.
* Read the clock before you write a time.

Read first, in this order.
1. `docs/STATE.md`: its head, with "CARRIED TO STAGE 4".
2. `CLAUDE.md`.
3. The plan's head, down to "## Task 0" (the owner's summary, Global Constraints, Review Focus, the File Map, the registry, the standing checks, "How to run this plan", the departures), and Task 14's section: its skeptic writes the task's one brief from the head and the section; the implementer gets that brief alone.
4. The ledger, `.superpowers/sdd/2026-10-06-window-redesign/progress.md` (gitignored). Its lines are very long: search it, do not read it through. Read its first 47 lines, the sections "## Stage 4 — steps 1 to 3, plans, Settings, applied against pending (plan)" and "(build of ...)", and "## Engine changes landed". Never add a line above the lines that begin "- HAND-OFF" (the last is "- HAND-OFF 2026-10-29"). The build's lines go just above "## Engine changes landed", in the section Task 0 began.
5. `stage-4/briefs/task-14-skeptic-dispatch.md` (written by the session that built Task 13: the eight questions and the carried items) and `stage-4/briefs/task-14-carried.md` (the first commit's three items, Task 13 as built, Task 13's proposal 2 for the skeptic, the exit's items from Task 13). Send the dispatch as it stands, with the base head, unless your own reading finds a question it lacks.
6. The evidence the plan was written from, in the ledger's `stage-4/plan-drafts/` (with `probes-report.md` and `probes/`): the skeptics' starting points; read them yourself only as a ruling needs.
7. The design's parts as each task cites them (`docs/superpowers/specs/2026-10-05-window-redesign-design/`), the fit-saving design `docs/superpowers/specs/2026-10-21-fit-saving-design.md`, and the prompts of 22, 23, 28 and 30 October and 3 and 5 November beside this file, for everything this one does not repeat. Where they differ the later wins, and this one wins over all.

Tools. The scratch folder is `C:\Users\J\AppData\Local\Temp\prism-redesign-scratch\`.
* `s4\gate.sh` (kinds fast, quiet, slow, display, collect) writes into `stage-4/gates/` (labels `s4taskNN-<sha>`, after a fix `s4taskNN-fix1-<sha>`). Run it from Bash, from the scratch folder (`bash s4/gate.sh fast <label>`), in the background; it refuses to start while a `python.exe` is alive (it asks Windows whether each has exited, so 46012 and 16612 do not stop it) or `QT_QPA_PLATFORM`, `QT_SCALE_FACTOR`, `PRISM_ARTIFACTS` or `PRISM_RESOURCES` is set. Before a gate, copy the last `stage-4/gates/artifacts-after-*.txt` (now `artifacts-after-s4task13-fix1-b51a27f.txt`) as `artifacts-before-<label>.txt`. After it, `powershell -NoProfile -ExecutionPolicy Bypass -File "C:\Users\J\AppData\Local\Temp\prism-redesign-scratch\s4\ground.ps1" after <label>`, then read the tree, `sbi-logs`, `Resources/` (no `Plans`, `Models` holding `SHM.json` and `SHM2.json`, nothing untracked), the desktop's count and `PRISM.lnk`'s absence. Text copies of the tools stand in `stage-4/tools/`.
* The ledger: insert lines above "## Engine changes landed" with `s4\ledger_fix1.py <lines.txt>` (standard library, LF only) only while no suite runs; while one runs, with the Edit tool after reading the place. A time in it is read from `date` first.
* The carried-over list is `.superpowers/sdd/2026-10-06-window-redesign/carried-over.md`: 1530 rows, twice none, missing 0, gone 59 at `b51a27f`. Its check, from that folder: `/c/Users/J/anaconda3/envs/biophys-env/python.exe -I /c/Users/J/AppData/Local/Temp/prism-redesign-scratch/task01_test_names.py --check carried-over.md`. `s4\rows_born.py <suite.py> "<task words>" "<sha range>" "<what>"` adds appendix rows ONLY for `test_shell.py`, `test_step_chain.py`, `test_look.py`, `test_plans.py`, `test_pages.py`, `test_applied_and_pending.py`, `test_before_you_start.py`, `test_run_text.py`, `test_run_states.py`, `test_stop_and_close.py`, `test_retry.py`; the nine first-table suites (`test_nav_and_gating`, `test_worker_dispatch`, `test_artifact_browser`, `test_settings_persistence`, `test_vt_progress`, `test_simulate`, `test_figures`, `test_user_models`, `test_docs`) take main-table rows by `s4\rows_born_nine.py`. A test renamed or gone gets its row closed by hand, naming its successor (Task 14 retires `tests/test_run_text.py:795`'s test).
* Departures: a skeptic writes its rows and its corrections in `stage-4/briefs/task-NN-departures.md` as fenced blocks; a short script applies them byte for byte: `s4\departures_287.py` (rows after the last row, corrections in the plan alone) or `s4\departures_300.py` (rows, corrections in both tables), or `departures_277.py` when the audit's list gains an item (guard on its last section only, `s4\departures_277_audit.py`). `git write-tree` after `git add` must equal the tree the skeptic walked on, and only then is it committed.
* The review seats' standing instructions: `stage-4/briefs/review-seats.md`; what every skeptic is told, `stage-4/briefs/skeptic-standing.md` (through "What Task 13 taught"). A dispatch names the seat, BASE, HEAD, the brief, the rulings, the report, the package (`stage-4/reviews/task-NN-package.txt`, from `git log --format="%h %T %s"`, `git diff --stat` and `git diff -U10`) and the output file, with the hardest questions for that task. `SendMessage` to a seat that said "waiting to run" gives it the machine once the gate has exited. A skeptic is resumed by `SendMessage` for its fix round. Tell every agent to put its report in its final message if a write under `stage-4/reports/` is refused, and save it yourself.
* The pictures: `stage-3/tools/exit3-pictures3.py.txt` is the newest model; `stage-2/briefs/task-11-pictures-dispatch.md` and its ledger lines hold the nine checks. The scratch-settings harness for any window script is the ledger's `stage-2/tools/task-07-fix1-harness.py.txt`.
* Copies for a trial: `git -C <repo> -c core.autocrlf=false archive <sha> | tar -x -C <folder>`, and restore `run.sh`'s 100755 in them. A scratch script that imports torch and must stay off the card sets `CUDA_VISIBLE_DEVICES=-1` and asserts CUDA is not available: an EMPTY value does not hide the card here. One test of Task 13 (`test_home_reads_the_picks_once_until_something_changes`) holds only with the card visible: Task 14's first commit cures it.

What to do. Task 14 through the loop of the plan's "How to run this plan", then the last review, then the exit:
* One skeptic (opus) first builds the task as a stand-in in a scratch copy, runs its tests red then green and its one-run faults, and brings what it finds to you; your rulings go INTO the one brief, walked again, rebuilt from the brief's own blocks byte for byte. `skeptic-standing.md` carries every lesson so far; add what Task 14 teaches. Task 13 taught, above all: one model in every test hides every place a model's name is read (eleven of seat three's fourteen faults green, five at once, every test on Nadrowski); read back what the person sees, not what the code was handed; a guard tested only where another guard already refuses is not tested; a press that fills changes more than the forms, so read every widget that shows what was filled; a fault's command must run what it names (pytest keeps the last of two `-k`).
* One implementer (sonnet), from that one brief. If its tree differs from the brief's by a slip it cannot amend away, a second commit that brings the tree to the brief's is the cure (Task 13's fix round did this for one blank line); never a reset or an amend.
* On DONE, at the same moment: the one-process fast gate in the background, and three readers: seat one (sonnet) the brief against what was built; seat two (opus) the product before against after; seat three (opus) the tests and the faults they would miss, its fault runs after the gate has exited.
* An Important goes to the task's skeptic, resumed, with your rulings in a file. One fix round, re-read by one reader (sonnet) while a fresh gate runs. A task is complete when its review is clean and its gate green on the same head.
* After every gate, the checks above. After the gate of a task that adds or changes a display-marked test or the start-up path, the real-screen run (hidden windows only).
* Then the last review over the whole stage (three readers) with one fix wave.
* Then Task 15, the exit, which needs me. First, one question to me: the order. Recommended, as I chose for stage 3: the pictures and my questions on them first (on hidden windows; nothing of mine is touched), any change I ask for as a small task of its own with the same loop, then the gates of record once, on the final head. Then: a quiet machine for the gate of record; the card test, five lines and run 5, no diagnostic card (tell me first: the commit, the scratch folder, how long the last run took (31 minutes with its diagnostics), and that nothing else may use the card); the real-screen run; the walkthrough rows written for me; my yes or no to the `CLAUDE.md` edits, each in exact words; the list of everything you decided without me (`stage-4/exit/rulings-for-the-owner.md`). The slow set waits for stage 5's exit.

What the build departs from in the design or the plan goes into the plan's table, the design's section 8 and the ledger (all three), from 307 on, with its cost, committed before the implementer starts; ask me when it touches something I decided.

Carried, each to its task (the ledger's build section has why). Each goes into that task's dispatch or brief.
* Task 14: all of `stage-4/briefs/task-14-carried.md` and the dispatch's "Carried into this task" (the first commit's three items; Task 13 as built; Task 13's proposal 2, Home saying which load the press could not make, for the skeptic to say whether Task 14's rows cover it; the 3 November prompt's items: a kept row's time stops at the last saved pass, the note's runs to the stop, the same UTC and raw words as Task 11, a name typed at Resume lost by a stop before a pass is saved, a reader of `settings` checks it is a dict, text stays text in a fill, `fit.save_error` and `fit.wall_seconds` only for a run that did not finish, the choice's "press Resume beside it on this page"; the "Open" buttons' stand-in openers; G23 to G25 with G25's two questions of stage 3).
* The last review: Task 11's fix reader's three Minors (`_a_kept_training` leaves patience and the epoch limit at their usual values; "no re-listing while live" pinned for the page's own load, not for a narrowing round, and `_run_ended` has no live guard of its own; `_exact` copies `fields._said`).
* Stage 5: "the number given on the Training page" for a round that typed one; the narrowed question's "inference on any other observation asks first" is untrue until stage 5.
* Stage 6: the Save window's wait for Enter (my answer, 256); one wording for a prior that is gone ("Simulation set for a prior that is no longer here" on Records against "a prior no longer in Records" in step 3's line).
* Stage 8's help copy (departure 135): the help marks' old words on step 1 and on Prior, `HELP["bounds_source"]` and `HELP["prior"]`, describe controls that are gone.
* The code audit: items 17 to 21 of `docs/superpowers/notes/2026-10-29-the-code-audits-list.md`'s last section.
* The exit, for me to hear: everything the 5 November prompt (`...-build-from-task-13.md`) listed under "The exit, for me to hear", and from Task 13, as `task-14-carried.md`'s "For the owner at the exit" gives them:
  * proposal 1: ask before "Continue with these" sets values I typed and did not apply (it touches my 7.1.7; about 25 lines and 2 tests, a small task on my yes);
  * the new words of 289 and 292 ("Next: continue with what was in use last time", the "Last time: …" sentence, "Nothing is loaded until you press the button.", the button "Continue with these");
  * what a person meets: 288, 291 and 302 (a pick the stage would refuse is not named), 293 (a load that starts nothing ends the press with no box: proposal 2 if Task 14 does not cover it), 295 and 303, 297, 298, 301 (after the press with a plan, step 1's list stands on it), 304 (a double-click on Home's main button presses it once, whatever it says);
  * not changed, to hear: a prior loaded on a bounds file of another name than it was built on is left out though it would load; after "Continue with these" by Enter, a second Enter opens the run page;
  * walkthrough rows, as built: `stage-4/briefs/task-13-brief.md`'s section 7 (G11's Home half), beside Tasks 10 to 12's.
* The exit, documents: everything the 5 November prompt listed, and the plan's Task 13 section as built (departures 287 to 306 hold it).

Things to check before they bite.
* THE FIRM REQUIREMENT (my "Leave it to stage 4", departure 125): "at the window's least size, and at every width above it, nothing of the run page is hidden under the form; the form and the run page share the width with neither drawn over the other". Task 14's list of kept trainings on the Training step is the next page to hold it, with a long list and a long name. The exit's pictures show it. Departure 187 says where the sum stops fitting: tell me at the exit, with my screens read first.
* The engine is closed: every task stays under `core/gui/`, `tests/` and the documents the exit lists. Task 14 reads `core.SBI.fit_state` (`kept_fits`, `KeptFit`) and `orchestrator.build_posterior` and edits neither; if it finds it must open an engine file, it stops and you bring it to me.
* No task edits a function of `core.SBI.fit_state.FIT_CODE`. If one finds it must, it stops and you ask me first. The seventeen digests are held by `test_the_named_training_code_is_as_recorded` in every gate.
* `Artifacts/` holds 428 entries as the gates list them, its kinds `diagnostics`, `fdt`, `priors`, `simulations`, no `fits/`; every gate's listing after equals the one before.
* `AskUserQuestion` waits for my answer: ask while long runs go, not before them.
* No file goes into `Resources/Plans`; the guard of Task 1 fails a run that writes there, and `_no_plans_written` binds the window's folder of plans to the session's and holds it empty.
* The plan's section for Task 14 was cross-checked and every test in it is unrun: expect the trial to correct it, and the review to find Important points the trial had not.

Facts the build stands on (measured; the ledger has each).
* The fast gate holds 3317 passed, 10 skipped, 5 deselected (3332 collected) at `b51a27f`, about 34 minutes. Trials have run far ahead of the plan's counts (Task 13: 31 functions against the plan's 16, and 11 more in its fix round).
* The last gate of record is still `e17f1ec`'s, 27 min 16 s. The real-screen run passed at `e7508da` with 10; at `b51a27f` 9 passed, 1 skipped (the keep-awake check, `tests/test_user_models.py:1445`, skips while another program keeps the computer awake; `powercfg /requests` needs an administrator and was not asked for).
* The last card run took 31 minutes with its diagnostics. While a gate runs, `Resources/*/sbitest*` and `Resources/Models/SBITEST.json` stand untracked, a test's own: they are gone at its end.
* On this computer `.write_text` makes every file CRLF, so a test that needs exact bytes writes bytes. Replacing a file another program holds open fails with "Access is denied" (error 5).
* On the offscreen platform `QTest.mouseDClick` presses nothing; the four events Windows sends for a double-click do (Task 13's fix round).
* In this checkout `run.bat` and `run.sh` read `i/lf w/crlf`.

The rules every agent was given, each learned the hard way:
* pytest is started by ONE tool call at a time, never two in one message;
* no python at all beside another python, not even a script of milliseconds that imports nothing (the controller and the skeptic each slipped once on 5 November), and readers run NO python while a gate runs; a seat that wants runs reads first, ends its turn with "waiting to run", and is given the machine when the gate has exited;
* a test that reads state the whole application or process shares takes a reading first, or works on a stand-in;
* a `cd` inside a Bash call moves the session's working directory: end it back at the repository root;
* never write a file into the repository while an implementer works there;
* GNU sed reads `\c` in a replacement as a control character: write a path with a backslash through the editor, and check a written document with `file` before committing it;
* times are read from `date`, never guessed (the session of 5 November twice wrote a time before reading the clock, and corrected each).

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
