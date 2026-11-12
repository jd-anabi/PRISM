# Prompt for the next chat — PRISM made a program for inference by simulation alone (Task 5b, the cut), then stage 5's last four tasks, the leaner way

Paste everything below the line into a new chat opened on this repository. It is written in the owner's voice by the session of 11 November 2026 that finished Task 5 of stage 5, from `docs/superpowers/prompts/2026-11-10-window-redesign-stage-5-build-from-task-5.md`. That session committed this prompt with STATE and the code audit's list as one documents commit, so the tree is clean when it is handed over. What in it is not the owner's own words:

- The owner's words in that session: the pasted prompt it started from; "How much longer on this? … I think I just want it to be a SBI parameter inference software, I don't think the CMR of FDT stuff needs to be in it as it just bloats the whole software. Those things can just be written as scripts when I need them down the line"; "Yes, along with the live simulation."; "How else can we speed up this whole refactoring? Can we combine stages and tasks and be more efficient?"; "Yes, all of it". Each is quoted where it stands below.
- The standing orders (the 16 October quotation and the usage rule, "Ask me what is mine", "What must not change", the standing rules) repeat the owner's words of the earlier prompts.
- Everything else is that session's record, in the owner's voice as these prompts are. Every count, time, cost and measurement is that session's or its agents', not something the owner measured.

---

PRISM — made a program for inference by simulation alone, and the redesign of its window finished the leaner way. Start with Task 5b, the cut.

Where things stand. Stages 1 to 4 of the window's redesign and the fit-saving stage are built; STATE has each. Stage 5's plan, `docs/superpowers/plans/2026-11-09-window-redesign-stage-5.md`, was approved on 9 November. TASKS 0 TO 5 ARE COMPLETE, each with its review clean and its gate green on the same head. Task 5, the Checks list, a check's page and every check's result: `434ee3b`, `729b1cf`, `73fe79f`, departures 450 to 462 `be61451`, its fix round `4c433ef` (the skeptic's trial commit, its tree checked equal to its brief's walk), departures 463 to 467 `d93c4e0`. Its fix round made every result inside its check's page before anything fills it: a result made with no parent was shown as a window of its own.

What I decided on 11 November, during Task 5's fix round:
* "I think I just want it to be a SBI parameter inference software, I don't think the CMR of FDT stuff needs to be in it as it just bloats the whole software. Those things can just be written as scripts when I need them down the line." Asked whether to take out the effective temperature (FDT, with its CrossVal study, sweeps and comparisons) and the Reduction map (its centre-manifold reduction, my "CMR") right after Task 5: "Yes, along with the live simulation." That yes opens the engine for the cut alone, removes the FDT settings, and lets `CLAUDE.md` be edited where it describes those pages.
* Asked how else to speed the work up ("Can we combine stages and tasks and be more efficient?") and offered six changes: "Yes, all of it." Stage 5's Tasks 6 to 13 are built as four; one Opus builder per task, two readers, the three-reader review once at the stage's end; departures written once, in the plan; `CLAUDE.md` shortened to its rules; the design's stages 6 to 9 become one last stage; the code audit and the retrain come right after stage 5.

The plan holds all of it in its section "## Re-plan of 11 November 2026 (the owner's)", just before Task 6: the order, the loop, Task 5b in full, what the controller does after it, the four merged tasks, and what is carried into the first of them. Where the plan's "How to run this plan" or an older prompt differs from that section, the section wins, and this prompt wins over both.

I pushed up to `cdc80d2`. Everything after it is not pushed (read `git status -sb`), and I push when I choose. The walkthrough rows are mine to run on the real screen when it suits me.

Before any work. Nothing is running as this is handed over. Check that:
* the tree is clean;
* the graphics card is free;
* no `python.exe` or `pythonw.exe` is alive. Two pytest processes, 46012 and 16612, may still be listed: they exited long ago and stale handles hold them. Leave them alone;
* no `PRISM.lock` stands beside my settings file, whose folder holds `PRISM.ini` and `PRISM.ini.bak-preband`;
* my desktop has its five names, hidden ones counted (`Research Plots`, `desktop.ini`, `ellie.jpg`, `TSNPE_round1_postmortem.md`, `Claude Science.lnk`), and no `PRISM.lnk`.

A PRISM started from `run.bat` or its shortcut (a `pythonw.exe -m core.gui`) is my own: leave it alone and wait. While it is open no gate runs. My real `PRISM.ini` is 3,296 bytes, sha256 79BF94DF…C83E.

How much to spend is decided: do not ask again.
* The leaner way, as I chose on 11 November ("Yes, all of it"), replaces the three readers for every task of 12 October and the "same rigor" of 26 October. No workflow unless I ask for one in my own words; every agent given its model by name, Sonnet where reading is enough.
* My words of 16 October stand: "Try to also finish things up at a clean state when usage limits are near their max, I won't buy anymore credits". So read the usage card when you start, tell me where the weekly bar stands, and read it again before every dispatch; do not count on extra usage; near about 90 %, start nothing that cannot be finished below it, and stop at a clean state (nothing half-built, the record written).
* The bar read 72 % at the hand-over and resets on 15 November at 04:00 UTC. Start Task 5b before the reset only if the bar leaves room to finish it below 90 % (expected 10 to 15 points the leaner way, not yet measured; its gates add wall time, not points); otherwise write its dispatch and wait for the reset. Task 5, a page task with its fix round, cost about 20 points the old way. Measure Task 5b and tell me.
* Hand over to a new chat, with STATE, the ledger and a prompt like this one, when the usage rule says stop or when the chat's room runs low.
* Read the clock before you write a time.

Read first, in this order.
1. `docs/STATE.md`: its head.
2. `CLAUDE.md`.
3. The plan's section "## Re-plan of 11 November 2026 (the owner's)", then the plan's head down to "## Task 0" (Global Constraints, Review Focus, the File Map, the registry, "How to run this plan", the departures). Each merged task's two sections when its turn comes, and the design's parts and the fit-saving design as those sections cite them.
4. The ledger, `.superpowers/sdd/2026-10-06-window-redesign/progress.md` (gitignored). Its lines are very long: search it, do not read it through. Read its first 47 lines and the end of the section "## Stage 5 — … (build of …)", just above "## Engine changes landed": the lines of 11 November run from "- OWNER ASKED, 2026-11-11" to "- HAND-OFF 2026-11-11". Never add a line above the lines that begin "- HAND-OFF". The build's lines go just above "## Engine changes landed", never at the file's end.
5. The carried notes in `stage-5/briefs/`: `task-10-carried.md`, `task-11-carried.md`, `task-12-carried.md`. Each goes into its merged task's builder dispatch word for word.
6. Task 5's records, as a ruling needs them: `stage-5/briefs/task-05-*`, `stage-5/reports/task-05-*`, `stage-5/reviews/task-05-*`.

Tools. The scratch folder is `C:\Users\J\AppData\Local\Temp\prism-redesign-scratch\`; stage 5's is `s5\`, with text copies of its tools in `stage-5/tools/`.
* `gate.sh <fast|quiet|slow|display|collect> <label>`, run from Bash in the background (`cd /c/Users/J/AppData/Local/Temp/prism-redesign-scratch/s5 && ./gate.sh fast s5task05b-<sha>`), writes `stage-5/gates/<kind>-<label>.log`; it refuses to start while a `python.exe` is alive (46012 and 16612 do not stop it) or while `QT_QPA_PLATFORM`, `QT_SCALE_FACTOR`, `PRISM_ARTIFACTS` or `PRISM_RESOURCES` is set. `ground.ps1 before|after <label>` runs through the PowerShell tool as `powershell.exe -NoProfile -ExecutionPolicy Bypass -File C:\Users\J\AppData\Local\Temp\prism-redesign-scratch\s5\ground.ps1 before <label>` (the machine refuses scripts otherwise; the switch is for that one process). `card\card-gate.ps1` is the card test; `exit_fence.py` the engine fence, against `stage-5/gates/digests-at-start.json`.
* After every gate, the plan's "After each gate" checks (the ground before and after, the listing of `Artifacts/`, `PRISM.ini`, the desktop, the gate's line in the ledger). After the gate of a task that adds or changes a display-marked test or the start-up path, the real-screen run (`gate.sh display`), only with no PRISM of mine open.
* The ledger: while an agent has the machine or a suite runs, write lines to a file in the scratchpad and insert them above "## Engine changes landed" with `sed -i "<n-2>r <file>"` (n the heading's line; the blank line before it stays), then read the place back. No python while a suite runs. A time in it is read from `date` first.
* Departures from now on: one row each in the plan's table only (after the last row, `| N | WHAT. … | why | cost | who decided |`), NEXT 469 (468 is my re-plan). A ruling is a ledger line.
* The carried-over list `.superpowers/sdd/2026-10-06-window-redesign/carried-over.md` (1771 rows, missing 0 at the hand-over) is kept current at every task's close. Its check, from that folder: `/c/Users/J/anaconda3/envs/biophys-env/python.exe -I /c/Users/J/AppData/Local/Temp/prism-redesign-scratch/s5/task01_test_names.py --check carried-over.md`. Born rows: `s5\rows_born.py <suite.py> <task words> <sha> <what>` for a whole suite, `s5\rows_born_nine.py` (same arguments) for the nine first-table suites, `s5\rows_named.py <suite.py> <task words> <sha> <what> <test names…>` for named tests of a suite the list keeps only some rows of. A row marked for a task is marked done when the task completes; a test renamed or gone gets its row closed by hand, naming its successor (Task 5's rows 189, 190 and 192 are the newest models); a test the cut removes gets its row closed naming the cut.
* The readers: their standing instructions are `stage-5/briefs/review-seats.md` (the skeptic's file `skeptic-standing.md` is retired with the skeptic). A reader's dispatch is a file naming BASE, HEAD, the plan's sections, the builder's report, the package (`stage-5/reviews/task-NN-package.txt`, from `git log`, `git diff --stat` and `git diff -U10`), the output file and the hardest questions for that task; `stage-5/reviews/task-05-seats-dispatch.md` and `task-05-fix1-reread-dispatch.md` are the newest models. A reader that wants to run python reads first, ends its turn with "waiting to run", and is given the machine by `SendMessage` once the gate has exited. Tell every agent to put its report in its final message if a write under `stage-5/` is refused, and save it yourself.
* A builder that commits in the repository needs the tree to itself: never write into the repository while it works.
* For the exit: the pictures' models are `stage-4/briefs/exit-pictures-3-dispatch.md` and its report, the nine checks `stage-2/briefs/task-11-pictures-dispatch.md`, the scratch-settings harness for any window script `stage-2/tools/task-07-fix1-harness.py.txt`. Copies for a trial: `git -C <repo> -c core.autocrlf=false archive <sha> | tar -x -C <folder>`, then restore `run.sh`'s 100755. A scratch script that imports torch and must stay off the card sets `CUDA_VISIBLE_DEVICES=-1` and asserts CUDA is not available.

What to do, one task at a time.
* Task 5b, the cut, as the Re-plan section writes it, by one Opus builder in the repository: what goes, what stays, the eight things that must not change each held by a test or a check, three commits each green, the fast gate, the slow set and the display tests on the final head. It may open engine files for the cut alone. It touches no function of `core.SBI.fit_state.FIT_CODE`, and comments inside those functions that name removed code stay as they are; comments elsewhere in kept files that name removed code are rewritten to say the same in words. In `core/config.py` only the FDT constants go. The card gate runs only if a file under `core/SBI/`, `core/Simulator/`, `core/Solvers/`, `core/Models/` or `core/diagnostics/` changes beyond comments. Its report lists every engine file it changed and the new count of `core.refusals.FIELDS`.
* After 5b, as the Re-plan section says: re-base the exit's checks on 5b's head (the fence's expected changes, `digests-at-start.json` taken again, the `FIELDS` count, Task 14's walkthrough rows of removed pages dropped). Then shorten `CLAUDE.md` to its rules. My yes of 11 November covers the builder's edits of sentences that describe the removed parts and your shortening with every rule's substance kept; a change to what a rule says still needs my yes on its exact words. The card gate's block stays whole, in `CLAUDE.md` or moved whole into `docs/guide/testing.md` with a pointer. Show me the result before you commit it. Then update STATE.
* Tasks 6-7, 8-9, 10-11, 12-13, each by one Opus builder from its two sections, two readers on DONE (Sonnet: the sections against what was built; Opus: the product and the tests' faults), one fix round through the builder, a Sonnet re-read while the fresh gate runs.
* Then the last review over the whole stage (three readers, seat three's fault runs) with one fix wave.
* Then Task 14, the exit, which needs me. First, one question to me: the order. Recommended: the pictures and my questions on them first (on hidden windows; nothing of mine is touched), a change I ask for as a small task, then the gates of record once, on the final head (a quiet machine; the slow set; the card test with the diagnostic card and one card test through the Checks page's call, telling me the commit, the folder, how long the last run took (25 min 30 s at stage 4's exit, without the diagnostics) and that nothing else may use the card; the real-screen run); the walkthrough rows written for me; the guide's pages for Checks, Inference and the round; my yes or no to the `CLAUDE.md` edits in exact words; the list of everything you decided without me (`stage-5/exit/rulings-for-the-owner.md`).
* After stage 5: the code audit, then the retrain from the window (started by me, from on-screen steps you give me), then one last stage for Records, Your own models, Help, the engine's sentences, the guide and the final gates.

Carried, each to its task.
* Task 6-7: the labels shown as windows of their own (the Re-plan section has the places, the cure and its standing guard); and from Task 5's re-read: a redraw that brings back the keyboard's row may scroll the page while the list has focus (`core/gui/panels/inference/validate_tab.py:122-124` with `:529-533`, read, not seen: look on the real devices); a kept result's source omits the units and force unit its words use (`validate_tab.py:327-337`); three docstrings in `tests/test_checks.py` say "check 3" (word them by the check's name); the item-1 test sees only results and forms (the standing guard covers every widget).
* Tasks 6-7 and 8-9: `RunSpec.check`, `last_check` and `result: {}` are unread so far.
* Task 8-9: the stand-in run accepts an order of stage events the engine never makes (build the refusal if the card test's page must refuse such an order); an `on_stage` that raises is untested; the card test's page can be driven with `stand_in.stage(...)`; the timing of a short fit is two runs, the second a turn after the first's `run_ended` (411); check 3 between a posterior's return and the end's reading; `health.comparison`; Checks reads each manifest once by its folder (414).
* Task 10-11: `task-10-carried.md` and `task-11-carried.md` whole.
* Task 12-13: `task-12-carried.md`. Tasks 12 and 13 were swapped at the plan's cross-check (12 is "Plans fill every step", 13 "A kept round after a reopening"); an older note may use the old numbers.
* The last review: Task 1's lines test checks `epoch_line` against itself; the own-words test in `tests/test_checks.py` still matches by key names, its two "no words" checks rest on the fixtures, and a lead comment is doubled at `:1551-1552`.
* Task 14: three guide sentences Task 2 made false (`docs/guide/rules-and-traps.md:125-128`, `docs/guide/command-line.md:244`, `:248-250`; the cut may already have rewritten that page: check); G8's words (Task 3's skeptic's report); the plan's diagnostic card at the exit is kept. For my list at the exit: departures 450 and 462 with 465, which narrows 462 (a check 3 "Not compared" because its counts could not be read holds step 4 open); 452, 453, 458, 459, 460; `CLAUDE.md`'s rule on what the inference pages remember and its "Drive F₀ (N)" (404); the input-use check judged on the simulations asked for (the code audit's item 10); departure 439's cost; the Inference grid's words; the first test that starts a pytest run of its own.

Things to check before they bite.
* THE FIRM REQUIREMENT (departure 125): "at the window's least size, and at every width above it, nothing of the run page is hidden under the form; the form and the run page share the width with neither drawn over the other". Each rebuilt page joins its two tests in the task that builds it.
* The engine is closed again after Task 5b: a later task that finds it must open a file under `core/` outside `core/gui/` stops, and you bring it to me. The code audit's list, `docs/superpowers/notes/2026-10-29-the-code-audits-list.md` (its stage 5 part has items 1 to 13), must not be lost; the cut removes the items that only concern removed code, saying so.
* No task edits a function of `core.SBI.fit_state.FIT_CODE`. If one finds it must, it stops and you ask me first. The seventeen digests are held by `test_the_named_training_code_is_as_recorded`.
* `Artifacts/` held 428 entries as the gates list them, its kinds `diagnostics`, `fdt` (an empty folder), `priors`, `simulations`, and no `fits/`; every gate's listing after equals the one before. Nothing deletes the empty `fdt/` folder.
* A page built on the step page is added by hand to the six lists that name the rebuilt pages, in the commit that builds it; `test_the_pages_walked_here_are_every_page_built_on_the_step_page` is the net.
* Nothing is read while a run is live; the controller's order (not live, then `status_changed`, then `run_ended`) is pinned. No code of the window reads a layout's items (`itemAt`, `takeAt`).
* Nothing passes `observation_name` until Task 10-11 rebuilds Inference.
* `AskUserQuestion` waits for my answer: ask while long runs go. I often follow over Remote Control: put a question that waits on me in the closing message too.

Facts the build stands on (measured; the ledger has each).
* The fast gate: 3981 passed, 12 skipped, 5 deselected at `d93c4e0` (3998 collected; 42 min 00 s), on a machine not certified quiet; the real-screen run 12 passed.
* `core.refusals.FIELDS` held 115 keys before the cut.
* The last card run took 25 min 30 s (five lines and run 5, no diagnostics) at `850dc23`.
* On this computer `.write_text` makes every file CRLF, so a test that needs exact bytes writes bytes, and git warns that the documents' LF will become CRLF (harmless). In this checkout `run.bat` and `run.sh` read `i/lf w/crlf`; an edit to `run.bat` is an edit to the pin in `tests/test_shell.py`.
* PySide6 6.9.3: a widget made visible before it has a parent is shown as a window of its own; a wrapper of every layout item read outlives the item.

The rules every agent is given, each learned the hard way:
* pytest is started by ONE tool call at a time, never two in one message;
* no python at all beside another python, not even a script of milliseconds, and readers run NO python while a gate runs;
* a test that reads state the whole application or process shares takes a reading first, or works on a stand-in;
* a `cd` inside a Bash call moves the session's working directory: end it back at the repository root;
* GNU sed reads `\c` in a replacement as a control character: write a path with a backslash through the editor, and check a written document with `file` before committing it;
* do not pipe Markdown or Python through a bash heredoc: write the file with the Write tool;
* times are read from `date`, never guessed, and never written ahead of the clock.

Ask me what is mine: what I decided or what the design leaves to me, what cannot be undone, and anything under "What must not change" below; one question at a time, in plain words, with a picture where it helps and your recommendation. Decide the small things yourself, write each as a ruling with its cost, and give me the list at the exit. Before you tell me a fact about my computer, check it.

What must not change without my say-so:
* The science: the models, the simulation, the features, the training and calibration methods, and every setting in `core/config.py` (the FDT settings excepted, by my yes of 11 November).
* What a simulated row holds, and the simulation set's identity: the finished set `Artifacts/simulations/eda5c2a1e468` stays usable.
* What a fit that is never interrupted computes.
* The rules in `CLAUDE.md`, edited only on my yes (my yes of 11 November covers the two edits named above, and no other).
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
