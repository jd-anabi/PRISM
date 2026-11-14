PRISM — the redesign of its window finished the leaner way, now that it is a program for inference by simulation alone. Start with Task 12-13's fix round.

Where things stand. Stages 1 to 4 of the window's redesign and the fit-saving stage are built; STATE has each. Stage 5's plan, docs/superpowers/plans/2026-11-09-window-redesign-stage-5.md, was approved on 9 November and re-planned by me on 11 November (its section "## Re-plan of 11 November 2026 (the owner's)", just before Task 6). TASKS 0 TO 5, 5B, 6-7, 8-9 AND 10-11 ARE COMPLETE, each with its review clean and its gate green on the same head: Task 6-7 on `7d532ee`, Task 8-9 on `fd89995`, Task 10-11 on `7c75891` ("Parameter inference" final: no page of the old frame left). TASK 12-13 IS BUILT, on `9fc11de` (Task 12 `e17631d`, `6650f8c`; Task 13 `0647131`, `1595719`, `f11f6da`, `7f8bd7d`; the guide `9fc11de`): its fast gate RED, 1 failed, 4130 passed, 14 skipped, 3 deselected (`tests/test_look.py::test_console_text_alone_is_set_in_the_fixed_width_face`: `FIXED_FACE_CALLERS` still lists `panels/inference/posterior_tab.py` and `screens/home_screen.py`, which Task 13 left without a command line), its real-screen run 14 passed, its two readers' reviews written (stage-5/reviews/task-12-13-reader-one.md, `task-12-13-reader-two.md`; reader two read only, its faults listed and not run). The chat of 12 and 13 November stopped there by the usage rule, at 93 % of the weekly bar with extra usage switched on. Departures 476 to 479 and the registry's "Amended as built" paragraph are in the plan. Where the plan's "How to run this plan" or an older prompt differs from the Re-plan section, the section wins, and this prompt wins over both.

I pushed up to `cdc80d2`. Everything after it is not pushed (read `git status -sb`), and I push when I choose. The walkthrough rows are mine to run on the real screen when it suits me.

Before any work. Nothing is running as this is handed over. Check that:

* the tree is clean;
* the graphics card is free;
* no `python.exe` or `pythonw.exe` is alive. Two pytest processes, 46012 and 16612, may still be listed: they exited long ago and stale handles hold them. Leave them alone;
* no `PRISM.lock` stands beside my settings file, whose folder holds `PRISM.ini` and `PRISM.ini.bak-preband`;
* my desktop has its five names, hidden ones counted (`Research Plots`, `desktop.ini`, `ellie.jpg`, `TSNPE_round1_postmortem.md`, `Claude Science.lnk`), and no `PRISM.lnk`.

A PRISM started from `run.bat` or its shortcut (a `pythonw.exe -m core.gui`) is my own: leave it alone and wait. While it is open no gate runs. My real `PRISM.ini` is 3,296 bytes, sha256 79BF94DF…C83E.

How much to spend is decided: do not ask again.

* The leaner way, as I chose on 11 November ("Yes, all of it"): one Opus builder per task, two readers, the three-reader review once at the stage's end. No workflow unless I ask for one in my own words; every agent given its model by name, Sonnet where reading is enough.
* My words of 16 October stand: "Try to also finish things up at a clean state when usage limits are near their max, I won't buy anymore credits". And of 13 November: "please stop insisting to wait for the reset when we have like 18% left, it wastes my time because you could have been doing this overnight". So: read the usage card when you start and before every dispatch, tell me where the weekly bar stands, keep the loop going task after task while it has room, never hold a task on an estimate, and stop only when it actually nears its end, at a clean state (nothing half-built, the record written). Extra usage is switched on for my account: nothing may run the bar past 100 %, so stop starting agents near about 95 %.
* Measured the leaner way: Task 6-7 about 4 points, Task 8-9 about 4, Task 10-11 about 5, Task 12-13's build about 2. Tell me what each task costs.
* Hand over to a new chat, with STATE, the ledger and a prompt like this one, when the usage rule says stop or when the chat's room runs low.
* Read the clock before you write a time, and never write one ahead of it.

Read first, in this order.

1. docs/STATE.md: its head.
2. `CLAUDE.md`, and docs/guide/testing.md for what it points to.
3. The plan's section "## Re-plan of 11 November 2026 (the owner's)", then the plan's head (Global Constraints, Review Focus, the registry and its "Amended as built" paragraph, the departures, the newest 476 to 479), then Task 12's and Task 13's sections.
4. The ledger, .superpowers/sdd/2026-10-06-window-redesign/progress.md (gitignored). Its lines are very long: search it, do not read it through. Read its first 47 lines and the section "## Stage 5 — … (build of …)" from the line "- CHAT STARTED 2026-11-12 at 14:59" to its end, just above "## Engine changes landed". Never add a line above the lines that begin "- HAND-OFF". The build's lines go just above "## Engine changes landed", never at the file's end.
5. Task 12-13's records: stage-5/briefs/task-12-13-builder-dispatch.md (the dispatch and rulings R1 to R11), stage-5/reports/task-12-13-builder.md, the two reviews, stage-5/reviews/task-12-13-package.txt; and the newest models of the loop: Task 10-11's dispatch, readers' dispatch (`task-10-11-readers-dispatch.md`), re-read dispatch, and its fix round's rulings (the ledger line "- Ruling, 2026-11-13 at 17:44 PST").

Tools. The scratch folder is `C:\Users\J\AppData\Local\Temp\prism-redesign-scratch\`; stage 5's is `s5\`, with text copies of its tools in `stage-5/tools/`.

* `gate.sh <fast|quiet|slow|display|collect> <label>`, run from Bash in the background (`cd /c/Users/J/AppData/Local/Temp/prism-redesign-scratch/s5 && ./gate.sh fast s5task12-13-fix1-<sha>`), writes `stage-5/gates/<kind>-<label>.log`; it refuses while a `python.exe` is alive (it asks Windows whether each has exited; a refusal "a python process is alive (1)" may be a process just ending: read again) or while `QT_QPA_PLATFORM`, `QT_SCALE_FACTOR`, `PRISM_ARTIFACTS` or `PRISM_RESOURCES` is set. `ground.ps1 before|after <label>` through the PowerShell tool as `powershell.exe -NoProfile -ExecutionPolicy Bypass -File C:\Users\J\AppData\Local\Temp\prism-redesign-scratch\s5\ground.ps1 before <label>`. `card\card-gate.ps1` is the card test; `exit_fence.py` the engine fence.
* After every gate, the plan's "After each gate" checks, AND the real-screen run (`gate.sh display`, about 20 s, only with no PRISM of mine open): builders cannot run display tests, and a stale display expectation slipped through twice.
* The ledger: while an agent has the machine or a suite runs, write lines to a file in the scratchpad and insert them above "## Engine changes landed" with `sed -i "<n-2>r <file>"`, then read the place back. No python while a suite runs.
* Departures: one row each in the plan's table only (after the last row), NEXT 480. A ruling is a ledger line.
* The carried-over list .superpowers/sdd/2026-10-06-window-redesign/carried-over.md (1913 rows, missing 0 after Task 10-11) is kept current at every task's close: born rows by `s5\rows_born.py` (appendix suites) and `s5\rows_born_nine.py` (the nine main-table suites: `test_nav_and_gating.py`, `test_settings_persistence.py` and the rest); gone rows found by `s5\task10-11\rows_gone.py` and closed naming their successors by a script like `s5\task10-11\close_rows.py`; the check `/c/Users/J/anaconda3/envs/biophys-env/python.exe -I /c/Users/J/AppData/Local/Temp/prism-redesign-scratch/s5/task01_test_names.py --check carried-over.md` from that folder.
* The readers' standing instructions are stage-5/briefs/review-seats.md; their dispatches name BASE, HEAD, the sections, the report, the package, the output file and the hardest questions. A reader that runs python reads first, ends its turn with "waiting to run", and is given the machine once the gate has exited.
* A builder that commits in the repository needs the tree to itself: never write into the repository while it works.

What to do, one task at a time.

* Task 12-13's fix round: rule every finding of the two reviews (reader two's listed faults run first, by a reader given the machine after a gate, if a finding rests on one), then ONE fresh Opus builder (the old one's context is gone) from a dispatch naming the reviews, the report and your rulings; then a Sonnet re-read while a fresh gate runs on the final head, and the real-screen run. Complete when the review is clean and the gate green on the same head. Then the carried-over rows and STATE.
* Then the last review over the whole stage (three readers, seat three's fault runs) with one fix wave.
* Then Task 14, the exit, which needs me. First, one question to me: the order. Recommended: the pictures and my questions on them first (on hidden windows; nothing of mine is touched), a change I ask for as a small task, then the gates of record once, on the final head (a quiet machine; the slow set, three tests, about 49 minutes; the card test with the diagnostic card and one card test through the Checks page's call, telling me the commit, the folder, how long the last run took (25 min 30 s at stage 4's exit, without the diagnostics) and that nothing else may use the card; the real-screen run); the walkthrough rows written for me (the words each builder drafted in its report); the guide's pages for Checks, Inference and the round; my yes or no to the `CLAUDE.md` edits in exact words; the list of everything you decided without me (stage-5/exit/rulings-for-the-owner.md).
* After stage 5: the code audit, then the retrain from the window (started by me, from on-screen steps you give me), then one last stage for Records, Your own models, Help, the engine's sentences, the guide and the final gates.

Carried, each to its task.

* The last review: from the earlier prompt, Task 1's lines test checks `epoch_line` against itself; the own-words test in tests/test_checks.py still matches by key names, its two "no words" checks rest on the fixtures, and a lead comment is doubled (about `:1551-1552`). From this chat: `_own_config` says any `ValueError` of a check's own configuration as an unreadable file; a forcing line's one long token cannot wrap; a test writes `sbitest` files into the real `Resources/` while it runs (gone at its end; a gate cut short would leave them); the values grid drawn as a three-rule table and the keyboard handed to the result's first next step (owed by the design, not built); a held carried observation overrides a pick made before the next reading; the run page's unused command slot (Task 13 hid it).
* Task 14: three guide sentences Task 2 made false (docs/guide/rules-and-traps.md:125-128, docs/guide/command-line.md:244, `:248-250`; check what is left after the cut); G8's words (Task 3's skeptic's report); the plan's diagnostic card at the exit is kept. `CLAUDE.md`'s plan rule (about `:144`, "A run plan fills the forms of steps 1 to 3 …") and its kept-round rule are stale after Task 12-13: their new words go to me for a yes on exact words, with the rule on what the inference pages remember and its "Drive F₀ (N)" (departure 404; Inference's box now reads "Drive amplitude").
* For my list at the exit: departures 450 and 462 with 465; 452, 453, 458, 459, 460; 469 (Records' key Ctrl+3, Help's Ctrl+6); 470 to 475 (the cut's rulings); 476 (the words under a check's field are the window's, the rule the engine's); 477 (the audit greyed while its record could not count); 478 (Inference's end and stop sentences the shared ones); 479 ("on this page" only where the thing is); seeds ungrouped; the repeated calibration's seed bounded lower by its repeats; the narrowing round compared on its run page with an audit of the whole prior; a timing press flashing the taskbar at each of its two fits; the Checks' cell lists opening empty until a cell is chosen in Inference; the input-use check judged on the simulations asked for (the code audit's item 10); departure 439's cost; the Inference grid's words; the first test that starts a pytest run of its own; the README's opening line; the five packages only removed code used (pyqtgraph, imageio, imageio-ffmpeg, h5py, pandas), still pinned: mine to drop or keep. The code audit's list gained items 14 to 17 in these tasks.

Things to check before they bite.

* THE FIRM REQUIREMENT (departure 125): "at the window's least size, and at every width above it, nothing of the run page is hidden under the form; the form and the run page share the width with neither drawn over the other". Held now by `test_nothing_of_the_run_page_is_cut_on_a_step_page_at_any_width` (every rebuilt step, Inference among them since `7c75891`) and `test_nothing_of_the_run_page_is_hidden_at_150_percent_display_scaling`. Both readers missed that Inference had dropped out of the first: check it yourself at every task.
* The engine is closed: a task that finds it must open a file under `core/` outside `core/gui/` stops, and you bring it to me. No task edits a function of `core.SBI.fit_state.FIT_CODE`; the seventeen digests are held by `test_the_named_training_code_is_as_recorded`.
* `Artifacts/` holds 428 entries as the gates list them; every gate's listing after equals the one before. Nothing deletes the empty `fdt/` folder.
* A builder runs every suite a commit edits WHOLE before the commit: Task 10-11's fix round let a regression through a suite it ran only in part.
* A guard's or a test's message never prints what it read from my computer (the clipboard guard once printed my clipboard into a gate log; the lines were removed).
* A page never holds a toolkit item (`QStandardItem`, a layout item) across its list's rebuild: Qt handed `QMainWindow.event` a stale one.
* `AskUserQuestion` waits for my answer: ask while long runs go. I often follow over Remote Control: put a question that waits on me in the closing message too.

Facts the build stands on (measured; the ledger has each).

* The fast gate: 4100 passed, 14 skipped, 3 deselected at `7c75891` (47 min 0 s; not certified quiet); Task 12-13's at `9fc11de` 1 failed, 4130 passed, 14 skipped, 3 deselected (47 min 41 s); the real-screen run 14 passed; the slow set, three tests, 3 passed in 49 min 12 s at `f2214da`.
* `core.refusals.FIELDS` holds 98 keys; the window's no-control set 14.
* The last card run took 25 min 30 s (five lines and run 5, no diagnostics) at `850dc23`.
* On this computer `.write_text` makes every file CRLF, so a test that needs exact bytes writes bytes, and git warns that the documents' LF will become CRLF (harmless).
* PySide6 6.9.3: a widget made visible before it has a parent is shown as a window of its own; a wrapper of every layout item read outlives the item; a combo's `QStandardItem` handle stays valid after `clear()`.

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

* The science: the models, the simulation, the features, the training and calibration methods, and every setting in core/config.py.
* What a simulated row holds, and the simulation set's identity: the finished set `Artifacts/simulations/eda5c2a1e468` stays usable.
* What a fit that is never interrupted computes.
* The rules in `CLAUDE.md`, edited only on my yes, on its exact words.
* Anything on disk under `Artifacts/`, and the console capture C:\Users\J\prism-retrain-console.log. Read freely; ask before anything deletes, renames, rewrites or trains on them.
* Nothing signs me out, locks or shuts down the computer, and nothing writes into my real desktop folder.

Do not start the retrain, or any training of real size, in this chat; the tiny trainings in the tests and the card gates are fine.

Standing rules:

* the local `main` branch;
* one commit per logical step, with a short subject, never amended, ending with the Co-Authored-By line the harness provides;
* I push;
* never two pytest processes at once, and never edit a source file while a suite runs;
* a run on the card has the card to itself;
* scratch files go under `C:\Users\J\AppData\Local\Temp\`, and no `.py` file goes under `.superpowers/` or `docs/`;
* update docs/STATE.md at the end of every session.
