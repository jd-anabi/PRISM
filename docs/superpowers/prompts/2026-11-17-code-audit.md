Paste everything below the line into a new chat opened on this repository. It is written in the owner's voice by the session of 16 and 17 November 2026 that finished stage 5's exit, and it is committed with STATE, so the tree is clean when it is handed over.

---

PRISM — the code audit, now that stage 5 of the window's redesign is closed and PRISM is a program for inference by simulation alone.

Where things stand. Stages 1 to 5 of the window's redesign and the fit-saving stage are built and closed; STATE has each. Stage 5's exit ended on 16 November with every gate of record green on the final code `e9c6b5cd`; the documents after it end with the commit that holds this prompt and STATE. Nothing from stage 5 waits on me: my "Yes, all six" to the `CLAUDE.md` and `README.md` edits is committed (`d847df3d`), and my "Yes, as recommended" to the rulings' A2 sends the keyboard handed to a result's first next step to the last stage and drops the design's three-rule table in the results. My list of what was decided without me is `.superpowers/sdd/2026-10-06-window-redesign/stage-5/exit/rulings-for-the-owner.md`. The order after this chat is: the retrain from the window, started by me, then one last stage for Records, Your own models, Help, the engine's sentences, the guide and the final gates.

I pushed up to `cdc80d2`. Everything after it is not pushed (read `git status -sb`), and I push when I choose. The walkthrough rows (`docs/checklists/display-walkthrough.md`) are mine to run on the real screen when it suits me.

Before any work. Nothing is running as this is handed over. Check that:

* the tree is clean;
* the graphics card is free (the desktop's share, about 2,100 MiB on 17 November);
* no `python.exe` or `pythonw.exe` is alive (an idle kernel worker under `.claude-science` is another tool's, not ours: leave it; `gate.sh` skips it);
* no `PRISM.lock` stands beside my settings file, whose folder holds `PRISM.ini` and `PRISM.ini.bak-preband`;
* my desktop holds what it held on 17 November, hidden names counted (`Research Plots`, `desktop.ini`, `ellie.jpg`), and no `PRISM.lnk`.

A PRISM started from `run.bat` or its shortcut (a `pythonw.exe -m core.gui`) is my own: leave it alone and wait. While it is open no gate runs. My real `PRISM.ini` is 3,296 bytes, sha256 79BF94DF…C83E.

How much to spend is decided: do not ask again.

* The leaner way, as I chose on 11 November ("Yes, all of it"): one Opus builder per task, two readers (Sonnet and Opus), the three-reader review once at the audit's end, bigger tasks. No workflow unless I ask for one in my own words; every agent given its model by name, Sonnet where reading is enough.
* My words of 16 October stand: "Try to also finish things up at a clean state when usage limits are near their max, I won't buy anymore credits". And of 13 November: "please stop insisting to wait for the reset when we have like 18% left, it wastes my time because you could have been doing this overnight". So: read the usage card when you start and before every dispatch, tell me where the weekly bar stands, keep the loop going task after task while it has room, never hold a task on an estimate, and stop only when it actually nears its end, at a clean state (nothing half-built, the record written). Extra usage is switched on for my account: nothing may run the bar past 100 %, so stop starting agents near about 95 %. The bar was at 15 % on 17 November, resetting on 22 November at 04:00 UTC.
* Measured the leaner way: stage 5's tasks cost 2 to 5 points each; its exit about 10. Tell me what each task costs.
* Hand over to a new chat, with STATE, the ledger and a prompt like this one, when the usage rule says stop or when the chat's room runs low.
* Read the clock before you write a time, and never write one ahead of it.

Read first, in this order.

1. docs/STATE.md: its head.
2. `CLAUDE.md`, and docs/guide/testing.md and docs/guide/architecture.md for what they point to.
3. The audit's list, docs/superpowers/notes/2026-10-29-the-code-audits-list.md, whole: the body gathered at stage 3's exit (to about line 237, with its "Taken off" and "Pointers I could not follow"), then three sections of items sent later: stage 3's exit's small tasks (from about line 238), stage 4's (about 333) and stage 5's (about 423). Each item is as recorded and none was checked against the code. Lines have moved, and the cut of 12 November removed `core/FDT/`, `core/Reduction/` and Live simulation, so many items name code that is gone. Its "To a later stage (not the audit)" and "For the exit's documents (not the audit)" sections are not the audit's.
4. The stage-5 ledger's last lines, .superpowers/sdd/2026-10-06-window-redesign/progress.md (gitignored; its lines are very long: search it, do not read it through), from the line "- THE GATES OF RECORD, ALL GREEN on the final code `e9c6b5cd`" to "## Engine changes landed", and that section, which registers every engine change of the redesign for this audit.
5. docs/guide/rules-and-traps.md, which names the rules each engine area must keep.
6. The loop's models: the readers' standing instructions, .superpowers/sdd/2026-10-06-window-redesign/stage-5/briefs/review-seats.md, and stage 5's newest dispatches beside it (a builder's, its readers', a re-read's); text copies of stage 5's tools are in `stage-5/tools/`.

What to do.

* A workspace of its own: `.superpowers/sdd/2026-11-17-code-audit/` (gitignored), with `progress.md` (the ledger: a head of what binds, then dated lines, and as its last section "## Engine changes landed by the audit", which registers every engine change the audit makes; dated lines go just above that heading, never at the file's end), and `briefs/`, `reports/`, `reviews/`, `gates/`. While an agent has the machine or a suite runs, write ledger lines to a scratch file and insert them with `sed -i "<n-2>r <file>"` (n the heading's line), then read the place back.
* The tools. The scratch is `C:\Users\J\AppData\Local\Temp\prism-redesign-scratch\audit\`: copy stage 5's from `s5\` (`gate.sh`, `ground.ps1`, `exit_fence.py`, `card\card-gate.ps1`, `card\diag-card.ps1`, `card\checks_card.py`, `rows_born.py`, `rows_born_nine.py`, `task01_test_names.py`, `carried_check.py`) and re-point each before its first use: `gate.sh` and `ground.ps1` hold stage 5's `gates` path on one line each (make it the audit's); the card scripts hold `s5\card-exit` (their scratch store) and `s5\card\logs-<label>`, and `diag-card.ps1` runs `checks_card.py` by its `s5` path (make them `audit\…`); `exit_fence.py` keeps its three comparison records in stage 4's and stage 5's `gates/` as they are, gains `stage-5/gates/digests-at-exit.json` as a fourth, and writes its own record into the audit's `gates/`; `rows_born.py` and `rows_born_nine.py` write "stage 5" into a born row's text (make it "the code audit"). `gate.sh <fast|quiet|slow|display|collect> <label>` is run from Bash in the background (`cd /c/Users/J/AppData/Local/Temp/prism-redesign-scratch/audit && ./gate.sh fast audit-taskNN-<sha>`) and writes `gates/<kind>-<label>.log`; it refuses an existing log, a live python, or a set `QT_QPA_PLATFORM`, `QT_SCALE_FACTOR`, `PRISM_ARTIFACTS` or `PRISM_RESOURCES`. A task's gate is `fast` (one process, about an hour); the gates of record at the exit are `quiet`.
* After every gate: `ground.ps1 after <label>` shows the `Artifacts/` listing identical to the one before (names, sizes and times; 428 is its count), and you check the rest by hand: tree clean; no `sbi-logs/`; `Resources/` as it was (`Resources/Models` holds `SHM.json` and `SHM2.json` only, and there is no `Resources/Plans` folder on 17 November); the real `PRISM.ini` 3,296 bytes, sha256 79BF94DF…C83E, its folder holding `PRISM.ini` and `PRISM.ini.bak-preband`; the desktop's three names and no `PRISM.lnk`; no python alive; the gate's line in the ledger with its count and time. Then the real-screen run (`gate.sh display`, about 20 s), run by you, only with no PRISM of mine open: builders cannot run display tests, and a stale display expectation slipped through twice.
* First, the sorting. Two Sonnet readers, read only, at the same time (no python), one for the engine half (`core/` outside `core/gui/`, the command line) and one for the window, the tests, the launchers and the documents. Each item is checked against the code at HEAD and given: still true (where it stands now), already fixed (by what), gone with removed code, or not a fault; its size; and whether it touches anything under "What must not change" below, a function of `core.SBI.fit_state.FIT_CODE`, a refusal's field, what a stage writes or reads, or the window's words (which are the last stage's). The body's "Taken off" items stay in the table, marked taken off. One sorted table, every item kept with its old number.
* Then the audit's plan, written by you from the sorted table: tasks by area, each naming the items it closes, its files, its tests and its gate; the items that are mine as one list of questions (below); the items that are the last stage's sent there, named in STATE. Bring me the plan for my yes, one question; nothing is built before it.
* Then the build, task after task: one Opus builder from a dispatch, two readers of its diff, one fix round, a fresh `fast` gate on the final head with the real-screen run; complete when the review is clean and the gate green on the same head. Then the three-reader review of the whole audit with one fix wave.
* Then the exit, which needs me: the gates of record once on the final head (the quiet fast gate, the slow set, the card gate with run 5's drill, the diagnostic card and one card test through the Checks page's call, telling me first the commit, the folder and how long the last run took; the real-screen run); the guide's pages each change made false; my yes or no to any `CLAUDE.md` edit, on its exact words; the list of everything you decided without me.

Mine to answer, each as one question in plain words with your recommendation:

* every edit inside `FIT_CODE` (no kept training exists today: `Artifacts/` has no `fits/` folder, so such an edit costs no training of mine, but it re-records the seventeen digests that `test_the_named_training_code_is_as_recorded` holds, and none may come after the retrain starts);
* anything that would change what a stage refuses, writes or reads back, or a record kind's manifest, or remove a command-line flag or subcommand;
* the five packages only removed code used, still pinned in `requirements.txt` (pyqtgraph, imageio, imageio-ffmpeg, h5py, pandas): drop or keep, one answer for all five (if you recommend splitting them, say which);
* any item that comes near the science, a simulated row, the simulation set's identity or what an uninterrupted fit computes: bring it to me before anything is drafted.

Things to check before they bite.

* The engine opens for this audit, and only for the items of the approved plan. A change that could move a number (a simulation, a feature, a fit, a calibration, an inference) is shown not to, on a tiny seeded run before and after, in its report; the card smoke gate runs after any change that moves tensors, and the diagnostic card after one under `core/diagnostics` that does (`CLAUDE.md`).
* `Artifacts/` holds 428 entries as the gates list them; every gate's listing after equals the one before. Nothing deletes the empty `fdt/` folder; the store never lists or touches it.
* The firm requirement of the window (departure 125 of the window's design): at the window's least size and at every width above it, nothing of the run page is hidden under the form. Held by `test_nothing_of_the_run_page_is_cut_on_a_step_page_at_any_width` and `test_nothing_of_the_run_page_is_hidden_at_150_percent_display_scaling`; an audit item under `core/gui/` keeps both green.
* The carried-over list (.superpowers/sdd/2026-10-06-window-redesign/carried-over.md, 1525 rows, missing 0 at stage 5's exit) stays current: a test the audit removes or renames gets its row closed, naming its successor; a new test is born into it.
* A builder runs every suite a commit edits WHOLE before the commit.
* A builder that commits in the repository needs the tree to itself: write nothing into the repository (STATE and the documents included) while it works.
* A page never holds a toolkit item (`QStandardItem`, a layout item) across its list's rebuild: Qt handed `QMainWindow.event` a stale one.
* No shippable file cites the working record (`tests/test_source_hygiene.py`); the audit's item numbers are labels, never written into code or tests.
* A guard's or a test's message never prints what it read from my computer.
* `AskUserQuestion` waits for my answer: ask while long runs go. I often follow over Remote Control: put a question that waits on me in the closing message too.

Facts the audit stands on (measured; the stage-5 ledger has each).

* The quiet fast gate on `e9c6b5cd`: 4182 passed, 14 skipped, 3 deselected, 59 min 5 s; the slow set 3 passed in 59 min 24 s; the real-screen run 14 passed; 4199 collected.
* The card gate with run 5's drill took 26 min, the diagnostic card 7 min, the card test through the Checks page about 6 min.
* `core.refusals.FIELDS` holds 98 keys; the training code's seventeen digests are recorded in `stage-5/gates/digests-at-exit.json`.
* The config's model keys are upper case (`config.VALID_MODELS` is `["BP", "NADROWSKI", "HOPF"]`); a script that spelt "Nadrowski" failed at the exit.
* On this computer `.write_text` makes every file CRLF, so a test that needs exact bytes writes bytes.
* PySide6 6.9.3: a widget made visible before it has a parent is shown as a window of its own; a wrapper of every layout item read outlives the item; a combo's `QStandardItem` handle stays valid after `clear()`.

The rules every agent is given, each learned the hard way:

* pytest is started by ONE tool call at a time, never two in one message;
* no python at all beside another python, not even a script of milliseconds, and readers run NO python while a gate runs;
* a test that reads state the whole application or process shares takes a reading first, or works on a stand-in;
* a `cd` inside a Bash call moves the session's working directory: end it back at the repository root;
* GNU sed reads `\c` in a replacement as a control character: write a path with a backslash through the editor, and check a written document with `file` before committing it;
* do not pipe Markdown or Python through a bash heredoc: write the file with the Write tool;
* `ground.ps1` runs as `powershell.exe -NoProfile -ExecutionPolicy Bypass -File <path> before|after <label>`;
* times are read from `date`, never guessed, and never written ahead of the clock.

Ask me what is mine: what I decided or what the design leaves to me, what cannot be undone, and anything under "What must not change" below; one question at a time, in plain words, with your recommendation. Decide the small things yourself, write each as a ruling with its cost, and give me the list at the exit. Before you tell me a fact about my computer, check it.

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
