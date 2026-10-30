# Prompt for the next chat — the window redesign: the plan of stage 4 (steps 1 to 3, plans, Settings, applied against pending)

Paste everything below the line into a new chat opened on this repository. It is written in the owner's voice by the
session of 29 and 30 October 2026 that ran stage 3's exit. What in it is not the owner's own words:

- The owner's own words in that session:
  - the order, "Pictures first";
  - on the pictures: "Fix it now" (the cut end time), "Add the arrows" (the figure tabs) and "Leave it to stage 4"
    (the form drawn over the run page at the window's least size);
  - "Start now" (the gates of record) and "Nothing of PRISM's" (`powercfg /requests`);
  - a yes to all four of `CLAUDE.md`'s changes;
  - "Write it now" (this prompt).

  "How much to spend", the usage rule and "What must not change" repeat the earlier prompts, whose words stand.
- Everything else is that session's record and its way of handing over. What stage 4 must carry was gathered from
  `docs/STATE.md`, the ledger and the last reviews of stages 1, 2 and 3 and of the fit-saving stage. Every count, time
  and measurement is that session's, not the owner's.

---

PRISM — the redesign of the window: the plan of stage 4, steps 1 to 3, plans, Settings, applied against pending.

**Where things stand.** Four pieces of the redesign are built: stage 1 (the look), stage 2 (the shell), stage 3 (the
run) and the fit-saving stage.

STAGE 3'S EXIT PASSED on 29 October. The code is final at `e17f1ec`, and every gate of record is green on it:
- the fast gate on a quiet machine: 2010 passed, 9 skipped, 5 deselected, in 27 min 16 s;
- the slow set: 5 passed, in 1 h 0 min 49 s;
- the card test with the diagnostic card and run 5: every criterion held, in 31 minutes;
- the real-screen run: 9 passed;
- the collection: 2024.

The named training code's seventeen digests are unchanged.

The documents followed the code:
- the guide, `5af4c23`;
- `CLAUDE.md`, on my yes, `e83bc6c`;
- the walkthrough's rows, `2d3bca6`;
- the plan and the design as built, with departures 122 to 125, `49621f2`;
- the code audit's list in one file, `1524c8e`;
- STATE, `3dc138d`, and then this prompt.

I pushed up to `5573d6f`. Everything after it is not pushed (read `git status -sb`), and I push when I choose. The
walkthrough rows G1, G2, G6, G7 and G34 to G36 are mine to run on the real screen when it suits me.

**Before any work.** Nothing is running as this is handed over. Check that:
- the tree is clean;
- the graphics card is free;
- no `python.exe` or `pythonw.exe` is alive;
- no `PRISM.lock` stands beside my settings file;
- my desktop has its ten names (hidden ones counted) and no `PRISM.lnk`.

A PRISM started from `run.bat` or its shortcut (a `pythonw.exe -m core.gui`) is my own: leave it alone and wait. While
it is open no gate runs. My real `PRISM.ini` is 3,296 bytes, sha256 79BF94DF…C83E.

**How much to spend is decided: do not ask again.**
- "In between", as I chose on 12 October:
  - no workflow unless I ask for one in my own words;
  - every agent given its model by name, Sonnet where reading is enough;
  - the review by three readers for every task that builds.
- On 26 October: "Let's just keep the same rigor, it doesn't seem worth it to rush things". So do not propose cutting
  the loop for speed.
- My words of 16 October stand: "Try to also finish things up at a clean state when usage limits are near their max, I
  won't buy anymore credits". So:
  - read the usage card when you start, tell me where the weekly bar stands, and read it again before every dispatch;
  - do not count on extra usage;
  - near about 90 %, start nothing that cannot be finished below it, and stop at a clean state.
- The week resets on 1 November at 04:00 UTC. The bar read 73 % at about 09:40 on 30 October.
- Stage 3's plan cost about 3.9 million tokens in background agents (the bar went from 69 % to 73 %). Your own long
  reading costs too: read with agents where reading is enough, and keep your own context for the plan's head.
- Hand over to a new chat at two thirds of the room, or when the usage rule says stop.
- Read the clock before you write a time.

**Read first, in this order.**

1. `docs/STATE.md`:
   - its head: stage 3 is done, and "CARRIED TO STAGE 4";
   - the paragraph "THE LAST REVIEW OF STAGE 3 AND TASK 14's THIRD FIX ROUND", for its "Carried to stage 4";
   - the fit-saving exit's "What waits for the window's stages", for its stage 4 line;
   - stage 2's exit paragraph (search "TO STAGE 4:").
2. `CLAUDE.md`.
3. The ledger, `.superpowers/sdd/2026-10-06-window-redesign/progress.md` (gitignored). Its lines are very long:
   search it, do not read it through. Read:
   - its first 47 lines;
   - the stage 3 section's exit lines, from the first line that begins "- NEW CHAT 2026-10-28" to the section's end;
   - "## Engine changes landed";
   - "## Deviations from the design", rows 1 to 125 (rows 122 to 125 were copied into it on 30 October);
   - the last ten lines of "## Gates".

   Never add a line above the lines that begin "- HAND-OFF" (the last is "- HAND-OFF 2026-10-29"). Stage 4's section
   heading goes just above "## Engine changes landed".
4. The design, `docs/superpowers/specs/2026-10-05-window-redesign-design.md`. Its section 8, the 125 departures, wins
   over the parts where they differ. Its parts are in `docs/superpowers/specs/2026-10-05-window-redesign-design/`:
   - `05-anatomy.md` whole: the page every step is built to (5.3 applied against pending, 5.4 "Before you start", 5.5
     name and note, 5.6 the one main button, 5.7 how a refusal is shown, 5.11 where a page hands over to the run);
   - `08-steps.md`: 8.1 to 8.4, and 8.7 (a saved simulation set's differences);
   - `07-plans.md` whole: plans (7.1); Settings' "This computer" (7.2), with the keep-awake switch (7.2.6); what the
     window remembers (7.3), and its rule for `CLAUDE.md` (7.3.3);
   - `03-shell.md`, for Home's "Continue with these" (U24) and its next step;
   - `06-run.md`: 6.8 ("Continue from batch N" after a reopening) and 6.10 (the keep-awake switch during a run);
   - `11-engine.md`: 11.3 (E2), 11.6 (E5 and E8), and 11.11 (the registry: `plan`, `chi_mode`, `reparam_rotate` and
     `expected_epochs` arrive with E5);
   - `13-tests-stages-docs.md`:
     - 13.1: Table A's per-step tests for "Model and options" and "Prior", Table B's stage 4 rows, the remembered
       table's inference rows, and the control table;
     - 13.2's "Stage 4";
     - 13.4: the walkthrough rows G9, G11, G12 and G15 to G25, runnable after stage 4;
   - what `04-look.md` and `12-findings.md` say these pages use;
   - each part's "Choices in this part the owner has not seen";
   - the locked records `decisions-as-approved.md` and `rulings-after-the-drafts.md`, in
     `docs/superpowers/specs/2026-10-05-window-redesign-assets/` (the later wins).

   Have readers read the parts against today's code. Read yourself what the plan's head needs.
5. The fit-saving design, `docs/superpowers/specs/2026-10-21-fit-saving-design.md`, for stage 4's share: sections 4.2,
   4.4, 5.3, 5.6, 6.2 and 8. With it, the fit-saving last review, in
   `.superpowers/sdd/2026-10-06-window-redesign/fit-saving/reviews/last-review-two.md`:
   - its section "### Stage 4 (the Training step, Home)": the calls, the Resume call and what it must not send, a
     round's route, the refusals to expect;
   - its table's CARRY rows to stage 4.
6. Stage 3 as built:
   - its plan, `docs/superpowers/plans/2026-10-22-window-redesign-stage-3.md`. It is the model of a plan: "How to run
     this plan", the registry "Names tasks share", and the departures table with costs. Its departures 85, 90, 97,
     101, 110, 114, 123 and 125 hand stage 4 its work.
   - section 7 of `.superpowers/sdd/2026-10-06-window-redesign/stage-3/reviews/stage-3-final-review-product.md`, "What
     stage 4's planner must know that is written nowhere".
7. The prompts of 22 October (stage 3's plan) and 28 October (stage 3's exit), beside this file. They hold everything
   this one does not repeat: the preparation and review loop, the tools, and the rules every agent was given. Where
   they differ the later wins, and this one wins over all.

**What to do.** Write the plan of stage 4 from the design's parts as they stand, with the `writing-plans` skill.
Commit it when it is written, and give it to me to approve. Build nothing before I have.

Stage 4 delivers, by part 13.2 (ruling D3):
- the pages of steps 1 to 3 ("Model and options", "Prior", "Training"), to U16's anatomy. Training has:
  - "Before you start";
  - name and note;
  - "Load this posterior";
  - "Start training".
- plans: the plan picker gets its page (7.1.6), and the plan's fill reaches steps 1 to 3.
- Settings with "This computer": the memory limit, and the keep-awake switch that stage 3 left always on.
- "In use" and "Changed, not applied" on every step's picks; each pick is written when it is applied (B6).
- "Continue with these" on Home (U24).
- after a reopening, "Continue from batch N" (C2, C4); pressing it fills these forms.
- the ask-first rule.

At its end:
- a prior is built or loaded, and a training is planned, named, started, watched, stopped and continued wholly in the
  new design;
- the forms of Checks, Inference and the Narrowing round are still today's.

From the fit-saving design, stage 4 also delivers:
- the choice among matching kept trainings when a training starts (its 4.4);
- Resume after a reopening, on Home and on the Training step (its 4.2);
- Home's next step.

The redesign's places that still say a stopped fit is lost (that design's section 9, `08-steps.md` among them) are
amended as this stage's departures, as departure 84 did for the run.

Engine:
- E2: every differing setting, per saved simulation set;
- E5: the plan reader, a new torch-free `core/plans.py`;
- E8: one path line, `PLANS_PATH`, in `core/config.py`'s PATHS block. It was approved in U21 and is a path, not a
  setting, but that file is mine: say it to me when you give me the plan.

Find their lines again at today's head: stage 3 and the fit-saving stage moved the code the design cites.
`core.refusals.FIELDS` holds 110 keys today; the design counted 109 before the fit-saving stage. Recount it before the
registry's rows are written.

Tests rewritten first:
- Table B, stage 4;
- Table A's per-step tests for "Model and options" and "Prior", re-pointed and their wording rewritten;
- the remembered table's inference rows.

New suites: `test_before_you_start.py`, `test_plans.py` and `test_applied_and_pending.py`.

The exit, by part 13:
- the gate, the display-marked run, and pictures;
- the walkthrough rows this stage creates;
- the slow set waits for stage 5's exit, which covers stage 4's changes.

Say in the plan whether the card test is owed, and why. It follows changes that touch training or move tensors, and
the window's Resume starts a training.

Say honestly how many tasks you expect:
- the design estimated 15 for stage 4, and 17 for stage 3;
- stage 3's plan had fourteen building tasks and held to them. It grew by fix rounds (Task 14 needed three), one fix
  wave after its last review, and three small tasks at its exit. It took seven chats, from 23 to 29 October.

Put the preparation that worked into the plan:
- For a task that touches threads, the run, the engine or what a run is handed, one skeptic (opus) first builds it as
  a stand-in in a scratch copy and runs its tests red then green. It brings what it finds to the controller, whose
  rulings go INTO the one brief.
- One implementer per task, from that one brief.
- On DONE, at the same moment:
  - the one-process fast gate;
  - three readers: seat one (sonnet) checks the brief against what was built; seat two (opus) compares the product
    before and after; seat three (opus) reads the tests and the faults they would miss, and runs its fault checks
    after the gate.
- An Important goes to one skeptic, with your conditional rulings in a file.
- One fix round, re-read by one reader while a fresh gate runs.
- A last review over the whole stage, with one fix wave.
- The exit, with me.

Ask me, one question at a time, in plain words, with a picture where it helps and your recommendation, what the parts
leave to me. Decide the small things yourself, and write each as a ruling with its cost.

**What the plan must carry.**

- THE FIRM REQUIREMENT (my "Leave it to stage 4", departure 125): "at the window's least size, and at every width
  above it, nothing of the run page is hidden under the form; the form and the run page share the width with neither
  drawn over the other".
  - Today the form's scroll area keeps its least width (`CONTROLS_MIN_W`, 360 px) while the run page asks for its
    326 px. Within about 29 px of the least width, the form is drawn over the run page's first pixels: its head, the
    stage names, the health measures and the command's first letters.
  - A test measures it at 900 x 600, at 100 % text and at 150 % display scaling, and the exit's pictures show it.
- From stage 3 (STATE's "Carried to stage 4"; the last review's rulings, "Not in the wave"; the product seat's
  section 7):
  - The run's history leaves `plan`, `settings` and `check` empty, and `health` short (departure 123). Stage 4 fills
    them. It must read every older history as holding none: Settings' and "Before you start"'s memory of the last run,
    and the "Continue" after a reopening that fills the forms from `settings`, find nothing in a history written
    before then.
  - `history.continuable` has no caller, and is tested on a shape the controller does not write. It knows nothing of
    a kept training, so Resume after a reopening reads `fit.kept_id`, the store's `fit` record and its resumability
    itself. A stopped run's offers live only in memory and die with the process.
  - The reopen command, and the failure's message beside it, can be selected by mouse only. Resume after a reopening
    may remove the command. If it stays, make it keyboard-selectable.
  - The window's opt-in for a taken name (`if_name_taken="unnamed"`) is passed at five sites and is inert, since the
    window names a record only after it is written. It begins to matter when a page names a record at the start.
  - The run controller's host is duck-typed (the product seat lists what it needs). A rebuilt Training page that
    calls the controller directly (departure 90) must meet all of it or change it.
  - Keep-awake's switch reaches the controller as an argument (the knob rule), and acts again when it changes during
    a run (6.10).
  - `RunSpec.expected_epochs` is always None today. The Training page's "Expected epochs" feeds the time axis
    (departure 101).
  - The failure page's "Close PRISM and open it again …" is reworded once Home offers the continuation (departures
    97 and 110).
  - The three "Before you start" lines about the fit land on the Training page (departure 114):
    - the fit is saved after every pass;
    - a resume first reads its simulation set back, about four minutes at full size;
    - a training whose simulation set is not kept cannot be resumed.
  - `training_kept()` is set only for "Training", never for a narrowing round, and is None after a reopening.
  - Helpers are defined two to five times across the suites, and training specs are typed by hand.
  - "Sign out anyway" waits for this stage's real-screen rows: how long the process then lives, and whether a shown
    window's close handler runs.
- From the fit-saving stage (STATE's fit-saving exit paragraph; that last review's "Stage 4" section):
  - offer Resume only where `can_resume` holds, and show `why_not` otherwise;
  - the Resume call sends `resume_fit` and none of the knobs unless they are the record's, with the configuration
    rebuilt from the record;
  - a torn kept training is listed as resumable and then refused, so show the refusal and offer Delete;
  - a round resumes through the narrowing round;
  - a name given with a resume is not yet carried into the kept record. Decide it here.
- From stage 2 (STATE's "TO STAGE 4:"; `stage-2/reviews/stage-2-final-review-sorting.md`'s "CARRY 4", ten rows):
  - Enter on a ringed button: my answer Q8. The plan writes the rule, and proposes that Enter presses a ringed button
    too.
  - The layout room is read once, from the primary screen. A window on another screen keeps the primary's thresholds
    above 100 % text, and "This computer" is the place to settle it.
  - Step 3 reads "Needs a prior" while loading a posterior needs none.
  - Step 3's row against its page after a failed prior build.
  - The Prior step's gaps, which stage 2's column cure (its Task 6) would also close; it was not built there.
  - Step 1 can always start while its apply button is greyed for a model of your own that inference cannot take.
  - Step 1's Alt+Space.
  - One model has two names until the pages are rebuilt.
  - Tab reaches the scrolling column, the figure tabs and the log pane before the first setting.
  - The first rebuilt page that puts a push button first needs `_to_page`'s test.
  - A first opening during a live run is tested.
  - The old places named in the step panels' strings go with their pages.
- From stage 1 (`stage-1/reviews/stage-1-final-review-sorting.md`, "Stage 4, the first rebuilt pages", 27 rows): and
  the greyed main button of the first rebuilt page is shown to me.
- The code audit's list, `docs/superpowers/notes/2026-10-29-the-code-audits-list.md`. This stage must not lose it,
  and adds to it: every engine change goes on it.
- The departures go on from 126.

**Mine to hear, still open:**
- `CLAUDE.md`'s rule on what the window remembers (7.3.3), in this stage, only on my yes;
- `CLAUDE.md`'s line that `Resources/` holds four input folders, which a plans folder changes, only on my yes;
- bundling Cascadia Mono (never asked).

The lists of what was decided without me are in the ledger's `stage-3/exit/rulings-for-the-owner.md` and the earlier
stages' `exit/` folders. I may undo any of it. Keep giving me such a list at each stage's exit.

**Facts the plan stands on** (measured; the ledger has each):

- The fast gate holds 2010 tests (2024 collected, 2019 not slow). It took 27 min 16 s on a quiet machine at
  `e17f1ec`; stage 2's took 19 min 21 s, and `CLAUDE.md` records both. The slow set took 1 h 0 min 49 s.
- The gate tools are in the scratch folder `C:\Users\J\AppData\Local\Temp\prism-redesign-scratch\`: `gate.sh`, and
  `ground.ps1`, whose commands are run inline. They write into `stage-3/gates/`. The plan's Task 0 points them at a
  workspace of stage 4's own, `stage-4/`; their text copies are in the ledger's `stage-3/tools/`.
- The carried-over list is `.superpowers/sdd/2026-10-06-window-redesign/carried-over.md`: 950 rows, missing 0 at
  `e17f1ec`.
- The desktop's share of the card is about 1.2 to 3.0 GiB.
- No kept training stands in `Artifacts/`: there is no `fits/` folder. The named training code
  (`core.SBI.fit_state.FIT_CODE`) is not on stage 4's path as the design draws it; E2 is in `training_checkpoint.py`
  and the training preview. Say in the plan which named functions a task would touch, if any, and ask me first if one
  would: an edit there makes every kept training unresumable.
- In this checkout `run.bat` and `run.sh` read `i/lf w/crlf`. An edit to `run.bat` is an edit to the pin in
  `tests/test_shell.py`, in the same commit.
- `AskUserQuestion` waits for my answer: ask while long runs go, not before them.
- The rules every agent was given (the 28 October prompt lists them in full):
  - pytest is started by one tool call at a time;
  - no `python -c` beside another python, and readers run NO python while a gate runs;
  - a test that reads state the whole process shares takes a reading first;
  - a `cd` inside a Bash call moves the session's working directory;
  - never write into the repository while an implementer works there;
  - GNU sed reads `\c` in a replacement as a control character;
  - make trial copies with `git -c core.autocrlf=false archive`, and restore `run.sh`'s 100755 in them;
  - times are read from `date`, never guessed.

**What must not change without my say-so:**
- The science: the models, the simulation, the features, the training and calibration methods, and every setting in
  `core/config.py`.
- What a simulated row holds, and the simulation set's identity: the finished set `Artifacts/simulations/eda5c2a1e468`
  stays usable.
- What a fit that is never interrupted computes.
- The rules in `CLAUDE.md`, edited only on my yes.
- Anything on disk under `Artifacts/`, and the console capture `C:\Users\J\prism-retrain-console.log`. Read freely;
  ask before anything deletes, renames, rewrites or trains on them.
- Nothing signs me out, locks or shuts down the computer, and nothing writes into my real desktop folder.

Do not start the retrain, or any training of real size, in this chat; the tiny trainings in the tests and the card
gates are fine.

Standing rules:
- the local `main` branch;
- one commit per logical step, with a short subject, never amended, ending with the Co-Authored-By line the harness
  provides;
- I push;
- never two pytest processes at once, and never edit a source file while a suite runs;
- a run on the card has the card to itself;
- scratch files go under `C:\Users\J\AppData\Local\Temp\`, and no `.py` file goes under `.superpowers/` or `docs/`;
- update `docs/STATE.md` at the end of every session;
- hand over to a new chat, with STATE, the ledger and a prompt like this one, at two thirds of the room or when the
  usage rule says stop, whichever comes first.
