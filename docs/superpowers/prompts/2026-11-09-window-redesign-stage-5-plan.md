# Prompt for the next chat — the window redesign: the plan of stage 5 (Checks, Inference, the Narrowing round)

Paste everything below the line into a new chat opened on this repository. It is written in the owner's voice by the
session of 5 to 9 November 2026 that ran stage 4's last review and its exit. What in it is not the owner's own words:

- The owner's own words in that session:
  - the exit's order, "Pictures first (Recommended)";
  - the sixteen answers on the pictures, in `stage-4/exit/owner-answers.md`, among them "Build it now" (Task 15e)
    and "Yes, keep the room (Recommended)";
  - "Yes, all of them (Recommended)" (the `CLAUDE.md` edits) and "Yes, record it (Recommended)" (the gate's time);
  - "Yes, write it" (this prompt).

  "How much to spend", the usage rule and "What must not change" repeat the earlier prompts, whose words stand.
- Everything else is that session's record and its way of handing over. What stage 5 must carry was gathered from
  `docs/STATE.md`, the ledger, stage 4's plan and its last review, and the last reviews of stages 2 and 3 and of the
  fit-saving stage. Every count, time and measurement is that session's, not the owner's.

---

PRISM — the redesign of the window: the plan of stage 5, Checks, Inference and the Narrowing round.

**Where things stand.** Five pieces of the redesign are built: stage 1 (the look), stage 2 (the shell), stage 3 (the
run), the fit-saving stage, and stage 4 (steps 1 to 3, plans, Settings, applied against pending).

STAGE 4 CLOSED on 9 November. The code is final at `850dc23`, and every gate of record is green on it:
- the fast gate on a quiet machine: 3572 passed, 11 skipped, 5 deselected, in 43 min 28 s;
- the card test (`CLAUDE.md`'s five lines and run 5; no diagnostic card, nothing under `core/diagnostics` changed):
  every criterion held, in 25 min 30 s;
- the real-screen run: 11 passed, none skipped;
- the collection: 3588.

The slow set was not run at stage 4: it waits for stage 5's exit, which covers stage 4's changes. The named training
code's seventeen digests are unchanged (`stage-4/gates/digests-at-exit.json`).

The documents followed the code:
- the guide pages for the rebuilt steps, `c45e7b7`;
- the walkthrough's rows, `7cd91bf`;
- the code audit's list, items 29 to 33, `c5c262d`;
- `CLAUDE.md` and `docs/guide/testing.md`, on my yes, `125b332`;
- STATE, `385f056`, and then this prompt.

I pushed up to `dc66861`. Everything after it is not pushed (read `git status -sb`), and I push when I choose. The
walkthrough rows G1, G2, G6, G7, G8, G9, G11, G12, G13 (as far as plans reach), G14 (its frame half), G15 to G25 and
G34 to G53 are mine to run on the real screen when it suits me.

**Before any work.** Nothing is running as this is handed over. Check that:
- the tree is clean;
- the graphics card is free;
- no `python.exe` or `pythonw.exe` is alive. Two pytest processes, 46012 and 16612, may still be listed: they exited
  long ago and stale handles hold them. Leave them alone;
- no `PRISM.lock` stands beside my settings file, whose folder holds `PRISM.ini` and `PRISM.ini.bak-preband`;
- my desktop has its four names (hidden ones counted) and no `PRISM.lnk`.

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
- The week resets on 15 November at 04:00 UTC. The bar read 21 % at 09:26 on 9 November.
- Stage 4's plan cost about 6.4 million tokens in background agents (the bar went from 73 % to 78 %); stage 3's cost
  about 3.9 million. Your own long reading costs too: read with agents where reading is enough, and keep your own
  context for the plan's head.
- Hand over to a new chat at two thirds of the room, or when the usage rule says stop.
- Read the clock before you write a time.

**Read first, in this order.**

1. `docs/STATE.md`:
   - its head: stage 4 is closed, and the exit as it stood before my last two answers;
   - the paragraph that begins "STAGE 4'S PLAN IS WRITTEN, COMMITTED AND APPROVED", for the plan's shape;
   - stage 2's exit paragraph (search "TO STAGE 5:").
2. `CLAUDE.md`.
3. The ledger, `.superpowers/sdd/2026-10-06-window-redesign/progress.md` (gitignored). Its lines are very long:
   search it, do not read it through. Read:
   - its first 47 lines;
   - the stage 4 build section's last chat, from the line that begins "- NEW CHAT 2026-11-05" to the section's end;
   - "## Engine changes landed";
   - "## Deviations from the design", rows 1 to 125 (the later departures are in stage 4's plan and the design's
     section 8);
   - the last ten lines of "## Gates".

   Stage 5's section heading goes just above "## Engine changes landed".
4. The design, `docs/superpowers/specs/2026-10-05-window-redesign-design.md`. Its section 8, the departures, wins over
   the parts where they differ (stage 4's plan holds every departure to 397; a few that only record tests as built,
   such as 392, 396 and 397, are in the plan alone). Its parts are in
   `docs/superpowers/specs/2026-10-05-window-redesign-design/`:
   - `09-checks.md` whole: the twelve checks (9.1), reading a deciding check (9.2), which records count and the
     headline (9.3), the page (9.4), the comparison after simulating (9.5), the short fit's timing (9.6), the card test
     as a check (9.7, E3's screen), refusals that say where to fix it (9.8, E6), what must never be shown (9.9);
   - `08-steps.md`: 8.1 (what the five pages share), 8.5 (Inference), 8.6 (Narrowing round), and 8.4's lines on the
     pre-run checks in Training's "Before you start" (C12);
   - `05-anatomy.md`, read against the page stage 4 built to it (`core/gui/panels/inference/step_page.py`,
     `core/gui/widgets/form.py`, `core/gui/step_facts.py`);
   - `07-plans.md`: 7.1 (the fill of every step) and 7.3 (what the window remembers, the inference rows);
   - `06-run.md` 6.8 ("Continue from batch N", for a round after a reopening) and `03-shell.md` (Home's "Continue with
     these" and its next step, the rail);
   - `11-engine.md`: 11.4 (E3), 11.5 (E4, an observation's name), 11.7 (E6), and 11.11 (the registry:
     `observation_name` arrives with E4);
   - `13-tests-stages-docs.md`:
     - 13.1: Table A's per-step tests for Checks, Inference and the Narrowing round, the settings with no control, the
       remembered table's inference rows, the control table, and `test_plans.py`'s row;
     - 13.2's "Stage 5";
     - 13.4: the walkthrough rows G8, G13, G26 and G28 to G30, runnable after stage 5;
   - what `04-look.md` and `12-findings.md` say these pages use;
   - each part's "Choices in this part the owner has not seen";
   - the locked records `decisions-as-approved.md` and `rulings-after-the-drafts.md`, in
     `docs/superpowers/specs/2026-10-05-window-redesign-assets/` (the later wins).

   Have readers read the parts against today's code. Read yourself what the plan's head needs.
5. The fit-saving design, `docs/superpowers/specs/2026-10-21-fit-saving-design.md`, for a kept narrowing round: 4.2,
   4.6, 6.2 and 8. With it, the fit-saving last review's table
   (`.superpowers/sdd/2026-10-06-window-redesign/fit-saving/reviews/last-review-two.md`), for its rows on a round.
6. Stage 4 as built:
   - its plan, `docs/superpowers/plans/2026-10-30-window-redesign-stage-4.md`. It is the model of a plan: "How to run
     this plan", the registry "Names tasks share", and the departures table with costs. Its departures 55, 72, 100,
     137, 141, 142, 149, 152 and 253 hand stage 5 its work.
   - its last review, in `.superpowers/sdd/2026-10-06-window-redesign/stage-4/reviews/`: the product seat's section 7,
     "What stage 5's planner must know that is written nowhere" (`stage-4-final-review-product.md`), and the sorting's
     rows B1 to B5 and H8 (`stage-4-final-review-sorting.md`).
7. The prompts of 30 October (stage 4's plan) and 5 November (stage 4 from Task 14: the last review and the exit),
   beside this file. They hold everything this one does not repeat: the preparation and review loop, the tools, and
   the rules every agent was given. Where they differ the later wins, and this one wins over all.

**What to do.** Write the plan of stage 5 from the design's parts as they stand, with the `writing-plans` skill.
Commit it when it is written, and give it to me to approve. Build nothing before I have.

Stage 5 delivers, by part 13.2 (ruling D3):
- the Checks list, each check's form and result, and the headline (C15);
- "Inference" and "Narrowing round" rebuilt on the step page stage 4 built;
- these forms join the plan's fill, so that one press fills every step (departure 141's three later headings);
- the pre-run checks' state in the Training step's "Before you start" (C12; departure 149), and the run's history's
  `check` filled;
- the comparison when simulation ends (B1, C16; departure 100's "Not compared");
- at its end, "Parameter inference" is final.

From stage 4's rulings with me (my answer of 30 October, and the last review's B1 to B5):
- Resume after a reopening for a kept narrowing round, and its "Continue from batch N" (departures 137 and 152);
- the rail marks a kept round (the second pictures' finding (b)6);
- the narrowed question: "inference on any other observation asks first";
- the epochs' words for a round (`expected_epochs` for a narrowing round);
- Enter on a probe row's close button, which today loses a typed row with no question (my answer 146: the rule is
  window-wide).

Engine:
- E4: an observation's name, with the registry key `observation_name`;
- the rest of E6, as B14 gives it: the input-use check's five plain errors and the third identifiability method's one;
- E3's screen: the card test as a check (its engine half landed in stage 3).

Find their lines again at today's head: stage 4 moved the code the design cites. `core.refusals.FIELDS` holds 114
keys today. Recount it before the registry's rows are written.

Tests rewritten first:
- Table A's per-step tests for Checks, Inference and the Narrowing round, re-pointed and their wording rewritten;
- the settings with no control;
- the engine's tests of an observation's name and of the six refusals.

Completed here: `test_plans.py`'s test that one press fills every step. New: `test_checks.py`.

The exit, by part 13:
- the gate, the display-marked run, and pictures;
- the walkthrough rows this stage creates, and G49's second half (it waits for the narrowing round's row);
- THE SLOW SET, owed at this exit for stages 4 and 5 together. It took 1 h 0 min 49 s at stage 3's exit, alone, and
  the machine has read slower since.

Say in the plan whether the card test is owed, and why. It follows changes that touch training or move tensors: a
round's Resume starts a training, and E3's screen runs the card test. If a task changes anything under
`core/diagnostics` that moves tensors (E6 is there), `CLAUDE.md`'s diagnostic card is owed too.

Say honestly how many tasks you expect:
- the design estimated 12 for stage 5, and 15 for stage 4;
- stage 4's plan had fourteen building tasks and held to them. It grew by fix rounds (Tasks 7, 9 and 12 needed two
  each), one fix wave after its last review, and three small tasks at its exit (15c, 15d and 15e, each with a fix
  round). It took seven chats after its plan's, from 30 October to 9 November.
- every fast gate now costs about 45 minutes. Count it in the plan's time.

Put the preparation that worked into the plan:
- For a task that touches threads, the run, the engine or what a run is handed, one skeptic (opus) first builds it as
  a stand-in in a scratch copy and runs its tests red then green. It brings what it finds to the controller, whose
  rulings go INTO the one brief.
- One implementer (sonnet) per task, from that one brief.
- On DONE, at the same moment:
  - the one-process fast gate;
  - three readers: seat one (sonnet) checks the brief against what was built; seat two (opus) compares the product
    before and after; seat three (opus) reads the tests and the faults they would miss, and runs its fault checks
    after the gate.
- An Important goes to one skeptic, with your conditional rulings in a file.
- One fix round, re-read by one reader while a fresh gate runs.
- A last review over the whole stage, with one fix wave.
- The exit, with me: pictures first, any change I ask for as its own small task through the same loop, then the gates
  of record once on the final head.

Ask me, one question at a time, in plain words, with a picture where it helps and your recommendation, what the parts
leave to me. Decide the small things yourself, and write each as a ruling with its cost.

**What the plan must carry.**

- THE FIRM REQUIREMENT stands for every page that hosts a run: "at the window's least size, and at every width above
  it, nothing of the run page is hidden under the form; the form and the run page share the width with neither drawn
  over the other". Stage 4's tests measure it at text 1.0 and 1.5, and at 150 % display scaling in a fresh
  interpreter. The rebuilt Inference, Narrowing round and Checks pages join them.
- From stage 4's last review (the product seat's section 7 and the sorting's B1 to B5 and H8; written nowhere else):
  - `InferenceScreen._fan_out` is the one place that hands a configuration to the Inference page, called by
    `take_prior`; `install_config` has no product caller. A rebuilt Inference page takes its configuration from there.
  - `load_chain(observation=…)` raises `ValueError`: the observation's link is stage 5's to build, after the
    posterior's. Home's "Continue with these" then gains the observation in `Picks` and `last_time_sentence`.
  - A kept round is offered today as kind "round", with "Open Training" and a command line (`continuing._offered`,
    `_draw_the_kept`, `retry.py`). `chain._ACTIONS` has no "round" action, `resume_call` hands only `resume_fit` and
    a name, and `step_states` lets "round" and "fit_again" change no step. The round's Resume must hand the narrowing
    stage `resume_fit` and nothing it compares: the stage refuses the directions and the HPD level given again.
  - `orchestrator.matching_fits` answers for an amortized training only. The round's page needs its own question
    before a new round.
  - `continuing._history_offer` reads only "Training": a round's history offers no Continue.
  - `PosteriorPanel._refusal` says a "resume_fit" refusal on the row of `_last_resumed`, which is never cleared. A
    round's page that resumes needs its own record of which row it pressed.
  - `fields.ON_TRAINING_ALONE` (`checkpoint_every`, `resume`, `max_num_epochs`, `expected_epochs`) name step 3 alone.
    A round page that gains those controls adds its place to their entries, as `num_runs` has, and drops them from
    `ON_TRAINING_ALONE`.
  - The round's run hands no `plan` or `settings` to its history. Its posterior reaches step 3's "In use" through
    `take_posterior` and `PosteriorPanel._run_ended`'s re-listing: a rebuilt round page keeps both ways in.
  - "No re-listing while live" relies on the controller's order (the status is no longer live before `run_ended`).
    The round's page gets that pin, and the plan says the order is load-bearing.
  - Pages hand the run names and numbers only, never a record or an array; `history.plain` copes with nothing else.
  - `_others_pending` on `StepPage` (step 2 saying that step 1 has changes not applied) is taken by Checks, Inference
    and the round for the steps before them.
  - The round's name rule is the training's: a kept round whose name another posterior has taken since meets the
    dead end stage 4 cured for Training. Cure it the same way.
- From stage 4's plan:
  - 55: step 4's states until the checks are counted;
  - 72: step 6's observation on offer, read today from the old round panel's own list;
  - 100: the "Unusable driven recordings" row reads "Not compared" until Checks gives the comparison;
  - 141: a plan's three later headings are read and shown, not filled;
  - 142: closing still writes what the unrebuilt parts write (`inference_validate`, `inference_infer`,
    `inference_tsnpe` and others in the remembered table);
  - 149: the pre-run checks' fact and question are left out of Training;
  - 253: the narrowed question's "inference on any other observation asks first" is untrue until Inference is
    rebuilt: today its page refuses unless "Run on a different observation" is ticked.
- From stage 2 (`stage-2/reviews/stage-2-final-review-sorting.md`, its three rows for stage 5):
  - 921b: the Infer step's remembered cell is written back as the folder's first cell when the step is built and the
    window closed with no configuration;
  - 951a: "the χ probe table" in `recording_probe`'s sentence names no caption;
  - 956f: old place words in `fields.py`'s docstrings and in the first sentences of the step 4 and 6 tests'
    docstrings.
- From the fit-saving stage's last review: the narrowing command's Ctrl-C advice is wrong for a `tsnpe` given
  `--directions` or `--level`, and a round's kept training resumed through `train` dumps the region. Both are on the
  code audit's list; say in the plan whether the round's Resume meets either.
- The code audit's list, `docs/superpowers/notes/2026-10-29-the-code-audits-list.md`. This stage must not lose it,
  and adds to it: every engine change goes on it.
- The departures go on from 398.

**Mine to hear, still open:**
- `CLAUDE.md`'s rule on what the inference tabs remember, which these pages change, only on my yes;
- the guide's pages for Checks, Inference and the round, at this stage's exit, as stage 4 did for steps 1 to 3.

The lists of what was decided without me are in the ledger's `stage-4/exit/rulings-for-the-owner.md` and
`rulings-for-the-owner-2.md`, and the earlier stages' `exit/` folders. I may undo any of it. Keep giving me such a list
at each stage's exit.

**Facts the plan stands on** (measured; the ledger has each):

- The fast gate holds 3572 tests (3588 collected). It took 43 min 28 s on a quiet machine at `850dc23`; stage 3's took
  27 min 16 s at 2010 tests, and `CLAUDE.md` records both. The machine reads slower even alone: the engine's slowest
  test took 257 s, against 182 s at stage 3's gate, why not established. The four suites stage 4 added take 3 min 30 s
  alone.
- The gate tools are in the scratch folder `C:\Users\J\AppData\Local\Temp\prism-redesign-scratch\s4\`:
  - `gate.sh <fast|quiet|slow|display|collect> <label>`, which logs one pytest process into `stage-4/gates/`;
  - `ground.ps1 before|after <label>`, run as `powershell -NoProfile -ExecutionPolicy Bypass -File ground.ps1 …`;
  - `card\card-gate.ps1` (the card test), `exit_fence.py` (the engine fence) and `task01_test_names.py --check
    carried-over.md`.

  Their text copies are in the ledger's `stage-4/tools/`. The plan's Task 0 points them at a workspace of stage 5's
  own, `stage-5/`.
- The carried-over list is `.superpowers/sdd/2026-10-06-window-redesign/carried-over.md`: 1642 rows, missing 0, gone
  63 (each closed), at `850dc23`.
- `core.refusals.FIELDS` holds 114 keys; there are 34 suites.
- No kept training stands in `Artifacts/`: it holds `diagnostics/`, `fdt/`, `priors/` and `simulations/`, and no
  `fits/` folder. The named training code (`core.SBI.fit_state.FIT_CODE`, seventeen functions in `core.SBI.chi`,
  `chi_encoder`, `embedded_network`, `summaries`, `train` and two in `core.orchestrator`) was not on stage 4's path.
  Say in the plan which named functions a task would touch, if any, and ask me first if one would: an edit there makes
  every kept training unresumable.
- In this checkout `run.bat` and `run.sh` read `i/lf w/crlf`. An edit to `run.bat` is an edit to the pin in
  `tests/test_shell.py`, in the same commit.
- `AskUserQuestion` waits for my answer: ask while long runs go, not before them. I often follow over Remote Control,
  where the next-step box may not show: put a question that waits on me in the closing message too.
- The rules every agent was given (the 28 October prompt lists them in full):
  - pytest is started by one tool call at a time;
  - no `python -c` beside another python, and readers run NO python while a gate runs;
  - a test that reads state the whole process shares takes a reading first;
  - a `cd` inside a Bash call moves the session's working directory;
  - never write into the repository while an implementer or a picture agent works there;
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
