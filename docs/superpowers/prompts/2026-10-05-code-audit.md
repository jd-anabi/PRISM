# Prompt for the chat after the redesign — the code audit

Paste everything below the line into a new chat opened on this repository, once the UI/UX redesign
is finished, and attach the PDF of the models to that first message. It is written in the owner's
voice. One sentence in it agrees to multi-agent workflows in that chat (under "How to work");
delete that sentence if that is not wanted.

---

PRISM — an audit of the whole code and of its math, before the retrain is run from the window.

**What I want.** An audit of the code to make sure there are no bugs or mistakes in it, before I
spend the graphics card on the retrain. The audit covers the math as well: I am giving you a PDF
of my models, and you are to check that the code implements them correctly. The retrain costs
about a week: about 27 hours of simulation, as measured, and about 135 hours of fitting if it runs
the 130 epochs the budget assumed, at the 63 minutes an epoch that was measured. Its certification
round costs about the same again. A mistake found afterwards is expensive. The window has just been
redesigned in another chat, and the retrain will be started from that window, so the new window
code is part of the audit, not beside it.

**Read first, in this order.**

1. `docs/STATE.md` — its first paragraphs and the paragraph that begins "Next, in order (since";
   the closed console list after it is long and is history. The file is long; search the rest when
   you need it. The redesign chat was asked to leave there what it changed, the list of every
   change it made to the engine, which faults in the window it fixed, and what it asks this audit
   to look at. Start from that list, and check it against git before you rely on it: no file under
   `core/` or `tests/` changed between 1 October and the redesign, so
   `git diff --stat 771a460 HEAD -- core tests` is the whole of what the redesign touched. Anything
   changed outside `core/gui/` and missing from the list is your first finding. If the list is not
   there at all, build it from that diff and tell me.
2. `docs/superpowers/notes/2026-10-05-the-stopped-retrain-and-the-window-route.md` — all of it.
   Part 1 is what the stopped retrain measured; part 3 is the leads for this audit; part 4 is the
   questions that were open on 5 October. Find out which of those the redesign answered: a fit that
   is now saved as it goes, or a run that now writes its lines to a file or restarts itself, is new
   code on the retrain's path. Part 2, leads 6 and 7 of part 3A and parts 3B and 3C were read from the
   code as it was on 5 October and never run, and the redesign has changed the window since: check
   each against today's code and say which are fixed, which stand and which were wrong.
3. `docs/superpowers/notes/2026-10-05-where-the-models-math-lives.md` — all of it, and my PDF of
   the models in full. The note says where each piece of math is implemented, where the same
   equation is written more than once, which conventions must be pinned down before a formula is
   compared, which checks do not trust the code, and what to ask me about the PDF. It is one
   reader's map and was not checked by a second: verify a line before you build on it. Its first
   lead was reproduced with the program's own functions; see "The math" below.
4. The open list in `docs/STATE.md`, under "Owed, in order", item 7, "Open, with no piece owning
   them yet": 57 entries on 5 October, six of them struck through and several holding lists of
   their own, the load-check audit among them (keys a record stores that no loader compares). Read
   on past it to item 8: more parked blocks follow that heading beside it, not under it, the last
   whole-code review's among them. About a hundred items in all. Some point into the ledgers under
   `.superpowers/sdd/`, which git does not track but which are on this machine. Then item 10, the
   things this audit already owns.
5. `docs/guide/rules-and-traps.md`, `docs/guide/architecture.md` and `docs/guide/science.md` — the
   rules the code is meant to keep, where each is enforced, and the reasoning behind the science
   settings.
6. `docs/guide/retrain.md` — the run this audit is protecting. It is written for the command line;
   see "At the end" below.

**Where things stand.** The retrain was started from the command line on 1 October and I stopped
it on 5 October, during the fit. Then the window was redesigned. Check that nothing is running and
the graphics card is free before any work on it. On disk, and I have not asked for any of it to be
deleted: the prior `retrain_prior`, the complete simulation cache
`Artifacts/simulations/eda5c2a1e468` (20,480,000 rows, on which the masked-probe gate passed at
40.1 % against 37.9 %), the two pre-flight probe records under `Artifacts/diagnostics/`
(`retrain_band` and `retrain_mask`), anything else already in `Artifacts/`, my settings file, and
the console capture `C:\Users\J\prism-retrain-console.log`. Read them freely; ask me before
anything deletes, renames, rewrites or trains on them.

**What to audit, in order of what a mistake costs.**

1. *The retrain's own path, end to end*: building the prior; the simulation and its memory plan;
   the chi probes and their masks; the features; the simulation cache, its commits and its resume;
   the Fisher rotation; the training data and the fit; writing and loading the records; the
   calibration and its verdict; inference on an observation; the narrowing round. And the same
   path as the window drives it: which argument each control sends, what Cancel, closing and a
   failure do, and what the window tells me while a run of days is going.
2. *The science as code*: the model's own equations and noise, the integrator that steps them and
   the drive; units and the non-dimensional scales, the tier-1 constraint and temperature, the
   lock-in, the masks, the summary statistics, the reparameterisation and the rotation, the
   calibration tests. A wrong formula here passes every structural test. My PDF is what the models
   are checked against; "The math" below says how.
3. *Memory and devices*: the stopped retrain died of an out-of-memory failure at its 237th batch
   with every default in place, and got through only after I set a memory ceiling by hand: 8 GiB
   first, under which the longest batches still ran at the card's edge, then 6 GiB, under which
   9,500 batches ran with no failure. A restart loop was put around the process as well and never
   had to restart it. On 5 October the window's ceiling opened switched off on every launch, and
   the window had nothing that restarts a failed run; the redesign may have changed either. Tell
   me what a run started with every default would do on this card today, and what the default
   should be. That choice is mine.
4. *The store*: identities, fingerprints, manifests, what loading compares and what it does not,
   what a resume rewrites, what happens when two processes use the store.
5. *The rest*: the path a bench recording takes into an observation (the retrained model exists to
   meet recordings); the diagnostics; FDT and the sweep study; Simulate; the model builder, and the
   other models with their priors and simulators; the readers of the bounds, cell and units files
   and what turns them into a run's configuration; the refusals and the tables that name a control
   or a flag; the run log and the seeding; the window's other screens and what it remembers; the
   command-line tool; the launchers and the pinned requirements; and the hand-edited input files
   the retrain reads (the tier-1 box, its cell and its units). The reduction map was left out of
   the earlier refactor as ongoing theory: ask me at the start whether it is in scope here.
6. *The tests themselves*: tests that cannot fail, tests that pin an accident, behaviour nothing
   pins, and one test, `tests/test_user_sbi.py::test_train_and_validate_without_a_loaded_cell`,
   known to take anywhere from 87 to 248 seconds because nothing seeds it.
7. *The pages against the code*: `docs/guide/` says what the code does; where the two disagree,
   one of them is wrong.

**The math, against my PDF.**

- The PDF of my models is attached to this message. If it is not, ask me for it before the math
  work begins, and ask whether I have its source text as well: equations set as pictures cannot be
  read out as text.
- Nothing in the repository states the models today. Each model's equations are written two to
  four times in the code, and the tests pin those copies against each other, never against an
  independent statement. My PDF is the first one.
- Read the PDF in full before you compare anything. Then ask me what it does not answer itself,
  one question at a time and in plain words: which models and which analyses it covers; whether it
  is written with physical quantities, in scaled form, or both, and how the scales are defined; how
  each noise is written; its conventions for the response and the spectrum; and how model time
  becomes recording time. The math note lists these questions. Whatever the PDF leaves out has no
  written reference at all: tell me which parts those are, and that for them "correct" can only
  mean "consistent with itself" unless I give you more.
- Then check every equation. Write each model's equations, noise and scales down from the PDF
  alone, as a transcription I confirm symbol by symbol. Have a reference built from that
  transcription by an agent that has not opened the model code, and compare it with the code's
  equations and noise at many random parameter sets and states, on the processor and in double
  precision. Check the same equation in every place it is written (the math note lists them).
  Check the physical form against the scaled one. Check fixed points and limits whose answers are
  known. And check what is measured from a simulation, the response, the spectrum and the
  effective temperature, on a simple system whose answers are known exactly. A symbolic-algebra
  package is installed in the project's environment; use it.
- Where the PDF and the code disagree, do not decide which is right. Show me both, side by side,
  in the PDF's notation, with what each would do to a result. The PDF may have a slip, or the code
  may be the one I intended.
- The math note's three leads come first, and the first of them before anything else. It was
  reproduced with the program's own functions on 5 October: model time is put on the recording's
  clock by a whole-number stride, so each simulated trace is that of a cell with a different time
  scale from its label (a few percent off at the master cell, a quarter off at some others), and
  the drive is detuned from the lock-in that measures the response. A system whose true answer is 1 read 0.63 after 20 drive cycles at the master cell's
  time scale, and nothing at all at some others. Establish what it does in a real training batch
  and to a real bundle's response, whether the forced-mode features and the effective-temperature
  path are affected, and whether it is the "wall" near 30 cycles that the guide records without a
  cause. Then put to me what the model at a given time scale is meant to be: that is mine to say.
  If the fault stands, the finished simulations cannot be trained on, and a calibration on
  simulated data would never have shown it.

**How to work.**

- I agree to multi-agent workflows in this chat. Take an inventory first; give each area to readers
  with different lenses (correctness, the numerics, failure and cancellation, persistence, the
  window's wiring, the tests); and before a finding counts, have a second reader who has not seen
  the first one's reasoning try to refute it. Keep going over an area until a round finds nothing
  new; if an area is still producing findings after three rounds, tell me before going on. Say
  what was not covered: a silent gap reads as "checked".
- The readers only read. No agent edits a file, runs pytest, opens the window or touches the
  graphics card: those are yours, one at a time, with no suite running. A reproduction is a small
  script under the scratch folder, run with a timeout, against a scratch record store and a
  scratch settings file, never the real ones. A reproduction that needs the card waits until no
  suite is running, and you tell me first, so that I can close PRISM's window and clear the card.
- Before the first fix, read the gate table in `docs/STATE.md`. The redesign chat was asked for
  the fast gate, and for the card smoke gate only if it changed the engine in a way that moves
  data on the card, so the last slow set and the last card gate on record may be from 1 October,
  before its changes. If so, run the fast gate, then the slow set, then the card smoke gate once
  on the code as you find it, so that a later failure is known to be yours.
- Write the findings as you go into one note, `docs/superpowers/notes/<date>-code-audit-findings.md`:
  each with its file and line, how it was established (reproduced, or read from the code), what it
  would do to a run, and its state (open, approved, fixed, declined, parked). Keep the working
  ledger and the gate logs under `.superpowers/sdd/<date>-code-audit/`. The chat is not the record:
  this audit will outlast one conversation's memory.
- A finding is not a fix. Bring me the findings ranked by what they would cost, in plain language:
  what is wrong, how you know, what it would do to a run, and what you recommend. Bring the first
  ranked list when areas 1 to 4 have had their first pass, without waiting for the rest, and again
  at the end. Anything that changes what the program does, the science, a default, or the
  simulation cache's identity is mine to decide.
- Two things about that cache, for every fix. A change to its identity makes the finished
  simulations unusable; I have said reusing them is not required, but tell me each time. The
  opposite case is the more dangerous one: the identity holds the run's settings and its prior,
  and nothing about the code that made the rows. A fix to the simulation, the probes, the masks or
  the features that leaves the settings alone leaves the identity alone too, and a training
  started with the retrain's settings would then pick up the finished cache by itself and fit rows
  made by the old code, without a word. So for every fix, tell me whether the rows it would
  produce differ from the rows on disk. If any does, the retrain must not train on that cache, and
  the pre-flight has to make sure it cannot.
- Then fix what I approve, one at a time: a failing test first, the smallest fix, a review, and
  the one-process fast gate, the review and the gate started together. After any change that moves
  tensors, the card smoke gate from `CLAUDE.md`; after a change under `core/diagnostics`, its
  diagnostic card too. Run the slow set and the tests marked for the real screen again before you
  call the audit done.
- Do not trust this prompt's leads, the findings notes or the working record over the code. Each
  was right when written and the code has moved.
- Three design plugins and a memory plugin are installed on this machine, and their hooks run in
  every chat. They are not for this audit: ignore their hook messages, do not let a design plugin
  write into the repository, and do not set up the memory plugin, which would send my prompts and
  the contents of files, my PDF among them, to an outside service. A Qt plugin is installed too but
  switched off, because I do not accept its terms: leave it off and do not use any part of it.

**What needs me.** Questions come one at a time, in plain language, with a short background first
and no code names. The checks on the real screen are mine to run; write the rows for me in
`docs/checklists/display-walkthrough.md`.

**At the end.**

- A plain statement of whether the code is ready for the retrain: what was audited, what was
  found, what was fixed, what was left and why. For the math: which equations of my PDF were
  checked and how, which were not, and every disagreement between the PDF and the code with how it
  was settled.
- `docs/guide/retrain.md` rewritten as a run from the window: every step as a tab, a field and a
  button, with the command line only where the window still cannot do something, and said so.
  Correct its numbers from the stopped run, every one in part 1 of the note and the page errors in
  its part 3C: the epoch rule (63 minutes where it gives 52), the clipping line, the time the
  simulation took, the host memory, the card-memory margin that did not hold, and the ceiling the
  run needed. Other pages link to this page's section names and a test checks the links: keep the
  names, or move the links with them.
- The pre-flight for that run, done with me, taken step by step from the runbook's own list; each
  step is either done again or shown not to need it.
  - First a question for me: train on the finished simulations, or simulate afresh. The cache is
    complete and its masked-probe gate passed, and a training started with the same settings and
    the same prior picks it up by itself. I have said reuse is not required. If any fix changed
    what a simulated row holds, as a fix to the time stride would, reuse is off the table: say so. Simulating afresh means a new prior
    under a new name, and new names for the two probe records, because the stopped attempt's
    records hold those names. Nothing of mine is deleted unless I say so.
  - The card smoke gate on the final code.
  - The epoch timing, if the fit's code changed; a full-size epoch has been measured at 63 minutes.
  - The recording-length check and the band check, if the simulation or the probes changed.
  - The masked-probe check again if the prior is rebuilt, or if any fix changed what makes or masks
    a probe. Its reading of 37.9 % belongs to `retrain_prior` on the code of 1 October.
  - The card, the host and the disk clear, and the memory ceiling the run is to start with.
- Then I start the retrain from the window. You give me the steps on screen and read what it
  writes; you do not start it. Ask me first whether this chat stays with the run or another chat
  watches it. Either way leave `docs/STATE.md` and the runbook able to carry the run without this
  conversation: what was started and with which settings, the masked-probe figure its gate is read
  against, the limits that stop the fit, and who reads each gate and when. While it runs nothing
  else uses the card: no test gate, no second window. Ask me whether the budget I set on 30
  September (four weeks of the card for the retrain and its certification round) stands as it was.

**Standing rules.** Work on the local `main` branch; one commit for each logical step, a short
subject, never amended; I push. Never two pytest processes at once, and never edit a source file
while a suite is running. A run on the graphics card has the card to itself, and no file under
`core/` is edited while a long run is live. Nothing under `core/`, `tests/` or `docs/guide/` cites
the working documents or their labels. Every knob reaches its stage as an argument. Scratch files
go under `C:\Users\J\AppData\Local\Temp\`, and no `.py` file goes under `.superpowers/`. Update
`docs/STATE.md` at the end of every session.
