# Saving the fit as it goes — design

**Status:** for the owner's approval. Written on 2026-10-21 in the chat that finished stage 2 of the window
redesign, from the owner's answers F1 to F10 (the redesign ledger's section "The design for saving the fit",
`.superpowers/sdd/2026-10-06-window-redesign/progress.md`) and one reader's measured evidence
(`.superpowers/sdd/2026-10-06-window-redesign/fit-saving/recon-report.md`, with every file:line; scripts in
`C:\Users\J\AppData\Local\Temp\prism-redesign-scratch\fit\`). The approved redesign
(`2026-10-05-window-redesign-design.md`, U4) left this to "a SEPARATE design, to accept or leave"; it is made
between stages 2 and 3, as the owner decided on 2026-10-06.

Words used here: a **fit** is the training of the neural posterior on the rows of a simulation set (sbi's training
loop, many passes over the data); a **pass** is one epoch; a **kept fit** is a fit saved partway, which can be
resumed.

## 1. Intent and why

A fit at the retrain's size takes days (62 passes of about 62 minutes each on 20,480,000 rows were measured on
5 October). Today it is one training call with no save: a crash, Windows restarting the machine, a power cut or the
window closed by mistake loses every pass already made (the simulation set survives). The owner stopped the
retrain of 5 October on purpose, to start fresh, so that loss cost nothing (F1); what this design protects against
is an ACCIDENTAL loss (F2). It matters more because the owner may change the network's architecture after the
redesign, so fits will be run again (F2).

## 2. Goals, non-goals, anti-goals

**Goals.**
- After any end of the program during a fit, at most the pass in progress is lost (at full size about an hour).
- A fit resumed in a new process continues the SAME fit: the same validation rows, the same optimizer state, the
  same pass count, best loss, early-stopping count and best weights (F7).
- The person resumes by hand, from a clear statement of where the fit stopped (F3); nothing resumes by itself
  (U4: no unattended restart).
- A kept fit is never resumed into anything that differs from what it was trained with: settings, network shape,
  the code that builds or trains the network, the rows (F8).

**Non-goals.**
- A posterior from an unfinished fit. Only a finished fit gives a posterior (F6).
- Making Stop act sooner than the end of the pass in progress (it stays up to an hour at full size; U12 stands).
- Restarting a fit by itself after a crash or a restart of the machine (U4).
- Keeping the machine awake, or stopping Windows from restarting it.
- A fit whose rows are not in a kept simulation set (see 4.7).
- Digit-for-digit equality of a resumed fit with an uninterrupted one ON THE GRAPHICS CARD (section 6.5).

**Anti-goals** (it fails, though it "works", if):
- a fit that is never interrupted trains differently from today: what a trained posterior is must not change;
- a kept fit is resumed after the network or a setting changed, silently;
- a save can leave a half-written kept fit that is then resumed;
- the simulation set a kept fit needs can be deleted under it;
- anything changes what a simulated row holds or the simulation set's identity (the finished set
  `Artifacts/simulations/eda5c2a1e468` must stay usable).

## 3. Constraints

- The science does not change (the owner's standing rule): sbi stays the trainer; the training method, the
  network, the early stopping and the best-weights rule are sbi's, as today (F10: PRISM hooks into sbi's own
  trainer; it does not run the passes itself).
- `CLAUDE.md`'s rules: every knob reaches its stage as an argument; a refusal is a `core.refusals.Refusal` with a
  field key and each front end names the control or flag from its own table; nothing printed or logged inside a
  save's commit steps; nothing shippable cites this design or its labels.
- sbi 0.26.1, torch 2.9.0 and nflows 0.14 as installed; the hook relies on private details of sbi 0.26.1's
  trainer, so the design must make an upgrade that breaks it fail loudly (7.1).
- The training side is built as ITS OWN SHORT STAGE BEFORE STAGE 3 (F9); the window's words and buttons go into
  stages 3, 4 and 6 as they rebuild their pages (section 8).

## 4. What a person meets

### 4.1 During a fit
At the end of every pass PRISM saves the fit (about 61 MB in 0.07 s at the retrain's size, against a pass of about
an hour). The run's page may say so in a line of its log ("Saved after pass 41."); nothing else changes during a
fit.

### 4.2 After an accident (F3)
When PRISM opens and a kept fit exists that was not stopped on purpose, Home's next step and the Training step say a
training stopped and show where: the pass it reached, when it was last saved, the best validation loss so far, the
passes since the best, and the simulation set and prior it trains on. A button "Resume" starts it again (after about
four minutes of reading the simulations back, said in "Before you start"). Nothing happens until it is pressed.

**A resume runs on the kept fit's OWN record, never on what the Training step's fields hold now** (F3: one button).
The window reopens every science setting at its default on each launch, so after a crash its fields no longer
show the fit's settings; the kept fit therefore records everything needed to run on: every fit setting, the bounds
and the cell, the prior, the simulation set, for a narrowing round its observation and parent posterior, and the
name and note given when the training started (a run's own record, not something the window remembers).

### 4.3 Stopping on purpose (F4)
"Stop training…" (U12) stops at the end of the pass in progress, after that pass is saved. Then PRISM asks:
**"Keep this training so you can resume it later?"** with "Keep" and "Discard". Kept, it becomes a kept fit;
discarded, its record is removed. The question's words are the run stage's to write; the choice is this design's.

### 4.4 Starting a training (F5)
When the person starts a training and one or more kept fits MATCH the settings as applied (section 5.3), PRISM
offers a choice: resume one of them (each shown with where it stopped) or start a new fit. A new fit leaves every
kept fit as it is. Kept fits that do not match are not offered.

### 4.5 Kept fits in Records (F5)
A kept fit is a record of its own, with a name and a date, listed in Records under its own kind (working word:
"Unfinished training"). Several may be kept, even with identical settings. One that can no longer be resumed says
why: "the code that builds or trains the network has changed since", a library changed, or its simulation set is
gone or changed (its own settings travel with it, so the window's fields never make it unresumable); the person
deletes it there. A kept fit is deleted by PRISM itself only when the fit it holds
finishes and its posterior is written (F7).

### 4.6 The narrowing round (F7)
A narrowing round trains a network too; its fit is saved and resumed the same way, from the round's own
simulation set.

### 4.7 Which fits can be kept
A fit can be kept only when its rows come from a kept simulation set (the window and the tool's `train` keep one by
default). A fit trained with the simulation set not kept (the tool's checkpointing switched off) cannot be resumed,
and "Before you start" (and the tool's opening lines) say so.

### 4.8 The command line (F7)
`train` and `tsnpe` can resume a kept fit named on the command line (working flag `--resume-fit <record>`, refused
with its field key when the record does not match, naming what differs). Without it the tool starts a new fit and
prints one line naming any kept fit that would match. Ctrl-C, a crash or a power cut leave the fit kept (the tool
cannot ask "Keep or Discard"; a kept fit not wanted is removed with `artifacts rm`). `resume` already names the
simulation set's policy, so the fit's field and word are new (one refusal key added).

## 5. What a kept fit is

### 5.1 What it holds
Written ONCE, when the fit starts:
- its identity (5.3);
- the random state of the CPU generator taken just before sbi draws the train/validation split (5,056 bytes), with a
  digest of the validation indices it produced, so a resume regenerates the identical split in 0.33 s and checks it
  (rather than storing the 164 MB permutation).

Written at the end of EVERY pass (one snapshot, taken where sbi summarises the pass):
- the network's weights (15.4 MB at the retrain's size: 3,791,805 parameters);
- the optimizer's state (30.3 MB);
- the best weights so far (15.4 MB);
- the next pass's number, the best and the last validation loss, the passes since the best;
- the loss histories (sbi's summary lists);
- the CPU generator's state at that moment (the shuffling of the next pass continues from it);
- the cumulative wall-clock time of the fit and the number of processes it has run in.

### 5.2 How it is written
Exactly as the simulation set's checkpoint already is (`core/SBI/training_checkpoint.py`, `file_manager._atomic_write`):
a sibling temporary file, flushed and fsynced, then an atomic replace; the snapshot of one pass back is kept beside
it (two generations, about 123 MB on disk with the split's state); a reader falls back to the older generation if
the newer one is unreadable; the whole save runs inside the cancel-deferred block, with nothing printed or logged
between its steps. A torn save therefore never leaves a kept fit that resumes from half a snapshot.

### 5.3 When it may be resumed: its identity (F8, strict)
The kept fit records ALL of these when it starts, and may be resumed only when the resume finds every one of them
the same (its settings are its own, 4.2; the rest is read from the code, the libraries and the simulation set as they
stand):
- **the rows**: the simulation set's identity digest and the number of its committed batches, in the order the
  shards are read (the same rows in the same order);
- **the rotation and the box**: the bijection probe the simulation set already checks (the rotation comes from the
  simulation set's header, as it does for a resumed simulation);
- **every fit setting**, by value, INCLUDING the sbi defaults PRISM does not pass today (they are recorded so a
  later change of default cannot pass unseen): hidden features, transforms, bins, the density estimator, learning
  rate, early-stopping patience, the pass ceiling, training batch size, validation fraction (0.1), gradient clip
  (5.0), the z-scoring of parameters and of the summaries, dropout, batch norm, the data device, the device and the
  number type;
- **the network's shape**: a digest of the names and shapes of the network's weights as built;
- **the code that builds or trains the network**: a digest of the source text of each named function and class
  that does (the reader's list: the training call in `core/SBI/train.py`, `core/SBI/embedded_network.py`,
  `core/SBI/chi_encoder.py`, the summaries' winsorising, the chi block's normalisation and width, the training
  stage's network builder and its expected forcing width in `core/orchestrator.py`), together with the resolved
  values of the configuration constants that shape the network or the fit but are not arguments (the density
  estimator, the bins, the batch size, the rank-gauss knots, the winsor percentile, the chi encoder's dimensions,
  knots and layout);
- **the libraries**: the versions of torch, sbi and nflows, and a digest of the installed sbi trainer files.

Any difference refuses the resume, naming the first that differs by its on-screen name (the window) or its flag
(the tool), or "the code that builds or trains the network has changed since this training was kept". The digest
covers named functions and classes, not whole files, so an edit elsewhere in `core/orchestrator.py` or
`core/config.py` does not refuse; any edit inside a named one does, even a comment (the owner chose strict).

**When each check runs.** At a click, before anything is spent (cheap, so Home, the Training step, Records and the
tool's line can say at once whether a kept fit can be resumed): the simulation set is there with the same digest and
committed batches, the code digest, the libraries; and, when a training is being SET UP (4.4), whether the applied
settings equal a kept fit's own. During the resume, after the rows are read back (about four minutes): the
bijection probe, the standardiser and chi encoder fitted to the same values, the split's digest, the network-shape
digest. A late check that fails refuses the resume with its reason, leaves the kept fit exactly as it was, and marks
it "cannot be resumed: <reason>" (deletable, never deleted by PRISM).

### 5.6 The states of a kept fit
- **Running**: a process is fitting it. Its folder holds a lock while a process runs the fit (the kind the
  one-PRISM-at-a-time lock uses, so a dead process's lock is recognised at once); a fit run by the tool while the
  window is open therefore reads as running and is never offered for resuming. (The store's "recently written"
  rule cannot tell, with passes of an hour.)
- **Stopped unexpectedly**: no process holds its lock and no stop was recorded (a crash, a restart, a power cut).
- **Kept on purpose**: stopped by Stop and kept with "Keep", or by Ctrl-C in the tool (recorded as a stop on
  purpose; the tool cannot ask).
- A Stop whose "Keep or Discard" was never answered (the program ended first) stays KEPT: nothing is discarded
  without an answer.
- Home and the Training step offer every kept fit that can be resumed, newest first, each saying how it stopped
  ("stopped unexpectedly", "kept when stopped"); Records lists them all.
- What the window's stages are handed by the engine: a listing of kept fits (state, where each stopped, whether it
  can be resumed and if not why; cheap, 5.3's checks at a click), "keep" and "discard" for a stopped fit, and the
  training stage's argument that resumes one.
- Between this stage and stage 3, today's window asks nothing: a fit stopped by Cancel is kept (nothing lost),
  today's Records screen lists it and deletes it, and stage 3 brings the question of 4.3.

### 5.4 Where it lives
A new kind in the record store, `fit` (directory `Artifacts/fits/<name>__<id>/`), PROGRESSIVE like the `fdt` kind:
its directory and a first manifest exist from the start, the manifest is refreshed at every pass, and a stop or a
crash leaves the folder marked unfinished. Its `log.txt` gathers every process's records. Its parents are the
simulation set and the prior, so the store refuses to delete a simulation set (or a prior) while a kept fit needs
it, as it already refuses to delete a prior a simulation set needs.

### 5.5 At the end
When a kept fit finishes (sbi's early stopping, or the pass ceiling), the posterior is written as today; its
manifest's training block adds whether it was resumed, the kept fit's id, the number of processes and the
cumulative wall time (today `wall_seconds` covers the last process only). Then the kept fit's record is deleted.
A fit that is never interrupted writes the same posterior as today, with "resumed: no".

## 6. How it works

### 6.1 The hook (F10)
sbi 0.26.1's trainer (`NPE.train`) is left to run the passes. PRISM wraps the trainer instance's per-pass summary
method (`_summarize_epoch`) to take the snapshot of 5.1 there; the training call stays sbi's own. The reader built
this at tiny size on the CPU and killed the fit after passes 0, 10, 37, 38 and 39: each resume in a new process
(another random seed, the data re-read, the network rebuilt) matched the uninterrupted fit exactly (41 passes,
identical loss histories, best loss, early-stopping count and final weights).

### 6.2 Resuming, in order
In the new process: read the simulation set's rows in shard order (as a resume of the simulation set does);
fit the standardiser and the chi encoder as today (deterministic, so this is also a check that the rows are the
same); build the network with PRISM's builder; move it to the device; create the optimizer and load its state
(after the move); restore the split from its random state and check its digest; restore the CPU generator's state,
the pass counter (the next pass's number), the best and the last loss, the passes since the best, the best weights
and the histories onto the trainer; then call sbi's `train(resume_training=True)`. Restoring the next pass's number
from a snapshot taken at the summary avoids a quirk of sbi's own in-process resume, which counts the last pass twice
towards the patience and was measured stopping one pass early.

### 6.3 Code changes
- `core/SBI/train.py`: the training call split into "prepare the rows and the network" and "fit", so a resume can
  restore between them; the hook; the restore.
- A fit-state writer and reader beside `core/SBI/training_checkpoint.py` (about 200 lines, its patterns).
- The record store: the `fit` kind (kind tables, manifest keys, the progressive writer, the delete rule).
- The training stage (`build_posterior`, and through it `tsnpe_round`): an argument naming a kept fit to resume and
  one switching saving off (default on when the rows are in a kept simulation set); the identity of 5.3; the
  refusals; the end of 5.5.
- The tool's `train` and `tsnpe`: the flag of 4.8; the refusal key in both front ends' field tables.
- No change to the simulation, the features, the simulation set's identity or what a row holds.

### 6.4 Stopping
The window's Cancel already takes effect where sbi prints the pass's line, that is after the pass's save: a stop
loses nothing finished. Ctrl-C in the tool, a crash or a power cut lose at most the pass in progress. (The window's
words "up to ~1 min during training" are wrong today, `main_window.py:771`, `base_panel.py:324`; stage 3 rewrites
them as U12 has it.)

### 6.5 What a resume costs, and what it does not promise
- About four minutes before the first resumed pass at full size (measured on a fifth of the real simulation set and
  scaled: reading 10 to 15 s, the winsorising and the standardiser's fit about 97 s each, the rest seconds); host
  memory as the original fit's (a flat 24.7 GB).
- On the graphics card a resumed fit is the same fit but not a digit-for-digit replay: the flow's spline uses an
  operation torch lists as non-deterministic on the card, and an uninterrupted fit on the card is not repeatable to
  the digit either. On the processor it is exact (measured). The card path was read, not run: the stage's card gate
  runs it (7.2).
- A fit that died of a non-finite loss resumes from its last saved pass; on the processor it will meet the same loss
  again (not a goal to cure here).

### 6.6 A save that fails
A save that fails (a full disk, a file held by another program) never stops the fit: stopping would lose the fit in
memory, which is what this design protects. The fit goes on; a warning goes to the run's log and its health line
("the last save failed: <reason>; the kept training stands at pass N"); the save is tried again at the next pass;
the kept fit on disk stays at its last good pass (its two generations untouched by the failed write).

## 7. Tests and gates

### 7.1 In the fast gate (CPU, seconds)
- **Resume equals uninterrupted**: a tiny fit (the cheapest real `train_nn` the suites have, about a second) run
  whole, and again killed after a pass and resumed in a fresh state; every loss, the counters and the final weights
  equal. It is also the alarm for an sbi upgrade that breaks the hook.
- A fit never interrupted writes a posterior identical to today's (the snapshot touches nothing the fit reads).
- Each item of 5.3 changed alone refuses the resume and names itself; an edit inside a named function refuses; an
  edit outside the named ones does not.
- A save torn at each of its steps leaves a readable kept fit (the older generation).
- The `fit` kind: written progressively, listed, marked unfinished after a stop, refused deletion of its simulation
  set, deleted after its posterior is written; the tool's flag and its refusal.

### 7.2 On the card
A gpu-marked test that resumes a tiny fit on the card (same split, counters and optimizer state on the device);
and in the card smoke gate a training interrupted and resumed (its lines and pass criteria are written by the
stage's plan; `CLAUDE.md`'s gate block changes only on the owner's yes). The slow set runs at the stage's exit (the
chi full-pipeline test trains through the hook).

## 8. The stage, and the window's part

- **A short stage before stage 3** (F9), planned like the others and approved by the owner before it is built:
  the engine change of 6.3 with the tests of 7, the tool's flag, the `fit` kind in the store (and, since the
  Records screen of today lists every kind, a kept fit shows there in today's form). It is an engine change named
  for the code audit. Its exit runs the fast gate of record, the slow set and the card gates.
- **The window**, as the stages that rebuild each page are planned: stage 3 (the run): the "Keep / Discard"
  question after a stop, the close question's words during a fit, "Before you start" saying the fit is saved after
  each pass and how long a resume reads back, the run history naming a kept fit; stage 4 (the Training step):
  the choice of 4.4 and the "Resume" of 4.2, Home's next step; stage 6 (Records): the kind's page (where it stopped,
  what differs, Delete).

## 9. What this changes in the approved redesign and elsewhere

To be amended when the stage that touches each lands (each change recorded as that stage's departure, citing this
design):
- the redesign: U10's "that a fit cannot be resumed"; U12's "Stop training…" (now followed by Keep / Discard);
  section 7's first bullet; finding W172 (closed by this design); the parts `06-run.md`, `08-steps.md`,
  `10-records-and-parts.md` and `12-findings.md` where they say a stopped fit is lost (the reader's list);
- `CLAUDE.md`: "A fit is not checkpointed: any stop during it loses the whole fit" (the owner's to approve, with the
  stage); the card gate's block if a resume run is added to it;
- `docs/guide/retrain.md` (lines 433-434, 550-552, 581) and the comment in `core/SBI/train.py` near line 361.

## 10. The controller's calls (for the owner to check)

1. The split kept as a random state plus a digest of the validation indices (5 KB), not the 164 MB permutation.
2. Two generations of snapshots; about 125 MB per kept fit.
3. The code fingerprint over named functions and classes plus resolved constants, not whole files and not the git
   revision (which moves with any commit, documents included).
4. The libraries in the identity, with a digest of sbi's installed trainer files.
5. A new progressive record kind `fit`, whose parents protect the simulation set and the prior from deletion.
6. The tool starts a new fit unless told to resume, and keeps a fit stopped by Ctrl-C.
7. The working word "Unfinished training" for the kind on screen, and the flag `--resume-fit`; the run stage and
   the tool's help may word them better.
8. A posterior's training block records whether it was resumed and the cumulative time.
9. The narrowing round treated the same (F7).
10. A resume runs on the kept fit's own record (4.2), so a kept fit becomes unresumable only through the code, the
    libraries or its simulation set, never through the window's fields.
11. The cheap checks at a click and the costly ones during the resume; a late failure marks the kept fit "cannot be
    resumed" and deletes nothing (5.3).
12. The states of 5.6: a lock in the fit's folder tells running from stopped; a Ctrl-C in the tool counts as a stop
    on purpose; an unanswered Keep or Discard keeps; until stage 3, today's window keeps a cancelled fit without
    asking.
13. A failed save warns and goes on, never stops the fit (6.6).
14. Settled with the builder's reader: the kept fit's record is created when the fit starts (after the simulations),
    so a crash before the first save leaves nothing to resume and nothing behind; the narrowing round's fit also
    names its observation and parent posterior as parents; a fit refuses to start, loudly, if sbi's hook points are
    missing (an upgrade).

## 11. Left to the builder

The names of the modules, classes and functions; the order of the stage's tasks; the exact test fixtures; the
manifest's key names within the `fit` kind; the log lines' words; how the tool prints the matching kept fits.

## 12. The owner's answers (2026-10-21)

F1 the 5 October stop was to start fresh; F2 protect against an accidental loss, and new fits will follow changes
of architecture; F3 after an accident, where it stopped and a Resume button; F4 a fit stopped on purpose: ask to
keep it, resumable while nothing has changed; F5 kept fits are records of their own, several may be kept, a new fit
leaves them, a non-matching one is never resumed and is deleted in Records; F6 only a finished fit gives a
posterior; F7 the playback (saved each pass, Home and the Training step, the narrowing round, the command line, the
same fit not to the last digit, deleted after its posterior); F8 the strict code check; F9 the training side as its
own short stage before stage 3; F10 PRISM hooks into sbi's own trainer.
