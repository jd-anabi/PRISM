# The retrain runbook

Checked against commit ce0d6d7f.

The next full-size training: a posterior of the frequency response on the tier-1 box, 10,000 batches
of 2,048 simulations, with a network 256 wide and 10 deep, followed by one narrowing round that
certifies it. It runs in the window, and its owner starts it. Every step below names the control by
the words the window shows: the part of the rail (Home, Parameter inference, Records, Settings), the
step (1 to 6), the field and the button. The few things the window cannot do are gathered in
[What stays on the command line](#what-stays-on-the-command-line), each with its reason. The
reasoning behind each setting is in [The science behind the settings](science.md), and every field
of the window has its help beside it.

## Decisions

Each decision, with its reason.

**A new simulation set, simulated from the start.** The simulator now integrates each cell at its
exact time scale: every recorded point is exactly one frame after the last in the cell's own time,
where each frame once took the nearest whole number of fixed steps
([Nondimensionalisation](science.md#nondimensionalisation),
[The record clock](architecture.md#the-record-clock)). A simulation set's identity names the clock it
was made with, so the finished set of the last full-size training, made at the earlier fixed step,
is not continued, and nothing of it is read again. It stays on disk; Records lists it with the other
simulation sets.

**The tier-1 box, `master_tier1.txt`.** The master box samples the gating-spring energy twice, and
nothing holds the two samples to agree, so the bath temperature they imply is spread over decades.
The tier-1 box infers the temperature T, in (280, 310) K, in place of f_scale, and derives the force
scale from it ([The tier-1 constraint and temperature](science.md#the-tier-1-constraint-and-temperature)).
Temperature is an assumed input: every report marks it "(assumed input)", and it stays in the
calibration verdict, judged like every other parameter.

**A fresh prior, built on the tier-1 box with the frequency response.** A prior's record names what
it was built from, its bounds file by path and hash and the kind of observation, so a prior built
for this run names this run's own box. A simulation set is keyed on the prior it was made with, so a
fresh prior cannot continue any set made before it by accident. And the audit of unusable driven
recordings is read on the prior in use, and the comparison after the simulation reads that audit:
both belong to the prior this training uses.

**10,000 batches of 2,048 simulations.** Every simulation of a batch shares the batch's
(t_scale, recording length) pair, so the batch count is the training set's diversity of operating
points and the width is replication at each point
([The solver's physics check](science.md#the-solvers-physics-check) sets count against width). The
August 2026 retrain ran 5,000 batches; this run doubles the operating points at the same width,
20.48 million simulations. The width is the card's own batch, which "Simulations per batch" set to
"Automatic" takes. It costs almost nothing in simulation time, because the solver's time is set by
its sequential steps, but the fit's time grows with the rows, so the width is paid for in fitting: a
width of 1,024 would about halve the fit.

**A network 256 wide and 10 deep, from the start.** The August 2026 retrain's reading that more
capacity would not help was made before the conditioning repair, by a network that could not see two
of its own inputs ([The August 2026 retrain](science.md#the-august-2026-retrain)); a larger network
may use what the repair restored. The network is not part of the simulation set's identity, so if
the larger network does not help, the complete set is fitted again at 128 by 8 without simulating
again.

**Six driven recordings supplied into twelve slots.** A simulated observation holds six driven
recordings ("Driven recordings per observation") in twelve slots ("Slots for driven recordings"),
and training draws its number for each batch from 2 to 12. The slot count stays: it is frozen into
every posterior of the frequency response and into the set's identity, so a change means another
full training, and whether a matched slot count would buy information has never been measured
([Open questions](science.md#open-questions)). Changed here, its effect could not be told apart from
the run's other changes.

**The certification round.** Before this posterior meets a bench recording, one narrowing round on
the simulated tier-1 cell, whose truth is known, at the training's own size. It checks end to end
that a round drawn from this posterior keeps the truth inside every direction it narrows and
calibrates on its region; on this code a round has run only at small size
([Narrowing rounds](science.md#narrowing-rounds)).

**A budget of four weeks of the card, 672 hours, for the training and its certification round**,
set by the owner on 30 September 2026. The round is a second full-size training, so the two cost
about twice the fit, plus the two generations, the calibrations and the calibrations at fixed
numbers of driven recordings. What is left after those, halved, is each fit's limit;
[The figures that depend on the simulator's step](#the-figures-that-depend-on-the-simulators-step)
gives it in hours and per epoch. The last full-size fit of this network on this card, in October
2026, took about 62 minutes an epoch on 20.48 million simulations, well inside it; the August 2026
fit stopped by itself after 130 epochs.

**The verdict rule.** The calibration decides by one fixed rule: it passes when every parameter's
rank test reaches KS p ≥ 0.05 ÷ 13 and the joint coverage test reaches KS p ≥ 0.05
([Reading calibration honestly](science.md#reading-calibration-honestly)). It is fixed before the
run, so the gate cannot be bent to the result, and the calibration is seeded, so it can be repeated
on the same card, though not bit for bit. The calibrations at a fixed number of driven recordings
characterise the posterior and decide nothing. They run because a pooled rank test can be flat while
each number of driven recordings is miscalibrated in compensating directions; they decide nothing
because each carries only the rank half of the verdict, and no rule was set for them before the run.

**The settings travel in a plan.** `core/config.py`'s defaults stay at 128 by 8 and 5,000 batches,
and they go on serving every other run: every setting reaches its stage as an argument
([Rules the code keeps](rules-and-traps.md#rules-the-code-keeps)), so this run needs no edit of the
code. The network's width and depth open at the defaults each time PRISM starts, so this run's
settings are saved once as a plan ([Run plans](recordings.md#run-plans)), committed with the inputs,
and filled into the forms from it. A Resume of a kept training needs no plan: it runs on the settings
the training was kept with.

**A new simulation set after a setting of the frequency response changes.** The drive, the band,
the lock-in limit and the slot count are part of the set's identity, so a change to one of them
starts a new set, and the store refuses a posterior trained at another value of any of them. Two,
the cycle floor `CHI_MIN_CYCLES` and the smallest number of driven recordings `CHI_K_MIN_TRAIN`, are
in neither ([The artifact store](architecture.md#the-artifact-store)): after a change to either, a
training at unchanged settings would continue the old set and its rows drawn under the old value.
The route is a fresh prior, which no old set matches
([The chi probe set and its Fisher](rules-and-traps.md#the-chi-probe-set-and-its-fisher) has the
general rule). Turning the saving of the simulations off would also stay clear of an old set, but a
10,000-batch run that cannot resume is not acceptable.

## The figures that depend on the simulator's step

These figures move whenever the simulator's step or its rule for taking a step changes, and they are
kept in this one place. At this commit:

- **Generation**, 10,000 batches of 2,048 at a memory limit of 6 GiB: about 29 card hours, about
  10 s a batch. An estimate, not yet measured on this code: the solver's steps counted over a fresh
  draw of 10,000 batches, a count that gives the last full-size set's measured 26.85 hours at the
  earlier fixed step. A narrowing round of 10,000 batches simulates at about the same pace.
- **A full-size calibration**, 2,000 test datasets at 200 operating points: under an hour (43
  minutes, measured at the earlier fixed step). Each repeat of a calibration at a fixed number of
  driven recordings costs about as much, so the four calibrations at fixed numbers (2, 6 and 12
  driven recordings, and mixed), each of ten repeats, take about 30 hours.
- **The audit of unusable driven recordings**, 100 batches of 32: about six minutes.
- **The set on disk**: about 10.3 GiB. The run's full text states the size and the free space when
  the set starts.
- **The last full-size set's readings**, at the earlier fixed step: 0.15 % failed simulations, and
  0.083 % of its summary elements clipped to their percentiles.
- **The rest of the programme**: about 95 hours, the two generations, the calibrations and a few
  hours for the inferences and the other checks. Each fit may then take about 290 hours,
  (672 − 95) ÷ 2: about 2.2 hours an epoch over 130 epochs, which the timing of a short fit reaches
  at about 16 s an epoch (2.2 hours ÷ 500).

## Pre-flight

In this order. Nothing here trains, and any item can stop the run.

1. **The card smoke gate**, every line passing on the code the run will use, from the command line
   ([What stays on the command line](#what-stays-on-the-command-line)). Its run 4 trains this box at
   this network size end to end, and its run 5 ends a training like a power cut and resumes it. If
   code under `core/diagnostics` has changed since the last card run,
   [The diagnostic card](testing.md#the-diagnostic-card) passes as well.

2. **This computer, in Settings.** Under "This computer", **Memory limit**: choose "Limit each batch
   to" and type 6 (GiB). With no limit the 1 October run died at batch 237, out of the card's
   memory; at 6 GiB 9,500 batches went through. The limit is remembered for this computer, and the
   Training step's "Before you start" names it: "Memory limit: 6 GiB for each batch of simulations (set in
   Settings, This computer)." `PRISM_VRAM_CEILING_GIB`, when it holds a number in the environment the
   window starts in, takes precedence, and Settings then says "Set outside PRISM by
   PRISM_VRAM_CEILING_GIB = …"; the desktop shortcut starts PRISM with Explorer's environment, so the
   variable reaches the window only through `run.bat` started from a shell that set it
   ([Launching PRISM](getting-started.md#launching-prism)). Leave "Keep this computer awake while a
   run is live" ticked: PRISM then asks Windows not to sleep until the run ends, and a shutdown or a
   sign-out shows PRISM's reason and waits for an answer.

3. **The forms, saved as a plan.** Fill the forms once, then save them:
   - Step 1, Model and options: Model "Nadrowski", Kind of observation "Frequency response". Leave
     "Driven recordings per observation" at 6, "Slots for driven recordings" at 12, "Lock-in limit" at
     64 and Rotation "On", their usual values. Press **Apply model and options**.
   - Step 2, Prior: Bounds file `master_tier1.txt`.
   - Step 3, Training: Batches 10,000; "Simulations per batch" "Automatic"; Network width 256;
     Network depth 10; "Continue or start new" at "Continue the saved simulation set if there is
     one". Under "Rarely changed", "Expected epochs (for the time estimate only)" 130, the August 2026
     fit's count: no finished training of this kind is in Records to take it from. Leave the rest at
     their usual values, "Save the simulations every … batches" at 50 above all.
   - Step 6, Narrowing round: Batches 10,000; Network width 256; Network depth 10; "Region level"
     0.999 and "Directions narrowed" 5, their usual values. The round takes nothing from step 3's
     form.
   - Back on step 1, under Model, **Save as a plan…**, a "Plan name" such as `retrain`, **Save**. The
     file is `Resources/Plans/nadrowski/retrain.txt`. A plan writes only the settings that differ from
     their usual values, so **Show what it holds** lists these and no others: Kind of observation; the
     bounds file; under Training, Batches, Network width, Network depth and "Expected epochs"; under
     Narrowing round, Batches, Network width and Network depth. Every setting left at its usual value
     above is not written, and filling from the plan sets it to that usual value.

   A later launch opens the plan list on "No plan" and the network at 128 by 8: choose the plan and
   press **Fill the forms from this plan** before any start that is not a Resume.

4. **The inputs committed, the revision noted.** Commit the plan, with the bounds file and the cell
   files the run reads (`master_tier1.txt`, `master_spont_tier1.txt`) if either has changed, and see
   that `git status --short` prints nothing: an untracked file counts as a change too. A folder git
   does not track still counts, an editor's settings folder say. Two ways clear it without touching a
   tracked file: list it in `.git/info/exclude`, the repository's own list of files to pass over, or
   move it out of the repository. A line in `.gitignore` would do it too, but that is a tracked file,
   so it is the owner's choice, and it is committed before the run. Every record stamps the
   checkout's revision and whether the tree has changes: a simulation set when it starts, a prior, a
   check or a posterior when it is written, the posterior at the very end of the run. Then note
   `git rev-parse HEAD` with the time. Both are command lines.

5. **The prior.** Step 2, Prior: **Build a new prior instead…**, the bounds file as above, a Name not
   taken (an earlier attempt's prior may hold `retrain_prior`; "Before you start" says "Fine. This
   name is free." of a free one), then **Build a new prior**. Its sweep has no retry when the card's
   memory runs out, so close what uses the card first (item 7). Its record's text says how much of the
   box was stable: on this box nearly all of it, and a warning that no clusters were found, defaulting
   to one, is expected here and does not stop the run ([The prior](science.md#the-prior)).

6. **The checks before the run**, in step 4, Checks, whose list groups them under "Before the run".
   Each runs on the prior in use; select the check, fill its form, press its button.
   - **Frequency band**, Cell file `master_spont_tier1.txt`, **Run the frequency-band check**. It
     must read "Passed": the band (0.03, 0.3) and the drive 0.15 both hold for this cell. The cell
     shares the master cell's non-dimensional values and time scale, and every driven recording is
     made at the same non-dimensional amplitude, so it should read as the master cell does
     ([Chi probe design](science.md#chi-probe-design)). Any other reading stops the run until it is
     understood.
   - **Unusable driven recordings: audit**, at its usual 100 batches of 32, **Run the audit**. It
     runs training's own generator over this prior. It reads
     "Passed" within 8 points of the expected 37 % (29 % to 45 %); "Look closer" from 8 to 12 points
     away, or when any driven recording gave no finite reading: the owner reads its table, the share
     by cause and the per-batch figures, and decides; "Failed" beyond 12 points, or when one of its
     consistency records does not hold: stop. Do not run it again with another seed to average: the
     seed changes neither the prior nor the layout of the driven recordings. Its share is the one the
     comparison after the simulation reads. Why these widths: the reference 37 % is itself uncertain
     by about 3 points, this reading by about 1.2, and a prior's build may move the share by up to
     about 2; combined, about 3.5 to 4 points, so ±8 is about two standard deviations and ±12 about
     three ([Chi probe design](science.md#chi-probe-design) says where each comes from).
   - **What the observations can pin down**, Cell file `master_spont_tier1.txt`, Recording length
     4.5 (s), **Run this check**. Its result says "Driven recordings at the low end of the band hold
     2 cycles from <s> s": that length must be under 4.5 s, the length the certification round
     simulates ([Recording length](recordings.md#recording-length)). Below it, the lowest driven
     recordings of the band fall under the cycle floor and are masked.
   - **Timing of a short fit**, at its usual 20 batches and 10 epochs, **Run this check**. It fits
     step 3's network, 256 by 10, then the built-in 128 by 8, each on 20 batches of the card's width,
     1/500 of the run, in a store of its own that it removes. Its result gives each network's seconds
     per epoch, and a projection: "About … of fitting, if the fit takes 130 epochs". The 128 by 8
     reading shows whether the card is as fast as it was in August 2026, about 5.1 s an epoch; the
     larger network read about 6.2 s in October 2026.
     - Above the timing's limit for 256 by 10, in
       [the figures that move with the simulator](#the-figures-that-depend-on-the-simulators-step),
       **stop for the owner's decision**. Nothing is simulated yet, so a stop costs only the timing.
       The owner's options: look for a cause (another program on the card, host memory short) and
       time again; a width of 1,024, which about halves the fit but halves the replication at each
       operating point, a science choice; or fit 128 by 8 first from the complete set and fit again
       at 256 by 10 later without simulating again.
     - Above about 7 s an epoch for 128 by 8, the card is slower than it was: tell the owner before
       the run, whatever the larger network read.
     - Whatever the reading, tell the owner the programme's projected total, twice the projected fit
       plus the rest of the programme in the figures that move with the simulator, before the run
       starts.

7. **The card and the computer to the run**, just before the start, and kept so while it runs.
   - Close every program that uses the card, browsers first: the longest batches use most of the
     card's memory even under the limit, which bounds the simulator's own buffers and not everything
     a batch holds. In Settings, "Free graphics-card memory" with **Read now** reads what is free. A
     batch that does not fit is split into smaller pieces, at several times the cost
     ([Memory planning and out-of-memory recovery](architecture.md#memory-planning-and-out-of-memory-recovery)).
   - While the run is live the card is its own: no pytest (the fast gate runs its card tests on the
     card), no card gate, no other PRISM command. PRISM keeps one window; a second start brings the
     open one to the front.
   - Host memory: close other programs. The last full-size fit's process held about 25 GB of the
     machine's 63 GiB through its fit.
   - Disk: the set's size in the figures that move with the simulator, with room to spare.
   - **No edit under `core/`, of the bounds file or of the units file, and no commit while the run
     is live.** The posterior stamps the checkout's revision and whether the tree has changes when it
     is written at the end, so a commit of any file during the run, documents included, moves that
     stamp. Whether an import late in a run reads an edited file is not established. The same holds
     for as long as a training is kept, live or stopped: an edit inside the code that builds or trains
     the network, a comment included, makes every training kept before it unresumable, and so does a
     change to the bounds file or the units file it began with; and do not delete the run's prior from
     Records, since a kept training whose prior is gone cannot be resumed ([Resuming](#resuming)).

## The run

Step 3, Training, with the plan's settings in the form and the new prior in use. Read
"Before you start" first. It should say, in this order:

- "Model Nadrowski; bounds file master_tier1.txt; prior <the new prior>; frequency response with 12
  slots; rotation on."
- "10,000 batches of 2,048 make 20,480,000 simulations. Each simulates one spontaneous recording and
  2 to 12 driven recordings."
- That it starts a new simulation set, naming each saved set of the model it does not continue with
  what it differs in: the finished set of the last full-size training among them, made on another
  prior, at a lock-in limit of 20 and by another version of PRISM.
- "The fit is saved after every epoch, so a stop or a failure keeps it." and "A resume first reads
  the simulation set back, about four minutes at full size."
- How long it should take, when the run history or the timing of a short fit can say; the memory
  limit in force; and what the frequency band and the audit read for this prior, "Fine." when both
  passed.

Type the Name (`retrain`, say) and a one-line Note, then press **Start training**. Two questions may
come first:

- "Start training although a check failed?" only when the frequency band or the audit failed: cancel,
  and go back to the pre-flight.
- "Start a new simulation set?", listing each saved set with what it differs in. Answer
  **Start training on a new simulation set**; "Cancel" is preselected.

The run's page then stands in the step's place, and the foot of the rail says where it is
("Simulating, batch 6,150 of 10,000", "Fitting, epoch 62 of about 130"); Ctrl+R goes back to it from
anywhere. The page shows the run's four stages, Rotation, Simulation, Fitting and Saving; one sentence
that says where it stands; a time axis with when the stage should end; the training's health; the
fit's loss by epoch; and the messages, newest first. **Show the full text** shows every line the run
wrote, and **Open the file** opens its dated text file, kept in the `runs` folder beside the settings
file (`%APPDATA%\PRISM`). Closing the window asks first, "Keep running" preselected.

## What to watch

Stage by stage, as the run's page says it. The engine's own lines are in the full text, and the
records' lines are also in the record's `log.txt`.

- **Rotation.** "The rotation is being computed before any simulation." The full text says over how
  many operating points the rotation was averaged (8 of 8 expected) and its eigenvalue spread; a
  warning that an operating point was skipped means one fewer.
- **Simulation.** "Simulating: batch … of 10,000, … s per batch", at about the pace that
  [the figures that move with the simulator](#the-figures-that-depend-on-the-simulators-step) give.
  The full text's first line of the set states its size and the free disk space; a warning after it
  means the disk is short. The set is saved every 50 batches.
- **Training health**, its rows in the window's order:
  - "Failed simulations": "Fine" up to 1 %; the figures that move with the simulator give the last
    full-size set's share.
  - "Unusable driven recordings", shown for a frequency response, as this run is: "Counted when
    simulation ends", then the comparison with the audit ([Gates](#gates)).
  - "Graphics-card memory": the largest batch's peak so far. "Batch … did not fit in the graphics
    card's memory and was run in smaller pieces." now and then is survivable; a steady stream means
    the card is fuller than the splitting absorbs: look for another program on it.
  - "Training saved": "after epoch …", once the fit has begun.
- **Two lines a continued simulation can bring.** When an earlier session's save was cut off before
  it committed, the full text says it "moved … uncommitted shard files past batch … aside to …,
  which no load reads": nothing is asked of you, and nothing was deleted. When the set's newest
  commit file could not be read, a message from the engine says that "state.pt could not be read, so
  this run resumes from state.prev.pt, a generation back": act on it before the first batch ends, as
  [Resuming](#resuming) says.
- **When simulation ends**, the comparison is made at once: within 3 points of the audit is fine, and
  the fit goes on ([Gates](#gates) says what the other two bands ask). The full text then says how
  many summary elements were clipped to their percentiles: a few tenths of a percent at most.
- **Fitting.** The stage says "Now, about … per epoch"; the loss by epoch grows on the page; "Training
  saved" follows each epoch. A silence during an epoch is said as normal; one twice as long as the
  last epoch took is a warning. A full-size epoch takes about 500 times the timing's seconds per
  epoch. Slower than the limit per epoch in the figures that move with the simulator, stop the run
  for the owner; so does a fit still running past its limit in hours after simulation ended. Either
  way the set is complete and the fit is kept after every epoch, so a stop loses at most the epoch
  in progress, never a simulated batch ([Resuming](#resuming)).
- **Saving.** "Saving the posterior" cannot be stopped. The page then says the posterior is saved,
  and how the fit ended: by itself, once validation had not improved for 20 epochs. The record holds
  the loss figure, `figures/training_loss.png`, and its numbers, `loss.npz`.

## Resuming

A run that stops keeps what it saved:

- **During the simulation**, every batch its set saved, the last save at most 50 batches back. A
  stop from the page saves the batches finished since, too.
- **During the fit**, every batch, and the fit at its last saved epoch, kept as a training of its
  own: a record of the kind `fit`, which Records calls an unfinished training. The fit is
  saved after every epoch, so a stop, a failure, a power cut or a closed window loses at most the
  epoch in progress. A training whose simulations are not saved keeps no fit at all.

**Stopping on purpose.** **Stop training…** asks first. During the simulation it says that the
batches simulated so far are kept and the batch in progress is not: **Stop and keep the
simulations**. During the fit, that the fit stops when the epoch in progress ends and every finished
epoch is kept: **Stop training**. "Keep training" is preselected each time, and until the stop is
taken **Don't stop after all** withdraws it. Once a stop in the fit has landed, the page asks "Keep
this training so you can resume it later?": **Keep**. **Discard** deletes the kept training, not the
simulations. After a stop during the simulation the page says "No training was kept: the fit had not
begun.", and the simulations are kept all the same.

**Resuming.** A kept training is resumed by **Resume** in three places: on the run's page ("Resume
the training") until PRISM is closed; after a reopening, on step 3, Training, under "Kept trainings",
whose row says how and when it stopped, its best validation loss and what it trains on; and on Home,
whose next step offers it. A Resume runs on the training's own record ("It uses the settings it was
kept with"), whatever the forms hold, and puts its prior in use. It first reads the simulation set
back, about four minutes at full size, then continues from the epoch after the last one saved; the
full text says from which pass it resumes, and "Training saved" moves on once the next epoch is saved.

**Continuing a simulation.** A training stopped or failed during its simulation keeps its set, and
step 3 says "Simulations saved to batch …". Its run's page offers **Continue from batch …** until
PRISM is closed, and Home offers it for one that failed or was not seen to end; either fills the
forms with the settings the run was started with, leaving the name and the note blank. Or fill the
forms from the plan and load the prior the set was made with (step 2, **Use a saved prior
instead…**, **Load this prior**): "Before you start" then says "Continues the saved simulation set:
… of 10,000 batches are done.", and **Start training** continues it. A continued set reuses the
rotation stored with it, and step 3 says of the rotation's three settings "Not used: this training
reuses the saved simulation set's rotation."

A training that failed or was not seen to end with its set whole and no kept training on that set
(its fit had not saved an epoch) is offered **Fit again on the saved simulations**, on Home and, after
a reopening, on its run's page; it fills the forms in the same way, and fits again on the saved set
without simulating. On step 3 the main button reads so whenever the set of these settings is whole and
"Continue or start new" is "Only continue: do not start if there is nothing to continue".

**What makes a kept training unresumable.** A kept training resumes only while what it was trained
with is unchanged. Its row on step 3 says why it cannot, in place of **Resume**:

- a process is training it now;
- the code that builds or trains the network has changed since it was kept, an edit of a comment
  included (the functions named in `core.SBI.fit_state.FIT_CODE`): finish a kept training you want
  before such an edit;
- a configuration constant that code reads has changed;
- the libraries have changed, sbi's own trainer files among them;
- its simulation set is gone from the store, or no longer holds the batches it was trained on;
- its prior is no longer in Records.

A press of **Resume** checks more before it loads anything, and says on the row what stops it:

- the bounds file it began with is gone, or has changed since the training began;
- the units file it began with is gone, or has changed;
- its model is no longer one PRISM can train;
- it was trained on the graphics card, and this computer has none now.

Its settings and its network's shape are checked as well: the window resumes it on its own settings,
so these cannot differ there, and a resume whose rows, read back, do not give the kept fit is refused
and the training marked as one that cannot be resumed: stop for the owner. A kept training that cannot
be resumed is deleted with **Delete** on its row. Once the posterior is written, the kept training is
removed.

**Keep the prior.** A simulation set is keyed on its prior, and a prior built again is another prior:
continue a set with the prior it was made with. A Resume puts the right one in use itself.

**When the set's newest commit could not be read**, the continued run says so in a message from the
engine, naming the shard files it leaves in place. Stop the run while it is still on its first
batch, where a stop commits nothing, and continue again: if `state.pt` reads then, nothing is lost.
Only if it is really damaged, copy `state.prev.pt` over it in the set's folder before continuing;
that run then sets those files aside. Left to run, it simulates those batches again, and a later load
may refuse the leftovers as overlapping.

**A set that a continuation refuses as holding two shards that overlap** names both: move the one
that does not fit the committed run, with its `th_` twin, into the set's `uncommitted` folder, as the
message says, and continue again.

## Gates

On step 4, Checks: "A posterior passes when all six deciding checks have passed." The list shows each
check's state. Two passed before the run; one is read when the simulation ends; three run once the
posterior is in use, each from its page, with the posterior and its own prior in use.

1. **Frequency band**: passed in the pre-flight.
2. **Unusable driven recordings: audit**: passed in the pre-flight.
3. **Unusable driven recordings: simulation set against the audit.** Made by itself when simulation
   ends, on the run's page and in this list. With D the difference between the set's share and the
   audit's, in points:
   - D at most 3: "Passed"; the fit goes on.
   - D above 3 and at most 5: "Look closer before leaving the fit to run". The comparison's page here
     shows the per-batch figures of the set and of the audit side by side; once they are understood,
     **Accept this difference**, which the run's history keeps with its moment.
   - D above 5: the advice is to stop the fit and find out why, and the run's page asks for an
     answer: **Keep training**, or stop with **Stop training…**. The window never stops the fit
     itself. The set is complete and the fit is kept, so a stop loses at most the epoch in progress.

   D spreads as the audit's reading does, about 1.2 to 1.6 points: the set's own share, over 10,000
   batches, is far steadier, and neither the reference's uncertainty nor the prior's build enters,
   since both readings are of the same prior. So 3 points is at least about two standard deviations
   and 5 at least about three. A calibration's own count of unusable driven recordings, scoped to its
   own run, is not this gate.
4. **Calibration**, at its usual 2,000 test datasets, 200 operating points and 1,000 samples per
   dataset: **Run the calibration**. A calibration made smaller never decides. It passes by the
   verdict rule ([Decisions](#decisions)); temperature's row is marked "(assumed input)" and counts
   like the rest. Its seed is drawn and recorded; to repeat a calibration set, type that seed in
   "Seed", under "Rarely changed".
5. **Use of each part of the input**, at its usual 200,000 simulations read and 33 points a sweep:
   **Run the input-use check**. It reads the posterior's own simulation set, one reason the run keeps
   it. It passes when the two inputs the August 2026 network could not see, `A1_mean` and
   `D3_bimodality` ([Conditioning features](science.md#conditioning-features)), respond as the
   typical input does; any other input invisible to the network makes it "Look closer".
6. **Identifiability of the posterior**: **Run the identifiability check**. It simulates nothing. It
   passes when the record holds how strongly each direction is pinned down: the rotation's
   eigenvalues, stored with the set and carried by the posterior.

Two readings are written down, not passed:

- **The loss curve**, on the run's page and in the posterior's record. A clean plateau well before
  the best epoch, the validation loss no longer descending as it nears its best, reads as a limit in
  the data; a validation loss still falling near the best epoch reads as under-fitting
  ([Identifiability limits](science.md#identifiability-limits)), answered by fitting again on the
  complete set, with no new simulation.
- **Informativeness**, total and per parameter, as the first baseline: the reading under the
  calibration's table. Read its sign first: a negative total means the network gives the truth less
  density than the prior does. Temperature's line is marked as an assumed input, and the total is
  not adjusted for it ([Reading calibration honestly](science.md#reading-calibration-honestly)). No
  earlier posterior of this code exists to compare with, so this is the number later runs are
  compared against, always on fresh calibration sets. Under one prior, a calibration set repeats with
  the same seed, test datasets and operating points; a narrowed posterior draws its set from its
  region, so its set is never its parent's.

And one characterisation: **Calibration repeated at fixed numbers of driven recordings**, under "After
the fit", once at each of 2, 6 and 12 in "Fixed number of driven recordings" and once at "Mixed, as
in training", each at 10 repeats of 2,000 test datasets ([Decisions](#decisions) says why it decides
nothing).

## The certification round

A narrowing round from the passed posterior, drawn around the simulated tier-1 cell at 4.5 s.

1. **The observation.** Step 5, Inference: "Run on" "A simulated cell", Cell file
   `master_spont_tier1.txt`, Recording length 4.5 (s), a "Name of the inference" and a "Name of the
   observation", then **Run inference**. The inference records the parent's intervals, beside each
   parameter's true value, and its observation is the one the round is drawn around.
2. **The round.** Step 6, Narrowing round, with the plan filled: Observation the one just made (an
   observation an inference makes is chosen in the list once its run has ended), Region level 0.999,
   Directions narrowed 5, Batches 10,000, Network width 256, Network depth 10, a Name. Read its
   "Before you start", then press **Start the narrowing round**. The round is a second
   full-size training: it inherits neither its parent's network nor its batch count, which is why
   both are in its form. Its region is part of its set's identity, so it simulates a set of its own,
   and it costs about what the training does; watch it the same way, against the same limits. It
   reuses its parent's rotation, the one its region was measured in, and computes none.
3. **Its calibration.** The round's posterior comes into use on step 3 when its run ends. Step 4,
   Checks: **Run the calibration**, at the usual sizes. It draws its test datasets inside the region
   the posterior was narrowed to, and its record keeps the region's share of the prior,
   `results.kept_fraction`.
4. Optional, **a child inference**: step 5 again on the same cell and length with the narrowed
   posterior in use. Simulating the cell again draws new noise, so the observation is never the
   region's own, and the press asks "Run on another observation?": **Run inference anyway**. Its
   intervals are a record only: drawn on independent noise, they cannot be compared with the parent's
   interval by interval.

The round passes when both hold:

- the truth lies inside every narrowed direction: the round's full text says for each "truth
  direction …: … inside [lo, hi]", a truth outside is a warning among its messages, and the
  posterior's record keeps the answers as `training.truth_containment`. A direction loaded on t_scale
  is left at full width, and the full text says that it was not narrowed;
- its calibration passes, made inside its region.

A round that stops is resumed like the training: its run's page offers to resume or continue it
until PRISM is closed, and after a reopening step 6 lists it under "Kept narrowing rounds", with
**Resume** and **Delete**, Home's **Open Narrowing round** leading there. A continued round draws its
region again from a seed taken from the observation and the round's settings, so that it finds its
own set; before leaving a continued round, see that its Simulation stage started from the batch it
stopped at, not from the first. Its first start may ask "Start the narrowing round beside a saved
simulation set?": answer **Start the narrowing round**.

That a round narrows the posterior no more than the data support is not a reading of this run. The
round's proposal guarantees it: the prior restricted to the region, never the posterior, which a test
in the suite pins on a case whose answer is known ([Narrowing rounds](science.md#narrowing-rounds)).
The run checks it through the two criteria above, and the round's informativeness is inflated by
−log P(A), as that section says.

## Afterwards

In Records, select the posterior: type its result in the Note box (the verdict and the gates in one
line) and press **Set**; then **Lineage report…** saves the posterior and every record it was made
from, back through the chain, to a file you choose.

Which record answers each gate. Records shows each record's manifest; the keys are under its `body`.

- The calibration verdict: the calibration's `results.verdict` (`passed`, a row per parameter, the
  coverage row) and `results.seed`.
- The inputs: the input-use diagnostic's `results.channels`, the `A1_mean` and `D3_bimodality`
  entries and their `verdict`.
- The comparison: the posterior's `log.txt`, its total of masked driven recordings over every batch
  of the set; the audit's `results.masked_fraction`, a fraction, not a percentage.
- The eigenvalues: the posterior's `transform.fisher_eigenvalues`; the identifiability diagnostic's
  `results.eigenvalues`.
- The loss curve: the posterior's `figures/training_loss.png`, `loss.npz`, `training.epochs_trained`
  and `training.best_validation_loss`.
- Informativeness: the calibration's `results.informativeness` (`total_nats`, `sem_nats`,
  `per_param`, `per_direction`).
- The calibrations at fixed numbers: each diagnostic's `results.rank_verdict`.
- The round's truth: the round's posterior's `training.truth_containment`; its calibration's
  `results.verdict`, and `results.kept_fraction`, set for a calibration inside a region.
- The round's widths, a record: each inference's `results.posterior_summary` (`q05`, `median`, `q95`)
  and `results.ground_truth`.
- The cost: the simulation set's `wall_seconds`, from its start to its completion across sessions;
  the posterior's `training.fit_wall_seconds`, the fit across its sessions, and
  `training.wall_seconds`, the last session alone.

Every line a run wrote is in its dated text file in the `runs` folder, the sessions of a resumed run
each in its own.

## What stays on the command line

Only what the window cannot do, one line each, from the repository root in PowerShell.

- **The card smoke gate**: `docs/guide/testing.md`'s five command lines and run 5's drill, each
  checked with `$LASTEXITCODE` ([The card smoke gate](testing.md#the-card-smoke-gate)). The window's
  Card test, under "Tools", runs one small pipeline on the card; the gate also proves a resume, a
  refusal one setting away, the forced mode and a training ended like a power cut, in throwaway
  stores, which no check of the window does.
- **The diagnostic card**, when code under `core/diagnostics` has changed
  ([The diagnostic card](testing.md#the-diagnostic-card)), for the same reason.
- **The revision**: `git rev-parse HEAD`, and `git status --short`, which must print nothing (item 4
  of the [Pre-flight](#pre-flight) says how to clear an untracked folder). The window reads no git; it
  only stamps the revision into each record.
- **The commit of the inputs**: `git add Resources/Plans/nadrowski/retrain.txt`, then `git commit`.
  The window saves a plan but commits nothing.
- **The card's memory while a run is live**: `nvidia-smi --query-gpu=memory.used --format=csv`.
  Settings reads the free memory only while no run is live, and the run's health row gives the run's
  own peak, not what other programs hold.

The two file moves a damaged simulation set can ask for, copying `state.prev.pt` over `state.pt` and
moving an overlapping shard aside, are made in Explorer, as the engine's message says.

## Ready, and not ready

The code at this commit is ready for this run from the window: the plan holds every setting it needs,
the twelve checks hold the pre-flight's readings and the six gates, the run's page watches the
simulation, the comparison and the fit, the fit is saved after every epoch, and a kept training or
round resumes from the window. The simulator integrates each cell at its exact time scale, so the
finished simulation set of the last full-size training no longer matches, and this run simulates its
set again from the start. A long run is started by its owner, from the window.

It is not ready for an inference on bench recordings: PRISM reads every recording as sampled exactly
once a millisecond, and a bench recording is to be read at its camera's own measured frame interval
before any inference on it
([Recording files and drive frequencies](recordings.md#recording-files-and-drive-frequencies)). And
the narrowing round has run on this code only at small size: the certification round is its first
full-size run.
