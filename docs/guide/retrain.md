# The retrain runbook

Checked against commit 95745d2.

The next full-size training run: a chi posterior on the tier-1 box, 10,000 batches of 2,048 rows,
with a flow of 256 hidden features × 10 transforms, followed by one narrowing round that certifies
it. It runs on the command line, from the repository root, one command at a time, and every command
is written here in full. What each flag means is in [The command-line tool](command-line.md), and the
reasoning behind each setting in [The science behind the settings](science.md).

The commands use these PowerShell variables, set once in the session that runs them. Replace
`<scratch>` with a folder outside the repository: the pre-flight writes its throwaway records there,
and deletes it at the end.

```powershell
$py   = "C:\Users\J\anaconda3\envs\biophys-env\python.exe"
$Box  = "--bounds","Resources/Bounds/nadrowski/master_tier1.txt"
$Cell = "--cell","Resources/Cells/nadrowski/master_spont_tier1.txt"
$M    = "--bounds","Resources/Bounds/nadrowski/master.txt"
$MC   = "--cell","Resources/Cells/nadrowski/master_spont.txt"
$S    = "<scratch>"
```

## Decisions

Each decision, with its reason.

**The tier-1 box, `master_tier1.txt`.** The master box samples the gating-spring energy twice, and
nothing holds the two samples to agree, so the bath temperature they imply is spread over decades.
The tier-1 box infers the temperature T, in (280, 310) K, in place of f_scale, and derives the force
scale from it ([The tier-1 constraint and temperature](science.md#the-tier-1-constraint-and-temperature)).
Temperature is an assumed input: every report marks it as one, and it stays in the calibration
verdict, judged like every other parameter.

**A fresh prior, `retrain_prior`, built on the tier-1 box with `--chi`.** The stability screen
reads the non-dimensional block alone (`core.orchestrator.build_prior`), and the two boxes share it,
so the screen is the same as a `master.txt` prior's. A `master.txt` prior would even load:
`core.artifacts.store.ArtifactStore.load_prior` compares the model, the non-dimensional parameters,
their order and box, and the log-box mask, and neither the mode nor the rescale block. But a prior's
record names what it was built from: its bounds file by path and hash, and a configuration block
with the mode, all thirteen parameter names and the tier-1 constraint
(`core.artifacts.manifest.config_from_cfg`). Built on the tier-1 box in chi mode, the record names
the retrain's own box and mode. A new prior is also a new fit, and a simulation cache is keyed on its
prior's fit, so no cache made before it can be picked up by accident.

**10,000 batches × 2,048 rows.** Every row of a batch shares the batch's (t_scale, T_obs) pair, so
the batch count is the training set's operating-point diversity and the width is replication at each
point ([The solver's physics check](science.md#the-solvers-physics-check) sets count against width).
The August 2026 retrain ran 5,000 batches; this run doubles the operating points at the same width,
20.48 million rows. The width is the card's hardware batch, which `--run-size 0` takes, and it costs
almost nothing in time, because the solver's time is set by its sequential steps.

**256 hidden features × 10 transforms, from the start.** The August 2026 retrain's reading that more
capacity would not help was made before the conditioning repair, by a flow that could not see two of
its own channels ([The August 2026 retrain](science.md#the-august-2026-retrain)); a larger flow may
use what the repair restored. The network's settings are not part of the simulation cache's identity,
so if the larger flow does not help, the complete cache refits at 128 × 8 without simulating again.

**Six probes supplied into twelve slots.** A simulated observation supplies `CHI_N_FREQS` = 6 probes
into `CHI_K_PAD` = 12 slots, and training draws its probe count for each batch from 2 to 12. The slot
count stays: it is frozen into every chi record and into the cache's identity, so a change means
another full retrain, and whether a matched slot count would buy information has never been measured
([Open questions](science.md#open-questions)). Changing it here would change two things in one run.

**The certification round.** Before this posterior meets a bench recording, one narrowing round on
the simulated tier-1 cell, whose truth is known, at the retrain's own size. It checks end to end that
a round drawn from this posterior keeps the truth inside every direction it truncates and calibrates
on its region; on today's code a round has run only at smoke size
([Narrowing rounds](science.md#narrowing-rounds)).

**The verdict rule.** The calibration gate is one fixed rule, `validate`'s verdict: PASS when every
parameter's rank test reaches KS p ≥ 0.05 ÷ 13 and the joint coverage test reaches KS p ≥ 0.05
([Reading calibration honestly](science.md#reading-calibration-honestly)). It is fixed before the run,
so the gate cannot be bent to the result, and the calibration is seeded, so a reviewer can repeat it.
The stratified `sbc` runs characterise the posterior and gate nothing. They run because a pooled rank
test can be flat while each probe count is miscalibrated in compensating directions; they gate
nothing because each carries only the rank half of the verdict, and no rule was set for them before
the run.

**`core/config.py`'s defaults stay** at 128 × 8 and 5,000 batches. Every setting reaches its stage as
an argument ([Rules the code keeps](rules-and-traps.md#rules-the-code-keeps)), so the retrain needs no
edit, and the defaults go on serving every other run. That is also why the retrain runs on the command
line: in the window, Hidden features and Transforms open at `core/config.py`'s values on every launch,
so the retrain, and every resume of it, would need them typed into the Posterior tab each time; the
tab remembers Batches ([What the window remembers](window.md#what-the-window-remembers)).

**A new simulation cache after a chi constant changes.** If any chi constant in `core/config.py` has
changed since a cache was written, train against a new cache. The drive, the band, the lock-in
ceiling and the slot count are part of the cache's identity, so a change to one of them re-keys the
cache (a committed cache exactly one setting away is refused until `--new-run` is given), and the
store refuses a posterior trained at another value of any of them. The cycle floor,
`CHI_MIN_CYCLES`, is neither in the identity nor recorded with a posterior, so a plain `train` finds
the old cache and reuses rows masked at the old floor;
[The chi probe set and its Fisher](rules-and-traps.md#the-chi-probe-set-and-its-fisher) has the
trap, and [The artifact store](architecture.md#the-artifact-store) the other keys a record carries
that no loader compares. No flag forces a new cache under an unchanged identity: `--resume never` refuses
when this run's own cache holds batches, and `artifacts rm` refuses a cache while any record names it.
Take one of two routes:

- delete the records that depend on the cache from the leaves up (the calibrations, inferences,
  diagnostics and narrowing rounds built on each posterior trained from it, then those posteriors),
  then the cache itself with `& $py -m core artifacts rm simulation <digest>`; each refusal names the
  next record in the way, and the prior may stay;
- or train in a fresh records root: point `PRISM_ARTIFACTS` at a new folder, and build the prior again
  there.

Training with `--checkpoint-every 0` also stays clear of the old cache, but a 10,000-batch run that
cannot resume is not acceptable. For this retrain the question arises only if a cache from an earlier
attempt of it exists in the real store, since a freshly built `retrain_prior` matches no older cache;
`& $py -m core artifacts list simulation` lists what is there.

## Pre-flight

In order. Nothing here trains the retrain, and any step can stop it.

1. **The card smoke gate, run 4 included.** All five command lines of
   [The card smoke gate](testing.md#the-card-smoke-gate) pass, on the code the retrain will run. Run 4
   matters most here: it runs this box at this network size, end to end. The retrain's checks below
   lean on the diagnostics too, so if code under `core/diagnostics` has changed since the last card
   run, [The diagnostic card](testing.md#the-diagnostic-card) passes as well.

2. **The epoch timing**, in a scratch store: how long one epoch of the 256 × 10 flow takes on this
   card, before anything costs weeks.

   ```powershell
   $env:PRISM_ARTIFACTS = "$S/timing"
   & $py -m core prior @Box --chi --device cuda --name timing_prior
   & $py -m core train @Box --chi --device cuda --prior timing_prior --num-runs 20 --hidden-features 256 --num-transforms 10 --max-epochs 3 --checkpoint-every 0 --name timing
   ```

   Twenty batches of 2,048 rows are 40,960 rows, 1/500 of the retrain's. The fit's span is read from
   the posterior's `log.txt`:

   - It starts when generation ends. Generation's last record, from
     `core.SBI.pipeline.gen_training_data`, is the `[chi] masked probes` run total; then
     `core.SBI.train.train_nn` logs `[winsor] clipped …`, builds the flow and fits it.
   - No record is logged after the fit. `core.orchestrator.build_posterior` goes straight on to write
     the record, and the record's `log.txt` is written as it commits, so the fit ends a few seconds
     before `log.txt`'s last-write time, with the saving of the posterior and its loss figure between
     the two.
   - sbi counts epochs from zero: at `--max-epochs 3`, sbi 0.25 trains four. Divide by the count the
     record keeps, `training.epochs_trained`, never by `--max-epochs`.

   ```powershell
   $dir = (Get-ChildItem "$S/timing/posteriors" -Directory -Filter "timing__*").FullName
   Select-String -Path "$dir/log.txt" -Pattern "masked probes"
   (Get-Item "$dir/log.txt").LastWriteTime.ToString("HH:mm:ss")
   (Get-Content "$dir/manifest.json" -Raw | ConvertFrom-Json).body.training.epochs_trained
   ```

   Seconds per epoch is the span between the two times, divided by the epoch count. The span also
   holds the flow's set-up and the record's writing, so it slightly overstates an epoch, which errs
   toward stopping.

   - **The projection**, in hours, is seconds per epoch × 500 × 130 ÷ 3,600: about 18 hours for
     every second an epoch takes. 500 is 20,480,000 ÷ 40,960, the retrain's rows over the timing
     run's, and 130 is the epoch count the last full run trained to at a patience of 20 epochs.
   - **The reference.** The August 2026 retrain fitted its 128 × 8 flow on 10.24 million rows in 46.1
     hours, so the same flow on 20.48 million rows would take about 92 hours. At 256 × 10 the fit is
     expected to cost about four times that, about 370 hours: sbi's spline flow at 256 × 10 has about
     3.9 times the trainable parameters of the 128 × 8 one (3.7 million against 0.96 million, for
     thirteen parameters), and a training step's work grows with them. Generation comes on top:
     11.4 hours for the August retrain's 5,000 batches, and about 31 for the 10,000-batch run of
     29 August 2026.
   - **The stop point.** If the projection is more than twice the expected cost, about 740 hours,
     **stop for the owner's decision**.

3. **The recording-length check**, still in the scratch store.

   ```powershell
   & $py -m core identifiability jacobian --chi @M @MC --t-obs 4.5 --m 32 --m-noise 128 --seed 0 --device cuda
   & $py -m core identifiability jacobian --chi @Box @Cell --t-obs 4.5 --m 32 --m-noise 128 --seed 0 --device cuda
   ```

   Read the line that begins `low edge` in each:

   ```
     low edge 0.03x clears 2 cycles at T_obs >= <s> s; high edge 0.3x stays under the 20-cycle ceiling below T_obs = <s> s.
   ```

   The first length is T*: below it, the band's lowest probes fall under the cycle floor and are
   masked. It must be under 4.5 s, the length the certification round simulates
   ([Recording length](recordings.md#recording-length)). The two cells share their non-dimensional
   values and time scale, so both maps should give the same two lengths. The tier-1 map also has a
   temperature column, recorded as a measurement
   ([The tier-1 constraint and temperature](science.md#the-tier-1-constraint-and-temperature)).

4. **The retrain prior, then the probe checks**, with `PRISM_ARTIFACTS` unset, so that the prior is
   written into the real store:

   ```powershell
   Remove-Item Env:PRISM_ARTIFACTS -ErrorAction SilentlyContinue
   & $py -m core prior @Box --chi --device cuda --name retrain_prior
   & $py -m core probes band @Box @Cell --device cuda --name retrain_band
   & $py -m core probes mask @Box --prior retrain_prior --device cuda --name retrain_mask
   ```

   - `probes band` must report that the configured band (0.03, 0.3) and the configured drive 0.15
     both hold for the tier-1 cell. The cell shares the master cell's non-dimensional values and time
     scale, and every probe is driven at the same non-dimensional amplitude, so the verdicts should
     read as they do on `master_spont` ([Chi probe design](science.md#chi-probe-design)); one that
     does not hold stops the retrain until it is understood.
   - `probes mask` must find the masked probes within ±12 points of 37 %, with the cycle floor the
     only cause, over the retrain's own prior ([probes mask](command-line.md#probes-mask)).

   Then delete the scratch folder: `Remove-Item -Recurse -Force $S`.

5. **Resources.**

   - **Card memory.** Read it with `nvidia-smi --query-gpu=memory.used --format=csv`, never with
     `torch.cuda.mem_get_info()`, which on Windows overstates free memory by the desktop's share
     ([Memory and devices](rules-and-traps.md#memory-and-devices)). The margin is about 1 GiB: at
     2,048 rows the worst batch geometry needs about 11.44 GiB against a budget of about 12.48 GiB on
     an idle card. Close every desktop application that uses the card, browsers first, before the
     prior build and before the run. A run survives a busy card by splitting its batches, at several
     times the cost
     ([Memory planning and out-of-memory recovery](architecture.md#memory-planning-and-out-of-memory-recovery)).
   - **Host memory.** Training is expected to hold about 50 of the machine's 63 GiB as it starts,
     because sbi copies the training data several times; close other applications.
   - **Disk.** The cache is about 10.3 GiB. The `[checkpoint] writing to …` line states the size and
     the free space, and a warning follows it when the space is short.
   - **The prior's sweep has no out-of-memory retry**, so the card must be clear before `prior` runs
     too.

6. **Capture the console.** A resumed run's first process prints lines that exist nowhere else, and
   sbi's prints and the tool's own lines never reach a `log.txt`. This keeps the live display and
   appends both output streams to one file; it was checked on this machine's Windows PowerShell 5.1
   with the two commands below, which import no torch:

   ```powershell
   $ErrorActionPreference = "Continue"
   $env:PYTHONUNBUFFERED = "1"
   filter Plain { if ($_ -is [System.Management.Automation.ErrorRecord]) { $_.Exception.Message } else { $_ } }
   $Log = "$HOME\prism-retrain-console.log"
   & $py -m core --help 2>&1 | Plain | Tee-Object -FilePath $Log -Append
   $LASTEXITCODE
   & $py -m core validate 2>&1 | Plain | Tee-Object -FilePath $Log -Append
   $LASTEXITCODE
   ```

   The first prints the help and 0, the second a usage error and 2, and `$Log` then holds both: the
   help from the output stream and the usage error from the error stream.

   - `2>&1` joins the error stream, where warnings, refusals and progress bars go, to the output
     stream, and `Tee-Object -Append` shows each line and appends it to the file, so a resumed run's
     lines follow the first process's.
   - Windows PowerShell wraps every line from a program's error stream in an error record; `Plain`
     gives back the line. Without it, the first such line prints inside PowerShell's
     `NativeCommandError` block, in the file too.
   - `$ErrorActionPreference` must stay `Continue`, its default: under `Stop`, the first line on the
     error stream ends the command.
   - `PYTHONUNBUFFERED=1`: without it, Python holds its printed lines while its output goes into a
     pipe, and they reach the screen and the file only when the run ends.
   - `$LASTEXITCODE` still holds the tool's exit code after the pipe.
   - Progress bars redraw as new lines instead of in place, and since the two streams are merged as
     they arrive, a warning can land a line or two away from the record it followed.
   - Set the first four lines again in every new PowerShell session, a resume's included.

## The run

```powershell
& $py -m core train @Box --chi --device cuda --prior retrain_prior --num-runs 10000 --hidden-features 256 --num-transforms 10 --checkpoint-every 50 --name retrain --note "<one line>" 2>&1 | Plain | Tee-Object -FilePath $Log -Append
$LASTEXITCODE
```

Left at their defaults ([train](command-line.md#train)): `--model` NADROWSKI, from the bounds file's
folder; `--run-size 0`, the hardware batch, 2,048 rows on this card, so the `[budget]` line must read
`2,048 rows`; `--learning-rate 0.001`; `--stop-after-epochs 20`; `--max-epochs`, no ceiling;
`--fisher-m 48`, `--fisher-dz 0.1` and `--fisher-points 8`; `--resume auto`, and no `--new-run`; and
`--chi-k 6`, which in chi mode sets the Fisher rotation's probe count and not the training rows'
probe count: each batch draws its own, from 2 to 12.

Then the calibration and the two diagnostics that read the record:

```powershell
& $py -m core validate @Box --chi --device cuda --posterior retrain --n-cal 2000 --cal-n-scales 200 --posterior-samples 1000 --seed <N> --name retrain_cal
& $py -m core identifiability rotation @Box --chi --posterior retrain --name retrain_rotation
& $py -m core ablation @Box --chi --device cuda --posterior retrain --name retrain_ablation
```

- `validate` ([validate](command-line.md#validate)): `<N>` is a whole number you choose and write
  down; the record keeps it. `--n-cal`, `--cal-n-scales` and `--posterior-samples` are written at
  their defaults. Left at its default: `--chi-k 6`, which a calibration does not read, since it
  draws its probe count for each batch as training does.
- `identifiability rotation` ([identifiability rotation](command-line.md#identifiability-rotation))
  reads the record and simulates nothing. Left at their defaults: `--n-worst 3`, `--top-n 4`,
  `--device auto`.
- `ablation` ([ablation](command-line.md#ablation)) reads the posterior's own simulation cache, one
  reason the run keeps one. Left at their defaults: `--rows 200000`, `--n-sweep 33`.

Then the stratified SBC, a characterisation and not a gate (the reason is under
[Decisions](#decisions)): once at each fixed probe count, and once pooled.

```powershell
& $py -m core sbc @Box --chi --device cuda --posterior retrain --chi-k-fixed 2 --name retrain_sbc_k2
& $py -m core sbc @Box --chi --device cuda --posterior retrain --chi-k-fixed 6 --name retrain_sbc_k6
& $py -m core sbc @Box --chi --device cuda --posterior retrain --chi-k-fixed 12 --name retrain_sbc_k12
& $py -m core sbc @Box --chi --device cuda --posterior retrain --name retrain_sbc_pooled
```

Left at their defaults ([sbc](command-line.md#sbc)): `--repeats 10`, `--n-cal 2000`,
`--posterior-samples 1000`, `--cal-n-scales 200`, `--seed 0`, and `--chi-k 6`, which `sbc` does not
read either. Each repeat costs about what one `validate` does.

## Resuming

A run that stops keeps every batch its cache committed: every 50 batches, and on Ctrl-C or a crash
during generation, the completed batches since the last commit too. A power cut or a killed process
loses at most the batches since the last commit. A run that dies while the flow trains loses the fit
and none of the rows. To continue, open a PowerShell session, set the variables at the top of this
page and the first four lines of step 6 of the pre-flight, and run the same command with
`--resume require`:

```powershell
& $py -m core train @Box --chi --device cuda --prior retrain_prior --num-runs 10000 --hidden-features 256 --num-transforms 10 --checkpoint-every 50 --name retrain --note "<one line>" --resume require 2>&1 | Plain | Tee-Object -FilePath $Log -Append
$LASTEXITCODE
```

- **It prints, in order:**
  - `Reusing the Fisher rotation stored with the training checkpoint (<k>/10000 batches) — NOT recomputing it: the rotation's operating points are not reproducible across processes, …`,
    with `— COMPLETE, so generation will be skipped` after the batch count when every batch was
    committed;
  - `[checkpoint] the stored rotation's eigenvalues come with it (spread best/worst <x>); …`: the
    rotation's eigenvalues were found beside it;
  - `[checkpoint] resuming at batch <k>/10000 from <dir> (reusing the stored rotation V)`, from the
    simulation.

  A resumed process prints no `[fisher]` line, and its record writes the three Fisher settings as
  null ([The Fisher settings on a resumed run](window.md#the-fisher-settings-on-a-resumed-run)).
- **Keep the prior.** Resume with the same `retrain_prior`: the cache is keyed on its fit, and a
  rebuilt prior is another fit.
- **The two refusals** ([Training-cache flags](command-line.md#training-cache-flags)):
  - with `--resume require` and no cache of this run's own,
    `prism train: refused: resume='require' but there is no resumable cache at <dir>.`, followed by
    any committed cache one setting away. Something in the command differs from the first run's, and
    the cache it lists names the setting;
  - without `require`, a run one setting away from a committed cache is refused before the Fisher
    step: `This run would start a NEW simulation cache at <dir> from zero, but a committed cache ONE setting away exists:`,
    a line naming the setting and both values, and a line ending `(--new-run)`. Set the setting back
    to continue that cache. `--new-run` is right only when that one setting was changed on purpose
    and a new run from zero is what you want; it never forces a fresh start over this run's own cache.
- **Ctrl-C** prints
  `[checkpoint] stopping: saving <n> completed batches (<from> -> <to>) before unwinding…`, then
  `prism train: interrupted: the artifact being written was removed. If a [checkpoint] line above says batches were saved, the same command with --resume require continues them.`,
  and exits 130.
- **Capture every process, with `-Append`.** The posterior's `log.txt` holds only the last
  process's records, so a resumed run's first process's lines, its `[fisher]` lines among them, exist
  nowhere but in the captured file.

## What to watch

In the order the run prints them, quoted as the code prints them. The tool's own lines (`[cfg]`,
`[prism]`) and sbi's prints reach only the console; every other line is in the posterior's `log.txt`
as well ([Conventions](command-line.md#conventions)).

- `[cfg] chi: K=6 F0=0.15 range=(0.03, 0.3) x Omega_0  -> 12 probe slots, conditioning block = 72 features`,
  with `[cfg] bounds=Resources/Bounds/nadrowski/master_tier1.txt` and
  `[cfg] rescale order: ['x_scale', 't_scale', 'T']`. Anything else, stop.
- `[tier1] f_scale is DERIVED as N*beta*k_B*T/x_scale: implied range over the prior p1/p50/p99 = <a> / <b> / <c> (cell force units), min <lo> max <hi>`
  and `[tier1] chi drive amplitude = CHI_F0 * f_scale = <a> / <b> / <c> at those quantiles`: the
  force scale the prior implies, and the drives that follow. They report and never refuse
  ([The tier-1 constraint and temperature](science.md#the-tier-1-constraint-and-temperature)).
- `Training batch COUNT overridden: 10000 batches (config default 5000) — 20,480,000 training rows.`
- `[budget] 10,000 batches x 2,048 rows = 20,480,000 training rows`: anything but 2,048 rows, stop.
- In a fresh process only: `Computing decorrelating Fisher rotation (REPARAM_ROTATE=True)...`,
  `[fisher] averaged simulation Fisher over <n>/8 operating points (prior median + <n-1> prior draw(s))`
  and `[fisher] eigenvalue spread (best/worst direction): <x>`. A
  `warning: [fisher] operating point <k> …` line means that point was skipped.
- `[checkpoint] writing to <dir> every 50 batches (~10.3 GiB total, <free> GiB free)`, when the
  cache starts; a `UserWarning: Only <free> GiB free …` after it means the disk is short.
- For each batch that masks any probe, a notice on the error stream:
  `<file>:<line>: UserWarning: training batch <k>/10000 [t_scale=<…>, T=<…>, n_fine=<…>, N_points=<…>, rows=2048]: chi: <m>/<n> probes masked (below 2.0 drive cycles, at/above Nyquist, out of band, or a non-finite lock-in).`
  Here T is the batch's recording length, not the temperature.
- `warning: [patho] batch <k>: <n> new pathological trajectorie(s) -- …`, for each batch that has
  any, and at the end of generation
  `[patho] run total: <n> pathological of <rows> simulated trajectories (<p>%) -- …`.
- `[mem] training batch <k>/10000 [...]: peak allocated <a> GiB, peak reserved <b> GiB, <free>/<total> GiB reported free (optimistic on Windows), learned cap <cap>`,
  on the first batch and every 250th after it. Peak reserved far above peak allocated means the
  allocator cannot hand memory back. A few `OOM at simulation batch …` warnings are survivable; a
  steady stream means the card is fuller than the splitting absorbs.
- `[chi] masked probes: <m> of <n> (<p>%) over 10000 batches, every committed batch of the simulation cache; per batch <lo>-<hi>%, median <med>%`,
  the run total the gate reads.
- `[checkpoint] complete: 10000 batches in <dir>. Safe to delete once the posterior is saved; keeping it lets you retrain the flow without re-simulating.`
  Keep it: `ablation` reads it.
- `[winsor] clipped <n> of <m> summary elements (<p>%) to their per-column 0.1%/99.9% percentiles`:
  a few tenths of a percent.
- sbi's epoch counter, `Training neural network. Epochs trained: <n>`, and at the end
  `Neural network successfully converged after <n> epochs.`: prints of sbi's, on the console only.
- `[prism] posterior retrain__<id>  <path>`: the record is written.
- In the record: `figures/training_loss.png`, the loss curve with the best epoch and the early-stop
  window marked, and `loss.npz`, its numbers.

## Gates

A retrained model passes when all six hold; [Afterwards](#afterwards) says which record answers each.

1. **The calibration verdict passes.** `retrain_cal`'s `[verdict]` record reads
   `[verdict] PASS: every parameter's rank test at KS p >= 0.05/13 = 0.003846 and the joint coverage test at KS p >= 0.05 (calibration seed <N>)`.
   Temperature's row is marked "(assumed input)" and counts like the rest.
2. **The dead channels are revived.** In `retrain_ablation`, `A1_mean` and `D3_bimodality`, the two
   channels the August 2026 flow could not see
   ([Conditioning features](science.md#conditioning-features)), read `healthy`: each moves the
   embedding at least a tenth as far as the median channel, the same order as this run's healthy
   channels.
3. **The masked total is within ±12 points of 37 %**, read from the training run's
   `[chi] masked probes` line, the one scoped to every committed batch of the simulation cache.
   `validate` logs a line of its own, scoped to its own process, which is not this gate.
4. **The eigenvalues are recorded.** `retrain_rotation` prints `[eigenvalues] max … min … spread …`
   and the participation ratio; `[eigenvalues] NOT STORED for this artifact.` fails the gate.
5. **The loss curve is read, and the reading written down.** A clean plateau well before the best
   epoch, the validation loss no longer descending as it nears its best, reads as a limit in the
   data; a validation loss still falling near the best epoch reads as under-fitting
   ([Identifiability limits](science.md#identifiability-limits)).
6. **Informativeness is written down**, total and per parameter, as the first baseline: the
   `Informativeness` block `retrain_cal` prints. Read its sign first; temperature's line is marked
   "(assumed input)", and the total is not adjusted for it
   ([Reading calibration honestly](science.md#reading-calibration-honestly)). No earlier posterior
   exists to compare with, so this is the number later runs are compared against, always on fresh
   calibration sets.

Dropped, each with its reason:

- **The implied temperature in 280 to 310 K.** On the tier-1 box it is 100 % by construction: T is
  drawn inside that box, and the force scale is derived from it.
- **A dry run before the run.** The tool has none. The epoch timing replaces it, and the smoke gate's
  run 4 has already run this box and this network end to end.
- **Old against new on one calibration set.** No old posterior exists: the August 2026 retrain's
  records are gone.

## The certification round

A narrowing round from `retrain`, drawn around the simulated tier-1 cell at 4.5 s:

```powershell
& $py -m core infer @Box --chi --device cuda --posterior retrain @Cell --t-obs 4.5 --name cert_parent
& $py -m core tsnpe @Box --chi --device cuda --posterior retrain --observation <id> --directions 5 --level 0.999 --num-runs 10000 --hidden-features 256 --num-transforms 10 --checkpoint-every 50 --name cert_round 2>&1 | Plain | Tee-Object -FilePath $Log -Append
& $py -m core validate @Box --chi --device cuda --posterior cert_round --accept-truncated --n-cal 2000 --cal-n-scales 200 --posterior-samples 1000 --seed <N> --name cert_cal
```

- `<id>` comes from `infer`'s `[prism] observation _unnamed__<id>  <path>` line: the part after
  `_unnamed__`.
- The round is a second full-size training. It inherits neither its parent's network nor its batch
  count, so both are passed ([tsnpe](command-line.md#tsnpe)), and it costs about what the retrain's
  fit does. Left at their defaults: `infer`'s `--chi-k 6`, the probes the simulated cell is measured
  at, and `--n-samples 1000`; `tsnpe`'s `--run-size 0`, `--learning-rate 0.001`,
  `--stop-after-epochs 20`, `--max-epochs` with no ceiling, and `--resume auto`; `validate`'s as
  above.
- A round resumes like the retrain: the same `tsnpe` command with `--resume require`. It draws its
  region again from a seed taken from the observation and the round's settings, so that the same
  command finds its own cache (`core.SBI.truncate.region_from_posterior`). No round has been resumed
  on the card yet, so check for the `[checkpoint] resuming at batch …` line before leaving it.
- An optional child inference shows the narrowed posterior on the cell. Re-simulating the cell draws
  new noise, so the observation is never the region's own, and the command needs both acceptances:

  ```powershell
  & $py -m core infer @Box --chi --device cuda --posterior cert_round @Cell --t-obs 4.5 --accept-truncated --accept-other-observation --name cert_child
  ```

The round passes when:

- its `[tsnpe] truth direction <d>: <value> inside [<lo>, <hi>]` lines say inside for every truncated
  direction, as `cert_round`'s `training.truth_containment` records. A direction loaded on t_scale is
  left full width, and `[tsnpe] direction <j> NOT truncated: …` says so;
- its calibration verdict passes: `cert_cal`'s `[verdict] PASS`, with `results.accepted` showing
  `truncated`;
- the widths shrink no more than the data supports: wherever the cell's truth lies inside
  `cert_parent`'s 90 % interval, it lies inside `cert_child`'s too. A round that narrows past the
  truth has cut support, not added information ([Narrowing rounds](science.md#narrowing-rounds)).

## Afterwards

Note the posterior with its result, and save its lineage report:

```powershell
& $py -m core artifacts note posterior retrain --note "<one line: the verdict and the gates>"
& $py -m core artifacts summary posterior retrain --out <path>
```

Which record answers each gate. The keys are under each record's manifest `body`, which
`& $py -m core artifacts show <kind> <ref>` prints with the tail of its `log.txt`.

| gate | record | key or file |
|---|---|---|
| the calibration verdict | calibration `retrain_cal` | `results.verdict` (`passed`, one row per parameter, the coverage row), `results.seed`; the `[verdict]` record in its `log.txt` |
| the revived channels | diagnostic `retrain_ablation` | `results.channels`, the `A1_mean` and `D3_bimodality` entries and their `verdict`; `results.median_disp` |
| the masked total | posterior `retrain` | the `[chi] masked probes` record in its `log.txt` |
| the eigenvalues | posterior `retrain`; diagnostic `retrain_rotation` | `transform.fisher_eigenvalues`; `results.eigenvalues` |
| the loss curve | posterior `retrain` | `figures/training_loss.png`, `loss.npz`, `training.epochs_trained`, `training.best_validation_loss` |
| informativeness | calibration `retrain_cal` | `results.informativeness` (`total_nats`, `sem_nats`, `per_param`, `per_direction`) |
| the stratified SBC, a characterisation | diagnostics `retrain_sbc_k2`, `retrain_sbc_k6`, `retrain_sbc_k12`, `retrain_sbc_pooled` | `results.rank_verdict` |
| the round's truth | posterior `cert_round` | `training.truth_containment` |
| the round's calibration | calibration `cert_cal` | `results.verdict`, `results.accepted` |
| the round's widths | inferences `cert_parent`, `cert_child` | `results.posterior_summary` (`q05`, `median`, `q95`), `results.ground_truth` |
| the run's cost | the simulation cache; posterior `retrain` | the cache's `wall_seconds`, from its creation to its completion across resumes; the posterior's `training.wall_seconds`, the last process only |
| the lines no record keeps | the console capture | `$Log`: the tool's and sbi's prints, and every process's records but the last |
