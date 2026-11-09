# Getting started

Checked against commit 531aae3.

This page takes you from a fresh installation to a first run in each front end, and shows where
PRISM keeps what it reads and what it writes.

## Launching PRISM

PRISM runs in the conda environment the root README's [installation steps](../../README.md#installation)
create (`biophys-env`). On Windows the launcher, `run.bat`, finds that environment itself. Every other
command below runs the `python` it finds, so activate the environment first: only its `python` has
PRISM's dependencies installed.

| to start | on Windows (PowerShell) | on macOS and Linux (bash) |
|---|---|---|
| the window | `run.bat`, or the desktop shortcut it makes | `bash run.sh` |
| the window, directly | `python -m core.gui` | `python -m core.gui` |
| the command-line tool | `python -m core <subcommand>` | `python -m core <subcommand>` |

- **The launchers start the window with no console.** `run.bat` starts it with the environment's
  `pythonw.exe` and returns at once, and `run.sh` starts it detached from the terminal, which may then
  be closed. Only one PRISM runs at a time, so a second start brings the open window to the front.
  What the window prints outside a
  run, a record another thread logs during one, the toolkit's own messages, an error nothing else
  caught and a crash of the interpreter go to `window-log.txt` beside the settings file, after a dated
  line for each launch; a run's full text is on the run's page and in the dated file it opens, in the
  `runs` folder there. A block in that file that opens `Windows fatal exception:` and lists the
  threads' Python stacks, followed by more of the same launch's lines, is not a crash: Python's fault
  handler, which PRISM points at that file to keep a crash of the interpreter, writes such a block
  for an exception Windows reports to the process even when the code that raised it handles it, and
  PRISM went on. After a real crash nothing of that launch follows the block; the next line, if any,
  is the next launch's dated one. A start that fails before its window is shown writes why to that
  file and, on Windows only, says so in a plain Windows message that names it, and a second start from
  `run.bat` or its shortcut that the open PRISM does not answer says so in one too. On macOS and
  Linux no message is shown: `run.sh` prints where to look as it starts. A direct launch keeps its
  console and writes there instead.
- **Which Python `run.bat` uses.** The active environment's `pythonw.exe`, when it is started from a
  prompt whose active environment holds PySide6 and torch (a prompt left in conda's `base`
  environment holds neither); else `envs\biophys-env\pythonw.exe` under `anaconda3` or `miniconda3`
  in your user folder, `anaconda3` in your local application data folder, or `anaconda3` or
  `miniconda3` in the folder shared by every user. When it finds none, a Windows message says what to
  do. `run.bat --where` prints the Python it would use, and starts nothing.
- **The desktop shortcut.** The first time `run.bat` runs it puts `PRISM.lnk` on the desktop (the
  environment's `pythonw.exe -m core.gui`, started in the repository, with PRISM's icon) and leaves a
  mark, `shortcut-made`, beside the settings file; it never makes the shortcut again, even when it was
  deleted. `run.bat --make-shortcut` makes it, without starting PRISM, when it was never made; to make
  it again, delete the mark first (`run.bat --where` names it). The shortcut starts PRISM with
  the environment Windows gives it, so a variable set in a shell (`PRISM_ARTIFACTS`,
  `PRISM_VRAM_CEILING_GIB`) reaches the window only through `run.bat` started from that shell.
  The memory limit needs no variable: Settings, under This computer, keeps it for the computer it was
  set on, and a variable that holds a number takes precedence over it
  ([Memory planning](architecture.md#memory-planning-and-out-of-memory-recovery)).
- **`KMP_DUPLICATE_LIB_OK=TRUE`.** torch and MKL each ship an OpenMP runtime; without this variable
  the first simulation aborts with "OMP Error #15" and no traceback. Both entry modules, the window's
  `core.gui.__main__` and the tool's `core.__main__`, set it themselves (unless it is already set) and
  switch matplotlib to its non-interactive Agg backend, both before anything imports torch; the
  launchers set it too. Neither front end needs anything set around it.
- **The working directory does not matter** to PRISM's own folders: `core.config` finds them from its
  own location in the repository. It does matter to a path you type. A flag such as
  `--bounds Resources/Bounds/nadrowski/master.txt` is read relative to where you are, so the examples
  in this guide are run from the repository root.

## Inputs and records

PRISM keeps its inputs apart from the records its runs generate.

- **Inputs** live under `Resources/`, in five folders: `Bounds/` (which parameters are inferred, in
  what order and over what box), `Cells/` (one cell's initial conditions and parameter values),
  `Units/` (the unit system a model's files are written in), `Models/` (the definitions of
  user-defined models) and `Plans/` (run plans: the settings of the first three steps, one folder per
  model). `Plans/` does not exist until the first plan is saved from the window, and no plan ships.
  Bounds, cell, units and plan files are edited by hand, and are tracked in git like the rest of the
  inputs: commit a plan you saved before a long run, or every record the run writes is stamped as made
  from a changed tree. The window's model builder,
  under Your own models, writes a user model's definition into `Models/`, and a first bounds file,
  cell file and units file for that model into `Bounds/`, `Cells/` and `Units/`
  (`core.Helpers.model_store.save_user_model`). [Bringing recordings](recordings.md#input-files)
  describes each format.
- **Records**, what every stage, diagnostic and FDT run writes, live under the records root
  (`Artifacts/` unless overridden), one directory per record:
  `<records root>/<kind directory>/<name>__<id>/`. [The artifact store](#the-artifact-store) below
  describes them, and the few outputs that are not records.

`core.config` resolves both roots from its own location, never from the working directory, and an
environment variable overrides each:

| root | default | override | when the override is read |
|---|---|---|---|
| inputs: `core.config.RESOURCES_ROOT` | `<repo>/Resources` | `PRISM_RESOURCES` | once, when `core.config` is imported |
| records: `core.config.artifacts_root()` | `<repo>/Artifacts` | `PRISM_ARTIFACTS` | at every call |

Which records root a run writes to:

- **the window**: the `PRISM_ARTIFACTS` root, resolved once, when the window starts
  (`core.gui.app.build_app`);
- **`smoke`**: the root `--store-root` names; without it, a fresh temporary directory made for that run,
  whose path the first `[smoke]` line prints and which is left on disk unless the run failed before
  writing anything;
- **`fdt` and `crossval`**: the root `--store-root` names; without it, the `PRISM_ARTIFACTS` root;
- **every other subcommand**, the `artifacts` family included: the `PRISM_ARTIFACTS` root, and no
  other (`core.tool.main`).

To keep a first try away from the real store, point `PRISM_ARTIFACTS` at a scratch folder before you
start anything, and delete the folder when you are done:

```powershell
$env:PRISM_ARTIFACTS = "$HOME\prism-scratch"
```

```bash
export PRISM_ARTIFACTS="$HOME/prism-scratch"
```

## A first run in the window

The window opens on Home, which says what to do next. The inference chain is the six steps of
Parameter inference, which the rail lists under that part while it is open; each step is a page of
its own, Alt+1 to Alt+6 open them from anywhere, and they are used in order. The first three steps
are the rebuilt pages described here; steps 4 to 6 are still the older pages, with their own
controls. [The window](window.md) describes the window as it was before its layout was redesigned and
names the steps by their old tab names, Config, Prior, Posterior, Validate, Infer and TSNPE;
[The Parameter Inference tabs](window.md#the-parameter-inference-tabs) says what each needs before it can start.

**Home.** Home's heading says the next step and its button goes there. PRISM starts every launch
with nothing loaded. When a model was in use last time and nothing is loaded yet, Home's button reads
**Continue with these** and its sentence lists what that is: the model, the kind of observation, the
bounds file, the prior, the posterior and the plan last used, each only while it can still be found as
it was used. One press applies step 1, loads the prior, then the posterior, one after another, each
once the one before has returned its record; nothing is loaded until you press. If you changed the
forms and did not apply them, the press first asks and lists what it would set, with "Keep what is
there" as its safe answer. A chain that stops says where, under Home's heading, and the step it
stopped at says why.

**A step's page.** Each of the first three steps has a heading, a line that says what the step holds
("In use"), the form, and a foot that never scrolls: **Before you start** lists what stops the start
(marked Error or Blocked), what needs attention and what is fine, with the words that go beside the
main button, and **Check again** reads the saved records again. A page reads the disk when it is
shown, 300 ms after your last edit, when a run ends, when the store changes, at **Check again** and
at the press, and never while a run is live. Under each setting a small word says where its value came
from: usual, changed from the usual, kept from last time, "from plan NAME", or changed from the plan
("from run NAME" after a Continue).

1. **Model and options.** Choose the model, the kind of observation ("Frequency response" is chi
   mode) and the units, then press **Apply model and options**. Applying while records are in use asks
   first, because it starts a new session. The **Plan** row fills all
   three steps' settings from a plan file ([Run plans](recordings.md#run-plans)).
2. **Prior.** Choose a bounds file, give the prior a Name, and press **Build a new prior**, which
   runs the stability sweep and fits the prior. **Use a saved prior instead…** switches the page to
   its list of saved priors, and **Load this prior** loads the one you chose.
3. **Training.** Set the training budget (Batches, Simulations per batch, and the network's width and
   depth), give the posterior a Name, and press **Start training**. **Use a saved posterior
   instead…** switches the page to the saved posteriors, and **Load this posterior** loads one. A
   training that finds its simulation set whole or partly saved says so under **Before you start**,
   and **Start training** continues it. With **Only continue** chosen, the button reads **Continue
   from batch N**, or **Fit again on the saved simulations** when the set is whole.
4. **Checks.** Press **Run calibration** to calibrate the posterior on data simulated from its own
   prior.
5. **Inference.** Choose "Simulated (cell ground truth)" and a cell, or "Experimental data" and your
   recordings, set the recording length T_obs, and press **Run inference**.
6. **Narrowing round**, when you want it: **Run TSNPE round** trains a narrowed posterior around a
   recorded observation (step 5 records one each time it runs).

The name is typed before the run and the record is written under it when the run completes; there is
no Save afterwards on these steps. The name is checked as you type against the store's rule (the
letters a to z in either case, digits, `_`, `.` and `-`, at most 64 characters, starting with a letter
or a digit) and against the names already taken. Under a choice on its form that differs from what
the step holds, a step says "Changed, not applied" and what is in use; "In use" counts such choices,
and the rail marks the step.

While a step's run goes, the run's own page stands in place of the step's form: where the run stands,
its messages and the figures it made, and a stop button that asks before it stops anything that would
be lost. The foot of the rail says what is running from any part, and pressing it goes to the run.
Each stage writes its record when it completes.

**The memory limit.** In Settings, under This computer, **Memory limit** is either "No limit" or
"Limit each batch to" a number of GiB: the most graphics-card memory one batch of simulations may plan
to use. It changes how a batch is split, never what is simulated, it is remembered for the computer it
was set on, and the Training step's "Before you start" names the limit in force. A full-size
simulation on a card shared with the desktop needs it set ([The retrain runbook](retrain.md)). The
same page holds the switch that keeps the computer awake while a run is live. The limit is locked while a
run is live.

**After a reopening.** When the newest training failed or was cut off (PRISM or the computer closed
during it), the next launch's Home offers **Continue from batch N** if its saved simulations stop
short of the end. If they are whole and no kept training stands on them, it offers **Fit again on
the saved simulations**. A training you stopped during its simulations is not offered on Home: the
Training step says how far its simulations reached, and its own button continues them. A training
kept after some epochs, stopped, failed or cut off during its fit, is offered with **Resume**.
**Continue from batch N** and **Fit again** ask first if something is in use,
apply step 1 with the run's own model and options, fill the Training step's form from the settings
the run was started with, load its prior and open the Training step with "Only continue" chosen and
the name and note blank; you press the Training step's own button to start. The Training step lists
every kept training with **Resume** and **Delete** beside it; **Resume** puts the kept training's own
model and prior in use first, asking if something is in use, then resumes it, and a resume first
reads the simulation set back. A kept narrowing round is
resumed from the command line: Home and the Training step print the `tsnpe --resume-fit` command for
it; the narrowing round's page has no Resume button.

A small training budget (a few batches in the Training step's Batches box) gives a posterior that
shows the chain working and nothing more: at that size neither the posterior nor its calibration
means anything. The default budget is a full-size training run. In chi mode, give a simulated cell a
T_obs of a few seconds rather than the 1 s default, for the reason given for `--t-obs` below.
[The window](window.md) covers every setting, and
[The inference settings](window.md#the-inference-settings) says what each setting really changes.

## A first run on the command line

The tool's own end-to-end run is `smoke`, which builds a prior, trains a posterior, calibrates it and
runs an inference on a simulated cell. The training and the calibration run at tiny sizes, while the
prior is built at its full default size, so the run takes several minutes even on a graphics card:

```bash
python -m core smoke --chi --t-obs 4.5 --bounds Resources/Bounds/nadrowski/master.txt --cell Resources/Cells/nadrowski/master_spont.txt
```

- **`--bounds`** is required by every subcommand of the inference chain: the five stages, the
  diagnostics, `probes` and `smoke`. The bounds file declares which parameters are inferred, in what
  order and over what box, and a posterior loads only under the bounds file it was trained with.
- **`--chi`** selects chi mode. Without it the tool follows `core.config.CHI_MODE`, which is False,
  and this bounds file, which has a Forcing section, would select forced mode instead.
- **`--t-obs 4.5`** sets the length, in seconds, of the observation the infer stage simulates.
  `smoke`'s default is 1 s (`core.config.T_MIN_EXP_S`), and that is too short for chi mode on this
  cell: a probe locked in over fewer than `core.config.CHI_MIN_CYCLES` (two) drive cycles is masked,
  and at 1 s the probes at the low end of the band get fewer.
  [Recording length](recordings.md#recording-length) explains how long is long enough.

The first `[smoke]` line prints the temporary store the run writes (`store=...`). The last lines say
that the chain runs, and that the run says nothing about calibration: at these sizes it cannot.
[smoke](command-line.md#smoke) lists what to watch in its output.

A real chain runs the stages one by one, each writing a record the next one names:

```bash
python -m core prior    --chi --bounds Resources/Bounds/nadrowski/master.txt --name first_prior
python -m core train    --chi --bounds Resources/Bounds/nadrowski/master.txt --prior first_prior --name first_posterior
python -m core validate --chi --bounds Resources/Bounds/nadrowski/master.txt --posterior first_posterior
python -m core infer    --chi --bounds Resources/Bounds/nadrowski/master.txt --posterior first_posterior --cell Resources/Cells/nadrowski/master_spont.txt --t-obs 4.5
```

Keep `--chi` and `--bounds` the same along the chain: a posterior loads only under the observation
mode and the bounds file it was trained with. At its defaults `train` is a full-size training run
(`--num-runs` 5000 batches); add a small `--num-runs` for a first look.
[Worked examples](command-line.md#worked-examples) has complete chains, and every subcommand's
`--help` lists its flags.

## The artifact store

Every record belongs to one of nine kinds, the keys of `core.artifacts.store.KIND_DIRS`. A kind is
named in the singular; eight have a plural directory, and `fdt` keeps its own name.

| kind | directory under the root | written by the tool | written by the window |
|---|---|---|---|
| `prior` | `priors/` | `prior` | step 2, Prior |
| `simulation` | `simulations/` | `train`, `tsnpe` | steps 3 and 6, Training and Narrowing round |
| `posterior` | `posteriors/` | `train`, `tsnpe` | steps 3 and 6, Training and Narrowing round |
| `observation` | `observations/` | `infer` | step 5, Inference |
| `calibration` | `calibrations/` | `validate` | step 4, Checks |
| `inference` | `inferences/` | `infer` | step 5, Inference |
| `diagnostic` | `diagnostics/` | `sbc`, `identifiability`, `ablation`, `probes` | none |
| `fdt` | `fdt/` | `fdt`, `crossval`, `compare` | Effective temperature |
| `fit` | `fits/` | `train`, `tsnpe` | steps 3 and 6, Training and Narrowing round |

`smoke` writes the first six kinds into its own root; it writes a simulation cache only with
`--checkpoint`. A few things are saved outside the kinds, with no manifest:

- the Reduction map writes plain files into `reduction/` under the records root
  (`core.config.artifacts_root()`);
- Live simulation's **Save video…**, a figure window's **Save As…**, Records' **Save…** and **Lineage
  report…**, and `artifacts summary --out` each write where you choose;
- the model builder writes into the inputs root, as described under
  [Inputs and records](#inputs-and-records).

**What a record holds.** A record's directory holds:

- `manifest.json`: what the record is and how it was made, including the git revision and the
  environment, the input files and their hashes, its parents, its payloads' digests, its settings and
  its results;
- `log.txt`: the run's records, one per line, stamped `HH:MM:SS info`, `HH:MM:SS warning` or
  `HH:MM:SS error`, with every Python warning raised meanwhile (`core.runs`);
- its payload files, such as `prior.pt`, `posterior.pt`, `results.json` or `data.h5`, and a
  `figures/` folder of its figures.

A record is written when its run completes. A run that fails or is stopped leaves no record, with
three exceptions:

- **The simulation cache** holds a training run's simulated rows. Its directory is named by a
  12-character digest of the run's identity, it is committed batch by batch, and a training run with
  the same identity resumes from it. It has no `log.txt`: it has no writer of its own.
- **A kept training** (`fit`) is a training saved after every pass while its rows are in a kept
  simulation set; one that stops or fails partway stays kept at its last saved pass, to be resumed,
  and it is removed once its posterior is written.
- **An `fdt` record is progressive**: its directory and a first manifest exist from the moment the
  run starts and are rewritten as it goes, so a stop or a crash keeps the folder, marked unfinished.

**Names and ids.**

- A record's id is the UTC time it was created, `YYYYMMDDTHHMMSS`, with `-2`, `-3` and so on added when
  two land in the same second. A simulation cache's id is its digest.
- A record directory is `<name>__<id>`, or `_unnamed__<id>` until it is named; a simulation cache's
  directory is its digest alone.
- A name starts with a letter or a digit, holds letters, digits, `_`, `.` and `-`, runs to at most 64
  characters, and may not be shaped like an id. Names are unique within a kind.
- A reference to a record, wherever a flag asks for one (`--prior`, `--posterior`, `--observation`,
  `--record`, an `artifacts` mode's `<ref>`), is its name or its bare id. The directory name
  `_unnamed__<id>` is not a reference.
- A taken name is refused when the stage starts, before anything is spent; the window checks it as it
  is typed, and the first three steps ask for it before the run.
- Naming a record after the fact is a rename (`ArtifactStore.rename`): the manifest is rewritten, then
  the directory is renamed. The window names a record when its run starts and has no control that
  renames one.

**Loading refuses a mismatch.** Loading checks a record against the configuration in use before
anything is spent: the model and the parameter set and its order; for a prior or a posterior also
the box; for a posterior or an observation the mode and the conditioning width.
[The artifact store](architecture.md#the-artifact-store) lists every check each loader makes, and
the recorded settings it leaves unchecked. The load path has exactly two escape hatches, the fields
of `core.artifacts.Accept`, and neither lets a mismatch through:

- `truncated`: load a posterior that a narrowing round (TSNPE) trained, which is valid only near the
  observation its region was drawn around (`--accept-truncated`; in the window, a confirmation at
  step 3, Training);
- `other_observation`: run such a posterior on an observation other than its region's
  (`--accept-other-observation`; in the window, "Run on a different observation" at step 5,
  Inference).

Each use is recorded downstream: an inference, a calibration and every diagnostic that loads a
posterior record the acceptances they ran under, and a narrowing round records the ones its parent
was loaded under.

## Browsing the store

**In the window**, the part Records (Ctrl+6), whose page is headed Artifacts, reads the store:

- choose a kind, and a table lists its records;
- select a record, and the detail pane shows its manifest and its run's log; **Save…** writes what is
  shown to a text file, and **Lineage report…** writes the record and its parents, back through the
  chain;
- the Actions box sets or clears the record's note (**Set**) and deletes it (**Delete…**, refused
  while any other record depends on it);
- **Sweep this kind…** removes the kind's directories that have no manifest at all and any loose file
  inside its folder; **Sweep all kinds…** does that for every kind and also removes the legacy
  `crossval/` directory an older build left beside the kind directories
  (`core.artifacts.store.LEGACY_DIRS`). Each asks before it removes anything.

Reading is never blocked. The actions that change the store are refused while a run in the window is
live (`core.gui.screens.artifact_screen.ArtifactScreen`). A run in another process, such as a
command-line run, is guarded only by the sweeps' recency check: a directory or file written in the
last five minutes is refused rather than removed (`core.artifacts.store.RECENT_WRITE_SECONDS`),
because an ordinary record's manifest is written last and a run in flight looks like a leftover
until then. A kept training is the exception: one that a running process holds, in this window or
any other process, cannot be deleted (`core.SBI.fit_state.running`).

**On the command line**, the `artifacts` family does the same against the `PRISM_ARTIFACTS` root:

| command | what it does |
|---|---|
| `python -m core artifacts list [<kind>]` | one line per record of a kind, or of all nine kinds under headings |
| `python -m core artifacts show <kind> <ref>` | the record's manifest and its run's log |
| `python -m core artifacts note <kind> <ref> --note TEXT` | sets the note; `--note ''` clears it |
| `python -m core artifacts rm <kind> <ref>` | deletes one record; refused while anything depends on it |
| `python -m core artifacts sweep [<kind>] [--yes]` | removes directories with no manifest and loose files, and, with no kind, the legacy `crossval/` directory; anything written in the last five minutes is refused and named; a dry run until `--yes` |
| `python -m core artifacts summary <kind> <ref> [--out PATH]` | the lineage report: the record, then its parents |

The family loads nothing, so it cannot say whether a posterior matches your bounds file; loading it
does. [artifacts](command-line.md#artifacts) has every mode in full (`core.tool.browse`).
