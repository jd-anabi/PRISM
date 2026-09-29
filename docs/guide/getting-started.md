# Getting started

Checked against commit 04fbe54.

This page takes you from a fresh installation to a first run in each front end, and shows where
PRISM keeps what it reads and what it writes.

## Launching PRISM

PRISM runs in the conda environment the root README's [installation steps](../../README.md#installation)
create (`biophys-env`). Activate it before you launch anything: every command below runs the
`python` it finds, and only this environment's has PRISM's dependencies installed.

| to start | on Windows (PowerShell) | on macOS and Linux (bash) |
|---|---|---|
| the window | `run.bat` | `bash run.sh` |
| the window, directly | `$env:KMP_DUPLICATE_LIB_OK="TRUE"; python -m core.gui` | `KMP_DUPLICATE_LIB_OK=TRUE python -m core.gui` |
| the command-line tool | `python -m core <subcommand>` | `python -m core <subcommand>` |

- **`KMP_DUPLICATE_LIB_OK=TRUE`.** torch and MKL each ship an OpenMP runtime; without this variable
  the first simulation aborts with "OMP Error #15" and no traceback. `run.bat` and `run.sh` set it
  before they start the window. A direct launch of the window must set it by hand, as in the table:
  the window's entry module, `core.gui.__main__`, sets only matplotlib's backend.
- **The launchers' interpreter.** `run.bat` runs the first `python` on the PATH and `run.sh` runs
  `python3`, so start either from a terminal in which the environment is active.
- **The tool needs nothing set around it.** Its entry module, `core.__main__`, sets
  `KMP_DUPLICATE_LIB_OK` itself (unless it is already set) and switches matplotlib to its
  non-interactive Agg backend, both before anything imports torch.
- **The working directory does not matter** to PRISM's own folders: `core.config` finds them from its
  own location in the repository. It does matter to a path you type. A flag such as
  `--bounds Resources/Bounds/nadrowski/master.txt` is read relative to where you are, so the examples
  in this guide are run from the repository root.

## Inputs and records

PRISM keeps its inputs apart from the records its runs generate.

- **Inputs** live under `Resources/`, in four folders: `Bounds/` (which parameters are inferred, in
  what order and over what box), `Cells/` (one cell's initial conditions and parameter values),
  `Units/` (the unit system a model's files are written in) and `Models/` (the definitions of
  user-defined models). Bounds, cell and units files are edited by hand. The window's model builder,
  reached from Settings, writes a user model's definition into `Models/`, and a first bounds file,
  cell file and units file for that model into the other three folders
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
  whose path the first `[smoke]` line prints and which is left on disk afterwards;
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

The window opens on Home, which offers five sections (`core.gui.screens.home_screen.SECTIONS`):
Reduction Map, FDT Analysis, Parameter Inference, Simulate and Artifacts. The inference chain lives in
Parameter Inference, whose six tabs are used in order: Config, Prior, Posterior, Validate, Infer and
TSNPE. A tab stays greyed until the tabs before it have produced what it needs; hovering over a greyed
tab says what is missing (`core.gui.screens.inference_screen.InferenceScreen.refresh_gates`).

1. **Config.** Choose the model and its options (tick "Multi-frequency χ(ω) conditioning" for chi
   mode), then press **Apply model & options**.
2. **Prior.** Choose a bounds file, leave the Prior picker on its "(from scratch)" entry and press
   **Build / Load prior**, which runs the stability sweep and fits the prior. Picking a saved prior
   instead loads it.
3. **Posterior.** Leave the Posterior picker on its "(from scratch)" entry, set the Training budget,
   and press **Train / Load posterior**.
4. **Validate.** Press **Run calibration** to calibrate the posterior on data simulated from its own
   prior.
5. **Infer.** Choose "Simulated (cell ground truth)" and a cell, or "Experimental data" and your
   recordings, set the recording length T_obs, and press **Run inference**.
6. **TSNPE**, when you want it: **Run TSNPE round** trains a narrowed posterior around a recorded
   observation (the Infer tab records one each time it runs).

Each stage writes its record when it completes. The Save button beside a name box on the Prior and
Posterior tabs names the record just made; saving is a rename, and nothing is simulated again.

A small training budget (a few batches in the Posterior tab's Batches box) gives a posterior that
shows the chain working and nothing more: at that size neither the posterior nor its calibration
means anything. The default budget is a full-size training run. In chi mode, give a simulated cell a
T_obs of a few seconds rather than the 1 s default, for the reason given for `--t-obs` below.
[The window](window.md) covers every screen and setting, and
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

Every record belongs to one of eight kinds, the keys of `core.artifacts.store.KIND_DIRS`. A kind is
named in the singular; seven have a plural directory, and `fdt` keeps its own name.

| kind | directory under the root | written by the tool | written by the window |
|---|---|---|---|
| `prior` | `priors/` | `prior` | the Prior tab |
| `simulation` | `simulations/` | `train`, `tsnpe` | the Posterior and TSNPE tabs |
| `posterior` | `posteriors/` | `train`, `tsnpe` | the Posterior and TSNPE tabs |
| `observation` | `observations/` | `infer` | the Infer tab |
| `calibration` | `calibrations/` | `validate` | the Validate tab |
| `inference` | `inferences/` | `infer` | the Infer tab |
| `diagnostic` | `diagnostics/` | `sbc`, `identifiability`, `ablation`, `probes` | none |
| `fdt` | `fdt/` | `fdt`, `crossval`, `compare` | the FDT Analysis screen |

`smoke` writes the first six kinds into its own root; it writes a simulation cache only with
`--checkpoint`. Three things are saved outside the kinds: the Reduction Map writes plain files, with
no manifest, into `reduction/` under the records root (`core.config.artifacts_root()`); Simulate
saves a video wherever you choose; and the model builder writes into the inputs root, as described
under [Inputs and records](#inputs-and-records).

**What a record holds.** A record's directory holds:

- `manifest.json`: what the record is and how it was made, including the git revision and the
  environment, the input files and their hashes, its parents, its payloads' digests, its settings and
  its results;
- `log.txt`: the run's records, one per line, stamped `HH:MM:SS info`, `HH:MM:SS warning` or
  `HH:MM:SS error`, with every Python warning raised meanwhile (`core.runs`);
- its payload files, such as `prior.pt`, `posterior.pt`, `results.json` or `data.h5`, and a
  `figures/` folder of its figures.

A record is written when its run completes. A run that fails or is cancelled leaves no record, with
two exceptions:

- **The simulation cache** holds a training run's simulated rows. Its directory is named by a
  12-character digest of the run's identity, it is committed batch by batch, and a training run with
  the same identity resumes from it. It has no `log.txt`: it has no writer of its own.
- **An `fdt` record is progressive**: its directory and a first manifest exist from the moment the
  run starts and are rewritten as it goes, so a cancel or a crash keeps the folder, marked unfinished.

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
- A taken name is refused when the stage starts, before anything is spent.
- Naming a record after the fact (the window's Save) is a rename: the manifest is rewritten, then the
  directory is renamed.

**Loading refuses a mismatch.** Loading a record refuses any mismatch it can verify against the
configuration in use, before anything is spent: the model, the parameter set and its order, the box,
the observation mode, the conditioning width. The load path has exactly two escape hatches, the
fields of `core.artifacts.Accept`, and neither lets a mismatch through:

- `truncated`: load a posterior that a narrowing round (TSNPE) trained, which is valid only near the
  observation its region was drawn around (`--accept-truncated`; in the window, a confirmation on the
  Posterior tab);
- `other_observation`: run such a posterior on an observation other than its region's
  (`--accept-other-observation`; in the window, "Run on a different observation" on the Infer tab).

Each use is recorded downstream: an inference, a calibration and every diagnostic that loads a
posterior record the acceptances they ran under, and a narrowing round records the ones its parent
was loaded under.

## Browsing the store

**In the window**, the Artifacts screen (Home's fifth section) reads the store:

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
until then.

**On the command line**, the `artifacts` family does the same against the `PRISM_ARTIFACTS` root:

| command | what it does |
|---|---|
| `python -m core artifacts list [<kind>]` | one line per record of a kind, or of all eight kinds under headings |
| `python -m core artifacts show <kind> <ref>` | the record's manifest and its run's log |
| `python -m core artifacts note <kind> <ref> --note TEXT` | sets the note; `--note ''` clears it |
| `python -m core artifacts rm <kind> <ref>` | deletes one record; refused while anything depends on it |
| `python -m core artifacts sweep [<kind>] [--yes]` | removes directories with no manifest and loose files, and, with no kind, the legacy `crossval/` directory; anything written in the last five minutes is refused and named; a dry run until `--yes` |
| `python -m core artifacts summary <kind> <ref> [--out PATH]` | the lineage report: the record, then its parents |

The family loads nothing, so it cannot say whether a posterior matches your bounds file; loading it
does. [artifacts](command-line.md#artifacts) has every mode in full (`core.tool.browse`).
