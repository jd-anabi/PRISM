# The command-line tool

Checked against commit 878109c.

`python -m core <subcommand> [<mode>] <flags>` is PRISM's command-line tool (`core.tool`). It takes
flags only and never prompts, so a run can be written down and repeated. This page lists every
subcommand, mode and flag: the conventions and the shared flag groups first, then one section per
command, in the order `python -m core --help` lists them, then worked examples. Where a command's
`--help` states a flag's default, it ends the flag's help with a `(default ...)` clause, and the
flag's entry here ends with the same clause. The examples are run from the repository root, as
[Launching PRISM](getting-started.md#launching-prism) explains.

## Conventions

- **References.** Wherever a flag or an `artifacts` mode asks for a record (`--prior`,
  `--posterior`, `--observation`, `--record`, `<ref>`), it takes the record's name or its bare id,
  as [The artifact store](getting-started.md#the-artifact-store) describes. That section also says
  what a record holds and what the kinds, names and ids are, and
  [Inputs and records](getting-started.md#inputs-and-records) says where each run saves.
- **What goes where.** The tool's own framing lines (`[cfg]`, `[prism]`, `[smoke]`, `[prism fdt]`,
  `[prism crossval]`, and the `artifacts` family's listings and reports) are prints on stdout. They
  are not records, so no `log.txt` holds them. A stage's messages are records: information goes to
  stdout as the bare message, and warnings and errors go to stderr with the prefix `warning: ` or
  `error: ` (`core.tool.logging_console`); the record the run writes keeps them all in its
  `log.txt`. A library's logging records go to stderr whatever their level
  (`core.tool._library_record_sink`), but a library's own prints still reach stdout: during `train`,
  `tsnpe` and `smoke`'s posterior stage, sbi prints its epoch counter, its convergence line and its
  training summary there, and no log keeps them. A Python warning, such as a masked-probe notice or
  a pre-flight warning about a recording length, prints in Python's own form on stderr, and the
  run's `log.txt` keeps it too. Refusals and usage errors go to stderr ([Exit codes](#exit-codes)).
- **The parameter order is printed on every run.** Every command that builds a configuration opens
  with its `[cfg]` lines: the model, the observation mode and the device; the cell and bounds files;
  in chi mode the probe settings; then the ND order, the rescale order and the forcing order
  (`core.tool.config_args.describe`). The simulators bind parameter columns by position, so a bounds
  file whose order differs from the one a command assumes mis-binds values with no error and no
  crash. The printed order, beside the bounds path, is the only place that difference shows.
- **Defaults stay with their stage.** A value flag is forwarded only when it is given
  (`core.tool.config_args.knobs`), so the stage's own default applies: the one `core/config.py` or
  the stage's signature holds, which is the one each entry below states. The one exception is
  `validate --seed`: when none is given the tool draws a seed and passes it, so the calibration
  record names the seed it ran with. `smoke` is the other way round: its defaults are its own small
  sizes, which it always passes.
- **Flags in full.** argparse accepts an unambiguous prefix of a long flag on every command except
  the `probes` modes, which refuse abbreviations so that a flag typed from habit cannot silently set
  a longer one. This page always writes flags in full.
- **`--help` costs nothing.** It builds the parser only, which imports no torch, so it answers at
  once and needs no graphics card. A command imports its heavy modules after its flags parse.

## Shared flags

Four groups of flags are added by helpers in `core.tool.config_args` and mean the same on every
command that takes them. Each command's section lists them again, briefly, with the default its own
`--help` states.

### Configuration flags

Taken by `prior`, `train`, `validate`, `infer`, `tsnpe`, `sbc`, the three `identifiability` modes,
`ablation` and `smoke`. The `probes` modes take `--bounds`, `--model` and `--device` only, because
each sets its own observation mode. `fdt`, `crossval`, `compare` and `artifacts` take none of them.

- `--bounds PATH` — required. The bounds file declares which parameters are inferred, in what order
  and over what box, and whether a drive is declared (a Forcing section). With `--no-chi`, a bounds
  file with a Forcing section gives forced mode and one without gives spontaneous mode; `--chi` gives
  chi mode either way ([Observation modes](recordings.md#observation-modes)). What a command loads
  is checked against it. A posterior is refused on a mismatch in model, parameter set or order, box
  or mode (`core.artifacts.store.ArtifactStore.load_posterior`). A prior is refused on a mismatch in
  model, ND parameter set or order, ND box or log-box mask; its mode is not checked, so one prior
  serves every mode over the same box (`core.artifacts.store.ArtifactStore.load_prior`).
- `--model NAME` — the model. Without it, the bounds file's parent folder, upper-cased:
  `Bounds/nadrowski/` gives NADROWSKI. A name that is neither a built-in model nor a user model
  eligible for inference is refused (`core.tool.config_args.make_cfg`).
- `--chi` / `--no-chi` — chi mode on or off. The default is `core.config.CHI_MODE`, which is off, so
  every chi command passes `--chi`. A posterior loads only in the mode it was trained in, so keep the
  flag the same along a chain.
- `--chi-k K` — the probes per observation, 6 by default: the count a simulated observation is
  measured at, and the count the Fisher rotation and `identifiability jacobian` use. Training draws
  its own count for every batch, from 2 up to the network's 12 probe slots, so `--chi-k` does not
  reach it.
- `--device auto|cpu|cuda` — where the run computes. `auto`, the default, takes a CUDA card of
  compute capability 8.0 or above, else Apple's MPS, else the CPU; `cpu` forces the CPU; `cuda`
  requires the card and is refused `(--device)` when it is absent or below 8.0
  (`core.config.detect_device`). The device sets the hardware batch, 2,048 rows on a CUDA card and 64
  elsewhere. The device type and its number type are part of a simulation cache's identity: a run
  on another device starts a cache of its own, while `cuda` and `auto` on a qualifying card share
  one.

### Naming flags

Taken by every command that writes a record except `smoke`, whose `--save` names what it builds:
the five stages, `sbc`, the `identifiability` modes, `ablation`, the `probes` modes, `fdt`,
`crossval` and the `compare` modes.

- `--name NAME` — names the record the command writes; `''`, the default, leaves it unnamed. A taken
  name is refused when the stage starts, before anything is spent. `infer` names only its inference,
  and `crossval` makes two names from one ([crossval](#crossval)). What a name may hold is in
  [The artifact store](getting-started.md#the-artifact-store).
- `--note TEXT` — free text recorded in the record's manifest; `''`, the default, means no note. It
  must be one line of at most 200 characters (`core.refusals.NOTE_MAX_CHARS`), checked once, before
  the command does anything else (`core.tool.main`): a longer or multi-line note is refused
  `(--note)`, never trimmed.

### Training-cache flags

Taken by `train`, `tsnpe` and `smoke`, the commands that simulate training rows. A training run
commits its rows to a simulation cache batch by batch (`train` and `tsnpe` every
`--checkpoint-every` batches, `smoke` only with `--checkpoint`). The cache's directory is named by
the run's identity: the prior's fit, the box, the observation mode, the batch count, the rows per
batch, the device and more (`core.artifacts.identity.SimulationIdentity`). A later run with the same
identity finds that cache; the flow's settings are not part of it.

- `--resume auto|require|never` — what to do with this run's own cache. `auto`, the default,
  resumes it when it holds committed batches and starts it otherwise. `require` resumes it, and
  refuses when there is none to resume, so a run you meant to continue cannot silently start again
  from zero; the refusal names the directory it looked in and any cache one setting away. `never`
  starts a new cache, and refuses `(--resume)` when this run's own cache already holds batches,
  rather than overwrite them. With checkpointing off there is no cache, so `require` and `never` are
  refused.
- `--new-run` — consent to start a new cache when a committed cache exactly one setting away
  exists. Without it that case is refused before the Fisher step and before any simulation, naming
  the setting and both values `(--new-run)`: a cache one setting away is almost always the same
  experiment with something nudged by accident. It silences that refusal and nothing else; it never
  forces a fresh start over this run's own cache. A cache two or more settings away is taken as a
  different experiment: the run starts its own cache with a warning that lists the other caches and
  the first setting each differs in.

A resume never simulates a committed batch again, and prints
`[checkpoint] resuming at batch <k>/<N> from <dir> (...)`. On `train` and `smoke`, a rotated run's
resume reuses the Fisher rotation stored with the cache instead of computing a new one, because the
rotation is not reproducible across processes and a new one would put the stored rows in another
coordinate; it prints `Reusing the Fisher rotation stored with the training checkpoint (...)` before
the resume line. A `tsnpe` round never computes a rotation, resumed or not: it prints
`[tsnpe] basis: reusing the PARENT posterior's rotation ...` instead.

### Acceptance flags

A posterior that a narrowing round (`tsnpe`) trained is valid only near the observation its region
was drawn around, so the tool refuses to load one, and refuses to run one on any other observation,
unless it is told to. These two flags are the only ways past those two refusals: `--accept-truncated`
answers the load refusal (`core.artifacts.store.ArtifactStore.load_posterior`), and
`--accept-other-observation` the other. They map one to one onto `core.artifacts.Accept`, and each
use is recorded in what the run writes ([The artifact store](getting-started.md#the-artifact-store)).

- `--accept-truncated` — load a non-amortized (narrowed) posterior. Taken by `validate`, `infer`,
  `tsnpe`, `sbc`, `identifiability rotation`, `identifiability laplace` and `ablation`, the commands
  that load a posterior.
- `--accept-other-observation` — `infer` only: run a narrowed posterior on an observation other
  than its region's. A simulated inference always needs it for a narrowed posterior: re-simulating a
  cell draws new noise, so the observation can never be the region's own.

## Environment variables

The tool reads no environment variable itself; every setting is a flag. PRISM reads four, and
`python -m core --help` names them:

| variable | what it sets | when it is read |
|---|---|---|
| `PRISM_RESOURCES` | the inputs root | see [Inputs and records](getting-started.md#inputs-and-records) |
| `PRISM_ARTIFACTS` | the records root | see [Inputs and records](getting-started.md#inputs-and-records); the tool resolves it once, when a command starts, except for `smoke`, and for `fdt` and `crossval` given `--store-root`, which use another root |
| `PRISM_VRAM_CEILING_GIB` | the GiB one simulation batch may plan to hold on the card; 0 or unset is automatic | afresh for every batch plan |
| `PRISM_MEM_LOG_EVERY` | the batches between `[mem]` memory lines, 250 when unset | once, when the simulation pipeline is first imported |

The last two are read by the simulation pipeline (`core.SBI.pipeline`), never by the tool, and are
deliberately not flags: they change how a batch is planned in memory, or how often memory is
reported, never the rows it produces. [The inference settings](window.md#the-inference-settings)
says what the ceiling buys and how to size it.

Two more are set rather than read as settings:

- `KMP_DUPLICATE_LIB_OK` is set to `TRUE` by the tool's entry module, `core.__main__`, before torch
  is imported, unless it is already set; [Launching PRISM](getting-started.md#launching-prism) says
  why.
- `PYTORCH_CUDA_ALLOC_CONF` is given an allocator policy by `core.config` when it is imported
  (`roundup_power2_divisions:8,garbage_collection_threshold:0.6`), which lets the card's memory be
  reused and reclaimed from batch to batch. A value you export wins.

## Exit codes

`core.tool.main` returns one of four codes, and `core.__main__` exits with it:

| code | meaning |
|---|---|
| `0` | success, or `--help` |
| `1` | a refusal, or a bug |
| `2` | a usage error |
| `130` | Ctrl-C |

- **0.** A command that finished. A result the command judged is not a failure: a calibration whose
  verdict is FAIL, an empty `artifacts list` and a probe check that recorded "no clear oscillation"
  all exit 0.
- **2, a usage error.** Most are argparse's own: an unknown subcommand, mode or flag, a missing
  required flag, a value that is not one of the choices or does not parse, an unknown
  `smoke --stages` entry. argparse prints a usage line, then
  `python -m core [<command>]: error: <message>`. A flag combination that only the built
  configuration can judge is a usage error too (`core.tool.config_args.UsageError`), printed as
  `prism <subcommand>: usage: <message>`: the recording rules of [infer](#infer), the resume rules
  of [smoke](#smoke), `fdt --skip-sanity` with `--no-production`, and a `crossval` grid whose point
  count is not a whole number of at least 2.
- **1, a refusal.** A refusal prints one line, `prism <subcommand>: refused: <message> (<flag>)`,
  ending with the flag that answers it, for example `(--n-cal)` (`core.tool.fields.FLAG`). A missing
  or blank `--bounds`, `--cell`, `--spont` or `--forced` file is such a refusal, and names its flag
  (`core.refusals.require_file`). A refusal that carries no flag ends at the message: a record
  reference that resolves to nothing (`--prior`, `--posterior`, `--observation`, or an `artifacts`
  mode's `<ref>`), `--resume require` with no cache to resume, the narrowed-parent rule of
  [tsnpe](#tsnpe), a missing units file (the tool has no units flag), and a rule two settings answer
  together. An error raised as a plain `ValueError`, such as an unsupported `--model`
  (`core.tool.config_args.make_cfg`) or the cache refusals of [ablation](#ablation), prints
  `prism <subcommand>: refused: <error class>: <message> [raised at <file>:<line>]`, and so does a
  missing file that no check covers. Anything else is a bug: the traceback, then
  `prism <subcommand>: *** FAILED ***`. An `artifacts sweep` that could not read or remove something
  exits 1 too.
- **130, Ctrl-C.** `prism <subcommand>: interrupted: ` and advice for what was cut short: the
  `interrupt_note` a family's parser sets, where it sets one (`core.tool.fdt`, `core.tool.probes`),
  else the generic advice.
  - `smoke`: for the case where a `[checkpoint]` line said batches were saved, the flags to re-run
    with to continue them ([smoke](#smoke), `core.tool._smoke_interrupt_advice`);
  - `fdt`, `crossval` and `compare`: the record is kept, marked unfinished, and nothing resumes
    ([fdt](#fdt));
  - the `probes` modes: a check writes its record only when it finishes, so nothing is left, and the
    same command starts it again;
  - every other command: the generic advice, that the record being written was removed and, if a
    `[checkpoint]` line said batches were saved, the same command with `--resume require` continues
    them. The `artifacts` family prints it too, although it fits none of its modes
    ([artifacts](#artifacts)).

## prior

Builds a stability-screened prior over the bounds file's parameters: a global census of the box, a
random-walk flood-fill of the stable region, then a Gaussian mixture fitted to the accepted sets,
with as many components as HDBSCAN finds clusters among them.

- `--bounds PATH` — required; see [Configuration flags](#configuration-flags).
- `--model NAME` — see [Configuration flags](#configuration-flags) (default the bounds file's parent folder, upper-cased).
- `--chi` / `--no-chi` — see [Configuration flags](#configuration-flags) (default --no-chi).
- `--chi-k K` — see [Configuration flags](#configuration-flags) (default 6).
- `--device auto|cpu|cuda` — see [Configuration flags](#configuration-flags) (default auto).
- `--name NAME` — see [Naming flags](#naming-flags).
- `--note TEXT` — see [Naming flags](#naming-flags).
- `--num-iterations N` — rounds of the global census; each round screens its candidates over the first half of the stability duration, so rounds cost time (default 50).
- `--sweep-batch N` — candidate parameter sets per round; 0 = automatic, the hardware batch (default 0).
- `--max-sets N` — accepted sets at which the flood-fill stops: the point cloud the prior is fitted to (default 175000).
- `--walk-step X` — the flood-fill's random-walk step, in the parameters' own units: one absolute size for every parameter, not a fraction of its range (default 0.01).
- `--stability-units X` — ND time units a candidate must stay bounded for (default 1000).
- `--min-cluster-size N` — HDBSCAN's minimum cluster size (default 50).
- `--min-samples N` — HDBSCAN's min_samples: the neighbours a point needs to count as a cluster core; larger declares more points noise (default 10).

Several of these change the prior rather than the time it takes:
[Settings that are not speed dials](window.md#settings-that-are-not-speed-dials).

**Writes** one prior record and prints `[prism] prior <name>__<id>  <path>`. `train --prior`,
`smoke --prior` and `probes mask --prior` load it by name or id.

**Refuses**, before the sweep starts, a taken name and any setting out of range, naming its flag: a
count below 1, a `--min-cluster-size` below 2, a negative `--sweep-batch`, a `--walk-step` or
`--stability-units` that is not positive.

**Exit codes** as in [Exit codes](#exit-codes).

## train

Trains an amortized posterior (neural posterior estimation) on a stored prior: it computes the
Fisher rotation when the configuration rotates (`core.config.REPARAM_ROTATE`, on), simulates the
training rows batch by batch, and fits the flow.

- `--bounds PATH` — required; see [Configuration flags](#configuration-flags).
- `--model NAME` — see [Configuration flags](#configuration-flags) (default the bounds file's parent folder, upper-cased).
- `--chi` / `--no-chi` — see [Configuration flags](#configuration-flags) (default --no-chi).
- `--chi-k K` — see [Configuration flags](#configuration-flags); in chi mode it sets the Fisher rotation's probe count only, and the training rows draw their own (default 6).
- `--device auto|cpu|cuda` — see [Configuration flags](#configuration-flags) (default auto).
- `--name NAME` — see [Naming flags](#naming-flags).
- `--note TEXT` — see [Naming flags](#naming-flags).
- `--prior REF` — required: the prior to train on, by name or id.
- `--num-runs N` — training batches to simulate; every row of a batch shares one (t_scale, T_obs) operating point, so this is also the training set's operating-point diversity (default 5000).
- `--run-size N` — ceiling on simulations per training batch; 0 = the hardware batch (default 0).
- `--hidden-features N` — flow width per transform (default 128).
- `--num-transforms N` — flow depth (default 8).
- `--learning-rate X` — Adam learning rate (default 0.001).
- `--stop-after-epochs N` — early-stopping patience, in epochs (default 20).
- `--max-epochs N` — hard ceiling on training epochs (default no ceiling).
- `--fisher-m N` — ensemble per latent perturbation for the Fisher rotation (default 48).
- `--fisher-dz X` — latent central-difference step of the Fisher rotation (default 0.1).
- `--fisher-points N` — operating points the Fisher rotation is averaged over (default 8).
- `--checkpoint-every N` — batches between commits of the simulation cache; 0 keeps no cache, so nothing can resume (default 50).
- `--resume auto|require|never` — see [Training-cache flags](#training-cache-flags) (default auto).
- `--new-run` — see [Training-cache flags](#training-cache-flags).

What each training setting really changes, and which are part of the cache's identity, is in
[The inference settings](window.md#the-inference-settings).

**Writes** a posterior record whose parents are the prior and, with checkpointing on, the simulation
cache, then prints `[prism] posterior <name>__<id>  <path>`. The cache itself is committed as the
run goes and keeps no `log.txt`. On the way it prints:

- `[budget] <N> batches x <R> rows = <T> training rows`, always, before anything is simulated;
- when it starts a new cache, `[checkpoint] writing to <dir> every <n> batches (...)`, with the
  cache's expected size and the free disk;
- on a run that resumes nothing, the Fisher rotation's lines
  (`Computing decorrelating Fisher rotation ...` and the `[fisher]` lines); on a resume, the resume
  lines of [Training-cache flags](#training-cache-flags) instead;
- in chi mode, `[chi] masked probes: ...`, the run total of masked probes over the batches its scope
  names;
- on a box that declares temperature in place of the force scale, `[tier1]` lines: the force scale
  it derives over the prior and, in chi mode, the chi drive amplitude that follows from it.

**Rules**, each checked before the Fisher step and before any simulation:

- the prior is loaded against `--bounds` and refused on a mismatch in model, ND parameter set or
  order, ND box or log-box mask;
- every setting above is refused out of range, naming its flag;
- a committed cache one setting away is refused unless `--new-run` is given, and one two or more
  settings away gives a warning only ([Training-cache flags](#training-cache-flags));
- with `--resume require`, a run with no cache to resume is refused, and with
  `--checkpoint-every 0` both `--resume require` and `--resume never` are refused, because no cache
  is read or written;
- a resume reuses the cache's stored rotation, so `--fisher-m`, `--fisher-dz` and `--fisher-points`
  have no effect on it, and the record writes them as null.

**Exit codes** as in [Exit codes](#exit-codes). After Ctrl-C, if a `[checkpoint]` line said
batches were saved (`[checkpoint] stopping: saving ...` is the one an interrupt prints), the same
command with `--resume require` continues from the last saved batch; with checkpointing off, nothing
simulated is kept.

## validate

Calibrates a posterior without an observation: a calibration set drawn from the posterior's own
prior and simulated, then the rank-uniformity test (SBC) for every parameter and the joint coverage
test (TARP), ending on one verdict.

- `--bounds PATH` — required; see [Configuration flags](#configuration-flags).
- `--model NAME` — see [Configuration flags](#configuration-flags) (default the bounds file's parent folder, upper-cased).
- `--chi` / `--no-chi` — see [Configuration flags](#configuration-flags) (default --no-chi).
- `--chi-k K` — see [Configuration flags](#configuration-flags) (default 6).
- `--device auto|cpu|cuda` — see [Configuration flags](#configuration-flags) (default auto).
- `--name NAME` — see [Naming flags](#naming-flags).
- `--note TEXT` — see [Naming flags](#naming-flags).
- `--posterior REF` — required: the posterior to calibrate, by name or id; its own training prior is loaded with it.
- `--n-cal N` — calibration datasets to simulate (default 2000).
- `--cal-n-scales N` — (t_scale, T_obs) operating points the calibration set is spread over: t_scale's effective sample size, not a speed dial (default 200).
- `--posterior-samples N` — posterior draws per calibration dataset (default 1000).
- `--seed N` — the calibration set's random seed, a whole number from 0 up to 2^64 − 1 (default none: one is drawn and recorded).
- `--accept-truncated` — see [Acceptance flags](#acceptance-flags).

**Writes** a calibration record whose parents are the posterior and its prior. After the `[cfg]`
lines it prints `[prism] prior: <ref> (the posterior's training prior)`, then the rank tests
(`SBC uniformity checks:`), the coverage test (`TARP: ...`) and the informativeness estimate, then
the `[verdict]` record, and last `[prism] calibration <name>__<id>  <path>`. For a narrowed
posterior the calibration set is drawn from the prior restricted to its region, and the fraction of
prior draws the region kept is printed.

- The verdict is PASS when every parameter's rank test reaches KS p ≥ 0.05 divided by the number of
  parameters and the joint coverage test reaches KS p ≥ 0.05. The `[verdict]` record gives the rule,
  the seed, one row per parameter and the coverage row; the record keeps the same in
  `results.verdict`. A FAIL is a result: the record is written the same way and the command exits 0.
  [Reading calibration honestly](science.md#reading-calibration-honestly) says what a verdict can
  and cannot show.
- The record keeps the seed in `results.seed` and the acceptances the posterior was loaded under in
  `results.accepted`.
- With `--seed`, a calibration set repeats: on one device the same seed draws the same set and
  reaches the same verdict, bit for bit on the CPU and not bitwise on a CUDA card, where a verdict at
  its threshold can differ. The stream never replays the one a training run with that seed used.
  [For a reviewer](README.md#for-a-reviewer) gives the procedure for repeating a calibration from
  its record.
- The prior is always the posterior's own, read from its manifest, so `validate` takes no `--prior`.

**Refuses** a taken name; a count below 1; a seed outside its range `(--seed)`; a posterior that
does not match `--bounds`; and a narrowed posterior without `--accept-truncated`.

**Exit codes** as in [Exit codes](#exit-codes); a FAIL verdict exits 0.

## infer

Runs a posterior on one observation: simulated from a cell's ground truth (`--cell`), or built from
measured recordings (`--spont` and its companions).

- `--bounds PATH` — required; see [Configuration flags](#configuration-flags).
- `--model NAME` — see [Configuration flags](#configuration-flags) (default the bounds file's parent folder, upper-cased).
- `--chi` / `--no-chi` — see [Configuration flags](#configuration-flags) (default --no-chi).
- `--chi-k K` — see [Configuration flags](#configuration-flags); in chi mode, the probes a simulated cell is measured at, while recordings bring their own count, one per `--forced` (default 6).
- `--device auto|cpu|cuda` — see [Configuration flags](#configuration-flags) (default auto).
- `--name NAME` — names the inference only; see [Naming flags](#naming-flags).
- `--note TEXT` — see [Naming flags](#naming-flags).
- `--posterior REF` — required: the posterior to run, by name or id; its own training prior is loaded with it.
- `--cell PATH` — simulated: the cell file whose ground truth is re-simulated at `--t-obs`. Exactly one of `--cell` and `--spont` is required.
- `--spont PATH` — experimental: the passive (unforced) recording.
- `--t-obs S` — required: the observation length in seconds, the recordings' length or the length the cell is re-simulated at.
- `--forced PATH[@HZ]` — experimental: a driven recording, and in chi mode the frequency in Hz it was driven at; repeatable.
- `--drive NAME=VALUE` — forced mode: the drive the recording was made at, in SI units; once per forcing parameter the bounds file declares.
- `--f0-si X` — chi mode: the physical drive amplitude every probe was driven at, in newtons.
- `--n-samples N` — posterior draws for the corner plot, the posterior predictive check and the summary (default 1000).
- `--accept-truncated` — see [Acceptance flags](#acceptance-flags).
- `--accept-other-observation` — see [Acceptance flags](#acceptance-flags).

**Writes** two records, an observation and an inference, and prints one line for each:
`[prism] observation _unnamed__<id>  <path>`, then `[prism] inference <name>__<id>  <path>`. The
observation is never named on this path, so refer to it by the `<id>` part of its line, as
`tsnpe --observation` does; `python -m core artifacts list observation` lists it too. Pre-flight
warnings go to stderr: a recording length outside the training range, a cell value the bounds file
does not declare (ignored), a simulated truth outside the training distribution, and a truth outside
a narrowed posterior's region.

**Rules** for the recordings, set by the observation mode the bounds file and `--chi` give
([Observation modes](recordings.md#observation-modes)). They are checked before the posterior loads,
and each is a usage error. The three mode rules are `core.tool.config_args.recording_set`'s:

- chi mode: at least one `--forced PATH@HZ`, every one with the frequency it was driven at, plus
  `--f0-si`; `--drive` is not taken. The path is split at its last `@`.
- spontaneous mode: `--spont` alone; `--forced`, `--drive` and `--f0-si` are not taken.
- forced mode: exactly one `--forced PATH` without `@HZ`, plus `--drive NAME=VALUE` naming exactly
  the forcing parameters the bounds file declares; `--f0-si` is not taken.

The fourth is `core.tool.stages._infer`'s: `--cell` excludes `--forced`, `--drive` and `--f0-si`,
which describe measured recordings. A simulated cell brings its own drive: in forced mode it is
re-simulated with the drive its cell file states, and in chi mode it is probed at the configured
multiples of its own peak frequency, and the drive in its cell file is ignored.

[Recording files and drive frequencies](recordings.md#recording-files-and-drive-frequencies) and
[Recording length](recordings.md#recording-length) say what the recordings must be.

**Refuses** a taken name, a `--t-obs` of 0 or less, an `--n-samples` below 1, a posterior that does
not match `--bounds`, a narrowed posterior without `--accept-truncated`, and a narrowed posterior run
on any observation but its region's own without `--accept-other-observation`, which a simulated cell
always is.

**Exit codes** as in [Exit codes](#exit-codes).

## tsnpe

One narrowing round (TSNPE): it draws a region around an observation along a parent posterior's
Fisher directions, then trains a new posterior on the parent's prior restricted to that region. The
result is valid only near that observation.

- `--bounds PATH` — required; see [Configuration flags](#configuration-flags).
- `--model NAME` — see [Configuration flags](#configuration-flags) (default the bounds file's parent folder, upper-cased).
- `--chi` / `--no-chi` — see [Configuration flags](#configuration-flags) (default --no-chi).
- `--chi-k K` — see [Configuration flags](#configuration-flags); in chi mode the observation's own probe count replaces it (default 6).
- `--device auto|cpu|cuda` — see [Configuration flags](#configuration-flags) (default auto).
- `--name NAME` — see [Naming flags](#naming-flags).
- `--note TEXT` — see [Naming flags](#naming-flags).
- `--posterior REF` — required: the parent posterior the region is measured in, by name or id.
- `--observation REF` — required: the observation the region is drawn around, by name or id; [infer](#infer) prints its id.
- `--directions N` — Fisher directions to truncate; the flat ones keep the full prior width (default 5).
- `--level X` — HPD level of the region along each truncated direction; below 0.99 warns, because truncation permanently deletes prior support (default 0.999).
- `--num-runs N` — training batches to simulate for this round (default 5000).
- `--run-size N` — ceiling on simulations per training batch; 0 = the hardware batch (default 0).
- `--hidden-features N` — flow width per transform (default 128).
- `--num-transforms N` — flow depth (default 8).
- `--learning-rate X` — Adam learning rate (default 0.001).
- `--stop-after-epochs N` — early-stopping patience, in epochs (default 20).
- `--max-epochs N` — hard ceiling on training epochs (default no ceiling).
- `--checkpoint-every N` — batches between commits of the round's simulation cache; 0 keeps no cache (default 50).
- `--resume auto|require|never` — see [Training-cache flags](#training-cache-flags) (default auto).
- `--new-run` — see [Training-cache flags](#training-cache-flags).
- `--accept-truncated` — see [Acceptance flags](#acceptance-flags); needed when the parent is itself narrowed.

**Writes** a non-amortized posterior (its manifest says `amortized: false`) whose parents are the
prior, the parent posterior, the observation and, with checkpointing on, the round's own simulation
cache. The region is part of the cache's identity, so a round never resumes an amortized run's rows.
It prints `[tsnpe] region from observation <ref>: TruncationRegion(...)`, the basis line, the truth
lines below, the fraction of prior draws the region accepted and how many training targets lie
inside it, then `[prism] posterior <name>__<id>  <path>`.

**Rules:**

- A round takes no Fisher flag. It reuses the parent's rotation, carried with the region, and never
  computes one: the region is measured in the parent's basis and means nothing in another. Its
  record carries the parent's eigenvalues.
- A round does not inherit its parent's network size or batch count: `--hidden-features`,
  `--num-transforms` and `--num-runs` fall back to the defaults above, so pass them to match a parent
  trained at another size.
- A `--level` below 0.99 raises a pre-flight warning.
- A narrowed parent drawn around another observation is refused, and no flag overrides it: a region
  drawn from it around another observation would sit where its flow extrapolates. Start from an
  amortized posterior, or draw the round around the parent's own observation.
- Truth containment: when the observation was simulated from a cell, the round logs, for every
  truncated direction, whether the cell's truth lies inside or outside the region, with the
  direction's interval, and records the answers in the new posterior's `training.truth_containment`.
  A truth outside also raises a pre-flight warning. A recording has no truth, and nothing is said.
- The observation is checked against the configuration's mode and conditioning width when it loads,
  and the cache rules are those of [train](#train).

**Refuses**, before any simulation: a taken name; a `--directions` below 1 or above the posterior's
latent width `(--directions)`; a `--level` not strictly between 0 and 1 `(--level)`; a training
setting out of range, naming its flag, as [train](#train) does; a parent posterior or an observation
that does not match `--bounds`; and a narrowed parent without `--accept-truncated`.

**Exit codes** as in [Exit codes](#exit-codes).

## sbc

The rank-uniformity test (SBC) repeated on one posterior, each repeat on a fresh calibration set, to
measure how far its KS p-values vary from run to run. A single flat SBC can be a lucky draw;
repeats tell sampling noise from a real miscalibration.

- `--bounds PATH` — required; see [Configuration flags](#configuration-flags).
- `--model NAME` — see [Configuration flags](#configuration-flags) (default the bounds file's parent folder, upper-cased).
- `--chi` / `--no-chi` — see [Configuration flags](#configuration-flags) (default --no-chi).
- `--chi-k K` — see [Configuration flags](#configuration-flags) (default 6).
- `--device auto|cpu|cuda` — see [Configuration flags](#configuration-flags) (default auto).
- `--name NAME` — see [Naming flags](#naming-flags).
- `--note TEXT` — see [Naming flags](#naming-flags).
- `--posterior REF` — required: the posterior to calibrate repeatedly, by name or id.
- `--repeats N` — independent SBC runs (default 10).
- `--n-cal N` — calibration datasets to simulate per repeat (default 2000).
- `--posterior-samples N` — posterior draws per calibration dataset (default 1000).
- `--cal-n-scales N` — (t_scale, T_obs) operating points each calibration set is spread over (default 200).
- `--chi-k-fixed K` — hold the chi probe count at K, from 1 to the probe slots, instead of pooling over the training mixture; chi mode only (default pooled over the training mixture).
- `--seed N` — the random seed for the whole run; repeat r runs at seed + r (default 0).
- `--accept-truncated` — see [Acceptance flags](#acceptance-flags).

**Writes** a diagnostic record whose parents are the posterior and its prior, its variant `pooled`,
or `k<K>` for a fixed probe count, with the pooled rank histogram and every repeat's p-values. It
prints `[sbc] repeat <r>/<R>: n_valid=...  worst KS p=...` per repeat, then each parameter's p-value
distribution over the repeats (median, minimum, fraction below 0.05), then the rank-uniformity half
of the calibration verdict: a repeat passes when every parameter's KS p reaches 0.05 divided by the
number of parameters. The record keeps each repeat's answer and the fraction that passed in
`results.rank_verdict`. No joint coverage test runs here: that half of the verdict is
[validate](#validate)'s.

- The stratified protocol: run once with `--chi-k-fixed` at each probe count of interest and once
  pooled, then compare. A pooled SBC over a mixture of counts can be flat while each count is
  miscalibrated in compensating directions.
- For a narrowed posterior the repeats share one restricted prior, so `--repeats 1 --seed r` does
  not reproduce repeat r of a longer run.

**Refuses** a taken name, a count below 1, a `--chi-k-fixed` outside chi mode or outside its range
`(--chi-k-fixed)`, a posterior that does not match `--bounds`, and a narrowed posterior without
`--accept-truncated`.

**Exit codes** as in [Exit codes](#exit-codes).

## identifiability

What a posterior measured (`rotation`), or whether the information is there at all (`laplace`,
`jacobian`). Each mode is a parser of its own, so a flag that belongs to another mode, such as
`--posterior` on `jacobian` or `--zero-tol` on `laplace`, is a usage error rather than a setting
silently ignored. The configuration flags are shared by all three, though, so `--chi-k` is accepted
and ignored where a mode has no use for it: by `rotation`, and by `laplace`, which runs only without
chi. Each mode writes a diagnostic record whose variant is the mode's name, and prints
`[prism] diagnostic <name>__<id>  <path>`.

`laplace` and `jacobian` simulate at the cell's ground truth, so both need `--cell` and `--t-obs`,
and their configuration loads the cell's values: the `[cfg]` lines name any cell value the bounds
file does not declare, which is ignored. On a box that declares temperature in place of the force
scale, both derive the force scale from the temperature as training does
(`core.SBI.derived.for_simulation`).

### identifiability rotation

Decomposes a trained posterior's Fisher eigenbasis. It reads the record and simulates nothing.

- `--bounds PATH` — required; see [Configuration flags](#configuration-flags).
- `--model NAME` — see [Configuration flags](#configuration-flags) (default the bounds file's parent folder, upper-cased).
- `--chi` / `--no-chi` — see [Configuration flags](#configuration-flags) (default --no-chi).
- `--chi-k K` — see [Configuration flags](#configuration-flags) (default 6).
- `--device auto|cpu|cuda` — see [Configuration flags](#configuration-flags) (default auto).
- `--name NAME` — see [Naming flags](#naming-flags).
- `--note TEXT` — see [Naming flags](#naming-flags).
- `--posterior REF` — required: the posterior whose Fisher eigenbasis is decomposed, by name or id.
- `--n-worst N` — worst directions totalled per parameter (default 3).
- `--top-n N` — loadings shown per direction (default 4).
- `--accept-truncated` — see [Acceptance flags](#acceptance-flags).

**Writes** a record whose parent is the posterior. It prints the eigenvalue spread and the
participation ratio, the directions best-constrained first with their largest loadings, each
parameter's share of its identifiability in the worst directions with a verdict (good, moderate,
poor, very poor or UNMEASURED), and the parameters that are their own near-null direction.

When the posterior's record holds no eigenvalues, the mode warns
`[eigenvalues] NOT STORED for this artifact.`, and what follows is an ordering and a set of
loadings without the scale. The eigenvalues go unrecorded when training resumed a cache written
before caches kept them, and on a narrowing round trained before rounds inherited them, built with
no parent posterior, or whose parent held none. The warning names no record, but says which records
may hold the same rotation with its eigenvalues: the amortized posterior a narrowing round descends
from, and, for a resumed run, the posterior of the run that started its cache, if that run computed
the rotation and finished.

**Refuses** a posterior that records no rotation `(--posterior)`, and an `--n-worst` above the
posterior's latent width `(--n-worst)`.

**Exit codes** as in [Exit codes](#exit-codes).

### identifiability laplace

The Laplace marginal standard deviation of every parameter at the cell's ground truth and at draws
from the posterior's own training prior: whether the data could pin each parameter down there,
whatever the posterior learned.

- `--bounds PATH` — required; see [Configuration flags](#configuration-flags). It must declare a Forcing section.
- `--model NAME` — see [Configuration flags](#configuration-flags) (default the bounds file's parent folder, upper-cased).
- `--chi` / `--no-chi` — see [Configuration flags](#configuration-flags); this mode refuses chi mode (default --no-chi).
- `--chi-k K` — see [Configuration flags](#configuration-flags) (default 6).
- `--device auto|cpu|cuda` — see [Configuration flags](#configuration-flags) (default auto).
- `--name NAME` — see [Naming flags](#naming-flags).
- `--note TEXT` — see [Naming flags](#naming-flags).
- `--posterior REF` — required: the posterior whose training prior supplies the evaluation points beyond the ground truth, by name or id.
- `--cell PATH` — required: the cell file whose ground truth is the first evaluation point.
- `--t-obs S` — required: the observation length, in seconds, the metric is measured at.
- `--n-points N` — evaluation points including the ground truth (default 6).
- `--m N` — ensemble per perturbation arm (default 32).
- `--m-noise N` — ensemble for the single-trajectory feature-noise floor, at least 10 (default 128).
- `--rel X` — perturbation as a fraction of the prior range (default 0.02).
- `--min-valid X` — valid-member fraction an arm needs before it is used (default 0.5).
- `--seed N` — the random seed for the whole run (default 0).
- `--sd-identified X` — an SD below this counts as identified (default 0.3).
- `--accept-truncated` — see [Acceptance flags](#acceptance-flags).

**Writes** a record whose parent is the posterior. For each evaluation point it prints how many
parameters were measurable there, then a table with one row per parameter: its SD at every point,
the median, and the fraction of points where it falls below `--sd-identified`. A rescale parameter
whose name contains `scale` is log-uniform in the prior, so its SD is in log-range units; every
other SD is in range units, and the record names which is which.

**Rules:** the mode measures the single-frequency feature set and reads the cell's own drive, so it
runs in forced mode only: chi mode is refused, and so is a bounds file with no Forcing section. It
also refuses a setting out of range, naming its flag, and a `--t-obs` too short to give a single
sample `(--t-obs)`.

**Exit codes** as in [Exit codes](#exit-codes).

### identifiability jacobian

The degeneracy map at the cell's ground truth, over the mode's own feature set: how strongly each
parameter moves the features the posterior conditions on, and which parameters move them alike. It
needs no posterior.

- `--bounds PATH` — required; see [Configuration flags](#configuration-flags).
- `--model NAME` — see [Configuration flags](#configuration-flags); this mode takes the Nadrowski model only (default the bounds file's parent folder, upper-cased).
- `--chi` / `--no-chi` — see [Configuration flags](#configuration-flags) (default --no-chi).
- `--chi-k K` — see [Configuration flags](#configuration-flags); in chi mode, the probes the map is measured at (default 6).
- `--device auto|cpu|cuda` — see [Configuration flags](#configuration-flags) (default auto).
- `--name NAME` — see [Naming flags](#naming-flags).
- `--note TEXT` — see [Naming flags](#naming-flags).
- `--cell PATH` — required: the cell file whose ground truth the map is measured at.
- `--t-obs S` — required: the observation length, in seconds, the map is measured at; compare only maps made at the same length.
- `--m N` — ensemble per perturbation arm (default 32).
- `--m-noise N` — ensemble for the single-trajectory feature-noise floor, at least 10 (default 128).
- `--rel X` — perturbation as a fraction of the prior range (default 0.02).
- `--min-valid X` — valid-member fraction an arm needs before it is used (default 0.5).
- `--seed N` — the random seed for the whole run (default 0).
- `--zero-tol X` — a parameter whose sensitivity, in units of the feature noise, has a norm below this carries no local information (default 0.05).
- `--noise-eps X` — a feature channel whose spread across the ensemble is below this fraction of its size is treated as dead and zeroed in the map (default 1e-06).

**Writes** a record with no parent. In chi mode it first prints the probe budget: a line
`=== probe budget: T_obs=... = <s> s, Omega_0=... Hz ... ===`, one row per probe with its drive
cycles, marked `DRIFT` under the 2-cycle floor and `PINNED` above the lock-in ceiling, then the
recording-length line, which gives the shortest recording at which the band's low edge clears the
floor and the longest at which its high edge stays under the ceiling:
`low edge <m>x clears 2 cycles at T_obs >= <s> s; high edge <m>x stays under the <n>-cycle ceiling below T_obs = <s> s.`
A dead feature channel is warned and zeroed.

**Rules:**

- It is refused for every model but Nadrowski, because the parameter names it prints are
  Nadrowski's (`core.diagnostics.feature_sets.assert_nadrowski`).
- Without `--chi` the mode reads the cell's own drive, so it needs a bounds file with a Forcing
  section; spontaneous mode is refused.
- It refuses a setting out of range, naming its flag, and a `--t-obs` too short to give a single
  sample `(--t-obs)`.
- Two failures can come after the simulation has begun, and leave no record: fewer than 10 finite
  baseline runs at the truth is a refusal, and in chi mode a probe measurement that is not finite,
  not positive, or at or above 0.9 times Nyquist is an error, where [probes band](#probes-band)
  would mask that probe instead.

**Exit codes** as in [Exit codes](#exit-codes).

## ablation

Which conditioning channels the trained flow can see. It sweeps each summary channel across its
real range, from the 1st to the 99th percentile of the rows the posterior was trained on, and
measures how far the flow's embedding moves; it reads the records and simulates nothing.

- `--bounds PATH` — required; see [Configuration flags](#configuration-flags).
- `--model NAME` — see [Configuration flags](#configuration-flags) (default the bounds file's parent folder, upper-cased).
- `--chi` / `--no-chi` — see [Configuration flags](#configuration-flags) (default --no-chi).
- `--chi-k K` — see [Configuration flags](#configuration-flags) (default 6).
- `--device auto|cpu|cuda` — see [Configuration flags](#configuration-flags) (default auto).
- `--name NAME` — see [Naming flags](#naming-flags).
- `--note TEXT` — see [Naming flags](#naming-flags).
- `--posterior REF` — required: the posterior whose flow is probed, by name or id.
- `--rows N` — leading rows read from the posterior's own simulation cache to set each channel's range (default 200000).
- `--n-sweep N` — points per channel sweep, at least 2 (default 33).
- `--accept-truncated` — see [Acceptance flags](#acceptance-flags).

**Writes** a diagnostic record whose parents are the posterior and its simulation cache: one row per
summary channel with its displacement and a verdict (healthy, compressed, severely compressed,
invisible, constant in training, or non-finite), and a count of the usable channels.

**Rules:** the ranges must be the ones this network was trained on, so the mode reads the
posterior's own simulation cache: the posterior must have trained with checkpointing on. One that
names no cache is refused, as is a cache whose rows are not as wide as the posterior's conditioning.

**Exit codes** as in [Exit codes](#exit-codes).

## probes

Checks of the chi probe settings on a lab's own cell or prior: whether the configured band and drive
hold for a cell ([probes band](#probes-band)), why training throws probes out over a prior
([probes mask](#probes-mask)), and how hard a lab can drive a cell ([probes drive](#probes-drive)).

- The checks measure and never change a setting. The lengths, frequencies, drive strengths and
  lock-in ceilings a check takes are grids of its own, never settings: none reaches a training run's
  configuration. Every record states the configured band, drive, probe slots, cycle floor, cycle
  ceiling and probe count it judged, and a deliberate change to any of them is made in
  `core/config.py`.
- `band` and `mask` build their configuration in chi mode themselves, and `drive` builds it with chi
  mode off, so no mode takes `--chi`, `--no-chi` or `--chi-k`. Each takes `--bounds`, `--model` and
  `--device` from the [Configuration flags](#configuration-flags).
- Each mode writes one diagnostic record, its variant the mode's name, and prints
  `[prism] diagnostic <name>__<id>  <path>`.
- A finding is a result, not a refusal: a probe at or above 0.9 times the sampling limit (Nyquist)
  is masked and reported, "no clear oscillation" and "nothing captured" are recorded, and the command
  exits 0.
- A check writes its record only when it finishes, so Ctrl-C leaves nothing, and the same command
  starts it again.
- Flags must be written in full on these modes.

### probes band

Do the configured chi band and drive hold for this cell? At every recording length, probe frequency
and drive strength, an ensemble of the cell's own noisy runs is driven and locked in on, at full
length and under the configured lock-in ceiling. Four criteria are judged: the spread of |chi| over
the runs, the phase scatter, the signal over the same lock-in on the undriven runs, and capture, the
share of the cell's own peak power that survives the drive. Only capture is physical evidence; the
other thresholds are conventions.

- `--bounds PATH` — required; see [Configuration flags](#configuration-flags).
- `--model NAME` — see [Configuration flags](#configuration-flags) (default the bounds file's parent folder, upper-cased).
- `--device auto|cpu|cuda` — see [Configuration flags](#configuration-flags) (default auto).
- `--name NAME` — see [Naming flags](#naming-flags).
- `--note TEXT` — see [Naming flags](#naming-flags).
- `--cell PATH` — required: the cell file whose ground truth the probes are measured on.
- `--lengths S [S ...]` — recording lengths, in seconds, each measured on an ensemble of its own; one beyond the cell's training ceiling is measured and marked (default five log-spaced from the shortest expected recording to this cell's training ceiling).
- `--multipliers X [X ...]` — probe frequencies, as multiples of each run's own peak frequency (default four across the configured band plus one control at twice its top).
- `--drives X [X ...]` — non-dimensional drive amplitudes; a probe frequency inside the band that is captured at any of them fails the band verdict, while the drive verdict judges the configured drive alone (default the configured chi drive).
- `--repeats N` — noise repeats per point, the ensemble every measure is taken over (default 24).
- `--cycle-caps N [N ...]` — lock-in ceilings, in drive cycles, re-measured to bracket where a longer lock-in stops helping; the configured ceiling is always among them (default the configured cycle ceiling only).
- `--cv-max X` — largest spread of |chi| over the runs, as a fraction of its mean, that passes (default 0.2).
- `--phase-max X` — largest circular spread of the phase of chi, in radians, that passes; without it the phase is reported and not judged (default not judged).
- `--snr-min X` — smallest mean |chi| driven, over the same lock-in on the undriven runs, that passes (default 3.0).
- `--sup-min X` — smallest share of the cell's own peak power that must survive the drive, in (0, 1]; below it the probe captured the cell (default 0.5).
- `--peak-window X` — half-width of the own-peak window, as a fraction of the peak frequency, in (0, 1); never narrower than two frequency bins either side (default 0.1).
- `--seed N` — the random seed for the whole run (default 0).

**Writes** a record with no parent: every point's four measures and verdict, the band verdict, the
drive verdict and the configured settings it judged. A probe at or above 0.9 times Nyquist is
masked, never moved to a frequency the cell can be sampled at, and the check warns which points it
masked. A length longer than the pre-simulated time grid holds is measured over the part that fits,
with a warning that gives the length measured; [probes drive](#probes-drive) refuses such a length
instead.

**Refuses** a setting out of range, naming its flag, before anything is simulated.

**Exit codes** as in [Exit codes](#exit-codes).

### probes mask

Why are training probes thrown out, over a prior? It runs a few batches of the training generator
itself over the prior, at the configured band and drive, and splits every masked probe by cause:
too slow for the cycle floor even at the band's top over the full recording, shortened below the
floor by the duration draw, or a lock-in that came back non-finite or zero. Each row range's count
is checked against the generator's own masked-probe warning, and every batch and row asked for must
reach the audit.

- `--bounds PATH` — required: the bounds file the prior was built with; the store refuses a prior whose model, parameter order, box or log-box mask differs. See [Configuration flags](#configuration-flags).
- `--model NAME` — see [Configuration flags](#configuration-flags) (default the bounds file's parent folder, upper-cased).
- `--device auto|cpu|cuda` — see [Configuration flags](#configuration-flags) (default auto).
- `--name NAME` — see [Naming flags](#naming-flags).
- `--note TEXT` — see [Naming flags](#naming-flags).
- `--prior REF` — required: the prior to audit, by name or id; it is loaded, never built.
- `--num-runs N` — training batches to simulate and audit; their count is the effective sample size of the per-batch masked fraction (default 12).
- `--run-size N` — rows per audited batch (default 32).
- `--chi-k-fixed K` — audit one chi probe count, from 2 to the number of probe slots, instead of training's own mixture of counts (default pooled over the training mixture).
- `--seed N` — the random seed for the whole run; the probe layout keeps training's own fixed seed (default 0).

**Writes** a record whose parent is the prior. No simulation cache is written.

**Refuses** a prior that does not match `--bounds`, and a setting out of range, naming its flag.

**Exit codes** as in [Exit codes](#exit-codes).

### probes drive

How hard can a lab drive this cell? The cell's own noisy runs are simulated undriven, which gives
the peak frequency, the peak's clarity (the peak bin's power over the median power) and the cycles
in the recording. Then, one ensemble per strength, they are driven at the detune times the peak
frequency, and each strength is judged by the share of the undriven own-peak power the driven runs
keep inside the own-peak window: free-running at the free-running threshold or more, captured at the
captured threshold or less, in between otherwise.

- `--bounds PATH` — required: it must declare a Forcing section, the section this check suggests values for. See [Configuration flags](#configuration-flags).
- `--model NAME` — see [Configuration flags](#configuration-flags) (default the bounds file's parent folder, upper-cased).
- `--device auto|cpu|cuda` — see [Configuration flags](#configuration-flags) (default auto).
- `--name NAME` — see [Naming flags](#naming-flags).
- `--note TEXT` — see [Naming flags](#naming-flags).
- `--cell PATH` — required: the cell file whose ground truth is driven.
- `--t-obs S` — recording length, in seconds, of the undriven runs and of every driven ensemble; one longer than the pre-simulated time grid holds is refused (default 5.0).
- `--repeats N` — noise repeats per ensemble, undriven and at each strength; the fewer the runs, the higher the clarity pure noise reaches, and the record states that level (default 16).
- `--detune X` — the drive frequency, as a multiple of the cell's own peak frequency; it must differ from 1 by more than the own-peak window (default 1.4).
- `--strengths X [X ...]` — non-dimensional drive strengths, increasing; each drives at that multiple of the cell's force scale (default sixteen from 0.01 to 20, plus the configured chi drive).
- `--free-min X` — smallest share of the undriven own-peak power the driven runs keep that counts as free-running, in (0, 1] (default 0.7).
- `--captured-max X` — largest share of the undriven own-peak power the driven runs keep that counts as captured, in (0, 1) and below the free-running threshold (default 0.1).
- `--peak-window X` — half-width of the own-peak window, as a fraction of the peak frequency, in (0, 1); never narrower than two frequency bins either side (default 0.02).
- `--clarity-min X` — smallest peak clarity that counts as an oscillation; below it no strength is judged, and a threshold that does not clear the clarity pure noise reaches is warned (default 3.0).
- `--seed N` — the random seed for the whole run (default 0).

**Writes** a record with no parent. It names the strongest free-running strength, the top of the
unbroken run of free-running strengths from the weakest up, and the weakest captured one, in model
units and in the cell's force unit. It logs suggested Forcing lines for a cell file, and records
them; it never writes them to any file. The phase locking is reported, never judged, and there is
no linearity test. The default strengths include the configured chi drive, but that row cannot
certify it, because its verdict depends on the detune: [probes band](#probes-band) judges the chi
drive.

Recorded with nothing judged, and exit 0: no clear oscillation (a clarity below the threshold), a
drive at or above 0.9 times Nyquist, or a drive within a frequency bin of the own-peak window.
Warned while the strengths are still judged: a clarity threshold that does not clear the clarity
pure noise reaches with this many runs and samples, fewer than 30 cycles of the peak in the
recording, and a drive whose second to fifth harmonic lands within a bin of the window. When no
strength in the grid captures the cell, the check says so, and that too is a result.

**Refuses**, before anything is simulated, a bounds file with no Forcing section (the check states
each strength as a force in the cell's unit, so it needs one); a `--t-obs` longer than the
pre-simulated time grid holds `(--t-obs)`; a `--detune` within the own-peak window of 1
`(--detune)`; strengths that do not increase; a captured threshold not below the free-running one;
and any other setting out of range, naming its flag.

**Exit codes** as in [Exit codes](#exit-codes).

## smoke

Every stage end to end at tiny sizes, on the real bounds and cell files: it builds a prior, trains a
posterior, calibrates it and runs an inference on a simulated cell. Run it on the graphics card
after changing code that moves tensors; [The card smoke gate](testing.md#the-card-smoke-gate) gives
the command lines and what each must print.

- `--bounds PATH` — required; see [Configuration flags](#configuration-flags).
- `--model NAME` — see [Configuration flags](#configuration-flags) (default the bounds file's parent folder, upper-cased).
- `--chi` / `--no-chi` — see [Configuration flags](#configuration-flags) (default --no-chi).
- `--chi-k K` — see [Configuration flags](#configuration-flags) (default 6).
- `--device auto|cpu|cuda` — see [Configuration flags](#configuration-flags) (default auto).
- `--cell PATH` — required: the cell file whose ground truth the infer stage simulates; when that stage runs, the cell is checked against `--bounds` before the prior is built.
- `--t-obs S` — observation length, in seconds, of the observation the infer stage simulates; in chi mode give it a few seconds, as [A first run on the command line](getting-started.md#a-first-run-on-the-command-line) explains (default 1.0).
- `--seed N` — the random seed for the whole run: one seeded stream through every stage (default 0).
- `--stages STAGE[,STAGE...]` — the stages to run, a comma-separated subset of prior,posterior,validate,infer. A stage left out prints `[skip] <stage>`; the prior stage is also where `--prior` is loaded, so a run without `prior` stops there and a run without `posterior` stops after the prior, both with exit 0 (default prior,posterior,validate,infer).
- `--num-runs N` — training batches to simulate (default 4).
- `--run-size N` — ceiling on simulations per training batch; the prior sweep keeps the hardware batch (default 32).
- `--n-cal N` — calibration datasets to simulate (default 40).
- `--max-epochs N` — hard ceiling on training epochs (default 5).
- `--hidden-features N` — flow width per transform (default 128).
- `--num-transforms N` — flow depth (default 8).
- `--checkpoint` — keep a simulation cache, committed every max(1, num_runs // 2) batches. Off unless given, so a second run of one configuration simulates again, which is the path this command exists to exercise.
- `--save` — name what this run builds `smoke_prior` and `smoke_posterior`; a second `--save` run against the same store is refused by name.
- `--store-root PATH` — the store this run writes; reuse one, with `--prior` and `--checkpoint`, to resume (default a fresh temporary directory, left on disk).
- `--prior REF` — a prior in the store to load instead of building, by name or id, loaded by the prior stage; a resume needs it, because the cache is keyed on the prior's fit and two fits of one box differ.
- `--resume auto|require|never` — see [Training-cache flags](#training-cache-flags) (default auto).
- `--new-run` — see [Training-cache flags](#training-cache-flags).

**Writes** into a store of its own, never the `PRISM_ARTIFACTS` root:
[Inputs and records](getting-started.md#inputs-and-records) says which, and
[The artifact store](getting-started.md#the-artifact-store) what it writes there. A failed run
removes the temporary directory it made only when nothing was written into it.

**What to watch.** The run shows that the chain runs, not that it is calibrated:

- `[smoke] conditioning width = <features>+<flags> + 1 + <block> = <total>` adds up the vector the
  network conditions on: the summary features and their validity flags, one channel for the
  logarithm of the recording length, and the mode's block. In chi mode that is the chi block, whose
  width is fixed by the probe slots rather than the probe count; in forced mode, one column per
  forcing parameter; in spontaneous mode, none. Read it with the mode the `[cfg]` line names. The
  infer stage checks the observation it builds against this width and fails the run on a mismatch,
  which is the error this run exists to catch before a long one.
- `[cfg] bounds=<path>` and `[cfg] rescale order: [...]`, read together. The width cannot tell two
  boxes apart in chi mode: boxes that differ only in a rescale parameter, such as the force scale,
  report the same mode and the same width. Only the bounds path beside the rescale order shows which
  box the run used.
- `[chi] masked probes: ...`, a run total of masked probes. Read the training stage's line: with
  `--checkpoint` it is scoped to every committed batch of the simulation cache, resumed batches
  included; without it, both lines are scoped to this process only, and the training line is the
  first. The calibration stage logs a line of its own, which is not the training figure. The
  reference is about 37 % of training probes, and one run within about 12 percentage points of it
  says nothing either way: every row of a batch shares one operating point and one probe set, so the
  effective sample size is the batch count, not the probe count. Compare the mean of a few runs.
- The pre-flight warnings on stderr: a recording length outside the training range, and a truth
  outside the training distribution.
- The calibration at these sizes has no power, so its verdict means nothing; the last `[smoke]`
  line says so. Only a crash means something here.
- `[ok] <stage> in <s>s` after each stage and `[smoke] ALL STAGES COMPLETED in ...` at the end; on
  a failure, `[smoke] *** FAILED in stage <stage> ***` before the traceback.

A seeded run is not bitwise-reproducible on a CUDA card or across devices.

**Rules:**

- With the posterior stage among the stages, `--resume require` or `--resume never` without
  `--checkpoint` is a usage error, since no cache is read or written; so is `--resume require`
  without `--prior`, since a new prior's fit is part of the cache's identity and no cache can match.
  Both are refused before the prior is built.
- A resume uses the same store, the same `--num-runs` and `--run-size`, and `--prior` naming the
  prior the first run built (`smoke_prior` under `--save`), with `prior` in `--stages`, because the
  prior stage is what loads it. The usage checks above do not ask for `prior`: without it the run
  prints `[skip] prior`, trains nothing and exits 0. A run one setting away is refused unless
  `--new-run` is given ([Training-cache flags](#training-cache-flags)).
- Given `--store-root` without `--checkpoint`, the run prints that nothing will be resumable.
- After Ctrl-C it prints advice for the case where a `[checkpoint]` line said batches were saved:
  the flags to re-run with to continue them, `--store-root` and `--prior` naming this run's store
  and prior, `--checkpoint`, `--stages prior,posterior`, `--resume require`, and the same
  `--num-runs` and `--run-size` (`core.tool._smoke_interrupt_advice`). It names no configuration
  flag and no `--cell`: those stay as they were.

**Exit codes** as in [Exit codes](#exit-codes).

## fdt

The effective-temperature analysis (FDT) for one cell: the sanity checks, then the production sweep
of driven responses, set against the cell's spontaneous fluctuations.

- `--cell PATH` — required: the cell file whose ground truth the analysis runs at.
- `--model NAME` — model name (default the cell file's parent folder, upper-cased).
- `--store-root PATH` — the store this run writes its record into (default the artifacts root).
- `--name NAME` — see [Naming flags](#naming-flags).
- `--note TEXT` — see [Naming flags](#naming-flags).
- `--n-freqs N` — drive frequencies in the driven-response sweep (default 60).
- `--ensemble-m N` — trajectories per frequency (default 256).
- `--freqs-per-batch N` — frequencies packed into one simulator call (default 1).
- `--f0 X` — non-dimensional drive amplitude; keep it inside the linear regime (default 0.05).
- `--seed N` — the random seed for the whole run, a whole number from 0; the record carries it, so the run can be repeated (default none: one is drawn and recorded).
- `--skip-sanity` — skip the sanity checks and go straight to the production sweep.
- `--no-production` — stop after the sanity checks.

**Writes** one fdt record, progressively. It prints `[prism fdt] writing record <id> at <dir>`
before anything is spent and before the folder exists. The folder and a first manifest appear when
the run opens the record, which it logs as `Writing fdt record <id> at <dir>`, and the manifest is
rewritten as the run goes. It prints `[prism fdt] record <id> at <path>` when it finishes. The
record holds the figures (the power spectrum, the chi components, T_eff/T and the spontaneous
trajectory), the numbers in `data.h5`, the settings and seed, and the log. A cancel or a crash after
the record opens keeps the folder, marked unfinished; a cancel or a refusal before then leaves
nothing (`core.FDT.fdt_pipeline.run_fdt`).

**Rules:**

- The command takes no configuration flags: no prior is built and nothing is trained, so it has no
  observation mode and no `--bounds`. The bounds file is resolved from the cell
  ([How a cell finds its bounds file](recordings.md#how-a-cell-finds-its-bounds-file)), and the
  record names it by path and hash.
- It runs on the CPU only and takes no `--device`: its solver steps a small ensemble one time step
  at a time, which at 256 trajectories runs about 3.4 times faster on the CPU than on a card.
- The model comes from the cell's folder unless `--model` is given. A model FDT cannot run is refused
  with the reason `(--model)`: FDT drives the observable itself, so a user model needs additive,
  non-zero observable noise and no intrinsic forcing.
- The pair `--skip-sanity` and `--no-production` is a usage error: together they run nothing.
- An `--n-freqs`, `--ensemble-m` or `--freqs-per-batch` below 1, an `--f0` of 0 or less, and a
  `--seed` outside 0 to 2^64 − 1 are refused, naming the flag (`core.cli.make_fdt_config`).

**Exit codes** as in [Exit codes](#exit-codes). After Ctrl-C the record named by the
`writing record` line is kept, marked unfinished, unless the run was stopped before its folder
existed; then there is nothing to clear. `python -m core artifacts list fdt` lists it and
`python -m core artifacts rm fdt <id>` removes it. Nothing resumes; the same command starts the
analysis again, and with `--name` only once the unfinished record is removed, because it keeps the
name. The `artifacts` family reads only the `PRISM_ARTIFACTS` root, so after a run given
`--store-root`, point `PRISM_ARTIFACTS` at that root first.

## crossval

The FDT parameter-sweep study on the Nadrowski model, whose model is fixed: an S sweep, holding
T_a/T = 1 and varying S (FDT is restored as S goes to 0), then a T_a/T sweep, holding S = 0 and
varying T_a/T (restored as T_a/T goes to 1).

- `--cell PATH` — required: a Nadrowski cell file whose ground truth the sweeps start from.
- `--preset exploratory|production` — resolution preset (default exploratory).
- `--s-grid MIN MAX N` — required: the S sweep grid.
- `--t-grid MIN MAX N` — required: the T_a/T sweep grid.
- `--store-root PATH` — the store this run writes its records into (default the artifacts root).
- `--name NAME` — a base name for the study's two records, `NAME-s` and `NAME-temp`, one per swept parameter; `''` leaves both unnamed. Both names are claimed before anything is spent, so a taken one is refused before either sweep starts.
- `--note TEXT` — see [Naming flags](#naming-flags).
- `--n-freqs N` — drive frequencies of each sweep's one common grid, as a floor: the grid covers every resonant operating point's band, and its point count grows with the span of their resonances (default the preset's: exploratory 30, production 60).
- `--ensemble-m N` — trajectories per frequency (default 256).
- `--freqs-per-batch N` — frequencies packed into one simulator call (default 1).
- `--f0 X` — non-dimensional drive amplitude; keep it inside the linear regime (default 0.05).
- `--seed N` — the random seed for the whole run, a whole number from 0; the records carry it (default none: one is drawn and recorded).

**The presets** set the resolution the flags do not expose, and two flags' defaults
(`core.cli.SWEEP_PRESETS`):

| preset | drive band, in multiples of ω₀ | drive window, in drive periods | spontaneous recording, ND time | `--n-freqs` | `--ensemble-m` |
|---|---|---|---|---|---|
| exploratory | 0.2 to 30 | 20 | 4000 | 30 | 256 |
| production | 0.1 to 30 | 30 | 8000 | 60 | 256 |

The operating points' resonances set ω₀, not the cell's: a sweep's one grid runs from the band's
low multiple of the lowest resonant operating point's ω₀ to its high multiple of the highest's
(`core.FDT.cross_validation._build_common_grid`).

**Writes** two fdt records, `NAME-s` and `NAME-temp`, each progressive like the record of
[fdt](#fdt), and each with its `data.h5`, its 3-D plot and its point counts. The S sweep's record is
finished and readable before the T_a/T sweep starts. Before anything is spent, it prints one
`[prism crossval] writing record <id> at <dir> (the S sweep)` line and one for the T_a/T sweep; then
one `[prism crossval] sweep record <id> at <path>` line per finished record.

**Rules** for each grid, `MIN MAX N`: N must be a whole number of at least 2, else a usage error;
MIN must be below MAX; and the T_a/T grid's MIN may not be below 0, because a negative temperature
ratio is unphysical and the simulation diverges there. The last two are refusals naming the grid's
flag; S has no floor. The resolution settings and the seed are refused as [fdt](#fdt) refuses them
(`core.cli.make_param_sweep_config`). A sweep whose every operating point failed keeps its
unfinished record and lets the other sweep run; when both sweeps measured nothing, the study is
refused `(--s-grid)`, exit 1, with both unfinished records kept
(`core.FDT.cross_validation.run_param_study_cli`). Like `fdt`, the study runs on the CPU only and
takes no configuration flags.

**Exit codes** as in [Exit codes](#exit-codes). After Ctrl-C a sweep that had finished keeps its
finished record, the one in progress keeps an unfinished one, and a sweep not yet started left
nothing; the advice under [fdt](#fdt) applies to each.

## compare

Draws saved fdt records into a comparison record of its own (kind fdt, study "comparison"): its
figures are its output, and its `data.h5` holds the common grid and the interpolated curves.

- Nothing is simulated, and no record it draws is modified. It has no `--store-root` and no
  configuration flags: it reads and writes the `PRISM_ARTIFACTS` root.
- Runs land on different frequencies, since each detects its own resonance, so the modes that draw
  two or more runs interpolate the curves onto a grid log-spaced over the intersection of their
  spans; a point whose bracketing samples include a blank stays blank rather than being drawn
  through. Runs that share no frequency band are refused. `renormalise` draws its one run on the
  run's own grid and interpolates nothing.
- Every mode takes each record with its own `--record REF`, by name or id. A record whose run did
  not finish is refused, naming it `(--record)`, as are one run named twice, a record of the wrong
  study (a single-cell run or a sweep) and a record with no data file.
- A comparison names the records it drew in its body, not as its parents, so it does not protect
  them: [artifacts rm](#artifacts-rm) of one succeeds, and the comparison's lineage report then
  prints that record as `MISSING`.
- It logs `Writing comparison record <id> at <dir>` when it opens its record and prints
  `[prism] fdt <name>__<id>  <path>` when it finishes. After Ctrl-C the opened record is kept,
  marked unfinished, and nothing resumes.

### compare cells

Two or more single-cell runs' ratio curves on one axis, labelled by cell.

- `--record REF` — required, once per record: a saved single-cell fdt record, by name or id; at least two.
- `--name NAME` — see [Naming flags](#naming-flags).
- `--note TEXT` — see [Naming flags](#naming-flags).

**Exit codes** as in [Exit codes](#exit-codes).

### compare repeats

Several runs of one cell, with the spread across them drawn as a band. The mode does not enforce one
cell: it names the cells it drew, as a notice in the record when there is more than one. It also
notes records that share a seed: they are one run drawn twice, and their spread measures nothing.

- `--record REF` — required, once per record: a saved single-cell fdt record, by name or id; at least two.
- `--name NAME` — see [Naming flags](#naming-flags).
- `--note TEXT` — see [Naming flags](#naming-flags).

**Exit codes** as in [Exit codes](#exit-codes).

### compare renormalise

One run's T_eff/T recomputed with a supplied normalisation constant, drawn against the original.

- `--record REF` — required: exactly one saved single-cell fdt record, by name or id; it must record the constant it used.
- `--name NAME` — see [Naming flags](#naming-flags).
- `--note TEXT` — see [Naming flags](#naming-flags).
- `--prefactor X` — required: the normalisation constant to recompute T_eff/T with; one that is not greater than 0 is refused `(--prefactor)`.

**Exit codes** as in [Exit codes](#exit-codes).

### compare sweeps

Two sweep records together, and a slice of both at one operating point.

- `--record REF` — required, once per record: exactly two saved sweep records, sweeping the same parameter over ranges that overlap.
- `--name NAME` — see [Naming flags](#naming-flags).
- `--note TEXT` — see [Naming flags](#naming-flags).
- `--at X` — the operating point to slice both sweeps at; without it, the middle of the range they share. A point outside that range is refused.

**Exit codes** as in [Exit codes](#exit-codes).

## artifacts

Reads, annotates and tidies the store of records: the command-line twin of the window's Artifacts
screen. [Browsing the store](getting-started.md#browsing-the-store) sets the two side by side.

- The family takes no configuration flags and no `--store-root`: it reads the root
  `PRISM_ARTIFACTS` names. A listing must not be able to fail on a bounds file it does not need, so
  the family loads nothing and cannot tell you whether a posterior matches your bounds file; loading
  it does.
- `<kind>` is one of the eight kinds of [The artifact store](getting-started.md#the-artifact-store),
  named in the singular. It is a plain positional, so an unknown kind is a refusal (exit 1), not a
  usage error. `<ref>` is a record's name or its id.
- Its output is plain aligned text on stdout, meant for a person or a script. It is not a record and
  reaches no `log.txt`.
- After Ctrl-C the family prints the generic advice of [Exit codes](#exit-codes), about a removed
  record and `--resume require`. It does not apply here: these modes write no record and keep no
  cache, so there is nothing to resume.

### artifacts list

`python -m core artifacts list [<kind>]`: one line per record of a kind, or, with no kind, of all
eight under `== <kind> ==` headings. The mode takes no flags.

The columns are the window's, lower-cased: name and created for every kind; then progress and
finished for a simulation cache, mode, width and amortized for a posterior, mode and width for an
observation, variant for a diagnostic, and study, points and finished for an fdt record; then the
note. A directory with no usable manifest comes last, as `incomplete  <dir>  -- <reason>`. With no
kind, the kinds that hold nothing are named on a trailing `nothing in:` line.

**Exit codes** as in [Exit codes](#exit-codes): an empty listing exits 0, so a script can tell
"nothing on disk" from a mistake.

### artifacts show

`python -m core artifacts show <kind> <ref>`: one record's manifest, rendered as the window's detail
pane renders it, then `== records ==` and the tail of its `log.txt`, at most the last 1 MiB. The
mode takes no flags. A note follows the manifest when the directory's name disagrees with it, which
a refused rename leaves. A training cache keeps no log, and a record written outside a run has none;
each says so.

**Exit codes** as in [Exit codes](#exit-codes); a reference that names no complete record is refused.

### artifacts note

`python -m core artifacts note <kind> <ref> --note TEXT`: sets or clears one record's note, and
prints `[prism] <kind> <label>: note = '<text>'` or `note cleared`.

- `--note TEXT` — required: one line of at most 200 characters; `''` clears the note. A note with a newline, or a longer one, is refused `(--note)` rather than trimmed. It is required rather than defaulted, so leaving it off can never clear a note.

A reference that names a leftover directory, rather than a record, is refused with a pointer to
[artifacts sweep](#artifacts-sweep).

**Exit codes** as in [Exit codes](#exit-codes).

### artifacts rm

`python -m core artifacts rm <kind> <ref>`: deletes one record and prints
`[prism] removed <kind> <ref>: <path>`. The mode takes no flags.

It is refused while any other record depends on it. The refusal names every dependent and why: it
names this record as a parent, or, for a prior, it is a training cache generated against it, found
by the prior's fit alone. Delete those first; no flag overrides the refusal. A comparison is not a
dependent: deleting a record a comparison drew succeeds, and the comparison's lineage report then
prints it as `MISSING` ([compare](#compare)).

**Exit codes** as in [Exit codes](#exit-codes).

### artifacts sweep

`python -m core artifacts sweep [<kind>] [--yes]`: removes every directory with no `manifest.json`
at all, every loose file sitting directly inside a kind directory, and, with no kind, the legacy
directory an older build left beside the kind directories.

- `--yes` — actually remove them. Without it this is a dry run: it prints exactly what it would remove and removes nothing.

The dry run prints `[prism] would remove ...` lines and then
`[prism] dry run: nothing was removed. Re-run with --yes to remove <n> items.`; with `--yes` each
removal prints `[prism] removed ...`. A directory that holds a `manifest.json` is never swept, even
one this build cannot read or one that declares another kind: it is reported as
`[prism] kept <kind> <dir> -- <reason>` and left where it is (`core.tool.browse._sweep`, and
`core.artifacts.store.ArtifactStore.remove_incomplete` refuses it too). Nothing inside a record's
own folder is offered: the loose-file read never descends into a record, so an unfinished fdt
record's measurements cannot be reached from here (`core.artifacts.store.ArtifactStore.loose_files`).
Anything written recently is refused at removal, as
[Browsing the store](getting-started.md#browsing-the-store) describes.

**Exit codes** as in [Exit codes](#exit-codes): 0 when everything could be read and every removal
asked for succeeded, or there was nothing to sweep; 1 when something could not be read or removed,
each named on stderr.

### artifacts summary

`python -m core artifacts summary <kind> <ref> [--out PATH]`: the lineage report, this record and
then its parents back through the chain, oldest last; a parent that is gone prints as `MISSING`.

- `--out PATH` — write the report to this file instead of stdout.

With `--out` the report is written with LF line endings, byte for byte what the window's Lineage
report writes, and the mode prints `[prism] lineage report: <path>`. A write that fails is refused
rather than raised.

**Exit codes** as in [Exit codes](#exit-codes).

## Worked examples

The three examples write into whatever records root is current when they run, so the first line
points `PRISM_ARTIFACTS` at a scratch folder. Run all three in that same shell, so it stays set and
the real store gains nothing.

**A stage chain at small sizes.** The prior builds at its full default size, which takes about a
minute and a half on the graphics card and longer on the CPU. Training, calibration and inference
run at sizes that show the chain working and mean nothing more.

```bash
export PRISM_ARTIFACTS="$HOME/prism-scratch"    # PowerShell: $env:PRISM_ARTIFACTS = "$HOME\prism-scratch"
python -m core prior    --chi --bounds Resources/Bounds/nadrowski/master.txt --name demo_prior
python -m core train    --chi --bounds Resources/Bounds/nadrowski/master.txt --prior demo_prior --num-runs 8 --run-size 32 --max-epochs 5 --name demo_posterior
python -m core validate --chi --bounds Resources/Bounds/nadrowski/master.txt --posterior demo_posterior --n-cal 100 --seed 7
python -m core infer    --chi --bounds Resources/Bounds/nadrowski/master.txt --posterior demo_posterior --cell Resources/Cells/nadrowski/master_spont.txt --t-obs 4.5 --name demo_inference
python -m core artifacts summary inference demo_inference
```

The last line prints the inference's lineage, back through its observation and its posterior to the
prior and the simulation cache the posterior was trained from. Running `validate` again with
`--seed 7` repeats its calibration set on the same device.

**A probe check on one cell.** Before recording from a preparation described by a cell file, ask
whether the configured chi band and drive hold for it, then how hard it can be driven:

```bash
python -m core probes band  --bounds Resources/Bounds/nadrowski/master.txt --cell Resources/Cells/nadrowski/master_spont.txt --name spont_band
python -m core probes drive --bounds Resources/Bounds/nadrowski/master.txt --cell Resources/Cells/nadrowski/master_spont.txt --name spont_drive
```

Each writes one diagnostic record; `python -m core artifacts show diagnostic spont_band` prints its
verdicts and the settings it judged. The bounds file must declare a Forcing section for
`probes drive`, and this one does.

**An FDT run twice, then compared.** Each run draws and records a seed of its own, so the two are
independent repeats of one cell. The sizes here (4 drive frequencies, 8 trajectories each, no
sanity checks) show the commands working and mean nothing more. A real run takes its defaults, 60
frequencies of 256 trajectories after the sanity checks, and is long: it runs on the CPU only.

```bash
python -m core fdt --cell Resources/Cells/nadrowski/master_spont.txt --n-freqs 4 --ensemble-m 8 --skip-sanity --name spont_fdt_a
python -m core fdt --cell Resources/Cells/nadrowski/master_spont.txt --n-freqs 4 --ensemble-m 8 --skip-sanity --name spont_fdt_b
python -m core compare repeats --record spont_fdt_a --record spont_fdt_b --name spont_fdt_repeats
```

The comparison record draws both ratio curves on one axis, with their spread as a band. When you
are done, delete the scratch folder.

## Where the old scripts went

The tool replaced a folder of scripts, each of which read its settings from the environment, and an
interactive prompt interface. `smoke_train.py` became `smoke`. `sbc_characterize.py` became `sbc`.
`posterior_identifiability.py`, `identifiability_offgt.py` and `degeneracy_map.py` became the three
`identifiability` modes, `rotation`, `laplace` and `jacobian`. `channel_ablation.py` became
`ablation`, which reads the posterior's own simulation cache. `chi_f0_sweep.py`, `chi_mask_audit.py`
and `build_master_cells.py`, which measured the chi band and drive, audited masked probes and chose
the master cells, were archived and rebuilt as `probes band`, `probes mask` and `probes drive`. The
remaining scripts were archived with no replacement, and the interactive prompt interface was
retired: every setting is now a flag, and the tool never prompts.
