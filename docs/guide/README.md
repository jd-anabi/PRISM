# The PRISM guide

Checked against commit 04fbe54.

This guide explains how to use PRISM, how to run it at full size, and how it is built. Each fact is
written once, on the page it belongs to; the reading paths below take each kind of reader through the
pages in the order that reader needs them.

## What PRISM is

PRISM is a research tool built around a simulated inner-ear hair bundle: a stochastic model of the
bundle's motion. It serves two lines of work.

- **The generalised fluctuation-dissipation theorem (GFDT).** PRISM measures how far a simulated
  bundle's spontaneous fluctuations and its response to a small drive depart from the equilibrium
  fluctuation-dissipation relation, expressed as an effective temperature, for one cell or across a
  sweep of model parameters.
- **Simulation-based inference.** PRISM trains a neural posterior on simulated bundles, calibrates
  it, and runs it on a recording, either simulated from a cell whose parameters are known or measured
  at the bench, to infer the model parameters that produced it.

Three models are built in (Nadrowski, Hopf and BP), and a user can define more.

## Two front ends

PRISM has two front ends over one core.

- **The window**, a PySide6 desktop application: `run.bat` on Windows, `bash run.sh` on macOS and
  Linux, or `python -m core.gui`. Its Artifacts screen browses the store of everything PRISM has
  generated.
- **The command-line tool**, `python -m core <subcommand>`, which takes flags only and never
  prompts. `python -m core --help` lists the subcommands; its `artifacts` family is the command-line
  twin of the Artifacts screen.

Both front ends call the same stages (`core.orchestrator`) and the same FDT analysis (`core.FDT`);
the diagnostics (`core.diagnostics`) are reached from the tool alone. Every setting either front end
offers reaches its stage as an argument, never as an edit to `core.config`, so a run started
from the window and the same run started from the tool do the same thing.

## Reading paths

### For the owner-scientist

You decide the science and run the full-size training.

1. [The science behind the settings](science.md), read in order: why each setting has the value it
   has, what the August 2026 retrain measured, and the questions still open.
2. [The retrain runbook](retrain.md): the decisions, the pre-flight checks, the exact commands, what
   to watch while it runs, and the gates a new model must pass.
3. [The inference settings](window.md#the-inference-settings): every setting on the Parameter
   Inference tabs, with its flag, the argument that receives it, its default and what it really
   changes.
4. [Settings that are not speed dials](window.md#settings-that-are-not-speed-dials): the settings
   that look like a trade of time for precision but change the prior or the measurement instead.
5. [validate](command-line.md#validate), [tsnpe](command-line.md#tsnpe) and
   [probes](command-line.md#probes): the calibration, a narrowing round, and the checks of the chi
   probe settings, flag by flag.
6. [The card smoke gate](testing.md#the-card-smoke-gate): the short end-to-end run on the graphics
   card after any change to code that moves tensors.

### For a lab member bringing recordings

You have recordings of a hair bundle and want to know what they say about its parameters.

1. [Getting started](getting-started.md): launching PRISM, where its inputs and records live, and a
   first run in each front end.
2. [Bringing recordings](recordings.md), read in order: the input files, the observation modes and
   what each needs, drive frequencies, recording lengths, and the chi assumptions and how to check
   them.
3. [probes](command-line.md#probes): check a preparation, described by its cell file, before you
   record: whether the configured chi band and drive hold for it, and how hard it can be driven.
4. [infer](command-line.md#infer): run a trained posterior on your recordings.
5. [What the window remembers](window.md#what-the-window-remembers) and
   [Supported models](window.md#supported-models): which boxes keep their values from one session to
   the next, and which models each screen accepts.
6. [Chi probe design](science.md#chi-probe-design) and
   [Reading calibration honestly](science.md#reading-calibration-honestly): why the probes are placed
   as they are, and how much a calibration result can tell you.

### For a successor

You will maintain and extend the code.

1. [Getting started](getting-started.md): launching, the two roots, and the store's records, names
   and ids.
2. [Architecture](architecture.md): the module map, the stages and the compositions that chain them,
   the store, both front ends, the solver, memory planning and the FDT pipeline.
3. [Rules and traps](rules-and-traps.md): the rules the code keeps, among them that a bad input is
   refused before anything is spent, as a `core.refusals.Refusal` that names the setting at fault by
   a key each front end maps to its own control or flag, and that every setting travels as an
   argument; and the traps that have already cost a run.
4. [Testing](testing.md): the interpreter, the test gates and their markers, the smoke gate on the
   card, and what a green suite does not certify.
5. [Conventions](command-line.md#conventions) and [Exit codes](command-line.md#exit-codes): how the
   tool reports a refusal, with the flag that answers it, and what each exit code means.

### For a reviewer

You want to judge what a trained model can claim, and check it for yourself.

1. [Reading calibration honestly](science.md#reading-calibration-honestly),
   [Identifiability limits](science.md#identifiability-limits) and
   [The August 2026 retrain](science.md#the-august-2026-retrain): what a calibration result does and
   does not show, which parameters the data cannot pin down, and what the last full retrain measured.
2. [The artifact store](architecture.md#the-artifact-store): how each record carries its settings,
   its parents and its log, and what loading a record refuses.
3. [Narrowing-round safety rules](rules-and-traps.md#narrowing-round-safety-rules): each rule a
   narrowing round keeps, where the code enforces it, which test pins it, and which rules are still
   only written down.
4. [Gates](retrain.md#gates) and [Afterwards](retrain.md#afterwards): the gates a retrained model must
   pass, and which record answers each one.
5. [artifacts](command-line.md#artifacts) and [validate](command-line.md#validate): to reproduce a
   calibration, print its record with `python -m core artifacts show calibration <ref>`, read the
   posterior among its parents and the seed among its results, then run `validate` on that posterior
   with `--seed` set to that seed and the calibration settings the record lists.

## The pages

- [Getting started](getting-started.md): launching PRISM, inputs and records, a first run in each
  front end, and the artifact store.
- [The window](window.md): each screen, the inference settings, what the window remembers, and which
  models each screen supports.
- [The command-line tool](command-line.md): every subcommand, mode and flag, with worked examples.
- [Bringing recordings](recordings.md): the input files, observation modes, recording lengths, the
  chi assumptions, and defining a model of your own.
- [The science behind the settings](science.md): the reasoning behind each setting, what the last
  retrain measured, and the open questions.
- [Architecture](architecture.md): how the code is put together, from the module map to the FDT
  pipeline.
- [Rules and traps](rules-and-traps.md): the rules the code keeps, the narrowing-round safety rules,
  and the traps that still bite.
- [Testing](testing.md): the environment, the test gates and markers, and the checks on the graphics
  card.
- [The retrain runbook](retrain.md): the next full-size training run, from its decisions to its
  gates.
