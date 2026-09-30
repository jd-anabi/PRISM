# Testing

Checked against commit 95745d2.

This page is for whoever changes the code: the interpreter and the environment a check needs, the
test gates and their markers, the two runs on the graphics card that no suite replaces, what a green
suite does not certify, and notes for writing tests. What each flag means is on
[The command-line tool](command-line.md), and the rules the tests pin are on
[Rules and traps](rules-and-traps.md).

## Environment and interpreter

- The interpreter is the conda environment `biophys-env`'s,
  `C:\Users\J\anaconda3\envs\biophys-env\python.exe`: Python 3.12, torch 2.9.0+cu130 and PySide6
  6.9.3. The `python` on PATH is a different interpreter, with none of the dependencies. Every
  `python` on this page means `biophys-env`'s; the card's command lines below call it `$py`.
- sbi 0.25.0 is installed, while `requirements.txt` pins 0.26.1. An upgrade must check again that
  sbi's per-epoch print still fires: the window's cooperative cancel hooks into it, and without it a
  Cancel pressed while the flow trains waits for the whole fit instead of one epoch
  ([The window](architecture.md#the-window)). `requirements.txt` names the other thing to check
  after that upgrade: calibration numbers from before it are not comparable with those after it.
- Run every check from the repository root. The code finds its two roots from its own location
  ([Inputs and records](getting-started.md#inputs-and-records)), but the command lines here give their
  input files as paths relative to the root.

## Environment variables

What each variable sets is under [Environment variables](command-line.md#environment-variables); this
is who sets which, for a check.

- `KMP_DUPLICATE_LIB_OK=TRUE` is needed by anything that imports torch
  ([Launching PRISM](getting-started.md#launching-prism) says why). The root `conftest.py` defaults
  it for pytest, the tool's entry (`core.__main__`) sets it before torch is imported, and `run.bat`
  and `run.sh` set it for the window. A bare `python -c` or a script sets neither this nor the next
  one: set them yourself there.
- `QT_QPA_PLATFORM=offscreen` is for headless checks that import PySide6, and only those. The root
  `conftest.py` defaults it; the launchers do not set it, because the window needs a real platform
  plugin.
- `PRISM_RESOURCES` and `PRISM_ARTIFACTS` move the inputs root and the records root
  ([Inputs and records](getting-started.md#inputs-and-records) says when each is read, and which root
  each run writes to). Point `PRISM_ARTIFACTS` at a scratch folder, or wrap the code in
  `core.artifacts.use_store`, to keep a check's records out of the real `Artifacts/`. Inside a running
  process, the default store keeps the root it was first built with, so set the variable before
  anything builds it, or use `core.artifacts.use_store`.
- `PRISM_VRAM_CEILING_GIB` is read afresh for every batch plan.
- `PRISM_MEM_LOG_EVERY` is read once, when `core.SBI.pipeline` is first imported, so setting it
  inside a running window changes nothing.
- `PYTORCH_CUDA_ALLOC_CONF` is given an allocator policy by `core/config.py` when it is imported,
  unless one is already exported ([Memory and devices](rules-and-traps.md#memory-and-devices)).

## Test gates and markers

The configuration is in three files: `pytest.ini` (the test folders and the markers), the root
`conftest.py` (the environment defaults and the marker skips) and `tests/conftest.py` (the fixtures
and the session guards).

- **The fast gate** is `python -m pytest -m "not slow" -q`, run as one process, with a target of 16
  minutes. One process, because the session installs its temporary records root once: only a single
  process can catch code that swaps the default store and never puts it back, and a gate split
  across processes hides it.
- **The full run** is `python -m pytest`; budget about an hour, and log it to a file.
  `python -m pytest -m slow --collect-only -q` lists the slow set, the real simulations the fast gate
  leaves out.
- **The markers**, registered in `pytest.ini` and turned into skips by the root `conftest.py`:
  - `slow`: left out of the fast gate;
  - `gpu`: needs CUDA, and is skipped when `torch.cuda.is_available()` is false; with a card
    present, the gpu-marked tests run on it inside every fast gate;
  - `display`: needs a real screen, and is skipped offscreen. In PowerShell,
    `$env:QT_QPA_PLATFORM = "windows"; python -m pytest -m display; Remove-Item Env:QT_QPA_PLATFORM`
    runs these on the real screen, since the root `conftest.py` only defaults the variable; they
    create hidden native windows, and nothing shows.
- **Counting.** `python -m pytest --collect-only -q` counts the tests. Count them; never trust a
  number written down.
- **One pytest process at a time.** Two would share the card and the test model the fixtures install
  in the real `Resources/` ([Writing tests](#writing-tests)), and one's teardown would remove the
  other's files.
- **Never edit a source file while a suite runs.** Several tests read source with
  `inspect.getsource`, which reads the file as it is now with the line numbers the function was loaded
  with, so an edit mid-run produces a failure that is not real.
- **The session guards**, autouse fixtures in `tests/conftest.py`:
  - `_sandbox_default_store`: a temporary records root for the whole session, and a check at teardown
    that the real `Artifacts/` gained nothing;
  - `_checkpointing_off_unless_asked`: training keeps no simulation cache, so a complete cache left by
    one test cannot skip another's generation; a test that wants a cache passes `checkpoint_every` to
    `build_posterior`;
  - `_settings_home`, with `_isolated_settings` for each test: a temporary settings file, so no panel
    reads the real `PRISM.ini` and no test writes it, checked at teardown;
  - `_no_modal_dialogs`: a message box records itself and returns at once, because offscreen a modal
    dialog nobody expected stalls the run instead of failing it;
  - `_no_sbi_logs`: `<repo>/sbi-logs` must not exist, checked at setup as well as teardown, so a
    leftover tree makes every run error before its first test.
- **The source scans**, which read files rather than run code:
  - `tests/test_source_hygiene.py`: no code file, test or guide page cites a working document or a
    label one coined ([Rules the code keeps](rules-and-traps.md#rules-the-code-keeps) has the comment
    policy);
  - `tests/test_artifact_store.py::test_no_literal_resource_paths_outside_config`: no code outside
    `core/config.py` builds a literal `Resources/` or `Artifacts/` path;
  - `tests/test_docs.py`: every relative link in this guide resolves, every page opens with its
    commit stamp, and the command-line page names every subcommand and flag, with each default as its
    `--help` states it.

## The card smoke gate

The suites run on the CPU apart from the gpu-marked tests, so a tensor built on the wrong device
passes them and fails only on the card. After touching code that moves tensors, and before any
full-size run, run these five command lines on the card, from the repository root, and check
`$LASTEXITCODE` after each one. Replace `<scratch>` with a folder outside the repository.

```powershell
$py = "C:\Users\J\anaconda3\envs\biophys-env\python.exe"
$B  = "--bounds","Resources/Bounds/nadrowski/master.txt"
$C  = "--cell","Resources/Cells/nadrowski/master_spont.txt"
$S  = "<scratch>"
# run 1: chi; builds and names smoke_prior/smoke_posterior; writes the simulation cache
& $py -m core smoke --chi --t-obs 4.5 @B @C --checkpoint --save --store-root "$S/smoke"
# run 2: same store, same --num-runs; must resume
& $py -m core smoke --chi --t-obs 4.5 @B @C --checkpoint --store-root "$S/smoke" --prior smoke_prior --stages prior,posterior --resume require
# run 2b: one setting away; must exit 1 naming n_runs
& $py -m core smoke --chi --t-obs 4.5 @B @C --checkpoint --store-root "$S/smoke" --prior smoke_prior --stages prior,posterior --num-runs 2
# run 3: forced mode, its own store
& $py -m core smoke --no-chi --t-obs 4.5 @B --cell Resources/Cells/nadrowski/master_weak.txt --checkpoint --save --store-root "$S/smoke_chi0"
# run 4: the tier-1 box (temperature inferred, the force scale derived) at the larger network, its own store
& $py -m core smoke --chi --t-obs 4.5 --bounds Resources/Bounds/nadrowski/master_tier1.txt --cell Resources/Cells/nadrowski/master_spont_tier1.txt --hidden-features 256 --num-transforms 10 --checkpoint --save --store-root "$S/smoke_t1"
```

Pass criteria, read off `$LASTEXITCODE` and the printed lines after each command:
- runs 1, 3 and 4: `$LASTEXITCODE` 0, no OOM lines, stage timings near the last recorded card
  run; on the two chi runs (1 and 4) the training `[chi] masked probes` run-total line (the one
  scoped to every committed batch of the simulation cache; the calibration stage logs a second
  one, scoped to this process only, which is not the criterion) within ±12 pp of 37 %;
- run 2: prints `Reusing the Fisher rotation stored with the training checkpoint` and
  `[checkpoint] resuming at batch 4/4`, `$LASTEXITCODE` 0;
- run 2b: `$LASTEXITCODE` 1, the refusal names `n_runs`, no `[fisher]` line, no new directory
  under `$S/smoke/simulations/`;
- run 4, also: both `[tier1]` lines, `rescale order: ['x_scale', 't_scale', 'T']` in the banner,
  and `smoke_posterior`'s `manifest.json` lists `T` in `body.transform.param_keys` and carries a
  `config.tier1` block.

`--bounds` is required: the same-named sibling rule would otherwise resolve the 12-dim
spontaneous box. After a change under `core/diagnostics` that moves tensors, also run the
diagnostic card, each command with `$env:PRISM_ARTIFACTS` set to the store named. Against
`$S/smoke`: `sbc --chi @B --posterior smoke_posterior --repeats 1 --n-cal 20`,
`identifiability jacobian --chi @B @C --t-obs 4.5` and `ablation --chi @B --posterior
smoke_posterior`. Against `$S/smoke_chi0`: `identifiability laplace --no-chi @B --posterior
smoke_posterior --cell Resources/Cells/nadrowski/master_weak.txt --t-obs 4.5`. Against
`$S/smoke_t1`: `identifiability jacobian --chi --bounds Resources/Bounds/nadrowski/master_tier1.txt
--cell Resources/Cells/nadrowski/master_spont_tier1.txt --t-obs 4.5` (its temperature column is a
measurement, not a failure). Against an empty scratch store: `probes band @B @C` and `probes drive
@B @C`. Against `$S/smoke`: `probes mask @B --prior smoke_prior`. Each exits 0 and writes one
record under `diagnostics/`. Delete `$S` afterwards.

What each run checks, and what passing looks like:

- **Run 1**, chi mode on the master box and the spontaneous master cell, builds a prior and a
  posterior, names them `smoke_prior` and `smoke_posterior` (`--save`), and writes a simulation cache
  (`--checkpoint`) into `$S/smoke`. It passes with exit 0, no out-of-memory line, stage timings near
  the last card run's (below), and the masked-probe criterion.
- **Run 2** loads `smoke_prior` from the same store, at the same batch count, and runs only the prior
  and posterior stages with `--resume require`. It must resume: it prints
  `Reusing the Fisher rotation stored with the training checkpoint (4/4 batches — COMPLETE, so generation will be skipped) — NOT recomputing it: …`
  and `[checkpoint] resuming at batch 4/4 from <dir> (reusing the stored rotation V)`, and exits 0.
  It is the card's check that a resume reuses the stored rotation instead of computing a new one
  ([Randomness and reproducibility](rules-and-traps.md#randomness-and-reproducibility) says why that
  matters).
- **Run 2b** asks for 2 batches against the 4-batch cache: one setting away. It must be refused before
  the Fisher step and before any simulation, with exit 1, a refusal that reads
  `differs only in n_runs: this run 2, that cache 4` and ends `(--new-run)`, no `[fisher]` line, and no
  new directory under `$S/smoke/simulations/`
  ([Training-cache flags](command-line.md#training-cache-flags)).
- **Run 3**, forced mode (`--no-chi`) on the weakly driven master cell, in a store of its own. It
  passes with exit 0 and no out-of-memory line; there are no probes, so no masked-probe line.
- **Run 4**, the tier-1 box, with temperature inferred and the force scale derived from it, at the
  retrain's network of 256 hidden features × 10 transforms, in a store of its own. It passes with
  exit 0, no out-of-memory line, the masked-probe criterion, both `[tier1]` lines, the rescale order
  above in the `[cfg]` banner, and the two manifest entries the criteria name
  ([The tier-1 constraint and temperature](science.md#the-tier-1-constraint-and-temperature)). Its
  calibration verdict means nothing at this size ([smoke](command-line.md#smoke)).
- **Run 4's resume** is not part of the recipe. The last card run also ran run 4 again against its
  own store, which checks that a resumed tier-1 run carries the stored rotation's eigenvalues: it
  printed `[checkpoint] the stored rotation's eigenvalues come with it (…)`, and the record kept
  them. Repeat it after a change to the resume path:

```powershell
& $py -m core smoke --chi --t-obs 4.5 --bounds Resources/Bounds/nadrowski/master_tier1.txt --cell Resources/Cells/nadrowski/master_spont_tier1.txt --hidden-features 256 --num-transforms 10 --checkpoint --store-root "$S/smoke_t1" --prior smoke_prior --stages prior,posterior --resume require
```

**Why `--bounds` names the master box.** `smoke` always takes `--bounds`, and the choice matters:
the cell `master_spont.txt` also has a bounds file of its own name, the 12-parameter spontaneous box,
which the runs that find a bounds file from the cell alone would pick
([How a cell finds its bounds file](recordings.md#how-a-cell-finds-its-bounds-file)). Named here, that
box would, in chi mode, report the same mode and the same conditioning width, and drop `f_scale`
without a word.

**The masked-probe criterion.** The training run's `[chi] masked probes` run-total line, the one
scoped to every committed batch of the simulation cache, must be within ±12 points of 37 %; the
calibration stage logs a line of its own, scoped to its own process, which is not the criterion.
[Chi probe design](science.md#chi-probe-design) says where 37 % comes from and why the band is that
wide. A single run's count can also move by one row's probes from process to process on the card, so
an exact count is never the criterion.

**The last card run**, on 29 September 2026, for the timings, in seconds per stage:

| run | prior | posterior | validate | infer | masked probes |
|---|---|---|---|---|---|
| 1 | 92 | 194 | 22 | 76 | 254 of 704 (36.1 %) |
| 3 | 91 | 60 | 13 | 20 | |
| 4 | 92 | 150 | 20 | 45 | 285 of 704 (40.5 %) |

Run 2 exited 0 in 7.7 s, its masked line equal to run 1's, and run 2b exited 1 in 5.8 s.

Delete the scratch folder when the gate, and the diagnostic card if you run it, are done:
`Remove-Item -Recurse -Force $S`.

## The diagnostic card

The diagnostics under `core/diagnostics` simulate on the card too, and the smoke gate does not reach
them. After changing code there that moves tensors, run the diagnostic card once the smoke gate has
passed, against the stores it left. The recipe above gives the card as a paragraph; this expands that
paragraph into command lines, with `$py`, `$B`, `$C` and `$S` as the recipe sets them. Check
`$LASTEXITCODE` after each: every one exits 0 and writes one record under `diagnostics/` in the store
`PRISM_ARTIFACTS` names. The empty scratch store the probe checks write into is `$S/probes`, which
the first of them creates.

```powershell
$env:PRISM_ARTIFACTS = "$S/smoke"
& $py -m core sbc --chi @B --posterior smoke_posterior --repeats 1 --n-cal 20
& $py -m core identifiability jacobian --chi @B @C --t-obs 4.5
& $py -m core ablation --chi @B --posterior smoke_posterior
$env:PRISM_ARTIFACTS = "$S/smoke_chi0"
& $py -m core identifiability laplace --no-chi @B --posterior smoke_posterior --cell Resources/Cells/nadrowski/master_weak.txt --t-obs 4.5
$env:PRISM_ARTIFACTS = "$S/smoke_t1"
& $py -m core identifiability jacobian --chi --bounds Resources/Bounds/nadrowski/master_tier1.txt --cell Resources/Cells/nadrowski/master_spont_tier1.txt --t-obs 4.5
$env:PRISM_ARTIFACTS = "$S/probes"
& $py -m core probes band @B @C
& $py -m core probes drive @B @C
$env:PRISM_ARTIFACTS = "$S/smoke"
& $py -m core probes mask @B --prior smoke_prior
Remove-Item Env:PRISM_ARTIFACTS
```

The last line puts the shell back on the real records root. What each should show:

- `sbc` and `ablation` at these sizes show only that the code runs: one repeat of 20 calibration
  datasets has no power, and a flow trained on four batches has no channel table worth reading.
- The two `identifiability jacobian` maps print the probe budget and the recording-length line
  ([identifiability jacobian](command-line.md#identifiability-jacobian)). The tier-1 map has a
  temperature column: it is a measurement, not a failure
  ([The tier-1 constraint and temperature](science.md#the-tier-1-constraint-and-temperature)).
- `identifiability laplace` prints its table of standard deviations at the cell's truth and at
  draws from the smoke prior; at this size it shows that the forced-mode path runs.
- `probes band`: the configured band (0.03, 0.3) and the configured drive 0.15 both hold for the
  master cell, and the 0.6× control is captured at every length.
- `probes drive`: free-running at about 0.02, and captured from 0.15 on the check's default grid,
  which contains 0.15. On the original sixteen-strength grid, which has no point between 0.1 and 0.2,
  capture reads 0.2; [Chi probe design](science.md#chi-probe-design) has both readings.
- `probes mask` over `smoke_prior`: masked probes within ±12 points of 37 %, the cycle floor the
  only cause (35.1 % on the last card run).

Then delete the scratch folder, as the recipe says.

## What a green suite does not certify

- **The GPU path.** With a card present, the gpu-marked tests (`tests/test_gpu_paths.py`, and some in
  `tests/test_user_sbi.py` and `tests/test_diagnostics.py`) run on it inside every fast gate. But they
  resume no simulation cache and read no real bounds or cell file: a tiny test model stands in. The
  card smoke gate covers those.
- **TorchScript's first runs.** On CUDA a scripted step's first calls run unoptimised and later ones
  fused, and the two differ by about one unit in the last place, so they are not bitwise-reproducible;
  warm up before any bitwise comparison
  ([The solver, CUDA graphs and reproducibility](architecture.md#the-solver-cuda-graphs-and-reproducibility)).
- **A seeded run on CUDA or across devices.** A seed repeats an experiment on one device and gives a
  bitwise repeat on the CPU only; the suites' reproducibility checks run on the CPU (same link).
- **Calibration at small sizes.** SBC at smoke or test sizes has no power: a pass or a FAIL there says
  only that the code ran ([Reading calibration honestly](science.md#reading-calibration-honestly)).
- **The window on a screen.** Offscreen tests cannot see layout, painting or a native dialog. Window
  features are checked by hand on a real screen, and the display-marked tests run only with
  `QT_QPA_PLATFORM=windows`.

## Writing tests

- **Stand-ins, not real simulations.** The fast gate has little headroom under its target, so a new
  fast test uses stand-ins. `tests/_fixtures.py` holds them: `stubbed_training` (a training stage that
  stops before the network), `stub_calibration_battery` (everything a calibration simulates or
  samples), `stand_in_gen_obs` (a cheap simulator of the right shape) and `build_fdt_record` (an fdt
  record in seconds); and `build_tiny_run`, behind the `tiny_run` fixture, when a test needs a real
  prior and posterior at tiny size.
- **The test model lives in the real `Resources/` while it is installed.** `install_sbitest`
  (`tests/_fixtures.py`), which `build_tiny_run` and `tests/test_tool.py`'s module fixture call,
  writes a one-variable model, SBITEST, into `Resources/Bounds/sbitest/`, `Resources/Cells/sbitest/`,
  `Resources/Units/sbitest/` and `Resources/Models/SBITEST.json`, so that the tool can be driven with
  real `--bounds` and `--cell` files; its teardown removes them in a `finally`. Other suites write
  throwaway models there the same way: `tests/test_user_models.py`'s UMTEST models, and
  `tests/test_user_sbi.py`'s full user-model pipeline. A killed run never reaches the `finally` and
  leaves them behind: delete whatever `git status` lists as untracked under `Resources/`. They are not
  inputs.
- **A widget that was never shown** reports a placeholder size from `width()`: 640×480 for one built
  without a parent, 100×30 for one built inside another. The suites run offscreen and show nothing,
  so assert on `sizeHint()`, `minimumSizeHint()` or `maximumWidth()`, or show the widget, resize it
  and pump the events first (`qt_app` and `pump` in `tests/_fixtures.py`).
- **Assert order, not adjacency.** `src.index(a) < src.index(b)` survives a line inserted between the
  two; a check that they are neighbours does not.
- **A test that reads source must keep seeing the words it looks for.** Several tests read the source
  of a function, so a rename or a rewording fails them by design; change the test in the same commit.
  Assert on code through `code_only` (`tests/_fixtures.py`), which strips comments and docstrings: a
  comment that documents a fix contains the very text the check forbids.
- **Pin that a knob reaches its target, not just the signature.** A setting a function accepts and
  never passes on satisfies a signature check. Follow it to where it is used, as
  `tests/test_settings_persistence.py::test_the_training_budget_reaches_build_posterior_as_arguments_not_via_config`
  does: it sets the boxes, presses the button, and asserts the values arrive as the stage's keyword
  arguments.
- **Slow-mark a real simulation that takes more than about a minute** (`@pytest.mark.slow`), so the
  fast gate stays fast.
