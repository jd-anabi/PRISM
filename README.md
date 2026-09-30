# PRISM

Research for running GFDT theory and experiments on a simulated biophysical model of the inner-ear hair-cell bundles. The application (**PRISM**) has two front ends over one core: a PySide6 desktop GUI and the command-line tool `python -m core`.

Start with the guide in [docs/guide/](docs/guide/): it explains how to use, run and maintain PRISM, with a reading path for each kind of reader.

## Installation

### Requirements

- **Python 3.12** recommended (validated). The pinned stack requires Python **≥ 3.11** (e.g. NumPy 2.3, SciPy 1.16); PyTorch 2.9 supports 3.10–3.13.
- **conda** (Miniforge / Miniconda / Anaconda) is recommended — it matches the development environment and provides the `ffmpeg` binary used for MP4 video export. A plain `venv` also works, but MP4 export then depends on a system `ffmpeg`.
- **git** to clone the repository.
- **GPU note:** on Windows/Linux the default install pulls the CUDA (`+cu130`) PyTorch build for NVIDIA GPUs; on macOS it uses the CPU/MPS build (there is no CUDA on Mac). See the per-platform notes below.

Clone the repository first (all platforms):

```bash
git clone https://github.com/jd-anabi/PRISM.git
cd PRISM
```

Then follow the section for your platform.

---

### Apple Silicon Mac (M1 / M2 / M3 / M4 — "M-series")

> **You must use a _native arm64_ Python.** PyTorch 2.9 ships **arm64-only** macOS wheels — there is no Intel (x86_64) macOS build. If you install with an x86_64 Python (e.g. an older Intel Anaconda), `pip install` fails with `No matching distribution found for torch==2.9.0`. Use **Miniforge**, which is arm64-native on Apple Silicon.

1. **Install Miniforge (arm64).** With Homebrew (which is arm64 on Apple Silicon):
   ```bash
   brew install miniforge
   ```
   Or without Homebrew:
   ```bash
   curl -L -o /tmp/Miniforge3-MacOSX-arm64.sh \
     https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh
   bash /tmp/Miniforge3-MacOSX-arm64.sh -b -p "$HOME/miniforge3"
   ```

2. **Create and activate an arm64 environment:**
   ```bash
   conda create -n biophys-env python=3.12 -y
   conda activate biophys-env
   ```

3. **Confirm the interpreter is arm64** — this must print `arm64`:
   ```bash
   python -c "import platform; print(platform.machine())"
   ```

4. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. _(Optional)_ **ffmpeg** for MP4 video export:
   ```bash
   conda install -c conda-forge ffmpeg -y
   ```

6. **Run:** `bash run.sh`

<details>
<summary><b>Troubleshooting: step 3 printed <code>x86_64</code> instead of <code>arm64</code></b></summary>

Your `conda` is pointing at an existing **Intel Anaconda** (common if you already had Anaconda installed — it auto-activates its `base` and shadows Miniforge on your `PATH`), so it created an x86_64 environment. Use Miniforge's conda by **full path** instead:

```bash
# Remove the Intel env that was created (this runs your Anaconda conda):
conda env remove -n biophys-env -y

# Create the env with Miniforge's conda explicitly (Homebrew install path shown):
/opt/homebrew/Caskroom/miniforge/base/bin/conda create -n biophys-env python=3.12 -y

# Activate it via Miniforge's activate script:
source /opt/homebrew/Caskroom/miniforge/base/bin/activate biophys-env

python -c "import platform; print(platform.machine())"   # -> arm64
pip install -r requirements.txt
```

If you installed Miniforge with the script, replace `/opt/homebrew/Caskroom/miniforge/base` with `$HOME/miniforge3`. In new terminals, re-activate with the same `source .../activate biophys-env` line before running the app (a fresh shell starts with Anaconda active and cannot see the Miniforge env).
</details>

---

### Intel Mac (x86_64)

> **Not supported by the pinned dependencies.** PyTorch discontinued Intel-macOS (x86_64) builds after **2.2.2**, so no `torch==2.9.0` wheel exists for Intel Macs and `pip install -r requirements.txt` will fail with `No matching distribution found for torch==2.9.0`. (This is the same error M-series users hit when they use an Intel Python.)

Options:

- **Recommended:** run the project on a supported platform — an Apple Silicon Mac, Linux, or Windows.
- **Advanced / untested:** relax the `torch` pin and source an Intel build from conda-forge. The pinned stack (`torch==2.9.0`, `sbi==0.26.1`) has not been tested against older Intel-compatible PyTorch and is not supported here.

---

### Linux

1. **Create and activate an environment** (conda recommended; `venv` also works):
   ```bash
   conda create -n biophys-env python=3.12 -y
   conda activate biophys-env
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   This installs the CUDA build `torch==2.9.0+cu130` (from the PyTorch index configured in `requirements.txt`). It uses an **NVIDIA GPU** if present and still runs on CPU otherwise (the wheel is large because it bundles the CUDA libraries).
3. _(Optional)_ **ffmpeg** for MP4 export: `conda install -c conda-forge ffmpeg -y`
4. **Run:** `bash run.sh`

**CPU-only / smaller install (no NVIDIA GPU):** install the CPU PyTorch build from the CPU index instead of the bundled CUDA build:
```bash
pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cpu
```
The `requirements.txt` torch/torchvision lines pin `+cu130` for Linux, so for a pure-CPU install, run the command above first and skip those two lines when installing the rest (see the comments in `requirements.txt`).

---

### Windows

1. **Create and activate an environment** (conda recommended — matches the development environment; Miniconda / Anaconda / Miniforge all work):
   ```bat
   conda create -n biophys-env python=3.12 -y
   conda activate biophys-env
   ```
2. **Install dependencies:**
   ```bat
   pip install -r requirements.txt
   ```
   This installs `torch==2.9.0+cu130` (CUDA build) for **NVIDIA GPUs**; it also runs CPU-only if you have no NVIDIA GPU (large download).
3. _(Optional)_ **ffmpeg** for MP4 export: `conda install -c conda-forge ffmpeg`
4. **Run:** double-click **`run.bat`**, or from a terminal:
   ```bat
   run.bat
   ```

**CPU-only / smaller install:** same as Linux — install `torch` / `torchvision` from `https://download.pytorch.org/whl/cpu` first and skip those two lines in `requirements.txt`.

---

## Running the app

Use the launchers below (they `cd` to the repository root for you). The working directory does not matter to the code: `core/config.py` resolves the `Resources/` inputs and the `Artifacts/` root from its own location, and `PRISM_RESOURCES` / `PRISM_ARTIFACTS` override them.

| Platform        | Launch the GUI                                 |
| --------------- | ---------------------------------------------- |
| macOS / Linux   | `bash run.sh`                                  |
| Windows         | `run.bat`                                      |
| Any OS (direct) | `KMP_DUPLICATE_LIB_OK=TRUE python -m core.gui` |

The launchers set `KMP_DUPLICATE_LIB_OK=TRUE` for you (on Windows, `set KMP_DUPLICATE_LIB_OK=TRUE` first, or `$env:KMP_DUPLICATE_LIB_OK="TRUE"` in PowerShell); torch and MKL ship two OpenMP runtimes under conda and without it the first simulation aborts with OMP Error #15 and no traceback.

On macOS/Linux, remember to activate the environment (`conda activate biophys-env`, or the `source .../activate biophys-env` line if you set up Miniforge by full path) before launching.

## The command-line tool

PRISM has two front ends over one core: the desktop window and the command-line tool, `python -m core <subcommand> [<mode>] <flags>`. The tool takes flags only and never prompts, so a run can be written down and repeated. Both call the same stages and the same FDT analysis; the diagnostics are reached from the tool alone, and its `artifacts` family is the twin of the window's Artifacts screen. Run it from the repository root, in the environment the window uses. No flag edits a configuration constant: each becomes an argument to a stage, to the configuration or to the store the tool builds (or, for `smoke --stages`, the choice of stages), so a flag you leave off gives the stage's own default, except `validate --seed` (the tool draws and records one) and `smoke` (which always passes its own small sizes; [the rule](docs/guide/command-line.md#conventions)). `--bounds` is required by every command that builds a configuration (all but `fdt`, `crossval`, `compare` and `artifacts`): the bounds file declares which parameters are inferred and in what order, and what a stored prior or posterior is checked against. `--chi` is off by default, so every chi command passes it, and a chain keeps it the same from prior to inference.

One row per subcommand, and per mode where a subcommand has modes; each name links to its section of the command-line page. The store's kinds are `prior`, `simulation`, `posterior`, `observation`, `calibration`, `inference`, `diagnostic` and `fdt`.

| command | what it does | what it writes |
|---|---|---|
| [`prior`](docs/guide/command-line.md#prior) | Builds a stability-screened prior over the bounds file's parameters | `prior` |
| [`train`](docs/guide/command-line.md#train) | Trains an amortized posterior on a stored prior | `posterior`, and the `simulation` cache as it runs |
| [`validate`](docs/guide/command-line.md#validate) | Calibrates a posterior without an observation: SBC, TARP, one verdict | `calibration` |
| [`infer`](docs/guide/command-line.md#infer) | Runs a posterior on one observation, simulated from a cell or built from recordings | `observation`, `inference` |
| [`tsnpe`](docs/guide/command-line.md#tsnpe) | One narrowing round: a region around an observation, then training on the prior restricted to it | `posterior` (not amortized), and its `simulation` cache |
| [`sbc`](docs/guide/command-line.md#sbc) | SBC repeated on one posterior, to show its run-to-run spread | `diagnostic` |
| [`identifiability rotation`](docs/guide/command-line.md#identifiability-rotation) | Decomposes a trained posterior's Fisher eigenbasis; simulates nothing | `diagnostic` |
| [`identifiability laplace`](docs/guide/command-line.md#identifiability-laplace) | Laplace marginal SD of every parameter at a cell's ground truth and at draws from the training prior | `diagnostic` |
| [`identifiability jacobian`](docs/guide/command-line.md#identifiability-jacobian) | The degeneracy map at a cell's ground truth; needs no posterior | `diagnostic` |
| [`ablation`](docs/guide/command-line.md#ablation) | Which conditioning channels the trained flow can see; simulates nothing | `diagnostic` |
| [`probes band`](docs/guide/command-line.md#probes-band) | Do the configured chi band and drive hold for this cell? | `diagnostic` |
| [`probes mask`](docs/guide/command-line.md#probes-mask) | Why training throws probes out, over a prior | `diagnostic` |
| [`probes drive`](docs/guide/command-line.md#probes-drive) | How hard a lab can drive this cell | `diagnostic` |
| [`smoke`](docs/guide/command-line.md#smoke) | Every stage end to end at tiny sizes; run it on the card after changing code that moves tensors | `prior`, `posterior`, `calibration`, `observation`, `inference` (one per stage that builds it) and, with `--checkpoint`, the `simulation` cache, in a store of its own: a fresh temporary folder, or the one `--store-root` names |
| [`fdt`](docs/guide/command-line.md#fdt) | The effective-temperature analysis for one cell: sanity checks, then the production sweep | `fdt`, kept unfinished after a cancel |
| [`crossval`](docs/guide/command-line.md#crossval) | The FDT parameter-sweep study on the Nadrowski model: an S sweep, then a `T_a/T` sweep | two `fdt` records |
| [`compare cells`](docs/guide/command-line.md#compare-cells) | Two or more single-cell runs' ratio curves on one axis | `fdt` (a comparison) |
| [`compare repeats`](docs/guide/command-line.md#compare-repeats) | Several runs of one cell, their spread drawn as a band | `fdt` (a comparison) |
| [`compare renormalise`](docs/guide/command-line.md#compare-renormalise) | One run's ratio recomputed with a supplied normalisation constant | `fdt` (a comparison) |
| [`compare sweeps`](docs/guide/command-line.md#compare-sweeps) | Two sweep records together, and a slice of both at one operating point | `fdt` (a comparison) |
| [`artifacts list`](docs/guide/command-line.md#artifacts-list) | One line per record of a kind, or of all eight | nothing |
| [`artifacts show`](docs/guide/command-line.md#artifacts-show) | One record's manifest and the log of the run that wrote it | nothing |
| [`artifacts note`](docs/guide/command-line.md#artifacts-note) | Sets or clears one record's note | nothing new (the note in the manifest) |
| [`artifacts rm`](docs/guide/command-line.md#artifacts-rm) | Deletes one record; refused while another record depends on it | nothing (removes a record) |
| [`artifacts sweep`](docs/guide/command-line.md#artifacts-sweep) | Removes leftover directories with no manifest; a dry run until `--yes` | nothing (removes leftovers) |
| [`artifacts summary`](docs/guide/command-line.md#artifacts-summary) | The lineage report: a record, then its parents back | nothing (`--out` writes the report to a file) |

Four groups of flags mean the same wherever they appear:

- [Configuration](docs/guide/command-line.md#configuration-flags): `--bounds`, `--model`, `--chi` / `--no-chi`, `--chi-k`, `--device`. Every command that builds a configuration takes them; the `probes` modes take `--bounds`, `--model` and `--device` only.
- [Naming](docs/guide/command-line.md#naming-flags): `--name`, `--note`. Every command that writes a record but `smoke`; a taken name is refused before anything is spent.
- [Training cache](docs/guide/command-line.md#training-cache-flags): `--resume auto|require|never`, `--new-run`. On `train`, `tsnpe` and `smoke`; a run resumes its own simulation cache, and `--new-run` consents to a new one beside a cache exactly one setting away.
- [Acceptance](docs/guide/command-line.md#acceptance-flags): `--accept-truncated` (on `validate`, `infer`, `tsnpe`, `sbc`, `identifiability rotation`, `identifiability laplace` and `ablation`, the commands that load a posterior) and `--accept-other-observation` (on `infer` only). The only ways past the two refusals on a narrowed (`tsnpe`) posterior; each use is recorded in what the run writes.

Four environment variables, the ones `python -m core --help` names; the last two are never flags: they change how a batch is planned in memory, or how often memory is reported, and never the rows it produces ([details](docs/guide/command-line.md#environment-variables)):

- `PRISM_RESOURCES`: the inputs root (default `<repo>/Resources`).
- `PRISM_ARTIFACTS`: the records root (default `<repo>/Artifacts`). Every subcommand writes there except `smoke`, and `fdt` and `crossval` given `--store-root`.
- `PRISM_VRAM_CEILING_GIB`: the GiB one simulation batch may plan to occupy; 0 or unset is automatic; read afresh for every batch plan.
- `PRISM_MEM_LOG_EVERY`: the batches between memory log lines; read once, when the simulation pipeline is first imported.

Exit codes ([details](docs/guide/command-line.md#exit-codes)): `0` success, and a result the command judged, such as a calibration whose verdict is FAIL; `1` a refusal, printed as `prism <subcommand>: refused: <message>`, usually ending with the flag that answers it (an error raised as a plain `ValueError`, or a missing file that no check names, adds its class before the message and `[raised at <file>:<line>]` after it), or a bug; `2` a usage error; `130` Ctrl-C.

Three examples, run from the repository root in one shell. The first line points `PRISM_ARTIFACTS` at a scratch folder, so the real `Artifacts/` gains nothing; delete that folder when you are done.

**A stage chain.** The prior builds at its full default size, about a minute and a half on the card and longer on the CPU; training, calibration and inference run at sizes that show the chain working and mean nothing more. The last line prints the inference's lineage back to the prior.

```bash
export PRISM_ARTIFACTS="$HOME/prism-scratch"    # PowerShell: $env:PRISM_ARTIFACTS = "$HOME\prism-scratch"
python -m core prior    --chi --bounds Resources/Bounds/nadrowski/master.txt --name demo_prior
python -m core train    --chi --bounds Resources/Bounds/nadrowski/master.txt --prior demo_prior --num-runs 8 --run-size 32 --max-epochs 5 --name demo_posterior
python -m core validate --chi --bounds Resources/Bounds/nadrowski/master.txt --posterior demo_posterior --n-cal 100 --seed 7
python -m core infer    --chi --bounds Resources/Bounds/nadrowski/master.txt --posterior demo_posterior --cell Resources/Cells/nadrowski/master_spont.txt --t-obs 4.5 --name demo_inference
python -m core artifacts summary inference demo_inference
```

**A probe check on one cell.** Do the configured chi band and drive hold for it, and how hard can it be driven? Each writes one `diagnostic` record, and `python -m core artifacts show diagnostic spont_band` prints its manifest.

```bash
python -m core probes band  --bounds Resources/Bounds/nadrowski/master.txt --cell Resources/Cells/nadrowski/master_spont.txt --name spont_band
python -m core probes drive --bounds Resources/Bounds/nadrowski/master.txt --cell Resources/Cells/nadrowski/master_spont.txt --name spont_drive
```

**An FDT run twice, then compared.** Each run draws and records a seed of its own, so the two are independent repeats of one cell. The sizes here (4 drive frequencies, 8 trajectories each, no sanity checks) show the commands working and mean nothing more; a real run takes the defaults, 60 frequencies of 256 trajectories after the sanity checks, and is long, on the CPU only.

```bash
python -m core fdt --cell Resources/Cells/nadrowski/master_spont.txt --n-freqs 4 --ensemble-m 8 --skip-sanity --name spont_fdt_a
python -m core fdt --cell Resources/Cells/nadrowski/master_spont.txt --n-freqs 4 --ensemble-m 8 --skip-sanity --name spont_fdt_b
python -m core compare repeats --record spont_fdt_a --record spont_fdt_b --name spont_fdt_repeats
```

[The command-line page](docs/guide/command-line.md) lists every subcommand, mode and flag with what each writes, refuses and prints. `python -m core <subcommand> --help` (`python -m core <family> <mode> --help` for a mode) lists a command's flags with their defaults, imports no torch and needs no graphics card.
