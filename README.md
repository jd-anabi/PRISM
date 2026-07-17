# GFDTResearch

Research for running GFDT theory and experiments on a simulated biophysical model of the inner-ear hair-cell bundles. The application (**PRISM**) is a PySide6 desktop GUI.

## Installation

### Requirements

- **Python 3.12** recommended (validated). The pinned stack requires Python **≥ 3.11** (e.g. NumPy 2.3, SciPy 1.16); PyTorch 2.9 supports 3.10–3.13.
- **conda** (Miniforge / Miniconda / Anaconda) is recommended — it matches the development environment and provides the `ffmpeg` binary used for MP4 video export. A plain `venv` also works, but MP4 export then depends on a system `ffmpeg`.
- **git** to clone the repository.
- **GPU note:** on Windows/Linux the default install pulls the CUDA (`+cu130`) PyTorch build for NVIDIA GPUs; on macOS it uses the CPU/MPS build (there is no CUDA on Mac). See the per-platform notes below.

Clone the repository first (all platforms):

```bash
git clone https://github.com/jd-anabi/GFDTResearch.git
cd GFDTResearch
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

Always launch from the **repository root** — the launchers `cd` there for you, and `core/config.py` builds all `Resources/` paths from the current working directory.

| Platform        | Launch the GUI            |
| --------------- | ------------------------- |
| macOS / Linux   | `bash run.sh`             |
| Windows         | `run.bat`                 |
| Any OS (direct) | `python -m core.gui`      |

Use `python -m core` instead for the interactive CLI. On macOS/Linux, remember to activate the environment (`conda activate biophys-env`, or the `source .../activate biophys-env` line if you set up Miniforge by full path) before launching.
