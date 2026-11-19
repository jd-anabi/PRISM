# Bringing recordings

Checked against commit d1d8bbfb.

This page is for a lab member who brings a preparation and its recordings to PRISM. It covers the
files that describe a cell, which bounds file a cell is checked against, the three observation modes and
the recordings each needs, how long to record, what chi mode assumes about a cell and how to check
it, and how to define a model of your own. Read [Getting started](getting-started.md) first. Every
flag is described in [The command-line tool](command-line.md), and every control of the window in
[The window](window.md); the commands below are run from the repository root.

## Input files

The hand-edited inputs live under `Resources/`, one folder per kind and, inside it, one folder per
model ([Inputs and records](getting-started.md#inputs-and-records)). The bounds, cell and units
files share one layout: a line that starts with `#` opens a section, and each line under it holds
one entry, except in the units file, whose one line lists its tokens separated by spaces. The
examples are the Nadrowski master files.

**Bounds files**, `Resources/Bounds/<model>/<name>.txt`, say which parameters are inferred, in what
order, and over what box. This is `master.txt`:

```text
# Non-dimensional Parameters
k in (0.01, 5)
lam in (0.1, 50)
f_max in (0.05, 30)
tau in (0.0001, 2)
tau_c in (0.005, 5)
s in (0, 3)
delta_E in (0, 30)
beta in (0.5, 100)
n in (10, 300)
temp in (0.05, 10)

# Dimensional Parameters
x_scale in (10, 100)
t_scale in (1, 40)
f_scale in (1, 1000)

# Forcing Parameters
amp in (0, 6)
freq in (0.00226, 0.226)
phase in (0, 6.28319)
offset in (-50, 50)
```

- The three section headers are `# Non-dimensional Parameters`, `# Dimensional Parameters` and
  `# Forcing Parameters`, and every entry is a line `name in (lo, hi)`.
- The first two sections are the inferred parameters. The non-dimensional section binds by
  position: the simulator hands its values to the model's parameters in the file's order, so the
  same numbers in another order are another model. The Dimensional section is read by name. The
  order of both is recorded with a posterior, which loads only under a bounds file in the same
  order. Every command of the inference chain prints the order it read on its `[cfg]` lines
  ([Conventions](command-line.md#conventions)).
- The Forcing section declares a drive. It is not inferred: it names the drive a recording was made
  at, and its box is the range training draws drives from in forced mode. Whether a bounds file has
  one decides the observation mode ([Observation modes](#observation-modes)).
- A line of any other shape in one of the three sections, or one whose brackets hold other than two
  numbers, is refused, naming its line by its number and its words
  (`core.Helpers.file_manager.parse_bounds_file`): the command line ends the refusal `(--bounds)`, and
  the Prior step says it under its list of bounds files. A `#` line is always read as a header, by
  the words it contains: one that names no section ends the current one, so the lines under it are
  skipped without a word, and a misspelt header can file the lines under it in the wrong section. A
  comment line or a misspelt header can therefore drop parameters or misfile them silently: write no
  comments, and
  compare the order the `[cfg]` lines print with the file.
- No units are written here. The model's units file declares them once for all its files; a unit
  written after a name, as in `x_scale (nm) in (10, 100)`, is read and ignored.
- `master_tier1.txt` is `master.txt` with `T in (280, 310)`, a temperature in kelvin, in place of
  `f_scale`: the force scale is then derived from the temperature instead of inferred
  ([The tier-1 constraint and temperature](science.md#the-tier-1-constraint-and-temperature)).

**Cell files**, `Resources/Cells/<model>/<name>.txt`, hold one preparation's values: its initial
conditions and one value per parameter, as `name = value`. This is `master_weak.txt`:

```text
# Non-dimensional Initial Conditions
x_init = -0.11
xa_init = -1.32
c_init = 0.5

# Non-dimensional Parameters
k = 0.8
lam = 3.57
f_max = 1.32
tau = 0.027
tau_c = 0.268
s = 0.95
delta_E = 10
beta = 14.1
n = 50
temp = 1.5

# Dimensional Parameters
x_scale = 62.14
t_scale = 3.73
f_scale = 10

# Forcing Parameters
amp = 0.2
freq = 0.03164
phase = 0
offset = 0
```

- A cell gives a value for every parameter the bounds file declares, and a missing one is refused,
  by name (`core.cli.load_and_validate_gt`, through `core.sim_config.SimConfig.inject_ground_truth`).
  The inferred values must also lie inside the box; the drive values need not.
- Values the bounds file does not declare are ignored, and the tool names them: on a `[cfg]` line
  for the commands that measure a cell, and in a warning from `infer --cell`. That is what lets one
  cell serve several bounds files. `master_spont.txt` carries `f_scale` and a Forcing section, and the
  bounds file of the same name, which declares neither, simply ignores them.
- A cell meant for the tier-1 box gives `T` instead of `f_scale`, as `master_spont_tier1.txt` does.

**Units files**, `Resources/Units/<model>/units.txt`, hold a `# Units` header and the unit tokens.
This is the Nadrowski one:

```text
# Units
nm ms pN kHz
```

- The tokens declare what the numbers in the model's bounds and cell files, and in your recordings,
  are written in. Nothing is converted. For Nadrowski, `x_scale = 62.14` is 62.14 nm per model unit
  of length, `t_scale = 3.73` is 3.73 ms per model unit of time, and a cell's `freq = 0.03164` is
  0.03164 cycles per millisecond, 31.64 Hz.
- Tokens are matched to quantities by their dimension, not their order. A time token is required.
- A drive frequency is read as the reciprocal of the time unit, whatever frequency token the file
  declares. A frequency token, if the file declares one, must be the reciprocal of the time token
  (kHz with ms, Hz with s); it is only a label and may be left out. The window's step 2, Prior, says
  so under "Before you start" (marked Attention) when the configuration in use declares tokens that
  disagree (`core.sim_config.SimConfig.check_unit_consistency`); the command-line tool does not check.
- A force token is needed wherever a force in newtons is converted: a forced-mode drive, the chi
  drive amplitude, and a tier-1 box.
- The window's step 1, Model and options, can take the tokens typed in place of the file ("Typed
  here"), or a units file from any folder ("Choose a units file…"), which it reads at once and
  refuses in words if it holds no units ([The inference settings](window.md#the-inference-settings)).

**Model files**, `Resources/Models/<NAME>.json`, define user models. The model builder writes them,
together with a first bounds, cell and units file for the model
([Defining a model of your own](#defining-a-model-of-your-own)).

**A file's model is its folder.** A cell or bounds file belongs to the model its parent folder
names: `Cells/nadrowski/` holds Nadrowski cells. The inference chain takes the model from the bounds
file's folder, and `--model` overrides it
([Configuration flags](command-line.md#configuration-flags)).

## Run plans

A run plan is a text file of the science settings of the window's six steps, so that a
training can be set up again, or set up the same way for another cell, in one press. It is an input
like a bounds or cell file: it lives at `Resources/Plans/<model>/<name>.txt`, one folder per model with
the model's name in lower case (`core.config.PLANS_PATH`), it is hand-edited, and it is tracked in git.
No plan ships, and the folder is made when the first plan is saved. A plan saved from the window is
named with the letters a to z in either case, digits, `_` and `-`, at most 40 characters; a file
named by hand is listed and read whatever its name. Only the window reads plans
(`core.plans` is the reader, with the standard library alone); the command-line tool reads none, and
no stage does. A plan only fills the forms: it never starts a run, applies a step or changes which
record is picked.

A plan reads like a cell file, `#` heading lines and `name = value` lines. What stands under the first
heading is free text and is shown as the plan's description; what follows white space and `#` on a
setting line is a remark and is never read:

```
# About this plan
The temperature box at the larger network.

# Model and options
chi_mode = on                  # Kind of observation: Frequency response

# Prior
bounds = master_tier1.txt      # Bounds file

# Training
num_runs = 10000               # Batches
hidden_features = 256          # Network width
```

- **The headings** are About this plan, Model and options, Prior, Training, Calibration, Inference
  and Narrowing round, and each setting stands under its own: a name under another heading is a
  problem. Each name is a setting's key in the refusal registry (`core.refusals.FIELDS`), the same key
  that names the setting in a refusal. The window fills each heading into its own step: Calibration
  into the calibration's form on Checks, Inference into its posterior draws, Narrowing round into the
  round's form. **Save as a plan…** writes the settings of every step, with no description, so saving
  over a plan that holds one drops it, after asking.
- **Values.** A setting the file does not hold means the built-in value. `chi_mode` and
  `reparam_rotate` take `on` or `off`; `bounds` is the name of a file in the model's bounds folder (a
  sub-folder is written with a forward slash); `units` is the units' words; a whole-number setting is
  digits alone (10000, never 10000.0 or 1e4), and a number takes a point as its decimal mark,
  whatever Windows' number format is (0.0005, or 5e-4). A value
  that breaks the setting's own rule is refused with that rule's sentence.
- **Strict.** The cell file's reader skips what it does not know, and the bounds file's skips the lines
  under a header it does not know; the plan's reader does
  not, because a plan decides days of work on the card. Every line is understood or named as a
  problem with its line number, all of them at once, with the nearest known name when one is close, and
  a plan with any problem is refused whole. A setting no plan can hold is refused with the reason: it
  is fixed by measurement (the chi band and drive), belongs to this computer (the device), to one
  run (the name, the note, the seed), is a consent, is a pick of a record or a recording and not a
  setting, is the folder the plan stands in (the model), or is no setting of the inference steps.
- **Checked beyond the file.** The window also checks what the reader cannot: that the bounds file is
  in the model's bounds folder, with the same capitals, and that the units resolve.

In the window, step 1's **Plan** row lists the model's plans, opens on "No plan" at every launch
(the plan last used with the model, and when, is remembered only as a hint beside the list), shows a plan's description and what it
holds, and **Fill the forms from this plan** fills every step's form, asking first what it would replace;
**Save as a plan…** writes every setting of every step that differs from its usual value, the
bounds file chosen in the list (never limits typed by hand), each with a remark naming its label.
Under each setting a small word says where its value came from (usual, changed from the usual, kept
from last time, "from plan NAME", or changed from the plan), and each step's "Before you start" says the plan in use and each
setting that has since departed from it. A run's history records the plan, with the file's hash.
Commit a plan before a long run: a plan that is saved or edited and not committed makes every record
the run writes read as made from a changed tree.

## Which bounds file a cell is checked against

A cell is checked against the bounds file it is given, and against no other: no run looks one up from
a cell alone. Every inference, diagnostic and probe command takes `--bounds`
([Configuration flags](command-line.md#configuration-flags)), and in the window step 2, Prior,
chooses it with its Bounds file list ([The Parameter Inference tabs](window.md#the-parameter-inference-tabs)). A
cell given to `infer --cell`, to `probes` or to step 5, Inference, is checked against that bounds file
by `SimConfig.inject_ground_truth` (`core.sim_config`): a cell that lacks a parameter the bounds file
infers, or puts one outside its box, is refused (field `cell`); values the bounds file does not
declare are ignored and reported, so one cell serves several bounds files, and several cells can
share one box. Step 5 reads the cell it is handed against the same rule and says what it finds before
its run starts.

The tier-1 cell `master_spont_tier1.txt` goes with
`--bounds Resources/Bounds/nadrowski/master_tier1.txt`, whose dimensional section infers `T` in
`f_scale`'s place. Given `master.txt` instead, it is refused: `master.txt` infers `f_scale`, and a
tier-1 cell has none ("Cell file is missing rescale parameters the bounds file requires: f_scale.").

## Observation modes

The bounds file and the chi switch together choose the observation mode
(`core.sim_config.SimConfig.observation_mode`), and the mode fixes what one observation is. A
posterior loads only in the mode it was trained in, so choose the mode before you record. A
posterior's bounds file and mode are in its record: `python -m core artifacts show posterior <ref>`
prints `inputs.bounds` and `config.mode`.

| mode | chosen by | what you record | what you note at the bench |
|---|---|---|---|
| spontaneous | a bounds file with no Forcing section, chi off | one passive (undriven) recording | its length |
| forced | a bounds file with a Forcing section, chi off | a passive recording, and one driven recording of the same length | the length, and the drive: one value per forcing parameter the bounds file declares, in SI units (amplitude and offset in newtons, frequency in hertz, phase in radians) |
| chi | chi on, whatever the bounds file's Forcing section | a passive recording, and one or more single-tone driven recordings, up to the posterior's probe slots (12 unless it was trained with another number) | the length, the frequency each driven recording was actually driven at, in hertz, and the physical drive amplitude, in newtons: one value shared by every probe, so drive every probe at the same amplitude |

- **Forced mode.** Keep the drive inside the bounds file's Forcing box: training drew its drives
  from that box, and nothing checks a recording's drive against it. The box is written in the
  cell's units; `master.txt`'s spans 0 to 6 pN and 2.26 to 226 Hz. The phase you give is the
  drive's phase at the driven recording's first sample
  ([Recording files and drive frequencies](#recording-files-and-drive-frequencies)).
- **Chi mode.** A Forcing section, if the bounds file has one, is ignored: the probes sit at the
  frequencies you drove at. χ is the response divided by the drive: the lock-in divides by the
  amplitude you give, so give the amplitude actually applied. Training drove every probe at the
  configured chi drive, and an active bundle does not respond linearly, so the |χ| a cell gives, and
  its scatter, change with the amplitude: drive as near that amplitude, in newtons for this cell, as
  you can ([The chi assumptions, and checking them](#the-chi-assumptions-and-checking-them) gives
  it). A much stronger drive can also capture the bundle, which then abandons its own rhythm and
  follows the drive, so that χ measures the drive rather than the bundle. Start each driven
  recording with the drive at the top of its cycle
  ([Recording files and drive frequencies](#recording-files-and-drive-frequencies)).

In the window, step 1's "Kind of observation" list is the chi switch ("Frequency response" is chi
mode) and step 2's Bounds file list chooses the bounds file. Step 5, Inference, has two modes, "A
simulated cell" and "Your own recordings", and the second shows the form the posterior in use reads,
with a sentence saying which kind that is; with no posterior in use it says so and shows no form.
For a spontaneous posterior it asks for the **Spontaneous recording** alone. For a forced one it
asks for that and the **Driven recording**, with a row for each parameter of the drive in SI units.
For a frequency response it asks for the **Spontaneous recording** and a table of **Driven
recordings**, each row a file and the frequency it was driven at, in Hz, and for the **Drive
amplitude**, in N. **Add a driven recording** adds a row, and a row's remove button asks first, with
"Keep it" as the safe answer, when the row holds a file or a frequency you typed. "Before you start"
says how many driven recordings the posterior takes at most, and how many to remove when there are
more. **Plan the driven recordings…**, once the spontaneous recording and the recording length are
given, measures the cell's own frequency and says the band, fills only the frequencies left blank
from the band, and says under each row what the engine will do with it (used whole, cut at the
lock-in limit, too short to be used, or refused, with the reason). The **Recording length**, in s, and
the **Drive amplitude** open blank at every launch: type them for each recording you bring
([The inference settings](window.md#the-inference-settings)). The observation the run makes needs a
name, and the inference may have one. On the command line,
[infer](command-line.md#infer) lists the flags each mode takes and refuses. One example per mode;
the recording paths and posterior names are placeholders, the forced drive is `master_weak.txt`'s in
SI units, and the chi amplitude is the configured chi drive on a master cell:

```bash
python -m core infer --bounds Resources/Bounds/nadrowski/master_spont.txt --posterior my_spont_posterior --spont passive.csv --t-obs 4.5
python -m core infer --bounds Resources/Bounds/nadrowski/master.txt --posterior my_forced_posterior --spont passive.csv --forced driven.csv --drive amp=2e-13 --drive freq=31.64 --drive phase=0 --drive offset=0 --t-obs 4.5
python -m core infer --chi --bounds Resources/Bounds/nadrowski/master.txt --posterior my_chi_posterior --spont passive.csv --forced probe_a.csv@0.8 --forced probe_b.csv@2.4 --forced probe_c.csv@6.5 --f0-si 1.5e-12 --t-obs 4.5
```

## Recording files and drive frequencies

- **File types.** A recording is a `.npy` file holding one 1-D array, or a `.csv` file of
  comma-separated numbers whose last column is the trace. Earlier columns, such as a time column, are
  ignored, and a header row of column names cannot be read
  (`core.Helpers.file_manager.load_experimental_data`).
- **Sampling.** Every recording must be sampled once a millisecond, 1,000 samples a second
  (`core.config.DT_EXP_S`, the camera frame interval the network was trained on). Nothing reads a
  rate from the file: resample a recording made at another rate first, or every frequency and time
  in it is misread.
- **A recording is read as if its samples are exactly 1 ms apart.** Training simulates every cell
  on exact frames, and PRISM does not yet read a recording at its camera's own measured interval. A
  driven response locked in over the ceiling's 64 cycles reads at |sinc(π · 64 · e)| of its size
  when the camera's interval is off by a fraction e: about 0.45 when it is 1 % off, 0.993 when it is
  0.1 % off (over 20 cycles, 0.94 and 0.9993). An interval off by 1 % also moves every feature
  measured in time units by log 1.01. A bench recording is to be read at its camera's own measured
  frame interval before any inference on it: know your camera's real frame interval before you
  trust a reading.
- **Every driven recording starts with the drive at a known point of its cycle**, because the
  lock-in counts time from the recording's first sample, as training does:
  - a frequency-response probe starts with the drive at the top of a cosine: at its first sample the
    drive is at its largest, F0 · cos(2π · f · t) with t = 0 there;
  - a forced-mode recording's drive is amp · sin(2π · freq · t + phase) + offset with t = 0 at its
    first sample, so the phase you give is the drive's phase at that sample.

  A recording that starts elsewhere in the cycle reads every phase off by the drive's phase at its
  first sample (each probe's cos and sin channels; the forced response's phase features), and a
  probe's |χ| a little too unless it holds a whole number of cycles, and nothing warns of it.
  Starting the camera on the drive's own trigger is the simplest way to meet it.
- **Length.** T_obs (`--t-obs`, or the Recording length box, in s) is the recording's length in seconds. The
  passive recording should hold T_obs times 1,000 samples, and a warning says so when it is more than
  one sample off. In forced mode the driven recording must be exactly as long as the passive one,
  and a mismatch is refused. In chi mode each driven recording may have its own length, and is
  locked in over its whole length up to the lock-in ceiling.
- **Displacement unit.** The trace is read in the length unit the units file declares, nanometres
  for Nadrowski: a simulated trace is the model's observable times `x_scale`, in that unit, and a
  recording is compared with it as it stands.
- **Drive frequencies are given in hertz**, both a chi probe's (`--forced PATH@HZ`, or the frequency
  box of a driven recording's row) and a forced-mode drive's (`--drive freq=`). PRISM converts them into the
  cell's own frequency unit, the reciprocal of its time unit
  (`core.sim_config.SimConfig.freq_si_to_cell`): 31.64 Hz is 0.03164 cycles per millisecond in a
  Nadrowski cell.
- **Give the frequency you actually drove at**, not the one you aimed for. The lock-in locks where
  it is told, and at a wrong frequency its answer decays like a sinc function of the error: an error
  of a fraction of 1/T_obs spoils the estimate while every number still looks plausible, and nothing
  in PRISM detects it.

Each chi probe is then masked, truncated or refused by one rule set (`core.SBI.chi.probe_verdict`,
applied in `core.SBI.observations.build_experiment_obs_chi`). Ω₀ is the peak frequency of the
cell's own oscillation, measured from the passive recording.

- **Masked**, with a warning that says how long to record at that frequency: a probe with fewer than
  two drive cycles in its recording. It stays in the set and contributes nothing. Training masks such
  probes too, so the network has learned to read a set with probes missing, and refusing would throw
  away an observation it handles.
- **Truncated**, with a warning: a probe with more drive cycles than the lock-in ceiling, 64 cycles
  by default. The cycles up to the ceiling are used and the rest of the recording is dropped.
- **Refused**, because each is a mistake to fix rather than a limit of the recording: a frequency
  that is not finite or not positive; a probe at or above 0.9 times the sampling limit, 450 Hz at
  1,000 samples a second; a probe outside the band, judged against this recording's own Ω₀ with a
  margin (about 0.0225 to 0.4 times Ω₀ is accepted); a passive recording with no usable peak; more
  driven recordings than the posterior's probe slots (12 unless it was trained with another number);
  and a set in which no probe survives.

The window's **Plan the driven recordings…**, on step 5's frequency-response form, applies the same
rules before you make the driven recordings: from the passive recording it measures Ω₀ and gives the
band in hertz, and under each driven recording it says whether the recording is used whole, cut at
the lock-in ceiling, masked as too short (with how long it must be at its frequency) or refused. The
command line has no equivalent: no subcommand measures Ω₀ from a recording.

## Recording length

Training does not see every length from 1 s to 60 s at every t_scale, and a lab member has to allow
for that.

- Training draws each batch's length T_obs log-uniformly from `core.config.T_MIN_EXP_S` to
  `core.config.T_MAX_EXP_S`, 1 s to 60 s, together with its t_scale. It keeps a batch only if the
  batch's fine integration steps fit both the per-batch cost cap (`core.config.N_ND_MAX`, 300,000
  steps) and the pre-simulated time grid, whichever is shorter (`core.SBI.pipeline._batch_schedule`),
  and a smaller t_scale needs more fine steps per recorded sample. So the longest recording training
  holds depends on t_scale. On the master bounds files, whose t_scale box runs from 1 to 40:
  - about 7.4 s at t_scale 1, the bottom of the box;
  - 26.9 s at the master cells' t_scale of 3.73 (`core.diagnostics.probe_math.training_ceiling_s`);
  - the 60 s draw limit toward the top of the box.
- On the master boxes, a recording no longer than about 7.4 s is therefore inside the training
  range whatever the preparation's t_scale.
- What is checked for you:
  - both inference paths warn when T_obs is outside 1 s to 60 s
    (`core.orchestrator.simulated_inference`, `core.orchestrator.experimental_inference`);
  - a simulated observation also warns when it is longer than training holds at the cell's own
    t_scale (`core.orchestrator.generate_observations`);
  - a recording carries no t_scale, so a passive or chi recording is checked against 1 s to 60 s
    only. In forced mode the recording's builder also warns, when the length is beyond what the
    bottom of the t_scale box allows, the smallest t_scale training covered at that length
    (`core.SBI.observations.build_experiment_obs`), which you must weigh against what you expect of
    the preparation.
- So check the length yourself. `probes band` measures a cell at lengths up to that cell's own
  ceiling by default, and marks with a warning any length you give beyond it
  ([probes band](command-line.md#probes-band)).

**Chi mode has a shortest useful length too.** The band's lowest probe (0.03 Ω₀) first reaches the
two-cycle floor at one length, T* = 2 / (0.03 Ω₀), about 66.7 / Ω₀ in seconds when Ω₀ is in hertz;
the band's two edges are ten times apart, so there the highest probe (0.3 Ω₀) has twenty cycles,
well under the default 64-cycle ceiling. Below T*, the probes at the bottom of the band are masked;
above it, every probe is used, and those at the top are truncated to the ceiling only once they pass
64 cycles, which costs nothing. Record for longer than T*. On the `master_weak` cell T* is 2.93 s.
`identifiability jacobian --chi` prints it for any Nadrowski cell, on the line that begins `low edge`
([identifiability jacobian](command-line.md#identifiability-jacobian)); the same command also
measures a whole degeneracy map:

```bash
python -m core identifiability jacobian --chi --bounds Resources/Bounds/nadrowski/master.txt --cell Resources/Cells/nadrowski/master_weak.txt --t-obs 4.5
```

The window's Plan the driven recordings… says how long a driven recording that is too short must be
at its frequency. This project's own examples and checks use 4.5 s, above T* on the master cells and
under the 7.4 s every t_scale allows.

## The chi assumptions, and checking them

Chi mode rests on four settings in `core/config.py`, chosen by measurements on the master cells
([Chi probe design](science.md#chi-probe-design) has the measurements). The band, the drive and the
ceiling were measured on the simulator's old clock, which ran the master cells 2.6 % fast, and again
on the record clock: the band and the drive hold, and the ceiling is 64 cycles, where on the old
clock it was 20:

| setting | value | what it means for a recording |
|---|---|---|
| `CHI_FREQ_BOUNDS` | (0.03, 0.3) | the band: probes lie between 0.03 and 0.3 times the observation's own Ω₀, so the band moves with each cell |
| `CHI_F0` | 0.15 | the drive amplitude in model units: 0.15 times the cell's force scale, 1.5 pN on `master.txt`'s cells, whose `f_scale` is 10 pN. On the tier-1 box the force scale is derived from the temperature ([The tier-1 constraint and temperature](science.md#the-tier-1-constraint-and-temperature)), so for the values in `master_spont_tier1.txt` it is about 47.0 pN and the drive about 7.0 pN; a chi training run prints the drive its prior implies on its second `[tier1]` line. Strong enough for a reproducible lock-in, and weak enough to leave the bundle oscillating on its own |
| `CHI_MIN_CYCLES` | 2 | the floor: a probe with fewer drive cycles is masked |
| `CHI_MAX_CYCLES` | 64 | the lock-in ceiling: each probe is locked in over at most its first 64 cycles. A longer lock-in reads χ more reproducibly, but it is more selective in frequency, so it asks for the camera's frame interval to be exact (above), and past 64 cycles the gain flattens while that cost keeps growing |

The band, the drive and the cycle floor come only from `core/config.py`. The lock-in ceiling, and
the number of probe slots beside it, also have boxes on step 1, Model and options ("Lock-in limit" and "Slots for driven recordings")
([The inference settings](window.md#the-inference-settings)); a posterior records both, and the
store refuses a mismatch when it loads one. The command-line tool and the `probes` checks always use
`core/config.py`'s values.

A preparation whose oscillation is much sharper or noisier than the master cells' may not share
these values. The `probes` checks measure whether it does, from the Checks step of the window or
the command line; they
replaced the scripts that first measured the values
([Where the old scripts went](command-line.md#where-the-old-scripts-went)). Each takes the cell file
that describes your preparation, or, for `mask`, a prior. The file names below are the master
examples.

A cell file must give a value for every parameter the bounds file declares, each inferred one inside
its box (`core.sim_config.SimConfig.inject_ground_truth` refuses otherwise). The checks judge the
values you give, not the preparation. For a preparation whose values are not known, a master cell or
the medians an earlier inference recorded (`results.posterior_summary`) are stand-ins, and each
verdict, and the physical drive in newtons, is only as good as the stand-in.

**Do the band and the drive hold for this cell?**

```bash
python -m core probes band --bounds Resources/Bounds/nadrowski/master.txt --cell Resources/Cells/nadrowski/master_spont.txt --name my_cell_band
```

It drives an ensemble of the cell's own noisy runs at probe frequencies across the band, and one
above it, at several recording lengths. It judges every point on three criteria: how much |χ|
varies between runs, the signal over the same lock-in on undriven runs, and capture, how much of the
cell's own oscillation survives the drive. It also reports the scatter of χ's phase, which it judges
only when given a threshold. Its two verdicts read "the configured band (0.03, 0.3) holds for this
cell" or "does not hold for this cell" (or "was not judged", when nothing measured could decide
it), and the same for the configured drive, 0.15. Read them with
the two caveats the record carries: the spread, signal and phase thresholds are conventions, and
only capture is physical evidence against a probe; and a probe with a low harmonic near the cell's
own peak is marked, because the harmonic's power inflates the not-captured measure and can hide
capture.

**How hard can you drive this cell?**

```bash
python -m core probes drive --bounds Resources/Bounds/nadrowski/master.txt --cell Resources/Cells/nadrowski/master_spont.txt --name my_cell_drive
```

The bounds file must have a Forcing section. The check drives the cell off its own peak at a range
of strengths, and reports the strongest at which the bundle still oscillates freely and the weakest
at which the drive captures it, each in model units and in the cell's force unit. It logs and
records suggested Forcing lines for a cell file, for a forced-mode preparation; copy them by hand,
because the check never writes a file ([probes drive](command-line.md#probes-drive)). To judge the
chi drive, use [probes band](command-line.md#probes-band).

**Why are training probes masked?**

```bash
python -m core probes mask --bounds Resources/Bounds/nadrowski/master.txt --prior my_prior
```

This check looks at a prior, not a cell. It runs a few batches of training's own generator over the
prior and splits the masked probes by cause: too slow for the floor even at the band's top,
shortened below it by training's draw of lock-in durations, or a lock-in that came back non-finite
or zero. It reports the spread of the rows' own peak frequencies, and names the lever the dominant
cause points at. The probe count, the multipliers and the lock-in durations come from training's own
fixed seed, so they do not change with `--seed`; where the probes land depends on each row's own
peak and recording length, and that placement does change with `--seed`
([probes mask](command-line.md#probes-mask)).

Each check writes one diagnostic record ([probes](command-line.md#probes) has every flag and every
line of output), into the store's `diagnostics/` folder
([The artifact store](getting-started.md#the-artifact-store)). This prints a record's verdicts and the
settings it judged:

```bash
python -m core artifacts show diagnostic my_cell_band
```

**When a check fails.** A check measures and changes nothing. The band, the drive and the cycle
floor change only by a deliberate edit of `core/config.py`, followed by a retrain
([The retrain runbook](retrain.md)); the lock-in ceiling and the number of probe slots change there
too, or on step 1, Model and options, and a change to either also means a retrain. The store refuses
to load a chi posterior trained at another band, drive, lock-in ceiling or number of probe slots
(`core.artifacts.store.ArtifactStore.load_posterior`). The cycle floor is neither recorded with a
posterior nor part of the simulation cache's identity
([The artifact store](architecture.md#the-artifact-store)). So after changing it, retrain against a
new simulation cache, and rebuild any chi observation saved before the change: a plain `train`
would resume the cache simulated at the old floor
([The chi probe set and its Fisher](rules-and-traps.md#the-chi-probe-set-and-its-fisher) gives the
routes). The grids a check takes (`--multipliers`, `--drives`, `--cycle-caps`) measure
alternatives and nothing more: none of them reaches a training run.

## Defining a model of your own

The model builder defines a new stochastic model without code. Open it from Settings (the gear, then
Full settings…) with Open model builder, or with Edit on a saved model
([Screens](window.md#screens)).

1. Give the model a name (a letter first, then letters, digits and `_`, at most 24 characters; it is
   stored in capitals and may not be a built-in name) and its state variables, comma-separated, then
   press **Set variables**. The first variable is the observable: the quantity a recording measures.
2. For each variable, type the right-hand side of its equation dx/dt; its noise strength D, whose
   noise amplitude is √(2D), where numbers and parameters give additive noise and a state variable
   makes it state-dependent; its initial condition; and optionally a drive (sinusoidal, step,
   triangular or exponential). The equations are non-dimensional, and nothing converts them. A
   second-order equation is written as two first-order ones. An expression may use names, numbers,
   `+ - * / ^ ( )`, the functions `sin cos tan asin acos atan sinh cosh tanh exp log sqrt abs sign`
   and `pi`.
3. **Detect parameters** lists every other name as a parameter, each with its value, its box (min
   and max, or auto) and the box's coordinate (linear, or log, which needs a minimum above 0).
4. Under Display scales, `x_scale` is nanometres per model unit of length and `t_scale` seconds per
   model unit of time; `t_scale` must be below 1.5 s.
5. **Validate** parses the model and runs a short integration; **Save model** writes its files:
   `Resources/Models/<NAME>.json`, a `default.txt` in `Bounds/<name>/` and in `Cells/<name>/`, and
   `Units/<name>/units.txt`, which always declares `nm s`. The bounds file keeps the boxes you set,
   and gives `x_scale` and `t_scale` boxes from half to twice their values. Give that bounds file in
   step 2, and the cell is checked against it
   ([Which bounds file a cell is checked against](#which-bounds-file-a-cell-is-checked-against)).

Training, the prior's stability sweep and the Fisher rotation start a user model from the initial
conditions its definition declares; a simulated observation starts it from its cell file's
(`core.registry`). So an edit to the cell file's initial conditions does not reach training.

The rules the builder keeps (`core.Models.user_model.parse_user_model`,
`core.Helpers.model_store.save_user_model`), which a hand edit can break:

- **Numbers are removed before parameter names are looked for.** `1e-3` is a number, not a
  parameter named `e`, and so is `0x1F`. A digit joined to a name belongs to the name: `k2` is a
  parameter.
- **`E` is an ordinary parameter**, not Euler's number, because physics uses the name for a modulus,
  a field or an energy. The only constant is `pi`; write `exp(1)` for Euler's number. The function
  names, `pi`, `t`, `force` and `I` are reserved, and so are `Symbol`, `Integer`, `Float`, `Rational`
  and `Function`, the parser's own names: none can name a parameter.
- **The parameter order is part of the model.** Parameters are ordered by first appearance: the
  first variable's equation, then its noise, then the next variable's. The builder writes the bounds
  file's non-dimensional section in that order, and the simulator binds values to parameters by
  position, so a bounds file in another order would run every simulation with its values in the
  wrong places, and no error. Never reorder that section by hand. The prior stage and `probes mask`
  refuse a bounds file whose order has drifted from the model's
  (`core.SBI.run_guards._assert_user_model_in_sync`); saving the model again rewrites it.
- **A save is committed by its JSON.** The three text files are written first and the JSON last.
  PRISM knows a model by its JSON alone, so a save that fails part-way leaves at most stray text
  files, which the next save overwrites, and never a model with a file missing. Deleting a model
  keeps the same order, its folders first and its JSON last, so an interrupted delete leaves the
  model registered and a second delete finishes it.
- **Not every screen accepts every model.** Which ones take yours depends on its drives, its noise
  and its parameters, as [Supported models](window.md#supported-models) sets out.
