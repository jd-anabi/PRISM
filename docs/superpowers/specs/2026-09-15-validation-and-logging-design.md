# PRISM piece 3: Validation, logging and the private copy (design)

**Date:** 2026-09-15 · **Status:** approved in the brainstorming session of 2026-09-15; self-reviewed by
four adversarial readers and 55 refutations before the owner's review · **Owner:** J
**Parent:** `2026-09-10-pre-retrain-hardening-decomposition.md` §4 row 3 · **Read at:** HEAD `83545b8`. Every
`file:line` below was checked against that tree; a line number is a pointer, not a promise (the plan
re-reads before it edits).

## 1. Purpose and scope

Piece 3 makes the application proof against misuse on the paths piece 2 unified. Five things are
wrong today, each recorded in `docs/STATE.md` "Owed" item 7:

- **One shared configuration.** The window builds one `SimConfig` at Build/Load prior
  (`prior_tab.py:142-150`, `inference_screen.py:96-104`) and every later run writes onto that same
  object: the observation length, the loaded cell's truth, the recording's drive values and probe
  count (`orchestrator.py:165,172,235,305-314,2054-2063`; `store.py:114-149`; `cli.py:87`). Nothing
  clears them. A refused or failed run leaves its writes behind; an amortized training after an
  inference anchors its Fisher rotation on the leaked truth (`decorrelate.py:290-298`); a bench chi
  inference with three probes makes the next simulated chi inference simulate three
  (`build_experiment_observation` sets `cfg.chi_n_freqs`, `orchestrator.py:312`). The command-line
  tool builds a fresh config per command and never trains on a truth (`core/tool/config_args.py:87-123`).
- **Severity by channel, not by message.** There is no `logging` in `core/`. 217 `print` sites in 22
  files talk to the operator; the window shows stdout at info and stderr at warning
  (`streams.py:240-241`), so the memory-statistics line wears a triangle and "loaded a NON-AMORTIZED
  posterior" does not. The console is also the cancel mechanism (`streams.py:188-194`).
- **Blank boxes read as zero.** `FloatField.value()`/`IntField.value()` return 0 on blank or
  half-typed text (`labeled_inputs.py:19-23,45-49`). Tabs clamp (`max(1, …)`), default (`… or
  config.X`) or forward the zero; a blank observation length reaches `math.log(0.0)` after the
  simulation was spent (`statistics.py:604`); the TSNPE tab forwards a blank as 0 and the stage refuses
  with a sentence naming neither the box nor the default (`orchestrator.py:2153-2159`).
- **Two kinds of failure, one red dialog.** `Worker.run` flattens every exception to `(str(e),
  traceback)` (`worker.py:54-55`), so a refusal and a crash both open the Critical "Error" box with a
  traceback behind Details (`base_panel.py:381-392`). The tool guesses from the type: any `ValueError`
  prints as a refusal (`core/tool/__init__.py:122-131`).
- **Saved settings overwrite the science file.** The Config tab seeds the chi drive amplitude and band
  from `config.py` and then restores whatever the last session saved (`config_tab.py:57-58,262-264,
  283-285`); since D11 every non-default value is refused seconds into the prior build. The same
  seed-then-restore pattern covers seven prior-sweep knobs, seven network and rotation knobs and the
  two calibration counts. The TSNPE tab saves five keys and never restores them (`tsnpe_tab.py:38-91`).

The piece also settles four small carried items (§6) and adds the test fixtures the work needs (§8).

### 1.1 Decisions (binding)

| # | decision | where |
|---|---|---|
| V1 | **The private copy is a promise of the core.** No public stage, composition or diagnostic mutates the configuration it is handed: each copies it on entry and works on the copy. The window keeps its session configuration exactly as built; training in the window is truth-free, as on the command line. A refused, failed or cancelled run leaves nothing on the session. | §2 |
| V2 | **Bad inputs are refused at the click**, before any worker starts, by one shared rule set that the stages and compositions run again at entry. A blank box is a refusal, never a zero or a default; the two "0 = automatic" boxes accept a typed 0 only. Every silent clamp and fallback in the inference tabs goes. | §3 |
| V3 | **One refusal kind, neutral message, front-end fix.** Every pre-spend refusal in the modules §3.6 lists raises `Refusal(message, field=…)`. The message names the setting in neutral words, the allowed values, what was given and the default; each front end appends its own "how to fix here" from a table keyed by field. The window shows a refusal as the yellow "Check your inputs" box with no traceback; anything else stays the red box with the traceback. The tool prints `refused:` for a `Refusal`, a traceback for a bug. | §3 |
| V4 | **Standard logging with three levels, and a log file inside each artifact's folder.** Every in-stage print becomes a logging call at information, warning or error. The window feeds records into the log pane at their level and keeps the cancel checkpoint; the tool prints information to stdout and warning/error to stderr with a prefix. Records emitted since the outermost public entry point began are written to `log.txt` in every artifact the store's writer commits for that entry point. `PreflightWarning` stays as piece 2 built it. | §4 |
| V5 | **Selections and budget are remembered; nothing else.** Chi drive amplitude and band become read-only displays of `config.py`'s values. Chi probe count, slots, lock-in ceiling, the prior-sweep, network, rotation and calibration knobs, the HPD level and the direction count open at `config.py`'s value on every launch. Batches and rows-per-batch stay remembered. Consents are never remembered, including the FDT panel's two boxes, which open at their construction defaults. The TSNPE tab restores its observation picker and budget. | §5 |
| V6 | The three derived lines under the training budget display only what one orchestrator preview function computes with the training run's own resolution. | §6.1 |
| V7 | A posterior's record carries the three Fisher settings only when the rotation ran in that process; otherwise it records them as empty. | §6.2 |
| V8 | Loading a prior with no figure sink closes its figure, as the build branch does. | §6.3 |
| V9 | The training library's TensorBoard writer is replaced by one that discards the curves; the repository-root `sbi-logs/` tree is deleted, its ignore line and scan exception go, and the test session asserts no such directory appears. | §6.4 |

### 1.2 How the design reads the decisions

| topic | reading | why |
|---|---|---|
| V1 "copies it on entry" | One decorator on the public entry points (§2.2) does the copy; internal helpers do not copy again. A composition therefore copies once and the public stages it calls copy again from the composition's copy. | Under a millisecond per copy once the caches are dropped first (§2.1); the double copy keeps the promise local to every public function without a "was it copied" flag. |
| V1 and the tool | The tool's fresh config is copied once more at each stage. Nothing about the copy changes what the tool prints. `smoke` runs its four stages against one config and never reads it between them. | The promise is core-level, so both front ends get it without their own code. |
| V1 and the observation context | The observation artifact is the durable copy of a run's context (`_write_observation`, `orchestrator.py:325-344`), and `LoadedObservation.install` re-applies it to a run's copy (`store.py:114-149`). No new session object is added. A caller that wants a stage's writes reads the artifact when the stage is real, or captures the copy the stubbed stage received. | The TSNPE tab already reads its context from disk; Infer builds its own. |
| V1 and the ground-truth Fisher branch | After V1 no production path reaches `decorrelate`'s ground-truth anchor (`:290-298`): the tool's `train` and `smoke` build truth-free configs, a truncated round never runs the Fisher, the identifiability modes that load a truth never call the rotation, and the window's session config is never written by a stage. The branch stays live and tested through callers that inject a truth onto their own config before the call (`build_tiny_run`, `_fixtures.py:226-230`; the `test_user_sbi` pipelines). The truncated-round guardrail-5 warning (`orchestrator.py:1038-1046`) is fed by `tsnpe_round`'s install on the round's own copy and is unchanged. | The piece-2 gate at `0d96d2f` already measured the truth-free rotation the window will now produce. |
| V2 "the same rules again at entry" | The field rules are pure functions on values in a torch-free module (§3.1). The window calls them on the GUI thread before dispatch; the stages and compositions call them before any spend, in the order §3.4 fixes. Stage-level checks that need loaded objects stay in the stage. | The window is protected even if a tab forgets; the tool gets the rules for free. |
| V2 and the live derived lines | The three budget lines (§6.1), the Prior tab's sweep note (`prior_tab.py:196-205`) and the Infer tab's probe planner read `value_or_none()`. A blank, half-typed or out-of-rule box renders the total or sweep line as "<label> is blank" or "<label> must be <rule>" and leaves the memory and checkpoint lines empty; no clamp, no default, no raise. `training_preview` takes ints only and `_sync_budget` does not call it until both boxes pass their §3.3 rule. The planner ("Plan probes…", `infer_tab.py:240-304`) is a click: it runs `require_positive("t_obs", …)` and reports through `_refusal` like any button. | A live line cannot pop a dialog mid-typing; V2's "blank is a refusal" is a click-time rule. |
| V2 and the VRAM ceiling | Unchanged: not persisted, applied live on every keystroke, and its note beside the field is the feedback (`config_tab.py:96-132`). A blank or negative box reads as "off" and the note says so. | Same reason. |
| V2 and a stage with a load branch | A tab whose stage has a load branch (Prior: `build_prior`; Posterior: `build_posterior`) validates at the click only what the chosen branch will read, decided as the stage decides it (`is_new` from the picker, mirroring `trains = train_new or ref is None`, `orchestrator.py:736`). On a load that is the picker (and, for Posterior, the accept dialog); the sweep, budget, flow and Fisher boxes are then not forwarded (the stage resolves `None` to its default and the load branch never reads them). On a build or train every box the branch reads is validated. | The click-side twin of the stage-side exemption (§3.4); the two front ends agree. |
| V3 and the existing control-naming sentences | The near-miss (`orchestrator.py:409-418`), other-observation (`:1785-1789` and its duplicate in `simulated_inference`, `:2047-2053`), non-amortized-load (`store.py:625-634`) and resume-needs-checkpoint (`orchestrator.py:755-758`) messages lose their embedded button labels and flags and gain a field key (`new_run`, `accept_other_observation`, `accept_truncated`; the resume message answers to two controls and has `field=None`); the front-end tables carry the labels. A Python keyword mention ("pass `new_run=True`", "pass `Accept(truncated=True)`") is the core API's own name, not a front-end control, and stays. The `[tsnpe]` prefix and the rest of each sentence stay. | The tables are the single place a control is named in the converted modules; a renamed control cannot go stale. |
| V3 and the tool's ladder | `UsageError` stays exit 2. `Refusal` prints `prism <cmd>: refused: <message> <fix>`, exit 1, no class name and no `[raised at …]`; `fix_sentence` owns the parentheses and is empty for `field=None` or a key with no flag, so the line then ends at the message. A bare `ValueError` or `FileNotFoundError` keeps today's `refused: <Class>: <msg> [raised at …]`, exit 1. Anything else is a traceback. | Backward compatible: an unconverted refusal still reads as a refusal; the class name and location were hedges for bugs disguised as refusals, which a dedicated class no longer needs. |
| V3 and `core/diagnostics` | The five diagnostics are public entry points (§2.2), so their pre-spend knob checks (`sbc.py:109-118`, `identifiability.py:62-66,220,373,378,836,392-395,849-852,880-884`, `ablation.py:91-96`, `feature_sets.py`'s `assert_*` guards) convert too, through the §3.1 rules with the tool-only keys §3.1 lists; `identifiability_rotation`'s "records no Fisher rotation" becomes `Refusal(field="posterior")`. A tool-only key has `CONTROL[key] = None` and renders no window sentence. | One refusal shape on the tool; the "every key in both tables" pin holds with the `None` entry. |
| V4 and the cancel checkpoint | The window's logging handler calls `cancel.check()` before it sinks a record, exactly as `_SignalStream.write` does. A record emitted between a checkpoint shard's fsync and its state replace would therefore raise mid-commit, so the rule at `training_checkpoint.py:44-50` now reads "do not print or log between steps 1 and 3". | Cancel still lands at the next message; the constraint moves with the channel. |
| V4 "since the outermost public entry point began" | A composition writes two artifacts; the observation's file holds the records up to its commit, the inference's file holds the whole composition's records including the pre-spend warnings. `smoke` calls stages one by one, so each stage is its own outermost entry. The tool's own framing prints (§4.1) never reach the file. | The writer opens only around the write (`orchestrator.py:538,1165,1596,1807`), so a buffer from stage entry is the only way to capture the stage. |
| V4 and the simulation cache | The cache gets no `log.txt`: the store refuses the writer for that kind (`store.py:399-401`), its manifest is `training_checkpoint`'s and is refreshed per batch across resumes, and one cache is shared by every posterior that names it. Its records land in the posterior's `log.txt` when `build_posterior` commits; an interrupted process's records are lost with the process, as today. | No single commit holds one entry's records; piece 4's browser must not expect the file there. |
| V4 and Python warnings | The buffer tees `warnings.showwarning` while active, so `PreflightWarning` judgements and library warnings reach the file. The window and tool routes for warnings are unchanged. | The judgement channel is what a reviewer wants in the file. |
| V4 and the logger level | `core/runs.py` sets `logging.getLogger("core").setLevel(logging.INFO)` once at import; it is imported by every §2.2 entry point and both front ends. The window handler and the tool's `main` install and remove handlers only and never touch the level. Propagation stays on. | Python's root logger sits at WARNING; without this the window and the artifact file would drop every information record while `caplog` hid it in the suites. |
| V5 and the Infer tab's observation fields | The observation lengths, the physical chi drive amplitude and the recording paths describe the recording, not the science file; they are selections and stay remembered. | They are seeded from `T_MIN_EXP_S` and `1.0`, but `config.py` owns neither as a science constant. |
| V5 and the FDT consents | "Never persisted" and "opens unticked" are different things. Both boxes are never persisted and open at their construction defaults: `skip_sanity` unticked, `confirm_production` ticked (`fdt_panel.py:83-85`), matching the tool where `--no-production` is opt-in (`core/tool/fdt.py:103-104,142-143`). | Otherwise a default click would stop after the sanity checks where today and on the command line it runs the production sweep. |
| V6 and the checkpoint cadence | The preview reads the training stage's own binding of `TRAINING_CHECKPOINT_EVERY` (the import-time snapshot in `orchestrator`), not `config`'s live one (`base.py:163`, `tsnpe_tab.py:136`). `tests/conftest.py:54-76` patches the orchestrator binding to 0 for the session, so every GUI test now sees "off" on both tabs; the one test that rebinds `config`'s copy to flip the line (`test_settings_persistence.py:239-259`) rebinds the orchestrator's instead. The "off" line keeps the word "config" (`"Checkpointing is off (config.TRAINING_CHECKPOINT_EVERY = 0) …"`). | The line and the run can never disagree. |
| device | `--device` gains `cuda` beside `auto` and `cpu`. `cuda` resolves `hw = config.detect_device()` in `make_cfg` (`config_args.py:121`) and raises `refuse("device", …)` when `hw.device.type != "cuda"`, saying CUDA is absent or the card's compute capability is below 8.0 (`config.py:66-70`). `auto` keeps passing `hw=None`. The cuda `DeviceConfig` is therefore exactly `detect_device()`'s, never a hand-built one: its batch size and dtype enter the simulation identity (`identity.py:53,72-73`), so `cuda` and `auto` on a qualifying card share one cache identity. The window keeps auto-detection and no device control. | The decomposition names "device" among the fields; today there is no way to ask for the card explicitly, and the prior sweep falls back to the CPU with a warning when it cannot have it (`prior.py:31`). |
| Repeated warnings | The out-of-memory notices stay repeated per event (they become `warning` records, which have no once-per-location registry); `PreflightWarning` keeps its `always` filter; `chi_probes.py:263-267`'s masked-probe count stays a `warnings.warn` per batch. | The GPU gate reads the masked-probe counts and the OOM lines. |

### 1.3 Out of scope, and who owns it

| item | owner |
|---|---|
| The Simulate, FDT, CrossVal, Reduction and model-builder fields (`simulate_panel.py:69-89`, `fdt_panel.py:72-101`, `crossval_panel.py:71-104`, `reduction_panel.py:49-58`, `model_builder_screen.py:134-192`). The rules module (§3.1) and the front-end tables are built so those panels adopt them. Their builder-failure path stays on `BasePanel._config_error` (§3.5). The three edits this piece makes inside those files are this piece's: §3.5's ffmpeg refusal, §3.6's removal of the `FDTModelError` translation, §5.1's unpersisted FDT consents. | piece 5 |
| The artifact browser, annotate, cleanup of incomplete directories, `log.txt` shown in the browser (never for a simulation cache). | piece 4 |
| A chosen-cell anchor for the Fisher rotation in the window. Under V1 training is truth-free; if a truth-anchored rotation is ever wanted it becomes an explicit control. | later, if ever |
| A verbosity switch on the tool (spec §3.9 of piece 2: "There is no `--quiet`"). `PRISM_MEM_LOG_EVERY` stays a once-read environment setting (D13). | none |
| Qt's own C++ diagnostics (`qWarning`, `qt.qpa.*`), which bypass Python's streams; no `qInstallMessageHandler` is installed. | none |
| The chi override (D11) and the D12 escape hatch: standing refusals, not reopened. | none |
| Recording the parent's Fisher settings in a resumed run's record (the checkpoint header carries `V` but not `m`/`dz`/`points`, `training_checkpoint.py:159-180`). V7 records "not run" honestly and loses that provenance. | the owner decides later |
| The optional tidy-up of `decorrelate.py:93-95`'s `or` fallbacks and `prior.py:102-105`'s clamps once the stages pass resolved values (§3.4). | the plan, if cheap; else none |
| `calibration` and TSNPE children recording accept flags; `identifiability jacobian`'s NaN column; repeat-SBC's fixed probe seed; the multi-GPU `seeded()`; the enumerated skip list of the code-directory guard (`docs/STATE.md` "Open, with no piece owning them yet"). | as listed there |

## 2. The private copy (V1)

### 2.1 `SimConfig.copy_for_run()`

```python
_CACHED = ("t", "_ureg", "length_unit", "time_unit", "force_unit", "freq_unit")

def copy_for_run(self) -> "SimConfig":
    """A deep copy for one run. The cached properties are dropped BEFORE the deep copy, so the
    2.4M-point grid and the pint registry are never duplicated; the copy recomputes them lazily on
    the thread that uses it."""
    c = copy.copy(self)
    for key in _CACHED: c.__dict__.pop(key, None)
    return copy.deepcopy(c)
```

- Measured on a real `_nad_cfg` with `t`, `_ureg` and a `chi_obs_freqs` tensor set: 19 ms and a
  transient 9.6 MB when the caches are copied, 0.1 ms with them popped first; the `OrderedDict`s,
  `hw`, `sources`, `labels` and `chi_obs_freqs` are independent and equal afterwards, and `t`
  recomputes equal. A CUDA tensor deep-copies onto its device.
- `config.unit_registry`'s docstring (`config.py:641-647`) and the "shared singleton" comments at
  `cli.py:187,214` describe a process-wide registry that the code never had (the function builds a
  new one per call); they are corrected in the same change.

### 2.2 The entry points that copy

A decorator `core.runs.public_entry` (torch-free module, see §4.4 for its second job) wraps every
public function that takes a `SimConfig` and may write to it or to an artifact:

| module | functions |
|---|---|
| `core/orchestrator.py` | `generate_observations`, `build_experiment_observation`, `build_prior`, `build_posterior`, `validate_calibration`, `infer_and_visualize`, `build_truncation_region` (takes no cfg; captures logs only), `simulated_inference`, `experimental_inference`, `tsnpe_round` |
| `core/diagnostics/*.py` | `sbc_repeats`, `identifiability_rotation`, `identifiability_laplace`, `identifiability_jacobian`, `channel_ablation` |

The decorator replaces the first positional argument (or the `cfg` keyword) with `cfg.copy_for_run()`
when it is a `SimConfig`; stubs and `object()` sentinels pass through untouched (the gate tests put
`object()` on `session.cfg`). It uses `functools.wraps`, so `inspect.signature` follows `__wrapped__`
and `inspect.getsource` returns the original's source with the decorator in `decorator_list` (probed:
the AST pins at `test_conditioning_repair.py:786,1038`, `test_settings_persistence.py:120`,
`test_nav_and_gating.py:293` keep finding their needles). Monkeypatched stage names still take effect
inside compositions, which look the stage up in module globals at call time. The keyword assertions of
`test_the_compositions_forward_every_keyword_unchanged` hold; its positional identity assertion
(`seen["args"][0] is cfg`) does not and is re-pinned (§8.2).

Internal helpers (`_write_observation`, `_draw_calibration_set`, `check_observation_in_distribution`,
`fresh_run_near_misses`, `training_identity`, `training_preview` of §6.1) do not copy: they read.

### 2.3 What changes for callers

- The window needs no new plumbing. `session.cfg` stays as `install_config` set it. The Infer tab's
  probe rows (`infer_tab.py:207-238`), the Prior tab's "K drive frequencies" line
  (`prior_tab.py:172`) and the budget group (`base.py:118-140`) keep reading the pristine config.
- A caller that reads a field a stage wrote is a defect this piece fixes the same way: it reads the
  artifact when the stage is real, or captures the copy the stubbed stage received. The plan greps,
  after every stage or composition call in `core/` and `tests/`, for reads of `T_obs`, `n_obs`,
  `chi_obs_freqs`, `chi_n_freqs`, `inits_dict`, `has_ground_truth`, `sources` (both `sources["cell"]`
  and `in cfg.sources`), and for identity pins (`is cfg`, `is cfg.hw`) on an object passed through a
  public entry. The seven tests found so far are in §8.2.
- `build_tiny_run` sets `cfg.T_obs = 1.0` and injects the cell on its own config before calling
  stages (`_fixtures.py:210-255`); that is a caller's own write and stays.
- The `[fisher]` summary line names the anchor (`decorrelate.py:330-332`); after V1 a window training
  logs "prior median", never "GT".
- `TrainingPlan` shares `cfg.t` and `cfg.hw` by reference (`orchestrator.py:1061,1103`); with the copy
  it shares the copy's, and the checkpoint dict's `"hw"` compares equal to the caller's by value
  (`DeviceConfig` is an `eq` dataclass).

### 2.4 Pins

- Stage level, for every function in §2.2's table: the caller's config compares equal, field by field
  (tensors by `torch.equal`, cached keys excluded), before and after (a) a success, (b) a refusal
  raised before the spend, (c) an exception raised inside the stage (monkeypatched to raise after the
  first write). One parametrised test over a `_nad_cfg` plus the existing composition stubs.
- Window level, on the real-session fixture (§8.1), whose session config is a truth-free copy: a
  simulated inference dispatched through the Infer tab leaves `session.cfg` equal to its snapshot,
  `has_ground_truth` False and `T_obs` unchanged. (With a truth-carrying session config the clause
  would be vacuous: the composition re-injects the same cell.)
- Chi probe count: in `test_an_experimental_observation_records_its_own_length_and_drive_frequencies`'s
  setup, after `build_experiment_observation` with three probes the caller's `cfg.chi_n_freqs` is
  still the config's count and `cfg.n_obs` is still `None`; the observation manifest records three
  probes and 200 samples.
- The decorator passes non-`SimConfig` first arguments through unchanged.

## 3. Refusals and field validation (V2, V3)

### 3.1 `core/refusals.py` (torch-free)

```python
class Refusal(ValueError):
    """Something asked for that the program will not do. `field` names the setting at fault by a key
    every front end can map to its own control or flag; None for a refusal with no single field."""
    def __init__(self, message: str, *, field: str | None = None): ...

@dataclass(frozen=True)
class Field:
    key: str            # "t_obs"
    what: str           # "the observation length, in seconds"
    default: str | None # "none: it must be given" / "5000" / None when the rule has no default

FIELDS: dict[str, Field]   # the registry; every key raised anywhere in core is here (pinned)

def require_given(key, value)                       # None (a blank box) -> Refusal "… is blank"
def require_positive(key, value)                    # blank, or <= 0 -> Refusal
def require_at_least(key, value, minimum)           # blank, or < minimum -> Refusal "… at least <minimum>"
def require_between(key, value, lo, hi, *, open=…)  # blank, or outside -> Refusal
def require_finite(key, value)                      # NaN/inf -> Refusal
def require_file(key, path, what)                   # blank or missing -> Refusal naming the input kind
def refuse(key, message)                            # a Refusal with the field's what/default appended
```

Message shape, fixed by test: `"<What> must be <rule>; got <value> (default <default>)."` with the
default clause only when the field has one, and `"<What> is blank (default <default>)."` for a blank.
`require_at_least`'s rule text is `at least <minimum>`. Neutral: in the modules §3.6 converts, no box,
tab, flag or button is ever named; a Python keyword may be.

Field keys (the registry, initial set): `t_obs`, `num_runs`, `run_size_cap`, `n_directions`,
`hpd_level`, `n_cal`, `cal_n_scales`, `num_posterior_samples`, `n_samples`, `num_iterations`,
`sweep_batch`, `max_sets`, `walk_step`, `stability_units`, `min_cluster_size`, `min_samples`,
`hidden_features`, `num_transforms`, `learning_rate`, `stop_after_epochs`, `max_num_epochs`,
`fisher_m`, `fisher_dz`, `fisher_points`, `checkpoint_every`, `resume`, `new_run`, `accept_truncated`,
`accept_other_observation`, `name`, `bounds`, `cell`, `units`, `recording_spont`, `recording_forced`,
`recording_probe`, `drive_amplitude`, `drive_frequency`, `drive_phase`, `chi_f0_si`, `chi_n_freqs`,
`chi_k_pad`, `chi_max_cycles`, `chi_f0`, `chi_freq_bounds`, `device`, `model`, `observation`,
`posterior`, `prior`; tool-only (diagnostics): `repeats`, `n_points`, `n_worst`, `top_n`, `m`,
`m_noise`, `rel`, `min_valid`, `rows`, `n_sweep`, `chi_k_fixed`. The plan may add keys; the pin is that
every key used is registered and mapped.

### 3.2 The front-end tables

- `core/gui/fields.py`: `CONTROL: dict[str, tuple[str, str] | str | None]` — for a box, `(tab, label)`;
  for a consent, picker or dialog, one sentence ("tick 'Run on a different observation' on the Infer
  tab"); `None` for a tool-only key. `label(key)` returns the label, and **the inference tabs build
  their static rows from it** (`add_help_row(form, label("t_obs"), …)`), so a renamed control is
  renamed in one place. The three drive keys' entries are `("Infer", labels.gui_forcing_label(n,
  config.FORCING_DISPLAY_UNITS[n]))` for amp/freq/phase (static strings; `_rebuild_forcing_fields`
  keeps building its rows from the config's forcing names). The read-only `chi_f0` and
  `chi_freq_bounds` are sentence entries ("fixed by measurement; edit config.py"). `fix_sentence(key)`
  renders `"Set it in the '<label>' box on the <tab> tab."`, the sentence, or nothing.
- `core/tool/fields.py`: `FLAG: dict[str, str]` (`"t_obs": "--t-obs"`, `"new_run": "--new-run"`,
  `"accept_truncated": "--accept-truncated"`, `"cell": "--cell"`, `"repeats": "--repeats"`, …).
  `fix_sentence(key)` renders `"(<flag>)"` and owns the parentheses; a key with no flag, or
  `field=None`, renders `""`.
- Pinned: every `FIELDS` key has an entry in both tables (`None` counts); every `field=` literal under
  `core/` (AST scan over `CODE_ROOTS`) is a `FIELDS` key; every `(tab, label)` in `CONTROL` names a
  label the named tab's form shows. The read-back reads the `QLabel` inside each row's holder widget
  (`help_badge.py:36-63`) and compares against `labels.pretty_gui(label)`; the Infer tab is read after
  `install_config` with a forced stub config so its drive rows exist.

### 3.3 The rules, per field

| field | rule | today |
|---|---|---|
| `t_obs` (three boxes, `--t-obs`) | given, finite, > 0. Outside `[T_MIN_EXP_S, T_MAX_EXP_S]` stays a `PreflightWarning` on the simulated path (walkthrough B6) and becomes the same warning on the experimental path | warned at 0, crashed in `math.log` after the spend |
| `num_runs` | integer ≥ 1 | Posterior refuses via log line, TSNPE clamps to 1 |
| `run_size_cap` | integer ≥ 0; typed 0 = automatic; blank refused | blank silently 0 |
| `n_directions` | integer ≥ 1; the stage adds ≤ latent width with the width and the default in the message | blank → 0 → stage refusal naming nothing |
| `hpd_level` | 0 < x < 1; below 0.99 stays the tight-HPD warning | blank → 0.0 → stage refusal |
| `n_cal`, `num_posterior_samples`, `n_samples`, `repeats` | integer ≥ 1 | clamped or refused by keyword name |
| `cal_n_scales` | integer ≥ 1, refused (its docstring says it changes what is measured) | clamped in `analysis.py:159` and `validate_tab.py:48` |
| `num_iterations`, `max_sets`, `min_samples`, `stop_after_epochs`, `max_num_epochs`, `hidden_features`, `num_transforms`, `fisher_m`, `fisher_points`, `n_points`, `n_worst`, `top_n`, `m`, `m_noise`, `rows`, `n_sweep` | integer ≥ 1 | clamped in the tabs; unchecked or `or default` in core |
| `min_cluster_size` | integer ≥ 2 | clamped |
| `sweep_batch`, `checkpoint_every` | integer ≥ 0; typed 0 = automatic / off | clamped |
| `walk_step`, `stability_units`, `learning_rate`, `fisher_dz`, `rel` | finite, > 0 | `or default` (a blank became the default; a negative passed) |
| `resume` | one of `auto`/`require`/`never` | already refused, by keyword name |
| `bounds`, `cell`, `units`, `recording_spont`, `recording_forced`, `recording_probe` | given and the file exists, message naming the input kind | `File not found: <path>` or `the spont recording was not found: ''` inside the worker |
| `drive_amplitude`, `drive_frequency` (driven bench branch) | finite, > 0 (a zero drive is the passive branch's job) | unchecked; 0 Hz reaches the lock-in |
| `drive_phase` | finite | unchecked |
| `chi_f0_si` | finite, > 0 | blank → division by zero inside the worker |
| `chi_n_freqs` | 2 ≤ K ≤ `CHI_K_MAX` and ≤ `chi_k_pad` | checked at Apply, by ad-hoc strings |
| `chi_k_pad` | 2 ≤ pad ≤ `CHI_K_MAX` | checked one tab later, in `__post_init__` |
| `chi_max_cycles` | > `CHI_MIN_CYCLES` | checked at Apply |
| `device` | `cuda` only when `detect_device()` yields it | not requestable |
| `name` | the store's name rule and "not taken" | `StoreError` naming nothing |

The stage-only checks (direction count against the latent width, the D7 near miss, D12, the
store's load mismatches, the mode/width mismatch, the other-observation guardrail) keep their logic
and become `Refusal`s with a field key where one control answers them (`n_directions`, `new_run`,
`accept_other_observation`, `accept_truncated`, `posterior`, `observation`) and `field=None` otherwise.

### 3.4 Where the checks run

- **At the click.** Each inference tab gains `_read_inputs() -> dict` that reads every box the chosen
  branch will read (§1.2, "a stage with a load branch") through `value_or_none()` (`IntField` gains
  one) and the §3.1 rules, and raises the first `Refusal`. The click handler is `try: v =
  self._read_inputs() except Refusal as e: self._refusal(e); return`, then dispatch. The Config tab's
  Apply is a click handler under this scheme: its K, slots, lock-in-ceiling and typed-units checks
  (`config_tab.py:197-226`) move into its `_read_inputs` (`field="chi_n_freqs" | "chi_k_pad" |
  "chi_max_cycles" | "units"`; `UnitParseError` is already a `Refusal` under §3.6). The existing
  pick-time cell check (`infer_tab.py:141-160`) and the probe-row `problems` (`rows.py:59`) stay as
  they are, rephrased through the rules where they overlap.
- **At stage entry, and where.** Each stage's resolution block and each composition's pre-spend block
  call the same rule functions on the values they resolve, before any file is hashed, any Fisher runs
  or any simulation starts. Order inside each block, which the "refused before any spend" pins rely
  on: numeric knobs first, then the name, then the files. Placement:
  - `build_prior` (`orchestrator.py:507-519`) resolves `max_sets`, `walk_step`, `min_cluster_size`,
    `min_samples` itself (`None` → the `config.PRIOR_*` constant) beside `num_iterations` and
    `sweep_batch`, runs the rules there, and passes the resolved values to `gen_prior` (`:527-529`)
    and into the prior manifest's `knobs` (`:543-545`).
  - `build_posterior` (`:718-758`) resolves `fisher_m`, `fisher_dz`, `fisher_points` itself (`None` →
    `config.REPARAM_FISHER_*`), runs the rules on the flow and Fisher knobs only when the call
    trains (`trains`, `:736`), and passes the resolved values to `decorrelate.build_latent_fisher_rotation`
    (`:1010-1012`); the `or` re-resolution at `:1185-1191` goes (§6.2).
  - `simulated_inference` (`:2037-2082`): `n_samples`, `t_obs`, then the name, then
    `require_file("cell", …)` beside the cell/`gt_values` exclusivity check; `cli.load_and_validate_gt`
    and `validate_gt_file` keep their parse-failure behaviour.
  - `experimental_inference` (`:2102-2104`) adds `n_samples` and `t_obs` and forwards `rec` unchecked;
    the recording rules stay in `build_experiment_observation`'s loop (`:273-277`, today's
    `FileNotFoundError` becoming `Refusal(field="recording_spont" | "recording_forced" |
    "recording_probe")`), and the drive rules in the forced builder.
  - `validate_calibration` (`:1585-1594`), `tsnpe_round` (`:2146-2177`) and the diagnostics' guards,
    as today, with the rules.
- **The load path.** `build_posterior`'s load branch no longer range-checks `num_runs`,
  `run_size_cap` and `checkpoint_every` (they are already exempt for the flow knobs, `:736-748`);
  `build_prior`'s load branch already skips the sweep knobs (`:496-512`).
- **The VRAM ceiling** is the one field outside this scheme (§1.2).

### 3.5 How each front end shows a refusal

- **Window.** `BasePanel._refusal(exc: Refusal)`: a log line at `warning` carrying the message and the
  fix sentence, then a `QMessageBox` Warning titled "Check your inputs", `setText(message)`,
  `setInformativeText(fix_sentence(exc.field))` (or nothing), one "OK" button. No traceback. The
  inference tabs stop calling `_config_error` (`prior_tab.py:117,145,227`; `posterior_tab.py:272`;
  `infer_tab.py:137`): a `Refusal` from a builder or a rename goes to `_refusal` (a rename failure is
  a `StoreError` and reads as such), a bare exception to `_on_error`. `BasePanel._config_error` keeps
  its body (`base_panel.py:362-379`) for the Simulate, Reduction, CrossVal and FDT panels, whose
  builders still raise bare `ValueError`/`FileNotFoundError` for a bad cell; piece 5 retires it. The
  missing-ffmpeg `RuntimeError` (`simulate_panel.py:135`) becomes a `Refusal` shown through `_refusal`.
- **Window, from the worker.** `WorkerSignals.error` becomes `Signal(object, str)` carrying the
  exception; `_on_error(exc, tb)` routes a `Refusal` to `_refusal` and everything else to the red
  Critical box with the traceback in Details, as today. `_on_error` also accepts a plain string.
  `tsnpe_tab.py:100-104` passes the caught exception through unchanged (no wrapper, no re-typing, its
  `Could not load observation '<key>':` prefix dropped: every store refusal already names the
  observation, `store.py:679,684-698`), so a store `Refusal` reaches the yellow box and a bug the red
  one. Tests that stub `_on_error = lambda message, tb: …` receive the exception object.
- **Tool.** `main`'s ladder (§1.2): `Refusal` → `prism <cmd>: refused: <message> <fix>` on stderr,
  exit 1.
- **The dialog's default button** is the safe one; for a one-button box that is "OK".

### 3.6 Conversions in core

- `StoreError`, `ManifestError`, `UnitParseError`, `FDTModelError`, `ModelParseError` and `UsageError`
  become subclasses of `Refusal` (`UsageError` keeps exit 2 by being caught first). In the FDT panel's
  guard (`fdt_panel.py:39-50`) the `FDTModelError → RuntimeError` clause goes (the error is a `Refusal`
  and the worker routes it to the yellow box); the `KeyError` backstop stays as it is.
- The bare `ValueError`s raised as refusals in `orchestrator.py`, `sim_config.py` (`__post_init__`,
  `_fill_checked`), `cli.py` (`make_sim_config`, `load_and_validate_gt`), `observations.py` (the
  builders' refusals, including the `KeyError` at `:99-101`), `run_guards.py`, `truncate.py:164-207`
  (`check_basis`), `pipeline.py:1361-1369`, and `core/diagnostics/*.py`'s pre-spend guards (§1.2)
  become `Refusal`s with a field key where one control answers them. `smoke`'s post-spend integrity
  checks (`smoke.py:236-240`) stay `RuntimeError`: they are bugs.
- `assert_name_free` raises `Refusal(field="name")`. `file_manager._read_lines`'s `FileNotFoundError`
  stays; the callers check the file first with `require_file`, so it is reached only by a race.
- `_assert_chi_config_is_deliberate`'s text drops the "MOST LIKELY CAUSE: stale persisted GUI settings
  … Check the [inference_config] chi_lo / chi_hi / chi_f0 keys in PRISM.ini" paragraph and its "Config
  tab" mentions (`run_guards.py:154-158`): after V5 no front end can produce the mismatch; the advice
  becomes "edit config.py deliberately" and the per-field values table stays. Its docstring's history
  (`:111-118`) stays as history.
- `validate_gt_file` (`cli.py:46-74`) stays the non-mutating dry run; its strings come from the same
  rules so the two copies cannot drift.

## 4. Logging (V4)

### 4.1 Levels and the conversion rule

Each module gets `log = logging.getLogger(__name__)` (the family root is `core`, level `INFO` set by
`core/runs.py`, §1.2). Three levels are used:

| level | what | examples |
|---|---|---|
| `info` | banners, timings, results tables, progress notices, budget and checkpoint lines | `[budget] …`, `[checkpoint] resuming at batch k/n …`, `Reusing the Fisher rotation …`, `[fisher] eigenvalue spread …`, the SBC/TARP tables, the diagnostics' report tables, `[mem] …`, the every-5-s wait line |
| `warning` | anything the operator should act on or notice | OOM retries and device errors (`pipeline.py:698-707,788-793,895-906,1652,1682`), `[patho]`, `could not snapshot the RNG`, `[tsnpe] loaded a NON-AMORTIZED posterior`, `No clusters found`, the CPU fallback of the sweep (`prior.py:31`), `[tsnpe] direction j NOT truncated`, `[fisher] operating point k failed`, `!! DEAD FEATURE CHANNELS`, `[eigenvalues] NOT STORED`, the solver's CUDA-graph fallback, `Adaptive batching: … capped`, the plot helpers' "dropping n non-finite points", `[tier1] could not describe`, `describe_siblings`'s near-miss account |
| `error` | a failure reported rather than raised | `[checkpoint] could not save on the way out`, `batch FAILED after both halving retries`, `Campaign 2 FAILED` per point |

The plan carries the full 217-site table (one row per site with its level; the brainstorming map's
`print-sites` inventory is its source and is copied into the plan). **Not converted:** the tool's own
framing prints stay `print`s on their current streams — `[smoke] …`, `[skip] …`, `=== <stage> ===`,
`[ok] <stage> in Xs`, `[smoke] *** FAILED in stage …` in `smoke.py`, the `[prism] <kind> …` and
`[cfg] …` lines in `config_args.py`, and `main`'s ladder — because they are not in-stage prints and
`public_entry`'s buffer never sees them. `file_manager.list_dir` stops printing and returns its
listing; the `ArtifactPicker.refresh` wrapper that swapped stdout process-wide to silence it
(`artifact_picker.py:52`) goes, and the hazard note at `base_panel.py:279-287` is reduced to the
model-combo case.

The two judgements said twice (`orchestrator.py:1046-1052` and `:1781-1792`) keep only the
`warnings.warn`; the accepted-other-observation sentence keeps "Running anyway (accepted)." (walkthrough
B3). `PreflightWarning`, its `always` filter and `_preflight_warn` are unchanged. `pipeline._log_memory`
keeps its name (two tests monkeypatch it by name).

### 4.2 The window handler

`redirect_streams` installs, beside the two `_SignalStream`s, a `logging.Handler` on the `core` logger
for the duration of the run: `emit` calls `cancel.check()` when a token is armed, then
`pump.sink("log", (record.getMessage(), level))` with `level` in `info`/`warning`/`error`. It is removed
in the same `finally` that restores the streams; it never sets a level. The console redirect stays
for the library's own prints, the progress bars and sbi's epoch counter. A record emitted on the GUI
thread during a run lands in the pane like a GUI-thread print does today.

### 4.3 The tool's handlers

`main` installs two handlers on the `core` logger around the handler call and removes them in a
`finally` (the suite calls `main` repeatedly in-process): `info` records to `sys.stdout` with
`%(message)s`; `warning` and `error` to `sys.stderr` as `warning: %(message)s` / `error: %(message)s`.
Both resolve `sys.stdout`/`sys.stderr` at emit time (capsys swaps them per test; the window swaps them
per run). `main` never touches the logger's level.

### 4.4 The per-artifact log file

`core/runs.py` holds a thread-local stack of run buffers. `public_entry` (§2.2) pushes a buffer when
none is active and pops it on exit; while active, the buffer receives every `core` record at `info`
and above (formatted `HH:MM:SS level message`) and tees `warnings.showwarning` (calling the previous
hook, which under the window is `streams`' own and under `pytest.warns` is pytest's; both restore
theirs afterwards, so the tee nests cleanly). `ArtifactWriter._commit` (`store.py:265-289`) writes the
active buffer's lines so far to `<dir>/log.txt` before the manifest. The file is not a payload (not
hashed, not in `payloads`); the store indexes by `manifest.json` only (`store.py:304-324`) and
`validate()` checks dict keys only, so loading ignores it and the manifest schema is unchanged. The
simulation cache gets no file (§1.2).

What is not in the file: sbi's printed epoch counter and table, the progress bars, and anything
printed by a library or by the tool's framing. What is: every stage record and every Python warning
raised meanwhile, including `PreflightWarning`.

### 4.5 Lines whose text and stream are fixed

The GPU recipe in `CLAUDE.md` and `docs/STATE.md` and the tool suite read these; each keeps its exact
words, and on the tool its stream: `Reusing the Fisher rotation stored with the training checkpoint`
(info, stdout), `[checkpoint] resuming at batch k/n` (info, stdout; `test_tool.py:504,838`), the
absence of any `[fisher]` line on run 2b, the OOM notices (warning, stderr), the masked-probe counts
(`warnings.warn`, stderr), `=== <stage> ===`, `[ok] <stage> in Xs`, `[smoke] ALL STAGES COMPLETED`,
`[smoke] *** FAILED in stage …` (prints, stdout), the `[budget]` line, `[tsnpe] region from
observation …`, `[prism] <kind> <name__id> <path>`, `differs only in n_runs: this run 2, that cache 4`,
and every string a passed walkthrough row quotes (`docs/checklists/display-walkthrough.md:36-72`).

### 4.6 What stays as it is

`PRISM_MEM_LOG_EVERY` (D13); no `--quiet`; no global warnings filter in core; the three local
`catch_warnings` blocks in `gen_training_data` (`pipeline.py:969,1026,1080`), which do not silence
logging records (an OOM notice inside them is now visible, which is the intent); sbi's root-logger
warnings, which reach stderr through `lastResort` as today; `config.QUIET_SEGMENT_BAR`.

## 5. Remembered settings (V5)

### 5.1 Classes and keys

| class | keys | after |
|---|---|---|
| selections | `inference_config`: `model`, `units_mode`, `units_text`, `chi_mode`, `reparam_rotate`; `inference_prior`: `prior`, `bounds`; `inference_posterior`: `posterior`; `inference_infer`: `cell`, `infer_mode`, `sim_tobs`, `exp_tobs`, `chi_tobs`, `exp_spont`, `exp_forced`, `chi_spont`, `chi_f0_si`; `inference_tsnpe`: `observation` | saved and restored, as today; TSNPE's now restored |
| budget | `inference_posterior` and `inference_tsnpe`: `num_runs`, `run_size_cap` | saved as the box's text (`settings.save_field`) and restored through `get_int` with the `config.py` default, so a box left blank at close opens at the default on the next launch, never as a `0` nobody typed (`posterior_tab.py:297-298`, `tsnpe_tab.py:158-159` save the int today) |
| fixed by measurement | `chi_f0`, `chi_lo`, `chi_hi` | read-only displays; keys neither written nor read |
| frozen into the network | `chi_k_pad`, `chi_max_cycles` | open at `config.py`; keys neither written nor read |
| science knobs | `chi_k`; `inference_prior`: `sweep_iters`, `sweep_batch`, `sweep_max_sets`, `sweep_step`, `sweep_units`, `cluster_size`, `cluster_samples`, and the dead `bounds_source`; `inference_posterior`: `flow_hidden`, `flow_transforms`, `flow_lr`, `flow_patience`, `fisher_m`, `fisher_dz`, `fisher_points`; `inference_validate`: `cal_n`, `cal_scales`; `inference_tsnpe`: `hpd`, `n_dirs` | open at `config.py` (or `truncate.DEFAULT_*`); keys neither written nor read; a stale key in an old `PRISM.ini` is ignored |
| consents | TSNPE `new_run`, Infer `other_obs` (already unpersisted); `fdt`: `skip_sanity`, `confirm_production` (`fdt_panel.py:150-151,161-162` go) | never persisted; open at construction defaults (§1.2) |
| preferences and layout | `appearance/*`, `window/geometry`, `layout/*` | unchanged |

The Validate tab's docstring ("Persists: nothing") becomes true. `settings.get_bool` gains the
"unparseable → default" branch `get_int` has (a corrupt `reparam_rotate` restored as False,
`settings.py:55-59`).

### 5.2 The Config tab's fixed values

`chi_f0` and the band row's two fields stay `FloatField`s with `setReadOnly(True)` and the caption
style, showing `config.CHI_F0` and `config.CHI_FREQ_BOUNDS`, under a caption "fixed by measurement;
change it in config.py"; `_sync_chi_enabled`'s loop is unchanged. The draft is built with
`chi_f0=None, chi_freq_bounds=None` so `make_sim_config` takes `config.py`'s values
(`cli.py:283-286` already accepts `None`). The Apply log line (`config_tab.py:235-240`) reads
`config.CHI_F0` and `config.CHI_FREQ_BOUNDS`, not the draft; the Apply-time "F₀ > 0" and "0 < lo < hi"
checks (`:206-207`) go and the class docstring (`:26-27`) stops claiming them.
`_assert_chi_config_is_deliberate` stays as the last line of defence (its text per §3.6).
`HELP["chi_f0"]` and `HELP["chi_range"]` (`help_text.py:140-151`) and the `config.py:572` comment
("TUNABLE per config in the Config tab") say the values are fixed by measurement and change only by
editing `config.py`.

### 5.3 The TSNPE tab

`__init__` ends with `self.restore_settings(settings.settings())` like every other panel;
`save_settings` writes `observation`, `num_runs`, `run_size_cap` (as text, §5.1); `restore_settings`
reads those three. `hpd` and `n_dirs` open at `truncate.DEFAULT_HPD` and `DEFAULT_N_DIRECTIONS`.

### 5.4 Test isolation

The settings location is never the real `PRISM.ini` during a test process, at any fixture scope
(§8.1), so a test that asserts a default asserts `config.py`, never the developer's last session.

## 6. The small carried items

### 6.1 The training preview (V6)

```python
@dataclass(frozen=True)
class TrainingPreview:
    n_runs: int; width: int; hw_batch: int        # rows per batch, resolved by _training_run_size
    n_fine: int | None; n_vars: int | None        # n_vars from _observation_inits(cfg).shape[-1]
    need_elements: int | None; have_elements: int | None; estimate_error: str | None
    checkpoint: str                               # "off" | "needs_config" | "resume_complete" |
                                                  # "resume_partial" | "new_with_siblings" | "new"
    batches_done: int; siblings: str; cadence: int; checkpoint_error: str | None

def training_preview(cfg, prior, *, num_runs: int, run_size_cap: int, truncation=None,
                     checkpoint_every=None, store=None, hw=None) -> TrainingPreview
```

It resolves exactly as `build_posterior` and `fresh_run_near_misses` do (`orchestrator.py:361-396,
718-727,849,877-880`): `_training_run_size`, `training_identity`, `store.kind_dir("simulation")` of
the given or default store, `training_checkpoint.peek`/`describe_siblings`, the orchestrator's
cadence binding, and the planner's own `peak_sim_elements`/`sim_memory_budget_elements`. The
no-config branch is explicit, because `_sync_budget` runs at launch with `session.cfg = None`: the
hardware is `getattr(cfg, "hw", None) or hw or detect_device()` (the tab memoises `detect_device()`
as today, `base.py:75-88`, and passes it as `hw=`), so `width` and `hw_batch` are always ints and the
total line always formats; with no config the memory estimate uses the hardware defaults (`n_fine =
N_ND_MAX`, `n_vars = 3`, `steady = 0`) and the tab appends "(Estimated from hardware defaults until a
config is built.)" as today; any failure inside the memory arithmetic lands in `estimate_error`, any
failure in the identity, store, `peek` or `describe_siblings` lands in `checkpoint_error`, and
`checkpoint="needs_config"` when the config or the prior is `None`. It never raises.
`_TrainingBudgetMixin._sync_budget` calls it and only formats; `_effective_width`, `_budget_memory`'s
arithmetic and `_budget_checkpoint`'s derivation go. The TSNPE tab's override takes the preview and
returns its rule sentence only when `preview.checkpoint != "off"`, falling through to the mixin's
formatting otherwise; it never reads `config.TRAINING_CHECKPOINT_EVERY` (`tsnpe_tab.py:136` goes).

Strings kept, beyond those the walkthrough rows A3, A8 and B4 quote: the total line's `"<n:,>
simulations"`, `"diversity"` and `"capped from"`; the memory line's `"CUDA-only"` on a non-CUDA
device and, on CUDA, the GiB figure, `"<n_fine:,>"` and `"upper bound"`; the checkpoint line's word
`"config"` when there is no config or prior, and `"(config.TRAINING_CHECKPOINT_EVERY = 0)"` when off
(`test_settings_persistence.py:152-159,216-229,237`).

### 6.2 Fisher settings in the record (V7)

`orchestrator.py:1185-1194` records `fisher_m`, `fisher_dz`, `fisher_points` as the values
`build_posterior` resolved at entry (§3.4) only when `fisher_evals is not None` (the one witness that
the rotation ran here, `:701,908`), and as `None` otherwise: a resumed run, a truncated round, an
unrotated run. `body.transform.fisher_eigenvalues` and `body.training.fisher_spread` are already
`None`-honest. The pin at `test_artifact_store.py:741` (an unrotated run recording the default) flips.

### 6.3 The loaded prior's figure (V8)

`build_prior`'s load branch (`orchestrator.py:496-497`) draws with a sink that closes the figure when
`fig_sink` is None, matching `ArtifactWriter.fig_sink` (`store.py:233-244`). `visualize_dist` and
`emit_figure` in `visualizers.py:99-105,479-485` keep their bare-library `plt.show()` fallback; the
three `fig_sink` docstrings (`orchestrator.py:450,623-624,1758-1759`) say one thing: every front end
passes a sink, and a stage given None closes what it draws. Pinned as the FDT precedent is
(`test_fdt_plot_functions_close_a_saved_figure_instead_of_show`): no new open figure, no
"non-interactive" warning.

### 6.4 sbi-logs (V9)

`train_nn` (`train.py:132-141`) gains `summary_writer=None`, resolved to a `_NoSummary` object with
`add_scalar(*, tag, scalar_value, global_step)`, `flush()` and `close()` that do nothing, and passes it
to `SNPE(...)` (`train.py:182`; sbi 0.25's `NeuralInference.__init__` does no `isinstance` check,
`trainers/base.py:212-214`, and `_summarize` uses only `add_scalar` and `flush`, `:932-1000`).
`build_posterior` passes nothing (the default is the stub). The repository-root `sbi-logs/` (878
timestamped directories) is deleted in the task that lands the stub; `.gitignore:1` and the scan skip
at `test_artifact_store.py:1103` go; `tests/conftest.py` gains a session-autouse guard that asserts at
teardown that `<repo>/sbi-logs` does not exist, beside `_sandbox_default_store`.

## 7. What changes on screen

- The Config tab's chi drive amplitude and band are read-only displays with a caption; its χ probe
  count, probe slots and lock-in ceiling open at `config.py`'s values every launch.
- A refused input opens the yellow "Check your inputs" box naming the box and tab; the log pane
  carries the same sentence. Refusals from a run (the direction count, a near miss, D12, a load
  mismatch) open the same yellow box instead of the red one; the red box with a traceback is for bugs.
- The TSNPE tab remembers its observation and budget; its HPD and direction boxes open at the defaults.
- The Prior, Posterior and Validate tabs' science knobs open at `config.py`'s values every launch.
- The log pane shows the memory-statistics line plain and "loaded a NON-AMORTIZED posterior" with a
  triangle.
- The FDT panel's two check boxes open at their defaults every launch (skip unticked, proceed ticked).
- Nothing else moves. No new control is added.

## 8. Tests

### 8.1 Fixtures

- `tests/conftest.py`, settings: a session-autouse `_settings_home(tmp_path_factory)` calls
  `settings.use_ini_file(str(tmp_path_factory.mktemp("settings") / "prism.ini"))` (a path that does
  not exist until Qt writes it) and asserts at teardown that the real user-scope `PRISM.ini`'s bytes
  are unchanged, mirroring the real-`Artifacts/` assertion; a function-autouse `_isolated_settings(tmp_path)`
  points at a fresh `tmp_path / "prism.ini"` per test and its teardown restores the SESSION path, never
  `None`. So a module- or session-scoped fixture that builds a panel (pytest instantiates higher scopes
  first) reads an empty file too, and §5.4 holds at every scope. A fresh path per test is the
  primitive; a shared file cleared per test would fight QSettings' file cache.
- `tests/conftest.py`, dialogs: a session-autouse guard (same `pytest.MonkeyPatch()` shape as
  `_checkpointing_off_unless_asked`) sets `QMessageBox.exec` on the class to record the box in a
  module-level `SHOWN` list and return 0; a function-autouse companion clears the list. An offscreen
  modal is then an inspectable record instead of a stall past the ten-minute tool-call limit (no
  timeout plugin is installed). Tests that want their own fake `exec` layer on top per test as today
  (`test_nav_and_gating.py:1057-1085`, `test_worker_dispatch.py:273-284`); returning 0 with no clicked
  button reads as Cancel on the consent dialogs and as the safe branch in `main_window.py:231`.
- `tests/conftest.py`: session-autouse `_no_sbi_logs` (§6.4).
- `tests/_fixtures.py`: `qt_app()` and `pump(app, seconds)` (the seven `_app` and five `_pump` copies
  move here); `code_only(obj)` (the four docstring-stripping helpers collapse into one);
  `snapshot_cfg(cfg)` and `assert_cfg_unchanged(cfg, snap)` for §2.4; `PaneCapture(panel)` recording
  `(level, text)` off `log_pane.append_line`. The four `_temp_settings` helpers and their 11 call sites
  go (`test_settings_persistence.py` ×7, `test_nav_and_gating.py` ×2, `test_worker_dispatch.py` ×1,
  `test_simulate.py` ×1); the two tests that call `use_ini_file` directly (`test_settings_persistence.py:139-167`,
  `test_user_models.py:718-739`) may drop their own redirect.
- `tests/conftest.py`: module-scoped `screen_run(tiny_run)` — an `InferenceScreen` whose session
  holds a truth-free copy of the tiny config (`cfg = tiny_run.cfg.copy_for_run(); cfg.clear_ground_truth()`;
  a copy, because `tiny_run` is module-scoped and its siblings need the truth) with `tiny_run.prior`
  and `.posterior`, for the window-level pins of §2.4 and §3 (one build per file that requests it; used
  by `test_nav_and_gating.py`).
- Core records are asserted with `caplog` (the `core` logger is at `INFO` by import, so
  `caplog.set_level` is not needed, and the window-handler and artifact-file tests must not call it);
  the tool's with `capsys` on the unchanged stdout text; the window's with `PaneCapture` or
  `log_pane.toPlainText()`. A negative pin on a message is re-pinned as `caplog.records` being free of
  the text after `caplog.clear()`; a negative left on a stdout buffer passes vacuously.

### 8.2 Tests that change

| test | change |
|---|---|
| **The verbatim print pins** (all named): `test_user_sbi.py::test_the_reported_kept_fraction_is_measured_after_the_t_scale_override`, `::test_a_tsnpe_round_reuses_the_parents_basis_and_refuses_every_mismatch` (the four non-GROUND-TRUTH stdout pins `:2371-2374,2426`), `::test_calibration_theta_star_lies_inside_the_region_when_one_is_given`, `::test_the_sweep_device_degrades_to_the_cpu_instead_of_raising` (level `warning`), `::test_the_oom_notice_is_printed_before_the_release`, `::test_log_memory_can_never_kill_a_run`, `::test_the_batch_retry_survives_a_failing_rng_restore`; `test_conditioning_repair.py::test_a_t_scale_loaded_direction_is_excluded_from_the_region`, `::test_the_round_announces_the_region_it_drew`; `test_diagnostics.py::test_the_chi_feature_set_drops_group_g_and_adds_the_fisher_block`, `::test_the_calibration_draw_is_three_helpers_with_the_stratification_seam`, `::test_sbc_prints_small_p_values_as_numbers_and_bins_ranks_at_least_ten_wide`, `::test_identifiability_rotation_reports_absent_eigenvalues_and_refuses_an_absent_rotation`; `test_artifact_store.py::test_the_budget_line_prints_with_default_arguments` | re-pinned on `caplog` records (same text, plus the level); negatives per §8.1 |
| `test_a_tsnpe_round_reuses_the_parents_basis_and_refuses_every_mismatch` (`GROUND TRUTH` in stdout AND warnings, `:2532,2541`) | the warning only |
| **V1, stubbed downstream** (capture the copy the stub receives): `test_conditioning_repair.py::test_the_round_reads_this_observations_truth_and_no_other` (`:1230-1237`: the `build_posterior` stub records its first argument; the three pins move onto it; the caller's config still `has_ground_truth`); `test_artifact_store.py::test_hand_entered_values_replace_the_recorded_cell` (`:1458-1461`: the `generate_observations` stub records its config; `"cell" not in captured.sources and captured.has_ground_truth`; the caller's `sources["cell"]` still names master_weak) | as described |
| **V1, real downstream** (read the artifact): `test_artifact_store.py::test_generate_observations_writes_an_artifact_that_reinstalls_its_context` (`:832`: `fresh.n_obs == obs.manifest.body["n_obs"] == 200`); `::test_an_experimental_observation_records_its_own_length_and_drive_frequencies` (`:1550`: `sim.manifest.body["n_obs"] == 200`, plus §2.4's caller pins) | as described |
| **V1, identity pins:** `test_artifact_store.py::test_the_compositions_forward_every_keyword_unchanged` (`:1658`) | `isinstance(seen["args"][0], SimConfig) and seen["args"][0] is not cfg`; the forwarded copy carries the composition's writes (`has_ground_truth`, `sources["cell"]`, `T_obs == 1.0 × factor`); `assert_cfg_unchanged(cfg, snap)` after each leg (never `==` between copy and caller: the composition wrote on its copy). The `x.npy` leg passes as written because the recording rule lives in the stage (§3.4) |
| `test_artifact_store.py::test_a_training_run_records_the_prior_as_the_simulation_cache_parent` (`:598`, `ck["hw"] is cfg.hw`) | `ck["hw"] == cfg.hw` |
| `test_chi_mode_full_sbi_pipeline` (slow, `:426,473-478`) | reads `chi_obs_freqs` from the observation manifest |
| **Refusal shape:** `test_tool.py::test_a_taken_name_is_refused_with_exit_1_and_nothing_written` | the line ends `(--name)`, contains no `((` and no `raised at` |
| `test_artifact_store.py::test_a_non_amortized_posterior_needs_accept_and_the_flag_is_recorded` (`:678-680`), `::test_inference_refuses_a_foreign_observation_for_a_truncated_posterior_unless_accepted` (`:985-987`), `::test_a_near_miss_cache_is_refused_before_the_fisher` (`:1330`) | `pytest.raises(Refusal, match=…)` with `field == "accept_truncated"` / `"accept_other_observation"` / `"new_run"`; no tab label or flag in the message; the `"new_run"` substring stays (Python keyword) |
| `test_tool.py::test_validate_refuses_a_non_amortized_posterior_without_accept_truncated` (`:460`), `::test_smoke_runs_the_four_stages_and_the_resume_drill_is_loud` (`:851`) | `--accept-truncated` / `--new-run` present (from the FLAG table), `raised at` absent |
| `test_tool.py::test_the_rotation_diagnostic_refuses_a_posterior_without_a_rotation` (`:771`) | keeps "no Fisher rotation", drops `raised at` |
| `test_artifact_store.py::test_zero_posterior_samples_or_calibration_points_are_refused_before_any_spend` (`:1628-1631`), `::test_flow_knobs_are_range_checked_only_when_the_call_trains` (`:1228`), `::test_resume_is_validated_and_refused_before_any_spend` (`:1208,1211,1253`) | `knob in str(e.value)` and the `match=` needles become `e.value.field == knob`; `"at least 1"` stays; `made == []` stays; the load branch also skips the budget checks |
| `test_diagnostics.py::test_sbc_refuses_before_the_spend` (`:370-376`), the guard pins at `:113-138,530-542,746-804,1256-1273` | `Refusal` + `field` (`repeats`, `n_cal`, `num_posterior_samples`, `n_points`, …); message text loses the flag and file names |
| `test_artifact_consistency.py::test_a_chi_run_at_a_non_default_band_is_refused_before_the_simulation_spend` (`:262-263`) | the `"PRISM.ini" … "QSettings"` clause becomes a pin that the message names none of "PRISM.ini", "QSettings", "Config tab"; the field-name and "config.py" pins stay |
| `test_conditioning_repair.py::test_tsnpe_round_refuses_bad_direction_counts_and_hpd_levels_before_any_spend` | asserts `Refusal` with `field == "n_directions"`/`"hpd_level"` and the default in the message |
| **Dialogs and the worker:** `test_worker_dispatch.py::test_on_error_puts_the_traceback_in_details_not_the_body` | `_on_error(RuntimeError("Something failed"), tb)`; plus a sibling for a `Refusal` → yellow box, no Details, read from `SHOWN[-1]` |
| `test_nav_and_gating.py::test_the_tsnpe_tab_dispatches_a_loaded_observation` (`:1115,1140-1145`) | the `_on_error` stub receives the exception; `_boom` raises `Refusal("Observation '20260910T120000' is 61 wide …", field="observation")`; pin `isinstance(errors[-1], Refusal)`, its field, and `"width 61" in str(errors[-1])`; a sibling asserts a bare `RuntimeError` reaches `_on_error` unwrapped |
| `test_nav_and_gating.py::test_fdt_panel_guard_translates_model_error_and_gate_admits_builtins` (`:87-113`) | a stubbed `FDTModelError` propagates unwrapped and `isinstance(e, Refusal)`; a stubbed `KeyError` still arrives as the `RuntimeError` naming the parameter and model |
| **Click-time rules:** `test_nav_and_gating.py::test_the_infer_tab_dispatches_the_compositions` (`:908-974`) | its `/tmp/…` cell and recording paths become real files under `tmp_path` (touched; dispatch is stubbed), because the click now refuses a missing file |
| `test_nav_and_gating.py::test_simulated_inference_emits_the_ground_truth_figure` (`:857-905`) | `cell=str(tmp_path / "cell.txt")` after touching it (`load_and_validate_gt` is stubbed) |
| `test_nav_and_gating.py::test_config_units_control_declares_units_and_validates_them` (`:636-638`) | stub `_refusal` (recording the exception); assert `field == "units"` and the draft unchanged |
| `test_settings_persistence.py::test_the_budget_refuses_a_batch_count_below_one`, `test_nav_and_gating.py::test_the_new_tab_knobs_are_forwarded_and_not_written_to_config` | the refusal is a yellow box (stubbed `_refusal`); blank boxes are refused rather than clamped; a load click with a blank knob box still dispatches (§1.2) |
| `test_settings_persistence.py::test_the_budget_memory_line_reads_pipelines_own_cost_model` (`:203`) | its `_budget_cfg` stub gains `inits_tensor = torch.zeros(1, 3)` so `_observation_inits` works on it |
| `test_settings_persistence.py::test_the_budget_lines_never_raise_on_a_config_they_do_not_understand` (`:231`) | unchanged in intent; now pins the preview's fail-soft branch; gains a sibling with `num_runs` `""` and `"-"` and `run_size_cap` `"-5"` asserting the lines name the box and nothing raises |
| `test_settings_persistence.py::test_the_tsnpe_tab_never_claims_it_will_resume_the_amortized_checkpoint` | rebinds `orchestrator.TRAINING_CHECKPOINT_EVERY` via monkeypatch, not `config`'s |
| `test_artifact_store.py::test_build_posterior_auto_persists_and_returns_the_loaded_wrapper` (`:741`) | an unrotated run records `fisher_m is None` |
| `test_settings_persistence.py::test_settings_round_trip_reduction_and_fdt` (`:300-301,313-314`) | after saving `skip_sanity=True`/`confirm_production=False` and rebuilding, `skip_sanity` is False and `confirm_production` is True |
| `test_the_tsnpe_new_run_box_is_not_persisted`, `test_the_vram_ceiling_is_on_config_live_and_NOT_persisted` | unchanged in intent; the source pins still hold |
| `test_the_source_scans_cover_every_code_directory` | `sbi-logs` leaves the skip set |

### 8.3 New tests, by area (names indicative)

- **V1:** `test_every_public_entry_leaves_the_callers_config_untouched` (parametrised over §2.2's
  table × success/refusal/mid-run exception); `test_copy_for_run_drops_the_caches_first_and_keeps_chi_obs_freqs`;
  `test_a_bench_chi_observation_does_not_change_the_configs_probe_count`;
  `test_a_dispatched_inference_leaves_the_session_config_pristine` (screen_run);
  `test_the_public_entry_decorator_passes_a_sentinel_through`.
- **V2/V3:** `test_the_rules_refuse_blank_zero_negative_and_missing_files_with_the_field_named`;
  `test_every_field_key_is_registered_and_mapped_by_both_front_ends` (AST scan + tables);
  `test_the_gui_control_table_matches_the_tabs_labels`; one per tab:
  `test_the_<tab>_tab_refuses_bad_boxes_at_the_click_and_dispatches_nothing` (for Prior and Posterior
  also the load click that dispatches with a blank knob box); `test_a_blank_t_obs_is_refused_on_all_three_infer_branches`;
  `test_the_driven_branch_refuses_a_zero_drive_and_a_missing_recording`;
  `test_the_compositions_refuse_t_obs_at_or_below_zero_before_any_spend`;
  `test_cal_n_scales_and_the_fisher_knobs_are_refused_not_clamped` (build_posterior resolves and refuses
  `fisher_m=0`; build_prior refuses `min_cluster_size=1`);
  `test_the_direction_refusal_names_the_width_and_the_default`;
  `test_a_refusal_opens_the_yellow_box_without_a_traceback_and_a_bug_the_red_one` (from `SHOWN`);
  `test_the_tool_prints_a_refusal_with_its_flag_and_a_bug_with_a_traceback` (one `field=None` refusal
  whose line carries no `()`); `test_device_cuda_is_refused_when_unavailable` (`detect_device`
  monkeypatched to `cpu_device()`; exit 1, `refused:`, `(--device)`); `test_a_rename_failure_reads_as_a_name_refusal`;
  `test_the_secondary_panels_still_show_a_bad_cell_as_check_your_inputs` (the four piece-5 panels).
- **V4:** `test_in_stage_messages_are_records_with_levels` (a sample per level via `caplog`);
  `test_the_core_logger_is_at_info_by_import_and_stays_so_after_main_and_a_redirect`;
  `test_the_window_handler_feeds_the_pane_at_the_records_level_and_checks_cancel` (an info record
  reaches the pane without `caplog.set_level`);
  `test_the_tool_routes_info_to_stdout_and_warnings_to_stderr_and_removes_its_handlers`;
  `test_each_artifact_gets_the_log_of_the_entry_that_wrote_it` (a composition: two files, the second a
  superset with the pre-spend warning; and a `build_posterior` with `checkpoint_every=1` leaves the
  simulation cache with no `log.txt`); `test_python_warnings_reach_the_artifact_log`;
  `test_the_gate_lines_keep_their_text_and_stream`; `test_list_dir_returns_and_the_picker_no_longer_swaps_stdout`;
  `test_the_duplicated_judgements_are_said_once`.
- **V5:** `test_science_knobs_open_at_config_and_are_not_written` (Config, Prior, Posterior, Validate,
  TSNPE; from an empty file and from a file holding stale keys);
  `test_the_chi_drive_and_band_are_read_only_and_the_draft_carries_config` (Apply with χ mode ON; the
  "Model applied" line carries `config.py`'s amplitude and band);
  `test_the_tsnpe_tab_restores_its_observation_and_budget_only` (with a blank-at-save case restoring
  the default); `test_consents_are_never_persisted` (all four); `test_get_bool_falls_back_on_an_unparseable_value`.
- **V6–V9:** `test_the_training_preview_agrees_with_the_stage_on_width_directory_and_cadence`;
  `test_the_budget_lines_only_format_the_preview`; `test_fisher_settings_are_recorded_only_when_the_rotation_ran`
  (three no-run cases); `test_loading_a_prior_with_no_sink_closes_its_figure`;
  `test_training_creates_no_sbi_logs_directory` (chdir to tmp, train, assert absent).

### 8.4 Count and budget

- **Baseline:** 460 collected at `d34997c` (457 passed, 1 skipped, 2 slow), 13 min 38 s.
- **Change:** about 80 ± 20 tests added across `test_artifact_store.py`, `test_nav_and_gating.py`,
  `test_settings_persistence.py`, `test_worker_dispatch.py`, `test_vt_progress.py`, `test_tool.py`,
  `test_diagnostics.py`, and one new suite `tests/test_refusals.py` for the torch-free rules and tables;
  about 40 existing tests change (§8.2).
- **Target:** the one-process fast gate at or under **16 minutes** (raised from 14; the real-session
  fixture costs one `tiny_run` build per file that requests it).
- **Recording:** the count and `--durations=15` at the final gate, in `docs/STATE.md`.

## 9. Gates and verification

- After each task: one `pytest -m "not slow" -q` in the background, logging to a scratch file; no
  source edit until it exits; never two pytest processes; commit the tested tree.
- Before the piece is called done: the slow set (`pytest -m slow -q`, ~31 min).
- The GPU smoke gate of `CLAUDE.md` after the piece (V1 and V4 sit on the training path): the four
  command lines and their four pass criteria unchanged. Run 1's posterior timing compares with the
  piece-2 gate (`0d96d2f`, truth-free), not with `bb38f1a`.
- A whole-piece review with a fix loop, then the slow set and the final fast gate, then the documents.
- Every refusal message change that a walkthrough row quotes is listed in §10.

## 10. Display walkthrough rows (section "Piece-3 GUI checks")

| row | check |
|---|---|
| C1 | Config tab: chi drive amplitude and band are read-only with the caption; Apply with χ mode on logs `config.py`'s values |
| C2 | A blank observation length on the Infer tab: the yellow "Check your inputs" box names the box and the Infer tab; nothing runs |
| C3 | TSNPE tab: 0 directions gives the yellow box naming "Directions truncated" and the default 5 (supersedes B5's "an error dialog") |
| C4 | TSNPE tab: after a relaunch the observation picker and budget are restored; HPD and directions show the defaults |
| C5 | D12 on the TSNPE tab arrives as the yellow box (supersedes B7's "an error dialog"; the text and "no checkbox" stand) |
| C6 | Posterior tab after a relaunch with edited network knobs: they show `config.py`'s values |
| C7 | The budget group's three lines after picking a prior: the checkpoint verdict matches what Train then does (A3/A8/B4 strings) |
| C8 | A bench chi inference with fewer probes than the Config tab's K, then a simulated chi inference: the observation records K probes |
| C9 | The log pane during training: `[mem]` lines plain, an OOM notice (if any) and "loaded a NON-AMORTIZED posterior" with a triangle; a `log.txt` exists in the posterior's folder and none in the simulation cache |
| C10 | FDT panel after a relaunch with both boxes flipped before closing: skip unticked, proceed ticked |
| C11 | Config tab after a relaunch with edited K, slots and lock-in ceiling (χ mode ticked): the three boxes show `config.py`'s values; the model, units and χ-mode tick are still remembered |

## 11. Deviations made during execution (piece 3)

Every ruling that departs from this spec, with the task that made it. Rows whose task reads "plan (Tn)"
were ruled while the implementation plan was written (`docs/superpowers/plans/2026-09-16-validation-and-logging.md`)
and are carried out by that task; every other row is appended by the task that made the ruling, in its
own commit, numbered one past the last row.

| # | task | deviation from this spec | why |
|---|---|---|---|
| 1 | plan (T4) | §3.2's `(tab, label)` entry names one tab. `num_runs` and `run_size_cap` name two, `(("Posterior", "TSNPE"), label)`, and their fix sentence reads "Set it in the '<label>' box on the Posterior or TSNPE tab." | Both tabs show the budget boxes. Naming one tab would send a user refused on the TSNPE tab to the Posterior tab. |
| 2 | plan (T6) | §3.6 says `validate_gt_file`'s strings come from the same rules as the cell refusal. In this piece they stay two wordings: T6 makes `load_and_validate_gt`'s refusal a `Refusal(field="cell")` and leaves the dry run's problem strings as they are. | Sharing them needs one message builder that the pick-time cell check and the secondary panels' cell pickers both consume; piece 5 adopts the rules for those pickers and owns it. |
| 3 | T2 | §2.1's "the function builds a new one per call" is false: `config.unit_registry` is `@lru_cache(maxsize=1)` and `unit_registry() is unit_registry()` holds (probed at `ca583f8`), so its docstring and the two `cli.py:187,214` comments describe the registry the code HAS. They were not "corrected" into an error: the docstring gains the sentence V1 relies on (a deep copy would duplicate the singleton; `copy_for_run` pops `_ureg` first and the copy recomputes to the same instance), and the `cli.py` comments stay as they are. | The plan re-reads before it edits, and a docstring must describe the code. |
