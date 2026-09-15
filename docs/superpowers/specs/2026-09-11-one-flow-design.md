# PRISM piece 2: One flow underneath (design)

**Date:** 2026-09-11 · **Status:** approved in the brainstorming session of 2026-09-11 · **Owner:** J
**Parent:** `2026-09-10-pre-retrain-hardening-decomposition.md` §4 row 2 · **Read at:** HEAD `44e6081`. Every `file:line` below was checked against that tree.

## 1. Purpose and scope

Today the inference flow exists in three copies:
- the GUI runners (`core/gui/panels/inference/runners.py`, 58 lines);
- the prompt CLI's `orchestrator.run` (`core/orchestrator.py:67-131`);
- `scripts/_common.py` plus the twelve scripts beside it, driven by environment variables (thirteen files in all).

Each copy carries checks the others lack. After piece 2 there is ONE flow:
- the existing orchestrator stages;
- three composition functions that sit beside them;
- two front ends over those: the GUI and a flags-only command-line tool, `python -m core <subcommand>`.

The piece also does the following:
- It retires the prompt CLI.
- It folds six scripts into four subcommands, archives six, and removes `scripts/`.
- Diagnostics write a new store kind.
- The GPU-gate incident of 2026-09-11 becomes a refusal inside the training stage. In that incident, run 2 silently started a new cache under an identity that differed from run 1's in one field.

### 1.1 Decisions (binding)

| # | decision | where |
|---|---|---|
| D1 | Retire the prompt CLI whole: `core/__main__.py`'s menu, `orchestrator.run`, `core/app.py`, the prompt half of `core/cli.py`, the callers of `helpers.clear_screen`, and `run_fdt`'s `input()` calls. `run_fdt`'s booleans become required. The tool gets thin `fdt` and `crossval` subcommands, tested at tiny size, with no new science. Reduction gets no subcommand. | §2.12, §3.5, §6 |
| D2 | Build the full command-line tool the decomposition planned. | §3 |
| D3 | Keep the step functions and the module identity `core.orchestrator`. Add `simulated_inference`, `experimental_inference` and `tsnpe_round`, and move into them the checks that today live in only one front end. The GUI's calls become one line. There is no load/build split. | §2 |
| D4 | Six scripts become four subcommands: `smoke`, `sbc`, `identifiability` (three modes, from three scripts) and `ablation`. Six are archived: moved on disk, then `git rm --cached`. `_common` dissolves. The stage subcommands are `prior`, `train`, `tsnpe`, `validate` and `infer`. | §3, §4, §6 |
| D5 | Add a new store kind, `diagnostic`, with its own directory and manifest. Its parents are the artifacts it read, and its loader has no refusals. | §4.1 |
| D6 | The tool takes argparse flags only, and every flag reaches its stage as a keyword argument. `--bounds` is required wherever a prior is built or training runs. The tool reads only `PRISM_RESOURCES` and `PRISM_ARTIFACTS` from the environment. The GPU gate becomes a set of command lines. `main(argv) -> int` can be tested in-process. | §3 |
| D7 | `build_posterior` refuses a near miss before any simulation and before the Fisher: a sibling cache that differs in exactly one field. `new_run=True` overrides the refusal. The GUI dialog passes that choice, and the tool gets `--new-run`. The resume policy `auto`/`require`/`never` becomes a stage argument and the `--resume` flag. | §2.7 |
| D8 | The tool refuses by default. `--accept-truncated` and `--accept-other-observation` map 1:1 onto Accept. In the GUI, the Posterior tab asks before it loads a stored non-amortized posterior; a round the user just ran installs itself without asking; the Infer tab gets an other-observation checkbox. | §3.4, §5 |
| D9 | Every driven chi recording states its frequency in Hz. The legacy branch without a frequency is deleted, and the slow test passes the frequencies it simulated. | §2.10 |
| D10 | Invocation is `python -m core <subcommand>`. The entry sets `KMP_DUPLICATE_LIB_OK` (with setdefault) and the Agg backend before any torch or core import; `registry.load_user_models()` then runs inside `main`, after parsing and before any handler, so `--help` stays torch-free. | §3.1, §3.2 |
| D11 | The chi override hatch is deleted, not threaded as an argument. A non-default chi band or drive amplitude is refused on every surface; changing one means editing `config.py` deliberately. | §2.9 |
| D12 | A TSNPE round whose parent is itself non-amortized and whose observation is not that parent's is refused, with no escape hatch. | §2.5 step 3 |
| D13 | `PRISM_VRAM_CEILING_GIB` and `PRISM_MEM_LOG_EVERY` stay as core-level environment settings. They are documented in the tool's `--help` epilog and in `CLAUDE.md`, and they are not turned into arguments. | §1.2, §9.5 |

### 1.2 How the design reads the decisions

These readings go beyond the literal text of a decision. They are recorded here so that the plan does not treat them as settled by the decisions themselves.

| topic | reading | why |
|---|---|---|
| D3 "GUI runners shrink to one-line calls" | `runners.py` is deleted and the tabs dispatch the compositions directly. The dispatch line is the one-line call. | The file existed only to be module-level and Qt-free with an injectable `fig_sink` (`runners.py:1-2`), and the compositions meet both conditions. A wrapper would add a hop where positional order can drift. |
| D7 "differs in EXACTLY ONE field" | The existing rule, `training_checkpoint.near_miss_siblings`, is kept unchanged. It exempts a sibling that differs in `truncation` alone (`training_checkpoint.py:315`, reasoned at `:307-310`). | A TSNPE round at its parent's budget differs from the parent's cache only in the region. That round is legitimate and is pinned (`test_user_sbi.py:2437-2441`). |
| D6 "every setting is a flag that travels to the stage" | `--seed` exists only on `smoke` and on the diagnostics. Stages take no seed (§2.8). | The only seeded caller being folded is `smoke_train.py:172`, which seeds once. Per-stage seeds would restart the calibration and training streams from the same state (§2.8). |
| D6 "no environment" | `PRISM_VRAM_CEILING_GIB` (`pipeline.py:470`) and `PRISM_MEM_LOG_EVERY` (`pipeline.py:155`) are read by core, not by the tool, and they stay. They are listed in the `--help` epilog and in `CLAUDE.md`. | D13. The GUI Config tab reads the ceiling too (`config_tab.py:117`). |
| hand-entered truth | `simulated_inference` pops `cfg.sources["cell"]` when it is given hand-entered values. | Every hand-entered GUI inference now passes through this new function, which writes provenance. `inject_ground_truth` never touches `sources` (`sim_config.py:253-278`), so without the pop the earlier cell is recorded and hashed (`orchestrator.py:279-282`, `provenance.py:103-106`). The fix is one line. |

### 1.3 Out of scope, and who owns it

| item | owner |
|---|---|
| Boundary validation of every field. Logging with severity in place of `print`, including the in-stage prints this piece leaves. Copy-on-run session config, which also covers an amortized training in a session whose cfg carries a truth: that training anchors the Fisher on the truth (`decorrelate.py:291-292`). Error dialogs that name the fix. `tsnpe_tab.restore_settings`, which is never called (`tsnpe_tab.py:154-169`). The Config tab's chi drive and band fields, editable and QSettings-restored while every non-default value is refused (§2.9). The budget status line's separate identity derivation (`base.py:154-182`). The TSNPE manifest recording Fisher defaults that did not run (`orchestrator.py:1075-1080`). `build_prior`'s load branch calling `plt.show()` when `fig_sink` is None (`orchestrator.py:440`, `visualizers.py:100-103`; every caller passes a sink). sbi writing `<cwd>/sbi-logs` (`train.py:182`). | piece 3 |
| The artifact browser, including a listing of diagnostics. Annotate (`set_note` has no GUI caller). Cleanup of incomplete directories. | piece 4 |
| FDT/CrossVal hardening, and wrapping them in the store. Their outputs stay in `artifacts_root()/fdt` and `/crossval`. | piece 5 |
| The `docs/` split of the handoff, including the `PRISM_HANDOFF.md` lines D1 makes false (`:49,53-54,76,190-191`). A README reference section for the tool. | piece 6 |
| Reduction on the command line. Its GUI panel stays. | out of the programme |

## 2. The stage API (core)

### 2.1 Shape

The seven step functions keep their names, their positional shapes and their module:

| function | line |
|---|---|
| `generate_observations` | `:135` |
| `build_experiment_observation` | `:295` |
| `build_prior` | `:370` |
| `build_posterior` | `:541` |
| `build_truncation_region` | `:1119` |
| `validate_calibration` | `:1330` |
| `infer_and_visualize` | `:1581` |

The three compositions are defined **in `core/orchestrator.py`**, after `infer_and_visualize` and before the `observations` re-export (`:1834-1838`). The suite patches stages by assigning to attributes of the module object:
- `tests/test_nav_and_gating.py:815-827`
- `tests/_fixtures.py:184-198`
- `tests/test_user_sbi.py:52,78-79,417-418`

and it reaches core's guards through that same module object (`tests/test_artifact_consistency.py:163-172,253-288`).

A function defined inside the module resolves `generate_observations`, `pipeline.X` and `cli.X` through the module globals at call time, so every such patch reaches it. A re-export from another module would be a second binding, and the patches would miss it.

Rules every composition follows:
- Look up stages and `cli.*` at call time; never `from`-import them.
- Refuse before the first simulation.
- Take `cfg` first and positionally.
- Return only `Loaded*` wrappers.
- Forward an optional keyword to a stage only when it is set, so that stage defaults live in one place.

### 2.2 One message channel

```python
class PreflightWarning(UserWarning):
    """A judgement reported before or instead of refusing: out-of-distribution truth, a truth outside a
    non-amortized posterior's region, T_obs outside the training range, an HPD tighter than recommended,
    cell values the bounds ignore."""
warnings.filterwarnings("always", category=PreflightWarning)       # module level
def _preflight_warn(msg: str) -> None: warnings.warn(msg, PreflightWarning, stacklevel=3)
```

Why this channel:
- The GUI routes `warnings.showwarning` to the log pane at warning severity (`core/gui/streams.py:244-247`), while a `print` lands at info (`:240`).
- `orchestrator.run` already used `warnings.warn` for these messages (`:102-113`).
- Tests can assert them with `pytest.warns`.
- The `"always"` filter defeats Python's once-per-location registry. Without it, a second GUI inference would stay silent about a repeated out-of-distribution truth.

Core installs no global `ignore` filter; the pipeline's filters are local `catch_warnings` blocks (`pipeline.py:969,1026,1080`). The tool therefore does **not** port `_common.enable_warnings`, which existed only to undo the scripts' own blanket ignore (`_common.py:61-79`), and Python's default `showwarning` prints to stderr. Converting the existing in-stage prints is piece 3's work.

### 2.3 `simulated_inference`

```python
def simulated_inference(cfg, posterior: LoadedPosterior, T_obs_s: float, *,
                        cell=None, gt_values=None, prior: LoadedPrior | None = None,
                        accept: Accept | None = None, n_samples: int | None = None,
                        name: str = "", note: str = "", fig_sink=None, store=None
                        ) -> tuple[LoadedObservation, LoadedInference]
```

- `gt_values` has the `(inits, params, rescale, forcing)` shape that `parse_values_file` returns.
- Give exactly one of `cell` and `gt_values`; anything else raises `ValueError`.
- `name` and `note` name the inference. The observation stays unnamed, as it is in the GUI today.
- `n_samples` is forwarded to `infer_and_visualize` only when set. Its default stays that stage's literal 1000 (`:1583`).

Steps:
1. `store = resolve_store(store)`, then `store.assert_name_free("inference", name)`. Today this check first runs inside `infer_and_visualize` (`:1602`), which is after the observation has been simulated and written.
2. `accept = accept or Accept()`. If `posterior.posterior.x_obs_digest is not None and not accept.other_observation`, raise `ValueError`. A freshly simulated observation carries new noise, so it can never carry the region's digest. The message names the Infer-tab box "Run on a different observation", `--accept-other-observation`, and `Accept(other_observation=True)`. Today the refusal fires at `:1617-1626`, after the observation has been written, and leaves an orphan behind. `x_obs_digest`, `truncation` and `amortized=False` are written together (`:1092-1093`) and are never set independently, so this predicate, the Infer tab's gate on `truncation` (§5.5) and the store's load refusal on `amortized` pick out the same posteriors.
3. Inject the truth:
   - with `cell`: `ignored = cli.load_and_validate_gt(cfg, cell)`, which sets `cfg.sources["cell"]` (`core/cli.py:248`);
   - with `gt_values`: `ignored = cfg.inject_ground_truth(*gt_values)`, then `cfg.sources.pop("cell", None)` (§1.2).
4. If `ignored` is non-empty, warn once: `_preflight_warn("the bounds file does not declare …; those cell values were ignored (the bounds file defines the inferred set)")`. This one wording replaces three: `orchestrator.py:100-101`, `runners.py:17-18` and `_common.py:201-203`.
5. `cfg.T_obs = T_obs_s * cfg.get_unit_conversion_factor("s")`.
6. **T_obs range check.** If `T_obs_s` is below `T_MIN_EXP_S` or above `T_MAX_EXP_S`, call `_preflight_warn` with the text from `:106-113`. Today only `run` has this check. It reads only `T_obs_s` and the two imported constants, so the minimal `Cfg` stub in `test_nav_and_gating.py:801-805` still works.
7. **Out-of-distribution check**, only when `prior` is given. Every message from `check_observation_in_distribution(cfg, prior.prior, prior.force_prior)` (`:1253`) goes to `_preflight_warn`. With no prior the check is skipped silently, as `runners.py:21` does today. The tool always supplies a prior (§3.4), and so does the GUI (the session's `inf_prior`).
8. **Region check.** This applies only when `region = posterior.posterior.truncation` is not None, so it is reachable only after an accepted step 2. `check_observation_in_distribution` samples the FULL base prior, while a non-amortized posterior was trained on that prior restricted to its region. So for each message from `_truth_outside_region(posterior.posterior.T, region, cfg.ground_truth_tensor)` (§2.11), call `_preflight_warn`, prefixed "this truth lies outside the region the NON-AMORTIZED posterior was trained on; the flow extrapolates there:". This path is new: D8 makes `Accept(other_observation=True)` reachable for the first time. Its only caller today is `test_artifact_store.py:894`.
9. `obs = generate_observations(cfg, fig_sink=fig_sink, store=store)`.
10. `inf = infer_and_visualize(cfg, posterior, obs, name=name, note=note, fig_sink=fig_sink, store=store, accept=accept, **({"n_samples": n_samples} if n_samples is not None else {}))`.
11. Return `(obs, inf)`, the payload shape that `infer_tab._on_observation` unpacks (`infer_tab.py:376`).

### 2.4 `experimental_inference`

```python
def experimental_inference(cfg, posterior, rec: RecordingSet, *, accept=None, n_samples=None,
                           name="", note="", fig_sink=None, store=None) -> tuple[...]
```

1. Resolve the store and check that the inference name is free, as in §2.3.
2. `obs = build_experiment_observation(cfg, rec, fig_sink=fig_sink, store=store)`. D9's refusal lives in that stage (§2.10).
3. Call `infer_and_visualize(...)`, forwarding `n_samples` only when set, and return.

This composition adds no new checks:
- A recording has no truth, so there is no out-of-distribution check.
- The T_obs range check is not added here. No path checks recordings today, and D3 moves existing checks; it does not create new ones.
- There is no up-front non-amortized refusal. Rebuilding the same recordings reproduces the region's digest, which is the legitimate match. The refusal stays in `infer_and_visualize`, and building the observation costs no simulation.

### 2.5 `tsnpe_round`

```python
def tsnpe_round(cfg, posterior: LoadedPosterior, prior: LoadedPrior, observation: LoadedObservation, *,
                n_directions=None, level=None, num_runs=None, run_size_cap=None,
                hidden_features=None, num_transforms=None, learning_rate=None, stop_after_epochs=None,
                max_num_epochs=None, checkpoint_every=None, resume="auto", new_run=False,
                name="", note="", fig_sink=None, store=None) -> LoadedPosterior
```

The positional order `(posterior, prior)` matches `validate_calibration` and today's runner. The function takes **no Fisher knobs, by design**: with `truncation` set, `build_posterior` takes the first arm (`:784-861`), reuses `truncation.V` (`:839`) and never reaches the Fisher (`:885`).

1. `store = resolve_store(store)`; `store.assert_name_free("posterior", name)`.
2. Resolve `n = truncate.DEFAULT_N_DIRECTIONS if None else int(...)` and `q = truncate.DEFAULT_HPD if None else float(...)` (`core/SBI/truncate.py:68,70`). With `P = len(cfg.params_dict) + len(cfg.rescale_params)`:
   - `n < 1` raises `ValueError("At least one direction must be truncated")`. Today `truncate.py:340` silently clamps it to 1.
   - `n > P` raises `ValueError(f"{n} directions requested but the latent has {P}; truncating every direction deletes support along the flat ones too")`. `truncate.py:340` also clamps this side silently, and guardrail 3 needs the flat directions left full width.
   - `q` outside `(0, 1)` raises `ValueError("HPD level must be strictly between 0 and 1")`. Today `truncate.py:341` has no guard.
   - `q < 0.99` calls `_preflight_warn("HPD {q:g} is tighter than the recommended 0.999. Truncation permanently deletes prior support; no later round can recover it.")`.

   These checks leave `tsnpe_tab.py:90-102`, so only this copy remains.
3. **New refusal** (D12; there is no escape hatch). If `posterior.posterior.x_obs_digest is not None` and it differs from `observation.digest`, raise `ValueError`. A non-amortized parent is valid only near its own observation, so a region drawn around another observation would be drawn where the flow extrapolates. This case is reachable today by loading a posterior on the Posterior tab and then using the TSNPE tab, and `build_truncation_region` does not check it (`:1143-1212`).
4. `observation.install(cfg)` (`store.py:110-130`). After §2.11's fix, a simulated observation puts back its own truth and an experimental one clears any truth left on the cfg. Either way, the round's ground-truth-in-region check (`:913-933`) reads this observation's truth or none. Training reads nothing that `install` sets: the identity holds no `T_obs`, `n_obs` or `chi_n_freqs` (`identity.py:41-75`).
5. `t_scale_idx = len(cfg.params_dict) + cfg.rescale_idx["t_scale"]`, as `runners.py:53` computes it.
6. `region = build_truncation_region(posterior, observation, n_directions=n, level=q, t_scale_idx=t_scale_idx)`. A region with no eligible direction is refused inside `truncate.region_from_posterior` (pinned at `test_conditioning_repair.py:890-895`). The draw is seeded from the observation under `fork_rng` (`truncate.py:316-334`).
7. `print(f"[tsnpe] region from observation {observation.name or observation.id}: {region!r}")`. This line is kept from `runners.py:54`; walkthrough row A7 quotes it.
8. `return build_posterior(cfg, prior, None, True, truncation=region, observation=observation, parent_posterior=posterior, …)`, forwarding every knob above only when it is set (`resume` and `new_run` are always forwarded) together with `name=`, `note=`, `fig_sink=` and `store=`.

**Guardrail audit (PRISM_HANDOFF.md §11.6)**

| guardrail | why `tsnpe_round` cannot break it |
|---|---|
| propose from the truncated prior | Only `truncation=region` is passed. `build_posterior` wraps the latent prior in `TruncatedLatentPrior` (`:934`), and the posterior supplies samples only for the box (`truncate.py:334`). Pinned at source (§8.1 row 2). |
| 1 region only around a recorded observation | The round takes a `LoadedObservation`, which is hashed at load. `build_truncation_region` re-checks the digest (`:1144-1148`). Refusal 3 covers a non-amortized parent. |
| 2 NON-AMORTIZED marked | `build_posterior` writes `amortized=False` and the region (`:1092-1093`). The store refuses to load it without `Accept(truncated=True)` (`store.py:600-610`). |
| 3 Fisher eigenbasis; flat directions left full width | The region is measured in the parent's latent coordinates (`:1162-1212`). The `n > P` refusal applies. |
| 4 unweighted draws | `region_from_posterior` samples the flow. |
| 5 generous HPD; a truth outside the region is reported | `DEFAULT_HPD` is the default, and an HPD below 0.99 warns. The truth check reads the observation's own truth after step 4. The same wording is used at inference (§2.3 step 8). |
| 6 show the cost | The GUI budget group is unchanged. On the command line, `build_posterior`'s budget line is now unconditional (§2.6). |
| 7 the basis travels, never the Fisher | Unchanged (`:793-861`). Fisher knobs are not accepted. |
| 8 calibrate on the truncated prior | `validate_calibration` reads `.posterior.truncation` (`:1380,1399-1406`). `sbc_repeats` shares that code (§4.3). |

### 2.6 New keyword arguments on existing stages

These follow the existing pattern, which is pinned by `tests/test_user_sbi.py:3301-3336`:
- keyword-only, with default `None`;
- resolved in the body as `CONST if x is None else cast(x)`.

`resume` (a literal default) and `new_run` (a consent boolean) fall outside that pattern, so each gets its own tests.

| stage | new kwarg | resolves / replaces | recorded in manifest `config` | part of simulation identity |
|---|---|---|---|---|
| `build_posterior` | `checkpoint_every: int \| None = None` (0 = off) | `TRAINING_CHECKPOINT_EVERY`, read at `:766` and `:982`; replaces the rebinds at `smoke_train.py:227,234` | yes | **no** (the checkpoint header stores no cadence either, `training_checkpoint.py:169-180`) |
| `build_posterior` | `max_num_epochs: int \| None = None` | `TRAINING_MAX_NUM_EPOCHS` (`:1020`); `smoke_train.py:194` | yes | no |
| `build_posterior` | `resume: str = "auto"` | refuses anything outside `{"auto","require","never"}`; replaces the hard-coded `"auto"` (`:982`). `pipeline.gen_training_data` already honours all three values (`core/SBI/pipeline.py:1359-1368`). | no (the outcome is recorded as `training.resumed_from_batch`, `:1095`) | no |
| `build_posterior` | `new_run: bool = False` | D7 consent (§2.7) | no | no |
| `cli.make_sim_config` | `hw: DeviceConfig \| None = None` | `detect_device()` when None; replaces the hard-coded call at `cli.py:552` | already recorded (`device`, `dtype`) | already (`identity.py:72-73`) |

Where these land in `build_posterior`:
- `ck_every` is resolved beside `n_runs` (`:649-655`), and `resume` is validated there.
- If `resume != "auto"` while `ck_every == 0` and `train_new`, raise `ValueError`: the policy has no cache to act on. Without this refusal, a `--resume require` drill with checkpointing off exits 0 having tested nothing.
- `:766` becomes `if ck_every and train_new:`.
- `:982` becomes `"every": ck_every, "resume": resume`.
- `:1020` passes `max_num_epochs=max_ep`.
- `:1078` gains `checkpoint_every` and `max_num_epochs`. `config` is free-form (`manifest.py:129-131,147`).
- **Budget line.** After `run_size` is resolved (`:746-755`), print this line unconditionally: `f"[budget] {n_runs:,} batches x {run_size:,} rows = {n_runs * run_size:,} training rows"`. The two conditional announcements stay. This is guardrail 6 on the command line. No test pins these prints (grep).

**Not added (YAGNI):**
- `seed` on any stage (§2.8).
- `SBC_N_CAL` and `TRAINING_NUM_RUNS`, which are already arguments.
- `num_posterior_samples` and `n_samples`, which have literal defaults.
- `DENSITY_ESTIMATOR`, `NSF_NUM_BINS`, `TRAINING_BATCH_SIZE` and `TRAINING_SHOW_SUMMARY`.
- `chi_k_fixed` on `validate_calibration`; stratification belongs to `sbc_repeats` (§4.4).
- An argument for `smoke_train.py:272`'s `cfg.hw.batch_size = RUN_SIZE`: passing `run_size_cap=RUN_SIZE` gives the identical identity, because `run_size = min(hw.batch_size, cap)` (`:746-750`).

### 2.7 D7: the near-miss refusal and the resume policy

**One detector, shared by the stage and the GUI:**

```python
def fresh_run_near_misses(cfg, prior, *, num_runs=None, run_size_cap=None, truncation=None,
                          checkpoint_every=None, store=None) -> list[dict]:
    """training_checkpoint.near_miss_siblings for the identity build_posterior WOULD use. [] when
    checkpointing resolves to off, when the run's own cache already holds batches (a resume), or when
    no committed sibling is one field away. Pure; fails open for a stub prior (identity.py:30-32)."""
```

How the detector resolves things:
- `n_runs` and `run_size` are resolved exactly as `build_posterior` resolves them. `run_size` comes from a new pure helper, `_training_run_size(cfg, size_cap)`, which `build_posterior` (`:746-750`) also calls; the prints stay in `build_posterior`.
- The identity is `training_identity` (`:360`), and the root is `resolve_store(store).kind_dir("simulation")`.
- The sibling rule is `near_miss_siblings`, unchanged (`training_checkpoint.py:294-316` over `_sibling_diffs` `:231-252`). A sibling is any committed directory, complete or partial, with `batches_done >= 1` that differs in EXACTLY one identity field, where that field is not `truncation` alone (§1.2).
- Two or more differing fields mean a different experiment and are never refused (`test_settings_persistence.py:585-591`).

**Where the check sits in `build_posterior`.** It goes inside the `if ck_every and train_new:` block, after `_st = training_checkpoint.peek(ckpt_dir)` (`:772`) and before `rotate = cfg.reparam_rotate` (`:780`). That puts it:
- after every entry refusal: the taken name (`:624`, pinned at `test_artifact_store.py:505-513`) and the region without a digest (pinned at `:980-990`);
- before the Fisher (`:881-894`), before any simulation, and before the checkpoint's `create()` (`pipeline.py:1414`).

```python
if _st and _st.get("batches_done"):
    ckpt_resumed = training_checkpoint.read_header(ckpt_dir)      # unchanged
else:
    near = training_checkpoint.near_miss_siblings(ident, store.kind_dir("simulation"))
    if resume == "require":
        raise ValueError(_no_cache_message(ckpt_dir, near))
    if near and not new_run:
        raise ValueError(_near_miss_message(ckpt_dir, near))
```

`build_posterior` shares the detector's internals rather than restating them. The `never` refusal stays only in the pipeline (`pipeline.py:1362-1366`): when the run's own directory has batches, the Fisher is skipped anyway, so that refusal costs nothing. Today the pipeline's `require` refusal (`:1367-1368`) fires only after a freshly computed Fisher, so it is hoisted here to fire before the Fisher; the pipeline keeps its copy as a second line.

`_near_miss_message`:

```
This run would start a NEW simulation cache at <ckpt_dir> from zero, but a committed cache ONE setting away exists:
  <name>: <batches> batches (<complete|partial>) -- differs only in <field>: this run <mine!r>, that cache <theirs!r>
  (up to three, richest first)
If you meant to continue that cache, set <field> back to <theirs!r>. To start a new cache anyway, pass new_run=True
(GUI: "Start a new run anyway"; command line: --new-run).
```

`_no_cache_message`: "resume='require' but there is no resumable cache at <dir>", followed by the same near-miss lines.

How the settings interact:

| setting | behaviour |
|---|---|
| `auto` | Resumes its own directory; otherwise refuses on a near miss unless `new_run`. |
| `require` | Resumes its own directory, or refuses and names the near misses. |
| `never` | Refused by the pipeline if its own directory has batches; otherwise behaves like `auto`. |
| `new_run=True` | Only silences the near-miss refusal. It never forces a fresh start over a resumable directory of the run's own. |
| `ck_every == 0` | No near-miss check, because nothing is read or written; any `resume` but `auto` is refused up front by §2.6, since no cache exists for a policy to act on. |

**Defect fixed by sharing the detector.** The GUI dialog's identity check (`posterior_tab.py:119`) is wrong in two ways:
- it uses `run_size = cap or _hw_batch(cfg)`, while the stage uses `min(hw, cap)`;
- it looks in `_default_root()` (`training_checkpoint.py:93-95`, via `resolve_dir(ident)` at `posterior_tab.py:161`), not `store.kind_dir("simulation")`.

With a cap above the hardware batch, the GUI therefore asks about a directory the run never touches. Both callers now use the one detector.

### 2.8 Device, and why stages take no seed

**Device.** `make_sim_config(..., hw=None)`. The GUI (`session.py:32-37`) passes nothing, so its behaviour is unchanged. The tool builds `hw` from `--device` (§3.3).

**No per-stage seed.** Seeding every stage with one value would start draws that should be independent from identical RNG states:
- Training and calibration both draw initial conditions from numpy's global RNG (`pipeline.py:1113-1117`, and `analysis.py:151-163` through the same `gen_training_data`).
- Both draw their Sobol `(t_scale, T)` scramble from torch's global RNG (`pipeline.py:1161`).

On a TSNPE round, or with the rotation off, nothing consumes the stream between the seed and the pipeline. The calibration set would then replay the training strata, and trap X5 makes those operating points t_scale's effective SBC sample size.

`smoke_train` seeded once and let the streams run on (`smoke_train.py:172`). The tool does the same:
- `smoke` runs its whole stage sequence inside one `seeded(seed, device)` context (§4.2), with a default seed of 0.
- Diagnostics seed inside the same context, per repeat or per mode.
- The context restores the caller's RNG, so an in-process `main` never changes the pytest process's streams.

Seeded runs are not bitwise-reproducible on CUDA or across devices; the smoke `--help` says so.

### 2.9 The chi override hatch is deleted (D11)

The override is an environment variable that no retained caller uses. Its one sanctioned user, `chi_f0_sweep.py`, is archived. D6 forbids the tool from reading it, and CLAUDE.md forbids settings that are not arguments. So:
- Delete `CHI_OVERRIDE_ENV` (`core/SBI/run_guards.py:21`), its environment branch (`:152-155`), and its re-import (`orchestrator.py:41`).
- `_assert_chi_config_is_deliberate` refuses unconditionally. Its last line (`:165`) becomes: "A non-default band or drive amplitude is not supported: set the Config tab back to config.py's values (or edit config.py itself, deliberately, for every future run)."
- Rewrite the `:raises` clause of `_assert_chi_config_is_deliberate` (`run_guards.py:133-135`), which still documents the hatch, to `:raises ValueError: on any band/drive mismatch. There is no override: a non-default band or drive amplitude means editing config.py deliberately (D11).` This belongs to T3, with the deletion, not to T23's comment sweep.
- The tool offers `--chi` / `--no-chi` and `--chi-k` only. There is no `--chi-f0` and no `--chi-band`, because any value other than config.py's would be refused.
- The GUI needs no code change: a non-default band is refused there today unless the variable is set, and after this deletion it is refused unconditionally.

The alternative -- `chi_override: bool = False` threaded through `_assert_chi_config_is_deliberate`, `build_prior`, `build_posterior`, `ArtifactStore.load_posterior` and `tsnpe_round`, recorded in the manifest, with `--chi-f0`, `--chi-band` and `--chi-override` in the tool -- was DECLINED (D11): a task and five signatures for a hatch with no remaining caller.

**Consequence for the GUI, recorded here because piece 2 does not fix it.** The Config tab's chi drive and band fields stay editable and are restored from QSettings (`config_tab.py:57-58,262-264,283-285`), so a value other than `config.CHI_F0` / `config.CHI_FREQ_BOUNDS` is now accepted by the form and refused at `build_prior`. The refusal message names the fields, so the run is never silently wrong. Making those two fields read-only belongs to piece 3's "science constants never restored from QSettings" (§1.3).

### 2.10 D9 inside `build_experiment_observation`

- In the file-check loop (`orchestrator.py:302-309`), before any load. The loop iterates `named = [(rec.spont, "spont", None)] + [(p, "forced", f) for p, f in rec.forced]` (`:302`), so its FIRST element is the passive recording with `f is None` BY CONSTRUCTION: the refusal must be role-scoped or it rejects every chi observation. `if cfg.observation_mode == "chi" and role == "forced" and f is None: raise ValueError(f"chi mode: forced recording {p!r} has no drive frequency; every driven chi recording must state the frequency (Hz) it was driven at")`.
- `:313` collapses to `(x, float(f))`.
- `build_experiment_obs_chi` takes pairs only. Delete `paired` and `mults` (`core/SBI/observations.py:219-223`), the `else` arm (`:229-232`), and the docstring's "legacy" clause (`:193-195`).
- **Forced (single-drive) mode keeps `(path, None)`.** Its drive is `forcing_params_si` (`observations.py:21-28`; GUI `infer_tab.py:370`; `test_artifact_store.py:818,826` unchanged).
- The slow test (`tests/test_user_sbi.py:443`) passes the frequencies it simulated: `forced=tuple((str(p), float(f) / cfg.freq_si_to_cell) for p, f in zip(forced_paths, cfg.chi_obs_freqs))`. `generate_observations` stores `cfg.chi_obs_freqs` in cell units (`:264-265`), and `observations.py:232` converts from cell units to Hz by dividing by `freq_si_to_cell`.

### 2.11 Two science fixes on paths piece 2 creates

1. **An experimental observation clears a stale truth.**
   - *The defect.* `LoadedObservation.install` injects a truth only when `source.kind == "simulated"` (`store.py:124-127`), and nothing ever clears one (`sim_config.py:253-278`). So after a simulated inference, an experimental observation's round reports the stale cell as "the loaded cell's GROUND TRUTH lies OUTSIDE the truncation region" (`orchestrator.py:913-933`), or is silently satisfied by it. The same stale inits feed the experimental PPC (`_observation_inits`, `:1317-1318`).
   - *The fix.* Add `SimConfig.clear_ground_truth()`, which sets every value slot in `params_dict` and `rescale_params` to None, empties `inits_dict`, and pops `sources["cell"]`. `has_ground_truth` then reads False (`sim_config.py:181-184`), and the PPC falls back to the documented experimental inits (`:1319-1325`). `install` calls it when `body["source"]["kind"] != "simulated"`. Its contract remains "put back THIS observation's context".
   - *Callers.* `install` has one caller today (`orchestrator.py:1632`), after the refusals, and `tsnpe_round` becomes the second.
2. **One wording for "the truth lies outside the region".** Factor build_posterior's `:913-933` message builder into `_truth_outside_region(T, region, truth) -> list[str]` (the latent inverse, `region.contains` on CPU float64, and the per-direction list). Two places call it:
   - the round's guardrail-5 block, whose printed text and `warnings.warn` stay;
   - `simulated_inference` step 8.

### 2.12 Deleted from core (D1)

| delete | where |
|---|---|
| `orchestrator.run` | `core/orchestrator.py:66-131`. Its only callers are `__main__.py:25` and `app.py:39`. **Keep** the `T_MIN_EXP_S`, `T_MAX_EXP_S` and `warnings` imports, which §2.3 uses. |
| `core/app.py` | the whole file (44 lines, no callers) |
| the menu body of `core/__main__.py` | `:1-34`. In T10 the file becomes a stub that prints "the command-line tool arrives with the next commits; the GUI is `python -m core.gui`" and exits 2. T11 rewrites it as the tool entry (§3.1), so no commit leaves an entry that fails at import. |
| the prompt half of `core/cli.py` | `_prompt_index` 31, `select_model` 61, `select_cell_file` 85, `select_bounds_file` 97, `get_time_params` 121, `_pick_artifact` 132, `select_or_build_prior` 152, `select_or_train_posterior` 167, `prompt_save_name` 182, `select_inference_mode` 193, `get_inference_inputs` 256, `get_inference_inputs_chi` 282, `get_inference_inputs_spontaneous` 303, `select_mode` 315, `_prompt_int` 341, `_prompt_float` 345, `build_sim_config` 558-567, `build_fdt_config` 594-611, `build_reduction_config` 614-631, `_select_sweep_preset` 667-679, `build_param_sweep_config` 682-725. The imports left unused go, including `helpers` at `:19`. The module docstring (`:1-6`) is rewritten. **The module keeps the name `core.cli`**; renaming it would touch about 70 test sites for a name alone. |
| `helpers.clear_screen` | `core/Helpers/helpers.py:48-61`. All 18 callers are in deleted code. `os` and `sys` go if nothing else uses them. |
| the `_emit` import | `orchestrator.py:35` (unused) |
| `CHI_OVERRIDE_ENV` | §2.9 |
| `run_fdt`'s prompts | `core/FDT/fdt_pipeline.py:58-59,77-78,86-88`. The signature becomes `run_fdt(cfg, *, skip_sanity: bool, confirm_production: bool)`, with no defaults. The GUI already passes both (`test_nav_and_gating.py:97-104`). |

About 350 prompt-free lines survive in `core/cli.py`:
- `UnitParseError`, `resolve_units_file`, `validate_gt_file`, `load_and_validate_gt`, `INFERENCE_PROMPT_UNITS`;
- `_merge_vals_bounds`, `MASTER_BOUNDS_NAME`, `resolve_bounds_for_cell`, `parse_cell`, `units_to_factors`;
- `make_sim_config`, `make_fdt_config`, `make_reduction_config`, `SWEEP_PRESETS`, `make_param_sweep_config`.

### 2.13 Two refusal messages reworded (D8)

- `store.py:604-610` (loading a non-amortized posterior): the text "Pass Accept(truncated=True) to load it anyway (the Posterior tab does)" becomes "To load it anyway: confirm the load on the Posterior tab, pass --accept-truncated on the command line, or pass Accept(truncated=True)". The rest is unchanged.
- `orchestrator.py:1624-1626` (inference on another observation): the text becomes "Use the recorded observation or an amortized posterior. To run anyway, tick 'Run on a different observation' on the Infer tab, pass --accept-other-observation, or pass Accept(other_observation=True); the inference will record it."

Both are pinned (§8.2 T8, §8.4).

### 2.14 What piece 3 inherits

- The compositions and stages keep no reference to `cfg`, so copy-on-run can pass `copy.deepcopy(session.cfg)`.
- They mutate `cfg` in these ways:
  - T_obs;
  - truth injection;
  - `sources.pop`;
  - `observation.install`, which now also clears the truth for an experimental observation.

  The last two are new relative to today. An amortized training in a session whose cfg carries a truth still anchors its Fisher on that truth (`decorrelate.py:291-292`). That is left to copy-on-run.
- `_preflight_warn` is the seam for logging.
- Each composition's pre-spend block is where field validation goes.

## 3. The command-line tool (`python -m core`)

### 3.1 Layout and entry

```
core/__main__.py          entry (below)
core/tool/__init__.py     main(argv=None) -> int, build_parser(), UsageError
core/tool/config_args.py  shared flags, build_cfg(), model_for(), describe(), parse_forced(),
                          recording_set(), close_sink()
core/tool/stages.py       prior, train, tsnpe, validate, infer
core/tool/diagnostics.py  sbc, identifiability, ablation
core/tool/smoke.py        smoke (run_smoke)
core/tool/fdt.py          fdt, crossval
```

- Each module exposes `register(subparsers)`, and each handler has the shape `fn(args, store) -> int | None`.
- Every heavy import (orchestrator, torch, diagnostics) is inside a handler, so `--help` and the parser tests never import torch.
- `build_parser()` attaches `parser.subcommands = {name: subparser}` for the tests, and sets a top-level `epilog` (with `RawDescriptionHelpFormatter`) naming the only environment PRISM reads: `PRISM_RESOURCES` and `PRISM_ARTIFACTS` (the roots, §3.2) and the two core-level settings `PRISM_VRAM_CEILING_GIB` (`pipeline.py:470`) and `PRISM_MEM_LOG_EVERY` (`pipeline.py:155`), which core reads live and the tool never does (D13).
- `core/__init__.py` is empty.
- Nothing in `core/tool` imports `core.gui`.

```python
"""``python -m core <subcommand>``: the PRISM command-line tool (core/tool/). The GUI is ``python -m core.gui``."""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # before torch: OMP Error #15 otherwise
import matplotlib
matplotlib.use("Agg")                                   # before any core import
if __name__ == "__main__":
    from core.tool import main
    raise SystemExit(main())
```

### 3.2 `main(argv) -> int`

1. **Parse** with `prog="python -m core"`. argparse's `SystemExit` is caught and its code returned: 0 for `--help`, 2 for bad usage. `main` never calls `sys.exit`.
2. Call `registry.load_user_models()`, which is idempotent.
3. **Store.**
   - The root is `args.store_root` for `smoke` and `config.artifacts_root()` for every other subcommand; `PRISM_ARTIFACTS` is read at call time.
   - Create it with `mkdir(parents=True, exist_ok=True)`, so a root that cannot be created fails at start-up.
   - Run the handler under `with use_store(ArtifactStore(root)) as store:` (`store.py:765-773`), and ALSO pass `store=store` to every stage call.
   - `set_default_store` is never called; that call was `smoke_train.py:219-222`'s leak.
   - There is no global `--artifacts` flag. The `fdt` and `crossval` outputs follow the environment, not the store (`fdt_pipeline.py:28-33`, `cross_validation.py:40-41`), so a flag for the store alone would split one run across two roots.
4. **Dispatch.** A handler that returns `None` means exit code 0.

Exit codes (every message goes to stderr with the prefix `prism <subcommand>:`):

| code | when | output |
|---|---|---|
| 0 | success or `--help` | the artifacts written (`kind name__id`, path) on stdout |
| 2 | an argparse error, or a `UsageError` raised after parsing (a flag combination that depends on the built config, §3.6, or an unknown `--stages` entry) | the usage line, or `usage: <message>` |
| 1 | a refusal: `ValueError`, which includes `StoreError` (`store.py:37`), `cli.UnitParseError` (`cli.py:22`), `FDTModelError` (`campaigns.py:26`) and every stage refusal; or `FileNotFoundError` | `refused: <ExceptionClass>: <message> [raised at <file>:<line>]`, where the location is the innermost frame of the traceback. There is no full traceback, but a genuine bug that raises `ValueError` still says where it came from. |
| 1 | any other `Exception` | the full traceback, then `*** FAILED ***` (`smoke` names the stage) |
| 130 | `KeyboardInterrupt` | §3.9 |

### 3.3 Shared flags and the config builder

`add_config_flags(p)` adds these flags to every SBI subcommand (`prior`, `train`, `tsnpe`, `validate`, `infer`, `sbc`, every `identifiability` mode, `ablation`, `smoke`):

| flag | dest | default / rule |
|---|---|---|
| `--bounds PATH` | bounds | **required** on every SBI subcommand. It is one rule, and it removes the trap in which an unset bounds file picks the 12-dimensional spontaneous box (`smoke_train.py:46-59`). |
| `--model NAME` | model | `--model`, else the name of the `--bounds` parent folder upper-cased (the `Bounds/<model>/` layout; the same rule as `_common.model_for_cell`, `_common.py:94-106`) |
| `--chi` / `--no-chi` | chi_mode | None, which means `config.CHI_MODE` (`BooleanOptionalAction`) |
| `--chi-k K` | chi_n_freqs | None |
| `--device {auto,cpu}` | device | `auto` means `hw=None` (detect); `cpu` means `config.cpu_device()` (`config.py:88`) |

Which subcommands take `--cell` is listed in §3.5. The bounds file and the mode flags must describe the posterior a command loads. The store refuses a mismatch in model, order, box or mode loudly (`store.py:521-633`), and `--help` says so.

`build_cfg(args, *, load_gt: bool) -> tuple[SimConfig, list]` is `_common.script_cfg` (`_common.py:128-173`) with every environment read removed:
1. `model = model_for(args)`. An SBI command refuses (exit 1) a model that is neither built in nor `registry.is_sbi_user_model(model)` (`core/registry.py:63-74`).
2. Labels, as at `_common.py:152-153`.
3. `cfg = cli.make_sim_config(model, labels, registry.state_dep_drift(model), bounds, chi_mode=, chi_n_freqs=, hw=hw)`. The None fallbacks stay in `make_sim_config`.
4. `ignored = cli.load_and_validate_gt(cfg, args.cell) if load_gt else []`. Only `identifiability laplace|jacobian` pass `load_gt=True`. `infer` and `smoke` leave the truth to `simulated_inference`, so the note about ignored values prints once.
5. **The builder never sets `cfg.T_obs`**, unlike `_common.py:168-169`. `--t-obs` travels only to the functions that use it.
6. `describe(cfg, cell=, bounds=, ignored=)`, moved from `_common.py:176-203` and always printed. Its `T_obs` field prints `T_obs=(unset)` when `cfg.T_obs is None`; the verbatim move would raise a TypeError (`_common.py:186`). The banner shows the ND and rescale order, which makes a box mistake visible.

A single figure sink is passed to every call: `close_sink(title, fig): plt.close(fig)`. Every stage, composition and diagnostic accepts `fig_sink`, so the tool never has to know which of them draws. On write paths `ArtifactWriter.fig_sink` saves the figure and then forwards to this sink (`store.py:206-216`).

### 3.4 Rules shared by every subcommand

- **Refs.** `--prior`, `--posterior` and `--observation` take a name or an id (`store._find`, `store.py:313-317`; a name shaped like an id is refused when naming, `:361-367`).
- **The prior is always the posterior's own.** Wherever a posterior is loaded, the prior is `store.get("posterior", ref).parents["prior"]`. It is always recorded (`:1063`), and a TSNPE child records its base prior. The tool prints `[prism] prior: <ref> (the posterior's training prior)`. `--prior` exists only on `train` (required) and `smoke` (the drill's run 2). An explicit prior elsewhere could only ever be refused (`:673-682`; `test_artifact_store.py:830-849`).
- **Loading.**
  - A prior is loaded with `build_prior(cfg, ref, False, fig_sink=close_sink, store=store)`.
  - A posterior is loaded with `build_posterior(cfg, prior, ref, False, accept=accept, fig_sink=close_sink, store=store)`, which is the Posterior tab's call.
  - An observation is loaded with `store.load_observation(cfg, ref)`.
  - `store.load_posterior` is never called directly.
- **Accept (D8).**
  - `--accept-truncated` is on every subcommand that loads a posterior.
  - `--accept-other-observation` is on `infer` only.
  - The object is `Accept(truncated=args.accept_truncated, other_observation=getattr(args, "accept_other_observation", False))`. It goes to the posterior load and, for `infer`, to the composition.
- **Names.** `--name` and `--note` default to `""`. A taken name exits 1 before anything is spent, and nothing is written.
- **Knobs.** Every knob flag's `dest` equals the stage's keyword, set explicitly wherever the flag spelling differs (`--run-size` → `run_size_cap`, `--max-epochs` → `max_num_epochs`, `--posterior-samples` → `num_posterior_samples`, `--t-obs` → `t_obs_s`). The two crossval grid flags are the exception: `--s-grid` and `--t-grid` keep their own dest and are passed as `s_spec=` and `t_spec=`. Value flags default to None and are passed only when set. Booleans are always passed. The tool restates no default from `config.py`; `smoke` is the exception (§3.7).
- **Resume.** `--resume {auto,require,never}` (passed only when given) and `--new-run` are on `train`, `tsnpe` and `smoke`.
- **No environment reads and no module writes.** `core/tool` contains no `os.environ` or `os.getenv`, no assignment to an upper-case module attribute, and no assignment to `.hw.batch_size` (§8.3).

### 3.5 The subcommands

Every SBI subcommand also takes the config flags of §3.3.

| subcommand | flags (dest = stage kwarg) | call |
|---|---|---|
| `prior` | `--name --note --num-iterations --sweep-batch --max-sets --walk-step --stability-units --min-cluster-size --min-samples` | `build_prior(cfg, None, True, …)` |
| `train` | `--prior REF` (required); `--name --note --num-runs --run-size`(→`run_size_cap`) `--hidden-features --num-transforms --learning-rate --stop-after-epochs --max-epochs`(→`max_num_epochs`) `--fisher-m --fisher-dz --fisher-points --checkpoint-every --resume --new-run` | loads the prior (§3.4); `build_posterior(cfg, prior, None, True, …)` |
| `tsnpe` | `--posterior REF --observation REF` (required); `--directions`(→`n_directions`) `--level --num-runs --run-size --hidden-features --num-transforms --learning-rate --stop-after-epochs --max-epochs --checkpoint-every --resume --new-run --accept-truncated --name --note`. No Fisher flags. | `tsnpe_round(cfg, posterior, prior, store.load_observation(cfg, ref), …)` |
| `validate` | `--posterior` (required); `--n-cal --cal-n-scales --posterior-samples`(→`num_posterior_samples`) `--accept-truncated --name --note` | `validate_calibration(cfg, posterior, prior, …)` |
| `infer` | `--posterior` (required); a required, mutually exclusive choice of `--cell PATH` (simulated) or `--spont PATH` (experimental); `--t-obs S` (required, →`T_obs_s`); for experimental only: `--forced PATH[@HZ]` (append), `--drive NAME=VALUE` (append, SI units), `--f0-si N`; `--n-samples --accept-truncated --accept-other-observation --name --note` | simulated: `simulated_inference(cfg, posterior, args.t_obs_s, cell=args.cell, prior=prior, accept=, …)`; experimental: `experimental_inference(cfg, posterior, recording_set(cfg, args), accept=, …)` |
| `sbc` | `--posterior` (required); `--repeats --n-cal --posterior-samples --cal-n-scales --chi-k-fixed --seed --accept-truncated --name --note` | `core.diagnostics.sbc_repeats(...)` (§4.4) |
| `identifiability rotation` | `--posterior` (required); `--n-worst --top-n --accept-truncated --name --note` | `identifiability_rotation(...)` |
| `identifiability laplace` | `--posterior --cell --t-obs` (required); `--n-points --m --m-noise --rel --min-valid --sd-identified --seed --accept-truncated --name --note` | `identifiability_laplace(...)`; the cfg is built with `load_gt=True` |
| `identifiability jacobian` | `--cell --t-obs` (required; no posterior); `--m --m-noise --rel --min-valid --zero-tol --noise-eps --seed --name --note` | `identifiability_jacobian(...)`; `load_gt=True` |
| `ablation` | `--posterior` (required); `--rows --n-sweep --accept-truncated --name --note` | `core.diagnostics.channel_ablation(...)` |
| `smoke` | §3.7 (`--cell` required) | `run_smoke` |
| `fdt` | `--cell` (required); `--model` (default: the cell's parent folder, gated by `registry.fdt_support`); `--n-freqs --ensemble-m`(dest `ensemble_M`) `--freqs-per-batch --f0`(dest `F0`), passed when set (`make_fdt_config` defaults, `cli.py:571-573`); `--skip-sanity` (default False); `--no-production` (sets `confirm_production=False`). No `--device`: `make_fdt_config` forces the CPU (`cli.py:589`). No SBI config flags. | `run_fdt(cli.make_fdt_config(model, registry.state_dep_drift(model), cell, **knobs), skip_sanity=args.skip_sanity, confirm_production=not args.no_production)` |
| `crossval` | `--cell` (required); `--preset {exploratory,production}` (default exploratory; `cli.SWEEP_PRESETS`, `cli.py:660-665`); `--s-grid MIN MAX N` and `--t-grid MIN MAX N` (both required, `nargs=3`); `--n-freqs --ensemble-m`(dest `ensemble_M`), defaulting to the preset's values; `--freqs-per-batch --f0`(dest `F0`) | `cfg, s, t = cli.make_param_sweep_config(cell, preset=dict(SWEEP_PRESETS[p]), s_spec=, t_spec=, n_freqs=, ensemble_M=, freqs_per_batch=, F0=)` (`cli.py:728-756`), then `run_param_study_cli(cfg, s, t)` (`cross_validation.py:299-332`); prints the two `.h5` paths |

- `identifiability`'s mode is a **nested positional subparser**, so a flag that means nothing to a mode is an argparse error (exit 2).
- `--t-obs` has dest `t_obs_s` everywhere.
- There is no Reduction subcommand.

### 3.6 `infer --spont`: which recordings each observation mode takes

`recording_set(cfg, args) -> RecordingSet` is a pure function of `cfg.observation_mode` and `cfg.force_params_dict`. `parse_forced("x.npy@12.5")` splits at the LAST `@`; with no `@` the frequency is None. A frequency that is not a float is a `UsageError`.

| mode | rules | result |
|---|---|---|
| chi | `--spont`; at least one `--forced`, **each with `@HZ`** (a missing one is a `UsageError` citing D9); `--f0-si` required; `--drive` forbidden | `RecordingSet(spont, forced=tuple((p, hz)…), T_obs_s=, F0_si=)`, the form the GUI uses (`infer_tab.py:356-358`) |
| spontaneous | `--spont` only | `RecordingSet(spont, T_obs_s=)` |
| forced | exactly one `--forced`, without `@`; `--drive` for exactly the keys of `cfg.force_params_dict` (a missing or extra key is a `UsageError` naming both sets); `--f0-si` forbidden | `RecordingSet(spont, forced=((p, None),), T_obs_s=, forcing_params_si={…})` (`infer_tab.py:369-371`) |

### 3.7 `smoke`

`core/tool/smoke.py::run_smoke(args, store)` is a composition of its own, not a chain of `main(argv)` calls: run 2 loads a prior into the same cfg, `--stages` can stop early, and one process holds one store. It calls `build_prior`, `build_posterior`, `validate_calibration` and `simulated_inference` with keyword arguments and no module rebinds, which replaces `smoke_train.py:192-194,227,234,272`. The whole sequence runs inside `seeded(args.seed, cfg.hw.device)` (§2.8).

| flag | replaces | default |
|---|---|---|
| `--bounds`, `--cell` (both required), `--model`, `--chi`/`--no-chi`, `--chi-k`, `--device` | `BOUNDS CELL MODEL CHI*` | no default cell |
| `--seed N` | `SEED` | 0 (`smoke_train.py:153`) |
| `--t-obs S` | `TOBS_S` | None, meaning `config.T_MIN_EXP_S` |
| `--stages LIST` | `STAGES` | `prior,posterior,validate,infer`; an unknown entry exits 2 |
| `--num-runs N` | `NUM_RUNS` | 4, passed as `num_runs` |
| `--run-size N` | `RUN_SIZE` | 32, passed as `run_size_cap`. The identity is the same as smoke_train's, because `min(hw, cap)` is 32 both on the card and on the CPU. The prior sweep keeps the hardware batch (`:238-243`). |
| `--n-cal N` | `N_CAL` | 40, passed as `n_cal` |
| `--max-epochs N` | `EPOCHS` | 5, passed as `max_num_epochs` |
| `--checkpoint` | `CHECKPOINT=1` | off. On means `checkpoint_every = max(1, num_runs // 2)`; off means 0 (`:227,234`). |
| `--save` | `SAVE=1` | off. Names the prior `smoke_prior` (only when this run built it) and the posterior `smoke_posterior`. |
| `--store-root DIR` | `CKPT_DIR` | a fresh `tempfile.mkdtemp(prefix="prism_smoke_")`, printed and left on disk |
| `--prior REF` | `PRIOR` | None, meaning build |
| `--resume`, `--new-run` | new (D7) | `auto`; off. `--resume require` or `--resume never` without `--checkpoint` is refused by the stage (exit 1, §2.6). |

**Banner:**
- `describe(cfg)`;
- `[smoke] stages= num_runs= run_size= n_cal= epochs= seed= save= checkpoint= prior= store=`;
- the chi ceiling and conditioning-width lines (`:178-183`);
- one store-root warning, "--store-root given without --checkpoint: nothing will be resumable";
- the note "--run-size caps the training batch; the prior sweep keeps the hardware batch".

smoke_train's second warning (a reused store without a prior, `:228-237`) is dropped. Under D7 a rebuilt prior against a reused cache is a near miss in `prior_fingerprint` (`identity.py:44`), and it now refuses.

**Stages.** Each stage is wrapped as `[skip]`, then `=== name ===`, then `[ok] name in Xs`. On an exception the tool prints `[smoke] *** FAILED in stage <name> ***` and re-raises.
- **prior**, then **posterior**: `build_posterior(..., num_runs=, run_size_cap=, max_num_epochs=, checkpoint_every=, new_run=args.new_run, **resume_if_given)`. Early returns follow `:267-279`.
- **validate**.
- **infer**: `obs, inf = simulated_inference(cfg, post, t_obs_s, cell=args.cell, prior=prior, …)`. It keeps smoke_train's two checks (`:285-287`) as `RuntimeError`s:
  - the width equals `SUMMARY_WIDTH + 1 + expected_forcing_dim(cfg)`;
  - `torch.isfinite(obs.x_obs).all()`.

  This stage now also runs the out-of-distribution and T_obs warnings, which the old docstring told the operator to watch for (`:39-40`).
- The completion lines follow `:292-295`.

smoke_train's guidance on what to watch (`:15-63`) moves into the module docstring and the `--help` epilog:
- the ~37 % masked-probe reference, with its ±12 pp band;
- the mode banner;
- the pairing of bounds and rescale;
- "SBC at these sizes has no power".

### 3.8 The GPU gate as command lines

These replace the recipe in `docs/STATE.md` Owed item 3. Run them from the repository root.

```powershell
$py = "C:\Users\J\anaconda3\envs\biophys-env\python.exe"
$B  = "--bounds","Resources/Bounds/nadrowski/master.txt"
# run 1: chi; builds and names smoke_prior/smoke_posterior; writes the simulation cache
& $py -m core smoke --chi --t-obs 4.5 @B --cell Resources/Cells/nadrowski/master_spont.txt --checkpoint --save --store-root <scratch>/smoke
# run 2: the resume drill -- same store, same --num-runs; a non-resume is a refusal (exit 1)
& $py -m core smoke --chi --t-obs 4.5 @B --cell Resources/Cells/nadrowski/master_spont.txt --checkpoint --store-root <scratch>/smoke --prior smoke_prior --stages prior,posterior --resume require
# run 2b: the 2026-09-11 incident, now loud -- must exit 1 naming n_runs 4 vs 2, no [fisher] line, no new simulations/ dir
& $py -m core smoke --chi --t-obs 4.5 @B --cell Resources/Cells/nadrowski/master_spont.txt --checkpoint --store-root <scratch>/smoke --prior smoke_prior --stages prior,posterior --num-runs 2
# run 3: forced mode, its own store
& $py -m core smoke --no-chi --t-obs 4.5 @B --cell Resources/Cells/nadrowski/master_weak.txt --checkpoint --save --store-root <scratch>/smoke_chi0
```

Pass criteria:
- Runs 1 and 3 each finish within a few seconds per stage of the `bb38f1a` figures.
- Run 2 prints "Reusing the Fisher rotation stored with the training checkpoint" and `[checkpoint] resuming at batch 4/4`, and exits 0 in seconds.
- Run 2b exits 1 as annotated.
- No OOM lines appear, and the masked-probe counts sit within ±12 pp of 37 %.

**What is comparable with `bb38f1a` and what is not.** smoke_train loaded the cell's truth into the cfg, so its Fisher was anchored at master_spont's ground truth (`decorrelate.py:291-292`). The new `smoke` builds a truth-free cfg, so its Fisher anchors at the prior median, which is the behaviour training is meant to have. From T19 onward only timings and the masked-probe count are compared with `bb38f1a`; V and `fisher_spread` are not. The gate table in STATE.md says so.

`PRISM_ARTIFACTS` is not needed for the four smoke runs, which touch only `--store-root`; the diagnostic card runs of §9.3 need it pointed at the same directory. Delete `<scratch>` afterwards.

### 3.9 Ctrl-C and progress

**Ctrl-C.**
- A `KeyboardInterrupt` propagates.
- The pipeline's `except BaseException` (`pipeline.py:1758-1775`) commits the completed batches and announces it with a `[checkpoint]` line.
- `ArtifactWriter.__exit__` removes the directory being written (`store.py:223-236`).
- `main` returns 130 with the message "interrupted: the artifact being written was removed. If a [checkpoint] line above says batches were saved, the same command with --resume require continues them."
- There is no signal handler.

**Progress.**
- The tool never assigns `config.QUIET_SEGMENT_BAR`; it stays False (`config.py:361`), so the per-segment and solver bars render to stderr. The pin `tests/test_vt_progress.py:433-445` stays true.
- sbi's epoch table stays on (`config.py:352`).
- There is no `--quiet`.

## 4. Diagnostics and the `diagnostic` kind

### 4.1 The kind (`core/artifacts/`)

`manifest.py`:
- `KINDS` (`:17`) gains `"diagnostic"`.
- `BODY_KEYS["diagnostic"] = ("diagnostic", "variant", "settings", "results")`.
- `validate` requires those keys only. There is no closed registry of diagnostic names.
- The body key is `variant`, not `mode`, because `ArtifactStore.list` fills `Summary.mode` from `body.get("mode")` (`store.py:307`), and there `mode` means the observation mode.

`store.py`:
- `KIND_DIRS` (`:28-29`) gains `"diagnostic": "diagnostics"`, and `_PARENT_KEYS` (`:32-34`) gains `"diagnostic": ()`.
- New `LoadedDiagnostic(Loaded)` with fields `diagnostic: str`, `variant: str | None`, `settings: dict` and `results: dict`, exported from `core/artifacts/__init__.py`.
- New `load_diagnostic(self, ref) -> LoadedDiagnostic`, following the `load_calibration` pattern (`:688-698`): no cfg, no verification, no refusal (D5). A missing ref raises `StoreError`.
- `dependents` already walks every `KIND_DIRS` entry (`:418-423`). So deleting a parent that a diagnostic names is refused without `force=True` (`:434-444`), the same policy as for calibrations and inferences.

**The writer shape.** Every diagnostic function follows these steps:
1. `store = resolve_store(store)`.
2. `store.assert_name_free("diagnostic", name)`.
3. Every refusal.
4. `with store.create("diagnostic", cfg, name=, note=) as w:`. The computation happens inside; figures go through `w.fig_sink(fig_sink)` and payloads through `w.payload`.
5. Set `w.parents`, `w.fingerprints`, `w.config.update(settings)` and `w.body = {diagnostic, variant, settings, results}`.
6. `return store.load_diagnostic(w.id)`.

Rules for the payload and results:
- Every float goes through `orchestrator._num` (`:1571-1577`).
- Per-parameter records are lists of `{"name": …}` entries.
- Arrays go into `.npz` payloads via `file_manager.atomic_savez`.
- `results` is a short summary, and piece 4 can open the payload for the rest.
- Every function that takes a posterior records `results["accepted"] = list(getattr(posterior, "accepted", []))`.

### 4.2 The `core/diagnostics/` package

It holds `__init__.py` (re-exporting the five functions), `feature_sets.py`, `rng.py`, `sbc.py`, `identifiability.py` and `ablation.py`. The function names differ from the module names, so no function shadows its submodule.

**`feature_sets.py`** receives these from `scripts/_common.py:273-354`:
- `GROUP_G_PREFIX`;
- `summary_keep_idx()`, keeping the `n_dropped == 11` assertion;
- `feature_labels(cfg)`, via `chi.CHI_FISHER_CHANNELS` (`core/SBI/chi.py:75`) and `chi.chi_labels` (`:88`);
- `n_features` and `describe_features`;
- `assert_not_chi`, `assert_nadrowski` and `assert_forced`.

**The guards raise `ValueError`, not `SystemExit`.** Their messages name subcommands, not environment variables. `assert_forced` suggests `config.CELL_PATH / "nadrowski" / "master_weak.txt"`.

**`rng.py`** provides `seeded(seed, device)`, a context manager used by `smoke` and by every diagnostic. It:
- runs inside `torch.random.fork_rng(devices=[device] if device.type == "cuda" else [])`;
- saves and restores `np.random.get_state()`;
- calls `torch.manual_seed(seed)` and `np.random.seed(seed % 2**32)`.

### 4.3 Factoring the calibration draw out of `validate_calibration` (no change in behaviour)

Three private helpers go just above `validate_calibration`, with their bodies moved verbatim:
- `_calibration_prior(cfg, posterior, prior) -> (val_latent_prior, T, truncation)`: `:1380,1388-1406`.
- `_draw_calibration_set(cfg, val_latent_prior, T, force_prior, *, n_cal, cal_n_scales, chi_k_fixed=None) -> (x_cal, theta_star)`: `:1408-1432`.
- `_sbc_reference_sample(cfg, val_latent_prior, T, truncation, inferred_prior, theta_star)`: `:1442-1459`.

`validate_calibration` calls them in today's order: prior, then the draw (with `chi_k_fixed=None`), then `run_sbc`, then the reference sample. The helpers call `analysis.gen_cal_data` and `run_sbc` as orchestrator module globals, so the patches in `test_user_sbi.py:2536-2597` still reach them. The comment at `:1424-1427` now names `sbc_repeats(chi_k_fixed=)`.

### 4.4 `sbc_repeats` (from `scripts/sbc_characterize.py`)

```python
def sbc_repeats(cfg, posterior: LoadedPosterior, prior: LoadedPrior, *, repeats: int = 10, n_cal: int = 2000,
                num_posterior_samples: int = 1000, cal_n_scales: int | None = None,
                chi_k_fixed: int | None = None, seed: int = 0,
                name="", note="", fig_sink=None, store=None) -> LoadedDiagnostic
```

The pickled `gen_dist` already IS the `TruncatedLatentPrior` (`PRISM_HANDOFF.md:3913-3919`). Drawing through §4.3's code adds four things the script lacked: `check_basis`, the t_scale-override mirror in the reference sample, the kept-fraction line, and the `LoadedPrior` path. `sbc.py` calls `orch.run_sbc`, `orch.check_sbc` and `orch.sbc_rank_plot` **through the module**.

**Refusals, before anything is spent:**
- the name is free;
- `repeats >= 1` and `n_cal >= 1`;
- `chi_k_fixed` is refused outside chi mode (`sbc_characterize.py:86-88`); its range is `gen_training_data`'s own check (`pipeline.py:1330-1335`);
- `run_guards._assert_prior_used_matches_posterior(posterior.posterior, prior.prior, "SBC")` (`run_guards.py:67-84`).

**Loop.** `_calibration_prior` runs **once**. Then, for each `r`, inside `seeded(seed + r, cfg.hw.device)`: draw the set, `run_sbc(..., show_progress_bar=False)`, take the reference sample, and run `check_sbc`. The per-parameter KS table is printed, and so is the kept fraction when the prior is truncated.

**Output.**
- One figure: "SBC ranks pooled over repeats (histogram)".
- The payload `sbc_repeats.npz` holds `ks`, `c2st_ranks` and `c2st_dap` (each K×P), plus `ranks` (N×P), `repeat`, `n_valid`, `labels` and `nps`.
- Settings: `repeats`, `n_cal`, `num_posterior_samples`, `cal_n_scales`, `chi_k_fixed`, `seed`.
- Results:
  - `per_param [{name, ks_p_median, ks_p_min, frac_ks_below_05, c2st_ranks_median}]`;
  - `n_valid`;
  - `stratum` (`"pooled"` or `"k<N>"`);
  - `kept_fraction`;
  - `accepted`.
- Parents are `{posterior, prior}`; fingerprints are `{gmm}`.
- The run is all-or-nothing, like every stage.

**Not carried over:** the t_offset deep-dive (`:207-253`; no current bounds file declares a `t_offset` rescale), the CSV, and the incremental saves.

### 4.5 `identifiability`: three functions in `core/diagnostics/identifiability.py`

| variant | posterior | cell (truth) | simulates | guards |
|---|---|---|---|---|
| `rotation` | yes | no | no | none |
| `laplace` | yes | yes | yes | `assert_not_chi`, `assert_forced` |
| `jacobian` | no | yes | yes | `assert_nadrowski`; `assert_forced` when not chi |

**`identifiability_rotation(cfg, posterior, *, n_worst=3, top_n=4, name="", note="", fig_sink=None, store=None)`**, from `posterior_identifiability.py`:
- It reads `posterior.manifest.body["transform"]` (`param_keys`, `V` stored in columns, `fisher_eigenvalues`; `orchestrator.py:1084-1091`) and `body["conditioning"]`.
- The sidecar reconciliation (`:69-100`) is replaced by the load itself, which refused any V inconsistent with the pickled prior (`store.py:597-598`).
- It refuses `V is None` (the rotation was off).
- `fisher_eigenvalues is None` is reported, not refused; every TSNPE round carries None, and the message says to run the diagnostic on the parent.
- The computation is `:103-182`, verbatim.
- Results: `P`, `orthogonality`, `eigenvalues`, `directions [{index, eigenvalue, loadings:[{name, loading}]}]`, `per_param [{name, bottom_share, top4_share, peak_dir, verdict}]`, `flat_axes`, `accepted`.
- Parents are `{posterior}`.

**`identifiability_laplace(cfg, posterior, *, n_points=6, m=32, m_noise=128, rel=0.02, min_valid=0.5, sd_identified=0.3, t_obs_s=None, seed=0, name="", note="", fig_sink=None, store=None)`**, from `identifiability_offgt.py`:
- The bijection is `posterior.posterior.T`.
- The latent training prior comes from `_training_latent_prior(posterior.latent)`, which walks `.prior` until it finds `gen_dist` and raises `ValueError` if there is none. For a TSNPE posterior that prior is the truncated one.
- Refusals: both guards; `cfg.ground_truth` is touched up front (`sim_config.py:193-199`); `n_points >= 1`.
- The script's `:74-189` is lifted into `_laplace_raw(cfg, nd, res, force, m, crn, n_obs)`, which tests patch, and `_analyze_point`. Both run inside `seeded`.
- Corrections:
  - Names come from `list(cfg.params_dict) + list(cfg.rescale_params)`, not the pre-rename `ND_LBL` (`:65-67`).
  - The unused `build_inferred_bijection` call (`:169`) is removed.
- **Units.** `sd` is in "prior-range units", and the unit is chosen by the script's name heuristic: log-range for a rescale name containing "scale", linear range otherwise (`:71`). It is not chosen by the posterior's box (`config.py:400`, `REPARAM_LOG_PARAMS = []`). The arithmetic is kept. Each `per_param` record carries `unit: "log-range" | "range"`, and `settings["log_range_params"]` lists the names the heuristic chose.
- `N_obs` is computed locally from `(t_obs_s or config.T_MIN_EXP_S)`. **The function never writes `cfg.T_obs`.**
- Output:
  - payload `laplace_sd.npz` (`SD`, `points`, `force`);
  - results `points [{tag, measurable}]`, `per_param [{name, unit, sd, median_sd, frac_identified}]`, `accepted`;
  - parents `{posterior}`.
- The "Track A/B" verdict (`:200-216`) is not carried over.

**`identifiability_jacobian(cfg, *, m=32, m_noise=128, rel=0.02, min_valid=0.5, zero_tol=0.05, noise_eps=1e-6, t_obs_s=None, seed=0, name="", note="", fig_sink=None, store=None)`**, from `degeneracy_map.py`. That script runs at module level, so its code is lifted into functions over a context `_JacCtx(cfg, n_obs, mults, forcing_gt, keep_idx, feat_labels, n_force_ch)`:
- `_jacobian_features` (`:131-222`, both branches; the CRN seeds under `fork_rng`; tests patch it);
- `_probe_budget` (`:267-311`; its cos²+sin² check raises `ValueError`);
- `_dead_channels` (`:347-357`);
- `_jacobian` (`:360-404`);
- `_summaries` (`:405-529`).

It is chi-aware through `feature_sets` and `chi.chi_multipliers_for(cfg)`. Its output:
- payload `degeneracy_map.npz`, holding the script's arrays (the singular values, the per-parameter tables, the pairs, the top features);
- two figures (`:496-516`);
- results limited to `observation_mode`, `T_obs_s`, `n_features`, `condition_number`, `unmeasurable`, `degenerate_pairs` and `dead_channels`;
- parents `{}`; the header hashes the bounds and cell files.

### 4.6 `channel_ablation` (from `scripts/channel_ablation.py`)

```python
def channel_ablation(cfg, posterior: LoadedPosterior, *, rows: int = 200_000, n_sweep: int = 33,
                     name="", note="", fig_sink=None, store=None) -> LoadedDiagnostic
```

**The rows come from the cache the posterior names.**
- If `parents.get("simulation") is None`, raise `ValueError`: the posterior was trained with checkpointing off, so its rows no longer exist.
- `sm = store.get("simulation", digest)` supplies `batches_done` and `identity.run_size` (`store.py:735-740`).
- **`training_checkpoint.load_rows` gains keyword-only `x_only=False` and `max_rows=None`** (`training_checkpoint.py:365-394`). With `x_only`, it skips loading the `th_` shards and returns `(x, None)`. With `max_rows`, it stops after the shard that reaches the count, truncates, and skips the final `got != batches_done` check. There remains one walk over the shard naming.
- The row width must equal the posterior's `conditioning.width`; otherwise `ValueError`. The pre-flag widening branch (`:107-113`) is dropped.

**The network.**
- `est = posterior.latent.posterior_estimator`, and the net is found with `isinstance(m, embedded_network.EmbeddedNet)`.
- **Summary width.** `n_sum = int(net.input_dim)` (`:117-118`). The labels are `FEATURE_LABELS + VALID_FLAG_LABELS + ["logT"]` when `n_sum == SUMMARY_WIDTH + 1`; any other summary width raises `ValueError`.
- **The sweep.** It runs over columns `[0, n_sum)` of full-width rows (`:162-170`). The forcing or chi block stays at the base row's values, so forced and chi posteriors, whose widths include `forcing_dim`, are measured.
- **A defect fixed on this path.** The sweep calls `est.embedding_net`, the whole conditioning path. For spontaneous and forced posteriors sbi puts a standardizer ahead of the net (`z_score_x="independent"`, `train.py:177-179`), so the bare-net sweep was feeding raw values to a net trained on z-scores. The two coincide under chi, where every historical measurement was taken.
- Rows move to the net's device and dtype under `no_grad`. The sweep is deterministic and takes no seed.

**Results** (no figures, no payload): `n_rows`, `base_row`, `live_probes`, `median_disp`, `channels [{label, max_disp, rel_median, p1, p99, verdict}]`, `counts`, `accepted`.

**Parents** are `{posterior, simulation}`, so deleting the cache is refused while this diagnostic names it.

## 5. GUI changes

What changes on screen:
- a dialog on the Posterior tab before it loads a stored non-amortized posterior;
- a checkbox on the Infer tab;
- a near-miss checkbox on the TSNPE tab.

Everything else is rewiring. Nothing new is persisted to QSettings.

### 5.1 The GUI names no store

No GUI dispatch passes `store=`. `build_app` installs the default once (`core/gui/app.py:48-51`), and nothing swaps it afterwards. The GUI's own reads use `default_store()` (`prior_tab.py:225`, `posterior_tab.py:209`). This settles the TSNPE half of artifact-store spec §11.10.

### 5.2 The runners are deleted; the tabs call the compositions

| site | after |
|---|---|
| `inference_tabs.py:21-22` | the re-export lines go; the comment at `:15-18` no longer mentions the runners |
| `infer_tab.py:19` | DELETED: `:19` is the `from .runners import …` line, and the replacement is not written there. Line 5 already reads `from core import cli, config, forcing` and gains `orchestrator` (imported nowhere in this file today, though the four dispatch rows below name it); `from core.artifacts import Accept` joins it. |
| `infer_tab.py:332-334` | `dispatch(orchestrator.simulated_inference, cfg, post, self.sim_tobs.value(), cell=cell, gt_values=gt_dicts, prior=self.session.inf_prior, accept=self._accept(), provide_fig_sink=True, on_result=self._on_observation)`; exactly one of `cell` and `gt_dicts` is set, per the branch at `:318-331` |
| `infer_tab.py:359,366,372` | `dispatch(orchestrator.experimental_inference, cfg, post, rec, accept=self._accept(), provide_fig_sink=True, on_result=self._on_observation)` |
| `tsnpe_tab.py:11`, `:86-107` | §5.4 |
| `simulate_runner.py:42` | its docstring names `orchestrator.simulated_inference` |

No composition keyword collides with a parameter of `dispatch` or `Worker` (`base_panel.py:172-173,205-206`; `worker.py:35`). The ignored-cell note and the out-of-distribution warning (`runners.py:16-23`) now come from the composition, together with the T_obs warning the GUI never had. The pick-time `cli.validate_gt_file` (`infer_tab.py:146`) stays.

### 5.3 Posterior tab (D7, D8)

`_build_posterior` (`posterior_tab.py:104-137`) keeps its guards at `:108-118`. Then:

```python
new_run, accept = False, Accept()
if is_new:
    go, new_run = self._confirm_fresh_run(cfg, n_runs, cap)
    if not go: return
else:
    accept = self._accept_for_load(entry)
    if accept is None: return
self.session.reset_downstream("posterior")      # only after both questions: Cancel leaves the session intact
self._screen.refresh_gates()
self.dispatch(orchestrator.build_posterior, cfg, self.session.inf_prior, entry, is_new,
              num_runs=n_runs, run_size_cap=cap, accept=accept, new_run=new_run, <the seven knobs as :130-136>,
              provide_fig_sink=True, on_result=self._on_posterior)
```

**`_confirm_fresh_run(self, cfg, n_runs, cap) -> tuple[bool, bool]`** replaces `:139-188`. It returns `(True, False)` in three cases:
- there is no prior;
- the detector raises (logged as a warning: "…continuing -- the training stage checks again before it simulates");
- `orchestrator.fresh_run_near_misses(cfg, self.session.inf_prior, num_runs=n_runs, run_size_cap=cap)` returns `[]`.

Otherwise it returns `(True, True)` if `self._ask_new_run(near, n_runs)` says yes (the existing QMessageBox, `:172-188`, moved verbatim into a method tests can patch), and `(False, False)` if not. These go away: the direct `peek`, `resolve_dir` and `near_miss_siblings` calls, the live `config.TRAINING_CHECKPOINT_EVERY` short-circuit (`:157`), and the `training_checkpoint` and `_hw_batch` imports (`:5,12`).

**`_accept_for_load(self, entry) -> Accept | None`:**
- It reads `m = default_store().get("posterior", entry)` (`store.py:319`); the picker's `userData` is the id (`artifact_picker.py:144`).
- It returns `Accept()` on an exception (the load then names the error) and for an amortized manifest.
- Otherwise it returns `Accept(truncated=True)` if `self._ask_load_non_amortized(m)` says yes, else `None`.

The dialog is a Warning box:
- title: "Load a NON-AMORTIZED posterior?";
- text: "'<name or id>' was trained by a TSNPE round.";
- informative text built from `m.body["truncation"]` (`level`, `dims`, `x_obs_digest`): the posterior is valid only near that observation; Validate restricts its prior to the region; Infer refuses another observation unless "Run on a different observation" is ticked; and a TSNPE round can only be drawn around this posterior's own observation, with no override (D12);
- buttons: "Load it", and "Cancel" as the default.

A round's own result installs with no dialog: `tsnpe_tab._on_round` (`:109-122`) sets the session directly, and `build_posterior`'s read-back uses `Accept(truncated=True)` internally (`orchestrator.py:1107`).

Strings updated:
- `:123-126`: the comment now describes `_accept_for_load`.
- `:196-197`: "…unless 'Run on a different observation' is ticked on the Infer tab".
- `:257-258`: "the stage defaults".

The GUI does not expose the resume policy.

### 5.4 TSNPE tab (`TSNPEPanel`, `tsnpe_tab.py:17`)

```python
def _round(self):
    s = self.session
    if s.posterior is None or s.inf_prior is None or not self.obs_picker.key():
        return
    try:
        obs = default_store().load_observation(s.cfg, self.obs_picker.key())   # re-hashed, mode/width checked
    except Exception as e:                                                      # noqa: BLE001
        self._on_error(f"Could not load observation '{self.obs_picker.key()}': {e}", ""); return
    n_runs, cap = self._budget_values()
    self.dispatch(orchestrator.tsnpe_round, s.cfg, s.posterior, s.inf_prior, obs,
                  n_directions=self.n_dirs.value(), level=self.hpd.value(),
                  num_runs=max(1, n_runs), run_size_cap=max(0, cap), new_run=self.new_run.isChecked(),
                  provide_fig_sink=True, on_result=self._on_round)
    self.new_run.setChecked(False)
```

- A load failure goes through `_on_error` (`base_panel.py:381-392`), not `_config_error`, whose text begins "The configuration could not be built" (`:377`).
- The checks at `:90-102` leave the tab (§2.5 step 2). The stage refuses before it simulates, so the error dialog arrives within seconds.
- **`self.new_run`** is an unpersisted `QCheckBox("Start a new simulation even if a cache one setting away exists")` in the budget group. It is unchecked by default and cleared after each dispatch. It is needed because D7 also fires for rounds: a 2-batch test round and a 5000-batch round on the same parent differ only in `n_runs`, and the region is drawn inside the worker, so an up-front dialog is impossible.

### 5.5 Infer tab (D8)

- `self.other_obs = QCheckBox("Run on a different observation")`, added with `with_badge(..., HELP["infer_other_obs"])` between the input stack and the Run button (between `infer_tab.py:105` and `:107`). The help text says three things:
  - a TSNPE posterior is valid only near its region's observation;
  - ticking the box runs it anyway and records `"accepted": ["other_observation"]`;
  - a re-simulated cell draws new noise, so simulated mode needs the box for any TSNPE posterior.
- The box is enabled only for a non-amortized session posterior. `refresh_local_gates` (`:385-392`) does this:
  ```python
  post = self.session.posterior
  truncated = getattr(getattr(post, "posterior", None), "truncation", None) is not None
  if post is not self._gated_posterior or not truncated:
      self.other_obs.setChecked(False)
  self._gated_posterior = post
  self.other_obs.setEnabled(truncated)
  ```
  `self._gated_posterior = None` is set in `__init__`. The `getattr` chain matters because the gate tests put `object()` on `session.posterior` (`test_nav_and_gating.py:242,267`).
- `_accept(self)` returns `Accept(other_observation=True) if self.other_obs.isEnabled() and self.other_obs.isChecked() else None`. It is passed at all four dispatch sites.
- The box is not persisted.
- D9 needs no GUI change: the rows already send `(path, Hz)` pairs (`rows.py:56-71`).

### 5.6 What the GUI passes of the new keywords

| kwarg | GUI passes |
|---|---|
| `new_run` on `build_posterior` | `_confirm_fresh_run`'s answer |
| `new_run` on `tsnpe_round` | the TSNPE checkbox |
| `accept` | Infer tab: `_accept()`; Posterior load: `_accept_for_load` |
| `resume`, `checkpoint_every`, `max_num_epochs`, `store`, `name`, `note`, device | nothing (today's behaviour) |

## 6. What is deleted or archived

| path | fate | task |
|---|---|---|
| `orchestrator.run`, `core/app.py`, the `core/__main__.py` menu, the prompt half of `core/cli.py`, `helpers.clear_screen`, `run_fdt`'s prompts, the `_emit` import | deleted (§2.12) | T10 |
| `CHI_OVERRIDE_ENV` and its branch | deleted (§2.9) | T3 |
| `core/gui/panels/inference/runners.py` | deleted (§5.2) | T6, T7 |
| the legacy branch of `build_experiment_obs_chi` | deleted (§2.10) | T9 |
| `scripts/smoke_train.py` | FOLDED into `smoke`; `git rm` | T19 |
| `scripts/sbc_characterize.py` | FOLDED into `sbc`; `git rm` | T16 |
| `scripts/posterior_identifiability.py`, `identifiability_offgt.py`, `degeneracy_map.py` | FOLDED into `identifiability`; `git rm` | T17 |
| `scripts/channel_ablation.py` | FOLDED into `ablation`; `git rm` | T18 |
| `scripts/_common.py` | DISSOLVED. The feature helpers go to core (T14); the config builder and `describe` go to the tool (T11). `require_mode`, `load_posterior` and `enable_warnings` are not carried anywhere. Then `git rm`. | T21 |
| `scripts/retrain_convergence.py`, `chi_mask_audit.py`, `chi_f0_sweep.py`, `feature_candidate_test.py`, `build_master_cells.py`, `generate_bundle_videos.py` | ARCHIVED: moved on disk into the gitignored `archive/scripts/` (`.gitignore:2`), then `git rm --cached`, **never `git mv`**. No name collides with the seven files already there. | T21 |
| `scripts/` | removed, including its ignored `__pycache__` | T21 |

A folded script is `git rm`'d and not also archived, for three reasons:
- git history is its archive;
- nine of the thirteen scripts already fail at import (`core/config.py:179-186`);
- an archived copy would drift from its replacement.

T21 in PowerShell, from the repository root:

```
foreach ($f in 'retrain_convergence','chi_mask_audit','chi_f0_sweep','feature_candidate_test','build_master_cells','generate_bundle_videos') {
  Move-Item "scripts\$f.py" "archive\scripts\$f.py"; git rm --cached -q "scripts/$f.py" }
git rm -q scripts/_common.py
Remove-Item -Recurse -Force scripts\__pycache__; Remove-Item scripts
```

Before committing, check all four:
- `git ls-files scripts` is empty;
- `git ls-files -i -c --exclude-standard` is empty;
- `git status` shows exactly seven staged deletions;
- `Test-Path scripts` is False.

Nothing in `core/` or `tests/` imports `scripts/`.

## 7. Carried items

| carried item (`docs/STATE.md:80-86`; artifact-store spec §11.4, §11.10) | resolution | task | pinned by |
|---|---|---|---|
| Extend the literal-path source scan to wherever tool code lives | The tool lives under `core/`, which is already scanned. T21 introduces `CODE_ROOTS = ("core",)` in `tests/_fixtures.py` and points the rotation-reader scan at it. T22 points the literal-path scan at it and adds a guard that every top-level directory holding `*.py` is in `CODE_ROOTS`. | T21, T22 | `test_no_literal_resource_paths_outside_config` (walks `CODE_ROOTS`), `test_the_source_scans_cover_every_code_directory` |
| Retire `_common.require_mode` and the unconditional `Accept(truncated=True)` in `_common.load_posterior` | Neither is carried: the file is removed, and the tool refuses by default, with `--accept-*` mapping 1:1 | T12, T21 | `test_validate_refuses_a_non_amortized_posterior_without_accept_truncated` |
| Thread the store into the TSNPE runner | `tsnpe_round` takes a `LoadedObservation` plus `store=`. The tab and the tool load the observation, and no stage loads by ref. | T5, T7, T12 | `test_tsnpe_round_uses_the_store_it_is_given`, `test_the_tsnpe_tab_dispatches_a_loaded_observation` |
| The ten stale comments left by the clean break | the table below | T23 | review |
| Make the resume drill loud | the D7 refusal plus `resume="require"` (§2.7); GPU run 2b (§3.8) | T1, T2 | the §8 D7 tests; the GPU gate |

**Stale comments and references** (T23 unless marked). The rules:
- A folded script is renamed to its subcommand.
- An archived script is cited BY COMMIT (`scripts/<file>.py at <T21-sha>^`).
- Approved specs, plans and `PRISM_HANDOFF.md` are not edited.

| location | fix |
|---|---|
| `core/gui/screens/inference_screen.py:121-124` | "depends on what the artifact store holds (the observation picker lists `observations/`), not on the session" |
| `tests/test_nav_and_gating.py:272-276` | "rather than through the store's `observations/` listing … `refresh()` repopulates it from the store … only while no observation had been recorded" |
| `core/SBI/pipeline.py:1411`, `core/SBI/training_checkpoint.py:163` | "a read-only artifact root (`Artifacts/`, or `PRISM_ARTIFACTS`)" |
| `core/SBI/statistics.py:505-506` | "training checkpoint `train_98aebd93ed17` (pre-piece-1 layout, deleted by the 2026-09-11 clean break; the figures stand as history)" |
| `tests/test_user_sbi.py:43-52` | the block goes in T1; its reasoning moves to the conftest fixture's docstring |
| `tests/test_user_sbi.py:2209` | "`scripts/migrate_checkpoint_flags.py` at `e37df41^`" (`docs/STATE.md:112`) |
| `scripts/smoke_train.py:6`, `scripts/generate_bundle_videos.py:82-84` | vanish with their files |
| `run.sh:4-6` | "the working directory does not matter to the code (`core/config.py` resolves both roots from its own location); the cd only keeps typed paths relative to the repo" |
| **D1 references (T10):** `README.md:153` | one line: "`python -m core --help` lists the command-line tool's subcommands" |
| `run.bat:8`, `run.sh:12-13` | "`python -m core <subcommand>` is the command-line tool" |
| `core/gui/__init__.py:4-5` | "the GUI and the command-line tool (`python -m core`, `core/tool/`) are two front ends over the same orchestrator stages and compositions" |
| `core/orchestrator.py:4` | "No `input()` anywhere; the front ends (`core/gui`, `core/tool`) call these stages" |
| `core/cli.py:1-6`, `:25-27` | the prompt-free config builders; the front ends catch `UnitParseError` |
| `core/config.py:357-358`, `core/gui/app.py:39` | "the command-line tool never touches it and keeps both bars" |
| `core/registry.py:6` | "from `core.gui.app.build_app` and `core.tool.main`" |
| `core/Reduction/__init__.py:11` | "the Reduction panel's entry point (no command-line subcommand)" |
| `core/gui/panels/fdt_panel.py:6-7,56`, `crossval_panel.py:108`, `artifact_picker.py:1-2`, `labeled_inputs.py:1`, `inference_screen.py:112`, `core/Helpers/visualizers.py:479` | each names a deleted prompt function; reword to the surviving one |
| `orchestrator.py:1597-1598` | the `fig_sink` doc claims "(CLI unchanged)" for a CLI D1 deletes: drop the clause; every front end passes a sink, and None is the bare-library fallback |
| `tests/test_vt_progress.py:436` | "the command-line tool (`python -m core <subcommand>`) has nothing else"; the assertions are unchanged |
| **Script citations:** `orchestrator.py:564-565,609-612` | "every caller passes it as an argument (the GUI, `python -m core train/smoke`)" |
| `orchestrator.py:735,741`, `config.py:332` | jacobian becomes `identifiability jacobian`; the batch-size note names only the three pipeline tests |
| `orchestrator.py:890` (cites a file that never existed) | "See `python -m core identifiability rotation`" |
| `orchestrator.py:1424-1427`, `config.py:270` | `python -m core sbc` (`--chi-k-fixed`) |
| `config.py:446,513,561`, `run_guards.py:134` | `chi_f0_sweep.py` by commit; there is no override any more (§2.9) |
| **T3, with the deletion:** `core/config.py:170`, `core/SBI/pipeline.py:513`, `tests/test_user_sbi.py:3747` | each explains `PRISM_VRAM_CEILING_GIB` by analogy to `PRISM_CHI_OVERRIDE`, which T3 deletes: drop the analogy and say "read live from the environment, like `PRISM_MEM_LOG_EVERY`" (D13 keeps both) |
| `config.py:468`, `decorrelate.py:112,224`, `chi.py:41,67,499`, `tests/test_chi_set_encoder.py:434,467` | `identifiability jacobian` |
| `analysis.py:246`, `truncate.py:100`, `tests/test_user_sbi.py:2128` | `identifiability rotation` |
| `decorrelate.py:24`, `tests/test_gpu_paths.py:4` | `python -m core smoke` |
| `train.py:3` | drop `retrain_convergence.py` |
| `chi.py:42` | `core/diagnostics/feature_sets.py` |
| `observations.py:6` | "for the GUI and the command-line tool" |
| `tests/test_user_sbi.py:2018`, `tests/test_artifact_consistency.py:206` | the archived script, by commit |
| `CLAUDE.md:42-43,82`, the suite count, the `docs/STATE.md` recipe | T19/T25 (§9.5) |

## 8. Tests

### 8.1 Tests that change

| # | test | change | task |
|---|---|---|---|
| 1 | `test_conditioning_repair.py:928-948` (an ast pin on `orchestrator.run`) | **DELETE.** Its premise is impossible since piece 1: the posterior either refuses to load or loads WITH its region (`store.py:600-632`), and `validate_calibration` reads the region (`:1380`). The behaviour is re-pinned by the tool's `validate` test and by row 4. | T10 |
| 2 | `test_nav_and_gating.py:285-299` | RETARGET to the docstring-stripped `ast.unparse(inspect.getsource(orchestrator.tsnpe_round))`: it contains `build_truncation_region` and `truncation=region`, and neither `set_default_x` nor `proposal`. PLUS: `TSNPEPanel._round`'s code contains `orchestrator.tsnpe_round` and none of `build_posterior`, `build_truncation_region`, `set_default_x`, `proposal`. Import `orchestrator` and `TSNPEPanel` instead of `inference_tabs`' runner (`:261`). | T7 |
| 3 | `test_nav_and_gating.py:301-347` | add `inf.posterior_panel._ask_load_non_amortized = lambda m: pytest.fail(...)` before `_on_round` | T8 |
| 4 | `test_nav_and_gating.py:350-387` | gains the `store` fixture; `:377-382` is rewritten on real artifacts (`_nad_cfg(chi_mode=True)`, `_posterior_artifact(store, cfg, amortized=False, region=…)`, plus an amortized one). (a) The ask returns True: `accept.truncated is True` and `args[3] is False`. (b) It returns False: nothing is dispatched and the session posterior is unchanged. (c) The amortized id, with an ask that fails if called: `accept.used() == []`. | T8 |
| 5 | `test_nav_and_gating.py:787-829` | call `orchestrator.simulated_inference(Cfg(), SimpleNamespace(posterior=SimpleNamespace(x_obs_digest=None, truncation=None)), 0.1, cell="cell.txt", fig_sink=…)`; the three assignment stubs stay; rename it `test_simulated_inference_emits_the_ground_truth_figure` | T6 |
| 6 | `test_settings_persistence.py:596-611` | keep `:600-605`; add `"new_run=" in` the `_build_posterior` code; the fail-open count becomes `body.count("return True, False") >= 3`, plus `"fresh_run_near_misses" in body` and none of `near_miss_siblings`, `resolve_dir`, `peek` | T2 |
| 7 | `test_user_sbi.py:43-52` (the import-time `orchestrator.TRAINING_CHECKPOINT_EVERY = 0`) | DELETE; replaced by a session-autouse `_checkpointing_off_unless_asked` in `tests/conftest.py` (`pytest.MonkeyPatch`, restored at teardown). Today the rebind applies only when that file is collected, so a single-file run differs from the gate, and under D7 a stray cache becomes a refusal in another test. **Every test that wants a cache passes `checkpoint_every` explicitly.** | T1 |
| 8 | `tests/_fixtures.py:171-205` (`build_tiny_run` rebinds three knobs) | keep only the `pipeline.gen_prior` stub; call `build_posterior(..., num_runs=2)`. The model-install half becomes `install_sbitest() -> (bounds, cell, teardown)`. **Every `tiny_run` consumer that trains or calibrates passes `num_runs`/`n_cal` explicitly** (the current ones do: `test_artifact_store.py:510-511,836,918,984`). The other `orchestrator.*` rebinds in the test files stay; they restore and are still honoured. | T1, T11 |
| 9 | `test_user_sbi.py:3316-3322` | `build_posterior` gains `checkpoint_every` and `max_num_epochs` | T1 |
| 10 | `test_user_sbi.py:443` (slow) | `(path, Hz)` pairs from `cfg.chi_obs_freqs` (§2.10) | T9 |
| 11 | `test_artifact_consistency.py:277-288` | with `PRISM_CHI_OVERRIDE=1` set in the environment (monkeypatched), the non-default cfg STILL raises, and the message no longer names the variable | T3 |
| 12 | `test_conditioning_repair.py:922-924` (`("core", "scripts")`) | walks `CODE_ROOTS` | T21 |
| 13 | `test_artifact_store.py:937-962` | walks `CODE_ROOTS`; the matcher is unchanged | T22 |
| 14 | `test_artifact_store.py:139-157`, `:60-73` | the `diagnostic` body in the round trip; a missing body key is refused | T13 |
| 15 | `test_user_sbi.py:2514-2659` (guardrail 8) | gains a `sbc_repeats` leg under the SAME stubs: `cap["prior"]` is the `TruncatedLatentPrior` with `.region is region`; every θ* and reference draw lies in the region; the t_scale column of the reference is a permutation of θ*'s; "PRIOR RESTRICTED" is printed | T16 |
| — | **unchanged, and they must stay green:** `test_user_sbi.py:2437-2441` (it now ALSO pins D7's truncation-alone exemption), `:2662-2681`, `test_conditioning_repair.py:786-802,890-895`, `test_artifact_store.py:161` (the `_Cancel` writer test, which covers the store half of §3.9), `:353-354,505-513,818,826,830-849,980-990`, `test_settings_persistence.py:169-201,239-261,558-594`, `test_nav_and_gating.py:97-112,389-435`, `test_user_models.py:838`, `test_vt_progress.py:433-445` | — | — |

### 8.2 New tests, by task (names are indicative)

- **T1:**
  - `resume` values are validated and refused before any spend: an unknown value, and `require` with checkpointing off.
  - `checkpoint_every` and `max_num_epochs` reach the plan; extend `test_artifact_store.py:560-570`, which captures `plan.checkpoint`.
  - Both are recorded in the manifest `config`.
  - `make_sim_config(hw=cpu_device())` is honoured.
  - The `[budget]` line prints with default arguments.
- **T2:** `test_a_near_miss_cache_is_refused_before_the_fisher(store, monkeypatch)`.
  - Setup: `_nad_cfg()` with rotation on and a prior artifact. A sibling under `store.kind_dir("simulation")` has header identity `SimulationIdentity.from_cfg(cfg, lp, 4, 3).to_dict()` and `batches_done=2`. `decorrelate.build_latent_fisher_rotation` is patched to raise `AssertionError("Fisher reached")`, and `pipeline.train_nn` is stubbed.
  - (a) `build_posterior(cfg, lp, None, True, num_runs=2, run_size_cap=4, checkpoint_every=1)` raises `ValueError` naming `n_runs`, 3, 2 and `new_run`.
  - (b) With `new_run=True`, it reaches the Fisher stub.
  - (c) `resume="require"` refuses before the Fisher.
  - (d) `checkpoint_every=0` does not refuse.
  - (e) `fresh_run_near_misses(..., run_size_cap=4096)` with `hw.batch_size=4` returns the same row.
- **T2:** `test_a_confirmed_near_miss_dispatches_new_run`, which monkeypatches the detector and `_ask_new_run`:
  - one row and "yes" dispatch with `new_run is True`;
  - "no" dispatches nothing;
  - `[]` dispatches with `new_run is False` and asks nothing;
  - a raising detector dispatches with `False` and logs a warning.
- **T4:**
  - `pytest.warns(PreflightWarning)` for the T_obs range, the ignored values and an out-of-distribution truth.
  - The region check: a truncated stub posterior plus `Accept(other_observation=True)` and a region that excludes the truth warns, and `generate_observations` is still reached.
  - `gt_values` pops `sources["cell"]`.
  - A non-amortized posterior without accept is refused before `generate_observations` is called.
  - A taken inference name is refused before the simulation.
  - `install` of an experimental observation leaves `cfg.has_ground_truth is False` and `"cell" not in cfg.sources`.
  - **Second-hop pin:** with recorders assigned to `orchestrator.generate_observations`, `build_experiment_observation` and `infer_and_visualize`, each composition forwards `accept`, `n_samples` (when set; absent when None), `name`, `note`, `fig_sink` and `store` unchanged.
- **T5:**
  - Zero directions, more than P directions, and an HPD outside (0,1) are refused before any spend.
  - HPD 0.95 warns.
  - A non-amortized parent with another observation is refused.
  - The round uses the store it is given.
  - `observation.install` runs before training.
  - An experimental observation on a cfg carrying `master_weak`'s truth prints no `[tsnpe] WARNING ... GROUND TRUTH` line, while a simulated observation whose truth lies outside a synthetic region still warns.
  - **Second-hop pin:** with recorders on `orchestrator.build_truncation_region` and `orchestrator.build_posterior`, every keyword reaches `build_posterior` unchanged, with no Fisher keyword and `truncation is region`.
- **T7:** `test_the_tsnpe_tab_dispatches_a_loaded_observation`, which stubs `default_store`:
  - `fn is orchestrator.tsnpe_round` and `args[3] is sentinel`;
  - a raising load dispatches nothing and calls `_on_error`;
  - ticked: `kwargs["new_run"] is True`, and the box is unticked after the call;
  - unticked: `kwargs["new_run"] is False`;
  - the `save_settings` text does not contain `new_run`.
- **T8:**
  - `test_the_infer_tab_other_observation_box`: (a) an amortized stub leaves the box disabled with `accept is None`; (b) a truncated stub with the box ticked gives `other_observation is True`; (c) a new posterior plus `refresh_gates()` leaves it unticked; (d) `save_settings` does not mention `other_obs`.
  - `infer_and_visualize` with `Accept(other_observation=True)` records `results.accepted == ["other_observation"]`.
  - Both reworded refusals name their GUI control and their flag (§2.13).
- **T9:** a chi set whose forced entry carries no frequency is refused before any load, and the refusal names that file; a chi set whose forced entries all carry one is NOT refused, though its passive recording has no frequency (the loop's first element, `orchestrator.py:302`).
- **T10:** `test_the_prompt_cli_is_retired`: there is no `ast.Call` of `input` under `CODE_ROOTS`, and `run_fdt`'s `skip_sanity` and `confirm_production` are KEYWORD_ONLY with no default.
- **T13:** `test_a_diagnostic_loads_without_a_config_and_blocks_deleting_what_it_names`. `load_diagnostic`'s signature is exactly `(self, ref)`; the parent's `dependents` include the diagnostic; the delete is refused, then succeeds with `force=True`.
- **T14:** `test_the_chi_feature_set_drops_group_g_and_adds_the_fisher_block`; `test_the_diagnostic_guards_refuse_with_value_errors`; `test_seeded_restores_the_callers_rng`.
- **T16:**
  - `test_sbc_writes_one_diagnostic_naming_its_posterior_and_prior(tiny_run)`: 2 repeats and `n_cal=8`; the npz shape; one figure; a second seed-0 run gives an equal `ks`.
  - `test_sbc_refuses_before_the_spend`: a recorder on `gen_cal_data` stays empty.
- **T17:**
  - rotation on a synthetic manifest: participation ratio, a flat axis, eigenvalues None reported, V None refused;
  - the tool test (§8.4) covers refusal on a real posterior without a rotation;
  - laplace and jacobian with `_laplace_raw` and `_jacobian_features` patched, on a forced Nadrowski cfg with `master_weak`;
  - the per-parameter `unit` is pinned;
  - the guards refuse before `create`;
  - `_probe_budget` refuses a miswired cos/sin pair.
- **T18:**
  - `test_load_rows_x_only_stops_at_max_rows` (the `th_` shards are never opened);
  - `test_ablation_reads_the_cache_its_posterior_names(tiny_run)`: it trains with `num_runs=2, run_size_cap=8, hidden_features=8, num_transforms=1, stop_after_epochs=1, checkpoint_every=1, new_run=True`, and it refuses a posterior without a cache;
  - `test_ablation_sweeps_the_summary_columns_of_a_forced_width_row`: an untrained `build_embedding_net` for a forced Nadrowski cfg, rows of width `SUMMARY_WIDTH+1+3`; the forcing columns are never swept.
- **T11-T20:** `tests/test_tool.py` (§8.4).
- **T22:** the code-roots guard (§8.3).

The tests for T13-T18 live in a new `tests/test_diagnostics.py`.

### 8.3 Source scans

- **`CODE_ROOTS = ("core",)`** in `tests/_fixtures.py`, used by the literal-path scan and by the rotation-reader scan.
- **`test_the_source_scans_cover_every_code_directory`:** the set of top-level repository directories that contain any `*.py` equals `set(CODE_ROOTS)`. Excluded: `tests`, `archive`, `.claude`, `.git`, `.pytest_cache`, `sbi-logs`, `Artifacts`, `Resources`, `__pycache__`.
- **`test_the_tool_reads_no_environment_and_writes_no_knob`** (in `test_tool.py`) checks three things:
  - `core/tool` contains no `os.environ` or `os.getenv`;
  - `core/tool` contains no assignment whose target attribute is upper-case or `batch_size`;
  - `core/__main__.py` holds exactly one `os.environ.setdefault("KMP_DUPLICATE_LIB_OK", …)` and one `matplotlib.use("Agg")`, both before its first `core` import.

### 8.4 `tests/test_tool.py` (new)

Everything runs in-process through `main(argv)`, because a subprocess could not catch an unrestored default store.
- One module-scoped `MonkeyPatch.context()` holds the `PRISM_ARTIFACTS` temp directory, the `orchestrator.pipeline.gen_prior = _fixtures._tiny_gen_prior` stub, and `install_sbitest()`'s teardown. All three are restored at module teardown.
- Every test asserts that `default_store()` after `main` is still the session sandbox object.

With `CFG = --bounds b --device cpu` (what every SBI subcommand takes; `LOAD` is the same list, named for the loading subcommands) and `SCFG = *CFG --cell c` (the subcommands that also declare `--cell`: `smoke`, `infer`, `identifiability laplace|jacobian`), the module fixture `tool_run` runs:
- `prior *CFG --name tp`
- `train *CFG --prior tp --name tpost --num-runs 2 --run-size 8 --hidden-features 8 --num-transforms 1 --stop-after-epochs 1 --checkpoint-every 1`

| test | asserts |
|---|---|
| `test_every_knob_flag_reaches_its_stage_as_a_keyword` | Each callee is replaced by a recorder: the stages, the three compositions, the diagnostics functions, `run_fdt`, `make_fdt_config`, `make_param_sweep_config` and `run_param_study_cli`. An explicit table maps each option dest to (callee, keyword) or to a marker (`positional`, `RecordingSet`, `accept`, `ref`, `smoke-only`). Examples: `t_obs_s` maps to `simulated_inference` positional 2; `ensemble_M` and `F0` map to `make_fdt_config` and `make_param_sweep_config`; `s_grid` maps to `s_spec`. Each subcommand runs with every non-shared option set to a distinctive value, and each value arrives where the table says. Value flags default to None (smoke's documented defaults excepted). `accept == Accept(...)` exactly as the flags say. |
| `test_the_tool_reads_no_environment_and_writes_no_knob` | §8.3 |
| `test_the_help_epilog_names_the_core_environment_settings` | `build_parser().epilog` contains `PRISM_RESOURCES`, `PRISM_ARTIFACTS`, `PRISM_VRAM_CEILING_GIB` and `PRISM_MEM_LOG_EVERY` (D13) |
| `test_usage_errors_exit_2` | an unknown subcommand; `--cell` together with `--spont`; `identifiability rotation --cell x`; `smoke --stages prior,bogus`; `prior` without `--bounds`; `validate` without `--bounds`; the `parse_forced` and `recording_set` unit cases, including chi `--forced x.npy` without `@HZ` |
| `test_a_taken_name_is_refused_with_exit_1_and_nothing_written` | `prior *CFG --name tp` again returns 1 and `already exists`; the count of priors is unchanged |
| `test_validate_and_simulated_infer` | `validate *LOAD --posterior tpost --n-cal 60` and `infer *LOAD --posterior tpost --cell c --t-obs 1.0` both return 0; the parents name tpost and its training prior |
| `test_validate_refuses_a_non_amortized_posterior_without_accept_truncated` | with a truncated artifact: exit 1, and stderr names `--accept-truncated` (§2.13) and a `raised at` location; with `--accept-truncated`, `validate_calibration` (a recorder) receives a posterior whose `.posterior.truncation` is set |
| `test_near_miss_is_refused_before_simulation` | `train *CFG --prior tp --num-runs 3 --run-size 8 --checkpoint-every 1` returns 1; stderr names `n_runs`, 2 and 3; no new `simulations/` or `posteriors/` entry appears |
| `test_ctrl_c_mid_simulation_keeps_the_committed_batches` | `pipeline._rows_with_oom_retry` (`pipeline.py:1634`) is wrapped to raise `KeyboardInterrupt` on its third call. `train *CFG --prior tp --num-runs 4 --run-size 8 --hidden-features 8 --num-transforms 1 --stop-after-epochs 1 --checkpoint-every 1 --new-run` returns 130; `posteriors/` is unchanged; the new cache is partial with `batches_done == 2`. Unwrapped, the same argv plus `--resume require` returns 0 and prints `resuming at batch 2/4`. |
| `test_smoke_runs_the_four_stages_and_the_resume_drill_is_loud` | `smoke *SCFG --store-root tmp --num-runs 2 --run-size 8 --n-cal 60 --max-epochs 2 --t-obs 1.0 --checkpoint --save` returns 0 and writes all six artifacts; the drill with `--resume require` resumes; `--num-runs 3` returns 1 naming `n_runs`; a patched, failing `validate_calibration` returns 1 with `FAILED in stage validate` |
| `test_tsnpe_subcommand_runs_a_round` | `infer` makes an observation; then `tsnpe *LOAD --posterior tpost --observation <id> --num-runs 1 --run-size 8 --checkpoint-every 1 --new-run` returns 0 and writes a non-amortized posterior whose parents name the observation |
| `test_identifiability_rotation_refuses_a_posterior_without_a_rotation` | `identifiability rotation *LOAD --posterior tpost` returns 1 and names the rotation. The SBITEST posterior has none, because a model without forcing disables the rotation (`orchestrator.py:727-728`). Nothing is written. The writing path is covered in `test_diagnostics.py`. |
| `test_fdt_and_crossval_run_at_tiny_size` | `fdt --cell <HOPF cell> --n-freqs 2 --ensemble-m 8 --skip-sanity` and `crossval --cell …/master_spont.txt --s-grid 0 0.1 2 --t-grid 1 1.1 2 --n-freqs 2 --ensemble-m 8` both return 0, and their outputs land under the temp root. If the pair measures above ~15 s it is marked `slow`, and the recorder test keeps the fast-gate coverage. |

### 8.5 Count and budget

- **Baseline:** 361 collected at `df7491e` (359 passed, 1 skipped, 1 deselected).
- **Change:** piece 2 adds about 42 ± 8 tests in two new suites (`test_tool.py`, `test_diagnostics.py`), which makes sixteen: `tests/` holds fourteen today, and `CLAUDE.md:79`'s "thirteen" is stale by one (`test_gpu_paths.py` landed at `bb38f1a`). It deletes one test.
- **Target:** the one-process fast gate stays at or under 14 minutes. `test_tool.py` should take at most ~90 s, plus fdt/crossval as measured, and `test_diagnostics.py` at most ~90 s.
- **Recording:** durations are measured with `--durations` at the first gate after each suite lands, and recorded in `docs/STATE.md`.

## 9. Gates and verification

### 9.1 After each task

- Run one `pytest -m "not slow" -q` in the background, logging to a scratch file.
- Touch no source until it exits, and never run two pytest processes at once.
- Commit the tested tree and check `git log -1 --stat`.
- Record in `docs/STATE.md`: the count; pass, skip and deselect; wall time; and confirmation that the real `Artifacts/` gained nothing.

### 9.2 The slow test

`pytest tests/test_user_sbi.py -m slow -q` (~31 min, in the background) runs once, at T25, before the piece is called done.

### 9.3 GPU

GPU checks run after every task that moves tensors:

| when | what |
|---|---|
| **Interim, after T2** | The still-live `scripts/smoke_train.py`, with `PRISM_ARTIFACTS` at scratch. Its module rebinds still work under the None default resolved at call time. Run 1 is STATE item 3's recipe. Run 2 uses the same `NUM_RUNS` and must resume. **Run 2b**, with `NUM_RUNS=2` against run 1's 4, must exit nonzero naming `n_runs` 4 vs 2, with no `[fisher]` line and no new `simulations/` directory. |
| **Interim, after T15** | The same script, runs 1 and 3. This covers T3-T8 and T15 — the calibration draw moved into helpers (T15), which is the code that failed on the card before. It does NOT cover T9: `smoke_train` simulates its observation (`smoke_train.py:284`) and never calls `build_experiment_observation`. |
| **Gate of record, at T19** | The four command lines of §3.8. Then, with `$env:PRISM_ARTIFACTS` set to `<scratch>/smoke` (and to `<scratch>/smoke_chi0` for the laplace run, because the diagnostics read `config.artifacts_root()` and no subcommand but `smoke` takes `--store-root`), one card run each, which covers the tensor-moving diagnostics of T16-T18: `sbc --repeats 1 --n-cal 20`, `identifiability jacobian` (chi) and `ablation` against run 1's `smoke_posterior`, and `identifiability laplace` against run 3's forced posterior. Then delete the scratch stores. |
| **At T25** | The gate is repeated only if a tensor-moving change landed after T19; T20-T24 move none. |

`tests/test_gpu_paths.py` stays green in every fast gate on this machine.

### 9.4 Display walkthrough

Append a section "Piece-2 GUI checks" to `docs/checklists/display-walkthrough.md`. The rows are added in the commits that create the behaviour, and the user runs them once at the end. The section's intro says two things:
- B1 supersedes A7's "Loading it needs no dialog".
- B3 supersedes A7's quoted refusal string: simulated inference now refuses up front, with §2.3's text.

Rows A1-A9 are not edited.

| # | surface | do | expect |
|---|---|---|---|
| B1 | Posterior: load a stored non-amortized posterior | Pick a round's posterior, then Train/Load. Cancel. Repeat and choose "Load it". Then load an amortized one. | A dialog titled "Load a NON-AMORTIZED posterior?" naming the level, directions and digest. Cancel: no "Posterior ready", and the other tabs keep their state. Load it: the NON-AMORTIZED lines. Amortized: no dialog. |
| B2 | TSNPE round result | Run a round at Batches = 2. | No load dialog; "TSNPE round complete…"; the Infer box becomes enabled and stays unticked. |
| B3 | Infer "Run on a different observation" | Use an amortized posterior. Then use the round's posterior on another cell, first unticked, then ticked. Then load another posterior. Then restart. | Amortized: the box is greyed. Unticked: the refusal dialog names the box, and nothing new appears under `inferences/` or `observations/`. Ticked: it runs, the log says "Running anyway (accepted).", `results.accepted` is `["other_observation"]`, and if the cell's truth lies outside the region a warning line says so. After the reload and the restart: unticked. |
| B4 | Near-miss dialog (D7) | With checkpointing on, train at Batches 2 to completion. Then train at Batches 3 with the same prior: Cancel, then "Start a new run anyway". Then train at Batches 2. | The "differs only in n_runs" dialog with both values. Cancel dispatches nothing. Confirming trains in a new digest directory. Batches 2: no dialog, and "Resumes a COMPLETE checkpoint". |
| B5 | TSNPE stage checks | Set Directions to 0 and Run. Set HPD to 0.95 and Run. Repeat a round at Batches 3, get refused, tick the new box, and run. | 0 directions: an error dialog, and nothing written. HPD 0.95: a warning line, and the round runs. Near miss: the refusal names `n_runs`; with the box ticked the round runs and the box unticks. |
| B6 | Simulated inference with T_obs outside `[T_MIN_EXP_S, T_MAX_EXP_S]` | Infer at T_obs 0.5 s. | A warning line in the log pane, new to the GUI, alongside the out-of-distribution warning as before. |
| B7 | A TSNPE round on a non-amortized parent (D12) | Load a round's posterior (accept the B1 dialog), pick a DIFFERENT observation on the TSNPE tab, Run. | An error dialog within seconds naming the parent's own observation digest; nothing written under `posteriors/` or `simulations/`; no `[tsnpe] region…` line. There is no checkbox for this. |

### 9.5 Documents at the end (T25)

- **`docs/STATE.md`:**
  - piece 2 DONE, with its commit range;
  - Owed item 5's piece-2 items struck through, with their commits;
  - new piece-3 items (§1.3) and the handoff lines for piece 6;
  - the clean-break "left alone" list marked fixed;
  - the gate table: the new count, the fast gate, the slow test, the GPU gate as command lines including run 2b, the note that the Fisher is now truth-free, and the walkthrough rows;
  - the decisions log: D1-D13, §1.2's readings, the rule for folding versus archiving, the conftest checkpoint default, and every §11 ruling. D11 (no chi override anywhere) and D12 (no escape hatch for a non-amortized parent on another observation) are recorded as standing refusals a later piece must not re-open.
- **`CLAUDE.md`:**
  - the tool is `python -m core <subcommand>` and sets `KMP_DUPLICATE_LIB_OK` and Agg itself;
  - the GPU smoke line becomes §3.8's command lines;
  - sixteen suites (correcting the stale "thirteen");
  - tool flags map 1:1 onto stage kwargs;
  - the conftest checkpointing-off fixture is the only test-side knob default;
  - `PRISM_VRAM_CEILING_GIB` and `PRISM_MEM_LOG_EVERY` are core-level environment settings (D13);
  - archiving means moving the file and then `git rm --cached`, while a folded script is `git rm`'d;
  - the `scripts/` line becomes `core/tool/` and `python -m core --help`.
- **The auto-memory files** `memory/dev-environment.md` and `memory/prism-refactor-state.md`: the recipe becomes `python -m core smoke`, the prompt CLI is retired, and `scripts/` is gone.
- **This spec's §11:** filled in.

## 10. Task order (for the plan)

One logical step per commit, with the fast gate after each.

| # | commit | depends on | GPU | walkthrough |
|---|---|---|---|---|
| T1 | stage kwargs (§2.6: `checkpoint_every`, `max_num_epochs`, `resume` and its refusals, `make_sim_config(hw=)`, the budget line); the conftest checkpointing-off fixture; `build_tiny_run` kwargs; pin extension | — | after T2 | — |
| T2 | D7: `fresh_run_near_misses`, `_training_run_size`, the refusal and the hoisted `require` in `build_posterior`; `_confirm_fresh_run` returns `(go, new_run)`, plus `_ask_new_run`; tests | T1 | **interim** | B4 |
| T3 | delete the chi override hatch (§2.9); test row 11 | — | — | — |
| T4 | `PreflightWarning`, `_truth_outside_region`, `SimConfig.clear_ground_truth` and its use in `install`, `simulated_inference`, `experimental_inference` | T1 | — | — |
| T5 | `tsnpe_round` (the checks, the new refusal, `install`, `t_scale_idx`) | T2, T4 | — | B7 |
| T6 | the Infer tab dispatches the compositions; the two inference runners go; test row 5 | T4 | — | B6 |
| T7 | the TSNPE tab dispatches `tsnpe_round` with a loaded observation, plus the `new_run` box; `runners.py` deleted; pin row 2 | T5, T6 | — | B5, B7 |
| T8 | D8: the Posterior tab's `_accept_for_load` dialog; the Infer checkbox; the two reworded refusals (§2.13) | T6, T7 | — | B1, B2, B3 |
| T9 | D9, plus the slow test's frequencies | T4 | — (no GPU path: the recording builder loads files and moves no new tensor; the CPU test plus the slow test at T25 cover it) | — |
| T10 | D1 retirement (§2.12, with the `__main__` stub), `run_fdt` booleans, pin row 1 deleted, the prompt-retired pin, the D1 reference sweep | T4, T5 | — | — |
| T11 | tool skeleton (§3.1-3.4), the entry in `core/__main__.py`, the `--help` epilog (D13), `prior`, `train`, `install_sbitest` | T1-T3, T10 | — | — |
| T12 | `validate`, `infer`, `tsnpe`; `--accept-*` | T5, T8, T9, T11 | — | — |
| T13 | the `diagnostic` kind and `load_diagnostic` | — | — | — |
| T14 | `core/diagnostics/feature_sets.py` and `rng.py` (the helpers leave `_common` in the same commit; every remaining importer already fails at import, each on a path constant the clean break deleted, and each is either archived at T21 or folded at T16-T18) | — | — | — |
| T15 | factor `validate_calibration` (no change in behaviour) | — | **interim** | — |
| T16 | `sbc_repeats`, the `sbc` subcommand, the guardrail-8 leg; `git rm sbc_characterize.py` | T12-T15 | at T19 | — |
| T17 | `identifiability` (three functions and the subcommand); `git rm` of three scripts | T12-T14 | at T19 | — |
| T18 | `load_rows(x_only=, max_rows=)`, `channel_ablation`, `ablation`; `git rm channel_ablation.py` | T12-T14 | at T19 | — |
| T19 | `smoke`; `git rm smoke_train.py`; the recipe becomes command lines in `CLAUDE.md` and `docs/STATE.md` | T12, T14 | **gate of record + diagnostic card runs** | — |
| T20 | `fdt`, `crossval` | T10, T11 | — | — |
| T21 | archive the six, `git rm _common.py`, remove `scripts/`, introduce `CODE_ROOTS`, point the rotation scan at it | T14-T20 | — | — |
| T22 | the literal-path scan over `CODE_ROOTS`, and the code-roots guard | T21 | — | — |
| T23 | stale comments and script citations (§7) | T21 | — | — |
| T24 | the piece-2 walkthrough section intro, including its two notes on what B1 and B3 supersede in A7 (row A7 itself is not edited), if not already added with B1-B7 | T8 | — | — |
| T25 | documents of record (§9.5); the slow test; the final fast gate | all | only if a tensor-moving change landed after T19 | the user runs B1-B6 |

## 11. Deviations made during execution (piece 2)

Every ruling made during execution that departs from this spec, with the task that made it. Piece 2's commits are `0016dae`..`d34997c`. Rows 1–43 were ruled during tasks T1–T25; rows 44–57 come from the whole-branch final review and its fixes (`2010250`..`d34997c`); row 58 was found while writing the documents. Labels such as R-A or M1 name entries in the execution ledger and the final review, which are gitignored scratch under `.superpowers/sdd/2026-09-12-one-flow/`; this table and the decisions log in `docs/STATE.md` are the durable record. Where a row says a section's text is now false, that section was left as approved and this row governs.

| # | task | deviation from this spec | why |
|---|---|---|---|
| 1 | T2 | §2.7 says the near-miss check sits "after every entry refusal: the taken name and the region without a digest". It sits where §2.7's concrete placement puts it (inside `if ck_every and train_new:`, after `peek`, before `rotate`), which is BEFORE the truncation refusals and the Fisher; the code comment now states that order (91aa51a). | The concrete placement is precise and binding; the "after the region without a digest" clause contradicts it. Every refusal still precedes any spend, and the region-without-digest pin runs with checkpointing off. |
| 2 | T2 | §2.7 says `build_posterior` shares the detector's internals rather than restating them; the stage restates the identity/root/`peek`/`near_miss_siblings` logic (`core/orchestrator.py:180-182` vs `:867-884`) while `fresh_run_near_misses`' docstring says "ONE DETECTOR". | Plan-mandated code; the two agree field for field today. Left open by the final review. |
| 3 | T4 | §8.2 T4 asks each composition's forwarding pin to cover `n_samples` both set and unset; each test covers one branch (simulated: set; experimental: unset). | Deferred minor; the forwarding code is shared in shape, so the gap is test coverage only. |
| 4 | T5 | §2.5 warns when `q < 0.99`; the code writes the threshold as `truncate.DEFAULT_HPD - 0.009` (0.99 today). | Plan-mandated expression; identical today, but it moves if `DEFAULT_HPD` changes. Deferred minor. |
| 5 | T5 | §7 (and §9) cite `test_tsnpe_round_uses_the_store_it_is_given`, which does not exist; the coverage lives in `test_tsnpe_round_installs_the_observation_then_forwards_every_knob_to_build_posterior`. | The §8.2 names were indicative. §7 and §9 keep the old name; this row is the correction. |
| 6 | T7 | §2.7's `_near_miss_message` names only the Posterior tab's "Start a new run anyway"; the message also names the TSNPE tab's "Start a new simulation even if a cache one setting away exists" box beside `--new-run` (R-O, e622ce7). | D7 also fires for rounds, and a round refused on the TSNPE tab otherwise named a button that exists only on the Posterior tab. |
| 7 | T10 | §7's stale-reference table missed D1 prose: `core/cli.py:91, :232, :246, :301`, `core/Helpers/visualizers.py:37,71`, `core/Reduction/sweep.py:2,5,163`. Carried into T23's sweep and fixed there (7c5ff48). | Found by T10's review; the spec table was incomplete. |
| 8 | T11 | §3.1 (and the tool's `--help` epilog) say core reads both core-level settings live; only `PRISM_VRAM_CEILING_GIB` is live, `PRISM_MEM_LOG_EVERY` is read once at import (`pipeline.py:155`). Corrected in the code and documents by the final review (row 47). | A slip in the spec copied into the epilog. |
| 9 | T11, T12, T16, T18, T19, T20 | §8.4's single table-driven `test_every_knob_flag_reaches_its_stage_as_a_keyword` was built by no task. It is replaced by per-subcommand pins that check the exact keyword set each callee receives (prior/train T11; validate/infer/tsnpe T12, 7d91ab0; sbc T16; ablation T18; smoke T19, 0d96d2f; fdt/crossval T20) (R-E; parked at T12). | The plan never created the table; a per-callee exact keyword-set pin catches the dest drift that `knobs()` would swallow silently. |
| 10 | T12 | §8.4's `test_tsnpe_subcommand_runs_a_round` command line has no `--directions`; the test passes `--directions 3`. | The SBITEST latent is 4 wide, so the default of 5 directions is refused by §2.5's `n > P` refusal. |
| 11 | T12 | §3.5 builds `recording_set(cfg, args)` inside the `experimental_inference` call, after the posterior and prior are loaded; `infer` validates the recording flags BEFORE loading anything (R-L; ordering pinned in 7d91ab0). | A usage error then costs no load (refuse before the spend). |
| 12 | T12 | New usage error: `infer --cell` together with any of `--forced`, `--drive`, `--f0-si` exits 2 (`core/tool/stages.py:187`); §3.5 only says those flags are "for experimental only" (7d91ab0). | Silently ignoring a flag the operator gave is the D6 trap. |
| 13 | T14 | §4.2 names the moved constant `GROUP_G_PREFIX`; the code keeps the private `_GROUP_G_PREFIX` (`core/diagnostics/feature_sets.py:23`). | The plan kept the script's name for a byte-identical move; noted as spec drift by the preflight scan. |
| 14 | T14-T19, T23 | §7's rule renames folded-script citations to their subcommands; provenance docstrings and history comments written in piece 2 ("Folded from scripts/x.py", "Moved from scripts/_common.py", "that was scripts/smoke_train.py's leak") keep the script names, and the leftover searches accept them (R-B). | They record where code came from, not an instruction to run a retired thing; rewording them to satisfy a count would lose the provenance. |
| 15 | T15, T23 | §4.3 says the comment at `:1424-1427` names `sbc_repeats(chi_k_fixed=)` and §7 gives `python -m core sbc` (`--chi-k-fixed`) at 12-space indentation; T15 moved the comment into `validate_calibration`'s body and it names `python -m core sbc --chi-k-fixed` (`core.diagnostics.sbc_repeats`); T23 keeps that wording and indentation (R-G; 50a9bb6). | The call site the comment annotated moved into a helper; the wording names both the command and the function. |
| 16 | T16 | `sbc_repeats` also refuses `num_posterior_samples < 1` up front (`core/diagnostics/sbc.py:78`), a refusal §4.4 does not list (cc2041e). | Without it the bad value failed only after simulating the calibration set. |
| 17 | T16 | §3.4 says the tool restates no default from `config.py`; the `sbc` help strings restate the stage defaults (values are still passed only when set). | Plan-mandated operator text; stale risk only. Deferred minor. |
| 18 | T16, T17, T18 | The diagnostics subcommands declare `--accept-truncated` with `config_args.add_accept_flags(p)` and build the Accept with `config_args.accept_from(args)` instead of restating the flag and hand-building `Accept` as §3.4 spells it (R-T). | T12 made those helpers the one definition of the D8 flags and their help text; a second copy is how the two drift. |
| 19 | T17 | §4.5 puts `fork_rng` around the CRN seeds of `_jacobian_features` only; `identifiability_laplace` also wraps its CRN seed calls in `fork_rng`, leaving its noise-floor ensemble on the running seeded stream (2a8d178, corrected in 8e25978). | Each CRN `manual_seed` otherwise reset the stream, so `--seed` stopped mattering after point 1; the first fix also wrapped the noise-floor call and made every point replay identical noise, which round 2 undid. |
| 20 | T17 | New refusals beyond §4.5: `identifiability_rotation` refuses `n_worst > P`; laplace and jacobian refuse `m_noise < 10` and a `t_obs_s` whose computed `n_obs` is non-finite or below 1; jacobian refuses fewer than 10 finite baseline rows (`core/diagnostics/identifiability.py:67, :360, :368, :817, :825, :855`) (2a8d178, 8e25978). | Each input previously gave a negative slice, RuntimeWarnings, or a zero-size reduction after the spend; a bare `t_obs_s <= 0` check let a tiny or NaN value through. |
| 21 | T17 | §8.2 asks the rotation test to pin a participation ratio, but §4.5's rotation results have no such key; the tests pin §4.5's keys. | Spec inconsistency between §8.2 and §4.5; results stay limited to §4.5. |
| 22 | T17 | §7's table misses the citations of the deleted `degeneracy_map.py` at `core/orchestrator.py:~807` and `core/SBI/pipeline.py:~1338` (and `scripts/feature_candidate_test.py:8`, archived at T21). Carried into T23's sweep and fixed there (7c5ff48). | Found by T17's review; the spec table was incomplete. |
| 23 | T18 | `channel_ablation` refuses `rows < 1` and `n_sweep < 2` before reading the cache (`core/diagnostics/ablation.py:92-100`); §4.6 lists no such refusal (a6dae3f). | `--rows -5` read `x[:-5]` and recorded `rows=-5`; `--n-sweep 1` wrote a p1-only table; 0 crashed inside `create` after the read. |
| 24 | T18 | §4.6 says `load_rows(max_rows=)` skips the final `got != batches_done` check; it skips it only when the cap actually stopped the walk early, and still refuses a cache that runs out of shards before the cap (`core/SBI/training_checkpoint.py:373-378, :409`) (a6dae3f). | Running dry before the cap is corruption, not a partial read. |
| 25 | T18 | Beyond §4.6's verdicts: a non-finite displacement gets its own verdict, is recorded as None, and `counts` gains `nonfinite` (a6dae3f, tests b070619). | A NaN displacement fell through to "healthy" and the `allow_nan=False` manifest write raised after the sweep. |
| 26 | T19 | §10 lists T19 as one commit; it landed as 187a096 (smoke), 0d96d2f (review fixes) and 93f5585 (the STATE.md GPU gate-table row) (R-U). | The ~30-minute GPU gate of record cannot run inside the task's own step, and the row must carry the measured values. |
| 27 | T19 | §9.3's card runs of `sbc` and `ablation` against run 1's chi store carry `--chi`, as the `identifiability jacobian` run does (R-F). | Without it the tool builds a forced-mode config (the default) and the store refuses the chi posterior, so the run exits 1. |
| 28 | T19 | The recipe of record in `CLAUDE.md` defines `$C` for the cell and uses `@B @C` instead of §3.8's inline `--cell`; it is the single recipe text, copied verbatim into `docs/STATE.md` (R-I). | One text keeps the two copies identical; the plan had two differing copies. |
| 29 | T19 | The recipe's comments are short per-run labels (`# run 2b: one setting away; must exit 1 naming n_runs`) instead of §3.8's annotations; the third command is called run 2b, as §3.8 does, where the plan's prose had called it run 3 (R-V). | Run 3 is the forced run; the per-line labels make each command's expected exit plain. |
| 30 | T19 | The recipe block uses a `$S` scratch variable, gives each command's pass criteria read off `$LASTEXITCODE` (including run 2b's no `[fisher]` line and no new `simulations/` directory), and points at the diagnostic card runs after a `core/diagnostics` change, which §3.8's block omits (0d96d2f). | §3.8's bare `<scratch>` is a PowerShell parse error, and the review found the recipe of record dropped §9.3's card runs and run 2b's criteria. |
| 31 | T19 | §3.8's criterion "runs 1 and 3 each finish within a few seconds per stage of the bb38f1a figures" holds only for prior and validate: run 1's posterior took 198 s vs 750 s and infer 85 s vs 59 s. The STATE.md GPU row (93f5585) stands as measured (R-X). | Not a regression: smoke's truth-free cfg anchors the Fisher at the prior median plus 7 prior draws, so the simulation cost follows different operating points (1341 vs 608 segments); the truth-anchored interim check at 50a9bb6 matched on every stage. |
| 32 | T19 | An unknown `--stages` entry is rejected by an argparse `type=` validator at parse time; §3.2 classes it as a `UsageError` raised after parsing (exit 2 either way) (0d96d2f). | Validating at parse time refuses before `mkdtemp` and the config build. |
| 33 | T19 | §3.7 says smoke's default store root is a fresh `mkdtemp` "printed and left on disk"; `main` now removes that auto-created root (rmdir only) when it is still empty after a failure (`core/tool/__init__.py:56-63, :84-86`) (0d96d2f). | An empty `prism_smoke_*` directory leaked on every early failure, and its path was never printed. |
| 34 | T19 | §3.9 gives one Ctrl-C message; `smoke` gets its own advice, keyed on the `--store-root` flag (`core/tool/__init__.py:43, :106`) (0d96d2f). The final review changed what it says (row 50). | The generic "--resume require continues them" advice was wrong for smoke's temp store. |
| 35 | T19 | `--resume` and `--new-run` are defined once, in `config_args.add_resume_flags` (`core/tool/config_args.py:54`), and used by `train`, `tsnpe` and `smoke` (0d96d2f). | The review found the two flags defined in three places with different help text. |
| 36 | T20 | `fdt` and `crossval` set their own interrupt note through `set_defaults`: partial outputs stay under `<root>/fdt` or `<root>/crossval`, and neither has `--resume` (ea83fa2). | §3.9's message would tell an FDT operator the output was removed and to use `--resume require`, both false. |
| 37 | T20 | `crossval`'s grid point count must be a finite whole number of at least 2; a non-finite value such as `--s-grid 0 0.1 inf` exits 2 (`core/tool/fdt.py:73-76`) (ea83fa2). | It raised an OverflowError traceback instead of a usage error. |
| 38 | T20 | The fdt flag test sets `skip_sanity` and `confirm_production` to differing values in both directions (ea83fa2); the plan's two-call test set them equal both times. | A cross-wiring passed the plan's test (a plain `fdt --cell` would abort after sanity with the suite green). |
| 39 | T20 | `core/FDT/plots.py` closes a figure when `save_path` is set and calls `plt.show()` only when it is not (the `cross_validation_plots.py:134-141` split), with a test (bb22ac4); §1.3 leaves FDT hardening to piece 5. | The tool made the Agg "non-interactive" warning operator-facing on every fdt run, and in-process runs leaked figures; every caller passes `save_path`. |
| 40 | T22 | §8.3's guard compares top-level directories only; the literal-path scan also walks top-level `*.py` files, the guard pins them to `CODE_FILES = ("conftest.py",)` beside `CODE_ROOTS` (`tests/_fixtures.py:28`), a missing entry fails with a message naming it, and the skip reasons describe `.claude/` generically (e12c5ff, 4094e5a). The final review made the `input()` and rotation-reader scans walk `CODE_FILES` too (f5ab205). | `conftest.py` is live code that the scans skipped; a docstring naming machine-local contents goes stale. |
| 41 | T24 | The walkthrough gains row B8 (the FDT panel's full run, re-checked) beyond §9.4's B1-B7 (d6b1f03); STATE.md's walkthrough row names B1-B8 (R-X), where §10's T25 row says the user runs B1-B6. | bb22ac4 changed the plots the GUI FDT panel's full run uses, and that check last passed on 2026-09-11, before it. |
| 42 | T3-T25 | §9.1 records each task's gate in `docs/STATE.md`; the per-task gate results live in the execution ledger, STATE.md gets the GPU gate-of-record row at T19, and T25 appends one final fast-gate row (preflight ruling, R-C). | The gate ran after each commit, so a task could not record its own row, and 25 transient rows are churn. |
| 43 | T25 | §9.2 runs only `pytest tests/test_user_sbi.py -m slow` at T25. The end order is the whole-branch review and its fix, then `pytest -m slow` (the chi pipeline test AND T20's slow-marked FDT tiny-size test) and the final fast gate with `--durations`, then T25 writes the documents from those numbers (R-W). | The gates recorded in STATE.md must certify the final code, and the ~45-minute slow set runs once. |
| 44 | final review (F1; 11361cb, d8e9585) | New refusals beyond §2.6 and §3.5: on a call that trains, `build_posterior` refuses `max_num_epochs`, `hidden_features`, `num_transforms` or `stop_after_epochs` below 1, and a `learning_rate` that is not a finite positive number, before the Fisher and any simulation. A call that loads a stored posterior skips these checks. | They were resolved only after the whole spend: `--max-epochs -1` simulated everything and then crashed inside sbi, and `--learning-rate 0` or `--stop-after-epochs 0` silently wrote an untrained posterior. Checking them on a load would block loading a posterior while a GUI field holds a bad value. |
| 45 | final review (F2; 11361cb) | New refusals beyond §2.3, §2.4 and §4.3: `validate_calibration` refuses `n_cal < 1` and `num_posterior_samples < 1` before `store.create`; `simulated_inference`, `experimental_inference` and `infer_and_visualize` refuse `n_samples < 1` right after the name check. | Zero samples failed only after the calibration set was simulated, or after an orphan observation was written. |
| 46 | final review (F3; 9aaed26) | §2.11's `install` contract is extended to the cell: installing a simulated observation puts back `cfg.sources["cell"]` only when the recorded cell file exists and its sha256 matches the record, and removes the entry otherwise. | In the GUI, a TSNPE round around observation A after a simulated inference on cell B recorded B as the round's cell. Restoring an unchecked path would turn a moved cell file into a refusal after the Fisher. |
| 47 | final review (F4; e21e035) | §3.1's "which core reads live" and §7's T3 row ("read live from the environment, like `PRISM_MEM_LOG_EVERY`") are false for `PRISM_MEM_LOG_EVERY`. `PRISM_VRAM_CEILING_GIB` is read live on each batch plan; `PRISM_MEM_LOG_EVERY` is read once, when `core.SBI.pipeline` is imported. `config.py`, the `pipeline.py` docstring, a `test_user_sbi.py` docstring, both `--help` epilogs (the tool's and `smoke`'s) and `CLAUDE.md` now say so. D13 is unchanged. | Setting `PRISM_MEM_LOG_EVERY` inside a running GUI changes nothing, so "read live" misleads the operator. |
| 48 | final review (F5; e21e035) | `smoke` checks `--cell` against a throwaway config (`config_args.make_cfg`), when `infer` is among the stages, before the prior is built; §3.3 step 4 and §3.7 leave the cell to `simulated_inference`. The training config stays truth-free. | A mistyped cell, or one outside `--bounds`, cost the prior, training and calibration before exiting 1 and left orphans behind. |
| 49 | final review (F6; e21e035) | `smoke` refuses `--resume require` or `--resume never` without `--checkpoint`, and `--resume require` without `--prior`, as usage errors (exit 2) before the prior is built, when `posterior` is among the stages. §3.7's flag table (its `--resume` row) is now false: it says the stage refuses with exit 1. | The stage's refusal came after the prior build and named `--checkpoint-every N`, a flag `smoke` lacks; a resume without `--prior` fits a new prior, whose fingerprint can never match the cache. |
| 50 | final review (F7; e21e035) | `smoke`'s Ctrl-C advice (row 34) names the prior actually used (`--prior`, else `smoke_prior` under `--save`, else the unnamed prior's id), the store root in use (the temp root included), `--checkpoint`, and the same `--num-runs` and `--run-size`. | The advice always said `--prior smoke_prior` and that a temp root cannot be resumed; both were false in common cases, and the temp root is kept whenever it holds committed batches. |
| 51 | final review (F8; e21e035) | `fdt --skip-sanity --no-production` is a usage error (exit 2) before any config build; §3.5's `fdt` row allows the pair. The help no longer says `--no-production` is ignored with `--skip-sanity`. | The pair silently ran the full production sweep. |
| 52 | final review (F9; af47063) | `assert_forced` advises `--bounds …/nadrowski/master.txt` together with a forced cell `--cell …/nadrowski/master_weak.txt`; §4.2 says it suggests only the `master_weak.txt` cell. | The refusal depends on the bounds file, which the tool never resolves from the cell, so changing `--cell` alone could not clear it. |
| 53 | final review (F10; af47063) | New refusals beyond §4.5: laplace and jacobian refuse `m < 1`, a `rel` that is not a finite positive number, and `min_valid` outside (0, 1]; `identifiability_rotation` refuses `n_worst < 1` and `top_n < 1`, where the lifted code clamped them to 1. | `--m 0` ran the whole noise ensemble first; `--rel 0` with a zero-valued truth divided by zero after the spend; a silent clamp quietly changes what the operator asked for (the D6 trap). |
| 54 | final review (M3, M4; 9c9a397, follow-ups d8e9585, d34997c) | `sbc`'s KS table prints p-values as numbers (`8.2e`) and `frac<.05` as `9.3f`. The pooled rank histogram's bin count has a cap of `min(N // 20, (nps + 1) // 10)` (at least 1) and is the largest divisor of `nps + 1` between half the cap and the cap, else the cap itself (`_rank_hist_bins`). §4.4 names no bin rule; the landed code used `N // 20`. | Truncating `str(float)` printed 3.2e-20 as `3.212345`, on exactly the miscalibrated rows the table sorts first. 950 bins over 1001 integer ranks drew comb spikes on a calibrated posterior; a count that divides `nps + 1` gives every bin the same number of ranks, and a pure largest-divisor rule fell to 1 bin when `nps + 1` is prime. |
| 55 | final review (M5; eb60fc0) | Both dialogs make Cancel the default button. For the D8 dialog this restores §5.3 ("Cancel" as the default); for the D7 dialog, which §5.3 moves verbatim without naming a default, it is new. The landed code made the destructive button ("Load it", "Start a new run anyway") the default in both. | Pressing Enter took the destructive path: a fresh cache one setting away from a committed one, or loading a non-amortized posterior. |
| 56 | final review (M6; 8d48236) | Walkthrough row B5 is rewritten: reload the same amortized posterior on the Posterior tab before the Batches 3 round; the refusal then names `n_runs` (this run 3, that cache 2). §9.4's B5 text ("Repeat a round at Batches 3, get refused") is now false. | Every round installs its result as the session's posterior, so the repeated round had a different parent and a different region; two fields differed, and the one-setting refusal could never fire. |
| 57 | final review (M1, M2; 2010250, ea38593) | Bug fix where §2.10 is silent: `build_experiment_observation` sets `cfg.n_obs` to the recording's length and, in chi mode, `cfg.chi_obs_freqs` to the stated drive frequencies (cell units, in recording order) and `cfg.chi_n_freqs` to their count; other modes set `cfg.chi_obs_freqs` to None. | The observation recorded the previous observation's `n_obs`, so every inference on it failed after the PPC, even in a fresh process. A chi experimental observation recorded no frequencies (or the previous cell's), so the PPC re-drove a different experiment. |
| 58 | T25 | §8.5's count and recording: 460 tests are collected at `d34997c`, 99 more than the 361 at `df7491e`, where §8.5 budgeted about 42 ± 8 added. Per-suite durations for `test_tool.py` and `test_diagnostics.py` were not measured; the whole fast gate took 13 min 38 s, inside the 14-minute target. | Review and fix rounds added tests beyond the plan's list (the final review's fixes alone took the fast gate from 441 to 457 passed). `--durations=15` lists the slowest tests, not suites. |
