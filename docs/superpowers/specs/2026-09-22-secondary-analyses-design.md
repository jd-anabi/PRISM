# PRISM piece 5: the secondary analyses — hardening, records and comparison (design)

**Status:** approved 2026-09-22 (brainstormed with the owner the same day; every decision in §1.1 is
the owner's answer to a plain-language question). Reviewed against the code by four independent
readers and a judge before the owner read it; twelve must-fix findings and about thirty corrections
are folded in.
**Piece:** 5 of the seven-piece pre-retrain hardening programme
(`docs/superpowers/specs/2026-09-10-pre-retrain-hardening-decomposition.md` §4, row 5).
**Depends on:** piece 1 (the artifact store), piece 2 (the one-flow stage API and the tool),
piece 3 (`core/refusals.py`, `core.runs.public_entry`, logging), piece 4 (the Artifacts screen and
the `artifacts` subcommand family). It runs after all four, not in parallel with any.
**Followed by:** piece 6 (documentation and the retrain runbook).

**Anchor rule.** Every `file:line` below was checked on 2026-09-22, but **the quoted text is the
anchor, not the number** — line numbers go stale between the plan and the task. An implementer who
finds the quoted text elsewhere in the named file has found the right place.

---

## 1. Purpose and scope

Two analyses sit outside everything pieces 1–4 built.

- **The single-cell measurement** (`core/FDT/fdt_pipeline.run_fdt`) asks, for one cell, whether the
  hair bundle obeys the fluctuation–dissipation theorem. It runs an unforced ensemble and takes a
  Welch spectrum, then drives the bundle at each of a log-spaced grid of frequencies and lock-in
  detects the complex susceptibility, and reports the effective-temperature ratio — 1 at
  equilibrium, far from 1 where the bundle is active.
- **The parameter sweep** (`core/FDT/cross_validation.run_fdt_param_sweep`, reached as
  `run_param_study_cli`) repeats that measurement across a grid of operating points in one
  parameter, twice — once over the motor's activity and once over temperature.

Neither is hardened. Measured on 2026-09-22:

- **No input is checked anywhere.** `FDTConfig` has no `__post_init__`
  (`core/sim_config.py:562`, contrast `SimConfig`), no `require_*` rule from `core/refusals.py`
  is called in `core/FDT/` or `core/tool/fdt.py`, and the window's numeric boxes return `0` for a
  blank (`core/gui/widgets/labeled_inputs.py:47-51`). Zero is not harmless: `freqs_per_batch = 0`
  makes `_plan_adaptive_batches` append `(start, 0)` for ever (`core/FDT/campaigns.py:76-88`) — the
  application looks hung, not failed; `ensemble_M = 0` raises `ZeroDivisionError` in `_pick_n_segs`
  (`campaigns.py:47-51`, `FDT_MAX_ELEMENTS_PER_SEG // batch_size`); `F0 = 0` divides by zero in
  `lock_in_chi` (`core/FDT/spectral.py:86`, `2.0 / (F0 * T_obs)`); `n_freqs = 0` yields an empty
  grid, an empty figure and exit 0.
- **The cell's normalisation constant is checked after both campaigns.**
  `observable_noise_prefactor` is called at `core/FDT/fdt_pipeline.py:128-134`, after
  `run_campaign2_chi` returns. On the `--skip-sanity` path nothing calls it earlier, so a cell
  missing `n` or `beta` costs the whole run before it is refused — and the refusals it raises carry
  no field key (`campaigns.py:103,127,133,136,141,145`), so `Refusal.field` is `None` and neither
  front-end table can name a control or a flag.
- **Neither is a public entry.** `grep` for `public_entry` over `core/` finds it, outside its own
  definition at `core/runs.py:231` and a docstring mention at `core/sim_config.py:540`, only in
  `core/orchestrator.py` and the three diagnostics. So neither gets piece 3's private settings copy
  (V1) — `run_fdt` writes `cfg.omega_0` on the caller's object twice, at `fdt_pipeline.py:73` and
  `:117` — and neither is inside `capture_run()`, so neither produces a `log.txt`.
- **Nothing records what produced a result.** The single-cell run has five figure-save sites — the
  sanity-only passive plot (`fdt_pipeline.py:84`), the spontaneous trajectory (`:104`) and the three
  written at the end (`:138-140`) — into a flat `<artifacts root>/fdt` whose filenames carry only a
  second-resolution wall-clock stamp, and it returns `None`. The numbers — the frequency grid, the
  spectrum, the susceptibility, the ratio — are computed, plotted and discarded. The sweep writes
  one HDF5 per swept parameter into `<artifacts root>/crossval` and records `psd_T_obs_nd` in it but
  not `freq_bounds`, `T_obs_periods`, `dt_nd`, `burn_in_nd`, the preset name, the device or the cell.
- **Nothing reads either output back.** `load_param_sweep`'s only callers are
  `cross_validation.py:320` and `:333`, inside the run that wrote the file. The one function in the
  tree written to consume this data, `core/Reduction/plots.plot_cross_validation_3d`, has no caller
  anywhere.
- **There is a live instance of the hazard in the repository.** `Artifacts/fdt/` holds two PNGs from
  one run stamped `20260915_153042` — the passive-baseline plot and the spontaneous trajectory, the
  first two of the five — and none of the three a complete run writes at the end. It carries no
  record of which cell or which settings produced it, and no command in either front end can list or
  delete it: `_entries` iterates directories only (`store.py:381`), so a file inside a kind
  directory is invisible to the listing, and the store never walks its own root, so a plain
  `crossval/` beside the kind directories is invisible too.
- **A documented guarantee is false.** `_interp_log` (`core/FDT/sanity.py:28-48`) promises "NaN
  outside the grid"; the Welch grid's first entry is exactly `0.0`
  (`core/FDT/spectral.py:56-58`, `omegas = 2π·rfftfreq(nperseg, dt)`), the helper clamps it to
  `1e-30` before taking logarithms, and the in-range test is
  `log_x_new >= log_x_old[0]` — which every positive frequency passes. The low end therefore never
  returns a blank; it returns a value blended from the zero-frequency bin, and the callers' count of
  excluded points reports zero. `PRISM_HANDOFF.md:2485-2489` and `:6754` state the opposite
  guarantee and name the consequence ("widen `freq_bounds` past the PSD resolution and `T_eff/T`
  acquires a smooth, plausible-looking, entirely fabricated tail"). At the default band the lowest
  probe sits above the first real bin, so it does not bite today; widen the band downward, shorten
  the spontaneous recording, or work at a lower resonance and it does, invisibly.
- **The Nadrowski sanity path is reached by no test.** `tests/test_tool.py:1363-1366` runs the
  single-cell leg on `Resources/Cells/hopf/cell.txt` with `--skip-sanity`, so the Nadrowski-only
  sanity checks and the passive-baseline plot they draw (`sanity.py:292-307`) are exercised nowhere.
  (The Nadrowski *normalisation* branch is covered — `tests/test_fdt_user.py:53` asserts
  `observable_noise_prefactor` directly, and the sweep leg of the same slow test runs a real
  Nadrowski cell.)

### 1.1 Decisions (binding)

Each is the owner's answer, given 2026-09-22, to a question asked in plain language.

| # | decision |
|---|---|
| **E1** | **Both analyses move into the artifact store, fully, with a reading side.** Named records, provenance with file fingerprints, listing, deletion, the outputs living inside the record's folder, the numbers saved and not just the pictures, and a way to pick an earlier run from the screens. |
| **E2** | **An interrupted run keeps its folder, marked unfinished.** The run writes into its own record folder from the start; a cancel or a crash leaves the folder with everything written so far, plainly marked unfinished, and never deletes it. This is the training cache's rule, adopted for the same reason. |
| **E3** | **One new kind covers both analyses**, with a field inside each record saying which of the two it is. |
| **E4** | **Nothing measurable is a refusal; some points failed is a completed record carrying the count.** A run that measured nothing — every operating point failed, or every frequency off-grid — refuses, naming the setting to change, and leaves its unfinished folder behind with whatever was collected. A run with some points failed completes, and the count is in the record and in both listings. |
| **E5** | **Checks refuse what breaks; a setting too thin to trust warns, and the warning is recorded.** Floors only where the computation is otherwise undefined. A deliberately tiny quick look stays possible and is visibly marked as one. |
| **E6** | **The fix-hint table is widened to understand screens**, not only the six inference tabs, and an input that appears in several places lists them all. Labels and hint sentences stay generated from one entry. |
| **E7** | **Every run records the seed it used**, drawing one when none is supplied, with reseeding confined the way the diagnostics confine theirs. Repeats of a cell use different seeds; their spread is the measurement error. |
| **E8** | **Comparing covers all four modes**: several cells' ratio curves on one axis; repeats of one cell with their spread; one run re-normalised with a supplied constant; two sweeps together. |
| **E9** | **The off-grid range test is fixed, and the band is refused before the driven campaign.** Frequencies outside what the spontaneous spectrum resolves come back blank and are counted; once the grid is built — the first moment it is knowable — a band reaching below that resolution refuses before the driven campaign is spent. |
| **E10** | **The tidy-up command learns to see the legacy loose files** so the owner can clear them; nothing is deleted on their behalf. |
| **E11** | **Both subcommands gain the destination flag**, and the three behaviours that key off its presence are reworked for them deliberately rather than flipping by accident. |
| **E12** | **One piece, ordered so every part of the hardening lands before the first line of the comparison facility.** Every point before the comparison work is a coherent place to stop. |

### 1.2 How the design reads the decisions

Rulings this design makes, which the owner did not have to decide but which follow from E1–E12 and
which a reviewer should check against.

- **The progressive record is a MODE on the existing writer, not a second writer** (E2). The
  training cache bypasses `ArtifactWriter` entirely and has its own manifest function
  (`store.write_simulation_manifest`), which costs it the payload, figure, `fig_sink` and `log.txt`
  handling. This kind needs all four, so `ArtifactWriter` gains an explicit per-kind progressive
  mode instead. The six ordinary kinds keep today's behaviour exactly — directory created at entry,
  manifest last, directory removed on any exception — and a test pins that they do (§8.2).
- **The FRONT END creates the writer; the STAGE enters it** (E1, E2, and forced by V4). A
  progressive record's directory name must be known before the run is dispatched, so the figure
  watcher can be pointed at it (§5.4); but `log.txt` is written from `runs.current_run_log()`
  (`store.py:343-347`), which is **thread-local** (`core/runs.py:137`,
  `_active = threading.local()`) and is only populated inside `capture_run()` on the worker thread.
  A writer entered and exited on the window's own thread would therefore write no `log.txt` at all.
  The shape that satisfies both: `store.create(...)` mints the id, runs `assert_name_free` before
  anything is spent and fills `writer.dir` **without creating the directory**
  (`store.py:546-550`, `:257`; `__enter__` does the `mkdir`), and the stage takes that writer and
  does `with writer:` itself, so `__enter__`, every `refresh()` and `__exit__` all run where the run
  log lives. Both front ends create; the stage enters.
- **`FDTConfig` gains two fields and one method.** The fields `sources` (so
  `provenance.inputs_from_cfg` works) and `seed`; the method `copy_for_run()`. The method is not
  optional: `public_entry` copies the config **only when it has one** (`core/runs.py:243-247`,
  `if hasattr(kwargs["cfg"], "copy_for_run")`), and `copy_for_run` is defined on `SimConfig` alone
  (`core/sim_config.py:539`). Without it the decorator gives `run_fdt` the run log and **not** V1's
  private copy, and `cfg.omega_0 = …` keeps writing on the caller's object — the exact defect §1
  names, claimed and silently absent. A plain deep copy of the dataclass suffices: `FDTConfig`
  carries no `cached_property` cache, so it needs none of `SimConfig.copy_for_run`'s `_CACHED`
  popping. Fields with defaults and one added method leave the reduction path — which shares this
  dataclass and is outside the programme — working untouched.
- **`manifest.config_from_cfg` gains one branch for the FDT settings shape.** Without it,
  `store.create("fdt", cfg, …)` raises `AttributeError` on `cfg.observation_mode`, the second key it
  reads (`manifest.py:211-238`). The block records the settings **as given to the builder**; the
  resonance the run discovers goes in `body.grid`, not in `config`, because the writer computes the
  config block at `create()` time from the caller's object while the stage runs on its private copy.
- **The checks live in the two builder functions** — `cli.make_fdt_config` (`core/cli.py:319`) and
  `cli.make_param_sweep_config` (`core/cli.py:377`) — not on `FDTConfig` and not in the screens.
  Both front ends inherit one wording, and `cli.make_reduction_config` (`core/cli.py:342`), the
  reduction map's builder and the only `FDTConfig` in the tree not pinned to the CPU, is untouched.
- **A two-parameter study writes TWO records, one per swept parameter** (E4). Today the study runs
  the activity sweep, plots it, then runs the temperature sweep (`cross_validation.py:304-337`), and
  an all-failed activity sweep raises before the first plot and before the second sweep starts. Two
  records make each sweep answerable on its own. A study that produced one listing entry now
  produces two; that is the visible cost.
- **A comparison is itself a record of the same kind**, with `study = "comparison"`. It names the
  runs it drew **in its body, not in `parents`**: the parents block is a flat `{key: id}` map read as
  `m.parents.get(pk) == id_` (`store.py:605`), so it cannot carry an arbitrary number of ids without
  widening the store's contract for every kind. Deleting a run a comparison used is therefore **not**
  refused, and `report.render_lineage` gains a branch that resolves the body's `compared` ids and
  prints `MISSING <kind> [<id>]` for one the store no longer holds, in the same wording its parents
  walk already uses (`report.py:222-224`) — otherwise the promise is unreachable, because
  `render_lineage` walks `m.parents` only and never touches the body.
- **`fdt` and `crossval` gain `--seed`, and both screens a Seed box** (E7, E8). Piece 2 recorded a
  reading that "only `smoke` and the diagnostics take `--seed`" (`docs/STATE.md:518`); E7 and E8
  widen it deliberately, because a recorded seed you cannot supply back does not make a run
  reproducible, and E8's repeat comparison needs a run to be identifiable as a repeat. The box is
  **not** remembered between launches (§5.5): a remembered seed would silently turn every run into a
  repeat of the last one. Blank means "draw one and record it".
- **Five settings are checked but are not front-end knobs.** `freq_bounds`, `burn_in_nd`,
  `T_obs_periods`, `dt_nd` and `psd_T_obs_nd` are parameters of neither builder and are exposed by
  neither front end (`cli.py:319-321`, `core/tool/fdt.py:83-92`, `fdt_panel.py:82-86`); for the
  sweep, three arrive from the closed `--preset` (`cli.py:404-408`). They are registered in `FIELDS`
  like any other key and carry `None` in **both** front-end tables, so `fix_sentence` returns the
  empty string and a refusal names the setting and offers no fix — which is honest, because there
  is no control and no flag to name. They are **not** unregistered: every `require_*` rule builds
  its sentence through `describe(key)`, which raises a bare `KeyError` for a key `FIELDS` does not
  hold (planning ruling **P2**, which corrects this bullet's first draft).
- **The model builder is inside §5's set.** Piece 3 handed five surfaces by name
  (`2026-09-15-validation-and-logging-design.md:88`) and `docs/STATE.md` repeats them. The builder is
  a `QWidget`, not a `BasePanel`, so its refusal goes through
  `core/gui/widgets/refusal_box.show_refusal` (`refusal_box.py:24`) rather than `_refusal`.
- **The main-cell-type gap closed here is the Nadrowski SANITY path** (§1's last bullet), not the
  normalisation branch, which is already covered.
- **The artifact table's numeric columns gain a sort key.** They sort as text today
  (`docs/STATE.md`'s piece-5 carry-forward: `10/12 batches` before `9/12`), and this piece adds a
  `Points` column of exactly that shape, so it would manufacture a fresh instance of the defect it
  inherits.
- **Walkthrough rows 15, 16 and B8 are marked superseded**, and this piece writes a new set. Row 15
  asserts how an all-failed run is reported, which E4 changes; rows 15 and 16 assert where the
  figures land, which E1 changes; B8 re-checks row 15 and asserts that the figures reach the panel,
  which E1 changes, and its no-"non-interactive"-line half is carried into the new rows.
- **Every test that runs a real campaign is `slow`-marked.** The fast gate's one-process target
  stays at 16 minutes (§9); the slow set's budget rises.

### 1.3 Out of scope, and who owns it

| thing | why | owner |
|---|---|---|
| The reduction map (`core/Reduction/`) | Excluded by the programme (decomposition §2). It shares `FDTConfig`, so every addition to that dataclass is a defaulted field or a method, and `make_reduction_config` is left alone. A test pins that the reduction builder still works. | nobody yet |
| Resuming an interrupted sweep | Not asked for. E2 makes the folder survive so its data file can be read, not so a later run can continue it. The sweep has never had a resume and none is added. | a later piece, if wanted |
| Re-deriving the science of the effective-temperature ratio | No new science. The off-grid fix (E9) corrects code against a guarantee the science reference already states; it changes no formula. | — |
| The stale memory figures in the science reference (`PRISM_HANDOFF.md:2615-2616` against `:6756`, opposite status for the same two items) | A documentation defect in a file piece 6 is splitting. Recorded, not fixed here. | piece 6 |
| A cross-process lock on a record being written | Out of scope in piece 4 and still out of scope. §2.5 says what the recency guard now means. | nobody yet |
| Deleting the owner's legacy files | E10 is "let the owner clear them", not "clear them". | the owner |
| The third wording of "the cell lacks what the bounds file requires", in `cli._merge_vals_bounds` (P53) | It carries the cell PATH, which the FDT builders need and the other two wordings do not, so folding it into `missing_values_phrase` would drop that or change the other two. §6.2 unifies two and knowingly leaves this one. | nobody yet |
| ~~The Live simulation tab crashes into the red box, `KeyError: "Forcing parameter 'amp' missing for kind 'sin'."`, on a spontaneous built-in cell such as `nadrowski/master_spont.txt`: `simulate_runner.run_simulation_stream` builds the sinusoidal force tensor for every built-in model, while `build_stream_config` leaves `forcing_idx` empty for a cell with no drive (found by T23's review, handed on by T35, §5.6)~~ **Struck: fixed after all, at the end of the piece, by the whole-piece review's fix dispatch (N17, ruling R-F4) — a departure from §5.6's bound, recorded as §12 row 34.** | ~~Not a refusal or data-loss defect, so §5.6's bounded mandate hands it on (P58). Cost if it is left: no spontaneous built-in cell — the passive recordings the inference is fitted to — can be streamed on the Live simulation tab; the operator gets a traceback where a trace should be, though nothing is lost or written.~~ | ~~nobody yet~~ — |
| In an unfinished single-cell record, the passive-baseline figure (the Nadrowski sanity path) is on disk from the end of the passive check (a check the whole-piece review measured at 466 s) but is listed in the manifest only at the next refresh, after the remaining sanity checks and Campaign 1, against §2.2 step 2's "after each figure". A cancel or a crash in that window still lists it (the keep branch's final manifest lists every figure on disk); only a hard kill — the process killed outright — leaves it on disk and unlisted (`core/FDT/fdt_pipeline.py`, the passive-plot path in `run_fdt`; found by the scoped re-review of the whole-piece fix dispatch, its minor 2) | Found after the piece's one fix dispatch and parked by ruling: the owner's end-of-piece process is one fix dispatch and one scoped re-review, and nothing is lost — the figure is on disk, it is merely not named in a record that is already marked unfinished. The fix is one `writer.refresh()` after `run_all_sanity` whenever the passive figure's path was handed out (P58). | nobody yet |

---

## 2. The new artifact kind

### 2.1 The kind, its directory, and the legacy files

The kind is **`fdt`**, its directory is **`fdt/`**, and both analyses and their comparisons live
there, told apart by the body's `study` field: `"single"`, `"sweep"` or `"comparison"`.

`crossval` is deliberately not used as a kind or directory name: `core/Reduction/plots.py:192`
already writes `reduction_crossval_{stamp}.png` into the reduction directory, and a second meaning
for the word would collide with existing vocabulary.

Reusing `fdt/` as the kind directory is chosen, not accidental. The owner's existing stray pictures
then sit as **loose files inside a kind directory**, which is where §6.3's tidy-up can reach them.
`<artifacts root>/crossval/` does not exist on the owner's machine and is handled as a legacy
directory wherever it does.

### 2.2 Progressive records: the writer's second mode

`ArtifactWriter` today creates the directory at `__enter__`, and on `__exit__` either removes the
whole directory (any exception, a cancel included — the window's cancel is a `BaseException`
subclass, `core/gui/streams.py:38-45`) or commits: hash the payloads, validate the manifest, write
`log.txt`, write `manifest.json` last and atomically (`store.py:294-349`).

The `fdt` kind opts into a **progressive** mode:

1. **`__enter__` creates the directory and writes a first manifest carrying EVERY key of
   `manifest.BODY_KEYS["fdt"]`.** `validate` refuses a partial set — `want = set(BODY_KEYS[…])` then
   `if set(d["body"]) != want: raise ManifestError` (`manifest.py:145-147`) — so the first write
   carries `complete` false, the facts already known (`study`, `settings`, `seed`) filled in,
   `notices` opening as `[]`, and `grid`, `points`, `offgrid`, `compared` and `results` `null` until
   a refresh fills them. The caller sets `w.body` between `create()` and the `with`;
   `ArtifactWriter.body` is already a plain public attribute (`store.py:259`), so `create` needs no
   new argument.
2. **`w.refresh()`** re-validates and re-writes the manifest, and re-writes `log.txt`, at points the
   stage chooses — after the spontaneous campaign, after each operating point, after each figure.
   Each refresh is the same atomic write `_write_manifest` already performs.
3. **`__exit__` on an exception keeps the directory**, leaves `complete` at `False`, writes one final
   refresh so the log up to the failure is on disk, and re-raises — **except** when the exception is
   a `Refusal` raised before the first payload or figure was written, in which case the directory is
   removed. A pre-spend refusal must not leave a permanent empty record that §2.5 says the sweep can
   never clear.
4. **`__exit__` on the clean path** commits as today, with `complete` set to `True`.

Constraints this mode inherits and does not change: payloads are flat files directly inside the
directory (`store.py:263-268`), so the sweep's `.h5` is stored as-is with no new code; `figures/` is
the one subdirectory and holds only PNGs (`store.py:270-279`); figures are recorded by name and
never hashed, while payloads are hashed at commit. A progressive record's payload hashes are
therefore written at the final commit only — an unfinished record lists its payloads with a null
hash, which the listing shows rather than hides.

The six ordinary kinds do not opt in. §8.2 pins that a failure inside any of them still removes the
directory, and that a failure inside an `fdt` record does not.

### 2.3 What a record holds

**Body keys** (`manifest.BODY_KEYS["fdt"]`; the set must match exactly at every write):

| key | holds | null when |
|---|---|---|
| `study` | `"single"`, `"sweep"` or `"comparison"` | never |
| `settings` | every knob the run resolved: `n_freqs`, `ensemble_M`, `freqs_per_batch`, `F0`, `freq_bounds`, `burn_in_nd`, `T_obs_periods`, `dt_nd`, `psd_T_obs_nd`, `skip_sanity`, `confirm_production`, the preset **name** for a sweep, and the sweep's grid as `(min, max, N)` | never |
| `seed` | the integer the run used (E7) | a comparison |
| `grid` | `omega_0`, `omega_0_source` (the model estimate or the spectrum peak), `n_freqs`, `bounds`, `omega_min`, `omega_max` | a sweep (each point's grid is in `data.h5`); until the spontaneous campaign finishes |
| `points` | a sweep: `{param, planned, done, failed}` | a single run; a comparison |
| `offgrid` | `{blanks, of}` — how many grid frequencies came back blank, of how many (E9) | a comparison; until the driven campaign finishes |
| `notices` | the "too thin to trust" warnings this run raised, as sentences (E5) | never — `[]` when there are none |
| `compared` | a comparison: the mode, and the ids and names of the records it drew | a single run; a sweep |
| `complete` | `False` until a clean finish (E2) | never |
| `results` | a short summary — the peak ratio and where it occurred, the ratio at the resonance, the fraction of the grid usable; for a sweep, a summary across its points. **Finite numbers only**: `validate` refuses any non-finite float in the body or the config (`manifest.py:107-118,157-158`) and this ratio legitimately carries NaN, so a summary that would be NaN is recorded as `null` | until the run finishes |

The swept parameter is recorded **once**, in `points.param`; `settings` carries the grid, not the
parameter name.

**Payloads** (flat, hashed at commit):

- `data.h5` — the numbers (E1). **The layout is chosen by `body.study`, and the file writes a
  `study` attribute at its root** so a reader can check which it holds. A single-cell run stores the
  frequency grid, the spontaneous spectrum and its own frequency axis, the complex susceptibility,
  the ratio, and the normalisation prefactor. A sweep keeps the layout `load_param_sweep` already
  reads, so the one function written to consume it has something to read. A comparison stores the
  common grid and the interpolated curves. `h5py` is already pinned (`requirements.txt:17`).

**Figures:** `figures/*.png`, written through **`w.figure_path(title)`** — not `w.fig_sink`
(**P48**). None of the four drawing functions ever yields a live `Figure`: each takes a `save_path`
and calls `savefig` itself, so a sink would mean rewriting all four, which is in no task's mandate.
`figure_path` delivers what this section needs — the PNG inside the record's `figures/` and its
name in the manifest — and the window still receives the figures through its watcher.

**Header blocks**, filled by the writer as for every other kind: the git revision, the versions, the
device, and the input files — the cell, the units and **the bounds file that resolved**, each by
path and SHA-256, and the model **by name**, exactly as every other kind records it
(`provenance.py:106-107` puts only `bounds`, `cell` and `units` through `file_ref`).

**Parents:** none. `_PARENT_KEYS["fdt"] = ()`. Nothing can depend on an `fdt` record and an `fdt`
record depends on nothing, so its lineage report is a single step — this record's inputs and knobs,
with no parents to follow. A comparison's `compared` ids live in the body and are resolved by
§1.2's new `render_lineage` branch.

### 2.4 The kind-addition checklist

Measured against the code on 2026-09-22. Every item is required; the ones marked **silent** fail at
run time rather than in the suite, and each gains a test here.

**Inside `core/artifacts/`:**

1. `manifest.KINDS` (`manifest.py:19`) — the validator's allow-list.
2. `manifest.BODY_KEYS` (`manifest.py:27-44`) — §2.3's keys. A kind in `KINDS` but not here raises a
   bare `KeyError` inside `validate`, not a `ManifestError`.
3. `store.KIND_DIRS` (`store.py:31-33`) — the directory map and the canonical display order.
4. `store._PARENT_KEYS` (`store.py:52-54`) — `()`. A **missing** entry silently means the same thing,
   so the entry is added explicitly.
5. `store.LoadedFdt` — a `Loaded` subclass carrying the record's body and the path to `data.h5`.
6. `store.load_fdt` — **verifies every recorded payload hash that is not null, and nothing else**
   (**P47**). A null recorded hash means "not yet", never a mismatch: a progressive record's hashes
   are written at the final commit only, so a loader that refused a null hash would refuse exactly
   the records E2 keeps the folder for. Nothing else is checked — no config match, no parent walk —
   as with `load_diagnostic` (`store.py:1056-1058`), because this kind records a measurement and
   does not constrain a later run. (This item's first draft said both "verifies the payload hash"
   and "like `load_diagnostic`", which verifies nothing.)
7. `core/artifacts/__init__.py:6-9` — the package re-export, so nothing outside the store imports the
   submodule (pinned at `tests/test_artifact_store.py:3289-3290`).
8. `store.list`'s `finished` branch (`store.py:412`) — today
   `bool(body.get("complete")) if kind == "simulation" else True`; it becomes the two kinds that
   carry `complete`.
9. `Summary` gains `study` and `points_done` / `points_planned` / `points_failed`
   (`store.py:75-105`), and `store.list` fills them.
10. **`ArtifactWriter`'s progressive mode** (§2.2) and `w.refresh()`.
11. **`store.loose_files(kind)` / `store.remove_loose(kind, filename)`** and
    **`store.legacy_dirs()` / `store.remove_legacy(name)`** against a closed literal list of legacy
    directory names — the only route E10 can travel: `_entries` iterates directories only
    (`store.py:381`, `for sub in sorted(p for p in d.iterdir() if p.is_dir())`) so a listing never
    reports a file, and `remove_incomplete` refuses a non-directory (`store.py:694-696`). Both
    removers share `RECENT_WRITE_SECONDS` and `remove_incomplete`'s samefile+realpath hardening;
    `remove_loose` can never reach a directory and `remove_legacy` can never reach a kind directory.
12. **`report.render_lineage`'s `compared` branch** (§1.2).

**Outside `core/artifacts/`:**

13. `core/gui/screens/artifact_screen.py:40-48` `KIND_LABELS` — **silent**: a kind in `KIND_DIRS`
    with no label here is a `KeyError` at `:133-135`, i.e. at window launch, and nothing pins it.
14. `core/gui/widgets/artifact_table.py:24-32` `_EXTRA_COLUMNS` and the matching `cells_for` branch
    at `:104-111` — the columns are `Study`, `Points`, `Finished` — plus §1.2's numeric sort key.
15. `core/gui/screens/artifact_screen.py:98-102` — the delete confirmation hard-codes the one kind
    that can be unfinished (`if s.kind == "simulation" and not s.finished:`). An unfinished `fdt`
    record — the whole point of E2 — would otherwise be deleted behind a prompt that says nothing
    about what is half-written.
16. `core/tool/browse.py:49` `KINDS` — a literal, restated so `--help` stays torch-free.
17. `core/tool/browse.py:56-64` `COLUMNS` and `_cells` (`:132-151`).
18. The hand-written kind counts in the tool's `--help`: the epilog prose at `browse.py:101`, the
    two "all seven" phrases at `:87` and `:94`, `list`'s help at `:442` and `sweep`'s at `:468`.
    **Silent**: nothing pins any of them.
19. `core/gui/widgets/artifact_picker.py` — see §5.4: a row predicate, and filtering on `finished`
    rather than `complete`. (The picker needs no per-kind knowledge, but it does need those two.)
20. Nothing else in the rendering path. `report.render_manifest` walks any body generically
    (`core/artifacts/report.py:147-174`), so the detail pane and `artifacts show` need no per-kind
    change.

**Tests:**

21. `tests/_fixtures.py:384-459` `build_browse_store` — a body for the new kind, or
    `tests/test_tool.py:1820-1826` and `tests/test_artifact_browser.py:86-95` fail.
22. `tests/test_artifact_store.py:152-172` `_bodies` — used by the per-kind round trip at `:184` and
    the finished-per-kind test at `:3292`. Neither is closed against `BODY_KEYS`, so absence here is
    silently uncovered; §8.2 closes it **against `BODY_KEYS` minus the simulation kind**, which that
    dict omits on purpose (`:154-155`).
23. `tests/test_worker_dispatch.py:699` — the closed set that forces the GUI columns — **and
    `:700-706`, which spells out one `columns_for("<kind>") == (…)` assertion per kind.**
24. `tests/test_tool.py:1753-1757` — pins the tool's kind list and columns against `KIND_DIRS` and
    against the GUI's, by order and membership.
25. `tests/test_artifact_browser.py:13-17,84` — derives its list from `KIND_DIRS` and follows
    automatically, except the `browse_<kind>` naming assertion at `:86-95`.
26. `tests/test_refusals.py:91` (the registry count) and `:805` (the None-flag set, asserted by
    **equality** against a six-key literal) — every field key §5.3 adds forces both.
27. **`tests/test_artifact_store.py:2540-2566`** — the `public_entry` source scan, which walks
    `CODE_ROOTS = ("core",)` (`tests/_fixtures.py:38`) and therefore `core/FDT/`, and asserts
    `found == want` against a hard-coded set of fifteen, with `_UNTOUCHED_LEGS` (`:2497-2513`) closed
    against the same names by equality and `_REFUSAL_FIELDS` (`:2521-2537`) beside it.
    `run_fdt` and `run_param_study_cli` join all three, each with its success / refusal / boom leg
    against a stubbed simulator. **This reds the fast gate the moment the first decorator lands**, so
    it is part of the same task.

**Documents:**

28. `CLAUDE.md`'s kind list and its "Every stage writes its artifact at completion" sentence, which
    the progressive mode makes false for `fdt` as well as for the cache;
    `core/tool/browse.py:18-23`'s `--store-root` docstring, which E11 changes (and whose own line
    numbers are already stale); `docs/STATE.md`'s piece-5 row.

### 2.5 Listing, and the sweep's recency guard

The browser and `artifacts list` show an `fdt` record's study, its point counts where it has them,
and whether it finished — the same treatment the training cache gets, through the `Summary.finished`
field piece 4 added.

One consequence of E2 needs saying plainly. Piece 4's leftover sweep removes only a directory
carrying **no manifest at all** (`store.NO_MANIFEST_REASON`) and refuses one whose tree was touched
in the last five minutes (`store.RECENT_WRITE_SECONDS = 300.0`). A progressive record carries a
manifest from its first moment, so **the sweep can never remove it** — finished or not. That is the
right answer, and it removes the worry the five-minute guard was carrying for long runs: an `fdt`
record in progress is protected by having a manifest, not by being recently written. It is also why
§2.2 step 3 removes the directory for a pre-spend refusal: an empty record nothing can clear would
otherwise accumulate.

---

## 3. The single-cell measurement

### 3.1 The run becomes a public entry

`run_fdt` is decorated with `core.runs.public_entry` and its signature becomes

```
run_fdt(cfg, *, skip_sanity, confirm_production, writer, seed=None) -> LoadedFdt
```

It takes the **open-but-not-entered writer**, not the store and a name, for the reason §1.2 gives:
the caller needs `writer.dir` before dispatching, and the run log only exists on the worker thread.
`run_fdt` does `with writer:` itself, so `__enter__`, every `refresh()` and `__exit__` run where
`runs.current_run_log()` is populated. It returns the record rather than `None`, so both front ends
can say what was written.

With `FDTConfig.copy_for_run()` in place (§1.2) the decorator delivers V1: the run works on a private
copy, and `cfg.omega_0` is no longer written on the caller's object.

### 3.2 Provenance

`FDTConfig` gains `sources`, the same shape `SimConfig` carries, so `provenance.inputs_from_cfg`
records the cell, the units and the resolved bounds file by path and SHA-256, and the model by name,
with no new provenance code.

Filling it is not free: `cli.parse_cell` returns a 7-tuple carrying neither the resolved bounds path
nor the units path (`core/cli.py:167-168`), and `bounds_path = resolve_bounds_for_cell(cell_file,
model)` at `:174` is local to it. `parse_cell` gains a way to report the two paths it resolved, and
the builders fill `sources` from it. On the legacy inline-bounds branch there is no bounds or units
file, which `inputs_from_cfg` already records as `None` (`provenance.py:107-110`).

The two sentences in `core/tool/fdt.py` that say these analyses "have no bounds file" (`:10`, `:56`)
are corrected with it: `parse_cell` does resolve one, and on the decoupled path "the bounds file
defines the param set + order" (`core/cli.py:156-161`), so without recording which one resolved, the
parameter set a result was measured under is not recoverable.

### 3.3 The checks, and where they live

In `cli.make_fdt_config`, before the settings object is built, through `core/refusals.py`'s rules:

| setting | rule | why |
|---|---|---|
| `n_freqs` | `require_at_least(…, 1)` | 0 yields an empty grid, an empty figure and exit 0 |
| `ensemble_M` | `require_at_least(…, 1)` | 0 raises `ZeroDivisionError` in `_pick_n_segs` |
| `freqs_per_batch` | `require_at_least(…, 1)` | 0 loops for ever; the application looks hung |
| `F0` | `require_positive` | 0 divides by zero in `lock_in_chi` |
| `dt_nd`, `T_obs_periods`, `psd_T_obs_nd` | `require_positive` | a zero step or a zero window makes the spectrum undefined |
| `burn_in_nd` | `require_at_least(…, 0)` | a zero burn-in is a well-defined setting, not a broken one; E5 forbids the over-floor |
| `freq_bounds` | both positive, **and the lower below the upper** | the one shape no existing rule covers |
| the cell file | `require_file` | a missing path today surfaces as `FileNotFoundError` |
| the model | `refuse("model", reason)` on `registry.fdt_support`'s `(ok, reason)` | already gated, as a bare `ValueError`; converted to a `Refusal` carrying `field="model"`, keeping `fdt_support`'s own per-model reason (`registry.py:77-103`). The where-the-name-came-from hint moves to `core/tool/fields.py`'s entry, because a core message may not name a flag and a source scan pins that (**P26**). **Not** `require_choice`: `fdt_support` is a predicate returning a tailored diagnostic sentence, not a list of choices |

Of these, `freq_bounds`, `burn_in_nd`, `T_obs_periods`, `dt_nd` and `psd_T_obs_nd` are not
front-end knobs (§1.2), so they are registered in `FIELDS` with `None` in both front-end tables, and a
refusal on one names the setting and offers no fix (P2, P75; corrected at pre-flight, F36).
`require_below`'s justification is §4.4's grid pair, which both front ends do expose.

**One new rule** is added to `core/refusals.py`: `require_below(key, lo, hi, …)` for an ordered pair,
message "the lower bound must be below the upper bound (got …)". It is the only rule the existing
eight do not cover.

**Thin settings warn and are recorded** (E5). Below a documented threshold — fewer than two grid
frequencies, fewer than eight trajectories — the run raises a `PreflightWarning` whose sentence goes
into `body.notices`, so the record says the answer is a quick look. The thresholds are named
constants in `config.py`. Eight is chosen because the existing end-to-end test runs at eight and must
keep passing.

### 3.4 The two pre-spend refusals

- **The normalisation constant, at the start.** `observable_noise_prefactor(cfg)` is called once at
  the top of `run_fdt`, before the spontaneous campaign, and its result carried to step 8. A cell
  missing `n` or `beta` is then refused in a second rather than after both campaigns, and the
  `FDTModelError`s it raises gain field keys.
- **The band, after the grid is built.** Once the spontaneous spectrum exists and `omega_0` is
  refined from its peak, the grid's lowest frequency is known and so is the spectrum's first real
  bin. If the grid reaches below it, the run refuses **before the driven campaign**, naming the band
  setting and the spontaneous duration that sets the resolution. This is the first moment the
  condition is knowable, and it sits before the expensive half of the run.

### 3.5 The off-grid fix (E9)

`_interp_log`'s in-range test compares against the **first strictly positive** frequency of the
spectrum's grid rather than against the clamped zero bin. Out-of-range frequencies then return NaN,
as the docstring and the science reference both already say they do, and the callers' count of
excluded points becomes honest. That count is `body.offgrid`.

§3.4 gates the lower end for a single-cell run, so the blanks that remain there come from the **upper**
end — the test is two-sided (`sanity.py:47`, `log_x_new <= log_x_old[-1]`) and the chi grid tops out
at `bounds[1]·omega_0` (`spectral.py:109-111`) against a spectrum Nyquist of `π/dt_nd`, which an
active cell can exceed. For a **sweep**, `_campaign2_ratio` (`cross_validation.py:91`) interpolates
onto a grid common to every operating point and is not behind §3.4's gate, so a point whose own
spectrum does not cover the common grid blanks at either end.

The change alters numbers for any run whose band reached below the spectrum's resolution: values that
were fabricated become blanks the run reports.

### 3.6 Nothing measurable (E4)

If every grid frequency comes back blank, the run refuses — a calm one-line refusal naming the band
and the spontaneous duration, not a crash — and its unfinished folder stays behind holding the
spontaneous spectrum, which is what diagnoses the failure. Today this case saves an empty figure and
reports success.

### 3.7 The seed (E7)

`run_fdt` takes `seed`; when none is given it draws one from the ambient source and records it. The
reseeding is confined the way `core/diagnostics/rng.py:27-36` and
`core/diagnostics/identifiability.py:259-273` confine theirs, because `torch.manual_seed` is global
and reseeds every device — a hazard the repository documents at `core/SBI/decorrelate.py:192-198`.
`--seed` and a Seed box expose it (§1.2); blank draws one.

### 3.8 The numbers (E1)

The frequency grid, the spontaneous spectrum with its own frequency axis, the complex
susceptibility, the ratio and the normalisation prefactor are written to `data.h5` inside the
record. Everything the pictures show is therefore recoverable, which is what makes §7 possible.

---

## 4. The parameter sweep

### 4.1 One record per swept parameter

`run_param_study_cli` becomes a public entry that writes **two** records, one per swept parameter,
each a complete `fdt` record with `study = "sweep"` and `body.points.param` naming the parameter.
`run_fdt_param_sweep` takes the open writer instead of an `output_path`, and enters it, as §3.1's
run does.

The study takes **one** `seed`, draws it once when none is supplied, and records the same integer on
both records; each operating point derives its own stream from it, so a point is reproducible from
the seed and its index.

The special case where the first sweep's plot is saved at the study's midpoint
(`cross_validation.py:304-337`) disappears: each sweep plots into its own record when it finishes.

### 4.2 The data file inside the record

The HDF5 is written into `w.payload("data.h5")` and, as today, written **as the run proceeds** — the
spectra in the first phase, the susceptibility and ratio added in the second. E2 is what makes that
safe: the record's folder survives an interruption, so the partial file the sweep's own docstring
calls a used property (`cross_validation.py:176-186`) is still there, and now listed and deletable
rather than invisible.

The knobs the file does not record today — `freq_bounds`, `T_obs_periods`, `dt_nd`, `burn_in_nd`, the
preset name, the device, the cell — are all in `body.settings` and the manifest's input block, so the
file no longer has to carry them.

### 4.3 Some points failed; all points failed (E4)

The second phase already catches, logs and counts per-point failures
(`cross_validation.py:262-273`); the first phase catches nothing (`:219-245`). Both phases now count,
and the counts become `body.points.done` and `body.points.failed`, visible in both listings.

All points failed is a refusal — a calm operator line naming the grid and the cell, not the
`RuntimeError` the command line currently reports as a crash — raised **after** the record's final
refresh, so the spectra the message tells the reader to look at are on disk. Because each sweep is
its own record (§4.1), an all-failed activity sweep no longer costs the temperature sweep.

Two smaller defects on the same path are fixed with it: the per-point peak log line sits outside the
try that guards the second phase, so an empty ratio raises `ValueError` from `np.nanmax` after the
first phase's whole cost (`cross_validation.py:264-285`); and the dead helper `_fdt_measure`
(`cross_validation.py:97-126`, whose `cfg.omega_0 = omega_0_emp` at `:109` is the only mutation of a
caller's settings object in the module) has no caller anywhere and is removed.

### 4.4 The checks

`cli.make_param_sweep_config` (`core/cli.py:377`) applies §3.3's rules, and additionally: each grid
is `(min, max, N)` with `N` at least 2 (the tool checks this already, `core/tool/fdt.py:73-77`; the
window checks nothing), `min` and `max` finite, and `min` below `max`. The preset remains a closed
choice in both front ends; the builder gains a `preset_name: str` keyword that both front ends pass,
because it takes `preset` as an already-resolved dict and the call site drops the name
(`core/tool/fdt.py:151-153`) — and `body.settings["preset"]` must hold the name.

---

## 5. The five screens

The FDT, CrossVal, Simulate and Reduction panels, **and the model-builder screen**.

### 5.1 The refusal treatment, and retiring the generic box

All five stop wrapping their builder in a broad `except` that ends at `BasePanel._config_error` with
"The configuration could not be built." (`base_panel.py:398-416`). A `Refusal` from a builder goes to
`BasePanel._refusal` — the yellow "Check your inputs" box naming the setting and the fix, as the
inference tabs have had since piece 3 — and a bare exception goes to the red box with its traceback.
The model builder is a `QWidget`, not a `BasePanel`, so its refusal goes through
`core/gui/widgets/refusal_box.show_refusal` (`refusal_box.py:24`).

`_config_error` is then uncalled and is removed, which is what piece 3's spec said piece 5 would do
(`2026-09-15-validation-and-logging-design.md:322-323`).

**Corrected at planning time (P4).** This section's first draft said `cli.parse_cell` raises two
bare `ValueError`s that must be converted. It does not: `grep -n "raise ValueError" core/cli.py`
returns nothing, and both mistakes `base_panel.py:403-407` names by hand already raise
`Refusal(field="cell")` — piece 3 converted them. They therefore reach the yellow box the moment
this section's routing lands, and nothing needs converting. What the conversion was there to buy
is still owed, and is delivered directly: `core.refusals.missing_values_phrase(label, missing)`,
the one wording §6.2 needs, placed in `core/refusals.py` because `core/sim_config.py` cannot
import `core/cli.py`.

**The Simulate panel needs the care the Reduction panel was given (P29).** Its builder raises two
bare `ValueError`s (`simulate_runner.py:50-55`, `:68-72`) for the two most plausible cell and model
mistakes on that screen, so removing `_config_error` without converting them moves both to the red
crash box — the regression this section exists to prevent, on a panel it did not name. They become
`Refusal(field="cell")` and `Refusal(field="model")`; both keys already exist.

`tests/test_nav_and_gating.py:1688-1733` pins today's unconverted behaviour in a loop over all four
panels and both exception shapes; it is rewritten rather than extended, because a half-converted
state cannot be expressed in it.

### 5.2 The fix-hint table widened to screens (E6)

`core/gui/fields.py`'s tuple entry is `(tab, label)` where the first element must be one of the six
inference tab titles, pinned at `tests/test_nav_and_gating.py:1481-1500`. It widens so an entry may
name a **place** that is either an inference tab or a screen, and `fix_sentence` says "on the … tab"
or "on the … screen" accordingly. As today, a place may be a tuple of several — which is how
`num_runs` and `run_size_cap` already span the Posterior and TSNPE tabs.

The inputs that collide list every place they appear, so a bad cell chosen on the measurement
screen no longer sends the owner to the Infer tab. **They are `cell`, `model` and `t_obs`, not
`units` (P32, P33):** there is exactly one units control in the application, and on the four other
screens the units file is resolved from the model with no control at all, so widening it would
name a box that does not exist; the Simulate panel's observation length, by contrast, is the
existing `t_obs` key and does collide. The
sentence-shaped entry stays available, but it is no longer the only option for a non-tab surface, so
`label(key)` keeps working for the new screens — which is what stops a box label and its hint
sentence drifting apart.

### 5.3 The new field keys

`core/refusals.FIELDS` gains one key per new **front-end** setting: `n_freqs`, `ensemble_m`,
`freqs_per_batch`, `f0`, `preset`, `s_grid`, `t_grid`, `seed`, the Simulate and Reduction panels'
own numeric fields, and the model builder's (the parameter row's value, minimum and maximum, the
initial condition, `x_scale`, `t_scale`, and each forcing field). Each gets a neutral description and
a default clause in `core/refusals.py`, a place entry in `core/gui/fields.py`, and a flag entry in
`core/tool/fields.py`. The five non-knob settings of §1.2 are registered here too, with `None` in
both tables (**P2**).

`tests/test_refusals.py:91` hard-codes the registry's size and `:805` asserts the None-flag set by
**equality** against a six-key literal, so both move with this section; §8.2 replaces the equality
with a subset assertion and a reason, so the next piece to add a window-only knob does not face a
false red.

### 5.4 The pickers, and showing an earlier run (E1)

The FDT and CrossVal screens each gain a `StorePicker` over the `fdt` kind, filtered to their own
study.

`StorePicker` is generic over a kind string and every fact it shows is a `Summary` field with a
`None` default (`artifact_picker.py:139-200`), so a new kind needs no new widget — but **two
additions are needed, and checklist item 19 names them**:

- `refresh` lists every row of the kind with no filter hook (`:179-181`), so `StorePicker` gains an
  optional row predicate for the study filter: `row_filter: Callable[[Summary], bool] | None`,
  applied **after** the `finished` rule and placed before `parent` so existing positional calls are
  unaffected (**P9**).
- It filters on `Summary.complete`, which means "has a valid manifest", **not** "the run finished"
  (`store.py:76-86`). For this kind `complete` is true from a progressive record's first moment, so
  the picker must filter on `Summary.finished`, or it would silently offer half-written records as
  results.

Selecting a record shows its cell, settings, seed and notices, and re-opens its figures from the
record's `figures/`.

**The record is created before the run is dispatched.** The panel calls
`store.create("fdt", cfg, name=…, note=…)` — which mints the id, runs `assert_name_free` before
anything is spent, and fills `writer.dir` without creating the directory — sets the first body, and
hands the writer to the run, which enters it on the worker thread (§1.2, §3.1). The figure watcher is
pointed at `<record>/figures`, a directory whose name is therefore known before the run starts. The
watcher globs one directory and does not recurse (`core/gui/plot_watcher.py:61-64`), which pointing
it at `figures/` satisfies.

### 5.5 What each screen remembers

Unchanged in kind from piece 3's V5, which `CLAUDE.md` already records: these panels keep remembering
their own numeric fields, and consents are never persisted. The new pickers remember their selection,
as the inference pickers do. **The seed is not remembered** — a remembered seed would silently turn
every run into a repeat of the last one.

### 5.6 Simulate and the model builder

Round-trip tests that drive a model through the builder and back, and an export through Simulate and
back. **A defect a round trip exposes is fixed here only if it is a refusal defect or a data-loss
defect**; anything else is recorded in §1.3 and handed on, so the mandate is bounded (**P58**).

Their input checking is converted with the other three (§5.1): the Simulate panel today dispatches
`max(1, frame_steps.value())` and `float(max(1, fps.value()))` — silent clamps — and passes its
observation length unchecked (`simulate_panel.py:105-118`). The clamps are replaced by refusals that
name the box.

---

## 6. The command line

### 6.1 The destination flag, and the three behaviours it flips (E11)

`core/tool/__init__.py:114-116` decides **three** things from whether a subcommand declares
`--store-root` — as `core/tool/browse.py:20-23` itself says ("``main`` keys three of smoke's
behaviours on the FLAG's presence"):

- an unset flag sends the run to a fresh `tempfile.mkdtemp` (`:116`) — replaced by an explicit
  `args.temp_store_root`, set by `smoke.register` alone (**P10**);
- `auto_root = has_store_root and not args.store_root` (`:115`) removes an auto-created root that is
  still empty after a failure (`:183-184`, `_remove_if_still_empty(root)`);
- `:146` selects the Ctrl-C advice — already bypassed for these two, which set `interrupt_note`
  through `set_defaults` (`core/tool/fdt.py:118,129`) and take `main`'s `note is not None` branch at
  `:142-143` first.

`fdt` and `crossval` declare the flag, and all three are reworked for them:

- **No throwaway root.** With no `--store-root`, these two follow `artifacts_root()` — the
  environment — as they do today and as every screen does. The `mkdtemp` branch stays `smoke`'s
  alone; the dispatch becomes an explicit per-subcommand property rather than `hasattr`.
- **No auto-root cleanup.** `auto_root` stays `smoke`'s alone. With an unset flag these two must not
  set it, or a failed run would call `rmdir` on the operator's real `Artifacts/` root. (`rmdir`
  refuses a non-empty directory, so nothing is lost — but this is exactly the accidental flip E11
  exists to prevent.)
- **Their own interrupt advice**, which already bypasses the flag branch, is rewritten for what now
  survives an interrupt: a named, listed, deletable unfinished record, not loose files under a
  directory, and there is no resume.

### 6.2 Refusals, and the unsupported-model hint

Both subcommands' refusals become `Refusal`s with field keys, so they print as the tool's one
operator line naming the flag rather than as a bare `ValueError` reported with its class name and
raise site (`core/tool/fdt.py:132-139`). The unsupported-model hint keeps both branches — a plain
cell file and a cell folder — and §8.2 covers the folder branch, one of the two gaps `docs/STATE.md`
names.

`load_and_validate_gt`'s `Refusal(field="cell")` and the dry run's problem strings stop being two
wordings for one rule: both go through `core.refusals.missing_values_phrase` (**P4**), closing the
item piece 3 handed on (`2026-09-15-validation-and-logging-design.md:732`). **A third wording of
the same rule, in `cli._merge_vals_bounds`, is knowingly left alone (P53)**: it carries the cell
*path*, which the FDT builders need and the other two do not, so folding it in would either drop
that or change the other two. It is handed on in §1.3.

### 6.3 The tidy-up command and the legacy files (E10)

`artifacts sweep` today offers only directories inside a kind directory that carry no
`manifest.json` at all. It gains a second, clearly separated category, over §2.4 item 11's new store
calls: **loose files** inside a kind directory, and a **legacy directory** beside the kind
directories where one exists. The legacy directory is offered only by the all-kinds `sweep` — the
form with no `<kind>` argument — because `sweep` takes the kind positionally
(`core/tool/browse.py:467-468`) and a legacy directory is under no kind. The Artifacts screen's sweep
gains the same category, so the two front ends stay in step, which is the property piece 4
established.

As today, the preview names each item, the removal is bound to the list the confirmation showed, a
dry run is the default unless `--yes` is passed, and the recency guard applies. Nothing is removed on
the owner's behalf, and a valid record's payload files are never offered.

---

## 7. Comparing (E8)

This is the last part of the piece (E12); nothing before it depends on it.

### 7.1 What it draws

One new subcommand with four modes, mirroring `identifiability`'s shape:

| mode | draws |
|---|---|
| `compare cells` | the ratio curves of two or more single-cell records on one axis, labelled by cell |
| `compare repeats` | several records of the same cell, with the spread across them shown as a band |
| `compare renormalise` | one record's ratio recomputed with a normalisation constant supplied on the command line, drawn against the original |
| `compare sweeps` | two sweep records together, and a slice of both at a common operating point |

The FDT screen carries the first three over its picker; the CrossVal screen carries the fourth.
Because `StorePicker` is single-selection, **the comparison controls hold a list the picker appends
to** — no multi-select widget is built.

A comparison draws only records whose `complete` is true; an unfinished one is refused, naming it.

### 7.2 The common grid

Every run detects its own resonance and builds its grid around it, so two runs of the same cell land
on different frequencies. Curves from different records are interpolated onto a common grid: **log-
spaced over the intersection of their spans, with N the smallest `n_freqs` among the records**. A
target point whose bracketing samples in a source record include a blank is blank in that record's
interpolated curve — blanks are never interpolated across. The drawing says so, in the axis label and
in the record's notices.

### 7.3 A comparison is a record

A comparison writes an `fdt` record with `study = "comparison"`, whose `body.compared` names the mode
and the ids and names of the records it drew, whose figures are its output, and whose `data.h5` holds
the common grid and the interpolated curves. It is listed, noted and deleted like any other.

Deleting a record a comparison used is **not** refused; `report.render_lineage`'s new branch (§1.2,
checklist item 12) resolves each `compared` id and prints `MISSING <kind> [<id>]` for one the store no
longer holds.

---

## 8. Tests

### 8.1 Fixtures

- `tests/_fixtures.py` gains `build_fdt_record(store, *, study, finished=True, …)` — a tiny `fdt`
  record with a real manifest, a small `data.h5` and one figure — and `build_browse_store` gains an
  `fdt` body (checklist 21).
- A tiny real single-cell run at the smallest legal size, `slow`-marked, for the end-to-end tests.
- `tests/test_artifact_store.py:152-172` `_bodies` gains the kind, and §8.2 closes that dict against
  `BODY_KEYS` **minus the simulation kind** (checklist 22).

### 8.2 New tests, by area (names indicative)

**The kind and the progressive writer**

- an `fdt` record round-trips through create, list and get, like every other kind
- **the first manifest on disk validates** — every body key present, five of them null
- a failure inside an `fdt` record **keeps** the directory, with `complete` false and the log written
- a `Refusal` raised before the first payload or figure **removes** the directory (§2.2 step 3)
- a failure inside each of the six ordinary kinds still **removes** the directory — the pin that the
  new mode changed nothing for them
- an unfinished `fdt` record is listed as unfinished in both front ends, and the leftover sweep never
  offers it (it carries a manifest)
- the delete confirmation names what is half-written for an unfinished `fdt` record (checklist 15)
- `KIND_LABELS` is closed against `KIND_DIRS` — checklist 13, silent
- every hand-written kind count in the rendered `--help` names the new kind — checklist 18, silent;
  the test scans the rendered help, not the epilog constant
- `_bodies` is closed against `BODY_KEYS` minus the simulation kind — checklist 22
- the `Points` column sorts numerically in both directions (§1.2)

**The run entries**

- `run_fdt` leaves the caller's `cfg.omega_0` at its entry value on success, on a refusal and on a
  boom — V1, which `copy_for_run` is what delivers
- the `public_entry` source scan and its two companion sets include both new entries, each with its
  three legs — checklist 27
- a record's `log.txt` holds the run's records, written from the worker thread

**The checks**

- each floor in §3.3 refuses, at the click and at the flag, naming its setting: shown failing before
  the change with the behaviour it prevents (the batch planner's unbounded loop is asserted with a
  bounded call count, not by running it)
- a thin setting warns, completes, and the sentence is in `body.notices`
- a cell missing the normalisation constant is refused **before** the spontaneous campaign — asserted
  by the simulator never being called
- a band below the spectrum's resolution is refused **before** the driven campaign — asserted the
  same way
- the unsupported-model refusal keeps `fdt_support`'s own reason and carries `field="model"`
- `require_below` refuses an inverted pair and accepts an ordered one
- §4.4's grid checks: a one-point grid and a non-finite bound each refuse, naming the grid field

**The off-grid fix**

- a probe outside the spectrum's positive span returns a blank, not a blended value — the empirical
  demonstration that it does **not** today
- the reported count of blanks matches the number of blanks

**Failures**

- a sweep with some points failed completes, and the counts are in the body and in both listings
- a sweep with every point failed refuses, the message names the grid, and the record is on disk
  holding the spectra
- an activity sweep that fails entirely does not prevent the temperature sweep — the two-record split
- a single-cell run with every frequency off-grid refuses rather than saving an empty figure

**Provenance and seeds**

- a record names its cell, units and resolved bounds file each by path and hash, and its model by
  name
- two runs with the same seed agree; two runs with different seeds differ; the seed is recorded
- a study's two records carry the same seed, and a point is reproducible from the seed and its index
- the seed is not remembered between launches

**The screens**

- each of the five surfaces shows a builder `Refusal` in the yellow box naming the setting, and a bug
  in the red box — replacing `tests/test_nav_and_gating.py:1688-1733`; the model builder's goes
  through `show_refusal`
- `_config_error` has no callers and is gone
- the Reduction panel's two cell mistakes still reach the yellow box (§5.1)
- the widened table: a screen entry produces "on the … screen", a shared input names every place, and
  `label(key)` still works for every box entry
- the picker filters to its own study and offers only finished records
- the picker remembers its selection across a relaunch
- the figure watcher points at the record's `figures/` and the panel names the record it wrote
- the reduction builder still builds a working settings object (§1.3)

**The command line**

- `--store-root` is honoured; without it these two follow the environment and **not** a temp root
- an `fdt` run exiting non-zero with no `--store-root` leaves `artifacts_root()` in place
- the interrupt note names the unfinished record
- refusals print as one operator line with the flag
- **the cell-folder branch of the unsupported-model hint** — `docs/STATE.md`'s first named gap
- the cell refusal's wording is identical from `load_and_validate_gt` and from the dry run (§6.2)

**The tidy-up (E10)**

- a loose PNG inside a kind directory is previewed and removed; a recently written one is refused
- a dry run removes nothing; the removal is bound to the previewed list
- a valid record's payload files are never offered
- a legacy directory is offered by the all-kinds sweep only
- the Artifacts screen offers the same categories as the tool

**The drawing functions**

- **tests only (P17).** All four functions already close a saved figure — `bb22ac4` touched all
  four — so what is missing is three TESTS, not three behaviours; this bullet's first draft said
  otherwise. Each new test asserts the same three properties the existing one does (the figure is
  closed, the interactive path is not taken, no "non-interactive" warning), and the task proves its
  tests have teeth by deleting one `plt.close(fig)`, watching the test fail, and restoring it.

**The Nadrowski sanity path** (§1.2)

- a `slow`-marked end-to-end run on a Nadrowski cell **with** the sanity checks, covering the
  Nadrowski-only checks and the passive-baseline plot that no test reaches today

**Round trips**

- a model through the builder and back; an export through Simulate and back

**Comparing**

- each of the four modes draws and writes a record
- the common grid is the log-spaced intersection at the smallest `n_freqs`, and blanks are not
  interpolated across
- an unfinished record is refused, naming it
- a comparison whose source record was deleted still lists, and its lineage prints `MISSING`

### 8.3 Tests that change

- `tests/test_tool.py:1344-1378` `test_fdt_and_crossval_run_at_tiny_size` — its assertions glob
  `artifacts_root()/fdt` and `/crossval` for PNGs and `.h5` files, which is exactly what this piece
  moves. Rewritten to assert the records, their payloads and their figures. Its docstring's 598 s
  figure is stale against every gate since (243, 208, 229 s) and is corrected from measurement.
- `tests/test_artifact_store.py:2540-2566` — the `public_entry` scan and its two companion sets
  (checklist 27).
- `tests/test_nav_and_gating.py:1688-1733` — rewritten (§5.1).
- `tests/test_nav_and_gating.py:1481-1500` — widened for the place shape (§5.2).
- `tests/test_refusals.py:91,788-790,805` — the count, and the None-flag equality replaced by a subset
  assertion with its reason (§5.3).
- `tests/test_artifact_browser.py:864` — `sorted(seen) == [...]` becomes a subset assertion, the
  carry-forward `docs/STATE.md` already records: this piece adds pickers, which is exactly the false
  red it predicted.
- `tests/test_worker_dispatch.py:699-706` — the closed set and the per-kind column assertions.

### 8.4 Count and budget

Piece 4 added 129 tests (603 → 732) against a stated band of 130 ± 30. This piece is larger in
surface: **about 170 ± 40 added**, for roughly 900 collected. Every test that runs a real campaign is
`slow`-marked; the unit-level tests stub the simulator, as the existing fast FDT tests do.

---

## 9. Gates

- **The one-process fast gate** (`pytest -m "not slow"`) after every task, run by the controller and
  never by a subagent, with no source edited while it runs. **Target: 16 minutes**, unchanged from
  pieces 3 and 4 — which is what forces the campaign-running tests into the slow set.
- **The slow set** (`pytest -m slow`) before the piece is called done. Its budget rises from ~31
  minutes to **about 45**, with the new Nadrowski end-to-end run and the rewritten tiny-size test.
- **The GPU smoke gate** (`CLAUDE.md`'s four command lines) if any line that creates or moves a tensor
  changes. This path is pinned to the CPU (`cli.make_fdt_config` forces `cpu_device()`), so the
  expectation is that only the seed work qualifies; the judgement is made against the diff at the end
  and **recorded either way**, as pieces 3 and 4 recorded theirs.
- **The display walkthrough** — §10, run by the owner.

---

## 10. The display walkthrough

Rows 15, 16 and B8 of `docs/checklists/display-walkthrough.md` are marked **superseded** for the
reasons §1.2 gives, and this piece writes a new **E-row** section covering: a run's record appearing
in the browser while it runs; a cancelled run leaving an unfinished record; a bad box refused in the
yellow box naming the box and the screen; the two pre-spend refusals; a sweep with some points failed
and one with all of them failed; the picker showing an earlier run and re-opening its figures; each
of the four comparison modes; the tidy-up command offering the legacy files; and B8's carried half,
that no library line appears without its prefix during a full run.

No task fills those rows in. They are the owner's, on a real screen.

---

## 11. Risks, and the order of work

| risk | mitigation |
|---|---|
| The kind-addition checklist has three silent members (items 13, 15, 18) — a miss crashes the window at launch or prints a wrong help page, and no test catches any of them | All three gain a pin in the same task that adds the kind (§8.2) |
| The progressive mode touches `ArtifactWriter`, which every kind uses, and piece 4's review caught a delete that destroyed a finished artifact | The six ordinary kinds' behaviour is pinned by a test written **before** the mode exists, and the mode is opt-in per kind, never a default |
| The `public_entry` source scan reds the fast gate the moment the first decorator lands | Checklist item 27 makes the scan and its three legs part of that same task, not a later one |
| The fast gate is already at 13 min 57 s of a 16-minute target, and this piece adds ~170 tests | Every campaign-running test is `slow`-marked; the gate is measured after every task and the trend watched, as piece 4 watched its own |
| `FDTConfig` is shared with the reduction map, which is out of scope and is the only `FDTConfig` not pinned to the CPU | Every addition is a defaulted field or a method; no `__post_init__`; `make_reduction_config` is not touched; a test pins that the reduction builder still works |
| The comparison facility is new capability and can grow without limit | It is last (E12), it is four named modes and no more, and every point before it is a coherent stopping place |
| The plan's quoted line numbers go stale, as they did in pieces 3 and 4 | Quoted **text** is the anchor (the rule is stated at the head of this document), and a pre-flight scan of the plan against the real code runs before task 1 — it found six blocking conflicts in piece 4 |

**Order of work:** the store kind, the progressive writer and the three silent pins; the settings
object's `sources`, `seed` and `copy_for_run`, with the `public_entry` scan; the checks and the new
rule; the two pre-spend refusals and the off-grid fix; the single-cell run as a public entry writing a
record; the sweep, split into two records, with its failure counts; the five surfaces' refusals and
the widened table; the pickers and the watcher; the command line's flag, notes and refusals; the
tidy-up command; the two round trips; the test gaps; **then** the comparison facility; then the
documents, the gates of record and the walkthrough rows.

**Documents edited in place at the end**, each with a pointer to the section that changed it:
`CLAUDE.md` (the kind list and the "writes its artifact at completion" sentence),
`core/tool/browse.py:18-23` (the `--store-root` docstring),
`core/tool/fdt.py:10,56` (the "no bounds file" sentences), and
`docs/superpowers/specs/2026-09-11-one-flow-design.md:425` (whose stated reason for having no store
flag on `fdt`/`crossval` — that their outputs are not store artifacts — stops being true; corrected
in place with a pointer to §6.1).

---

## 12. Deviations (filled during execution)

Every ruling that differs from what is written above goes here, with what it costs if it is wrong —
the practice pieces 2, 3 and 4 followed. Rows 1–10 were ruled at PLANNING time, before any code was
written: rows 1–9 when ten drafters read the code for their tasks and raised 69 objections (rulings
**P1–P69** in the plan), row 10 when three lenses and a judge read the assembled plan (**P72**, one
of P70–P82). The full texts are in the ledger. Rows from 11 on were written at the end of the piece,
from the whole ledger and the whole-piece review, and each names the ruling it rests on: a
**P**-number is a planning ruling that departs from this document but was not recorded here at
planning time (rows 11, 12 and 21), an **F**-number a pre-flight ruling (made against the real code after the plan and before
the first task), a **T**-number the task during which it was ruled, and **M**/**N** with **R-F** the
whole-piece review's fix dispatch (rows 27–35, `final-review-synthesis.md` and `final-fix-brief.md`
in the ledger's folder).

Ten further rulings corrected this document rather than deviating from it, and are applied inline
above with their P-numbers: **P2** (the five non-knob settings are registered, with `None` in both
tables), **P4** (`parse_cell` already refuses; the message builder is delivered directly), **P9** and
**P10** (two names the document left unfixed), **P17** (the drawing functions are a tests-only item),
**P26** (a core message may not name a flag), **P32** (`units` does not collide; `t_obs` does),
**P47** (what `load_fdt` verifies), **P48** (`figure_path`, not `fig_sink`) and **P53** (a third
wording is knowingly left alone). Two more were applied the same way later: **F36** (at pre-flight,
§3.3's sentence brought into line with P2 and P75) and **P58** (at the end of the piece, §5.6 hands a
round-trip defect on through §1.3, not through this table).

| # | where | deviation | why |
|---|---|---|---|
| 1 | §1.2, §3.1, §4.1 (P3) | T17 and T19 each carry the bridging edits to their own call sites, in the same commit as the signature change | A required `writer` keyword with the call sites left behind reds the fast gate, which the plan forbids after any task. The bridging shape is this document's own, so nothing is undone later. *Cost: T25/T26 find the writer creation already there.* |
| 2 | §4.1, §2.4 item 27 (P27) | `run_param_study_cli` gets §3.4's normalisation check at its top, and `_REFUSAL_FIELDS["run_param_study_cli"] = "cell"` | The `public_entry` scan's three sets are closed by equality and demand a refusal leg; this document names no pre-spend refusal for the study. The justification is §3.4's own. *Cost: one extra cheap check per study.* |
| 3 | §4.2 (P28) | `plot_fdt_3d_vs_param` takes its destination instead of building one under `<artifacts root>/crossval` | Otherwise the sweep keeps writing into the very directory E10 offers for tidy-up, and a "legacy" directory refills itself. *Cost if wrong: a figure lands outside its record and the tidy-up offers it.* |
| 4 | §7.1, §7.3 (P11) | The comparison facility is ONE public entry with per-mode drawers, and the whole subcommand parser lands in the first of its five tasks | Forced twice over: the tool-table pin walks the real parser, so a flag no subcommand defines fails immediately; and four entries would mean four tasks editing the scan's three equality-closed sets. *Cost: that task is larger than its neighbours.* |
| 5 | §5.4 (P30, P31) | Both screens gain a "Record name" and a "Note" box, neither persisted; a two-record study's name is a STEM, and its records are `<stem>-s` and `<stem>-temp` | §5.4 mandates `store.create(name=…)` and Review Focus item 2 turns on `assert_name_free`, but no section gave these screens a name control. Two records cannot share one name. *Cost: two controls to remove.* |
| 6 | §3.6 (P22) | `plot_psd` moves to just before Campaign 2 | §3.6 promises the unfinished folder holds the spontaneous spectrum; today that figure is drawn after both refusals could fire, so the promise was unreachable and the folder would hold only a time series. *Cost: one figure drawn earlier; no numbers change.* |
| 7 | §3.1 (P55) | `run_fdt` on the `confirm_production=False` branch returns its record, finished, with `grid` and `offgrid` null | This document never said what that branch returns, and it is the cheap way to reach the sanity checks. *Cost: a record that says the sanity checks ran and nothing else did.* |
| 8 | §7.1 (P60) | `compare repeats` does not enforce "the same cell"; it reports the cells it drew and refuses only on arity | A record's cell lives in the manifest's inputs block and nothing stops a caller passing two. *Cost if wrong: a misleading band — visible, because the drawing names the cells.* |
| 9 | §3.5, §8.2 (P23) | `check_passive_baseline`'s bare `ValueError`, newly reachable from the low end once the off-grid fix lands, is converted only if the new Nadrowski test reds on it | Converting it needs a field-key decision §3.6 does not supply. *Cost if wrong: one test fails loudly in the piece that caused it.* |
| 10 | §1.2 ("`FDTConfig` gains two fields and one method"), §4.4 (P72, planning time) | `FDTConfig` gains a third defaulted field, `preset_name: str \| None = None`; `make_param_sweep_config` passes it into the construction and the sweep reads `cfg.preset_name` for `body.settings["preset"]` | §4.4 requires the preset NAME in the record, and the builder's keyword alone carried it nowhere, so `settings["preset"]` would always have been null. A defaulted field, so §1.3's reduction-map constraint holds. *Cost if wrong: one more field on a shared dataclass.* |
| 11 | §1.2 ("Walkthrough rows 15, 16 and B8 are marked superseded"), §10 (P42, planning) | Rows 15, 16 and B8 of `docs/checklists/display-walkthrough.md` are not marked in place; the new E-row section's preamble states that they are superseded, and why, and names E8, E9 and E16 as their replacements (with walkthrough C2, whose quoted sentence the widened fix-hint table changed) | An older row is a dated record of what was seen on the code as it then was, and a later piece editing it would rewrite that record. *Cost if wrong: a reader who opens rows 15, 16 or B8 alone, without the piece-5 preamble, does not learn they no longer describe the application.* |
| 12 | §3.2 ("`parse_cell` gains a way to report the two paths it resolved"; the plan's interface contract, planning; F31, T7) | `cli.parse_cell` keeps its 7-tuple unchanged, and a sibling, `cli.cell_sources(cell, model=…)`, re-derives the resolved bounds and units paths the builders record in `sources`; its `model` argument keeps a default, as `parse_cell`'s does | Every other caller of `parse_cell` stays untouched. *Cost if wrong: two resolutions of the same paths that could drift apart; the fix dispatch's N28 guard test pins that they name the same files for every shipped cell `parse_cell` accepts.* |
| 13 | §3.3 (`burn_in_nd` with `require_at_least(…, 0)`; F33, T10) | `burn_in_nd` is checked with `require_finite` and then a CLOSED `require_between(…, 0, inf)` | `require_at_least` coerces with `int()`, so -0.5 would pass as 0 (the silent clamp V2 forbids) and NaN would raise a bare `ValueError`; the closed form refuses NaN, ±inf and every negative and accepts 0, and F33's literal `open_hi=True` would print "(exclusive)" beside a legal 0. *Cost if wrong: the refusal reads "between 0 and inf".* |
| 14 | §3.3 (the model row: "the where-the-name-came-from hint moves to `core/tool/fields.py`'s entry"), §6.2 | The hint ("--model named it" or "the cell's parent folder named it; pass --model to override that") is built in `core/tool/fdt.py`'s handler, in a `fdt_support` pre-check ahead of the builder, and raised as `Refusal(field="model")`; `core/tool/fields.py` maps `model` to `--model` alone | A static table entry cannot know which of the two sources named the model, and those are two different mistakes; only the handler knows. *Cost if wrong: `fdt_support` runs twice on the tool's path (the handler, then the builder), and the hint lives outside the one table that names flags.* |
| 15 | §1.2 ("The block records the settings as given to the builder"; T17) | The manifest's config block records the RESOLVED seed — the drawn one when none was supplied — set beside `body["seed"]` before the writer is entered | Every stage updates its resolved knobs onto `writer.config`, and a lineage report that printed "(none)" for the seed E7 records would contradict the body. *Cost if wrong: the config block shows a seed the operator never typed, so it no longer reads as the settings exactly as given.* |
| 16 | §3.4 ("naming the band setting and the spontaneous duration that sets the resolution"; T14, T15) | The band refusal and the passive-baseline warning name the Welch SEGMENT (`nperseg = min(2**14, n_obs)`) as what sets the lowest resolved frequency, and offer "lengthen the recording" only while the recording is below the segment cap; above it they say lengthening would not help. The high-frequency check's warning no longer offers it at all: the spectrum's top is π/dt_nd, which no recording length moves. The shipped Hopf cell is therefore refused at the default band (its peak ≈ 0.268 ND puts the lowest probe below the fixed first bin, 0.0383 ND), accepted as E9's consequence and left for the owner | §3.4's premise is false above the cap: 200, 4000, 8000 and 16000 ND all give the same first bin, so the old advice sent the operator to double a spontaneous campaign for an identical band. *Cost if wrong: one conditional clause per message; and the Hopf cell needs a deliberate `freq_bounds` edit, which no front end exposes, before it can be FDT-analysed.* |
| 17 | §2.3 (`offgrid` null "until the driven campaign finishes"), §3.6 (T16) | The nothing-measurable check and `body.offgrid` move ABOVE Campaign 2, directly below the band refusal; a single-cell record carries `offgrid` from before the driven campaign starts | The refuse-before-the-spend constraint: whether any probe is measurable depends only on the grid and Campaign 1's spectrum, and the result was shown to be bitwise identical. *Cost if wrong: none measured; an interrupted Campaign 2 leaves a record whose `offgrid` is filled though the ratio was never measured.* |
| 18 | §3.4 ("the two pre-spend refusals"; T16) | A THIRD pre-spend refusal: a spontaneous spectrum with no finite value at any positive frequency is refused right after Campaign 1, before the peak search, as a diverged simulation (field `cell`); its folder keeps the spontaneous-trajectory figure and no spectrum figure | The peak search returns the first bin on an all-NaN spectrum, so every later sentence — the band advice, "the band it resolves" — would describe a band that does not exist. *Cost if wrong: a new refusal sentence, and a diverged record has one figure fewer.* Extended to the sweep by row 27. |
| 19 | §3.7 (reseeding "confined the way `core/diagnostics/rng.py` … confine theirs"; T17) | `seeded` MOVES to a new light module `core/rng.py` (`core/diagnostics/rng.py` re-exports the same object), and on a CPU device it seeds ONLY the CPU generator (`torch.default_generator` inside `fork_rng(devices=[])`) and numpy; on a CUDA device it is unchanged. This changes smoke's and the diagnostics' CPU seeding too | `torch.manual_seed` on a CPU run clobbers every CUDA stream, which §3.7's confinement forbids, and importing the diagnostics package from an FDT run pulled in the SBI stack. A CPU run draws only from the CPU generator, so no CPU result changes. *Cost if wrong: a CPU diagnostic that silently relied on CUDA being reseeded would lose that (none known).* |
| 20 | §4.1 ("a point is reproducible from the seed and its index"; T19→T20) | Each operating point's stream is `SeedSequence([seed, sweep, phase, index])`, not `seed + index` / `seed + n_points + index`; a point is reproducible from the study's seed, its sweep, its phase and its index | Under the old rule the S and T sweeps' point k drew the same Phase-A stream (point 0 of both was bit-identical) and one sweep's Phase B could collide with the other's Phase A — a hidden correlation `compare sweeps` would read as signal. *Cost if wrong: none; every point's draws changed once, before any record of this shape existed.* |
| 21 | §4.3 ("All points failed is a refusal — a calm operator line naming the grid and the cell"), E4 (P77, planning; F14/F45, T20; amended by N9) | Inside a study, a sweep that measured nothing is logged at error, its record kept unfinished, and the other sweep still runs; the study returns only the FINISHED records, so the tool exits 0 and the window shows a pane line, not a yellow box. The study refuses only when BOTH sweeps measured nothing, with ONE refusal (field `s_grid`) whose sentence names both grids | An all-failed S sweep must not cost the T sweep (§4.1's reason for two records), and a refusal naming one grid sent the operator to the S grid alone when the T_a/T grid had failed too. *Cost if wrong: a one-sided failure is an error line that names no box or flag, which qualifies E4's "refuses, naming the setting".* |
| 22 | §5.4 ("Selecting a record … re-opens its figures from the record's `figures/`"; F47, T27) | Figures re-open only on the user's own pick (the picker combo's `activated` signal); a selection restored at launch, or moved programmatically after a run, fills the summary line only | Re-opening an old run's pictures on every launch is behaviour nobody asked for, and after a run the figure watcher already shows that run's figures. *Cost if wrong: after a relaunch the owner picks the run once more to see its figures.* |
| 23 | §6.1 (their own interrupt advice), §8.2 ("the interrupt note names the unfinished record"; F20, T29; amended by N1 and N24) | A note is fixed text and cannot carry an id, so each handler prints `writing record <id> at <dir>` when it creates the record, before anything is spent, and the note points back at that line (and says to set `PRISM_ARTIFACTS` to a `--store-root` for the `artifacts` commands); both run entries also log `Writing fdt record <id> at <dir>` as their first record inside the writer, so the window's pane and the record's `log.txt` name it too; the notes are hedged for a run stopped before its folder existed | §8.2's promise is unreachable for a static note, and the `artifacts` family reads only the environment. *Cost if wrong: one framing line per record.* |
| 24 | §7.1 (`compare sweeps`: "a slice of both at a common operating point"; T39) | The slice takes each sweep's nearest finished row to the requested point (a stated tie rule on rounded distances), draws it on the ABSOLUTE ND frequency axis with each curve's `omega_0` its row's own resonance, and records a notice naming the rows compared when they differ from the request or from each other; the side-by-side surfaces keep each sweep's own normalised axis | Normalising by each sweep's own reference made one physical measurement look like two disagreeing curves (driven: identical physics differed by up to 0.80 in T_eff/T), and two grids need not share the requested point exactly. *Cost if wrong: the slice's x axis is ND frequency rather than a normalised one (the legend names each row's parameter value).* |
| 25 | §5.4 ("The figure watcher is pointed at `<record>/figures`"; F18, T26, against P35's "one watcher re-pointed") | The sweep screen runs ONE WATCHER PER RECORD, one on each of the study's two `figures/` folders | It delivers P35's purpose — both sweeps' figures reach the panel — with no new worker-to-window signal, which no interface provided. *Cost if wrong: two pollers instead of one during a study.* |
| 26 | §2.3 ("written through `w.figure_path(title)` — not `w.fig_sink`", P48; T36–T39) | The comparison drawers draw through `w.fig_sink`, which saves through `figure_path` and then forwards the figure to the window | P48's rule was written for the four FDT drawing functions, which save their own figures; a comparison yields a live figure the window also shows. *Cost if wrong: none found; the PNG lands in the record's `figures/` and is listed, as `figure_path` alone would do.* |
| 27 | §4.3, §4.4 (extends row 18 to the sweep; the whole-piece review's M1, ruling R-F1) | A sweep point whose spontaneous spectrum holds no finite positive-frequency value, or whose χ holds no finite value, is a FAILED point carrying the diverged sentence as its `error`, not a done one; and `make_param_sweep_config` refuses, before anything is spent, a T_a/T grid whose minimum is below 0 (field `t_grid`). 0 is allowed, and there is no S floor | A sweep in which every point had diverged committed as finished with every point "done", never reached the all-failed refusal, and was offered by the picker and by `compare`; a negative T_a/T end was its one known trigger, and it is unphysical. *Cost if wrong: an owner who wants T_a/T = 0 excluded, or an S floor, adds one rule; a diverging point is still counted failed, never done.* |
| 28 | §2.3 (`results`: "for a sweep, a summary across its points"; the whole-piece review's M2, ruling R-F2, re-litigating P78) | A sweep's `results` holds one entry per FINISHED operating point (`param_value`, `omega_0_resonance`, `is_resonant`, `ratio_at_resonance`) beside `usable_fraction`, `offgrid_blanks`, `peak_ratio`, `peak_omega` and a new `peak_param_value`; the top-level `ratio_at_resonance` is null; and `offgrid.of` counts the probes of the points that finished, not every planned point's slots | P78's premise, that the single-run summary carries over to a concatenation, was false: the recorded value was the first point's ratio at the sweep's largest resonance — no point's own — and `of` counted probes never measured. *Cost if wrong: a body shape change made before any owner record of this shape existed; readers print one more nested block.* |
| 29 | §2.2 step 3 ("writes one final refresh so the log up to the failure is on disk"; the whole-piece review's N1) | A kept unfinished record's `log.txt` ends with one line the store writes itself, `HH:MM:SS error stopped: <Type>: <message>`, naming what stopped it (nothing when no run log is active); it is not a logging record | Nothing on disk said why an unfinished record stopped, so the browser could not either; a logging record would have been shown twice by the front ends. *Cost if wrong: the record's `log.txt` holds one line the pane never showed.* |
| 30 | §2.2 step 2 (refresh "after each figure"), §2.3 (the manifest lists the figures; the whole-piece review's N2) | An UNFINISHED record's manifest lists only the figures that are on disk (a committed manifest is byte-identical to before); the single-cell run refreshes once just before Campaign 2 (after the spectrum figure and the `offgrid` count) and again after writing `data.h5`, the sweep once its `data.h5`'s root attributes exist, and the χ figure's path is requested just before it is drawn | A figure path requested but never drawn (a KeyboardInterrupt inside the passive check, a failing plot) left the manifest naming a file that did not exist, and the manifest lagged what was on disk. *Cost if wrong: the refreshes are not literally after every figure — the passive-baseline figure is listed one refresh late, handed on in §1.3.* |
| 31 | §2.2 step 2 ("Each refresh is the same atomic write"; the whole-piece review's N4) | A mid-run refresh the operating system refuses with `PermissionError` (a Windows file lock) is warned about and skipped; `FileNotFoundError` — the record deleted under the run — still propagates and stops it; the commit and the keep branch stay strict | A reader holding a record's file open for a moment would otherwise kill an hours-long run at its next refresh. *Cost if wrong: one refresh's state can be missing from disk until the next succeeds; a lock that never lifts is caught by the strict commit.* |
| 32 | §2.3 (`settings` records `skip_sanity` and `confirm_production`; the whole-piece review's N6) | `settings.confirm_production` is recorded as null when `skip_sanity` is true | The value is not consulted on the skip-sanity path, so recording it claimed a choice that had no effect. *Cost if wrong: the box's state on a skip-sanity run is not recoverable from the record (it changed nothing).* |
| 33 | §7.1, row 8 (P60; the whole-piece review's N11) | `compare repeats`' sentence naming the cells it drew stays in `body.notices` as row 8 says, but is LOGGED at info when the records share one cell and at warning only when they span several | A warning on every correct use — repeats of one cell — taught the operator to ignore the pane's triangle. *Cost if wrong: the one-cell confirmation is easier to miss in the pane; the record still holds it.* |
| 34 | §5.6 ("fixed here only if it is a refusal defect or a data-loss defect"); the §1.3 row it strikes (F54; the whole-piece review's N17, ruling R-F4) | The Live simulation tab's stream builds a zero drive of the sinusoid's shape for a built-in cell with no forcing, instead of the sinusoidal force tensor, so a spontaneous built-in cell streams instead of crashing into the red box | The crash was on a shipped cell in a tab this piece's screens touch, the ruling that handed it on recommended the fix, and it is one line on a CPU-pinned path; the owner's item was taken at the end of the piece. *Cost if wrong: a departure from §5.6's bounded mandate — a defect that is neither a refusal nor a data loss was fixed inside the piece.* |
| 35 | §5.6, §5.3 (the model builder's forcing field, `forcing_value`; the whole-piece review's N18, ruling R-F5) | A negative forcing amplitude is refused under `forcing_value` at Validate, at Save and when a saved model JSON is loaded | Validate and Save accepted it, and the bounds line saved with it then excluded its own value; refusing at load too keeps one rule for the one field. *Cost if wrong: an owner's private saved model with a negative amplitude must be edited before it loads (the shipped SHM and SHM2 have no forcing and still load).* |
