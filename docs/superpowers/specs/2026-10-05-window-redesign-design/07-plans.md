## 7. Plans, "This computer", and what the window remembers

Three things decide which settings a run starts with: run plans (U8, E5, E8), the Settings group "This computer" (U9), and what is stored between launches (U15, B6, B12). Nothing here starts a stage, changes what a simulated row holds or a simulation set's identity, or changes a setting in `core/config.py`. Of the engine it needs E5, E8 and the registry names of U21.

### 7.1 Run plans

#### 7.1.1 What a plan is and where it lives

A plan is a hand-editable text file holding science settings for the inference steps (U8).

- Location: `Resources/Plans/<model>/<name>.txt`, under E8's `config.PLANS_PATH` (11.6). One sub-folder per model, the model being the parent folder's name as for a cell (`core/cli.py:201-204`); a plan never names the model, and the list offers the plans of the model shown on "Model and options" (B13; a choice the owner has not seen).
- `<name>` is the file stem: letters, digits, `_` and `-`, 1 to 40 characters (a choice the owner has not seen; a record's name has its own rule, `core/artifacts/manifest.py:21`).
- Plan files are tracked in git like bounds and cell files and are committed before a long run, so its records are not stamped as made from a changed tree (C27; a choice the owner has not seen): a plan just saved from the window is untracked, and until it is committed a record written afterwards stamps `git_dirty: true` (`core/artifacts/provenance.py:28-30`).
- No plan file is shipped; the first is written with "Save as a plan…" (B13; a choice the owner has not seen).
- Not designed in: deleting one of your own models removes its bounds, cells and units folders only (`core/Helpers/model_store.py:410`), so its plans folder stays. Harmless: the list is read only for a model that exists.

#### 7.1.2 The file's form

It reads like its neighbours: `#` heading lines and `name = value` lines (`Resources/Cells/nadrowski/master_spont_tier1.txt:1-27`, parsed by `core/Helpers/file_manager.py:265-294`). As the window writes one, its description added by hand (7.1.10):

```
# About this plan
The 13-parameter retrain on the temperature box: 10,000 batches, network 256 by 10.

# Model and options
chi_mode = on                  # Kind of observation: Frequency response

# Prior
bounds = master_tier1.txt      # Bounds file

# Training
num_runs = 10000               # Batches
hidden_features = 256          # Network width
num_transforms = 10            # Network depth

# Narrowing round
hidden_features = 256          # Network width
num_transforms = 10            # Network depth
```

| rule | detail |
|---|---|
| encoding, lines | UTF-8 (as `file_manager._read_lines`, `file_manager.py:187-192`); each line stripped, blank lines skipped (as its callers do, `:279-281`) |
| heading | a line starting with `#`. Its text is one of seven, compared without case: `About this plan`, `Model and options`, `Prior`, `Training`, `Calibration`, `Inference`, `Narrowing round`. Any other heading is a problem; the neighbours treat one as the silent end of a section (`file_manager.py:231-233`). The fifth is `Calibration`, not the step's name, Checks: U8 lets a plan hold the calibration's sizes and no other check's (a choice the owner has not seen) |
| setting line | `name = value`; `name` is `\w+` (the name group of `ASSIGNMENT_PATTERN`, `file_manager.py:91`); the value is the rest of the line up to a remark |
| remark | on a setting line, a space, `#` and everything after it; never read. A value therefore cannot hold ` #` |
| names | the engine's setting names (B13; a choice the owner has not seen): the registered field keys (`core/refusals.py:79-214`), which are the stage arguments' names except `bounds` (reaches `bounds_file`), `units` (`units_override`), `hpd_level` (`level`) and `expected_epochs` (reaches no stage). A plan the window saves carries each setting's on-screen label as the remark on its line, and after it the on-screen value where the file's reads differently (`on`, `0`) |
| whole number | optional sign and digits only; no separators |
| number | the form of `FLOAT_REGEX` (`file_manager.py:89`), read with `float()`: the decimal mark is a point whatever Windows' number format is |
| switch | `on` or `off` |
| free text | every line under `About this plan`; read back as one description, never applied, never a record's note (a choice the owner has not seen) |
| strictness | a line that is neither a heading nor `name = value`, a setting above the first heading, and a setting named twice under one heading are problems. The neighbours skip an unmatched line silently (`file_manager.py:235-236`) |
| absence | no setting is required. A setting not in the file means the built-in value |

#### 7.1.3 What a plan may hold

Every row but one reaches its stage as an argument. "Identity" says whether the value is part of the simulation set's identity (`core/artifacts/identity.py:87-122`), which is what U10's ask-first reads.

| heading | name (`core/refusals.py` line) | value and rule | reaches | built-in | identity |
|---|---|---|---|---|---|
| Model and options | `chi_mode` (new) | switch | `make_sim_config(chi_mode=)`, `core/cli.py:275`, `:335` | off, `core/config.py:462` | yes, `:111` |
| | `chi_n_freqs` (`:122`) | whole number, 2 to 24, at most `chi_k_pad` (`config_tab.py:230-241`). The floor is the window's 2, not the engine's own 1 (`core/sim_config.py:121`) (a choice the owner has not seen) | `cli.py:275`, `:336` | 6, `config.py:463` | no |
| | `chi_k_pad` (`:123`) | whole number, 2 to 24 (`sim_config.py:116`) | `cli.py:277`, `:340` | 12, `config.py:528` | yes, `:113` |
| | `chi_max_cycles` (`:124`) | number above 2 (`sim_config.py:130`) | `cli.py:277`, `:341` | 20.0, `config.py:538` | yes, `:117` |
| | `reparam_rotate` (new) | switch | `cli.py:278`, `:342` | on, `config.py:404` | yes, `:98` |
| | `units` (`:114`) | unit words; the window checks that they resolve (7.1.5) | `make_sim_config(units_override=<tuple>)`, `cli.py:274`, `:320-321` | absent: the model's units file, `cli.py:314-316` | yes, `:121` |
| Prior | `bounds` (`:112`) | a file name under `Bounds/<model>/`, written with forward slashes as the picker shows it (`artifact_picker.py:53`). The picker's stored key is the raw relative path with the system's separators (`:53`, `:73-77`), so the window converts before it selects; it also checks that the file exists (7.1.5) | `make_sim_config(bounds_file=)`, `cli.py:273`, `:311-312` | none | yes, `:92-96` |
| | `num_iterations`, `sweep_batch`, `max_sets`, `walk_step`, `stability_units`, `min_cluster_size`, `min_samples` (`:89-95`) | at least 1, 0, 1; above 0, above 0; at least 2, 1 (`prior_tab.py:29-37`) | `build_prior(...)`, `core/orchestrator.py:598-604` | 50, 0, 175000, 0.01, 1000, 50, 10 (`config.py:632`, `:643`, `:650`, `:654`, `:628`, `:671`, `:672`) | only through a newly built prior's fingerprint, `:90` |
| Training | `num_runs`, `run_size_cap` (`:81-82`) | at least 1; at least 0 | `build_posterior(...)`, `orchestrator.py:793`, `:946-947` | 5000, 0 (`config.py:282`, `:328`) | yes, `:99-100` |
| | `hidden_features`, `num_transforms`, `learning_rate`, `stop_after_epochs` (`:96-99`) | at least 1, 1; above 0; at least 1 | `orchestrator.py:796-797`, `:949-952` | 128, 8, 0.001, 20 (`config.py:363-369`) | no |
| | `max_num_epochs`, `checkpoint_every` (`:102`, `:106`) | at least 1; at least 0 | `orchestrator.py:798`, `:948`, `:953` | no ceiling, 50 (`config.py:370`, `:303`) | no (`orchestrator.py:839-840`) |
| | `fisher_m`, `fisher_dz`, `fisher_points` (`:103-105`) | at least 1; above 0; at least 1 | `orchestrator.py:800-801`, `:954-956` | 48, 0.1, 8 (`config.py:405-410`) | no |
| | `expected_epochs` (new) | whole number, at least 1 | no stage: the Training page's "Expected epochs (for the time estimate only)" (B9; 8.4). It never limits the fit | blank; B9 says what a blank means | no |
| Calibration | `n_cal`, `cal_n_scales`, `num_posterior_samples` (`:85-87`) | at least 1 each | `validate_calibration(...)`, `orchestrator.py:1875-1876`, `:1945-1947` | 2000, 200, 1000 (`config.py:279`, `:276`; `orchestrator.py:1876`) | no |
| Inference | `n_samples` (`:88`) | at least 1 (`orchestrator.py:2430`) | `simulated_inference(n_samples=)`, `experimental_inference(n_samples=)`, `orchestrator.py:2451`, `:2537` | 1000 (`orchestrator.py:2152`) | no |
| Narrowing round | `n_directions`, `hpd_level` (`:83-84`) | at least 1 (the upper end, the posterior's width, is the stage's: `orchestrator.py:2611-2614`); between 0 and 1, both excluded | `tsnpe_round(n_directions=, level=)`, `orchestrator.py:2568`, `:2608-2616` | 5, 0.999 (`core/SBI/truncate.py:73`, `:71`) | the round's own set, `:120` |
| | `num_runs`, `run_size_cap`, `hidden_features`, `num_transforms`, `learning_rate`, `stop_after_epochs`, `max_num_epochs`, `checkpoint_every` | as under Training | `tsnpe_round(...)`, `orchestrator.py:2569-2572`, forwarded at `:2648-2652` | as Training | as Training |

Two rules across lines: `chi_n_freqs`, `chi_k_pad` and `chi_max_cycles` are a problem unless the same plan holds `chi_mode = on` (with the switch off the window sends none of them, `config_tab.py:225-226`); `chi_n_freqs` is checked against `chi_k_pad`, taking the built-in for whichever is absent (the registry's default strings, `refusals.py:122-123`, since the reader reads no `core/config.py`).

U8 names "the training budget and network" and "the calibration sizes". The table takes every setting of each step's form that 7.1.4 does not exclude, so it adds the learning rate, the patience, the epoch limit, how often simulations are saved, the rotation's three measurement settings, the draws per calibration dataset and an inference's posterior draws (a choice the owner has not seen), and B9's expected epochs (a choice the owner has not seen; its name in the file is this part's). U8's list does not name the units choice; U15 says it comes "from the plan or the built-in value", so `units` is allowed (a choice the owner has not seen). Typed units give another simulation set identity than the model's units file even when the words are the same (`identity.py:28-32`, `:56-67`; W141, not closed here), so the window never writes `units` when the choice is the model's file.

Every row has a control in part 8's tables (8.2 to 8.6) or, for the `Calibration` values, on the calibration form of Checks (9.4, which says which checks take them). The plan code touches a form only through the page's `plan_fields()`, a mapping from name to a row with `value()`, `set_value(v)`, `usual()` and `set_origin(text)` (5.2).

#### 7.1.4 What a plan may not hold

A registered key that is not in 7.1.3 is "not allowed in a plan"; a name that is not registered is "unknown".

| not allowed | why |
|---|---|
| `chi_f0`, `chi_freq_bounds` (`refusals.py:125-126`) | the two measured definitions; read-only displays of `core/config.py`; U8 |
| the memory limit, `device` (`:127`), the keep-awake switch | they belong to the computer; U8, U9 |
| `name`, `note`, `seed` (`:111`, `:134`, `:151`) | they belong to one run; U8 |
| `new_run`, `accept_truncated`, `accept_other_observation` (`:108-110`) | consents, answered by the session that runs; U8 |
| `resume` (`:107`), on screen "Continue or start new" | it answers what is on disk when one run starts, as a consent does (a choice the owner has not seen) |
| `prior`, `posterior`, `observation`, `cell` (`:129-131`, `:113`), the recordings and their description (`t_obs`, `chi_f0_si`, `recording_*`, `drive_*`) | picks and the description of one recording; a record's id means nothing in another store. Of the Inference step only the number of posterior draws is a plan setting (a choice the owner has not seen) |
| the model | it is the plan's folder (B13) |
| bounds typed by hand | a plan names a file; values typed into the bounds grid are in none |
| every key of Effective temperature, Live simulation, Reduction map, Your own models and the checks other than calibration | U8 limits a plan to the inference steps and names the calibration sizes only |

#### 7.1.5 The reader in the engine (E5) and the window's two checks

Part 11 fixes the place and the contract (11.6): a new module `core/plans.py` that imports the standard library and `core.refusals` and nothing else under `core/`, reads one file, no store and no `core/config.py`, and raises every failure as a `Refusal` with `field="plan"`. This part gives the interface.

```python
HEADINGS: tuple[str, ...]                 # the seven, in file order
ALLOWED: dict[str, tuple[str, ...]]       # heading -> the names of 7.1.3

def read_plan(path) -> Plan
def plan_text(values, about="", remarks=None) -> str   # read_plan(plan_text(v, a)) gives v and a back

@dataclass(frozen=True)
class Plan:
    name: str; path: Path; sha256: str; about: str
    values: dict[tuple[str, str], object]     # (heading, name) -> checked value; only what the file gives

@dataclass(frozen=True)
class PlanProblem:
    line: int | None; heading: str | None; key: str | None; message: str

class PlanRefusal(Refusal):                   # field is always "plan"
    problems: tuple[PlanProblem, ...]
```

- `read_plan` reads the whole file, collects every problem, and returns a `Plan` or raises one `PlanRefusal` listing them all with their line numbers; never part of a plan. The refusal's field is `plan`, the thing to correct being the file; each problem carries its setting's own key.
- `values` is keyed by heading and name, because Training and the narrowing round share names.
- Range rules run in the reader, through the registry's own `require_*` functions (`refusals.py:274-361`); the stage still runs its own at entry. The two limits that are `core/config.py` constants, 24 slots and 2 cycles (`config.py:509`, `:530`), are literals in the module, pinned against `core.config` by a test, as the registry's defaults are (`refusals.py:22-25`).
- `plan_text` takes the description and each setting's remark from the window; the on-screen labels live in `core/gui/fields.py`, which the engine does not read. It strips each `about` line, drops blank ones, and refuses (a `PlanRefusal`) an `about` line that starts with `#` and a value that holds ` #`: that is what makes the round trip above exact, and the save dialog shows the refusal under the field at fault. That the engine's module holds a plan's text form as well as its reader is a choice the owner has not seen (11.6).

| what is wrong | problem's key | sentence (after "line N: ") |
|---|---|---|
| file missing or unreadable | (none; `line` is None) | "The plan was not found: '<path>'." |
| unknown heading | (none) | "'# Trainng' is not a heading a plan has. The headings are: …" |
| not `name = value`, or above the first heading | (none) | "'num_runs 10000' is not a setting line. Write name = value under a heading." |
| unknown setting | (none) | "'num_run' is not a setting PRISM has." followed, when one name is close, by "The nearest is num_runs." |
| setting not allowed | that key | "<What> cannot be set by a plan: it is fixed by measurement." (or "it belongs to the computer", "it belongs to one run", "it is a consent", "it is a pick, not a setting", and for a key of another part of the program "it is not a setting of the inference steps") |
| under the wrong heading | that key | "<What> belongs under '# Prior', not '# Training'." |
| named twice | that key | "<What> is set twice under '# Training' (first on line 9)." |
| wrong kind | that key | "<What> must be a whole number; got 'abc'." (or "a number", "on or off"). The module's own sentence: the registry's rules convert with `float()` and `int()` and raise a bare `ValueError` on such a text (`refusals.py:279`, `:302`), so the reader checks the kind before it calls one |
| out of range | that key | the registry rule's own sentence. The three limits no registry rule has (at most 24, twice; above 2) take the form the window gives them today, "<What> must be at most 24; got 30 (default 6)." (`config_tab.py:232-249`) |
| `chi_*` without `chi_mode = on`; `chi_n_freqs` above `chi_k_pad` | that key | "<What> needs chi_mode = on in the same plan." / "<What> must be at most the number of slots, 12; got 14." |

"<What>" is the registry's phrase for the key. The module's own sentences are in U7's words from the start; the registry's phrases and rule sentences ("the number of chi probe frequencies") are reworded with the rest by E11, in the last stage (U22).

The registry gains four keys for the reader under U21 (counted in 11.11): `plan` ("the plan"), `chi_mode` ("the kind of observation"), `reparam_rotate` ("the rotation") and `expected_epochs` ("the expected number of epochs"). The reader uses each as a key, and the scan requires every key used under `core/` to be registered (`tests/test_refusals.py:1067-1078`). `core/tool/fields.py` maps all four to `None` (U4: no plan flag). `core/gui/fields.py` maps `plan` to the sentence "Correct the file, then choose the plan again." and the other three to their rows on "Model and options" and Training.

**The window's side** is `core/gui/plans.py`. Its functions that list, read and save take the folder as an argument, so a test never writes beside the real inputs (part 13); the window passes `config.PLANS_PATH / model.lower()`. After `read_plan` it makes the two checks that need something outside the file: the bounds file exists in `config.BOUNDS_PATH / model.lower()` ("The bounds file master_tier1.txt is not in this model's bounds folder."), and the units resolve with `cli.units_to_factors` (`core/cli.py:246-269`; "The units are not usable: <reason>", as `config_tab.py:259`). All problems are shown as one list under the picker, and nothing is filled.

#### 7.1.6 The plan picker

On the page "Model and options", directly under the "Model" row (8.1), because plans are per model. Each step page shows under its heading the plan line: "Plan: retrain", "Plan: retrain, 2 changes" or "No plan".

| element | words and behaviour |
|---|---|
| row | label "Plan"; a list holding "No plan", then the model's plans by name. The folder is re-read each time the list opens and when the model changes; there is no rescan button |
| at launch | always "No plan". The choice is never stored and nothing is filled (U8) |
| hint | beside the list while it shows "No plan": "Last used: retrain, 5 Oct 2026, 14:02". Plain text, not a control. If the file is gone: "Last used: retrain (the file is no longer there)". Read from the keys of 7.3.2: there is no hint when the last step applied, or the last run started, had no plan |
| description | under the list, for the plan shown: its `about` text and "14 settings in 4 steps" |
| "Show what it holds" | folded; a three-rule table: step, setting (the on-screen label, with its name in the file beside it), the plan's value, the usual value |
| a plan with problems | "This plan cannot be used: 2 problems in retrain.txt", the problems one per line, and "Open the plans folder". The fill button is greyed with the reason beside it |
| what is in use | "In use: plan retrain, filled 6 Oct 2026, 12:04" or "In use: no plan". When the list shows something else: "Changed, not applied" (U14's words) |
| buttons | "Fill the forms from this plan"; with "No plan" shown it reads "Return the forms to the usual values". "Save as a plan…". "Open the plans folder", greyed until the folder exists. None carries the accent. The fill button is U8's "one action" (C27); choosing in the list alone changes nothing on the forms, so a stray movement of the list cannot rewrite them (a choice the owner has not seen) |
| during a run | locked with the forms (U11) |
| model changed | the list returns to "No plan" and no plan is in use; values already on the forms stay |

#### 7.1.7 Filling the forms

Filling is the only thing a plan does. It never starts a stage, never applies a step, never changes a pick of a record, and never replaces the session (`InferenceScreen.new_draft`, `core/gui/screens/inference_screen.py:163`, is not called). It is one function, `fill_forms(values, source, keep_remembered=False)`, with `source` the words after "from"; it asks nothing. At the picker's button:

1. The file is read and checked again (7.1.5).
2. `would_replace(values)` lists the typed or kept values the fill would change (a form value that is neither its usual value nor the value of the source in use). If there are any, the shared question dialog (B2; 4.6) asks: title "Fill the forms from plan retrain?" (the button's words: U7 has an action keep one name); sentence "Filling from plan retrain sets every setting a plan can hold: 14 from the plan, the rest to their usual values."; then each one in full under the sentence, one to a line as "Batches: 9,500 becomes 10,000", scrolling beyond eight and never behind a link (C10); answers "Keep what is there" (the default; Escape) and "Fill the forms", not drawn as destructive, since no record or run is lost (B27) (a choice the owner has not seen).
3. Every setting of 7.1.3 is set through `plan_fields()`: the plan's value where it has one, otherwise the built-in value. A fill therefore leaves the plan over the built-in values, whatever was typed before (a choice the owner has not seen). `bounds` selects that file in the bounds list and is left alone when the plan names none.
4. The window keeps the source in use: name, path, `sha256`, the time, and the values as filled. The inference screen holds it, not the session object, which is replaced whole when step 1 is applied (`inference_screen.py:163`).
5. A selection row whose form now differs from what its step holds reads "Changed, not applied" (5.3) until that step's own button is pressed.

"Return the forms to the usual values" is the same with an empty plan; its question is titled "Return the forms to the usual values?", says "This sets every setting a plan can hold to its usual value." and answers "Keep what is there" (the default; Escape) or "Return to the usual values". Three other callers fill through the same function, and what they fill counts as the source in use exactly as a plan does. The fill asks nothing more in any of them:

- Records' "Set up a run like this one" (10.2), with the record's settings and the source "record retrain_posterior". Its one question, asked up front (B4, C9), carries the list of `would_replace` under its sentence (C10).
- "Continue from batch N" (6.8; after a failure in the fit the button reads "Fit again on the saved simulations", C2), with the history's `settings` (7.1.11) and the source "run retrain_3". It fills settings only: a name and a note are never refilled, also not after a restart, so the page opens with both blank and the earlier name shown as a hint (C4).
- Home's "Continue with these" (U24, B16; 3.6). `remembered_picks()` (7.3.1) names the plan that was in use last time; the press fills it with `keep_remembered=True` before anything is applied. The values kept from last time (the kind of observation, the bounds file, the two budgets) are then not replaced, each reading "from plan retrain" or "changed from the plan" where the plan holds it and "kept from last time" where it does not (7.1.8), and nothing is asked, the press being the explicit act (a choice the owner has not seen). Step 1 is then applied with the kind of observation that was in use (B12) and every other option at its built-in value, or the plan's (B16); Home's sentence names the kind (C1; a choice the owner has not seen).

#### 7.1.8 Where each value came from

Part 5 draws the origin beside each field and computes it (5.2); this part fixes the words and supplies the source in use, `plan_value(heading, name)` (by heading and name, since two steps share names) and the usual value.

| condition | words |
|---|---|
| the source in use holds the setting and the form has its value | "from plan retrain" ("from record retrain_posterior", "from run retrain_3") |
| the source holds it and the form differs | "changed from the plan" ("changed from the record", "changed from the run") |
| the source does not hold it, or there is none, and the form has the built-in value | "usual" |
| the same, and the form differs | "changed from the usual" |
| put back from the settings file and not touched since (7.3) | "kept from last time" |

The first three are U8's, said of a plan. Their record and run forms and the last two rows are added: a typed value with no plan, and a remembered budget, would otherwise read "usual" falsely (a choice the owner has not seen). Where the source in use holds the setting, its two rows win; otherwise a value put back and not touched since reads "kept from last time" whether or not it equals the usual value. Each field so has exactly one origin, which a test checks over every combination. When a step is applied, the origin of each of its values is kept with the applied value.

#### 7.1.9 What "Before you start" says about the plan

Part 5 owns the frame (5.4) and part 8 the other contents. This part supplies `plan_facts(headings)` as facts for that frame: the source in use, every departure among the settings the run reads (the applied values of earlier steps, the form values of the page's own step), and whether the file still has the `sha256` taken at the fill. None blocks. Every departure is named and none cut, against the plan's value where the plan holds the setting and the usual one otherwise. The lead words are a choice the owner has not seen.

- "Information. No plan. Every setting is at its usual value."
- "Information. No plan. 2 settings differ from the usual: Batches 10,000 (usual 5,000); Rotation off (usual on)."
- "Information. Plan retrain. Its 14 settings are in use as written."
- "Information. Plan retrain, with 2 changes: Batches 9,500 (plan 10,000); Network width 128 (plan 256)."
- "Attention. retrain.txt has been changed since its settings were filled in on 6 Oct 2026, 12:04. The forms hold the earlier values; fill again to take the changes."
- With a record or a run as the source, "Plan retrain" reads "Settings from record retrain_posterior" or "Settings from run retrain_3".

#### 7.1.10 Saving the current settings as a plan

"Save as a plan…" opens the shared dialog in its form that asks for one line of text (ruling D4; 4.6): the line is "Plan name", checked as it is typed, and "What will be written" is the list under its sentence. The dialog asks for no description: the text under `# About this plan` is written by hand in the file (a choice the owner has not seen).

- Written: every setting of 7.1.3 whose form value differs from its built-in value, and the bounds file whenever one is selected, each line with its remark (7.1.2). Settings at their usual values are left out (B13; a choice the owner has not seen): one written out would keep overriding the built-in value after that value is deliberately changed, the pattern that trained the 2026-08-19 run on a retired band (`config_tab.py:137-143`). The dialog says: "Only the settings that differ from their usual values are written. A setting left out follows the usual value."
- The three settings of the driven recordings (`chi_n_freqs`, `chi_k_pad`, `chi_max_cycles`) are written only when the kind of observation is the frequency response: without `chi_mode = on` the reader refuses them (7.1.3). A test reads back every file the dialog can write.
- A form that reads "Changed, not applied" is saved as the form shows it; the dialog names those settings.
- Bounds typed by hand: no `bounds` line, and the dialog says "The bounds were typed by hand, so this plan names no bounds file."
- Nothing to write: the button is greyed, "Every setting is at its usual value, so there is nothing to save."
- A name already in use is never replaced without asking, through the shared question dialog (B2): "Replace plan retrain?", "retrain.txt was last changed on 3 Oct 2026, 14:02. The earlier file cannot be brought back.", answers "Keep the existing plan" (the default; Escape) and "Replace it", drawn as destructive (B27).
- The text is `core.plans.plan_text(values, about, remarks)`, so reader and writer cannot drift. The window writes it to a sibling temporary file and moves that into place. The saved plan becomes the plan in use; no value changes.

#### 7.1.11 What the run history records about the plan

Part 6 owns the history file (6.5) and takes three entries from this part at the start of a run that keeps one (C3):

| entry | content |
|---|---|
| `plan` | none, or: the name; the file's path and its `sha256` at the start of the run, as `provenance.file_ref(path, relative_to=config.RESOURCES_ROOT, missing_ok=True)` gives them (`provenance.py:69-99`; without `missing_ok` a file removed since the fill raises there, `:80-82`, and would stop the run from starting); the `sha256` and the time of the fill; whether the two differ. For a record or a run as the source: its kind, id and name |
| `settings` | everything the run was started with, enough to set the same run up again (6.5, 6.8): for each setting of 7.1.3 the run reads, its step, name, on-screen label, value, origin (7.1.8) and, where they differ, the plan's and the usual value; and the model, the records picked, the name, the note and "Continue or start new", as the page handed them to `dispatch`. The name and the note are kept as the record of that run and are never filled back (C4) |
| `memory_limit` | 7.2.3 |

### 7.2 Settings, "This computer"

A group on the Settings page (10.7) for what belongs to the machine and not to the science: the memory limit and the keep-awake switch. Both are remembered, per computer.

#### 7.2.1 The memory limit today

A box on the Config tab, opening at the environment variable's value or 0 (`config_tab.py:108-113`; W112). Every keystroke assigns `core.config.SIM_VRAM_CEILING_GIB`, a blank or half-typed value being 0, which is off (`:117`, `:148`; W016), and runs `nvidia-smi` on the window's thread, as launch does (`:119`, `:151`; W043, W166). The note and the help advise leaving it off (`:159-161`; `help_text.py:47-48`; W018), and a variable that is not a number is said to override the box while the planner ignores it (`:154-157`; `core/SBI/pipeline.py:589-595`; W019). The limit is never stored and recorded nowhere (`:137-143`; W098).

#### 7.2.2 On screen

| element | words |
|---|---|
| label and one sentence | "Memory limit". "The most graphics-card memory one batch of simulations may plan to use." |
| the choice | two options: "No limit" and "Limit each batch to" with a field and the unit "GiB". "No limit" is a choice of its own, never a blank and never a typed 0 |
| in force | "In force: 6 GiB." or "In force: no limit." |
| the variable set to a number | the options are greyed and the line reads "Set outside PRISM by PRISM_VRAM_CEILING_GIB = 6.5. It takes precedence; the limit remembered here (6 GiB) is not in use." The variable's name is the one code name on screen in this part: it is the only handle for changing it, and today's note names it too (`config_tab.py:156`) (a choice the owner has not seen). At 0 or below, which the planner reads as no limit (`pipeline.py:592`), the first sentence ends "= 0, which means no limit." |
| the variable set to something that is not a number | "PRISM_VRAM_CEILING_GIB is set to '6,5', which is not a number. PRISM ignores it. In force: 6 GiB." |
| free memory | "Free graphics-card memory: not read yet." and a button "Read now"; after a press, "Free graphics-card memory: 12.5 GiB, read 6 Oct 2026, 12:04." and "Read again" |
| the last run | "The last run on this computer (retrain, ended 5 Oct 2026, 09:12) ran with a limit of 6 GiB. Its largest batch held 5.4 GiB.", the second sentence left out when the run measured none, with "It ended for lack of graphics-card memory." when it did; or "No run on this computer has been recorded yet." |
| during a run | the memory limit is greyed: "Locked while a run is live. This run started with a limit of 6 GiB." |

#### 7.2.3 In code

- Meaning, unchanged: the cap on what one simulation batch may plan to hold on the card (`core/config.py:146`). The planner's budget is the smallest of its own free-memory figure, its learned cap and this limit (`pipeline.py:635-640`). A lower limit splits batches and costs time; it changes neither the rows nor the identity (`config.py:164-168`).
- One module, `core/gui/this_computer.py`, is the only place in the window that assigns `core.config.SIM_VRAM_CEILING_GIB`: the one existing exception to "every knob is an argument", read live by the planner. Making it an argument would be an engine change and is not designed in.
- `set_memory_limit(gib or None)` assigns the constant (0.0 for none) and stores the value, once per accepted change. `apply_remembered()` is called from `build_app` after the settings store exists and before the main window is built (3.10); it reads the settings file and never the card.
- `memory_limit_in_force()` returns the value and its source, `setting`, `environment` or `none`. The value is always the planner's own answer, `pipeline.vram_ceiling_gib()` (`pipeline.py:584-603`), so the screen cannot disagree with the run. Only the source is the window's: the variable counts when it is non-blank and `float()` accepts it, the planner's own test (`pipeline.py:589-592`), and a test pins the two against each other over a table of texts. The environment variable still wins; no engine line changes.
- Stored as `computer/<host>/memory_limit_gib`, text `none` or a number, `<host>` being `socket.gethostname()`, the name each record already carries (`provenance.py:55`). The settings file is per user (`core/gui/settings.py:20-23`) and can travel with a profile; keying by the computer's name keeps one card's limit from being applied on another (a choice the owner has not seen). An absent value means no limit. So does an unreadable one, and the group then says "The remembered limit could not be read; no limit is in force."
- Free memory is read by `provenance._nvidia_smi_free_gib` (`provenance.py:114-128`, unchanged) on a thread of its own, only when the button is pressed: never at launch, never when the page is built or shown (3.10), and never while a run is live, when the button is greyed with "Not read while a run is live." (a choice the owner has not seen).
- At the start of every run that keeps a history (6.5; C3) the window hands part 6 its `memory_limit` entry: the limit in force, its source and the computer's name. Part 6 writes it and heads the run's full text with it (6.4, 6.5). A record's own manifest still has no such key (W098's remaining half; engine, not designed in).
- The last-run sentence reads the newest history file whose `memory_limit` names this computer: its limit, and from `health` the largest memory peak and whether the run ended for lack of memory (6.5).
- The command line does not read the window's settings: a run started there has no limit unless the variable is set, as now.

The memory fact stands only in the "Before you start" of a run that simulates on the card (C13, where U9 said every one: a choice the owner has not seen; 5.4, and for Training 8.4). It is drawn from `memory_limit_in_force()` at each draw, so it cannot be stale. Always led by "Information." (a choice the owner has not seen): "Memory limit: 6 GiB for each batch of simulations.", one of the environment sentences of 7.2.2, or "No memory limit is set on this computer."; then the last-run sentence. It shows no free-memory reading and no verdict (U10).

#### 7.2.4 A blank or mistyped value

The field is checked when editing ends (Enter, or leaving the field), not on each keystroke. A blank, a text that is not a number, a 0, a negative or a non-finite value is not applied: the limit in force stays what it was, and the line beside the field says why.

| typed | line beside the field |
|---|---|
| nothing | "The field is empty. In force, unchanged: 6 GiB. To run without a limit, choose 'No limit'." |
| `6,5` | "Use a point as the decimal sign. In force, unchanged: 6 GiB." The comma is not swallowed as it is typed (ruling D9; 4.6) |
| `abc`, `-` | "That is not a number. In force, unchanged: 6 GiB." |
| `0`, `-2` | "The limit must be above 0. To run without a limit, choose 'No limit'." |

#### 7.2.5 What the note says and does not say

Says, under the options: "A lower limit makes PRISM split a batch into smaller pieces, which takes longer and changes nothing in what is simulated. A limit above what PRISM would plan anyway changes nothing. With no limit, PRISM plans from the free-memory figure Windows gives it, which can count memory other programs are using as free." (`pipeline.py:635-640`; `config.py:150-154`.)

Does not say: that any value "fits"; that no limit "is right on an idle card"; "only if you need the desktop"; a recommended figure derived from free memory. The planner's arithmetic is left for the audit (U18), and the advice of free memory minus 1 GiB (`config.py:158`) is not established as safe (W025). The "?" text for the limit (W115; 10.8) and the guide's line (`docs/guide/window.md:219`, W168; part 13) take these words.

#### 7.2.6 The keep-awake switch

- A tick box: "Keep this computer awake while a run is live". On unless turned off (U11). Stored as `computer/<host>/keep_awake`.
- Note: "PRISM asks Windows not to go to sleep from the start of a run until it ends. The screen may still turn off. Shutting down or signing out is separate: Windows then shows PRISM's reason and waits for your answer, whatever this switch says." The last sentence is U27, whose mechanism is the run part's (B28; 6.10).
- While a run is live: "Windows has been asked to stay awake since 6 Oct 2026, 12:04." If Windows refuses (the call returns 0): "Windows did not accept the request. This computer may go to sleep during the run."
- `this_computer.keep_awake(on)` calls `SetThreadExecutionState` with `ES_CONTINUOUS | ES_SYSTEM_REQUIRED`, or with `ES_CONTINUOUS` alone to end the request, through `ctypes`, on the window's own thread (the request belongs to the thread that made it, and Windows drops it when the process ends). Nothing under `core/` calls it today (W004); when it is called is part 6's (6.10).
- The switch stays usable during a run, unlike the memory limit (C29; 6.12): it changes nothing about the run or its record (a choice the owner has not seen).
- Off Windows the box is greyed: "Only on Windows."

### 7.3 What the window remembers

#### 7.3.1 Rules

1. A remembered value is written by one function, `put(key, value)` in `core/gui/remembered.py` (set, then `sync()`), at one of two moments (U15, B6): a pick when it comes into use, which is when the run returns its record (B3) or, for step 1, at Apply; the two budgets and the description of the recordings when a run starts with them. (A setting of the Settings page, 7.2 and 10.7, is written when it is changed.) Closing writes the window's size and place and nothing else (3.10; today `main_window.py:447-456` writes everything then, as each list happens to show). A session that only looks therefore changes nothing (the look-and-close overwrite of the bounds file, named in W138), a crash loses nothing that had been applied, and a driven-recordings table filled in and never run is not kept.
2. A remembered value is put back on its form as a choice, marked "kept from last time", when its part is first built (B21). The window never applies it on its own: the session is new at every launch (`inference_screen.py:49`). Home's "Continue with these" (U24, B16, C1) is the one press that applies the picks together. `remembered.remembered_picks()` gives Home what was in use last time, so Home names no key: the model, the kind of observation, the bounds file, the prior, the posterior and the plan, each left out when its file or record is gone or, for the plan, has problems. The press then goes on without that plan (a choice the owner has not seen).
3. A list that is re-read keeps its selection. `ArtifactPicker.refresh` clears and refills and re-selects nothing (`artifact_picker.py:44-57`), where `StorePicker.refresh` keeps its key (`:181-183`, `:219`); the file list is given the same behaviour. When a list is pointed at a folder again it re-selects the pick in use now, not the one stashed at launch (`prior_tab.py:66`, `:141`, `:373`; `inference/base.py:37-40`; `infer_tab.py:545`). When the remembered or selected file is gone, or nothing is remembered, the list shows no selection ("Choose a bounds file") and never falls to the folder's first file, whose order is the file system's (`file_manager.py:316-322`; today an empty key leaves the first row selected, `artifact_picker.py:82-83`) (a choice the owner has not seen). The widget is shared, so the rule holds for the cell lists of part 10's pages too.
4. A key that this design removes is neither read nor written, and is left in the file. A stale key from an older version is ignored, not restored and not migrated.
5. No science setting is restored except those U15 and B12 name. A consent, a seed, a record name, a note, a comparison list and the plan picker's choice are never stored.

#### 7.3.2 The table

Keys are those of the settings file (`settings.py:20-23`). "When" is the moment of writing after this design. Part 13's remembered-table test takes its rows from here.

| key today | written, read (today) | after | when, and why |
|---|---|---|---|
| `window/geometry` | `main_window.py:450`, `:188` | kept | at close (B6) |
| `appearance/mode` | `settings.py:75`, `:71` | changed | a stored `auto` is read as `system` (U13; 3.9) |
| `appearance/system_accent`, `appearance/force_inter` | `settings.py:84`, `:93` | removed | U13 |
| `layout/<panel>/splitter`, `/results` | `base_panel.py:369-370`, `:382-387` | removed | the dividers go (5.1) |
| `inference_config/model` | `config_tab.py:403`, `:415` | kept | when "Model and options" is applied |
| `inference_config/units_mode`, `units_text` | `config_tab.py:404-405`, `:418-420` | removed | U15; the units choice comes from a plan or the model's file. Closes W146 |
| `inference_config/reparam_rotate` | `config_tab.py:407`, `:422` | removed | U15; opens on (`config.py:404`). Closes W008 |
| `inference_config/chi_mode` | `config_tab.py:406`, `:421` | kept | when "Model and options" is applied (B12) |
| `inference_prior/bounds` | `prior_tab.py:363`, `:373` | kept | when a prior built or loaded with that file comes into use |
| `inference_prior/prior` | `prior_tab.py:362`, `:370` | kept | the id of the prior that comes into use (`prior_tab.py:316`); after a build this is the new prior, not "" (W015) |
| `inference_posterior/posterior` | `posterior_tab.py:347`, `:354` | kept | the id of the loaded or trained posterior that comes into use (`posterior_tab.py:288`) |
| `inference_posterior/num_runs`, `run_size_cap` | `posterior_tab.py:348-349`, `:357-358` | kept | when a training starts with them (U15) |
| `inference_infer/cell` | `infer_tab.py:530`, `:545` | kept | when an inference on a simulated cell starts with it (a choice the owner has not seen: U15's list does not name the cell file) |
| `inference_infer/infer_mode` | `infer_tab.py:531`, `:547` | kept | when an inference starts |
| `inference_infer/exp_spont`, `exp_forced`, `chi_spont` | `infer_tab.py:534-536`, `:552-554` | kept | when an inference on recordings starts |
| `inference_infer/sim_tobs`, `exp_tobs`, `chi_tobs`, `chi_f0_si` | `infer_tab.py:532-538`, `:550-556` | removed | U15, B12: the three length boxes (one field key, `t_obs`) and the drive amplitude open blank at every launch, not at 1 s and 1 N (`infer_tab.py:96`, `:108`, `:120-121`; `config.py:223`); a blank is refused at the press (`infer_tab.py:399`, `:413-414`, `:432`). U15 and B12 name the recordings' lengths; the simulated cell's length, the same setting under the same key, goes with them (a choice the owner has not seen) |
| (none) | rows are never stored (`infer_tab.py:540-541`) | added | `inference_infer/driven/size`, `driven/<n>/file`, `driven/<n>/frequency`: every row of the driven-recordings table, the frequency as typed, when an inference on recordings starts (U15) |
| `inference_tsnpe/observation` | `tsnpe_tab.py:219`, `:226` | kept | when a round starts with it. An observation made in this session is preselected over the remembered one (U16) |
| `inference_tsnpe/num_runs`, `run_size_cap` | `tsnpe_tab.py:220-221`, `:229-230` | kept | when a round starts (B12) |
| `fdt/model`, `cell`, `record`, `n_freqs`, `ensemble_m`, `freqs_per_batch`, `f0`; `crossval/preset`, `cell`, `record`, `f0`, `freqs_per_batch`, `s_points`, `t_points`; `reduction/cell`, `f0`; `simulate/model`, `cell`, `tobs`, `frame_steps`, `fps` | `fdt_panel.py:340-345`, `:355-361`; `crossval_panel.py:353-360`, `:367-376`; `reduction_panel.py:98-99`, `:104-105`; `simulate_panel.py:179-183`, `:190-195` | kept | as part 10 has it (10.3 to 10.5): at the start click, and `record` when a saved run is opened. `simulate/tobs` is a simulated length, not a recording's, and keeps its opening value |
| `artifacts/kind`, `sort_col`, `sort_order` | `artifact_screen.py:959-961`, `:969-971` | changed | as part 10 has it (10.2): the sort is stored by column name and an old number is ignored |
| (none) | | added | `computer/<host>/memory_limit_gib` (U9) and `computer/<host>/keep_awake`, on each accepted change |
| (none) | | added | `plans/last_used/<model>`, `plans/last_used_at/<model>`: the plan in use, or blank, and the time, written with every key of the `inference_*` groups above. They feed the picker's hint (U8) and "Continue with these" (B16) and are never applied at launch (a choice the owner has not seen: a plan counts as used when a step is applied or a run starts with it, not when it is filled in) |

Tests that pin today's rule and are rewritten first: `tests/test_settings_persistence.py:729-792` (the limit is not stored, and the note names the variable, `:785`), `:973-1046` (the first tab's keys at a save, `units_mode`, `units_text` and `reparam_rotate` among them, `:1046`), `:1049-1145` (the keys each other panel writes at a save, and `_saved_bounds_key` at `:1123`), `:407`, `:524-539` (a list whose remembered file is gone must not be left without a selection, the opposite of rule 3), `:662-686` (the layout keys), `:1148`, and `tests/test_artifact_browser.py:154-185`.

#### 7.3.3 The rule in `CLAUDE.md` after this

For the owner to approve; this part edits nothing. It replaces the bullet beginning "The inference tabs remember SELECTIONS", up to and not including its sentence on the Simulate, FDT and CrossVal panels (part 13 gives that sentence its new names). Four clauses depend on choices flagged above: the cell file; the simulated cell's length among the three observation lengths; plan files being tracked and committed (C27); and which "Before you start" states the limit (C13).

> The window remembers what was last USED, and never a science setting on its own. Remembered, each written when it comes into use or when a run starts with it, never at closing (closing writes the window's size and place only): the model, the bounds file, the cell file and the saved records picked; the kind of observation; the description of the recordings (the mode, the recording paths, and every row of the driven-recordings table, file and frequency); the training budget and the narrowing round's (batches and rows per batch). A remembered value is put back on its form as a choice, marked as remembered; nothing is applied or loaded until the step's own button, or Home's "Continue with these", is pressed. The three observation lengths and the physical drive amplitude `Drive F₀ (N)` are NOT remembered: they open blank at every launch and must be typed. Every other science setting opens at `config.py` on every launch (the driven-recording count, slots and lock-in ceiling, the rotation switch, the units choice, the sweep, network, rotation and calibration settings, the HPD level and the direction count), and the non-dimensional χ drive amplitude and band are read-only displays of it. Beyond those, science settings pass from one session to the next only when a press fills the forms: from a run plan (`Resources/Plans/<model>/<name>.txt`, read by `core/plans.py`; tracked in git like the bounds and cell files and committed before a long run), at the picker or through "Continue with these" when one was in use last time; from a record ("Set up a run like this one"); or from a run's history ("Continue from batch N", which fills settings only: the name and the note open blank). The picker opens on "No plan" at every launch, only the NAME of the plan is stored, and a plan cannot hold the two measured χ definitions, anything about the computer, a name, a note, a seed or a consent. The memory limit and the keep-awake switch belong to the computer (Settings, "This computer") and ARE remembered, per computer: the "Before you start" of every run that simulates on the card states the limit in force, a run's history records it, and `PRISM_VRAM_CEILING_GIB`, when set to a number, still wins. A consent, a seed, a record name, a note and a comparison list are never remembered. A stale key an older build left in `PRISM.ini` is ignored, not restored.

In the bullet on long runs, "and the window's field opens at 0 unless `PRISM_VRAM_CEILING_GIB` is set in the environment the window starts in" becomes:

> it is set once in Settings, "This computer", is remembered for this computer and is shown in the "Before you start" of every run that simulates on the card; `PRISM_VRAM_CEILING_GIB`, set in the environment the window starts in, takes precedence. A run started from the command line reads the variable only.

The bullet on `Resources/` gains `Plans/` among the input folders (E8); that edit is part 13's.

### Choices in this part the owner has not seen

1. Filling from a plan sets every setting a plan can hold, to the plan's value or else the usual one, so values typed before do not survive a fill.
2. Saving a plan writes only the settings that differ from their usual values, plus the bounds file, so a setting left out keeps following the built-in value.
3. "Continue with these" applies the kind of observation that was in use last time, and every other option at its built-in value or the plan's.
4. A plan is filled into the forms by the button "Fill the forms from this plan" beside the list; choosing in the list alone changes nothing.
5. Plan files are tracked in git like bounds and cell files and are committed before a long run, so that its records are not stamped as made from a changed tree.
6. The memory lines stand only in the "Before you start" of runs that simulate on the card, where the decision on the memory limit said every "Before you start".
7. A plan may hold every setting on a step's form that is not excluded, including the learning rate, the patience, the epoch limit, how often simulations are saved, the rotation's three measurement settings, the draws per calibration dataset and an inference's posterior draws.
8. A plan may hold typed units.
9. Plans are kept in one folder per model, and a plan never names or changes the model.
10. A plan file names each setting by the engine's name; a plan the window saves writes the on-screen label beside it as a remark.
11. No plan file is shipped; the first one is saved from the window.
12. A plan may not hold "Continue or start new"; it is treated like a consent.
13. Of the Inference step a plan holds only the number of posterior draws, never the cell, the recordings, their lengths or the drive amplitude.
14. A plan may hold the expected number of epochs, which feeds the time estimate only and never limits the fit.
15. Before a fill replaces a value that was typed or kept from last time, the window asks, lists each one, and preselects "Keep what is there".
16. "Continue with these" fills last time's plan without a further question and leaves in place the values kept from last time: the kind of observation, the bounds file and the two budgets.
17. The memory limit and the keep-awake switch are stored under the computer's name, so a settings file carried to another machine applies no limit there.
18. Free graphics-card memory is read only when its button is pressed in Settings: never at launch, when the page opens, or while a run is live.
19. In "Before you start" the plan line and the memory line are marked "Information"; the only "Attention" is for a plan file changed since it was filled in.
20. The keep-awake switch stays usable during a run, while the memory limit is locked.
21. Two origin phrases are added to the decision's three, "changed from the usual" and "kept from last time", and the plan's two phrases are also said of a record or a run that filled the forms.
22. A plan counts as used, for the "Last used" hint and for "Continue with these", when a step is applied or a run starts with it, not when it is filled in; a step applied or a run started with no plan clears the hint.
23. The cell file picked for a simulated inference stays remembered.
24. When a remembered or selected file is gone, or nothing is remembered, a file list shows no selection instead of falling to the folder's first file.
25. A plan file may carry a free-text section, "About this plan", shown under the picker and never used as a record's note.
26. The file's heading for the calibration sizes is "Calibration", not the step's name, "Checks".
27. A plan's name is letters, digits, "_" and "-", at most 40 characters.
28. A plan file is held to the window's floor of 2 driven recordings per observation, not the engine's 1.
29. The engine's plan module holds a plan's text form, which the window's save uses, as well as its reader.
30. The memory-limit lines in Settings and "Before you start" name the environment variable that overrides the limit, when it is set: the one code name on screen in this part.
31. The observation length of a simulated cell opens blank at every launch, with the two recordings' lengths.
32. When the plan in use last time is gone or has problems, "Continue with these" leaves it out and goes on.
33. "Save as a plan…" asks for the plan's name only; a description is written by hand in the file, under "About this plan".

### What this part fixes

- W002, W029, W050, W054: a bounds or cell list that is re-read or pointed at its folder again keeps the pick in use, not the first file or the one stashed at launch (7.3.1, rule 3).
- W004: the computer is kept awake during a run (7.2.6). A sign-out or a shutdown is U27's, in the run part.
- W008: the rotation switch is no longer remembered (7.3.2).
- W015: the remembered prior is the one built or loaded, never an empty pick (7.3.2). The list's own display after a build is the Prior page's.
- W016: a blank, mistyped, zero or negative limit is not applied and never means "no limit" (7.2.4).
- W018, W168 (the words): the note no longer advises running without a limit (7.2.5); the guide's page is part 13's.
- W019: a variable that is not a number is said to be ignored, with the limit really in force (7.2.2).
- W043 (the per-keystroke half; the launch half is the shell part's) and W166: free memory is read off the window's thread, only on a press, and shown with its time; the note gives no advice (7.2.2, 7.2.3, 7.2.5).
- W055: every row of the driven-recordings table is remembered (7.3.2).
- W056: the recording lengths and the drive amplitude open blank at every launch (7.3.2). The missing check on the amplitude is not this part's.
- W098: the limit a run used is in its history, its full text and "Before you start" (7.2.3, with 6.5). The record's manifest is unchanged.
- W112 (its second half): the memory limit leaves the first step's page for Settings (7.2).
- W146 (across launches): the typed units text is no longer stored (7.3.2). Following a change of model within a session is the first step's page's.
- The look-and-close overwrite of the remembered bounds file (first note 2A; no id of its own, named in W138): nothing is written at closing (7.3.1, rule 1).

Helped here and closed elsewhere: W003, W017, W024, W026, W141, W142, W194 (the steps' pages and "Before you start"); W115 (Help).
