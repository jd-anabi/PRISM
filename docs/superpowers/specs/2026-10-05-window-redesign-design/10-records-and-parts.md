## 10. Records, the other parts, Settings and Help

Everything outside Parameter inference: Records; Effective temperature (One cell, Sweep study,
Compare); Live simulation; Reduction map; Your own models; Settings; Help. This part gives these
pages' content; the anatomy (5), the run and its stop (6), the question dialog (look 4.6), the
rail and Home (3), plans, "This computer" and what is remembered (7) and the refusal table (5.7)
are other parts'. Of the engine changes it needs E9 alone; points marked "Engine:" are named and
not designed in. Each part here is built when first opened (B21), reads no card when built (shell
3.10) and asks the run controller for its lock (run 6.12).

In the control tables, **Place** is M main setting, R "Rarely changed", N name and note, F stated
as text, not a field (look 4.5). The new labels in those tables, and the sentences of this part's
questions and glossary, are its own wording (a choice the owner has not seen); each label goes into
`fields.CONTROL` (`fields.py:150-177`) in the task that rebuilds its page, its place named as in
"Effective temperature, Sweep study" (5.7). Dates are local, written "3 Oct 2026, 14:02" in every
table, record and sentence here (B22, C18).
Every question is the one question dialog (B2), asked through the look's `ask` (4.6): its first
answer is the safe one, the default and Escape's, and a second answer marked "loses work" is drawn
as destructive (`loses_work`, B27). Where a question names several things (leftovers to clear, what
a next step loads and lets go, the settings a fill would replace, the records that stop a delete),
they are the dialog's list under its sentence (`items`, 4.6), one to a line, scrolling beyond eight
and never behind a link (C10).

### 10.1 Words for records

One word per store kind, in every sentence, list and question (W131; `artifact_screen.py:40-51`,
`:171`, `:326`).

| store kind (`store.py:33-41`) | one record is called | entry in the kind filter |
|---|---|---|
| `prior` | prior | "Priors" |
| `simulation` | simulation set | "Simulation sets" |
| `posterior` | posterior | "Posteriors" |
| `observation` | observation | "Observations" |
| `calibration` | calibration | "Calibrations" |
| `inference` | inference result | "Inference results" |
| `diagnostic` | check | "Checks" |
| `fdt`, `study` "single" / "sweep" / "comparison" | effective-temperature measurement / sweep / comparison | "Effective temperature" |

**A record's label**, one function for Records, Home's "Latest records" (shell 3.6), every list of
saved records and every sentence that names a record:

- named: its name;
- unnamed: "Unnamed calibration, 3 Oct 2026, 14:02", in place of "(unnamed 20261003T140211)"
  (`store.py:160-162`); the id stays on the record's page (W088, W190);
- a simulation set with no name: "Simulation set for retrain_prior", from
  `Summary.parents["prior"]` (`store.py:723`) looked up among the priors; "Simulation set for a
  prior that is no longer here" when it is not found; the id's first six characters follow when two
  sets share a prior (W130). A set gets a name only by "Rename…", which E9 makes last (11.12).

(The two forms for a record with no name: a choice the owner has not seen.) The "In use" lines
(5.3), the from-zero question (8.4) and the Checks page (9.3) call this function.

The kind words U7 does not give ("inference result", "check" for `diagnostic`, "measurement",
"sweep", "comparison") are this part's (a choice the owner has not seen); E11 writes them into
the refusal sentences the command line prints (11.13).

**State words**, each with one of the look part's ten markers (4.5). In a sentence the word is
written small ("2 unfinished").

| word | marker | when | from |
|---|---|---|---|
| "Finished" | `done` | its run reached the end | `Summary.finished` (`store.py:700`) |
| "Checks only" | `info` | a finished measurement that ran its checks and nothing else | `finished`, `study` "single", no `data.h5` in the folder (`fdt_pipeline.py:348-350`, `:477`) |
| "Unfinished" | `half` | a simulation set or effective-temperature record whose run did not reach the end | `complete and not finished` (`store.py:132-147`) |
| "Being written" | `now` | a run in this window is writing it | below |
| "Leftover" | `info` | a folder with no manifest at all | `reason == NO_MANIFEST_REASON` (`store.py:51`, `:667-668`) |
| "Unreadable" | `err` | a folder whose manifest this PRISM cannot use | any other reason (`store.py:672-677`) |

("Checks only" as a state of its own: a choice the owner has not seen; W070.)

**"Being written"** (W030). Today a calibration or an inference in flight is listed as a folder
"nothing can load" (`artifact_screen.py:373-379`): its writer opens before the work
(`orchestrator.py:1950`, `:2227`) and writes the manifest last (`store.py:627-629`). New: while a
run is live here a row reads "Being written" when the run's `spec` names its id, or when its kind
is one the run writes (run 6.2; for a training or a narrowing round also the simulation set), it
is a leftover or unfinished, and its folder was created or last written after the run's `started`
(run 6.5). Its page says
"Being written by the run in this window: Training retrain_posterior. It becomes a record when the
run finishes." A posterior has no folder until its fit ends (`orchestrator.py:1420`), so during a
training the list's foot says "A posterior is added when the training finishes." A command-line
run writing the same kind at that time reads the same (a choice the owner has not seen).

### 10.2 Records

For reading everything PRISM has written and acting on one record. Replaces
`screens/artifact_screen.py`, `widgets/artifact_table.py` and the text pane fed by
`core/artifacts/report.py`. Heading "Records"; sentence "Everything PRISM writes is kept here, with
its settings and what it was made from." Layout as in `structure-chosen-rail-5-records.png`: the
list on the left, the selected record's page on the right; below a page width of 940 px (5.1) the
record's page takes the list's place, with "All records" to return (a choice the owner has not
seen; the Checks page does the same in the small layout, 9.4). The part's `show_page` (shell 3.2)
takes a record's kind and id and opens it: so arrive a name on Home, a "Made from" link and the
link of a taken name.

#### The list

One ruled table over all kinds (today one kind at a time, each with its own columns:
`artifact_screen.py:200-210`, `artifact_table.py:20-39`).

| column | content | from |
|---|---|---|
| "Name" | the label (10.1) | `Summary.name`, `.id`, `.parents` |
| "Kind" | the kind word | `Summary.kind`, `.study` |
| "State" | marker and word | 10.1 |
| "Made" | local date and time | `Summary.created`, a UTC ISO string (`store.py:331-332`, `:614`), converted (W032) |
| "Holds" | one phrase | simulation set: "4,200 of 10,000 batches" or "10,000 batches, 20,480,000 simulations" (`artifact_table.py:87-109`); posterior, observation: the kind of observation, and "narrowed" for `amortized` false; calibration, inference result, check: "of retrain_posterior"; sweep: "10 of 12 points, 2 failed" (`artifact_table.py:124-138`) |
| "Note" | the note | `Summary.note` |

Above it: "Kind" ("All kinds", then the entries of 10.1); a search field, "Search names, notes and
ids", which hides rows whose name, note, id, kind word and "Holds" all lack the text, as it is
typed (W088); "Refresh"; at the right "Clear leftovers…". Below it: "38 records: 35 finished, 2
unfinished, 1 being written. 1 leftover." (W129; `artifact_screen.py:337-338`).

- Read by `ArtifactStore.list(kind)`, once per kind (`store.py:681-735`), never while the page is
  built (kept: `artifact_screen.py:279-284`); again when the page is shown, on "Refresh", after a
  change made here, and on a change of the run part's `RunStatus` (W082; today the screen takes no
  signal from a run, as its imports show, `artifact_screen.py:22-36`).
- The selection survives a re-read, found again by kind and id (a leftover by folder name), and the
  page keeps its place (W086; `artifact_table.py:220-223`, `:276`). If the
  record is gone: "The record that was open is no longer there." Not remembered between launches
  (kept: `artifact_screen.py:12-14`).
- Header click sorts; "Made", newest first, at the start; leftover and unreadable rows last under
  every column (kept: `artifact_table.py:169-210`). The header is the look's
  `RuleHeader(sortable=True)` (4.6).
- Remembered in the `artifacts` group (`artifact_screen.py:942-976`): the kind filter, with "all"
  added, and the sort, stored by column name, each written when it is changed and not at closing
  (B6). Today the sort is a column number, which names a different column for each kind; an old
  number is ignored. The search text is not remembered.

The command line keeps its per-kind columns. `tests/test_tool.py:3072` and `:3958`, which hold the
tool's columns and three of its sentences equal to the window's, are re-expressed as the tool's own
literals (13.1.2), so the command line is still proven unchanged.

#### A record's page

Built from the record's `Manifest` (`store.get(kind, id)`, `store.py:743-749`) and its folder, not
from `render_manifest`'s text, so the cut of lists over 12 entries (`report.py:30`, `:53-65`) never
applies on screen and the command line's output is untouched. Top to bottom:

1. **Heading** (the label) and **one sentence**. For a prior, simulation set, posterior,
   observation and calibration: "A prior: " and the glossary's sentence (10.8); a posterior with
   `amortized` false adds "This one was narrowed around one observation and is valid only near
   it." For a check: the Checks part's sentence for it (9.1), found by `body.diagnostic` and
   `body.variant`. Otherwise:

   | kind | sentence |
   |---|---|
   | inference result | "An inference result: the parameter estimates a posterior gave for one observation." |
   | measurement | "An effective-temperature measurement of one simulated cell." |
   | sweep | "One half of a sweep study: a cell's effective temperature measured while S is varied." (or "T_a/T"; `body.points.param`, `cross_validation.py:379`) |
   | comparison | "A comparison of saved effective-temperature measurements on one set of axes." |

2. **State**: marker, word and count ("unfinished: 4,200 of 10,000 batches saved"), and "In use in
   Parameter inference" when the session holds it.
3. **Facts**, a ruled table: "Made", with what the date means ("written at the end of its run" for
   a prior, a posterior and an observation, whose writers open only for the final write,
   `orchestrator.py:726`, `:1420`, `:371`; "simulating started" for a simulation set, whose
   manifest is first written before its first simulation, `training_checkpoint.py:216-219`,
   `store.py:1575`; "run started" otherwise);
   "Id"; "Kept in", the folder's path as selectable text, with "Copy path" (a choice the owner has
   not seen; opening the folder was declined on 2026-09-17, `artifact_screen.py:16-18`, and is not
   added); "Made by PRISM", the revision and whether the code had uncommitted changes
   (`Manifest.prism`); "Made by the run", a link to that run's page (run 6.3), and "Memory limit",
   both from the run-history file whose `result` or `simulation_set` names this record (run 6.5),
   or "not recorded" when none does, as for a record made at the command line (11.9); for a
   simulation set "Size on disk", measured off the window's thread when the page opens, and
   "Took", the time from its start to its last batch, pauses between runs included
   (`body.wall_seconds`, `store.py:1565-1566`, `:1575-1579`).
4. **Result**: the block the record's own part shows after a run of that kind, built from the
   record's manifest and folder: a step's "Afterwards" without its "In use" line (8.1), Checks'
   block (9.4), or one of this part's three effective-temperature blocks (10.3).
5. **Note**: the note or "No note", with "Edit note".
6. **Settings**, in full, as ruled tables:
   - "Input files": model, bounds, cell and units file from `Manifest.inputs`, each with its path
     and "unchanged since this record was made", "changed since", "no longer there" or "was not read
     when the record was written" (a null hash, `report.py:94-108`), by hashing the file now with
     `provenance.sha256_file`, as `store.py:1552` does (a choice the owner has not seen);
   - "Parameters": one row per parameter, in order, with lower bound, upper bound and linear or
     logarithmic scale, joined from `config.param_keys`, `nd_lows`, `nd_highs`, `rescale_lows`,
     `rescale_highs`, `log_params` (`manifest.py:276-281`); all thirteen names show (W081); a
     measurement or a sweep holds values, not bounds (`param_values`, `manifest.py:231-233`), and
     shows those;
   - "Settings": every other key of `Manifest.config` under its on-screen name and unit, from
     `fields.label(key)` (5.7) and the steps part's table of a simulation set's settings (8.7); a
     key with no on-screen name is listed under its stored name, so nothing is hidden (a choice
     the owner has not seen). A test holds the keys shown equal to the keys of `Manifest.config`
     and `Manifest.body` (U17's "in full");
   - "Rotation", on a posterior: every eigenvalue, as direction and value
     (`body.transform.fisher_eigenvalues`, `orchestrator.py:1466`; W183, W184), and the matrix,
     whole, behind "Show the rotation matrix";
   - "Everything else recorded": the remaining keys of `Manifest.body` as a tree, lists whole, a
     text of several lines wrapped under its key (W133; printed raw today, `report.py:50`, `:90`).
     A simulation set's settings show once; its manifest holds them twice (`store.py:1588`,
     `:1591`; W158).
7. **"Made from"**: each parent in `Manifest.parents` as a link with its kind word. The keys are
   `prior`, `simulation`, `posterior`, `parent_posterior`, `observation` (`store.py:81-89`); the
   window keeps a copy of that table, held equal to the store's by a test. A parent no longer held
   reads "prior 20261001T152900, no longer here" (as `report.py:283-288`). A comparison lists the
   measurements it drew (`body.compared.records`, `compare.py:392-393`).
   **"Used by"**: `ArtifactStore.dependents(kind, id)` (`store.py:891-916`), each a link with why:
   "trained from it", or, for a simulation set matched by its prior's fingerprint alone, "its
   simulations were drawn from this prior" (as `artifact_screen.py:660-666`). An
   effective-temperature record also lists the comparisons that drew it, read from their
   manifests; the store does not count them as dependents (`compare.py:5-9`).
8. **Figures**: every file in `Manifest.figures` (`store.py:397`, `:624`), titled from its file name
   (`plot_watcher.py:36-39`), in the anatomy part's figure viewer (5.9). A stored figure is a light
   picture and is shown as one in both themes (U23, B18); one an earlier PRISM stored in the dark
   theme stays as it was drawn. A simulation set says "A simulation set has no figures." (W085;
   `record_view.py:105-117`, `:211-227`.)
9. **Log**: all of `log.txt`, in the fixed-width face (B19), by `read_log(kind, id)` with no limit
   (`store.py:757-799`; today the last 1 MiB, `artifact_screen.py:55`, `:389-395`; W083), read off
   the window's thread when the section opens. Above it "Find in the log" (Enter next, Shift+Enter
   previous, "3 of 17"), "Copy all" and "Times are local, hours:minutes:seconds, without a date."
   (`runs.py:59-60`; U18 leaves the dates for the audit). With no log the page says why (kept:
   `artifact_screen.py:57-65`); for a simulation set: "A simulation set keeps no log of its own: it
   is written batch by batch and shared by every posterior trained from it. Each run that wrote it
   kept its full text.", with a link to each run `history.runs_for` gives for the set (run 6.5).

A leftover's page: "This folder is not a record: a run left it behind without finishing.", its
path, when it was last written, and "Clear leftovers… removes it." A posterior's leftover that
holds `posterior.pt` (`orchestrator.py:1421`) adds "It holds a trained network whose record was
never written. PRISM cannot load it." (W011; the unguarded write is left for the audit). An
unreadable folder's page says "PRISM cannot read this folder's record and leaves it alone." (a
choice the owner has not seen), with the store's reason beneath as console text (B19): it is the
engine's wording (`store.py:672-677`) and no refusal, so E11 does not reword it (11.13).

#### Actions

| button | what it does | today |
|---|---|---|
| "Rename…" | every kind, a simulation set included (B26, E9): the shared dialog in its form that asks for one line of text (ruling D4; look 4.6), the name checked as typed by `assert_name_free(kind, name, allow=id)` (`store.py:815-844`), as the anatomy part's name check (5.5); "Rename" calls `ArtifactStore.rename(kind, id, name)` (`store.py:852-872`) | "Save" boxes on two tabs (`prior_tab.py:328`, `posterior_tab.py:307`; W044, W052, W053, W190) |
| "Edit note" | the note becomes a field with "Save note" and "Cancel"; Enter saves (W088); `require_note("note", text)`, then `set_note` (`refusals.py:370-387`, `store.py:874-889`); a draft survives a re-read (kept: `artifact_screen.py:531-553`) | "Note" box and "Set" |
| the next step | loads the record and opens its step (below) | none (declined, `artifact_screen.py:16-18`; granted by U17) |
| "Set up a run like this one" | the same, then fills that run's form from this record (below) | none |
| "Save report…" | "This record…" writes `render_manifest` and the whole log; "This record and what it was made from…" writes `render_lineage` (`report.py:147-174`, `:253-317`); both are the engine's text, unchanged, so lists over 12 entries are still cut in the files (a choice the owner has not seen) | "Save…", "Lineage report…" (`artifact_screen.py:404-475`) |
| "Delete…" | below | "Delete…" |

"Rename…": the check's own sentence sits under the field (5.5; in U7's words after E11), and
"Rename" is greyed until the name holds; a blank is not accepted (`store.py:853-855`). If the
folder cannot be moved because another program has it open (tolerated, `store.py:862-871`):
"Renamed. The folder keeps its old name on disk until nothing has it open; the record is found by
its new name all the same." (kept from `artifact_screen.py:75-87`). A record saved under its id by
E7 is named here. The two save dialogs open through the one file-chooser function of 13.1.4
(static calls today, `artifact_screen.py:416`, `:463`).

#### The next step

| record | button | loads | opens |
|---|---|---|---|
| prior | "Use this prior" | the prior | Prior |
| simulation set | "Train from these simulations" | as "Set up a run like this one" | Training |
| posterior | "Check this posterior"; "Run it on recordings" (narrowed: "Run it on its observation") | the posterior (narrowed: and its observation) | Checks; Inference |
| observation | "Narrow a posterior around it" | the observation | Narrowing round, which uses the posterior in use |
| calibration; a check of a posterior | "Check this posterior" | the posterior it tested (`parents["posterior"]`) | Checks, at that check's page |
| inference result | "Narrow around its observation" | its observation and the posterior that gave the result | Narrowing round |
| measurement, sweep, finished | "Compare it with others" | nothing | Compare, this record ticked |
| measurement, sweep, unfinished or checks only (neither can be compared, 10.3) | "Measure again with these settings" | nothing; fills the form | One cell or Sweep study |
| comparison | "Compare these again" | nothing | Compare, the same records ticked where they still exist |

A check with no posterior has no next step. (The buttons beyond the three U17 names: a choice the
owner has not seen.) A simulation set made by a narrowing round (`config.truncation` is set,
`core/artifacts/identity.py:120`) offers neither "Train from these simulations" nor "Set up a run
like this one": a training never continues it (8.7), and only the same round, run again, does
(C11). (A choice the owner has not seen.) An observation's record names no prior and no posterior
(its writer sets no parents, `orchestrator.py:371-385`), so `load_chain`, which needs a prior
(8.1), cannot be fed from it: "Narrow a posterior around it" is greyed, with "Load the posterior
to narrow first.", until a posterior is in use (C9), and then loads the observation alone; the
store refuses one that does not fit the settings in use (`store.py:1449-1463`). (A choice the
owner has not seen.)

A button that loads asks one question up front, then loads and opens (B4, C9); nothing is left
"Changed, not applied". The question, titled for example "Load retrain_posterior and open
Checks?", carries as its list (C10):

- what is loaded: the record, and whatever its step needs that Parameter inference does not hold,
  which after a launch is everything: the model and options the record was made with, its bounds
  file, and the records it was made from, each by its label;
- what Parameter inference holds now and would release, in the phrases of `session_contents()`
  (5.3).

Answers "Keep things as they are" and "Load and open". On yes Records calls the steps part's
`load_chain` (8.1) with what the record was made with: its own options, not those of "Continue
with these" (B16, C1), because a posterior loads only under the options it was trained with
(`store.py:1343-1366`). It asks nothing more about what is let go and runs each step's own "Load
this …" in order, each showing "Loading…" at the rail's foot and leaving nothing to acknowledge
(C3). A consent is never implied (C9): the narrowed-posterior question is asked inside the chain
when that load arises (8.1, 8.4), the other-observation question at "Run inference" (8.5). A load
that is refused or fails ends the chain, the earlier ones stay (B3), and the step that failed
opens and says so. The button is greyed, with the reason beside it, when a run is live, when
something it needs is gone ("Its prior is no longer here.") or when its model is not installed.
(Loading what the step needs along with the record: a choice the owner has not seen; B4 names the
record alone, and after a launch nothing is held.)

#### "Set up a run like this one"

Asks and loads as a next step does, but loads what the run that made the record started from (for
a posterior: up to its prior, not the posterior itself), then fills that run's form through the
plans part's `fill_forms(values, source)` with the source "record retrain_posterior" (7.1.7), and
opens that step: each filled value reads "from record retrain_posterior". It starts nothing.
`fill_forms` asks nothing itself: the one question's list also carries the typed settings the fill
would replace (`would_replace`, 7.1.7). A list on the record's page then says what was filled and
what was not, with the reason. For a measurement or a sweep it fills that page's form and loads
nothing.

Never filled: a name and a note; a seed (shown beside the field: "this record used seed 123456");
any consent; the memory limit (U9); the two measured definitions. Where the record's `chi_f0` or
`chi_freq_bounds` (`manifest.py:290-291`) differ from today's: "This record was made with a
frequency band of 0.1 to 10; the band is now 0.03 to 0.3 and cannot be set here. A run set up now
will not match it."

| from | fills | read from | cannot fill |
|---|---|---|---|
| posterior | model; bounds file; kind of observation and its settings; rotation on or off; the prior; batches and simulations per batch; network, learning rate, patience; the rotation's three settings | `inputs`; `config` (`manifest.py:274-292`, `orchestrator.py:1450-1455`); `body.conditioning.chi_n_freqs` (`manifest.py:197`); `parents["prior"]` (`orchestrator.py:1432`) | a bounds file gone or changed (said, not filled); the rotation's settings where the rotation was reused, recorded as null (`orchestrator.py:1440-1454`) |
| simulation set | the same, without the network, the rotation's three settings and the count of driven recordings (`chi_n_freqs`) | `config`, the set's identity (`store.py:1588`; `core/artifacts/identity.py:87-122`); `parents["prior"]` (`orchestrator.py:1357`) | those three: no part of a set's record (its saved rotation carries no settings, `orchestrator.py:1440-1444`) |
| prior | bounds file; the seven build settings | `inputs`; `config` (`orchestrator.py:731-734`) | as above |
| measurement | model, cell file, the four numbers | `config.model`, `inputs["cell"]`, `body.settings` (`fdt_pipeline.py:60-66`) | "What to run" (a consent); the fixed settings |
| sweep | cell file, resolution, this sweep's range and points, the four numbers | `body.settings` with `preset`, `sweep_grid`, `held` (`cross_validation.py:370-374`) | the other sweep's range, which is in the twin record |

For a simulation set, Training's "Before you start" then says whether the run continues this set
and names every difference if not (E2).

#### Deleting

"Delete…" is greyed, with the reason beside it, when a run is live here; when the record is being
written here; when records were made from it ("Used by 2 records. Delete those first."); on a
leftover (W087, `artifact_screen.py:580-590`; W132, `:699-703`).

Otherwise it asks: title "Delete this posterior?" (the kind word); answers "Keep it" and "Delete
the posterior" (loses work); text, the record's label and then:

| record | sentences |
|---|---|
| prior | "This removes the prior and its figure. Nothing has been made from it. Building it again needs its bounds file, master_tier1.txt." |
| simulation set, finished | "It holds 20,480,000 simulations in 10,000 batches (10.3 GiB). 26 h 51 min passed from its start to its last batch, pauses between runs included. No posterior has been trained from it. A training with these settings would simulate all of it again." (W006; `body.wall_seconds`, `store.py:1565-1566`, `:1575-1577`) |
| simulation set, unfinished | "4,200 of 10,000 batches are saved (4.3 GiB). A training with the same settings would continue from batch 4,200; without the set it starts from zero.", then the paragraph below |
| posterior | "It is the trained network: 131 epochs on 20,480,000 simulations; the run that fitted it took 5 days 3 h. Nothing has been made from it. Its simulation set is kept, so it can be fitted again without simulating, but a fit starts from its first epoch." (`orchestrator.py:1469-1482`; the time is that whole run's, from the start of the training call, `:900`, any simulating included; the sentence about the set only where `parents["simulation"]`, `:1433-1434`, names a set still in Records) |
| observation | "It is the observation cell_a. Nothing has been made from it." |
| calibration | "It is the calibration of retrain_posterior: its verdict, its table and 4 figures. Running it again simulates 2,000 cells." (`orchestrator.py:2093`) |
| inference result | "It is the estimate retrain_posterior gave for cell_a: its samples and 3 figures." |
| check | "It is the result of this check, with 2 figures." |
| measurement or sweep, finished | "It is the measurement of master_spont at 60 frequencies, 256 trajectories each: its numbers and 4 figures. Measuring again takes as long as it did." Where it applies: "2 comparisons drew it. They keep their own figures and will show it as no longer here." |
| measurement or sweep, unfinished | "It holds 3 of 8 points and the figures it had written. A measurement cannot be continued: deleting it loses what it measured.", then the paragraph below |
| comparison | "It is a drawing of 3 saved measurements. Nothing measured is lost: comparing them again draws it again." (kept from `artifact_screen.py:137-140`) |

For every unfinished record (W007, W137; today only the effective-temperature sentence says it,
`artifact_screen.py:128-136`): "Another program may still be writing it. A run started at the
command line looks exactly like this until it finishes, and PRISM cannot tell. If one is running,
let it finish or stop it first: deleting a simulation set under a running training loses its
simulations, and that training does not say so." (the last clause is W140,
`training_checkpoint.py:371-372`, `:465-474`; under U26 a second window is no longer a case). For
a set made by a narrowing round, "A training with …" in its two rows reads "The same narrowing
round with …" (C11). The last line is always "This removes the folder <path> and everything in
it. It cannot be undone."

A record held in Parameter inference adds "Parameter inference is holding this prior. Deleting it
lets go of it there, and of the posterior and the observation loaded with it.", and on "Delete"
the session lets go, through the steps part's `SbiSession.holds(kind, id)` and `release(kind)`
(8.1; W033; today it keeps a deleted record and writes children naming it, `session.py:39-57`,
`main_window.py:331-344`). (A choice the owner has not seen.)

The call is `ArtifactStore.delete(kind, id)`, never forced (kept: `artifact_screen.py:716`), made
off the window's thread under "Deleting…", since removing a large set blocks (`store.py:937-947`;
W134). If the store then refuses because a dependent appeared meanwhile, `ask` with one button,
titled "This prior cannot be deleted yet", says "2 records were made from it. Delete those first.
PRISM never deletes a record that others were made from.", with the two as its list (C10), each by
kind word, label and why. The store's sentence, which ends "pass force=True"
(`store.py:930-933`), is not shown, and the refusal does not go through `show_refusal` (5.7).

#### Clearing leftovers

One button for all kinds, "Clear leftovers…", replaces "Sweep this kind…" and "Sweep all kinds…"
(`artifact_screen.py:499-507`) (a choice the owner has not seen). It lists, then removes, exactly
three things:

| listed as | what it is | found by | removed by |
|---|---|---|---|
| "A folder a run left behind: calibrations/_unnamed__20261003T140211, last written 3 Oct 2026, 14:02" | a folder in a kind's folder with no manifest at all | `list(kind)` rows whose reason is `NO_MANIFEST_REASON` (`artifact_screen.py:731-762`) | `sweep_incomplete([(kind, folder)])` (`store.py:1134-1166`) |
| "A file that belongs to no record: fdt/fdt3d_vs_S.png, 212 KB, last written 12 Sep 2026, 09:40" | a file directly inside a kind's folder | `loose_files(kind)` (`store.py:1051-1077`) | `remove_loose(kind, name)` (`store.py:1079-1132`) |
| "A folder an older PRISM wrote: crossval" | the one legacy folder beside the kinds' folders | `legacy_dirs()` (`store.py:79`, `:1168-1181`) | `remove_legacy(name)` (`store.py:1183-1225`) |

A posterior's leftover that holds `posterior.pt` is listed as "A folder a run left behind that
holds a trained network: posteriors/retrain_3__20261008T190211, 48 MB, last written 13 Oct 2026,
14:10" (W011). Never listed: a folder that carries a manifest, whether finished, unfinished or
unreadable (an unreadable one is named under "Left alone", with the store's reason as console
text, as on its page), nor anything inside a record's folder. The list shown is the list removed
(kept: `artifact_screen.py:887-909`).

The question: title "Clear 3 leftovers?"; as its list (C10), the lines above; as its text, where it
applies, "1 of these holds a trained network whose record was never written. Clearing removes the
only copy of that fit."; then "PRISM
will not remove anything written in the last five minutes. That is its only protection for a run
at the command line: a calibration or an inference there can go longer than that without writing.
If PRISM is running anything at the command line on this computer, keep them." Answers "Keep
them" and "Clear them" (loses work). Today's text promises that nothing still being written is removed
(`artifact_screen.py:877-881`), which the five-minute test (`store.py:58`, `:1024-1034`) cannot
keep for an idle calibration folder (W031); the guard is the engine's and stays. Afterwards:
"Cleared 2 of 3. calibrations/… was written 40 s ago and was left."

#### While a run is live

Greyed, with the lock's sentence (run 6.12) beside them: "Rename…", "Edit note", "Delete…", "Clear
leftovers…" (U17), "Set up a run like this one" and every next step, since each loads or fills and
settings are locked (U11), "Compare it with others" because a new comparison waits (U25). The test
is the run controller's flag (run 6.1; today `BasePanel._running`, `artifact_screen.py:617`,
`:682`, `:829`). Open throughout: the list, filter and search, every page, figures, logs, "Save
report…", "Copy path".

#### Kept from today's code

What depends on a record is read before anything is asked (`artifact_screen.py:692-703`); after a
change the order is re-read, tell the window, then say what happened, and only if the re-read
worked (`:555-578`); `store_changed`, on which every list of saved records, every "In use" line
and Home re-read (`main_window.py:124`, `:331-344`; shell 3.6); a note is refused, never cut
(`:492-494`). The widget names `tests/test_artifact_browser.py` reaches for change, so those tests
are rewritten first.

### 10.3 Effective temperature

Three pages under one rail entry: "One cell" (today the tab "FDT analysis", `fdt_panel.py`), "Sweep
study" ("Sweep study cross-validation", `crossval_panel.py`) and "Compare" (today a box on each
tab, `fdt_panel.py:180-213`, `crossval_panel.py:175-199`). The choice among them is this part's
(shell 3.2): a row of the three names above the page's heading, the current one marked as the
current place is in the rail, moved along with the arrow keys and reached by the part's
`show_page`; the part opens on "One cell", and the page showing is not remembered (a choice the
owner has not seen).

#### Rules the three pages share

- **Anatomy** (5.1), with, in region 2, a ruled table of the page's saved runs and the selected one
  shown under it. After a run the record just written is selected and shown, with its next step.
  The lock leaves region 2 open (run 6.12; today it covers the saved-run list too,
  `base_panel.py:323-335`; W077). On the page whose own run is live the run's view stands in the
  page's place (run 6.3); that page's saved runs are read in Records meanwhile.
- **Saved runs**: every record of the page's study, unfinished ones too, with "Name", "State",
  "Made", "Holds", "Note" (W069, W150; `artifact_picker.py:192-198`). A selected row is shown with
  the widget Records uses for a record's page (W089, W124; `record_view.py:63-74`). The table is
  re-read whenever a run ends, however it ended (today only on a result,
  `crossval_panel.py:335-349`), and on `store_changed`. `record_view.py` and
  `widgets/compare_list.py` go.
- **Result block**: one sentence ("Measured at 57 of 60 frequencies; 3 came back blank because the
  spontaneous spectrum does not reach them.", `body.offgrid`, `fdt_pipeline.py:449-450`); a table
  (highest ratio and its frequency, the ratio at the cell's own frequency, the share of usable
  frequencies, `body.results`, `fdt_pipeline.py:69-90`; for a sweep, points measured and failed);
  the notices; the figures. Unfinished: "This measurement did not finish.", what it holds, and "A
  measurement cannot be continued."; then "Show the full error", which opens the log's last line
  as console text (B19), not as the window's own words: after a stop or a failure the store
  writes that line with the error's type name (`store.py:567-570`), seen only there and in the
  log (C7). Checks only: "Only the checks were run; there is no measurement." (W070).
- **Figures** of a live run show on the run's page as they are written, through `dispatch`'s
  `watch_dir` as today (`fdt_panel.py:319-320`, `crossval_panel.py:332`); a saved record's show in
  its view, light in both themes (U23). The pile of tabs under repeated titles goes (W073).
- **Name**: optional (5.5); checked as typed with `assert_name_free("fdt", name)`. When taken:
  "Taken by an unfinished measurement of 3 Oct 2026, 14:02.", with a link to that record, where
  "Rename…" and "Delete…" free the name (W069; `fields.py:96-97`). The anatomy part empties "Name"
  and "Note" after a run that made its record (5.5); here a record exists once the run's thread
  enters its writer (`fdt_pipeline.py:308`), so they are emptied whenever the run leaves a record,
  however it ended, and "Seed" with them, the result stating the seed used. After a refusal made
  before anything was written they are kept (5.5): it leaves no record and the name free
  (`fdt_pipeline.py:277`; `store.py:434-437`, `:473-478`). (A choice the owner has not seen; today
  a name left there is refused at the next click and a seed left there repeats the run,
  `fdt_panel.py:326-336`.)
- **Start button**: greyed with the reason beside it: "Choose a cell file."; the sentence of
  `registry.fdt_support(model)`; the lock's sentence (W078; `fdt_panel.py:256-263`, `:266-269`).
  The model is judged again when the list of models changes (W151). That engine sentence says
  "FDT" today (`core/registry.py:101-112`) and is also the refusal the builder raises
  (`cli.py:401-405`); E11 rewords it to "effective temperature" although it sits outside any
  raise (C23; 11.13).
- **Stopping**: the run part's stop (6.6) with these words:

  | page | button | question | text | answers |
  |---|---|---|---|---|
  | One cell | "Stop measurement…" | "Stop measurement?" | "It has run for 1 h 12 min. The record cell_a_3oct is kept, marked unfinished, with what it has measured: the spontaneous half and 2 figures. A measurement cannot be continued; to finish it, it is measured again from the start." | "Keep measuring"; "Stop and keep what is measured" (loses work) |
  | Sweep study | "Stop sweep study…" | "Stop sweep study?" | "It has run for 5 h 40 min. The S sweep is finished and saved. The T_a/T sweep has 3 of 8 points and is kept, marked unfinished; it cannot be continued." | "Keep measuring"; "Stop the study" (loses work) |

  Today one click on "Cancel" ends either (`base_panel.py:291-300`; W005). The run page says how a
  run ended; the line "FDT run finished." (`fdt_panel.py:253`; W120) goes.
- **Remembered** (7.3.2): the keys of group `fdt` (`fdt_panel.py:338-348`) and group `crossval`
  (`crossval_panel.py:351-361`) as today, written when the main button starts a run, as the values
  that run used, and `record` when a saved run is opened; nothing at closing (B6). Never, as now:
  seed, name, note, "What to run", anything on Compare. A rescan of the cell list keeps the
  selection (7.3.1, rule 3; W029).
- **For the run's `spec`** (6.2). One cell: "Measurement"; stages "Checks" (left out after
  `Skipping sanity checks.`, `fdt_pipeline.py:336`), "Spontaneous" (from `Campaign 1: …`, `:357`),
  "Driven" (from `Campaign 2: …`, `:464`). Sweep study: "Sweep study"; stages "S sweep" and "T_a/T
  sweep", each from its `--- Phase A (… sweep)` line (`cross_validation.py:415`), with a count of
  points from `[A i/n]` and `[B i/n]` (`:419`, `:497`). Compare: "Comparison", no stages.
  The run part takes these lines through `spec` (6.2), and its history's `result` holds both
  records of a sweep study (6.5). Each sweep passes its points twice, `[A i/n]` for the
  spontaneous half and `[B i/n]` for the driven one, and a point is measured only when its driven
  half lands (`cross_validation.py:531`, `:542`): the count names the half it counts, and "points
  measured" is read from the driven pass alone.

#### One cell

Sentence: "Measure how far one simulated cell is from thermal equilibrium. Its spontaneous
fluctuations are compared with its response to a small drive, frequency by frequency; the result
is the ratio T_eff/T, which is 1 at equilibrium." Stage: `run_fdt` (`fdt_pipeline.py:246-317`) on
a configuration from `cli.make_fdt_config` (`cli.py:380-425`).

| Place | new label | today's label; key | built-in value | argument |
|---|---|---|---|---|
| M | "Model" | "Model"; `model` | NADROWSKI (`fdt_panel.py:124`) | `make_fdt_config(model, registry.state_dep_drift(model), …)` |
| M | "Cell file" | "Cell"; `cell` | none chosen | `cell_file` |
| M | "Drive frequencies" | "n_freqs"; `n_freqs` | 60 | `n_freqs` |
| M | "Trajectories per frequency" | "M_ensemble"; `ensemble_m` | 256 | `ensemble_M` |
| M | "What to run": "The checks, then the measurement" / "The measurement only" / "The checks only" | tick boxes "Skip sanity checks" and "Proceed to the production sweep after sanity" (`fdt_panel.py:144-148`) | the first | `run_fdt(skip_sanity=, confirm_production=)`: False, True / True, True / False, False |
| R | "Frequencies simulated together" | "freqs / batch"; `freqs_per_batch` | 1 | `freqs_per_batch` |
| R | "Drive amplitude F₀ (non-dimensional)" | "F0 (ND forcing amplitude)"; `f0` | 0.05 | `F0` |
| R | "Seed" | "Seed"; `seed` | blank: one is drawn and recorded | `seed`, to the builder and to `run_fdt` |
| F | "Frequency range", "Settling time", "Driven length", "Time step", "Spontaneous length" | shown nowhere | 0.1 to 30 times the cell's own frequency; 100; 30 drive periods; 0.01; 8,000 (`sim_config.py:611-618`) | none; no front end sets them (`cli.py:354-377`) |
| N | "Name" | "Record name" | blank | `default_store().create("fdt", cfg, name=)`, at the click (`fdt_panel.py:301-302`) |
| N | "Note" | "Note" | blank | `note=`, after `require_note` |

"What to run" replaces two tick boxes whose fourth combination shows "do not proceed" beside a run
that proceeds (`fdt_pipeline.py:292-297`; W157). It is a consent: it opens on its first choice at
every launch. (A choice the owner has not seen.)

"Before you start", recomputed as the form changes (W072; today nothing is stated):

- "Simulates the cell's spontaneous motion once, 256 trajectories, then its motion driven at 60
  frequencies, 256 trajectories each: 61 sets, 15,616 trajectories."
- "First runs 5 checks of the method on this cell." (3 for a model other than Nadrowski,
  `sanity.py:372-382`), or "The checks are skipped.", or "Runs the 5 checks only. No measurement
  is made and nothing can be compared."
- "Runs on the processor, not the graphics card." (`cli.py:420`); so no page of this part carries
  a memory line (C13).
- "The last finished measurement with these settings took 2 h 14 min (cell_a, 3 Oct 2026, 14:02)."
  or "No finished measurement with these settings is saved, so PRISM cannot say how long this
  takes." The engine records no duration for these runs (`manifest.py:54-55`); the window takes
  the time from the record's "Made" to the last writing of its `log.txt`, which the final save
  writes (`store.py:548-550`) and a rename or a note does not. (A choice the owner has not seen.)
- "It can be stopped at any time. What is measured by then is kept in a record marked unfinished;
  a measurement cannot be continued."

Main button "Start measurement". The click keeps today's order (`fdt_panel.py:277-324`): each
field read with `value_or_none`, the configuration built, the note judged, the writer created
(which claims the name), then the writer handed to the stage, which enters it on the run's thread.

#### Sweep study

Sentence: "Measure the effective temperature of a Nadrowski cell again and again while one
parameter is stepped from its equilibrium value to the cell's own: S from 0, then T_a/T from 1."
Stage: `run_param_study_cli` (`cross_validation.py:595-673`) on `cli.make_param_sweep_config`
(`cli.py:487-550`). It writes two records, one per sweep.

| Place | new label | today's label; key | built-in value | argument |
|---|---|---|---|---|
| F | "Model: Nadrowski. The study varies two of that model's parameters." | "Model is fixed to NADROWSKI." (`crossval_panel.py:151`) | | fixed in the builder (`cli.py:530`) |
| M | "Cell file" | "Cell"; `cell` | none chosen | `cell_file` |
| F | "This cell: S = 0.1234, T_a/T = 1.5000" | "Cell values" | from `cli.parse_cell` (`crossval_panel.py:239-251`) | |
| M | "Resolution": "Exploratory" / "Production" | "Preset"; `preset` | Exploratory | `preset=dict(cli.SWEEP_PRESETS[name])`, `preset_name=name` |
| M | "S sweep (T_a/T held at 1)": "From", "To", "Points" | "S grid  (T_a/T = 1)": min, max, n; `s_grid` | 0 and the cell's S, the lower as "From"; 8 or 12 points | `s_spec=(from, to, points)` |
| M | "T_a/T sweep (S held at 0)": "From", "To", "Points" | "T_a/T grid  (S = 0)"; `t_grid` | 1 and the cell's T_a/T, the lower as "From" (kept: `crossval_panel.py:252-258`; the builder refuses a "From" not below its "To", `cli.py:483`); 8 or 12 | `t_spec` |
| R | "Drive frequencies", "Trajectories per frequency", "Frequencies simulated together", "Drive amplitude F₀ (non-dimensional)", "Seed" | as on One cell | as on One cell, but 30 or 60 drive frequencies | as on One cell |
| F | "Set by the resolution": frequency range, driven length, spontaneous length | shown nowhere (W071) | Exploratory 0.2 to 30, 20 periods, 4,000; Production 0.1 to 30, 30 periods, 8,000 (`cli.py:455-460`) | inside `preset` |
| N | "Name" | "Record name" | blank | `store.create("fdt", cfg, name=)` twice, as `<name>-s` and `<name>-temp` (`crossval_panel.py:305-307`) |
| N | "Note" | "Note" | blank | `note=` on both |

Values that follow another choice (W071, W029). The four numbers a resolution sets ("Points"
twice, "Drive frequencies", "Trajectories per frequency") carry the remark "from Exploratory", the
two "From" and "To" pairs "from the cell", in the origin's place on the row (5.2). Changing
"Resolution" or "Cell file" replaces only values still so marked; a typed value reads "changed" and
is kept, and a line says what happened: "Production set Points to 12 and Drive frequencies to 60.
Kept your Trajectories per frequency, 128." (Today all are overwritten, `crossval_panel.py:256-265`.
A choice the owner has not seen.)

The name is checked for both records as typed: "retrain-s is taken by a sweep of 2 Oct 2026,
09:15."

"Before you start": "Two sweeps of 8 points each. Every point simulates the cell's spontaneous
motion once and its motion driven at 30 frequencies or more, 256 trajectories each: at least 496
sets, 126,976 trajectories." (one grid is built to cover every point's own frequency, never smaller
than the number asked for, `cross_validation.py:115-142`); "Runs on the processor."; the time, as
for One cell, from the last finished study at the same resolution and counts; "Each finished sweep
is saved as its own record. Stopping during the second keeps the first; a sweep cannot be
continued."

Main button "Start sweep study"; the click keeps today's order (`crossval_panel.py:275-333`).

#### Compare

Sentence: "Draw saved measurements on one set of axes. Nothing is simulated; the comparison is
saved as a record of its own." Stage: `compare(mode, refs, name=, note=, **options)`
(`compare.py:438-491`), started as a run, as today (`fdt_panel.py:234`, `crossval_panel.py:218`).

| new label | today's label; key | built-in value | argument |
|---|---|---|---|
| "What to compare": "Several cells" / "Repeats of one cell" / "One measurement, another normalisation constant" / "Two sweeps" | "Comparison" (cells, repeats, renormalise) on one tab; the button "Compare saved sweeps" on the other | "Several cells" | `mode`: "cells", "repeats", "renormalise", "sweeps" |
| a ruled table of the saved measurements (or sweeps), a tick on each row | "Runs to compare" / "Sweeps to compare", filled by "Add selected", "Remove", "Clear" (`compare_list.py:25-32`); `compare_records` | none ticked | `refs`: the ticked ids, in table order |
| "Normalisation constant" (third choice only) | the same; `prefactor` | blank | `prefactor=` |
| "Slice at" (fourth choice only) | the same; `slice_at` | blank: the middle of the range the sweeps share | `at=`, left out when blank |
| "Name", "Note" | "Comparison name", "Comparison note" | blank | `name=`, `note=` |

(One page, one choice of four and a tick on each row: a choice the owner has not seen.)

The table lists what the stage accepts: records of the right study, finished and holding their
numbers (`compare.py:238-250`). The rest are counted under it: "2 unfinished and 1 checks-only
measurement cannot be compared." Main button "Compare", greyed with the reason until the count
fits (`compare.py:64-69`): "Tick at least 2 measurements.", "Tick exactly 1 measurement.", "Tick
exactly 2 sweeps." (W126.) A ticked record deleted in Records leaves the
table, and a line says so. In region 2: "Saved comparisons". Nothing here is remembered (U15).

**While a run is live** (U25, B15) saved measurements, sweeps and comparisons stay open on all
three pages: tables, record views, figures, logs. A new comparison waits: the form is locked with
the others and beside "Compare" stands "A new comparison can be made when the run ends: Training
retrain_posterior is live."

### 10.4 Live simulation

Today `simulate_panel.py`, `simulate_runner.py`, `simulate_export.py`,
`widgets/live_hair_bundle.py`; kept, on the shared anatomy and look. Sentence: "Watch a simulated
hair bundle move. The settling-in time is simulated before the recording length and is shown
marked. Nothing is saved unless you save a video." (the middle sentence: ruling D2).

| Place | new label | today's label; key | built-in value | argument |
|---|---|---|---|---|
| M | "Model" | "Model" | NADROWSKI (`simulate_panel.py:74`) | `build_stream_config(model, cell)` (`simulate_runner.py:40-76`) |
| M | "Cell file" | "Cell"; `cell` | none chosen | the same |
| M | "Recording length", s: the key's one label on every page (ruling D2) | "T_obs (s)"; `t_obs`, stored as `tobs` (`simulate_panel.py:181`) | 1.0 (`config.T_MIN_EXP_S`); the length of a simulation that is only watched, so B12 is read as not reaching it and it stays remembered (7.3.2) (a choice the owner has not seen) | `run_simulation_stream(cfg, t_obs_s, …)` (`simulate_runner.py:187-188`) |
| R | "Simulation steps per frame" | "Steps / frame"; `frame_steps` | 2000 | `frame_steps` |
| R | "Frames per second, at most" | "Max FPS"; `fps` | 30 | `fps` |
| R | "Seed" | none (W074) | blank: a new stream each time | a new keyword `seed` of the window's runner, which wraps its loop in `core.rng.seeded(seed, device)` (`rng.py:82-107`) and draws one when blank; never remembered (a choice the owner has not seen); a test holds that two streams at one seed give the same trace |

The fixes the findings name:

- **The settling-in stretch is marked** (W074). The stream shows the settling steps before the
  recording (`simulate_runner.py:89`, `:128-131`) on one time axis that starts at the first sample
  (`:262`). New: time is zero where the recording starts; the settling stretch runs at negative
  times in the secondary ink; a line at zero carries "settling in" on its left and "recording" on
  its right, in the live view and in the saved video (a choice the owner has not seen; an axis and
  its lettering, U4).
- **Colours** (U23). The live view follows the theme, reading the look's table as now
  (`widgets/live_hair_bundle.py:109-122`). The saved video follows the light figure settings like
  any stored figure (C20; a choice the owner has not seen): it stays on a white page (kept:
  `simulate_export.py:91-116`) and its trace, which names no colour (`:100`), takes those
  settings' ink in both themes (B18), so its code needs no change (look 4.9).
- **Progress** (W074): the runner reports its plan once, settling samples and recording samples
  (the stages "Settling in" and "Recording"), and the run page shows samples done against the
  total, through `RunView.set_count(done, total, unit)` (run 6.3.3), in place of a pulsing bar.
  A Live simulation leaves no history file and no text file, and neither does a video export (C3;
  run 6.5; a choice the owner has not seen).
- **"Before you start"**: "Shows a settling-in stretch, then 1 s of recording, at up to 30 frames
  a second. The settling-in stretch depends on the cell; its length is stated when the simulation
  starts."; "Runs on the processor."; "While it plays, no other run can start." (today a label in
  the form, `simulate_panel.py:93`).
- **Start button** "Start simulation", greyed with "Choose a cell file." when the model has none,
  judged when the page is built, when the model changes and after a rescan (W127;
  `simulate_panel.py:56-66`, `:173-175`). The configuration is built with `hw=cpu_device()` passed
  to `cli.make_sim_config` (`cli.py:273-279`), so the click no longer asks the graphics card
  (`simulate_runner.py:63`, `:75`).
- **Stopping**: "Stop simulation" stops at once and asks nothing, because nothing is lost (B24, a
  choice the owner has not seen); it takes effect within a frame and has no "Don't stop after all"
  (run 6.6). A video being written is different, since its stop loses the file so far: "Stop
  saving…" asks "Stop saving the video?", "The unfinished file is removed. The trace stays and can
  be saved again."
  (`simulate_export.py:151-160`), "Keep saving", "Stop saving" (loses work).
- **Result**: "Showed 0.31 s of settling in and 1.00 s of recording of master_spont (Nadrowski),
  seed 123456.", in place of "Simulation complete."; after a stop, "Stopped after 0.31 s of
  settling in and 0.42 s of recording. The trace shown is kept and can be saved as a video." "Save
  video…" stays; its file chooser opens through the one function of 13.1.4 (a static call today,
  `simulate_panel.py:139-140`).
- Remembered (7.3.2): group `simulate` as today (`simulate_panel.py:177-184`), written at the
  start click.

### 10.5 Reduction map

Today `reduction_panel.py`; its science and its files are out of scope. Sentence: "Reduce a
Nadrowski cell to the Hopf normal form and show how the result moves as f_max is varied. Nothing
is simulated."

| Place | new label | today's label; key | built-in value | argument |
|---|---|---|---|---|
| F | "Model: Nadrowski. The map is a property of that model's equations." | the same statement (`reduction_panel.py:58`) | | |
| M | "Cell file" | "Cell"; `cell` | none chosen | `cli.make_reduction_config(cell, F0=)` (`cli.py:428-444`) |
| M | "Drive amplitude F₀ (non-dimensional)" | "F0 (ND forcing amplitude)"; `f0` | 0.05 (`reduction_panel.py:53`) | `F0` |

"Before you start": "Computed from the cell's parameters; no simulation."; "Writes a table and a
figure into the reduction folder. They are files, not records: they carry no name, note or
settings and do not appear under Records." Main button "Compute the map" (`run_reduction_map(cfg)`,
`sweep.py:162-201`). "Stop computing…" asks "Stop computing the map?", "The map will not be shown.
If its table is already written, that file stays in the reduction folder.", "Keep computing",
"Stop computing" (U17 has this stop ask; B24 lifts it for Live simulation alone). A stop takes
effect at the engine's next printed line, and one follows the table's write (`sweep.py:190-191`),
so the text cannot say that nothing was written.

Result on the page (W079, in part): the values at the cell's own parameters as a ruled table, from
the `ReductionRecord` the stage returns, which today is kept and shown nowhere
(`reduction_panel.py:92-94`) while its report is only printed (`sweep.py:126-159`); this run's
figure, which the engine draws, light in both themes (B18); "Saved as", with the two files' full
paths and "Copy path" (`sweep.py:186-190`, `:197`). The stage returns the record alone (`:201`),
so the window finds the two as the table and the figure written in the reduction folder since the
run began. Below, "Earlier results": the files
`reduction_sweep_<stamp>.png` in the reduction folder, newest first, dated from the stamp in their
names (`core/Reduction/plots.py:99-104`); other files there are not listed (a choice the owner has
not seen). Only this run's figure is the result, which ends the pile of tabs all titled "reduction
sweep" (W073). Remembered (7.3.2): group `reduction` as today (`reduction_panel.py:96-100`),
written at the click.

Engine: making a reduction a record is outside this redesign; and `make_reduction_config` takes no
device, so its click still asks the graphics card (`cli.py:443`; W127). Neither is designed in.

### 10.6 Your own models

Today `screens/model_builder_screen.py`, reached only through Settings
(`main_window.py:143-146`). New: a rail entry (U5), with two parts on one page.

**"Your models"**, a ruled table: name, variables, parameters, and what each can be used for
("Live simulation"; "Effective temperature", by `registry.fdt_support`; "Parameter inference", by
`registry.is_sbi_user_model`), with "New model", "Edit" and "Delete…". Files that failed to load
are listed under it with the reason (today `settings_screen.py:178-182`). The list moves here from
Settings so the models and their builder are in one place. (A choice the owner has not seen.)

**The builder**, its steps kept in order (`model_builder_screen.py:251-330`):

| step | fields (new label; today's) | button |
|---|---|---|
| 1 "Name and variables" | "Name" ("name"); "Variables" ("variables") | "Set variables" |
| 2 "Equations", a block per variable headed "dx/dt" | "Right-hand side" ("drift"); "Noise strength D" ("D"); "Starting value" ("init"); "Forcing" ("forcing") and its own fields | |
| 3 "Parameters" | per parameter "Value" ("value"), "Automatic bounds" ("auto"), "Lowest" ("min"), "Highest" ("max"), "Scale" ("box": linear, log) | "Find parameters" ("Detect parameters") |
| 4 "Scales" | "Length scale (nm)" ("x_scale (nm)"); "Time scale (s)" ("t_scale (s)") | |
| 5 the pinned bar | the result line, then "Check the model" ("Validate") and "Save model" | |

What the buttons do is unchanged: `_set_variables`, `_detect_params`, `_validate`, `_save`
(`model_builder_screen.py:333-534`). "Back to settings" goes.

- **The result line sits in the pinned bar**, left of the two buttons (W075; today inside the
  scrolled form, `model_builder_screen.py:286-287`, `:309`). It reads "Not checked yet.", "DUFFING
  is valid: 2 variables, 5 parameters. Ready to save.", "Saved DUFFING. It is now in every list of
  models.", or the problem.
- **Unsaved edits** (W076). The form is unsaved when it differs from what was last loaded, saved
  or started blank. "New model", and "Edit" on another model, then ask first
  (`main_window.py:254-256`, `:274-280`). Title "Throw away the unsaved changes?"; text "DUFFING
  has changes that are not saved." (or "This new model has not been saved."); answers "Keep
  editing" and "Throw them away" (loses work). Leaving by the rail does not ask: only `reset` and
  `load_existing` replace the form (`model_builder_screen.py:537-569`).
  Closing the window throws unsaved edits away too, so it asks the same question (U17), after any
  question about a live run (C8): the close is the run part's (6.7), the words are these.
  (That "Keep editing" then keeps the window open: a choice the owner has not seen.)
- **Saving over a model** asks (a choice the owner has not seen); today the model being edited is
  overwritten silently (`model_builder_screen.py:520-524`). Title "Save over DUFFING?"; text "This
  rewrites the model and the three files PRISM generates for it: its bounds file and its cell
  file, each named default.txt, and its units file. Other files in its folders are not touched.
  Records already made with DUFFING stay as they are; PRISM cannot tell them from records made
  with the new equations." (`model_store.py:384-399`); answers "Keep editing" and "Save over it"
  (loses work).
- **"New model" starts from the built-in scales**, 10 and 0.01; `reset` leaves the last model's
  (`model_builder_screen.py:556-569`; W156).
- **"Delete…"** says what goes (W152): `delete_user_model` removes three whole folders and the
  definition (`model_store.py:403-416`), where today's question speaks of "generated" files
  (`main_window.py:288-294`). Title "Delete the model DUFFING?"; text "This removes its definition
  and its three folders with everything in them: its bounds files (2), its cell files (3) and its
  units file. 3 of these were not written by PRISM: slow.txt, fast.txt, wide.txt. Records made
  with DUFFING are kept, but can no longer be used in a new run."; answers "Keep it" and "Delete
  the model" (loses work). The counts are read from the model's folders under
  `config.BOUNDS_PATH`, `CELL_PATH` and `UNITS_PATH`.
- Greyed while a run is live, with the lock's sentence: "Check the model", "Save model", "Delete…"
  (run 6.12; kept: `model_builder_screen.py:394-398`, `:510-512`; `main_window.py:283-287`); the
  form stays usable.
- After a save or a delete every built page that lists models judges its model again (W151),
  through `_refresh_model_combos` (`main_window.py:310-329`; shell 3.10); a page built later reads
  the list when it is built.

### 10.7 Settings

Today `screens/settings_screen.py`, reached from the gear's menu (`main_window.py:212-213`). New: a
rail entry with two headed sections and no boxes.

- **"Appearance"**: three radio buttons (U13), "Follow Windows", "Light", "Dark", each calling
  `MainWindow.set_appearance_mode(mode)`, which stores `appearance/mode` and applies it at once
  (shell 3.9, which also lists what leaves `theming.py` and `settings.py`; W101). It can be
  changed while a run is live, as can the keep-awake switch; every other setting is locked (C29;
  run 6.12; a choice the owner has not seen: U11 locks settings during a run). Gone from the page
  with "Auto (time of day)": "Use the Windows accent colour", "Use the Inter font everywhere" and
  its caption
  (`settings_screen.py:87-102`). One line: "Text size follows the text size set in Windows.",
  shown once the look part has made it true (B20; look 4.8; W102).
- **"This computer"**: the memory limit and the keep-awake switch, as part 7 defines them (7.2).

The list of your own models moves to that part's page (10.6) and the Help text to Help.

### 10.8 Help

A rail entry. It also opens from F1 at the section for the page showing (shell 3.7), from the
"Help" button of a page's heading at that page's section (5.1), and from "More in Help" in a
setting's explanation at that setting's entry (5.8). Today Help is a box in Settings built from
docstrings and `HELP` dictionaries under their own keys (`settings_screen.py:23-29`, `:184-205`;
W103, W125).

New: the text is written for the reader and held in one table in the window's code, an entry per
section and an entry per setting, keyed by field key so that every heading is the on-screen label
(`fields.label(key)`, 5.8). It replaces the six `HELP` dictionaries
(`panels/inference/help_text.py:7`, `fdt_panel.py:32-63`, `crossval_panel.py:35-60`,
`simulate_panel.py:33-43`, `reduction_panel.py:28-31`, `model_builder_screen.py:40-60`), whose
texts are the starting text, reworded into U7's words. This part writes the entries for 10.3 to
10.6. Those for the settings of parts 7, 8 and 9 are written in the Help stage from those parts'
tables and words: the memory limit and the keep-awake switch from 7.2.5 and 7.2.6, the
explanations the steps give once (8.2 to 8.6) word for word. Three stale texts go (W115): the
observation entry's claim of a comparison nothing makes (`help_text.py:22-25`), the hour figures
in the network-width entry (`:87-89`), and the advice to leave the memory limit off (`:47-48`;
W018).

#### Sections

1. "What PRISM does": four sentences. Then "Where to start", the longer words the first-time
   opening leaves to Help (B5): "A live simulation takes seconds on the processor and saves
   nothing unless you save a video. Parameter inference trains a network on simulated hair
   bundles, checks it, then runs it on your recordings: the training takes days on a graphics
   card, the steps before it minutes. An effective-temperature measurement runs on the processor.
   Whatever you choose, PRISM says what it will cost before it starts, and everything it writes
   appears under Records."
2. "Getting around": the rail, the run's state at its foot, "Continue with these" on Home (U24),
   and the keys, printed from `keys.KEYS` (shell 3.7).
3. One section per part, in the rail's order; Parameter inference with a paragraph per step,
   Effective temperature with one per page. Each opens with the line Home gives for the part
   (shell 3.6), says how long its runs take and what it writes, and lists its settings with their
   explanations.
4. "Runs": one at a time, in one PRISM (U26); what stopping keeps and loses, by kind of run; what
   closing the window, or shutting Windows down, does during a run (U12, U27); where a run's full
   text and history are kept (B8) and which runs leave none (C3); and "A record's log is what
   PRISM itself reported while making it. The full text of a run also holds its progress lines."
5. "Your files": bounds, cell and units files, models, plans (a folder per model, B13); where
   records are kept.
6. "Glossary".

"Find in Help" searches the whole page. The first-time opening itself is the shell part's and
final (B5; 3.6); its fourth choice opens Help at section 1.

#### Glossary

One sentence each, for the science terms; a term a step explains once on its page (8) has the same
words here. The machinery's words (record, run, plan, memory limit, "Before you start", the full
text) are explained in sections 3 to 5, in the words of the parts that own them.

| term | explanation |
|---|---|
| bounds file | A text file that lists the parameters to be estimated, in order, with the lowest and highest value each may take; it also settles which kind of observation is used. |
| cell file | A text file with one cell's parameter values, used to simulate that cell. |
| non-dimensional | Written in the model's own units of length, time and force, not in nanometres, seconds and newtons. |
| prior | The range of parameter values a training draws from: the bounds file's limits, cut down to the values for which the model is stable. |
| observation | What is measured from one cell, real or simulated, reduced to the numbers a posterior reads. |
| driven recording | A recording of the bundle while it is pushed sinusoidally at one frequency. |
| unusable driven recording | A driven recording too short or too slow to cover enough cycles of its drive; PRISM leaves it out and counts it. |
| frequency response, χ(ω) | How strongly, and with what delay, the bundle follows a small sinusoidal push at frequency ω; an observation of this kind is one spontaneous recording and several driven recordings, each at one frequency. |
| slots | The most driven recordings one observation can hold; fixed in a posterior once it is trained. |
| lock-in limit | The longest stretch of a driven recording, in drive cycles, used to measure the response. |
| training | The run that makes a posterior: it works out the rotation, simulates the cells, fits the network to them and saves it. |
| simulation set | The simulated cells a training or a narrowing round has made, saved batch by batch; the same run with the same settings continues from it instead of simulating again. |
| rotation | A change of coordinates, worked out before simulating from how strongly the observations respond to each parameter, so that the network learns nearly independent combinations of the parameters. |
| fit | The part of a training in which the network is adjusted to the simulation set, one pass (an epoch) after another; a stopped fit cannot be resumed. |
| posterior | The trained network that turns an observation into parameter estimates with their uncertainty. |
| calibration | A check of whether a posterior's stated uncertainty is honest: it is run on many simulated cells whose true parameters are known, and passes when the truth falls where the posterior says, as often as it says. |
| narrowing round | A second training for one observation only: the prior is cut down to the region the posterior in use finds believable for that observation, and training is repeated there (truncated sequential neural posterior estimation, TSNPE); its posterior is valid only near that observation. |
| trajectory | One simulated run of the bundle through time; a measurement averages many. |
| effective temperature | The ratio T_eff/T that compares a bundle's spontaneous fluctuations with its response to a small drive at each frequency; it is 1 at thermal equilibrium, and its distance from 1 measures how active the bundle is. |
| sweep study | The effective temperature measured again and again while one parameter of a cell is stepped from its equilibrium value to the cell's own. |

#### The "?" beside a setting

Every setting in the tables of 10.3 to 10.6 has an entry in the help table, and its "?" opens that
entry in the anatomy part's panel (5.8): the same text Help lists under the part, short enough to
read there (W051; `help_badge.py:24-33`). A test walks the built pages and fails on a setting
whose key has no entry.

### Choices in this part the owner has not seen

Most consequential first. Items 5, 12, 26 and 36 rest on rulings of part C (C3, C20, C9, C29), 17
in part on C8, and 28 on B24.

1. A next-step button on a record's page also loads whatever its step needs and is not loaded yet:
   the record's model and options (its own, not the built-in ones), its bounds file and the
   records it was made from.
2. The words for the kinds of record that U7 does not give ("inference result", "check",
   "measurement", "sweep", "comparison") also go into what the command line prints, when the
   engine's refusal sentences are reworded.
3. Deleting a record that Parameter inference holds asks with one more sentence, then makes it let
   go; today the session goes on holding the deleted record.
4. One "Clear leftovers…" button for all kinds replaces the two sweep buttons.
5. A Live simulation and the saving of a video leave no run-history file and no full-text file,
   and neither does a run that only loads a record, as each load behind a next-step button is;
   this narrows U17's "one small file per run".
6. A simulation set made by a narrowing round offers neither "Train from these simulations" nor
   "Set up a run like this one", because a training never continues it; its delete question
   speaks of the round, not of a training.
7. A measurement's likely length is taken from the last finished one with the same settings, as
   the time from its making to the last writing of its log.
8. The measurement page's two tick boxes become one choice of three, "What to run", which opens on
   "The checks, then the measurement" at every launch.
9. Comparing is one page: one choice of four kinds and a tick on each saved measurement, in place
   of an add-and-remove list on two tabs.
10. In Live simulation time is counted from the start of the recording; the settling-in stretch is
    drawn before zero and labelled, on screen and in the saved video.
11. Live simulation gains a "Seed" field; blank means a new stream each time, and it is never
    remembered.
12. The saved video is drawn with the light figure settings like any stored figure, in both
    themes: a white page, and its trace in the page's ink where it is blue today.
13. When an effective-temperature run leaves a record, finished or not, Name, Note and Seed are
    emptied and the result states the seed used; after a refusal that leaves no record they stay.
14. On Sweep study a typed number survives a change of resolution or cell file; only numbers still
    following them are replaced, and a line says which.
15. A measurement that ran only its checks has its own state, "Checks only", and cannot be
    compared.
16. Saving over an existing model asks first and says that records made with the earlier equations
    cannot be told from new ones.
17. Closing the window with an unsaved model asks "Throw away the unsaved changes?" after any
    question about a live run, and "Keep editing" keeps the window open.
18. The list of your own models moves from Settings to "Your own models".
19. The three Effective temperature pages are chosen from a row of names above the page, which
    opens on "One cell" every time.
20. A record with no name is called by its kind and date, and a simulation set with no name by its
    prior.
21. In a narrow window a record's page takes the place of the list, with "All records" to return.
22. A record's page shows its folder with "Copy path" and has no button that opens the folder.
23. A record's page says whether each input file is unchanged, changed or gone, by reading the
    file again when the page opens.
24. "Save report…" writes the command line's text, so lists over 12 entries are still shortened in
    the saved file.
25. The next-step buttons beyond the three the owner named: "Use this prior", "Train from these
    simulations", "Run it on its observation", "Narrow a posterior around it", "Narrow around its
    observation", "Compare it with others", "Measure again with these settings", "Compare these
    again".
26. "Narrow a posterior around it" is greyed until a posterior is in use, because an observation's
    record names none.
27. Live simulation's "Recording length" stays remembered: B12's "recording lengths" is read as the
    Inference step's alone.
28. Live simulation's stop asks nothing, because nothing is lost; stopping the saving of a video
    does ask.
29. The plain-word labels of the settings on these pages, for example "Drive frequencies",
    "Trajectories per frequency", "Right-hand side".
30. The sentences of this part's questions (deleting each kind of record, clearing leftovers,
    loading from a record, stopping a measurement or the saving of a video, a model's three
    questions) and of the glossary.
31. The Reduction map lists earlier results as its own figures in its folder and leaves the
    folder's other files unlisted.
32. A folder of the kind the live run writes, changed since that run began, reads "Being written"
    even when a command-line run is writing it.
33. An unreadable folder gets a sentence of the window's, with the store's reason beneath as
    console text; an unfinished measurement's last log line, which names the error's type, is
    console text behind "Show the full error".
34. A setting with no on-screen name is listed on a record's page under its stored name.
35. The words of the Reduction map's stop question.
36. Appearance, like the keep-awake switch, can be changed while a run is live; every other
    setting is locked then.

### What this part fixes

- W005: stopping a measurement or a sweep study asks first and says what is kept.
- W006: deleting a finished simulation set states its simulations, its size and the time from its
  start to its last batch.
- W007, W137: deleting an unfinished record says another program may be writing it.
- W011 (in part): a leftover holding a trained network is named as one before it is cleared.
- W018, W115: the three out-of-date help texts are replaced.
- W029: a rescan keeps the cell; the sweep's ranges change only while "from the cell".
- W030: a record a run in this window is writing reads "Being written".
- W031: clearing leftovers no longer promises what the five-minute test cannot keep.
- W032: dates in the list and on the page are local, with the date.
- W033: deleting a record the session holds makes the session let go.
- W044, W052, W053, W190: "Rename…" on the record's page, for every kind (E9).
- W051 (with the anatomy part): an explanation is short and under its on-screen label.
- W069: unfinished runs are listed; a taken name says by what, with a link.
- W070: a checks-only record is called "Checks only" and cannot be compared.
- W071: values set by the resolution or the cell are marked; hidden settings are shown.
- W072: "Before you start" states the simulations and, where known, the time.
- W073: each run's figures show with that run.
- W074: the settling-in stretch is marked, progress is counted, a seed can be given.
- W075: the builder's result line sits beside its buttons.
- W076: "New model", "Edit" and closing the window ask before unsaved edits are lost.
- W077: saved measurements can be browsed during a run; a new comparison waits (U25).
- W078: the reason for a greyed start button is beside it.
- W079 (in part): the reduction's result and its files' paths show on its page.
- W081, W183, W184: parameter names and eigenvalues show whole.
- W082: Records re-reads when a run starts, changes stage or ends.
- W083: the whole log, searchable.
- W084: the record's folder is shown and can be copied.
- W085: figures for every kind of record.
- W086: the selection and the page survive a re-read.
- W087: actions refused during a run are greyed with the reason.
- W088: a search field; Enter saves a note; unnamed records are told apart.
- W089, W124: a saved run is shown as a record's page, in on-screen words and local time.
- W101 (with the shell part): Settings offers three appearance choices.
- W103, W125: Help is written for the reader, under on-screen labels.
- W120: no "finished" line after a stop or a failure.
- W126: "Compare" is greyed with the reason until the right number is ticked.
- W127 (in part): Live simulation's start is judged at once and does not ask the card.
- W128, W153 (in part): the stale comment at `crossval_panel.py:241-243` goes with the Sweep study
  page's rewrite (12.2).
- W129: the count under the list tells finished from unfinished.
- W130, W131: one word per kind; an unnamed simulation set is called by its prior.
- W132: a refused delete is a plain message with its own title.
- W133: a text of several lines is wrapped under its key.
- W134: the log is read, and a delete made, off the window's thread.
- W150: a study stopped in its second sweep lists its finished first sweep.
- W151: a model is judged again for each page when models change.
- W152: deleting a model says what is really removed.
- W156: "New model" starts from the built-in scales.
- W157: one choice replaces two tick boxes that could contradict the run.
- W158: a simulation set's settings show once.
