## 12. What becomes of each finding

This part designs no screen. It says, for each of the 198 findings of
`docs/superpowers/notes/2026-10-05-the-window-as-read-findings.md` (W001 to W198), what the design
does with it and which part carries it; then what is still exposed, what the audit chat starts
from, and how the first note's part 2 and part 4 are answered. B1 to B28 and C1 to C31 are the
rulings of RULINGS.md, parts B and C.

### 12.1 How to read the table

**Outcomes.** One per row.

| outcome | meaning |
|---|---|
| FIXED | The words of a decision (a U or an E) cover what a person meets. The *by* column names it. |
| FIXED IN PASSING | No decision names the fault, but the thing at fault is replaced, or mended by the carrying part, in building what a decision specifies, or a ruling of part B or C settles it. The *by* column names the decision or the ruling. |
| LEFT TO THE AUDIT | On U18's audit list, or an engine fault outside E1 to E12 (handing the second kind to the audit is a choice the owner has not seen). |
| LEFT AS IT IS | The reason is given: by design, out of scope under U4, not a fault, or not measured. |
| DROPPED | The second reader found it wrong, or it was not settled by reading and nothing has settled it since. |

**The dagger.** A dagger (†) in the *by* column means the row closes only with a rule the owner has
not seen: one the carrying part's closing list gives as such a choice, or a ruling of part B or C
marked so. A fault its part simply mends (a crash, a stale line, a blocked thread) is no such
choice and carries none. If the owner refuses the rule, the row becomes LEFT AS IT IS (12.2).

**Remainder.** Where a finding has two halves and a decision covers one, the row takes the outcome
of the covered half and its note says **Remainder:**; 12.6 lists every one. (Counting a two-halved
finding this way is a choice the owner has not seen.)

**Quotes.** Double quotes hold words of the new window; today's labels are in single quotes and are
called today's. Things are named by U7 even where a finding describes today's window.

**Code.** Checked on 2026-10-06 at commit `39c37c1`. `git diff 71a53b5 HEAD -- core tests run.bat
run.sh` is empty, so the findings' own line numbers stand. Every file:line here was read for the
first draft; those in the rows the rulings changed or added were read again.

**Checking.** This part adds no test and no walkthrough row. A row is closed in the build when
the text of each part named in its row is built and that part's test or walkthrough row for it
passes (U19: test-first; part 13); the auditor holds the row against that text, not against this
table, and a row for which the named parts give neither a test nor a walkthrough row is reported,
not counted as closed.

**Parts.** The *part* column names every part whose own text closes the row or a stated half of
it. It was checked both ways against the closing lists of parts 3 to 11 and 13 as finalised:
every closed row is named in the list of each part its row names, and every id a list names has
that part in its row. Two ids are named by a list that does not close them: part 7's names W138
only to place a bullet of the first note with no id of its own (the look-and-close overwrite;
12.8, 2A), and part 10's names W011 for one half, so that row stays LEFT TO THE AUDIT and names
part 10 for that half alone. Part 7 is left out of the rows its list calls helped there and
closed elsewhere.

### 12.2 The rows that rest on a rule the owner has not seen

These 25 rows carry the dagger. The rule is in the closing list of the part named.

| whose rule | rows |
|---|---|
| part 3 | W041, W180 and the launch half of W043 (B21) |
| part 4 | W110 |
| part 5 | W061, W073, W090, W100, W105, W121, W145, W147, W197 |
| part 7 | the per-keystroke half of W043 |
| part 8 | W048, W091 |
| part 10 | W033, W070, W071, W074, W079, W126, W157 |
| this part | W119, W128, W153 |

Unseen rules a row leans on without a dagger, because a decision closes the row or its other half:
C11, amending B11 (W154's refusal shown as a question); B19 (the fixed-width face, W042); B23 (the
"Seed" field, 12.8); C22 (only the window takes E7's fallback, W010); part 8's refilling of the
typed-units box when the model changes (W146, within a session); part 4's full stop as the decimal
mark on every computer (W167); part 5's figure viewer (the figures' half of W067, which stands or
falls with W073).

W119, W128 and W153 are wrong comments, with nothing on screen, as is half of W105. W105's other
half, the divider dragged just before quitting, closes only with part 5's rule that no page keeps a
divider the user can drag (B6 alone would leave it: closing writes the window's size and place
only); so the row carries the dagger. This part counts a comment-only finding as FIXED IN PASSING
when the build removes or rewrites the lines that hold it (a choice the owner has not seen):
`core/gui/panels/base_panel.py:359` and `:368` (W105, W119, W128) go with the dividers part 5
removes; `core/gui/panels/crossval_panel.py:241-243` (W128, W153) goes with part 10's rewrite of
that page; part 13 corrects W153's two test docstrings and W119's two comments at
`core/gui/vt.py:139`, `:234`, and names the stage of each (13.2.2).

### 12.3 The table

#### A1. Faults the first note does not list: can lose days of work (W001 to W011)

| id | the finding | outcome | by | part | note |
|---|---|---|---|---|---|
| W001 | Close question is generic; its preselected answer cancels the run | FIXED | U12 | 6, 13 | Today `core/gui/main_window.py:424-435`. |
| W002 | Rescan button puts the bounds list back on its first file | FIXED | U15 | 5, 7, 8 | Today `core/gui/widgets/artifact_picker.py:36`, `:44-57`. |
| W003 | Boxes show pending choices; nothing marks what the session holds | FIXED | U14, U5 | 5, 8 | |
| W004 | Nothing keeps Windows awake or reacts to sign-out or restart | FIXED | U11, U9, U27 | 6, 7, 13 | Keep-awake, and a reason registered with Windows that holds a sign-out or shutdown until the owner answers. What is still lost: 12.5. |
| W005 | One click on today's 'Cancel' ends a run for good | FIXED | U12, U17 | 6, 10 | **Remainder:** an Effective temperature run still cannot be resumed. |
| W006 | Deleting a finished simulation set asks one generic question | FIXED | U17 | 10 | |
| W007 | Delete question for an unfinished set omits a writer elsewhere | FIXED | U17 | 10 | **Remainder:** the store deletes with no in-use check (`core/artifacts/store.py:918-947`). |
| W008 | Rotation tick is remembered; a remembered off trains unrotated | FIXED | U15 | 7, 8 | U8 and U10 state it before the run. |
| W009 | After an inference the narrowing tab holds the earlier observation | FIXED | U16 | 8 | E4 lets that observation be named (W190). |
| W010 | Name checked again at the end; if taken, the network is lost | FIXED | E7 | 11 | Reachable from the window once U10 puts the name first; only the window takes the fallback (C22). **Remainder:** a command-line run still fails there and loses the network. |
| W011 | Final record write fails; network left in a sweepable folder | LEFT TO THE AUDIT | | 10 | The commit is unguarded (`core/artifacts/store.py:482`); E10 does not cover it. Not on U18's list. Part 10 closes one half only, naming such a leftover before it is cleared (12.5). |

#### A2. Can mislead a long run (W012 to W037)

| id | the finding | outcome | by | part | note |
|---|---|---|---|---|---|
| W012 | What the session holds shows only on today's first tab | FIXED | U5, U14 | 3, 5 | |
| W013 | Prior button lets go of posterior, observation and prior unasked | FIXED | U14, B3 | 5, 8 | Nothing is let go until the run returns its record (B3). |
| W014 | Bounds list says nothing about the file it shows | FIXED | U16 | 8 | |
| W015 | After a build the prior list still says from scratch | FIXED | U14, U15 | 5, 7, 8 | |
| W016 | Blank or half-typed memory limit silently means off | FIXED IN PASSING | U9, U10 | 7 | Today `core/gui/panels/inference/config_tab.py:148`. "No limit" becomes a choice of its own. |
| W017 | Typing a limit leaves the planner-budget line stale | FIXED IN PASSING | U10 | 8 | The line is dropped. |
| W018 | Help and note advise leaving the limit off | FIXED IN PASSING | U9, U17 | 7, 10 | Today `core/gui/panels/inference/help_text.py:47-48`. Part 7 gives the words Help prints. |
| W019 | Mistyped environment limit: note claims an override not in force | FIXED IN PASSING | U9, U10 | 7 | The planner's fallback stays (`core/SBI/pipeline.py:589-595`); part 7 shows it truthfully. |
| W020 | With the frequency response on, the applied line omits rotation | FIXED IN PASSING | U14, U10 | 8 | The line gives way to "In use" and "Before you start". |
| W021 | Training starts on one click; a second click refits | FIXED | U14, U16, U10 | 5, 8 | |
| W022 | Later tabs never say which posterior and prior they use | FIXED | U14, U5 | 5, 8, 9, 3 | |
| W023 | New-run warning speaks in folder codes and internal keys | FIXED | U10, E2 | 8, 11 | |
| W024 | Memory line stale after a limit change; a misfit asks nothing | FIXED IN PASSING | U10 | 5, 8 | The line is dropped. Its second half was by design. |
| W025 | The fits-in-one-piece line reassured about the run that died | FIXED | U10 | 5, 8 | **Remainder:** the planner's arithmetic (U18's list). |
| W026 | Narrowing tab: own budget; network neither shown nor settable | FIXED | U16, U8 | 8 | |
| W027 | A crash's full error lives only inside a dialog | FIXED | U12 | 6 | |
| W028 | The first hour of a fit looks hung | FIXED | U11, E1 | 6, 11 | E1 speaks at the end of each epoch; part 6 says the first has not finished. |
| W029 | Cell list's rescan button resets the choice; sweep grids follow | FIXED | U15 | 5, 7, 10 | |
| W030 | A calibration or inference in flight is listed as broken | FIXED | U17 | 10 | |
| W031 | Leftover sweep promises a safety its 300-second test cannot give | FIXED IN PASSING | U17 | 10 | The question is rebuilt under the delete rule. **Remainder:** the age test (`core/artifacts/store.py:1024-1034`). |
| W032 | Two clocks: raw universal-time stamps; log lines with no date | FIXED | U17 | 10 | **Remainder:** dates in a record's own log lines (U18's list). |
| W033 | A record a step holds can be deleted; the step goes on | FIXED IN PASSING | U17 † | 10 | Part 10: the session lets go of a deleted record. |
| W034 | The after-simulation comparison has no place in the window | FIXED | U11, U16 | 9, 6 | In three bands (B1), in the same words on the run's page and on the Checks list (C16). |
| W035 | An epoch-limit box would carry two traps | FIXED IN PASSING | U4 | 8, 9 | Part 8 offers the limit with both traps stated beside it. |
| W036 | A fit ended by the limit keeps its last network, unsaid | FIXED IN PASSING | U4 | 8, 6 | The end says the limit was reached. **Remainder:** which network is kept is the fitting library's rule (`sbi/inference/trainers/base.py:978`, `:1151-1153`, installed copy). |
| W037 | Set keyed by model name; edited equations reuse old rows | LEFT TO THE AUDIT | | | U18's list. Key at `core/artifacts/identity.py:89`. |

#### A3. Everyday use (W038 to W100)

| id | the finding | outcome | by | part | note |
|---|---|---|---|---|---|
| W038 | Nothing says a run has ended, how or when | FIXED | U12 | 3, 6 | |
| W039 | Run indicator is a small grey label with a clock | FIXED | U5, U11 | 3 | |
| W040 | No shortcut keys; Escape does nothing | FIXED | U13 | 3 | |
| W041 | One screen failing to build stops the whole window opening | FIXED IN PASSING | U5, B21 † | 3 | Each part is built at its first opening, inside a guard. |
| W042 | Today's log pane jumps to the bottom; proportional lettering | FIXED IN PASSING | U11 | 6 | Following the end only while at the end is part 6's mend; the fixed-width face is B19's (12.2). |
| W043 | Card asked for free memory on the window's thread, per keystroke | FIXED IN PASSING | U9, U10 † | 3, 7 | |
| W044 | Today's 'Save' on a prior is really a rename | FIXED | U14 | 5, 8, 10 | |
| W045 | A prior cannot be named or annotated before it is built | FIXED | U14, U16 | 5, 8 | |
| W046 | One button for a load and a build; nothing asked | FIXED | U14, U16 | 8 | |
| W047 | Input mistakes on the Prior tab answered only in today's log pane | FIXED IN PASSING | U10, U11 | 5, 8 | |
| W048 | Hand-typed bounds vanish unasked; cannot be saved as a file | FIXED IN PASSING | U14 † | 8 | **Remainder:** saving the grid as a bounds file. |
| W049 | Malformed bounds line breaks today's 'Edit values' | FIXED IN PASSING | U16 | 8 | |
| W050 | Each prior build resets the inference cell to launch time | FIXED | U15 | 7, 8 | |
| W051 | Help reachable by mouse only; long paragraphs in a bubble | FIXED | U17, U13 | 5, 10 | |
| W052 | Today's 'Save' on a posterior renames; its closing line misleads | FIXED | U14 | 5, 8, 10 | |
| W053 | Narrowing tab has no way to name its result | FIXED | U14, U16 | 5, 8, 10 | |
| W054 | Cell choice jumps back whenever a prior is built or loaded | FIXED | U15 | 7, 8 | |
| W055 | Driven-recordings table not remembered between launches | FIXED | U15 | 7, 8 | |
| W056 | Recording length and drive amplitude open at 1 s, 1 N | FIXED | U15 | 7, 8 | **Remainder:** the typed drive amplitude is still checked only for being above zero. |
| W057 | Calibration: no cost before, verdict only in today's log pane | FIXED | U16 | 9 | |
| W058 | Narrowing button greyed with no reason shown | FIXED | U10, U5 | 5, 8, 3 | |
| W059 | A refused or stopped training has already dropped the posterior | FIXED IN PASSING | U14, B3 | 5, 8 | Today `core/gui/panels/inference/posterior_tab.py:142`, before the dispatch at `:148`. |
| W060 | Rotation boxes checked even when the run will not read them | FIXED IN PASSING | U16 | 8 | |
| W061 | Each budget keystroke scans the store on the window's thread | FIXED IN PASSING | U10 † | 5 | |
| W062 | A run's end carries no time, no duration, no alert | FIXED | U12, U17 | 6 | |
| W063 | Today's log pane jumps to the bottom (as W042) | FIXED IN PASSING | U11 | 6 | |
| W064 | A run whose text capture cannot start ends without a word | FIXED IN PASSING | U12 | 6 | Today `core/gui/worker.py:207-210`. |
| W065 | Error box shows the message without the kind of error | FIXED | U12 | 6 | The page says in plain words what failed; the kind is behind "Show the full error" only (C7). |
| W066 | Ordinary progress lines shown as warnings | FIXED IN PASSING | U11 | 6 | Today `core/gui/streams.py:289`. |
| W067 | Runs pile up in one tab: unbroken log, old figures stay | FIXED IN PASSING | U11, U17 | 6 | Each run has its own view, text and, where one is kept, file (C3); the figures' half is W073's. |
| W068 | A pending cancel barely shows; header still says running | FIXED | U12 | 6 | |
| W069 | Unfinished Effective temperature record invisible; its name stays taken | FIXED | U17 | 10 | |
| W070 | Run stopped after its first checks is recorded as complete | FIXED IN PASSING | U17 † | 10 | **Remainder:** the engine marks it complete (`core/FDT/fdt_pipeline.py:316`, `:348-349`). |
| W071 | Sweep preset and cell overwrite typed numbers; three values unseen | FIXED IN PASSING | U17 † | 10 | |
| W072 | No cost stated before an Effective temperature run | FIXED | U17 | 10 | |
| W073 | Figure tabs pile up across runs under the same titles | FIXED IN PASSING | U16, U17 † | 5, 10 | |
| W074 | Live simulation: unmarked settling-in stretch, pulsing bar, no seed | FIXED IN PASSING | U17, U11 † | 10 | |
| W075 | Your own models: the result line is out of sight below the form | FIXED | U17 | 10 | |
| W076 | Your own models: unsaved edits are thrown away without asking | FIXED | U17 | 10, 6 | The window's close asks it too, after the question about a live run (C8). |
| W077 | Effective temperature tabs locked during a training | FIXED | U11, U25 | 6, 10, 13 | Saved runs stay open to read. **Remainder:** a new comparison waits until the run ends, and its button says so; no second run of any size. |
| W078 | Greyed Effective temperature button gives no reason | FIXED | U10, U17 | 10 | |
| W079 | Reduction map leaves loose files and no record | FIXED IN PASSING | U17 † | 10 | The result and the files' paths are on its page. **Remainder:** it is still not a record. |
| W080 | Stored figures keep the window's dark or light colours | FIXED | U23 | 4, 11, 13 | A figure saved into a record is always drawn light (B18), with E12 for its line colours. Today `core/gui/mpl_theme.py:16-33` follows the theme. |
| W081 | Parameter list cut to three of thirteen | FIXED | U17, U18 | 10 | The cut stays in the shared text (`core/artifacts/report.py:57-60`); the record's page draws from the manifest. |
| W082 | Records has nothing to read about a live training | FIXED IN PASSING | U11, U5, U17 | 10 | Records re-reads as a run goes; the run's own page, part 6's, is where a live run is read. |
| W083 | A log over 1 MiB cannot be read in full | FIXED | U17 | 10 | |
| W084 | A record's folder is never shown; its files unreachable | FIXED IN PASSING | U17 | 10 | Figures and the whole log are on the record's page; part 10 also shows the folder. |
| W085 | Stored figures viewable only for finished Effective temperature runs | FIXED | U17 | 10 | |
| W086 | Every refresh drops the selection and empties the pane | FIXED IN PASSING | U17 | 10 | |
| W087 | Note, delete and sweep look available during a run | FIXED IN PASSING | U17, U11 | 10 | Greyed with the reason, under part 6's lock. |
| W088 | No filter, search or multi-selection in the records list | FIXED | U17 | 10 | **Remainder:** multi-selection. |
| W089 | Saved-run line shows internal names and raw universal time | FIXED | U17 | 10 | |
| W090 | Loading a posterior with no prior held ends in a crash | FIXED IN PASSING | U14 † | 5, 8 | **Remainder:** the stage reads the prior before its load branch (`core/orchestrator.py:902`). |
| W091 | Prior and posterior paired by hand; never checked on load | FIXED IN PASSING | U16 † | 8 | **Remainder:** the engine's load takes no prior (`core/artifacts/store.py:1302`). |
| W092 | The short timing fit cannot be run from the window | FIXED | U16 | 9 | |
| W093 | Input-use check fails with plain errors, not refusals | FIXED | E6 | 11, 9 | A posterior trained with no simulation set kept names none (`core/orchestrator.py:1433-1434`) and cannot run it; parts 8 and 9 say so. |
| W094 | The must-continue refusal carries no setting to point at | FIXED | E6 | 11, 8, 9 | |
| W095 | Repeated-calibration screen would sit silent, stop unanswered | LEFT TO THE AUDIT | | | `core/diagnostics/sbc.py:172`. Not on U18's list. Part 9's stage line says so while it lasts. |
| W096 | Log lines carry a time of day and no date | LEFT TO THE AUDIT | | | U18's list (`core/runs.py:59-60`). The dated full-text file of U11 carries the date. |
| W097 | A zero or mistyped memory-line setting breaks a run or start-up | LEFT TO THE AUDIT | | | `core/SBI/pipeline.py:269`. No control is designed for it. |
| W098 | The memory limit a run used is recorded nowhere | FIXED | U9, U18 | 7, 6 | In the run's history and full text. **Remainder:** the record's own manifest has no such key. |
| W099 | The must-continue refusal names no control (as W094) | FIXED | E6 | 11, 8, 9 | |
| W100 | Budget keystrokes read other sets' files (as W061) | FIXED IN PASSING | U10 † | 5 | |

#### A4. Cosmetic, or a page or comment that is wrong (W101 to W136)

| id | the finding | outcome | by | part | note |
|---|---|---|---|---|---|
| W101 | Today's 'Auto (time of day)' restyles each minute, flips mid-run | FIXED | U13 | 3, 4, 10 | |
| W102 | Windows' text-size setting is ignored | FIXED | U13 | 4 | Today `core/gui/fonts.py:38`. A scratch test in stage 1 first establishes what the toolkit already follows (B20). |
| W103 | Help lists entries under internal names | FIXED | U17 | 10 | |
| W104 | Header changes height between Home and other screens | FIXED IN PASSING | U5 | 3 | No header; the rail keeps its size. |
| W105 | Divider dragged just before quitting is forgotten; comments wrong | FIXED IN PASSING | U16 † | 5 | Both dividers and their saved layout go, by part 5's rule, and the comments with them (12.2). |
| W106 | No colour for error, warning, success or a destructive action | FIXED | U6 | 4 | |
| W107 | Prior tab asks the card on every keystroke | FIXED IN PASSING | U16 | 8 | |
| W108 | Build note shows a question mark for the step count | FIXED IN PASSING | U16 | 8 | |
| W109 | Closing line tells a named prior to be named | FIXED IN PASSING | U14 | 5, 8 | |
| W110 | Read-only band and amplitude boxes look editable | FIXED IN PASSING | U8 † | 4, 5, 8 | Shown as text. |
| W111 | Units listed in a different order from launch to launch | FIXED IN PASSING | U16 | 8 | |
| W112 | First tab gives most of its width to an unused area | FIXED IN PASSING | U16, U9 | 5, 7 | |
| W113 | The simulation set has three names in one flow | FIXED | U7, U22 | 8, 11 | E11 gives the engine's own sentences the same words. |
| W114 | A refusal's advice points at a dialog that never appeared | FIXED IN PASSING | U10 | 5, 8 | |
| W115 | Three help texts are out of date | FIXED IN PASSING | U17 | 10 | |
| W116 | After the window closes, log lines pile up unread | FIXED IN PASSING | U12 | 6 | No run outlives the window. |
| W117 | A second run past the one-run guard: unreachable | LEFT AS IT IS | | | Not a fault: nothing reaches it (`core/gui/panels/base_panel.py:221-225`). It bounds part 6's retries; U25 keeps every second job out. |
| W118 | Stall notice counts raw seconds in a narrow label | FIXED IN PASSING | U11 | 6 | |
| W119 | Stale comments: layout saving; a library line number | FIXED IN PASSING | † | 5, 13 | 12.2. |
| W120 | A finished line is printed after a cancel or a crash | FIXED | U17 | 10, 6 | Today `core/gui/panels/fdt_panel.py:253`. |
| W121 | A refusal's fix line names every tab holding the control | FIXED IN PASSING | U16 † | 5 | |
| W122 | Saving a model recompiles every saved model on the window's thread | LEFT AS IT IS | | | Not measured; no decision; no part takes it. Call at `core/gui/screens/model_builder_screen.py:530`. |
| W123 | Stop and close messages say training whatever is running | FIXED | U12, U7 | 6 | |
| W124 | Saved-run details show internal names and universal time | FIXED | U17 | 10 | |
| W125 | Help headed by internal names and mode lines | FIXED | U17 | 10 | |
| W126 | Adding to a comparison and comparing fail silently or late | FIXED IN PASSING | U17, U10 † | 10 | |
| W127 | Start enabled with no cell file; card asked though unused | FIXED IN PASSING | U17, U10 | 10 | **Remainder:** the reduction map's half. |
| W128 | Two stale comments | FIXED IN PASSING | † | 5, 10 | 12.2. |
| W129 | Count line calls an unfinished record complete | FIXED IN PASSING | U17, U6 | 10 | |
| W130 | Every simulation set is shown as unnamed | FIXED | U17, E9 | 10, 11 | "Rename…" covers a set (B26) and E9 keeps the name; today every save writes it empty (`core/artifacts/store.py:1582`). |
| W131 | Messages use the store's internal words for kinds | FIXED | U7, U22 | 10, 11 | E11 for the engine's own sentences. |
| W132 | Refused delete shown as an input fault with stray advice | FIXED IN PASSING | U17 | 10 | |
| W133 | Multi-line text falls out of the pane's layout | FIXED IN PASSING | U17, U18 | 10 | |
| W134 | Selecting and deleting block the window's thread | FIXED IN PASSING | U17 | 10 | |
| W135 | The tracker argument is not a live per-epoch hook | FIXED IN PASSING | E1 | 11 | E1 takes the other route and corrects `core/SBI/train.py:208-211`. |
| W136 | A posterior is dated at its end; durations cover one process | LEFT TO THE AUDIT | | | `core/artifacts/store.py:846-850`. Not on U18's list. The run's history (U17) holds the start. |

#### B. Faults the second readers added (W137 to W160)

| id | the finding | outcome | by | part | note |
|---|---|---|---|---|---|
| W137 | Delete question omits a writer elsewhere (as W007) | FIXED | U17 | 10 | **Remainder:** as W007. |
| W138 | Loading a saved prior does not check the bounds file | FIXED | U16 | 8 | **Remainder:** the engine's check (U18's list; `core/artifacts/store.py:1275-1291`). |
| W139 | Closing the launcher's console window kills PRISM and its run | FIXED | U12 | 6 | Today `run.bat:9`, `:16`. |
| W140 | A set deleted under a live run is rebuilt broken, silently | LEFT TO THE AUDIT | | | `core/SBI/training_checkpoint.py:372`, `:465-474`. Needs a second program, which after U26 means the command line; U17's wording is the window's only part. |
| W141 | Typed units never match a set made under the units file | FIXED | U15, U10, E2 | 8 | **Remainder:** the two key differently by design. |
| W142 | A training without the rotation never says so | FIXED | U15, U11, U10 | 8, 6 | **Remainder:** the run's own log has no line (`core/orchestrator.py:1255-1256`). |
| W143 | Nothing stops or warns about a second PRISM window | FIXED | U26 | 3, 13 | A second start brings the open window to the front (B17). |
| W144 | Capture failing to start ends with no message (as W064) | FIXED IN PASSING | U12 | 6 | |
| W145 | Popped-out figure windows outlive the main window | FIXED IN PASSING | U12 † | 5, 6 | The main window owns and closes them. |
| W146 | Typed-units box never follows a change of model | FIXED | U15 | 7, 8 | U15 closes the half across launches; the half within a session is part 8's rule (12.2). |
| W147 | Loading a posterior without a prior crashes (as W090) | FIXED IN PASSING | U14 † | 5, 8 | **Remainder:** as W090. |
| W148 | Re-listing cells logs warnings about a cell nobody picked | FIXED IN PASSING | U15 | 8 | |
| W149 | Pressing today's 'Train' unloads the held posterior (as W059) | FIXED IN PASSING | U14, B3 | 5, 8 | |
| W150 | Finished first half of a stopped sweep study is not listed | FIXED IN PASSING | U17 | 10 | |
| W151 | Run button stays grey after the model is fixed | FIXED IN PASSING | U10, U17 | 10 | |
| W152 | Deleting a model removes hand-added files, unsaid | FIXED IN PASSING | U17 | 10 | Today `core/gui/main_window.py:292` against `core/Helpers/model_store.py:413`. |
| W153 | Stale comments and test docstrings about the error hook | FIXED IN PASSING | † | 13, 10 | 12.2. |
| W154 | Narrowing round's refusal is worded for a programmer | FIXED | U22 | 8, 11 | E11 puts the sentence in U7's words (today `core/orchestrator.py:567-587`); C11 shows it, as it stands, as a question at the click. **Remainder:** it still names a folder, an engine setting and "pass new_run=True". |
| W155 | Closing line suggests an unnamed result is not kept | FIXED IN PASSING | U14 | 5, 8 | |
| W156 | A new model opens with the last model's two scales | FIXED IN PASSING | U17 | 10 | |
| W157 | Greyed proceed box can show the opposite of what runs | FIXED IN PASSING | U17 † | 10 | |
| W158 | A simulation set's settings are printed twice | FIXED IN PASSING | U17, U18 | 10 | |
| W159 | Command line's comment claims a pairing check not made | LEFT TO THE AUDIT | | | With W091 (`core/tool/config_args.py:351-353`). U4 leaves the command line alone. |
| W160 | An unreadable set record is silently rewritten as new | LEFT TO THE AUDIT | | | `core/artifacts/store.py:1570-1588`. Not on U18's list. |

#### C. The first note's claims that did not simply hold (W161 to W195)

| id | the finding | outcome | by | part | note |
|---|---|---|---|---|---|
| W161 | The whole-window lock was overstated: only each form locks | LEFT AS IT IS | | | Not a fault: a correction. U11 keeps settings locked; its one change is W077. |
| W162 | Today's 'Quit anyway' in a fit loses the result, silently | FIXED | U12, U18 | 6 | The button goes; the figure hand-over is guarded in the window. |
| W163 | Today's 'Quit anyway' was never tried on a real screen | FIXED IN PASSING | U12, U19 | 13 | The new closing question gets walkthrough rows. |
| W164 | What the two quit buttons really do was unknown | FIXED | U12 | 6, 13 | Settled by the experiment, not by reading (see below). |
| W165 | Memory limit is locked during a run by the app-wide lock | LEFT AS IT IS | | | By design: U11 locks settings; U9 records the limit the run began with. |
| W166 | Free-memory figure stale; question on today's 'Apply' only | FIXED | U10, U9, U14 | 7 | |
| W167 | Untouched bounds grid gives the same key as its file | LEFT AS IT IS | | | Not a fault. Its one unsettled point, a comma typed as the decimal sign, is decided by part 4's rule (12.2) and ruling D9: it is refused in words when editing ends. A walkthrough row now types a comma into the bounds grid (part 13's row G9), and another into the memory limit (G11). |
| W168 | Guide, help and note all advise no limit | FIXED IN PASSING | U17, U9, U19 | 7, 13 | Part 7 gives the words; part 13 rewrites the guide page. |
| W169 | The settings line names one differing key, no values | FIXED | U10, E2 | 8, 11 | |
| W170 | The fit's progress is one caption: no clock, no loss | FIXED | U11, E1 | 6, 11 | |
| W171 | The lock was overstated (as W161) | LEFT AS IT IS | | | Not a fault. |
| W172 | One click cancels a fit and cannot be withdrawn | FIXED | U12 | 6 | **Remainder:** a confirmed stop still loses the fit. |
| W173 | Today's 'Quit anyway' loses the record at the end (as W162) | FIXED | U12, U18 | 6 | |
| W174 | Today's 'Cancel task & quit' closes at once, without waiting | FIXED | U12 | 6 | |
| W175 | During the fit only the header clock moves | FIXED | U11, E1 | 6 | |
| W176 | Only the left column is locked (as W161) | LEFT AS IT IS | | | Not a fault. |
| W177 | The right-hand column stays live (as W161) | LEFT AS IT IS | | | Not a fault. |
| W178 | Today's 'Quit anyway' is a gamble the dialog hides (as W162) | FIXED | U12, U18 | 6, 13 | |
| W179 | A training leaves no lasting record of its lines | FIXED | U11 | 6 | |
| W180 | Simply opening the window starts using the graphics card | FIXED IN PASSING | U10, B21 † | 3 | U10 drops the line that asked; nothing but Home is built at launch. |
| W181 | Reduction map and live simulation have no command-line form | LEFT AS IT IS | | | Out of scope under U4; not a fault. |
| W182 | No lock between programs: another window can delete under a run | LEFT TO THE AUDIT | | | `core/artifacts/store.py:52-58`. U26 leaves one window; the command line can still delete under a run. |
| W183 | Thirteen eigenvalues shown as three | FIXED | U17, U18 | 10 | |
| W184 | Thirteen eigenvalues shown as three (as W183) | FIXED | U17, U18 | 10 | |
| W185 | The writer removes its folder when the figure hand-over fails | FIXED | U18, U12 | 6 | The guard is the window's, round the hand-over that is unguarded today (`core/gui/panels/base_panel.py:62`). |
| W186 | Reusing a complete set overwrites its recorded generation time | LEFT TO THE AUDIT | | | `core/artifacts/store.py:1576-1577`. Not on U18's list. |
| W187 | Name, note and three run settings are window-only additions | FIXED | U10, U4, E6 | 8 | Part 8 offers all five. |
| W188 | The pre-run checks have nothing in the window | FIXED | U16, E3 | 9, 11 | |
| W189 | Calibration at fixed numbers of driven recordings has no screen | FIXED | U16 | 9 | |
| W190 | Calibrations, inferences and observations stay unnamed | FIXED | U14, U17, E4 | 10, 11 | Named before they are made on the pages of parts 8 and 9; "Rename…" afterwards. |
| W191 | The set's key holds nothing about the code | LEFT TO THE AUDIT | | | U18's list. |
| W192 | Today's 'Quit anyway' loses the record (as W162) | FIXED | U12, U18, E10 | 6, 11 | E10 guards the engine's own drawing and saving of the loss figure (today `core/orchestrator.py:1429-1431`); on either front end (C22). |
| W193 | Close-and-cancel does not wait for the rescue save | FIXED | U12 | 6 | The rescue save itself is kept as it is. |
| W194 | A narrowing round always trains the default network | FIXED | U16, U8 | 8 | |
| W195 | The grid round trip is traced and safe (as W167) | LEFT AS IT IS | | | Not a fault. |

The second readers could not settle W164, W173, W178 and W192 by reading. The experiment in part 2
of `docs/superpowers/notes/2026-10-05-the-window-as-read-for-the-redesign.md` settled them (its case
A: the process lived on and the figure hand-over raised), so they are FIXED, not DROPPED. (Treating
the experiment as settling them is a choice the owner has not seen.)

#### D. Dropped, or not settled by reading (W196 to W198)

| id | the finding | outcome | by | part | note |
|---|---|---|---|---|---|
| W196 | The last-resort error hook may open a dialog off its thread | DROPPED | | | Not settled by reading; nothing has settled it since. Part 3 keeps the hook where it is (`core/gui/app.py:106-107`). |
| W197 | Main window closed with a figure window open: PRISM lives on | FIXED IN PASSING | U12 † | 5, 6 | Settled since by the same experiment (its case C). The main window owns and closes the figure windows. |
| W198 | A cancel landing after the posterior is committed | DROPPED | | | The second reader found it wrong. |

### 12.4 Counts

| outcome | A1 | A2 | A3 | A4 | B | C | D | all |
|---|---|---|---|---|---|---|---|---|
| FIXED | 10 | 15 | 35 | 11 | 8 | 21 | 0 | 100 |
| FIXED IN PASSING | 0 | 10 | 25 | 22 | 13 | 3 | 1 | 74 |
| LEFT TO THE AUDIT | 1 | 1 | 3 | 1 | 3 | 3 | 0 | 12 |
| LEFT AS IT IS | 0 | 0 | 0 | 2 | 0 | 8 | 0 | 10 |
| DROPPED | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 2 |
| all | 11 | 26 | 63 | 36 | 24 | 35 | 3 | 198 |

Of the 74 FIXED IN PASSING, 25 carry the dagger (12.2). Of the 174 rows closed, 24 leave a
remainder (12.6).

The 10 LEFT AS IT IS: eight are corrections to the first note, kept by design or outside U4; one
is unreachable (W117); one is a fault that no decision and no part takes up (W122).

### 12.5 The findings that can lose days of work

One thing first, because it outweighs the list: **the fit is still not saved as it goes.** Under U4
that is a separate design, to accept or leave. Until it is built, any end of the program during a
fit loses the whole fit: a confirmed stop, a crash, a power cut, a shutdown the owner lets go
ahead. The simulation set survives each. Everything below narrows the ways to reach that loss by
accident; nothing below removes it.

Nothing in the window is still exposed for: a fit lost by closing the window and pressing Enter
(W001; U12), by today's 'Quit anyway' (W162, W164, W173, W178, W185, W192; U12, U18, E10) or by
closing the launcher's console window (W139; U12); batches lost by a close that did not wait
(W174, W193; U12); days without the rotation (W008, W142; U15); a narrowing round around the wrong
observation or at the wrong network size (W009, W194, W026; U16, U8); a finished simulation set
deleted on one generic question (W006; U17); a trained network lost to a name taken during a run
started from the window (W010; E7; from the command line it is still lost, C22).

| what could be lost | ids | outcome | what is still exposed |
|---|---|---|---|
| A fit, by one click on today's 'Cancel' | W172, W005 | FIXED, U12 | A confirmed stop still loses the fit. |
| Hours of an Effective temperature run or sweep, by one click | W005 | FIXED, U17 | An unfinished run still cannot be resumed. |
| Days on the wrong bounds file, shown as another | W003, W002, W138 | FIXED, U14, U15, U16 | The engine still loads a prior without checking the whole bounds file (the audit, item 4). |
| A fit, because Windows slept, signed out or shut down | W004 | FIXED, U11, U27 | **If the owner shuts down anyway during the fit, the fit is lost**: nothing of a fit can be saved, and the reason shown says so (B28). During simulation the batches done are saved for as long as Windows allows; after reopening the run goes on from them, from the Training page: a run ended by "shut down anyway" counts as stopped by the user (ruling D8; part 6, 6.10). A shutdown that Windows forces without asking still ends the run; a scratch test in stage 3 settles whether the registered reason makes Windows wait (6.10). |
| A set that a run in another program is writing, deleted under it | W007, W137, W140, W143, W182 | FIXED for the wording (U17) and for a second window (U26); LEFT TO THE AUDIT for the rest | **The store still deletes without asking whether anything is writing, and a run whose set was deleted still rebuilds a broken one without a word.** |
| A trained network, because the last write failed | W011 | LEFT TO THE AUDIT | **The network stays in a folder nothing can load, and clearing leftovers would remove it.** That is a deliberate click, and parts 6 and 10 say before it what the folder holds. |

### 12.6 Remainders: the halves no decision closes

Eleven go to the audit, under the item of 12.7 named: W025 (item 1), W032 (item 3), W091 and W138
(item 4), W007, W137 and W031 (item 7), W090 and W147 (item 14), W070 (item 15), W142 (item 16).
The other thirteen are left as they are:

| id | what remains | why it is left |
|---|---|---|
| W005 | An Effective temperature run or sweep cannot be resumed (nothing under `core/FDT` resumes one; the word's only mention there, `core/FDT/compare.py:28`, says so of a comparison). | It needs the engine, outside E1 to E12. U17 shows the unfinished run as unfinished. |
| W010 | From the command line a name taken during the run still costs the trained network (`core/artifacts/store.py:846-850`). | By ruling: E7 is opt-in and only the window opts in (C22, a choice the owner has not seen). |
| W036 | A fit ended by the epoch limit keeps its last network. | The fitting library's rule; changing it would change the training method. |
| W048 | Typed bounds cannot be saved as a bounds file. | A new feature no decision asks for. |
| W056 | The typed drive amplitude is checked only for being above zero (`core/SBI/observations.py:208-212`, `:224`). | No decision or part adds a check; U15 makes it typed at every launch. |
| W077 | No new comparison and no second run of any size beside a live run. | By decision (U25). |
| W079 | A reduction map's result is not a record: no name, note or settings are kept. The store has no such kind (`core/artifacts/store.py:33-41`). | A new kind in the store is an engine change beyond E1 to E12. |
| W088 | Records has no multi-selection. | No decision. |
| W098 | A record's own manifest holds no memory limit, so a run from the command line records none. | By decision: U18 has the window record it, not the engine. |
| W127 | The reduction map's half of the finding. | Part 10 leaves it, and the finding calls it small. |
| W141 | Typed units and a units file never share a simulation set. | By design (`core/artifacts/identity.py:28-32`); part 8 says so on the page. |
| W154 | The engine's sentence still names the saved set by its folder and the setting by the engine's name, and ends "pass new_run=True". | Structured facts on a refusal are an engine change beyond E1 to E12 (part 8, 8.6); the question's buttons say what to do. |
| W172 | A confirmed stop loses the fit, also one confirmed just before the fit ends; so does shutting down anyway during the fit (U27). | Closed only by saving the fit (the separate design, U4). |

### 12.7 Everything left to the audit

The audit chat starts from this list. The first five are U18's own; part 11 gives their code
(11.10). Items 18 and 19 are on it by B10 and U21. Items 6 to 17, and W091 and W159 beside item 4,
are this part's additions (a choice the owner has not seen).

| # | what | findings | code |
|---|---|---|---|
| 1 | The memory planner's arithmetic behind today's 'fits' line: one drive counted, four built. | W025 | part 11 |
| 2 | Failing fast after a memory failure that cannot be recovered. | none; the first note's lead 3A.2 | part 11 |
| 3 | Dates in a record's own log lines. | W096, W032 | part 11 |
| 4 | The prior is not checked against the whole bounds file inside the engine. | W138; W091 and W159 beside it | part 11; `core/artifacts/store.py:1302` (W091); `core/tool/config_args.py:351-353` (W159) |
| 5 | A simulation set knows nothing of the code, or of a user model's equations. | W191, W037 | part 11 |
| 6 | A failed final write leaves a trained network in a folder with no record. | W011 | `core/artifacts/store.py:482` |
| 7 | No lock or in-use check between programs: delete, set a note and clear leftovers under a live run. Since U26 the other program is the command line. | W182, W007, W137, W031 | `core/artifacts/store.py:52-58`, `:918-947`, `:1024-1034` |
| 8 | A simulation set deleted under its run is rebuilt as a broken partial set, and the failure is swallowed. | W140 | `core/SBI/training_checkpoint.py:372`, `:465-474` |
| 9 | An unreadable set record is rewritten as if new. | W160 | `core/artifacts/store.py:1570-1588` |
| 10 | Reusing a complete set replaces its recorded generation time. | W186 | `core/artifacts/store.py:1576-1577` |
| 11 | A posterior's id and date are those of its end; durations cover the last process only. | W136 | `core/artifacts/store.py:846-850` |
| 12 | The repeated calibration reports nothing and cannot be stopped during its rank step. | W095 | `core/diagnostics/sbc.py:172` |
| 13 | The memory-line setting at zero divides by zero; a non-number stops the program while it loads. | W097 | `core/SBI/pipeline.py:269` |
| 14 | The training stage reads the prior before its load branch, so a load with none is a crash, not a refusal. | W090, W147 | `core/orchestrator.py:902` |
| 15 | An Effective temperature run stopped after its first checks is marked complete. | W070 | `core/FDT/fdt_pipeline.py:316`, `:348-349` |
| 16 | A training with the rotation off logs no line saying so. | W142 | `core/orchestrator.py:1255-1256` |
| 17 | A calibration on a round just trained records an empty list of consents. | none; the first note's 2D and lead 15 | `core/orchestrator.py:1488` |
| 18 | The numbers the deciding checks are judged by are science numbers kept in the window's Checks code, not in the engine; B10 lists them. | none of its own | none in the engine; part 9 names the constants |
| 19 | The four engine additions the owner listed for the audit (U21): E8, E9, E10, and the setting names that E4 and E5 add to the engine's list. | W130 (E9), W192 (E10) | part 11; today `core/config.py:181-184`, `core/artifacts/store.py:1582`, `core/orchestrator.py:1429-1431` |

Not on this list, on purpose: the remainders of W036, W010 and W154 (12.6 gives each reason).

### 12.8 The first note's part 2 and part 4

The note is `docs/superpowers/notes/2026-10-05-the-stopped-retrain-and-the-window-route.md`.

#### 2A. Before the button

| bullet | answered by |
|---|---|
| The network size reopens at 128 by 8 on every launch. | U8: it still opens at the built-in values, by decision; a plan sets it and "Before you start" names every departure. |
| Protection against an accidental fresh run is thin. | U10, last bullet, with E2. |
| The passive line can name the wrong thing. | U10 with E2: every difference, by its on-screen name. |
| The remembered selections can be wrong for the run. | U15, U14, U16 (a saved prior shows the bounds file it was built on); U24: Home names last time's picks and loads nothing until "Continue with these" is pressed. |
| A look-and-close launch overwrites the remembered bounds file. | U15: a pick is saved when it is applied; B6: closing writes the window's size and place only. |
| Choosing a prior in the list loads nothing until the button. | U14: "In use" against "Changed, not applied". |
| Every confirmed press of today's 'Apply' re-selects the launch-time bounds file. | U15, U14. |
| The memory limit resets to 0 on every launch; the memory-line cadence has no control. | U9 for the limit. The cadence stays an environment setting: still open, and left so (the audit, item 13). |
| No name and no note before the run. | U10, U14, E7. |
| No control for an epoch limit, the save cadence, the continue policy or the device. | U4 puts them in scope; part 8 offers the first three on Training. The device: still open; no decision or part gives a control. |
| Launching the window touches the card. | U10 removes the line that asked; under B21 nothing but Home is built at launch (W180). |

#### 2B. During the run

| bullet | answered by |
|---|---|
| No lasting record of the run's lines. | U11: a dated file written as the run goes. |
| The fit's progress is one caption. | U11 with E1. |
| Today's stall notice stands through every epoch. | U11: the notice goes with today's progress pane; part 6 gives a quiet stretch in minutes or hours and says the first epoch has not finished. |
| The whole window is locked; nothing guards against another process. | U11 and U25 for the lock (narrowed; W161, W077). A second window: U26. The command line beside a run: still open (the audit, item 7). |
| A memory failure ends in a red box and waits; a relaunch resets the network and the limit. | U12 as C5, C2 and C4 narrow it (three tries only after the card runs out of memory in a training or a narrowing round; "Continue from batch N" on Home by C2's rule, the name and note typed again), U17 (run history), U9, U8; U24 with B16 and C1: one press brings back last time's picks and plan. Failing fast: the audit, item 2. |

#### 2C. Stopping and closing

| bullet | answered by |
|---|---|
| The fit is not saved as it goes. | Still open: U4 makes it a separate design. |
| A cancel lands at the next epoch; one click; cannot be withdrawn. | U12. |
| The close dialog's preselected button cancels. | U12. |
| Today's 'Quit anyway' probably loses the record. | U12 (it goes), U18 (the window's guard) and E10 (the engine's, on either front end: C22). |
| Today's 'Cancel task & quit' closes without waiting. | U12. The rescue save during simulation is kept; U27 relies on it when the owner shuts down anyway (B28). |

#### 2D. Around the fit

| row | answered by |
|---|---|
| The pre-run checks. | U16, E3. |
| Calibration: a seed, a name, a note. | U16 and U14 for the name and note. The seed: B23, a "Seed" field under "Rarely changed" (a choice the owner has not seen). |
| The rotation check, and lists cut to three of thirteen. | U16; U17 and U18 for the lists. |
| The input-use check. | U16, E6. |
| The four calibrations at fixed numbers of driven recordings. | U16. |
| Inference: a name, the draw count. | U16, E4, U14. The draw count: U4 puts it in scope; no decision names it; part 8 gives it a field, "Posterior draws", under "Rarely changed" (part 8's choice, which the owner has not seen). |
| The narrowing round: network size, continue policy, epoch limit, whether it continues. | U16, U8. Whether it continues: never a training's simulations, and its own only when the same round is started again with the same settings; its "Before you start" says so (C11), and the page has no continue setting. |
| A calibration straight after a round records no consent. | Still open (the audit, item 17). |
| Note and lineage: figures; the last 1 MiB of a log. | U17. |
| Unnamed calibrations, inferences and observations; the narrowing tab's remembered observation. | U14, U17, E4, U16. |

#### 2E. What the window already does well

| bullet | kept, or why not |
|---|---|
| The row total, the memory line, continue or new, before the button. | Kept in "Before you start" (U10), except the memory line, which U10 drops. |
| A question on a one-setting near miss; a question before a session is replaced. | Kept and widened: U10's last bullet for Training, C11 for the narrowing round, U14 for the session. |
| The memory limit as a visible field with a free-memory note. | Moved to Settings and remembered (U9); the "Before you start" of a run that simulates on the card says only what is known (U10, C13). |
| Live progress, the loss figure at the end, the finished posterior held. | Replaced by U11's live view and U6's loss curve; the result stays on the page with the next step offered (U16). |
| The rescue save on a stop during simulation. | Kept: the engine is not touched; U12's question says what is kept. |

#### 2F. Never exercised

U19, stage 9, adds the owner's walkthrough rows, which part 13 lists (13.4: the closing question,
a stop and a withdrawn stop, trainings of more than two batches, and by C25 every rebuilt surface);
today's 'Quit anyway' goes (U12); a training of real size from the window stays unexercised until
the retrain itself.

#### Part 4. The questions left for the owner

| question | answered by |
|---|---|
| Should the certification round match the retrain's network? | U16, U8: the round's network comes from the plan and is shown. The value is the plan's. |
| Should the checks behind gates 2 and 4, and the stratified calibrations, be reachable? | U16: yes, each has a screen. |
| Is a calibration seed drawn by the window acceptable for gate 1? | Still open, and the owner's. Under B23 it can be typed, or drawn and recorded. |
| Must calibration and inference records be nameable? | U14, U17, E4: named before they are made; "Rename…" afterwards. |
| What should happen after a failure in a days-long run? | U12, as C5 and C2 narrow it (2B above). |
| What replaces the console capture? | U11 (the dated file), U17 (run history). |
| Should the fit be saved as it goes? | Still open: the separate design of U4. |
| What should a run with every default do about card memory? | U9, U10: the limit is remembered, shown and recorded, and a run with none says so before it starts. Still open: the planner's arithmetic (the audit, item 1). |
| Reuse the complete simulation set or simulate afresh? | Still open: the owner's, after the audit. U10 says which the run would do. |
| Does the four-week budget stand? | Still open: not a matter of this design. |

### Choices in this part the owner has not seen

1. An engine fault outside E1 to E12 that U18's audit list does not name is handed to the audit,
   not designed into this redesign and not left as it is: items 6 to 16 of 12.7, and W159 with
   W091's remainder beside item 4. The one that can lose days is W011 (12.5).
2. A row that closes only through a rule the owner has not seen counts as fixed in passing, and
   becomes left as it is if the owner refuses the rule (25 rows). A fault its part simply mends,
   listing no rule for it, is fixed in passing with no dagger.
3. A finding with two halves is counted under the half a decision covers; the other half is listed
   as a remainder (24 rows).
4. Five findings the second readers could not settle by reading count as settled by the experiment
   and as fixed, not dropped (W164, W173, W178, W192, W197).
5. The audit's list carries one lead of the first note that has no finding of its own and is not
   on U18's list: the empty list of consents after a narrowing round (item 17).
6. That a fit ended by the epoch limit keeps its last network is left as the fitting library's rule
   and kept off the audit's list (W036), as are the remainders of W010 and W154.
7. The pause when a model is saved is left as it is: never measured, and no part takes it up
   (W122).
8. A finding that is only a wrong comment counts as fixed in passing when the build removes or
   rewrites the lines that hold it (W119, W128, W153, half of W105).

The rules behind the daggers, and those 12.2 names as leaned on, are the carrying parts' or the
controller's, not this part's.

### What this part fixes

This part changes no screen and no code, and closes no finding by itself. It records where each
of the 198 is closed or left (12.3, counted in 12.4), what can still lose days of work (12.5), and
the 19 items the audit starts from (12.7).
