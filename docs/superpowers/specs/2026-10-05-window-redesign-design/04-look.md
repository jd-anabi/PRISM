## 4. The look: colours, type, sizes, and the pieces that carry them

This part states the visual system of U6 and U13 so that it can be built and tested. Its values are those of the PySide6 sample, `docs/superpowers/specs/2026-10-05-window-redesign-assets/toolkit-sample.py.txt` ("sample:N"), which rebuilt the chosen mock-up `look-chosen-scientific-page.html` ("mock-up:N"); where the two differ the sample is taken and the difference named. Only its values and devices are taken: its on-screen words predate U7, U9 and U10. Paths with no folder are under `core/gui/`. The code was read at `c45cc4a`.

The look is ONE table of named colours, type and sizes. Four things read it and nothing else holds a colour or a size: one application style sheet, one palette, one small style over Fusion, and the painted pieces. One thing leaves `core/gui/`: matplotlib's process-wide settings, as today (mpl_theme.py:31-33). Under E12 the engine's plotting code imports nothing of the window: it holds its own copy of four of `SERIES`' colours, which a test holds equal (C20; 4.9). Nothing here reaches an engine argument.

### 4.1 What replaces what

| file | today | after |
|---|---|---|
| `design.py` | `LIGHT`/`DARK`, 18 colours each, none for a state (:50-89); `TYPE` (:39-46); `RADIUS_*`, `CTL_H*`, `SPACE` (:18-22); `system_accent()` and the `accent` argument (:96-141, :144, :384); `_QSS` (:191-326); `ui_scale()`, `scaled()` (:329-362) | the table of 4.2 to 4.4 (`COLOURS`, `TYPE`, `SIZE`, `GAP`, `STEP`, `AXIS`, `DIALOG`, `FIGURE`, `SERIES`). `tokens(dark)`, `build_palette(dark)`, `build_qss(dark)` keep their names and lose `accent`; `system_accent()` is deleted. New: `set_dark(bool)` and `colour(name)`, the `QColor` in the theme in force (light when none was set, so a painted piece works under tests, which build no `Appearance`: theming.py:9-10). `ui_scale()` becomes `text_scale()` (4.8). The old two-column layout constants (:28-35) stay until that layout is retired. |
| `theming.py` | four modes, a 60 s timer, the accent switch; `_reapply` restyles whether or not anything changed (:100-115) | the modes, what goes and the stored values are as part 3 defines (3.9). Here: `_reapply` sets the colour scheme, resolves light or dark, and stops there when that is the scheme already applied; whether it goes on is one pure function of the two, so its test needs no `Appearance`. The order of an apply is in 4.8. |
| `fonts.py` | Segoe UI first, Inter only when forced (:17, :27-34); pixel size pinned to 14 (:27, :38) | Inter always. Two factories, `inter(kind, weight=None)` and `book(bold=False)`, build each font once, hand out copies (sample:184-209) and register the two files with Qt on first use, as the icon font does today (icons.py:31-45), so a painted piece has its face under tests too. `load_app_font(app)` loses `prefer_inter`, sets `inter("body")` and registers the files with matplotlib; `apply_class_fonts()` (4.7 rule 8). |
| `icons.py` | five glyphs in an icon font, drawn as button text (:19-25, :62-73) | deleted with its last caller, at the stage part 13 gives (13.2.1); glyphs are painted (4.6). Callers: screens/nav_shell.py:65, :97; widgets/help_badge.py:23; widgets/artifact_picker.py:33, :167; panels/inference/rows.py:44. Six more files import the module and call nothing in it, and lose the import: panels/inference/config_tab.py:12, prior_tab.py:12, posterior_tab.py:9, validate_tab.py:8, infer_tab.py:13, tsnpe_tab.py:10. |
| `mpl_theme.py` | sixteen colour settings from the neutral tokens (:16-28), applied again at every change of theme (:41-42) | the LIGHT settings of 4.9, written once (B18). `apply_mpl_theme()` loses its `dark` argument. `install(appearance)` still does nothing without an `Appearance` (:36-40) and no longer connects to `theme_changed`. |
| `app.py` | `app.setStyle("Fusion")` (:98) | the order of `build_app` is part 3's (3.10). Here: the style is `PrismStyle()`, the object kept referenced, and the focus follower is installed after `mpl_theme.install`. None of it runs under the suites, as today. |
| `assets/` | `fonts/`; `icons/` (`prism-icons.ttf`, its builder, `LICENSE.txt`); `app/` | `fonts/` and `app/` untouched. `icons/` removed with `icons.py`; git history keeps it (a choice the owner has not seen). No file is added: the sheet names no image, and the SVG reader does not load here (app_icon.py:3-7). |

Which test is rewritten when is part 13's, and this part adopts its order (C30; 13.1.2, Table B; 13.2.1). In stage 1, before the code: tests/test_user_models.py:1042-1062 (accent override) and :1065-1072 (Inter only when forced), tests/test_worker_dispatch.py:1007-1033 (`QTreeView` rules, accent in the sheet) and tests/test_figures.py:400 read the new names; tests/test_figures.py:423 is inverted: a change of theme changes no matplotlib setting; :321, :435 and :456 pin what the engine and the video export do under dark settings, so they set those by hand; :363 runs once, under the light settings. Not in stage 1: tests/test_user_models.py:1076-1150 (icon font) are retired with the font's last caller, and tests/test_nav_and_gating.py:4251-4252 (`QToolButton#navRunning`) and :4257 (the two `apply_icon` calls it counts in the shell) hold on the interim rule until the shell is rebuilt.

For stage 1 of U19 every old token maps to the new name of the same job. Two split: `base` to `field` inside a control and `paper` elsewhere; `mid` to `rule_light` for a separator and `field_line` for an outline. Readers of the old keys outside `design.py`: mpl_theme.py:17-27 and widgets/live_hair_bundle.py:113-114. `QPalette.Mid` and `QPalette.Highlight` stay correct (`rule_light`, `accent`) for the sparkline (widgets/progress_pane.py:132, :152) until it is retired. Each old widget class the sheet styles today (design.py:220-226, :246-319) keeps one interim rule in the new names. So do the two properties the old screens set themselves, at 29 places (`type` on a label, design.py:193-196; `accent` on a button, :208-214), and `#iconButton` (:217): stage 1 edits no screen, so without them every old main button and heading would lose its look. When each goes is part 13's (C30; 13.2.1). `SPACE` stays for its two readers (screens/artifact_screen.py:511-513; screens/model_builder_screen.py:188) until those screens are rebuilt.

New modules (the plan may rename a file once): `paint.py` (`paint_marker`, `paint_indicator`, `paint_hatch`, `fill_px`: pure functions of a painter), `style.py` (`PrismStyle`, `FocusRing`, `FocusFollower`, `set_class`), and under `widgets/`: `marker.py`, `ruled.py`, `rule_table.py`, `time_axis.py`, `message_row.py`, `page.py` (`Page`, `PageHead`, `Scroller`, `Columns`: sample:1388-1474, :1611-1623; part 5 uses them by these names and says what they do), `ask.py`, `choice.py`, `plot.py`. `widgets/labeled_inputs.py` keeps `IntField` and `FloatField`.

### 4.2 Colours

Thirty names (sample:57-88). The mock-up's values are the same (mock-up:83-132); its four frame colours are not taken, and `warn_mark` is the sample's name for the #1A1A1A the mock-up wrote into the triangle itself (mock-up:372).

| name | light | dark | its one job |
|---|---|---|---|
| `paper` | #FFFFFF | #1C1C1C | the page: every working surface |
| `margin` | #F2F2F2 | #141414 | the rail's grey; the scroll bar's track |
| `field` | #FFFFFF | #141414 | inside a field |
| `raised` | #FFFFFF | #262626 | inside a question; a tool tip |
| `ink` | #1A1A1A | #EDEDED | text; the "done" marker |
| `ink_2` | #4D4D4D | #B8B8B8 | secondary text, tick labels, the information marker and the padlock |
| `ink_3` | #636363 | #9C9C9C | hints, "to come", the idle dash, the scroll handle |
| `rule` | #1A1A1A | #C8C8C8 | the three rules of a table; an axis |
| `rule_light` | #D4D4D4 | #3A3A3A | a quiet separator, never the only sign of anything |
| `field_line` | #6B6B6B | #8A8A8A | the outline of a control |
| `accent` | #0072B2 | #56B4E9 | what is now, what is selected, the main action |
| `on_accent` | #FFFFFF | #0D0D0D | text and marks on the accent |
| `accent_tint` | #E2EEF6 | #1A3447 | the ground of the step in hand and of a selected row |
| `accent_hover` | #00639B | #7CC5EE | the main button under the pointer |
| `accent_press` | #005382 | #3FA3DC | the main button pressed |
| `ok` | #007A5A | #2FBF94 | the "fine" marker |
| `warn` | #E69F00 | #E69F00 | the fill of the attention triangle |
| `warn_edge` | #1A1A1A | #E69F00 | the triangle's edge |
| `warn_mark` | #1A1A1A | #1A1A1A | the stroke inside the triangle, both themes |
| `err` | #C24E00 | #F0813A | the error marker; a field in error; an action that loses work |
| `on_err` | #FFFFFF | #0D0D0D | the cross inside the error marker |
| `focus` | #1A1A1A | #FFFFFF | the focus ring |
| `hover_fill` | #F2F2F2 | #262626 | an ordinary button under the pointer |
| `press_fill` | #DEDEDE | #343434 | an ordinary button pressed |
| `disabled_fill` | #F2F2F2 | #232323 | inside a disabled control |
| `disabled_line` | #C4C4C4 | #4A4A4A | the outline of a disabled control |
| `disabled_ink` | #636363 | #9C9C9C | disabled text and marks |
| `hatch` | #4D4D4D | #B8B8B8 | the hatched range on the time axis |
| `elapsed` | #1A1A1A | #C8C8C8 | the elapsed bar on the time axis |
| `series_2` | #0072B2 | #56B4E9 | the second colour of `SERIES` |

Rules, each testable against the code (4.10):

- Lettering is only ever `ink`, `ink_2`, `ink_3`, `disabled_ink`, `on_accent` (on the accent) or `accent` (a link). `ok`, `warn` and `err` colour markers, outlines, edges and a figure's lines, never words.
- The accent has the three jobs of U6 and, as the chosen mock-up drew it (mock-up:178) and the sample built it (`toolkit-run-light.png`), a fourth: a link's words (a choice the owner has not seen: U6's own sentence names the three only). A figure's blue is `series_2`, of U23's figure set, not the accent, though the value is the same.
- The palette is built from the table, role by role as the sample's is (sample:219-240), with its Disabled group from the three `disabled_` names.

`SERIES` is the colour-blind-safe set of U23 for a figure's lines: `ink`, `series_2`, `err`, `ok`, `warn`, in that order, and nothing else: the set carries no line style (ruling D1). That it has five colours, and their order, are a choice the owner has not seen (the sample drew two lines and gave the first two). Its use is in 4.9.

### 4.3 Type

| name | face | size / line (px) | weight | used for |
|---|---|---|---|---|
| `small` | Inter | 12 / 16 | 400 | state words in the rail, tick labels, a second line in a table cell; at 600, a table's header |
| `body` | Inter | 14 / 20 | 400 | everything not named in another row |
| `strong` | Inter | 14 / 20 | 600 | the current place, a stage's name, the bold figure in a table |
| `head` | Inter | 16 / 22 | 600 | a section heading; at 400, the line that says where a run stands, under its stage line (C19; sample:1853-1854) |
| `brand` | Inter | 18 / 24 | 600 | the word "PRISM" in the rail, letters spaced at 104 % |
| `title` | Inter | 24 / 30 | 600 | a page's title |
| `book` | Cambria | 16 / 22 | 400, bold lead words | the four places below |

Five sizes and nothing between (sample:89-98). Two weights, because two files are bundled (`assets/fonts/Inter-Regular.ttf`, `Inter-SemiBold.ttf`). No italics.

**The two-face rule.** Inter is the face of everything operated or read off. Cambria, always 16 on 22, is used in FOUR PLACES ONLY, each one widget class (mock-up:50-55; sample:97):

1. the one sentence under a page's title;
2. a figure's caption;
3. the sentence of a message row (so every line under "Before you start", and a check's verdict sentence: C19);
4. the sentence of a question.

Everything in the rail, a table, a settings row, a button, an axis, the error sentence under a field (sample:1773), the reason beside a greyed button (sample:1657) and the line that says where a run stands (C19) is Inter. A whole sentence the program says anywhere else (the explanation "?" opens; Help's running text) uses the message row's sentence class, with or without a marker, so the classes stay four (a choice the owner has not seen); a fifth goes to the owner first. If Cambria does not resolve, the four places take Georgia, then Inter `body` (a choice the owner has not seen; mock-up:54-55, :112).

A third face (B19; a choice the owner has not seen): text quoted verbatim from the console (a run's full text, a record's log, the full error a question can show) is set in the system's fixed-width face at 14 on 20, as screens/artifact_screen.py:225 does today.

**Digits of equal width** are switched on in code for every Inter font: `font.setFeature(QFont.Tag.fromString("tnum"), 1)` (sample:193-196; PySide6 6.9.3 takes only a `Tag` here). The application font is `inter("body")`, so the feature is inherited wherever no font is set. Cambria's digits were measured equal already. Lettering drawn by matplotlib keeps proportional digits (4.11).

**Numbers and dates.** A whole-number quantity of four digits or more carries a comma, in a field while it is typed, in a table and in a sentence: "20,480,000". An identifier (a seed, a record's id) does not (a choice the owner has not seen). A number and its unit are separated by a space, per cent included: "40.1 %", "6 GiB". A date in a table, a record or a sentence is written "3 Oct 2026, 14:02" (B22). On the run's page and at the rail's foot a time within seven days of now carries its weekday and drops its year, "Sun 4 Oct, 12:10", as the approved pictures did; a tick of the time axis is shorter still (C18; 4.6). All read the same on every computer, whatever Windows' regional setting: a full stop for the decimal mark, a comma between thousands, three English letters for the month and for the weekday (a choice the owner has not seen). So that this can be held, each is written by one function of `design.py` (`number_text`; `date_text`, which gives the weekday form only when it is handed the present moment and the date lies within seven days of it; the plan may rename them), which every part calls and 4.10 pins.

**Rich text.** In Inter, bold is `font-weight:600`; a bare `<b>` asks for 700, which the two files do not have. A fixed line height needs `<p style='line-height:Npx'>` and a negative top margin of half the extra (sample:1290-1300).

### 4.4 Sizes

Sizes are the sample's `SIZE`, `GAP`, `STEP`, `AXIS`, `DIALOG` and `FIGURE` tables unchanged (sample:102-151), in logical pixels, held in `design.py`. The ones other parts lay out by are below; what the small layout, the two-column switch and a page's pinned foot do is as parts 3 and 5 define.

| group | values |
|---|---|
| window | opens at 1300 x 820 or what the screen offers (main_window.py:44-49, :92); smallest 900 x 600 (:95) at Windows' ordinary text size, growing with the text size and held to the screen (C17; 4.8); below 700 high the small layout; a page narrower than 940 sets two columns one above the other |
| page and rail | page padding 22, 20, 20, 14 (small layout 16, 16, 14, 14); rail 220 wide; a part's row 36 high (small 32); a step's row 46 (one line: 32) |
| controls and forms | field 32 and button 34 high; corner radius 3; number field 120 wide; label column 200 (165 in a side column); row gap 12 |
| markers, lines, tables | marker 16 (24 in a question); heavy rule 2; hairline 1; accent bar 4; focus ring 2, clear of its control by 2; a table's row at least 30 and its header 27; a question 448 wide |

### 4.5 States

One marker shape per state, always with its word. Ten shapes, drawn by one function (sample:379-447; `toolkit-markers-light.png`, `-dark.png`). No part adds an eleventh.

| kind | shape | colours | word | says |
|---|---|---|---|---|
| `done` | a disc with a tick | `ink`, tick `paper` | "Done" | complete and saved |
| `now` | a dot | `accent` | "Now", then what: "Now: running" | the step in hand; what is running |
| `later` | an open, broken circle | `ink_3` | "To come", or what it needs | not started |
| `lock` | a padlock | `ink_2` | "Blocked"; "Locked" when the cause is a live run | cannot be used or changed until something else is done or over |
| `half` | a half-filled disc | `ink`, `paper` | "Unfinished" | stopped before its end; what was written is kept |
| `none` | a dash | `ink_3` | "Nothing is running" | idle |
| `fine` | a ring with a tick | `ok` | "Fine" | a check passed; a figure within its limits |
| `info` | a ring with an i | `ink_2` | "Information" | to know; nothing to do |
| `warn` | a triangle with a mark | `warn`, edge `warn_edge`, mark `warn_mark` | "Attention" | read before going on; nothing has failed |
| `err` | a square with a cross | `err`, cross `on_err` | "Error" | failed, or stops a start |

The sample's key sheet prints "Blocked" beside the padlock (sample:2058); the mock-up prints "Locked" (mock-up:764, :793, :825), U11's word for settings during a run. Both are kept for the one shape, each for its cause (a choice the owner has not seen). A step that cannot start yet is not a padlock: it is "To come" with what it needs (sample:1516-1518).

States the other parts name, which no picture drew, take these shapes with their own words (a choice the owner has not seen):

| state words, and whose they are | shape |
|---|---|
| "Needs you" (U11); "Look closer" (B1); "Not compared" (C16); "Changed, not applied" as a step's state in the rail (U14; part 3) | `warn` |
| a check (part 9): "Not run yet", "Waiting for …" / "Running" / "Passed" / "Failed" / "Does not apply" / "Done" for an informing check or a tool that has run | `later` / `now` / `fine` / `err` / `none` / `done` |
| a record (part 10): "Finished" / "Checks only", "Leftover" / "Unfinished" / "Being written" / "Unreadable" | `done` / `info` / `half` / `now` / `err` |
| a run that finished / was stopped / failed (part 3) | `done` / `half` / `err` |

**Focus against selection.** They never share a sign.

| | what is seen | made by |
|---|---|---|
| keyboard focus | a ring 2 px wide in `focus`, 2 px clear of the control, corners 6; on a control that fills its row (a part of the program, a step, a table's row, a scrolling page) the ring is just inside the edge | `FocusRing` |
| a selected row | ground `accent_tint`, a 1 px `accent` line above and below, a 4 px `accent` bar at the left, the first cell semibold | the table's delegate |
| the current place | one row of the rail carries `accent_tint` and the 4 px bar, its name semibold; which row is as part 3 defines (3.3) | the step row; the rail |

The ring is shown after Tab, Shift+Tab or a shortcut, and after a mouse click only in a field one types in; an opened list marks its own row (sample:848-874). The toolkit's dotted focus mark is never drawn. On a selected row the ring starts to the right of the bar, so both are whole (sample:1103-1109). Today's sign is a border in the selection's own colour.

**Disabled.** Ground `disabled_fill`, outline `disabled_line`, text, markers and glyphs `disabled_ink`; a disabled main button has no accent; a ticked box keeps its tick (sample:470-475). Disabled text measures 5.37 to 1 in light and 5.72 in dark, so a form locked for days stays readable; today it is 2.44 and 2.48 (design.py:178-180).

**Error on a field.** `set_class(field, state="error")` gives a 2 px `err` outline, taken from the inner height and the padding so that neither the field nor its text moves (sample:301, :327); the look's test finds the field's size the same with and without the class. Six pixels below stand the error marker and one sentence in Inter saying what to do (sample:1762-1774). The outline alone is never the sign.

**A value that cannot be edited is not drawn as a field.** It is text in the field's column, the value semibold, a remark in `ink_2` (sample:1783-1785, which drew it for one row; as a rule, a choice the owner has not seen). A setting that is a field but fixed for now (set automatically; locked by a run) is a disabled field with its reason beside it (sample:1723). Part 8 applies this to the two measured definitions of U8, which closes W110.

### 4.6 The shared pieces

What a row, a list or an axis CONTAINS, and which old widget it retires, belongs to the part that owns the page. Of this part's own, the pieces replace the "⚠ " and "✖ " prefixes (widgets/log_pane.py:8) and box icons as the only sign of severity (W106), group boxes drawn as cards (design.py:264-271), and focus shown as a 1 px border (design.py:206, :213, :226, :235).

| piece | what is seen | made as |
|---|---|---|
| marker | 4.5; glyphs: chevron right and down, and up for a sorted column (not in the sample), a circled question mark, the vermilion square on an action that loses work | painted: `paint_marker(painter, kind, box, disabled=False)` in a 16-unit box (sample:379-462); `Marker(kind, size)`; `PaintedIcon`, an icon engine for buttons, sharp at any scaling (sample:525-562) |
| ruled line; section heading | a line of one weight wherever it falls; a heading is 16 px semibold with a 2 px `rule` line 6 px under it | painted: `Ruled(top, bottom, colour)` paints its own edges through `fill_px`, on whole device pixels (sample:565-603, :1336-1340) |
| page heading | the title; under it one sentence in the book face, `ink_2`; actions at the right, 12 apart | `PageHead` (sample:1388-1403) |
| settings row; facts list | as part 5 defines (5.2, 5.4). A row's remark is `ink_2`; the facts' reading time is `small`, at the right of their heading | a grid row, label column 200, row gap 12 (sample:1719-1734); layout over message rows (sample:1625-1681) |
| fold | a hairline above and below; a chevron and a semibold line, "Rarely changed (7 settings)"; under it the values in `small`, `ink_2`, indented 22. Opened, the chevron points down and the rows stand in place of the summary (only the folded state was pictured: a choice the owner has not seen) | `Ruled` in `rule_light`; a `fold` button as wide as its words (sample:1736-1746) |
| message row | a marker; one sentence in the book face whose first word, with its date when it has one, is bold: on the run's page "Attention, Sun 4 Oct, 12:10." (C18). An error row has a 2 px `err` edge at the left. No tinted ground | `MessageRow`: a marker and one label; the edge is painted (sample:1406-1422) |
| step row | as part 3 defines (3.4): the number in `ink_2`, the state words in `small`, a 2 px line on to the next step, solid `rule` after a finished step, dotted `ink_3` otherwise | painted: an abstract button (sample:758-818) |
| focus ring | 4.5 | painted: the toolkit's focus frame, placed by the style and shown by one application event filter (sample:493-522, :822-902) |
| "?" | the circled question mark, 16 px | a frameless button with a `PaintedIcon`; not a stop for the Tab key, and what opens it is as part 5 defines (5.8) |
| figure in the page | 4.9 | `Plot`: Agg at the widget's size and scaling, kept until size, scaling, theme or options change (sample:1215-1253) |

#### Buttons

Stock push buttons, coloured by the sheet through a class set with `set_class` (sample:272-294).

| kind | class | resting | under the pointer / pressed |
|---|---|---|---|
| ordinary | none | `paper`, 1 px `field_line`, `ink`, 34 high | `hover_fill` / `press_fill` |
| main, one to a page | `primary` | `accent` ground and outline, `on_accent`, semibold | `accent_hover` / `accent_press` |
| loses work | `danger` | `paper`, 1 px `err`, `ink`, semibold, the vermilion square before its words | as ordinary |
| link | `link` | no frame; `accent`, underlined; 20 high; it is a button, so it takes the focus and the ring | none |

The rail's rows and a fold's line are frameless buttons of the classes `nav` and `fold`, with no pointer-over state (4.11). The accent comes from the class alone, never from `:default`, which follows the keyboard focus from button to button (sample:257-259). A button names its outcome (U7); an ellipsis ends the name of one that asks first: "Stop training…".

#### Fields

| field | made as | what is seen |
|---|---|---|
| text | stock line edit, sheet rule (sample:296-301); hint and selection colours from the palette | 32 high, 1 px `field_line`, radius 3, text 8 in; hint text in `ink_3` |
| number | `IntField` and `FloatField`, line edits, never spin boxes: no stepping on the wheel or the arrow keys | right-aligned, 120 wide; in a whole number the comma appears as the digits are typed and the caret keeps its place (sample:1363-1385); a decimal is as typed |
| drop-down | `Choice`, a stock combo box that paints its own chevron (sample:1352-1360) | the chevron in a 16 px box, 8 inside the right outline; the opened list on `field` with a 1 px `field_line` edge |
| tick box, radio | stock; the 18 px indicator is painted by `PrismStyle` (sample:465-522) | empty: `field`, 1 px `field_line`; set: `accent` with an `on_accent` tick or dot; 10 (radio 8) to its words |

Today `IntField` holds a `QIntValidator` and reads its text with `int()` (widgets/labeled_inputs.py:61-85); with the comma in the text `value()` would return 0 and `value_or_none()` None. So both strip the comma, the validator goes (the field keeps digits and a leading minus itself), and a whole number set from code is grouped the same way. What tests/test_settings_persistence.py:954-970 pins stays true: "-3" reads −3, and text set from code that is not a whole number ("1.5", "-") is shown as given and reads None. The constructor still takes a number or its text, as today's callers pass both (panels/inference/prior_tab.py:105-107). A seed is an identifier and today's seed boxes are `IntField`s (panels/fdt_panel.py:134; panels/crossval_panel.py:133), so `IntField(grouped=False)` shows digits as typed. A remembered text may carry commas (settings.py:96-107); reading strips them. The decimal mark is a full stop and the separator a comma on every computer (4.3), so `FloatField` reads its text by the C locale's rules; today its validator takes the system's (widgets/labeled_inputs.py:22). A comma typed as the decimal sign is not silently swallowed (ruling D9): the key shows, the text is never read as a number, and when editing ends the field is in error (4.5) with the sentence "Use a point as the decimal sign.". A tool tip is `raised`, `ink`, with a 1 px `field_line` edge (sample:261); nothing is said in a tool tip alone. A switch between a few views of one thing is a row of radio buttons, as "Recent" and "All epochs" (sample:1882-1888); the look has no tab bar.

#### The three-rule table

`RuleTable` paints a 2 px `rule` line above and below; `RuleHeader` paints its own words (`small`, semibold) and one 1 px `rule` line under them in a single stroke; there are no lines between rows (sample:992-1020, :1112-1118). A cell is a small description, painted by one delegate (sample:924-989): text (plain or `ink_2`; a name too long is cut in the middle, a note at the end; or wrapped), a value (the figure semibold, its unit plain, an optional second line in `small`; right-aligned), or a state (marker and word). A column is fixed, fitted to its widest cell (at most 180) or stretching. The table is as tall as its rows; the page scrolls. Row heights are set explicitly: the toolkit's own fitting made each row 1 px too tall (sample:1089-1098). The sample's header cannot be clicked: for a table that sorts (Records, today widgets/artifact_table.py:41-53, :249) `RuleHeader(sortable=True)` makes its sections clickable and paints the chevron, up or down, after the sorted column's words (not pictured: a choice the owner has not seen).

#### The time axis

`axis_layout(start, now, end_from, end_by, width, text_width)` returns plain numbers and is tested without a screen; `TimeAxis(start, now, end_from, end_by)` only paints what it returns (sample:637-726; eight moments in `toolkit-time-axis-light.png`). The track is outlined in `rule`; elapsed time is filled in `elapsed`; the range in which the run should end is hatched from `end_from` to the end of the axis, and shows through the bar in `paper` strokes once "now" is inside it. The "now" mark is `accent` with 1 px of `paper` each side, stands above and below the track, and never leaves the axis. A run past its range gets an axis a whole day longer and is marked late.

Ticks: every 30 s under 5 minutes; every 15 minutes under 2 hours; every 6 hours on the clock under 2 days, the day before the clock at a midnight ("2 Oct 00:00"); beyond that at each midnight, with the day ("2 Oct") between two midnights. A tick stays this short, with no weekday and no year (C18), where the pictures had the weekday's name (a choice the owner has not seen). A label is left out sooner than printed over its neighbour.

Under the axis one line (sample:729-755), its times written as the run's page writes them (C18; 4.3): left "Started Thu 1 Oct, 19:02"; right the hatch key and "Expected to end between Tue 6 Oct, 19:00 and Wed 7 Oct, 19:00", or "Expected to end by Wed 7 Oct, 19:00" once inside the range, or the attention marker and "Later than expected: it was due by Wed 7 Oct, 19:00". The accessible name gives the same in words.

Two forms the sample did not draw (a choice the owner has not seen). A firm end is `end_from == end_by`: no hatch, and the words "Expected to end by …". No end known is `end_by=None` (B9): no hatch, and nothing is called late; the axis runs from the start to `end_from` when one is given (the time before which the run cannot end), or to now when none is given or now has passed it, and never less than 30 s, so its span is never zero; the right-hand words are the run part's (6.3.3).

#### The question

One class serves every question, every notice and "Check your inputs" (B2); the window no longer uses `QMessageBox`, and every modal dialog it opens is this class or the one file-chooser function of 13.1.4 (ruling D4). `Ask` is a `QDialog` instance in the system's own frame, modal to its window, laid out as the sample's (sample:1964-2010; `toolkit-stop-question-light.png`): a 24 px marker, the sentence in the book face, and the answers at opposite ends, the safe one at the right as the main button. The safe answer is the default and stays the default: Enter, Escape and the frame's close button all mean it wherever the focus is; the other answer is pressed with the mouse or the space bar (sample:1993-2005), and is drawn as `danger` only when it loses work (B27). The title bar follows the theme through `styleHints().setColorScheme` (theming.py:107).

`ask(parent, title, sentence, safe, other=None, *, loses_work=False, marker="warn", items=None, beneath=None, detail=None)` is the one way in; part 5's `show_refusal` calls it. It returns in line: True for the other answer, False for the safe one. With `other=None` it tells and has one button. `items` is a list under the sentence, one item to a line in the sentence's face, none cut, scrolling inside the question beyond eight (C10): what a decision must name, as U10's every difference, goes there, never behind a link. `beneath` is a closing paragraph in the book face, `ink_2`, under the list when there is one: the fix sentence of a refusal (5.7). `detail` is verbatim text, a traceback, behind a link "Show the full error", which opens it under the answers in the fixed-width face; an error's kind, its type name, is shown there and nowhere else (C7). The marker is `warn` for a question and a refusal, `err` for a failure, `info` for a notice. (`items`, `beneath`, `detail` and its link were not pictured: a choice the owner has not seen.)

The class has one further form, which asks for one line of text (ruling D4); "Save as a plan…" (7.1.10) and "Rename…" (10.2) use it. `ask_text(parent, title, sentence, label, other, *, text="", check=None, items=None, beneath=None)` is the same `Ask` with one labelled text field under its sentence: `check` judges the text as it is typed, its sentence stands under the field as on a page (5.5), and the answer `other` is greyed until the text holds. It returns the text for `other` and None for the safe answer, "Cancel", which is Escape's and the frame's close button's. In this one form Enter in the field means `other` once the text holds, and does nothing while it does not: the form loses no work (neither use replaces anything without a question of its own), and a line of text that Enter throws away would be a trap (a choice the owner has not seen). It opens through `Ask.exec` like every question.

The sample opened its question with `open()` and went on in a slot (sample:2008-2010, :2053). Here `ask` blocks, through `Ask.exec`, because its callers need the answer before they return: a closing window must accept or ignore its event, and a start click must know before it dispatches. Nothing seen depends on the call. The session guard, which patches `QMessageBox.exec` only today (tests/conftest.py:170-190), is extended to `Ask.exec` in the task that introduces the class, as part 13 defines (13.1.4; B2), and so covers both forms (ruling D4).

### 4.7 Rules for the style sheet

Learned in the sample. Rules 1 to 4 and 10 are checked on the sheet's own text by the look's test (4.10).

1. A selector names a widget class, narrowed where it has variants by a property (`cls`, `state`, `current`, `small`); never `*` or a bare `QWidget` (today's rule, design.py:188-190).
2. Colours, borders, padding and least heights. Sizes and weights by class. Never a family, letter-spacing or a font feature: a sheet cannot do the last two, and faces are set in code.
3. No weight or size inside a pseudo-state (`:hover`, `:checked`, `:default`): the control is not measured again and its text is cut.
4. No image: no `url(`. The tick, the dot and the chevron are painted.
5. No border or padding on a widget that holds a layout, and no font on a header section: `Ruled` and the header paint their own.
6. A class is set only through `set_class`, which polishes the widget again; a property set after the first polish otherwise changes nothing (sample:243-254).
7. A height the sheet gives cannot be undone in code; a variant is a class property (sample:1585-1587).
8. Setting the application sheet puts the system's font back on lists, tables, headers, menus and tool tips (and on message boxes while any remain); `apply_class_fonts()` sets Inter on those classes again after every `setStyleSheet` (sample:2118-2123).
9. A rule for a class's pseudo-state comes after the class's own rule; at equal weight the later rule wins.
10. The sheet is filled by `string.Template` (design.py:13, :191), where a size and a colour passed under one name silently replace each other. `build_qss` gives sizes their own prefix, and the test finds no `$` left, as tests/test_worker_dispatch.py:1026 does today.
11. With a border of the sheet's own on a drop-down the style is no longer asked for its arrow (sample:1352-1354), so `Choice` paints the chevron. This replaces today's warning not to clear the arrow (design.py:229).

### 4.8 Appearance, display scaling, text size

**Appearance** has three choices, "Follow Windows", "Light", "Dark" (U13); part 3 defines the modes and part 10 the page that offers them. An apply does, in order (sample:359-375): `setColorScheme`, so the frames Windows draws follow; `design.set_dark`; `setPalette`; `setStyleSheet`; `apply_class_fonts()`; `retheme()` on each widget that has it (only the two kinds that cache: a figure the window drew, a label whose rich text holds a colour); an update of every top-level widget; the ring put back on its control; then `theme_changed(dark)`, to which the Live simulation view listens as today (widgets/live_hair_bundle.py:109). Every other painted piece reads `design.colour()` when it paints and keeps no colour. The appearance stays changeable while a run is live (C29): nothing the engine draws depends on it.

**Display scaling.** PRISM sets no scaling policy, so Qt passes 125 % and 150 % through. What must hold, as pictured in `toolkit-run-light-150-percent.png`: a painted rule is a whole number of device pixels, the same at every position (`fill_px` works the thickness out once, rounds it down and never below one, sample:565-582), so a table's rules are 2, 1, 2 device pixels at 100 % and 125 %, 3, 1, 3 at 150 %, 4, 2, 4 at 200 %; a figure the window draws is rendered again when the window moves to a screen with another scaling (sample:1236-1252); and on a 1920 x 1080 screen at 150 % the window is lower than 700, so the small layout is what a person gets.

**Text size** (U13; W102). Today `fonts.py:38` pins the application font at 14 px, which holds `ui_scale()` at 1.0 for ever (design.py:346-355). Whether the toolkit's default font already follows Windows' text-size setting is established by a scratch test in stage 1, before any mechanism is built (B20): at 100 % and at one larger step it reads the size of a fresh application's default font, and whether a change made while it runs arrives. One factor, `design.text_scale()`, is then built on what the test finds: the default font's size against its size at 100 % if the toolkit follows the setting, and otherwise what Windows itself reports, read so that it can never raise (the candidate, the `TextScaleFactor` value under `HKEY_CURRENT_USER\Software\Microsoft\Accessibility`, is not established). The source, the factor's upper limit, and whether a change is seen while PRISM is open or at its next start wait for the test.

What holds either way: the factor is 1.0 off Windows and on any failure, and `scaled(n)` is `floor(n * text_scale() + 0.5)`. Every size and line height in `TYPE` and every length in `SIZE`, `GAP`, `STEP`, `AXIS`, `DIALOG` and `FIGURE` is scaled before the first widget, so the sheet, the fonts and the painted pieces all read scaled values. Not scaled: line weights and radii (the rules, the accent bar, the ring and its gap, the hatch step, the scroll bar, the "now" mark, a figure's axis and line widths). The window's opening size, its smallest size and the two layout thresholds are scaled and then held to what the screen offers; the rail then scrolls its list between the name and the run's state, as part 3 defines (3.8). That lengths grow with the lettering, and that the smallest window is therefore 900 by 600 at Windows' ordinary text size only (C17), is a choice the owner has not seen.

### 4.9 Figures

**Figures the engine builds** (corner plots, calibration grids, effective-temperature plots, the loss figure at the end of a training) are always drawn on a light page, in both themes (U23, B18). They are built on the run's thread under matplotlib's process-wide settings, rendered to PNG and shown as pictures (panels/base_panel.py:31-63; widgets/figure_stack.py:19-46). `mpl_theme` writes the LIGHT column of the table into those settings once, when the application is built, and a change of theme no longer touches them. So:

- a figure stored in a record never carries the dark colours: this closes W080;
- a run that straddles a change of theme can no longer split a figure between two themes, the failure `save_figure` guards against (core/Helpers/visualizers.py:13-35); it and its callers stay as they are;
- in the dark theme a stored or newly built figure is shown as it is, a light picture on the dark page, neither inverted nor tinted (U23), in part 5's viewer (5.9);
- the video export pins its own white page and black lettering (panels/simulate_export.py:91-116) and needs no change in code. It pins nothing else, so its frames follow the light figure settings like any stored figure (C20): they take the settings' axis style and face, and the trace, which names no colour (:100), turns from matplotlib's blue to the light `ink` in both themes (a choice the owner has not seen).

`mpl_theme._rc()` becomes the sample's settings (sample:335-356) at their light values: page, saved page and axes ground `paper`; spines and ticks `rule`; text, axis names and titles `ink`; tick labels `ink_2`; top and right spines off; ticks outward, 3.6 pt long, 0.72 pt wide like the spines; lines 1.44 pt with round joins; `font.family` "Inter", set only when both files were registered with matplotlib's font manager (sample:2110-2112); `axes.unicode_minus` on. Added: `grid.color` `rule_light`, and `axes.prop_cycle`: the five colours of `SERIES` and nothing else (ruling D1; below). Three departures (a choice the owner has not seen):

- `axes.facecolor` is `paper`, not "none": the engine's helpers read it as a solid colour for a table's cells and a summary box (core/Helpers/visualizers.py:202, :301-310), and tests/test_figures.py:391 requires the cells opaque. On the page the two look the same.
- Legends keep a frame: `legend.facecolor` and `legend.edgecolor` `paper`, `legend.framealpha` 1.0, a patch of the page's colour with no outline. The sample switched frames off, having no legend; the engine's figures have them, often over the data (core/Helpers/visualizers.py:178; core/FDT/compare.py:546), and core/Reduction/plots.py:34-39 relies on the ground.
- The three text sizes (sample:340, :345, :352) are NOT set process-wide: the engine's figures are composed at their own sizes, rendered at 110 dots per inch or more and shown scaled down (panels/base_panel.py:59; core/FDT/plots.py:87; widgets/figure_stack.py:73-76), so a smaller base size would shrink their lettering on screen.

**The lines' colours (E12; C20).** The plotting code stops naming its own line colours (`steelblue`, `darkorange`, `crimson`, `red`, `green`, the `tab:` colours: for example core/Helpers/visualizers.py:157-172, :441-449; core/FDT/plots.py:63). One engine module holds four colours by role, the light values of `SERIES`' second to fifth (blue #0072B2, vermilion #C24E00, green #007A5A, orange #E69F00); `tests/test_look.py` holds them equal to `SERIES` (13.1.5); the first, the ink #1A1A1A, comes from the figure settings. Which line takes which colour, and what stays as it is, are part 11's (11.14).

A line that names no colour takes the settings' cycle, where matplotlib's own, which such lines take today, has ten colours: a comparison of many saved runs (core/FDT/compare.py:545, :607, :880), and the calibration's cumulative rank figure, which the fitting library draws with one line per parameter on one axes, each coloured by its place in the cycle (core/orchestrator.py:2013). The cycle is the five colours of `SERIES` and carries no line style: C20's dashed repeat is withdrawn (ruling D1), and E12 edits no line style, as 11.14 has it (B18, U4). A figure with more unnamed lines than colours repeats colours, as any cycle does, so thirteen parameters show each colour two or three times; telling many curves apart by style is not designed in.

Nothing else about a figure changes (U4): not which figures exist, their titles, axis names and units, their sizes and stored resolution, their legends' place, their colour maps or the pop-out window.

**Figures the window draws itself** (the loss curve on a run's page, from the per-epoch lines of E1; the Live simulation view) follow the theme (U23). The live view keeps reading the table at each `theme_changed` (widgets/live_hair_bundle.py:111-122: `paper` ground, `ink` axes and trace). The loss curve is drawn like a journal figure, as the sample's is (sample:1138-1202): left and bottom axes only, ticks outward, no grid and no legend box, each line named where it ends; validation in `ink`, training in `series_2` and dashed, so the two differ without colour; the best epoch an open ring with its label. In its empty top-right corner sit two radio buttons, "Recent" and "All epochs", the second set at first. What "Recent" covers, and the figure's data, are as part 6 defines (6.3.7).

Because the process-wide settings are now the light ones and belong to the engine's figures, `Plot` passes every colour, weight, dash and size to each call itself (the page, the spines, the ticks, each text, each line), from `design.colour()` when it draws, and draws again on `retheme()`. It never changes the process-wide settings, even for a moment: another thread's figure could pick them up (the reason panels/simulate_export.py:93 gives). It is a `matplotlib.figure.Figure` on an Agg canvas, never pyplot, as part 6 has it; its lettering is Inter at `small`. Such a figure is rendered on the window's thread while the run's thread may be building one of its own; the pop-out window already does this (widgets/figure_window.py:72-79), but that the two never collide is not established by reading. So part 13 lists, as a task of stage 3 before the loss curve is built, a test that renders such a figure on the window's thread many times while another thread builds and saves a figure (13.2.2). If they can collide, the loss curve is drawn from its numbers by a painted widget and no figure library runs on the window's thread (ruling D7).

### 4.10 The contrast that must hold, and its test

Ratios by the WCAG 2 formula, worked from the table in 4.2; the formula reproduces the mock-up's own stated figure (17.4 to 1, mock-up:61). One torch-free module, `tests/test_look.py` as part 13 names it, computes every pair from `design.COLOURS`, so a changed colour fails by name.

| what | floor | pairs | lowest now, light / dark |
|---|---|---|---|
| lettering | 4.5 | `ink` on `paper`, `margin`, `field`, `raised`, `accent_tint`, `hover_fill`, `press_fill`; `ink_2` on `paper`, `margin`, `raised`, `accent_tint`; `ink_3` on `paper`, `margin`, `field`; `disabled_ink` on `disabled_fill`, `paper`, `margin`; `on_accent` on `accent`, `accent_hover`, `accent_press`; `accent` on `paper`, `raised` | 5.19 (`accent` on `paper`; `on_accent` on `accent`) / 5.72 (`disabled_ink` on `disabled_fill`) |
| marks and outlines | 3.0 | `field_line` on `paper`, `field`, `raised`, `margin`; `rule`, `hatch`, `elapsed`, `series_2` on `paper`; `accent` on `paper`, `margin`, `accent_tint`; `ok`, `err` on `paper`, `raised`, `margin`, `accent_tint`; `on_err` on `err`; `warn_edge` on `paper`, `raised`, `margin`, `accent_tint`; `warn_mark` on `warn`; `focus` on `paper`, `margin`, `raised`, `accent_tint`; `ink_2`, `ink_3` on `paper`, `margin` | 4.06 (`err` on `accent_tint`: a marker in a selected row or in the step in hand) / 4.38 (`field_line` on `raised`) |

Named exceptions. Each is asserted as it stands, BELOW its floor (3.0 for a mark, 4.5 for lettering) in the theme where its figure here is under it, together with what carries it:

| pair | light / dark | carried by |
|---|---|---|
| `warn` on `paper`; and on `raised`, `margin`, `accent_tint`, where the triangle also stands | 2.25 / 7.57; 2.25, 2.01, 1.91 / 6.72, 8.18, 5.73 | the triangle's edge (17.40 / 7.57; on the other three no less than 14.75 / 5.73) and its mark (7.73 on the amber) |
| `rule_light` on `paper` | 1.48 / 1.50 | it is never the only sign of anything |
| `disabled_line` on `disabled_fill` | 1.56 / 1.77 | a disabled control; its text passes |
| `accent` on `elapsed` | 3.36 / 1.38 | the "now" mark has 1 px of `paper` each side and stands above and below the track |
| `focus` on `accent` | 3.36 / 2.31 | the ring is 2 px clear of the main button; an inset ring starts after the accent bar |
| `accent_tint` on `paper` | 1.18 / 1.32 | a selected row also has two lines, a bar and a semibold name |
| `accent` as lettering on `accent_tint` | 4.39 / 5.60 | not allowed: a link never stands on a selected row |
| `SERIES` fifth colour (`warn`) on `paper` | 2.25 / 7.57 | a figure's orange lines only (a line drawn orange today, or a fifth or tenth line that names no colour), always on a light page, where today's oranges measure 1.97 to 2.53; the other four are 4.79 or more |

Beside what part 13 lists for it (13.1.5), the module asserts, without a screen: the sheet in both themes has no `$`, no `url(`, no `font-family`, no `letter-spacing`, no `:default`, no `*` or bare `QWidget` selector, no weight or size in a rule with a pseudo-state, and no text colour but a lettering colour of 4.2 (4.7); no `.py` file under `core/gui` but `design.py` writes a hex colour, none but `theming.py` sets a sheet, and `book()` is called by its four classes only; `number_text` and both forms of `date_text` as 4.3 writes them, whatever the regional setting; the figure settings are the light column, their cycle is the five colours and carries no line style (ruling D1), and a change of theme changes none of them; `axis_layout` at the eight moments of sample:2072-2089, with a firm end and with no end; `fill_px` gives the weights of 4.8 at scalings 1, 1.25, 1.5 and 2; typing 20480000 into an `IntField` shows "20,480,000" with the value 20480000; a question answers Enter and Escape with the safe answer from either button, and with nine items it scrolls and cuts none; `scaled()` at factors 1.0, 1.25 and 2.0.

### 4.11 What the toolkit costs

Named when the sample was built and still true:

- Lettering inside a figure is drawn by matplotlib: the same face and size, a little softer at 100 %, with proportional digits.
- Inter is fitted to whole pixels: lines run about 1 % wider than in the mock-up, so no width may be copied from it.
- The hatching is greyer in dark than in the mock-up.
- An opened drop-down paints its current row in the full accent, not the tint the sheet asks for.
- Typing in a number field clears that field's undo history, because the text is set again at each key.
- The rail's rows and the fold give no sign of the pointer over them or of being pressed.
- The focus follower is one event filter on the whole application; its cost in the real window was not measured.

Not yet seen by anyone, so owed as walkthrough rows in part 13's list (13.4): the system's frame round a question in light and dark; a light figure on the dark page; each kind of figure the engine builds, and a frame of the exported video, under the new figure settings (only the loss curve and one trace were pictured); pointer-over states; the ring as a real keyboard moves it, arrow keys inside a radio group included; a field's 1 px outline at 125 % and 150 %; any text size but 100 %; a decimal number field.

### Choices in this part the owner has not seen

- When Windows' text size is raised, spacing, controls and the smallest window grow with the lettering; line weights and corner radii do not. The smallest window is therefore 900 by 600 only at Windows' ordinary text size, and larger above it, though never larger than the screen.
- A figure's lines are coloured from a set of five, in the order ink, blue, vermilion, green, orange; a figure with more lines than that repeats the colours, and nothing but colour tells such lines apart (today such lines take the plotting library's ten colours). It shows first in the calibration's cumulative rank figure, which has one curve per parameter.
- The exported video keeps its white page and black lettering; its axes and typeface take the same style as every other stored figure, and its trace turns from blue to ink.
- In a dialog that asks for one line of text ("Save as a plan…", "Rename…"), Enter means the main answer once the text is acceptable and does nothing before; Escape cancels. In every other question Enter means the safe answer.
- States that no picture drew reuse the ten shapes: "Needs you", "Look closer", "Not compared" and "Changed, not applied" get the attention triangle, "Being written" and a running check the blue dot, a check that does not apply the dash, a leftover or checks-only record the information ring, an unreadable record the error square.
- The padlock carries two words: "Blocked" in general, "Locked" when the cause is a live run.
- Text quoted from the console (a run's full text, a record's log, a full error) is set in a fixed-width typeface, a third beside the look's two.
- The question dialog can also carry a list under its sentence (one item to a line, scrolling beyond eight), a closing line and, for a failure, the full error behind a link "Show the full error"; none of the three was pictured.
- When no end can be estimated, the time axis has no hatched range and runs to the earliest moment the run could end, or only to now; a firm end has no hatched range either. Neither was pictured.
- On the run's page a time within seven days is written "Sun 4 Oct, 12:10", where the pictures had "Sun 12:10"; on the time axis's ticks a day is written "2 Oct", where the pictures had the weekday's name.
- Figures the engine builds keep their own text sizes; only their typeface, colours and axis style change.
- A figure's legend stands on a patch of the page's colour with no outline (the sample's figures had no legend).
- The area inside a figure's axes is filled with the page's colour instead of being left transparent; the two look the same.
- Quantities of four or more digits carry a comma; identifiers such as a seed or a record's id do not.
- Numbers and dates are written the same on every computer (a full stop, a comma, English letters for the month and the weekday), whatever Windows' regional setting.
- A setting's explanation and Help's running text are set in the book face in the same way as a message's sentence, with or without a marker, so that the book face keeps its four uses.
- Links are in the accent blue, a fourth use beside the current place, the selection and the main action; the chosen mock-up drew them so.
- A value that cannot be edited is shown as text, never as a greyed field; the sample showed this for one row only, and the two measured definitions are shown so.
- A fold, when opened, shows its settings rows in place of the one-line summary, its chevron pointing down.
- A table that can be sorted shows a small chevron after the name of the column it is sorted by.
- If Cambria is missing on a computer, its four places fall back to Georgia and then to Inter.
- The small icon font and its builder script leave the repository (git history keeps them); every glyph is painted.

### What this part fixes

- W080: a figure stored in a record is always drawn on a light page, and a change of theme no longer rewrites the figure settings (4.9; U23, B18).
- W101: an apply that changes nothing does nothing; the "Auto (time of day)" mode and its timer go as part 3 defines (4.1, 4.8).
- W102: the text size follows Windows; the pinned 14 px and the scale that could never leave 1.0 go, by the mechanism B20's test selects (4.8).
- W106: states have colours and shapes, and an action that loses work has its own button (4.2, 4.5, 4.6).
- W110: a value that cannot be edited is never drawn as a field (4.5), applied by part 8 to the two measured definitions.
