# Display walkthrough checklist

Features the headless suites cover by wiring only, never by a human looking at the screen. Each
row is checked off on a real display in piece 4 of the hardening programme, with the date and
what was seen. Seeded 2026-09-10 from `PRISM_HANDOFF.md` §6 (the ONGOING row and row 3).

| # | surface | do | expect | date | result |
|---|---|---|---|---|---|
| 1 | App icon | launch `run.bat` | taskbar and window show the PRISM icon, not the interpreter's | | |
| 2 | Navigation | click every Home tile, the back arrow, the gear | each screen mounts; back returns; the gear reaches Settings and the model builder | | |
| 3 | Config → Prior | apply a model, pick a bounds file, build a prior from scratch | corner figure appears; every panel's controls lock during the run; the picker lists the prior after | | |
| 4 | Prior save | name and save | it appears in the picker; a second save under the same name is refused | | |
| 5 | Training-budget group | vary Batches and Max rows | the three derived lines wrap without overflowing the controls column | | |
| 6 | Posterior train and save | 2 batches from scratch, then save | loss figure; the checkpoint line names the directory; the save works | | |
| 7 | Cancel during live NN training | press Cancel mid-training | "Cancelling…", a clean stop within about a minute, controls unlock, no error dialog | | |
| 8 | Validate | run calibration at small sizes | SBC and TARP figures; no dialog | | |
| 9 | Infer, simulated | pick a cell, run | corner, PPC, overlays, eye test | | |
| 10 | Infer, experimental (passive, driven, chi) | point at recordings | it runs, or a dialog names the missing or invalid file | | |
| 11 | TSNPE round | pick the observation, run at 2 batches | a non-amortized posterior; loading it logs the NON-AMORTIZED line | | |
| 12 | Progress pane | during any run | one overall bar with a caption; a solver rate with no bar; no re-layout jitter | | |
| 13 | Simulate | start streaming, watch trace and heatmap, cancel | live update; cancel stops cleanly | | |
| 14 | Save video | mp4 and gif | the files play; a partial file is removed on cancel | | |
| 15 | FDT full run | small settings | figures land; a run where every row failed is reported, not "complete" | | |
| 16 | CrossVal | S-sweep and T-sweep at small sizes | figures land | | |
| 17 | Model builder round trip | declare a two-variable model, validate, save, use it in Simulate, delete it | the parameter row's two lines are readable; the sticky action bar is visible; delete confirms | | |
| 18 | Settings | theme Light / Dark / Auto, OS accent, force Inter | figures rendered after the flip adopt the theme; the help text scrolls to its end | | |
| 19 | Layout | resize to 1366×768; drag every splitter | nothing clipped; splitter positions survive a restart | | |
| 20 | Close while running | close the window with a run live | the three-button dialog; "Cancel task & quit" stops cleanly | | |
