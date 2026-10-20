"""The window's places and its keys, as tables with no widget.

The parts of the rail in their order, the six steps of parameter inference, the pages of a part that has
several, the keys that open each, and the words a sentence names a place by ("step 3, Training", "Effective
temperature, Sweep study"). The rail, Home, the window's shortcuts, Help's table of keys and the refusal
tables all read them here, so a place is named in one table and nowhere else. Nothing here imports the
toolkit or the engine: the refusal tables are read without a window.
"""
from typing import NamedTuple


class Place(NamedTuple):
    """One place a person can open: its key, its name on screen, the keys that open it, and the one line
    Home's table of parts gives it."""
    key: str
    name: str
    keys: tuple[str, ...] = ()
    line: str = ""


# the parts of the rail, in its order; Home has no line, since Home's table of parts lists the others
PARTS: tuple[Place, ...] = (
    Place("home", "Home", ("Ctrl+1",)),
    Place("inference", "Parameter inference", ("Ctrl+2",),
          "Infer a cell's model parameters from recordings, in six steps."),
    Place("temperature", "Effective temperature", ("Ctrl+3",),
          "Measure it on one cell, or in a sweep study over two parameters; compare saved runs."),
    Place("live", "Live simulation", ("Ctrl+4",), "Watch a cell's bundle move as it is simulated; save a video."),
    Place("reduction", "Reduction map", ("Ctrl+5",),
          "The analytical map from the Nadrowski model to the Hopf normal form."),
    Place("records", "Records", ("Ctrl+6",), "Everything PRISM has written, and what each record was made from."),
    Place("models", "Your own models", ("Ctrl+7",), "Define a model of your own."),
    Place("settings", "Settings", ("Ctrl+8",), "The appearance."),
    Place("help", "Help", ("Ctrl+9", "F1"), "What each part's settings mean, and the keys."),
)

# the six steps of parameter inference, in their order; each opens from anywhere with its key
STEPS: tuple[Place, ...] = (
    Place("model", "Model and options", ("Alt+1",)),
    Place("prior", "Prior", ("Alt+2",)),
    Place("training", "Training", ("Alt+3",)),
    Place("checks", "Checks", ("Alt+4",)),
    Place("infer", "Inference", ("Alt+5",)),
    Place("narrowing", "Narrowing round", ("Alt+6",)),
)

# the pages of a part that has several, chosen on the part's own page and not in the rail
PAGES: dict[str, tuple[Place, ...]] = {
    "temperature": (Place("one_cell", "One cell"), Place("sweep", "Sweep study")),
}

# the rows Help prints: a key, or a run of keys, and what it does
KEYS: tuple[tuple[str, str], ...] = (
    ("F6, Shift+F6", "Moves the keyboard between the rail and the page."),
    ("Ctrl+1 to Ctrl+9", "Opens the nine parts in rail order."),
    ("Alt+1 to Alt+6", "Opens steps 1 to 6 from anywhere."),
    ("Ctrl+R", "Goes to the run, whenever the foot is a button."),
    ("F1", "Opens Help."),
    ("Up, Down, Home, End", "In the rail, moves through the parts and, when shown, the steps, in the order seen."),
    ("Enter, Space", "In the rail, opens the row."),
    ("Escape", "In a question, the safe answer; closes an open list; in the rail, returns the keyboard to the "
               "page; otherwise nothing. It never moves between pages and never stops a run."),
)

# one row per shortcut of the window: the key and what it does, "part <key>", "step <key>", "rail or page" or
# "run". The keys that act inside the rail (the arrows, Home, End, Enter, Space, Escape) are the rail's own.
BINDINGS: tuple[tuple[str, str], ...] = (
    tuple((sequence, f"part {place.key}") for place in PARTS for sequence in place.keys)
    + tuple((sequence, f"step {place.key}") for place in STEPS for sequence in place.keys)
    + (("F6", "rail or page"), ("Shift+F6", "rail or page"), ("Ctrl+R", "run")))


def part(name: str) -> Place:
    """The part whose key is ``name``; KeyError for a name that is no part's."""
    for place in PARTS:
        if place.key == name:
            return place
    raise KeyError(name)


def step(name: str) -> Place:
    """The step whose key is ``name``; KeyError for a name that is no step's."""
    for place in STEPS:
        if place.key == name:
            return place
    raise KeyError(name)


def step_number(name: str) -> int:
    """The number of the step whose key is ``name``, 1 to 6."""
    return STEPS.index(step(name)) + 1


def place_words(place: str) -> str:
    """A place as a sentence names it: a part by its name ("Records"), a step by its number and name ("step 3,
    Training"), a page of a part as "<part>/<page>" by both names ("Effective temperature, Sweep study"); a
    step named as a page of Parameter inference reads as the step alone. KeyError for a place that is none of
    these."""
    if "/" in place:
        whole, page = place.split("/", 1)
        if whole == "inference" and any(named.key == page for named in STEPS):
            return place_words(page)
        for named in PAGES.get(whole, ()):
            if named.key == page:
                return f"{part(whole).name}, {named.name}"
        raise KeyError(place)
    if any(named.key == place for named in STEPS):
        return f"step {step_number(place)}, {step(place).name}"
    return part(place).name


def places_words(places: tuple[str, ...]) -> str:
    """Several places in one sentence, as it names them after its "in", each as ``place_words`` writes it,
    joined by ", or ": "step 3, Training, or step 6, Narrowing round". A step after another place begins with
    its own number; a part, or a page of a part, after another place takes its own "in", so that it does not
    read as one more name of the place before it: "step 5, Inference, or in Live simulation"."""
    said = [place_words(place) for place in places]
    return ", or ".join(words if number == 0 or words.startswith("step ") else f"in {words}"
                        for number, words in enumerate(said))
