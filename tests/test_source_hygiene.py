"""No shippable file points at a working document.

The program, the tests, the launchers, the README, the requirements file and the reader pages under
docs/guide must explain themselves in words. A comment that says why by naming a design document, a
decision id, a numbered rule or a review finding stops making sense the day that document is
archived, and the reader it was written for never had it. This suite reads the prose of every such
file -- comments, docstrings and string literals in Python (never code identifiers, where short math
names would be false positives), every line of the prose files and reader pages, and the names of
test functions and classes -- and fails on anything shaped like such a reference.

Every pattern below is assembled from adjacent literals, each of which is clean on its own, so this
module stays under its own scan.
"""
import ast
import functools
import io
import re
import tokenize
from collections import namedtuple
from pathlib import Path

from core.SBI.statistics import FEATURE_LABELS
from tests._fixtures import CODE_FILES, CODE_ROOTS

_REPO = Path(__file__).resolve().parents[1]
_SELF = Path(__file__).resolve()

Hit = namedtuple("Hit", "path line family match")

_SECTION_SIGN = chr(0xA7)  # the section sign, never written literally here
_RULING, _WALKTHROUGH = "rul" "ing", "walk" "through"  # two family names that are labels themselves

PROSE_PATTERNS: dict[str, re.Pattern] = {name: re.compile(source) for name, source in {
    "doc": (r"(?i)PRISM_" r"HAND" r"OFF|\bhand" r"off\b|\bSTATE\.md\b|\bCLAUDE\.md\b|super" r"powers"
            r"|\bsdd\b|\bprogress\.md\b|\bled" r"ger\b"),
    "piece": (r"(?i)\bpieces?[ -]?\d+\b|\bwhole-" r"piece\b"
              r"|\b(?:this|that|each|same|earlier|later|next|previous) " r"piece\b"),
    "section": _SECTION_SIGN + r"|\b(?:[Ss]ec|SEC)(?:tion|TION|\.)? ?\d+(?:\.\d+)+\b",
    "spec": (r"\bspec(?:'s)? (?:" + _SECTION_SIGN + r"|[Ss]ec|section|\d)|\b(?:design|piece-\d+) " r"spec\b"
             r"|\bthe spec" r"'s\b|\bspec " r"NOTES\b|\bplan(?:ning)? " r"rul" r"ing"),
    _RULING: r"\brul" r"ings?\b|\bowner " r"ruled\b",
    "review": (r"\breview" r"'s\b|\b[Ff]ix " r"round\b|\bReview " r"Focus\b|\bfinding \d|\breviewers?(?:'s)? " r"probe"
               r"|\breviewers? " r"probed|\bprobed by (?:\w+ )?" r"reviewers\b|\bthe review " r"of\b"
               r"|\bper the " r"review\b|\b[Cc]hecklist(?: item)? \d|\bBack" r"log\b"
               r"|\b(?:CRITICAL|IMPORTANT|Critical|Important) \d\b|\bdefect [A-Z]-?\d+\b"),
    "task": r"\bTasks? \d+\b|\btasks? T\d+",
    "trap": r"\b(?:trap|TRAP)s? [A-Z]+\d+[a-z]?\b|\bCHI\d+\b",
    "guardrail": r"(?i)\bguardrails?[ -]\d+\b",
    "series": r"(?<![\w-])[CS]-\d+\b",
    "appendix": r"\bAppendix " r"[A-Z]\b",
    _WALKTHROUGH: r"\bwalk" r"through\b|\brows? [A-Z]\d+\b",
    "programme": r"\bhardening " r"programme\b|\bone-flow " r"design\b|\bclean[- ]break\b",
    "id": r"(?<![\w./])(?<![a-z]-)(?:R-F\d+|FE\d+|[A-Z]\d{1,3}[a-z]?)(?!\w)(?!-[a-z])",
}.items()}

TEST_NAME_PATTERN = re.compile(
    r"(?i)(?:^|_)(?:piece\d+|[a-z]\d{1,3}[a-z]?|rf\d+|fe\d+|chi\d+|task\d+|guardrail\d*|round\d+"
    r"|whole_piece|fix_round|walk" r"through|hand" r"off|appendix|rul" r"ing)(?:_|$)")

# The summary features' short ids, allowed only in the two files that define and compute them.
FEATURE_IDS = frozenset(label.split("_", 1)[0] for label in FEATURE_LABELS)

ALLOW_TOKENS = frozenset({"F0", "L2", "F1", "F6"})  # a force amplitude, a norm, two keys
ALLOW_PER_FILE: dict[str, frozenset[str]] = {
    "core/SBI/statistics.py": FEATURE_IDS,
    "core/SBI/summaries.py": FEATURE_IDS,
}
ALLOW_SHAPES: tuple[re.Pattern, ...] = (
    re.compile(r"U\+[0-9A-F]{4}"),                  # Unicode code points
    re.compile(r"\bM[1-4](?: / M[1-4]){3}\b"),      # the Apple chips the README names
)
ALLOW_TEST_NAME_TOKENS = frozenset({"f0", "d0"})
ALLOW_TEST_NAMES: frozenset[str] = frozenset({
    "test_the_B7_pair_shares_one_flag",                 # names a summary feature by its id
    "test_v1_params_migrate_to_placeholder_boxes",      # model-file schema versions
    "test_v2_params_migrate_to_a_linear_box",
})

PROSE_FILES = ("README.md", "requirements.txt", "pytest.ini", "run.bat", "run.sh")
READER_DOCS_DIR = "docs/guide"

_DIRECTIVE = re.compile(r"noqa(?::[ \t]*[A-Z]+[0-9]+(?:[ \t]*,[ \t]*[A-Z]+[0-9]+)*)?"
                        r"|type:[ \t]*ignore(?:\[[^\]]*\])?")
_PROSE_TOKENS = {tokenize.COMMENT, tokenize.STRING, tokenize.FSTRING_MIDDLE}


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(_REPO).as_posix()
    except ValueError:
        return path.as_posix()


def _scan_text(text: str, rel: str, line: int = 1, used: set | None = None) -> list[Hit]:
    """Every match in one line of prose. Allowed shapes are blanked first; allowed tokens and a file's
    own tokens are spared from the id family. ``used`` collects which allowlist entries spared
    something."""
    for i, shape in enumerate(ALLOW_SHAPES):
        text, n = shape.subn(lambda m: " " * len(m.group()), text)
        if n and used is not None:
            used.add(("shape", i))
    hits = []
    for family, pattern in PROSE_PATTERNS.items():
        for m in pattern.finditer(text):
            token = m.group()
            if family == "id" and token in ALLOW_TOKENS:
                if used is not None:
                    used.add(("token", token))
                continue
            if family == "id" and token in ALLOW_PER_FILE.get(rel, ()):
                if used is not None:
                    used.add(("file", rel))
                continue
            hits.append(Hit(rel, line, family, token))
    return hits


def scan_python(path: Path, *, used: set | None = None) -> list[Hit]:
    """The hits in a Python file's comments, docstrings and string literals (never its code). A lint
    directive (``noqa: CODE`` or ``type: ignore[...]``) is stripped; an explanation after it is read."""
    rel = _rel(path)
    hits = []
    source = path.read_text(encoding="utf-8")
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type not in _PROSE_TOKENS:
            continue
        text = _DIRECTIVE.sub("", tok.string) if tok.type == tokenize.COMMENT else tok.string
        for i, segment in enumerate(text.split("\n")):
            hits += _scan_text(segment, rel, tok.start[0] + i, used)
    return hits


def scan_prose(path: Path, *, used: set | None = None) -> list[Hit]:
    """The hits in every line of a prose file."""
    rel = _rel(path)
    hits = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        hits += _scan_text(line, rel, n, used)
    return hits


def _scan_file(path: Path, *, used: set | None = None) -> list[Hit]:
    return scan_python(path, used=used) if path.suffix == ".py" else scan_prose(path, used=used)


def scanned_files() -> list[Path]:
    """Every file the rule covers: the Python under CODE_ROOTS, CODE_FILES and tests/ (this module
    included), the prose files, and every page under docs/guide once it exists."""
    python = {p for root in CODE_ROOTS for p in (_REPO / root).rglob("*.py")}
    python |= {_REPO / name for name in CODE_FILES}
    python |= set((_REPO / "tests").rglob("*.py"))
    files = sorted(python) + [_REPO / name for name in PROSE_FILES]
    guide = _REPO / READER_DOCS_DIR
    if guide.is_dir():
        files += sorted(guide.glob("*.md"))
    return files


@functools.lru_cache(maxsize=None)
def _scan_everything() -> tuple[dict[str, list[Hit]], frozenset]:
    """Hits per file over the whole rule, and the allowlist entries that spared something outside
    this module (whose own literals must not keep an entry alive)."""
    by_file, used = {}, set()
    for path in scanned_files():
        by_file[_rel(path)] = _scan_file(path, used=None if path.resolve() == _SELF else used)
    return by_file, frozenset(used)


def _test_name_labels(name: str) -> list[str]:
    labels, pos = [], 0
    while (m := TEST_NAME_PATTERN.search(name, pos)):
        token = m.group().strip("_").lower()
        if token not in ALLOW_TEST_NAME_TOKENS:
            labels.append(token)
        pos = m.start() + 1
    return labels


@functools.lru_cache(maxsize=None)
def _test_definitions() -> tuple[tuple[str, int, str], ...]:
    out = []
    for path in scanned_files():
        if path.suffix != ".py":
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                    and node.name.lower().startswith("test")):
                out.append((_rel(path), node.lineno, node.name))
    return tuple(out)


def _sparing_lists(family: str) -> str:
    """The allowlists that can spare a hit of ``family``: the token lists are consulted for the id
    family alone, while a shape is blanked before every family is matched."""
    return "ALLOW_TOKENS, ALLOW_PER_FILE or ALLOW_SHAPES" if family == "id" else "ALLOW_SHAPES"


def _failure_message(hits: list[Hit]) -> str:
    listing = "\n  ".join(f"{h.path}:{h.line} {h.family} {h.match!r} -> {_sparing_lists(h.family)}"
                          for h in hits)
    return (f"{len(hits)} reference(s) to a working document or one of its labels:\n  {listing}\n"
            "State the reason in words instead of pointing at a document. If a match is a legitimate "
            "token shaped like a label (a physics name, a new feature id, a published paper's section "
            "number), add the narrowest entry that spares it to tests/test_source_hygiene.py, from the "
            "lists named after its line -- ALLOW_TOKENS (one token, everywhere) and ALLOW_PER_FILE (one "
            "token, one file) spare the id family only; ALLOW_SHAPES (a pattern) spares any family -- in "
            "the commit that introduces it.")


def _test_name_failure_message(offenders: list[tuple[str, int, str]]) -> str:
    """The failure for test names carrying a label, each offender given as (path, line, name)."""
    listing = "\n  ".join(f"{path}:{line} {name} {_test_name_labels(name)}" for path, line, name in offenders)
    return ("test names carrying a label -- rename each so the name states the behaviour:\n  " + listing + "\n"
            "A legitimate name goes in ALLOW_TEST_NAMES (one name) in tests/test_source_hygiene.py; a "
            "physics token that any test name may carry goes in ALLOW_TEST_NAME_TOKENS there.")


def test_no_scanned_file_cites_a_working_document():
    """The rule itself: every file it covers must be clean, with no exemption. The walk is checked
    first: it must reach the top-level code files, the five prose files and this module, and at least
    150 Python files, so a walk that silently stopped cannot pass."""
    files = scanned_files()
    missing = [name for name in (*CODE_FILES, *PROSE_FILES) if not (_REPO / name).is_file()]
    assert not missing, f"a file the rule names does not exist: {missing}"
    assert _SELF in {p.resolve() for p in files}, "this module must stay under its own scan"
    walked = sum(p.suffix == ".py" for p in files)
    assert walked >= 150, f"the scan walked only {walked} Python files"
    by_file, _used = _scan_everything()
    hits = [h for rel, found in sorted(by_file.items()) for h in found]
    assert not hits, _failure_message(hits)


_GAP = chr(0xB7)  # written inside each label below and removed at run time, so no literal matches


def _u(text: str) -> str:
    return text.replace(_GAP, "")


def test_the_matchers_catch_each_family_and_spare_the_allowed_tokens(tmp_path):
    """The matchers are checked on samples first: with the tree clean, the tree walk alone would go on
    passing if a pattern stopped matching anything."""
    catch = {
        "doc": ["see PRISM_HAND·OFF.md", "noted in STATE·.md", "the super·powers workspace"],
        "piece": ["added in pie·ce 3", "this pie·ce keeps it"],
        "section": ["as " + _SECTION_SIGN + "3.4 says", "Sec·tion 11.6 covers it"],
        "spec": ["spec " + _SECTION_SIGN + "4.1 fixes it", "the design ·spec says"],
        _RULING: ["the controller rul·ing", "the owner rul·ed so"],
        "review": ["the whole-code review·'s finding", "found in a fix ·round", "defect D·3 again"],
        "task": ["done in Ta·sk 7", "tasks T·13 to T·18"],
        "trap": ["see trap X·5", "see CHI·10"],
        "guardrail": ["guard·rail 2 holds", "GUARD·RAIL 5 again"],
        "series": ["the C·-11 cache", "an S·-1 note"],
        "appendix": ["Appen·dix A tells it"],
        _WALKTHROUGH: ["the walk·through checklist", "row D·15 checks it"],
        "programme": ["the hardening program·me", "since the clean ·break"],
        "id": ["refused (D·7)", "the rows R-F·1 and FE·3", "ids T·13-T·18"],
    }
    for family, samples in catch.items():
        for sample in samples:
            found = {h.family for h in _scan_text(_u(sample), "core/example.py")}
            assert family in found, (family, _u(sample), found)
    spare = {
        "doc": ["the state of the bundle", "a ledgerless cache"],
        "piece": ["the piece removes", "a two-piece model"],
        "section": ["Fig. 3", "the section headings"],
        "spec": ["carried on the spec", "the batch plan"],
        _RULING: ["an owner guard", "the ruler's edge"],
        "review": ["a reviewer reads it", "round 2 of a narrowing"],
        "task": ["the batch task", "tasks run in turn"],
        "trap": ["a trap for the unwary", "trapped in a well"],
        "guardrail": ["a guardrail against drift", "the guardrails hold"],
        "series": ["a C-contiguous array", "an S-shaped curve"],
        "appendix": ["an appendix to the paper", "Appendix of results"],
        _WALKTHROUGH: ["rows A to D", "walk through the steps"],
        "programme": ["a clean breakpoint", "the training programme"],
        "id": ["the F0 amplitude and the L2 norm", "A1_mean and D3_bimodality", "UTF-8, SHA-256, HDF5",
               "V's columns, 0..K-1, B=64/T=20000", "U+E000..U+E004", "M1 / M2 / M3 / M4"],
    }
    assert set(spare) == set(catch) == set(PROSE_PATTERNS)
    for family, samples in spare.items():
        for sample in samples:
            assert _scan_text(sample, "core/example.py") == [], (family, sample, _scan_text(sample, "core/example.py"))
    feature = _u("the A·3 feature")
    assert _scan_text(feature, "core/SBI/statistics.py") == []
    assert [h.family for h in _scan_text(feature, "core/SBI/chi.py")] == ["id"]
    # a lint directive is stripped; an explanation after it is still read
    lint = tmp_path / "lint.py"
    lint.write_text(_u("import os  # noqa: E·402, F·401 -- kept for Ta·sk 7\nimport re  # noqa: F·401\n"),
                    encoding="utf-8")
    assert [(h.line, h.family) for h in scan_python(lint)] == [(1, "task")]
    # test names: every label shape is caught, the physics tokens and plain words are spared, and an
    # allowed token does not hide a label right beside it
    for name in ("test_the_d7_and_d8_dialogs_default_to_cancel", "test_the_contract_c11_depends_on",
                 "test_piece5_records", "test_task12_gate", "test_guardrail_holds", "test_after_the_fix_round",
                 "test_round2_findings", "test_rf1_fix", "test_fe3_fix", "test_chi10_channel",
                 "test_the_handoff_is_gone", "test_the_walkthrough_rows", "test_the_appendix_incident",
                 "test_the_ruling_holds", "test_the_whole_piece_gate"):
        assert _test_name_labels(name), name
    for name in ("test_the_f0_grid_is_log_spaced", "test_the_d0_constant", "test_the_sha256_digest",
                 "test_log_t_obs_widens_the_row"):
        assert _test_name_labels(name) == [], name
    assert _test_name_labels("test_the_f0_d7_case") == ["d7"]


def test_every_reader_page_is_scanned():
    pages = sorted((_REPO / READER_DOCS_DIR).glob("*.md"))
    assert len(pages) >= 10, f"expected the ten reader pages under {READER_DOCS_DIR}, found {len(pages)}"
    scanned = {p.resolve() for p in scanned_files()}
    missing = [p.name for p in pages if p.resolve() not in scanned]
    assert not missing, f"reader pages the reference scan does not read: {missing}"


def test_every_allowlist_entry_still_matches_something():
    """An entry that spares nothing is stale, and would hide the next real reference of its shape.
    Only uses outside this module count: its own literals would otherwise keep every entry alive.
    ALLOW_TEST_NAME_TOKENS is part of the test-name matcher rather than an entry: it keeps a physics
    name such as the drive amplitude's usable in a test name, whether or not one uses it today."""
    _by_file, used = _scan_everything()
    stale = [f"ALLOW_TOKENS {t!r}" for t in sorted(ALLOW_TOKENS) if ("token", t) not in used]
    stale += [f"ALLOW_PER_FILE {rel!r}" for rel in sorted(ALLOW_PER_FILE) if ("file", rel) not in used]
    stale += [f"ALLOW_SHAPES {s.pattern!r}" for i, s in enumerate(ALLOW_SHAPES) if ("shape", i) not in used]
    names = {name for _path, _line, name in _test_definitions()}
    stale += [f"ALLOW_TEST_NAMES {n!r}" for n in sorted(ALLOW_TEST_NAMES) if n not in names]
    assert not stale, "allowlist entries that spare nothing -- delete them:\n  " + "\n  ".join(stale)


def test_no_test_name_carries_a_process_label():
    """A test's name is what a failing run prints, so it states the behaviour, never a label."""
    offenders = [(path, line, name) for path, line, name in _test_definitions()
                 if name not in ALLOW_TEST_NAMES and _test_name_labels(name)]
    assert not offenders, _test_name_failure_message(offenders)


def test_a_failure_names_the_file_line_family_and_the_allowlist(tmp_path):
    """A legitimate new token shaped like a label -- a future feature id, a published paper's section
    number -- fails with everything needed to act on it: where, which family, what matched, and the
    allowlists that can spare it. Each hit names only the lists that can: the token lists spare the id
    family alone, and a shape spares any family. A test name that carries a label fails the same way,
    naming the name, the label and the list that can spare it."""
    module = tmp_path / "new_feature.py"
    new_id = "Q" + "7"
    module.write_text('"""A new module."""\n\nDRIFT = 1  # the ' + new_id + " channel\n"
                      "GAIN = 2  # as in Sec" + "tion 4.2 of Gardiner\n", encoding="utf-8")
    hits = scan_python(module)
    where = module.as_posix()
    assert hits == [Hit(where, 3, "id", new_id), Hit(where, 4, "section", "Sec" + "tion 4.2")]
    message = _failure_message(hits)
    assert f"{where}:3 id '{new_id}'" in message
    assert f"{where}:4 section 'Sec" + "tion 4.2'" in message
    for entry in ("ALLOW_TOKENS", "ALLOW_PER_FILE", "ALLOW_SHAPES", "tests/test_source_hygiene.py"):
        assert entry in message, entry
    lines = message.splitlines()
    id_line = next(ln for ln in lines if f"{where}:3 id" in ln)
    section_line = next(ln for ln in lines if f"{where}:4 section" in ln)
    assert id_line.endswith("-> ALLOW_TOKENS, ALLOW_PER_FILE or ALLOW_SHAPES"), id_line
    assert section_line.endswith("-> ALLOW_SHAPES") and "ALLOW_TOKENS" not in section_line, section_line

    name = "test_the_" + "H" + "1_channel_is_finite"
    named = _test_name_failure_message([("tests/x.py", 12, name)])
    for entry in ("tests/x.py:12", name, "'h" + "1'", "ALLOW_TEST_NAMES", "ALLOW_TEST_NAME_TOKENS",
                  "tests/test_source_hygiene.py"):
        assert entry in named, (entry, named)


# ── what the program no longer has ────────────────────────────────────────────────────────────────
# The effective temperature (its measurement, its sweep study and its comparisons), the reduction map and the live
# simulation were taken out of PRISM, which infers a cell's parameters and does nothing else. No file the rule
# covers imports one of their modules or names one of their modules or files, by its path or its dotted name: a
# mention would send a reader to a file that is not there. Each name is assembled from adjacent literals, so this
# module stays under its own scan. A plain word ("the effective temperature") is not what is hunted.
REMOVED_MODULES = ("core." "FDT", "core." "Reduction", "core.Helpers." "fdt", "core.tool." "fdt",
                   *(f"core.gui.panels.{name}" for name in ("fdt" "_panel", "crossval" "_panel", "reduction" "_panel",
                                                            "simulate" "_panel", "simulate" "_runner",
                                                            "simulate" "_export", "record" "_view")),
                   "core.gui.screens." "temperature" "_screen", "core.gui.widgets." "compare" "_list",
                   "core.gui.widgets." "live_hair" "_bundle")
REMOVED_NAMES = ("core/" "FDT", "core." "FDT", "core/" "Reduction", "core." "Reduction", "Helpers/" "fdt",
                 "Helpers." "fdt", "tool/" "fdt", "tool." "fdt", "fdt" "_pipeline", "cross" "_validation",
                 "fdt" "_panel", "crossval" "_panel", "reduction" "_panel", "simulate" "_panel", "simulate" "_runner",
                 "simulate" "_export", "record" "_view", "temperature" "_screen", "compare" "_list",
                 "live_hair" "_bundle", "test_fdt" "_user", "test_fdt" "_compare", "test" "_simulate")
_REMOVED_NAME = re.compile("|".join(rf"(?<!\w){re.escape(name)}(?!\w)" for name in REMOVED_NAMES))
# The lines that name a removed module and stay as they are, each by its file and its exact words, with why.
REMOVED_NAMES_KEPT = {
    # the configuration's comments stay byte for byte, as every setting of the configuration does
    ("core/config.py", "    Originally FDT-only (" "core/" "FDT/campaigns.py); lifted here because the SBI path "
                       "needs the same"): "a comment of core/config.py, every line of which is kept as it was",
    ("core/config.py", "# data-driven grid; see " "core/" "FDT/spectral.gen_freqs_log / find_spectral_peak). Each "
                       "chi(omega_k)"): "a comment of core/config.py, every line of which is kept as it was",
}


def _removed_scan_files() -> list[Path]:
    """Every file the reference rule covers but the reader pages and the README, whose words follow in a change of
    their own."""
    return [path for path in scanned_files()
            if _rel(path) != "README.md" and not _rel(path).startswith(READER_DOCS_DIR + "/")]


def _imports_of(path: Path, rel: "str | None" = None) -> list[tuple[int, str]]:
    """``(line, module)`` for every module a Python file imports, relative imports resolved from the file's own
    package (``rel``, its path from the repository, read off the file when not handed), and for
    ``from <package> import <name>`` the module ``<package>.<name>`` as well."""
    rel = _rel(path) if rel is None else rel
    package = rel[:-3].replace("/", ".").split(".")
    if package[-1] == "__init__":
        package = package[:-1]
    else:
        package = package[:-1]
    found = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found += [(node.lineno, alias.name) for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = package[:len(package) - node.level + 1] if node.level else []
            module = ".".join([*base, *([node.module] if node.module else [])])
            found.append((node.lineno, module))
            found += [(node.lineno, f"{module}.{alias.name}") for alias in node.names]
    return found


def test_no_file_imports_or_names_a_module_the_program_no_longer_has():
    """The modules of the effective temperature, the reduction map and the live simulation are gone from disk, no
    file imports one of them, and no line of any file the rule covers names one of their modules or files by its
    path or its dotted name, but the lines kept as they are, each listed with its reason."""
    for module in REMOVED_MODULES:
        where = _REPO / module.replace(".", "/")
        assert not where.exists() and not where.with_suffix(".py").exists(), f"{module} is still on disk"
    imported, named, kept = [], [], set()
    for path in _removed_scan_files():
        rel = _rel(path)
        if path.suffix == ".py":
            imported += [f"{rel}:{line} imports {module}" for line, module in _imports_of(path)
                         if any(module == gone or module.startswith(gone + ".") for gone in REMOVED_MODULES)]
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not _REMOVED_NAME.search(line):
                continue
            if (rel, line) in REMOVED_NAMES_KEPT:
                kept.add((rel, line))
                continue
            named.append(f"{rel}:{number} {_REMOVED_NAME.search(line).group()!r}")
    assert imported == [], "a file imports a module the program no longer has:\n  " + "\n  ".join(imported)
    assert named == [], ("a line names a module or file the program no longer has; say it in words, or list a line "
                         "kept as it is in REMOVED_NAMES_KEPT with its reason:\n  " + "\n  ".join(named))
    assert kept == set(REMOVED_NAMES_KEPT), f"entries that spare nothing -- delete them: {set(REMOVED_NAMES_KEPT) - kept}"


def test_the_scan_of_removed_modules_catches_an_import_and_a_name_and_spares_a_plain_word(tmp_path):
    """The scan's two halves are checked on a file of its own first, so that it cannot pass by reading nothing: an
    import of a removed module, absolute, relative or through its package, and a name of one in a comment or a
    string, are each caught; a plain word, a longer name that merely contains one, and a kept module are not."""
    module = tmp_path / "core" / "gui" / "panels" / "inference" / "probe.py"
    module.parent.mkdir(parents=True)
    module.write_text("\n".join([
        "from core." "FDT import spectral",                    # 1: absolute
        "from .. import " "fdt" "_panel",                       # 2: through its package, relative
        "import core." "Reduction" ".sweep",                    # 3: a submodule
        "from .base import _StagePanel",                        # 4: a kept module
        "# the effective temperature's pages were taken out",   # 5: a plain word
        "# see " "simulate" "_panel._save_video",               # 6: a name in a comment
        "owned_by_the_" "simulate" "_panel = 1",                # 7: a longer name that contains one
    ]), encoding="utf-8")
    imports = _imports_of(module, rel="core/gui/panels/inference/probe.py")
    caught = sorted({line for line, name in imports
                     if any(name == gone or name.startswith(gone + ".") for gone in REMOVED_MODULES)})
    assert caught == [1, 2, 3], imports
    named = [n for n, line in enumerate(module.read_text(encoding="utf-8").splitlines(), 1)
             if _REMOVED_NAME.search(line)]
    assert named == [1, 2, 3, 6], named
