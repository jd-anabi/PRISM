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

ALLOW_TOKENS = frozenset({"F0", "D0", "L2"})  # a force amplitude, a diffusion constant, a norm
ALLOW_PER_FILE: dict[str, frozenset[str]] = {
    "core/SBI/statistics.py": FEATURE_IDS,
    "core/SBI/summaries.py": FEATURE_IDS,
    "core/FDT/spectral.py": frozenset({"S" "2"}),  # scipy's name for the window's sum of squares
}
ALLOW_SHAPES: tuple[re.Pattern, ...] = (
    re.compile(r"\bPhase [A-Z]\d\b"),               # the reduction map's phases
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
        "id": ["the F0 amplitude, the L2 norm and D0", "A1_mean and D3_bimodality", "UTF-8, SHA-256, HDF5",
               "V's columns, 0..K-1, B=64/T=20000", "Phase B1", "U+E000..U+E004", "M1 / M2 / M3 / M4"],
    }
    assert set(spare) == set(catch) == set(PROSE_PATTERNS)
    for family, samples in spare.items():
        for sample in samples:
            assert _scan_text(sample, "core/example.py") == [], (family, sample, _scan_text(sample, "core/example.py"))
    feature = _u("the A·3 feature")
    assert _scan_text(feature, "core/SBI/statistics.py") == []
    assert [h.family for h in _scan_text(feature, "core/SBI/chi.py")] == ["id"]
    assert _scan_text(_u("scipy's S·2"), "core/FDT/spectral.py") == []
    assert [h.family for h in _scan_text(_u("scipy's S·2"), "core/FDT/plots.py")] == ["id"]
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
