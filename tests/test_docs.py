"""The reader pages under docs/guide, and the root README: every relative link resolves, every page
carries the commit it was checked against, and the command-line page matches the tool's parser.

Nothing here imports torch: the parser and its default table are torch-free by design.
"""
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GUIDE = REPO / "docs" / "guide"
PAGES = ("README.md", "getting-started.md", "window.md", "command-line.md", "recordings.md",
         "science.md", "architecture.md", "rules-and-traps.md", "testing.md", "retrain.md")

_FENCE = re.compile(r"^\s*(```|~~~)")
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_CODE_SPAN = re.compile(r"`[^`\n]*`")
_LINK = re.compile(r"\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
_REF_DEF = re.compile(r"^\s{0,3}\[[^\]]+\]:\s*\S")
_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
_DRIVE = re.compile(r"^[A-Za-z]:")     # one letter and a colon: a Windows drive, not a web scheme
_STAMP = re.compile(r"^Checked against commit [0-9a-f]{7,40}\.$")


def _prose_lines(text):
    """(line number, line) for every line outside a fenced code block."""
    in_fence = False
    for n, line in enumerate(text.splitlines(), 1):
        if _FENCE.match(line):
            in_fence = not in_fence
            continue
        if not in_fence:
            yield n, line


def _slug(title):
    """The anchor GitHub gives a heading: code and link markup reduced to their text, lower case,
    everything but letters, digits, spaces, hyphens and underscores dropped, spaces to hyphens."""
    text = _CODE_SPAN.sub(lambda m: m.group(0)[1:-1], title)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"[^\w\- ]", "", text.strip().lower())
    return text.replace(" ", "-")


def _anchors(text):
    """Every heading anchor of a page; a repeated heading is numbered -1, -2, ... as GitHub does."""
    seen, out = {}, set()
    for _n, line in _prose_lines(text):
        m = _HEADING.match(line)
        if m:
            base = _slug(m.group(2))
            k = seen.get(base, 0)
            out.add(base if k == 0 else f"{base}-{k}")
            seen[base] = k + 1
    return out


def _rel(path, root):
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def _exact_path(start, path_part):
    """``(path, None)`` for ``path_part`` walked from ``start`` one component at a time, each name
    required exactly as its folder lists it; ``(None, name)`` for the first component that is not
    there. The walk is the case check: GitHub serves names case-sensitively, and on Windows both a
    lookup and ``resolve`` find a mistyped name and hand back the disk's own spelling."""
    cur = start
    for part in path_part.split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            cur = cur.parent
            continue
        if not cur.is_dir() or part not in {p.name for p in cur.iterdir()}:
            return None, part
        cur = cur / part
    return cur, None


def _link_problems(md, root):
    """One line per link in ``md`` that does not resolve: a drive-letter or absolute path, a target
    outside ``root``, no such file (every folder and file name matched in its exact case), a folder
    with no README.md, an anchor no heading of the target carries, and a reference-style definition,
    which the pages do not use. Links to web addresses are skipped."""
    problems = []
    for n, line in _prose_lines(md.read_text(encoding="utf-8")):
        where = f"{_rel(md, root)}:{n}"
        if _REF_DEF.match(line):
            problems.append(f"{where}: reference-style link definition; write the link inline")
            continue
        for target in _LINK.findall(_CODE_SPAN.sub("", line)):
            if _DRIVE.match(target):
                problems.append(f"{where}: {target}: a drive-letter path; write it relative to the page")
                continue
            if _SCHEME.match(target):
                continue
            path_part, _, anchor = target.partition("#")
            if path_part.startswith("/"):
                problems.append(f"{where}: {target}: an absolute path; write it relative to the page")
                continue
            dest = (md.parent / path_part).resolve() if path_part else md.resolve()
            if dest != root.resolve() and root.resolve() not in dest.parents:
                problems.append(f"{where}: {target}: points outside the repository")
                continue
            dest, missing = _exact_path(md.parent, path_part) if path_part else (md, None)
            if dest is None:
                on_disk = (md.parent / path_part).exists()
                problems.append(f"{where}: {target}: " + (
                    f"'{missing}' is not the name on disk, whose case differs (GitHub's names are "
                    f"case-sensitive)" if on_disk else "no such file"))
                continue
            if dest.is_dir():
                dest = dest / "README.md"
                if not dest.is_file():
                    problems.append(f"{where}: {target}: a folder with no README.md")
                    continue
            if anchor and (dest.suffix != ".md"
                           or anchor not in _anchors(dest.read_text(encoding="utf-8"))):
                problems.append(f"{where}: {target}: no heading with the anchor #{anchor}")
    return problems


def test_every_relative_link_in_the_guide_and_readme_resolves():
    missing = [p for p in PAGES if not (GUIDE / p).is_file()]
    assert not missing, f"reader pages missing from docs/guide: {missing}"
    pages = [REPO / "README.md", *sorted(GUIDE.glob("*.md"))]
    problems = [p for md in pages for p in _link_problems(md, REPO)]
    assert not problems, "links that do not resolve:\n" + "\n".join(problems)


def test_every_guide_page_opens_with_its_commit_stamp():
    bad = []
    for name in sorted(set(PAGES) | {p.name for p in GUIDE.glob("*.md")}):
        page = GUIDE / name
        lines = [l for l in page.read_text(encoding="utf-8").splitlines() if l.strip()] \
            if page.is_file() else []
        if len(lines) < 2 or not lines[0].startswith("# ") or not _STAMP.match(lines[1]):
            bad.append(name)
    assert not bad, ("each page opens with its '# Title' line, then 'Checked against commit "
                     f"<hash>.': {bad}")


def test_the_link_checker_catches_a_missing_file_and_a_missing_anchor(tmp_path):
    """Also a name whose case differs from the disk's, for a file and for a folder: GitHub serves
    names case-sensitively, while Windows finds the file either way. And a drive-letter path, which
    is not a web address and does not exist for anyone else."""
    (tmp_path / "other.md").write_text("# Other\n\n## A section\n", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "inner.md").write_text("# Inner\n", encoding="utf-8")
    page = tmp_path / "page.md"
    page.write_text("# Page\n\n"
                    "[fine](other.md#a-section) [self](#page) [web](https://example.org)\n"
                    "[down](sub/inner.md) [up and back](sub/../other.md#a-section)\n"
                    "[gone](missing.md) [typo](other.md#nowhere)\n"
                    "[file case](Other.md) [folder case](Sub/inner.md) [drive](C:/x.md)\n"
                    "[ref]: other.md\n"
                    "`[in code](missing.md)`\n"
                    "```\n[in a fence](missing.md)\n```\n", encoding="utf-8")
    problems = _link_problems(page, tmp_path)
    assert len(problems) == 6, problems
    assert any("missing.md: no such file" in p for p in problems)
    assert any("#nowhere" in p for p in problems)
    assert any("reference-style" in p for p in problems)
    assert any(": Other.md: " in p for p in problems)
    assert any(": Sub/inner.md: " in p for p in problems)
    assert any("C:/x.md: a drive-letter path" in p for p in problems)
