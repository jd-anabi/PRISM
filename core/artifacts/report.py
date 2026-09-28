"""Two renderers over an artifact's manifest: the artifact as text, and its lineage.

A saveable run summary is two documents -- what the browser's detail pane shows (and
``python -m core artifacts show`` prints), and a lineage walk back through an artifact's parents. Both
are FILES, never a store kind of their own, and both are pure functions of manifests the store
already holds: nothing here writes, creates or resolves anything.

It lives beside the manifest code it reads and NOT in ``store.py``, because a report DESCRIBES the
store and must never be mistaken for something the store holds. ONE renderer for both front ends, so
the document a reviewer receives is the same bytes whichever one made it.

DETERMINISM IS THE CONTRACT. Every mapping renders in sorted key order and every sequence in its own
order -- a parameter list's order is meaning, a dict's is not. ``manifest.to_json_text`` already
writes with ``sort_keys=True``, so a manifest read back from disk is in that order anyway; sorting
here is what makes a manifest built in memory print identically to the same manifest read back.

TORCH-FREE in the same qualified sense as ``manifest.py``: its own module-level imports are nothing at
all, the store's parent table is imported lazily inside ``render_lineage``, and a caller reaches it
through ``core.artifacts``, which imports torch. No claim is made that it can be imported without
torch.
"""
from __future__ import annotations

INDENT = "  "
NONE = "(none)"
RULE = "=" * 72
# Numbers printed inline before a sequence is summarised instead. A rotation matrix is 13x13 float64
# lists -- 169 numbers whose digest is printed two sections above -- and inlining it would bury every
# knob that decides anything.
MAX_INLINE = 12
# The widest packed knob line. ``config`` carries the ~35-key identity vocabulary plus the stage's own,
# so the lineage packs them rather than spending a line each.
KNOB_WIDTH = 100


def _is_scalar(v) -> bool:
    return v is None or isinstance(v, (bool, int, float, str))


def _fmt_scalar(v) -> str:
    """One manifest value as text. ``bool`` before ``int`` (it is a subclass), and the JSON spelling of
    it, so a line diffs against ``manifest.json`` itself; ``repr`` for a float, which is
    shortest-round-trip and therefore exact."""
    if v is None:
        return NONE
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return repr(v)
    return str(v)


def _fmt_list(v) -> str:
    """One line for a sequence: inline while it is short, otherwise its shape and its first entries."""
    if not v:
        return "[]"
    if all(_is_scalar(x) for x in v):
        if len(v) <= MAX_INLINE:
            return "[" + ", ".join(_fmt_scalar(x) for x in v) + "]"
        return "[" + ", ".join(_fmt_scalar(x) for x in v[:3]) + f", ... ({len(v)} entries)]"
    if all(isinstance(x, list) for x in v):
        widths = sorted({len(x) for x in v})
        shape = str(widths[0]) if len(widths) == 1 else f"{widths[0]}-{widths[-1]}"
        return f"[{len(v)} x {shape} numbers]"
    return f"[{len(v)} entries]"


def _tree(value, indent: str) -> list:
    """``key: value`` lines for a nested manifest value: dicts in sorted key order, a list of records
    indexed in its own order (``sbc.per_param`` and ``posterior_summary`` are lists precisely so the
    parameter order survives the manifest's recursive key sort)."""
    if not isinstance(value, dict):
        return [indent + _fmt_scalar(value)]
    lines = []
    for key in sorted(value):
        v = value[key]
        if isinstance(v, dict) and v:
            lines.append(f"{indent}{key}:")
            lines.extend(_tree(v, indent + INDENT))
        elif isinstance(v, dict):
            lines.append(f"{indent}{key}: {{}}")
        elif isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
            lines.append(f"{indent}{key}:")
            for i, item in enumerate(v):
                lines.append(f"{indent}{INDENT}[{i}]")
                lines.extend(_tree(item, indent + INDENT + INDENT))
        elif isinstance(v, list):
            lines.append(f"{indent}{key}: {_fmt_list(v)}")
        else:
            lines.append(f"{indent}{key}: {_fmt_scalar(v)}")
    return lines


def _input_lines(inputs: dict) -> list:
    """The input files with their hashes. ``model`` is a name and not a file; a None entry is a file the
    artifact does not have (a bounds-built config carries no cell). An UNHASHED file says so in words:
    the writer records ``sha256: None`` for an input that had moved by the commit
    (``ArtifactWriter._commit``), and a report that printed nothing there would read as a file that was
    checked."""
    lines = []
    for key in sorted(inputs):
        v = inputs[key]
        if isinstance(v, dict):
            h = v.get("sha256")
            lines.append(f"{INDENT}{key}: {v.get('path')}  sha256 " + (h if h else "NOT READ AT COMMIT"))
        else:
            lines.append(f"{INDENT}{key}: {_fmt_scalar(v)}")
    return lines


def _pack(d: dict, indent: str, width: int = KNOB_WIDTH) -> list:
    """``key=value`` pairs in sorted key order, packed onto lines no wider than ``width``.

    The WHOLE ``config`` block, not a chosen subset: the manifest's ``config`` is "the knobs it ran
    under" -- the identity vocabulary every artifact records (``manifest.config_from_cfg``) plus the
    stage's own ``w.config.update(...)`` -- and a report that printed a hand-picked subset would be a
    second copy of that key list, one stage away from going stale.
    """
    if not d:
        return [indent + NONE]
    parts = []
    for key in sorted(d):
        v = d[key]
        if isinstance(v, list):
            shown = _fmt_list(v)
        elif isinstance(v, dict):
            shown = "{...}" if v else "{}"
        else:
            shown = _fmt_scalar(v)
        parts.append(f"{key}={shown}")
    lines, cur = [], ""
    for part in parts:
        if cur and len(indent) + len(cur) + 2 + len(part) > width:
            lines.append(indent + cur)
            cur = part
        else:
            cur = part if not cur else f"{cur}  {part}"
    if cur:
        lines.append(indent + cur)
    return lines


def _section(title: str, lines: list) -> list:
    return [title, *(lines or [INDENT + NONE]), ""]


def render_manifest(m) -> str:
    """The artifact as text: the header, the git revision and the environment, the input files with
    their hashes, the parents, the payloads with their digests, the figures by name, the fingerprints,
    the knobs and the kind's own body.

    Takes a ``Manifest`` (``store.get(kind, ref)``), never a path or a ref: the caller has already
    resolved and validated it, and a renderer that resolved refs of its own would be a second index.
    It therefore says nothing about the DIRECTORY -- a folder name that disagrees with
    ``Manifest.dir_name`` is the browser's notice to give, off ``Summary.dir_name``, because
    only a listing knows it.
    """
    out = [f"{m.kind} {m.name or '(unnamed)'}  [{m.id}]", RULE,
           f"schema:  {m.schema}",
           f"kind:    {m.kind}",
           f"id:      {m.id}",
           f"name:    {m.name or NONE}",
           f"created: {m.created}",
           f"note:    {m.note or NONE}", ""]
    out += _section("PRISM", _tree(m.prism, INDENT))
    out += _section("Environment", _tree(m.env, INDENT))
    out += _section("Inputs", _input_lines(m.inputs))
    out += _section("Parents", [f"{INDENT}{k}: {m.parents[k]}" for k in sorted(m.parents)])
    out += _section("Payloads", [f"{INDENT}{k}  sha256 {m.payloads[k] or NONE}" for k in sorted(m.payloads)])
    out += _section("Figures", [f"{INDENT}{f}" for f in m.figures])
    out += _section("Fingerprints", [f"{INDENT}{k}: {m.fingerprints[k] or NONE}" for k in sorted(m.fingerprints)])
    out += _section("Knobs", _tree(m.config, INDENT))
    out += _section("Body", _tree(m.body, INDENT))
    return "\n".join(out).rstrip("\n") + "\n"


def _accepted(m) -> list:
    """The escape hatches the artifact records (``Accept.used()``), or ``[]``.

    An inference writes them under ``body.results.accepted`` (``orchestrator.py:2176``) and so does
    every diagnostic (``sbc.py:220``, ``identifiability.py:182,464``, ``ablation.py:203``); a
    calibration records none -- there the flag only ever unlocked the load, and what is on record is
    the posterior's own ``amortized: false``. So the key is read with ``.get`` and its absence is not a
    gap.
    """
    results = m.body.get("results")
    if not isinstance(results, dict):
        return []
    return [str(a) for a in (results.get("accepted") or [])]


def _compared_lines(store, m) -> list:
    """What a comparison DREW, resolved.

    A comparison names its sources in the body and not in ``parents``: that block is a flat
    ``{key: id}`` map read as ``m.parents.get(pk) == id_``, so it cannot carry an arbitrary number of
    ids without widening the store's contract for every kind. The consequence is deliberate --
    deleting a record a comparison used is NOT refused -- and these lines are what keep the
    comparison honest about it, printing ``MISSING <kind> [<id>]`` in the wording the parents walk
    above already uses.

    Reads ``body["compared"]`` with ``.get``: every other kind's body has no such key, and a body
    that has it null is a record of this kind that is not a comparison.

    Any other shape than ``{"mode": ..., "records": [{"kind", "id", "name"}, ...]}`` prints a line
    saying WHAT is unreadable. Only a hand-edited manifest or a writer bug gets
    here, but both of ``render_lineage``'s promises still hold for it. Printing nothing would make the
    comparison read as having drawn nothing. Iterating a string or a mapping would print a false
    ``MISSING fdt []`` for every character or key. Letting a ``TypeError`` out would break "the one
    refusal is the head ref", and the browser catches only a Refusal.
    """
    from .store import KIND_DIRS, StoreError
    comp = m.body.get("compared") if isinstance(m.body, dict) else None
    if comp is None:
        return []
    if not isinstance(comp, dict):
        return [f"{INDENT}compared: UNREADABLE ({type(comp).__name__}, not a mapping of mode and records)"]
    out = [f"{INDENT}compared ({comp.get('mode') or NONE}):"]
    records = comp.get("records")
    if not isinstance(records, list):
        shown = "null" if records is None else type(records).__name__
        out.append(f"{INDENT}{INDENT}UNREADABLE records ({shown}, not a list of entries)")
        return out
    for rec in records:
        if not isinstance(rec, dict):
            # Its repr, never {}: a bare id string is still an id, and the reader should see it.
            out.append(f"{INDENT}{INDENT}UNREADABLE entry {rec!r} (not a mapping of kind, id and name)")
            continue
        # "fdt" is the default because a comparison of measurements is what this block exists for;
        # a kind this build does not know is named rather than skipped, like an unknown parent key.
        ck = str(rec.get("kind") or "fdt")
        if ck not in KIND_DIRS:
            out.append(f"{INDENT}{INDENT}(compared entry names kind {ck!r}, which is no artifact "
                       f"kind this build knows)")
            continue
        if rec.get("id") in (None, ""):
            # Not "MISSING fdt []": nothing was looked up, and the entry's repr keeps the name it gave.
            out.append(f"{INDENT}{INDENT}UNREADABLE entry {rec!r} (names no id)")
            continue
        cid = str(rec["id"])
        try:
            cm = store.get(ck, cid)
        except StoreError:
            cm = None
        if cm is None:
            out.append(f"{INDENT}{INDENT}MISSING {ck} [{cid}]  -- no complete {ck} with that id "
                       f"under {store.kind_dir(ck)}")
        else:
            out.append(f"{INDENT}{INDENT}{ck} {cm.name or '(unnamed)'} [{cm.id}]")
    return out


def render_lineage(store, kind: str, ref: str) -> str:
    """The artifact and its parents transitively, each artifact BEFORE its own parents, so every chain
    reads newest first and oldest last.

    Each step: the kind, the name, the id, the creation time, the input files with their hashes, the
    knobs that decided it, any Accept it recorded, and -- for a comparison -- the records it drew
    (``_compared_lines``), which are listed but not walked. A parent a manifest names but the store
    does not hold prints as MISSING with its id and with who named it -- skipping it would make a
    broken chain read as a complete one, which is the one thing a provenance report must never do.

    Parents are older ids by construction, so a cycle is impossible; the visited set is carried anyway,
    because the alternative to a two-line guard is an endless report out of one hand-edited manifest.
    It also collapses the diamond every trained posterior has: the prior is named by the posterior AND
    by the training cache, and belongs in the report once.

    The one refusal is the head ref: ``store.get``'s own StoreError, unchanged.
    """
    # Lazily, so this module's own imports stay standard library (see the module docstring), and
    # INVERTED from the store's table -- ``parent key -> kind`` -- so a parent key added there is
    # followed here without a second list to keep in step.
    from .store import StoreError, _PARENT_KEYS
    key_kind = {key: k for k, keys in _PARENT_KEYS.items() for key in keys}
    head = store.get(kind, ref)
    out = [f"Lineage of {kind} {head.name or '(unnamed)'} [{head.id}]", RULE, ""]
    queue = [(kind, head.id, head, "")]
    seen = {(kind, head.id)}
    step = 0
    while queue:
        k, id_, m, named_by = queue.pop(0)
        step += 1
        if m is None:
            out.append(f"{step}. MISSING {k} [{id_}]")
            out.append(f"{INDENT}named by {named_by}; no complete {k} with that id under "
                       f"{store.kind_dir(k)}")
            out.append("")
            continue
        label = m.name or "(unnamed)"
        out.append(f"{step}. {k} {label} [{m.id}]  created {m.created}")
        out.append(f"{INDENT}inputs:")
        out += [INDENT + line for line in _input_lines(m.inputs)]
        out.append(f"{INDENT}knobs:")
        out += _pack(m.config, INDENT + INDENT)
        acc = _accepted(m)
        if acc:
            out.append(f"{INDENT}accepted: {', '.join(acc)}")
        pairs = [(pk, m.parents[pk]) for pk in sorted(m.parents) if m.parents[pk]]
        out.append(f"{INDENT}parents: " + (", ".join(f"{pk}={pid}" for pk, pid in pairs) or NONE))
        out += _compared_lines(store, m)
        for pk, pid in pairs:
            pkind = key_kind.get(pk)
            if pkind is None:
                # Not silence: a parent key no kind claims is a writer this reader does not know about,
                # and a chain that dropped it would read as complete.
                out.append(f"{INDENT}(parent key {pk!r} names no artifact kind this build knows)")
                continue
            if (pkind, pid) in seen:
                continue
            seen.add((pkind, pid))
            try:
                pm = store.get(pkind, pid)
            except StoreError:
                pm = None
            queue.append((pkind, pid, pm, f"{k} {label!r} as {pk!r}"))
        out.append("")
    return "\n".join(out).rstrip("\n") + "\n"
