"""Who produced an artifact, from what, on which machine: git, package versions, input file hashes."""
from __future__ import annotations

import hashlib
import platform
import socket
import subprocess
import sys
from pathlib import Path

UNKNOWN_GIT = {"git_rev": "unknown", "git_dirty": None, "branch": "unknown"}


def _git(args, root) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=str(root), capture_output=True, text=True, timeout=5)
    except Exception:                        # noqa: BLE001 -- no git binary, a hung index, a timeout
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def git_info(root) -> dict:
    """``{git_rev, git_dirty, branch}`` for the repo containing ``root``; UNKNOWN_GIT outside one.
    Never raises: provenance must not be the thing that loses a multi-day run."""
    rev = _git(["rev-parse", "HEAD"], root)
    if not rev:
        return dict(UNKNOWN_GIT)
    status = _git(["status", "--porcelain"], root)
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], root) or "unknown"
    return {"git_rev": rev, "git_dirty": None if status is None else bool(status), "branch": branch}


def device_name(hw) -> str:
    dev = getattr(hw, "device", None)
    if getattr(dev, "type", None) == "cuda":
        try:
            import torch
            return torch.cuda.get_device_name(dev)
        except Exception:                    # noqa: BLE001
            return "cuda"
    return str(getattr(dev, "type", dev) or "cpu")


def _version(mod: str) -> str:
    try:
        return str(__import__(mod).__version__)
    except Exception:                        # noqa: BLE001 -- an optional package that is absent
        return "absent"


def env_info(hw) -> dict:
    return {
        "python": sys.version.split()[0], "torch": _version("torch"), "sbi": _version("sbi"),
        "PySide6": _version("PySide6"), "numpy": _version("numpy"),
        "os": f"{platform.system()} {platform.release()}", "hostname": socket.gethostname(),
        "device_type": str(getattr(getattr(hw, "device", None), "type", "cpu")),
        "device_name": device_name(hw), "dtype": str(getattr(hw, "dtype", "unknown")),
    }


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_ref(path, relative_to=None) -> dict:
    """``{"path": <relative when under relative_to, else absolute>, "sha256": ...}``. Refuses a missing
    file: an artifact must never claim an input it could not read."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"input file not found: {p}")
    shown = str(p.resolve())
    if relative_to is not None:
        try:
            shown = p.resolve().relative_to(Path(relative_to).resolve()).as_posix()
        except ValueError:
            pass
    return {"path": shown, "sha256": sha256_file(p)}


def inputs_from_cfg(cfg) -> dict:
    """The bounds / cell / units files a config was built from (``SimConfig.sources``), hashed."""
    from core import config
    src = getattr(cfg, "sources", None) or {}
    out = {"model": cfg.model}
    for key in ("bounds", "cell", "units"):
        out[key] = file_ref(src[key], relative_to=config.RESOURCES_ROOT) if src.get(key) else None
    return out


def _nvidia_smi_free_gib() -> "float | None":
    """Free VRAM in GiB according to ``nvidia-smi``, or None if it cannot be read.

    DELIBERATELY NOT ``torch.cuda.mem_get_info``: that overstates free VRAM on Windows by roughly the
    size of the desktop (measured 15037 MiB against nvidia-smi's 5814 at the same instant), and it is
    the number that green-lit the batch which killed the first chi retrain.
    """
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=2.0)
        if out.returncode != 0:
            return None
        return float(out.stdout.strip().splitlines()[0]) / 1024.0
    except Exception:                        # noqa: BLE001 -- no driver, no binary, a timeout: all "unknown"
        return None
