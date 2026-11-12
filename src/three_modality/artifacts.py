"""Shared artifact-safety helpers for the three-modality suite: atomic writes, content hashes, run identity, completed-run discovery, verified
reuse and never-overwrite run directories. A run is COMPLETE only when its manifest.json exists (written last) and every file listed in it still
matches its recorded SHA-256."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def atomic_write(path, data) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    mode = "wb" if isinstance(data, bytes) else "w"
    with open(tmp, mode, **({} if mode == "wb" else {"encoding": "utf-8"})) as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_save_npy(path, arr) -> None:
    """np.save straight to a temp file then os.replace -- no in-memory copy of a large array (the raw/transformed TCD matrices are ~1 GB each on the full cohort)."""
    import numpy as np
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, arr)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fingerprint(path) -> list:
    """[size, mtime_ns] -- a changed or re-copied input yields a new identity (cache miss = the safe direction)."""
    st = os.stat(path)
    return [int(st.st_size), int(st.st_mtime_ns)]


def hash_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def new_run_dir(base, prefix: str, identity: str, run_id: "str | None" = None) -> tuple:
    """(run_id, run_dir) that does not exist yet. An explicit run_id that exists is an error (never an overwrite); an auto id gets a _vN suffix."""
    base = Path(base)
    if run_id:
        d = base / run_id
        if d.exists():
            raise FileExistsError(f"{d} already exists -- completed runs are never overwritten; pick another --run-id (or omit it)")
        return run_id, d
    stem = f"{prefix}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{identity[:8]}"
    rid, v = stem, 1
    while (base / rid).exists():
        v += 1
        rid = f"{stem}_v{v}"
    return rid, base / rid


def list_runs(base_dir) -> list:
    """Completed runs (manifest.json present) under base_dir, oldest first."""
    base = Path(base_dir)
    out = []
    for d in sorted(base.glob("*")) if base.exists() else []:
        mf = d / "manifest.json"
        if mf.is_file():
            try:
                out.append({**json.loads(mf.read_text(encoding="utf-8")), "run_dir": str(d)})
            except json.JSONDecodeError:
                continue
    return out


def find_reusable_run(base_dir, identity: str) -> "dict | None":
    """A completed run with this exact identity whose files still match their recorded hashes (provenance verified), newest first, else None."""
    for r in reversed(list_runs(base_dir)):
        if r.get("identity") != identity:
            continue
        d = Path(r["run_dir"])
        if all((d / name).is_file() and sha256_file(d / name) == h for name, h in r.get("files", {}).items()):
            return r
    return None
