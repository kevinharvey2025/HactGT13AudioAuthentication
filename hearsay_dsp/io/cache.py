"""Content-addressed feature cache with atomic writes.

Layout: <root>/features/<sha[:2]>/<sha>.json holds one record per audio
content hash with per-module entries {key, result}; large arrays live in
<sha>.<module>.<key>.npz. A module entry is reused only when its key (module
version + parameters + preprocessing identity) matches and every array it
lists exists. Files are written to a temporary name and renamed, so a crash
never leaves a partial entry that looks complete.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def atomic_write_json(path: Path, obj) -> None:
    atomic_write_bytes(Path(path), json.dumps(obj, indent=1, sort_keys=True).encode("utf-8"))


def atomic_save_npz(path: Path, arrays: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".npz")
    os.close(fd)
    try:
        np.savez(tmp, **arrays)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


class FeatureCache:
    def __init__(self, root: str | Path):
        self.root = Path(root) / "features"

    def _dir(self, sha: str) -> Path:
        return self.root / sha[:2]

    def record_path(self, sha: str) -> Path:
        return self._dir(sha) / f"{sha}.json"

    def array_path(self, sha: str, module: str, key: str) -> Path:
        return self._dir(sha) / f"{sha}.{module}.{key}.npz"

    def load_record(self, sha: str) -> dict:
        path = self.record_path(sha)
        if not path.exists():
            return {}
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError):
            return {}  # unreadable -> treated as absent and recomputed

    def valid_entry(self, record: dict, sha: str, module: str, key: str) -> dict | None:
        entry = record.get("modules", {}).get(module)
        if not entry or entry.get("key") != key:
            return None
        if entry.get("arrays") and not self.array_path(sha, module, key).exists():
            return None
        return entry

    def save_record(self, sha: str, record: dict) -> None:
        atomic_write_json(self.record_path(sha), record)

    def save_arrays(self, sha: str, module: str, key: str, arrays: dict) -> None:
        atomic_save_npz(self.array_path(sha, module, key), arrays)

    def load_arrays(self, sha: str, module: str, key: str) -> dict | None:
        path = self.array_path(sha, module, key)
        if not path.exists():
            return None
        with np.load(path) as data:
            return {k: data[k] for k in data.files}
