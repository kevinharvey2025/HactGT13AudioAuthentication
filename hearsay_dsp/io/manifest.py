"""Manifest reading, explicit label normalization and validation.

Ground truth comes only from an explicit label column mapped through
LABEL_MAP (or a user-supplied mapping). Filenames and directories are never
used to infer labels.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

AUDIO_EXTENSIONS = (".wav", ".mp3", ".m4a", ".mp4", ".aac", ".ogg", ".oga", ".opus",
                    ".flac", ".aif", ".aiff", ".wma", ".webm", ".amr", ".3gp")

# real / bona fide -> 0, synthetic / spoof -> 1
LABEL_MAP = {
    "real": 0, "bonafide": 0, "bona-fide": 0, "bona fide": 0, "genuine": 0, "0": 0,
    "fake": 1, "spoof": 1, "synthetic": 1, "1": 1,
}

OPTIONAL_COLUMNS = ("split", "group_id", "speaker_id", "source_id", "attack_type")
EVAL_SPLITS = {"eval", "evaluation", "test", "holdout"}


class ManifestError(ValueError):
    pass


def normalize_label(value, label_map: dict | None = None):
    """Map a raw label to 0 (real) / 1 (synthetic); None for missing.

    Raises ManifestError for values not in the explicit mapping, rather than
    guessing (e.g. "maybe", "partial", "2" are rejected).
    """
    mapping = label_map or LABEL_MAP
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    key = str(value).strip().lower()
    if key == "" or key == "nan":
        return None
    if key.endswith(".0") and key[:-2] in mapping:  # "1.0" from numeric CSV columns
        key = key[:-2]
    if key not in mapping:
        raise ManifestError(f"ambiguous or unknown label {value!r}; allowed: {sorted(mapping)}")
    return int(mapping[key])


def read_manifest(path: str | Path, label_map: dict | None = None,
                  path_column: str = "path") -> pd.DataFrame:
    """Read a CSV/TSV manifest. Relative paths resolve against the manifest's directory."""
    path = Path(path)
    sep = "\t" if path.suffix.lower() in (".tsv", ".tab") else None
    df = pd.read_csv(path, sep=sep, engine="python", dtype=str, keep_default_na=False)
    if path_column not in df.columns:
        raise ManifestError(f"manifest {path} lacks required column {path_column!r}")
    base = path.parent.resolve()
    df = df.rename(columns={path_column: "path"})
    df["path"] = [p if os.path.isabs(p) else str((base / p)) for p in df["path"]]
    df["filename"] = [os.path.basename(p) for p in df["path"]]
    if "label" in df.columns:
        df["label_raw"] = df["label"]
        df["label"] = [normalize_label(v, label_map) for v in df["label"]]
        df["label"] = df["label"].astype("float")  # NaN for unlabeled rows
    else:
        df["label"] = np.nan
    for col in OPTIONAL_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df


def infer_directory(directory: str | Path, extensions=AUDIO_EXTENSIONS) -> pd.DataFrame:
    """Unlabeled manifest from a directory listing (sorted, recursive)."""
    directory = Path(directory)
    exts = tuple(e.lower() for e in extensions)
    paths = sorted(str(p) for p in directory.rglob("*")
                   if p.is_file() and p.suffix.lower() in exts and not p.name.startswith("."))
    df = pd.DataFrame({"path": paths})
    df["filename"] = [os.path.basename(p) for p in paths]
    df["label"] = np.nan
    for col in OPTIONAL_COLUMNS:
        df[col] = ""
    return df


def load_inputs(manifest: str | None = None, input_dir: str | None = None,
                label_map: dict | None = None) -> pd.DataFrame:
    if bool(manifest) == bool(input_dir):
        raise ManifestError("provide exactly one of --manifest or --input")
    return read_manifest(manifest, label_map) if manifest else infer_directory(input_dir)


def select_training_rows(df: pd.DataFrame, allow_splits: set | None = None) -> pd.DataFrame:
    """Rows usable for fitting: labeled and not designated evaluation/test."""
    split = df["split"].astype(str).str.strip().str.lower()
    blocked = split.isin(EVAL_SPLITS)
    if allow_splits is not None:
        blocked |= ~split.isin({s.lower() for s in allow_splits})
    return df[~blocked & df["label"].notna()].copy()


def validate_manifest(df: pd.DataFrame, check_files: bool = True) -> dict:
    """Structural checks. Returns a report; `errors` non-empty means unusable."""
    report: dict = {"n_rows": int(len(df)), "errors": [], "warnings": []}
    dup_paths = df["path"][df["path"].duplicated()].tolist()
    if dup_paths:
        report["errors"].append(f"{len(dup_paths)} duplicate paths, e.g. {dup_paths[:3]}")
    dup_names = df["filename"][df["filename"].duplicated()].unique().tolist()
    report["duplicate_basenames"] = len(dup_names)
    if dup_names:
        report["warnings"].append(
            f"{len(dup_names)} basenames occur more than once (export by basename would collide),"
            f" e.g. {dup_names[:3]}")
    if check_files:
        missing = [p for p in df["path"] if not os.path.isfile(p)]
        report["missing_files"] = len(missing)
        if missing:
            report["errors"].append(f"{len(missing)} files missing, e.g. {missing[:3]}")
    labeled = df["label"].notna()
    report["n_labeled"] = int(labeled.sum())
    report["n_real"] = int((df["label"] == 0).sum())
    report["n_synthetic"] = int((df["label"] == 1).sum())
    if 0 < labeled.sum() < len(df):
        report["warnings"].append(f"{int((~labeled).sum())} rows have no label")
    split = df["split"].astype(str).str.strip().str.lower()
    report["splits"] = split.value_counts().to_dict()
    if (df["group_id"].astype(str) == "").all():
        report["warnings"].append("no group_id column: splits cannot keep related recordings together")
    return report
