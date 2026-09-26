"""Submission TSV: exact schema, coverage checks and blocking validation.

Required format: a header row "filename<TAB>cm-score", one unique row per
expected filename (with extension), finite scores in [0, 1], 1 = synthetic.
Export is refused (ExportError) if any expected file lacks a valid score;
failures are never silently dropped or filled with a constant.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd

from .cache import atomic_write_bytes

HEADER = ("filename", "cm-score")


class ExportError(ValueError):
    pass


def read_reference(path: str | Path) -> list[str]:
    """Filenames (in order) from a prefilled/template TSV with the submission header."""
    text = Path(path).read_text(encoding="utf-8-sig")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise ExportError(f"reference {path} is empty")
    header = tuple(c.strip() for c in lines[0].split("\t"))
    if header != HEADER:
        raise ExportError(f"reference header {header} != {HEADER} (tab-delimited expected)")
    names = [ln.split("\t")[0].strip() for ln in lines[1:]]
    dups = pd.Series(names)[pd.Series(names).duplicated()].tolist()
    if dups:
        raise ExportError(f"reference has {len(dups)} duplicate filenames, e.g. {dups[:3]}")
    return names


def check_scores(filenames, scores, reference: list[str] | None = None) -> list[str]:
    problems = []
    s = pd.Series(np.asarray(scores, dtype=float), index=list(filenames))
    if s.index.duplicated().any():
        problems.append(f"duplicate filenames: {s.index[s.index.duplicated()].tolist()[:5]}")
    bad = s[~np.isfinite(s.values)]
    if len(bad):
        problems.append(f"{len(bad)} non-finite scores, e.g. {bad.index.tolist()[:5]}")
    out = s[(s < 0) | (s > 1)]
    if len(out):
        problems.append(f"{len(out)} scores outside [0,1], e.g. {out.index.tolist()[:5]}")
    if reference is not None:
        ref, got = set(reference), set(s.index)
        missing, extra = sorted(ref - got), sorted(got - ref)
        if missing:
            problems.append(f"{len(missing)} reference files have no score, e.g. {missing[:5]}")
        if extra:
            problems.append(f"{len(extra)} scored files not in reference, e.g. {extra[:5]}")
    return problems


def build_submission(filenames, scores, reference: list[str] | None = None) -> pd.DataFrame:
    problems = check_scores(filenames, scores, reference)
    if problems:
        raise ExportError("export blocked: " + "; ".join(problems))
    df = pd.DataFrame({HEADER[0]: list(filenames), HEADER[1]: np.asarray(scores, dtype=float)})
    if reference is not None:
        df = df.set_index(HEADER[0]).loc[reference].reset_index()
    return df


def write_submission(df: pd.DataFrame, path: str | Path, float_format: str = "%.6f") -> None:
    if tuple(df.columns) != HEADER:
        raise ExportError(f"columns {tuple(df.columns)} != {HEADER}")
    buf = io.StringIO()
    df.to_csv(buf, sep="\t", index=False, float_format=float_format, lineterminator="\n")
    atomic_write_bytes(Path(path), buf.getvalue().encode("utf-8"))


def validate_submission_file(path: str | Path, reference: list[str] | None = None) -> dict:
    df = pd.read_csv(path, sep="\t", dtype={HEADER[0]: str})
    report = {"path": str(path), "n_rows": int(len(df)), "columns": list(df.columns)}
    problems = []
    if tuple(df.columns) != HEADER:
        problems.append(f"columns {tuple(df.columns)} != {HEADER}")
    else:
        problems += check_scores(df[HEADER[0]], df[HEADER[1]], reference)
        if reference is not None and not problems:
            if df[HEADER[0]].tolist() != list(reference):
                problems.append("row order differs from reference")
    report["problems"] = problems
    report["valid"] = not problems
    return report
