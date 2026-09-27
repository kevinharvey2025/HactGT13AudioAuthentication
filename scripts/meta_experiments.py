"""Metadata-only detectors and interventions (plans/metadata_analysis_prompt.md: M0, M1, M2, X0 inputs).

    python scripts/meta_experiments.py fit             # M0/M1/M2 (+X0 inputs) on the shared split
    python scripts/meta_experiments.py interventions   # tag strip / tag replace / format re-encode, then rescore

Inputs: cache/manifest_pool.parquet + cache/triage_pool.parquet (ffprobe + decoded-signal facts per file).
Every model sees only its allowlisted namespaces (checked before fitting):
  meta.technical   container, codec, native_sr, channels, sample_fmt, bits, bitrate
  meta.sizedur     header_duration, size_bytes            (collection shortcuts, separate ablation group)
  meta.tags        encoder family, n_tags                 (editable)
  meta.consistency RIFF header arithmetic (byte rate, block align, RIFF size), extension vs container
  cross            occupied bandwidth / declared Nyquist, header vs decoded duration (X0 only)
Never: paths, filenames, ids, MAC times, hashes. Split: hearsay.splits.shared_split (train+val = fit pool with
grouped 5-fold OOF; holdout = reserved evaluation). Test scores are reported; with one format for all test
files every metadata model gives them a single constant score.
Writes runs/meta/{metrics.json, oof.parquet, holdout.parquet, test.parquet, interventions.parquet}.
"""
import argparse
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hearsay import config, metrics, splits  # noqa: E402
from hearsay.forensics import triage  # noqa: E402

OUT = config.REPO / "runs" / "meta"
NS = {
    "meta.technical": (["container", "codec", "sample_fmt"], ["native_sr", "channels", "bits", "bitrate"]),
    "meta.sizedur": ([], ["header_duration", "size_bytes"]),
    "meta.tags": (["encoder_family"], ["n_tags"]),
    "meta.consistency": (["ext_container"], ["riff_size_ok", "byte_rate_ok", "block_align_ok"]),
    "cross": ([], ["bw_frac_of_nyquist", "duration_mismatch_s"]),
}
EXPERIMENTS = {
    "M0": ["meta.technical", "meta.sizedur"],
    "M0-sizedur": ["meta.technical"],
    "M1": ["meta.technical", "meta.sizedur", "meta.tags"],
    "M2": ["meta.technical", "meta.sizedur", "meta.consistency"],
    "X0-inputs": ["meta.technical", "meta.sizedur", "meta.tags", "meta.consistency", "cross"],
}
FORBIDDEN = {"path", "file", "filename", "uid", "mtime", "ctime", "atime", "birthtime", "sha256"}


def riff_checks(path):
    try:
        b = open(path, "rb").read(4096)
        size = Path(path).stat().st_size
    except OSError:
        return dict(riff_size_ok=np.nan, byte_rate_ok=np.nan, block_align_ok=np.nan)
    if b[:4] != b"RIFF" or b[8:12] != b"WAVE":
        return dict(riff_size_ok=np.nan, byte_rate_ok=np.nan, block_align_ok=np.nan)
    r = dict(riff_size_ok=float(struct.unpack("<I", b[4:8])[0] == size - 8), byte_rate_ok=np.nan, block_align_ok=np.nan)
    i = b.find(b"fmt ")
    if i >= 0 and i + 24 <= len(b):
        tag, ch, rate, brate, align, bits = struct.unpack("<HHIIHH", b[i + 8:i + 24])
        if tag == 1:
            r.update(byte_rate_ok=float(brate == rate * ch * bits // 8), block_align_ok=float(align == ch * bits // 8))
    return r


def encoder_family(e):
    e = (e or "").strip().lower()
    if not e:
        return "absent"
    for fam in ("lavf", "lavc", "lame", "sox", "audacity", "ffmpeg", "praat", "libsndfile"):
        if e.startswith(fam) or fam in e:
            return fam
    return "other"


def features(man, tri):
    d = man.merge(tri, on="uid", how="left")
    d["encoder_family"] = d.encoder.fillna("").map(encoder_family)
    ext = d.path.str.rsplit(".", n=1).str[-1].str.lower()
    d["ext_container"] = np.where(ext.eq("wav") & ~d.container.fillna("").str.contains("wav"), "mismatch",
                                  np.where(ext.eq("mp3") & ~d.container.fillna("").str.contains("mp3"), "mismatch", "match"))
    d["bitrate"] = d.bitrate.replace(0, np.nan)
    d["bw_frac_of_nyquist"] = d.cutoff_hz / (d.native_sr.replace(0, np.nan) / 2)
    rc = pd.DataFrame([riff_checks(p) for p in d.path])
    return pd.concat([d.reset_index(drop=True), rc], axis=1)


def columns(exp):
    cat, num = [], []
    for ns in EXPERIMENTS[exp]:
        cat += NS[ns][0]
        num += NS[ns][1]
    assert not (set(cat) | set(num)) & FORBIDDEN, "diagnostic field in a model"
    return cat, num


def model(kind, cat, num):
    if kind == "lr":
        pre = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), cat),
            ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), num)])
        return make_pipeline(pre, LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000))
    pre = ColumnTransformer([("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                                    encoded_missing_value=-2), cat),
                             ("num", "passthrough", num)])
    return make_pipeline(pre, HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, class_weight="balanced",
                                                             categorical_features=list(range(len(cat))), random_state=0))


def cmd_fit(a):
    OUT.mkdir(parents=True, exist_ok=True)
    man = pd.read_parquet(config.CACHE / "manifest_pool.parquet")
    tri = pd.read_parquet(config.CACHE / "triage_pool.parquet")
    d = features(man, tri)
    lab = d[d.label >= 0].reset_index(drop=True)
    test = d[d.label < 0].reset_index(drop=True)
    lab["split"] = splits.shared_split(lab)
    fitp = lab[lab.split.isin(["train", "val"])].reset_index(drop=True)
    hold = lab[lab.split == "holdout"].reset_index(drop=True)
    itw = lab[lab.split == "itw"].reset_index(drop=True)   # In-the-Wild: web audio in other formats, never fitted
    y, yh = fitp.label.to_numpy(), hold.label.to_numpy()
    groups = fitp.text_group.to_numpy()
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(fitp, fitp.generator, groups))
    res, oof_all, hold_all, test_all = [], {}, {}, {}
    for exp in EXPERIMENTS:
        cat, num = columns(exp)
        for kind in ("lr", "hgb"):
            name = f"{exp}:{kind}"
            oof = np.zeros(len(fitp))
            for tr, te in folds:
                oof[te] = model(kind, cat, num).fit(fitp.iloc[tr][cat + num], y[tr]).predict_proba(fitp.iloc[te][cat + num])[:, 1]
            full = model(kind, cat, num).fit(fitp[cat + num], y)
            ph, pt = full.predict_proba(hold[cat + num])[:, 1], full.predict_proba(test[cat + num])[:, 1]
            r = dict(system=name, features=",".join(EXPERIMENTS[exp]))
            pi = full.predict_proba(itw[cat + num])[:, 1] if len(itw) else np.zeros(0)
            for part, yy, ss in [("oof", y, oof), ("holdout", yh, ph)] + ([("itw", itw.label.to_numpy(), pi)] if len(itw) else []):
                m = metrics.summary(yy, ss, prob=ss)
                r.update({f"{part}_{k}": round(float(v), 4) for k, v in m.items() if k in ("auc", "eer", "min_dcf", "logloss", "brier")})
            r.update(test_unique_scores=int(np.unique(pt.round(6)).size), test_mean=round(float(pt.mean()), 4))
            res.append(r)
            oof_all[name], hold_all[name], test_all[name] = oof, ph, pt
            print(json.dumps(r), flush=True)
    pd.DataFrame(res).to_json(OUT / "metrics.json", orient="records", indent=1)
    pd.DataFrame({"uid": fitp.uid, "label": y, "split": fitp.split, **oof_all}).to_parquet(OUT / "oof.parquet")
    pd.DataFrame({"uid": hold.uid, "label": yh, "generator": hold.generator, **hold_all}).to_parquet(OUT / "holdout.parquet")
    pd.DataFrame({"filename": test.filename, **test_all}).to_parquet(OUT / "test.parquet")
    # what the fit pool looks like per class (the shortcut, in one table)
    print(fitp.groupby(["label", "container", "native_sr", "encoder_family"]).size().to_string())


def _ff(args):
    subprocess.run([config.ffmpeg_bin(), "-v", "error", "-nostdin", "-y", *args], check=True)


def _pcm(path):
    """Decoded samples at the native rate/channels with one fixed decoder (for equivalence checks)."""
    return subprocess.run([config.ffmpeg_bin(), "-v", "error", "-nostdin", "-i", path, "-f", "s16le", "-"],
                          capture_output=True, check=True).stdout


OPS = {  # name -> (ffmpeg args after -i SRC, output extension or None = keep, metadata-only?)
    "strip_tags": (["-map_metadata", "-1", "-fflags", "+bitexact", "-c:a", "copy"], None, True),
    "set_lavf_tag": (["-map_metadata", "-1", "-metadata", "encoder=Lavf58.29.100", "-c:a", "copy"], None, True),
    "reencode_16k_pcm_wav": (["-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le"], "wav", False),
}


def cmd_interventions(a):
    """Holdout sample through metadata-only edits (PCM identity verified) and one signal-changing re-encode;
    the metadata models fitted on the fit pool rescore each edited file."""
    OUT.mkdir(parents=True, exist_ok=True)
    man = pd.read_parquet(config.CACHE / "manifest_pool.parquet")
    tri = pd.read_parquet(config.CACHE / "triage_pool.parquet")
    d = features(man, tri)
    lab = d[d.label >= 0].reset_index(drop=True)
    lab["split"] = splits.shared_split(lab)
    fitp = lab[lab.split.isin(["train", "val"])].reset_index(drop=True)
    hold = lab[lab.split == "holdout"].groupby("label", group_keys=False).sample(a.n // 2, random_state=0).reset_index(drop=True)
    fitted = {}
    for exp in ("M0", "M1", "M2"):
        cat, num = columns(exp)
        fitted[exp] = (model("lr", cat, num).fit(fitp[cat + num], fitp.label), cat + num)
    base = {exp: m.predict_proba(hold[cols])[:, 1] for exp, (m, cols) in fitted.items()}
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, r in hold.iterrows():
            for op, (args, ext, meta_only) in OPS.items():
                out = str(Path(tmp) / f"{op}.{ext or Path(r.path).suffix.lstrip('.')}")
                try:
                    _ff(["-i", r.path, *args, out])
                    same = _pcm(r.path) == _pcm(out) if meta_only else False
                    pr = triage.probe(out)
                    e = r.copy()
                    for k in ("container", "codec", "native_sr", "channels", "sample_fmt", "bits", "bitrate", "header_duration", "n_tags"):
                        e[k] = pr[k]
                    e["bitrate"] = pr["bitrate"] or np.nan
                    e["size_bytes"] = pr["size_bytes"]
                    e["encoder_family"] = encoder_family(pr["encoder"])
                    e.update(pd.Series(riff_checks(out)))
                    ext_ = Path(out).suffix.lstrip(".")
                    e["ext_container"] = "match" if (ext_ in (pr["container"] or "")) else "mismatch"
                    row = dict(uid=r.uid, label=r.label, generator=r.generator, op=op, metadata_only=meta_only,
                               pcm_identical=same, encoder_after=pr["encoder"], sr_after=pr["native_sr"])
                    for exp, (m, cols) in fitted.items():
                        pe = float(m.predict_proba(pd.DataFrame([e[cols]]))[:, 1][0])
                        row.update({f"{exp}_before": float(base[exp][i]), f"{exp}_after": pe,
                                    f"{exp}_flip": (base[exp][i] >= 0.5) != (pe >= 0.5)})
                    rows.append(row)
                except Exception as ex:
                    rows.append(dict(uid=r.uid, label=r.label, op=op, error=repr(ex)[:200]))
    res = pd.DataFrame(rows)
    res.to_parquet(OUT / "interventions.parquet")
    agg = {"n": ("uid", "size"), "pcm_identical": ("pcm_identical", "mean")}
    for exp in fitted:
        agg[f"{exp}_flip_rate"] = (f"{exp}_flip", "mean")
        agg[f"{exp}_mean_after"] = (f"{exp}_after", "mean")
    print(res.groupby(["op", "label"]).agg(**agg).round(3).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "interventions"])
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    {"fit": cmd_fit, "interventions": cmd_interventions}[a.cmd](a)


if __name__ == "__main__":
    main()
