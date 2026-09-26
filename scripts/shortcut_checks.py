"""Plan section 13: how far do trivial, non-content features go on the labeled data?

For each feature group a small gradient-boosting model is scored with grouped 5-fold CV, once on
the raw decoded clips and once on the canonical view every model sees (trim, test-like crop,
7 kHz low-pass, peak-norm, dither). A high AUC on the canonical view means a shortcut survived.

    python scripts/shortcut_checks.py      -> reports/diffusion/shortcuts.md, runs/diffusion/shortcuts.json
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, PredefinedSplit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hearsay import audio, config, manifest, metrics, splits  # noqa: E402
from hearsay.forensics.triage import signal_stats  # noqa: E402


def edge_silence(x, sr=config.SR, top_db=40.0):
    frame = int(0.02 * sr)
    n = len(x) // frame
    e = (x[: n * frame].astype(np.float64) / 32768).reshape(n, frame)
    db = 10 * np.log10(np.mean(e ** 2, 1) + 1e-12)
    act = np.where(db > db.max() - top_db)[0]
    return (act[0] * frame / sr, (n - 1 - act[-1]) * frame / sr) if len(act) else (n * frame / sr, 0.0)


def views(uid):
    x = audio.load_cached(uid)
    raw = signal_stats(x)
    raw["lead_sil"], raw["trail_sil"] = edge_silence(x)
    durs = audio.TestLikeDurations.from_cache()
    c = audio.canonical(x, audio.uid_rng(uid), durs)
    ci = np.clip(np.round(c * 32768), -32768, 32767).astype(np.int16)
    can = signal_stats(ci)
    can["lead_sil"], can["trail_sil"] = edge_silence(ci)
    return uid, raw, can


GROUPS = {
    "container (sr, codec, bitrate, tags)": ["native_sr", "codec_id", "bitrate", "bits", "n_tags", "has_encoder", "bytes_per_s"],
    "filesystem MAC times": ["mtime", "ctime", "birthtime"],
    "duration": ["decoded_duration"],
    "level (peak, RMS, clipping, DC)": ["peak", "rms_dbfs", "clip_frac", "dc_offset"],
    "edge silence + digital zeros": ["lead_sil", "trail_sil", "zero_frac", "zero_run_s"],
    "bandwidth (cutoff, HF drop, HF ratio)": ["cutoff_hz", "hf_7k_drop_db", "hf_ratio_4k"],
}
SIGNAL_COLS = ["decoded_duration", "peak", "rms_dbfs", "clip_frac", "dc_offset", "lead_sil", "trail_sil",
               "zero_frac", "zero_run_s", "cutoff_hz", "hf_7k_drop_db", "hf_ratio_4k"]


def cv_auc(X, y, folds):
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=15)
    p = cross_val_predict(clf, X, y, cv=PredefinedSplit(folds), method="predict_proba")[:, 1]
    return p


def main():
    man = manifest.load()
    tri = pd.read_parquet(config.CACHE / "triage.parquet").set_index("uid")
    lab = manifest.labeled(man)
    uids = list(man.uid)
    with ProcessPoolExecutor(9) as ex:
        res = list(ex.map(views, uids, chunksize=32))
    raw = pd.DataFrame([r for _, r, _ in res], index=[u for u, _, _ in res])
    can = pd.DataFrame([c for _, _, c in res], index=[u for u, _, _ in res])

    t = tri.loc[uids]
    meta = pd.DataFrame(index=uids)
    meta["native_sr"] = t.native_sr.to_numpy()
    meta["codec_id"] = t.codec.astype("category").cat.codes.to_numpy()
    meta["bitrate"] = t.bitrate.to_numpy()
    meta["bits"] = t.bits.to_numpy()
    meta["n_tags"] = t.n_tags.to_numpy()
    meta["has_encoder"] = (t.encoder.fillna("") != "").astype(int).to_numpy()
    meta["bytes_per_s"] = (t.size_bytes / t.decoded_duration).to_numpy()
    for c in ["mtime", "ctime", "birthtime"]:
        meta[c] = t[c].to_numpy()

    y = lab.label.to_numpy()
    f_text = splits.text_folds(lab)
    f_spk = splits.speaker_folds(lab)
    L = lab.uid
    rows = []
    for gname, cols in GROUPS.items():
        for view, src in [("raw", None), ("canonical", can)]:
            if view == "canonical" and not set(cols) <= set(SIGNAL_COLS):
                continue  # container / MAC facts don't change with the audio view
            X = (meta.join(raw) if src is None else meta.join(src)).loc[L, cols].to_numpy(dtype=float)
            for proto, f in [("text-CV", f_text), ("speaker-CV", f_spk)]:
                p = cv_auc(X, y, f)
                m = metrics.summary(y, p)
                rows.append(dict(group=gname, view=view, protocol=proto, auc=round(m["auc"], 3), eer=round(m["eer"], 3)))
    X = can.loc[L, SIGNAL_COLS].to_numpy(dtype=float)
    p = cv_auc(X, y, f_spk)
    rows.append(dict(group="ALL signal-trivial features", view="canonical", protocol="speaker-CV",
                     auc=round(metrics.summary(y, p)["auc"], 3), eer=round(metrics.summary(y, p)["eer"], 3)))
    res_df = pd.DataFrame(rows)

    # descriptive tables by generator, plus the unlabeled test set
    by = man.set_index("uid")[["generator"]].join(t[["native_sr", "codec", "encoder"]]).join(raw[SIGNAL_COLS])
    desc = by.groupby("generator").agg(
        native_sr=("native_sr", lambda s: "/".join(map(str, sorted(s.unique())))),
        codec=("codec", lambda s: "/".join(sorted(s.dropna().unique()))),
        encoder=("encoder", lambda s: "/".join(sorted({e or "-" for e in s}))[:30]),
        dur=("decoded_duration", "median"), peak=("peak", "median"), rms_dbfs=("rms_dbfs", "median"),
        lead_sil=("lead_sil", "median"), trail_sil=("trail_sil", "median"),
        zero_run_s=("zero_run_s", "median"), hf_7k_drop_db=("hf_7k_drop_db", "median"), cutoff_hz=("cutoff_hz", "median"),
    ).round(3)
    can_desc = man.set_index("uid")[["generator"]].join(can[SIGNAL_COLS]).groupby("generator")[
        ["decoded_duration", "peak", "rms_dbfs", "lead_sil", "trail_sil", "zero_run_s", "hf_7k_drop_db", "cutoff_hz"]].median().round(3)

    out = Path("reports/diffusion"); out.mkdir(parents=True, exist_ok=True)
    config.RUNS.mkdir(parents=True, exist_ok=True)
    res_df.to_json(config.RUNS / "shortcuts.json", orient="records", indent=1)
    with open(out / "shortcuts.md", "w") as fh:
        fh.write("# Shortcut and leakage checks (plan section 13)\n\n")
        fh.write("Grouped 5-fold CV AUC of a small gradient-boosting model on non-content features. "
                 "0.5 = no information. *raw* = decoded clip as delivered; *canonical* = the view every "
                 "model in this track sees (trim, test-like crop, 7 kHz low-pass, peak-norm, dither).\n\n")
        fh.write(res_df.to_markdown(index=False) + "\n\n## Raw clips by generator (medians)\n\n")
        fh.write(desc.to_markdown() + "\n\n## Canonical view by generator (medians)\n\n")
        fh.write(can_desc.to_markdown() + "\n")
    raw.to_parquet(config.CACHE / "signal_raw.parquet")
    can.to_parquet(config.CACHE / "signal_canonical.parquet")
    print(res_df.to_string(index=False))
    print(desc.to_string())
    print(can_desc.to_string())


if __name__ == "__main__":
    main()
