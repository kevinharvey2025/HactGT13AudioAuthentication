"""Build DSP-track manifests with leakage-safe dev/holdout splits.

Sources (read-only):
  data/DiffSSD/metadata.csv + audio       kharvey33/DiffSSD (242 real LJSpeech @16 kHz + fakes)
  data/external/LJSpeech-1.1              original 22.05 kHz LJSpeech (public domain)
  data/external/librispeech_10spk         LibriSpeech train-clean-360 utterances of the 10
                                          DiffSSD voice-clone reference speakers (CC BY 4.0)
  data/hearsay_test + score template      unlabeled HEARSAY test set

Split design (deterministic hashes of seed + group id, independent of row order):
  - 20% of DiffSSD sentence ids are held out; every fake with a held-out sentence
    id is in holdout, so no sentence text crosses splits.
  - 2 of the 10 clone speakers are held out entirely: their LibriSpeech reals and
    all fakes cloning them (sampled only from held-out sentences) are holdout.
  - LJSpeech reals are held out by chapter (LJ001..LJ050), 20%.
  - LibriSpeech reals of the other speakers are held out by chapter, 20%.
Fakes are sampled per generator with the same sentence ids across generators
(content-paired): LJ-voice TTS share one id set; clone generators share one id
set per speaker. OpenVoiceV2 style cycles deterministically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LJ_VOICE = ["diffgan_tts", "grad_tts", "pro_diff", "wavegrad2"]
CLONE = ["elevenlabs", "openvoicev2", "playht", "unit_speech", "xtts_v2", "your_tts"]
STYLES = ["en-au", "en-br", "en-default", "en-india", "en-us"]


def u01(seed: int, key: str) -> float:
    h = hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()
    return int(h[:15], 16) / float(16 ** 15)


def rel(path: Path, base: Path) -> str:
    return os.path.relpath(path, base)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "runs/dsp/manifests"))
    ap.add_argument("--seed", type=int, default=20260926)
    ap.add_argument("--fakes-per-generator", type=int, default=600)
    ap.add_argument("--ljspeech-extra", type=int, default=1000)
    ap.add_argument("--holdout-frac", type=float, default=0.2)
    ap.add_argument("--heldout-speakers", type=int, default=2)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seed, frac = args.seed, args.holdout_frac
    rng = np.random.default_rng(seed)

    ds_root = ROOT / "data/DiffSSD"
    md = pd.read_csv(ds_root / "metadata.csv", dtype=str)
    rows = []

    # --- held-out sets -----------------------------------------------------
    sent_held = {s for s in range(5000) if u01(seed, f"S{s}") < frac}
    clone_speakers = sorted(md.loc[md.generator.isin(CLONE), "speaker"].unique(), key=int)
    spk_held = set(sorted(clone_speakers, key=lambda s: u01(seed, f"P{s}"))[: args.heldout_speakers])

    def lj_chapter(uid: str) -> str:
        return uid.split("-")[0]

    # --- DiffSSD real ------------------------------------------------------
    for _, r in md[md.label == "real"].iterrows():
        ch = lj_chapter(r.utterance_id)
        rows.append(dict(path=rel(ds_root / r.file_name, out), label="real",
                         split="holdout" if u01(seed, f"LJ{ch}") < frac else "dev",
                         group_id=f"LJ:{ch}", speaker_id="LJSpeech",
                         source_id="diffssd_real_ljspeech_16k", attack_type="bonafide",
                         dataset="diffssd", family="real_lj", sentence_id="", style="",
                         item_id=r.utterance_id))

    # --- DiffSSD fakes -----------------------------------------------------
    fakes = md[md.label == "fake"].copy()
    fakes["sid"] = fakes.sentence_id.astype(float).astype(int)
    by_key = {(r.generator, r.speaker, r.sid, (r.style if isinstance(r.style, str) else "")): r.file_name
              for r in fakes.itertuples()}
    n = args.fakes_per_generator
    lj_ids = rng.choice(5000, size=n, replace=False)
    for gen in LJ_VOICE:
        for sid in lj_ids:
            fn = by_key[(gen, "LJSpeech", int(sid), "")]
            rows.append(dict(path=rel(ds_root / fn, out), label="fake",
                             split="holdout" if int(sid) in sent_held else "dev",
                             group_id=f"S:{int(sid)}", speaker_id="LJSpeech",
                             source_id=f"diffssd_{gen}", attack_type=gen, dataset="diffssd",
                             family="lj_voice_tts", sentence_id=str(int(sid)), style="",
                             item_id=f"{gen}:{int(sid)}"))
    per_spk = n // len(clone_speakers)
    held_pool = np.array(sorted(s for s in sent_held if s < 500))
    for spk in clone_speakers:
        pool = held_pool if spk in spk_held else np.arange(500)
        sids = rng.choice(pool, size=min(per_spk, pool.size), replace=False)
        for gen in CLONE:
            for sid in sids:
                style = STYLES[int(u01(seed, f"style{spk}:{sid}") * len(STYLES))] if gen == "openvoicev2" else ""
                fn = by_key[(gen, spk, int(sid), style)]
                rows.append(dict(path=rel(ds_root / fn, out), label="fake",
                                 split="holdout" if (int(sid) in sent_held or spk in spk_held) else "dev",
                                 group_id=f"S:{int(sid)}", speaker_id=f"LS{spk}",
                                 source_id=f"diffssd_{gen}", attack_type=gen, dataset="diffssd",
                                 family="voice_clone", sentence_id=str(int(sid)), style=style,
                                 item_id=f"{gen}:{spk}:{int(sid)}:{style}"))

    # --- external LJSpeech originals (22.05 kHz) ---------------------------
    lj_root = ROOT / "data/external/LJSpeech-1.1"
    lj_md = pd.read_csv(lj_root / "metadata.csv", sep="|", header=None, quoting=3, dtype=str)
    in_diffssd = set(md.loc[md.label == "real", "utterance_id"])
    cand = sorted(set(lj_md[0]) - in_diffssd)
    pick = rng.choice(len(cand), size=min(args.ljspeech_extra, len(cand)), replace=False)
    for i in sorted(pick):
        uid = cand[i]
        ch = lj_chapter(uid)
        rows.append(dict(path=rel(lj_root / "wavs" / f"{uid}.wav", out), label="real",
                         split="holdout" if u01(seed, f"LJ{ch}") < frac else "dev",
                         group_id=f"LJ:{ch}", speaker_id="LJSpeech",
                         source_id="ljspeech11_original_22k", attack_type="bonafide",
                         dataset="ljspeech11", family="real_lj", sentence_id="", style="",
                         item_id=uid))

    # --- external LibriSpeech (10 clone reference speakers) -----------------
    ls_root = ROOT / "data/external/librispeech_10spk"
    ls = pd.read_csv(ls_root / "index.csv", dtype=str)
    for r in ls.itertuples():
        held = r.speaker in spk_held or u01(seed, f"LS{r.speaker}-{r.chapter}") < frac
        rows.append(dict(path=rel(ls_root / r.file, out), label="real",
                         split="holdout" if held else "dev",
                         group_id=f"LS:{r.speaker}-{r.chapter}", speaker_id=f"LS{r.speaker}",
                         source_id="librispeech_train_clean_360", attack_type="bonafide",
                         dataset="librispeech", family="real_librispeech", sentence_id="",
                         style="", item_id=r.utt_id))

    pool = pd.DataFrame(rows)
    pool["speaker_heldout"] = pool.speaker_id.str[2:].isin(spk_held) & pool.speaker_id.str.startswith("LS")
    pool.to_csv(out / "pool.tsv", sep="\t", index=False)

    # --- leakage checks ----------------------------------------------------
    checks = {}
    for col in ("group_id",):
        splits_per = pool.groupby(col).split.nunique()
        checks[f"{col}_in_both_splits"] = int((splits_per > 1).sum())
    fk = pool[pool.label == "fake"]
    s_per = fk.groupby("sentence_id").split.nunique()
    checks["sentence_ids_in_both_splits"] = int((s_per > 1).sum())
    dev_spk = set(pool.loc[pool.split == "dev", "speaker_id"])
    checks["heldout_speakers_in_dev"] = sorted(f"LS{s}" for s in spk_held if f"LS{s}" in dev_spk)
    assert checks["group_id_in_both_splits"] == 0, checks
    assert checks["sentence_ids_in_both_splits"] == 0, checks
    assert not checks["heldout_speakers_in_dev"], checks

    # --- test manifest -----------------------------------------------------
    test_dir = ROOT / "data/hearsay_test"
    test_files = sorted(p for p in test_dir.iterdir() if p.suffix.lower() == ".wav")
    tdf = pd.DataFrame({"path": [rel(p, out) for p in test_files]})
    tdf["split"] = "test"
    tdf.to_csv(out / "hearsay_test.tsv", sep="\t", index=False)

    summary = {
        "seed": seed, "holdout_frac": frac, "heldout_speakers": sorted(spk_held, key=int),
        "n_heldout_sentence_ids": len(sent_held), "checks": checks,
        "counts": pool.groupby(["split", "label", "family"]).size().rename("n").reset_index()
                      .to_dict(orient="records"),
        "per_generator": pool.groupby(["attack_type", "split"]).size().unstack(fill_value=0)
                             .to_dict(orient="index"),
        "n_test_files": len(test_files),
    }
    (out / "split_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("heldout_speakers", "checks", "n_test_files")}, indent=1))
    print(pool.groupby(["split", "family", "label"]).size().to_string())


if __name__ == "__main__":
    main()
