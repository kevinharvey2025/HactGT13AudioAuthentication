"""Track D6-R: copy-synthesis fakes — real training clips resynthesized by pretrained mel vocoders.

    python scripts/run_d6r.py --vocoder hifigan_16k [--n 3000] [--shard 0 --n-shards 1]
    python scripts/run_d6r.py --merge            # -> cache/manifest_d6r.parquet (all vocoders found on disk)

Why: vocoded real speech (real content and speaker, vocoder artifacts only) is the Track D option with the
strongest evidence (Wang & Yamagishi 2023: ITW EER 26.65 -> 7.55%; plans/diffusion_cf_prompt.md A.3). Only
mel vocoders are used (hifigan_16k, hifigan_lj, diffwave_lj = a diffusion vocoder, vocos); neural codecs are
left out because labeling codec-resynthesized real speech as spoof is contested and could turn codec'd real
test clips into false alarms. Sources: real clips of the shared split's *train* part only (so val, holdout
and In-the-Wild stay identical with or without D6-R), stratified over LJSpeech, the cloned LibriSpeech
speakers and the extra LibriSpeech speakers. Each copy keeps its original's speaker and text group (paired).

Output audio: cache/wav16k/resynth_<vocoder>/<original uid>.wav (16 kHz PCM16, delay-aligned to the input),
so audio.load_cached(uid) works with uid = "resynth_<vocoder>/<original uid>".
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hearsay import audio, config, splits  # noqa: E402
from hearsay.diffusion.resynth import Resynthesizer, align  # noqa: E402

VOCODERS = ("hifigan_16k", "hifigan_lj", "diffwave_lj", "vocos")
SOURCES = {"real_lj": 0.5, "real_libri": 0.2, "real_extra": 0.3}


def sources(n, seed=0):
    man = pd.read_parquet(config.CACHE / "manifest_pool.parquet")
    tri = pd.read_parquet(config.CACHE / "triage_pool.parquet")[["uid", "decode_ok", "decoded_duration"]]
    lab = man[man.label >= 0].merge(tri, on="uid", how="left").reset_index(drop=True)
    lab["split"] = splits.shared_split(lab)
    real = lab[(lab.label == 0) & (lab.split == "train") & lab.decode_ok.fillna(False) &
               (lab.decoded_duration >= 3.3) & lab.family.isin(list(SOURCES))]
    parts = [real[real.family == f].sample(min(int(n * s), int((real.family == f).sum())), random_state=seed)
             for f, s in SOURCES.items()]
    return pd.concat(parts).sort_values("uid").reset_index(drop=True)


def generate(a):
    src = sources(a.n)
    src = src.iloc[a.shard::a.n_shards]
    torch.backends.cuda.matmul.allow_tf32 = True
    model = Resynthesizer(a.vocoder, device=torch.device("cuda"))
    done = 0
    for r in src.itertuples():
        uid = f"resynth_{a.vocoder}/{r.uid}"
        out = audio.cache_path(uid)
        if out.exists():
            continue
        x = audio.to_float(audio.load_cached(r.uid))
        y = model.reconstruct(x)
        _, ya, _ = align(x, y)                       # vocoders add delay; keep the original's timing
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(out, np.clip(ya, -1, 1), config.SR, subtype="PCM_16")
        done += 1
        if done % 200 == 0:
            print(f"{a.vocoder}: {done}/{len(src)}", flush=True)
    print(f"{a.vocoder}: wrote {done} new files ({len(src)} in this shard)")


def merge(a):
    src = sources(a.n)
    rows = []
    for v in VOCODERS:
        for r in src.itertuples():
            uid = f"resynth_{v}/{r.uid}"
            p = audio.cache_path(uid)
            if p.exists():
                rows.append(dict(uid=uid, path=str(p), label=1, generator=f"resynth_{v}", family="resynth",
                                 speaker=r.speaker, text_group=r.text_group, source=f"d6r:{r.family}",
                                 orig_uid=r.uid, decode_ok=True, decoded_duration=sf.info(p).duration))
    df = pd.DataFrame(rows)
    df.to_parquet(config.CACHE / "manifest_d6r.parquet")
    print(df.groupby(["generator", "source"]).size().to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocoder", choices=VOCODERS)
    ap.add_argument("--n", type=int, default=3000, help="real source clips (the same set for every vocoder)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    merge(a) if a.merge else generate(a)


if __name__ == "__main__":
    main()
