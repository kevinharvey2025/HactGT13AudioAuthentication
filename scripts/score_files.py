"""Zero-shot AntiDeepfake scores for arbitrary audio files (no manifest needed).

    python scripts/score_files.py --models adf_mms_300m adf_xlsr_2b \
        --glob 'data/hearsay_test/*.wav' --glob 'data/DiffSSD/real_speech/*.wav' --out runs/diffusion/zeroshot/x.parquet

Each file is decoded to 16 kHz mono (ffmpeg, as the test set was made). --view raw feeds it as is (the
model card's recipe); --view canonical applies the track's canonical view (trim, 7 kHz low-pass,
peak-normalize, dither). Output: one row per file and model with the logits [fake, real] and
synthetic_logit = fake - real (higher = synthetic).
"""
import argparse
import glob
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hearsay import antideepfake, audio, ssl  # noqa: E402


def load(path, view):
    x = audio.decode(path)
    if view == "canonical":
        return audio.canonical(x, audio.uid_rng(Path(path).name), durations=None)
    return audio.to_float(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["adf_mms_300m"])
    ap.add_argument("--glob", action="append", required=True)
    ap.add_argument("--view", choices=["raw", "canonical"], default="raw")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    files = sorted({f for g in args.glob for f in glob.glob(g)})
    print(f"{len(files)} files, view {args.view}", flush=True)
    with ThreadPoolExecutor(16) as ex:
        wavs = list(ex.map(lambda f: load(f, args.view), files))
    order = np.argsort([len(w) for w in wavs])  # similar lengths per batch; quantize() pads within 0.25 s
    rows = []
    for key in args.models:
        enc = ssl.SSLEncoder(key, device=torch.device("cuda"))
        out = np.zeros((len(files), 2), np.float32)
        with torch.inference_mode():
            for i in range(0, len(order), args.batch):
                idx = order[i: i + args.batch]
                n = min(len(ssl.quantize(wavs[j])) for j in idx)          # crop the batch to its shortest clip
                x = torch.from_numpy(np.stack([ssl.quantize(wavs[j])[:n] for j in idx])).cuda()
                h = enc.model(antideepfake.standardize(x)).last_hidden_state.mean(1)
                out[idx] = enc.head(h).float().cpu().numpy()
        rows.append(pd.DataFrame(dict(model=key, file=[Path(f).name for f in files], path=files,
                                      dur=[len(w) / 16000 for w in wavs], logit_fake=out[:, 0], logit_real=out[:, 1],
                                      synthetic_logit=out[:, 0] - out[:, 1])))
        print(key, "done", flush=True)
        del enc
        torch.cuda.empty_cache()
    df = pd.concat(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out)
    print(df.groupby("model").synthetic_logit.describe().to_string())


if __name__ == "__main__":
    main()
