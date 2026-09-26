"""Phase 0: manifest -> decoded 16 kHz cache -> T0 triage table -> test duration distribution.

    python scripts/prepare_data.py [--workers 8] [--allow-missing] [--no-test]

Writes cache/manifest.parquet, cache/wav16k/<uid>.wav, cache/triage.parquet, cache/test_durations.npy.
Needs the full DiffSSD and the NSA test set (HANDOFF_DIFFUSION.md section 11). --allow-missing drops
DiffSSD files that aren't on disk (e.g. the incomplete HF copy); --no-test builds a training-only
cache and takes the crop-length distribution from configs/test_durations.txt.
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hearsay import audio, config, manifest  # noqa: E402
from hearsay.forensics import triage  # noqa: E402


def work(row):
    uid, path = row
    out = audio.cache_path(uid)
    try:
        if out.exists():
            x = sf.read(out, dtype="int16")[0]
        else:
            x = audio.decode(path)
            out.parent.mkdir(parents=True, exist_ok=True)
            sf.write(out, x, config.SR, subtype="PCM_16")
        t = triage.triage(path, x)
        t.update(uid=uid, decode_ok=True)
    except Exception as e:  # keep going; a decode failure is itself a triage finding
        t = dict(uid=uid, decode_ok=False, error=repr(e)[:200])
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--allow-missing", action="store_true", help="drop DiffSSD files that are not on disk")
    ap.add_argument("--no-test", action="store_true", help="build without the NSA test set")
    args = ap.parse_args()
    config.CACHE.mkdir(parents=True, exist_ok=True)

    man = manifest.build(allow_missing=args.allow_missing, require_test=not args.no_test)
    man.to_parquet(config.CACHE / "manifest.parquet")
    print(man.groupby(["family", "generator"]).size().to_string())

    with ProcessPoolExecutor(args.workers) as ex:
        rows = list(tqdm(ex.map(work, zip(man.uid, man.path), chunksize=16), total=len(man)))
    tri = pd.DataFrame(rows)
    tri.to_parquet(config.CACHE / "triage.parquet")
    print("decode failures:", int((~tri.decode_ok).sum()))

    test = tri[tri.uid.str.startswith("test/")]
    if len(test):
        d = test.decoded_duration.to_numpy()
    else:  # no test set on this machine: use the committed distribution (1,671 NSA sample-test clips)
        d = np.loadtxt(config.REPO / "configs" / "test_durations.txt")
    np.save(config.CACHE / "test_durations.npy", d)
    print("test durations: median %.2f s, range %.2f-%.2f s" % (np.median(d), d.min(), d.max()))


if __name__ == "__main__":
    main()
