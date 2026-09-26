"""Phase 0: manifest -> decoded 16 kHz cache -> T0 triage table -> test duration distribution.

    python scripts/prepare_data.py [--workers 8]

Writes cache/manifest.parquet, cache/wav16k/<uid>.wav, cache/triage.parquet, cache/test_durations.npy.
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
    args = ap.parse_args()
    config.CACHE.mkdir(parents=True, exist_ok=True)

    man = manifest.build()
    man.to_parquet(config.CACHE / "manifest.parquet")
    print(man.groupby(["family", "generator"]).size().to_string())

    with ProcessPoolExecutor(args.workers) as ex:
        rows = list(tqdm(ex.map(work, zip(man.uid, man.path), chunksize=16), total=len(man)))
    tri = pd.DataFrame(rows)
    tri.to_parquet(config.CACHE / "triage.parquet")
    print("decode failures:", int((~tri.decode_ok).sum()))

    test = tri[tri.uid.str.startswith("test/")]
    np.save(config.CACHE / "test_durations.npy", test.decoded_duration.to_numpy())
    print("test durations: median %.2f s, range %.2f-%.2f s" % (
        test.decoded_duration.median(), test.decoded_duration.min(), test.decoded_duration.max()))


if __name__ == "__main__":
    main()
