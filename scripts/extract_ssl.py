"""Cache pooled SSL embeddings for every clip-view (Track A front-end; D2/D5 embedding space).

    python scripts/extract_ssl.py --model wavlm_base_plus --views 3

Labeled clips get views 0..V-1 (0 = clean canonical, >=1 = augmented); test clips get view 0.
Writes cache/emb/<model>/{pooled.npy [N, L, 2, D] float16, index.parquet}. Resumable.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hearsay import config, manifest, ssl, views  # noqa: E402


class ViewDataset(Dataset):
    def __init__(self, index, man):
        self.index, self.man, self.maker = index, man, None

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        if self.maker is None:  # built lazily inside each worker
            torch.set_num_threads(1)
            self.maker = views.ViewMaker(views.babble_pool(self.man, n=150))
        r = self.index.iloc[i]
        x, params = self.maker(r.uid, int(r.view), is_test=bool(r.is_test))
        return i, x, json.dumps(params)


def build_index(man, n_views):
    rows = []
    for uid, label in zip(man.uid, man.label):
        for v in range(n_views if label >= 0 else 1):
            rows.append((uid, v, label < 0))
    return pd.DataFrame(rows, columns=["uid", "view", "is_test"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="wavlm_base_plus", choices=list(ssl.MODELS))
    ap.add_argument("--views", type=int, default=3)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    man = manifest.load()
    out = config.CACHE / "emb" / args.model
    out.mkdir(parents=True, exist_ok=True)
    idx_path = out / "index.parquet"
    index = pd.read_parquet(idx_path) if idx_path.exists() else build_index(man, args.views)
    if "channel" not in index:
        index["channel"], index["params"], index["done"] = "", "", False

    enc = ssl.SSLEncoder(args.model)
    shape = (len(index), enc.n_layers, 2, enc.dim)
    arr_path = out / "pooled.npy"
    arr = (np.lib.format.open_memmap(arr_path, mode="r+") if arr_path.exists()
           else np.lib.format.open_memmap(arr_path, mode="w+", dtype=np.float16, shape=shape))
    todo = np.where(~index.done.to_numpy())[0]
    print(f"{args.model}: {len(todo)}/{len(index)} clip-views to embed")

    ds = ViewDataset(index, man)
    dl = DataLoader(torch.utils.data.Subset(ds, todo), batch_size=None, num_workers=args.workers,
                    prefetch_factor=8, persistent_workers=False)
    done = index.done.to_numpy().copy()
    ch, pr = index.channel.to_numpy().copy(), index.params.to_numpy().copy()
    buckets = {}  # quantized length -> [(row, audio, params)]; equal lengths batch well on the GPU

    def run(items):
        emb = enc.pooled_batch([x for _, x, _ in items])
        for (i, _, params), e in zip(items, emb):
            arr[i] = e
            ch[i], pr[i], done[i] = json.loads(params)["channel"], params, True

    def checkpoint():
        arr.flush()
        index["channel"], index["params"], index["done"] = ch, pr, done
        index.to_parquet(idx_path)

    for n, (i, x, params) in enumerate(tqdm(dl, total=len(todo), mininterval=30)):
        x = x.numpy() if torch.is_tensor(x) else x
        b = buckets.setdefault(len(ssl.quantize(x)), [])
        b.append((i, x, params))
        if len(b) == args.batch:
            run(b)
            b.clear()
        if n % 4000 == 3999:
            checkpoint()
    for b in buckets.values():
        if b:
            run(b)
    checkpoint()
    print("done:", int(index.done.sum()), "/", len(index))


if __name__ == "__main__":
    main()
