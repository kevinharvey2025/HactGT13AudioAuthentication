"""Cache pooled embeddings of an SSL encoder (optionally a fine-tuned detector) for every clip-view: the
concept-formation space (scripts/run_concepts.py).

    python scripts/extract_ssl.py --model wavlm_base_plus --views 3
    python scripts/extract_ssl.py --model adf_xlsr_1b --checkpoint runs/diffusion/ft/xlsr1b_ft/best.pt \
        --manifest manifest_pool --layers last --views 2 --name xlsr1b_ft_last

Labeled clips get views 0..V-1 (0 = clean canonical, >=1 = augmented); test clips get view 0.
Writes cache/emb/<name or model>/{pooled.npy [N, L, 2, D] float16, index.parquet}. Resumable.
--checkpoint loads fine-tuned weights (scripts/finetune_ssl.py best.pt: enc.* and head.*); --layers keeps
all layers, only the last one, or a comma-separated list.
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
    ap.add_argument("--manifest", default="manifest", help="cache/<name>.parquet (manifest or manifest_pool)")
    ap.add_argument("--checkpoint", default="", help="fine-tuned state dict (finetune_ssl.py best.pt)")
    ap.add_argument("--layers", default="all", help="all | last | comma-separated hidden-state indices")
    ap.add_argument("--name", default="", help="output directory name under cache/emb (default: --model)")
    args = ap.parse_args()

    man = manifest.load(args.manifest)
    out = config.CACHE / "emb" / (args.name or args.model)
    out.mkdir(parents=True, exist_ok=True)
    idx_path = out / "index.parquet"
    index = pd.read_parquet(idx_path) if idx_path.exists() else build_index(man, args.views)
    if "channel" not in index:
        index["channel"], index["params"], index["done"] = "", "", False

    enc = ssl.SSLEncoder(args.model)
    if args.checkpoint:
        sd = torch.load(args.checkpoint, map_location="cpu")
        enc.model.load_state_dict({k[4:]: v.float() for k, v in sd.items() if k.startswith("enc.")})
        if enc.head is not None:
            enc.head.load_state_dict({k[5:]: v.float() for k, v in sd.items() if k.startswith("head.")})
        print("loaded fine-tuned weights from", args.checkpoint)
    keep = (list(range(enc.n_layers)) if args.layers == "all" else [enc.n_layers - 1] if args.layers == "last"
            else [int(v) for v in args.layers.split(",")])
    json.dump(dict(layers=keep, model=args.model, checkpoint=args.checkpoint, manifest=args.manifest),
              open(out / "meta.json", "w"))
    shape = (len(index), len(keep), 2, enc.dim)
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
        emb = enc.pooled_batch([x for _, x, _ in items])[:, keep]
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
