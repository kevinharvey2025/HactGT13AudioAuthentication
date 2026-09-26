"""Access to cached pooled SSL embeddings (cache/emb/<model>/pooled.npy + index.parquet)."""
import numpy as np
import pandas as pd

from . import config


class EmbeddingStore:
    def __init__(self, model="wavlm_base_plus"):
        root = config.CACHE / "emb" / model
        self.index = pd.read_parquet(root / "index.parquet")
        self.arr = np.load(root / "pooled.npy", mmap_mode="r")      # [N, L, 2, D] float16
        self.n_layers, self.dim = self.arr.shape[1], self.arr.shape[3]
        self.row = {(u, v): i for i, (u, v) in enumerate(zip(self.index.uid, self.index.view))}

    def rows(self, uids, view=0):
        return np.array([self.row[(u, view)] for u in uids])

    def get(self, rows, layers=None, stats=("mean", "std")):
        """-> float32 [n, len(layers) * len(stats) * D] (layer-major)."""
        layers = list(range(self.n_layers)) if layers is None else list(layers)
        s = [{"mean": 0, "std": 1}[k] for k in stats]
        x = np.asarray(self.arr[np.sort(rows)][:, layers][:, :, s], dtype=np.float32)
        order = np.argsort(np.argsort(rows))  # restore caller's row order after the sorted mmap read
        return x[order].reshape(len(rows), -1)

    def layered(self, rows):
        """-> float32 [n, L, 2*D] for heads that learn their own layer weighting."""
        x = np.asarray(self.arr[np.sort(rows)], dtype=np.float32)[np.argsort(np.argsort(rows))]
        return x.reshape(len(rows), self.n_layers, -1)
