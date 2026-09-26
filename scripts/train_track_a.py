"""Track A: SSL front-end + layer-weighted MLP head, evaluated the ways the plan asks for.

    python scripts/train_track_a.py --model wavlm_base_plus [--logo]

Protocols: text-grouped 5-fold CV and speaker-disjoint 5-fold CV (splits.py). Training uses every
view (clean + augmented) of the training folds; each held-out clip is scored on every view, so
view 0 gives clean results and views >= 1 give robustness results. Optional leave-one-generator-
out on the speaker folds. A final model trained on all labeled data scores the test set.

Writes runs/diffusion/track_a/<model>/{oof_<protocol>.parquet, logo.parquet, metrics.json,
layer_weights.json, test_scores.parquet, head.pt}.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hearsay import config, embeddings, heads, manifest, metrics, splits  # noqa: E402


def evaluate(lab, oof, name):
    """oof: [n_clips, n_views] scores. Returns flat metric rows."""
    y = lab.label.to_numpy()
    rows = []
    masks = {"all": np.ones(len(lab), bool), **splits.subset_masks(lab)}
    for mname, m in masks.items():
        for vname, cols in [("clean", [0]), ("augmented", list(range(1, oof.shape[1])))]:
            s = oof[m][:, cols].reshape(-1)
            yy = np.repeat(y[m], len(cols))
            ok = np.isfinite(s)
            r = metrics.summary(yy[ok], s[ok], n_boot=200 if vname == "clean" else 0)
            rows.append(dict(protocol=name, subset=mname, view=vname, **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}))
    # each generator against all bona fide in the same folds (clean view)
    real = y == 0
    for g in config.GENERATORS:
        m = real | (lab.generator == g).to_numpy()
        r = metrics.summary(y[m], oof[m, 0])
        rows.append(dict(protocol=name, subset=f"gen:{g}", view="clean", auc=round(r["auc"], 4), eer=round(r["eer"], 4), n=r["n"]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="wavlm_base_plus")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--logo", action="store_true")
    args = ap.parse_args()

    out = config.RUNS / "track_a" / args.model
    out.mkdir(parents=True, exist_ok=True)
    man = manifest.load()
    lab = manifest.labeled(man)
    E = embeddings.EmbeddingStore(args.model)
    n_views = int(E.index[~E.index.is_test].view.max()) + 1
    y = lab.label.to_numpy()

    # all labeled views in memory as float16 [n_clips, n_views, L, 2D]
    X = np.stack([E.arr[E.rows(lab.uid, v)] for v in range(n_views)], axis=1)
    X = X.reshape(len(lab), n_views, E.n_layers, -1)
    print("features", X.shape, X.dtype, f"{X.nbytes / 1e9:.1f} GB")

    all_rows, lw = [], {}
    for proto, folds in [("text", splits.text_folds(lab)), ("speaker", splits.speaker_folds(lab))]:
        oof = np.full((len(lab), n_views), np.nan, np.float32)
        for k in range(folds.max() + 1):
            tr, te = np.where(folds != k)[0], np.where(folds == k)[0]
            model = heads.train_layer_mlp(X[tr].reshape(-1, E.n_layers, X.shape[-1]), np.repeat(y[tr], n_views),
                                          epochs=args.epochs, seed=k)
            for v in range(n_views):
                oof[te, v] = heads.predict(model, X[te, v])
            lw[f"{proto}_fold{k}"] = heads.layer_weights(model).round(4).tolist()
            print(proto, k, "clean AUC %.4f" % metrics.summary(y[te], oof[te, 0])["auc"], flush=True)
        metrics.check_polarity(y, oof[:, 0], f"track_a/{proto}")
        pd.DataFrame({"uid": np.repeat(lab.uid.to_numpy(), n_views), "view": np.tile(np.arange(n_views), len(lab)),
                      "score": oof.reshape(-1), "fold": np.repeat(folds, n_views)}).to_parquet(out / f"oof_{proto}.parquet")
        all_rows += evaluate(lab, oof, proto)

    if args.logo:
        folds = splits.speaker_folds(lab)
        logo = []
        for g, k, tr, te in splits.logo_splits(lab, folds):
            model = heads.train_layer_mlp(X[tr].reshape(-1, E.n_layers, X.shape[-1]), np.repeat(y[tr], n_views),
                                          epochs=args.epochs, seed=k)
            s = heads.predict(model, X[te, 0])
            logo.append(pd.DataFrame(dict(generator=g, fold=k, uid=lab.uid.to_numpy()[te], label=y[te], score=s)))
        logo = pd.concat(logo)
        logo.to_parquet(out / "logo.parquet")
        for g, d in logo.groupby("generator"):
            r = metrics.summary(d.label, d.score)
            all_rows.append(dict(protocol="logo_speaker", subset=f"heldout:{g}", view="clean", auc=round(r["auc"], 4), eer=round(r["eer"], 4), n=r["n"]))
            print("LOGO", g, round(r["auc"], 4), round(r["eer"], 4), flush=True)

    # final head on everything -> test scores
    model = heads.train_layer_mlp(X.reshape(-1, E.n_layers, X.shape[-1]), np.repeat(y, n_views), epochs=args.epochs, seed=0)
    test = man[man.label < 0]
    Xt = E.arr[E.rows(test.uid, 0)].reshape(len(test), E.n_layers, -1)
    pd.DataFrame(dict(uid=test.uid, filename=test.filename, score=heads.predict(model, Xt))).to_parquet(out / "test_scores.parquet")
    torch.save(model.cpu(), out / "head.pt")
    lw["final"] = heads.layer_weights(model).round(4).tolist()

    res = pd.DataFrame(all_rows)
    res.to_json(out / "metrics.json", orient="records", indent=1)
    json.dump(lw, open(out / "layer_weights.json", "w"), indent=1)
    pd.set_option("display.width", 220)
    print(res.to_string(index=False))
    print("final layer weights:", lw["final"])


if __name__ == "__main__":
    main()
