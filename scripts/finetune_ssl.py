"""Fine-tune an anti-spoofing SSL detector (AntiDeepfake encoder + its [fake, real] linear head).

    python scripts/finetune_ssl.py --backbone adf_mms_300m --name mms300m \
        [--holdout-gens pro_diff,xtts_v2] [--epochs 4] [--lr 1e-5] [--batch 32] [--workers 16]
    python scripts/finetune_ssl.py --backbone adf_xlsr_1b --name xlsr1b_final --final --epochs 3  # frozen config, all data

Data: the pool (scripts/prepare_data.py --pool -> cache/manifest_pool.parquet, audio in cache/wav16k).
Training audio is the canonical view (hearsay/audio.py: trim, crop, [augment], 7 kHz low-pass, DC removal,
peak-normalize, dither) of a random crop whose length is drawn per batch from the test-duration
distribution (clipped to 3.0-4.5 s), with a random channel augmentation (hearsay/augment.py) with
probability --p-aug. Both classes get exactly the same processing. Items are drawn with fixed family
shares (--shares) so neither LJSpeech's voice nor the cloned LibriSpeech voices can stand in for the label.

RawBoost (hearsay/rawboost.py): --rawboost-algo 1-8 applies it to an item with probability --p-rawboost, after the
chain (if drawn) and before the low-pass; --rawboost-params overrides the reference parameters ("maxF=7000,SNRmin=5").
Its decision and draws come from a separate stream per item, so crop, chain and dither are the same with RawBoost on
or off, and with it off the batches are bit-identical to earlier runs. The applied rate per class and family is
logged every epoch.

Splits: splits.shared_split, shared by every track (its holdout = the DSP track's holdout rules: hashed
sentence ids, held-out clone speakers 2061/5448, LJ chapters, LibriSpeech chapters, extra-real speakers).
Training uses 'train' rows; 'val' selects the checkpoint (mean of clean and augmented minDCF, organizers'
costs); 'holdout' is reported every epoch but never used for any choice. --holdout-gens removes whole
generators from training (their clips stay in val/holdout, as unseen generators). --final trains on
every labeled row for exactly --epochs epochs (configuration frozen beforehand) and keeps the last epoch.
Epoch 0 evaluates the pretrained model before any update (zero-shot benchmark). Evaluation audio: view 0
(clean canonical) and view 1 (fixed augmentation); --eval-views 0,1,100[,101] adds the unseen-channel views
(hearsay/unseen.py), scored in a separate pass so views 0 and 1 are batched exactly as without them, and never used
for selection. Test clips view 0, uncropped. Scores are synthetic logits (logit_fake - logit_real).

Writes runs/diffusion/ft/<name>/{config.json, log.jsonl, <set>_epoch<k>.parquet, test_epoch<k>.parquet, best.pt};
the per-epoch parquets hold score_<view> for each evaluated view, and channel_<view> for the added ones.
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hearsay import antideepfake, audio, augment, config, metrics, provenance, rawboost, splits, views  # noqa: E402

SR = config.SR
DEFAULT_SHARES = "lj_voice=0.25,clone=0.25,real_lj=0.2,real_libri=0.15,real_extra=0.15"
RAWBOOST_STREAM = 0x5242   # "RB": an item's RawBoost stream is default_rng([seed, epoch, row, RAWBOOST_STREAM])


# ----------------------------------------------------------------------------- data

class TrainSet(Dataset):
    """Item (row, crop_samples) -> (float32 audio of exactly crop_samples, class index: 0 fake, 1 real, row,
    whether RawBoost was applied)."""

    def __init__(self, uids, labels, p_aug, seed, babble_uids, rawboost_algo=0, p_rawboost=0.0, rawboost_params=None):
        self.uids, self.labels, self.p_aug, self.seed = list(uids), np.asarray(labels), p_aug, seed
        self.babble_uids, self.babble, self.epoch = list(babble_uids), None, 0
        self.rb_algo, self.p_rb = rawboost_algo, p_rawboost
        self.rb_params = rawboost_params or rawboost.RawBoostParams()

    def __len__(self):
        return len(self.uids)

    def rawboost_rng(self, i):
        """The item's RawBoost generator if RawBoost applies to it, else None (depends on seed, epoch and row only)."""
        if not self.rb_algo or self.p_rb <= 0:
            return None
        r = np.random.default_rng([self.seed, self.epoch, i, RAWBOOST_STREAM])
        return r if r.random() < self.p_rb else None

    def __getitem__(self, item):
        i, n = item
        if self.babble is None:  # per worker; real speech only
            torch.set_num_threads(1)
            self.babble = [audio.load_cached(u) for u in self.babble_uids]
        rng = np.random.default_rng([self.seed, self.epoch, i])
        x = audio.load_cached(self.uids[i])
        chain, boost = rng.random() < self.p_aug, self.rawboost_rng(i)
        aug = None
        if chain or boost is not None:
            def aug(y, r):  # the channel chain, then RawBoost; never sees the label
                prm = {}
                if chain:
                    y, prm = augment.random_chain(y, r, babble_pool=self.babble)
                if boost is not None:
                    y, _ = rawboost.apply(y, boost, self.rb_algo, self.rb_params)
                return y, prm
        y = audio.canonical(x, rng, durations=audio.TestLikeDurations([n / SR]), aug=aug)
        if len(y) < n:  # rare: clip shorter than the crop after trimming
            y = np.pad(y, (0, n - len(y)))
        return torch.from_numpy(y[:n].astype(np.float32)), int(1 - self.labels[i]), i, boost is not None


class FamilyBatches(Sampler):
    """Batches drawn with fixed family shares; one crop length per batch (equal lengths, no padding)."""

    def __init__(self, families, shares, batch, n_batches, durations, seed):
        self.batch, self.n_batches, self.seed, self.epoch = batch, n_batches, seed, 0
        fam = np.asarray(families)
        present = {f: s for f, s in shares.items() if (fam == f).any()}
        tot = sum(present.values())
        self.groups = {f: np.where(fam == f)[0] for f in present}
        self.p = {f: s / tot for f, s in present.items()}
        self.d = np.clip(np.asarray(durations), 3.0, 4.5)

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        rng = np.random.default_rng([self.seed, self.epoch, 7])
        fams = list(self.groups)
        p = np.array([self.p[f] for f in fams])
        for _ in range(self.n_batches):
            n = int(round(float(rng.choice(self.d)) * SR)) // 320 * 320   # whole 20 ms encoder frames
            picks = rng.choice(len(fams), size=self.batch, p=p)
            yield [(int(rng.choice(self.groups[fams[k]])), n) for k in picks]


def collate(items):
    x, y, rows, boosted = zip(*items)
    return torch.stack(x), torch.tensor(y), np.array(rows), np.array(boosted)


class EvalSet(Dataset):
    """(uid, view, is_test) -> exactly the audio views.ViewMaker gives every other module, and its channel label."""

    def __init__(self, rows, babble_uids):
        self.rows, self.babble_uids, self.maker = rows, list(babble_uids), None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        if self.maker is None:
            torch.set_num_threads(1)
            self.maker = views.ViewMaker([audio.load_cached(u) for u in self.babble_uids])
        uid, view, is_test = self.rows[i]
        x, prm = self.maker(uid, view, is_test=is_test)
        return i, x.astype(np.float32), prm["channel"]


# ----------------------------------------------------------------------------- model

class Detector(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        if not backbone.startswith("adf_"):
            raise ValueError("only AntiDeepfake backbones carry a pretrained head; use adf_*")
        self.enc, self.head = antideepfake.load(backbone[len("adf_"):])

    def forward(self, x):                                  # x [B, T] float32 at 16 kHz
        h = self.enc(antideepfake.standardize(x)).last_hidden_state.mean(1)
        return self.head(h.float())                        # [B, 2] logits [fake, real]


@torch.inference_mode()
def score(model, ds, device, batch, workers, channels=None):
    """Synthetic logits for every item; batches of similar length cropped to their shortest clip. Fills `channels`
    (a list) with each item's channel label when given."""
    dl = DataLoader(ds, batch_size=None, shuffle=False, num_workers=workers, prefetch_factor=4)
    xs, ch = [None] * len(ds), [None] * len(ds)
    for i, x, c in dl:
        xs[i], ch[i] = (x.numpy() if torch.is_tensor(x) else x), c
    if channels is not None:
        channels[:] = ch
    order = np.argsort([len(x) for x in xs])
    out = np.zeros(len(xs), np.float32)
    model.eval()
    for k in range(0, len(order), batch):
        idx = order[k: k + batch]
        n = min(len(xs[j]) for j in idx) // 320 * 320
        x = torch.from_numpy(np.stack([xs[j][:n] for j in idx])).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = model(x).float()
        out[idx] = antideepfake.synthetic_logit(lg).cpu().numpy()
    return out


def dev_metrics(dev, s0, s1):
    y = dev.label.to_numpy()
    r = {}
    for name, s in [("clean", s0), ("aug", s1)]:
        m = metrics.summary(y, s)
        r.update({f"{name}_{k}": m[k] for k in ("auc", "eer", "min_dcf")})
        for g in sorted(dev.generator[dev.label == 1].unique()):
            mask = (y == 0) | (dev.generator == g).to_numpy()
            r[f"{name}_mindcf_{g}"] = metrics.min_dcf(y[mask], s[mask])
    r["select"] = (r["clean_min_dcf"] + r["aug_min_dcf"]) / 2
    return r


def view_metrics(dev, extra):
    """AUC, EER and minDCF of the added views ({name: scores}); reported, never used for selection."""
    y = dev.label.to_numpy()
    return {f"{name}_{k}": v for name, s in extra.items() for k, v in metrics.summary(y, s).items()
            if k in ("auc", "eer", "min_dcf")}


def rawboost_rates(tr, rows, boosted):
    """Share of the epoch's training items that got RawBoost: overall, per class and per family."""
    lab, fam = tr.label.to_numpy()[rows], tr.family.to_numpy()[rows]
    return dict(n=int(len(rows)), all=float(boosted.mean()), fake=float(boosted[lab == 1].mean()),
                real=float(boosted[lab == 0].mean()), family={f: float(boosted[fam == f].mean()) for f in sorted(set(fam))})


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="adf_mms_300m")
    ap.add_argument("--name", required=True)
    ap.add_argument("--final", action="store_true", help="train on every labeled row; keep the last epoch")
    ap.add_argument("--holdout-gens", default="", help="comma-separated generators never trained on (evaluation only)")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--steps-per-epoch", type=int, default=800)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-6)
    ap.add_argument("--head-lr", type=float, default=1e-4)
    ap.add_argument("--p-aug", type=float, default=0.6)
    ap.add_argument("--shares", default=DEFAULT_SHARES)
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--freeze-layers", type=int, default=0,
                    help="freeze the bottom K transformer layers (+ positional conv, feature projection); 2B needs ~24 on a 40 GB A100")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-eval", type=int, default=5000, help="subsample val/holdout clips (stratified by generator) for speed")
    ap.add_argument("--select", default="val,itw", help="eval sets whose mean (clean+aug)/2 minDCF picks the checkpoint")
    ap.add_argument("--wise-from", default="", help="fine-tuned best.pt to interpolate with the pretrained weights (WiSE-FT)")
    ap.add_argument("--wise-alpha", type=float, default=1.0, help="weight of the fine-tuned weights (0 = pretrained, 1 = fine-tuned)")
    ap.add_argument("--extra-fakes", default="", choices=["", "d6r"],
                    help="d6r: add copy-synthesis fakes (cache/manifest_d6r.parquet; train-split reals only)")
    ap.add_argument("--rawboost-algo", type=int, default=0, choices=range(9),
                    help="RawBoost (hearsay/rawboost.py): 0 off, 1 LnL, 2 ISD, 3 SSI, 4 1->2->3, 5 1->2, 6 1->3, 7 2->3, 8 1||2")
    ap.add_argument("--p-rawboost", type=float, default=0.0, help="probability that an item gets RawBoost")
    ap.add_argument("--rawboost-params", default="", help="overrides of the reference parameters, e.g. maxF=7000,SNRmin=5")
    ap.add_argument("--eval-views", default="0,1", help="views scored every epoch: 0,1 plus 100 (unseen), 101 (device)")
    args = ap.parse_args()
    rb_params = rawboost.RawBoostParams.parse(args.rawboost_params)
    eval_views = [int(v) for v in args.eval_views.split(",")]
    if eval_views[:2] != [0, 1] or any(v not in views.NAMES for v in eval_views):
        raise ValueError(f"--eval-views must start with 0,1 and use views {sorted(views.NAMES)}")
    extra_views = eval_views[2:]
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev_ = torch.device("cuda")
    out = config.RUNS / "ft" / args.name
    out.mkdir(parents=True, exist_ok=True)
    json.dump(dict(vars(args), code_version=provenance.code_version(),
                   rawboost=rb_params.to_json() if args.rawboost_algo and args.p_rawboost > 0 else None),
              open(out / "config.json", "w"), indent=1)
    log = open(out / "log.jsonl", "a")

    man = pd.read_parquet(config.CACHE / "manifest_pool.parquet")
    tri = pd.read_parquet(config.CACHE / "triage_pool.parquet")[["uid", "decode_ok", "decoded_duration"]]
    man = man.merge(tri, on="uid", how="left")
    if args.extra_fakes == "d6r":
        d6 = pd.read_parquet(config.CACHE / "manifest_d6r.parquet")
        man = pd.concat([man, d6[[c for c in d6.columns if c in man.columns or c in ("decode_ok", "decoded_duration")]]],
                        ignore_index=True)
    man = man[man.decode_ok.fillna(False)].reset_index(drop=True)
    lab = man[man.label >= 0].reset_index(drop=True)
    test = man[man.label < 0].reset_index(drop=True)
    lab["split"] = splits.shared_split(lab)
    held = [g for g in args.holdout_gens.split(",") if g]
    use = np.array(lab.split != "itw") if args.final else np.array(lab.split == "train")   # writable copies (pandas CoW)
    use &= ~lab.generator.isin(held).to_numpy()
    # training rows must allow a >= 3 s crop after edge-silence trimming
    tr = lab[use & (lab.decoded_duration >= 3.3).to_numpy()].reset_index(drop=True)

    def subsample(d):
        if len(d) <= args.max_eval:
            return d.reset_index(drop=True)
        return d.groupby("generator", group_keys=False).sample(frac=args.max_eval / len(d), random_state=0).reset_index(drop=True)
    evals = {"val": subsample(lab[lab.split == "val"]), "holdout": subsample(lab[lab.split == "holdout"])}
    if (lab.split == "itw").any():
        evals["itw"] = subsample(lab[lab.split == "itw"])
    if args.final:  # every row is trained on: only the test set is scored
        evals = {}
    if held and not args.final:  # unseen generators: their clips from every split, against all non-train reals
        unseen = lab[lab.generator.isin(held) | ((lab.label == 0) & (lab.split != "train"))]
        evals["unseen_gen"] = subsample(unseen)
    print(f"train {len(tr)} ({tr.groupby('family').size().to_dict()}); eval " +
          ", ".join(f"{k} {len(v)}" for k, v in evals.items()) + f"; test {len(test)}; held-out generators {held}", flush=True)
    babble = tr[tr.label == 0].sample(min(150, int((tr.label == 0).sum())), random_state=0).uid

    shares = {kv.split("=")[0]: float(kv.split("=")[1]) for kv in args.shares.split(",")}
    durations = audio.TestLikeDurations.from_cache().d
    train_ds = TrainSet(tr.uid, tr.label, args.p_aug, args.seed, babble, args.rawboost_algo, args.p_rawboost, rb_params)
    sampler = FamilyBatches(tr.family, shares, args.batch, args.steps_per_epoch, durations, args.seed)
    eval_ds = {k: EvalSet([(u, v, False) for v in (0, 1) for u in d.uid], babble) for k, d in evals.items()}
    extra_ds = {k: EvalSet([(u, v, False) for v in extra_views for u in d.uid], babble) for k, d in evals.items() if extra_views}
    test_ds = EvalSet([(u, 0, True) for u in test.uid], babble)

    model = Detector(args.backbone)
    if args.wise_from:  # WiSE-FT (Wortsman et al. 2022): theta = (1 - a) * pretrained + a * fine-tuned
        ft = torch.load(args.wise_from, map_location="cpu")
        zs = model.state_dict()
        model.load_state_dict({k: ((1 - args.wise_alpha) * v.float() + args.wise_alpha * ft[k].float()) if v.is_floating_point()
                               else ft[k] for k, v in zs.items()})
        print(f"WiSE-FT: alpha {args.wise_alpha} between pretrained and {args.wise_from}", flush=True)
    model = model.to(dev_)
    model.enc.feature_extractor._freeze_parameters()            # CNN front-end stays as pretrained
    if args.freeze_layers:
        frozen = [model.enc.feature_projection, model.enc.encoder.pos_conv_embed] + list(model.enc.encoder.layers[: args.freeze_layers])
        for mod in frozen:
            for prm in mod.parameters():
                prm.requires_grad_(False)
    if args.grad_ckpt:
        model.enc.gradient_checkpointing_enable()
    enc_params = [p for p in model.enc.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": args.lr}, {"params": model.head.parameters(), "lr": args.head_lr}],
                            weight_decay=0.01)
    total = max(1, args.epochs * args.steps_per_epoch)   # epochs 0 = zero-shot evaluation only
    warm = max(1, int(0.1 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    # class weights from the family shares actually used (fake = class 0)
    p_fake = sum(v for k, v in sampler.p.items() if k in ("lj_voice", "clone", "resynth"))
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([0.5 / p_fake, 0.5 / (1 - p_fake)], device=dev_))

    best, step = None, 0
    for ep in range(args.epochs + 1):
        train = {}
        if ep > 0:
            model.train()
            train_ds.epoch = sampler.epoch = ep
            dl = DataLoader(train_ds, batch_sampler=sampler, num_workers=args.workers, collate_fn=collate,
                            prefetch_factor=4, persistent_workers=False)
            t0, tot, n, rows, boosted = time.time(), 0.0, 0, [], []
            for x, y, i, b in dl:
                x, y = x.to(dev_, non_blocking=True), y.to(dev_, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = loss_fn(model(x).float(), y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                step += 1
                tot, n = tot + loss.item(), n + 1
                rows.append(i)
                boosted.append(b)
                if step % 100 == 0:
                    print(f"ep {ep} step {step} loss {tot / n:.4f} ({(time.time() - t0) / n:.2f} s/step)", flush=True)
            train = dict(train_loss=tot / n, s_per_step=(time.time() - t0) / n)
            if args.rawboost_algo and args.p_rawboost > 0:
                train["rawboost_applied"] = rawboost_rates(tr, np.concatenate(rows), np.concatenate(boosted))
        t0 = time.time()
        r = dict(epoch=ep, step=step, **train)
        for k, d in evals.items():
            ch = []
            s = score(model, eval_ds[k], dev_, 32, args.workers, channels=ch)
            s0, s1 = s[: len(d)], s[len(d):]
            r.update({f"{k}_{m}": v for m, v in dev_metrics(d, s0, s1).items()})
            cols = dict(uid=d.uid, generator=d.generator, family=d.family, label=d.label, split=d.split,
                        score_clean=s0, score_aug=s1, channel_aug=ch[len(d):])
            if extra_views:  # the added views, scored apart so views 0 and 1 are batched as without them
                ch = []
                se = score(model, extra_ds[k], dev_, 32, args.workers, channels=ch)
                extra = {views.NAMES[v]: se[j * len(d): (j + 1) * len(d)] for j, v in enumerate(extra_views)}
                r.update({f"{k}_{m}": v for m, v in view_metrics(d, extra).items()})
                for j, v in enumerate(extra_views):
                    cols[f"score_{views.NAMES[v]}"] = extra[views.NAMES[v]]
                    cols[f"channel_{views.NAMES[v]}"] = ch[j * len(d): (j + 1) * len(d)]
            pd.DataFrame(cols).to_parquet(out / f"{k}_epoch{ep}.parquet")
        st = score(model, test_ds, dev_, 16, args.workers)
        pt = 1 / (1 + np.exp(-st))
        r.update(test_frac_gt_0_5=float((pt > 0.5).mean()), test_logit_median=float(np.median(st)), eval_s=time.time() - t0)
        summary = {k: round(v, 4) for k, v in r.items() if isinstance(v, float) and ("mindcf_" not in k)}
        print(json.dumps(dict(epoch=ep, step=step, **summary)), flush=True)
        log.write(json.dumps(r) + "\n")
        log.flush()
        pd.DataFrame(dict(uid=test.uid, filename=test.filename, score=st)).to_parquet(out / f"test_epoch{ep}.parquet")
        sel = [k for k in args.select.split(",") if f"{k}_select" in r]
        r["select"] = float(np.mean([r[f"{k}_select"] for k in sel])) if sel else np.nan
        better = args.final or best is None or r["select"] < best["select"]
        if ep > 0 and better:
            best = r
            torch.save({k: v.to(torch.bfloat16) for k, v in model.state_dict().items()}, out / "best.pt")
            json.dump(best, open(out / "best.json", "w"), indent=1)
        elif ep == 0 and best is None and not args.final:
            best = r  # zero-shot is the bar a fine-tuned checkpoint must beat on val
            if args.wise_from:  # an interpolated model is a system of its own: keep its weights
                torch.save({k: v.to(torch.bfloat16) for k, v in model.state_dict().items()}, out / "best.pt")
                json.dump(best, open(out / "best.json", "w"), indent=1)
    print("best:", json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in best.items()}))


if __name__ == "__main__":
    main()
