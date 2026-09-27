"""Evaluation protocols over the labeled manifest (reset index; one row per clip).

- text_folds: 5-fold CV, stratified by generator, grouped by text (a sentence id never spans folds).
- speaker_folds: the cloned LibriSpeech speakers are split 2 per fold, so clone fakes AND the
  matching real speakers in a test fold were never seen in training. LJ rows (one speaker) keep
  their text folds. This is the closest proxy for unseen test speakers.
- logo: leave-one-generator-out on top of a fold assignment.
- subsets: speaker-matched comparisons, where voice identity cannot separate the classes.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from . import config

K = 5


def text_folds(lab, k=K, seed=0):
    f = np.full(len(lab), -1)
    sgk = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    for i, (_, te) in enumerate(sgk.split(lab, lab.generator, lab.text_group)):
        f[te] = i
    return f


def speaker_folds(lab, k=K, seed=0):
    f = text_folds(lab, k, seed)
    spk = sorted(lab.loc[lab.family.isin(["clone", "real_libri"]), "speaker"].unique())
    rng = np.random.default_rng(seed)
    order = list(rng.permutation(spk))
    spk_fold = {s: i % k for i, s in enumerate(order)}
    m = lab.family.isin(["clone", "real_libri"])
    f[m.to_numpy()] = lab.loc[m, "speaker"].map(spk_fold).to_numpy()
    # extra bona fide speakers (fine-tuning pool): whole speakers per fold too
    ex = lab.family == "real_extra"
    if ex.any():
        es = list(np.random.default_rng(seed + 1).permutation(sorted(lab.loc[ex, "speaker"].unique())))
        f[ex.to_numpy()] = lab.loc[ex, "speaker"].map({s: i % k for i, s in enumerate(es)}).to_numpy()
    return f


SHARED_SEED = 20260926   # scripts/dsp_build_manifests.py


def _u01(seed, key):
    import hashlib
    return int(hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()[:15], 16) / float(16 ** 15)


def shared_split(lab, seed=SHARED_SEED, frac=0.2, val_frac=0.15, n_heldout_speakers=2):
    """The split every track shares: 'holdout' (reserved evaluation), 'val' (selection, calibration,
    fusion training) and 'train'. The holdout rules are exactly those of scripts/dsp_build_manifests.py,
    so the DSP pool's holdout rows are holdout here too:
      fakes: sentence id hashed into the held-out 20%, or a clone of a held-out speaker (2061, 5448);
      LJSpeech (organizer + LJSpeech-1.1): LJ chapter hashed; LibriSpeech speakers: held-out speaker or
      (speaker, chapter) hashed; extra reals: speaker hashed. 'val' takes a further val_frac of the
      remaining groups with an independent hash. In-the-Wild rows form their own split 'itw' (never trained on)."""
    sent_held = {s for s in range(5000) if _u01(seed, f"S{s}") < frac}
    clone_spk = sorted(lab.loc[lab.family == "clone", "speaker"].unique(), key=int)
    spk_held = set(sorted(clone_spk, key=lambda s: _u01(seed, f"P{s}"))[:n_heldout_speakers])
    out, key = [], []
    for r in lab.itertuples():
        if r.family in ("lj_voice", "clone"):
            sid = int(r.sentence_id)
            held = sid in sent_held or (r.family == "clone" and r.speaker in spk_held)
            k = f"S{sid}"
        elif r.family == "real_lj":
            ch = r.uid.split("/")[-1].split("-")[0]
            held, k = _u01(seed, f"LJ{ch}") < frac, f"LJ{ch}"
        elif r.family == "real_libri":
            ch = r.uid.split("/")[-1].split("-")[1]
            held = r.speaker in spk_held or _u01(seed, f"LS{r.speaker}-{ch}") < frac
            k = f"LS{r.speaker}-{ch}"
        elif r.family == "itw":  # In-the-Wild: evaluation only
            out.append("itw")
            key.append(f"ITW{r.uid}")
            continue
        elif r.family == "resynth":  # D6-R copies exist only for train-split reals: always train
            out.append("resynth")
            key.append(f"RS{r.uid}")
            continue
        else:  # real_extra: whole speakers
            held, k = _u01(seed, f"EX{r.speaker}") < frac, f"EX{r.speaker}"
        out.append("holdout" if held else "train")
        key.append(k)
    out = np.array(out, dtype=object)
    val = np.array([_u01(seed + 1, k) < val_frac for k in key]) & (out == "train")
    out[val] = "val"
    out[out == "resynth"] = "train"
    return out


def logo_splits(lab, folds):
    """Yield (generator, fold, train_idx, test_idx): train never sees `generator` or fold k."""
    real = (lab.label == 0).to_numpy()
    for g in config.GENERATORS:
        is_g = (lab.generator == g).to_numpy()
        for k in range(folds.max() + 1):
            te = np.where((folds == k) & (real | is_g))[0]
            tr = np.where((folds != k) & ~is_g)[0]
            yield g, k, tr, te


SUBSETS = {
    # voice held constant: LJSpeech real vs TTS models trained on LJSpeech
    "lj_matched": lambda d: d.family.isin(["lj_voice", "real_lj"]),
    # voice held constant: the 10 LibriSpeech speakers vs their clones
    "clone_matched": lambda d: d.family.isin(["clone", "real_libri"]),
    # only the organizers' 242 real clips as bona fide (what the provided data alone supports)
    "nsa_real_only": lambda d: (d.label == 1) | (d.generator == "real_lj_nsa"),
}


def subset_masks(lab):
    return {name: fn(lab).to_numpy() for name, fn in SUBSETS.items()}


def fold_table(lab, folds):
    return pd.crosstab(lab.generator, folds)
