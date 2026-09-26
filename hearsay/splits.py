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
    return f


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
