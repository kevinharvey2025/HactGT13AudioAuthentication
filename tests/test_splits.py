"""The shared split: no group straddles train / val / holdout; In-the-Wild and copy-synthesis rows are placed by rule."""
import numpy as np
import pandas as pd

from hearsay import splits


def manifest():
    rows = []
    for spk in ["19", "26", "103", "1034", "2061", "5448"]:                    # voice clones of LibriSpeech speakers
        rows += [dict(uid=f"clone/{spk}_{s}", family="clone", speaker=spk, sentence_id=s, label=1) for s in range(60)]
    rows += [dict(uid=f"lj_voice/{s}", family="lj_voice", speaker="LJ", sentence_id=s, label=1) for s in range(200)]
    rows += [dict(uid=f"real_lj/LJ{c:03d}-{k:04d}", family="real_lj", speaker="LJ", label=0) for c in range(1, 51) for k in range(5)]
    rows += [dict(uid=f"real_libri/{spk}-{100 + c}-{k:04d}", family="real_libri", speaker=spk, label=0)
             for spk in ["19", "26", "2061", "5448"] for c in range(5) for k in range(4)]
    rows += [dict(uid=f"real_extra/{1000 + s}-1-{k}", family="real_extra", speaker=str(1000 + s), label=0) for s in range(40) for k in range(3)]
    rows += [dict(uid=f"itw/{k}.wav", family="itw", speaker="itw", label=k % 2) for k in range(20)]
    rows += [dict(uid=f"resynth/vocos/LJ001-{k:04d}", family="resynth", speaker="LJ", label=1) for k in range(30)]
    return pd.DataFrame(rows)


def test_rules():
    lab = manifest()
    sp = pd.Series(splits.shared_split(lab), index=lab.index)
    assert set(sp) == {"train", "val", "holdout", "itw"}
    assert (sp[lab.family == "itw"] == "itw").all() and (sp[lab.family != "itw"] != "itw").all()
    assert (sp[lab.family == "resynth"] == "train").all()
    groups = {
        "sentence": lab.sentence_id.where(lab.family.isin(["clone", "lj_voice"])),
        "lj_chapter": lab.uid.str.extract(r"real_lj/(LJ\d+)-")[0],
        "libri_chapter": lab.uid.str.extract(r"real_libri/(\d+-\d+)-")[0],
        "extra_speaker": lab.speaker.where(lab.family == "real_extra"),
    }
    held_spk = set(lab.speaker[(lab.family == "clone") & (sp == "holdout")].value_counts().loc[lambda c: c == 60].index)
    assert len(held_spk) == 2                                  # every clone of the two held-out speakers is held out
    for name, g in groups.items():
        m = g.notna() & ~(lab.speaker.isin(held_spk) & (lab.family == "clone"))
        n_splits = sp[m].groupby(g[m]).nunique()
        assert (n_splits == 1).all(), f"{name} groups straddle splits"
    assert (sp[(lab.family == "real_libri") & lab.speaker.isin(held_spk)] == "holdout").all()


def test_copy_synthesis_rows_do_not_move_anything():
    lab = manifest()
    base = lab[lab.family != "resynth"]
    np.testing.assert_array_equal(splits.shared_split(lab)[: len(base)], splits.shared_split(base))


def test_deterministic_and_seeded():
    lab = manifest()
    a, b = splits.shared_split(lab), splits.shared_split(lab)
    assert (a == b).all() and (a != splits.shared_split(lab, seed=1)).any()
