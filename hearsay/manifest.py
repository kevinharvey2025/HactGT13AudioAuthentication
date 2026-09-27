"""One table for every clip the track uses: DiffSSD fakes, bona fide references and the test set.

label: 1 = spoof (synthetic), 0 = bona fide, -1 = unknown (test). Everything downstream is keyed
by `uid`, which also names the cached 16 kHz file (cache/wav16k/<uid>.wav). The `path` column is
absolute and machine-specific; scripts/prepare_data.py rebuilds it on each machine.
"""
import os
import warnings

import numpy as np
import pandas as pd

from . import config

REAL_FAMILIES = {"real_lj", "real_libri"}
SETUP_HINT = "see HANDOFF_DIFFUSION.md section 11 (moving to another machine)"


def _require(df, what, allow_missing):
    """The HF copy of DiffSSD lacks 5 generators; fail loudly instead of silently sampling missing files."""
    exists = np.array([os.path.exists(p) for p in df.path])
    if exists.all():
        return df
    counts = df[~exists].groupby("generator").size().to_dict()
    msg = f"{what}: {int((~exists).sum())} of {len(df)} files missing, by generator {counts}; {SETUP_HINT}"
    if not allow_missing:
        raise FileNotFoundError(msg)
    warnings.warn(msg + " -- dropping them (--allow-missing)")
    return df[exists]


def _read_diffssd():
    # speaker mixes "LJSpeech" and numeric ids: read it as str, or chunked parsing yields int/str mixes
    m = pd.read_csv(config.DIFFSSD / "metadata.csv", dtype={"speaker": str, "style": str, "utterance_id": str})
    m["path"] = [str(config.DIFFSSD / f) for f in m.file_name]
    m["uid"] = m.file_name.str.replace("generated_speech/", "", regex=False) \
                          .str.replace("real_speech/", "real_lj_nsa/", regex=False) \
                          .str.rsplit(".", n=1).str[0]
    return m


def _label_diffssd(fakes, real):
    fakes = fakes.copy()
    fakes["family"] = ["lj_voice" if g in config.LJ_VOICE else "clone" for g in fakes.generator]
    fakes["text_group"] = "s" + fakes.sentence_id.astype(int).astype(str)
    fakes["label"] = 1
    real = real.copy()
    real["generator"], real["family"], real["label"] = "real_lj_nsa", "real_lj", 0
    real["text_group"] = "lj:" + real.utterance_id
    out = pd.concat([fakes, real])
    out["source"] = "diffssd"
    return out


def _diffssd(n_lj_voice_sentences, n_clone_per_speaker, seed):
    m = _read_diffssd()
    fakes = m[m.label == "fake"].copy()
    # LJ-voice TTS: the same sentence ids for all four generators (paired comparisons)
    lj = fakes[fakes.generator.isin(config.LJ_VOICE)]
    sent = lj.sentence_id.drop_duplicates().sample(n_lj_voice_sentences, random_state=seed)
    lj = lj[lj.sentence_id.isin(sent)]
    # cloning systems: a fixed number of clips per (generator, speaker); openvoicev2 styles mixed in
    cl = fakes[fakes.generator.isin(config.CLONE)]
    cl = cl.groupby(["generator", "speaker"]).sample(n=n_clone_per_speaker, random_state=seed)
    return _label_diffssd(pd.concat([lj, cl]), m[m.label == "real"])


def _ljspeech_extra(n, exclude, seed):
    """n LJSpeech-1.1 originals not among `exclude` (n=None: all of them)."""
    root = config.EXTERNAL / "LJSpeech-1.1"
    if not (root / "metadata.csv").exists():
        raise FileNotFoundError(f"{root} missing: run scripts/fetch_ljspeech.sh ({SETUP_HINT})")
    meta = pd.read_csv(root / "metadata.csv", sep="|", header=None, quoting=3, names=["id", "text", "norm"])
    meta = meta[~meta.id.isin(exclude)]
    meta = meta if n is None else meta.sample(n, random_state=seed)
    return pd.DataFrame(dict(
        uid="real_lj/" + meta.id, path=[str(root / "wavs" / f"{i}.wav") for i in meta.id], label=0,
        generator="real_lj", family="real_lj", speaker="LJSpeech", text_group="lj:" + meta.id, source="ljspeech"))


def _libri10():
    root = config.EXTERNAL / "librispeech_10spk"
    if not (root / "index.csv").exists():
        raise FileNotFoundError(f"{root} missing: run scripts/fetch_librispeech_speakers.py ({SETUP_HINT})")
    idx = pd.read_csv(root / "index.csv")
    return pd.DataFrame(dict(
        uid="real_libri/" + idx.speaker.astype(str) + "/" + idx.utt_id, path=[str(root / f) for f in idx.file],
        label=0, generator="real_libri", family="real_libri", speaker=idx.speaker.astype(str),
        text_group="libri:" + idx.utt_id, source="librispeech"))


def _extra_reals():
    """Optional extra bona fide speech (data/external/extra_reals/index.csv: file, speaker, source[, chapter]),
    e.g. LibriSpeech dev/test-clean speakers that no DiffSSD system clones (scripts/fetch_extra_reals.sh)."""
    root = config.EXTERNAL / "extra_reals"
    if not (root / "index.csv").exists():
        return pd.DataFrame()
    idx = pd.read_csv(root / "index.csv", dtype=str)
    return pd.DataFrame(dict(
        uid="real_extra/" + idx.file.str.rsplit(".", n=1).str[0], path=[str(root / f) for f in idx.file],
        label=0, generator="real_extra", family="real_extra", speaker=idx.source + ":" + idx.speaker,
        text_group="extra:" + idx.file, source=idx.source))


def test_manifest(required=True):
    tpl_path = config.TEST_DIR / "HGT_Hearsay_score_template.csv"
    if not tpl_path.exists():
        if required:
            raise FileNotFoundError(f"test set missing at {config.TEST_DIR} ({SETUP_HINT})")
        warnings.warn(f"no test set at {config.TEST_DIR}: building a training-only manifest")
        return pd.DataFrame()
    tpl = pd.read_csv(tpl_path, sep="\t")
    return pd.DataFrame(dict(
        uid="test/" + tpl.filename.str.rsplit(".", n=1).str[0], path=[str(config.TEST_DIR / f) for f in tpl.filename],
        filename=tpl.filename, label=-1, generator="unknown", family="test", speaker="unknown",
        text_group="test:" + tpl.filename, source="test"))


def build(n_lj_voice_sentences=1000, n_clone_per_speaker=100, n_lj_extra=2000, seed=0,
          allow_missing=False, require_test=True):
    # sample first, then check files: the sample is identical on every machine that has the full data
    d = _require(_diffssd(n_lj_voice_sentences, n_clone_per_speaker, seed), "DiffSSD", allow_missing)
    nsa_ids = set(d.loc[d.generator == "real_lj_nsa", "utterance_id"])
    parts = [d, _ljspeech_extra(n_lj_extra, nsa_ids, seed), _libri10(), test_manifest(require_test)]
    cols = ["uid", "path", "label", "generator", "family", "speaker", "style", "sentence_id", "text_group", "source", "filename"]
    out = pd.concat(parts, ignore_index=True).reindex(columns=cols)
    out["speaker"] = out.speaker.astype(str)
    assert out.uid.is_unique
    return out


def build_pool(require_test=True):
    """Every labeled clip on disk, for fine-tuning: all DiffSSD audio present, all LJSpeech-1.1 originals
    (minus the 242 the organizers resampled), the 10 LibriSpeech speakers, optional extra reals, the test set.
    uids match build(), so both share cache/wav16k."""
    m = _read_diffssd()
    m = m[[os.path.exists(p) for p in m.path]]
    d = _label_diffssd(m[m.label == "fake"], m[m.label == "real"])
    nsa_ids = set(d.loc[d.generator == "real_lj_nsa", "utterance_id"])
    parts = [d, _ljspeech_extra(None, nsa_ids, 0), _libri10(), _extra_reals(), test_manifest(require_test)]
    cols = ["uid", "path", "label", "generator", "family", "speaker", "style", "sentence_id", "text_group", "source", "filename"]
    out = pd.concat([p for p in parts if len(p)], ignore_index=True).reindex(columns=cols)
    out["speaker"] = out.speaker.astype(str)
    assert out.uid.is_unique
    return out


def load(name="manifest"):
    return pd.read_parquet(config.CACHE / f"{name}.parquet")


def labeled(man=None):
    man = load() if man is None else man
    return man[man.label >= 0].reset_index(drop=True)
