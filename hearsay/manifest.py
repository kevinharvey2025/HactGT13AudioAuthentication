"""One table for every clip the track uses: DiffSSD fakes, bona fide references and the test set.

label: 1 = spoof (synthetic), 0 = bona fide, -1 = unknown (test). Everything downstream is keyed
by `uid`, which also names the cached 16 kHz file (cache/wav16k/<uid>.wav).
"""
import pandas as pd

from . import config

REAL_FAMILIES = {"real_lj", "real_libri"}


def _diffssd(n_lj_voice_sentences, n_clone_per_speaker, seed):
    m = pd.read_csv(config.DIFFSSD / "metadata.csv")
    m["path"] = [str(config.DIFFSSD / f) for f in m.file_name]
    m["uid"] = m.file_name.str.replace("generated_speech/", "", regex=False) \
                          .str.replace("real_speech/", "real_lj_nsa/", regex=False) \
                          .str.rsplit(".", n=1).str[0]
    fakes = m[m.label == "fake"].copy()
    # LJ-voice TTS: the same sentence ids for all four generators (paired comparisons)
    lj = fakes[fakes.generator.isin(config.LJ_VOICE)]
    sent = lj.sentence_id.drop_duplicates().sample(n_lj_voice_sentences, random_state=seed)
    lj = lj[lj.sentence_id.isin(sent)]
    # cloning systems: a fixed number of clips per (generator, speaker); openvoicev2 styles mixed in
    cl = fakes[fakes.generator.isin(config.CLONE)]
    cl = cl.groupby(["generator", "speaker"]).sample(n=n_clone_per_speaker, random_state=seed)
    fakes = pd.concat([lj, cl])
    fakes["family"] = ["lj_voice" if g in config.LJ_VOICE else "clone" for g in fakes.generator]
    fakes["text_group"] = "s" + fakes.sentence_id.astype(int).astype(str)
    fakes["label"] = 1

    real = m[m.label == "real"].copy()
    real["generator"], real["family"], real["label"] = "real_lj_nsa", "real_lj", 0
    real["text_group"] = "lj:" + real.utterance_id
    out = pd.concat([fakes, real])
    out["source"] = "diffssd"
    return out


def _ljspeech_extra(n, exclude, seed):
    root = config.EXTERNAL / "LJSpeech-1.1"
    meta = pd.read_csv(root / "metadata.csv", sep="|", header=None, quoting=3, names=["id", "text", "norm"])
    meta = meta[~meta.id.isin(exclude)].sample(n, random_state=seed)
    return pd.DataFrame(dict(
        uid="real_lj/" + meta.id, path=[str(root / "wavs" / f"{i}.wav") for i in meta.id], label=0,
        generator="real_lj", family="real_lj", speaker="LJSpeech", text_group="lj:" + meta.id, source="ljspeech"))


def _libri10():
    root = config.EXTERNAL / "librispeech_10spk"
    idx = pd.read_csv(root / "index.csv")
    return pd.DataFrame(dict(
        uid="real_libri/" + idx.speaker.astype(str) + "/" + idx.utt_id, path=[str(root / f) for f in idx.file],
        label=0, generator="real_libri", family="real_libri", speaker=idx.speaker.astype(str),
        text_group="libri:" + idx.utt_id, source="librispeech"))


def test_manifest():
    tpl = pd.read_csv(config.TEST_DIR / "HGT_Hearsay_score_template.csv", sep="\t")
    return pd.DataFrame(dict(
        uid="test/" + tpl.filename.str.rsplit(".", n=1).str[0], path=[str(config.TEST_DIR / f) for f in tpl.filename],
        filename=tpl.filename, label=-1, generator="unknown", family="test", speaker="unknown",
        text_group="test:" + tpl.filename, source="test"))


def build(n_lj_voice_sentences=1000, n_clone_per_speaker=100, n_lj_extra=2000, seed=0):
    d = _diffssd(n_lj_voice_sentences, n_clone_per_speaker, seed)
    nsa_ids = set(d.loc[d.generator == "real_lj_nsa", "utterance_id"])
    parts = [d, _ljspeech_extra(n_lj_extra, nsa_ids, seed), _libri10(), test_manifest()]
    cols = ["uid", "path", "label", "generator", "family", "speaker", "style", "sentence_id", "text_group", "source", "filename"]
    out = pd.concat(parts, ignore_index=True).reindex(columns=cols)
    out["speaker"] = out.speaker.astype(str)
    assert out.uid.is_unique
    return out


def load():
    return pd.read_parquet(config.CACHE / "manifest.parquet")


def labeled(man=None):
    man = load() if man is None else man
    return man[man.label >= 0].reset_index(drop=True)
