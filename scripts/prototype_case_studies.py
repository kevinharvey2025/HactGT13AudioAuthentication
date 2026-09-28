"""Case studies of the diffusion prototypes: which synthetic ideas each prototype family captures, and by what criteria.

    python scripts/prototype_case_studies.py [--features runs/dsp/extract_hearsay_test/features.tsv]
        [--scores submission/candidates/SideQuests_predictions_final_A.tsv] [--out results/concepts/case_studies]

Inputs: the per-clip explanations of the NSA test set (results/concepts/test_explanations.jsonl: basic-level concept and
three TTCG prototypes per clip, each named by its COBWEB concept), the submitted scores, and the signal-processing
features of the test clips (hearsay_dsp).

Every prototype is assigned a family by the dominant source of the concept that names it (e.g. "commercial cloner:
ElevenLabs", "LJ-voice diffusion TTS", "real LibriSpeech, cloned speakers"). A clip's dominant family is the family
with the largest summed prototype share. For each family the script reports how many clips it dominates, the
detector's scores on them, the noise levels and concept sizes of its prototypes, and the channels of its concepts.
It then compares the clips each synthetic family dominates with the clips the real-LibriSpeech families dominate on
interpretable acoustic features (Cliff's delta with a bootstrap interval): these are the family's measurable criteria.
No test labels exist; the comparison describes the audio each family explains, not ground truth.

Writes families.csv, criteria.csv, cases.json (representative and special clips) and provenance.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hearsay import config, provenance  # noqa: E402

FAMILY = {
    "elevenlabs": "Commercial cloner: ElevenLabs", "playht": "Commercial cloner: PlayHT",
    "xtts_v2": "Open zero-shot cloners: XTTS v2, YourTTS", "your_tts": "Open zero-shot cloners: XTTS v2, YourTTS",
    "unit_speech": "Unit-based cloning and conversion: UnitSpeech, OpenVoice v2",
    "openvoicev2": "Unit-based cloning and conversion: UnitSpeech, OpenVoice v2",
    "wavegrad2": "LJ-voice diffusion TTS", "pro_diff": "LJ-voice diffusion TTS", "grad_tts": "LJ-voice diffusion TTS",
    "diffgan_tts": "LJ-voice diffusion TTS",
    "real:ljspeech": "Real LJSpeech", "real:librispeech_cloned_speakers": "Real LibriSpeech, cloned speakers",
    "real:librispeech_other_speakers": "Real LibriSpeech, other speakers",
}
REAL_REF = ("Real LibriSpeech, cloned speakers", "Real LibriSpeech, other speakers")
FEATURES = {  # interpretable features and their plain names
    "bg.frame_level_std_db": "frame loudness variability (dB)",
    "bg.snr_db": "estimated SNR (dB)",
    "bg.noise_flatness_db": "background-noise flatness (dB)",
    "bg.floor_std_db": "noise-floor variability (dB)",
    "spec.fb_hiflatness_db_p90": "high-band spectral flatness, p90 (dB)",
    "spec.fb_flux_iqr": "spectral flux spread (IQR)",
    "spec.fb_centroid_hz_med": "spectral centroid (Hz)",
    "spec.fb_rolloff95_hz_med": "95% roll-off (Hz)",
    "spec.band_6000_7000_db_med": "energy 6-7 kHz (dB)",
    "spec.band_50_250_db_med": "energy 50-250 Hz (dB)",
    "spec.ltas_ripple_std_db": "long-term spectral ripple (dB)",
    "pros.f0_st_std": "pitch variability (semitones)",
    "pros.f0_dst_std": "pitch-change variability (semitones)",
    "pros.pauses_per_active_s": "pauses per active second",
    "pros.jitter_local": "jitter",
    "pros.shimmer_local": "shimmer",
    "pros.hnr_db": "harmonics-to-noise ratio (dB)",
    "lpc.res_kurtosis_med": "LPC residual kurtosis",
    "lpc.pred_gain_db_med": "LPC prediction gain (dB)",
    "phase.ifjump_hz_med": "instantaneous-frequency jumps (Hz)",
}


def cliffs_delta(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 5 or len(b) < 5:
        return np.nan
    allv = np.concatenate([a, b])
    ranks = pd.Series(allv).rank().to_numpy()
    u = ranks[: len(a)].sum() - len(a) * (len(a) + 1) / 2
    return float(2 * u / (len(a) * len(b)) - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--explanations", default=str(config.REPO / "results" / "concepts" / "test_explanations.jsonl"))
    ap.add_argument("--features", default=str(config.REPO / "runs" / "dsp" / "extract_hearsay_test" / "features.tsv"))
    ap.add_argument("--scores", default=str(config.REPO / "submission" / "candidates" / "SideQuests_predictions_final_A.tsv"))
    ap.add_argument("--out", default=str(config.REPO / "results" / "concepts" / "case_studies"))
    ap.add_argument("--boot", type=int, default=500)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    ex = [json.loads(line) for line in open(a.explanations)]
    P = pd.read_csv(a.scores, sep="\t").set_index("filename")["cm-score"]
    feats = pd.read_csv(a.features, sep="\t", low_memory=False).set_index("filename")

    protos, clips = [], []
    for r in ex:
        fam = {}
        for p in r["diffusion_prototypes"]:
            top = p["concept_sources"][0]["source"]
            ch = p["concept_channels"][0] if p["concept_channels"] else {"channel": "", "share": 0}
            f = FAMILY.get(top, top)
            fam[f] = fam.get(f, 0) + p["share"]
            protos.append(dict(filename=r["filename"], family=f, t=p["noise_level"], share=p["share"],
                               depth=p["concept_depth"], size=p["concept_size"], p_fake=p["concept_p_fake"],
                               channel=ch["channel"], channel_share=ch["share"]))
        dom = max(fam, key=fam.get)
        synth_share = sum(p["share"] * p["concept_p_fake"] for p in r["diffusion_prototypes"])
        clips.append(dict(filename=r["filename"], P=float(P[r["filename"]]), family=dom, purity=fam[dom],
                          n_families=len(fam), proto_synth_share=synth_share, ttcg_p_fake=r["ttcg_p_fake"],
                          cobweb_p_fake=r["cobweb_p_fake"], basic_depth=r["concept"]["depth"],
                          basic_size=r["concept"]["size"], basic_p_fake=r["concept"]["p_fake"]))
    pr, cl = pd.DataFrame(protos), pd.DataFrame(clips)

    fam_rows = []
    for f, g in cl.groupby("family"):
        pg = pr[pr.family == f]
        chans = pg[pg.channel_share >= 0.5].channel.value_counts(normalize=True)
        fam_rows.append(dict(family=f, clips=len(g), share_of_test=len(g) / len(cl), median_P=g.P.median(),
                             flagged=float((g.P > 0.2).mean()), mean_purity=g.purity.mean(),
                             prototypes=len(pg), median_t=pg.t.median(), share_t_ge_150=float((pg.t >= 150).mean()),
                             median_concept_size=pg["size"].median(), median_concept_depth=pg.depth.median(),
                             concept_p_fake=pg.p_fake.mean(),
                             non_clean_channel=float(((pg.channel != "clean") & (pg.channel_share >= 0.5)).mean()),
                             top_channels="; ".join(f"{k} {v:.0%}" for k, v in chans.head(3).items())))
    fams = pd.DataFrame(fam_rows).sort_values("clips", ascending=False)
    fams.to_csv(out / "families.csv", index=False, float_format="%.4f")

    ref = cl[cl.family.isin(REAL_REF) & (cl.purity >= 0.66)].filename
    crit = []
    for f in fams.family:
        if f in REAL_REF:
            continue
        fs = cl[(cl.family == f) & (cl.purity >= 0.66)].filename
        if len(fs) < 8:
            fs = cl[cl.family == f].filename
        for col, name in FEATURES.items():
            if col not in feats:
                continue
            x, y = feats.loc[feats.index.intersection(fs), col], feats.loc[feats.index.intersection(ref), col]
            d = cliffs_delta(x, y)
            bs = [cliffs_delta(rng.choice(x.to_numpy(), len(x)), rng.choice(y.to_numpy(), len(y))) for _ in range(a.boot)]
            crit.append(dict(family=f, feature=col, name=name, n_family=int(x.notna().sum()), n_real=int(y.notna().sum()),
                             median_family=float(x.median()), median_real=float(y.median()), cliffs_delta=d,
                             lo=float(np.nanpercentile(bs, 2.5)), hi=float(np.nanpercentile(bs, 97.5))))
    crit = pd.DataFrame(crit)
    crit["abs_delta"] = crit.cliffs_delta.abs()
    crit.sort_values(["family", "abs_delta"], ascending=[True, False]).to_csv(out / "criteria.csv", index=False, float_format="%.4f")

    def card(fn):
        r = next(e for e in ex if e["filename"] == fn)
        c = cl.set_index("filename").loc[fn]
        return dict(filename=fn, P=round(float(c.P), 4), family=c.family, purity=round(float(c.purity), 3),
                    ttcg_p_fake=r["ttcg_p_fake"], basic=r["concept"], prototypes=r["diffusion_prototypes"])

    cases = {"representative": {}, "special": {}}
    for f in fams.family:
        g = cl[cl.family == f].sort_values(["purity", "P"], ascending=[False, f in REAL_REF])
        cases["representative"][f] = [card(fn) for fn in g.filename.head(2)]
    mixed = cl[(cl.proto_synth_share > 0.2) & (cl.proto_synth_share < 0.8)].sort_values("P")
    disagree = cl[(cl.ttcg_p_fake - cl.P).abs() > 0.5].sort_values("P")
    uncertain = cl[(cl.P > 0.05) & (cl.P < 0.8)].assign(d=lambda d: (d.P - 0.5).abs()).sort_values("d")
    high_t = pr.groupby("filename").t.max().sort_values(ascending=False)
    chan_synth = pr[(pr.p_fake > 0.9) & (pr.channel != "clean") & (pr.channel_share >= 0.6)].sort_values("share", ascending=False)
    chan_real = pr[(pr.p_fake < 0.1) & (pr.channel != "clean") & (pr.channel_share >= 0.6)].sort_values("share", ascending=False)
    cases["special"] = {
        "counts": dict(mixed=len(mixed), disagree=len(disagree), uncertain=len(uncertain), clips=len(cl),
                       multi_family=int((cl.n_families > 1).sum()),
                       synthetic_channel_prototypes=len(chan_synth), real_channel_prototypes=len(chan_real)),
        "mixed": [card(fn) for fn in mixed.filename.head(3)],
        "disagree": [card(fn) for fn in disagree.filename],
        "uncertain": [card(fn) for fn in uncertain.filename.head(3)],
        "high_noise_level": [card(fn) for fn in high_t.index[:2]],
        "synthetic_under_channel": [card(fn) for fn in chan_synth.filename.drop_duplicates().head(2)],
        "real_under_channel": [card(fn) for fn in chan_real.filename.drop_duplicates().head(2)],
    }
    t_rows = pr.assign(tb=pd.cut(pr.t, [0, 50, 100, 200, 400], labels=["50", "75-100", "125-200", "225-375"])) \
        .groupby("tb", observed=True).agg(prototypes=("t", "size"), median_size=("size", "median"),
                                          median_depth=("depth", "median"), synthetic_concepts=("p_fake", lambda v: float((v > 0.5).mean())))
    cases["noise_level_table"] = t_rows.reset_index().to_dict(orient="records")
    json.dump(cases, open(out / "cases.json", "w"), indent=1, default=float)
    provenance.write(out, inputs=[a.explanations, a.features, a.scores], bootstrap_reps=a.boot)
    print(fams.round(3).to_string(index=False))
    print(json.dumps(cases["special"]["counts"]))
    print(pd.DataFrame(cases["noise_level_table"]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
