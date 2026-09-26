import json
import os

import numpy as np
import pytest
import soundfile as sf

from hearsay_dsp.evaluation.augment import codec_roundtrip
from hearsay_dsp.io.cache import FeatureCache
from hearsay_dsp.io.decode import DecodeError, decode_audio, sniff_format
from hearsay_dsp.pipeline import analyze_file, records_to_table
from hearsay_dsp.routing import eligibility

from dsp_testutils import SR, speechlike


def test_pcm16_amplitude_convention_and_stereo_downmix(write_wav, cfg):
    ints = np.array([0, 16384, -32768, 32767], dtype=np.int16)
    p = write_wav(np.tile(ints, 1000), name="i.wav")
    dec = decode_audio(p, cfg)
    assert dec.analysis[:4].tolist() == [0.0, 0.5, -1.0, 32767 / 32768]
    st = np.stack([np.full(SR, 0.5), np.full(SR, -0.5)], axis=1)
    dec2 = decode_audio(write_wav(st, name="s.wav"), cfg)
    assert dec2.provenance["channels"] == 2
    assert dec2.quality["downmix_cancellation_risk"] is True  # anti-phase channels flagged


def test_nonfinite_float_samples_are_zeroed_and_counted(write_wav, cfg):
    x = speechlike(2.0)
    x[100:110] = np.nan
    dec = decode_audio(write_wav(x, name="f.wav", subtype="FLOAT"), cfg)
    assert dec.quality["nonfinite_samples"] == 10 and np.isfinite(dec.analysis).all()
    x[: x.size // 2] = np.inf
    with pytest.raises(DecodeError):
        decode_audio(write_wav(x, name="g.wav", subtype="FLOAT"), cfg)


def test_clipping_is_measured(write_wav, cfg):
    x = np.clip(5 * speechlike(2.0), -1, 1)  # peak 1.5 before clipping
    assert decode_audio(write_wav(x, name="c.wav"), cfg).quality["clipped_frac"] > 0.01


def test_mp3_in_wav_extension_decodes_and_is_flagged(tmp_path, cfg):
    import av
    x = speechlike(2.0)
    p = tmp_path / "looks_like.wav"
    pcm = (x * 32767).astype(np.int16)[None, :]
    with av.open(str(p), mode="w", format="mp3") as out:
        s = out.add_stream("libmp3lame", rate=SR)
        s.layout = "mono"
        fr = av.AudioFrame.from_ndarray(pcm, format="s16", layout="mono")
        fr.sample_rate = SR
        for pk in s.encode(fr):
            out.mux(pk)
        for pk in s.encode(None):
            out.mux(pk)
    assert sniff_format(str(p)) == "mp3"
    rec = analyze_file(str(p), cfg, None, modules=("container", "lfcc"))
    assert rec["decode"]["provenance"]["decoder"] == "pyav"
    assert rec["modules"]["container"]["diagnostics"]["diag.container.extension_mismatch"] is True


def test_codec_roundtrips_preserve_length_roughly():
    x = speechlike(2.0)
    for codec, br in (("libmp3lame", 32000), ("aac", 32000), ("libopus", 16000)):
        y = codec_roundtrip(x, SR, codec, br)
        assert abs(y.size - x.size) < 0.1 * SR


def test_unreadable_file_is_a_decode_error_not_a_score(tmp_path, cfg):
    p = tmp_path / "junk.wav"
    p.write_bytes(os.urandom(4096))
    rec = analyze_file(str(p), cfg, FeatureCache(cfg["run"]["cache_dir"]))
    assert rec["error"].startswith("decode failed")
    t = records_to_table([rec])
    assert t["decode_status"].iloc[0] == "error"


def test_decoded_silence_differs_from_unreadable(write_wav, cfg):
    rec = analyze_file(write_wav(np.zeros(SR * 2), name="z.wav"), cfg, None)
    assert "error" not in rec
    assert rec["modules"]["activity"]["status"] == "insufficient_signal"
    for m in ("lpc", "background", "phase", "prosody"):
        assert rec["modules"][m]["status"] == "insufficient_signal", m
    assert rec["modules"]["compression"]["status"] == "not_applicable"


def test_short_clip_modules_abstain_with_reasons(write_wav, cfg):
    rec = analyze_file(write_wav(speechlike(0.3), name="short.wav"), cfg, None)
    for m in ("lpc", "background", "phase"):
        assert rec["modules"][m]["status"] in ("insufficient_signal", "ok")
        if rec["modules"][m]["status"] != "ok":
            assert rec["modules"][m]["reason"]


def test_cache_reuse_invalidation_and_partial_entries(write_wav, cfg):
    p = write_wav(speechlike(2.0), name="k.wav")
    cache = FeatureCache(cfg["run"]["cache_dir"])
    r1 = analyze_file(p, cfg, cache)
    r2 = analyze_file(p, cfg, cache)
    assert all(d["decision"] == "cached" for d in r2["routing"])
    assert r1["modules"]["spectral"] == r2["modules"]["spectral"]
    cfg2 = json.loads(json.dumps(cfg))
    cfg2["features"]["lpc"]["order"] = 12  # changes only the LPC key
    r3 = analyze_file(p, cfg2, cache)
    decisions = {d["module"]: d["decision"] for d in r3["routing"]}
    assert decisions["lpc"] == "run" and decisions["spectral"] == "cached"
    # a record whose array file vanished is treated as incomplete and recomputed
    sha = r1["sha256"]
    key = r1["module_keys"]["lfcc"]
    cache.array_path(sha, "lfcc", key).unlink()
    r4 = analyze_file(p, cfg, cache)
    assert {d["module"]: d["decision"] for d in r4["routing"]}["lfcc"] == "run"
    assert cache.array_path(sha, "lfcc", key).exists()
    assert not list(cache.record_path(sha).parent.glob(".*tmp*"))  # no leftover temp files


def test_router_abstains_without_background(cfg):
    class A:  # activity with active frames but no background frames
        active = np.ones(300, bool)
        background = np.zeros(300, bool)
        hop_s = 0.01
    run, status, reason = eligibility("background", A(), cfg)
    assert not run and status == "insufficient_signal" and "low-energy frames" in reason
    assert eligibility("compression", A(), cfg)[1] == "not_applicable"
