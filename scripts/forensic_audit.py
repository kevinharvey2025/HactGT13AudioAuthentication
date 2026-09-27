"""Forensic integrity audit of the data before any training (hashes, structure, signal history,
processing-pipeline identification, content matching). Nothing here is a classifier input.

    python scripts/forensic_audit.py structure  --workers 64     # RIFF layout, byte/PCM hashes, duplicates
    python scripts/forensic_audit.py signal     --workers 64     # lengths, levels, histogram/LSB, band edge
    python scripts/forensic_audit.py pipeline   --workers 64     # which resampling chain reproduces the test set
    python scripts/forensic_audit.py fingerprint --workers 64    # landmark-hash content matches test -> corpora
    python scripts/forensic_audit.py report                      # reports/forensics/audit.md from the tables

Sets: test (1,671 NSA clips), nsa_real (242 organizer reals), diffssd (whatever DiffSSD audio is on disk,
up to --per-gen files per generator for the per-file stages), ljspeech (LJSpeech-1.1 originals),
librispeech (the 10 clone speakers). Outputs go to runs/forensics/ (parquet/npz/json).
"""
import argparse
import hashlib
import json
import os
import struct
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal as sps

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hearsay import config  # noqa: E402

OUT = config.REPO / "runs" / "forensics"
SR = 16000


# ----------------------------------------------------------------------------- file sets

def file_sets(per_gen=None, seed=0):
    d = config.DATA
    sets = {
        "test": sorted((d / "hearsay_test").glob("*.wav")),
        "nsa_real": sorted((d / "DiffSSD" / "real_speech").glob("*.wav")),
        "ljspeech": sorted((d / "external" / "LJSpeech-1.1" / "wavs").glob("*.wav")),
        "librispeech": sorted((d / "external" / "librispeech_10spk").rglob("*.flac")),
        "extra_reals": sorted((d / "external" / "extra_reals").rglob("*.flac")),
    }
    fakes = []
    for root in [d / "DiffSSD" / "generated_speech", d / "subset_stream" / "DiffSSD" / "generated_speech"]:
        if root.exists():
            fakes += [p for p in root.rglob("*") if p.suffix in (".wav", ".mp3")]
    by_rel = {}
    for p in fakes:  # the same file may exist in both trees; keep one
        by_rel.setdefault(p.as_posix().split("generated_speech/", 1)[1], p)
    fk = pd.DataFrame({"rel": list(by_rel), "path": list(by_rel.values())})
    fk["generator"] = fk.rel.str.split("/").str[0]
    if per_gen:
        fk = fk.sample(frac=1.0, random_state=seed).groupby("generator").head(per_gen)
    sets["diffssd"] = sorted(fk.path)
    return sets


def table(sets):
    rows = []
    for s, paths in sets.items():
        for p in paths:
            gen = p.as_posix().split("generated_speech/", 1)[1].split("/")[0] if s == "diffssd" else s
            rows.append(dict(set=s, source=gen, path=str(p), file=p.name))
    return pd.DataFrame(rows)


def decode16k(path):
    """ffmpeg -> 16 kHz mono int16 (identity for files that already are 16 kHz mono PCM16 WAV)."""
    path = str(path)
    if path.endswith(".wav"):
        info = sf.info(path)
        if info.samplerate == SR and info.channels == 1 and info.subtype == "PCM_16":
            return sf.read(path, dtype="int16")[0]
    out = subprocess.run([config.ffmpeg_bin(), "-v", "error", "-nostdin", "-i", path, "-ac", "1", "-ar", str(SR),
                          "-f", "s16le", "-acodec", "pcm_s16le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(out, np.int16).copy()


def pool_map(fn, items, workers, chunksize=8):
    with ProcessPoolExecutor(workers) as ex:
        return list(ex.map(fn, items, chunksize=chunksize))


# ----------------------------------------------------------------------------- 1. structure

def riff_chunks(b):
    """[(id, size, offset)] of top-level RIFF chunks, plus LIST sub-ids; None if not RIFF/WAVE."""
    if len(b) < 12 or b[:4] != b"RIFF" or b[8:12] != b"WAVE":
        return None, {}
    chunks, info, off = [], {}, 12
    while off + 8 <= len(b):
        cid, size = b[off:off + 4].decode("latin1"), struct.unpack("<I", b[off + 4:off + 8])[0]
        chunks.append((cid, size, off))
        if cid == "LIST" and b[off + 8:off + 12] == b"INFO":
            o = off + 12
            while o + 8 <= off + 8 + size:
                sid, ss = b[o:o + 4].decode("latin1"), struct.unpack("<I", b[o + 4:o + 8])[0]
                info[sid] = b[o + 8:o + 8 + ss].rstrip(b"\0").decode("latin1", "replace")
                o += 8 + ss + (ss & 1)
        if cid == "fmt ":
            f = struct.unpack("<HHIIHH", b[off + 8:off + 24])
            info.update(fmt_tag=f[0], channels=f[1], rate=f[2], byte_rate=f[3], block_align=f[4], bits=f[5])
        off += 8 + size + (size & 1)
    return chunks, info


def structure_one(row):
    path = row["path"]
    b = open(path, "rb").read()
    st = os.stat(path)
    r = dict(path=path, size=len(b), sha256=hashlib.sha256(b).hexdigest(), mtime=st.st_mtime,
             magic=b[:4].decode("latin1", "replace"))
    chunks, info = riff_chunks(b)
    if chunks is not None:
        r["layout"] = " ".join(f"{c}({s})" if c != "data" else "data" for c, s, _ in chunks)
        r["riff_size_ok"] = struct.unpack("<I", b[4:8])[0] == len(b) - 8
        data = [c for c in chunks if c[0] == "data"]
        if data:
            end = data[0][2] + 8 + data[0][1]
            r["trailing_bytes"] = len(b) - end
            r["data_bytes"] = data[0][1]
        r.update({k: v for k, v in info.items()})
        if info.get("fmt_tag") == 1:
            r["byte_rate_ok"] = info["byte_rate"] == info["rate"] * info["channels"] * info["bits"] // 8
            r["block_align_ok"] = info["block_align"] == info["channels"] * info["bits"] // 8
    try:
        x = decode16k(path)
        r["pcm16k_sha256"] = hashlib.sha256(x.tobytes()).hexdigest()
    except Exception as e:  # a decode failure is itself a finding
        r["decode_error"] = repr(e)[:200]
    return r


def cmd_structure(a):
    if (OUT / "structure.parquet").exists() and not a.force:
        res = pd.read_parquet(OUT / "structure.parquet")
    else:
        t = table(file_sets(a.per_gen))
        res = pd.DataFrame(pool_map(structure_one, t.to_dict("records"), a.workers))
        res = t.merge(res, on="path")
        res.to_parquet(OUT / "structure.parquet")
    summ = {"n_files": res.groupby("set").size().to_dict(),
            "layouts": res.groupby(["set", "layout"], dropna=False).size().rename("n").reset_index().to_dict("records"),
            "ISFT": res.groupby(["set", "ISFT"], dropna=False).size().rename("n").reset_index().to_dict("records")}
    for key in ("sha256", "pcm16k_sha256"):
        sub = res[res[key].notna()]
        dup = sub[sub.duplicated(key, keep=False)]
        groups = [sorted(s + ":" + f for s, f in zip(g.set, g.file)) for _, g in dup.groupby(key)]
        cross = [g for g in groups if len({x.split(":")[0] for x in g}) > 1]
        summ[f"dup_groups_{key}"] = len(groups)
        summ[f"cross_set_dup_groups_{key}"] = cross[:50]
        summ[f"within_test_dup_groups_{key}"] = [g for g in groups if all(x.startswith("test:") for x in g)][:50]
    json.dump(summ, open(OUT / "structure_summary.json", "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in summ.items() if not k.startswith(("layouts", "ISFT"))}, indent=1, default=str)[:3000])
    print(res.groupby(["set", "layout"], dropna=False).size().to_string())


# ----------------------------------------------------------------------------- 2. signal

BANDS = [(0, 300), (300, 1000), (1000, 2000), (2000, 4000), (4000, 5000), (5000, 6000), (6000, 6500),
         (6500, 7000), (7000, 7250), (7250, 7500), (7500, 7750), (7750, 8000)]


def ltas(x, nfft=1024):
    f, p = sps.welch(x.astype(np.float64) / 32768.0, fs=SR, nperseg=nfft, noverlap=nfft // 2, window="hann")
    return f, p


def signal_one(row):
    try:
        x = decode16k(row["path"])
    except Exception as e:
        return dict(path=row["path"], error=repr(e)[:200])
    n = len(x)
    ax = np.abs(x.astype(np.int32))
    peak = int(ax.max()) if n else 0
    r = dict(path=row["path"], n=n, dur=n / SR, peak=peak, n_at_peak=int((ax == peak).sum()),
             n_fullscale=int((ax >= 32767).sum()), dc=float(x.mean() / 32768.0),
             rms_db=float(10 * np.log10(np.mean((x / 32768.0) ** 2) + 1e-20)),
             zeros_frac=float((x == 0).mean()), odd_frac=float((x & 1).mean()))
    # source lengths at 22.05 kHz that a 22.05k -> 16k resampler (floor/round/ceil rule) maps to n samples;
    # a pipeline that works in 256- or 512-sample frames at 22.05 kHz leaves a multiple in this window
    cands = np.arange(int(np.floor((n - 1) * 22050 / 16000)), int(np.ceil((n + 1) * 22050 / 16000)) + 1)
    for q in (256, 512):
        hit = cands[cands % q == 0]
        r[f"mult{q}_any"] = bool(len(hit))
        r[f"frames{q}"] = int(hit[0] // q) if len(hit) else -1
    # edges: level of first/last 50 ms relative to the loudest 50 ms (hard cuts vs. silence)
    fr = SR // 20
    if n >= 3 * fr:
        e = (x[: n // fr * fr].astype(np.float64).reshape(-1, fr) / 32768.0) ** 2
        db = 10 * np.log10(e.mean(1) + 1e-20)
        r.update(first50_rel_db=float(db[0] - db.max()), last50_rel_db=float(db[-1] - db.max()))
    # histogram forensics over the central value range: gaps/combs from gain scaling of integer audio
    h = np.bincount(np.clip(x.astype(np.int32) + 32768, 0, 65535), minlength=65536)
    core = h[32768 - 2048: 32768 + 2048]
    r["hist_empty_frac_core"] = float((core == 0).mean())
    spec = np.abs(np.fft.rfft(core - core.mean()))
    spec[:8] = 0
    k = int(np.argmax(spec))
    r["hist_comb_period"] = float(len(core) / k) if k else np.nan
    r["hist_comb_strength"] = float(spec[k] / (np.median(spec[8:]) + 1e-9))
    # LSB pairs-of-values chi-square (Westfeld & Pfitzmann): low p-values = equalized value pairs
    pairs = core.reshape(-1, 2)
    exp = pairs.mean(1)
    ok = exp > 5
    chi = float(((pairs[ok, 0] - exp[ok]) ** 2 / exp[ok]).sum()) if ok.any() else np.nan
    r["pov_chi2_per_pair"] = chi / max(1, ok.sum())
    # spectrum: band levels (dB re total) + edge measurements
    f, p = ltas(x)
    tot = p.sum() + 1e-30
    for lo, hi in BANDS:
        m = (f >= lo) & (f < hi)
        r[f"band_{lo}_{hi}"] = float(10 * np.log10(p[m].sum() / tot + 1e-30))
    ref = 10 * np.log10(p[(f >= 6000) & (f < 7000)].mean() + 1e-30)
    pd_ = 10 * np.log10(p + 1e-30) - ref
    for thr in (10, 20, 30, 40):
        above = np.where((f > 5000) & (pd_ > -thr))[0]
        r[f"edge_minus{thr}db_hz"] = float(f[above.max()]) if len(above) else np.nan
    return r, (np.float32(10 * np.log10(p + 1e-30)))


def cmd_signal(a):
    if (OUT / "signal.parquet").exists() and not a.force:
        print("signal.parquet exists (use --force to recompute)")
        return
    t = table(file_sets(a.per_gen))
    out = pool_map(signal_one, t.to_dict("records"), a.workers)
    rows, spectra = [], []
    for o in out:
        if isinstance(o, dict):
            rows.append(o)
            spectra.append(np.full(513, np.nan, np.float32))
        else:
            rows.append(o[0])
            spectra.append(o[1])
    res = t.merge(pd.DataFrame(rows), on="path")
    res.to_parquet(OUT / "signal.parquet")
    np.save(OUT / "ltas_db.npy", np.stack(spectra))
    cols = ["dur", "peak", "n_fullscale", "odd_frac", "zeros_frac", "hist_empty_frac_core", "pov_chi2_per_pair",
            "first50_rel_db", "last50_rel_db", "edge_minus20db_hz", "edge_minus40db_hz", "mult256_any", "mult512_any"]
    pd.set_option("display.width", 250)
    print(res.groupby("source")[cols].median(numeric_only=True).round(3).to_string())
    print(res.groupby("source")[["mult256_any", "mult512_any"]].mean().round(3).to_string())


# ----------------------------------------------------------------------------- 3. pipeline

def chains():
    """Candidate ways the organizers may have produced 16 kHz test audio from a source file."""
    import librosa
    import soxr
    import torch
    import torchaudio.functional as AF

    def ff(path):
        return decode16k(path).astype(np.float32) / 32768.0

    def lib(res_type):
        def f(path):
            y, _ = librosa.load(path, sr=22050, mono=True)           # librosa's default rate
            return librosa.resample(y, orig_sr=22050, target_sr=SR, res_type=res_type)
        return f

    def ta(kind):
        kw = dict(kaiser_best=dict(lowpass_filter_width=64, rolloff=0.9475937167399596, resampling_method="sinc_interp_kaiser",
                                   beta=14.769656459379492),
                  kaiser_fast=dict(lowpass_filter_width=16, rolloff=0.85, resampling_method="sinc_interp_kaiser",
                                   beta=8.555910),
                  default={})[kind]

        def f(path):
            y, _ = librosa.load(path, sr=22050, mono=True)
            return AF.resample(torch.from_numpy(y), 22050, SR, **kw).numpy()
        return f

    def sx(q):
        def f(path):
            y, _ = librosa.load(path, sr=22050, mono=True)
            return soxr.resample(y, 22050, SR, quality=q)
        return f

    return {"ffmpeg_swr": ff, "librosa_soxr_hq": lib("soxr_hq"), "librosa_kaiser_best": lib("kaiser_best"),
            "librosa_kaiser_fast": lib("kaiser_fast"), "librosa_polyphase": lib("polyphase"),
            "torchaudio_default": ta("default"), "torchaudio_kaiser_best": ta("kaiser_best"),
            "torchaudio_kaiser_fast": ta("kaiser_fast"), "soxr_vhq": sx("VHQ"), "soxr_lq": sx("LQ")}


def pipeline_one(item):
    name, path = item
    try:
        y = chains()[name](path)
    except Exception as e:
        return dict(chain=name, path=path, error=repr(e)[:200])
    y = np.clip(np.round(y / (np.abs(y).max() + 1e-12) * 32767), -32768, 32767).astype(np.int16)
    f, p = ltas(y)
    ref = 10 * np.log10(p[(f >= 6500) & (f < 7000)].sum() + 1e-30)
    top = 10 * np.log10(p[(f >= 7500) & (f < 8000)].sum() + 1e-30)
    pd_ = 10 * np.log10(p + 1e-30) - 10 * np.log10(p[(f >= 6000) & (f < 7000)].mean() + 1e-30)
    r = dict(chain=name, path=path, band_7500_8000_vs_6500_7000=float(top - ref))
    for thr in (10, 20, 30, 40):
        above = np.where((f > 5000) & (pd_ > -thr))[0]
        r[f"edge_minus{thr}db_hz"] = float(f[above.max()]) if len(above) else np.nan
    return r


def cmd_pipeline(a):
    sets = file_sets()
    rng = np.random.default_rng(0)
    src = list(rng.choice(sets["ljspeech"], 40, replace=False))
    fk = pd.Series(sets["diffssd"])
    gens = fk.map(lambda p: p.as_posix().split("generated_speech/", 1)[1].split("/")[0])
    for g in sorted(gens.unique()):
        src += list(fk[gens == g].sample(min(8, (gens == g).sum()), random_state=0))
    items = [(c, str(p)) for c in chains() for p in src]
    res = pd.DataFrame(pool_map(pipeline_one, items, a.workers, chunksize=2))
    res.to_parquet(OUT / "pipeline.parquet")
    sig = pd.read_parquet(OUT / "signal.parquet")
    test = sig[sig.set == "test"]
    tb = 10 * np.log10(10 ** (test["band_7500_7750"] / 10) + 10 ** (test["band_7750_8000"] / 10)) - test["band_6500_7000"]
    print("TEST: 7.5-8k vs 6.5-7k dB median %.1f (p10 %.1f, p90 %.1f); edge -20 dB %.0f Hz, -40 dB %.0f Hz" % (
        tb.median(), tb.quantile(0.1), tb.quantile(0.9), test.edge_minus20db_hz.median(), test.edge_minus40db_hz.median()))
    print(res.groupby("chain")[["band_7500_8000_vs_6500_7000", "edge_minus10db_hz", "edge_minus20db_hz",
                                "edge_minus40db_hz"]].median().round(1).to_string())


# ----------------------------------------------------------------------------- 4. fingerprint

FP_NFFT, FP_HOP, FP_FMAX_BIN, FP_FAN, FP_DT = 1024, 256, 256, 6, 48   # 16 ms hop, <= 4 kHz


def landmarks(x):
    """Shazam-style landmark hashes: (hash uint32, anchor frame int32) from spectral peaks <= 4 kHz."""
    from scipy.ndimage import maximum_filter
    x = x.astype(np.float32) / 32768.0
    if len(x) < FP_NFFT * 2:
        return np.zeros(0, np.uint32), np.zeros(0, np.int32)
    _, _, Z = sps.stft(x, fs=SR, nperseg=FP_NFFT, noverlap=FP_NFFT - FP_HOP, boundary=None, padded=False)
    S = np.log(np.abs(Z[1:FP_FMAX_BIN + 1]) + 1e-6)                     # [256 freq, T]
    peaks = (S == maximum_filter(S, size=(15, 9))) & (S > np.median(S) + 2.0)
    fi, ti = np.nonzero(peaks)
    order = np.argsort(ti, kind="stable")
    fi, ti = fi[order], ti[order]
    # keep the strongest ~30 peaks per second
    keep = np.zeros(len(fi), bool)
    for t0 in range(0, S.shape[1], 62):                                 # 62 frames ~ 1 s
        m = np.where((ti >= t0) & (ti < t0 + 62))[0]
        if len(m):
            keep[m[np.argsort(-S[fi[m], ti[m]])[:30]]] = True
    fi, ti = fi[keep], ti[keep]
    hs, ts = [], []
    for i in range(len(fi)):
        j = i + 1
        c = 0
        while j < len(fi) and c < FP_FAN:
            dt = ti[j] - ti[i]
            if dt > FP_DT:
                break
            if dt > 0 and abs(int(fi[j]) - int(fi[i])) < 64:
                hs.append((int(fi[i]) << 20) | (int(fi[j]) << 8) | int(dt))
                ts.append(int(ti[i]))
                c += 1
            j += 1
    return np.array(hs, np.uint32), np.array(ts, np.int32)


def fp_one(path):
    try:
        return landmarks(decode16k(path))
    except Exception:
        return np.zeros(0, np.uint32), np.zeros(0, np.int32)


def build_index(ref, workers):
    rfp = pool_map(fp_one, list(ref.path), workers, chunksize=16)
    H = np.concatenate([h for h, _ in rfp])
    T = np.concatenate([t for _, t in rfp])
    F = np.concatenate([np.full(len(h), i, np.int32) for i, (h, _) in enumerate(rfp)])
    o = np.argsort(H, kind="stable")
    return H[o], T[o], F[o]


def query(index, fps, names, ref):
    H, T, F = index
    rows = []
    for (h, t), name in zip(fps, names):
        lo, hi = np.searchsorted(H, h, "left"), np.searchsorted(H, h, "right")
        n = hi - lo
        keep = n <= 3000                                                    # drop uninformative (very common) hashes
        h, t, lo, hi, n = h[keep], t[keep], lo[keep], hi[keep], n[keep]
        if n.sum() == 0:
            rows.append(dict(name=name, n_hashes=len(h), best_votes=0))
            continue
        idx = np.concatenate([np.arange(a_, b_) for a_, b_ in zip(lo, hi)])
        key = F[idx].astype(np.int64) * 100000 + (T[idx] - np.repeat(t, n) + 50000)   # (file, time offset)
        u, c = np.unique(key, return_counts=True)
        best = np.argsort(-c)[:2]
        fid = int(u[best[0]] // 100000)
        rows.append(dict(name=name, n_hashes=len(h), best_votes=int(c[best[0]]),
                         best_ref=ref.file.iloc[fid], best_set=ref.set.iloc[fid], best_source=ref.source.iloc[fid],
                         best_offset_s=float(((u[best[0]] % 100000) - 50000) * FP_HOP / SR),
                         runner_up_votes=int(c[best[1]]) if len(best) > 1 else 0))
    res = pd.DataFrame(rows)
    res["vote_frac"] = res.best_votes / res.n_hashes.clip(lower=1)
    return res


def testlike(path, seed):
    """The test set's apparent pipeline: 22.05 kHz, trim (hop 512), start crop of 3-4 s in 512-sample frames,
    kaiser_fast-class resampling to 16 kHz, peak-normalize to 0.998."""
    import librosa
    import torch
    import torchaudio.functional as AF
    y, _ = librosa.load(path, sr=22050, mono=True)
    y, _ = librosa.effects.trim(y, top_db=60, frame_length=2048, hop_length=512)
    rng = np.random.default_rng(seed)
    n = int(rng.uniform(3.0, 4.0) * 22050) // 512 * 512
    y = y[:n]
    y = AF.resample(torch.from_numpy(y), 22050, SR, lowpass_filter_width=16, rolloff=0.85,
                    resampling_method="sinc_interp_kaiser", beta=8.555910).numpy()
    return np.round(y / (np.abs(y).max() + 1e-12) * 0.998 * 32767).astype(np.int16)


def fp_control_one(item):
    path, seed = item
    try:
        return landmarks(testlike(path, seed))
    except Exception:
        return np.zeros(0, np.uint32), np.zeros(0, np.int32)


def cmd_fingerprint(a):
    sets = file_sets()
    ref = table({k: v for k, v in sets.items() if k != "test"})
    test = table({"test": sets["test"]})
    print("fingerprinting", len(ref), "reference files and", len(test), "test clips;",
          ref.groupby("set").size().to_dict(), flush=True)
    index = build_index(ref, a.workers)
    print("reference hashes:", len(index[0]), flush=True)
    # positive control: reference files pushed through the test-like pipeline must find their source
    ctrl = ref.groupby("set", group_keys=False).sample(20, random_state=1).reset_index(drop=True)
    cfp = pool_map(fp_control_one, [(p, i) for i, p in enumerate(ctrl.path)], a.workers, chunksize=2)
    cres = query(index, cfp, list(ctrl.file), ref)
    cres["true_set"], cres["hit"] = ctrl.set, cres.best_ref.eq(ctrl.file)
    cres.to_parquet(OUT / "fingerprint_control.parquet")
    print("POSITIVE CONTROL (test-like copies of reference files):")
    print(cres.groupby("true_set").agg(n=("hit", "size"), hit_rate=("hit", "mean"),
                                       vote_frac_median=("vote_frac", "median"), vote_frac_min=("vote_frac", "min")).round(3).to_string())
    tfp = pool_map(fp_one, list(test.path), a.workers, chunksize=8)
    res = query(index, tfp, list(test.file), ref)
    res.to_parquet(OUT / "fingerprint.parquet")
    print("TEST:")
    print(res.vote_frac.describe().round(3).to_string())
    for thr in (0.05, 0.1, 0.2, 0.3):
        m = res.vote_frac >= thr
        print(f"vote_frac >= {thr}: {int(m.sum())} test clips; by source:", res[m].best_source.value_counts().to_dict())


# ----------------------------------------------------------------------------- report

def cmd_report(a):
    print("see runs/forensics/*.parquet; the markdown report is assembled by hand from these tables")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["structure", "signal", "pipeline", "fingerprint", "report"])
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--per-gen", type=int, default=150, help="DiffSSD files per generator for per-file stages")
    ap.add_argument("--force", action="store_true", help="recompute a stage whose output exists")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    globals()[f"cmd_{a.stage}"](a)


if __name__ == "__main__":
    main()
