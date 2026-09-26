"""Quick technical profile: DiffSSD generators (120 clips each) vs HEARSAY test (600 clips).

Measured at the 16 kHz analysis rate after resample_poly; no labels are used for the test set.
Run from the repo root: .venv/bin/python scripts/dsp_profile_quick.py
"""
import pandas as pd, os, random, numpy as np, soundfile as sf, av, glob, sys
from scipy.signal import resample_poly, welch
from math import gcd
from joblib import Parallel, delayed
def local_path(rel):
    return os.path.join("data/DiffSSD", rel)
def load(p):
    try:
        x, sr = sf.read(p, dtype="float64", always_2d=True)
    except Exception:
        with av.open(p) as c:
            s = c.streams.audio[0]; sr = s.sample_rate
            fr = [f.to_ndarray() for f in c.decode(s)]
            x = np.concatenate(fr, axis=1).T.astype(np.float64)
    x = x.mean(axis=1)
    if sr != 16000:
        g = gcd(16000, sr); x = resample_poly(x, 16000//g, sr//g); 
    return x
def prof(p, tag):
    x = load(p); n = len(x)
    fl = 400; hop = 160
    nf = 1 + (n - fl)//hop
    idx = np.arange(fl)[None,:] + hop*np.arange(nf)[:,None]
    fr = x[idx]; e = 10*np.log10(np.mean(fr**2, axis=1) + 1e-20)
    f, P = welch(x, fs=16000, nperseg=1024)
    Pdb = 10*np.log10(P + 1e-30)
    ref = np.median(Pdb[(f>=300)&(f<=3000)])
    band = lambda a,b: 10*np.log10(P[(f>=a)&(f<b)].sum()+1e-30)
    occ = f[np.where(Pdb >= ref - 50)[0].max()]
    return dict(tag=tag, dur=n/16000, rms_db=10*np.log10(np.mean(x**2)+1e-20), peak_db=20*np.log10(np.max(np.abs(x))+1e-12),
                digsil=np.mean(e < -90), e_first=e[:10].mean()-np.percentile(e,95), e_last=e[-10:].mean()-np.percentile(e,95),
                p10_minus_p95=np.percentile(e,10)-np.percentile(e,95), occ_bw50=occ,
                hf_4_8_rel=band(4000,8000)-band(0,4000), b78_rel_34=band(7000,8000)-band(3000,4000), b7575_8=band(7500,8000)-band(6500,7000))
md = pd.read_csv("data/DiffSSD/metadata.csv")
jobs=[]
for g, sub in md.groupby("generator"):
    files = sub.file_name.tolist(); random.Random(1).shuffle(files)
    jobs += [(local_path(r), g) for r in files[:120]]
test = sorted(glob.glob("data/hearsay_test/*.wav"))
random.Random(2).shuffle(test)
jobs += [(p,"TEST") for p in test[:600]]
res = Parallel(n_jobs=8)(delayed(prof)(p,t) for p,t in jobs)
df = pd.DataFrame(res)
pd.set_option("display.width", 250)
q = lambda s: f"{np.percentile(s,10):.1f}/{np.median(s):.1f}/{np.percentile(s,90):.1f}"
print(df.groupby("tag").agg({c: q for c in ["dur","rms_db","digsil","e_first","e_last","p10_minus_p95","occ_bw50","hf_4_8_rel","b78_rel_34","b7575_8"]}).to_string())
os.makedirs("reports/dsp", exist_ok=True)
df.to_csv("reports/dsp/profile_quick.csv", index=False)
