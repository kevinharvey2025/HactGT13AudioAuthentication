"""HEARSAY inference (team SideQuests): audio files -> <team>_predictions_final.tsv + per-file traces.

    python predict.py --input DIR --output DIR [--artifacts artifacts/diffusion] [--template FILE]
                      [--device auto|cpu|cuda] [--batch 8] [--label final]

For every audio file in --input (any container/codec ffmpeg reads):
  1. T0 triage: container, codec, rate, channels, encoder tag, file times (trace and explanation only; they are
     constant on the NSA test set and a shortcut in the training data, so they never enter the score);
  2. decode with ffmpeg to 16 kHz mono and build the canonical view the detectors were trained on
     (edge-silence trim, 7 kHz low-pass, DC removal, peak-normalize, 1-LSB dither);
  3. each fine-tuned anti-spoofing detector listed in <artifacts>/fusion.json gives a synthetic logit
     (logit_fake - logit_real); logits are z-normalized with the stored statistics, averaged, and mapped to
     P(synthetic) with the stored Platt calibration (fitted on validation + In-the-Wild clips, never on test);
  4. the router (agentic orchestration) decides per file which further analyses run, from the triage facts and
     the detector's confidence, and records every decision with its reason in the trace:
       - uncertain score (0.05 < P < 0.8)            -> windowed re-scoring (splice/partial-fake check) + concept explanation
       - lossy codec or high declared rate but narrow band -> compression/bandwidth cross-check (metadata vs signal)
       - clip longer than 6 s                          -> windowed re-scoring (voice/score drift across the clip)
       - confident score and clean container          -> stop (saves compute)
     Only the fused detector score is written to the TSV; the routed analyses explain it.
Output rows follow --template (the organizers' prefilled TSV) when given, else sorted filenames; files that fail
to decode are listed in <output>/failures.tsv and the export is refused (no invented scores).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from hearsay import antideepfake, audio, config, metrics, submission  # noqa: E402
from hearsay.forensics import triage  # noqa: E402

AUDIO_EXT = {".wav", ".mp3", ".m4a", ".mp4", ".aac", ".ogg", ".opus", ".flac", ".wma", ".webm", ".aiff", ".aif", ".amr"}


class Detector(torch.nn.Module):
    def __init__(self, backbone, pretrained=False):
        super().__init__()
        # architecture only: the fine-tuned checkpoint holds every tensor (no base-model download at inference)
        self.enc, self.head = antideepfake.load(backbone[len("adf_"):], pretrained=pretrained)

    def forward(self, x):
        h = self.enc(antideepfake.standardize(x)).last_hidden_state.mean(1)
        return self.head(h.float())


def load_systems(artifacts, device):
    spec = json.load(open(artifacts / "fusion.json"))
    systems = []
    for s in spec["systems"]:
        m = Detector(s["backbone"])
        ckpt = artifacts / Path(s["checkpoint"]).name if (artifacts / Path(s["checkpoint"]).name).exists() \
            else artifacts / s["name"] / "best.pt"
        sd = torch.load(ckpt, map_location="cpu")
        m.load_state_dict({k: v.float() for k, v in sd.items()})
        systems.append((s, m.eval().to(device)))
    return spec, systems


@torch.inference_mode()
def logits(model, xs, device, batch):
    order = np.argsort([len(x) for x in xs])
    out = np.zeros(len(xs), np.float32)
    for k in range(0, len(order), batch):
        idx = order[k: k + batch]
        n = min(len(xs[j]) for j in idx) // 320 * 320
        x = torch.from_numpy(np.stack([xs[j][:n] for j in idx])).to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            lg = model(x).float()
        out[idx] = antideepfake.synthetic_logit(lg).cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--artifacts", default=str(REPO / "artifacts" / "diffusion"))
    ap.add_argument("--template", default="", help="prefilled TSV (filename<TAB>cm-score) defining rows and order")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--team", default=config.TEAM)
    ap.add_argument("--label", default="final")
    a = ap.parse_args()
    device = torch.device("cuda" if (a.device == "auto" and torch.cuda.is_available()) else
                          ("cpu" if a.device == "auto" else a.device))
    inp, out = Path(a.input), Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    template = Path(a.template) if a.template else next(iter(sorted(inp.glob("*template*"))), None)
    files = (pd.read_csv(template, sep="\t").filename.tolist() if template and template.exists()
             else sorted(p.name for p in inp.iterdir() if p.suffix.lower() in AUDIO_EXT))
    print(f"{len(files)} files; device {device}; template {template}", flush=True)

    traces, xs, ok, failures = [], [], [], []
    for f in files:
        p = inp / f
        tr = dict(filename=f)
        try:
            raw = audio.decode(p)
            tr["triage"] = {k: v for k, v in triage.triage(p, raw).items() if not isinstance(v, float) or np.isfinite(v)}
            # same per-clip RNG as the training-time test view (uid "test/<stem>", view 0): identical audio
            xs.append(audio.canonical(raw, audio.uid_rng("test/" + Path(f).stem, 0), durations=None))
            ok.append(f)
        except Exception as e:  # recorded, never scored
            failures.append(dict(filename=f, error=repr(e)[:300]))
        traces.append(tr)

    spec, systems = load_systems(Path(a.artifacts), device)
    z = []
    for s, model in systems:
        lg = logits(model, xs, device, a.batch)
        z.append((lg - s["z_mean"]) / s["z_std"])
        for t, v in zip([t for t in traces if t["filename"] in set(ok)], lg):
            t.setdefault("detectors", {})[s["name"]] = round(float(v), 4)
    fused = np.mean(z, 0)
    prob = 1 / (1 + np.exp(-(spec["platt"]["coef"] * fused + spec["platt"]["intercept"])))
    prob = np.clip(prob, 1e-6, 1 - 1e-6)
    scores = dict(zip(ok, prob))

    # ---- router: targeted analyses per file (explanations; the TSV score stays the fused detector score)
    window_model = systems[0][1]
    for t, x, p in zip([t for t in traces if t["filename"] in scores], xs, [scores[f] for f in ok]):
        tri = t.get("triage", {})
        reasons = {}
        if 0.05 < p < 0.8:
            reasons["windowed_rescoring"] = f"detector score {p:.2f} in the uncertain band"
        if len(x) > 6 * config.SR:
            reasons.setdefault("windowed_rescoring", f"clip is {len(x) / config.SR:.1f} s long: check score drift")
        lossy = str(tri.get("codec", "")).lower() not in ("pcm_s16le", "pcm_s24le", "pcm_f32le", "flac", "")
        narrow = tri.get("native_sr", 0) >= 32000 and tri.get("cutoff_hz", 1e9) < 0.6 * tri.get("native_sr", 0) / 2
        if lossy or narrow:
            reasons["compression_crosscheck"] = ("lossy codec " + str(tri.get("codec"))) if lossy else \
                f"declared {tri.get('native_sr')} Hz but energy stops near {tri.get('cutoff_hz', 0):.0f} Hz"
        t["router"] = reasons or {"stop": "confident score, clean container: no further analysis"}
        if "windowed_rescoring" in reasons:  # 2 s windows, 1 s hop: max/min spread flags partial fakes
            w, h = 2 * config.SR, config.SR
            segs = [x[i: i + w] for i in range(0, max(1, len(x) - w + 1), h)] or [x]
            lw = logits(window_model, segs, device, a.batch)
            t["windows"] = dict(n=len(segs), max_logit=round(float(lw.max()), 3), min_logit=round(float(lw.min()), 3),
                                spread=round(float(lw.max() - lw.min()), 3),
                                finding=("score varies strongly across the clip (possible partial edit)"
                                         if lw.max() - lw.min() > 6 else "consistent across the clip"))
        if "compression_crosscheck" in reasons:
            t["compression"] = dict(finding=reasons["compression_crosscheck"],
                                    interpretation="channel history; not evidence of synthesis by itself")

    with open(out / "traces.jsonl", "w") as fh:
        for t in traces:
            if t["filename"] in scores:
                p = float(scores[t["filename"]])
                t["cm_score"] = round(p, 6)
                thr = metrics.bayes_threshold(calib_prior=spec["platt"].get("prior", 0.5))  # Pspoof 0.3, Cfa 4
                t["decision_at_bayes_threshold"] = "synthetic" if p > thr else "bona fide"
            fh.write(json.dumps(t, default=str) + "\n")
    if failures:
        pd.DataFrame(failures).to_csv(out / "failures.tsv", sep="\t", index=False)
        sys.exit(f"{len(failures)} files could not be scored (see {out / 'failures.tsv'}); export refused")
    name = submission.output_name(a.team).replace(".tsv", f"_{a.label}.tsv" if a.label else ".tsv")
    tpl = template if template and template.exists() else None
    if tpl:
        submission.write(pd.Series(scores), out / name, template=tpl)
    else:
        pd.DataFrame({"filename": list(scores), "cm-score": list(scores.values())}).to_csv(
            out / name, sep="\t", index=False, float_format="%.10f", lineterminator="\n")
    print("wrote", out / name, submission.describe(prob, calib_prior=spec["platt"].get("prior", 0.5)))


if __name__ == "__main__":
    main()
