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
     P(synthetic) with the stored Platt calibration (fitted on validation + In-the-Wild clips, never on test).
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
from hearsay import antideepfake, audio, config, submission  # noqa: E402
from hearsay.forensics import triage  # noqa: E402

AUDIO_EXT = {".wav", ".mp3", ".m4a", ".mp4", ".aac", ".ogg", ".opus", ".flac", ".wma", ".webm", ".aiff", ".aif", ".amr"}


class Detector(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.enc, self.head = antideepfake.load(backbone[len("adf_"):])

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

    with open(out / "traces.jsonl", "w") as fh:
        for t in traces:
            if t["filename"] in scores:
                p = float(scores[t["filename"]])
                t["cm_score"] = round(p, 6)
                t["decision_at_bayes_threshold"] = "synthetic" if p > 0.2 else "bona fide"  # organizers' costs
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
            out / name, sep="\t", index=False, float_format="%.6f", lineterminator="\n")
    print("wrote", out / name, submission.describe(prob))


if __name__ == "__main__":
    main()
