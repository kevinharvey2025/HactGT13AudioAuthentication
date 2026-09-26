"""Tiny supervised fixture workflow: checks mechanics only (decode -> features -> train ->
predict -> TSV). The two synthetic 'classes' differ by construction; nothing here is
evidence of real detection accuracy."""

import json

import numpy as np
import pandas as pd
import soundfile as sf
import yaml

from hearsay_dsp.cli import main
from hearsay_dsp.io.export import validate_submission_file

from dsp_testutils import SR, speechlike


def _make_fixture(tmp_path):
    rows = []
    rng = np.random.default_rng(0)
    for i in range(36):
        label = i % 2
        x = speechlike(2.5, f0=110 + 5 * (i % 7), seed=i, noise_db=-45)
        if label:  # 'synthetic' fixture class: smoothed high band + slight periodic ripple
            x = np.convolve(x, [0.5, 0.5], mode="same") + 0.002 * np.sin(2 * np.pi * 3000 * np.arange(x.size) / SR)
        name = f"clip_{i:02d}.wav"
        sf.write(tmp_path / name, x, SR, subtype="PCM_16")
        rows.append({"path": name, "label": "spoof" if label else "bonafide",
                     "split": "holdout" if i >= 30 else "dev", "group_id": f"g{i // 2}"})
    pd.DataFrame(rows).to_csv(tmp_path / "manifest.tsv", sep="\t", index=False)
    cfg = {"run": {"cache_dir": str(tmp_path / "cache"), "n_jobs": 2},
           "model": {"gmm": {"n_components": 2, "max_frames_per_clip": 40, "max_frames_total": 5000,
                             "max_iter": 50}, "lr": {"C_grid": [0.1, 1.0]}, "inner_folds": 3}}
    (tmp_path / "cfg.yaml").write_text(yaml.safe_dump(cfg))
    return tmp_path


def test_end_to_end_cli_fixture(tmp_path):
    d = _make_fixture(tmp_path)
    assert main(["train", "--manifest", str(d / "manifest.tsv"), "--config", str(d / "cfg.yaml"),
                 "--out", str(d / "model")]) == 0
    meta = json.loads((d / "model" / "bundle.json").read_text())["meta"]
    assert meta["n_train"] == 30 and meta["splits_used"] == ["dev"]  # holdout rows never trained on
    test_dir = d / "test_in"
    test_dir.mkdir()
    for i in range(30, 36):
        (test_dir / f"clip_{i:02d}.wav").write_bytes((d / f"clip_{i:02d}.wav").read_bytes())
    ref = d / "template.tsv"
    ref.write_text("filename\tcm-score\n" + "".join(f"clip_{i:02d}.wav\t0.006\n" for i in range(35, 29, -1)))
    out = d / "pred.tsv"
    assert main(["predict", "--input", str(test_dir), "--model", str(d / "model"), "--output", str(out),
                 "--reference", str(ref)]) == 0
    rep = validate_submission_file(out, [f"clip_{i:02d}.wav" for i in range(35, 29, -1)])
    assert rep["valid"], rep
    traces = [json.loads(l) for l in (d / "pred.traces.jsonl").read_text().splitlines()]
    assert len(traces) == 6 and all("explanation" in t and "analyses" in t for t in traces)
    # repeatability: a second prediction run gives identical scores
    out2 = d / "pred2.tsv"
    assert main(["predict", "--input", str(test_dir), "--model", str(d / "model"), "--output", str(out2),
                 "--reference", str(ref)]) == 0
    assert out.read_text() == out2.read_text()
    # confidence routing also yields a complete, valid file
    out3 = d / "pred3.tsv"
    assert main(["predict", "--input", str(test_dir), "--model", str(d / "model"), "--output", str(out3),
                 "--reference", str(ref), "--routing", "confidence"]) == 0
    assert validate_submission_file(out3, [f"clip_{i:02d}.wav" for i in range(35, 29, -1)])["valid"]


def test_predict_without_bundle_fails_actionably(tmp_path):
    (tmp_path / "in").mkdir()
    sf.write(tmp_path / "in" / "a.wav", speechlike(2.0), SR)
    rc = main(["predict", "--input", str(tmp_path / "in"), "--model", str(tmp_path / "nope"),
               "--output", str(tmp_path / "p.tsv")])
    assert rc == 3 and not (tmp_path / "p.tsv").exists()


def test_predict_blocks_export_on_undecodable_file(tmp_path):
    d = _make_fixture(tmp_path)
    assert main(["train", "--manifest", str(d / "manifest.tsv"), "--config", str(d / "cfg.yaml"),
                 "--out", str(d / "model")]) == 0
    (d / "in").mkdir()
    sf.write(d / "in" / "ok.wav", speechlike(2.5), SR)
    (d / "in" / "bad.wav").write_bytes(b"RIFF\x00\x00\x00\x00WAVEjunk")
    rc = main(["predict", "--input", str(d / "in"), "--model", str(d / "model"), "--output", str(d / "p.tsv")])
    assert rc == 2 and not (d / "p.tsv").exists()
    fails = pd.read_csv(d / "p.failures.tsv", sep="\t")
    assert fails["filename"].tolist() == ["bad.wav"]
