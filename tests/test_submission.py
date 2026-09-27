"""The challenge TSV: writer, validator, calibration, and the committed final file."""
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hearsay import metrics, submission

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "submission" / "SideQuests_predictions_final.tsv"


@pytest.fixture
def template(tmp_path):
    p = tmp_path / "template.csv"
    pd.DataFrame({"filename": [f"HGT{i}.wav" for i in (5, 3, 9, 1)], "cm-score": 0.5}).to_csv(p, sep="\t", index=False)
    return p


def test_write_follows_template_order(template, tmp_path):
    out = tmp_path / "pred.tsv"
    submission.write({"HGT1.wav": 0.1, "HGT9.wav": 0.9, "HGT3.wav": 1 / 3, "HGT5.wav": 0.0}, out, template)
    d = pd.read_csv(out, sep="\t")
    assert list(d.filename) == ["HGT5.wav", "HGT3.wav", "HGT9.wav", "HGT1.wav"]
    assert "0.3333333333" in out.read_text()                   # 10 decimals: no artificial ties
    assert submission.validate(out, template)


@pytest.mark.parametrize("bad", [{"HGT1.wav": 0.1}, {"HGT1.wav": np.nan}, {"HGT1.wav": 1.2}, {"HGT1.wav": -0.1}])
def test_write_refuses_missing_or_invalid(template, tmp_path, bad):
    scores = {"HGT5.wav": 0.2, "HGT3.wav": 0.2, "HGT9.wav": 0.2, **bad}
    if bad == {"HGT1.wav": 0.1}:
        scores.pop("HGT9.wav")
    with pytest.raises(ValueError):
        submission.write(scores, tmp_path / "pred.tsv", template)


@pytest.mark.parametrize("corrupt", ["header", "duplicate", "order", "crlf"])
def test_validate_catches(template, tmp_path, corrupt):
    good = pd.read_csv(template, sep="\t")
    if corrupt == "header":
        good = good.rename(columns={"cm-score": "score"})
    elif corrupt == "duplicate":
        good.loc[1, "filename"] = good.loc[0, "filename"]
    elif corrupt == "order":
        good = good.iloc[::-1]
    out = tmp_path / "pred.tsv"
    good.to_csv(out, sep="\t", index=False, lineterminator="\r\n" if corrupt == "crlf" else "\n")
    with pytest.raises(AssertionError):
        submission.validate(out, template)


def test_platt_shifts_log_odds_to_the_prior():
    rng = np.random.default_rng(0)
    y = (rng.random(4000) < 0.4).astype(int)
    s = rng.normal(2.0 * y, 1.0)
    a, b = submission.Platt(prior=0.5).fit(s, y), submission.Platt(prior=0.3).fit(s, y)
    assert a.coef == pytest.approx(b.coef)
    assert b.intercept - a.intercept == pytest.approx(np.log(0.3 / 0.7))
    assert np.all(np.diff(b(np.sort(s))) >= 0)                  # monotone: minDCF / EER unchanged


def _template_frame():
    f = ROOT / "data" / "hearsay_test" / "HGT_Hearsay_score_template.csv"
    if f.exists():
        return pd.read_csv(f, sep="\t")
    for z in sorted(ROOT.glob("data/*HackGTHearsayTesting*.zip")):
        with zipfile.ZipFile(z) as zf:
            name = next((n for n in zf.namelist() if n.endswith("HGT_Hearsay_score_template.csv")), None)
            if name:
                return pd.read_csv(io.BytesIO(zf.read(name)), sep="\t")
    return None


def test_committed_final_tsv():
    raw = FINAL.read_bytes()
    assert b"\r" not in raw
    d = pd.read_csv(FINAL, sep="\t")
    assert list(d.columns) == ["filename", "cm-score"] and len(d) == 1671 and d.filename.is_unique
    s = d["cm-score"].to_numpy(float)
    assert np.isfinite(s).all() and (s >= 0).all() and (s <= 1).all()
    assert 0.05 < (s > metrics.bayes_threshold(calib_prior=0.3)).mean() < 0.95
    ref = _template_frame()
    if ref is not None:
        assert (d.filename.to_numpy() == ref.filename.to_numpy()).all()
