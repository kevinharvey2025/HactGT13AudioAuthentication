import numpy as np
import pandas as pd
import pytest

from hearsay_dsp.evaluation.metrics import binary_metrics, eer, group_bootstrap, auc_w
from hearsay_dsp.evaluation.experiments import make_folds
from hearsay_dsp.io.export import (ExportError, build_submission, read_reference,
                                   validate_submission_file, write_submission)
from hearsay_dsp.io.manifest import (ManifestError, normalize_label, read_manifest,
                                     select_training_rows, validate_manifest)


# ---------------------------------------------------------------- labels
@pytest.mark.parametrize("raw,expected", [("real", 0), ("bonafide", 0), ("Bona-Fide", 0), ("0", 0),
                                          ("spoof", 1), ("fake", 1), ("Synthetic", 1), ("1.0", 1),
                                          ("", None), (np.nan, None)])
def test_label_normalization(raw, expected):
    assert normalize_label(raw) == expected


@pytest.mark.parametrize("raw", ["maybe", "partial", "2", "edited"])
def test_ambiguous_labels_rejected(raw):
    with pytest.raises(ManifestError):
        normalize_label(raw)


def test_manifest_relative_paths_and_eval_rows_blocked(tmp_path):
    (tmp_path / "a.wav").write_bytes(b"x")
    (tmp_path / "b.wav").write_bytes(b"x")
    (tmp_path / "c.wav").write_bytes(b"x")
    m = tmp_path / "m.tsv"
    m.write_text("path\tlabel\tsplit\tgroup_id\na.wav\treal\tdev\tg1\nb.wav\tspoof\tholdout\tg2\n"
                 "c.wav\tspoof\ttest\tg3\n")
    df = read_manifest(m)
    assert df["path"].iloc[0] == str(tmp_path / "a.wav")
    train = select_training_rows(df)
    assert train["filename"].tolist() == ["a.wav"]
    rep = validate_manifest(df)
    assert rep["errors"] == [] and rep["n_real"] == 1 and rep["n_synthetic"] == 2


def test_manifest_validation_flags_missing_and_duplicates(tmp_path):
    m = tmp_path / "m.tsv"
    (tmp_path / "d1").mkdir()
    (tmp_path / "d2").mkdir()
    (tmp_path / "d1" / "x.wav").write_bytes(b"x")
    (tmp_path / "d2" / "x.wav").write_bytes(b"x")
    m.write_text("path\tlabel\nd1/x.wav\treal\nd2/x.wav\tfake\nmissing.wav\treal\n")
    rep = validate_manifest(read_manifest(m))
    assert rep["missing_files"] == 1 and rep["duplicate_basenames"] == 1 and rep["errors"]


# ---------------------------------------------------------------- export
def test_submission_schema_order_and_blocking(tmp_path):
    ref = tmp_path / "ref.tsv"
    ref.write_text("filename\tcm-score\nb.wav\t0.006\na.wav\t0.006\n")
    names = read_reference(ref)
    assert names == ["b.wav", "a.wav"]
    sub = build_submission(["a.wav", "b.wav"], [0.2, 0.9], names)
    out = tmp_path / "pred.tsv"
    write_submission(sub, out)
    text = out.read_text().splitlines()
    assert text[0] == "filename\tcm-score"
    assert [ln.split("\t")[0] for ln in text[1:]] == ["b.wav", "a.wav"]
    assert validate_submission_file(out, names)["valid"]
    with pytest.raises(ExportError):
        build_submission(["a.wav"], [0.5], names)                 # missing reference file
    with pytest.raises(ExportError):
        build_submission(["a.wav", "b.wav"], [0.5, np.nan], names)  # non-finite
    with pytest.raises(ExportError):
        build_submission(["a.wav", "b.wav"], [0.5, 1.2], names)     # out of range
    with pytest.raises(ExportError):
        build_submission(["a.wav", "a.wav"], [0.5, 0.4], None)      # duplicate rows
    with pytest.raises(ExportError):
        build_submission(["a.wav", "b.wav", "c.wav"], [0.1, 0.2, 0.3], names)  # extra file


def test_reference_header_must_match(tmp_path):
    ref = tmp_path / "ref.csv"
    ref.write_text("filename,cm-score\na.wav,0.1\n")
    with pytest.raises(ExportError):
        read_reference(ref)


# ---------------------------------------------------------------- metrics
def test_known_answer_auc_eer():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.6, 0.4, 0.9])
    m = binary_metrics(y, s)
    assert m["auc"] == pytest.approx(0.75)
    assert m["eer"] == pytest.approx(0.5)
    assert eer(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(0.0)


def test_eer_interpolates_between_roc_points():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    s = np.array([1, 2, 3, 6, 4, 5, 7, 8], dtype=float)
    # one real above one fake -> FPR/FNR cross at 0.25
    assert eer(y, s) == pytest.approx(0.25)


def test_log_loss_brier_and_confusion():
    y = np.array([0, 1])
    p = np.array([0.2, 0.9])
    m = binary_metrics(y, p)
    assert m["log_loss"] == pytest.approx(-(np.log(0.8) + np.log(0.9)) / 2)
    assert m["brier"] == pytest.approx((0.04 + 0.01) / 2)
    assert m["confusion"] == {"tp": 1, "tn": 1, "fp": 0, "fn": 0}
    assert m["genuine_fpr"] == 0.0


def test_single_class_metrics_are_undefined_not_crashing():
    m = binary_metrics(np.array([1, 1, 1]), np.array([0.2, 0.7, 0.9]))
    assert m["auc"] is None and m["eer"] is None and "undefined_reason" in m
    assert m["specificity"] is None and m["sensitivity"] == pytest.approx(2 / 3)


def test_group_bootstrap_contains_point_estimate():
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1], 200)
    s = y + rng.normal(0, 1.0, y.size)
    g = np.concatenate([np.arange(200) // 4, 1000 + np.arange(200) // 4])
    ci = group_bootstrap(y, s, g, auc_w, n_boot=200, seed=1)
    point = auc_w(y, s, None)
    assert ci["lo"] <= point <= ci["hi"]
    assert ci["n_groups_real"] == 50 and ci["n_groups_synthetic"] == 50


def test_folds_never_split_a_group():
    df = pd.DataFrame({"label": np.repeat([0, 1], 60),
                       "group_id": [f"g{i // 3}" for i in range(120)]})
    folds = make_folds(df, 5, 0)
    assert (df.assign(f=folds).groupby("group_id")["f"].nunique() == 1).all()
    assert set(np.unique(folds)) == set(range(5))
