import json

import pandas as pd

from src import transform

COL = "TelephonyManager.getSimCountryIso"


def run_stage(tmp_path, monkeypatch, rows):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    pd.DataFrame(rows).to_csv(raw_dir / "batch.csv", index=False)
    (raw_dir / "_checkpoint.json").write_text(
        json.dumps({"batch_file": "batch.csv", "rows": len(rows)}), encoding="utf-8")
    monkeypatch.setattr(transform, "RAW_DIR", raw_dir)
    monkeypatch.setattr(transform, "STAGING_DIR", tmp_path / "staging")
    monkeypatch.setattr(transform, "QUARANTINE_DIR", tmp_path / "quarantine")
    return transform.stage()


def row(key, **changes):
    base = {COL: "0", "flag_a": "1", "flag_b": "0", "class": "B", "row_key": key}
    base.update(changes)
    return base


def test_clean_rows_are_not_quarantined(tmp_path, monkeypatch):
    good, bad = run_stage(tmp_path, monkeypatch, [row("k1"), row("k2", **{"class": "S"})])
    assert (good, bad) == (2, 0)


def test_question_mark_is_cleaned_not_rejected(tmp_path, monkeypatch):
    good, bad = run_stage(tmp_path, monkeypatch, [row("k1", **{COL: "?"})])
    assert (good, bad) == (1, 0)


def test_bad_feature_value_is_quarantined(tmp_path, monkeypatch):
    good, bad = run_stage(tmp_path, monkeypatch, [row("k1"), row("k2", flag_a="2")])
    assert (good, bad) == (1, 1)


def test_bad_label_is_quarantined(tmp_path, monkeypatch):
    good, bad = run_stage(tmp_path, monkeypatch, [row("k1"), row("k2", **{"class": "X"})])
    assert (good, bad) == (1, 1)


def test_identical_rows_are_all_kept(tmp_path, monkeypatch):
    rows = [row("k1"), row("k2"), row("k3")]
    good, bad = run_stage(tmp_path, monkeypatch, rows)
    assert good == 3