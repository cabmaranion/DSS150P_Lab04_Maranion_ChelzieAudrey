import json
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")
STAGING_DIR = Path("data/staging")
CURATED_DIR = Path("data/curated")
QUARANTINE_DIR = Path("data/quarantine")
COL = "TelephonyManager.getSimCountryIso"


def stage():
    saved = json.loads((RAW_DIR / "_checkpoint.json").read_text(encoding="utf-8"))
    raw = pd.read_csv(RAW_DIR / saved["batch_file"], dtype=str)
    assert len(raw) == saved["rows"], "raw row count differs from the checkpoint"

    df = raw.copy()
    df[COL] = df[COL].replace("?", "0")

    feature_cols = [c for c in df.columns if c not in ("class", "row_key")]
    numeric = df[feature_cols].apply(pd.to_numeric, errors="coerce")

    bad_feature = (~numeric.isin([0, 1])).any(axis=1)
    bad_label = ~df["class"].isin(["S", "B"])
    invalid = bad_feature | bad_label

    quarantine = raw[invalid].copy()
    quarantine["reason"] = "feature not 0/1 or label not S/B"

    good = df[~invalid].copy()
    good[feature_cols] = numeric[~invalid].astype(int)
    good = good.copy()
    good["is_malware"] = good["class"].map({"S": 1, "B": 0})
    assert good["row_key"].is_unique, "row keys are not unique"
    assert len(good) + len(quarantine) == saved["rows"], "rows were lost"

    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    good.to_csv(STAGING_DIR / "drebin_staged.csv", index=False)
    if len(quarantine):
        QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
        quarantine.to_csv(QUARANTINE_DIR / "drebin_rejected.csv", index=False)

    print("staged rows:", len(good))
    print("quarantined rows:", len(quarantine))
    print("class counts:", good["is_malware"].value_counts().to_dict())
    return len(good), len(quarantine)


def curate():
    """Model-ready layer: key, 215 integer flags, and the 0/1 label. Row order is kept."""
    staged = pd.read_csv(STAGING_DIR / "drebin_staged.csv")
    feature_cols = [c for c in staged.columns if c not in ("row_key", "class", "is_malware")]
    assert len(feature_cols) == 215, "unexpected number of features"

    curated = staged[["row_key"] + feature_cols + ["is_malware"]]
    assert curated["row_key"].is_unique, "row keys are not unique"
    assert curated[feature_cols + ["is_malware"]].isin([0, 1]).all().all(), "non 0/1 value"

    CURATED_DIR.mkdir(parents=True, exist_ok=True)
    curated.to_csv(CURATED_DIR / "drebin_curated.csv", index=False)
    print("curated rows:", len(curated), "features:", len(feature_cols))
    return len(curated)


if __name__ == "__main__":
    stage()
    curate()