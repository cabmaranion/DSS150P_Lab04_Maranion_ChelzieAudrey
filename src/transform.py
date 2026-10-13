import json
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")
STAGING_DIR = Path("data/staging")
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
    good["is_malware"] = good["class"].map({"S": 1, "B": 0})
    assert good["row_key"].is_unique, "row keys are not unique"
    assert len(good) + len(quarantine) == len(raw), "rows were lost"

    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    good.to_csv(STAGING_DIR / "drebin_staged.csv", index=False)
    if len(quarantine):
        QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
        quarantine.to_csv(QUARANTINE_DIR / "drebin_rejected.csv", index=False)

    print("staged rows:", len(good))
    print("quarantined rows:", len(quarantine))
    print("class counts:", good["is_malware"].value_counts().to_dict())
    return len(good), len(quarantine)


if __name__ == "__main__":
    stage()

    