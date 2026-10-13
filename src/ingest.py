import hashlib
import json
from pathlib import Path

import pandas as pd

from src.prepare import DATA, MANIFEST, sha256_of, expected_hash

RAW_DIR = Path("data/raw")
CHECKPOINT = RAW_DIR / "_checkpoint.json"


def row_keys(df):
    """One key per row: a hash of all its values, plus a copy number."""
    joined = df.apply("|".join, axis=1)
    base = joined.map(lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest())
    copy_no = base.groupby(base).cumcount() + 1
    return base + "-" + copy_no.astype(str)


def ingest():
    current_hash = sha256_of(DATA)
    if current_hash.lower() != expected_hash(MANIFEST).lower():
        raise SystemExit("Data hash does not match the manifest. Stop.")

    if CHECKPOINT.exists():
        saved = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
        batch_ok = (RAW_DIR / saved["batch_file"]).exists()
        if saved["sha256"] == current_hash and batch_ok:
            print("already ingested, new rows: 0")
            return 0

    df = pd.read_csv(DATA, dtype=str)
    df["row_key"] = row_keys(df)
    assert df["row_key"].nunique() == len(df), "row keys are not unique"

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    batch = RAW_DIR / f"drebin_{current_hash[:12]}.csv"
    df.to_csv(batch, index=False)

    CHECKPOINT.write_text(json.dumps({
        "sha256": current_hash,
        "rows": len(df),
        "unique_keys": int(df["row_key"].nunique()),
        "batch_file": batch.name,
    }, indent=2), encoding="utf-8")
    print("ingested rows:", len(df))
    return len(df)


if __name__ == "__main__":
    ingest()