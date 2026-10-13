import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import psycopg
from dotenv import load_dotenv

from src.prepare import sha256_of, DATA

CURATED = Path("data/curated/drebin_curated.csv")


def connect():
    load_dotenv()
    return psycopg.connect(
        host=os.environ["POSTGRES_HOST"],
        port=int(os.environ["POSTGRES_PORT"]),
        dbname=os.environ["POSTGRES_DB"],
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
    )


def record_failure(run_id, started, source_hash, message):
    with connect() as conn:
        conn.execute(
            "INSERT INTO repro.pipeline_runs "
            "(run_id, started_at_utc, finished_at_utc, status, source_sha256, message) "
            "VALUES (%s, %s, %s, 'failed', %s, %s)",
            (run_id, started, datetime.now(timezone.utc), source_hash, message),
        )


def load_curated():
    started = datetime.now(timezone.utc)
    run_id = started.strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    source_hash = sha256_of(DATA)

    df = pd.read_csv(CURATED)
    feature_cols = [c for c in df.columns if c not in ("row_key", "is_malware")]
    bits = df[feature_cols].astype(str).agg("".join, axis=1)
    rows = list(zip(df["row_key"].tolist(),
                    df["is_malware"].astype(int).tolist(),
                    bits.tolist()))

    try:
        with connect() as conn:
            with conn.cursor() as cur:
                before = cur.execute("SELECT count(*) FROM repro.curated_samples").fetchone()[0]
                cur.executemany(
                    "INSERT INTO repro.feature_names (position, name) VALUES (%s, %s) "
                    "ON CONFLICT DO NOTHING",
                    list(enumerate(feature_cols)),
                )
                cur.executemany(
                    "INSERT INTO repro.curated_samples (row_key, is_malware, features) "
                    "VALUES (%s, %s, %s) ON CONFLICT (row_key) DO NOTHING",
                    rows,
                )
                after = cur.execute("SELECT count(*) FROM repro.curated_samples").fetchone()[0]
                inserted = after - before
                cur.execute(
                    "INSERT INTO repro.pipeline_runs (run_id, started_at_utc, finished_at_utc, "
                    "status, source_sha256, rows_read, rows_inserted) "
                    "VALUES (%s, %s, %s, 'success', %s, %s, %s)",
                    (run_id, started, datetime.now(timezone.utc), source_hash, len(rows), inserted),
                )
    except Exception as exc:
        record_failure(run_id, started, source_hash, str(exc))
        raise

    print("run:", run_id, "rows read:", len(rows), "inserted:", inserted, "table total:", after)
    return inserted


if __name__ == "__main__":
    load_curated()

    