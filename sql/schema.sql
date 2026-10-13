CREATE SCHEMA IF NOT EXISTS repro;

CREATE TABLE IF NOT EXISTS repro.feature_names (
    position INTEGER PRIMARY KEY,
    name     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS repro.curated_samples (
    row_key    TEXT PRIMARY KEY,
    is_malware SMALLINT NOT NULL CHECK (is_malware IN (0, 1)),
    features   TEXT NOT NULL CHECK (length(features) = 215 AND features ~ '^[01]+$')
);

CREATE TABLE IF NOT EXISTS repro.pipeline_runs (
    run_id          TEXT PRIMARY KEY,
    started_at_utc  TIMESTAMPTZ NOT NULL,
    finished_at_utc TIMESTAMPTZ,
    status          TEXT NOT NULL CHECK (status IN ('running', 'success', 'failed')),
    source_sha256   TEXT NOT NULL,
    rows_read       INTEGER,
    rows_inserted   INTEGER,
    message         TEXT
);

