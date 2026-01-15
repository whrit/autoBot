-- registry_schema.sql — Postgres DDL for Strategy Registry & Audit
-- Compatible with PostgreSQL 14+ (recommended 16)

BEGIN;

-- 1) Enums
DO $$ BEGIN
  CREATE TYPE run_type AS ENUM ('ingest', 'features', 'labels', 'backtest', 'optimize', 'shadow', 'paper', 'canary', 'live');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
  CREATE TYPE artifact_kind AS ENUM ('json_strategy', 'onnx_model', 'allocator', 'cost_model', 'risk_profile');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
  CREATE TYPE lifecycle_state AS ENUM ('candidate', 'shadow', 'paper', 'promoted', 'retired', 'rejected');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
  CREATE TYPE gate_result AS ENUM ('pass', 'fail', 'abort');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- 2) Core tables

-- Immutable dataset snapshot record (raw+features+labels manifests)
CREATE TABLE IF NOT EXISTS dataset_snapshots (
  snapshot_id         UUID PRIMARY KEY,
  created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
  name                TEXT NOT NULL,
  description         TEXT,
  lake_uri            TEXT NOT NULL, -- e.g., s3://lake/...
  manifest_json       JSONB NOT NULL, -- partitions, counts, checksums, schema versions
  code_sha            TEXT NOT NULL,  -- git sha of pipeline code
  UNIQUE(name, created_at)
);

-- Feature schema versions (decision frame schema)
CREATE TABLE IF NOT EXISTS feature_schemas (
  feature_schema_version TEXT PRIMARY KEY,
  created_at             TIMESTAMPTZ NOT NULL DEFAULT now(),
  schema_json            JSONB NOT NULL, -- ordered feature list and types
  notes                  TEXT
);

-- Cost model versions
CREATE TABLE IF NOT EXISTS cost_models (
  cost_model_version TEXT PRIMARY KEY,
  created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
  model_json         JSONB NOT NULL, -- coefficients, assumptions
  notes              TEXT
);

-- Risk profiles
CREATE TABLE IF NOT EXISTS risk_profiles (
  risk_profile_id TEXT PRIMARY KEY,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  profile_json    JSONB NOT NULL, -- limits, kill-switch thresholds, per-symbol caps
  notes           TEXT
);

-- Strategy definitions (logical identity)
CREATE TABLE IF NOT EXISTS strategies (
  strategy_id     TEXT PRIMARY KEY,         -- stable name: "trend_5m"
  family          TEXT NOT NULL,             -- trend, mean_reversion, microstructure, ml
  description     TEXT,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  owner           TEXT
);

-- Concrete artifacts (versioned instances, including ONNX + JSON)
CREATE TABLE IF NOT EXISTS artifacts (
  artifact_id            UUID PRIMARY KEY,
  created_at             TIMESTAMPTZ NOT NULL DEFAULT now(),
  strategy_id            TEXT REFERENCES strategies(strategy_id),
  kind                   artifact_kind NOT NULL,
  lifecycle              lifecycle_state NOT NULL DEFAULT 'candidate',
  feature_schema_version TEXT REFERENCES feature_schemas(feature_schema_version),
  cost_model_version     TEXT REFERENCES cost_models(cost_model_version),
  risk_profile_id        TEXT REFERENCES risk_profiles(risk_profile_id),
  label_horizon_sec      INT NOT NULL,
  decision_interval_sec  INT NOT NULL,
  symbols                TEXT[] NOT NULL,
  artifact_uri           TEXT NOT NULL,     -- s3://.../artifact.onnx or .json
  metadata_json          JSONB NOT NULL,    -- thresholds, input order, calibration, etc.
  code_sha               TEXT NOT NULL,
  parent_artifact_id     UUID REFERENCES artifacts(artifact_id)
);

CREATE INDEX IF NOT EXISTS artifacts_strategy_kind_idx
  ON artifacts(strategy_id, kind, created_at DESC);

CREATE INDEX IF NOT EXISTS artifacts_lifecycle_idx
  ON artifacts(lifecycle, created_at DESC);

-- Runs (pipeline executions)
CREATE TABLE IF NOT EXISTS runs (
  run_id              UUID PRIMARY KEY,
  created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
  run_type            run_type NOT NULL,
  dataset_snapshot_id UUID REFERENCES dataset_snapshots(snapshot_id),
  artifact_id         UUID REFERENCES artifacts(artifact_id),
  status              TEXT NOT NULL, -- 'running' | 'success' | 'failed'
  params_json         JSONB,
  metrics_json        JSONB,
  logs_uri            TEXT,
  duration_ms         BIGINT
);

CREATE INDEX IF NOT EXISTS runs_type_time_idx
  ON runs(run_type, created_at DESC);

-- Backtest / paper / live metrics time series (optional, summary kept in runs.metrics_json)
CREATE TABLE IF NOT EXISTS metric_series (
  series_id    UUID PRIMARY KEY,
  created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
  artifact_id  UUID REFERENCES artifacts(artifact_id),
  run_id       UUID REFERENCES runs(run_id),
  name         TEXT NOT NULL,          -- e.g. "equity_curve"
  uri          TEXT NOT NULL,          -- s3://.../equity_curve.csv
  metadata_json JSONB NOT NULL
);

-- Gate evaluations
CREATE TABLE IF NOT EXISTS gates (
  gate_id       UUID PRIMARY KEY,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  artifact_id   UUID REFERENCES artifacts(artifact_id),
  gate_name     TEXT NOT NULL,            -- "offline", "shadow", "paper", "canary"
  result        gate_result NOT NULL,
  thresholds_json JSONB NOT NULL,
  observed_json   JSONB NOT NULL,
  notes         TEXT
);

CREATE INDEX IF NOT EXISTS gates_artifact_time_idx
  ON gates(artifact_id, created_at DESC);

-- Promotions: points to the active set (single strategy or library set)
CREATE TABLE IF NOT EXISTS promotions (
  promotion_id  UUID PRIMARY KEY,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  name          TEXT NOT NULL, -- "paper_current", "live_current"
  environment   TEXT NOT NULL, -- paper|live
  artifact_ids  UUID[] NOT NULL,
  weights       DOUBLE PRECISION[] NOT NULL, -- aligned with artifact_ids
  allocator_artifact_id UUID REFERENCES artifacts(artifact_id),
  notes         TEXT,
  UNIQUE(name, environment)
);

-- Audit log (append-only)
CREATE TABLE IF NOT EXISTS audit_log (
  audit_id    UUID PRIMARY KEY,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
  actor       TEXT NOT NULL,         -- service name or user
  action      TEXT NOT NULL,         -- "PROMOTE", "ROLLBACK", "KILL_SWITCH", "REGISTER_ARTIFACT"
  entity_type TEXT NOT NULL,         -- "artifact", "promotion", "run", ...
  entity_id   TEXT NOT NULL,
  payload     JSONB NOT NULL
);

CREATE INDEX IF NOT EXISTS audit_time_idx
  ON audit_log(created_at DESC);

COMMIT;
