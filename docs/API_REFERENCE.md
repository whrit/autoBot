# Registry API Reference

> **Version:** 0.1.0
> **Base URL:** `http://localhost:8080`
> **Description:** System memory and audit log for autonomous equities trading engine

---

## Table of Contents

1. [Overview](#overview)
2. [Authentication](#authentication)
3. [Common Response Patterns](#common-response-patterns)
4. [Endpoints](#endpoints)
   - [Health Check](#health-check)
   - [Strategies](#strategies)
   - [Dataset Snapshots](#dataset-snapshots)
   - [Artifacts](#artifacts)
   - [Artifact Storage](#artifact-storage)
   - [Backtest Runs](#backtest-runs)
   - [Gates](#gates)
   - [Gate Evaluation](#gate-evaluation)
   - [Promotions](#promotions)
   - [Promotion State Machine](#promotion-state-machine)
   - [Audit Log](#audit-log)
5. [Data Models](#data-models)
6. [Enumerations](#enumerations)
7. [Error Handling](#error-handling)
8. [Python Client Examples](#python-client-examples)

---

## Overview

The Registry API provides a comprehensive system for managing trading strategies throughout their lifecycle. It supports:

- **Strategy Management**: CRUD operations for trading strategy configurations
- **Artifact Storage**: Upload/download ONNX models, JSON configs, parameters, and metrics
- **Backtest Tracking**: Record and query backtest run results with GPU training metadata
- **Quality Gates**: Configurable gates for strategy evaluation and promotion
- **Promotion Lifecycle**: State machine for strategy promotion (candidate -> shadow -> paper -> retired)
- **Audit Logging**: Comprehensive audit trail for all state changes

---

## Authentication

Currently, the API does not implement authentication. All endpoints are publicly accessible.

> **Note:** In production environments, implement appropriate authentication mechanisms such as API keys, OAuth 2.0, or JWT tokens.

---

## Common Response Patterns

### Pagination

List endpoints support pagination with the following query parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `skip` | integer | 0 | Number of records to skip |
| `limit` | integer | 100 | Maximum records to return (max: 1000) |

**Paginated Response Schema:**
```json
{
  "items": [...],
  "total": 150,
  "skip": 0,
  "limit": 100
}
```

### Error Response

All errors follow this format:

```json
{
  "detail": "Error description message"
}
```

---

## Endpoints

---

### Health Check

#### GET /health

Check API health status.

**Response:** `200 OK`
```json
{
  "status": "ok"
}
```

**curl Example:**
```bash
curl -X GET http://localhost:8080/health
```

---

### Strategies

Trading strategy configurations with promotion state management.

#### POST /strategies

Create a new strategy.

**Request Body:**
```json
{
  "name": "momentum_v1",
  "family": "momentum",
  "version": "1.0.0",
  "parameters": {
    "lookback_period": 20,
    "threshold": 0.02,
    "position_size": 0.1
  }
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Unique strategy name (1-255 chars) |
| `family` | string | Yes | Strategy family (1-100 chars) |
| `version` | string | Yes | Strategy version (1-50 chars) |
| `parameters` | object | Yes | Strategy parameters as JSON |

**Response:** `201 Created`
```json
{
  "id": 1,
  "name": "momentum_v1",
  "family": "momentum",
  "version": "1.0.0",
  "parameters": {
    "lookback_period": 20,
    "threshold": 0.02,
    "position_size": 0.1
  },
  "state": "candidate",
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": null
}
```

**Error Responses:**
- `400 Bad Request`: Strategy with name already exists

**curl Example:**
```bash
curl -X POST http://localhost:8080/strategies \
  -H "Content-Type: application/json" \
  -d '{
    "name": "momentum_v1",
    "family": "momentum",
    "version": "1.0.0",
    "parameters": {"lookback_period": 20, "threshold": 0.02}
  }'
```

---

#### GET /strategies/{strategy_id}

Get a strategy by ID.

**Path Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `strategy_id` | integer | Strategy ID |

**Response:** `200 OK`
```json
{
  "id": 1,
  "name": "momentum_v1",
  "family": "momentum",
  "version": "1.0.0",
  "parameters": {"lookback_period": 20},
  "state": "candidate",
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": null
}
```

**Error Responses:**
- `404 Not Found`: Strategy with id {strategy_id} not found

**curl Example:**
```bash
curl -X GET http://localhost:8080/strategies/1
```

---

#### GET /strategies

List all strategies with optional filtering.

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `skip` | integer | Records to skip (default: 0) |
| `limit` | integer | Max records (default: 100, max: 1000) |
| `family` | string | Filter by strategy family |
| `state` | string | Filter by promotion state |

**Response:** `200 OK`
```json
{
  "items": [
    {
      "id": 1,
      "name": "momentum_v1",
      "family": "momentum",
      "version": "1.0.0",
      "parameters": {},
      "state": "candidate",
      "created_at": "2024-01-15T10:30:00Z",
      "updated_at": null
    }
  ],
  "total": 1,
  "skip": 0,
  "limit": 100
}
```

**curl Example:**
```bash
# List all strategies
curl -X GET "http://localhost:8080/strategies"

# Filter by family and state
curl -X GET "http://localhost:8080/strategies?family=momentum&state=shadow"
```

---

#### PUT /strategies/{strategy_id}

Update a strategy.

**Path Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `strategy_id` | integer | Strategy ID |

**Request Body:**
```json
{
  "name": "momentum_v1_updated",
  "family": "momentum",
  "version": "1.1.0",
  "parameters": {"lookback_period": 25},
  "state": "shadow"
}
```

All fields are optional. Only provided fields will be updated.

**Response:** `200 OK`
```json
{
  "id": 1,
  "name": "momentum_v1_updated",
  "family": "momentum",
  "version": "1.1.0",
  "parameters": {"lookback_period": 25},
  "state": "shadow",
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-15T11:00:00Z"
}
```

**Error Responses:**
- `400 Bad Request`: Strategy with name already exists
- `404 Not Found`: Strategy not found

**curl Example:**
```bash
curl -X PUT http://localhost:8080/strategies/1 \
  -H "Content-Type: application/json" \
  -d '{"version": "1.1.0", "parameters": {"lookback_period": 25}}'
```

---

#### DELETE /strategies/{strategy_id}

Delete a strategy.

**Path Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `strategy_id` | integer | Strategy ID |

**Response:** `204 No Content`

**Error Responses:**
- `404 Not Found`: Strategy not found

**curl Example:**
```bash
curl -X DELETE http://localhost:8080/strategies/1
```

---

### Dataset Snapshots

Versioned market data snapshots for backtesting.

#### POST /dataset-snapshots

Create a new dataset snapshot.

**Request Body:**
```json
{
  "name": "sp500_2023",
  "version": "1.0.0",
  "start_date": "2023-01-01T00:00:00Z",
  "end_date": "2023-12-31T23:59:59Z",
  "symbols": ["AAPL", "GOOGL", "MSFT", "AMZN"],
  "feature_schema_version": "v2",
  "row_count": 1000000,
  "checksum": "abc123def456"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Unique snapshot name |
| `version` | string | Yes | Snapshot version |
| `start_date` | datetime | Yes | Data range start |
| `end_date` | datetime | Yes | Data range end |
| `symbols` | array[string] | Yes | List of symbols |
| `feature_schema_version` | string | No | Feature schema version |
| `row_count` | integer | No | Number of rows |
| `checksum` | string | No | Dataset checksum |

**Response:** `201 Created`
```json
{
  "id": 1,
  "name": "sp500_2023",
  "version": "1.0.0",
  "start_date": "2023-01-01T00:00:00Z",
  "end_date": "2023-12-31T23:59:59Z",
  "symbols": ["AAPL", "GOOGL", "MSFT", "AMZN"],
  "feature_schema_version": "v2",
  "row_count": 1000000,
  "checksum": "abc123def456",
  "created_at": "2024-01-15T10:30:00Z"
}
```

**curl Example:**
```bash
curl -X POST http://localhost:8080/dataset-snapshots \
  -H "Content-Type: application/json" \
  -d '{
    "name": "sp500_2023",
    "version": "1.0.0",
    "start_date": "2023-01-01T00:00:00Z",
    "end_date": "2023-12-31T23:59:59Z",
    "symbols": ["AAPL", "GOOGL", "MSFT"]
  }'
```

---

#### GET /dataset-snapshots/{snapshot_id}

Get a dataset snapshot by ID.

**Response:** `200 OK`
```json
{
  "id": 1,
  "name": "sp500_2023",
  "version": "1.0.0",
  "start_date": "2023-01-01T00:00:00Z",
  "end_date": "2023-12-31T23:59:59Z",
  "symbols": ["AAPL", "GOOGL"],
  "feature_schema_version": "v2",
  "row_count": 1000000,
  "checksum": "abc123",
  "created_at": "2024-01-15T10:30:00Z"
}
```

**curl Example:**
```bash
curl -X GET http://localhost:8080/dataset-snapshots/1
```

---

#### GET /dataset-snapshots

List all dataset snapshots.

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `skip` | integer | Records to skip |
| `limit` | integer | Max records |
| `version` | string | Filter by version |

**Response:** `200 OK`
```json
{
  "items": [...],
  "total": 10,
  "skip": 0,
  "limit": 100
}
```

**curl Example:**
```bash
curl -X GET "http://localhost:8080/dataset-snapshots?version=1.0.0"
```

---

#### PUT /dataset-snapshots/{snapshot_id}

Update a dataset snapshot.

**Request Body:**
```json
{
  "name": "sp500_2023_updated",
  "version": "1.0.1",
  "feature_schema_version": "v3",
  "row_count": 1100000,
  "checksum": "newchecksum"
}
```

All fields are optional.

**Response:** `200 OK`

**curl Example:**
```bash
curl -X PUT http://localhost:8080/dataset-snapshots/1 \
  -H "Content-Type: application/json" \
  -d '{"row_count": 1100000}'
```

---

#### DELETE /dataset-snapshots/{snapshot_id}

Delete a dataset snapshot.

**Response:** `204 No Content`

**curl Example:**
```bash
curl -X DELETE http://localhost:8080/dataset-snapshots/1
```

---

### Artifacts

Database-tracked artifacts with metadata.

#### POST /artifacts

Create a new artifact record.

**Request Body:**
```json
{
  "strategy_id": 1,
  "artifact_type": "onnx",
  "path": "s3://bucket/models/strategy_1_v1.onnx",
  "checksum": "sha256:abc123def456..."
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `strategy_id` | integer | Yes | Associated strategy ID |
| `artifact_type` | string | Yes | Type of artifact |
| `path` | string | Yes | File path or S3 key |
| `checksum` | string | Yes | SHA-256 checksum |

**Response:** `201 Created`
```json
{
  "id": 1,
  "strategy_id": 1,
  "artifact_type": "onnx",
  "path": "s3://bucket/models/strategy_1_v1.onnx",
  "checksum": "sha256:abc123def456...",
  "created_at": "2024-01-15T10:30:00Z"
}
```

**Error Responses:**
- `404 Not Found`: Strategy not found

**curl Example:**
```bash
curl -X POST http://localhost:8080/artifacts \
  -H "Content-Type: application/json" \
  -d '{
    "strategy_id": 1,
    "artifact_type": "onnx",
    "path": "/models/strategy_1.onnx",
    "checksum": "abc123"
  }'
```

---

#### GET /artifacts/{artifact_id}

Get an artifact by ID.

**Response:** `200 OK`
```json
{
  "id": 1,
  "strategy_id": 1,
  "artifact_type": "onnx",
  "path": "/models/strategy_1.onnx",
  "checksum": "abc123",
  "created_at": "2024-01-15T10:30:00Z"
}
```

**curl Example:**
```bash
curl -X GET http://localhost:8080/artifacts/1
```

---

#### GET /artifacts

List all artifacts.

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `skip` | integer | Records to skip |
| `limit` | integer | Max records |
| `strategy_id` | integer | Filter by strategy |
| `artifact_type` | string | Filter by type |

**Response:** `200 OK`
```json
{
  "items": [...],
  "total": 5,
  "skip": 0,
  "limit": 100
}
```

**curl Example:**
```bash
curl -X GET "http://localhost:8080/artifacts?strategy_id=1&artifact_type=onnx"
```

---

#### DELETE /artifacts/{artifact_id}

Delete an artifact from database and/or storage.

**Response:** `200 OK`
```json
{
  "deleted": true,
  "artifact_id": 1,
  "message": "Artifact deleted successfully"
}
```

**curl Example:**
```bash
curl -X DELETE http://localhost:8080/artifacts/1
```

---

### Artifact Storage

File-based artifact upload and download.

#### POST /strategies/{strategy_id}/artifacts

Upload an artifact file for a strategy.

**Path Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `strategy_id` | integer | Strategy ID |

**Form Data:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | file | Yes | Artifact file |
| `artifact_type` | string | Yes | Type: `onnx`, `json`, `params`, `metrics` |

**Response:** `201 Created`
```json
{
  "id": 1,
  "strategy_id": 1,
  "artifact_type": "onnx",
  "filename": "model.onnx",
  "checksum": "sha256:abc123def456...",
  "size_bytes": 1048576,
  "message": "Artifact uploaded successfully"
}
```

**Error Responses:**
- `404 Not Found`: Strategy not found
- `422 Unprocessable Content`: Invalid artifact type

**curl Example:**
```bash
curl -X POST http://localhost:8080/strategies/1/artifacts \
  -F "file=@model.onnx" \
  -F "artifact_type=onnx"
```

---

#### GET /strategies/{strategy_id}/artifacts

List all artifacts for a strategy.

**Response:** `200 OK`
```json
{
  "items": [
    {
      "id": 1,
      "strategy_id": 1,
      "artifact_type": "onnx",
      "path": "artifacts/1/model.onnx",
      "checksum": "sha256:abc123...",
      "created_at": "2024-01-15T10:30:00Z"
    }
  ],
  "total": 1,
  "skip": 0,
  "limit": 1
}
```

**curl Example:**
```bash
curl -X GET http://localhost:8080/strategies/1/artifacts
```

---

#### GET /artifacts/{artifact_id}/download

Download an artifact file.

**Response:** `200 OK`
- Content-Type: `application/octet-stream` (ONNX) or `application/json` (JSON/params/metrics)
- Headers:
  - `Content-Disposition: attachment; filename="model.onnx"`
  - `X-Checksum-SHA256: abc123...`

**curl Example:**
```bash
curl -X GET http://localhost:8080/artifacts/1/download -o model.onnx
```

---

#### DELETE /artifacts/{artifact_id}/file

Delete an artifact file from storage.

**Response:** `200 OK`
```json
{
  "deleted": true,
  "artifact_id": 1,
  "message": "Artifact deleted successfully"
}
```

**curl Example:**
```bash
curl -X DELETE http://localhost:8080/artifacts/1/file
```

---

### Backtest Runs

Backtest execution results with GPU training metadata.

#### POST /backtest-runs

Create a new backtest run.

**Request Body:**
```json
{
  "strategy_id": 1,
  "dataset_snapshot_id": 1,
  "sharpe": 1.25,
  "sortino": 1.8,
  "max_drawdown": 0.12,
  "profit_factor": 1.5,
  "win_rate": 0.55,
  "num_trades": 150,
  "started_at": "2024-01-15T08:00:00Z",
  "completed_at": "2024-01-15T08:30:00Z",
  "metadata": {"notes": "Initial backtest"},
  "training_device": "cuda",
  "gpu_backend": "pytorch_cuda",
  "cuda_version": "12.1",
  "driver_version": "535.104.05",
  "seed": 42,
  "determinism_flags": "CUDA_LAUNCH_BLOCKING=1",
  "training_time_sec": 1800.5
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `strategy_id` | integer | Yes | Associated strategy ID |
| `dataset_snapshot_id` | integer | No | Dataset snapshot ID |
| `sharpe` | float | No | Sharpe ratio |
| `sortino` | float | No | Sortino ratio |
| `max_drawdown` | float | No | Maximum drawdown (0-1) |
| `profit_factor` | float | No | Profit factor (>= 0) |
| `win_rate` | float | No | Win rate (0-1) |
| `num_trades` | integer | No | Number of trades |
| `started_at` | datetime | No | Backtest start time |
| `completed_at` | datetime | No | Backtest end time |
| `metadata` | object | No | Additional metadata |
| `training_device` | string | No | `cpu` or `cuda` |
| `gpu_backend` | string | No | `xgboost_gpu`, `pytorch_cuda`, or `cpu` |
| `cuda_version` | string | No | CUDA version (e.g., "12.1") |
| `driver_version` | string | No | GPU driver version |
| `seed` | integer | No | Random seed for reproducibility |
| `determinism_flags` | string | No | Determinism flags |
| `training_time_sec` | float | No | Training duration in seconds |

**Response:** `201 Created`
```json
{
  "id": 1,
  "strategy_id": 1,
  "dataset_snapshot_id": 1,
  "sharpe": 1.25,
  "sortino": 1.8,
  "max_drawdown": 0.12,
  "profit_factor": 1.5,
  "win_rate": 0.55,
  "num_trades": 150,
  "started_at": "2024-01-15T08:00:00Z",
  "completed_at": "2024-01-15T08:30:00Z",
  "metadata": {"notes": "Initial backtest"},
  "training_device": "cuda",
  "gpu_backend": "pytorch_cuda",
  "cuda_version": "12.1",
  "driver_version": "535.104.05",
  "seed": 42,
  "determinism_flags": "CUDA_LAUNCH_BLOCKING=1",
  "training_time_sec": 1800.5
}
```

**curl Example:**
```bash
curl -X POST http://localhost:8080/backtest-runs \
  -H "Content-Type: application/json" \
  -d '{
    "strategy_id": 1,
    "sharpe": 1.25,
    "max_drawdown": 0.12,
    "num_trades": 150,
    "training_device": "cuda",
    "seed": 42
  }'
```

---

#### GET /backtest-runs/{run_id}

Get a backtest run by ID.

**Response:** `200 OK`

**curl Example:**
```bash
curl -X GET http://localhost:8080/backtest-runs/1
```

---

#### GET /backtest-runs

List all backtest runs.

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `skip` | integer | Records to skip |
| `limit` | integer | Max records |
| `strategy_id` | integer | Filter by strategy |
| `dataset_snapshot_id` | integer | Filter by dataset snapshot |
| `training_device` | string | Filter by training device (`cpu`, `cuda`) |

**Response:** `200 OK`
```json
{
  "items": [...],
  "total": 25,
  "skip": 0,
  "limit": 100
}
```

**curl Example:**
```bash
curl -X GET "http://localhost:8080/backtest-runs?strategy_id=1&training_device=cuda"
```

---

### Gates

Quality gates for strategy promotion.

#### POST /gates

Create a new gate.

**Request Body:**
```json
{
  "name": "min_sharpe_gate",
  "description": "Minimum Sharpe ratio requirement",
  "gate_type": "backtest",
  "criteria": {
    "metric": "sharpe",
    "operator": ">=",
    "threshold": 0.5
  },
  "is_active": true
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Unique gate name |
| `description` | string | No | Gate description |
| `gate_type` | string | Yes | Type: `backtest`, `shadow`, `paper` |
| `criteria` | object | Yes | Gate criteria as JSON |
| `is_active` | boolean | No | Whether gate is active (default: true) |

**Response:** `201 Created`
```json
{
  "id": 1,
  "name": "min_sharpe_gate",
  "description": "Minimum Sharpe ratio requirement",
  "gate_type": "backtest",
  "criteria": {
    "metric": "sharpe",
    "operator": ">=",
    "threshold": 0.5
  },
  "is_active": true,
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": null
}
```

**curl Example:**
```bash
curl -X POST http://localhost:8080/gates \
  -H "Content-Type: application/json" \
  -d '{
    "name": "min_sharpe_gate",
    "gate_type": "backtest",
    "criteria": {"metric": "sharpe", "threshold": 0.5}
  }'
```

---

#### GET /gates/{gate_id}

Get a gate by ID.

**Response:** `200 OK`

**curl Example:**
```bash
curl -X GET http://localhost:8080/gates/1
```

---

#### GET /gates

List all gates.

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `skip` | integer | Records to skip |
| `limit` | integer | Max records |
| `gate_type` | string | Filter by gate type |
| `is_active` | boolean | Filter by active status |

**Response:** `200 OK`
```json
{
  "items": [...],
  "total": 5,
  "skip": 0,
  "limit": 100
}
```

**curl Example:**
```bash
curl -X GET "http://localhost:8080/gates?gate_type=backtest&is_active=true"
```

---

#### PUT /gates/{gate_id}

Update a gate.

**Request Body:**
```json
{
  "name": "updated_gate",
  "description": "Updated description",
  "criteria": {"threshold": 0.6},
  "is_active": false
}
```

**Response:** `200 OK`

**curl Example:**
```bash
curl -X PUT http://localhost:8080/gates/1 \
  -H "Content-Type: application/json" \
  -d '{"is_active": false}'
```

---

#### DELETE /gates/{gate_id}

Delete a gate.

**Response:** `204 No Content`

**curl Example:**
```bash
curl -X DELETE http://localhost:8080/gates/1
```

---

### Gate Evaluation

Quality gate evaluation for strategies.

#### GET /gates/config

Get current gate evaluation configuration.

**Response:** `200 OK`
```json
{
  "gates": [
    {
      "gate_type": "min_sharpe",
      "threshold": 0.5,
      "required": true
    },
    {
      "gate_type": "max_drawdown",
      "threshold": 0.15,
      "required": true
    },
    {
      "gate_type": "min_trades",
      "threshold": 30,
      "required": true
    },
    {
      "gate_type": "min_win_rate",
      "threshold": 0.45,
      "required": false
    }
  ],
  "total": 4
}
```

**curl Example:**
```bash
curl -X GET http://localhost:8080/gates/config
```

---

#### PUT /gates/config

Update gate evaluation configuration.

**Request Body:**
```json
{
  "gates": [
    {"gate_type": "min_sharpe", "threshold": 0.6, "required": true},
    {"gate_type": "max_drawdown", "threshold": 0.10, "required": true},
    {"gate_type": "min_trades", "threshold": 50, "required": true}
  ]
}
```

**Valid Gate Types:**
- `min_sharpe` - Minimum Sharpe ratio
- `max_drawdown` - Maximum drawdown
- `min_trades` - Minimum number of trades
- `min_win_rate` - Minimum win rate
- `min_profit_factor` - Minimum profit factor
- `cost_sensitivity` - Cost sensitivity (lower is better)
- `min_sortino` - Minimum Sortino ratio

**Response:** `200 OK`
```json
{
  "updated": true,
  "message": "Gate configuration updated with 3 gates",
  "gates": [...]
}
```

**curl Example:**
```bash
curl -X PUT http://localhost:8080/gates/config \
  -H "Content-Type: application/json" \
  -d '{
    "gates": [
      {"gate_type": "min_sharpe", "threshold": 0.6, "required": true},
      {"gate_type": "max_drawdown", "threshold": 0.10, "required": true}
    ]
  }'
```

---

#### POST /strategies/{strategy_id}/evaluate-gates

Evaluate a strategy against quality gates.

**Path Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `strategy_id` | integer | Strategy ID |

**Request Body:**
```json
{
  "sharpe": 0.75,
  "sortino": 1.2,
  "max_drawdown": 0.08,
  "profit_factor": 1.5,
  "win_rate": 0.52,
  "num_trades": 120,
  "cost_sensitivity": 0.01
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `sharpe` | float | Yes | Sharpe ratio |
| `sortino` | float | Yes | Sortino ratio |
| `max_drawdown` | float | Yes | Max drawdown (0-1) |
| `profit_factor` | float | Yes | Profit factor (>= 0) |
| `win_rate` | float | Yes | Win rate (0-1) |
| `num_trades` | integer | Yes | Number of trades |
| `cost_sensitivity` | float | No | Cost sensitivity metric |

**Response:** `200 OK`
```json
{
  "strategy_id": 1,
  "passed": true,
  "passed_count": 4,
  "total_count": 4,
  "required_passed": 3,
  "required_total": 3,
  "optional_passed": 1,
  "optional_total": 1,
  "results": [
    {
      "gate_type": "min_sharpe",
      "passed": true,
      "actual_value": 0.75,
      "threshold": 0.5,
      "message": "min_sharpe passed: 0.7500 >= 0.5000",
      "required": true
    },
    {
      "gate_type": "max_drawdown",
      "passed": true,
      "actual_value": 0.08,
      "threshold": 0.15,
      "message": "max_drawdown passed: 0.0800 <= 0.1500",
      "required": true
    }
  ]
}
```

**curl Example:**
```bash
curl -X POST http://localhost:8080/strategies/1/evaluate-gates \
  -H "Content-Type: application/json" \
  -d '{
    "sharpe": 0.75,
    "sortino": 1.2,
    "max_drawdown": 0.08,
    "profit_factor": 1.5,
    "win_rate": 0.52,
    "num_trades": 120
  }'
```

---

### Promotions

Promotion records for strategy state transitions.

#### POST /promotions

Create a new promotion record.

**Request Body:**
```json
{
  "strategy_id": 1,
  "gate_id": 1,
  "from_state": "candidate",
  "to_state": "shadow",
  "passed": true,
  "evaluation_results": {
    "sharpe": {"passed": true, "value": 0.75},
    "max_drawdown": {"passed": true, "value": 0.08}
  },
  "promoted_by": "system"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `strategy_id` | integer | Yes | Strategy being promoted |
| `gate_id` | integer | Yes | Gate used for evaluation |
| `from_state` | string | Yes | Previous state |
| `to_state` | string | Yes | New state |
| `passed` | boolean | Yes | Whether gate was passed |
| `evaluation_results` | object | No | Evaluation details |
| `promoted_by` | string | No | User or system |

**Response:** `201 Created`
```json
{
  "id": 1,
  "strategy_id": 1,
  "gate_id": 1,
  "from_state": "candidate",
  "to_state": "shadow",
  "passed": true,
  "evaluation_results": {...},
  "promoted_at": "2024-01-15T10:30:00Z",
  "promoted_by": "system"
}
```

**curl Example:**
```bash
curl -X POST http://localhost:8080/promotions \
  -H "Content-Type: application/json" \
  -d '{
    "strategy_id": 1,
    "gate_id": 1,
    "from_state": "candidate",
    "to_state": "shadow",
    "passed": true
  }'
```

---

#### GET /promotions/{promotion_id}

Get a promotion by ID.

**Response:** `200 OK`

**curl Example:**
```bash
curl -X GET http://localhost:8080/promotions/1
```

---

#### GET /promotions

List all promotions.

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `skip` | integer | Records to skip |
| `limit` | integer | Max records |
| `strategy_id` | integer | Filter by strategy |
| `gate_id` | integer | Filter by gate |
| `passed` | boolean | Filter by pass status |

**Response:** `200 OK`
```json
{
  "items": [...],
  "total": 10,
  "skip": 0,
  "limit": 100
}
```

**curl Example:**
```bash
curl -X GET "http://localhost:8080/promotions?strategy_id=1&passed=true"
```

---

### Promotion State Machine

Strategy lifecycle state transitions.

#### POST /strategies/{strategy_id}/promote

Promote a strategy to a target state.

**Valid Transitions:**
- `candidate` -> `shadow` (requires passing backtest gates)
- `shadow` -> `paper` (requires successful shadow period)
- Any state -> `retired`

**Request Body:**
```json
{
  "target_state": "shadow"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `target_state` | string | Yes | Target state: `shadow`, `paper`, or `retired` |

**Response:** `200 OK`
```json
{
  "success": true,
  "from_state": "candidate",
  "to_state": "shadow",
  "message": "Successfully promoted from candidate to shadow",
  "requirements_met": {
    "min_sharpe": true,
    "max_drawdown": true,
    "min_trades": true
  }
}
```

**Error Responses:**
- `400 Bad Request`: Invalid transition or requirements not met
- `404 Not Found`: Strategy not found

**curl Example:**
```bash
curl -X POST http://localhost:8080/strategies/1/promote \
  -H "Content-Type: application/json" \
  -d '{"target_state": "shadow"}'
```

---

#### POST /strategies/{strategy_id}/demote

Demote a strategy to a lower state.

**Valid Demotions:**
- `paper` -> `shadow`
- `shadow` -> `candidate`

**Request Body:**
```json
{
  "reason": "Performance degradation detected"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `reason` | string | Yes | Reason for demotion |

**Response:** `200 OK`
```json
{
  "success": true,
  "from_state": "paper",
  "to_state": "shadow",
  "message": "Demoted from paper to shadow: Performance degradation detected",
  "requirements_met": {}
}
```

**curl Example:**
```bash
curl -X POST http://localhost:8080/strategies/1/demote \
  -H "Content-Type: application/json" \
  -d '{"reason": "Performance degradation detected"}'
```

---

#### POST /strategies/{strategy_id}/retire

Retire a strategy (terminal state).

**Request Body:**
```json
{
  "reason": "Strategy no longer profitable"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `reason` | string | Yes | Reason for retirement |

**Response:** `200 OK`
```json
{
  "success": true,
  "from_state": "paper",
  "to_state": "retired",
  "message": "Retired from paper: Strategy no longer profitable",
  "requirements_met": {}
}
```

**curl Example:**
```bash
curl -X POST http://localhost:8080/strategies/1/retire \
  -H "Content-Type: application/json" \
  -d '{"reason": "Strategy no longer profitable"}'
```

---

#### GET /strategies/{strategy_id}/promotion-history

Get promotion history for a strategy.

**Response:** `200 OK`
```json
[
  {
    "id": 1,
    "event_type": "promotion_approved",
    "entity_type": "strategy",
    "entity_id": 1,
    "user_id": "system",
    "old_value": {"state": "candidate"},
    "new_value": {"state": "shadow"},
    "message": "Strategy promoted from candidate to shadow",
    "timestamp": "2024-01-15T10:30:00Z",
    "metadata": {"requirements_met": {"min_sharpe": true}}
  }
]
```

**curl Example:**
```bash
curl -X GET http://localhost:8080/strategies/1/promotion-history
```

---

### Audit Log

Comprehensive audit trail for all state changes.

#### GET /audit

Query the audit log.

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `event_type` | string | Filter by event type |
| `entity_type` | string | Filter by entity type |
| `limit` | integer | Max entries (default: 100, max: 1000) |

**Valid Event Types:**
- `strategy_created`, `strategy_updated`, `strategy_deleted`
- `artifact_created`, `artifact_updated`, `artifact_deleted`
- `promotion_requested`, `promotion_approved`, `promotion_rejected`
- `demotion`, `retirement`
- `artifact_uploaded`, `gate_evaluation`
- `backtest_started`, `backtest_completed`
- `alert_triggered`, `rollback_triggered`
- `run_started`, `run_completed`

**Response:** `200 OK`
```json
[
  {
    "id": 1,
    "event_type": "strategy_created",
    "entity_type": "strategy",
    "entity_id": 1,
    "user_id": "system",
    "old_value": null,
    "new_value": {"name": "momentum_v1", "state": "candidate"},
    "message": "Strategy created: momentum_v1",
    "timestamp": "2024-01-15T10:30:00Z",
    "metadata": {}
  }
]
```

**curl Example:**
```bash
# Query all audit entries
curl -X GET http://localhost:8080/audit

# Filter by event type
curl -X GET "http://localhost:8080/audit?event_type=strategy_created&limit=50"

# Filter by entity type
curl -X GET "http://localhost:8080/audit?entity_type=strategy"
```

---

#### GET /strategies/{strategy_id}/audit

Get all audit entries for a strategy.

**Response:** `200 OK`
```json
[
  {
    "id": 1,
    "event_type": "strategy_created",
    "entity_type": "strategy",
    "entity_id": 1,
    "user_id": "system",
    "old_value": null,
    "new_value": {"name": "momentum_v1"},
    "message": "Strategy created",
    "timestamp": "2024-01-15T10:30:00Z",
    "metadata": {}
  },
  {
    "id": 2,
    "event_type": "promotion_approved",
    "entity_type": "strategy",
    "entity_id": 1,
    "user_id": "system",
    "old_value": {"state": "candidate"},
    "new_value": {"state": "shadow"},
    "message": "Strategy promoted from candidate to shadow",
    "timestamp": "2024-01-15T11:00:00Z",
    "metadata": {}
  }
]
```

**curl Example:**
```bash
curl -X GET http://localhost:8080/strategies/1/audit
```

---

## Data Models

### Strategy

| Field | Type | Description |
|-------|------|-------------|
| `id` | integer | Unique identifier |
| `name` | string | Unique strategy name |
| `family` | string | Strategy family |
| `version` | string | Strategy version |
| `parameters` | object | Strategy parameters |
| `state` | PromotionState | Current promotion state |
| `created_at` | datetime | Creation timestamp |
| `updated_at` | datetime | Last update timestamp |

### Artifact

| Field | Type | Description |
|-------|------|-------------|
| `id` | integer | Unique identifier |
| `strategy_id` | integer | Associated strategy |
| `artifact_type` | string | Type of artifact |
| `path` | string | File path or S3 key |
| `checksum` | string | SHA-256 checksum |
| `created_at` | datetime | Creation timestamp |

### DatasetSnapshot

| Field | Type | Description |
|-------|------|-------------|
| `id` | integer | Unique identifier |
| `name` | string | Unique snapshot name |
| `version` | string | Snapshot version |
| `start_date` | datetime | Data range start |
| `end_date` | datetime | Data range end |
| `symbols` | array[string] | List of symbols |
| `feature_schema_version` | string | Feature schema version |
| `row_count` | integer | Number of rows |
| `checksum` | string | Dataset checksum |
| `created_at` | datetime | Creation timestamp |

### BacktestRun

| Field | Type | Description |
|-------|------|-------------|
| `id` | integer | Unique identifier |
| `strategy_id` | integer | Associated strategy |
| `dataset_snapshot_id` | integer | Associated dataset |
| `sharpe` | float | Sharpe ratio |
| `sortino` | float | Sortino ratio |
| `max_drawdown` | float | Maximum drawdown |
| `profit_factor` | float | Profit factor |
| `win_rate` | float | Win rate |
| `num_trades` | integer | Number of trades |
| `started_at` | datetime | Start time |
| `completed_at` | datetime | End time |
| `metadata` | object | Additional metadata |
| `training_device` | string | `cpu` or `cuda` |
| `gpu_backend` | string | GPU backend |
| `cuda_version` | string | CUDA version |
| `driver_version` | string | Driver version |
| `seed` | integer | Random seed |
| `determinism_flags` | string | Determinism flags |
| `training_time_sec` | float | Training time |

### Gate

| Field | Type | Description |
|-------|------|-------------|
| `id` | integer | Unique identifier |
| `name` | string | Unique gate name |
| `description` | string | Gate description |
| `gate_type` | string | Type of gate |
| `criteria` | object | Gate criteria |
| `is_active` | boolean | Active status |
| `created_at` | datetime | Creation timestamp |
| `updated_at` | datetime | Last update timestamp |

### Promotion

| Field | Type | Description |
|-------|------|-------------|
| `id` | integer | Unique identifier |
| `strategy_id` | integer | Associated strategy |
| `gate_id` | integer | Associated gate |
| `from_state` | PromotionState | Previous state |
| `to_state` | PromotionState | New state |
| `passed` | boolean | Gate passed status |
| `evaluation_results` | object | Evaluation details |
| `promoted_at` | datetime | Promotion timestamp |
| `promoted_by` | string | User or system |

### AuditEntry

| Field | Type | Description |
|-------|------|-------------|
| `id` | integer | Unique identifier |
| `event_type` | string | Event type |
| `entity_type` | string | Entity type |
| `entity_id` | integer | Entity ID |
| `user_id` | string | User ID |
| `old_value` | object | Previous value |
| `new_value` | object | New value |
| `message` | string | Event description |
| `timestamp` | datetime | Event timestamp |
| `metadata` | object | Additional metadata |

---

## Enumerations

### PromotionState

| Value | Description |
|-------|-------------|
| `candidate` | Initial state for new strategies |
| `shadow` | Shadow mode (no real execution) |
| `paper` | Paper trading mode |
| `retired` | Terminal state (decommissioned) |

### ArtifactType

| Value | Description |
|-------|-------------|
| `onnx` | ONNX model files |
| `json` | JSON configuration files |
| `params` | Parameter files |
| `metrics` | Metrics files |

### GateType

| Value | Description |
|-------|-------------|
| `min_sharpe` | Minimum Sharpe ratio |
| `max_drawdown` | Maximum drawdown |
| `min_trades` | Minimum number of trades |
| `min_win_rate` | Minimum win rate |
| `min_profit_factor` | Minimum profit factor |
| `cost_sensitivity` | Cost sensitivity |
| `min_sortino` | Minimum Sortino ratio |

### AuditEventType

| Value | Description |
|-------|-------------|
| `strategy_created` | Strategy created |
| `strategy_updated` | Strategy updated |
| `strategy_deleted` | Strategy deleted |
| `artifact_created` | Artifact created |
| `artifact_updated` | Artifact updated |
| `artifact_deleted` | Artifact deleted |
| `artifact_uploaded` | Artifact uploaded |
| `promotion_requested` | Promotion requested |
| `promotion_approved` | Promotion approved |
| `promotion_rejected` | Promotion rejected |
| `demotion` | Strategy demoted |
| `retirement` | Strategy retired |
| `gate_evaluation` | Gate evaluation |
| `backtest_started` | Backtest started |
| `backtest_completed` | Backtest completed |
| `alert_triggered` | Alert triggered |
| `rollback_triggered` | Rollback triggered |
| `run_started` | Run started |
| `run_completed` | Run completed |

---

## Error Handling

### HTTP Status Codes

| Code | Description |
|------|-------------|
| `200` | Success |
| `201` | Created |
| `204` | No Content (successful deletion) |
| `400` | Bad Request (invalid input, duplicate name) |
| `404` | Not Found |
| `422` | Unprocessable Content (validation error) |
| `500` | Internal Server Error |

### Error Response Format

```json
{
  "detail": "Error message describing what went wrong"
}
```

---

## Python Client Examples

### Using httpx (Async)

```python
import httpx
import asyncio

BASE_URL = "http://localhost:8080"

async def main():
    async with httpx.AsyncClient(base_url=BASE_URL) as client:
        # Create a strategy
        strategy_data = {
            "name": "momentum_v1",
            "family": "momentum",
            "version": "1.0.0",
            "parameters": {"lookback": 20, "threshold": 0.02}
        }
        response = await client.post("/strategies", json=strategy_data)
        strategy = response.json()
        print(f"Created strategy: {strategy['id']}")

        # Create a backtest run
        backtest_data = {
            "strategy_id": strategy["id"],
            "sharpe": 1.25,
            "sortino": 1.8,
            "max_drawdown": 0.08,
            "profit_factor": 1.5,
            "win_rate": 0.55,
            "num_trades": 150,
            "training_device": "cuda",
            "seed": 42
        }
        response = await client.post("/backtest-runs", json=backtest_data)
        backtest = response.json()
        print(f"Created backtest run: {backtest['id']}")

        # Evaluate gates
        metrics = {
            "sharpe": 1.25,
            "sortino": 1.8,
            "max_drawdown": 0.08,
            "profit_factor": 1.5,
            "win_rate": 0.55,
            "num_trades": 150
        }
        response = await client.post(
            f"/strategies/{strategy['id']}/evaluate-gates",
            json=metrics
        )
        evaluation = response.json()
        print(f"Gates passed: {evaluation['passed']}")

        # Promote to shadow if gates passed
        if evaluation["passed"]:
            response = await client.post(
                f"/strategies/{strategy['id']}/promote",
                json={"target_state": "shadow"}
            )
            result = response.json()
            print(f"Promotion result: {result['message']}")

        # Get audit history
        response = await client.get(f"/strategies/{strategy['id']}/audit")
        audit_entries = response.json()
        print(f"Audit entries: {len(audit_entries)}")

asyncio.run(main())
```

### Using requests (Sync)

```python
import requests

BASE_URL = "http://localhost:8080"

def main():
    # Create a strategy
    strategy_data = {
        "name": "mean_reversion_v1",
        "family": "mean_reversion",
        "version": "1.0.0",
        "parameters": {"window": 30, "z_score_threshold": 2.0}
    }
    response = requests.post(f"{BASE_URL}/strategies", json=strategy_data)
    response.raise_for_status()
    strategy = response.json()
    print(f"Created strategy: {strategy['id']}")

    # Upload an artifact
    with open("model.onnx", "rb") as f:
        files = {"file": ("model.onnx", f)}
        data = {"artifact_type": "onnx"}
        response = requests.post(
            f"{BASE_URL}/strategies/{strategy['id']}/artifacts",
            files=files,
            data=data
        )
        response.raise_for_status()
        artifact = response.json()
        print(f"Uploaded artifact: {artifact['id']}, checksum: {artifact['checksum']}")

    # Download the artifact
    response = requests.get(f"{BASE_URL}/artifacts/{artifact['id']}/download")
    response.raise_for_status()
    with open("downloaded_model.onnx", "wb") as f:
        f.write(response.content)
    print(f"Downloaded artifact, checksum: {response.headers['X-Checksum-SHA256']}")

    # List strategies with filtering
    response = requests.get(
        f"{BASE_URL}/strategies",
        params={"family": "mean_reversion", "state": "candidate"}
    )
    response.raise_for_status()
    strategies = response.json()
    print(f"Found {strategies['total']} strategies")

    # Query audit log
    response = requests.get(
        f"{BASE_URL}/audit",
        params={"event_type": "strategy_created", "limit": 10}
    )
    response.raise_for_status()
    audit_entries = response.json()
    print(f"Found {len(audit_entries)} audit entries")

if __name__ == "__main__":
    main()
```

### Strategy Lifecycle Management

```python
import requests

BASE_URL = "http://localhost:8080"

def manage_strategy_lifecycle(strategy_id: int):
    """Demonstrate full strategy lifecycle management."""

    # 1. Get current strategy state
    response = requests.get(f"{BASE_URL}/strategies/{strategy_id}")
    strategy = response.json()
    print(f"Strategy '{strategy['name']}' is in state: {strategy['state']}")

    # 2. Evaluate gates
    metrics = {
        "sharpe": 0.75,
        "sortino": 1.2,
        "max_drawdown": 0.08,
        "profit_factor": 1.5,
        "win_rate": 0.52,
        "num_trades": 120
    }
    response = requests.post(
        f"{BASE_URL}/strategies/{strategy_id}/evaluate-gates",
        json=metrics
    )
    evaluation = response.json()

    print(f"Gate evaluation: {'PASSED' if evaluation['passed'] else 'FAILED'}")
    for result in evaluation["results"]:
        status = "PASS" if result["passed"] else "FAIL"
        print(f"  - {result['gate_type']}: {status} "
              f"(actual: {result['actual_value']:.4f}, threshold: {result['threshold']:.4f})")

    # 3. Promote to shadow if gates passed
    if evaluation["passed"] and strategy["state"] == "candidate":
        response = requests.post(
            f"{BASE_URL}/strategies/{strategy_id}/promote",
            json={"target_state": "shadow"}
        )
        result = response.json()
        print(f"Promotion: {result['message']}")

    # 4. Simulate shadow period success and promote to paper
    if strategy["state"] == "shadow":
        response = requests.post(
            f"{BASE_URL}/strategies/{strategy_id}/promote",
            json={"target_state": "paper"}
        )
        result = response.json()
        print(f"Promotion to paper: {result['message']}")

    # 5. Check promotion history
    response = requests.get(f"{BASE_URL}/strategies/{strategy_id}/promotion-history")
    history = response.json()
    print(f"\nPromotion history ({len(history)} events):")
    for entry in history:
        print(f"  - {entry['timestamp']}: {entry['event_type']}")
        print(f"    {entry['message']}")

# Example: Demote a strategy
def demote_strategy(strategy_id: int, reason: str):
    response = requests.post(
        f"{BASE_URL}/strategies/{strategy_id}/demote",
        json={"reason": reason}
    )
    if response.status_code == 200:
        result = response.json()
        print(f"Demotion successful: {result['from_state']} -> {result['to_state']}")
    else:
        print(f"Demotion failed: {response.json()['detail']}")

# Example: Retire a strategy
def retire_strategy(strategy_id: int, reason: str):
    response = requests.post(
        f"{BASE_URL}/strategies/{strategy_id}/retire",
        json={"reason": reason}
    )
    if response.status_code == 200:
        result = response.json()
        print(f"Retirement successful: {result['message']}")
    else:
        print(f"Retirement failed: {response.json()['detail']}")
```

---

## Environment Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `sqlite:///./registry.db` | Database connection string |
| `ARTIFACT_STORAGE_PATH` | `./artifacts` | Path for artifact storage |

---

## Running the API

```bash
# Install dependencies
pip install fastapi uvicorn sqlalchemy pydantic

# Run the server
cd services/registry_api_py
python -m registry_api_py

# Or with uvicorn directly
uvicorn registry_api_py.app:app --host 0.0.0.0 --port 8080 --reload
```

---

## OpenAPI Specification

The full OpenAPI 3.0 specification is available at:
- **Swagger UI:** `http://localhost:8080/docs`
- **ReDoc:** `http://localhost:8080/redoc`
- **OpenAPI JSON:** `http://localhost:8080/openapi.json`
