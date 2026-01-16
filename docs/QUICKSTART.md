# Quickstart Guide: autoBot Phase 1 Trading Engine

Get the autoBot autonomous equities taker engine running locally in under 30 minutes.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Clone and Initial Setup](#clone-and-initial-setup)
3. [Infrastructure Setup](#infrastructure-setup)
4. [Environment Configuration](#environment-configuration)
5. [Install Dependencies](#install-dependencies)
6. [Database Initialization](#database-initialization)
7. [Run Your First Data Ingestion](#run-your-first-data-ingestion)
8. [Start the Registry API](#start-the-registry-api)
9. [Verify the Setup](#verify-the-setup)
10. [Next Steps](#next-steps)
11. [Troubleshooting](#troubleshooting)

---

## Prerequisites

Before starting, ensure you have the following installed:

| Tool | Minimum Version | Check Command | Installation |
|------|-----------------|---------------|--------------|
| Python | 3.12+ | `python --version` | [python.org](https://www.python.org/downloads/) |
| UV | Latest | `uv --version` | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| Docker | 24+ | `docker --version` | [docker.com](https://docs.docker.com/get-docker/) |
| Docker Compose | 2.0+ | `docker compose version` | Included with Docker Desktop |
| Git | 2.0+ | `git --version` | [git-scm.com](https://git-scm.com/downloads/) |

### Alpaca Account Setup

You need an Alpaca account for market data access:

1. Sign up at [alpaca.markets](https://alpaca.markets/)
2. Navigate to **Paper Trading** dashboard
3. Generate API keys (you will get `API Key ID` and `Secret Key`)
4. Note: Free tier uses IEX feed; paid subscription enables SIP feed

**Verification:**

```bash
# Verify all prerequisites
python --version    # Should show Python 3.12.x or higher
uv --version        # Should show uv version
docker --version    # Should show Docker version
docker compose version  # Should show Docker Compose v2.x
```

---

## Clone and Initial Setup

```bash
# Clone the repository
git clone https://github.com/your-org/autoBot.git
cd autoBot

# Verify you're in the project root
ls -la
# You should see: pyproject.toml, README.md, services/, libs/, infra/
```

**Verification:**

```bash
# Confirm project structure
ls services/
# Expected output:
# backtester_py  feature_builder_py  ingestor_py  labeler_py
# monitor_py  optimizer_py  registry_api_py  runner_py

ls libs/
# Expected output:
# common_types  cost_models  risk_models
```

---

## Infrastructure Setup

Start the required infrastructure services using Docker Compose:

```bash
# Start PostgreSQL, MinIO (S3-compatible storage), and Redis
docker compose -f infra/docker-compose.yml up -d
```

This starts:
- **PostgreSQL 16**: Registry database (port 5432)
- **MinIO**: S3-compatible object storage for the data lake (ports 9000, 9001)
- **Redis 7**: Caching and task queue (port 6379)

**Verification:**

```bash
# Check all containers are running
docker compose -f infra/docker-compose.yml ps

# Expected output (all should show "running"):
# NAME                     STATUS
# autobot-postgres-1       running
# autobot-minio-1          running
# autobot-redis-1          running

# Test PostgreSQL connection
docker exec -it $(docker compose -f infra/docker-compose.yml ps -q postgres) \
    psql -U qt -d qt_registry -c "SELECT 1;"
# Expected: Returns "1"

# Test MinIO is accessible (opens console in browser)
echo "MinIO Console: http://localhost:9001"
echo "Username: minio"
echo "Password: minio123"
```

---

## Environment Configuration

Create and configure your environment file:

```bash
# Copy the example environment file
cp .env.example .env

# Edit the .env file with your Alpaca credentials
# Use your preferred editor (vim, nano, code, etc.)
nano .env
```

**Required `.env` settings:**

```bash
# =============================================================================
# ALPACA API CREDENTIALS (Required)
# =============================================================================
ALPACA_API_KEY=your_alpaca_api_key_here
ALPACA_API_SECRET=your_alpaca_api_secret_here
ALPACA_ENV=paper

# =============================================================================
# PostgreSQL Configuration
# =============================================================================
PGHOST=localhost
PGPORT=5432
PGDATABASE=qt_registry
PGUSER=qt
PGPASSWORD=qt

# For SQLAlchemy-based services
DATABASE_URL=postgresql://qt:qt@localhost:5432/qt_registry

# =============================================================================
# MinIO / S3 Configuration
# =============================================================================
S3_ENDPOINT_URL=http://localhost:9000
S3_ACCESS_KEY=minio
S3_SECRET_KEY=minio123
S3_BUCKET=lake

# =============================================================================
# Data Lake Path (local filesystem)
# =============================================================================
LAKE_PATH=./lake

# =============================================================================
# Registry API Configuration
# =============================================================================
REGISTRY_BIND=0.0.0.0
REGISTRY_PORT=8080
ARTIFACT_STORAGE_PATH=./artifacts
```

**Verification:**

```bash
# Verify .env file exists and has content
cat .env | grep -E "^ALPACA_API_KEY=" | head -1
# Should show: ALPACA_API_KEY=your_key (not empty)

# Load environment variables (for current shell session)
source .env && echo "Environment loaded successfully"
```

---

## Install Dependencies

Use UV to create the virtual environment and install all dependencies:

```bash
# Create virtual environment and sync all dependencies
uv venv
uv sync

# This installs:
# - All root project dependencies
# - All workspace member packages (services/, libs/)
# - Development tools (pytest, mypy, ruff)
```

**Verification:**

```bash
# Activate the virtual environment
source .venv/bin/activate

# Verify Python version in venv
python --version
# Should show: Python 3.12.x

# Verify key packages are installed
python -c "import alpaca; print('alpaca-py:', alpaca.__version__)"
python -c "import polars; print('polars:', polars.__version__)"
python -c "import pyarrow; print('pyarrow:', pyarrow.__version__)"
python -c "import fastapi; print('fastapi:', fastapi.__version__)"

# Verify workspace packages are importable
python -c "from ingestor_py.client import AlpacaDataClient; print('ingestor_py: OK')"
python -c "from registry_api_py.app import app; print('registry_api_py: OK')"
```

---

## Database Initialization

Initialize the PostgreSQL database with the required schema:

```bash
# Ensure environment is loaded
source .venv/bin/activate
source .env

# Create database tables (SQLAlchemy creates tables on first run)
# The Registry API will auto-create tables when started

# For explicit table creation, you can run:
python -c "
from registry_api_py.database import engine
from registry_api_py.models import Base
Base.metadata.create_all(bind=engine)
print('Database tables created successfully')
"
```

**Verification:**

```bash
# Connect to PostgreSQL and verify tables
docker exec -it $(docker compose -f infra/docker-compose.yml ps -q postgres) \
    psql -U qt -d qt_registry -c "\dt"

# Expected output (tables may vary):
#              List of relations
#  Schema |       Name        | Type  | Owner
# --------+-------------------+-------+-------
#  public | artifacts         | table | qt
#  public | audit_log         | table | qt
#  public | backtest_runs     | table | qt
#  public | dataset_snapshots | table | qt
#  public | gates             | table | qt
#  public | promotions        | table | qt
#  public | strategies        | table | qt
```

---

## Run Your First Data Ingestion

The ingestor service fetches market data from Alpaca and writes it to Parquet files.

### Option A: Backfill Historical Data (Recommended First Run)

```bash
# Ensure environment is active
source .venv/bin/activate
source .env

# Create the data lake directory
mkdir -p lake

# Run historical backfill for the last 7 days
INGESTOR_MODE=backfill \
BACKFILL_DAYS=7 \
SYMBOLS=SPY,QQQ \
LAKE_PATH=./lake \
ALPACA_API_KEY=$ALPACA_API_KEY \
ALPACA_API_SECRET=$ALPACA_API_SECRET \
python -m ingestor_py.main
```

**Expected output:**

```
2025-01-15 10:30:00 - ingestor_py.main - INFO - Ingestor starting in backfill mode
2025-01-15 10:30:00 - ingestor_py.main - INFO - Lake path: lake
2025-01-15 10:30:00 - ingestor_py.main - INFO - Symbols: ['SPY', 'QQQ']
2025-01-15 10:30:00 - ingestor_py.main - INFO - Starting backfill for ['SPY', 'QQQ'], last 7 days
2025-01-15 10:30:15 - ingestor_py.main - INFO - Backfill complete: {'trades': {'count': 12345, ...}, ...}
```

**Verification:**

```bash
# Check that Parquet files were created
ls -la lake/raw/

# Expected directories:
# trades/
# quotes/
# bars_provider/

# Check partitioned data structure
find lake/raw/trades -name "*.parquet" | head -5

# Expected paths like:
# lake/raw/trades/dt=2025-01-08/symbol=SPY/data.parquet
# lake/raw/trades/dt=2025-01-08/symbol=QQQ/data.parquet

# Verify Parquet content with Python
python -c "
import polars as pl
df = pl.read_parquet('lake/raw/trades/')
print(f'Total trades: {len(df)}')
print(f'Columns: {df.columns}')
print(df.head(3))
"
```

### Option B: Real-Time Streaming

```bash
# Start real-time streaming (runs continuously)
INGESTOR_MODE=stream \
SYMBOLS=SPY,QQQ \
LAKE_PATH=./lake \
ALPACA_API_KEY=$ALPACA_API_KEY \
ALPACA_API_SECRET=$ALPACA_API_SECRET \
python -m ingestor_py.main

# Press Ctrl+C to stop streaming
```

---

## Start the Registry API

The Registry API provides the system memory and audit log for the trading engine.

```bash
# Ensure environment is active
source .venv/bin/activate
source .env

# Start the Registry API server
uv run --package registry_api_py python -m registry_api_py

# Or alternatively:
python -m registry_api_py.app
```

**Expected output:**

```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)
```

**Verification (in a new terminal):**

```bash
# Health check
curl http://localhost:8080/health
# Expected: {"status":"ok"}

# List strategies (should be empty initially)
curl http://localhost:8080/strategies
# Expected: {"items":[],"total":0,"skip":0,"limit":100}

# Create a test strategy
curl -X POST http://localhost:8080/strategies \
  -H "Content-Type: application/json" \
  -d '{
    "name": "test_trend_v1",
    "family": "trend",
    "version": "1.0.0",
    "parameters": {"theta": 0.0015}
  }'
# Expected: {"id":1,"name":"test_trend_v1",...}

# View API documentation
echo "OpenAPI docs: http://localhost:8080/docs"
echo "ReDoc: http://localhost:8080/redoc"
```

---

## Verify the Setup

Run the complete verification checklist:

```bash
#!/bin/bash
# Save as: scripts/verify_setup.sh

echo "=== autoBot Setup Verification ==="
echo ""

# 1. Check Docker containers
echo "1. Docker Containers:"
docker compose -f infra/docker-compose.yml ps --format "table {{.Name}}\t{{.Status}}"
echo ""

# 2. Check environment
echo "2. Environment Variables:"
[ -n "$ALPACA_API_KEY" ] && echo "   ALPACA_API_KEY: Set" || echo "   ALPACA_API_KEY: NOT SET"
[ -n "$ALPACA_API_SECRET" ] && echo "   ALPACA_API_SECRET: Set" || echo "   ALPACA_API_SECRET: NOT SET"
[ -n "$DATABASE_URL" ] && echo "   DATABASE_URL: Set" || echo "   DATABASE_URL: NOT SET"
echo ""

# 3. Check Python environment
echo "3. Python Environment:"
python --version
echo "   UV: $(uv --version)"
echo ""

# 4. Check data lake
echo "4. Data Lake:"
if [ -d "lake/raw" ]; then
    echo "   Trades: $(find lake/raw/trades -name '*.parquet' 2>/dev/null | wc -l) files"
    echo "   Quotes: $(find lake/raw/quotes -name '*.parquet' 2>/dev/null | wc -l) files"
    echo "   Bars: $(find lake/raw/bars_provider -name '*.parquet' 2>/dev/null | wc -l) files"
else
    echo "   Data lake not initialized (run backfill first)"
fi
echo ""

# 5. Check Registry API
echo "5. Registry API:"
if curl -s http://localhost:8080/health > /dev/null 2>&1; then
    echo "   Status: Running"
    echo "   Health: $(curl -s http://localhost:8080/health)"
else
    echo "   Status: Not running (start with: uv run --package registry_api_py python -m registry_api_py)"
fi
echo ""

echo "=== Verification Complete ==="
```

Run the tests:

```bash
# Run all tests
uv run pytest

# Run specific service tests
uv run pytest services/ingestor_py/tests/ -v
uv run pytest services/registry_api_py/tests/ -v

# Run with coverage
uv run pytest --cov=services --cov-report=term-missing
```

---

## Next Steps

After completing this quickstart, you can:

1. **Build Features**: Run the feature builder to create microstructure bars
   ```bash
   python -m feature_builder_py.main
   ```

2. **Explore the API**: Open http://localhost:8080/docs to see all available endpoints

3. **Review the PRD**: Read `docs/PRD.md` for the full system architecture

4. **Run Backtests**: Use the backtester service once you have features built

5. **Monitor the System**: Start the monitor service for real-time metrics

---

## Troubleshooting

### Docker Issues

**Problem:** Containers fail to start

```bash
# Check logs
docker compose -f infra/docker-compose.yml logs postgres
docker compose -f infra/docker-compose.yml logs minio

# Reset containers
docker compose -f infra/docker-compose.yml down -v
docker compose -f infra/docker-compose.yml up -d
```

**Problem:** Port already in use

```bash
# Find process using port 5432
lsof -i :5432
# Kill if necessary, or change port in docker-compose.yml
```

### Alpaca API Issues

**Problem:** Authentication errors

```bash
# Verify API keys are set correctly
echo $ALPACA_API_KEY
echo $ALPACA_API_SECRET

# Test API connection
python -c "
from alpaca.data.historical import StockHistoricalDataClient
import os
client = StockHistoricalDataClient(
    api_key=os.environ['ALPACA_API_KEY'],
    secret_key=os.environ['ALPACA_API_SECRET']
)
print('API connection successful')
"
```

**Problem:** No data returned during market hours

- Free tier (IEX) has 15-minute delay
- Data is only available during market hours (9:30 AM - 4:00 PM ET)
- Use backfill mode to get historical data

### UV/Python Issues

**Problem:** UV not found

```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Reload shell
source ~/.bashrc  # or ~/.zshrc
```

**Problem:** Python version mismatch

```bash
# Check available Python versions
uv python list

# Install Python 3.12 if needed
uv python install 3.12

# Recreate venv with correct version
rm -rf .venv
uv venv --python 3.12
uv sync
```

### Database Issues

**Problem:** Cannot connect to PostgreSQL

```bash
# Check PostgreSQL is running
docker compose -f infra/docker-compose.yml ps postgres

# Test connection
docker exec -it $(docker compose -f infra/docker-compose.yml ps -q postgres) \
    psql -U qt -d qt_registry -c "SELECT version();"

# Check DATABASE_URL format
echo $DATABASE_URL
# Should be: postgresql://qt:qt@localhost:5432/qt_registry
```

**Problem:** Tables not created

```bash
# Manually create tables
python -c "
from registry_api_py.database import engine
from registry_api_py.models import Base
Base.metadata.create_all(bind=engine)
print('Tables created')
"
```

### Data Lake Issues

**Problem:** No Parquet files created

```bash
# Check lake directory permissions
ls -la lake/

# Create directory if missing
mkdir -p lake/raw/{trades,quotes,bars_provider}

# Run backfill with debug logging
INGESTOR_MODE=backfill \
BACKFILL_DAYS=1 \
SYMBOLS=SPY \
LAKE_PATH=./lake \
python -m ingestor_py.main 2>&1 | tee backfill.log
```

**Problem:** Cannot read Parquet files

```bash
# Verify file integrity
python -c "
import pyarrow.parquet as pq
# Replace with actual path
table = pq.read_table('lake/raw/trades/dt=2025-01-15/symbol=SPY/data.parquet')
print(table.schema)
print(f'Rows: {len(table)}')
"
```

---

## Quick Reference

### Common Commands

```bash
# Start infrastructure
docker compose -f infra/docker-compose.yml up -d

# Stop infrastructure
docker compose -f infra/docker-compose.yml down

# Activate environment
source .venv/bin/activate && source .env

# Run backfill
INGESTOR_MODE=backfill BACKFILL_DAYS=7 SYMBOLS=SPY,QQQ python -m ingestor_py.main

# Start Registry API
uv run --package registry_api_py python -m registry_api_py

# Run tests
uv run pytest

# Check API health
curl http://localhost:8080/health
```

### Service Ports

| Service | Port | URL |
|---------|------|-----|
| Registry API | 8080 | http://localhost:8080 |
| PostgreSQL | 5432 | localhost:5432 |
| MinIO API | 9000 | http://localhost:9000 |
| MinIO Console | 9001 | http://localhost:9001 |
| Redis | 6379 | localhost:6379 |

### Default Credentials

| Service | Username | Password |
|---------|----------|----------|
| PostgreSQL | qt | qt |
| MinIO | minio | minio123 |

---

**Need help?** Check the [PRD documentation](./PRD.md) or open an issue in the repository.
