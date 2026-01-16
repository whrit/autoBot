# Development Guide

This document provides comprehensive guidance for developers contributing to and extending the Phase 1 autonomous equities trading engine.

---

## Table of Contents

1. [Development Setup](#development-setup)
2. [Code Organization](#code-organization)
3. [Testing Guide](#testing-guide)
4. [Code Quality](#code-quality)
5. [Adding New Features](#adding-new-features)
6. [Git Workflow](#git-workflow)

---

## Development Setup

### Prerequisites

- **Python**: 3.12 or higher (required)
- **uv**: Modern Python package manager (recommended over pip)
- **Docker**: For running infrastructure services (PostgreSQL, MinIO, Redis)
- **Git**: Version control

### Cloning the Repository

```bash
# Clone the repository
git clone <repository-url> autoBot
cd autoBot

# Verify you're on the correct branch
git checkout Development-1-Phase1v0
```

### Setting Up Python Environment with UV

UV is the recommended package manager for this project. It provides faster dependency resolution and better workspace support.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv

# Activate the virtual environment
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows

# Sync all dependencies (including dev dependencies)
uv sync
```

### Installing Development Dependencies

The project uses a workspace structure with multiple packages. Install all workspace members:

```bash
# Install all workspace packages in development mode
uv sync

# For a specific service, navigate to it and install
cd services/feature_builder_py
uv sync
```

### Environment Configuration

Copy the example environment file and configure it:

```bash
cp .env.example .env
```

Edit `.env` with your configuration:

```bash
# Alpaca API credentials
ALPACA_API_KEY_ID=your_key
ALPACA_API_SECRET_KEY=your_secret
ALPACA_ENV=paper  # Use "paper" for development

# PostgreSQL (local Docker)
PGHOST=localhost
PGPORT=5432
PGDATABASE=qt_registry
PGUSER=qt
PGPASSWORD=qt

# MinIO S3-compatible storage (local Docker)
S3_ENDPOINT_URL=http://localhost:9000
S3_ACCESS_KEY=minio
S3_SECRET_KEY=minio123
S3_BUCKET=lake

# Registry API
REGISTRY_BIND=0.0.0.0
REGISTRY_PORT=8080
```

### Starting Infrastructure Services

The project uses Docker Compose for local infrastructure:

```bash
# Start all infrastructure services
docker compose -f infra/docker-compose.yml up -d

# Services started:
# - PostgreSQL (port 5432): Strategy registry database
# - MinIO (ports 9000, 9001): S3-compatible object storage
# - Redis (port 6379): Caching and message broker

# Check service status
docker compose -f infra/docker-compose.yml ps

# View logs
docker compose -f infra/docker-compose.yml logs -f

# Stop services
docker compose -f infra/docker-compose.yml down
```

### IDE Configuration

#### VSCode

Create `.vscode/settings.json`:

```json
{
    "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
    "python.analysis.typeCheckingMode": "strict",
    "python.analysis.extraPaths": [
        "${workspaceFolder}/libs/common_types",
        "${workspaceFolder}/libs/cost_models",
        "${workspaceFolder}/libs/risk_models",
        "${workspaceFolder}/services/feature_builder_py",
        "${workspaceFolder}/services/optimizer_py",
        "${workspaceFolder}/services/backtester_py",
        "${workspaceFolder}/services/registry_api_py",
        "${workspaceFolder}/services/monitor_py",
        "${workspaceFolder}/services/runner_py",
        "${workspaceFolder}/services/ingestor_py",
        "${workspaceFolder}/services/labeler_py"
    ],
    "[python]": {
        "editor.formatOnSave": true,
        "editor.codeActionsOnSave": {
            "source.organizeImports": "explicit"
        },
        "editor.defaultFormatter": "charliermarsh.ruff"
    },
    "ruff.enable": true,
    "ruff.lint.args": ["--config=${workspaceFolder}/pyproject.toml"],
    "mypy.runUsingActiveInterpreter": true
}
```

Recommended VSCode extensions:
- `charliermarsh.ruff` - Linting and formatting
- `ms-python.python` - Python language support
- `ms-python.mypy-type-checker` - Type checking
- `ms-python.vscode-pylance` - Language server

#### PyCharm

1. Open the project root directory
2. Configure Python interpreter:
   - Settings > Project > Python Interpreter
   - Select the `.venv/bin/python` interpreter
3. Enable Ruff:
   - Settings > Tools > Ruff
   - Enable Ruff for linting
4. Configure source roots:
   - Right-click on `libs/` and `services/` subdirectories
   - Mark Directory as > Sources Root

---

## Code Organization

### Project Structure

```
autoBot/
├── pyproject.toml          # Root workspace configuration
├── uv.lock                  # Lock file for reproducible builds
├── .env.example             # Environment variable template
├── README.md                # Project overview
│
├── docs/                    # Documentation
│   ├── OUTLINE.md           # Architecture outline
│   ├── PHASE1_PLAN.md       # Phase 1 implementation plan
│   ├── PHASE2_PLAN.md       # Phase 2 implementation plan
│   ├── PHASE3_PLAN.md       # Phase 3 implementation plan
│   ├── PRD.md               # Product requirements document
│   └── EVAL_PROTOCOL.md     # Evaluation protocol
│
├── libs/                    # Shared libraries
│   ├── common_types/        # Pydantic models and enums
│   ├── cost_models/         # Slippage and transaction costs
│   └── risk_models/         # Risk management models
│
├── services/                # Microservices
│   ├── ingestor_py/         # Market data ingestion
│   ├── feature_builder_py/  # Feature engineering
│   ├── labeler_py/          # Target label generation
│   ├── backtester_py/       # Strategy backtesting
│   ├── optimizer_py/        # Strategy optimization
│   ├── registry_api_py/     # Strategy registry API
│   ├── monitor_py/          # Monitoring and alerting
│   └── runner_py/           # Execution orchestration
│
├── configs/                 # Configuration files
│   ├── cost_model_cm_v1.json
│   ├── risk_profile_default.json
│   └── universe.yaml
│
├── infra/                   # Infrastructure
│   └── docker-compose.yml   # Local development services
│
├── artifacts/               # Build artifacts (git-ignored)
└── db/                      # Database migrations
```

### Service Directory Layout

Each service follows a consistent structure:

```
services/<service_name>/
├── pyproject.toml           # Service-specific dependencies
├── <service_name>/          # Source code package
│   ├── __init__.py          # Package exports
│   ├── main.py              # Entry point
│   └── *.py                 # Module files
└── tests/                   # Test suite
    ├── conftest.py          # Pytest fixtures
    └── test_*.py            # Test modules
```

### Shared Libraries

#### common_types

Central type definitions used across all services:

```python
from common_types import (
    # Enums
    LifecycleState,     # candidate, shadow, paper, promoted, retired
    RunType,            # backtest, shadow, paper, live
    ArtifactKind,       # json_rules, onnx_model

    # Market Data Models
    Trade,              # Raw trade ticks
    Quote,              # NBBO quotes with spread/microprice
    Bar,                # OHLCV bars (1m, 5m, 15m)
    MicroBar,           # Microstructure bars (5s, 15s, 30s)
    DecisionFrame,      # Multi-timeframe feature vector
    Label,              # Prediction targets

    # Strategy Models
    Strategy,           # Strategy configuration
    SignalLogic,        # Signal generation rules
    SizingConfig,       # Position sizing
    ExecutionConfig,    # Order execution settings
    TrainingMetadata,   # GPU/CPU training info

    # Configuration Models
    CostModelConfig,    # Slippage model coefficients
    RiskProfile,        # Risk limits

    # Artifact Models
    Artifact,           # Strategy artifact wrapper
    OnnxMetadata,       # ONNX model metadata
)
```

#### cost_models

Transaction cost and slippage modeling:

```python
from cost_models import (
    estimate_slippage,      # Estimate execution slippage
    calculate_total_cost,   # Total transaction cost
)
```

#### risk_models

Risk management and position limits:

```python
from risk_models import (
    check_position_limits,  # Validate position sizes
    calculate_exposure,     # Portfolio exposure
    check_drawdown_limit,   # Drawdown validation
)
```

---

## Testing Guide

### Running Tests with Pytest

```bash
# Run all tests from project root
pytest

# Run tests for a specific service
pytest services/optimizer_py/tests/

# Run a specific test file
pytest services/optimizer_py/tests/test_strategies.py

# Run a specific test class or function
pytest services/optimizer_py/tests/test_strategies.py::TestStrategySignal
pytest services/optimizer_py/tests/test_strategies.py::TestStrategySignal::test_strategy_signal_creation

# Run tests with verbose output
pytest -v

# Run tests with coverage report
pytest --cov=services --cov-report=html

# Run tests in parallel (faster)
pytest -n auto

# Run only failed tests from last run
pytest --lf

# Run tests matching a pattern
pytest -k "trend"
```

### Test Organization

Tests are organized by type:

```
tests/
├── conftest.py              # Shared fixtures
├── test_<module>.py         # Unit tests for each module
├── test_integration.py      # Integration tests
└── test_e2e.py              # End-to-end tests (if applicable)
```

#### Unit Tests

Test individual functions and classes in isolation:

```python
# services/optimizer_py/tests/test_strategies.py

class TestStrategySignal:
    """Tests for StrategySignal dataclass."""

    def test_strategy_signal_creation(self) -> None:
        """Test basic StrategySignal creation with required fields."""
        from optimizer_py.strategy_family import StrategySignal

        signal = StrategySignal(
            timestamp=datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC),
            symbol="AAPL",
            signal=1,
            confidence=0.85,
            metadata={"source": "trend"},
        )

        assert signal.symbol == "AAPL"
        assert signal.signal == 1
        assert signal.confidence == 0.85
```

#### Integration Tests

Test interactions between components:

```python
# services/optimizer_py/tests/test_optimizer_integration.py

class TestOptimizerIntegration:
    """Integration tests for optimizer service."""

    def test_full_optimization_pipeline(
        self,
        sample_decision_frame: pl.DataFrame,
    ) -> None:
        """Test complete optimization from data to ranked strategies."""
        # Test multiple components working together
        pass
```

### Writing New Tests

Follow TDD (Test-Driven Development) principles:

1. **Write the test first** - Define expected behavior before implementation
2. **Run the test (expect failure)** - Confirm the test catches missing functionality
3. **Implement the feature** - Write minimal code to pass the test
4. **Refactor** - Clean up while keeping tests green

#### Test Naming Convention

```python
def test_<what>_<condition>_<expected>() -> None:
    """Test that <what> does <expected> when <condition>."""
    pass

# Examples:
def test_strategy_signal_with_valid_values_creates_signal() -> None:
    """Test that StrategySignal creates successfully with valid values."""
    pass

def test_trend_strategy_on_uptrend_generates_long_signal() -> None:
    """Test that TrendStrategy generates long signal on uptrend."""
    pass
```

### Fixtures and Mocking Patterns

#### conftest.py Pattern

```python
# services/optimizer_py/tests/conftest.py
"""Pytest configuration and shared fixtures."""

import numpy as np
import polars as pl
import pytest
from datetime import UTC, datetime, timedelta


@pytest.fixture
def sample_decision_frame() -> pl.DataFrame:
    """Create a sample decision frame for testing strategies."""
    base_time = datetime(2024, 1, 15, 9, 30, 0, tzinfo=UTC)
    n_rows = 100

    np.random.seed(42)  # Reproducible randomness
    base_price = 100.0
    prices = base_price * np.cumprod(1 + np.random.randn(n_rows) * 0.002)

    return pl.DataFrame({
        "timestamp": [base_time + timedelta(minutes=i) for i in range(n_rows)],
        "symbol": ["AAPL"] * n_rows,
        "close": prices,
        "volume": np.random.randint(10000, 100000, size=n_rows).astype(float),
    })


@pytest.fixture
def mock_http_client() -> AsyncMock:
    """Create mock HTTP client for API testing."""
    from unittest.mock import AsyncMock, MagicMock

    client = AsyncMock()
    mock_response = AsyncMock()
    mock_response.json.return_value = {}
    mock_response.status_code = 200
    client.get.return_value = mock_response
    return client
```

#### Database Test Fixtures

```python
# services/registry_api_py/tests/conftest.py
"""Database fixtures for API tests."""

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

@pytest.fixture(scope="function")
def db_engine():
    """Create in-memory SQLite for testing."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def db_session(db_engine):
    """Create test database session."""
    TestingSessionLocal = sessionmaker(bind=db_engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
```

### Coverage Requirements

Target coverage levels:

| Component | Minimum Coverage |
|-----------|------------------|
| Core libraries (`libs/`) | 90% |
| Service business logic | 85% |
| API endpoints | 80% |
| Integration tests | 70% |

Generate coverage reports:

```bash
# HTML report
pytest --cov=services --cov-report=html
open htmlcov/index.html

# Terminal report
pytest --cov=services --cov-report=term-missing

# XML for CI
pytest --cov=services --cov-report=xml
```

---

## Code Quality

### Linting with Ruff

Ruff is the primary linter and formatter. Configuration is in `pyproject.toml`:

```toml
[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
extend-select = ["I", "UP", "B", "SIM", "RUF"]
ignore = ["RUF002", "RUF003"]

[tool.ruff.lint.isort]
known-first-party = ["common_types", "cost_models", "risk_models"]
```

Run linting:

```bash
# Check for issues
ruff check .

# Auto-fix issues
ruff check --fix .

# Format code
ruff format .

# Check formatting without changes
ruff format --check .
```

### Type Checking with Mypy

Strict type checking is enabled. Configuration in `pyproject.toml`:

```toml
[tool.mypy]
python_version = "3.12"
strict = true
warn_return_any = true
warn_unused_configs = true
no_implicit_optional = true
disallow_untyped_defs = true
disallow_any_generics = true
```

Run type checking:

```bash
# Check entire project
mypy .

# Check specific service
mypy services/optimizer_py/

# With verbose output
mypy --verbose services/optimizer_py/
```

### Pre-commit Hooks

Set up pre-commit hooks for automatic quality checks:

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run manually on all files
pre-commit run --all-files
```

Create `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.14.12
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.19.1
    hooks:
      - id: mypy
        additional_dependencies:
          - pydantic>=2.0.0
          - polars>=1.0.0
          - fastapi>=0.115.0

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
```

### Code Review Guidelines

When reviewing code, check for:

1. **Type Annotations**: All functions must have complete type hints
2. **Docstrings**: Public functions need docstrings with Args/Returns
3. **Test Coverage**: New code must have corresponding tests
4. **Error Handling**: Proper exception handling and logging
5. **Performance**: Avoid N+1 queries, unnecessary loops
6. **Security**: No hardcoded secrets, proper input validation

---

## Adding New Features

### Adding a New Strategy Family

1. **Create the strategy module**:

```python
# services/optimizer_py/optimizer_py/my_strategy.py
"""
My Custom Strategy Implementation (T4.XX).

Implements custom trading signals using...
"""

from __future__ import annotations
from dataclasses import dataclass
import polars as pl

from optimizer_py.strategy_family import StrategyFamily, StrategySignal


@dataclass
class MyStrategyParameters:
    """Parameters for my custom strategy."""
    lookback: int = 20
    threshold: float = 0.5


class MyStrategy(StrategyFamily):
    """
    Custom strategy implementation.

    Generates signals when...
    """

    def __init__(
        self,
        lookback: int = 20,
        threshold: float = 0.5,
    ) -> None:
        """Initialize strategy with parameters."""
        self._params = MyStrategyParameters(
            lookback=lookback,
            threshold=threshold,
        )

    @property
    def name(self) -> str:
        """Strategy family name."""
        return "my_strategy"

    @property
    def parameters(self) -> dict[str, float | int | str]:
        """Current strategy parameters."""
        return {
            "lookback": self._params.lookback,
            "threshold": self._params.threshold,
        }

    def get_parameter_grid(self) -> dict[str, list[float | int]]:
        """Get parameter grid for optimization."""
        return {
            "lookback": [10, 20, 50, 100],
            "threshold": [0.25, 0.5, 1.0, 2.0],
        }

    def generate_signals(self, decision_frame: pl.DataFrame) -> list[StrategySignal]:
        """Generate trading signals from decision frame."""
        self.validate_decision_frame(decision_frame)
        signals: list[StrategySignal] = []

        # Implementation here...

        return signals
```

2. **Export from package**:

```python
# services/optimizer_py/optimizer_py/__init__.py
from optimizer_py.my_strategy import MyStrategy, MyStrategyParameters

__all__ = [
    # ... existing exports
    "MyStrategy",
    "MyStrategyParameters",
]
```

3. **Write tests**:

```python
# services/optimizer_py/tests/test_my_strategy.py
"""Tests for MyStrategy implementation."""

import polars as pl
import pytest
from optimizer_py.my_strategy import MyStrategy


class TestMyStrategy:
    """Tests for MyStrategy."""

    def test_name_property(self) -> None:
        """Test strategy name is correct."""
        strategy = MyStrategy()
        assert strategy.name == "my_strategy"

    def test_default_parameters(self) -> None:
        """Test default parameter values."""
        strategy = MyStrategy()
        assert strategy.parameters["lookback"] == 20
        assert strategy.parameters["threshold"] == 0.5

    def test_generate_signals_on_sample_data(
        self,
        sample_decision_frame: pl.DataFrame,
    ) -> None:
        """Test signal generation on sample data."""
        strategy = MyStrategy(lookback=10)
        signals = strategy.generate_signals(sample_decision_frame)

        assert isinstance(signals, list)
        for signal in signals:
            assert signal.signal in [-1, 0, 1]
            assert 0.0 <= signal.confidence <= 1.0
```

### Adding New Features to feature_builder

1. **Add feature calculation function**:

```python
# services/feature_builder_py/feature_builder_py/my_features.py
"""Custom feature calculations."""

import polars as pl


def calculate_my_feature(df: pl.DataFrame, lookback: int = 20) -> pl.DataFrame:
    """
    Calculate custom feature.

    Args:
        df: Input DataFrame with OHLCV columns
        lookback: Lookback period for calculation

    Returns:
        DataFrame with new feature column added
    """
    return df.with_columns(
        pl.col("close")
        .rolling_mean(window_size=lookback)
        .alias("my_feature")
    )
```

2. **Integrate into decision frame builder**:

```python
# services/feature_builder_py/feature_builder_py/decision_frame.py

from feature_builder_py.my_features import calculate_my_feature


def build_decision_frame(bars: pl.DataFrame, ...) -> pl.DataFrame:
    """Build decision frame with all features."""
    df = bars

    # ... existing features ...

    # Add custom feature
    df = calculate_my_feature(df, lookback=20)

    return df
```

### Adding New Monitor Metrics

1. **Create metric module**:

```python
# services/monitor_py/monitor_py/my_metric.py
"""Custom monitoring metric."""

from prometheus_client import Counter, Gauge, Histogram


# Define Prometheus metrics
MY_METRIC_GAUGE = Gauge(
    "trading_my_metric",
    "Description of my metric",
    ["strategy_id", "symbol"],
)

MY_METRIC_HISTOGRAM = Histogram(
    "trading_my_metric_distribution",
    "Distribution of my metric",
    ["strategy_id"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
)


def calculate_my_metric(data: ...) -> float:
    """Calculate the custom metric."""
    # Implementation
    value = ...
    return value


def update_my_metric_gauge(
    strategy_id: str,
    symbol: str,
    value: float,
) -> None:
    """Update Prometheus gauge for my metric."""
    MY_METRIC_GAUGE.labels(
        strategy_id=strategy_id,
        symbol=symbol,
    ).set(value)


def observe_my_metric(strategy_id: str, value: float) -> None:
    """Record observation in histogram."""
    MY_METRIC_HISTOGRAM.labels(strategy_id=strategy_id).observe(value)
```

2. **Export from package**:

```python
# services/monitor_py/monitor_py/__init__.py
from monitor_py.my_metric import (
    calculate_my_metric,
    update_my_metric_gauge,
    MY_METRIC_GAUGE,
)
```

### Adding New API Endpoints

1. **Define schema**:

```python
# services/registry_api_py/registry_api_py/schemas.py

class MyResourceCreate(BaseModel):
    """Request schema for creating my resource."""
    name: str
    value: float
    metadata: dict[str, Any] | None = None


class MyResourceResponse(BaseModel):
    """Response schema for my resource."""
    id: int
    name: str
    value: float
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
```

2. **Create database model**:

```python
# services/registry_api_py/registry_api_py/models.py

class MyResource(Base):
    """Database model for my resource."""
    __tablename__ = "my_resources"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), unique=True)
    value: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
```

3. **Add endpoint**:

```python
# services/registry_api_py/registry_api_py/app.py

@app.post("/my-resources", response_model=MyResourceResponse, status_code=201)
def create_my_resource(
    data: MyResourceCreate,
    db: Session = Depends(get_db),
) -> MyResource:
    """Create a new my resource."""
    resource = MyResource(
        name=data.name,
        value=data.value,
    )
    db.add(resource)
    db.commit()
    db.refresh(resource)
    return resource


@app.get("/my-resources/{resource_id}", response_model=MyResourceResponse)
def get_my_resource(
    resource_id: int,
    db: Session = Depends(get_db),
) -> MyResource:
    """Get a my resource by ID."""
    resource = db.query(MyResource).filter(MyResource.id == resource_id).first()
    if not resource:
        raise HTTPException(status_code=404, detail="Resource not found")
    return resource
```

4. **Write tests**:

```python
# services/registry_api_py/tests/test_my_resource.py

def test_create_my_resource(client: TestClient) -> None:
    """Test creating a new resource."""
    response = client.post(
        "/my-resources",
        json={"name": "test", "value": 1.5},
    )
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "test"
    assert data["value"] == 1.5


def test_get_my_resource_not_found(client: TestClient) -> None:
    """Test getting non-existent resource returns 404."""
    response = client.get("/my-resources/9999")
    assert response.status_code == 404
```

---

## Git Workflow

### Branch Naming Conventions

| Branch Type | Pattern | Example |
|-------------|---------|---------|
| Main development | `Development-{version}-{phase}` | `Development-1-Phase1v0` |
| Feature | `feature/<ticket>-<description>` | `feature/T4-add-volatility-strategy` |
| Bugfix | `bugfix/<ticket>-<description>` | `bugfix/BUG-123-fix-null-check` |
| Hotfix | `hotfix/<version>-<description>` | `hotfix/1.0.1-security-patch` |
| Release | `release/<version>` | `release/1.0.0` |

### Commit Message Format

Follow conventional commit format:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Code style (formatting, semicolons, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**

```bash
# Feature
git commit -m "feat(optimizer): add volatility strategy family"

# Bug fix
git commit -m "fix(backtester): correct slippage calculation for limit orders"

# Documentation
git commit -m "docs: update API endpoint documentation"

# Test
git commit -m "test(registry): add integration tests for promotion endpoint"

# Refactor
git commit -m "refactor(feature_builder): extract microbar calculation to module"

# Chore
git commit -m "chore: update dependencies to latest versions"
```

### Pull Request Process

1. **Create feature branch**:
```bash
git checkout Development-1-Phase1v0
git pull origin Development-1-Phase1v0
git checkout -b feature/T4-my-feature
```

2. **Make changes and commit**:
```bash
# Make changes
git add .
git commit -m "feat(optimizer): add my new feature"
```

3. **Run quality checks**:
```bash
# Run linting
ruff check .
ruff format --check .

# Run type checking
mypy .

# Run tests
pytest

# Run tests with coverage
pytest --cov=services --cov-report=term-missing
```

4. **Push and create PR**:
```bash
git push -u origin feature/T4-my-feature
```

5. **PR Description Template**:
```markdown
## Summary
Brief description of changes.

## Changes
- Added X feature
- Fixed Y bug
- Updated Z documentation

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Manual testing performed

## Checklist
- [ ] Code follows project style guidelines
- [ ] All tests pass
- [ ] Type hints complete
- [ ] Documentation updated
- [ ] No sensitive data committed
```

### Code Review Checklist

Before approving a PR, verify:

- [ ] Tests pass and cover new code
- [ ] Type annotations are complete
- [ ] No linting errors
- [ ] Docstrings for public functions
- [ ] No hardcoded secrets or credentials
- [ ] Database migrations included (if needed)
- [ ] API documentation updated (if endpoints changed)
- [ ] Performance impact considered

---

## Running Services

### Registry API

```bash
# Run the registry API
uv run --package registry_api_py python -m registry_api_py

# Or with explicit module
cd services/registry_api_py
uv run python -m registry_api_py.app

# Access at http://localhost:8080
# OpenAPI docs at http://localhost:8080/docs
```

### Running Tests for Specific Services

```bash
# Feature builder tests
cd services/feature_builder_py
pytest tests/

# Optimizer tests
cd services/optimizer_py
pytest tests/

# Registry API tests
cd services/registry_api_py
pytest tests/

# Monitor tests
cd services/monitor_py
pytest tests/

# Runner tests
cd services/runner_py
pytest tests/
```

---

## Troubleshooting

### Common Issues

**Import errors when running tests:**
```bash
# Ensure you're in the project root with activated venv
source .venv/bin/activate
pytest
```

**Database connection errors:**
```bash
# Ensure Docker services are running
docker compose -f infra/docker-compose.yml up -d

# Check service health
docker compose -f infra/docker-compose.yml ps
```

**Type checking errors with workspace packages:**
```bash
# Install packages in editable mode
uv sync
```

**UV sync fails:**
```bash
# Clear cache and retry
uv cache clean
uv sync
```

---

## Additional Resources

- [Phase 1 Plan](./PHASE1_PLAN.md) - Detailed implementation plan
- [PRD](./PRD.md) - Product requirements document
- [Evaluation Protocol](./EVAL_PROTOCOL.md) - Strategy evaluation guidelines
- [UV Documentation](https://docs.astral.sh/uv/) - Package manager docs
- [Ruff Documentation](https://docs.astral.sh/ruff/) - Linter docs
- [Polars Documentation](https://docs.pola.rs/) - DataFrame library docs
- [FastAPI Documentation](https://fastapi.tiangolo.com/) - Web framework docs
