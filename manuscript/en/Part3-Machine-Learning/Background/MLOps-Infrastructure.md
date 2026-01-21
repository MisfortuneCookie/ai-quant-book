# Background: MLOps Infrastructure

> **Once deployed, models begin to decay. MLOps is not an optional advanced configuration. It is the infrastructure that keeps quantitative strategies alive.**

---

## From "Model Works" to "Model Is Useful"

In 2023, a quantitative researcher completed a momentum strategy model:

- Backtest Sharpe: 1.8
- IC mean: 0.04
- Clean code, tests passing

He excitedly deployed it to production.

**Three months later**:

- Month 1: Sharpe 1.2 ("market conditions")
- Month 2: Sharpe 0.4 ("let's observe more")
- Month 3: Sharpe -0.3 ("is the model broken?")

**What happened?**

Investigation revealed:
1. Production feature calculation code differed from backtest. A bug caused RSI to shift by one day
2. Model versioning was chaotic. Nobody knew which version was actually running
3. No way to trace the problem since feature snapshots and model inputs were not saved
4. By the time the issue was discovered, nobody knew when it had started

**Lesson**: Model development is just the beginning. **Reproducibility, version management, and drift monitoring** are the core of production systems. This is MLOps.

---

## 1. Why Quant Needs MLOps

### Quant-Specific Challenges

| Traditional ML | Quantitative ML |
|----------------|-----------------|
| Models are relatively stable after deployment | Market structure changes constantly; models inevitably decay |
| Data distribution is relatively fixed | Financial data is highly non-stationary |
| Model errors affect user experience | Model errors directly cause capital losses |
| Offline batch prediction is acceptable | Real-time inference required; latency-sensitive |
| Features come from stable data sources | Features come from multiple vendors; may be delayed or missing |

### The Three Pillars of MLOps

```
Quant MLOps = Feature Store + Model Registry + Drift Monitor

Functions:
1. Feature Store  -> Ensure backtest and live features are consistent (reproducibility)
2. Model Registry -> Track model versions and performance (auditability)
3. Drift Monitor  -> Detect model decay (timely stop-loss)
```

---

## 2. Feature Store

### Core Problem: Point-in-Time Correctness

The most insidious bug in quantitative finance is **look-ahead bias**.

```
Incorrect example (look-ahead bias):

2024-01-15 training sample:
  Feature: RSI = 65 (calculated using 2024-01-15 closing price)
  Label: Tomorrow's return

Problem:
  The 2024-01-15 closing price is not known until 16:00
  But RSI calculation used this value
  -> Model learned from "future information"

Correct approach:
  2024-01-15 training sample:
    Feature: RSI calculated using 2024-01-14 closing price
    Label: Return from 2024-01-15 to 2024-01-16
```

The core capability of a Feature Store is ensuring **Point-in-Time queries**: given any historical timestamp, return feature values that **were known at that time**.

### Dual Timestamp Design

```
Feature Events Table (feature_events):
+--------------+---------------+-----------------+-----------------+---------+
| entity_key   | feature_name  | event_time      | ingest_time     | value   |
+--------------+---------------+-----------------+-----------------+---------+
| AAPL.NASDAQ  | momentum_5d   | 2024-01-15      | 2024-01-15 20:00 | 0.035  |
| AAPL.NASDAQ  | rsi_14        | 2024-01-15      | 2024-01-15 20:00 | 62.5   |
+--------------+---------------+-----------------+-----------------+---------+

Meaning of the two timestamps:
- event_time: The business time the feature corresponds to (e.g., "this is RSI for 2024-01-15")
- ingest_time: When the feature was written to the system (e.g., "computed at 20:00")

Point-in-Time query rule:
  WHERE event_time <= as_of_time AND ingest_time <= as_of_time
```

**Why do we need two timestamps?**

```
Scenario: Backtest a trading decision at 2024-01-16 09:30

If using only event_time:
  Query: event_time <= '2024-01-16 09:30'
  May return data with event_time='2024-01-15' but ingest_time='2024-01-16 22:00'
  -> Look-ahead bias!

Correct dual-timestamp query:
  Query: event_time <= '2024-01-16 09:30' AND ingest_time <= '2024-01-16 09:30'
  Only returns features that were actually available at that time
```

### Database Design (TimescaleDB)

```sql
-- TimescaleDB is a PostgreSQL extension optimized for time-series data
CREATE TABLE IF NOT EXISTS feature_events (
    entity_key       TEXT NOT NULL,               -- e.g., 'AAPL.NASDAQ'
    feature_name     TEXT NOT NULL,               -- Feature name
    feature_version  INT  NOT NULL DEFAULT 1,     -- Version (increment when calculation logic changes)

    event_time       TIMESTAMPTZ NOT NULL,        -- Business time
    value_double     DOUBLE PRECISION,            -- Numeric features
    value_json       JSONB,                       -- Complex features (vectors, etc.)

    ingest_time      TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Traceability
    producer         TEXT,                        -- Producer (e.g., 'momentum_job')
    producer_version TEXT,                        -- Code version (git SHA)
    run_id           TEXT,                        -- Job ID

    PRIMARY KEY (entity_key, feature_name, feature_version, event_time)
);

-- Convert to hypertable for automatic partitioning
SELECT create_hypertable('feature_events', 'event_time', if_not_exists => TRUE);

-- Optimize for latest feature queries
CREATE INDEX IF NOT EXISTS idx_feature_events_latest
    ON feature_events (entity_key, feature_name, feature_version, event_time DESC);

-- Compression policy (compress after 7 days, saves 90%+ space)
ALTER TABLE feature_events SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'entity_key, feature_name, feature_version',
    timescaledb.compress_orderby = 'event_time DESC'
);
SELECT add_compression_policy('feature_events', INTERVAL '7 days');
```

### Python Implementation

```python
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

@dataclass
class FeatureValue:
    """Feature value returned from queries"""
    entity_key: str
    feature_name: str
    feature_version: int
    event_time: datetime
    value: float | dict[str, Any]


class FeatureStore:
    """
    TimescaleDB-backed Feature Store

    Core functions:
    1. write_features: Write features
    2. get_latest: Get latest feature values
    3. get_point_in_time: Batch Point-in-Time queries (for training set construction)
    """

    def __init__(self, conninfo: str, producer: str | None = None):
        self._conninfo = conninfo
        self._producer = producer

    def write_features(
        self,
        entity_key: str,
        timestamp: datetime,
        features: dict[str, float],
        *,
        feature_version: int = 1,
        availability_lag: timedelta | None = None,
    ) -> int:
        """
        Write feature values

        Args:
            entity_key: Entity identifier (e.g., 'AAPL.NASDAQ')
            timestamp: Business time of the feature (event_time)
            features: Feature dictionary {feature_name: value}
            feature_version: Feature version (increment when calculation logic changes)
            availability_lag: Data availability delay (used for backfilling)
                If a feature is only available at T+1, set availability_lag=timedelta(days=1)
                This makes ingest_time = event_time + 1 day

        Returns:
            Number of features written
        """
        if not features:
            return 0

        # Calculate ingest_time
        ingest_time = datetime.now()
        if availability_lag is not None:
            ingest_time = timestamp + availability_lag

        # Build batch insert (ON CONFLICT ensures idempotency)
        sql = """
            INSERT INTO feature_events
                (entity_key, feature_name, feature_version, event_time, value_double, ingest_time, producer)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (entity_key, feature_name, feature_version, event_time) DO NOTHING
        """

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                for name, value in features.items():
                    cur.execute(sql, [
                        entity_key, name, feature_version,
                        timestamp, float(value), ingest_time, self._producer
                    ])
            conn.commit()

        return len(features)

    def get_latest(
        self,
        entity_key: str,
        feature_names: list[str] | None = None,
        *,
        as_of: datetime | None = None,
    ) -> dict[str, FeatureValue]:
        """
        Get latest feature values for an entity

        Args:
            entity_key: Entity identifier
            feature_names: List of features to query (None for all)
            as_of: Point-in-Time timestamp (None for current)

        Returns:
            {feature_name: FeatureValue}
        """
        # Key: dual timestamp filtering
        sql = """
            SELECT DISTINCT ON (feature_name, feature_version)
                feature_name, feature_version, value_double, event_time
            FROM feature_events
            WHERE entity_key = %s
              AND feature_version = 1
        """
        params = [entity_key]

        # Point-in-Time filter
        if as_of is not None:
            sql += " AND event_time <= %s AND ingest_time <= %s"
            params.extend([as_of, as_of])

        # Feature name filter
        if feature_names:
            sql += " AND feature_name = ANY(%s)"
            params.append(feature_names)

        sql += " ORDER BY feature_name, feature_version, event_time DESC"

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

        return {
            row[0]: FeatureValue(
                entity_key=entity_key,
                feature_name=row[0],
                feature_version=row[1],
                event_time=row[3],
                value=row[2],
            )
            for row in rows
        }

    def get_point_in_time(
        self,
        entity_times: list[tuple[str, datetime]],
        feature_names: list[str] | None = None,
    ) -> list[FeatureValue]:
        """
        Batch Point-in-Time query (core method for building training sets)

        Args:
            entity_times: [(entity_key, as_of_time), ...]
            feature_names: List of features to query

        Returns:
            For each (entity, time) pair, returns the latest available features at that time
        """
        if not entity_times:
            return []

        # Use CTE and DISTINCT ON for efficient PIT queries
        values_sql = ", ".join(["(%s, %s)"] * len(entity_times))
        params = []
        for entity_key, as_of_time in entity_times:
            params.extend([entity_key, as_of_time])

        sql = f"""
        WITH entity_times(entity_key, as_of_time) AS (
            VALUES {values_sql}
        )
        SELECT DISTINCT ON (et.entity_key, fe.feature_name)
            et.entity_key,
            et.as_of_time,
            fe.feature_name,
            fe.feature_version,
            fe.value_double,
            fe.event_time AS feature_time
        FROM entity_times et
        JOIN feature_events fe
            ON fe.entity_key = et.entity_key
           AND fe.event_time <= et.as_of_time
           AND fe.ingest_time <= et.as_of_time
        WHERE fe.feature_version = 1
        ORDER BY et.entity_key, fe.feature_name, fe.event_time DESC
        """

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

        return [
            FeatureValue(
                entity_key=row[0],
                feature_name=row[2],
                feature_version=row[3],
                event_time=row[5],
                value=row[4],
            )
            for row in rows
        ]
```

### Usage Examples

```python
# Initialize
store = FeatureStore(
    conninfo="postgres://localhost:5432/trading",
    producer="momentum_job_v2"
)

# Write features
store.write_features(
    entity_key="AAPL.NASDAQ",
    timestamp=datetime(2024, 1, 15, 16, 0),  # Market close
    features={
        "momentum_5d": 0.035,
        "rsi_14": 62.5,
        "volume_ratio": 1.15,
    }
)

# Real-time inference: get latest features
latest = store.get_latest("AAPL.NASDAQ", ["momentum_5d", "rsi_14"])
print(f"Latest RSI: {latest['rsi_14'].value}")

# Build training set: Point-in-Time query
training_dates = [
    ("AAPL.NASDAQ", datetime(2024, 1, 10, 9, 30)),
    ("AAPL.NASDAQ", datetime(2024, 1, 11, 9, 30)),
    ("AAPL.NASDAQ", datetime(2024, 1, 12, 9, 30)),
    ("MSFT.NASDAQ", datetime(2024, 1, 10, 9, 30)),
    ("MSFT.NASDAQ", datetime(2024, 1, 11, 9, 30)),
]

features = store.get_point_in_time(training_dates, ["momentum_5d", "rsi_14"])
# Returns feature values available at each point in time, with no look-ahead bias
```

---

## 3. Model Registry

### Why Model Registration?

```
Scenario: Model performance drops, need to investigate

Without registry:
  - "Which version is running now?" -> Unknown
  - "What are this version's parameters?" -> Search through files
  - "Where is the previous version?" -> Possibly overwritten
  - "What was this version's backtest performance?" -> Run again

With registry:
  SELECT * FROM models WHERE name = 'momentum_v2';
  -> Version, parameters, metrics, training time, code version at a glance
```

### Database Design

```sql
-- Model metadata
CREATE TABLE IF NOT EXISTS models (
    model_id      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name          TEXT NOT NULL,
    version       INT NOT NULL,
    strategy_type TEXT,                  -- 'momentum', 'mean_reversion', etc.
    description   TEXT,
    created_at    TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(name, version)
);

-- Model metrics
CREATE TABLE IF NOT EXISTS model_metrics (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id      UUID REFERENCES models(model_id),
    metric_name   TEXT NOT NULL,         -- 'sharpe_ratio', 'ic', 'max_drawdown'
    value         DOUBLE PRECISION,
    dataset_type  TEXT,                  -- 'train', 'val', 'test', 'backtest', 'live'
    evaluated_at  TIMESTAMPTZ DEFAULT NOW()
);

-- Model artifacts (weight files, etc.)
CREATE TABLE IF NOT EXISTS model_artifacts (
    artifact_id   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id      UUID REFERENCES models(model_id),
    artifact_path TEXT NOT NULL,         -- 's3://models/momentum_v2/weights.pkl'
    artifact_type TEXT,                  -- 'weights', 'config', 'scaler', 'onnx'
    checksum      TEXT,                  -- SHA256
    size_bytes    BIGINT,
    created_at    TIMESTAMPTZ DEFAULT NOW()
);

-- Training run records
CREATE TABLE IF NOT EXISTS model_training_runs (
    run_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id      UUID REFERENCES models(model_id),
    params        JSONB,                 -- Training hyperparameters
    dataset_start TIMESTAMPTZ,
    dataset_end   TIMESTAMPTZ,
    started_at    TIMESTAMPTZ,
    finished_at   TIMESTAMPTZ,
    status        TEXT DEFAULT 'running' -- 'running', 'completed', 'failed'
);
```

### Python Implementation

```python
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import UUID
import hashlib
import json


@dataclass
class ModelInfo:
    """Model metadata"""
    model_id: UUID
    name: str
    version: int
    strategy_type: str | None
    description: str | None
    created_at: datetime


@dataclass
class ModelWithMetrics:
    """Model with its metrics"""
    model: ModelInfo
    metrics: dict[str, float]  # {metric_name_dataset: value}


class ModelRegistry:
    """
    Model Registry

    Functions:
    1. register_model: Register new model version
    2. log_metrics: Record evaluation metrics
    3. log_artifact: Record model artifacts
    4. get_best_model: Get best-performing model
    """

    def __init__(self, dsn: str):
        self.dsn = dsn

    def register_model(
        self,
        name: str,
        strategy_type: str | None = None,
        params: dict | None = None,
        description: str | None = None,
        version: int | None = None,
    ) -> UUID:
        """
        Register a new model version

        Args:
            name: Model name (e.g., 'momentum_v2')
            strategy_type: Strategy type
            params: Training parameters
            description: Description
            version: Version number (None for auto-increment)

        Returns:
            Model UUID
        """
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                # Auto version number
                if version is None:
                    cur.execute(
                        "SELECT COALESCE(MAX(version), 0) + 1 FROM models WHERE name = %s",
                        (name,)
                    )
                    version = cur.fetchone()[0]

                # Insert model
                cur.execute(
                    """
                    INSERT INTO models (name, version, strategy_type, description)
                    VALUES (%s, %s, %s, %s)
                    RETURNING model_id
                    """,
                    (name, version, strategy_type, description)
                )
                model_id = cur.fetchone()[0]

                # Record training parameters
                if params:
                    cur.execute(
                        """
                        INSERT INTO model_training_runs (model_id, params, started_at, status)
                        VALUES (%s, %s, %s, 'completed')
                        """,
                        (model_id, json.dumps(params), datetime.now())
                    )

            conn.commit()
            return model_id

    def log_metrics(
        self,
        model_id: UUID,
        metrics: dict[str, float],
        dataset_type: str | None = None,
    ) -> None:
        """
        Record model metrics

        Args:
            model_id: Model UUID
            metrics: {metric_name: value}, e.g., {'sharpe_ratio': 1.5, 'ic': 0.04}
            dataset_type: Dataset type ('train', 'val', 'test', 'backtest', 'live')
        """
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                for metric_name, value in metrics.items():
                    cur.execute(
                        """
                        INSERT INTO model_metrics (model_id, metric_name, value, dataset_type)
                        VALUES (%s, %s, %s, %s)
                        """,
                        (model_id, metric_name, value, dataset_type)
                    )
            conn.commit()

    def log_artifact(
        self,
        model_id: UUID,
        path: str | Path,
        artifact_type: str | None = None,
    ) -> UUID:
        """
        Record model artifact

        Args:
            model_id: Model UUID
            path: Artifact path (local or S3)
            artifact_type: Type ('weights', 'config', 'scaler')

        Returns:
            Artifact UUID
        """
        path = Path(path)
        checksum = None
        size_bytes = None

        if path.exists():
            size_bytes = path.stat().st_size
            # Calculate SHA256
            sha256 = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    sha256.update(chunk)
            checksum = sha256.hexdigest()

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO model_artifacts
                        (model_id, artifact_path, artifact_type, checksum, size_bytes)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING artifact_id
                    """,
                    (model_id, str(path), artifact_type, checksum, size_bytes)
                )
                artifact_id = cur.fetchone()[0]
            conn.commit()
            return artifact_id

    def get_model(self, name: str, version: int | None = None) -> ModelInfo | None:
        """Get model (defaults to latest version)"""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                if version is None:
                    cur.execute(
                        """
                        SELECT model_id, name, version, strategy_type, description, created_at
                        FROM models WHERE name = %s
                        ORDER BY version DESC LIMIT 1
                        """,
                        (name,)
                    )
                else:
                    cur.execute(
                        """
                        SELECT model_id, name, version, strategy_type, description, created_at
                        FROM models WHERE name = %s AND version = %s
                        """,
                        (name, version)
                    )

                row = cur.fetchone()
                if row:
                    return ModelInfo(*row)
                return None

    def get_best_model(
        self,
        strategy_type: str,
        metric_name: str,
        dataset_type: str = "test",
        higher_is_better: bool = True,
    ) -> ModelWithMetrics | None:
        """
        Get best-performing model for a strategy type

        Args:
            strategy_type: Strategy type
            metric_name: Sorting metric (e.g., 'sharpe_ratio')
            dataset_type: Dataset type
            higher_is_better: Whether higher values are better

        Returns:
            Best model with its metrics
        """
        order = "DESC" if higher_is_better else "ASC"

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT m.model_id, m.name, m.version, m.strategy_type,
                           m.description, m.created_at, mm.value
                    FROM models m
                    JOIN model_metrics mm ON m.model_id = mm.model_id
                    WHERE m.strategy_type = %s
                      AND mm.metric_name = %s
                      AND mm.dataset_type = %s
                    ORDER BY mm.value {order}
                    LIMIT 1
                    """,
                    (strategy_type, metric_name, dataset_type)
                )

                row = cur.fetchone()
                if not row:
                    return None

                model = ModelInfo(*row[:6])

                # Get all metrics for this model
                cur.execute(
                    """
                    SELECT metric_name, value, dataset_type
                    FROM model_metrics
                    WHERE model_id = %s
                    """,
                    (model.model_id,)
                )

                metrics = {
                    f"{r[0]}_{r[2]}": r[1]
                    for r in cur.fetchall()
                }

                return ModelWithMetrics(model=model, metrics=metrics)
```

### Usage Examples

```python
registry = ModelRegistry(dsn="postgres://localhost:5432/trading")

# Register new model
model_id = registry.register_model(
    name="momentum_xgb",
    strategy_type="momentum",
    params={
        "n_estimators": 100,
        "max_depth": 5,
        "learning_rate": 0.1,
        "features": ["ret_5d", "ret_20d", "vol_20d", "rsi_14"],
    },
    description="XGBoost momentum model with RSI features"
)

# Record backtest metrics
registry.log_metrics(model_id, {
    "sharpe_ratio": 1.65,
    "total_return": 0.28,
    "max_drawdown": 0.12,
    "ic": 0.042,
    "ir": 0.85,
}, dataset_type="backtest")

# Record test set metrics
registry.log_metrics(model_id, {
    "sharpe_ratio": 1.35,
    "ic": 0.035,
}, dataset_type="test")

# Save model artifacts
registry.log_artifact(model_id, "models/momentum_xgb_v3.pkl", "weights")
registry.log_artifact(model_id, "models/momentum_xgb_v3_config.json", "config")

# Get best momentum model
best = registry.get_best_model("momentum", "sharpe_ratio", "test")
if best:
    print(f"Best model: {best.model.name} v{best.model.version}")
    print(f"Test Sharpe: {best.metrics.get('sharpe_ratio_test', 'N/A')}")
```

---

## 4. Drift Monitor

### Three Dimensions of Drift

| Dimension | Detection Metric | Meaning | Threshold Recommendation |
|-----------|-----------------|---------|-------------------------|
| **Data Drift** | PSI | Feature distribution changes | < 0.10 normal, > 0.25 severe |
| **Prediction Drift** | IC | Correlation between predictions and actual returns | > 0.02 normal, < 0.01 severe |
| **Performance Drift** | Rolling Sharpe | Strategy risk-adjusted returns | > 0.5 normal, < 0 severe |

### Core Metric Calculations

```python
import numpy as np
from scipy.stats import spearmanr


def calculate_ic(signals: np.ndarray, returns: np.ndarray) -> float:
    """
    Calculate Information Coefficient

    IC = Spearman correlation(predicted signals, actual returns)

    Interpretation:
    - IC > 0.05: Excellent
    - IC 0.02-0.05: Good
    - IC < 0.02: Needs attention
    - IC < 0: Model may have issues
    """
    if len(signals) < 2:
        return 0.0

    # Remove NaNs
    mask = ~(np.isnan(signals) | np.isnan(returns))
    signals, returns = signals[mask], returns[mask]

    if len(signals) < 2:
        return 0.0

    ic, _ = spearmanr(signals, returns)
    return float(ic) if not np.isnan(ic) else 0.0


def calculate_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    bins: int = 10,
) -> float:
    """
    Calculate Population Stability Index (PSI)

    PSI = sum((actual% - expected%) * ln(actual% / expected%))

    Interpretation:
    - PSI < 0.10: Distribution stable
    - PSI 0.10-0.25: Mild drift, monitor
    - PSI > 0.25: Significant drift, action needed
    """
    eps = 1e-6

    # Create bins based on baseline distribution
    _, bin_edges = np.histogram(expected, bins=bins)

    # Calculate proportions in each bin
    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)

    expected_pct = expected_counts / len(expected) + eps
    actual_pct = actual_counts / len(actual) + eps

    # PSI formula
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))

    return float(psi)


def calculate_sharpe(
    returns: np.ndarray,
    periods_per_year: int = 252,
) -> float:
    """
    Calculate annualized Sharpe ratio

    Sharpe = mean(returns) / std(returns) * sqrt(252)
    """
    returns = returns[~np.isnan(returns)]

    if len(returns) < 2:
        return 0.0

    mean_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1)

    if std_ret < 1e-10:
        return 0.0

    return (mean_ret / std_ret) * np.sqrt(periods_per_year)
```

### Drift Monitor Implementation

```python
from dataclasses import dataclass
from datetime import date


@dataclass
class DriftMetrics:
    """Daily drift metrics"""
    date: date
    strategy_id: str
    ic: float | None = None
    ic_5d_avg: float | None = None
    psi: float | None = None
    sharpe_5d: float | None = None
    sharpe_20d: float | None = None
    ic_alert: bool = False
    psi_alert: bool = False
    sharpe_alert: bool = False


@dataclass
class AlertConfig:
    """Alert threshold configuration"""
    ic_warning: float = 0.02
    ic_critical: float = 0.01
    psi_warning: float = 0.10
    psi_critical: float = 0.25
    sharpe_warning: float = 0.5
    sharpe_critical: float = 0.0


class DriftMonitor:
    """
    Drift monitoring service

    Runs daily, calculates IC, PSI, Sharpe metrics, stores to database, triggers alerts.
    """

    def __init__(self, dsn: str, strategy_id: str = "default"):
        self.dsn = dsn
        self.strategy_id = strategy_id
        self.config = AlertConfig()

    def calculate_metrics(self, target_date: date) -> DriftMetrics:
        """Calculate drift metrics for a given date"""
        metrics = DriftMetrics(date=target_date, strategy_id=self.strategy_id)

        # Get signals and returns
        signals, returns = self._get_signals_and_returns(target_date)
        if len(signals) > 0:
            metrics.ic = calculate_ic(signals, returns)

        # Get historical returns to calculate Sharpe
        daily_returns = self._get_daily_returns(lookback_days=60)
        if len(daily_returns) >= 5:
            metrics.sharpe_5d = calculate_sharpe(daily_returns[-5:])
        if len(daily_returns) >= 20:
            metrics.sharpe_20d = calculate_sharpe(daily_returns[-20:])

        # Check alerts
        if metrics.ic is not None:
            metrics.ic_alert = metrics.ic < self.config.ic_critical
        if metrics.psi is not None:
            metrics.psi_alert = metrics.psi > self.config.psi_critical
        if metrics.sharpe_20d is not None:
            metrics.sharpe_alert = metrics.sharpe_20d < self.config.sharpe_critical

        return metrics

    def save_metrics(self, metrics: DriftMetrics) -> None:
        """Save metrics to database"""
        sql = """
            INSERT INTO drift_metrics (
                date, strategy_id, ic, ic_5d_avg, psi, sharpe_5d, sharpe_20d,
                ic_alert, psi_alert, sharpe_alert
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (date, strategy_id) DO UPDATE SET
                ic = EXCLUDED.ic,
                psi = EXCLUDED.psi,
                sharpe_5d = EXCLUDED.sharpe_5d,
                sharpe_20d = EXCLUDED.sharpe_20d,
                ic_alert = EXCLUDED.ic_alert,
                psi_alert = EXCLUDED.psi_alert,
                sharpe_alert = EXCLUDED.sharpe_alert
        """
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, [
                    metrics.date, metrics.strategy_id,
                    metrics.ic, metrics.ic_5d_avg, metrics.psi,
                    metrics.sharpe_5d, metrics.sharpe_20d,
                    metrics.ic_alert, metrics.psi_alert, metrics.sharpe_alert,
                ])
            conn.commit()

    def run_daily(self, target_date: date | None = None) -> DriftMetrics:
        """Daily drift monitoring job"""
        if target_date is None:
            target_date = date.today()

        print(f"Running drift monitoring for {target_date}")

        metrics = self.calculate_metrics(target_date)
        self.save_metrics(metrics)

        # Output alerts
        if metrics.ic_alert:
            print(f"[ALERT] IC = {metrics.ic:.4f} below threshold {self.config.ic_critical}")
        if metrics.psi_alert:
            print(f"[ALERT] PSI = {metrics.psi:.4f} above threshold {self.config.psi_critical}")
        if metrics.sharpe_alert:
            print(f"[ALERT] Sharpe = {metrics.sharpe_20d:.4f} below threshold {self.config.sharpe_critical}")

        return metrics
```

### Alert Response Matrix

| Alert Type | Severity | Recommended Action |
|------------|----------|-------------------|
| IC < 0.02 for 5 consecutive days | Warning | Check if feature calculations are correct |
| IC < 0.01 | Severe | Reduce position by 50%, start model diagnostics |
| IC < 0 for 3 consecutive days | Critical | Pause strategy, manual review |
| PSI > 0.10 | Warning | Monitor subsequent changes |
| PSI > 0.25 | Severe | Trigger retraining process |
| Sharpe < 0.5 for 10 consecutive days | Warning | Check market conditions |
| Sharpe < 0 for 5 consecutive days | Severe | Reduce position, prepare for retraining |

---

## 5. Integration: Research to Production

### Complete Workflow

```
+----------------------------------------------------------------------+
|                          Research Phase                               |
+----------------------------------------------------------------------+
|  1. Feature Development                                               |
|     +-> Write to Feature Store (set correct availability_lag)         |
|                                                                       |
|  2. Build Training Set                                                |
|     +-> FeatureStore.get_point_in_time()                              |
|     +-> Export Parquet (immutable snapshot)                           |
|                                                                       |
|  3. Model Training                                                    |
|     +-> Record parameters, code version                               |
|     +-> Register to Model Registry                                    |
|                                                                       |
|  4. Backtest Evaluation                                               |
|     +-> log_metrics(dataset_type='backtest')                          |
+----------------------------------------------------------------------+
                               |
                               v
+----------------------------------------------------------------------+
|                          Deployment Phase                             |
+----------------------------------------------------------------------+
|  5. Model Selection                                                   |
|     +-> get_best_model(strategy_type, metric, dataset_type='test')    |
|                                                                       |
|  6. Load Model                                                        |
|     +-> Load weights from artifact_path                               |
|     +-> Verify checksum                                               |
+----------------------------------------------------------------------+
                               |
                               v
+----------------------------------------------------------------------+
|                          Runtime Phase                                |
+----------------------------------------------------------------------+
|  7. Real-time Inference                                               |
|     +-> FeatureStore.get_latest() to get features                     |
|     +-> Model prediction                                              |
|     +-> Output signals                                                |
|                                                                       |
|  8. Daily Monitoring                                                  |
|     +-> Drift Monitor calculates IC/PSI/Sharpe                        |
|     +-> Trigger alerts                                                |
|                                                                       |
|  9. Retrain (if needed)                                               |
|     +-> Return to step 2                                              |
+----------------------------------------------------------------------+
```

### Reproducibility Checklist

| Check Item | How to Implement | Verification Method |
|------------|-----------------|---------------------|
| **Code version** | Record git SHA | `producer_version` field |
| **Feature version** | `feature_version` column | Specify version in queries |
| **Training data** | Parquet snapshot + fingerprint | Retraining should yield same results |
| **Model parameters** | `model_training_runs.params` | JSON storage |
| **Model weights** | `model_artifacts.checksum` | SHA256 verification |
| **Evaluation metrics** | `model_metrics` table | Trace by time |

### Daily Operations Script Example

```python
from datetime import date, datetime

def daily_mlops_job(
    feature_store: FeatureStore,
    model_registry: ModelRegistry,
    drift_monitor: DriftMonitor,
    strategy_id: str,
):
    """Daily MLOps job"""
    today = date.today()
    print(f"=== MLOps Daily Job: {today} ===")

    # 1. Feature health check
    print("\n[1] Feature Health Check")
    latest = feature_store.get_latest("AAPL.NASDAQ")
    for name, fv in latest.items():
        age_hours = (datetime.now() - fv.event_time).total_seconds() / 3600
        if age_hours > 24:
            print(f"  WARNING: {name} is {age_hours:.1f} hours old")
        else:
            print(f"  OK: {name} updated {age_hours:.1f} hours ago")

    # 2. Model status check
    print("\n[2] Model Status Check")
    current_model = model_registry.get_model("momentum_xgb")
    if current_model:
        print(f"  Current: {current_model.name} v{current_model.version}")
        print(f"  Created: {current_model.created_at}")

    # 3. Drift monitoring
    print("\n[3] Drift Monitoring")
    drift_metrics = drift_monitor.run_daily(today)
    print(f"  IC: {drift_metrics.ic}")
    print(f"  Sharpe (20d): {drift_metrics.sharpe_20d}")

    # 4. Decision
    if drift_metrics.ic_alert or drift_metrics.sharpe_alert:
        print("\n[ACTION REQUIRED] Consider retraining or reducing position size")
    else:
        print("\n[OK] All metrics within normal range")


# Scheduled job (e.g., cron)
# 0 6 * * * python -c "from mlops import daily_mlops_job; daily_mlops_job(...)"
```

---

## Further Reading

- *Machine Learning Design Patterns* by Lakshmanan, Robinson, and Munn
- [Model Drift and Retraining Strategies](Model-Drift-and-Retraining.md)
