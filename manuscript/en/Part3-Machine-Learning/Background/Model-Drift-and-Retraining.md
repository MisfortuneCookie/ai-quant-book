# Background: Model Drift and Retraining Strategies

> Financial markets are non-stationary. A model that works today may fail tomorrow. Detecting drift and timely retraining are essential capabilities for production-grade systems.

---

## 1. What is Model Drift?

Model drift refers to the phenomenon where a model's predictive performance gradually degrades over time after deployment.

### 1.1 Two Types of Drift

| Type | Definition | Financial Example |
|------|------------|-------------------|
| **Data Drift** | Distribution of input features changes | Volatility rises from 15% to 40% (COVID crisis) |
| **Concept Drift** | Relationship between features and target changes | Momentum factor becomes ineffective (regime switch) |

### 1.2 Root Causes of Drift in Financial Markets

```
Why do financial models inevitably drift?

1. Changes in Market Participant Structure
   - Retail investor influx → Momentum effects strengthen
   - More quant funds → Alpha decay

2. Macroeconomic Environment Changes
   - Interest rate cycle shifts (QE → Tightening)
   - Economic cycle transitions (Expansion → Recession)

3. Regulatory Policy Changes
   - Short-selling restrictions → Price discovery mechanism changes
   - HFT regulations → Market microstructure changes

4. Technology and Information Changes
   - New data sources emerge → Old factors get front-run
   - AI proliferation → Strategy homogenization
```

---

## 2. Drift Detection Methods

### 2.1 Performance Monitoring

**The most direct approach**: Monitor strategy performance over a rolling window.

```python
import numpy as np

class PerformanceMonitor:
    """Performance drift monitor"""

    def __init__(self, window: int = 30, sharpe_threshold: float = 0.5):
        self.window = window  # Rolling window (days)
        self.sharpe_threshold = sharpe_threshold
        self.returns = []

    def update(self, daily_return: float) -> dict:
        """Update and check for drift"""
        self.returns.append(daily_return)

        if len(self.returns) < self.window:
            return {'status': 'warming_up'}

        # Calculate rolling Sharpe
        recent = self.returns[-self.window:]
        rolling_sharpe = np.mean(recent) / np.std(recent) * np.sqrt(252)

        # Detect drift
        is_drifting = rolling_sharpe < self.sharpe_threshold

        return {
            'rolling_sharpe': rolling_sharpe,
            'is_drifting': is_drifting,
            'alert': 'DRIFT_DETECTED' if is_drifting else 'OK'
        }
```

**Threshold Setting Recommendations**:
| Metric | Warning Threshold | Critical Threshold | Triggered Action |
|--------|------------------|-------------------|------------------|
| Rolling Sharpe | < 0.5 | < 0 | Trigger retraining |
| Rolling Win Rate | < 45% | < 40% | Check signal quality |
| Rolling Return | < -5% | < -10% | Reduce position size |

---

### 2.2 Statistical Testing Methods

#### Kolmogorov-Smirnov Test (K-S Test)

Detects whether feature distributions have changed significantly.

```python
import numpy as np
from scipy.stats import ks_2samp

def detect_data_drift(
    training_data: np.ndarray,
    recent_data: np.ndarray,
    significance: float = 0.05
) -> dict:
    """
    K-S test for data drift detection

    Principle: Compare whether two samples come from the same distribution
    H0: Two samples come from the same distribution
    If p < significance, reject H0 and conclude drift has occurred
    """
    statistic, p_value = ks_2samp(training_data, recent_data)

    return {
        'ks_statistic': statistic,  # D-value, larger means greater distribution difference
        'p_value': p_value,
        'is_drifting': p_value < significance,
        'interpretation': 'DRIFT' if p_value < significance else 'STABLE'
    }

# Usage example
training_returns = returns['2020-01':'2022-12']
recent_returns = returns['2024-01':'2024-03']

result = detect_data_drift(training_returns, recent_returns)
print(f"K-S statistic: {result['ks_statistic']:.4f}")
print(f"P-value: {result['p_value']:.4f}")
print(f"Status: {result['interpretation']}")
```

#### Chi-Square Test

Suitable for drift detection in categorical features.

```python
from scipy.stats import chi2_contingency

def detect_categorical_drift(
    training_counts: dict,
    recent_counts: dict,
    significance: float = 0.05
) -> dict:
    """
    Chi-square test for categorical feature drift

    Example: Detect if market regime label distribution has changed
    training_counts = {'bull': 120, 'bear': 80, 'sideways': 50}
    recent_counts = {'bull': 10, 'bear': 35, 'sideways': 5}
    """
    # Build contingency table
    categories = set(training_counts.keys()) | set(recent_counts.keys())
    train_freq = [training_counts.get(c, 0) for c in categories]
    recent_freq = [recent_counts.get(c, 0) for c in categories]

    contingency_table = [train_freq, recent_freq]
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)

    return {
        'chi2_statistic': chi2,
        'p_value': p_value,
        'degrees_of_freedom': dof,
        'is_drifting': p_value < significance
    }
```

---

### 2.3 CUSUM Control Chart Method

**Cumulative Sum Control Chart**: Detects persistent shifts in prediction errors.

```python
class CUSUMDetector:
    """
    CUSUM (Cumulative Sum) drift detector

    Principle:
    - Accumulates deviation in prediction errors
    - If errors are random, cumulative sum should fluctuate around 0
    - If systematic bias exists, cumulative sum will drift persistently
    """

    def __init__(self, threshold: float = 5.0, drift: float = 0.5):
        """
        Parameters:
        - threshold: Alert trigger threshold
        - drift: Allowed drift amount (sensitivity control)
        """
        self.threshold = threshold
        self.drift = drift
        self.reset()

    def reset(self):
        self.s_pos = 0  # Positive cumulative sum
        self.s_neg = 0  # Negative cumulative sum
        self.history = []

    def update(self, error: float) -> dict:
        """
        Update CUSUM values

        Parameters:
        - error: Prediction error (predicted value - actual value)

        Returns:
        - Drift detection result
        """
        # Normalized error
        normalized_error = error

        # Update cumulative sums
        self.s_pos = max(0, self.s_pos + normalized_error - self.drift)
        self.s_neg = max(0, self.s_neg - normalized_error - self.drift)

        self.history.append({
            's_pos': self.s_pos,
            's_neg': self.s_neg,
            'error': error
        })

        # Detect drift
        drift_up = self.s_pos > self.threshold
        drift_down = self.s_neg > self.threshold

        if drift_up or drift_down:
            direction = 'UP' if drift_up else 'DOWN'
            return {
                'is_drifting': True,
                'direction': direction,
                'cusum_value': self.s_pos if drift_up else self.s_neg,
                'action': 'RETRAIN_RECOMMENDED'
            }

        return {
            'is_drifting': False,
            'cusum_pos': self.s_pos,
            'cusum_neg': self.s_neg,
            'action': 'CONTINUE_MONITORING'
        }

# Usage example
detector = CUSUMDetector(threshold=5.0, drift=0.5)

for pred, actual in zip(predictions, actuals):
    error = pred - actual
    result = detector.update(error)
    if result['is_drifting']:
        print(f"Drift detected! Direction: {result['direction']}")
        break
```

**Advantages of CUSUM**:
- Can detect gradual, small persistent shifts
- More sensitive than single-point detection
- Has solid statistical foundation

---

### 2.4 Multi-Indicator Comprehensive Detection

**Production-grade recommendation**: Combine multiple detection methods to reduce false positive rate.

```python
class ComprehensiveDriftDetector:
    """Comprehensive drift detector"""

    def __init__(self):
        self.performance_monitor = PerformanceMonitor()
        self.cusum_detector = CUSUMDetector()

    def check_drift(self,
                   daily_return: float,
                   prediction_error: float,
                   training_features: np.array,
                   recent_features: np.array) -> dict:

        results = {}

        # 1. Performance monitoring
        perf_result = self.performance_monitor.update(daily_return)
        results['performance'] = perf_result

        # 2. CUSUM detection
        cusum_result = self.cusum_detector.update(prediction_error)
        results['cusum'] = cusum_result

        # 3. K-S test (run periodically, e.g., weekly)
        ks_result = detect_data_drift(training_features, recent_features)
        results['ks_test'] = ks_result

        # Comprehensive judgment: majority voting
        drift_signals = [
            perf_result.get('is_drifting', False),
            cusum_result.get('is_drifting', False),
            ks_result.get('is_drifting', False)
        ]

        drift_count = sum(drift_signals)

        results['overall'] = {
            'drift_count': drift_count,
            'is_drifting': drift_count >= 2,  # At least 2 detectors alarming
            'confidence': drift_count / 3,
            'recommendation': self._get_recommendation(drift_count)
        }

        return results

    def _get_recommendation(self, drift_count: int) -> str:
        if drift_count == 0:
            return 'CONTINUE_NORMAL'
        elif drift_count == 1:
            return 'INCREASE_MONITORING'
        elif drift_count == 2:
            return 'PREPARE_RETRAIN'
        else:
            return 'IMMEDIATE_RETRAIN'
```

---

## 3. Retraining Strategies

### 3.1 Scheduled Retraining

**The simplest strategy**: Retrain models on a fixed schedule.

| Strategy Frequency | Period | Applicable Scenario | Pros | Cons |
|-------------------|--------|---------------------|------|------|
| Daily strategy | Monthly | Medium-low frequency factor strategies | Simple, predictable | May lag behind |
| Weekly strategy | Quarterly | Portfolio allocation strategies | Low cost | Cannot adapt to sudden changes |
| Minute-level strategy | Weekly | High-frequency trading | Timely updates | High cost |

```python
# Scheduled retraining scheduler
class ScheduledRetrainer:

    def __init__(self, retrain_frequency: str = 'monthly'):
        self.frequency = retrain_frequency
        self.last_retrain = None

    def should_retrain(self, current_date) -> bool:
        if self.last_retrain is None:
            return True

        if self.frequency == 'weekly':
            return (current_date - self.last_retrain).days >= 7
        elif self.frequency == 'monthly':
            return (current_date - self.last_retrain).days >= 30
        elif self.frequency == 'quarterly':
            return (current_date - self.last_retrain).days >= 90

        return False
```

---

### 3.2 Triggered Retraining

**A smarter strategy**: Trigger retraining only when drift is detected.

```python
class TriggeredRetrainer:
    """Triggered retrainer"""

    def __init__(self,
                 performance_threshold: float = 0.3,  # Sharpe threshold
                 cusum_threshold: float = 5.0,
                 min_interval_days: int = 7):  # Minimum retraining interval
        self.performance_threshold = performance_threshold
        self.cusum_threshold = cusum_threshold
        self.min_interval_days = min_interval_days
        self.last_retrain = None
        self.detector = ComprehensiveDriftDetector()

    def check_and_retrain(self, model, new_data, current_date) -> dict:
        """Check if retraining is needed, execute if necessary"""

        # Prevent overly frequent retraining
        if self.last_retrain:
            days_since = (current_date - self.last_retrain).days
            if days_since < self.min_interval_days:
                return {'action': 'SKIP', 'reason': 'Too soon since last retrain'}

        # Drift detection
        drift_result = self.detector.check_drift(...)

        if drift_result['overall']['is_drifting']:
            # Execute retraining
            new_model = self._retrain(model, new_data)
            self.last_retrain = current_date

            return {
                'action': 'RETRAINED',
                'drift_confidence': drift_result['overall']['confidence'],
                'new_model': new_model
            }

        return {'action': 'CONTINUE', 'drift_confidence': drift_result['overall']['confidence']}
```

---

### 3.3 Online Learning

**Continuous updates**: Instead of full retraining, incrementally update model parameters.

```python
class OnlineLearner:
    """
    Online learning updater

    Applicable scenarios:
    - Need to adapt quickly to market changes
    - Full retraining is too costly
    - Data streams arrive continuously

    Risks:
    - Catastrophic forgetting (losing historical patterns)
    - Sensitive to noise
    """

    def __init__(self, model, learning_rate: float = 0.001):
        self.model = model
        self.learning_rate = learning_rate
        self.update_count = 0

    def incremental_update(self, new_x, new_y):
        """
        Incrementally update the model

        Uses a small learning rate for single-step gradient descent
        """
        # Forward pass
        prediction = self.model.predict(new_x)
        error = new_y - prediction

        # Backward pass (simplified illustration)
        gradient = self._compute_gradient(new_x, error)

        # Parameter update
        for param, grad in zip(self.model.parameters(), gradient):
            param -= self.learning_rate * grad

        self.update_count += 1

        return {
            'prediction': prediction,
            'error': error,
            'update_count': self.update_count
        }

    def _compute_gradient(self, x, error):
        # Actual implementation depends on model type
        pass
```

**Pitfalls of Online Learning**:
1. **Catastrophic forgetting**: New data overwrites old knowledge
2. **Noise accumulation**: Single-sample updates are easily misled by noise
3. **Learning rate sensitivity**: Too large → unstable, too small → slow adaptation

---

### 3.4 Hybrid Strategy (Recommended)

**Best practice**: Combine scheduled and triggered retraining.

```python
class HybridRetrainer:
    """Hybrid retraining strategy"""

    def __init__(self):
        self.scheduled_interval_days = 30  # Scheduled: monthly
        self.drift_detector = ComprehensiveDriftDetector()
        self.last_scheduled_retrain = None
        self.last_triggered_retrain = None

    def should_retrain(self, current_date, metrics) -> dict:
        """Determine if retraining is needed"""

        # Check scheduled retraining
        scheduled_due = self._check_scheduled(current_date)

        # Check triggered retraining
        drift_result = self.drift_detector.check_drift(metrics)
        triggered_due = drift_result['overall']['is_drifting']

        if scheduled_due and triggered_due:
            return {
                'should_retrain': True,
                'reason': 'BOTH_SCHEDULED_AND_DRIFT',
                'priority': 'HIGH'
            }
        elif triggered_due:
            return {
                'should_retrain': True,
                'reason': 'DRIFT_DETECTED',
                'priority': 'HIGH'
            }
        elif scheduled_due:
            return {
                'should_retrain': True,
                'reason': 'SCHEDULED',
                'priority': 'NORMAL'
            }

        return {'should_retrain': False, 'reason': 'NO_TRIGGER'}
```

---

## 4. Best Practices for Retraining

### 4.1 Training Data Selection

| Strategy | Description | Pros | Cons |
|----------|-------------|------|------|
| **Expanding Window** | Use all historical data | Large sample size | Old data may be outdated |
| **Sliding Window** | Use only recent N days | Adapts to new patterns | May lose important history |
| **Weighted Window** | Higher weight for recent data | Balances history and present | Weight selection is difficult |

**Recommendation**: Sliding window + retain crisis period data

```python
def prepare_training_data(all_data, window_days=252*2, keep_crisis=True):
    """Prepare retraining data"""

    # Sliding window
    recent_data = all_data.iloc[-window_days:]

    if keep_crisis:
        # Retain important crisis period data
        crisis_periods = [
            ('2008-09', '2009-03'),  # Financial crisis
            ('2020-02', '2020-04'),  # COVID
            ('2022-01', '2022-06'),  # Rate hike shock
        ]

        crisis_data = []
        for start, end in crisis_periods:
            if start in all_data.index:
                crisis_data.append(all_data.loc[start:end])

        # Merge
        training_data = pd.concat([recent_data] + crisis_data)
        training_data = training_data.drop_duplicates()

    return training_data
```

### 4.2 Model Version Management

```python
# Model version management
class ModelVersionManager:

    def __init__(self, storage_path: str):
        self.storage_path = storage_path
        self.versions = []

    def save_version(self, model, metrics: dict, reason: str):
        """Save model version"""
        version_id = f"v{len(self.versions)+1}_{datetime.now():%Y%m%d_%H%M}"

        version_info = {
            'version_id': version_id,
            'timestamp': datetime.now(),
            'reason': reason,
            'metrics': metrics,
            'model_path': f"{self.storage_path}/{version_id}.pkl"
        }

        # Save model
        joblib.dump(model, version_info['model_path'])

        self.versions.append(version_info)
        return version_id

    def rollback(self, version_id: str):
        """Rollback to specified version"""
        for v in self.versions:
            if v['version_id'] == version_id:
                return joblib.load(v['model_path'])
        raise ValueError(f"Version {version_id} not found")
```

### 4.3 A/B Testing

After retraining, do not directly replace the old model. Instead, run a comparison test.

```python
class ABTester:
    """Model A/B testing"""

    def __init__(self, old_model, new_model, test_days: int = 5):
        self.old_model = old_model
        self.new_model = new_model
        self.test_days = test_days
        self.old_results = []
        self.new_results = []

    def run_comparison(self, data) -> dict:
        """Run comparison test"""

        for day_data in data:
            old_pred = self.old_model.predict(day_data)
            new_pred = self.new_model.predict(day_data)

            self.old_results.append(old_pred)
            self.new_results.append(new_pred)

        # Calculate performance comparison
        old_sharpe = calculate_sharpe(self.old_results)
        new_sharpe = calculate_sharpe(self.new_results)

        improvement = (new_sharpe - old_sharpe) / abs(old_sharpe) if old_sharpe != 0 else 0

        return {
            'old_sharpe': old_sharpe,
            'new_sharpe': new_sharpe,
            'improvement': improvement,
            'recommendation': 'DEPLOY_NEW' if improvement > 0.1 else 'KEEP_OLD'
        }
```

---

## 5. Production-Grade Drift Monitoring Architecture

The previous sections covered theoretical drift detection methods. This section presents a production-grade drift monitoring system implementation.

### 5.1 Core Design Patterns

Production systems require:
- **Multi-metric monitoring**: IC, PSI, and Sharpe tracked simultaneously
- **Configurable thresholds**: Different strategies have different tolerances
- **Persistent storage**: Drift history for analysis and audit
- **Alert levels**: Distinguish between warning and critical severity

#### AlertConfig Pattern

```python
from dataclasses import dataclass

@dataclass
class AlertConfig:
    """Alert threshold configuration"""

    # IC (Information Coefficient) thresholds
    ic_warning: float = 0.02    # IC < 0.02 triggers warning
    ic_critical: float = 0.01   # IC < 0.01 triggers critical alert

    # PSI (Population Stability Index) thresholds
    psi_warning: float = 0.10   # PSI > 0.10 indicates distribution shift
    psi_critical: float = 0.25  # PSI > 0.25 indicates significant shift

    # Sharpe thresholds
    sharpe_warning: float = 0.5   # Sharpe < 0.5 performance declining
    sharpe_critical: float = 0.0  # Sharpe < 0 strategy losing money
```

**Threshold Interpretation**:
| Metric | Warning Threshold | Critical Threshold | Business Meaning |
|--------|------------------|-------------------|------------------|
| IC | < 0.02 | < 0.01 | Signal predictive power declining |
| PSI | > 0.10 | > 0.25 | Feature distribution shifting |
| Sharpe | < 0.5 | < 0.0 | Risk-adjusted returns deteriorating |

### 5.2 DriftMetrics Data Structure

Drift metrics calculated and stored daily:

```python
from dataclasses import dataclass
from datetime import date

@dataclass
class DriftMetrics:
    """Daily drift metrics"""

    date: date
    strategy_id: str

    # IC metrics (Information Coefficient)
    ic: float | None = None           # Daily IC
    ic_5d_avg: float | None = None    # 5-day rolling average
    ic_20d_avg: float | None = None   # 20-day rolling average

    # PSI metrics (Distribution Stability)
    psi: float | None = None
    psi_5d_avg: float | None = None

    # Sharpe metrics (Risk-adjusted Returns)
    sharpe_5d: float | None = None    # 5-day Sharpe
    sharpe_20d: float | None = None   # 20-day Sharpe
    sharpe_60d: float | None = None   # 60-day Sharpe

    # Business metrics
    daily_return: float | None = None
    cumulative_return: float | None = None
    trade_count: int = 0
    signal_count: int = 0

    # Alert states
    ic_alert: bool = False
    psi_alert: bool = False
    sharpe_alert: bool = False
```

**Why Multiple Time Windows**:
- **5-day window**: Fast response, captures short-term drift
- **20-day window**: Filters noise, confirms trends
- **60-day window**: Long-term baseline, identifies structural changes

### 5.3 DriftMonitor Core Implementation

```python
import logging
import numpy as np
import psycopg
from psycopg.rows import dict_row

logger = logging.getLogger(__name__)

class DriftMonitor:
    """
    Production-grade drift monitoring service

    Responsibilities:
    1. Calculate IC, PSI, Sharpe metrics
    2. Compare against configured thresholds for alerts
    3. Persist to PostgreSQL
    4. Support per-strategy isolation
    """

    def __init__(self, dsn: str, strategy_id: str = "default"):
        """
        Args:
            dsn: PostgreSQL connection string
            strategy_id: Strategy identifier (supports multi-strategy isolation)
        """
        self.dsn = dsn
        self.strategy_id = strategy_id
        self._config: AlertConfig | None = None

    def load_config(self) -> AlertConfig:
        """Load alert configuration from database"""
        with psycopg.connect(self.dsn) as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """
                    SELECT ic_warning, ic_critical, psi_warning, psi_critical,
                           sharpe_warning, sharpe_critical
                    FROM drift_alert_config
                    WHERE strategy_id = %s
                    """,
                    (self.strategy_id,),
                )
                row = cur.fetchone()
                if row:
                    self._config = AlertConfig(**row)
                else:
                    self._config = AlertConfig()  # Use defaults
        return self._config

    def calculate_metrics(self, target_date: date) -> DriftMetrics:
        """
        Calculate all drift metrics for a given date

        Core logic:
        1. Get signals and returns, calculate IC
        2. Get historical returns, calculate rolling Sharpe
        3. Compare against thresholds for alert states
        """
        if self._config is None:
            self.load_config()

        metrics = DriftMetrics(date=target_date, strategy_id=self.strategy_id)

        # Calculate IC (signal-return correlation)
        signals, returns = self.get_signals_and_returns(target_date)
        if len(signals) > 0 and len(returns) > 0:
            metrics.ic = calculate_ic(signals, returns)
            metrics.signal_count = len(signals)

        # Calculate rolling Sharpe
        daily_returns = self.get_daily_returns(lookback_days=60)
        if len(daily_returns) >= 5:
            metrics.sharpe_5d = calculate_sharpe(daily_returns[-5:])
        if len(daily_returns) >= 20:
            metrics.sharpe_20d = calculate_sharpe(daily_returns[-20:])
        if len(daily_returns) >= 60:
            metrics.sharpe_60d = calculate_sharpe(daily_returns)

        # Determine alert states
        config = self._config or AlertConfig()
        if metrics.ic is not None:
            metrics.ic_alert = metrics.ic < config.ic_critical
        if metrics.psi is not None:
            metrics.psi_alert = metrics.psi > config.psi_critical
        if metrics.sharpe_20d is not None:
            metrics.sharpe_alert = metrics.sharpe_20d < config.sharpe_critical

        return metrics
```

### 5.4 PostgreSQL Persistence

Drift metrics need persistence for:
- Historical trend analysis
- Compliance auditing
- Retraining decision evidence

```python
def save_metrics(self, metrics: DriftMetrics) -> None:
    """Save metrics to database (supports idempotent upsert)"""
    with psycopg.connect(self.dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO drift_metrics (
                    date, strategy_id, ic, ic_5d_avg, ic_20d_avg,
                    psi, psi_5d_avg, sharpe_5d, sharpe_20d, sharpe_60d,
                    daily_return, cumulative_return, trade_count, signal_count,
                    ic_alert, psi_alert, sharpe_alert
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s
                )
                ON CONFLICT (date, strategy_id) DO UPDATE SET
                    ic = EXCLUDED.ic,
                    sharpe_20d = EXCLUDED.sharpe_20d,
                    ic_alert = EXCLUDED.ic_alert,
                    psi_alert = EXCLUDED.psi_alert,
                    sharpe_alert = EXCLUDED.sharpe_alert
                """,
                (
                    metrics.date, metrics.strategy_id, metrics.ic,
                    metrics.ic_5d_avg, metrics.ic_20d_avg, metrics.psi,
                    metrics.psi_5d_avg, metrics.sharpe_5d, metrics.sharpe_20d,
                    metrics.sharpe_60d, metrics.daily_return,
                    metrics.cumulative_return, metrics.trade_count,
                    metrics.signal_count, metrics.ic_alert,
                    metrics.psi_alert, metrics.sharpe_alert,
                ),
            )
        conn.commit()
    logger.info(f"Saved drift metrics for {metrics.date}")
```

**Database Schema**:

```sql
CREATE TABLE drift_metrics (
    date DATE NOT NULL,
    strategy_id VARCHAR(64) NOT NULL,
    ic FLOAT,
    ic_5d_avg FLOAT,
    ic_20d_avg FLOAT,
    psi FLOAT,
    psi_5d_avg FLOAT,
    sharpe_5d FLOAT,
    sharpe_20d FLOAT,
    sharpe_60d FLOAT,
    daily_return FLOAT,
    cumulative_return FLOAT,
    trade_count INT DEFAULT 0,
    signal_count INT DEFAULT 0,
    ic_alert BOOLEAN DEFAULT FALSE,
    psi_alert BOOLEAN DEFAULT FALSE,
    sharpe_alert BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (date, strategy_id)
);

CREATE TABLE drift_alert_config (
    strategy_id VARCHAR(64) PRIMARY KEY,
    ic_warning FLOAT DEFAULT 0.02,
    ic_critical FLOAT DEFAULT 0.01,
    psi_warning FLOAT DEFAULT 0.10,
    psi_critical FLOAT DEFAULT 0.25,
    sharpe_warning FLOAT DEFAULT 0.5,
    sharpe_critical FLOAT DEFAULT 0.0
);
```

### 5.5 Daily Monitoring Job

```python
def run_daily(self, target_date: date | None = None) -> DriftMetrics:
    """
    Daily drift monitoring job entry point

    Typical deployment: Run via cron or Airflow after market close
    """
    if target_date is None:
        target_date = date.today()

    logger.info(f"Running drift monitoring for {target_date}")
    metrics = self.calculate_metrics(target_date)
    self.save_metrics(metrics)

    # Alert logging
    if metrics.ic_alert:
        logger.warning(f"IC ALERT: IC={metrics.ic:.4f} below threshold")
    if metrics.psi_alert:
        logger.warning(f"PSI ALERT: PSI={metrics.psi:.4f} above threshold")
    if metrics.sharpe_alert:
        logger.warning(f"SHARPE ALERT: Sharpe={metrics.sharpe_20d:.4f} below threshold")

    return metrics
```

### 5.6 Integration Example: When to Trigger Retraining

Combining drift monitoring with retraining decisions:

```python
class RetrainOrchestrator:
    """Retraining orchestrator"""

    def __init__(self, drift_monitor: DriftMonitor):
        self.monitor = drift_monitor
        self.consecutive_alerts = 0
        self.alert_threshold = 3  # Trigger after 3 consecutive days

    def check_retrain_needed(self, target_date: date) -> dict:
        """
        Determine if retraining should be triggered

        Rules:
        1. IC < 0.01 for 3 consecutive days -> Trigger
        2. PSI > 0.25 single occurrence -> Trigger
        3. 20-day Sharpe < 0 -> Trigger
        """
        metrics = self.monitor.run_daily(target_date)

        # Track consecutive alerts
        if metrics.ic_alert or metrics.sharpe_alert:
            self.consecutive_alerts += 1
        else:
            self.consecutive_alerts = 0

        # Evaluate trigger conditions
        triggers = []

        if self.consecutive_alerts >= self.alert_threshold:
            triggers.append(f"IC/Sharpe alert for {self.consecutive_alerts} consecutive days")

        if metrics.psi_alert:
            triggers.append(f"PSI={metrics.psi:.3f} exceeds critical threshold")

        if metrics.sharpe_20d is not None and metrics.sharpe_20d < 0:
            triggers.append(f"20-day Sharpe={metrics.sharpe_20d:.2f} is negative")

        should_retrain = len(triggers) > 0

        return {
            'should_retrain': should_retrain,
            'triggers': triggers,
            'metrics': metrics,
            'action': 'RETRAIN' if should_retrain else 'CONTINUE'
        }

# Usage example
monitor = DriftMonitor(
    dsn="postgres://trading:trading@localhost:5432/trading",
    strategy_id="momentum_v2"
)
orchestrator = RetrainOrchestrator(monitor)

result = orchestrator.check_retrain_needed(date.today())
if result['should_retrain']:
    print(f"Triggering retrain, reasons: {result['triggers']}")
    # Call retraining pipeline
```

### 5.7 Architecture Summary

| Component | Responsibility | Key Design |
|-----------|----------------|------------|
| AlertConfig | Threshold configuration | Dataclass, supports DB loading |
| DriftMetrics | Metrics container | Multi-window, alert states |
| DriftMonitor | Core service | Calculate + Store + Alert |
| PostgreSQL | Persistence | Idempotent upsert, audit support |
| RetrainOrchestrator | Decision orchestration | Consecutive alerts, multi-condition triggers |

**Production Deployment Recommendations**:
1. Run T+30min after market close (wait for data readiness)
2. Connect alerts to Slack/PagerDuty
3. Dashboard showing IC/PSI/Sharpe trend charts
4. Retrain trigger automatically enters A/B testing flow

---

## 6. Summary

### Detection Methods Quick Reference

| Method | Detection Target | Sensitivity | Computational Cost | Recommended Scenario |
|--------|-----------------|-------------|-------------------|---------------------|
| Performance Monitoring | Strategy returns | Medium | Low | All strategies (essential) |
| K-S Test | Feature distribution | High | Medium | Periodic checks (weekly/monthly) |
| Chi-Square Test | Categorical features | High | Low | Market regime labels |
| CUSUM | Prediction errors | High | Low | Continuous monitoring (daily) |
| Comprehensive Detection | Multi-dimensional | Highest | Medium | Production systems (recommended) |

### Retraining Strategy Quick Reference

| Strategy | Trigger Type | Pros | Cons | Applicable Scenario |
|----------|-------------|------|------|---------------------|
| Scheduled | Time-driven | Simple, predictable | May lag behind | Stable markets |
| Triggered | Drift-driven | Timely response | Higher complexity | Volatile markets |
| Online Learning | Continuous update | Fastest adaptation | Unstable | High-frequency scenarios |
| Hybrid | Scheduled + Triggered | Balanced | Requires tuning | Production (recommended) |

---

> **Core Insight**: Model drift is not a question of "if" but "when." Establishing robust detection and retraining mechanisms is key to the long-term survival of quantitative strategies.
