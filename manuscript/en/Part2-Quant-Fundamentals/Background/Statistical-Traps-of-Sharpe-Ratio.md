# Background: Statistical Traps of Sharpe Ratio

> "Your Sharpe 2.0 strategy might just be statistical noise."

---

## 1. Estimation Error of Sharpe Ratio

### 1.1 Why Sample Sharpe Is Unreliable

The definition of Sharpe ratio is simple:

```
SR = (μ - rf) / σ

Where:
  μ  = Mean return
  rf = Risk-free rate
  σ  = Standard deviation of returns
```

**Problem**: Both μ and σ are estimated from finite samples, and both have estimation error.

### 1.2 Standard Error of Sharpe Ratio

Lo (2002) derived the approximate standard error of the Sharpe ratio:

```
SE(SR) ≈ √[(1 + SR²/2) / N]

Where:
  N = Number of observations (e.g., trading days)
```

> **Note**: This formula assumes IID (independent and identically distributed) returns.
> If returns exhibit autocorrelation (e.g., momentum effects, mean reversion), the
> standard error needs adjustment. Lo (2002) also provides formulas for the
> autocorrelation-adjusted standard error, which can be significantly larger than
> the IID version for strategies with persistent return patterns.

**Paper exercise**:

| True Sharpe | Observations N | Standard Error SE | 95% Confidence Interval |
|-------------|----------------|-------------------|-------------------------|
| 1.0 | 252 (1 year) | 0.077 | [0.85, 1.15] |
| 1.0 | 756 (3 years) | 0.045 | [0.91, 1.09] |
| 2.0 | 252 (1 year) | 0.109 | [1.79, 2.21] |
| 2.0 | 756 (3 years) | 0.063 | [1.88, 2.12] |

**Key findings**:
- 1 year of data has Sharpe estimation error of about ±0.15 (95% confidence)
- Even if true Sharpe is 0, the probability of sample Sharpe exceeding 0.06 is about 16% (exceeding 0.12 is only ~3%)

### 1.3 Code Implementation

```python
import numpy as np
from scipy import stats

def sharpe_ratio(returns: np.ndarray,
                 rf: float = 0.0,
                 periods_per_year: int = 252) -> float:
    """Calculate annualized Sharpe ratio"""
    excess_returns = returns - rf / periods_per_year
    return np.sqrt(periods_per_year) * excess_returns.mean() / excess_returns.std()


def sharpe_standard_error(sr: float, n: int) -> float:
    """Calculate standard error of Sharpe ratio"""
    return np.sqrt((1 + sr**2 / 2) / n)


def sharpe_confidence_interval(sr: float, n: int,
                                confidence: float = 0.95) -> tuple:
    """Calculate confidence interval for Sharpe ratio"""
    se = sharpe_standard_error(sr, n)
    z = stats.norm.ppf((1 + confidence) / 2)
    return (sr - z * se, sr + z * se)


def sharpe_p_value(sr: float, n: int) -> float:
    """
    Test whether Sharpe ratio is significantly greater than 0
    H0: True Sharpe = 0
    """
    se = sharpe_standard_error(0, n)  # Standard error under H0
    z = sr / se
    return 1 - stats.norm.cdf(z)
```

**Usage example**:

```python
# Assume you have 1 year of backtest data, sample Sharpe = 1.5
sr = 1.5
n = 252

se = sharpe_standard_error(sr, n)
ci = sharpe_confidence_interval(sr, n)
p = sharpe_p_value(sr, n)

print(f"Sample Sharpe: {sr:.2f}")
print(f"Standard Error: {se:.3f}")
print(f"95% Confidence Interval: [{ci[0]:.2f}, {ci[1]:.2f}]")
print(f"p-value: {p:.4f}")

# Output:
# Sample Sharpe: 1.50
# Standard Error: 0.081
# 95% Confidence Interval: [1.34, 1.66]
# p-value: 0.0000
```

---

## 2. Multiple Testing Problem

### 2.1 The Curse of Strategy Mining

**Scenario**: You tested 100 strategies and selected the one with the highest Sharpe.

```
Problem:
  - Each strategy is independent, all have true Sharpe of 0
  - After testing 100, expected maximum sample Sharpe ≈ 0.19 (pure noise!)

This is the statistical trap of "strategy mining"
```

### 2.2 Mathematical Principle

If testing K independent strategies, the expected value of maximum Sharpe:

```
E[max(SR₁, SR₂, ..., SRₖ)] ≈ SE × √(2 × ln(K))

Where SE is the standard error of a single Sharpe
```

**Paper exercise**:

| Strategies Tested K | Expected Max Sharpe when SE = 0.063 |
|---------------------|-------------------------------------|
| 10 | 0.063 × √(2×ln(10)) = 0.135 |
| 100 | 0.063 × √(2×ln(100)) = 0.191 |
| 1,000 | 0.063 × √(2×ln(1000)) = 0.234 |
| 10,000 | 0.063 × √(2×ln(10000)) = 0.270 |

**Conclusion**: The more strategies you test, the higher the sample Sharpe of the "best strategy"—even if they're all noise.

---

## 3. Deflated Sharpe Ratio

### 3.1 What Is Deflated Sharpe

A correction method proposed by Bailey & Lopez de Prado (2014), considering:

1. **Multiple testing**: How many strategies were tested
2. **Data length**: Sample size
3. **Skewness and kurtosis**: Non-normality of return distribution

### 3.2 Calculation Formula

```
DSR = Φ[(SR - SR* × √(1 + (γ₃/6)×SR + ((γ₄-3)/24)×SR²)) /
        √(1/N + (γ₃²/6)/N + ((γ₄-3)²/24)/N)]

Where:
  Φ = Standard normal CDF
  SR = Sample Sharpe
  SR* = Benchmark Sharpe (accounting for multiple testing)
  γ₃ = Skewness of returns
  γ₄ = Kurtosis of returns
  N = Sample size
```

**SR* calculation** (expected maximum Sharpe):

```
SR* = √(2 × ln(K)) × √(1/N) × (1 - γ × √(2 × ln(K)) + ...)

Simplified version:
SR* ≈ √(Var[SR]) × [(1-γ)×Z_K + γ×Z_K×exp(-Z_K²/2)]

Where Z_K = Φ⁻¹(1 - 1/K)
```

### 3.3 Full Implementation

```python
import numpy as np
from scipy import stats

def deflated_sharpe_ratio(
    returns: np.ndarray,
    n_trials: int,
    rf: float = 0.0,
    periods_per_year: int = 252
) -> dict:
    """
    Calculate Deflated Sharpe Ratio

    Parameters:
    -----------
    returns : Return series
    n_trials : Number of strategies/parameter combinations tested
    rf : Risk-free rate
    periods_per_year : Annualization factor

    Returns:
    --------
    dict : Contains original Sharpe, DSR, p-value, etc.
    """
    n = len(returns)
    excess = returns - rf / periods_per_year

    # Basic statistics
    mu = excess.mean()
    sigma = excess.std(ddof=1)
    skew = stats.skew(excess)
    kurt = stats.kurtosis(excess)  # Excess kurtosis

    # Annualized Sharpe
    sr = np.sqrt(periods_per_year) * mu / sigma

    # Expected maximum Sharpe (based on multiple testing)
    if n_trials > 1:
        z = stats.norm.ppf(1 - 1 / n_trials)
        euler_gamma = 0.5772156649
        sr_star = np.sqrt(1/n) * (
            (1 - euler_gamma) * z +
            euler_gamma * z * np.exp(-z**2 / 2)
        ) * np.sqrt(periods_per_year)
    else:
        sr_star = 0

    # Standard error of Sharpe ratio (accounting for non-normality)
    # Note: scipy.stats.kurtosis() returns EXCESS kurtosis (already subtracts 3),
    # so we use kurt/24 directly, not (kurt-3)/24
    sr_var = (1 + (skew/6)*sr + (kurt/24)*sr**2) / n
    sr_std = np.sqrt(sr_var) * np.sqrt(periods_per_year)

    # Deflated Sharpe test statistic
    if sr_std > 0:
        z_stat = (sr - sr_star) / sr_std
        p_value = 1 - stats.norm.cdf(z_stat)
    else:
        z_stat = np.nan
        p_value = 1.0

    # DSR = Probability that strategy Sharpe significantly exceeds benchmark
    dsr = 1 - p_value

    return {
        'sharpe_ratio': sr,
        'expected_max_sr': sr_star,
        'deflated_sr': dsr,
        'p_value': p_value,
        'z_statistic': z_stat,
        'n_observations': n,
        'n_trials': n_trials,
        'skewness': skew,
        'kurtosis': kurt,
        'is_significant': p_value < 0.05
    }
```

### 3.4 Usage Example

```python
# Simulate strategy returns
np.random.seed(42)
daily_returns = np.random.normal(0.0005, 0.01, 252)  # 1 year of data

# Assume we tested 50 parameter combinations
result = deflated_sharpe_ratio(daily_returns, n_trials=50)

print(f"Original Sharpe: {result['sharpe_ratio']:.2f}")
print(f"Expected Max Sharpe (50 trials): {result['expected_max_sr']:.2f}")
print(f"Deflated SR: {result['deflated_sr']:.2%}")
print(f"p-value: {result['p_value']:.4f}")
print(f"Is Significant: {result['is_significant']}")
```

---

## 4. Practical Guidelines

### 4.1 When to Worry

| Situation | Risk Level | Recommendation |
|-----------|------------|----------------|
| Sample Sharpe < 1.0, tested < 10 strategies | Low | Possibly valid, continue verification |
| Sample Sharpe 1.0-2.0, tested 10-100 strategies | Medium | Calculate DSR, beware of overfitting |
| Sample Sharpe > 2.0, tested > 100 strategies | High | Almost certainly overfitting |
| Sample Sharpe > 3.0 | Very High | Check for data errors or look-ahead bias |

### 4.2 How to Report Sharpe Ratio

**Wrong approach**:
> "My strategy has Sharpe 2.5"

**Correct approach**:
> "Based on 3 years of daily data (756 observations), sample Sharpe is 2.5,
> 95% confidence interval [2.3, 2.7], after testing 30 parameter combinations,
> Deflated SR = 0.92, p-value = 0.04"

### 4.3 Methods to Mitigate Multiple Testing

```
1. Pre-determine the number of strategies to test
   - Write down the list of strategies to test
   - Don't add strategies as you go

2. Out-of-sample validation
   - Reserve 20-30% of data not used in development
   - Only test the final strategy once on this data

3. Bonferroni correction
   - Significance level = 0.05 / K
   - K strategies require p < 0.05/K to be significant

4. Record all tests
   - Record even failed strategies
   - Use for calculating true n_trials
```

---

## 5. Common Misconceptions

**Misconception 1: Higher Sharpe is always better**

Extremely high Sharpe (>3) usually means:
- Data errors (duplicate calculations, look-ahead information)
- Overfitting
- Strategy capacity too small

**Misconception 2: 3 years of data is enough to accurately estimate Sharpe**

3 years of data (756 days) has standard error of about 0.04-0.05, meaning:
- A strategy with true Sharpe 1.0 may show sample Sharpe between 0.9-1.1
- Distinguishing between Sharpe 1.0 and 1.2 strategies is nearly impossible

**Misconception 3: Backtest Sharpe = Live Sharpe**

Backtests typically overestimate:
- Ignore slippage and market impact
- Subtle look-ahead bias
- Survivorship bias

**Rule of thumb**: Live Sharpe ≈ Backtest Sharpe × 0.5-0.7

---

## 6. Multi-Agent Perspective

In multi-agent architecture, the statistical issues of Sharpe ratio require special attention:

![Multi-Agent System Sharpe Evaluation](../assets/multi-agent-sharpe-evaluation.svg)

---

## 7. Summary

| Key Point | Description |
|-----------|-------------|
| Estimation error | 1 year of data has Sharpe standard error ≈ 0.07-0.08 |
| Multiple testing | Testing 100 strategies, expected max Sharpe ≈ 0.19 (pure noise) |
| Deflated SR | Significance test accounting for multiple testing |
| Practical advice | Report confidence intervals, calculate DSR, maintain skepticism |

---

## Further Reading

- [Lesson 03: Math and Statistics Fundamentals](../Lesson-03-Math-and-Statistics-Fundamentals.md) - Statistics basics
- [Lesson 07: Backtest System Pitfalls](../Lesson-07-Backtest-System-Pitfalls.md) - Other biases in backtesting
- Bailey, D. H., & Lopez de Prado, M. (2014). *The Deflated Sharpe Ratio*
- Lo, A. W. (2002). *The Statistics of Sharpe Ratios*
