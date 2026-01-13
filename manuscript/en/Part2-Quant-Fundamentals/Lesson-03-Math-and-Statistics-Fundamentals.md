# Lesson 03: Math and Statistics Fundamentals

> **The essence of quantitative trading is describing markets in mathematical language. If your mathematical assumptions are wrong, the more precise your model, the faster you lose money.**

---

## LTCM: The Nobel Laureates' Fatal Assumption

In 1998, Long-Term Capital Management (LTCM) had two Nobel Prize winners in Economics, Wall Street's top mathematicians, and $125 billion in assets.

Their models were exquisite, based on one core assumption: **market returns follow a normal distribution**.

According to the normal distribution, the probability of their strategy losing more than 1% in a single day was 1 in 100,000. With 252 trading days per year, theoretically this would happen once every 400 years.

**Then Russia defaulted on its debt.**

On August 21, 1998, LTCM lost $550 million in a single day. Over the next month, they lost $4.6 billion - nearly all their capital. According to the normal distribution, the probability of this event was 10^-27, roughly equivalent to something that wouldn't happen once per second since the birth of the universe.

But it happened.

**What went wrong?**

1. **Markets don't follow normal distributions**: Real markets have "fat-tailed distributions" - extreme events occur far more frequently than normal distribution predicts
2. **Correlations can spike suddenly**: During normal times, asset correlations are low; during crises, all assets spike to correlation of 1
3. **Volatility clusters**: After big drops, more big drops often follow - it's not independent "coin flipping"

LTCM's lesson: **The model wasn't imprecise - the mathematical assumptions were fundamentally wrong.**

This lesson's goal: Help you understand the true characteristics of financial data, and avoid using "coin flip" math to analyze markets.

---

## 3.1 Time Series Fundamentals

### Sequence vs IID Assumption

Traditional statistics assumes data is **IID (Independent and Identically Distributed)**:
- Independent: Today's data is unrelated to yesterday's
- Identically Distributed: Each day's data comes from the same distribution

**Financial data almost never satisfies the IID assumption**:

```python
import numpy as np

# IID data: coin flips
coin_flips = np.random.choice([0, 1], size=100)  # Each flip independent, probability constant

# Financial data: today's price depends on yesterday
prices = [100]
for _ in range(99):
    # Today's price = yesterday's price + random change
    prices.append(prices[-1] * (1 + np.random.normal(0, 0.02)))
```

**Why does this matter?**
- If you model with IID assumptions, you'll underestimate the probability of consecutive losses
- Real markets have "momentum": rises tend to continue rising, falls tend to continue falling

### Lag and Autocorrelation

**Autocorrelation**: The correlation between current values and past values.

**Paper Calculation Example**:

Assume 5 consecutive days of returns: +2%, -1%, +3%, +2%, -2%

| Today (t) | Yesterday (t-1) |
|-----------|-----------------|
| -1% | +2% |
| +3% | -1% |
| +2% | +3% |
| -2% | +2% |

Observed patterns:
- Yesterday up, today sometimes up sometimes down -> Autocorrelation near 0 (random)
- If yesterday up usually means today up -> Positive autocorrelation (momentum)
- If yesterday up usually means today down -> Negative autocorrelation (mean reversion)

**SPY (S&P 500 ETF) daily return autocorrelation is typically near zero** (e.g., ~0.02 for 2010-2020), indicating US stock daily returns are close to random walk, difficult to predict short-term. Exact values vary by sample period and methodology.

**Interpretation**:
- Autocorrelation near 0 -> Past has low predictive value for future (weak-form efficient market)
- Significantly positive autocorrelation -> Momentum effect (trend following may work)
- Significantly negative autocorrelation -> Mean reversion effect

<details>
<summary>Code Implementation (for engineers)</summary>

```python
import pandas as pd

def calculate_autocorrelation(series, lag=1):
    """Calculate autocorrelation coefficient at lag periods"""
    return series.autocorr(lag=lag)

# Example: SPY (S&P 500 ETF) daily return autocorrelation
returns = prices_series.pct_change().dropna()

print(f"Lag-1 autocorrelation: {calculate_autocorrelation(returns, 1):.3f}")
print(f"Lag-5 autocorrelation: {calculate_autocorrelation(returns, 5):.3f}")
```

</details>

### Stationarity Testing

**Stationarity**: Statistical properties (mean, variance) don't change over time.

```python
from statsmodels.tsa.stattools import adfuller

def check_stationarity(series, name="Series"):
    """ADF test: Determine if series is stationary"""
    result = adfuller(series.dropna())
    print(f"{name}:")
    print(f"  ADF Statistic: {result[0]:.4f}")
    print(f"  p-value: {result[1]:.4f}")
    print(f"  Conclusion: {'Stationary' if result[1] < 0.05 else 'Non-stationary'}")

# Price series are typically non-stationary
check_stationarity(prices_series, "Price Series")

# Return series are typically stationary
check_stationarity(returns_series, "Returns Series")
```

**Why test for stationarity?**
- Most statistical models assume stationary input
- Modeling non-stationary series produces "spurious regression"
- **Solution**: Use returns (differences) instead of prices

---

## 3.2 Mathematical Definition of Returns

### Simple Returns vs Log Returns

**Paper Calculation**:

Assume AAPL goes from $100 to $110:

```
Simple return = (110 - 100) / 100 = 10%
Log return = ln(110/100) = ln(1.1) ~ 9.53%
```

**Why do quants often use log returns?** See this example:

**Additivity Problem** (verify with calculator):

| | Day 1 | Day 2 | Total Return |
|--|-------|-------|--------------|
| Price | $100 -> $110 | $110 -> $100 | $100 -> $100 |
| Simple Return | +10% | -9.09% | **0%** |
| Direct Sum | | | 10% - 9.09% = 0.91% X |
| Log Return | +9.53% | -9.53% | **0%** |
| Direct Sum | | | 9.53% - 9.53% = 0% ✓ |

**Conclusion**: Log returns can be directly summed; simple returns must be multiplied. When calculating 100-day cumulative returns, log returns need only 100 additions, while simple returns need 100 multiplications.

| Property | Simple Return | Log Return |
|----------|---------------|------------|
| Additivity | X Multi-period returns can't be directly summed | O Multi-period returns can be directly summed |
| Symmetry | X Up 10% then down 10% != 0 | O More symmetric |
| Normality Assumption | Harder to satisfy | Closer to normal |

### Cumulative Return Calculation

**Paper Calculation**:

5 consecutive days of daily returns: +1%, -2%, +3%, +1%, -1%

**Simple return cumulative** (multiply):
```
(1+0.01) x (1-0.02) x (1+0.03) x (1+0.01) x (1-0.01) - 1
= 1.01 x 0.98 x 1.03 x 1.01 x 0.99 - 1
= 1.0194 - 1 = 1.94%
```

**Log return cumulative** (direct sum):
```
0.01 + (-0.02) + 0.03 + 0.01 + (-0.01) = 2%
```

Both methods give similar results, but log returns are simpler to calculate.

### Annualized Returns

**Paper Calculation**:

Assume you earned 5% over 60 trading days, what's the annualized return?

```
Annualized Return = (1 + 5%)^(252/60) - 1
                  = 1.05^4.2 - 1
                  = 1.227 - 1
                  = 22.7%
```

**Why 252?** US stocks have approximately 252 trading days per year (365 days - weekends - holidays).

**Common Mistakes**:
- X Directly multiplying monthly return by 12 as annualized (ignoring compounding)
- X Using calendar days instead of trading days (should use 252, not 365)
- X Annualizing before subtracting fees

<details>
<summary>Code Implementation (for engineers)</summary>

```python
def cumulative_return(returns, method='simple'):
    """Calculate cumulative return"""
    if method == 'simple':
        return (1 + returns).prod() - 1  # Multiply
    else:
        return returns.sum()  # Direct sum

def annualize_return(total_return, days, trading_days_per_year=252):
    """Annualized return"""
    years = days / trading_days_per_year
    return (1 + total_return) ** (1 / years) - 1

# Example: Earned 5% over 60 days
ann_return = annualize_return(0.05, 60)
print(f"Annualized return: {ann_return:.2%}")  # ~ 22.7%
```

</details>

---

## 3.3 Mathematical Definition of Risk

### Variance and Standard Deviation

**Paper Calculation**:

Assume 5 days of daily returns: +1%, -2%, +3%, 0%, -1%

**Step 1: Calculate mean**
```
Mean = (1 - 2 + 3 + 0 - 1) / 5 = 0.2%
```

**Step 2: Calculate variance** (average of squared deviations from mean)
```
Variance = [(1-0.2)^2 + (-2-0.2)^2 + (3-0.2)^2 + (0-0.2)^2 + (-1-0.2)^2] / 5
         = [0.64 + 4.84 + 7.84 + 0.04 + 1.44] / 5
         = 14.8 / 5 = 2.96 (%^2)
```

**Step 3: Calculate standard deviation** (volatility)
```
Daily volatility = sqrt(2.96) ~ 1.72%
```

**Step 4: Annualize**
```
Annualized volatility = Daily volatility x sqrt(252) ~ 1.72% x 15.87 ~ 27.3%
```

**Intuitive meaning of volatility**:

| Annualized Volatility | Typical Asset | Possible 1-Year Range (68% probability) |
|----------------------|---------------|----------------------------------------|
| 15% | Large-cap stocks (SPY) | -15% to +15% |
| 30% | Tech stocks (TSLA) | -30% to +30% |
| 80% | Cryptocurrency (BTC) | -80% to +80% |

**Remember this formula**: Annualized volatility = Daily volatility x sqrt(252) ~ Daily volatility x 16

<details>
<summary>Code Implementation (for engineers)</summary>

```python
def calculate_volatility(returns, annualize=True, trading_days=252):
    """Calculate volatility (standard deviation)"""
    daily_vol = returns.std()
    if annualize:
        return daily_vol * np.sqrt(trading_days)
    return daily_vol

# Example
daily_returns = pd.Series([0.01, -0.02, 0.03, 0, -0.01])
annual_vol = calculate_volatility(daily_returns)
print(f"Annualized volatility: {annual_vol:.2%}")
```

</details>

### Covariance and Correlation

**Intuitive Understanding**:

Correlation coefficient measures whether two assets "move together":

| Correlation | Meaning | Example |
|-------------|---------|---------|
| +1 | Perfectly synchronized: A up means B up | Same-sector stocks (AAPL vs MSFT) |
| 0 | Unrelated: A's movement unrelated to B | Gold vs tech stocks |
| -1 | Perfectly opposite: A up means B down | Stocks vs VIX (fear index) |

**Actual market correlations** (reference values):

| Asset Pair | Normal Period Correlation | Crisis Period Correlation |
|------------|---------------------------|---------------------------|
| AAPL vs MSFT | +0.7 | +0.9 |
| SPY vs TLT (bonds) | -0.3 | -0.5 or +0.8 |
| SPY vs GLD (gold) | +0.1 | -0.2 |

**"Correlation spike" during crises**: Assets uncorrelated during normal times can suddenly spike to 0.9 correlation during crises. This is what tripped up LTCM - they assumed stable correlations.

**Multi-Agent Perspective**: Portfolio Agent's core job is finding low-correlation assets to build portfolios, but Risk Agent must consider the risk of "correlation spikes during crises."

<details>
<summary>Code Implementation (for engineers)</summary>

```python
def analyze_correlation(returns_a, returns_b):
    """Analyze correlation between two assets"""
    corr = returns_a.corr(returns_b)
    print(f"Correlation coefficient: {corr:.3f}")

    if corr > 0.7:
        print("-> Highly positive correlated: Move together, poor diversification")
    elif corr < -0.3:
        print("-> Negatively correlated: Natural hedge, good for portfolio")
    else:
        print("-> Low correlation: Valuable for risk diversification")
```

</details>

### The Danger of Distribution Assumptions

**The normal distribution assumption will kill you.** This is the core lesson from LTCM.

**Paper Calculation**:

Assume SPY has 20% annualized volatility, daily volatility is about 1.26% (= 20% / sqrt(252)).

Under normal distribution:
- Probability of daily drop > 1 sigma (1.26%) = 16% (normal)
- Probability of daily drop > 2 sigma (2.52%) = 2.3% (0.5 times per month)
- Probability of daily drop > 3 sigma (3.78%) = 0.13% (once every 3 years)
- Probability of daily drop > 4 sigma (5.04%) = 0.003% (once every 125 years)

**But what actually happened?**

| Date | Event | SPY Single-Day Drop | "How many sigma" per normal distribution | Theoretical frequency |
|------|-------|---------------------|----------------------------------------|---------------------|
| 2020-03-16 | COVID crash | -12.0% | 9.5 sigma | 10^20 years |
| 2008-10-15 | Financial crisis | -9.0% | 7.1 sigma | 10^12 years |
| 2011-08-08 | US downgrade | -6.7% | 5.3 sigma | 15 million years |
| 2018-02-05 | Volatility crash | -4.1% | 3.3 sigma | 4 years |

**Conclusion**: Events the normal distribution says "once in 10^20 years" actually happen every few years.

**This is the power of "fat-tailed distributions"** - extreme events occur far more frequently than normal distribution predicts. If your risk model assumes normal distribution, you will inevitably blow up during some "black swan" event.

---

## 3.4 Special Characteristics of Financial Time Series

### Fat Tails

**Two key metrics**:

1. **Skewness**: Is the distribution symmetric?
   - Skewness = 0 -> Symmetric (similar magnitude ups and downs)
   - Skewness < 0 -> Left-skewed (more crashes than rallies)
   - **Stock market skewness is typically -0.5 to -1**, indicating crash risk exceeds rally opportunity

2. **Kurtosis**: How "fat" are the tails?
   - Kurtosis = 3 -> Normal distribution
   - Kurtosis > 3 -> Fat tails (more extreme events than normal distribution predicts)
   - **Stock market kurtosis is typically 5-10**, far exceeding normal distribution's 3

| Metric | Normal Distribution | SPY Actual | Meaning |
|--------|---------------------|------------|---------|
| Skewness | 0 | -0.7 | Crashes are more severe |
| Kurtosis | 3 | 8 | Frequent extreme events |

**Impact of fat tails on strategies**:
- Stop-losses must consider gap risk (overnight crashes may jump past your stop price)
- VaR models will severely underestimate risk
- Need to use Expected Shortfall (CVaR) instead of VaR

<details>
<summary>Code Implementation (for engineers)</summary>

```python
def analyze_tail_risk(returns):
    """Analyze tail risk"""
    skew = returns.skew()  # Skewness
    kurt = returns.kurtosis()  # Kurtosis (excess)

    print(f"Skewness: {skew:.3f} {'(negative skew, beware of crashes)' if skew < -0.5 else ''}")
    print(f"Kurtosis: {kurt+3:.3f} {'(fat tails!)' if kurt > 0 else ''}")
```

</details>

### Volatility Clustering

**GARCH Effect**: Large volatility tends to be followed by large volatility, small volatility tends to be followed by small volatility.

**Intuitive Understanding**:

Recall the COVID crash in March 2020:
- March 9: SPY down 7.6%
- March 12: SPY down 9.5%
- March 16: SPY down 12.0%

Crashes are not independent "coin flips" - once crashing begins, it often continues. This is volatility clustering.

**Volatility autocorrelation is as high as 0.7-0.9**, far higher than return autocorrelation (about 0). This means:
- Returns are nearly unpredictable (random walk)
- **Volatility is predictable** (tomorrow's volatility is likely similar to today's)

**Practical Implications**:
- Volatility predictability > Return predictability
- Trend Agent should build positions during low volatility, reduce during high volatility
- Risk Agent should dynamically adjust stop-losses based on recent volatility

<details>
<summary>Code Implementation (for engineers)</summary>

```python
def detect_volatility_clustering(returns, window=20):
    """Detect volatility clustering"""
    rolling_vol = returns.rolling(window).std()
    vol_autocorr = rolling_vol.autocorr(lag=1)
    print(f"Volatility lag-1 autocorrelation: {vol_autocorr:.3f}")
```

</details>

### Non-Stationarity and Regime Shifts

Markets switch between different "regimes," each with different statistical properties:

| Regime | Characteristics | Suitable Strategy |
|--------|-----------------|-------------------|
| **Bull Market** | Low volatility, positive returns, low correlation | Momentum, buy and hold |
| **Sideways** | Medium volatility, returns near 0 | Mean reversion |
| **Crisis** | High volatility, negative returns, high correlation | Reduce positions, hedge |

**Typical structural break cases (approximate values)**:

> Note: The table below shows typical magnitudes; specific values vary with methodology, window, and data source.

| Time | Event | Volatility Change | Correlation Change |
|------|-------|-------------------|-------------------|
| 2008.09 | Lehman collapse | 15% -> 80% | 0.3 -> 0.95 |
| 2020.03 | COVID outbreak | 12% -> 85% | 0.4 -> 0.90 |
| 2022.01 | Fed rate hikes | 15% -> 30% | 0.5 -> 0.70 |

**Impact of Regime Shifts on strategies**:
- Models trained on old regimes will fail in new regimes
- Correlation spikes mean diversification suddenly fails
- Volatility spikes mean stop-loss positions need recalculation

**This is why multi-agent is needed** - different regimes need different strategies; one model can't handle it all. Meta Agent's core task is identifying the current regime and dispatching to the appropriate specialist Agent.

<details>
<summary>Code Implementation (for engineers)</summary>

```python
def detect_regime_change(returns, window=60):
    """Simple regime change detection"""
    rolling_vol = returns.rolling(window).std()
    vol_change = rolling_vol.diff().abs()
    threshold = vol_change.quantile(0.95)
    regime_changes = vol_change > threshold
    print(f"Detected {regime_changes.sum()} potential regime change points")
    return regime_changes
```

</details>

---

## 3.5 Common Misconceptions

**Misconception 1: Normal distribution adequately describes markets**

Wrong. Real markets have fat-tailed distributions; extreme events occur far more frequently than normal distribution predicts. LTCM assumed normal distribution, and "once in a million years" events happened within weeks.

**Misconception 2: Lower volatility is always safer**

Not entirely. Low volatility can be the calm before the storm. More dangerously, low volatility periods often coincide with increased leverage; once volatility suddenly spikes, losses are amplified.

**Misconception 3: Correlations are stable and predictable**

Dangerous assumption. Assets with 0.3 correlation during normal times can spike to 0.9 during crises. Diversification may fail exactly when you need it most.

**Misconception 4: Historical data can predict the future**

Only partially. Markets experience Regime Shifts (structural breaks), and past statistical patterns can suddenly become invalid. Markets in 2008 and 2020 were completely different from before.

---

## 3.6 Multi-Agent Perspective

Different Agents can use different statistical assumptions:

| Agent | Statistical Assumption | Applicable Scenario |
|-------|----------------------|---------------------|
| **Trend Agent** | Momentum exists (positive autocorrelation) | Trending market |
| **Mean Reversion Agent** | Mean reversion (negative autocorrelation) | Sideways market |
| **Risk Agent** | Fat-tailed distribution + volatility clustering | All scenarios |
| **Regime Agent** | Non-stationary + structural breaks | Regime identification |

**Key Insight**:
- Don't let all Agents use the same statistical assumptions
- Risk Agent must use the most conservative assumptions (fat tails, frequent extreme events)
- Meta Agent's responsibility is identifying the current regime and dispatching to the appropriate Agent

---

## Lesson Deliverables

After completing this lesson, you will have:

1. **Correct understanding of financial data characteristics** - Know why "coin flip" models don't work in markets
2. **Return calculation ability** - Master correct calculation of log returns and annualized returns
3. **Intuition for risk measurement** - Understand practical meanings of volatility, correlation, fat-tailed distributions
4. **Vigilance about statistical assumptions** - Know how normal distribution assumptions lead to fatal errors

### Verification Checklist

| Checkpoint | Verification Standard | Self-Test Method |
|------------|----------------------|------------------|
| **Return calculation** | Can hand-calculate log returns and annualized returns | Given $100->$110->$99, calculate 2-day log returns and cumulative |
| **Volatility calculation** | Can hand-calculate standard deviation and annualize | Given 5 daily returns, calculate annualized volatility |
| **Fat tail understanding** | Can explain why normal distribution assumption is dangerous | Without notes, explain LTCM's statistical failure reason |
| **Regime awareness** | Can list 3 market regimes and corresponding strategies | Draw regime switching diagram, label each regime's characteristics |

**Comprehensive Exercise** (complete with calculator):

An intraday strategy:
- 10-day returns: +1%, -0.5%, +2%, -1%, +0.5%, -2%, +1.5%, 0%, -0.5%, +1%
- Calculate: (1) Cumulative return (2) Daily volatility (3) Annualized volatility (4) Is this a high or low volatility strategy?

<details>
<summary>Click to reveal answer</summary>

1. Cumulative return (simple method) ~ 2.0%
2. Daily volatility ~ 1.2%
3. Annualized volatility = 1.2% x sqrt(252) ~ 19%
4. Annualized volatility of 19% is medium volatility, close to SPY's volatility level

</details>

---

## Key Takeaways

- [x] Understand financial data doesn't satisfy IID assumption; autocorrelation and non-stationarity exist
- [x] Master the difference between log returns vs simple returns, and annualization methods
- [x] Recognize dangers of normal distribution assumption: fat tails, volatility clustering, extreme events
- [x] Understand impact of Regime Shifts on strategies, and how multi-agent systems respond

---

## Extended Reading

- [Background: Alpha and Beta](../Part1-Quick-Start/Background/Alpha-and-Beta.md) - Mathematical foundation of return decomposition
- [Background: Famous Quant Disasters](../Part1-Quick-Start/Background/Famous-Quant-Disasters.md) - The cost of ignoring fat tails
- [Background: Statistical Traps of Sharpe Ratio](Background/Statistical-Traps-of-Sharpe-Ratio.md) - Estimation error, multiple testing, and Deflated Sharpe

---

## Next Lesson Preview

**Lesson 04: The Real Role of Technical Indicators**

MACD, RSI, Bollinger Bands... Are these indicators actually useful? The answer: they're not "buy/sell signals," but **feature engineering**. Next lesson we reveal the true nature of technical indicators.
