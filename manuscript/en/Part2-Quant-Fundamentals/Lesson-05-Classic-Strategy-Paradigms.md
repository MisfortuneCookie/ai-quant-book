# Lesson 05: Classic Strategy Paradigms

> **Quantitative trading isn't about inventing new strategies - it's about using the right strategy in the right market.**

---

## The Fate of Two Traders

In March 2020, COVID-19 triggered a global stock market crash.

**Trader A** used a trend following strategy. At the beginning of the decline, his strategy went short promptly, profiting 40% in one month. Then when the market bottomed and rebounded, he went long with the trend, tripling his account by year-end.

**Trader B** used a mean reversion strategy. Seeing the big drop, he believed "what goes down must come up" and kept buying the dip. Result: the more he bought, the more it fell. He blew up his account the day before the bottom.

Same market, two strategies, completely opposite outcomes.

**In 2021, the situation reversed.**

The market entered a sideways period, and Trader A's trend strategy kept getting slapped: chasing highs and cutting lows, always buying at peaks and selling at troughs. He lost half of his 2020 profits.

Meanwhile Trader B (with new capital) saw his mean reversion strategy shine: selling high and buying low, steady profits. By year-end, his account grew 60%.

**What's the lesson?**

1. **No single strategy works for all markets**
2. **Trending markets need trend strategies; sideways markets need mean reversion strategies**
3. **Identifying market state is more important than choosing a strategy**

This is why we need multi-agent systems - let different specialist strategies handle different market states.

---

## 5.1 Trend Following

### Core Philosophy

> "The trend is your friend, until it isn't."

Trend following philosophy: **Markets have inertia - what goes up tends to keep going up, what goes down tends to keep going down.**

| Characteristic | Description |
|----------------|-------------|
| Profit model | Capture big trends, profit from "head" and "tail" |
| Win rate | Usually only 30-45% |
| Risk/reward | High, single win can be 3-10x the loss |
| Suitable market | Markets with clear trends (bull/bear) |
| Fatal scenario | Getting whipsawed in sideways markets |

### Dual Moving Average Strategy

The most classic trend following strategy:

```
Rules:
- Golden cross buy: Short-term MA crosses above long-term MA (e.g., MA5 > MA20)
- Death cross sell: Short-term MA crosses below long-term MA (e.g., MA5 < MA20)
```

**Intuitive understanding**:
- Short-term MA represents "current sentiment"
- Long-term MA represents "long-term trend"
- Short exceeds long = trend may be turning up

| Parameter Combo | Use Case | Characteristics |
|-----------------|----------|-----------------|
| MA5/MA20 | Short-term trading | Sensitive, many false signals |
| MA10/MA60 | Medium-term trading | Balanced |
| MA20/MA120 | Long-term investing | Stable, severe lag |

**Actual Performance**:

| Market State | Dual MA Performance |
|--------------|---------------------|
| Strong trend (bull/bear) | Profitable, captures most of the move |
| Weak trend | Small gains/losses |
| Sideways | Losses, frequent false signals |

**Optimization directions**:
- Add trend filter (only open when ADX > 25)
- Add volume confirmation (signals more reliable on high volume)
- Multi-timeframe confirmation (daily golden cross + weekly trend up)

### Intraday Trend Trading

Strategy suitable for full-time traders:

```
Rules:
1. Observe price direction for 30 minutes after open
2. If price breaks 0.5% above open, go long
3. Stop-loss at 0.5% below open price
4. Close before market close, no overnight risk
```

| Pros | Cons |
|------|------|
| No overnight risk | Requires constant monitoring |
| Clear signals | High transaction costs |
| Same-day settlement | May miss big trends |

### Risk Profile of Trend Following

![Trend Strategy Equity Curve](assets/trend-strategy-equity.svg)

**Key insights**:
- Trend strategies will have consecutive small losses during sideways periods
- Need mental preparation for 10-20 consecutive stop-losses
- Profit comes from a few big trend trades that recover all losses and more

---

## 5.2 Mean Reversion

### Core Philosophy

> "What goes up must come down, what goes down must come up."

Mean reversion philosophy: **Prices will eventually return to some "normal" level.**

| Characteristic | Description |
|----------------|-------------|
| Profit model | Buy low sell high, profit from "oscillation" |
| Win rate | Usually 55-70% |
| Risk/reward | Low, single win usually smaller than loss |
| Suitable market | Sideways, range-bound markets |
| Fatal scenario | Trend breakout, buying all the way down |

### Grid Trading Strategy

Set buy/sell points at equal intervals within a predefined price range:

![Grid Trading Structure](assets/grid-trading-structure.svg)

**Rules**:
- Price drops to 95 -> Buy 1 unit
- Price drops to 90 -> Buy another unit
- Price rises to 105 -> Sell 1 unit
- Price rises to 110 -> Sell another unit

| Parameter | Suggested Value | Explanation |
|-----------|----------------|-------------|
| Grid spacing | 3-5% | Too small and fees eat profits; too large and too few signals |
| Total grids | 5-10 | Too many spreads capital thin; too few leaves little room for error |
| Position per grid | Total capital / grid count | Ensure worst case doesn't blow up |

**The Truth About Grid Trading**:

| Market State | Grid Performance |
|--------------|------------------|
| Sideways | Stable profits, captures every oscillation |
| Uptrend | Small profit, but positions get sold out, miss further upside |
| Downtrend | **Major losses**, keeps buying as it falls, capital trapped |

**Important warning**: Grid trading's greatest fear is one-sided decline. Always set an overall stop-loss.

### Pairs Trading

Find two highly correlated assets and trade when their spread deviates from normal:

> **Important**: SPY (~$600) and IVV (~$550) have different absolute prices, so you cannot simply compare price ratios. Instead, use the **Z-score of return spread** with equal dollar amounts.

```
Example: SPY vs IVV (both track S&P 500)

Step 1: Calculate daily return spread
  - SPY daily return: +0.52%
  - IVV daily return: +0.48%
  - Return spread: +0.04%

Step 2: Calculate Z-score of the spread
  - Mean spread over 20 days: 0.00%
  - Std dev of spread: 0.02%
  - Today's Z-score: (0.04% - 0.00%) / 0.02% = +2.0

Step 3: Trade signal (Z-score > 2.0)
  - SPY "outperformed" temporarily
  - Short $10,000 SPY, Long $10,000 IVV
  - Wait for Z-score to return to 0, then close
```

**Why equal dollar amounts?**
- SPY price ~$600, IVV price ~$550
- To be dollar-neutral: Short 16.7 shares SPY ($10,000), Long 18.2 shares IVV ($10,000)
- This ensures equal exposure on both sides of the trade

**Why it works?**
- Both ETFs track the exact same index (S&P 500)
- Same macro factors affect them identically
- Return spread deviation is usually temporary and small
- By using returns (not prices), we normalize for different price levels

**Key concept: Cointegration**

| Concept | Definition | Example |
|---------|------------|---------|
| Correlation | Two series move together | Gold and Gold ETF |
| Cointegration | The difference between two series is stable | SPY and IVV |

Correlation doesn't mean cointegration. Cointegration is the foundation of pairs trading.

### Risk Profile of Mean Reversion

![Mean Reversion Equity Curve](assets/mean-reversion-equity.svg)

**Key insights**:
- Mean reversion is very stable during sideways periods
- But one trend breakout can wipe out months of profits
- Must have strict stop-loss, can't "hold and hope"

---

## 5.3 Multi-Strategy Portfolios

### Why Combine?

| Single Strategy Problem | Portfolio Solution |
|------------------------|-------------------|
| Trend strategy loses in sideways | Mean reversion compensates |
| Mean reversion loses in trending | Trend strategy compensates |
| Single strategy concentrates risk | Multi-strategy diversifies risk |

### Strategy Correlation

The key to combining is **low correlation**:

| Strategy A | Strategy B | Correlation | Portfolio Effect |
|------------|------------|-------------|------------------|
| Trend following | Trend following | High | No diversification |
| Trend following | Mean reversion | Low/Negative | Good diversification |
| Stock long | Bonds | Negative | Excellent diversification |

### Capital Allocation Methods

**Method 1: Equal Weight**
```
Strategy A: 33%
Strategy B: 33%
Strategy C: 34%
```
Simple, but doesn't consider strategy risk differences.

**Method 2: Risk Parity**
```
Allocate inversely proportional to volatility:
Strategy A volatility 20% -> weight proportional to 1/0.20 = 5
Strategy B volatility 10% -> weight proportional to 1/0.10 = 10
Strategy C volatility 40% -> weight proportional to 1/0.40 = 2.5

Normalized:
Strategy A: 5/17.5 ~ 29%
Strategy B: 10/17.5 ~ 57%
Strategy C: 2.5/17.5 ~ 14%
```
Makes each strategy's risk contribution to the portfolio equal.

**Method 3: Dynamic Adjustment**
- Trending market -> Increase trend strategy weight
- Sideways market -> Increase mean reversion strategy weight
- This requires accurate Regime Detection (detailed in Lesson 11)

---

## 5.4 High-Risk Strategy Warnings

### Martingale Strategy (Use with Extreme Caution)

```
Logic:
Bet $100 first time, lose
Bet $200 second time, lose
Bet $400 third time, lose
Bet $800 fourth time, win!

Won $800, lost $700 before, net profit $100
```

**Looks beautiful, actually dangerous**:

| Consecutive Losses | Cumulative Investment | Single Bet |
|-------------------|----------------------|------------|
| 1 | $100 | $100 |
| 2 | $300 | $200 |
| 3 | $700 | $400 |
| 4 | $1,500 | $800 |
| 5 | $3,100 | $1,600 |
| 6 | $6,300 | $3,200 |
| 7 | $12,700 | $6,400 |
| 8 | $25,500 | $12,800 |
| 9 | $51,100 | $25,600 |
| 10 | $102,300 | $51,200 |

**10 consecutive losses requires $100,000**, but the profit is only the initial $100.

**Why would you lose 10 times in a row?**
- At 50% win rate, probability of 10 consecutive losses = 0.5^10 ~ 0.1%
- Looks small, but trading 1000 times means you'll encounter it once
- Once is enough to blow up

**If you must use it**:
- Set maximum number of doublings (e.g., max 3 times)
- Set maximum daily loss (e.g., 10%)
- Understand this is essentially "trading blowup risk for high win rate"

---

## 5.5 Options Strategies Introduction (Advanced)

> Options are advanced instruments with high risk. This section is just an introduction; study Greeks thoroughly before trading.

### Options Basics

| Term | Meaning |
|------|---------|
| Call Option | Right to buy at agreed price |
| Put Option | Right to sell at agreed price |
| Strike Price | Agreed buy/sell price |
| Expiration | Date option expires |
| Premium | Price to buy the option |

### Bull Call Spread

**Operation**:
1. Buy lower strike call option (e.g., $100 Call)
2. Sell higher strike call option (e.g., $110 Call)

```
P&L Diagram:
Profit |        ____
       |       /
   0   |------*
       |     /|
Loss   |____/ |
       └──────┴────── Stock Price
          100  110
```

| Feature | Description |
|---------|-------------|
| Cost | Lower than buying Call directly |
| Max Loss | Net premium paid |
| Max Profit | Strike difference - Net premium |
| Use Case | Expecting moderate upside |

### Expiration Day Options (High Risk)

Betting on volatility explosion 1-3 days before expiration:

**Principle**:
- Near expiration, options time value decays rapidly
- But if big events occur (earnings, Fed meetings), volatility can explode
- Gamma is extremely high; small price moves create huge gains (or losses)

| Feature | Description |
|---------|-------------|
| Potential Gain | Can be 10x or more |
| Potential Loss | Premium goes to zero (100% loss) |
| Win Rate | Usually below 20% |
| Position Sizing | <= 5% of total capital |

### Gamma Scalping (Professional)

**Principle**: Hold options position, repeatedly buy/sell underlying to hedge Delta.

```
Simplified:
1. Hold Call options (long Gamma)
2. Price rises -> Delta increases -> Sell stock to hedge
3. Price falls -> Delta decreases -> Buy stock to hedge
4. Repeat, profit from oscillation
```

**Profit condition**: Realized volatility > Implied volatility

| Requirement | Description |
|-------------|-------------|
| Transaction fees | Must be extremely low |
| Trading frequency | High, possibly dozens of times daily |
| Technical barrier | Must master options pricing and Greeks |

---

## 5.6 Strategy Selection Framework

### How to Select Strategies? (Falsifiable Rules)

Vague "watching the trend" is meaningless; need quantifiable, verifiable rules:

| Indicator Condition | Determination | Recommended Strategy | Invalidation Signal |
|--------------------|---------------|---------------------|---------------------|
| ADX > 25 and sustained 5+ days | Trend confirmed | Trend following | ADX < 20 for 3 consecutive days |
| ADX < 20 and price oscillating within Bollinger Bands | Sideways confirmed | Mean reversion | Price breaks Bollinger 2 sigma and ADX rising |
| Volatility > 90th percentile historically | Crisis mode | Reduce positions/hedge | Volatility falls below 50th percentile |
| None of above satisfied | Uncertain | Reduce position 50% | Any condition satisfied |

**Falsifiable Strategy Selection Rules**:

```
If following conditions met -> Use trend following:
  1. ADX(14) > 25 for 5 consecutive days
  2. Price on same side of 20-day MA for 10 consecutive days
  3. Last 20 days return significantly != 0 (t-test p < 0.05)

If following conditions met -> Use mean reversion:
  1. ADX(14) < 20 for 5 consecutive days
  2. Price oscillating within Bollinger Bands (20-day, 2 sigma)
  3. Last 20 days return close to 0 (t-test p > 0.2)

If none of above conditions met -> Reduce position to 50%, wait for clear signal
```

### Handling Regime Transition Periods

**The most dangerous moment isn't trending or sideways - it's the transition.**

| Transition Scenario | Risk | Response Strategy |
|--------------------|------|-------------------|
| Sideways -> Trending | Mean reversion gets trapped, stop-loss cascade | Reduce mean reversion positions immediately when breakout signal appears |
| Trending -> Sideways | Trend strategy gets whipsawed, repeated stop-losses | Gradually reduce trend positions when ADX declining |
| Normal -> Crisis | All strategies lose simultaneously, correlations spike | Priority is reducing positions when volatility spikes, not switching strategies |

**Conservative Rules for Transition Periods**:

1. **Confirmation lag**: Regime change confirmation needs 3-5 days; don't chase day-one signals
2. **Reduce first**: In uncertain periods, reduce positions first then switch strategies, rather than full swap
3. **Accept transition cost**: Reserve 5-10% for "switching costs," accept these losses

**Must test in backtesting**:
- Percentage of total losses from transition periods (if >50%, switching logic has problems)
- Switching delay in days (if >5 days, consider more sensitive indicators)
- False switch count (if too frequent, add confirmation conditions)

### Strategy Comparison Summary

| Dimension | Trend Following | Mean Reversion | Grid Trading |
|-----------|-----------------|----------------|--------------|
| Win rate | 30-45% | 55-70% | 60-75% |
| Risk/reward | 3:1 or higher | About 1:2 | About 1:3 |
| Max drawdown | 20-40% | 15-30% | Can blow up |
| Capital efficiency | Average | Higher | Low (capital spread thin) |
| Psychological pressure | Consecutive stop-losses | Holding losers | Being trapped |
| Suitable market | Trending | Sideways | Sideways |

### Common Misconceptions

**Misconception 1: Higher win rate strategies are better**

Not necessarily. A strategy with 40% win rate but 3:1 risk/reward has far higher expected return than 70% win rate with 0.5:1 risk/reward. Key is Expected Return = Win Rate x Profit - Loss Rate x Loss.

**Misconception 2: Best backtested parameters are optimal**

Dangerous assumption. Optimal parameters are often "overfit." If returns change dramatically with +/-20% parameter changes, those parameters just happened to work on historical data.

**Misconception 3: Trend strategy can be profitable in sideways markets by "adjusting parameters"**

Can't. Trend strategy logic assumes "trends exist." Sideways markets have no trends; no amount of parameter tuning can make it profitable. Correct approach is switching to mean reversion.

**Misconception 4: Grid trading is a "guaranteed profit" strategy**

Extremely dangerous. Grid trading does profit steadily in sideways markets, but one-sided decline means buying all the way down, capital trapped. Must set overall stop-loss.

### Multi-Agent Perspective

In multi-agent systems:

| Agent | Main Strategy | Activation Condition |
|-------|---------------|---------------------|
| Trend Agent | Trend following | Regime Agent determines trending market |
| Mean Reversion Agent | Mean reversion | Regime Agent determines sideways market |
| Crisis Agent | Defensive strategy | Volatility spikes or anomalies appear |
| Portfolio Agent | Multi-strategy combo | Dynamically adjusts strategy weights |
| Risk Agent | Risk control | Always on, veto power |

---

## Code Implementation (Optional)

### Dual Moving Average Strategy Backtest Framework

```python
import pandas as pd
import numpy as np

def dual_ma_strategy(df, short_window=5, long_window=20):
    """
    Dual moving average strategy
    Returns position signal: 1=long, -1=short, 0=flat
    """
    df = df.copy()
    df['MA_Short'] = df['close'].rolling(short_window).mean()
    df['MA_Long'] = df['close'].rolling(long_window).mean()

    # Generate signals
    df['signal'] = 0
    df.loc[df['MA_Short'] > df['MA_Long'], 'signal'] = 1   # Golden cross long
    df.loc[df['MA_Short'] < df['MA_Long'], 'signal'] = -1  # Death cross short

    return df['signal']


def grid_trading_signal(price, grid_center, grid_step, num_grids):
    """
    Grid trading signal
    Returns suggested position change
    """
    position_change = 0
    for i in range(1, num_grids + 1):
        buy_level = grid_center * (1 - grid_step * i)
        sell_level = grid_center * (1 + grid_step * i)

        if price <= buy_level:
            position_change = i  # Buy more as it falls
        elif price >= sell_level:
            position_change = -i  # Sell more as it rises

    return position_change


def calculate_strategy_metrics(returns):
    """Calculate strategy evaluation metrics"""
    total_return = (1 + returns).prod() - 1
    annual_return = (1 + total_return) ** (252 / len(returns)) - 1
    volatility = returns.std() * np.sqrt(252)
    sharpe = annual_return / volatility if volatility > 0 else 0

    # Correct max drawdown calculation using multiplicative equity curve
    equity_curve = (1 + returns).cumprod()
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    return {
        'total_return': f'{total_return:.2%}',
        'annual_return': f'{annual_return:.2%}',
        'volatility': f'{volatility:.2%}',
        'sharpe_ratio': f'{sharpe:.2f}',
        'max_drawdown': f'{max_drawdown:.2%}'
    }
```

---

## Lesson Deliverables

After completing this lesson, you will have:

1. **Understanding of two major strategy paradigms** - The essential differences between trend following vs mean reversion
2. **Implementation approaches for classic strategies** - Dual moving average, grid trading, pairs trading
3. **Strategy portfolio framework** - Understanding how to diversify risk through low-correlation strategies
4. **Awareness of high-risk strategy pitfalls** - Martingale, expiration day options traps

### Verification Checklist

| Checkpoint | Verification Standard | Self-Test Method |
|------------|----------------------|------------------|
| **Strategy characteristics** | Can state win rate/risk-reward features of trend following and mean reversion | Fill in comparison table without notes |
| **Strategy selection** | Can determine which strategy to use based on ADX value | Given ADX=30, explain strategy choice rationale |
| **Risk identification** | Can explain why Martingale is dangerous | Calculate capital needed for 8 consecutive losses |
| **Transition cost** | Can state three risks during regime transitions | Draw position adjustment flowchart during transitions |

**Scenario Exercises**:

1. **Scenario A**: SPY's ADX rose from 15 to 28 over the past 2 weeks, price broke above 20-day MA
   - Question: Which strategy should you choose? Why?

2. **Scenario B**: Your grid strategy profited 15% steadily over the past 3 months, then SPY suddenly dropped 12% in 5 days
   - Question: How should you adjust? Keep adding or stop-loss?

3. **Scenario C**: Martingale strategy finally wins after 6 consecutive losses
   - Question: What's the total profit from these 7 trades? How much total capital was invested?

<details>
<summary>Click to reveal answers</summary>

1. **Scenario A Answer**: Switch to trend following strategy. Reason: ADX > 25 and rising, price broke MA - meets trend confirmation conditions.

2. **Scenario B Answer**: Should stop-loss! Grid trading's greatest fear is one-sided decline. 12% drop in 5 days far exceeds normal oscillation. Setting overall stop-loss (e.g., 5% loss) is essential.

3. **Scenario C Answer**:
   - Initial bet $100
   - 7th trade won $6,400, previously lost $6,300 cumulative
   - Net profit only $100, but total invested $12,700
   - Risk/reward is terrible: risking $12,700 to make $100

</details>

---

## Key Takeaways

- [x] Understand trend following characteristics: low win rate, high risk/reward, fears sideways
- [x] Understand mean reversion characteristics: high win rate, low risk/reward, fears trend breakouts
- [x] Master basic principles of grid trading and pairs trading
- [x] Recognize the fatal risks of Martingale strategy
- [x] Understand the value and methods of multi-strategy portfolios

---

## Extended Reading

- [Background: HFT and Market Microstructure](Background/HFT-and-Market-Microstructure.md) - More complex strategy types
- [Background: Cryptocurrency Trading Characteristics](Background/Cryptocurrency-Trading-Characteristics.md) - Strategy adjustments for 24/7 markets

---

## Next Lesson Preview

**Lesson 06: The Harsh Reality of Data Engineering**

No matter how good your strategy is, if your data has problems, it's all wasted. API rate limits, missing data, time alignment, survivorship bias... These problems kill more strategies than model issues. Next lesson we face the harsh reality of data.
