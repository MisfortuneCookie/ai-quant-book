# Candlestick Patterns and Volume Analysis

> **Core Point**: Candlestick patterns and volume-price relationships are cornerstones of traditional technical analysis, but have limited predictive power when used alone. The correct approach is to use them as feature engineering inputs, not direct trading signals.

---

## Why Understand These?

If you've used any trading software (TradingView, Thinkorswim, Interactive Brokers), you've certainly seen various candlestick pattern annotations and volume bar charts. Many traders rely on these visual patterns for decision-making.

**As a quant practitioner, you need to understand these concepts for several reasons**:

1. **Communication language**: Common vocabulary when talking with traders and analysts
2. **Feature engineering**: These patterns can be quantified as input features for ML models
3. **Strategy evaluation**: Understanding traditional methods' limitations helps you design better systems
4. **Market psychology**: Candlestick patterns reflect the battle between buyers and sellers

---

## 1. Candlestick Basics

### 1.1 Candlestick Components

A single candlestick contains four price points:

![Candlestick Anatomy](assets/candlestick-anatomy.svg)

### 1.2 Information Content of Candlesticks

| Component | Information Reflected |
|-----------|----------------------|
| Body length | Strength of buyer/seller dominance |
| Upper shadow | Selling pressure from above |
| Lower shadow | Buying support from below |
| Body position | Result of intraday buyer/seller battle |

---

## 2. Common Single Candlestick Patterns

### 2.1 Doji

```
    │
   ─┼─  ← Open ≈ Close
    │
```

**Characteristics**: Very small or no body, upper and lower shadows can vary in length

| Type | Shape | Meaning |
|------|-------|---------|
| Standard Doji | Similar upper/lower shadows | Buyer/seller equilibrium, indecision |
| Dragonfly Doji | Long lower shadow, no upper | Strong support below |
| Gravestone Doji | Long upper shadow, no lower | Strong resistance above |

**Traditional interpretation**: Signal of potential trend reversal
**Actual effectiveness**: Prediction accuracy around 50-55% when used alone

### 2.2 Hammer and Hanging Man

```
   │
   █  ← Small body at top
   │
   │ ← Long lower shadow (at least 2x body length)
```

**Difference is in context**:
- **Hammer**: Appears at bottom of downtrend → Potential upward reversal
- **Hanging Man**: Appears at top of uptrend → Potential downward reversal

### 2.3 Shooting Star

```
   │ ← Long upper shadow
   │
   █  ← Small body at bottom
```

**Meaning**: Appears at top of uptrend, suggests strong resistance above, possible top

### 2.4 Large Bullish and Bearish Candles

| Pattern | Characteristics | Meaning |
|---------|-----------------|---------|
| Large bullish | Long body, short shadows | Bulls in strong control |
| Large bearish | Long body, short shadows | Bears in strong control |

**Quantitative definition**: Body length > 1.5× recent ATR

---

## 3. Common Multi-Candlestick Patterns

### 3.1 Engulfing Pattern

```
Bullish Engulfing (at downtrend):    Bearish Engulfing (at uptrend):

    █                                      │
    █  First small bearish               ███ First small bullish
   ┌─┐                                   ┌─┐
   │ │ Second large bullish              █ █ Second large bearish
   │ │ Completely covers first           █ █ Completely covers first
   └─┘                                   └─┘
```

**Conditions**:
1. Second candle's body completely covers the first
2. Two candles are opposite colors
3. Appears in a clear trend

### 3.2 Morning Star and Evening Star

```
Morning Star (bottom reversal):

█     ← First: Large bearish
█
█
  ┼   ← Second: Small body (doji or small candle), gaps down
 ┌┐
 ││   ← Third: Large bullish, closes well into first candle's body
 ││
```

**Evening Star**: Mirror image of Morning Star, appears at tops

### 3.3 Three Black Crows and Three White Soldiers

| Pattern | Structure | Meaning |
|---------|-----------|---------|
| Three Black Crows | Three consecutive large bearish candles, each opens lower, closes lower | Strongly bearish |
| Three White Soldiers | Three consecutive large bullish candles, each opens higher, closes higher | Strongly bullish |

---

## 4. Volume Analysis Fundamentals

### 4.1 Meaning of Volume

Volume = Number of shares traded between buyers and sellers

**Core principle**: Price is direction, volume is momentum

| Volume-Price Relationship | Meaning | Reliability |
|---------------------------|---------|-------------|
| Price up + Volume up | Uptrend has capital support | Trend likely to continue |
| Price up + Volume down | Uptrend lacks momentum | Trend may exhaust |
| Price down + Volume up | Panic selling | May accelerate down or near bottom |
| Price down + Volume down | Downtrend momentum fading | May stabilize |

### 4.2 Key Volume-Price Patterns

#### Volume Breakout

```
Price ─────────┐
              └──────── Breakout
Volume       ████████ ← Volume significantly increases (>1.5× average)
```

**Meaning**: Breakout confirmed by capital, high credibility

#### Low-Volume Pullback

```
Price    /\
       /  \  ← Pullback
      /
Volume ██  █ ← Volume shrinks during pullback
```

**Meaning**: Pullback is normal profit-taking, not trend reversal

#### Volume-Price Divergence

```
Price   /\  /\  /\ ← Price makes new highs
      /  \/  \/
Volume ██ █  ▪ ← Volume decreases
```

**Meaning**: Uptrend momentum exhausting, possible top (similar to MACD divergence)

### 4.3 Common Volume Indicators

| Indicator | Formula | Purpose |
|-----------|---------|---------|
| **Volume Ratio** | Current volume / N-day average volume | Measure daily trading activity |
| **Turnover Rate** | Volume / Float shares | Measure share turnover |
| **OBV** | Cumulative volume (add on up days, subtract on down days, unchanged on flat) | Capital flow direction |
| **VWAP** | Σ(Price × Volume) / ΣVolume | Institutional trading benchmark |

---

## 5. Empirical Research on Pattern Recognition

### 5.1 Academic Research Conclusions

Multiple academic studies have tested the predictive power of candlestick patterns:

| Study | Sample | Conclusion |
|-------|--------|------------|
| Lo, Mamaysky & Wang (2000) | US stocks 1962-1996 | Some patterns statistically significant, but limited economic significance |
| Marshall, Young & Rose (2006) | 35 markets | Most patterns have no predictive power |
| Caginalp & Laurent (1998) | S&P 500 components | Certain patterns effective under specific conditions |

**Summary**:
- Pattern-only win rates typically 50-55%
- After trading costs, most pattern strategies aren't profitable
- Pattern effectiveness may decay over time (learned by market)

### 5.2 Why Patterns "Appear to Work"

1. **Confirmation bias**: People remember successes, forget failures
2. **Hindsight bias**: Looking at past charts, you can always find "perfect" patterns
3. **Survivorship bias**: Only successful traders share their stories
4. **Vague definitions**: What counts as a "standard" hammer? No precise definition

### 5.3 When Might Patterns Be Effective?

Research shows patterns perform better under these conditions:

- **Combined with volume**: Patterns with volume confirmation are more reliable
- **At key levels**: Appearing near support/resistance
- **Market environment**: In specific volatility regimes
- **Timeframes**: Longer timeframes (daily, weekly) more reliable than minute charts

---

## 6. Quantitative Implementation: Patterns as Features

### 6.1 Challenges in Quantifying Patterns

Traditional pattern analysis is "pattern matching" but with vague definitions:
- "Long lower shadow" - how long is long?
- "Small body" - how small is small?
- "In a trend" - how do you define trend?

**Solution**: Convert patterns to continuous features, not discrete signals

### 6.2 Feature Engineering Example

```python
def calculate_candlestick_features(df):
    """
    Convert candlestick patterns to continuous features
    df needs: open, high, low, close, volume columns
    """
    # Basic calculations
    df['body'] = df['close'] - df['open']
    df['body_abs'] = df['body'].abs()
    df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
    df['range'] = df['high'] - df['low']

    # Feature 1: Body ratio (0=doji, 1=no shadows)
    df['body_ratio'] = df['body_abs'] / df['range'].replace(0, np.nan)

    # Feature 2: Upper shadow ratio
    df['upper_shadow_ratio'] = df['upper_shadow'] / df['range'].replace(0, np.nan)

    # Feature 3: Lower shadow ratio
    df['lower_shadow_ratio'] = df['lower_shadow'] / df['range'].replace(0, np.nan)

    # Feature 4: Close position (where close sits in day's range)
    df['close_position'] = (df['close'] - df['low']) / df['range'].replace(0, np.nan)

    # Feature 5: Relative body size (compared to recent ATR)
    atr = calculate_atr(df, period=14)
    df['relative_body'] = df['body_abs'] / atr

    # Feature 6: Engulfing ratio (current body covers previous body)
    df['engulfing_ratio'] = df['body_abs'] / df['body_abs'].shift(1)

    return df

def calculate_volume_features(df, periods=[5, 20]):
    """
    Calculate volume-related features
    """
    for p in periods:
        # Volume ratio
        df[f'volume_ratio_{p}'] = df['volume'] / df['volume'].rolling(p).mean()

    # OBV (On-Balance Volume)
    df['obv'] = (np.sign(df['close'].diff()).fillna(0) * df['volume']).cumsum()

    # OBV trend (slope of OBV moving average)
    df['obv_slope'] = df['obv'].rolling(10).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0]
    )

    # Price-volume correlation (recent price change vs volume correlation)
    df['price_volume_corr'] = df['close'].pct_change().rolling(20).corr(
        df['volume'].pct_change()
    )

    return df
```

### 6.3 Feature Usage Recommendations

| Feature Type | Suitable Models | Notes |
|--------------|-----------------|-------|
| Continuous pattern features | Tree models, Neural networks | Need normalization |
| Binary pattern labels | Logistic regression, Rule systems | Definitions must be consistent |
| Sequential patterns | LSTM, Transformer | Need time window |

---

## 7. Multi-Agent Perspective

In multi-agent systems, candlestick patterns and volume analysis can be positioned as:

| Agent | Usage |
|-------|-------|
| **Trend Agent** | Use large candles to judge trend strength |
| **Mean Reversion Agent** | Use doji, hammer to identify potential reversal points |
| **Regime Agent** | Volume-price divergence as one signal for market regime changes |
| **Risk Agent** | Abnormal volume spikes as risk warning signals |

**Key principles**:
- Pattern features are one input, not the only basis
- Combine with other features (MACD, RSI, fundamentals) for comprehensive judgment
- Agent outputs probabilities, not deterministic signals

---

## 8. Common Misconceptions

### Misconception 1: Patterns are "Secret Codes"

> "Learning to read candlestick patterns guarantees consistent profits"

**Truth**: If patterns were so effective, the more people use them, the less effective they become (reflexivity)

### Misconception 2: Patterns Work Everywhere

> "A hammer always signals a bottom"

**Truth**: The same pattern may perform completely differently across market environments and instruments

### Misconception 3: Volume is Always Meaningful

> "High volume means institutional buying"

**Truth**: High volume could be retail panic, algorithmic trading, index rebalancing, and many other reasons

### Misconception 4: Patterns Can Predict Precisely

> "An evening star means tomorrow definitely drops"

**Truth**: Patterns only provide probabilistic edge (if any), not deterministic predictions

---

## Key Takeaways

- [x] Candlestick patterns are visual representations of buyer-seller battles
- [x] Pattern-only prediction power is limited (50-55% win rate)
- [x] Volume confirmation can improve pattern reliability
- [x] Correct usage is quantifying patterns as ML features, not direct signals
- [x] In multi-agent systems, patterns are one of many inputs

---

## Further Reading

- [Lesson 04: The Real Role of Technical Indicators](../Lesson-04-The-Real-Role-of-Technical-Indicators.md) - MACD, RSI and other indicators explained
- [Lesson 09: Supervised Learning in Quantitative Trading](../../Part3-Machine-Learning/Lesson-09-Supervised-Learning-in-Quantitative-Trading.md) - How to input features into ML models
- [Background: Feature Engineering Common Pitfalls](../../Part3-Machine-Learning/Background/Feature-Engineering-Common-Pitfalls.md) - Avoid feature engineering mistakes

---

## References

- Nison, S. (1991). *Japanese Candlestick Charting Techniques*
- Lo, A., Mamaysky, H., & Wang, J. (2000). Foundations of Technical Analysis. *Journal of Finance*
- Murphy, J. (1999). *Technical Analysis of the Financial Markets*
