# Background: Limitations of Machine Learning in Finance

> "If deep learning could predict stock prices, why aren't all top AI companies doing quantitative trading?"

---

## Core Limitation: Extremely Low Signal-to-Noise Ratio

| Domain | Signal-to-Noise Ratio | Achievable Accuracy |
|--------|----------------------|---------------------|
| Image Recognition | High | 95%+ |
| Speech Recognition | High | 90%+ |
| Natural Language | Medium | 80%+ |
| **Stock Price Prediction** | **Extremely Low** | **52-55% is already top-tier** |

**Why is financial signal-to-noise so low?**

1. **Markets are nearly efficient**: Obvious patterns are quickly arbitraged away
2. **Many participants**: Patterns you find, others are using too
3. **Noise dominates**: 90% of short-term price movement is random fluctuation
4. **Reflexivity**: The prediction itself changes what's being predicted

---

## Limitation 1: Not Enough Data

Deep learning requires massive data, but financial data is limited.

| Data Type | Sample Size | Deep Learning Suitability |
|-----------|-------------|---------------------------|
| 20 years daily data | 5,000 records | Far from enough |
| 5 years minute data | 500,000 records | Marginally usable |
| 1 year tick data | Millions of records | Can try |

**Comparison**: ImageNet has 14 million images, GPT was trained on trillions of tokens.

---

## Limitation 2: Unstable Distribution

Training data and prediction data have different distributions (Regime Shift).

```
Training Set (2015-2019):
  - Bull market mostly
  - Volatility 15%
  - VIX average 15

Test Set (2020):
  - COVID crash
  - Volatility spiked to 80%
  - VIX peak 82

-> Model completely fails
```

**Deep Learning Assumption**: Training and test data come from the same distribution. Financial markets violate this assumption.

---

## Limitation 3: Easy to Overfit

"Patterns" in financial data may just be noise.

| Model Complexity | Training Set Performance | Test Set Performance | Diagnosis |
|------------------|-------------------------|---------------------|-----------|
| Simple Linear | 8% annualized | 6% annualized | Normal |
| Random Forest | 25% annualized | 8% annualized | Slight overfitting |
| LSTM | 80% annualized | -5% annualized | Severe overfitting |
| Transformer | 150% annualized | -15% annualized | Catastrophic overfitting |

**Complex Model does not equal Better Prediction**; in finance, often the opposite.

---

## Limitation 4: Prediction does not equal Profit

52% accuracy sounds better than random, but may lose money after costs.

```
Assumptions:
  - Prediction accuracy 52%
  - Each win 1%, each loss 1%
  - Trading cost 0.3%

Expected return = 52% x 1% - 48% x 1% - 0.3%
               = 0.52% - 0.48% - 0.3%
               = -0.26% (losing money!)

Required win/loss ratio:
  Win 1.5%, lose 1%
  -> 52% x 1.5% - 48% x 1% - 0.3% = 0.28% (small profit)
```

---

## Limitation 5: Poor Interpretability

Deep learning is a black box; financial regulation and risk control need explanations.

| Scenario | Linear Model | Deep Learning |
|----------|--------------|---------------|
| Why buy this stock? | "High momentum factor score" | "Network output 0.7" |
| Loss attribution | "Value factor failed" | Unknown |
| Regulatory explanation | Can provide | Difficult |
| Risk control adjustment | Adjust single factor | Needs retraining |

---

## Limitation 6: Hardware and Cost

Training deep models requires significant compute power; quant returns may not cover costs.

| Resource | Cost | Required Return |
|----------|------|-----------------|
| GPU cluster training | $10,000+/month | Annualized > 10% |
| Data purchase | $50,000+/year | Annualized > 5% |
| Talent cost | $200,000+/year | Annualized > 20% |

**Comparison**: A simple moving average strategy costs near zero.

---

## When Does ML Actually Work?

| Scenario | ML Effectiveness | Reason |
|----------|------------------|--------|
| High-frequency trading | Limited | Latency matters more than model |
| Daily stock selection | Usable | Enough data, moderate complexity |
| Monthly asset allocation | Limited | Too little data |
| Alternative data mining | Valuable | Unstructured data processing |
| Risk modeling | Valuable | Predicting volatility easier than returns |

---

## Practical Recommendations

### 1. Simple Models First
```
First choice: Linear regression, Ridge regression, Logistic regression
Second: Random Forest, XGBoost
Last: LSTM, Transformer
```

### 2. Validation Over Model
```
Spend 80% of time on validation:
- Walk-Forward validation
- Multi-period stability
- Return after costs
```

### 3. Features Over Model
```
80% of Alpha comes from feature engineering
20% from model selection

Good features + Simple model > Poor features + Complex model
```

### 4. Predict Volatility Instead of Returns
```
Volatility is easier to predict:
- Volatility has clustering effect
- Volatility autocorrelation 0.7-0.9
- Return autocorrelation ≈ 0

Use ML to predict volatility -> Use rules to trade
```

---

## Summary

| Limitation | Impact | Coping Strategy |
|------------|--------|-----------------|
| Low signal-to-noise | Accuracy hard to exceed 55% | Lower expectations |
| Insufficient data | Easy to overfit | Simplify model |
| Distribution drift | Model failure | Rolling retraining |
| High costs | Returns eaten up | Reduce turnover |
| Black box | Hard to risk control | Maintain interpretability |

**Core Conclusion**: ML's value in quant is **signal enhancement**, not **predicting price movements**.
