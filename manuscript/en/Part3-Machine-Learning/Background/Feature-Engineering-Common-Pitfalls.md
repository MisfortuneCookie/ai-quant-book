# Background: Feature Engineering Common Pitfalls

> 80% of Alpha in quantitative trading comes from feature engineering, but 80% of failures also come from feature engineering.

---

## Pitfall 1: Future Information Leakage

**Problem**: Accidentally using future data in feature calculation.

```python
# Wrong Example: Using today's close for indicator, but deciding at open
df['ma20'] = df['close'].rolling(20).mean()  # Includes today's close
signal = df['close'] > df['ma20']  # Uses today's close to judge

# Correct Approach: shift(1) ensures only historical data is used
df['ma20'] = df['close'].rolling(20).mean().shift(1)
signal = df['close'] > df['ma20']
```

**Detection Methods**:
- Feature-label correlation > 0.9 -> Likely leakage
- Model accuracy > 90% -> Almost certainly leakage

---

## Pitfall 2: Global Normalization

**Problem**: Normalizing with whole-dataset statistics leaks future distribution information.

```python
# Wrong Example
mean = df['close'].mean()  # Includes future data
std = df['close'].std()
df['normalized'] = (df['close'] - mean) / std

# Correct Approach: Rolling window normalization
df['normalized'] = (
    (df['close'] - df['close'].rolling(20).mean()) /
    df['close'].rolling(20).std()
)
```

---

## Pitfall 3: Highly Correlated Feature Redundancy

**Problem**: Multiple features are highly correlated, creating information redundancy and overfitting risk.

| Feature Group | Correlation | Problem |
|---------------|-------------|---------|
| MA5, MA10, MA20 | 0.95+ | Nearly identical information |
| RSI, Stochastic | 0.8+ | Both measure overbought/oversold |
| Close, VWAP | 0.99 | Nearly identical |

**Solution**:
- Keep only one feature when correlation > 0.7
- Use PCA for dimensionality reduction
- Choose the most explanatory representative

---

## Pitfall 4: Overfitting to Noise

**Problem**: More features mean more chance of memorizing training data noise.

```
Feature count vs Sample count rule of thumb:

Samples / Features > 20  -> Safe zone
Samples / Features = 10  -> Warning zone
Samples / Features < 5   -> Danger zone

Example: 1000 samples, use at most 50 features
```

**Solution**:
- Recursive feature elimination, observe validation set performance
- Use L1 regularization for automatic selection
- Domain knowledge takes priority over statistical significance

---

## Pitfall 5: Categorical Feature Encoding Errors

**Problem**: Incorrect handling of categorical features leads to information loss or spurious relationships.

```python
# Wrong Example: Direct numeric encoding (implies ordering)
sector_map = {'Technology': 1, 'Healthcare': 2, 'Finance': 3}
df['sector'] = df['sector_name'].map(sector_map)  # Model thinks 3 > 2 > 1

# Correct Approach: One-Hot encoding
df = pd.get_dummies(df, columns=['sector_name'])
```

---

## Pitfall 6: Ignoring Time-Varying Correlation

**Problem**: Feature predictive power is unstable across different time periods.

| Period | Momentum Factor IC | Value Factor IC |
|--------|-------------------|-----------------|
| 2015-2017 | 0.05 | 0.02 |
| 2018-2019 | 0.01 | 0.04 |
| 2020-2021 | 0.06 | -0.02 |

**Solution**:
- Use rolling IC instead of single IC
- IC stability (IC/std(IC)) more important than absolute IC value
- Consider feature effectiveness conditional on regime

---

## Pitfall 7: Data Snooping

**Problem**: Repeatedly testing until finding "effective" features, actually overfitting.

```
Test 100 features
-> Expect 5 to be "significant" at p<0.05
-> These 5 may just be random noise
```

**Solution**:
- Use Bonferroni correction: p-value threshold = 0.05 / number of tests
- Reserve independent OOS dataset
- Record all tested features, not just "successful" ones

---

## Pitfall 8: Ignoring Trading Cost Impact

**Problem**: High-turnover features get eaten by costs in live trading.

```
Feature A: IC = 0.05, turnover 200%/month
Feature B: IC = 0.03, turnover 50%/month

Assuming one-way cost 0.2%:
Feature A cost: 200% x 0.2% x 2 = 0.8%/month
Feature B cost: 50% x 0.2% x 2 = 0.2%/month

After costs, Feature B may be better
```

---

## Feature Engineering Checklist

| Check Item | Passing Standard |
|------------|------------------|
| No future information | All features use shift(1) or earlier data |
| Rolling normalization | No global mean/std used |
| Low correlation | Inter-feature correlation < 0.7 |
| Sample ratio | Samples / Features > 20 |
| IC stability | IC / std(IC) > 0.5 |
| Cost feasibility | Still positive returns after turnover costs |
