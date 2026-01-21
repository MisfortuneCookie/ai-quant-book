# Background: Deep Learning Model Architecture Selection Guide

> Choosing the right architecture is half the battle. Different models suit different scenarios - there is no "universal model."

---

## 1. Model Architecture Quick Reference

| Model Type | Parameter Scale | Training Time | Inference Latency | Use Cases | Advantages | Disadvantages |
|------------|----------------|---------------|-------------------|-----------|------------|---------------|
| **LSTM** | 1-10M | Medium | <10ms | Short-term price prediction, HFT | Captures temporal dependencies, stable training | Performance degrades on long sequences |
| **GRU** | 0.5-5M | Faster | <8ms | Resource-constrained, real-time inference | Fewer parameters, faster training | Slightly less expressive than LSTM |
| **Transformer** | 10-100M | High | 10-50ms | Multi-asset portfolios, long-term trends | Parallel training, long-range dependencies | High data requirements, overfitting risk |
| **CNN** | 0.5-5M | Fast | <5ms | Technical pattern recognition, pattern matching | Local feature extraction, efficient | Weak temporal modeling |
| **CNN-LSTM Hybrid** | 5-20M | Medium-High | 10-30ms | Multi-timeframe analysis | Combines local and global features | High complexity, difficult to tune |

---

## 2. LSTM/GRU: The Workhorses of Temporal Modeling

### 2.1 Architecture Principles

LSTM (Long Short-Term Memory) controls information flow through **three gating mechanisms**:

```
Input Gate:  Decides what new information to write to memory
Forget Gate: Decides what old information to discard
Output Gate: Decides what memory information to output
```

**GRU (Gated Recurrent Unit)** is a simplified version of LSTM:
- Combines input and forget gates into a single "update gate"
- Approximately 25% fewer parameters, faster training
- Performs comparably to LSTM on small datasets

### 2.2 Typical Architecture Configurations

```
Single-asset daily strategy:
├── Input layer: 20-60 timesteps x 10-30 features
├── LSTM layer 1: 128 units + Dropout(0.2)
├── LSTM layer 2: 64 units + Dropout(0.2)
├── Dense layer: 32 units + ReLU
└── Output layer: 1 unit (regression) or 3 units (classification: up/down/flat)

High-frequency trading (minute-level):
├── Input layer: 60-120 timesteps x 50-100 features
├── GRU layer: 256 units (speed priority)
├── Dense layer: 64 units
└── Output layer: Discrete actions (buy/sell/hold)
```

### 2.3 When to Choose LSTM/GRU?

| Scenario | Recommendation | Rationale |
|----------|----------------|-----------|
| Data volume < 100K samples | LSTM/GRU | Transformers easily overfit on small datasets |
| Sequence length < 100 steps | LSTM/GRU | LSTM is sufficient; Transformer advantages are minimal |
| Inference latency < 10ms | GRU | Fewer parameters, faster inference |
| Single-asset strategy | LSTM | Captures temporal patterns of individual assets |

### 2.4 Important Findings

According to the arXiv paper "Vanilla LSTMs Outperform Transformer-based Forecasting":

> In financial time series prediction tasks, standard LSTMs often outperform more complex Transformer architectures in scenarios with **limited data** or **shorter sequences**.

**Reason**: Financial data has a low signal-to-noise ratio; complex models tend to learn noise rather than genuine patterns.

---

## 3. Transformer: The Choice for Long Sequences and Multi-Asset

### 3.1 Core Innovations

**Self-Attention Mechanism**:
- Attends to all positions in the sequence simultaneously
- Captures long-range dependencies
- Supports parallel computation for efficient training

**Positional Encoding**:
- Preserves temporal order information
- Compensates for the attention mechanism's inherent position-agnostic nature

### 3.2 Financial Domain Variants

| Variant | Improvements | Use Cases |
|---------|-------------|-----------|
| **Informer** | Sparse attention, reduced computational complexity | Long sequence prediction (>1000 steps) |
| **Autoformer** | Autocorrelation mechanism captures periodicity | Highly seasonal data |
| **StockFormer** | End-to-end reinforcement learning | Direct trading decision output |
| **Higher-Order Transformer** | Higher-order attention, feature interactions | Stock price prediction (+5-10% accuracy) |

### 3.3 When to Choose Transformer?

| Scenario | Recommendation | Rationale |
|----------|----------------|-----------|
| Multi-asset portfolio (>50 assets) | Transformer | Simultaneously models inter-asset relationships |
| Long sequences (>200 steps) | Transformer | Strong long-term dependency modeling |
| Data volume > 1M samples | Transformer | Fully utilizes model capacity |
| Macroeconomic forecasting | Transformer | Captures long-term trends |

### 3.4 Caveats

```
Transformer Pitfalls:
1. High overfitting risk → Requires strong regularization (Dropout >= 0.3)
2. High data requirements → Underperforms LSTM with insufficient samples
3. High computational cost → GPU training is essential
4. Positional encoding sensitivity → Requires adjustment for financial data
```

---

## 4. CNN: The Pattern Recognition Powerhouse

### 4.1 Application Approaches

**1D CNN**: Directly processes price sequences
```
Input: Past 60 days of OHLCV data (60x5 matrix)
Kernels: Multiple sizes (3, 5, 7 days) extract features at different periods
Pooling: Max pooling or average pooling
Output: Feature vector → Classification/regression head
```

**2D CNN**: Processes candlestick chart images
```
Input: Candlestick chart rendered as image (e.g., 224x224x3)
Architecture: Similar to ResNet or VGG
Purpose: Identifies head-and-shoulders, double bottoms, triangles, and other classic patterns
```

### 4.2 When to Choose CNN?

| Scenario | Recommendation | Rationale |
|----------|----------------|-----------|
| Technical pattern recognition | CNN | Excels at extracting local spatial features |
| Ultra-low latency requirements | CNN | Fastest inference speed |
| Correlation matrix analysis | 2D CNN | Visualizes multi-asset relationships |

### 4.3 Limitations

```
CNN Issues in Finance:
1. Ignores temporal order → Needs positional encoding or RNN combination
2. Local receptive field → Difficulty capturing long-term dependencies
3. Candlestick chart subjectivity → Different rendering methods affect results
```

---

## 5. Hybrid Architectures: Best of Both Worlds

### 5.1 CNN-LSTM

```
Architecture:
Input → CNN (extract local features) → LSTM (model temporal dependencies) → Output

Advantages:
- CNN quickly filters key features
- LSTM captures temporal evolution patterns
- Multi-timeframe fusion

Disadvantages:
- High tuning complexity
- Increased overfitting risk
```

### 5.2 LSTM-Transformer

```
Architecture:
Input → LSTM (local temporal) → Transformer (global context) → Output

Use Cases:
- Markets with both short-term momentum and long-term trends
- Strategies requiring regime switch detection
```

### 5.3 Hybrid Architecture Recommendations

| Data Characteristics | Recommended Architecture |
|---------------------|-------------------------|
| Strong short-term + weak long-term dependencies | LSTM-dominant |
| Weak short-term + strong long-term dependencies | Transformer-dominant |
| Both equally important | CNN-LSTM or LSTM-Transformer |
| Uncertain | Start with LSTM, gradually increase complexity |

---

## 6. Reinforcement Learning Algorithm Selection

### 6.1 Core Algorithm Comparison

| Algorithm | Annual Return | Sharpe Ratio | Max Drawdown | Sample Efficiency | Training Stability | Use Cases |
|-----------|--------------|--------------|--------------|-------------------|-------------------|-----------|
| **DQN** | 8-15% | 0.6-1.2 | 15-25% | Medium | Medium (prone to divergence) | HFT, discrete actions |
| **PPO** | 15-25% | 1.2-1.8 | 10-18% | Higher | High (stable convergence) | Medium/low frequency, continuous actions |
| **A3C** | 10-18% | 0.8-1.4 | 12-22% | Higher | Low (noticeable oscillation) | Parallel exploration, resource-constrained |
| **SAC** | 12-20% | 1.0-1.6 | 12-20% | Higher | Medium-High | HFT, encourages exploration |
| **DDPG** | 8-15% | 0.6-1.2 | 15-25% | Medium | Low | Continuous actions, precise positioning |

### 6.2 Selection Recommendations

```
Start with PPO → Best balance between stability and performance

If you need discrete actions (buy/sell/hold) → DQN
If you need continuous actions (position sizing) → PPO or SAC
If you want maximum exploration → SAC
If you have resources for parallelization → A3C
```

---

## 7. Practical Selection Workflow

### 7.1 Decision Tree

```
                    Data volume > 1M?
                    /            \
                  Yes             No
                   |               |
             Sequence >200?    Sequence <100?
             /        \         /        \
           Yes        No       Yes        No
            |          |        |          |
      Transformer   Hybrid   LSTM      GRU/LSTM
```

### 7.2 Quick Selection Table

| Your Situation | Recommended Architecture | Rationale |
|----------------|-------------------------|-----------|
| Beginner, want quick validation | **LSTM + PPO** | Mature, stable, abundant tutorials |
| Daily single-asset strategy | **LSTM** | Simple and effective |
| Minute-level HFT strategy | **GRU + DQN** | Low latency |
| Multi-asset portfolio optimization | **Transformer** | Captures inter-asset relationships |
| Technical pattern recognition | **CNN** | Excels at local patterns |
| Uncertain, want stability | **LSTM → gradually increase complexity** | Avoid premature optimization |

---

## 8. Common Misconceptions

**Misconception 1: Transformers are always better than LSTMs**

Not true. In finance, with limited data and low signal-to-noise ratio, LSTMs are often more robust.

**Misconception 2: More complex models are better**

The opposite is true. Financial data is noisy; complex models easily overfit. **Simple model + good features > Complex model + poor features**.

**Misconception 3: Copy NLP/CV architecture configurations directly**

Financial data has unique properties: non-stationarity, low signal-to-noise ratio, regime changes. Targeted adjustments are necessary.

**Misconception 4: Select models based only on backtest metrics**

Must also consider: inference latency, deployment complexity, interpretability requirements. In live trading, GRU may be more practical than Transformer.

---

## 9. Technical Selection Summary

| Complexity | Data Relationships | Recommended Architecture |
|------------|-------------------|-------------------------|
| Simple linear | Traditional factors | LightGBM/XGBoost |
| Medium complexity | Short-term temporal | LSTM/GRU |
| Highly nonlinear | Long-term dependencies | Transformer |
| Dynamic decision-making | Sequential decisions | Reinforcement Learning (PPO) |
| Multi-modal data | Text + numerical | LLM + LSTM hybrid |

### General Training Strategy Recommendations

1. **Experience Replay**: Breaks temporal correlation, stabilizes training
2. **Target Network**: Delayed updates reduce oscillation
3. **Gradient Clipping**: Prevents gradient explosion
4. **Model Ensembling**: Reduces single-point-of-failure risk
5. **Rigorous Historical Validation**: Walk-Forward testing is essential

---

## 10. Further Reading

- [Background: Reinforcement Learning in Trading](Reinforcement-Learning-in-Trading.md) - Detailed RL introduction
- [Background: Frontier ML and RL Methods (2025)](Frontier-ML-and-RL-Methods-2025.md) - Latest technical advances
- [Background: Time Series Cross-Validation (Purged CV)](Time-Series-Cross-Validation-Purged-CV.md) - Proper validation methods
- arXiv: "Vanilla LSTMs Outperform Transformer-based Forecasting"
- arXiv: "Higher Order Transformers: Enhancing Stock Movement Prediction"

---

> **Core Insight**: Model architecture selection is not about chasing the latest and most complex options, but about matching your data scale, latency requirements, and strategy type. Start simple, gradually increase complexity, and validate every decision with Walk-Forward testing.
