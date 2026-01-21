# Background: Reinforcement Learning in Trading

> "Supervised learning tells you whether it will go up or down tomorrow; reinforcement learning tells you how much to buy and when to stop-loss."

---

## Why is Trading Suitable for Reinforcement Learning?

| Trading Problem | RL Framework Mapping |
|-----------------|---------------------|
| Current position, market state | State |
| Buy/Sell/Hold/Position size | Action |
| Return after costs | Reward |
| Trading costs, slippage | Environment feedback |
| Maximize long-term returns | Objective function |

**Key Insight**: Trading is a **sequential decision problem**, not an independent prediction problem.

---

## RL vs Supervised Learning: Core Differences

| Dimension | Supervised Learning | Reinforcement Learning |
|-----------|---------------------|------------------------|
| Objective | Prediction accuracy | Maximize cumulative returns |
| Label Source | Historical data (known) | Action results (discovered through exploration) |
| Decision Continuity | Each prediction is independent | Considers future impact |
| Trading Costs | Deducted afterward | Built into decisions |
| Position Management | Additional logic | Native support |

**Comparison Example**:

```
Supervised Learning Approach:
1. Predict: 60% probability AAPL goes up tomorrow
2. Action: Buy
3. Problem: How much? What about existing positions?

Reinforcement Learning Approach:
1. State: Currently hold 100 shares, $10,000 balance, AAPL in uptrend
2. Decision: Based on Q-value, should add 50 shares
3. Consideration: This decision accounts for trading costs, position risk, and possible future movements
```

---

## Core Elements of Trading RL

### 1. State Space

| State Type | Examples | Dimensions |
|------------|----------|------------|
| Price Features | Returns, MA, RSI | 10-50 |
| Position Info | Current position, cost basis, unrealized P&L | 3-5 |
| Account Info | Cash ratio, margin used | 2-3 |
| Market State | Volatility, trend strength | 2-5 |

**State Design Principles**:
- Include enough information for decision-making
- Avoid dimension explosion
- Normalize values

### 2. Action Space

**Discrete Actions** (simple but coarse):
```
Actions = {Strong Sell, Sell, Hold, Buy, Strong Buy}
Mapping = {-100%, -50%, 0%, +50%, +100%}
```

**Continuous Actions** (precise but complex):
```
Action = Target position ratio in [-1, 1]
-1 = 100% short
 0 = Flat
+1 = 100% long
```

### 3. Reward Function — The Most Critical Design

Reward function design is the **make-or-break factor** of an RL trading system. The wrong reward will cause the Agent to learn completely opposite behaviors.

#### Four Types of Reward Functions and Their Use Cases

| Type | Formula | Pros | Cons | Use Case |
|------|---------|------|------|----------|
| **Profit-Oriented** | `r_t = position × return - cost` | Simple, intuitive | Ignores risk | Beginner/Baseline |
| **Risk-Adjusted** | `r_t = (return - rf) / volatility` | Considers risk | Ignores costs | Medium/Low frequency |
| **Cost-Penalized** | `r_t = profit - α|Δpos| - β×spread` | Controls turnover | Parameter sensitive | High frequency |
| **Multi-Objective** | `r_t = w1×profit + w2×sharpe - w3×drawdown` | Comprehensive balance | Complex tuning | Production-grade |

---

#### Type 1: Profit-Oriented

```python
# The simplest reward function
r_t = position_{t-1} × (P_t - P_{t-1}) - transaction_cost

# Example:
# Hold 100 shares, price goes from $100 to $101, commission $5
# r_t = 100 × (101 - 100) - 5 = $95
```

**Problem**: Agent learns "all-in" strategy—high volatility means high gains/losses, no risk consideration.

---

#### Type 2: Risk-Adjusted Return

```python
# Similar to instantaneous Sharpe ratio
r_t = (return_t - risk_free_rate) / volatility_t

# Where:
# return_t = position × price_change / portfolio_value
# volatility_t = rolling_std(returns, window=20)

# Example:
# Daily return 0.5%, risk-free rate 0.01% (daily), 20-day volatility 1%
# r_t = (0.5% - 0.01%) / 1% = 0.49
```

**Pros**: Encourages pursuing risk-adjusted returns rather than absolute returns.

**Problem**: Reward can be too large when volatility is low; needs smoothing.

---

#### Type 3: Cost-Penalized

```python
# Specifically for high-frequency strategies, penalizes excessive trading
r_t = profit_t - α × |position_t - position_{t-1}| - β × spread_t

# Parameters:
# α = turnover penalty coefficient (typically 0.001-0.01)
# β = spread penalty coefficient (typically 0.5-2.0)
# spread_t = bid-ask spread

# Example:
# Profit $100, position change 500 shares, spread $0.02/share
# α=0.005, β=1.0
# r_t = 100 - 0.005×500 - 1.0×0.02×500 = 100 - 2.5 - 10 = $87.5
```

**Key**: Choice of α and β must reflect real trading costs, otherwise Agent behavior will deviate from reality.

---

#### Type 4: Multi-Objective Composite Reward

```python
# Recommended design for production-grade systems
r_t = w1 × profit_t + w2 × sharpe_t - w3 × drawdown_t - w4 × turnover_t

# Parameters:
# w1 = profit weight (typically 1.0)
# w2 = Sharpe weight (typically 0.1-0.5)
# w3 = drawdown penalty (typically 0.5-2.0)
# w4 = turnover penalty (typically 0.01-0.1)

# Example configuration:
weights = {
    'profit': 1.0,       # Base return
    'sharpe': 0.3,       # Risk adjustment
    'drawdown': 1.5,     # Heavy drawdown penalty
    'turnover': 0.05     # Light turnover penalty
}
```

**Tuning Recommendations**:
1. Train baseline with profit-oriented reward first
2. Gradually add risk penalty terms
3. Adjust weights based on backtest results
4. Determine final weights using validation set

---

#### Common Reward Function Design Errors

| Error | Consequence | Solution |
|-------|-------------|----------|
| Only using returns as reward | Agent goes all-in | Add risk penalty terms |
| Ignoring trading costs | High frequency but loses money live | Add turnover penalty |
| Reward scale too large | Training instability | Normalize to [-1, 1] |
| Reward delay too long | Learning difficulty | Use intermediate rewards (unrealized P&L changes) |
| Reward too complex | Tuning difficulty | Start simple and iterate |

---

#### Practical Reward Function Code

```python
import numpy as np

class TradingReward:
    """Production-grade reward function"""

    def __init__(self, config: dict):
        self.w_profit = config.get('profit_weight', 1.0)
        self.w_sharpe = config.get('sharpe_weight', 0.3)
        self.w_drawdown = config.get('drawdown_weight', 1.5)
        self.w_turnover = config.get('turnover_weight', 0.05)

    def calculate(self, state: dict) -> float:
        """Calculate composite reward"""

        # Profit term
        profit = state['position'] * state['price_change']

        # Sharpe term (instantaneous version)
        if state['volatility'] > 0:
            sharpe = state['return'] / state['volatility']
        else:
            sharpe = 0

        # Drawdown penalty
        drawdown = max(0, state['peak_value'] - state['current_value'])
        drawdown_pct = drawdown / state['peak_value'] if state['peak_value'] > 0 else 0

        # Turnover cost
        turnover = abs(state['position'] - state['prev_position'])
        turnover_cost = turnover * state['transaction_cost']

        # Composite reward
        reward = (
            self.w_profit * profit
            + self.w_sharpe * sharpe
            - self.w_drawdown * drawdown_pct
            - self.w_turnover * turnover_cost
        )

        # Normalize (optional)
        return np.clip(reward, -1.0, 1.0)
```

---

**Simple Reward** (for beginners):
```
R_t = Position × Return - Trading Cost
```

**Risk-Adjusted Reward**:
```
R_t = Return - λ × Risk Penalty
    = Position × Return - λ × |Return - Mean|²
```

**Sharpe-Oriented Reward**:
```
R_t = (Return - Risk-free Rate) / Volatility
```

---

## Common RL Algorithms and Applicability

| Algorithm | Type | Suitable Scenario | Pros/Cons |
|-----------|------|-------------------|-----------|
| DQN | Discrete actions, value function | Simple trading decisions | Easy to implement, limited actions |
| DDPG | Continuous actions, policy gradient | Position optimization | Fine control, hard to train |
| PPO | General, policy gradient | Balanced choice | Stable, medium sample efficiency |
| A2C/A3C | Parallel training | Multi-asset simultaneous training | Faster training, needs resources |
| SAC | Continuous actions, max entropy | Encourage exploration | Prevents early stopping, compute intensive |

**Practical Recommendation**: Start with PPO, it balances stability and performance well.

---

## Special Challenges of Trading RL

### Challenge 1: Low Sample Efficiency

```
Comparison:
- Games: Can generate unlimited training samples
- Trading: Limited historical data

10 years daily data ≈ 2,500 samples
5 years minute data ≈ 500,000 samples

Solutions:
- Use simpler models
- Data augmentation (add noise, time warping)
- Multi-asset parallel training
```

### Challenge 2: Non-Stationary Environment

```
Game rules are fixed, market rules change:
- 2019: Low volatility environment
- 2020: COVID crash
- 2021: Retail frenzy
- 2022: Rate hike cycle

Solutions:
- Rolling training (update model every N days)
- Regime-conditioned training
- Multi-environment training
```

### Challenge 3: Sparse and Delayed Rewards

```
Problem:
- Buy today, know if it's right a week later
- Short-term losses may lead to long-term gains (averaging down)

Solutions:
- Use higher frequency intermediate rewards
- Introduce unrealized P&L changes as immediate feedback
- Adjust discount factor gamma
```

### Challenge 4: Overfitting

```
Symptoms:
- Training set 100%+ annualized
- Test set -20% annualized

Solutions:
- Simplify model structure
- Add regularization
- Use Purged CV validation
- Test across different market cycles
```

---

## Practical Application Examples

### Single Asset Position Management

```
State:
  - Past 20 days return sequence
  - Current position ratio
  - RSI, MACD indicators

Action:
  - Target position in [0, 1] (long only)

Reward:
  - Daily return x Position - Turnover cost

Results:
  - Agent learns to add to positions early in trends
  - Reduces position during high volatility
  - Automatically controls turnover
```

### Multi-Asset Allocation

```
State:
  - N assets' feature vectors
  - Current allocation weights
  - Portfolio volatility

Action:
  - Target weight vector in [0, 1]^N, summing to 1

Reward:
  - Portfolio Sharpe ratio

Results:
  - Agent learns to diversify
  - Dynamically adjusts risk exposure
  - Reduces high-beta assets during crises
```

---

## Multi-Agent Perspective

RL can replace or enhance specific Agents in multi-agent architecture:

```
Approach 1: RL Replaces Signal Agent
  - Signal Agent originally uses rules or supervised model
  - Replace with RL to directly output trading signals
  - Risk Agent still uses rule-based constraints

Approach 2: RL as Meta Agent
  - Signal Agent provides multiple signals
  - RL Meta Agent decides how to combine them
  - Learns weights of different signals across regimes

Approach 3: Multiple RL Agents Collaborate
  - One RL Agent per asset
  - Global Risk Agent coordinates positions
  - Shared experience pool accelerates learning
```

---

## Common Misconceptions

**Misconception 1: RL can completely replace manual rules**

Not realistic. RL needs:
- Large sample training
- Stable reward signals
- Reasonable state design

All of these require domain knowledge. RL is an enhancement tool, not a silver bullet.

**Misconception 2: More complex RL algorithms work better**

Often the opposite in finance. Simple algorithms (DQN, PPO) with good state design usually outperform complex algorithms (SAC, TD3) with crude design.

**Misconception 3: RL trained well can go directly to production**

Dangerous. RL may learn:
- Strategies that overfit historical data
- Behaviors that fail in extreme markets
- High-turnover strategies that can't actually be executed

Must undergo rigorous out-of-sample testing and paper trading.

---

## Practical Recommendations

### 1. Start Simple

```
Starting Configuration:
- Algorithm: PPO
- Actions: Discrete 5 levels
- State: < 20 dimensions
- Single asset
```

### 2. Reward Engineering is More Important Than Model

```
Bad Reward: Only looks at returns
  -> Agent learns to go all-in

Good Reward: Return - Risk penalty - Turnover cost
  -> Agent learns risk management
```

### 3. Validation Process

```
1. Training set: 60%
2. Validation set: 20% (tune hyperparameters)
3. Test set: 20% (final evaluation, use only once)

Baselines to Compare:
- Buy and hold
- Simple momentum strategy
- Supervised learning + rules
```

---

## Summary

| Key Point | Explanation |
|-----------|-------------|
| RL Advantage | End-to-end optimization, automatically considers costs and positions |
| Core Challenges | Few samples, changing environment, overfitting |
| Recommended Start | PPO + discrete actions + simple state |
| Key to Success | Reward design > Algorithm choice > Model structure |
| Multi-Agent Integration | Can replace Signal Agent or serve as Meta Agent |
