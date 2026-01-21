# Background: LLM Research in Quantitative Trading

> Large Language Models (LLMs) are changing the research paradigm in quantitative trading. This document summarizes the main research directions and representative works of LLMs in the quantitative field.

---

## 1. LLM Application Scenarios in Quantitative Trading

| Scenario | Traditional Method | LLM Method | Advantage |
|----------|-------------------|------------|-----------|
| Sentiment Analysis | Rules/Dictionary | Context understanding | More accurate, more flexible |
| News Understanding | NLP Models | Deep semantic understanding | Understands complex causality |
| Research Report Analysis | Manual reading | Automatic extraction | Massive efficiency improvement |
| Strategy Generation | Manual coding | Code generation | Rapid prototyping |
| Market Reasoning | Rule systems | Chain-of-thought reasoning | Multi-step logic |

---

## 2. Representative Research Works

### 2.1 FinGPT (2023)

**Source**: Columbia University + AI4Finance

**Position**: Open-source financial large language model

**Core Features**:
- Fine-tuned on LLaMA
- Trained on financial corpus
- Supports sentiment analysis, Q&A, summarization
- Open-source and available

**Architecture**:
```
Base Model (LLaMA)
    |
Financial Corpus Pre-training (Financial News, SEC Filings)
    |
Task Fine-tuning (Sentiment, QA, Summarization)
    |
FinGPT
```

**Limitations**:
- Mainly for NLP tasks
- Cannot directly generate trading signals
- Insufficient real-time capability

**Link**: https://github.com/AI4Finance-Foundation/FinGPT

---

### 2.2 BloombergGPT (2023)

**Source**: Bloomberg

**Position**: 50-billion parameter finance-specific LLM

**Core Features**:
- Proprietary model, not open-source
- Training data includes Bloomberg's exclusive data
- Outperforms general models on financial NLP tasks

**Training Data**:
| Data Type | Scale |
|-----------|-------|
| Financial Data | 363 billion tokens |
| General Data | 345 billion tokens |

**Performance**: Leads GPT-3 on financial sentiment analysis, NER, and other tasks

**Limitation**: Not open-source, cannot be used directly

---

### 2.3 FLAG-Trader (2024-2025)

**Source**: Academic research

**Position**: LLM + Reinforcement Learning trading system

**Core Innovation**:
- LLM generates trading hypotheses
- RL validates and optimizes
- Feedback loop improves LLM

**Architecture**:
```
Market Data + News
    |
LLM (Generate trading hypotheses)
    |
RL Agent (Execute and evaluate)
    |
Reward Feedback
    |
LLM Improvement (Learn from feedback)
```

**Advantages**:
- Combines LLM's reasoning ability with RL's optimization capability
- Good interpretability
- Can handle text + numerical multimodal data

---

### 2.4 TradingGPT / QuantGPT Series

**Position**: Using GPT-4 and similar models for trading decisions

**Method**:
- Directly use GPT-4 API
- Design specific prompts
- Generate trading recommendations

**Typical Prompt**:
```
You are a quantitative analyst. Based on the following market data:
- BTC 24-hour change: +5.2%
- RSI(14): 72
- Volume: Up 30% from yesterday

Please analyze the current market state and provide trading recommendations.
```

**Limitations**:
- Dependent on API, high latency
- High cost
- Hallucination issues

---

### 2.5 MM-DREX (2024)

**Position**: Multimodal trading system

**Core Innovation**:
- Directly "sees" candlestick charts
- Combines text + images
- Visual pattern recognition

**Input**:
```
Text: News, indicator values
Image: Candlestick charts, technical analysis charts
    |
Multimodal LLM
    |
Trading Decision
```

**Advantage**: Mimics human traders' ability to "read charts"

---

## 3. Main Research Directions

### 3.1 Enhanced Sentiment Analysis

**Traditional Method**:
```python
# Simple dictionary approach
positive_words = ['bullish', 'surge', 'rally']
negative_words = ['bearish', 'crash', 'plunge']
```

**LLM Method**:
```python
# Context understanding
prompt = """
Analyze the market sentiment of the following news (-1 to 1):
"Fed signals potential rate cuts, but warns inflation remains sticky"

Please consider:
1. Direct impact
2. Implied expectations
3. Possible market reaction
"""
```

**Advantage**: Understands complex, contradictory information

---

### 3.2 Factor Discovery and Hypothesis Generation

**Traditional Method**: Manual factor design

**LLM Method**:
```python
prompt = """
Based on the following market patterns, propose potentially effective quantitative factors:
1. Small-cap stocks outperform large-cap over the long term
2. High momentum stocks show continuation
3. Low volatility stocks have better risk-adjusted returns

Please propose 3 new factor hypotheses, including:
- Factor definition
- Theoretical basis
- Potential risks
"""
```

**Advantage**: Rapidly generate many hypotheses for testing

---

### 3.3 Code Generation

**Applications**:
- Strategy code generation
- Data processing scripts
- Visualization code

**Example**:
```python
prompt = """
Implement a dual moving average strategy in Python:
- Short-term MA: 10 days
- Long-term MA: 30 days
- Golden cross = buy, death cross = sell
- Use pandas for data processing
"""
```

**Note**: Generated code requires human review

---

### 3.4 Market Reasoning

**Application**: Multi-step logical reasoning

**Example**:
```
Question: If the Fed raises rates, what's the impact on tech stocks?

LLM Reasoning Chain:
1. Rate hike -> Interest rates rise
2. Rising rates -> Higher discount rate
3. Higher discount rate -> Future cash flow present value decreases
4. Tech stocks depend on future growth -> Valuations more affected
5. Conclusion: Tech stocks may fall, especially high-valuation growth stocks
```

---

## 4. Current Limitations

### 4.1 Hallucination Problem

LLMs may generate plausible-sounding but incorrect analysis:
- Fabricated data
- Incorrect causal relationships
- Overconfident predictions

**Coping Strategy**: Always verify with real data

### 4.2 Insufficient Real-Time Capability

- API call latency: 100-500ms
- Model inference time: 1-10 seconds
- Not suitable for high-frequency trading

### 4.3 Cost Issues

| Model | Cost | Suitable For |
|-------|------|--------------|
| GPT-4 | $30-60/1M tokens | Research |
| GPT-3.5 | $0.5-2/1M tokens | Production |
| Open-source models | Compute cost | Self-hosted |

### 4.4 Cannot Predict Prices

**LLM Can Do**:
- Understand news
- Analyze sentiment
- Generate hypotheses

**LLM Cannot Do**:
- Predict tomorrow's price
- Guarantee profits
- Replace traditional quantitative models

---

## 5. Practical Recommendations

### 5.1 Tasks Suitable for LLM

| Task | Recommendation Level | Notes |
|------|----------------------|-------|
| News Sentiment Analysis | High | LLM's core strength |
| Research Report Summary | High | Significant efficiency improvement |
| Strategy Code Generation | Medium | Requires human review |
| Factor Hypothesis Generation | Medium | Requires backtest validation |
| Direct Trading Decisions | Low | High risk, not recommended |

### 5.2 Recommended Architecture

```
Traditional Quantitative Model (Primary)
    ^
LLM Enhancement Layer (Auxiliary)
- Sentiment signals
- News filtering
- Hypothesis generation
    ^
Raw Data
```

### 5.3 Open-Source Options

| Model | Size | Suitable For |
|-------|------|--------------|
| LLaMA 3 | 8B-70B | General purpose |
| FinGPT | 7B | Financial NLP |
| Mistral | 7B | Lightweight deployment |
| Qwen | 7B-72B | Chinese support |

---

## 6. Future Trends

1. **Multimodal Fusion**: Text + Numerical + Image
2. **Agent-ification**: LLM as the "brain" of trading Agents
3. **Real-time Processing**: Lower latency inference
4. **Specialization**: More finance-specific models
5. **Compliance**: Meeting regulatory explainability requirements

---

> **Core Principle**: LLMs are powerful tools, but not magic. They can enhance your analytical capabilities but cannot replace solid quantitative foundations and strict risk control.
