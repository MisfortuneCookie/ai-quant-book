# Part 3: Machine Learning

> Goal for this stage: **From Models to Agents.** Understand the proper use of machine learning in quantitative trading, and how to evolve from "prediction models" to "decision-making Agents."

---

## Lesson List

| Lesson | Topic | Deliverables |
|--------|-------|--------------|
| [Lesson 09](Lesson-09-Supervised-Learning-in-Quantitative-Trading.md) | Supervised Learning in Quantitative Trading | ML Strategy Framework, IC/IR Evaluation |
| [Lesson 10](Lesson-10-From-Models-to-Agents.md) | From Models to Agents | Single Agent Strategy Module |

---

## Background Knowledge

| Document | Description | Suggested Reading Time |
|----------|-------------|------------------------|
| [LLM Research in Quantitative Trading](Background/LLM-Research-in-Quantitative-Trading.md) | Latest applications of large language models in quantitative trading | 15 min |
| [Triple Barrier Labeling Method](Background/Triple-Barrier-Labeling-Method.md) | Defining ML labels using take-profit and stop-loss | 15 min |
| [Time Series Cross-Validation (Purged CV)](Background/Time-Series-Cross-Validation-Purged-CV.md) | Preventing information leakage in time series | 15 min |
| [Reinforcement Learning in Trading](Background/Reinforcement-Learning-in-Trading.md) | Combining RL algorithms with trading | 20 min |
| [Alternative Data (NLP and Satellite)](Background/Alternative-Data-NLP-and-Satellite.md) | Non-traditional data sources: text sentiment, satellite imagery, etc. | 15 min |
| [Meta-Labeling Method](Background/Meta-Labeling-Method.md) | Secondary model for predicting signal reliability | 15 min |
| [Feature Engineering Common Pitfalls](Background/Feature-Engineering-Common-Pitfalls.md) | Future leakage, overfitting, and other common mistakes | 10 min |
| [Limitations of ML in Finance](Background/Limitations-of-ML-in-Finance.md) | Core challenges: low signal-to-noise ratio, distribution drift, etc. | 15 min |
| [Model Architecture Selection Guide](Background/Model-Architecture-Selection-Guide.md) | LSTM/GRU/Transformer/CNN comparison, RL algorithm selection | 20 min |
| [Model Drift and Retraining](Background/Model-Drift-and-Retraining.md) | K-S/CUSUM drift detection, retraining triggers and strategies | 20 min |
| [MLOps Infrastructure](Background/MLOps-Infrastructure.md) | Feature Store, Model Registry, Drift Monitor | 30 min |
| [Frontier ML and RL Methods (2025)](Background/Frontier-ML-and-RL-Methods-2025.md) | SOTA techniques: Decision Transformer, AlphaAgent, GNN, Diffusion Models | 30 min |

---

## After Completing This Stage

You will be able to:
- Understand why "predicting price" is the wrong objective
- Design proper labels (Triple Barrier Method)
- Understand the core components of an Agent: State, Action, Reward, Environment
- Build a single Agent strategy that generates trading signals

---

## Next Stage

→ [Part 4: Multi-Agent Systems](../Part4-Multi-Agent/README.md) - Building collaborative Agent systems
