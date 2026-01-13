# AI Quantitative Trading: From Zero to One | AI量化交易从0到1

> **Build production-ready quantitative trading systems with multi-agent architecture**
>
> **用多智能体架构构建可落地的量化交易系统**

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

**Author 作者**: [Wayland Zhang](https://waylandz.com)

---

## Language Versions | 语言版本

| Language | Status | Progress | Link |
|----------|--------|----------|------|
| 🇨🇳 中文 | ✅ Complete | 22 lessons + 30 background + 4 appendices | [阅读中文版 →](./manuscript/cn/) |
| 🇺🇸 English | ✅ Complete | 22 lessons + 30 background + 4 appendices | [Read English →](./manuscript/en/) |

---

<table>
<tr>
<td width="50%" valign="top">

## What's This Book About?

**Not a strategy holy grail, but teaching you to build production-ready quant systems.**

Most quant tutorials stop at:
- API translations of backtesting frameworks
- Stacking technical indicators and parameter optimization
- Showcasing overfitted "magic strategies"

These let you "play" quant, not "do" quant.

Real quant systems need to answer:
- **Where does data come from?** Rate limits, missing values, adjustments, timezones
- **How not to fool yourself in backtests?** Lookahead bias, overfitting, transaction costs
- **Why isn't one model enough?** Regime changes, signal conflicts, risk diversification
- **How to control risk?** Stop loss, position sizing, factor exposure, circuit breakers
- **How to go to production?** Execution slippage, monitoring, disaster recovery

This book uses **multi-agent architecture** to answer these questions: different agents handle different responsibilities (signals, risk, execution), collaborating to make trading decisions.

</td>
<td width="50%" valign="top">

## 这本书讲什么？

**不是策略圣杯，而是教你构建可落地的量化交易系统。**

市面上的量化教程大多停留在：
- 某个回测框架的 API 翻译
- 技术指标的堆砌和参数优化
- 过拟合的"神奇策略"展示

这些只能让你"玩"量化，不能让你"做"量化。

真正的量化系统需要回答：
- **数据从哪来？** API 限流、缺失值、复权、时区
- **回测怎么不骗自己？** 未来函数、过拟合、交易成本
- **为什么单一模型不够？** Regime 变化、信号冲突、风险分散
- **如何控制风险？** 止损、仓位、因子暴露、熔断机制
- **怎么上生产？** 执行滑点、系统监控、故障恢复

这本书用**多智能体架构**回答这些问题：不同的 Agent 负责不同的职责（信号、风控、执行），协作完成交易决策。

</td>
</tr>
</table>

---

## Content Overview | 内容概览

The book has **5 parts, 22 lessons** | 全书分为 **5 个部分、22 课**：

| Part | Topic 主题 | Lessons 课程 | Core Content 核心内容 |
|------|------------|--------------|----------------------|
| **1** | Quick Start 快速体验 | 1 | Quant landscape, multi-agent intuition 量化全景图、多智能体直觉 |
| **2** | Fundamentals 量化基础 | 7 | Markets, statistics, strategies, data, backtesting 市场、统计、策略、数据、回测 |
| **3** | Machine Learning 机器学习 | 2 | Supervised learning, from models to agents 监督学习、从模型到Agent |
| **4** | Multi-Agent 多智能体 | 7 | Architecture, regime detection, LLM, risk control 架构、Regime、LLM、风控 |
| **5** | Production 生产实战 | 5 | Costs, execution, operations, projects 成本、执行、运维、实战 |

Plus **30 background articles** and **4 appendices** | 另有 **30篇背景知识** 和 **4篇附录**

---

## Core Architecture | 核心架构

![Multi-Agent Architecture](manuscript/en/Part1-Quick-Start/assets/multi-agent-architecture.svg)

---

## Target Readers | 目标读者

<table>
<tr>
<td width="50%" valign="top">

| Reader Type | What You'll Get |
|-------------|-----------------|
| **Developers → Quant** | Complete path to build trading systems |
| **Quant Researchers** | Multi-agent architecture, production risk control |
| **Investors/PMs** | Understand quant system capabilities and risks |

**Prerequisites:**
- **Required**: Basic programming (Python)
- **Helpful**: Statistics, financial markets
- **Not needed**: ML/DL background

</td>
<td width="50%" valign="top">

| 读者类型 | 你会获得什么 |
|----------|-------------|
| **程序员想转量化** | 从零构建交易系统的完整路径 |
| **量化研究员** | 多智能体架构、生产级风控设计 |
| **投资者/PM** | 理解量化系统的能力边界和风险 |

**前置要求：**
- **必需**：基本编程概念（Python）
- **有帮助**：统计学基础、金融市场常识
- **不需要**：机器学习或深度学习背景

</td>
</tr>
</table>

---

## Repository Structure | 目录结构

```
ai-quant-book/
├── manuscript/
│   ├── cn/                      # 中文版 (Complete 已完成)
│   │   ├── Part1-快速体验/
│   │   ├── Part2-量化基础/
│   │   ├── Part3-机器学习/
│   │   ├── Part4-多智能体/
│   │   ├── Part5-生产与实战/
│   │   └── Resources & Links/
│   ├── en/                      # English (Complete)
│   │   ├── Part1-Quick-Start/
│   │   ├── Part2-Quant-Fundamentals/
│   │   ├── Part3-Machine-Learning/
│   │   ├── Part4-Multi-Agent/
│   │   ├── Part5-Production/
│   │   └── Resources-Links/
└── README.md
```

---

## Quick Start | 快速开始

<table>
<tr>
<td width="50%" valign="top">

### English Readers

→ [**Read English Version**](./manuscript/en/)

**Recommended Path:**
| Reader | Path |
|--------|------|
| Complete beginner | Part 1 → All Part 2 → Part 3-5 |
| With coding background | Lesson 01 → Skim Part 2 → Parts 3-5 |
| With quant background | Lesson 01 → Lesson 08 → Parts 3-5 |

</td>
<td width="50%" valign="top">

### 中文读者

→ [**阅读中文版**](./manuscript/cn/)

**推荐路径：**
| 读者类型 | 路径 |
|---------|------|
| 零基础入门 | Part 1 → Part 2 全部 → Part 3-5 |
| 有编程基础 | 第01课 → Part 2 快速浏览 → Part 3-5 |
| 有量化基础 | 第01课 → 第08课 → Part 3-5 |

</td>
</tr>
</table>

---

## Risk Disclaimer | 风险声明

> **Quantitative trading involves risk. Invest carefully.**
>
> **量化交易有风险，投资需谨慎。**

<table>
<tr>
<td width="50%" valign="top">

This book is educational and does not constitute investment advice.

- Strategies are for learning only, no profit guaranteed
- Fully understand risks before live trading
- Never trade with money you can't afford to lose
- Past performance ≠ future results

</td>
<td width="50%" valign="top">

本书是教育性质，不构成投资建议。

- 书中策略仅供学习，不保证盈利
- 实盘交易前请充分理解风险
- 永远不要用你输不起的钱交易
- 历史表现不代表未来收益

</td>
</tr>
</table>

---

## License | 许可证

This work is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

本书内容采用 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) 协议。

---

> *"Whatever happens in the stock market today has happened before and will happen again."*
> — Jesse Livermore

> *"We search through historical data looking for anomalous patterns that we believe are predictive of future price action."*
> — Jim Simons
