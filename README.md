# AI-Driven-Financial-News-Intelligence-System

<div align="center">

*Decomposed market analysis across parallel specialized agents — unified into a single explainable trading decision.*

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-Orchestration-1C3C3C?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)

</div>

---

## Description

This project implements a **multi-agent trading intelligence system** that analyzes financial markets using specialized agents and combines their reasoning into a unified, explainable trading decision.

Instead of relying on a single model or a single source of information, the system decomposes the problem into multiple **signal-specific agents**, where each agent is responsible for a different dimension of market analysis. The current system includes:

- a **Technical Analyst** for price- and momentum-based signals
- a **Fundamental Analyst** for company financial health and long-term valuation signals
- a **News + Sentiment Analyst** for external information flow using recency-weighted news and sentiment analysis

These agents run in parallel and feed into a **Chain-of-Thought (CoT) Synthesis Agent**, which aggregates their outputs, identifies agreement or conflict across signals, and produces a final interpretable market view.

The broader target architecture also includes memory, debate, risk management, trader decisioning, and reflection layers for building a more robust and adaptive trading pipeline.


---

## Input / Output

### Input

The system accepts any valid stock ticker symbol as input.
```bash
python main.py AAPL
python main.py TSLA
python main.py NVDA
```

Any publicly traded ticker supported by yfinance and Alpha Vantage will work.

### Output

The system produces a structured synthesis report covering technical, fundamental, and news + sentiment signals, with a final market bias, confidence level, and key risks.

---

## Architecture

### Architecture Diagram

### High-Level Flow

```
Input Ticker
│
▼
Memory Agent
(injects past decision history into analyst prompts)
│
▼
┌─────────────────────────────────────────────────────────┐
│               Parallel Analyst Layer                    │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │  Technical   │  │ Fundamental  │  │    News +    │   │
│  │   Analyst    │  │   Analyst    │  │  Sentiment   │   │
│  │  [yfinance]  │  │ [SEC EDGAR]  │  │[AlphaVantage]│   │
│  └──────────────┘  └──────────────┘  └──────────────┘   │
└─────────────────────────────────────────────────────────┘
│
▼
CoT Synthesis Node
(aggregates reports, identifies agreement and conflict,
produces a provisional bias)
│
▼
┌─────────────────────────────────────────────────────────┐
│                    Debate Layer                         │
│                                                         │
│  ┌────────────┐  ←debate rounds→  ┌────────────┐        │
│  │    Bull    │                   │    Bear    │        │
│  │ Researcher │                   │ Researcher │        │
│  └────────────┘                   └────────────┘        │
│                        ↓ (after debate ends)            │
│  ┌──────────────────────────────────────────────────┐   │
│  │                Neutral Arbiter                   │   │
│  │  (reads final bull + bear output, scores         │   │
│  │  argument quality, flags contradictions)         │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
│
▼
Trader Agent
(sees bull + bear arguments and arbiter verdict,
outputs rating + conviction score 1–5)
│
▼
Risk Manager
(market regime check, volatility flag,
may adjust or override decision)
│
▼
Final Decision
(Buy / Overweight / Hold / Underweight / Sell)
│
▼
Reflection + Memory Update
(stores decision to SQLite, fires at T+5 trading days
to check outcome, writes post-mortem back to memory DB)
```

---

## ✅ Current Implementation

### 📈 Technical Analyst
- Fetches historical market data
- Computes indicators such as RSI, MACD, Bollinger Bands, VWAP, and volume-based signals
- Generates a technical analysis report using an LLM

### 📊 Fundamental Analyst
- Fetches structured company data from EDGAR
- Analyzes revenue, earnings, assets, liabilities, and equity trends
- Generates a fundamental analysis report using an LLM

### 📰 News + Sentiment Analyst
- Uses Alpha Vantage news sentiment data
- Applies recency weighting so newer news has more impact
- Aggregates sentiment across articles
- Generates a combined external-signal report using an LLM

### 🧠 CoT Synthesis Agent
- Combines outputs from all currently implemented analyst agents
- Produces a final synthesis across technical, fundamental, and news/sentiment signals

### ⚡ Parallel Execution with LangGraph
- Analyst agents are executed concurrently
- Outputs are routed into the synthesis layer

---

## 🚀 Setup and Installation

### Prerequisites

- **Python 3.10 or above**

### Recommended Dependencies

| Package | Purpose |
|:--------|:--------|
| `langgraph` | Agent graph orchestration |
| `langchain` | LLM abstractions |
| `langchain-google-genai` or `langchain-openai` | LLM provider |
| `yfinance` | Market price data |
| `pandas` | Data processing |
| `numpy` | Numerical computation |
| `requests` | HTTP calls |
| `python-dotenv` | Environment variable management |

### Installation

**1. Clone the repository**
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

**2. Create and activate a virtual environment**
```bash
python -m venv env

# macOS / Linux
source env/bin/activate

# Windows
env\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

### Environment Setup

Create a `.env` file in the project root:

```env
LLM_PROVIDER=google
LLM_MODEL=gemini-2.5-flash
GOOGLE_API_KEY=your_google_api_key
ALPHAVANTAGE_API_KEY=your_alpha_vantage_api_key
```
> *Note:* You need an [Alpha Vantage API key](https://www.alphavantage.co/support/#api-key) for the news/sentiment analyst. The system supports multiple LLM providers — use whichever you prefer:
> 
>| Provider | LLM_PROVIDER value | Key to add |
> |:---------|:---------------------|:-----------|
> | Google Gemini (default) | google | GOOGLE_API_KEY from [Google AI Studio](https://aistudio.google.com/) |
> | OpenAI | openai | OPENAI_API_KEY from [OpenAI](https://platform.openai.com/api-keys) |
> | Anthropic | anthropic | ANTHROPIC_API_KEY from [Anthropic Console](https://console.anthropic.com/) |

### ▶️ Running the Project

Run the system with a ticker symbol:

```bash
python main.py AAPL
```
Replace AAPL with any valid ticker symbol (e.g., TSLA, MSFT, NVDA, GOOG)

---

## 🔮 Future Implementations

<details>
<summary><b>Memory Agent</b></summary>

- Retrieves past decisions and outcomes for the same ticker
- Injects relevant historical context into the pipeline

</details>

<details>
<summary><b>Debate Layer</b></summary>

- Bull Researcher
- Bear Researcher
- Neutral Arbiter to score and resolve conflicting views

</details>

<details>
<summary><b>Trader Agent</b></summary>

- Converts synthesized reasoning into a final rating such as Buy, Hold, or Sell
- Assigns a conviction score

</details>

<details>
<summary><b>Risk Manager</b></summary>

- Evaluates market regime and volatility conditions
- Adjusts final recommendations based on risk context

</details>

<details>
<summary><b>Reflection Agent</b></summary>

- Evaluates post-decision outcomes after a fixed horizon
- Generates post-mortem analysis
- Updates long-term memory for future runs

</details>

<details>
<summary><b>Persistence and API Layer</b></summary>

- Decisions database
- History endpoint
- Scheduler for multi-ticker watchlists
- Notification and monitoring support

</details>
