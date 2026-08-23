<div align="center">

# ⚖️ AI Debate Arena

**Two AI agents debate any topic. A third AI judges the winner.**

![Python](https://img.shields.io/badge/Python-3.14-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-🦜🔗-1C3C3C?style=for-the-badge)
![Groq](https://img.shields.io/badge/Groq-F55036?style=for-the-badge&logo=data:image/svg+xml;base64,&logoColor=white)
![Gemini](https://img.shields.io/badge/Gemini-8E75B2?style=for-the-badge&logo=googlegemini&logoColor=white)
![OpenRouter](https://img.shields.io/badge/OpenRouter-6467F2?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

</div>

---

## 🧠 What is this?

**AI Debate Arena** is a **multi-agent orchestration** system built with **LangChain**, where two AI debaters — one arguing **FOR** a topic, one arguing **AGAINST** it — go head-to-head for several rounds, directly rebutting each other's points. Once the debate ends, a neutral **Judge** agent reads the full transcript and delivers a verdict on who argued better.

This project follows the **debate pattern** of multi-agent orchestration — a technique where multiple LLM agents argue opposing sides and a supervisor (the judge) evaluates the outcome, often producing more rigorous reasoning than a single model working alone.

Each agent runs on a **different LLM provider**, so no single model is judging (or debating against) itself:

| Role | Provider | Model |
|------|----------|-------|
| 🟠 Debater A (FOR) | Groq | `openai/gpt-oss-20b` |
| 🔵 Debater B (AGAINST) | Google Gemini | `gemini-3.6-flash` |
| ⚪ Judge | OpenRouter | `nvidia/nemotron-3-super-120b-a12b:free` |

---

## ✨ Features

- 🥊 **Multi-round debate loop** — agents build on and rebut each other's previous arguments, not just repeat talking points
- 🌐 **Multi-agent orchestration** — Groq, Gemini, and OpenRouter agents coordinated through LangChain's unified chat model interface
- ⚖️ **Independent AI judge** — evaluates argument quality, evidence, and rebuttals rather than tone or confidence
- 📝 **Persistent shared transcript** — full debate history passed to every agent call so responses stay contextually grounded
- 🎯 **Stance-locked prompting** — agents are instructed to stay fully committed to their side, avoiding wishy-washy concessions
- 🆓 **100% free-tier APIs** — no paid API keys required to run this project

---

## 🏗️ How it works

```
                    ┌──────────────────┐
                    │   Debate Topic   │
                    └────────┬─────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
      ┌────────────────┐              ┌────────────────┐
      │  Debater A     │◄────────────►│  Debater B     │
      │  (FOR)         │  N rounds    │  (AGAINST)     │
      │  Groq          │  alternating │  Gemini        │
      └───────┬────────┘              └────────┬───────┘
              │                                │
              └───────────────┬────────────────┘
                              ▼
                     ┌────────────────────┐
                     │   Full Transcript  │
                     └─────────┬──────────┘
                               ▼
                     ┌─────────────────────┐
                     │   Judge (OpenRouter)│
                     │   → Final Verdict   │
                     └─────────────────────┘
```

Each round:
1. Debater A reads the full transcript so far → generates a rebuttal → appends to transcript
2. Debater B reads the *updated* transcript → generates a rebuttal → appends to transcript
3. Repeat for `N` rounds
4. Judge reads the complete transcript → declares a winner with reasoning

---

## 📂 Project Structure

```
ai-debate-arena/
├── main.py          # Entry point — models, prompts, debate loop, judge
├── .env             # API keys (not committed)
├── .gitignore
├── pyproject.toml   # uv-managed dependencies
└── README.md
```

---

## ⚙️ Setup

### 1. Clone & install dependencies

```bash
git clone <your-repo-url>
cd ai-debate-arena
uv sync
```

Or install manually:

```bash
uv add langchain langchain-groq langchain-google-genai langchain-openai python-dotenv
```

### 2. Get your free API keys

| Provider | Get key from |
|----------|--------------|
| Groq | [console.groq.com](https://console.groq.com) |
| Google Gemini | [aistudio.google.com](https://aistudio.google.com) |
| OpenRouter | [openrouter.ai](https://openrouter.ai) |

### 3. Add keys to `.env`

```env
GROQ_API_KEY=your_groq_key_here
GOOGLE_API_KEY=your_google_key_here
OPENROUTER_API_KEY=your_openrouter_key_here
```

### 4. Run it

```bash
uv run main.py
```

---

## 🖥️ Sample Output

```
Speaker A (FOR): Social media amplifies misinformation, fuels polarization, and
erodes mental well-being. Studies link excessive use to anxiety, depression,
and decreased attention spans...

Speaker B (AGAINST): Blaming social media for polarization ignores how it
democratizes information and empowers marginalized voices globally...

...

Judge's Verdict:
Winner: Speaker B (AGAINST)
Reason: Speaker B presented stronger, more concrete evidence and directly
rebutted Speaker A's points with specific data and examples...
```

---

## 🧩 Built With

- [LangChain](https://www.langchain.com/) — LLM orchestration & prompt templates
- [Groq](https://groq.com/) — ultra-fast inference
- [Google Gemini](https://ai.google.dev/) — multimodal reasoning model
- [OpenRouter](https://openrouter.ai/) — unified access to free open models
- [uv](https://docs.astral.sh/uv/) — Python package management

---

## 🚀 Ideas for Future Improvements

- [ ] Structured judge output (Pydantic schema for `winner` + `reason`)
- [ ] CLI input for custom topics & round count
- [ ] Streaming responses (token-by-token live output)
- [ ] Modular project structure (`src/debate_arena/`)
- [ ] Web UI (Streamlit / Gradio front-end)

---

<div align="center">

**Made with 🧠 + LangChain**

</div>