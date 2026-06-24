# AI Reliability Platform

A unified platform for evaluating LLM outputs against golden datasets and tracing AI pipeline failures.

## Why This Exists

Most AI teams build LLM features but have no systematic way to:
1. **Test** if their AI produces correct outputs
2. **Debug** when things go wrong in multi-step pipelines

This platform combines evaluation and observability into one tool.

## Features

- **Eval Runner** — Test LLM outputs against golden datasets with scoring
- **Trace Collector** — Visualize and debug LLM call traces
- **Multi-Provider** — Compare Groq, Mistral, Gemini, OpenRouter
- **Dashboard** — See everything at a glance

## Quick Start

```bash
# Clone
git clone https://github.com/yourusername/ai-reliability-platform
cd ai-reliability-platform

# Setup
cp .env.example .env  # Add your API keys
pip install -r requirements.txt

# Seed demo data
python scripts/seed_data.py

# Start
uvicorn app.main:app --reload

# Open http://localhost:8000
```

## API Documentation

Once running, visit http://localhost:8000/docs for interactive API docs.

## Architecture

```
Frontend (Jinja2) → FastAPI Backend → SQLite
                         ↓
              Provider Router (Groq/Mistral/Gemini/OpenRouter)
```

## Tech Stack

- **Python + FastAPI** — Async LLM calls, auto-generated docs
- **SQLite + SQLModel** — Zero setup, easy demo
- **Jinja2 Templates** — No frontend build step
- **sentence-transformers** — Local semantic similarity scoring

## What I Learned

- [Add your key insights here]

## License

MIT
