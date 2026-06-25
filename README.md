# pytest-llm

LLM-powered semantic assertions for pytest. Add AI-driven quality checks to any test suite with zero infrastructure.

## Install

```bash
pip install pytest-llm
```

## Quick Start

```python
from pytest_llm import assert_faithful, assert_safe, assert_tone

def test_llm_output():
    output = "Python was created by Guido van Rossum in 1991."
    source = "Guido van Rossum created Python, released in 1991."

    assert_faithful(output, source)        # factual accuracy
    assert_tone(output, "professional")     # tone check
    assert_safe(output)                     # safety check
```

## Assertions

| Assertion | Description |
|-----------|-------------|
| `assert_faithful` | Every factual claim in output is supported by source |
| `assert_no_hallucination` | Output contains no invented facts not in source |
| `assert_tone` | Output matches an expected tone (freeform string) |
| `assert_semantic_similarity` | Cosine similarity between output and expected text (no LLM) |
| `assert_contains_claim` | Output semantically contains a given claim |
| `assert_safe` | Output contains no harmful or offensive content |
| `assert_language` | Output is written in the expected language |
| `assert_regression` | Output is not worse than a baseline (similarity + quality) |

## Configuration

### Environment Variables

```bash
export LLM_JUDGE_PROVIDER=openai       # or anthropic, groq, ollama
export LLM_JUDGE_MODEL=gpt-4o-mini     # optional, defaults to provider best
export OPENAI_API_KEY=sk-...           # set for your chosen provider
```

### conftest.py

```python
from pytest_llm import pytest_configure_judge

pytest_configure_judge(provider="anthropic", model="claude-haiku-4-5-20251001")
```

### CLI Options

```bash
pytest --llm-judge-provider=anthropic --llm-judge-model=claude-haiku-4-5-20251001
pytest --llm-report   # print Rich summary table after tests
```

## Provider Support

| Provider | Default Model | Env Var for API Key |
|----------|---------------|---------------------|
| OpenAI | gpt-4o-mini | `OPENAI_API_KEY` |
| Anthropic | claude-haiku-4-5-20251001 | `ANTHROPIC_API_KEY` |
| Groq | llama-3.3-70b-versatile | `GROQ_API_KEY` |
| Ollama | llama3 | (local, no key needed) |

## CI/CD with GitHub Actions

```yaml
- name: Run tests with LLM assertions
  env:
    LLM_JUDGE_PROVIDER: openai
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
  run: |
    pip install pytest-llm
    pytest --llm-report
```
