# Multi-agent evaluation

Five agents score a submission with **separate LLM calls**, then a weighted total is written to CSV.

| Agent | Default weight | Role |
| --- | --- | --- |
| correctness | 0.35 | Does the code solve the problem and match I/O? |
| edge_cases | 0.20 | Boundaries, overflow, precision, invalid ops |
| counteragent | 0.20 | Adversarial review; hunt bugs and hardcoded answers |
| complexity | 0.15 | Algorithm vs constraints |
| style | 0.10 | Naming, comments, formatting |

Weights live in `config/weights.yaml` and are normalized if they do not sum to 1.

## Setup

```bash
pip install -r requirements.txt
copy .env.example .env
```

Put API keys in `.env`. Named judge models live in `config/models.yaml`.

## Test specific LLMs

List presets:

```bash
python -m evalsys presets
```

Ping one API (one cheap request):

```bash
python -m evalsys ping --preset groq
python -m evalsys ping --preset gemini
python -m evalsys ping --preset nvidia-gemma-4
python -m evalsys ping --preset moonshot-kimi-k3
python -m evalsys ping --preset cerebras
```

Run the five agents on dataset rows using the same judge model:

```bash
python -m evalsys batch --preset gemini --limit 3
python -m evalsys batch --preset groq --limit 3
python -m evalsys batch --preset nvidia-deepseek --limit 3
python -m evalsys batch --preset nvidia-laguna-xs --limit 3
python -m evalsys batch --preset nvidia-gemma-4 --limit 3
python -m evalsys batch --preset moonshot-kimi-k3 --limit 3
python -m evalsys batch --preset cerebras --limit 3
```

Sweep the default suite (Gemini, Groq, NVIDIA DeepSeek / Laguna XS / Gemma 4, Kimi K3, Cerebras):

```bash
python -m evalsys batch --sweep --limit 2
```

CSV rows include `preset`, `judge_model`, and `judge_provider` so you can compare models.

Kimi K3 on OpenRouter instead of Moonshot direct: `--preset openrouter-kimi-k3`.

Edit model IDs in `config/models.yaml` if a host renames them.

## Run

Mock pass (no keys, useful to verify CSV wiring):

```bash
python -m evalsys batch --sheet all_samples --limit 2 --mock
```

Live evaluation against the workbook:

```bash
python -m evalsys batch --sheet all_samples --limit 5
```

Single file:

```bash
python -m evalsys file path\to\submission.cpp --question "problem statement here"
```

Rows are **appended** to `results/evaluations.csv`. Full agent JSON (including raw model text) is also appended to `results/evaluations.jsonl`.

CSV includes agent scores, agent confidences, total score, overall confidence, predicted grade, and ground truth when present.

## Swap keys and models

`config/llm.yaml` controls provider, model, temperature, and which env var holds the key. Per-agent overrides are supported:

```yaml
agents:
  correctness:
    model: gpt-4o
  style:
    provider: groq
    model: llama-3.3-70b-versatile
    api_key_env: GROQ_API_KEY
    base_url_env: GROQ_BASE_URL
    base_url: https://api.groq.com/openai/v1
  counteragent:
    provider: anthropic
    model: claude-sonnet-4-20250514
    api_key_env: ANTHROPIC_API_KEY
  edge_cases:
    provider: gemini
    model: gemini-2.0-flash
    api_key_env: GEMINI_API_KEY
```

`openai_compat` works with OpenAI, Groq, OpenRouter, Together, DeepSeek, and Ollama. Set `base_url` / `base_url_env` accordingly.

Prompts are in `evalsys/prompts.py`.
