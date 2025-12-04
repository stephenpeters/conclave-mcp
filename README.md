# Conclave MCP

An MCP (Model Context Protocol) server that provides access to a "conclave" of LLM models, enabling any MCP-compatible client to consult multiple frontier models for diverse opinions, peer-ranked evaluations, and synthesized answers.

## Why This Exists

When working with an AI assistant, you're getting one model's perspective. Sometimes that's exactly what you need. But for important decisions—technical architecture, business strategy, creative direction, complex analysis, or any situation where blind spots matter—a plurality of opinions surfaces alternatives you might miss.

**Conclave brings democratic AI consensus to any workflow.**

Instead of manually querying multiple AI services, you can consult the conclave through Claude Desktop, Claude Code, or any MCP client. Get ranked opinions from multiple frontier models (GPT, Claude, Gemini, Grok, DeepSeek) and receive a synthesized answer representing collective AI wisdom.

**Use cases include:**
- **Technical**: Architecture decisions, code review, debugging, API design
- **Business**: Strategy analysis, proposal review, market research synthesis
- **Creative**: Writing feedback, brainstorming, editorial perspectives
- **Research**: Literature review, fact-checking, multi-perspective analysis
- **Decision-making**: Pros/cons analysis, risk assessment, option evaluation

> Inspired by [Andrej Karpathy's llm-council](https://github.com/karpathy/llm-council) concept. This project reimplements the core ideas as an MCP server for seamless integration with AI-assisted workflows.

## How It Works

The conclave operates in up to 3 stages:

```
┌─────────────────────────────────────────────────────────────────┐
│  Stage 1: OPINIONS                                              │
│  Query multiple LLMs in parallel for independent responses      │
│  (GPT, Claude, Gemini, Grok, DeepSeek, etc.)                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Stage 2: PEER RANKING                                          │
│  Each model anonymously evaluates and ranks all responses       │
│  Aggregate scores reveal best performers (lower = better)       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Stage 3: SYNTHESIS                                             │
│  Chairman model synthesizes final answer from collective wisdom │
│  Consensus level reported (strong/moderate/weak/split)          │
│  Tiebreaker vote cast if conclave is split                      │
└─────────────────────────────────────────────────────────────────┘
```

## Features

- **Tiered queries**: Choose cost/depth tradeoff (quick | ranked | full)
- **Three council tiers**: Premium (frontier), Standard (balanced), Budget (fast/cheap)
- **Consensus protocol**: Detects agreement level, triggers tiebreaker on splits
- **Odd conclave size**: Ensures tiebreaker votes can break deadlocks
- **Rotating chairmanship**: Weekly rotation prevents single-model bias
- **Chairman presets**: Context-aware chairman selection (code, creative, reasoning)
- **Cost estimation**: Know what you'll spend before querying
- **Eval-light**: Standalone benchmark runner for tracking performance over time

## Installation

### Prerequisites

1. Get an OpenRouter API key from https://openrouter.ai/keys
2. Add credits to your OpenRouter account (pay-as-you-go)

### Setup

```bash
# Clone the repository
git clone https://github.com/stephenpeters/conclave-mcp.git
cd conclave-mcp

# Create .env file with your API key
echo "OPENROUTER_API_KEY=sk-or-v1-your-key-here" > .env

# Install dependencies
uv sync
```

### Configure Claude Desktop

#### Option 1: Desktop Extensions (Recommended)

1. Open Claude Desktop
2. Go to **Settings > Extensions > Advanced settings > Install Extension...**
3. Navigate to the `conclave-mcp` directory
4. Follow prompts to configure your `OPENROUTER_API_KEY`
5. Restart Claude Desktop

#### Option 2: Manual Config

Open Claude Desktop, go to **Settings > Developer > Edit Config**, and add the following to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "conclave": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/conclave-mcp", "python", "server.py"],
      "env": {
        "OPENROUTER_API_KEY": "sk-or-v1-your-key-here"
      }
    }
  }
}
```

Replace `/path/to/conclave-mcp` with your actual path, save, and restart Claude Desktop.

### Configure Claude Code

Add the server using the CLI:

```bash
claude mcp add --transport stdio conclave -- uv run --directory /path/to/conclave-mcp python server.py --env OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

Or copy `.mcp.json.example` to `.mcp.json` and update paths:

```bash
cp .mcp.json.example .mcp.json
# Edit .mcp.json with your paths and API key
```

Verify with `/mcp` in Claude Code or `claude mcp list` in terminal.

## Available Tools

### `conclave_quick`

Fast parallel opinions (Stage 1 only). Queries all conclave models and returns individual responses.

**Cost**: ~$0.01-0.03 per query

**Use for**: Quick brainstorming, getting diverse perspectives fast

### `conclave_ranked`

Opinions with peer rankings (Stage 1 + 2). Shows which model performed best on this specific question.

**Cost**: ~$0.05-0.10 per query

**Use for**: Code review, comparing approaches, seeing which model "won"

### `conclave_full`

Complete conclave with synthesis (all 3 stages). Includes consensus detection and chairman tiebreaker.

**Cost**: ~$0.10-0.20 per query

**Options**:
- `tier`: Model tier - `"premium"`, `"standard"` (default), `"budget"`
- `chairman`: Override chairman model (e.g., `"anthropic/claude-sonnet-4"`)
- `chairman_preset`: Use a preset (`"code"`, `"creative"`, `"reasoning"`, `"concise"`, `"balanced"`)

**Use for**: Important decisions, architecture choices, complex debugging

### `conclave_config`

View current configuration: conclave members, chairman rotation status, consensus thresholds.

### `conclave_estimate`

Estimate costs before running a query.

## Configuration

Edit `config.py` to customize:

### Conclave Tiers

Each tier has unique models (no overlap) for proper price/performance differentiation:

```python
# Premium: 6 frontier models for complex questions (~$0.30-0.50/query)
COUNCIL_PREMIUM = [
    "anthropic/claude-opus-4.5",        # Claude Opus 4.5
    "google/gemini-3-pro-preview",      # Gemini 3 Pro
    "x-ai/grok-4.1",                    # Grok 4.1 (full reasoning)
    "openai/gpt-5.1",                   # GPT-5.1 (flagship)
    "deepseek/deepseek-v3.2-speciale",  # DeepSeek V3.2 Speciale
    "moonshotai/kimi-k2-thinking",      # Kimi K2 Thinking (1T MoE)
]

# Standard: 4 balanced models (default) (~$0.10-0.20/query)
COUNCIL_STANDARD = [
    "anthropic/claude-sonnet-4.5",      # Claude Sonnet 4.5
    "google/gemini-2.5-pro",            # Gemini 2.5 Pro
    "openai/o4-mini",                   # OpenAI o4-mini
    "deepseek/deepseek-v3.1",           # DeepSeek V3.1 Terminus
]

# Budget: 4 cheap/fast models (~$0.02-0.05/query)
COUNCIL_BUDGET = [
    "google/gemini-2.5-flash",          # Gemini 2.5 Flash
    "x-ai/grok-4.1-fast:free",          # Grok 4.1 Fast (free tier)
    "openai/gpt-4.1-mini",              # GPT-4.1 Mini
    "deepseek/deepseek-chat-v3-0324:free",  # DeepSeek Chat (free)
]
```

### Chairman Rotation

```python
CHAIRMAN_ROTATION_ENABLED = True
CHAIRMAN_ROTATION_DAYS = 7  # Rotate weekly

CHAIRMAN_POOL = [
    "deepseek/deepseek-chat",
    "x-ai/grok-3",
    "mistralai/mistral-large-2",
    "qwen/qwen3-235b-a22b",
]
```

### Consensus Thresholds

```python
CONSENSUS_STRONG_THRESHOLD = 0.75   # 75%+ agreement
CONSENSUS_MODERATE_THRESHOLD = 0.50  # 50-75% agreement
CHAIRMAN_TIEBREAKER_ENABLED = True   # Chairman breaks ties
```

## Eval-Light

A standalone benchmark runner for testing and comparing conclave performance across tiers and over time.

### Test Suite Overview

The eval suite includes **16 tasks** across **9 categories**, designed to test different model capabilities:

| Category | Tasks | Difficulty | What It Tests |
|----------|-------|------------|---------------|
| **math** | 2 | Easy-Medium | Arithmetic, word problems, step-by-step reasoning |
| **code** | 2 | Easy-Medium | Bug detection, concept explanation, code examples |
| **reasoning** | 2 | Medium-Hard | Syllogisms, multi-step logic puzzles |
| **analysis** | 2 | Medium | Logical fallacies, tradeoff analysis |
| **summarization** | 2 | Medium | Technical docs, business reports |
| **writing_business** | 2 | Easy-Medium | Professional emails, proposals |
| **writing_creative** | 2 | Easy-Medium | Story openings, original metaphors |
| **creative** | 1 | Easy | Analogies with explanations |
| **factual** | 1 | Easy | Science explanations for general audience |

### Running Evaluations

```bash
# Run all 16 tests at standard tier (default)
python eval.py

# Run at different tiers
python eval.py --tier premium    # 6 frontier models (~$0.30-0.50/query)
python eval.py --tier standard   # 4 balanced models (~$0.10-0.20/query)
python eval.py --tier budget     # 4 cheap/fast models (~$0.02-0.05/query)

# Different modes
python eval.py --mode quick      # Stage 1 only (fastest, cheapest)
python eval.py --mode ranked     # Stage 1 + 2 (adds peer rankings)
python eval.py --mode full       # All 3 stages (default, includes synthesis)

# Filter by category
python eval.py --category math
python eval.py --category code
python eval.py --category reasoning

# Don't save results to disk
python eval.py --no-save

# Combine options
python eval.py --tier premium --mode full --category reasoning
```

### Output Format

Results are saved to `evals/eval_<tier>_<mode>_<timestamp>.json` with:

- **metadata**: Timestamp, tier, mode, chairman model
- **summary**: Success rate, total time, average time per task
- **results**: Per-task details including:
  - Individual model responses
  - Peer rankings (for ranked/full modes)
  - Chairman synthesis (for full mode)
  - Consensus level

### Example Output

```
🏛️  Conclave Eval-Light
   Tier: standard | Mode: full | Tasks: 16
--------------------------------------------------

[1/16] Running: math_arithmetic (math)
   ✓ Completed in 12.34s

[2/16] Running: math_word_problem (math)
   ✓ Completed in 15.67s
...

==================================================
📊 EVAL SUMMARY
==================================================
Tier: standard | Mode: full
Chairman: deepseek/deepseek-chat
Tasks: 16/16 successful
Total time: 287.45s
Avg per task: 17.97s

📋 Results by Task:
  ✓ math_arithmetic (easy) - 12.34s
  ✓ math_word_problem (medium) - 15.67s
  ✓ code_debug (easy) - 11.23s
  ...

💾 Results saved to: evals/eval_standard_full_20251204_143052.json
```

### Comparing Tiers

Run the same eval across all tiers to compare model quality vs cost:

```bash
python eval.py --tier budget --category reasoning
python eval.py --tier standard --category reasoning
python eval.py --tier premium --category reasoning
```

Then compare the JSON outputs to see how different model tiers perform on the same tasks.

## Use Cases

| Scenario | Recommended Tool | Why |
|----------|------------------|-----|
| "Review this function" | `conclave_ranked` | See which model catches the most issues |
| "Redis vs PostgreSQL for sessions?" | `conclave_full` | Important decision, need synthesis |
| "Ideas for this feature" | `conclave_quick` | Fast diverse brainstorming |
| "Debug this error" | `conclave_quick` | Quick parallel diagnosis |
| "Rewrite this paragraph" | `conclave_full` + `chairman_preset="creative"` | Creative synthesis |
| "Is this architecture sound?" | `conclave_full` + `chairman_preset="code"` | Technical synthesis |

## Example Tool Output

```
## Conclave Full Result

**Consensus: ✅ STRONG** (75% agreement)

---

### Chairman's Synthesis

_Chairman: deepseek/deepseek-chat_

[Synthesized answer incorporating best points from all models...]

---

### Model Rankings (lower is better)

1. **claude-sonnet-4.5**: 1.50
2. **o4-mini**: 2.00
3. **gemini-2.5-pro**: 2.75
4. **deepseek-v3.1**: 3.75

_First-place votes:_ claude-sonnet-4.5=3, o4-mini=1
```

## Project Structure

```
conclave-mcp/
├── server.py      # MCP server entry point (5 tools)
├── conclave.py    # Core 3-stage council logic
├── config.py      # Model tiers, chairman rotation, cost estimates
├── eval.py        # Standalone benchmark runner
└── evals/         # Saved evaluation results
```

## Adding Models

OpenRouter supports 200+ models. Find model IDs at https://openrouter.ai/models

```python
# Add to COUNCIL_* lists in config.py
"x-ai/grok-4"                    # xAI Grok
"meta-llama/llama-4-maverick"    # Meta Llama
"mistralai/mistral-large-2"      # Mistral
"deepseek/deepseek-r1"           # DeepSeek reasoning
```

**Important**: Keep each tier's models unique (no overlap) for proper differentiation.

## How OpenRouter Works

OpenRouter is a unified API gateway—you don't need separate accounts with OpenAI, Google, Anthropic, etc. One API key, one credit balance, access to all models.

- Sign up: https://openrouter.ai
- Add credits (prepaid, or enable auto-top-up)
- Use your single API key for all models

## License

MIT

## Attribution

Inspired by [Andrej Karpathy's llm-council](https://github.com/karpathy/llm-council). The original is a web application for interactively exploring LLM comparisons. This project reimplements the council concept as an MCP server for integration with AI-assisted editors, adding consensus protocol and tiebreaker mechanics.
