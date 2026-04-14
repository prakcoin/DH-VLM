# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DH-Agent is a multi-agent fashion archive assistant specializing in the Dior Homme Autumn/Winter 2004 "Victim of the Crime" collection. It uses Amazon Bedrock (Nova 2 Lite for orchestration, Nova Pro for visual analysis) and the **Strands Agents framework** to answer natural language queries over a multimodal knowledge base of runway images and structured garment metadata.

## Commands

```bash
# Install dependencies
uv sync

# Run the server (dev)
uv run uvicorn agent:app --host 0.0.0.0 --port 8080

# Query the running agent
curl -X POST http://localhost:8080/invocations \
  -H "Content-Type: application/json" \
  -d '{"input": {"prompt": "What does look 1 consist of?"}}'

# Health check
curl http://localhost:8080/ping

# Run evaluation suites
uv run python evaluation/evaluate.py --mode aggregation
uv run python evaluation/evaluate.py --mode general
uv run python evaluation/evaluate.py --mode followups
```

## Architecture

### Three-Agent Orchestration

**Orchestrator** (`src/orchestration/orchestrator.py`) receives queries at `POST /invocations` and routes to two sub-agents as tools:

1. **Archive Assistant** (`src/agents/archive_agent.py`) — knowledge base expert for the AW04 collection
2. **Search Assistant** (`src/agents/search_agent.py`) — web research for marketplace listings and context

The orchestrator uses `BedrockModel("us.amazon.nova-2-lite-v1:0")` with `ProactiveSummarizingConversationManager` (auto-summarizes at 20+ messages, keeps 10 recent).

### Archive Tools (`src/tools/archive_tools/`)

- **`collection_inventory.py`** — Map-reduce aggregation over all 45 looks: splits CSV metadata into chunks, analyzes in parallel via sub-agents, reduces via aggregator agent. Supports filtering by subcategory/color.
- **`look_analysis.py`** — 3-step pipeline per look: (1) KB retrieval, (2) multi-image visual analysis via Nova Pro, (3) synthesis. No hallucination because it grounds every claim in KB data and images.
- **`image_input.py`** — Image-based similarity search using Bedrock Knowledge Base vector search, then visual comparison.

### Search Tools (`src/tools/search_tools/`)

- **`general_search.py`** — Simple Tavily web search for historical/contextual info.
- **`listing_search.py`** — 4-step pipeline: (1) retrieve KB metadata (reference codes, materials), (2) multi-variant Tavily search across US and Japan, (3) validate URLs are still active via Tavily extraction API, (4) filter against KB ground truth.

### Hooks & Plugins (`src/agents/hooks.py`, `src/agents/handlers.py`)

- **`LimitToolCounts`** (hook) — Caps per-tool call counts to prevent runaway tool use.
- **`NotifyOnlyGuardrailsHook`** (hook) — Runs Bedrock Guardrails in shadow mode (logs but does not block).
- **`AgentSteeringHandler`** (plugin) — Validates model output tone/format before returning; issues `Guide` actions to redirect the model if needed. Also enforces workflow prerequisites (e.g., `retrieve` must be called before `get_image_details`).

### Data Sources

- **S3 `aw04-data` bucket**: `/looks/look_{look_number}.csv` (garment metadata for 45 looks), `/images/look{look_number}_{image_number}.jpg` (runway photos served via CloudFront)
- **Bedrock Knowledge Base** (ID from env): Vector search over metadata embeddings
- **AWS Secrets Manager** (`dh-agent/config`): Runtime config including API keys

### Evaluation (`evaluation/`)

Uses **Strands Evals** with OpenTelemetry span collection. Evaluators: `OutputEvaluator`, `HelpfulnessEvaluator`, `FaithfulnessEvaluator`, `ToolSelectionAccuracyEvaluator`, `GoalSuccessRateEvaluator`. Results written to `evaluation/results/{mode}/{timestamp}/reports/`. Test cases are in `evaluation/datasets/eval_*.json` with expected tool trajectories.

### Agent Skills (`src/agents/skills/`)

Each sub-agent has a `skills/` directory with `SKILL.md` files that document tool capabilities for the Strands framework. These are passed to agents at construction time and influence tool selection behavior.

## Environment Variables

Secrets are loaded at startup from AWS Secrets Manager (`dh-agent/config`), which overrides env vars.
