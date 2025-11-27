#!/usr/bin/env python3
"""
LLM Council MCP Server

Exposes the LLM Council as MCP tools for use with Claude Desktop/Code.

Tools:
- council_quick: Fast parallel opinions (Stage 1 only)
- council_ranked: Opinions + peer rankings (Stage 1 + 2)
- council_full: Complete council with synthesis (all 3 stages)
- council_config: View/check current configuration
- council_estimate: Estimate cost before running
"""

import asyncio
import json
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
)

from config import (
    OPENROUTER_API_KEY,
    COUNCIL_MODELS,
    CHAIRMAN_PRESETS,
    CONSENSUS_STRONG_THRESHOLD,
    CONSENSUS_MODERATE_THRESHOLD,
    CHAIRMAN_TIEBREAKER_ENABLED,
    get_current_chairman,
    get_rotation_info,
    estimate_cost,
    validate_council_size,
    get_council_by_tier,
    get_tier_info,
)
from council import (
    run_council_quick,
    run_council_ranked,
    run_council_full,
)

# Initialize MCP server
server = Server("llm-council")


# =============================================================================
# TOOL DEFINITIONS
# =============================================================================

@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available council tools."""
    return [
        Tool(
            name="council_quick",
            description="""Query the LLM council for quick parallel opinions (Stage 1 only).

Fast and cheap - queries all council models in parallel and returns their individual responses.
No peer ranking or synthesis. Good for getting diverse perspectives quickly.

Use 'tier' to select model quality: "premium" (frontier), "standard" (default), "budget" (cheap/fast)

Cost: budget ~$0.01, standard ~$0.02, premium ~$0.10 per query""",
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question to ask the council",
                    },
                    "tier": {
                        "type": "string",
                        "enum": ["premium", "standard", "budget"],
                        "description": "Model tier: premium (complex), standard (default), budget (simple)",
                    },
                },
                "required": ["question"],
            },
        ),
        Tool(
            name="council_ranked",
            description="""Query the LLM council with peer rankings (Stage 1 + 2).

Medium cost - gets individual opinions, then has each model anonymously evaluate
and rank all responses. Returns aggregate "street cred" scores showing which
models performed best on this specific question.

Use 'tier' to select model quality: "premium" (frontier), "standard" (default), "budget" (cheap/fast)

Cost: budget ~$0.03, standard ~$0.08, premium ~$0.25 per query""",
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question to ask the council",
                    },
                    "tier": {
                        "type": "string",
                        "enum": ["premium", "standard", "budget"],
                        "description": "Model tier: premium (complex), standard (default), budget (simple)",
                    },
                },
                "required": ["question"],
            },
        ),
        Tool(
            name="council_full",
            description="""Run the full LLM council with synthesis (all 3 stages).

Most comprehensive - collects opinions, peer rankings, then has a Chairman model
synthesize the best possible answer from the collective wisdom.

Use 'tier' for model quality: "premium" (frontier models for complex questions),
"standard" (default), "budget" (cheap/fast for simple questions)

Use 'chairman' to override chairman model, or 'chairman_preset' for context-based
selection: "code", "creative", "reasoning", "concise", "balanced"

Cost: budget ~$0.05, standard ~$0.15, premium ~$0.40 per query""",
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question to ask the council",
                    },
                    "tier": {
                        "type": "string",
                        "enum": ["premium", "standard", "budget"],
                        "description": "Model tier: premium (complex), standard (default), budget (simple)",
                    },
                    "chairman": {
                        "type": "string",
                        "description": "Optional: Override chairman model (e.g., 'anthropic/claude-sonnet-4')",
                    },
                    "chairman_preset": {
                        "type": "string",
                        "enum": ["code", "creative", "reasoning", "concise", "balanced"],
                        "description": "Optional: Use a context-based chairman preset",
                    },
                },
                "required": ["question"],
            },
        ),
        Tool(
            name="council_config",
            description="""View current council configuration.

Shows:
- Council member models
- Current chairman (with rotation info)
- Available chairman presets
- API key status""",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="council_estimate",
            description="""Estimate cost for a council query before running it.

Provides approximate cost breakdown for quick/ranked/full tiers.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question (used to estimate token count)",
                    },
                    "tier": {
                        "type": "string",
                        "enum": ["quick", "ranked", "full"],
                        "description": "Which tier to estimate (default: all)",
                    },
                },
                "required": ["question"],
            },
        ),
    ]


# =============================================================================
# TOOL HANDLERS
# =============================================================================

@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls."""

    if name == "council_config":
        return await handle_config()

    if name == "council_estimate":
        return await handle_estimate(arguments)

    if name == "council_quick":
        return await handle_quick(arguments)

    if name == "council_ranked":
        return await handle_ranked(arguments)

    if name == "council_full":
        return await handle_full(arguments)

    return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def handle_config() -> list[TextContent]:
    """Return current configuration."""
    rotation_info = get_rotation_info()
    size_validation = validate_council_size()

    config = {
        "api_key_configured": bool(OPENROUTER_API_KEY),
        "tiers": get_tier_info(),
        "active_council": COUNCIL_MODELS,
        "council_size": {
            "total_members": size_validation["total_size"],
            "is_odd": size_validation["valid"],
            "chairman_in_council": size_validation["chairman_included"],
            "status": size_validation["message"],
        },
        "chairman": {
            "current": rotation_info["current_chairman"],
            "rotation_enabled": rotation_info["rotation_enabled"],
            "rotation_period_days": rotation_info["rotation_period_days"],
            "days_until_rotation": rotation_info["days_until_rotation"],
            "chairman_pool": rotation_info["chairman_pool"],
        },
        "consensus": {
            "strong_threshold": f"{CONSENSUS_STRONG_THRESHOLD:.0%}",
            "moderate_threshold": f"{CONSENSUS_MODERATE_THRESHOLD:.0%}",
            "tiebreaker_enabled": CHAIRMAN_TIEBREAKER_ENABLED,
        },
        "presets": CHAIRMAN_PRESETS,
    }

    return [TextContent(
        type="text",
        text=f"## LLM Council Configuration\n\n```json\n{json.dumps(config, indent=2)}\n```",
    )]


async def handle_estimate(arguments: dict) -> list[TextContent]:
    """Estimate costs for a query."""
    question = arguments.get("question", "")
    tier = arguments.get("tier")

    # Rough token estimate (4 chars per token)
    query_tokens = len(question) // 4 + 50

    tiers_to_estimate = [tier] if tier else ["quick", "ranked", "full"]

    estimates = {}
    for t in tiers_to_estimate:
        estimates[t] = estimate_cost(query_tokens, tier=t)

    output = "## Cost Estimates\n\n"
    for t, est in estimates.items():
        output += f"### {t.title()}\n"
        output += f"- Stage 1: ${est['stage1']:.4f}\n"
        output += f"- Stage 2: ${est['stage2']:.4f}\n"
        output += f"- Stage 3: ${est['stage3']:.4f}\n"
        output += f"- **Total: ${est['total']:.4f}**\n\n"

    output += f"_Estimates based on ~{query_tokens} input tokens. Actual costs may vary._"

    return [TextContent(type="text", text=output)]


async def handle_quick(arguments: dict) -> list[TextContent]:
    """Run quick council (Stage 1 only)."""
    question = arguments.get("question")
    tier = arguments.get("tier", "standard")

    if not question:
        return [TextContent(type="text", text="Error: 'question' is required")]

    if not OPENROUTER_API_KEY:
        return [TextContent(type="text", text="Error: OPENROUTER_API_KEY not configured")]

    # Get models for selected tier
    models = get_council_by_tier(tier)

    try:
        result = await run_council_quick(question, models=models)
        result["tier"] = tier
        return [TextContent(type="text", text=format_quick_result(result))]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]


async def handle_ranked(arguments: dict) -> list[TextContent]:
    """Run ranked council (Stage 1 + 2)."""
    question = arguments.get("question")
    tier = arguments.get("tier", "standard")

    if not question:
        return [TextContent(type="text", text="Error: 'question' is required")]

    if not OPENROUTER_API_KEY:
        return [TextContent(type="text", text="Error: OPENROUTER_API_KEY not configured")]

    # Get models for selected tier
    models = get_council_by_tier(tier)

    try:
        result = await run_council_ranked(question, models=models)
        result["tier"] = tier
        return [TextContent(type="text", text=format_ranked_result(result))]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]


async def handle_full(arguments: dict) -> list[TextContent]:
    """Run full council (all 3 stages)."""
    question = arguments.get("question")
    tier = arguments.get("tier", "standard")
    chairman = arguments.get("chairman")
    chairman_preset = arguments.get("chairman_preset")

    if not question:
        return [TextContent(type="text", text="Error: 'question' is required")]

    if not OPENROUTER_API_KEY:
        return [TextContent(type="text", text="Error: OPENROUTER_API_KEY not configured")]

    # Get models for selected tier
    models = get_council_by_tier(tier)

    try:
        result = await run_council_full(
            question,
            models=models,
            chairman=chairman,
            chairman_preset=chairman_preset,
        )
        result["tier"] = tier
        return [TextContent(type="text", text=format_full_result(result))]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]


# =============================================================================
# RESULT FORMATTERS
# =============================================================================

def format_quick_result(result: dict) -> str:
    """Format quick council result for display."""
    output = "## Council Quick Opinions\n\n"

    for resp in result["stage1"]:
        model_name = resp["model"].split("/")[-1]  # Just the model name
        output += f"### {model_name}\n\n{resp['content']}\n\n---\n\n"

    return output


def format_ranked_result(result: dict) -> str:
    """Format ranked council result for display."""
    output = "## Council Opinions with Rankings\n\n"

    # Stage 1 responses
    output += "### Individual Responses\n\n"
    for resp in result["stage1"]:
        model_name = resp["model"].split("/")[-1]
        output += f"#### {model_name}\n\n{resp['content']}\n\n---\n\n"

    # Aggregate rankings
    output += "### Aggregate Rankings (lower is better)\n\n"
    sorted_rankings = sorted(
        result["stage2"]["aggregate"].items(),
        key=lambda x: x[1]
    )
    for i, (model, score) in enumerate(sorted_rankings, 1):
        model_name = model.split("/")[-1]
        output += f"{i}. **{model_name}**: {score:.2f} avg rank\n"

    return output


def format_full_result(result: dict) -> str:
    """Format full council result for display."""
    output = "## Council Full Result\n\n"

    # Consensus status badge
    consensus = result.get("consensus", {})
    consensus_level = consensus.get("level", "unknown")
    consensus_emoji = {
        "strong": "✅",
        "moderate": "🟡",
        "weak": "🟠",
        "split": "⚖️",
    }.get(consensus_level, "❓")

    output += f"**Consensus: {consensus_emoji} {consensus_level.upper()}**"
    if consensus.get("ranking_agreement"):
        output += f" ({consensus['ranking_agreement']:.0%} agreement)"
    output += "\n\n"

    # Tiebreaker info if used
    tiebreaker = result.get("tiebreaker")
    if tiebreaker and tiebreaker.get("valid_vote"):
        output += f"⚖️ **Tiebreaker Vote**: Chairman selected **{tiebreaker['vote'].split('/')[-1]}** (Response {tiebreaker['vote_label']})\n\n"

    # Council size warning if even
    size_info = result.get("council_size", {})
    if size_info and not size_info.get("valid", True):
        output += f"⚠️ {size_info.get('message', 'Council size is even')}\n\n"

    output += "---\n\n"

    # Final synthesis (most important)
    output += "### Chairman's Synthesis\n\n"
    output += f"_Chairman: {result['stage3']['chairman']}_\n"
    if result['stage3'].get('tiebreaker_used'):
        output += "_Tiebreaker vote was cast_\n"
    output += f"\n{result['stage3']['synthesis']}\n\n"
    output += "---\n\n"

    # Aggregate rankings
    output += "### Model Rankings (lower is better)\n\n"
    sorted_rankings = sorted(
        result["stage2"]["aggregate"].items(),
        key=lambda x: x[1]
    )
    for i, (model, score) in enumerate(sorted_rankings, 1):
        model_name = model.split("/")[-1]
        # Mark tied models
        is_tied = consensus_level == "split" and model in consensus.get("split_details", {}).get("tied_models", [])
        tie_marker = " ⚖️" if is_tied else ""
        output += f"{i}. **{model_name}**: {score:.2f}{tie_marker}\n"

    # First place vote distribution
    if consensus.get("first_place_votes"):
        output += "\n_First-place votes:_ "
        votes = [f"{m.split('/')[-1]}={v}" for m, v in consensus["first_place_votes"].items()]
        output += ", ".join(votes)
        output += "\n"

    output += "\n---\n\n"

    # Individual responses (collapsed by default in most renderers)
    output += "<details>\n<summary>Individual Responses</summary>\n\n"
    for resp in result["stage1"]:
        model_name = resp["model"].split("/")[-1]
        output += f"#### {model_name}\n\n{resp['content']}\n\n---\n\n"
    output += "</details>\n"

    # Tiebreaker reasoning if available
    if tiebreaker and tiebreaker.get("reasoning"):
        output += "\n<details>\n<summary>Tiebreaker Reasoning</summary>\n\n"
        output += f"{tiebreaker['reasoning']}\n"
        output += "</details>\n"

    return output


# =============================================================================
# MAIN
# =============================================================================

async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
