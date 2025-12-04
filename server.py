#!/usr/bin/env python3
"""
Conclave MCP Server

Exposes the Conclave (LLM Council) as MCP tools for use with Claude Desktop/Code.

Tools:
- conclave_quick: Fast parallel opinions (Stage 1 only)
- conclave_ranked: Opinions + peer rankings (Stage 1 + 2)
- conclave_full: Complete conclave with synthesis (all 3 stages)
- conclave_config: View/check current configuration
- conclave_estimate: Estimate cost before running
"""

import json
from typing import Optional

from mcp.server.fastmcp import FastMCP

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
from conclave import (
    run_council_quick,
    run_council_ranked,
    run_council_full,
)

# Initialize FastMCP server
mcp = FastMCP("conclave")


# =============================================================================
# TOOLS
# =============================================================================

@mcp.tool()
async def conclave_quick(question: str, tier: str = "standard") -> str:
    """Query the conclave for quick parallel opinions (Stage 1 only).

    Fast and cheap - queries all conclave models in parallel and returns their
    individual responses. No peer ranking or synthesis. Good for getting
    diverse perspectives quickly.

    Args:
        question: The question to ask the conclave
        tier: Model tier - "premium" (frontier), "standard" (default), "budget" (cheap/fast)

    Returns:
        Individual responses from each conclave model
    """
    if not question:
        return "Error: 'question' is required"

    if not OPENROUTER_API_KEY:
        return "Error: OPENROUTER_API_KEY not configured"

    if tier not in ("premium", "standard", "budget"):
        tier = "standard"

    models = get_council_by_tier(tier)

    try:
        result = await run_council_quick(question, models=models)
        result["tier"] = tier
        return format_quick_result(result)
    except Exception as e:
        return f"Error: {str(e)}"


@mcp.tool()
async def conclave_ranked(question: str, tier: str = "standard") -> str:
    """Query the conclave with peer rankings (Stage 1 + 2).

    Medium cost - gets individual opinions, then has each model anonymously
    evaluate and rank all responses. Returns aggregate "street cred" scores
    showing which models performed best on this specific question.

    Args:
        question: The question to ask the conclave
        tier: Model tier - "premium" (frontier), "standard" (default), "budget" (cheap/fast)

    Returns:
        Individual responses plus aggregate rankings
    """
    if not question:
        return "Error: 'question' is required"

    if not OPENROUTER_API_KEY:
        return "Error: OPENROUTER_API_KEY not configured"

    if tier not in ("premium", "standard", "budget"):
        tier = "standard"

    models = get_council_by_tier(tier)

    try:
        result = await run_council_ranked(question, models=models)
        result["tier"] = tier
        return format_ranked_result(result)
    except Exception as e:
        return f"Error: {str(e)}"


@mcp.tool()
async def conclave_full(
    question: str,
    tier: str = "standard",
    chairman: Optional[str] = None,
    chairman_preset: Optional[str] = None,
) -> str:
    """Run the full conclave with synthesis (all 3 stages).

    Most comprehensive - collects opinions, peer rankings, then has a Chairman
    model synthesize the best possible answer from the collective wisdom.

    Args:
        question: The question to ask the conclave
        tier: Model tier - "premium" (complex), "standard" (default), "budget" (simple)
        chairman: Override chairman model (e.g., 'anthropic/claude-sonnet-4')
        chairman_preset: Use a context-based preset - "code", "creative", "reasoning", "concise", "balanced"

    Returns:
        Chairman's synthesis, consensus level, rankings, and individual responses
    """
    if not question:
        return "Error: 'question' is required"

    if not OPENROUTER_API_KEY:
        return "Error: OPENROUTER_API_KEY not configured"

    if tier not in ("premium", "standard", "budget"):
        tier = "standard"

    if chairman_preset and chairman_preset not in CHAIRMAN_PRESETS:
        return f"Error: Invalid chairman_preset. Valid options: {list(CHAIRMAN_PRESETS.keys())}"

    models = get_council_by_tier(tier)

    try:
        result = await run_council_full(
            question,
            models=models,
            chairman=chairman,
            chairman_preset=chairman_preset,
        )
        result["tier"] = tier
        return format_full_result(result)
    except Exception as e:
        return f"Error: {str(e)}"


@mcp.tool()
async def conclave_config() -> str:
    """View current conclave configuration.

    Shows conclave member models, current chairman with rotation info,
    available chairman presets, consensus thresholds, and API key status.

    Returns:
        Current configuration as formatted JSON
    """
    rotation_info = get_rotation_info()
    size_validation = validate_council_size()

    config = {
        "api_key_configured": bool(OPENROUTER_API_KEY),
        "tiers": get_tier_info(),
        "active_conclave": COUNCIL_MODELS,
        "conclave_size": {
            "total_members": size_validation["total_size"],
            "is_odd": size_validation["valid"],
            "chairman_in_conclave": size_validation["chairman_included"],
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

    return f"## Conclave Configuration\n\n```json\n{json.dumps(config, indent=2)}\n```"


@mcp.tool()
async def conclave_estimate(question: str, tier: Optional[str] = None) -> str:
    """Estimate cost for a conclave query before running it.

    Provides approximate cost breakdown for quick/ranked/full query types.

    Args:
        question: The question (used to estimate token count)
        tier: Which tier to estimate - "quick", "ranked", "full" (default: all)

    Returns:
        Cost estimates for each query type
    """
    # Rough token estimate (4 chars per token)
    query_tokens = len(question) // 4 + 50

    tiers_to_estimate = [tier] if tier in ("quick", "ranked", "full") else ["quick", "ranked", "full"]

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

    return output


# =============================================================================
# RESULT FORMATTERS
# =============================================================================

def format_quick_result(result: dict) -> str:
    """Format quick conclave result for display."""
    output = "## Conclave Quick Opinions\n\n"

    for resp in result["stage1"]:
        model_name = resp["model"].split("/")[-1]  # Just the model name
        output += f"### {model_name}\n\n{resp['content']}\n\n---\n\n"

    return output


def format_ranked_result(result: dict) -> str:
    """Format ranked conclave result for display."""
    output = "## Conclave Opinions with Rankings\n\n"

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
    """Format full conclave result for display."""
    output = "## Conclave Full Result\n\n"

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

if __name__ == "__main__":
    mcp.run(transport='stdio')
