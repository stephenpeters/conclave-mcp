"""
LLM Council MCP Server Configuration

Customize your council members, chairman rotation, and cost controls here.
"""

import os
from datetime import datetime
from typing import Optional

# =============================================================================
# OPENROUTER CONFIGURATION
# =============================================================================

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

# =============================================================================
# COUNCIL TIERS
# =============================================================================

# Premium council for complex questions (architecture, important decisions)
# 6 frontier models + 1 chairman = 7 (odd)
COUNCIL_PREMIUM = [
    "google/gemini-2.5-pro-preview-06-05",  # Gemini 3 Pro preview
    "anthropic/claude-sonnet-4-5-20250514", # Claude Sonnet 4.5
    "openai/gpt-5.1",                       # GPT 5.1
    "x-ai/grok-4-0414",                     # Grok 4.1
    "moonshotai/kimi-k2",                   # Kimi K2 (K3 when available)
    "deepseek/deepseek-r1",                 # DeepSeek R1 reasoning
]

# Standard council for typical questions (default)
# 4 models + 1 chairman = 5 (odd)
COUNCIL_STANDARD = [
    "openai/gpt-4.1",
    "anthropic/claude-sonnet-4",
    "google/gemini-2.5-pro",
    "moonshotai/kimi-k2",
]

# Budget council for simple questions (quick checks, brainstorming)
# 4 cheap/fast models + 1 chairman = 5 (odd)
COUNCIL_BUDGET = [
    "anthropic/claude-3-5-haiku-20241022",  # Claude Haiku
    "google/gemini-2.5-flash",              # Gemini Flash
    "deepseek/deepseek-chat",               # DeepSeek (very cheap)
    "qwen/qwen3-32b",                       # Qwen 32B
]

# Active council (change this to switch tiers, or use tier= parameter in tools)
COUNCIL_MODELS = COUNCIL_STANDARD

# =============================================================================
# COUNCIL MEMBERS (Legacy - use COUNCIL_TIERS above)
# =============================================================================

# Models that participate in the council (Stage 1 opinions + Stage 2 rankings)
# IMPORTANT: Total council size (members + chairman) must be ODD for tiebreaker votes
# By default, chairman is NOT in council (4 members + 1 separate chairman = 5, odd)

# =============================================================================
# CONSENSUS CONFIGURATION
# =============================================================================

# Consensus thresholds
CONSENSUS_STRONG_THRESHOLD = 0.75   # 75%+ agreement = strong consensus
CONSENSUS_MODERATE_THRESHOLD = 0.50  # 50-75% = moderate consensus
# Below 50% = split/weak consensus

# Chairman tiebreaker settings
CHAIRMAN_TIEBREAKER_ENABLED = True  # Chairman casts deciding vote on splits

# =============================================================================
# CHAIRMAN CONFIGURATION
# =============================================================================

# Pool of models eligible for chairmanship (Stage 3 synthesis)
# IMPORTANT: By default, chairman should NOT be in COUNCIL_MODELS to ensure odd total
# These are separate "judge" models that only synthesize, not participate in Stage 1/2
CHAIRMAN_POOL = [
    "deepseek/deepseek-chat",        # Cheap, good synthesis
    "x-ai/grok-3",                   # Alternative perspective
    "mistralai/mistral-large-2",     # European perspective
    "qwen/qwen3-235b-a22b",          # Large MoE model
]

# Rotation settings
CHAIRMAN_ROTATION_ENABLED = True
CHAIRMAN_ROTATION_DAYS = 7  # Rotate every N days

# Default chairman when rotation is disabled
DEFAULT_CHAIRMAN = "deepseek/deepseek-chat"

# Title generation model (fast and cheap)
TITLE_MODEL = "google/gemini-2.5-flash"

# =============================================================================
# CONTEXT-BASED CHAIRMAN PRESETS
# =============================================================================

# Use these with chairman_preset="code" etc. in tool calls
# NOTE: These override the default chairman - may result in even council if preset
# model is also in COUNCIL_MODELS. Use explicit chairman= for full control.
CHAIRMAN_PRESETS = {
    "default": "deepseek/deepseek-chat",
    "code": "deepseek/deepseek-chat",         # Good at code, cheap
    "creative": "x-ai/grok-3",                # Creative synthesis
    "reasoning": "qwen/qwen3-235b-a22b",      # Strong chain-of-thought
    "concise": "deepseek/deepseek-chat",      # Brief answers
    "balanced": "mistralai/mistral-large-2",  # Well-rounded
}

# =============================================================================
# COST ESTIMATION (approximate USD per 1K tokens)
# =============================================================================

# Input/output costs for estimation (update as pricing changes)
MODEL_COSTS = {
    # Format: "model": (input_per_1k, output_per_1k)
    "openai/gpt-4.1": (0.002, 0.008),
    "anthropic/claude-sonnet-4": (0.003, 0.015),
    "google/gemini-2.5-pro": (0.00125, 0.01),
    "google/gemini-2.5-flash": (0.00015, 0.0006),
    "moonshotai/kimi-k2": (0.0006, 0.0024),
    "deepseek/deepseek-chat": (0.00014, 0.00028),
    "qwen/qwen3-32b": (0.0003, 0.0003),
    "x-ai/grok-3": (0.003, 0.015),
}

# Default cost for unknown models
DEFAULT_MODEL_COST = (0.002, 0.01)

# =============================================================================
# COST CONTROLS
# =============================================================================

# Maximum spend per session (set to None to disable)
MAX_SESSION_SPEND_USD: Optional[float] = None  # e.g., 1.00 for $1 limit

# Require confirmation for full council queries
REQUIRE_CONFIRMATION_FOR_FULL = False

# =============================================================================
# CHAIRMAN ROTATION LOGIC
# =============================================================================

def get_current_chairman(
    override: Optional[str] = None,
    preset: Optional[str] = None,
) -> str:
    """
    Get the current chairman model.

    Priority:
    1. Explicit override (full model ID)
    2. Preset name (e.g., "code", "creative")
    3. Rotating chairman (if enabled)
    4. Default chairman

    Args:
        override: Explicit model ID to use as chairman
        preset: Preset name from CHAIRMAN_PRESETS

    Returns:
        Model ID string for the chairman
    """
    # Priority 1: Explicit override
    if override:
        return override

    # Priority 2: Preset selection
    if preset and preset in CHAIRMAN_PRESETS:
        return CHAIRMAN_PRESETS[preset]

    # Priority 3: Rotation
    if CHAIRMAN_ROTATION_ENABLED and CHAIRMAN_POOL:
        return _get_rotating_chairman()

    # Priority 4: Default
    return DEFAULT_CHAIRMAN


def _get_rotating_chairman() -> str:
    """Deterministic rotation based on day number."""
    # Days since Jan 1, 2025
    epoch = datetime(2025, 1, 1)
    day_number = (datetime.now() - epoch).days

    # Rotation index
    rotation_index = (day_number // CHAIRMAN_ROTATION_DAYS) % len(CHAIRMAN_POOL)

    return CHAIRMAN_POOL[rotation_index]


def get_rotation_info() -> dict:
    """Get information about current rotation state."""
    epoch = datetime(2025, 1, 1)
    now = datetime.now()
    day_number = (now - epoch).days

    days_into_rotation = day_number % CHAIRMAN_ROTATION_DAYS
    days_until_next = CHAIRMAN_ROTATION_DAYS - days_into_rotation

    return {
        "rotation_enabled": CHAIRMAN_ROTATION_ENABLED,
        "rotation_period_days": CHAIRMAN_ROTATION_DAYS,
        "current_chairman": get_current_chairman(),
        "days_until_rotation": days_until_next,
        "chairman_pool": CHAIRMAN_POOL,
    }


def estimate_cost(
    query_tokens: int,
    tier: str = "full",
    models: list[str] = None,
    chairman: str = None,
) -> dict:
    """
    Estimate cost for a council query.

    Args:
        query_tokens: Approximate tokens in the query
        tier: "quick" (stage 1), "ranked" (stage 1+2), or "full" (all stages)
        models: Council models to use (defaults to COUNCIL_MODELS)
        chairman: Chairman model (defaults to current chairman)

    Returns:
        Dict with cost breakdown and total
    """
    models = models or COUNCIL_MODELS
    chairman = chairman or get_current_chairman()

    def get_cost(model: str) -> tuple[float, float]:
        return MODEL_COSTS.get(model, DEFAULT_MODEL_COST)

    # Rough estimates for response sizes
    avg_response_tokens = 500
    ranking_tokens = 300
    synthesis_tokens = 800

    cost_breakdown = {
        "stage1": 0.0,
        "stage2": 0.0,
        "stage3": 0.0,
        "total": 0.0,
    }

    # Stage 1: Each model gets query, produces response
    for model in models:
        input_cost, output_cost = get_cost(model)
        cost_breakdown["stage1"] += (
            (query_tokens / 1000) * input_cost +
            (avg_response_tokens / 1000) * output_cost
        )

    if tier in ["ranked", "full"]:
        # Stage 2: Each model evaluates all responses
        eval_input_tokens = query_tokens + (avg_response_tokens * len(models))
        for model in models:
            input_cost, output_cost = get_cost(model)
            cost_breakdown["stage2"] += (
                (eval_input_tokens / 1000) * input_cost +
                (ranking_tokens / 1000) * output_cost
            )

    if tier == "full":
        # Stage 3: Chairman synthesizes everything
        synthesis_input = query_tokens + (avg_response_tokens * len(models)) + (ranking_tokens * len(models))
        input_cost, output_cost = get_cost(chairman)
        cost_breakdown["stage3"] = (
            (synthesis_input / 1000) * input_cost +
            (synthesis_tokens / 1000) * output_cost
        )

    cost_breakdown["total"] = sum([
        cost_breakdown["stage1"],
        cost_breakdown["stage2"],
        cost_breakdown["stage3"],
    ])

    return cost_breakdown


# =============================================================================
# COUNCIL VALIDATION
# =============================================================================

def validate_council_size(models: list[str] = None, chairman: str = None) -> dict:
    """
    Validate that council size is odd (including chairman).

    Args:
        models: Council models (defaults to COUNCIL_MODELS)
        chairman: Chairman model (defaults to current chairman)

    Returns:
        {
            "valid": bool,
            "total_size": int,
            "council_members": int,
            "chairman_included": bool,
            "message": str
        }
    """
    models = models or COUNCIL_MODELS
    chairman = chairman or get_current_chairman()

    council_count = len(models)
    chairman_in_council = chairman in models

    # Total voting members: council + chairman (if not already in council)
    total_size = council_count if chairman_in_council else council_count + 1

    is_odd = total_size % 2 == 1

    if is_odd:
        message = f"Council size is valid: {total_size} members (odd)"
    else:
        message = (
            f"WARNING: Council size is {total_size} (even). "
            f"Add or remove 1 model for proper tiebreaker support. "
            f"Current: {council_count} council members + "
            f"{'chairman already in council' if chairman_in_council else '1 chairman'}"
        )

    return {
        "valid": is_odd,
        "total_size": total_size,
        "council_members": council_count,
        "chairman_included": chairman_in_council,
        "message": message,
    }


def get_recommended_council_size(current_count: int, chairman_in_council: bool) -> dict:
    """
    Recommend how to adjust council size to be odd.

    Returns options for adding or removing models.
    """
    total = current_count if chairman_in_council else current_count + 1

    if total % 2 == 1:
        return {"adjustment_needed": False, "current_total": total}

    return {
        "adjustment_needed": True,
        "current_total": total,
        "options": [
            f"Add 1 model to council (total: {total + 1})",
            f"Remove 1 model from council (total: {total - 1})" if current_count > 2 else None,
            "Include chairman in council models" if not chairman_in_council else None,
        ],
    }


# =============================================================================
# COUNCIL TIER SELECTION
# =============================================================================

def get_council_by_tier(tier: str = "standard") -> list[str]:
    """
    Get council models by tier name.

    Args:
        tier: "premium", "standard", or "budget"

    Returns:
        List of model IDs for that tier
    """
    tiers = {
        "premium": COUNCIL_PREMIUM,
        "standard": COUNCIL_STANDARD,
        "budget": COUNCIL_BUDGET,
    }
    return tiers.get(tier.lower(), COUNCIL_STANDARD)


def get_tier_info() -> dict:
    """Get information about available council tiers."""
    return {
        "premium": {
            "models": COUNCIL_PREMIUM,
            "count": len(COUNCIL_PREMIUM),
            "description": "Frontier models for complex questions",
            "estimated_cost": "~$0.30-0.50 per full query",
        },
        "standard": {
            "models": COUNCIL_STANDARD,
            "count": len(COUNCIL_STANDARD),
            "description": "Balanced models for typical questions",
            "estimated_cost": "~$0.10-0.20 per full query",
        },
        "budget": {
            "models": COUNCIL_BUDGET,
            "count": len(COUNCIL_BUDGET),
            "description": "Fast/cheap models for simple questions",
            "estimated_cost": "~$0.02-0.05 per full query",
        },
    }
