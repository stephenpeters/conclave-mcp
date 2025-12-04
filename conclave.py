"""
Conclave Core Logic

Handles the 3-stage conclave process:
1. Collect individual opinions from all conclave models
2. Peer ranking with anonymized responses
3. Chairman synthesis of final answer
"""

import asyncio
import random
import re
from typing import Optional

import httpx

from config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    COUNCIL_MODELS,
    TITLE_MODEL,
    CONSENSUS_STRONG_THRESHOLD,
    CONSENSUS_MODERATE_THRESHOLD,
    CHAIRMAN_TIEBREAKER_ENABLED,
    get_current_chairman,
    validate_council_size,
)


# =============================================================================
# OPENROUTER API CLIENT
# =============================================================================

async def query_model(
    model: str,
    messages: list[dict],
    timeout: float = 120.0,
) -> dict:
    """Query a single model via OpenRouter."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/llm-council-mcp",
        "X-Title": "LLM Council MCP",
    }

    payload = {
        "model": model,
        "messages": messages,
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(
            OPENROUTER_BASE_URL,
            headers=headers,
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()

    return {
        "model": model,
        "content": data["choices"][0]["message"]["content"],
        "usage": data.get("usage", {}),
    }


async def query_models_parallel(
    models: list[str],
    messages: list[dict],
) -> list[dict]:
    """Query multiple models in parallel."""
    tasks = [query_model(model, messages) for model in models]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out failures
    successful = []
    for result in results:
        if isinstance(result, Exception):
            print(f"Model query failed: {result}")
        else:
            successful.append(result)

    return successful


# =============================================================================
# STAGE 1: COLLECT INDIVIDUAL OPINIONS
# =============================================================================

STAGE1_SYSTEM_PROMPT = """You are a helpful assistant participating in a council of AI models.
Answer the user's question thoughtfully and thoroughly.
Your response will be evaluated alongside responses from other AI models."""


async def stage1_collect_responses(
    user_query: str,
    models: list[str] = None,
) -> list[dict]:
    """
    Stage 1: Query all council models for their individual opinions.

    Returns list of {model, content, usage} dicts.
    """
    models = models or COUNCIL_MODELS

    messages = [
        {"role": "system", "content": STAGE1_SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ]

    responses = await query_models_parallel(models, messages)
    return responses


# =============================================================================
# STAGE 2: PEER RANKINGS
# =============================================================================

STAGE2_SYSTEM_PROMPT = """You are evaluating responses from multiple AI assistants to the same question.
Each response is labeled (Response A, Response B, etc.) and the identities are hidden.

Evaluate each response based on:
1. Accuracy and correctness
2. Completeness and thoroughness
3. Clarity and organization
4. Practical usefulness

Provide brief feedback on each response, then end with your final ranking in this exact format:

FINAL RANKING:
1. Response X
2. Response Y
3. Response Z
...

Rank from best (1) to worst. Include all responses in your ranking."""


def anonymize_responses(responses: list[dict]) -> tuple[str, dict]:
    """
    Convert model responses to anonymized format.

    Returns:
        - Formatted string with "Response A:", "Response B:", etc.
        - Mapping dict: {"A": "openai/gpt-4.1", "B": "anthropic/claude-sonnet-4", ...}
    """
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    # Shuffle to prevent position bias
    shuffled = responses.copy()
    random.shuffle(shuffled)

    label_to_model = {}
    formatted_parts = []

    for i, resp in enumerate(shuffled):
        label = labels[i]
        label_to_model[label] = resp["model"]
        formatted_parts.append(f"Response {label}:\n{resp['content']}")

    formatted_text = "\n\n---\n\n".join(formatted_parts)
    return formatted_text, label_to_model


def parse_ranking_from_text(text: str) -> list[str]:
    """
    Extract ranking from evaluator's response.

    Looks for "FINAL RANKING:" section and parses numbered list.
    Returns list of labels like ["B", "A", "C", "D"].
    """
    # Find FINAL RANKING section
    match = re.search(r"FINAL RANKING[:\s]*\n([\s\S]+?)(?:\n\n|$)", text, re.IGNORECASE)
    if not match:
        return []

    ranking_text = match.group(1)

    # Extract labels from numbered list
    rankings = []
    for line in ranking_text.strip().split("\n"):
        # Match patterns like "1. Response A" or "1. A" or "1) Response B"
        label_match = re.search(r"(?:Response\s+)?([A-Z])", line)
        if label_match:
            rankings.append(label_match.group(1))

    return rankings


async def stage2_collect_rankings(
    user_query: str,
    stage1_results: list[dict],
    models: list[str] = None,
) -> dict:
    """
    Stage 2: Have each model rank the anonymized responses.

    Returns:
        {
            "label_to_model": {"A": "model1", "B": "model2", ...},
            "rankings": [
                {"evaluator": "model1", "ranking": ["B", "A", "C"], "feedback": "..."},
                ...
            ],
            "aggregate": {"model1": 2.5, "model2": 1.5, ...}  # avg rank (lower is better)
        }
    """
    models = models or COUNCIL_MODELS

    # Anonymize responses
    anonymized_text, label_to_model = anonymize_responses(stage1_results)

    eval_prompt = f"""Original question: {user_query}

Here are the responses from different AI assistants:

{anonymized_text}

Please evaluate these responses and provide your ranking."""

    messages = [
        {"role": "system", "content": STAGE2_SYSTEM_PROMPT},
        {"role": "user", "content": eval_prompt},
    ]

    # Get rankings from all models
    eval_responses = await query_models_parallel(models, messages)

    rankings = []
    for resp in eval_responses:
        parsed_ranking = parse_ranking_from_text(resp["content"])
        rankings.append({
            "evaluator": resp["model"],
            "ranking": parsed_ranking,
            "feedback": resp["content"],
        })

    # Calculate aggregate scores
    aggregate = calculate_aggregate_rankings(rankings, label_to_model)

    return {
        "label_to_model": label_to_model,
        "rankings": rankings,
        "aggregate": aggregate,
    }


def calculate_aggregate_rankings(
    rankings: list[dict],
    label_to_model: dict,
) -> dict:
    """
    Calculate average rank position for each model.
    Lower score = better (ranked higher on average).
    """
    model_ranks = {model: [] for model in label_to_model.values()}
    label_to_model_inv = {v: k for k, v in label_to_model.items()}

    for ranking_data in rankings:
        ranking = ranking_data["ranking"]
        for position, label in enumerate(ranking, start=1):
            if label in label_to_model:
                model = label_to_model[label]
                model_ranks[model].append(position)

    # Calculate averages
    aggregate = {}
    for model, ranks in model_ranks.items():
        if ranks:
            aggregate[model] = sum(ranks) / len(ranks)
        else:
            aggregate[model] = float("inf")

    return aggregate


# =============================================================================
# CONSENSUS DETECTION
# =============================================================================

def detect_consensus(
    stage1_results: list[dict],
    stage2_results: dict,
) -> dict:
    """
    Analyze the level of consensus among council members.

    Looks at:
    1. Ranking agreement: Do evaluators agree on who's best?
    2. Score spread: How far apart are the aggregate rankings?

    Returns:
        {
            "level": "strong" | "moderate" | "weak" | "split",
            "top_ranked": "model_id",
            "score_spread": float,
            "ranking_agreement": float,  # 0-1, how much evaluators agree
            "split_details": {...} | None,  # If split, details on the factions
            "needs_tiebreaker": bool,
        }
    """
    aggregate = stage2_results["aggregate"]
    rankings = stage2_results["rankings"]

    if not aggregate:
        return {
            "level": "unknown",
            "needs_tiebreaker": False,
            "message": "No rankings available",
        }

    # Sort models by aggregate score (lower is better)
    sorted_models = sorted(aggregate.items(), key=lambda x: x[1])
    top_model = sorted_models[0][0]
    top_score = sorted_models[0][1]

    # Check for ties at the top
    tied_at_top = [m for m, s in sorted_models if abs(s - top_score) < 0.01]

    # Calculate score spread (difference between best and worst)
    scores = list(aggregate.values())
    score_spread = max(scores) - min(scores) if scores else 0

    # Calculate ranking agreement (how often do evaluators agree on #1?)
    first_place_votes = {}
    for r in rankings:
        if r["ranking"]:
            first = r["ranking"][0]
            model = stage2_results["label_to_model"].get(first)
            if model:
                first_place_votes[model] = first_place_votes.get(model, 0) + 1

    total_votes = len(rankings)
    max_first_place = max(first_place_votes.values()) if first_place_votes else 0
    ranking_agreement = max_first_place / total_votes if total_votes > 0 else 0

    # Determine consensus level
    if len(tied_at_top) > 1:
        level = "split"
        needs_tiebreaker = True
        split_details = {
            "tied_models": tied_at_top,
            "tied_score": top_score,
            "vote_distribution": first_place_votes,
        }
    elif ranking_agreement >= CONSENSUS_STRONG_THRESHOLD:
        level = "strong"
        needs_tiebreaker = False
        split_details = None
    elif ranking_agreement >= CONSENSUS_MODERATE_THRESHOLD:
        level = "moderate"
        needs_tiebreaker = False
        split_details = None
    else:
        level = "weak"
        needs_tiebreaker = False
        split_details = {
            "vote_distribution": first_place_votes,
            "no_clear_leader": True,
        }

    return {
        "level": level,
        "top_ranked": top_model,
        "score_spread": round(score_spread, 2),
        "ranking_agreement": round(ranking_agreement, 2),
        "first_place_votes": first_place_votes,
        "split_details": split_details,
        "needs_tiebreaker": needs_tiebreaker,
    }


# =============================================================================
# CHAIRMAN TIEBREAKER
# =============================================================================

TIEBREAKER_SYSTEM_PROMPT = """You are the Chairman of an AI council and must cast a TIEBREAKER VOTE.

The council is split - multiple responses received equal rankings and no clear winner emerged.

You must:
1. Carefully review the tied responses
2. Cast your deciding vote for ONE response
3. Briefly explain your reasoning

Your vote will break the tie and determine the council's position.

IMPORTANT: You MUST end your response with exactly this format:
TIEBREAKER VOTE: Response X

Where X is the letter of your chosen response."""


async def chairman_tiebreaker(
    user_query: str,
    stage1_results: list[dict],
    stage2_results: dict,
    tied_models: list[str],
    chairman: str = None,
    chairman_preset: str = None,
) -> dict:
    """
    Chairman casts tiebreaker vote when council is split.

    Args:
        user_query: Original question
        stage1_results: Individual responses from Stage 1
        stage2_results: Rankings from Stage 2
        tied_models: List of model IDs that are tied
        chairman: Explicit chairman override
        chairman_preset: Preset name

    Returns:
        {
            "chairman": "model_id",
            "vote": "model_id",  # The model chairman voted for
            "reasoning": "...",
            "vote_label": "A",  # The response label
        }
    """
    chairman_model = get_current_chairman(override=chairman, preset=chairman_preset)
    label_to_model = stage2_results["label_to_model"]
    model_to_label = {v: k for k, v in label_to_model.items()}

    # Build context showing only the tied responses
    tied_responses = []
    for resp in stage1_results:
        if resp["model"] in tied_models:
            label = model_to_label.get(resp["model"], "?")
            tied_responses.append(f"Response {label}:\n{resp['content']}")

    tied_text = "\n\n---\n\n".join(tied_responses)
    tied_labels = [model_to_label.get(m, "?") for m in tied_models]

    # Show vote distribution
    consensus = detect_consensus(stage1_results, stage2_results)
    vote_dist = consensus.get("first_place_votes", {})
    vote_text = "\n".join([
        f"  {model_to_label.get(m, '?')}: {v} first-place votes"
        for m, v in vote_dist.items()
    ])

    tiebreaker_prompt = f"""Original question: {user_query}

=== TIED RESPONSES ===

The following responses are TIED with equal aggregate rankings:

{tied_text}

=== CURRENT VOTE DISTRIBUTION ===

{vote_text}

=== YOUR TASK ===

As Chairman, you must cast the TIEBREAKER VOTE.
Choose between responses: {', '.join(tied_labels)}

Review carefully and cast your deciding vote."""

    messages = [
        {"role": "system", "content": TIEBREAKER_SYSTEM_PROMPT},
        {"role": "user", "content": tiebreaker_prompt},
    ]

    result = await query_model(chairman_model, messages)
    content = result["content"]

    # Parse the vote
    vote_match = re.search(r"TIEBREAKER VOTE:\s*Response\s+([A-Z])", content, re.IGNORECASE)
    vote_label = vote_match.group(1) if vote_match else None
    vote_model = label_to_model.get(vote_label) if vote_label else None

    return {
        "chairman": chairman_model,
        "vote": vote_model,
        "vote_label": vote_label,
        "reasoning": content,
        "valid_vote": vote_model in tied_models if vote_model else False,
    }


# =============================================================================
# STAGE 3: CHAIRMAN SYNTHESIS
# =============================================================================

STAGE3_SYSTEM_PROMPT = """You are the Chairman of an AI council. Your role is to synthesize
the best possible answer from the collective wisdom of the council.

You have access to:
1. The original question
2. Individual responses from each council member
3. Peer evaluations and rankings from each member

Create a comprehensive, well-structured final answer that:
- Incorporates the strongest points from each response
- Addresses any disagreements or nuances
- Provides a clear, actionable answer to the user

Be thorough but concise. The user wants the best possible answer, not a meta-discussion about the process."""


async def stage3_synthesize_final(
    user_query: str,
    stage1_results: list[dict],
    stage2_results: dict,
    chairman: str = None,
    chairman_preset: str = None,
    consensus: dict = None,
    tiebreaker: dict = None,
) -> dict:
    """
    Stage 3: Chairman synthesizes final answer from all inputs.

    Args:
        user_query: Original question
        stage1_results: Individual responses from Stage 1
        stage2_results: Rankings and feedback from Stage 2
        chairman: Explicit chairman model override
        chairman_preset: Preset name ("code", "creative", etc.)
        consensus: Consensus detection results
        tiebreaker: Tiebreaker vote results (if any)

    Returns:
        {"chairman": "model_id", "synthesis": "final answer text", "usage": {...}}
    """
    chairman_model = get_current_chairman(override=chairman, preset=chairman_preset)

    # Format Stage 1 responses (with model names)
    stage1_text = "\n\n---\n\n".join([
        f"Response from {resp['model']}:\n{resp['content']}"
        for resp in stage1_results
    ])

    # Format Stage 2 rankings
    stage2_text = "\n\n".join([
        f"Evaluation by {r['evaluator']}:\nRanking: {' > '.join(r['ranking'])}\n{r['feedback'][:500]}..."
        for r in stage2_results["rankings"]
    ])

    # Aggregate scores
    aggregate_text = "\n".join([
        f"  {model}: {score:.2f} avg rank"
        for model, score in sorted(stage2_results["aggregate"].items(), key=lambda x: x[1])
    ])

    # Build consensus context
    consensus_text = ""
    if consensus:
        consensus_text = f"""
=== CONSENSUS STATUS ===

Level: {consensus.get('level', 'unknown').upper()}
Ranking Agreement: {consensus.get('ranking_agreement', 0):.0%}
Top Ranked: {consensus.get('top_ranked', 'unknown')}
"""
        if consensus.get('level') == 'split':
            consensus_text += f"SPLIT DETECTED: {', '.join(consensus.get('split_details', {}).get('tied_models', []))}\n"

    # Build tiebreaker context
    tiebreaker_text = ""
    if tiebreaker and tiebreaker.get('valid_vote'):
        tiebreaker_text = f"""
=== CHAIRMAN TIEBREAKER VOTE ===

Your tiebreaker vote selected: {tiebreaker['vote']} (Response {tiebreaker['vote_label']})
This response should be weighted more heavily in your synthesis.
"""

    # Adjust system prompt based on consensus level
    if consensus and consensus.get('level') == 'split' and tiebreaker:
        system_prompt = STAGE3_SYSTEM_PROMPT + """

IMPORTANT: The council was SPLIT on this question. You cast a tiebreaker vote.
Your synthesis should favor the response you voted for while acknowledging
the valid points from other responses. Make the reasoning clear."""
    elif consensus and consensus.get('level') == 'weak':
        system_prompt = STAGE3_SYSTEM_PROMPT + """

NOTE: The council showed WEAK consensus on this question - there was significant
disagreement. Your synthesis should acknowledge this uncertainty and present
multiple valid perspectives where appropriate."""
    else:
        system_prompt = STAGE3_SYSTEM_PROMPT

    synthesis_prompt = f"""Original question: {user_query}

=== INDIVIDUAL RESPONSES ===

{stage1_text}

=== PEER EVALUATIONS ===

{stage2_text}

=== AGGREGATE RANKINGS (lower is better) ===

{aggregate_text}
{consensus_text}
{tiebreaker_text}
=== YOUR TASK ===

As Chairman, synthesize the best possible answer to the original question,
drawing on the council's collective wisdom and the peer evaluations."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": synthesis_prompt},
    ]

    result = await query_model(chairman_model, messages)

    return {
        "chairman": chairman_model,
        "synthesis": result["content"],
        "usage": result.get("usage", {}),
        "consensus_level": consensus.get("level") if consensus else None,
        "tiebreaker_used": tiebreaker is not None and tiebreaker.get("valid_vote", False),
    }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def run_council_quick(
    user_query: str,
    models: list[str] = None,
) -> dict:
    """
    Quick council: Stage 1 only (parallel opinions, no ranking or synthesis).

    Cheapest and fastest option.
    """
    stage1 = await stage1_collect_responses(user_query, models)

    return {
        "tier": "quick",
        "query": user_query,
        "stage1": stage1,
    }


async def run_council_ranked(
    user_query: str,
    models: list[str] = None,
) -> dict:
    """
    Ranked council: Stage 1 + Stage 2 (opinions + peer rankings).

    Medium cost, provides aggregate quality scores.
    """
    stage1 = await stage1_collect_responses(user_query, models)
    stage2 = await stage2_collect_rankings(user_query, stage1, models)

    return {
        "tier": "ranked",
        "query": user_query,
        "stage1": stage1,
        "stage2": stage2,
    }


async def run_council_full(
    user_query: str,
    models: list[str] | None = None,
    chairman: str | None = None,
    chairman_preset: str | None = None,
) -> dict:
    """
    Full council: All 3 stages with final synthesis.

    Most expensive but provides synthesized best answer.
    Includes consensus detection and chairman tiebreaker if needed.
    """
    models = models or COUNCIL_MODELS
    chairman_model = get_current_chairman(override=chairman, preset=chairman_preset)

    # Validate council size (warn if even)
    size_validation = validate_council_size(models, chairman_model)

    # Stage 1: Collect opinions
    stage1 = await stage1_collect_responses(user_query, models)

    # Stage 2: Peer rankings
    stage2 = await stage2_collect_rankings(user_query, stage1, models)

    # Detect consensus
    consensus = detect_consensus(stage1, stage2)

    # Handle tiebreaker if needed
    tiebreaker = None
    if consensus["needs_tiebreaker"] and CHAIRMAN_TIEBREAKER_ENABLED:
        tied_models = consensus["split_details"]["tied_models"]
        tiebreaker = await chairman_tiebreaker(
            user_query, stage1, stage2, tied_models,
            chairman=chairman,
            chairman_preset=chairman_preset,
        )

    # Stage 3: Chairman synthesis (with consensus context)
    stage3 = await stage3_synthesize_final(
        user_query, stage1, stage2,
        chairman=chairman,
        chairman_preset=chairman_preset,
        consensus=consensus,
        tiebreaker=tiebreaker,
    )

    return {
        "tier": "full",
        "query": user_query,
        "stage1": stage1,
        "stage2": stage2,
        "consensus": consensus,
        "tiebreaker": tiebreaker,
        "stage3": stage3,
        "council_size": size_validation,
    }


async def generate_title(user_query: str) -> str:
    """Generate a short title for a conversation."""
    messages = [
        {"role": "system", "content": "Generate a very short title (3-6 words) for this conversation. Reply with only the title, no quotes or punctuation."},
        {"role": "user", "content": user_query[:500]},
    ]

    result = await query_model(TITLE_MODEL, messages)
    return result["content"].strip()
