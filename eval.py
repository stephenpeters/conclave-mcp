#!/usr/bin/env python3
"""
Conclave Eval-Light: Standalone evaluation runner for testing council performance.

Runs a set of benchmark tasks at different council tiers and saves results
to disk for tracking over time. Does not require MCP - runs directly.

Usage:
    python eval.py                    # Run all tests at all tiers
    python eval.py --tier standard    # Run all tests at standard tier only
    python eval.py --quick            # Run quick (Stage 1 only) tests
    python eval.py --task math        # Run specific task category

Results saved to: evals/eval_YYYYMMDD_HHMMSS.json
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Ensure we can import from current directory
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    OPENROUTER_API_KEY,
    COUNCIL_PREMIUM,
    COUNCIL_STANDARD,
    COUNCIL_BUDGET,
    get_current_chairman,
)
from conclave import (
    run_council_quick,
    run_council_ranked,
    run_council_full,
)

# =============================================================================
# EVAL TASKS
# =============================================================================

EVAL_TASKS = [
    # ==========================================================================
    # MATH
    # ==========================================================================
    {
        "id": "math_arithmetic",
        "category": "math",
        "difficulty": "easy",
        "question": "What is 847 × 23? Show your work step by step.",
        "expected_answer": "19481",
        "eval_criteria": "Correct final answer of 19481 with clear working shown",
    },
    {
        "id": "math_word_problem",
        "category": "math",
        "difficulty": "medium",
        "question": "A train leaves Station A at 9:00 AM traveling at 60 mph. Another train leaves Station B (180 miles away) at 10:00 AM traveling toward Station A at 40 mph. At what time do they meet?",
        "expected_answer": "11:12 AM (or 11:11-11:13 range acceptable)",
        "eval_criteria": "Correct time calculation with proper reasoning about relative motion",
    },
    # ==========================================================================
    # CODE
    # ==========================================================================
    {
        "id": "code_debug",
        "category": "code",
        "difficulty": "easy",
        "question": """Find and fix the bug in this Python code:

```python
def calculate_average(numbers):
    total = 0
    for num in numbers:
        total += num
    return total / len(numbers)

result = calculate_average([])
print(result)
```

What's the bug and how would you fix it?""",
        "expected_answer": "ZeroDivisionError when empty list; add check for empty list",
        "eval_criteria": "Identifies division by zero error and provides reasonable fix",
    },
    {
        "id": "code_explain",
        "category": "code",
        "difficulty": "medium",
        "question": "Explain what a decorator is in Python. Give a simple example of a decorator that logs when a function is called.",
        "expected_answer": "Function that wraps another function; working @log example",
        "eval_criteria": "Clear explanation of decorator concept with working code example",
    },
    # ==========================================================================
    # REASONING
    # ==========================================================================
    {
        "id": "reasoning_logic",
        "category": "reasoning",
        "difficulty": "medium",
        "question": "If all Bloops are Razzles, and all Razzles are Lazzles, are all Bloops definitely Lazzles? Explain your reasoning.",
        "expected_answer": "Yes, by transitive property",
        "eval_criteria": "Correct answer with clear logical explanation (syllogism/transitivity)",
    },
    {
        "id": "reasoning_multistep",
        "category": "reasoning",
        "difficulty": "hard",
        "question": """A farmer has a fox, a chicken, and a bag of grain. He needs to cross a river in a boat that can only carry him and one item at a time. If left alone together, the fox will eat the chicken, and the chicken will eat the grain. How can he get everything across safely? List the steps.""",
        "expected_answer": "Classic river crossing: chicken first, return, fox/grain, bring chicken back, other item, return, chicken last",
        "eval_criteria": "Correct sequence of moves that prevents any eating; clear step-by-step solution",
    },
    # ==========================================================================
    # ANALYSIS (Critical thinking)
    # ==========================================================================
    {
        "id": "analysis_argument",
        "category": "analysis",
        "difficulty": "medium",
        "question": """Evaluate this argument and identify any logical flaws:

"Sales of ice cream increase during summer months. Crime rates also increase during summer months. Therefore, eating ice cream causes crime."

What's wrong with this reasoning?""",
        "expected_answer": "Correlation vs causation fallacy; third variable (heat/summer) causes both",
        "eval_criteria": "Identifies correlation vs causation fallacy; mentions confounding variable (weather/heat)",
    },
    {
        "id": "analysis_tradeoffs",
        "category": "analysis",
        "difficulty": "medium",
        "question": "A startup is deciding between building their app as a native mobile app (iOS/Android separately) or using a cross-platform framework like React Native. What are the key tradeoffs they should consider? List 3 pros and 3 cons of each approach.",
        "expected_answer": "Native: better performance, full API access, higher cost; Cross-platform: faster dev, one codebase, some limitations",
        "eval_criteria": "Balanced analysis covering performance, development speed, cost, maintenance, and platform-specific features",
    },
    # ==========================================================================
    # SUMMARIZATION
    # ==========================================================================
    {
        "id": "summarize_technical",
        "category": "summarization",
        "difficulty": "medium",
        "question": """Summarize the following technical document excerpt in 3-4 bullet points, capturing the key information:

The Model Context Protocol (MCP) is an open standard that enables seamless integration between AI applications and external data sources. It provides a unified protocol for connecting language models to tools, databases, and APIs. MCP uses a client-server architecture where the AI application acts as a client connecting to MCP servers that expose resources and capabilities. Key features include: resource discovery allowing clients to find available data sources, tool invocation for executing actions, and prompt templates for structured interactions. The protocol supports both local servers running on the same machine and remote servers accessed over the network. Security is handled through capability-based access control, where servers declare what operations they support and clients request only the capabilities they need. MCP aims to replace the fragmented landscape of custom integrations with a single, standardized approach that any AI application can adopt.""",
        "expected_answer": "Key points: open standard for AI-data integration, client-server architecture, features (resources/tools/prompts), security via capabilities",
        "eval_criteria": "Accurate summary capturing: what MCP is, architecture, key features, and security model",
    },
    {
        "id": "summarize_business",
        "category": "summarization",
        "difficulty": "medium",
        "question": """Summarize this quarterly report excerpt into an executive summary (2-3 sentences):

Q3 2025 showed mixed results for Acme Corp. Revenue increased 12% year-over-year to 4.2 billion dollars, driven primarily by strong performance in the cloud services division which grew 34%. However, the legacy hardware division continued its decline, dropping 18% as customers migrate to cloud solutions. Operating margins compressed by 2 percentage points to 15% due to increased R&D spending on AI initiatives. The company announced plans to acquire DataFlow Inc. for 800 million dollars to accelerate its data analytics capabilities. Customer retention remained strong at 94%, though new customer acquisition slowed compared to Q2. The company reaffirmed its full-year revenue guidance of 17 billion dollars but lowered profit guidance citing continued investment in strategic priorities.""",
        "expected_answer": "Revenue up 12% to 4.2B (cloud strong, hardware declining), margins down due to R&D spend, acquiring DataFlow for 800M",
        "eval_criteria": "Captures key metrics (revenue, growth rates, margins), strategic moves (acquisition), and outlook in concise format",
    },
    # ==========================================================================
    # WRITING - BUSINESS
    # ==========================================================================
    {
        "id": "writing_email",
        "category": "writing_business",
        "difficulty": "easy",
        "question": "Write a professional email (3-4 sentences) declining a meeting request due to a scheduling conflict, while suggesting alternative times next week.",
        "expected_answer": "Professional tone, clear decline, offers alternatives, polite closing",
        "eval_criteria": "Professional tone, clear communication, offers specific alternatives, appropriate length",
    },
    {
        "id": "writing_proposal",
        "category": "writing_business",
        "difficulty": "medium",
        "question": """Write an opening paragraph (4-5 sentences) for a business proposal to a potential client. The proposal is for a website redesign project. The client is a mid-sized law firm that wants to modernize their online presence and improve lead generation.""",
        "expected_answer": "Professional, addresses client needs, hints at solution, establishes credibility",
        "eval_criteria": "Professional tone, addresses specific client pain points, positions value proposition, appropriate for law firm audience",
    },
    # ==========================================================================
    # WRITING - CREATIVE
    # ==========================================================================
    {
        "id": "writing_story_opening",
        "category": "writing_creative",
        "difficulty": "medium",
        "question": "Write the opening paragraph (4-5 sentences) of a mystery story set in a small coastal town. Establish atmosphere and hint at something unusual.",
        "expected_answer": "Atmospheric, sets scene, introduces tension/mystery, engaging hook",
        "eval_criteria": "Evocative description, clear setting, creates intrigue, strong narrative voice",
    },
    {
        "id": "writing_metaphor",
        "category": "writing_creative",
        "difficulty": "easy",
        "question": "Create an original metaphor comparing time to something unexpected (not common metaphors like 'time is money' or 'time flies'). Explain why this metaphor works.",
        "expected_answer": "Original metaphor with thoughtful explanation of the comparison",
        "eval_criteria": "Originality (not cliché), apt comparison, clear explanation of how the metaphor illuminates something about time",
    },
    # ==========================================================================
    # CREATIVE (Original)
    # ==========================================================================
    {
        "id": "creative_analogy",
        "category": "creative",
        "difficulty": "easy",
        "question": "Complete this analogy and explain why: Book is to Library as _____ is to Museum.",
        "expected_answer": "Artifact/Exhibit/Painting (or similar collectible item)",
        "eval_criteria": "Reasonable completion with clear explanation of the relationship",
    },
    # ==========================================================================
    # FACTUAL
    # ==========================================================================
    {
        "id": "factual_science",
        "category": "factual",
        "difficulty": "easy",
        "question": "Why is the sky blue? Explain in 2-3 sentences suitable for a 10-year-old.",
        "expected_answer": "Rayleigh scattering; blue light scattered more than other colors",
        "eval_criteria": "Scientifically accurate but age-appropriate explanation",
    },
]


# =============================================================================
# EVAL RUNNER
# =============================================================================

async def run_single_eval(
    task: dict,
    tier: str = "standard",
    mode: str = "full",
) -> dict:
    """
    Run a single evaluation task.

    Args:
        task: Task definition from EVAL_TASKS
        tier: "premium", "standard", or "budget"
        mode: "quick", "ranked", or "full"

    Returns:
        Eval result with timing and response data
    """
    # Select council models
    if tier == "premium":
        models = COUNCIL_PREMIUM
    elif tier == "budget":
        models = COUNCIL_BUDGET
    else:
        models = COUNCIL_STANDARD

    start_time = datetime.now()

    try:
        if mode == "quick":
            result = await run_council_quick(task["question"], models=models)
        elif mode == "ranked":
            result = await run_council_ranked(task["question"], models=models)
        else:
            result = await run_council_full(task["question"], models=models)

        elapsed = (datetime.now() - start_time).total_seconds()

        return {
            "task_id": task["id"],
            "category": task["category"],
            "difficulty": task["difficulty"],
            "tier": tier,
            "mode": mode,
            "success": True,
            "elapsed_seconds": round(elapsed, 2),
            "question": task["question"],
            "expected": task["expected_answer"],
            "eval_criteria": task["eval_criteria"],
            "result": sanitize_result(result, mode),
        }

    except Exception as e:
        elapsed = (datetime.now() - start_time).total_seconds()
        return {
            "task_id": task["id"],
            "category": task["category"],
            "difficulty": task["difficulty"],
            "tier": tier,
            "mode": mode,
            "success": False,
            "elapsed_seconds": round(elapsed, 2),
            "error": str(e),
        }


def sanitize_result(result: dict, mode: str) -> dict:
    """Extract key information from result for storage."""
    sanitized = {
        "tier": result.get("tier"),
        "models_queried": len(result.get("stage1", [])),
    }

    # Stage 1: Individual responses
    if "stage1" in result:
        sanitized["responses"] = [
            {
                "model": r["model"],
                "content": r["content"][:2000],  # Truncate long responses
            }
            for r in result["stage1"]
        ]

    # Stage 2: Rankings
    if mode in ("ranked", "full") and "stage2" in result:
        sanitized["rankings"] = result["stage2"].get("aggregate", {})

    # Stage 3: Synthesis
    if mode == "full" and "stage3" in result:
        sanitized["synthesis"] = result["stage3"].get("synthesis", "")[:3000]
        sanitized["chairman"] = result["stage3"].get("chairman")
        sanitized["consensus"] = result.get("consensus", {}).get("level")

    return sanitized


async def run_eval_suite(
    tasks: Optional[list] = None,
    tier: str = "standard",
    mode: str = "full",
    category: Optional[str] = None,
) -> dict:
    """
    Run full evaluation suite.

    Args:
        tasks: Specific tasks to run (defaults to all)
        tier: Council tier to use
        mode: "quick", "ranked", or "full"
        category: Filter by category (math, code, reasoning, analysis, summarization, writing_business, writing_creative, creative, factual)

    Returns:
        Complete eval results with metadata
    """
    tasks = tasks or EVAL_TASKS

    # Filter by category if specified
    if category:
        tasks = [t for t in tasks if t["category"] == category]

    if not tasks:
        return {"error": "No tasks match the specified criteria"}

    print(f"\n🏛️  Conclave Eval-Light")
    print(f"   Tier: {tier} | Mode: {mode} | Tasks: {len(tasks)}")
    print("-" * 50)

    results = []
    for i, task in enumerate(tasks, 1):
        print(f"\n[{i}/{len(tasks)}] Running: {task['id']} ({task['category']})")

        result = await run_single_eval(task, tier=tier, mode=mode)
        results.append(result)

        if result["success"]:
            print(f"   ✓ Completed in {result['elapsed_seconds']}s")
        else:
            print(f"   ✗ Failed: {result.get('error', 'Unknown error')}")

    # Calculate summary stats
    successful = [r for r in results if r["success"]]
    total_time = sum(r["elapsed_seconds"] for r in results)

    summary = {
        "total_tasks": len(tasks),
        "successful": len(successful),
        "failed": len(tasks) - len(successful),
        "total_time_seconds": round(total_time, 2),
        "avg_time_seconds": round(total_time / len(tasks), 2) if tasks else 0,
    }

    return {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "tier": tier,
            "mode": mode,
            "category_filter": category,
            "chairman": get_current_chairman(),
        },
        "summary": summary,
        "results": results,
    }


def save_eval_results(eval_data: dict, output_dir: str = "evals") -> str:
    """Save evaluation results to disk."""
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tier = eval_data.get("metadata", {}).get("tier", "unknown")
    mode = eval_data.get("metadata", {}).get("mode", "unknown")
    filename = f"eval_{tier}_{mode}_{timestamp}.json"

    filepath = output_path / filename

    with open(filepath, "w") as f:
        json.dump(eval_data, f, indent=2, default=str)

    return str(filepath)


def print_summary(eval_data: dict):
    """Print a summary of eval results."""
    summary = eval_data.get("summary", {})
    metadata = eval_data.get("metadata", {})

    print("\n" + "=" * 50)
    print("📊 EVAL SUMMARY")
    print("=" * 50)
    print(f"Tier: {metadata.get('tier')} | Mode: {metadata.get('mode')}")
    print(f"Chairman: {metadata.get('chairman', 'N/A')}")
    print(f"Tasks: {summary.get('successful')}/{summary.get('total_tasks')} successful")
    print(f"Total time: {summary.get('total_time_seconds')}s")
    print(f"Avg per task: {summary.get('avg_time_seconds')}s")

    # Show individual results
    print("\n📋 Results by Task:")
    for r in eval_data.get("results", []):
        status = "✓" if r.get("success") else "✗"
        print(f"  {status} {r['task_id']} ({r['difficulty']}) - {r['elapsed_seconds']}s")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Conclave Eval-Light: Benchmark council performance"
    )
    parser.add_argument(
        "--tier",
        choices=["premium", "standard", "budget"],
        default="standard",
        help="Council tier to use (default: standard)"
    )
    parser.add_argument(
        "--mode",
        choices=["quick", "ranked", "full"],
        default="full",
        help="Eval mode (default: full)"
    )
    parser.add_argument(
        "--category",
        choices=[
            "math", "code", "reasoning", "analysis", "summarization",
            "writing_business", "writing_creative", "creative", "factual"
        ],
        help="Run only tasks in this category"
    )
    parser.add_argument(
        "--output",
        default="evals",
        help="Output directory for results (default: evals)"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to disk"
    )

    args = parser.parse_args()

    # Check API key
    if not OPENROUTER_API_KEY:
        print("Error: OPENROUTER_API_KEY not set")
        print("Set it in .env or as environment variable")
        sys.exit(1)

    # Run evaluation
    eval_data = asyncio.run(
        run_eval_suite(
            tier=args.tier,
            mode=args.mode,
            category=args.category,
        )
    )

    # Print summary
    print_summary(eval_data)

    # Save results
    if not args.no_save:
        filepath = save_eval_results(eval_data, args.output)
        print(f"\n💾 Results saved to: {filepath}")


if __name__ == "__main__":
    main()
