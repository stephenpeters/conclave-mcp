#!/usr/bin/env python3
"""
Test script for council-mcp.

Run with: python test_council.py

Requires OPENROUTER_API_KEY in .env or environment.
"""

import asyncio
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Check API key
if not os.getenv("OPENROUTER_API_KEY"):
    print("❌ OPENROUTER_API_KEY not set")
    print("   Create a .env file with: OPENROUTER_API_KEY=sk-or-v1-...")
    print("   Get your key at: https://openrouter.ai/keys")
    exit(1)

print("✓ API key configured")

from config import (
    COUNCIL_MODELS,
    get_current_chairman,
    get_rotation_info,
    validate_council_size,
    estimate_cost,
)

# Test config
print(f"✓ Council models: {len(COUNCIL_MODELS)}")
for m in COUNCIL_MODELS:
    print(f"    - {m}")

rotation = get_rotation_info()
print(f"✓ Current chairman: {rotation['current_chairman']}")
print(f"    Rotation: {'enabled' if rotation['rotation_enabled'] else 'disabled'}")
print(f"    Days until next: {rotation['days_until_rotation']}")

size = validate_council_size()
print(f"✓ Council size: {size['total_size']} ({'odd ✓' if size['valid'] else 'even ⚠️'})")

# Cost estimate
est = estimate_cost(100, tier="full")
print(f"✓ Estimated cost (full council): ${est['total']:.4f}")

# Test actual API call (optional - costs money)
async def test_quick_query():
    from conclave import run_council_quick

    print("\n--- Testing council_quick (will call OpenRouter API) ---")
    print("Question: What is 2+2?")

    try:
        result = await run_council_quick("What is 2+2? Answer briefly.")
        print(f"✓ Got {len(result['stage1'])} responses:")
        for resp in result['stage1']:
            model = resp['model'].split('/')[-1]
            content = resp['content'][:100].replace('\n', ' ')
            print(f"    {model}: {content}...")
        print("\n✓ API integration working!")
    except Exception as e:
        print(f"❌ API error: {e}")

if __name__ == "__main__":
    print("\n" + "="*50)
    response = input("Run live API test? (costs ~$0.01) [y/N]: ")
    if response.lower() == 'y':
        asyncio.run(test_quick_query())
    else:
        print("Skipped live test. Config validation passed!")
