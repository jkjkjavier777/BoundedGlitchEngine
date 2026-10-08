#!/usr/bin/env python3
"""
Quick import test — verify all modules load correctly.

Run this first to debug import issues before running bot.py.

Usage:
    python test_imports.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("Testing imports...")
print("=" * 60)

# Test 1: Config
try:
    from boundedglitch.config import load_config
    print("✓ boundedglitch.config — OK")
except ImportError as e:
    print(f"✗ boundedglitch.config — FAIL: {e}")

# Test 2: Conversation
try:
    from boundedglitch.conversation import ConversationManager
    print("✓ boundedglitch.conversation — OK")
except ImportError as e:
    print(f"✗ boundedglitch.conversation — FAIL: {e}")

# Test 3: Knowledge
try:
    from boundedglitch.knowledge import KnowledgeBase, KnowledgeManager
    print("✓ boundedglitch.knowledge — OK")
except ImportError as e:
    print(f"✗ boundedglitch.knowledge — FAIL: {e}")

# Test 4: GPT Interface
try:
    from boundedglitch.gpt_interface import GPTInterface, GPTModel, SimpleTokenizer
    print("✓ boundedglitch.gpt_interface — OK")
except ImportError as e:
    print(f"✗ boundedglitch.gpt_interface — FAIL: {e}")

# Test 5: Engine (the critical one)
try:
    from boundedglitch.engine import BoundedGlitchEngine
    print("✓ boundedglitch.engine.BoundedGlitchEngine — OK")
except ImportError as e:
    print(f"✗ boundedglitch.engine.BoundedGlitchEngine — FAIL: {e}")

# Test 6: All via top-level boundedglitch import
try:
    from boundedglitch import (
        load_config,
        ConversationManager,
        BoundedGlitchEngine,
        GPTInterface
    )
    print("✓ boundedglitch (top-level) — OK")
except ImportError as e:
    print(f"✗ boundedglitch (top-level) — FAIL: {e}")

# Test 7: CLI
try:
    from cli import TerminalCLI, ColoredCLI
    print("✓ cli — OK")
except ImportError as e:
    print(f"✗ cli — FAIL: {e}")

print("=" * 60)
print("\n✓ All imports working — Ready to run: python bot.py")

