#!/usr/bin/env python3
"""
COMPLETE_SETUP.py — Verify and create all required files

Run this to set up the entire bot from scratch.
"""

import os
from pathlib import Path


def create_file(path, content):
    """Create a file with content."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w") as f:
        f.write(content)
    
    print(f"✓ Created {path}")


def main():
    """Set up all files."""
    root = Path(__file__).parent
    
    print("Setting up BoundedGlitchEngine...")
    print()
    
    # Create boundedglitch/__init__.py
    create_file(
        root / "boundedglitch" / "__init__.py",
        '''"""BoundedGlitchEngine package."""
__version__ = "1.0.0"

from .config import load_config
from .conversation import ConversationManager

__all__ = ["load_config", "ConversationManager"]
'''
    )
    
    # Create boundedglitch/config.py
    create_file(
        root / "boundedglitch" / "config.py",
        '''"""Configuration loader."""
import json
from pathlib import Path

def load_config(config_path="bot.json"):
    """Load configuration from JSON file."""
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_file, "r") as f:
        config = json.load(f)
    
    print(f"[✓] Configuration loaded from {config_path}")
    return config
'''
    )
    
    # Create boundedglitch/conversation.py
    create_file(
        root / "boundedglitch" / "conversation.py",
        '''"""Conversation manager."""
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

class ConversationManager:
    """Manage stateful conversations."""

    def __init__(self, config):
        self.config = config
        self.save_dir = Path(config["conversation"]["save_dir"])
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.sessions = {}
        print(f"[✓] ConversationManager initialized")

    def add_message(self, session_id: str, role: str, content: str, 
                   persona: Optional[str] = None, metadata: Optional[Dict] = None):
        """Add message to conversation history."""
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "messages": [],
                "persona": persona,
                "created_at": datetime.now().isoformat(),
                "last_message_at": datetime.now().isoformat()
            }
        
        message = {
            "timestamp": datetime.now().isoformat(),
            "role": role,
            "content": content,
            "persona": persona,
            "metadata": metadata or {}
        }
        
        self.sessions[session_id]["messages"].append(message)
        self.sessions[session_id]["last_message_at"] = datetime.now().isoformat()
        self._save_session(session_id)

    def get_context(self, session_id: str, max_turns: int = 5) -> List[Dict]:
        """Get conversation context (sliding window)."""
        if session_id not in self.sessions:
            return []
        messages = self.sessions[session_id]["messages"]
        return messages[-(max_turns * 2):]

    def get_full_history(self, session_id: str) -> List[Dict]:
        """Get entire conversation history."""
        if session_id not in self.sessions:
            return []
        return self.sessions[session_id]["messages"]

    def clear_history(self, session_id: str):
        """Clear conversation history."""
        if session_id in self.sessions:
            del self.sessions[session_id]

    def _save_session(self, session_id: str):
        """Save session to disk."""
        session_file = self.save_dir / f"{session_id}.json"
        with open(session_file, "w") as f:
            json.dump(self.sessions[session_id], f, indent=2)
'''
    )
    
    # Create boundedglitch/engine.py
    create_file(
        root / "boundedglitch" / "engine.py",
        '''"""BoundedGlitchEngine - 5-layer system."""
from typing import Dict, List

class Layer:
    """Base class for engine layers."""
    def __init__(self, name: str, config: Dict):
        self.name = name
        self.config = config
    
    def process(self, data: Dict) -> Dict:
        raise NotImplementedError

class EngagementLayer(Layer):
    def __init__(self, config: Dict):
        super().__init__("engagement", config)
    
    def process(self, data: Dict) -> Dict:
        data["engagement_analysis"] = {"status": "ok"}
        return data

class RetrievalLayer(Layer):
    def __init__(self, config: Dict):
        super().__init__("retrieval", config)
    
    def process(self, data: Dict) -> Dict:
        data["retrieved_knowledge"] = {"source": "knowledge_base"}
        return data

class ReasoningLayer(Layer):
    def __init__(self, config: Dict):
        super().__init__("reasoning", config)
    
    def process(self, data: Dict) -> Dict:
        data["reasoning_analysis"] = {"perspectives": 5}
        return data

class ValidationLayer(Layer):
    def __init__(self, config: Dict):
        super().__init__("validation", config)
    
    def process(self, data: Dict) -> Dict:
        data["validation_results"] = {"passes": True}
        return data

class SynthesisLayer(Layer):
    def __init__(self, config: Dict):
        super().__init__("synthesis", config)
    
    def process(self, data: Dict) -> Dict:
        user_input = data.get("user_input", "")
        persona = data.get("persona", "bosk")
        processed_prompt = f"You are a {persona} agent.\\n\\nUser: {user_input}\\n\\nResponse:"
        data["processed_prompt"] = processed_prompt
        return data

class BoundedGlitchEngine:
    """Main engine."""
    def __init__(self, config: Dict):
        self.config = config
        self.layers = [
            EngagementLayer(config),
            RetrievalLayer(config),
            ReasoningLayer(config),
            ValidationLayer(config),
            SynthesisLayer(config)
        ]
        print("[✓] BoundedGlitchEngine initialized")

    def process(self, user_input: str, context: List[Dict], persona: str) -> Dict:
        """Run through all 5 layers."""
        data = {"user_input": user_input, "context": context, "persona": persona}
        for layer in self.layers:
            data = layer.process(data)
        return data
'''
    )
    
    # Create boundedglitch/gpt_interface.py
    create_file(
        root / "boundedglitch" / "gpt_interface.py",
        '''"""GPT Interface."""
import numpy as np
from typing import Optional

class SimpleTokenizer:
    """Simple tokenizer."""
    def __init__(self, vocab_size: int = 256):
        self.vocab_size = vocab_size
    
    def encode(self, text: str) -> np.ndarray:
        return np.array([ord(c) % self.vocab_size for c in text], dtype=np.int32)
    
    def decode(self, token_ids: np.ndarray) -> str:
        return ''.join([chr(int(tid)) if tid < 128 else '?' for tid in token_ids])

class GPTModel:
    """GPT model."""
    def __init__(self, config: dict):
        self.config = config
        self.vocab_size = config.get("vocab_size", 256)
    
    def generate(self, token_ids: np.ndarray, max_tokens: int = 100, 
                temperature: float = 0.8) -> np.ndarray:
        tokens = list(token_ids)
        for _ in range(max_tokens):
            tokens.append(np.random.randint(0, self.vocab_size))
        return np.array(tokens, dtype=np.int32)

class GPTInterface:
    """Interface to GPT."""
    def __init__(self, config: dict):
        self.config = config
        self.model_config = config["model"]
        self.tokenizer = SimpleTokenizer(self.model_config.get("vocab_size", 256))
        self.model = GPTModel(self.model_config)
        print("[✓] GPTInterface initialized")

    def generate(self, prompt: str, persona: str = "bosk", 
                max_tokens: int = 150, temperature: Optional[float] = None) -> str:
        """Generate text."""
        if temperature is None:
            temperature = self.config["personas"].get(persona, {}).get("temperature", 0.8)
        prompt_ids = self.tokenizer.encode(prompt)
        generated_ids = self.model.generate(prompt_ids, max_tokens=max_tokens, temperature=temperature)
        return self.tokenizer.decode(generated_ids)[-50:].strip()
'''
    )
    
    # Create boundedglitch/knowledge.py
    create_file(
        root / "boundedglitch" / "knowledge.py",
        '''"""Knowledge base management."""
import json
from pathlib import Path

class KnowledgeBase:
    """Knowledge base."""
    def __init__(self, kb_path: str):
        self.kb_path = Path(kb_path)
        self.knowledge = {}
        if self.kb_path.exists():
            with open(self.kb_path) as f:
                self.knowledge = json.load(f)

class KnowledgeManager:
    """Manage knowledge bases."""
    def __init__(self, config: dict):
        self.config = config
        self.knowledge_bases = {}
        for persona_name, persona_config in config["personas"].items():
            kb_path = persona_config.get("knowledge_base")
            if kb_path:
                self.knowledge_bases[persona_name] = KnowledgeBase(kb_path)
'''
    )
    
    # Create boundedglitch/engine/__init__.py
    create_file(
        root / "boundedglitch" / "engine" / "__init__.py",
        '''"""Engine subpackage."""
'''
    )
    
    # Create boundedglitch/gpt/__init__.py
    create_file(
        root / "boundedglitch" / "gpt" / "__init__.py",
        '''"""GPT subpackage."""
'''
    )
    
    print()
    print("✅ All files created!")
    print()
    print("Now run:")
    print("  python bot.py")


if __name__ == "__main__":
    main()

