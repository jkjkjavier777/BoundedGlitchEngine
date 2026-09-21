"""Knowledge base management."""
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
