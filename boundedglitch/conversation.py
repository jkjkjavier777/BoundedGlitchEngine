"""Conversation manager."""
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
