import json
import time
from config import CONVERSATIONS_PATH


def load_conversations():
    try:
        with open(CONVERSATIONS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_conversations(conversations):
    with open(CONVERSATIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(conversations, f, indent=2)


def remember(role, content):
    conversations = load_conversations()
    conversations.append({"role": role, "content": content, "timestamp": time.time()})
    save_conversations(conversations)
