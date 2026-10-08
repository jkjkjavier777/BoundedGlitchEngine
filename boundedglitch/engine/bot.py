"""BoundedGlitchEngine: taught answers first, GPT fallback, every reply tagged."""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.brain import load_replies, find_best_match, _pick_non_repeating, teach
from core.telemetry import log_event
from .gpt_interface import GPTInterface


class BoundedGlitchEngine:
    def __init__(self, weights_path=None):
        self.gpt = None
        try:
            self.gpt = GPTInterface(weights_path) if weights_path else GPTInterface()
        except Exception as e:
            print(f"[!] GPT unavailable, using taught replies only: {e}")
        self.conversation_history = []
        self.last_meta = {}

    def chat(self, user_input, max_tokens=100, temperature=0.8):
        user_input = user_input.strip()
        self.conversation_history.append(("user", user_input))
        t0 = time.time()
        source, matched_key = None, None

        if user_input.lower().startswith("teach:"):
            parts = user_input[6:].split("=", 1)
            reply = teach(*parts) if len(parts) == 2 else "Format: teach: your phrase = your answer"
            source = "teach"
        else:
            replies = load_replies()
            match = find_best_match(user_input, replies)
            if match and match != "default":
                reply = _pick_non_repeating(match, replies[match])
                source, matched_key = "taught", match
            elif self.gpt:
                reply = self.gpt.generate(user_input, max_tokens, temperature)
                source = "gpt"
            elif replies.get("default"):
                reply = _pick_non_repeating("default", replies["default"])
                source = "default"
            else:
                reply = "I don't understand. Teach me with: teach: your phrase = your answer"
                source = "default"

        latency_ms = round((time.time() - t0) * 1000)
        self.last_meta = {"source": source, "matched_key": matched_key}
        log_event({
            "prompt": user_input,
            "reply": reply,
            "source": source,
            "matched_key": matched_key,
            "latency_ms": latency_ms,
            "reply_chars": len(reply),
            "temperature": temperature if source == "gpt" else None,
        })
        self.conversation_history.append(("bot", reply))
        return reply

    def interactive_mode(self):
        print("BoundedGlitchEngine. 'quit' to exit.\n")
        while True:
            try:
                text = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n[*] Goodbye!")
                break
            if not text:
                continue
            if text.lower() in ("quit", "exit"):
                break
            reply = self.chat(text)
            print(f"\nBot [{self.last_meta['source']}]: {reply}\n")
