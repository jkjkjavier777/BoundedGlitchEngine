"""BoundedGlitchEngine: chat commands, history, and the model/inference glue."""

import threading

from inference import generate
from model import KnowledgeModel

HELP = ("teach: question = answer | forget: question | train | stats. "
        "Drop .txt/.md files into data/corpus/ then say 'train'.")


class BoundedGlitchEngine:
    def __init__(self, data_dir=None):
        self.model = KnowledgeModel(data_dir)
        self.history = []
        self.lock = threading.RLock()

    def respond(self, message: str) -> str:
        message = message.strip()
        if not message:
            return "Input required."
        with self.lock:
            self._log("user", message)
            reply = self._command(message) or generate(self.model, message)
            self._log("assistant", reply)
        return reply

    def teach(self, question, answer):
        with self.lock:
            return self.model.teach(question, answer)

    def train(self):
        with self.lock:
            return self.model.train()

    def stats(self):
        with self.lock:
            return {**self.model.stats(), "turns": len(self.history)}

    def _command(self, message):
        low = message.lower()
        if low.startswith("teach:"):
            q, sep, a = message[6:].partition("=")
            if not sep or not q.strip() or not a.strip():
                return "Format: teach: your question = your answer"
            return self.model.teach(q, a.strip())
        if low.startswith("forget:"):
            return self.model.forget(message[7:])
        if low in ("train", "/train"):
            return self.model.train()
        if low in ("stats", "/stats"):
            return ", ".join(f"{k}: {v}" for k, v in self.stats().items())
        if low in ("help", "/help"):
            return HELP
        return None

    def _log(self, role, content):
        self.history.append({"role": role, "content": content})
        del self.history[:-200]
