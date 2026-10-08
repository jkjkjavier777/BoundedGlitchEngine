"""GPTInterface: boundary between the engine and the numpy GPT models.

Primary:  The-BoundedGlitchGPT checkpoint  (data/model/glitchgpt.npz)
Fallback: legacy NumpyGPT                  (data/model/weights.npz)
"""
from pathlib import Path

import numpy as np

DATA_MODEL = Path(__file__).resolve().parents[2] / "data" / "model"
DEFAULT_WEIGHTS = DATA_MODEL / "weights.npz"    # legacy model
GLITCH_WEIGHTS = DATA_MODEL / "glitchgpt.npz"   # new model
DIALOGUE = True   # legacy path only: "You:/Bot:" prompt format
CHUNK = 16        # generate this many chars at a time, stop at end of reply


class GPTInterface:
    def __init__(self, weights_path=GLITCH_WEIGHTS):
        self.weights_path = str(weights_path)
        self.model = None
        self.tok = None
        self.backend = None
        self._rng = np.random.default_rng()
        try:
            from .glitch_model import GPT
            self.model, self.tok = GPT.load(self.weights_path)
            self.backend = "glitchgpt"
        except Exception as e:
            print(f"[GPTInterface] glitchgpt failed to load ({e}); using legacy weights")
            from .numpy_gpt import NumpyGPT
            self.weights_path = str(DEFAULT_WEIGHTS)
            self.model = NumpyGPT(self.weights_path)
            self.backend = "numpy_gpt"

    def generate(self, prompt, max_tokens=100, temperature=0.8):
        if self.backend == "glitchgpt":
            return self._generate_glitch(prompt, max_tokens, temperature)
        return self._generate_legacy(prompt, max_tokens, temperature)

    # ---- new model ---------------------------------------------------------

    def _generate_glitch(self, prompt, max_tokens, temperature):
        prompt = " ".join(str(prompt).split())
        ctx = self.model.config.block_size
        head, tail = "USER: ", "\nGPT:"
        room = max(8, ctx - len(head) - len(tail) - 16)
        ids = self.tok.encode(head + prompt[-room:] + tail)
        n_prompt = len(ids)

        out = list(ids)
        remaining = int(max_tokens)
        while remaining > 0:
            step = min(CHUNK, remaining)
            out = self.model.generate(out, max_new_tokens=step,
                                      temperature=temperature, top_k=20,
                                      rng=self._rng)
            remaining -= step
            if "\n" in self.tok.decode(out[n_prompt:]):
                break

        reply = self.tok.decode(out[n_prompt:]).split("\n")[0].strip()
        return reply or "..."

    # ---- legacy model (unchanged behavior) ---------------------------------

    def _generate_legacy(self, prompt, max_tokens, temperature):
        if DIALOGUE:
            prompt = " ".join(prompt.split())
            text = self.model.generate(f"You: {prompt}\nBot:", max_tokens,
                                       temperature, stop="\n")
            return text.strip() or "..."
        text = self.model.generate(prompt, max_tokens, temperature).strip()
        cut = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
        return text[: cut + 1] if cut > 20 else text

    def get_info(self):
        if self.backend == "glitchgpt":
            c = self.model.config
            return {"weights": self.weights_path, "vocab": self.tok.vocab_size,
                    "layers": c.n_layers, "heads": c.n_heads,
                    "context": c.block_size, "backend": self.backend}
        m = self.model
        return {"weights": self.weights_path, "vocab": m.vocab,
                "layers": m.layers, "heads": m.heads, "context": m.ctx,
                "backend": self.backend}
