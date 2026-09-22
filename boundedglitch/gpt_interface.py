"""GPT Interface: runs a trained BoundedGlitchGPT checkpoint with numpy only."""
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = "data/model/weights.npz"


class SimpleTokenizer:
    """Character tokenizer rebuilt from the vocab saved at export time."""

    def __init__(self, itos):
        self.itos = list(itos)
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self.unk = self.stoi.get("<UNK>", 0)
        self.special = [i for i, t in enumerate(self.itos)
                        if len(t) > 1 and t.startswith("<") and t.endswith(">")]

    def encode(self, text):
        return [self.stoi.get(c, self.unk) for c in text]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)


def _layer_norm(x, w, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


def _softmax(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


class GPTModel:
    """Numpy forward pass matching model/model.py (pre-norm GPT, no dropout)."""

    def __init__(self, path):
        data = np.load(path)
        self.itos = [str(t) for t in data["itos"]]
        self.num_heads = int(data["cfg_num_heads"])
        self.num_layers = int(data["cfg_num_layers"])
        self.max_ctx = int(data["cfg_max_context"])
        self.w = {k: data[k].astype(np.float32) for k in data.files
                  if k != "itos" and not k.startswith("cfg_")}
        self.pe = self.w.pop("positional_embedding.pe")[0]
        self.num_params = sum(v.size for v in self.w.values())

    def _logits(self, ids):
        W, H = self.w, self.num_heads
        T = len(ids)
        x = W["token_embedding.weight"][ids] + self.pe[:T]
        mask = np.tril(np.ones((T, T), dtype=bool))
        for i in range(self.num_layers):
            p = f"transformer_blocks.{i}."
            h = _layer_norm(x, W[p + "norm1.weight"], W[p + "norm1.bias"])
            q, k, v = (
                (h @ W[p + f"attention.{n}_proj.weight"].T
                 + W[p + f"attention.{n}_proj.bias"])
                .reshape(T, H, -1).transpose(1, 0, 2)
                for n in ("query", "key", "value")
            )
            s = (q @ k.transpose(0, 2, 1)) / np.sqrt(q.shape[-1])
            a = _softmax(np.where(mask, s, -np.inf))
            ctx = (a @ v).transpose(1, 0, 2).reshape(T, -1)
            x = x + ctx @ W[p + "attention.output_proj.weight"].T \
                + W[p + "attention.output_proj.bias"]
            h = _layer_norm(x, W[p + "norm2.weight"], W[p + "norm2.bias"])
            h = np.maximum(h @ W[p + "ffn.fc1.weight"].T + W[p + "ffn.fc1.bias"], 0)
            x = x + h @ W[p + "ffn.fc2.weight"].T + W[p + "ffn.fc2.bias"]
        x = _layer_norm(x[-1], W["final_norm.weight"], W["final_norm.bias"])
        return x @ W["output_head.weight"].T + W["output_head.bias"]

    def generate(self, ids, max_tokens=100, temperature=0.8, banned=()):
        ids, new = list(ids), []
        for _ in range(max_tokens):
            logits = self._logits(np.array(ids[-self.max_ctx:]))
            if len(banned):
                logits[list(banned)] = -np.inf
            if temperature <= 0:
                nxt = int(logits.argmax())
            else:
                p = _softmax(logits / temperature).astype(np.float64)
                nxt = int(np.random.choice(len(p), p=p / p.sum()))
            ids.append(nxt)
            new.append(nxt)
        return new


class GPTInterface:
    """Loads weights.npz exported from The-BoundedGlitchGPT and generates text."""

    def __init__(self, config):
        self.config = config
        self.model_config = config.get("model", {})
        path = Path(self.model_config.get("checkpoint", DEFAULT_CHECKPOINT))
        if not path.is_absolute():
            path = ROOT / path
        self.model = self.tokenizer = None
        if path.exists():
            self.model = GPTModel(path)
            self.tokenizer = SimpleTokenizer(self.model.itos)
            print(f"[✓] GPTInterface loaded {path.name} ({self.model.num_params:,} params)")
        else:
            print(f"[!] No weights at {path}. Train + export first; replies will be placeholders.")

    def generate(self, prompt, persona="bosk", max_tokens=100, **kwargs):
        if self.model is None:
            return "[no trained model loaded]"
        temperature = self.config["personas"].get(persona, {}).get("temperature", 0.8)
        ids = self.tokenizer.encode(prompt) or self.tokenizer.encode("\n")
        new = self.model.generate(ids, max_tokens=max_tokens, temperature=temperature,
                                  banned=self.tokenizer.special)
        return self.tokenizer.decode(new).strip()
