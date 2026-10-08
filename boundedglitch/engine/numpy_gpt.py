import numpy as np

SPECIAL = 4  # <PAD> <BOS> <EOS> <UNK>


def _ln(x, w, b, eps=1e-5):
    m = x.mean(-1, keepdims=True)
    v = x.var(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * w + b


def _softmax(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


class NumpyGPT:
    def __init__(self, path):
        d = np.load(path, allow_pickle=False)
        self.w = {k: d[k] for k in d.files}
        self.itos = [str(s) for s in self.w["itos"]]
        self.stoi = {s: i for i, s in enumerate(self.itos)}
        self.heads = int(self.w["cfg_num_heads"])
        self.layers = int(self.w["cfg_num_layers"])
        self.ctx = int(self.w["cfg_max_context"])
        self.vocab = len(self.itos)
        self.pe = self.w["positional_embedding.pe"][0]

    def encode(self, text):
        unk = self.stoi["<UNK>"]
        return [self.stoi.get(c, unk) for c in text]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids if i >= SPECIAL)

    def _lin(self, name, h):
        return h @ self.w[name + ".weight"].T + self.w[name + ".bias"]

    def forward_last(self, ids):
        w = self.w
        T = len(ids)
        D = w["token_embedding.weight"].shape[1]
        H = self.heads
        hd = D // H
        x = w["token_embedding.weight"][ids] + self.pe[:T]
        mask = np.triu(np.full((T, T), -1e9), k=1)
        for i in range(self.layers):
            p = f"transformer_blocks.{i}."
            n = _ln(x, w[p + "norm1.weight"], w[p + "norm1.bias"])

            def split(name):
                return self._lin(p + "attention." + name, n).reshape(T, H, hd).transpose(1, 0, 2)

            q, k, v = split("query_proj"), split("key_proj"), split("value_proj")
            att = _softmax(q @ k.transpose(0, 2, 1) / np.sqrt(hd) + mask)
            ctx = (att @ v).transpose(1, 0, 2).reshape(T, D)
            x = x + self._lin(p + "attention.output_proj", ctx)
            n2 = _ln(x, w[p + "norm2.weight"], w[p + "norm2.bias"])
            x = x + self._lin(p + "ffn.fc2", np.maximum(0, self._lin(p + "ffn.fc1", n2)))
        x = _ln(x, w["final_norm.weight"], w["final_norm.bias"])
        return self._lin("output_head", x[-1])

    def generate(self, prompt, max_new_tokens=100, temperature=0.8, top_k=20, stop=None):
        ids = self.encode(prompt) or [self.stoi["<BOS>"]]
        out = []
        for _ in range(max_new_tokens):
            logits = self.forward_last(np.array(ids[-self.ctx:]))
            logits[:SPECIAL] = -1e9
            logits = logits / max(temperature, 1e-3)
            if top_k and top_k < len(logits):
                cut = np.sort(logits)[-top_k]
                logits[logits < cut] = -1e9
            nxt = int(np.random.choice(len(logits), p=_softmax(logits)))
            ids.append(nxt)
            out.append(nxt)
            if stop and stop in self.decode(out):
                break
        text = self.decode(out)
        return text.split(stop)[0] if stop else text
