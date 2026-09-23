#!/usr/bin/env python3
"""
Train BoundedGlitchGPT and export weights.npz — run entirely from Colab.

Usage in Colab (one cell):
    !git clone https://github.com/jkjkjavier777/BoundedGlitchEngine
    %cd BoundedGlitchEngine
    !python train/colab_train.py

Produces weights.npz in the numpy format boundedglitch/gpt_interface.py
already expects — no .pt file, no separate vocab.json, no manual key
matching. Downloads automatically when done (if running in Colab).
"""
import glob
import math
import subprocess
import sys

try:
    import torch
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "-q"])
    import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ---------------------------------------------------------------------------
# Config: small and phone-friendly. Real vocab (77 chars), not GPT-2 sized.
# ---------------------------------------------------------------------------
VOCAB_SIZE = 77
MAX_CONTEXT = 128
EMBED_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 4
FFN_HIDDEN = 512
DROPOUT = 0.1
STEPS = 3000
BATCH_SIZE = 64
LR = 1e-3
OUT_PATH = "weights.npz"


class SimpleTokenizer:
    def __init__(self, vocab_size=VOCAB_SIZE):
        self.vocab_size = vocab_size
        self.stoi, self.itos = {}, {}
        chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?;:'\"-\n"
        idx = 0
        for tok in ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]:
            self.stoi[tok] = idx; self.itos[idx] = tok; idx += 1
        for c in sorted(set(chars)):
            if c not in self.stoi:
                self.stoi[c] = idx; self.itos[idx] = c; idx += 1

    def encode(self, text):
        unk = self.stoi["<UNK>"]
        return [self.stoi.get(c, unk) for c in text]

    def decode(self, ids):
        return "".join(self.itos.get(i, "<UNK>") for i in ids)


class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, dim):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2).float() * -(math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads, self.head_dim = heads, dim // heads
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.output_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, q, k, v, mask=None):
        B, T, D = q.size()
        Q = self.query_proj(q).view(B, T, self.heads, self.head_dim).transpose(1, 2)
        K = self.key_proj(k).view(B, T, self.heads, self.head_dim).transpose(1, 2)
        V = self.value_proj(v).view(B, T, self.heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = self.dropout(F.softmax(scores, dim=-1))
        ctx = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, T, D)
        return self.output_proj(ctx)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden, dropout):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, ffn_hidden, dropout):
        super().__init__()
        self.attention = MultiHeadAttention(dim, heads, dropout)
        self.ffn = FeedForward(dim, ffn_hidden, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        n1 = self.norm1(x)
        x = x + self.dropout(self.attention(n1, n1, n1, mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.positional_embedding = PositionalEmbedding(MAX_CONTEXT, EMBED_DIM)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(EMBED_DIM, NUM_HEADS, FFN_HIDDEN, DROPOUT) for _ in range(NUM_LAYERS)]
        )
        self.final_norm = nn.LayerNorm(EMBED_DIM)
        self.output_head = nn.Linear(EMBED_DIM, VOCAB_SIZE)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, ids, targets=None):
        T = ids.size(1)
        x = self.token_embedding(ids)
        x = self.positional_embedding(x)
        mask = torch.tril(torch.ones(T, T, device=ids.device)).view(1, 1, T, T)
        for block in self.transformer_blocks:
            x = block(x, mask)
        x = self.final_norm(x)
        logits = self.output_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, ids, n, temperature=0.8):
        for _ in range(n):
            ctx = torch.tensor([ids[-MAX_CONTEXT:]], dtype=torch.long, device=next(self.parameters()).device)
            logits, _ = self(ctx)
            probs = F.softmax(logits[0, -1] / temperature, dim=-1)
            ids.append(torch.multinomial(probs, 1).item())
        return ids


def load_corpus():
    files = sorted(glob.glob("data/corpus/*.txt"))
    if not files:
        raise SystemExit("No corpus files found at data/corpus/*.txt — check you're in the repo root.")
    text = ""
    for f in files:
        text += open(f, encoding="utf-8", errors="ignore").read() + "\n"
    print(f"Loaded {len(files)} files, {len(text):,} characters")
    return text


def main():
    tok = SimpleTokenizer()
    text = load_corpus()
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    unk_frac = (data == tok.stoi["<UNK>"]).float().mean().item()
    print(f"{len(data):,} tokens, {unk_frac:.1%} unknown chars")

    split = int(0.9 * len(data))
    train, val = data[:split], data[split:]
    if len(val) <= MAX_CONTEXT + 2:
        raise SystemExit(
            f"Corpus too small ({len(data)} chars) for context {MAX_CONTEXT}. "
            "Add more text to data/corpus/*.txt."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}")
    model = GPT().to(device)
    print(f"{sum(p.numel() for p in model.parameters()):,} parameters")
    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    def batch(d):
        ix = torch.randint(len(d) - MAX_CONTEXT - 1, (BATCH_SIZE,))
        x = torch.stack([d[i : i + MAX_CONTEXT] for i in ix])
        y = torch.stack([d[i + 1 : i + MAX_CONTEXT + 1] for i in ix])
        return x.to(device), y.to(device)

    for step in range(1, STEPS + 1):
        model.train()
        x, y = batch(train)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 250 == 0 or step == STEPS:
            model.eval()
            with torch.no_grad():
                _, vloss = model(*batch(val))
            print(f"step {step}/{STEPS}: train {loss.item():.3f}  val {vloss.item():.3f}")

    model.eval()
    print("\n--- sample ---")
    print(tok.decode(model.generate(tok.encode("hello"), 150, 0.8)))

    sd = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    sd["itos"] = np.array([tok.itos[i] for i in range(VOCAB_SIZE)])
    sd["cfg_num_heads"] = np.array(NUM_HEADS)
    sd["cfg_num_layers"] = np.array(NUM_LAYERS)
    sd["cfg_max_context"] = np.array(MAX_CONTEXT)
    np.savez_compressed(OUT_PATH, **sd)
    print(f"\nSaved {OUT_PATH}")

    try:
        from google.colab import files
        files.download(OUT_PATH)
        print("Download triggered.")
    except ImportError:
        print("Not running in Colab — find weights.npz in the working directory.")


if __name__ == "__main__":
    main()
