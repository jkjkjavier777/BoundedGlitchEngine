"""
GPTInterface: Clean boundary between BoundedGlitchEngine and The-BoundedGlitchGPT
"""

import sys
from pathlib import Path

# adjust to wherever The-BoundedGlitchGPT repo lives relative to this file
gpt_repo = Path(__file__).parent.parent.parent / "The-BoundedGlitchGPT"
if str(gpt_repo) not in sys.path:
    sys.path.insert(0, str(gpt_repo))

from gpt import GPT
from tokenizer import CharTokenizer


class GPTInterface:
    """Clean interface to trained BoundedGlitchGPT model (numpy, no torch)."""

    def __init__(self, checkpoint_path: str, tokenizer_path: str):
        self.checkpoint_path = checkpoint_path
        self.tokenizer_path = tokenizer_path

        print(f"[*] Loading tokenizer from {tokenizer_path}...")
        self.tokenizer = CharTokenizer()
        self.tokenizer.load(tokenizer_path)
        print(f"[✓] Tokenizer loaded: {self.tokenizer.vocab_size} tokens")

        print(f"[*] Loading model from {checkpoint_path}...")
        self.model = GPT(
            vocab_size=self.tokenizer.vocab_size,
            embedding_dim=128,
            num_layers=2,
            num_heads=4,
            ff_dim=256,
            max_seq_len=256
        )
        self.model.load(checkpoint_path)
        print("[✓] Model loaded")

    def generate(self, prompt: str, max_tokens: int = 150, temperature: float = 0.8) -> str:
        return self.model.generate(
            self.tokenizer, prompt,
            max_new_tokens=max_tokens,
            temperature=temperature
        )

    def get_info(self) -> dict:
        return {
            'checkpoint': self.checkpoint_path,
            'tokenizer': self.tokenizer_path,
            'vocab_size': self.tokenizer.vocab_size,
        }
