"""GPT Interface."""
import numpy as np
from typing import Optional

class SimpleTokenizer:
    """Simple tokenizer."""
    def __init__(self, vocab_size: int = 256):
        self.vocab_size = vocab_size
    
    def encode(self, text: str) -> np.ndarray:
        return np.array([ord(c) % self.vocab_size for c in text], dtype=np.int32)
    
    def decode(self, token_ids: np.ndarray) -> str:
        return ''.join([chr(int(tid)) if tid < 128 else '?' for tid in token_ids])

class GPTModel:
    """GPT model."""
    def __init__(self, config: dict):
        self.config = config
        self.vocab_size = config.get("vocab_size", 256)
    
    def generate(self, token_ids: np.ndarray, max_tokens: int = 100, 
                temperature: float = 0.8) -> np.ndarray:
        tokens = list(token_ids)
        for _ in range(max_tokens):
            tokens.append(np.random.randint(0, self.vocab_size))
        return np.array(tokens, dtype=np.int32)

class GPTInterface:
    """Interface to GPT."""
    def __init__(self, config: dict):
        self.config = config
        self.model_config = config["model"]
        self.tokenizer = SimpleTokenizer(self.model_config.get("vocab_size", 256))
        self.model = GPTModel(self.model_config)
        print("[✓] GPTInterface initialized")

    def generate(self, prompt: str, persona: str = "bosk", 
                max_tokens: int = 150, temperature: Optional[float] = None) -> str:
        """Generate text."""
        if temperature is None:
            temperature = self.config["personas"].get(persona, {}).get("temperature", 0.8)
        prompt_ids = self.tokenizer.encode(prompt)
        generated_ids = self.model.generate(prompt_ids, max_tokens=max_tokens, temperature=temperature)
        return self.tokenizer.decode(generated_ids)[-50:].strip()
