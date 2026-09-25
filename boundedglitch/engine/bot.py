"""
BoundedGlitchEngine Bot: Conversation management and orchestration.
"""

from .gpt_interface import GPTInterface


class BoundedGlitchEngine:
    """Main conversation engine."""

    def __init__(self, checkpoint_path: str, tokenizer_path: str, device: str = 'cpu'):
        self.gpt = GPTInterface(checkpoint_path, tokenizer_path, device=device)
        self.conversation_history = []

    def chat(self, user_input: str, max_tokens: int = 100, temperature: float = 0.8) -> str:
        self.conversation_history.append(("user", user_input))
        response = self.gpt.generate(user_input, max_tokens=max_tokens, temperature=temperature)
        self.conversation_history.append(("bot", response))
        return response

    def interactive_mode(self):
        print("\n" + "=" * 70)
        print("BoundedGlitchEngine - Interactive Mode")
        print("=" * 70)
        print("Type 'quit' to exit.\n")

        while True:
            try:
                user_input = input("You: ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ['quit', 'exit']:
                    print("\n[*] Goodbye!")
                    break
                print("\nBot: ", end="", flush=True)
                response = self.chat(user_input, max_tokens=150, temperature=0.8)
                print(response)
                print()
            except KeyboardInterrupt:
                print("\n\n[*] Interrupted.")
                break
            except Exception as e:
                print(f"\n[Error] {str(e)}\n")

    def get_history(self):
        return self.conversation_history

    def clear_history(self):
        self.conversation_history = []
