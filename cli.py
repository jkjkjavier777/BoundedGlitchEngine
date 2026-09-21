class TerminalCLI:
    def __init__(self, bot):
        self.bot = bot

    def run(self):
        self.bot.select_persona()
        print("Type 'menu' to switch persona, 'quit' to exit.")
        while True:
            try:
                text = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n[*] Goodbye!")
                break
            if not text:
                continue
            if text.lower() in ("quit", "exit"):
                print("[*] Goodbye!")
                break
            if text.lower() == "menu":
                self.bot.select_persona()
                continue
            reply = self.bot.process_message(text)
            if isinstance(reply, dict):
                reply = reply.get("response") or reply.get("output") or reply
            print(f"\nBot: {reply}")
