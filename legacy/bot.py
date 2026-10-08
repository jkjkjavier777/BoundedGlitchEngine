#!/usr/bin/env python3
"""BoundedGlitchEngine Chatbot — Terminal + Web Interface"""

import sys
import argparse
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Direct imports (now that engine folder is deleted)
from boundedglitch.config import load_config
from boundedglitch.conversation import ConversationManager
from boundedglitch import BoundedGlitchEngine
from boundedglitch.gpt_interface import GPTInterface
from cli import TerminalCLI

class BoundedGlitchBot:
    """Main bot class — orchestrates engine, model, and conversation state."""

    def __init__(self, config_path="bot.json"):
        """Initialize bot with configuration."""
        self.config = load_config(config_path)
        self.conversation_manager = ConversationManager(self.config)
        self.engine = BoundedGlitchEngine(self.config)
        self.gpt = GPTInterface(self.config)
        self.current_persona = None
        
        print("[✓] BoundedGlitchBot initialized")

    def select_persona(self):
        """Show menu and select persona."""
        print("\n" + "="*60)
        print("BoundedGlitchEngine Chatbot")
        print("="*60)
        print("\nSelect a persona:")
        print("  1. The-Book-of-Secret-Knowledge (BoSK)")
        print("  2. BoundedGlitchEngine (BGE)")
        print("  0. Exit")
        
        choice = input("\nEnter choice (0-2): ").strip()
        
        if choice == "1":
            self.current_persona = "bosk"
        elif choice == "2":
            self.current_persona = "bge"
        elif choice == "0":
            print("\n[*] Goodbye!")
            sys.exit(0)
        else:
            print("[!] Invalid choice. Try again.")
            return self.select_persona()
        
        persona_config = self.config["personas"][self.current_persona]
        print(f"\n[✓] Loaded persona: {persona_config['name']}")
        return self.current_persona

    def process_message(self, user_input):
        """
        Process user message through engine + model.
        
        Flow:
            user input
                ↓
            engine (5 layers)
                ↓
            gpt inference
                ↓
            response + save history
        """
        # Add to conversation history
        self.conversation_manager.add_message(
            session_id=self.session_id,
            role="user",
            content=user_input,
            persona=self.current_persona
        )
        
        # Build context from history
        context = self.conversation_manager.get_context(
            session_id=self.session_id,
            max_turns=5
        )
        
        # Run through engine (5 layers)
        engine_output = self.engine.process(
            user_input=user_input,
            context=context,
            persona=self.current_persona
        )
        
        # Generate response via GPT
        gpt_response = self.gpt.generate(
            prompt=engine_output["processed_prompt"],
            persona=self.current_persona,
            max_tokens=self.config["personas"][self.current_persona]["max_tokens"]
        )
        
        # Save assistant message
        self.conversation_manager.add_message(
            session_id=self.session_id,
            role="assistant",
            content=gpt_response,
            persona=self.current_persona,
            metadata={
                "engine_output": engine_output,
                "model": "BoundedGlitchGPT"
            }
        )
        
        return gpt_response

    def interactive_terminal(self):
        """Run interactive terminal chat loop."""
        import uuid
        self.session_id = str(uuid.uuid4())[:8]
        
        self.select_persona()
        
        persona_config = self.config["personas"][self.current_persona]
        print(f"\n[*] {persona_config['description']}")
        print("\nCommands:")
        print("  /switch     - Switch persona")
        print("  /history    - Show conversation history")
        print("  /clear      - Clear conversation")
        print("  /exit       - Exit")
        print("\nStart chatting:\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith("/"):
                    self._handle_command(user_input)
                    continue
                
                # Process message
                response = self.process_message(user_input)
                print(f"\nBot: {response}\n")
                
            except KeyboardInterrupt:
                print("\n\n[*] Interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n[!] Error: {e}\n")

    def _handle_command(self, command):
        """Handle special commands."""
        if command == "/switch":
            self.select_persona()
        elif command == "/history":
            history = self.conversation_manager.get_full_history(self.session_id)
            for msg in history:
                role = msg["role"].upper()
                content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
                print(f"{role}: {content}")
        elif command == "/clear":
            self.conversation_manager.clear_history(self.session_id)
            print("[✓] Conversation cleared")
        elif command == "/exit":
            print("[*] Goodbye!")
            sys.exit(0)
        else:
            print("[!] Unknown command")

    def run_server(self, host="localhost", port=5000):
        """Run Flask server (for web UI later)."""
        from flask import Flask, request, jsonify
        from flask_cors import CORS
        import uuid
        
        app = Flask(__name__)
        CORS(app)
        
        active_sessions = {}
        
        @app.route("/", methods=["GET"])
        def index():
            return {"status": "BoundedGlitchEngine API running"}
        
        @app.route("/session/new", methods=["POST"])
        def new_session():
            """Create new conversation session."""
            data = request.json
            persona = data.get("persona", "bosk")
            session_id = str(uuid.uuid4())[:8]
            active_sessions[session_id] = {"persona": persona}
            return {"session_id": session_id, "persona": persona}
        
        @app.route("/chat", methods=["POST"])
        def chat():
            """Process chat message."""
            data = request.json
            session_id = data.get("session_id")
            user_input = data.get("message")
            
            if session_id not in active_sessions:
                return {"error": "Invalid session"}, 400
            
            self.current_persona = active_sessions[session_id]["persona"]
            self.session_id = session_id
            
            try:
                response = self.process_message(user_input)
                return {"response": response, "status": "ok"}
            except Exception as e:
                return {"error": str(e)}, 500
        
        @app.route("/history", methods=["GET"])
        def get_history():
            """Get conversation history."""
            session_id = request.args.get("session_id")
            if not session_id:
                return {"error": "Missing session_id"}, 400
            
            history = self.conversation_manager.get_full_history(session_id)
            return {"history": history}
        
        print(f"\n[✓] Server starting at http://{host}:{port}")
        app.run(host=host, port=port, debug=False)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="BoundedGlitchEngine Chatbot")
    parser.add_argument("--web", action="store_true", help="Run Flask server instead of CLI")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--config", default="bot.json", help="Config file path")
    
    args = parser.parse_args()
    
    # Initialize bot
    bot = BoundedGlitchBot(config_path=args.config)
    
    # Run mode
    if args.web:
        bot.run_server()
    else:
        bot.interactive_terminal()


if __name__ == "__main__":
    main()

