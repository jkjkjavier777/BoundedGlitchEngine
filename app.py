from flask import Flask, request, jsonify, render_template

from config import HOST, PORT, DEBUG
from core.memory import remember
from boundedglitch.engine import BoundedGlitchEngine

app = Flask(__name__)
engine = BoundedGlitchEngine()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    user_input = (data.get("message") or "").strip()
    if not user_input:
        return jsonify({"error": "No message provided"}), 400
    remember("user", user_input)
    response = engine.chat(user_input)
    remember("bot", response)
    return jsonify({"response": response, "source": engine.last_meta.get("source")})


if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=False)
