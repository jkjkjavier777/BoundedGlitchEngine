from flask import Flask, request, jsonify, render_template

from config import HOST, PORT, DEBUG
from core.brain import collapse, teach
from core.memory import remember

app = Flask(__name__)


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

    if user_input.lower().startswith("teach:"):
        body = user_input[6:].strip()
        parts = body.split("=", 1)
        if len(parts) < 2:
            response = "Format: teach: your phrase = your answer"
        else:
            response = teach(parts[0], parts[1])
    else:
        response = collapse(user_input)

    remember("bot", response)

    return jsonify({"response": response})


if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=DEBUG)
