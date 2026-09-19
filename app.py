from flask import Flask, render_template, request, jsonify
from chatbot import BoundedGlitchEngine

app = Flask(__name__)
bot = BoundedGlitchEngine()


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    message = data.get("message", "").strip()
    if not message:
        return jsonify({"reply": "No message received."}), 400
    return jsonify({"reply": bot.respond(message)})


@app.route("/teach", methods=["POST"])
def teach():
    data = request.get_json(silent=True) or {}
    q = data.get("question", "").strip()
    a = data.get("answer", "").strip()
    if not q or not a:
        return jsonify({"reply": "Need both question and answer."}), 400
    return jsonify({"reply": bot.teach(q, a)})


@app.route("/train", methods=["POST"])
def train():
    return jsonify({"reply": bot.train()})


@app.route("/stats")
def stats():
    return jsonify(bot.stats())


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
