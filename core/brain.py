import json
import random
import re
from config import REPLIES_PATH

# Tracks, per key, which variant indices have already been shown
# this process. Cleared/reshuffled once a key's pool is exhausted.
_shown_indices = {}


def load_replies():
    try:
        with open(REPLIES_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_replies(replies):
    with open(REPLIES_PATH, "w", encoding="utf-8") as f:
        json.dump(replies, f, indent=2)


def _words(text):
    text = re.sub(r"[^\w\s]", "", text.lower())
    return [w for w in text.split() if w]


def find_best_match(user_input, replies):
    exact_key = user_input.strip().lower()
    if exact_key in replies:
        return exact_key

    input_words = set(_words(user_input))
    best_key = None
    best_score = 0.0

    for key in replies:
        key_words = _words(key)
        if not key_words:
            continue
        overlap = sum(1 for w in key_words if w in input_words)
        score = overlap / len(key_words)
        if score > best_score and score >= 0.6:
            best_score = score
            best_key = key

    return best_key


def _pick_non_repeating(key, options):
    shown = _shown_indices.setdefault(key, set())

    if len(shown) >= len(options):
        shown.clear()

    available = [i for i in range(len(options)) if i not in shown]
    chosen = random.choice(available)
    shown.add(chosen)

    return options[chosen]


def collapse(user_input):
    replies = load_replies()
    match = find_best_match(user_input, replies)

    if not match:
        fallback = replies.get("default")
        if fallback:
            return _pick_non_repeating("default", fallback)
        return "I don't understand. Teach me with: teach: your phrase = your answer"

    options = replies[match]
    return _pick_non_repeating(match, options)


def teach(phrase, answer):
    replies = load_replies()
    phrase = phrase.strip().lower()
    answer = answer.strip()

    if not phrase or not answer:
        return "Format: teach: your phrase = your answer"

    replies.setdefault(phrase, []).append(answer)
    save_replies(replies)

    # New variant added — reset that key's tracking so the fresh
    # answer is eligible immediately instead of waiting a full cycle.
    _shown_indices.pop(phrase, None)

    return f'Learned it. "{phrase}" now has {len(replies[phrase])} possible answer(s), saved permanently.'
