"""Answer selection: taught answers > corpus passage > Markov babble."""

import random
import re

from model import tokens

KB_THRESHOLD = 0.45      # min similarity to answer from taught knowledge
CORPUS_THRESHOLD = 0.12  # min similarity to answer from corpus
FALLBACK = "I don't know yet. Teach me: teach: your question = your answer"


def _clean(word):
    return re.sub(r"[^a-z0-9']", "", word.lower())


def _markov(model, q_tokens, max_words=40):
    """Babble from the chain, seeded on states containing a query word."""
    if not model.chain or not q_tokens:
        return ""
    wanted = set(q_tokens)
    seeds = [s for s in model.chain if any(_clean(w) in wanted for w in s)]
    if not seeds:
        return ""
    starts = [s for s in seeds if s[0][:1].isupper()]
    out = list(random.choice(starts or seeds))
    for _ in range(max_words):
        nxt = model.chain.get(tuple(out[-model.MARKOV_ORDER:]))
        if not nxt:
            break
        out.append(random.choice(nxt))
        if out[-1].endswith((".", "!", "?")) and len(out) > 10:
            break
    return " ".join(out)


def generate(model, message):
    q = tokens(message)
    if q:
        kb_score, kb_answers, c_score, c_text = model.search(q)
        if kb_answers and kb_score >= KB_THRESHOLD:
            return random.choice(kb_answers)
        if c_text and c_score >= CORPUS_THRESHOLD:
            return c_text
    return _markov(model, q) or FALLBACK
